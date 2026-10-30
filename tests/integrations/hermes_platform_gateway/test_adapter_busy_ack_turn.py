"""The turn must stay open while a tool call is in flight.

hermes' busy/steer ack arrives through ``_send_with_retry`` with ``reply_to``
set, so it used to terminate the turn: the pending call could no longer be
matched, later calls got "no active turn for session", and the run's own answer
was rejected with "no prior AGENT_START". The send that happens once nothing is
pending is now the one that closes the turn.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from domyn.agents.core import (
    METADATA_VISIBILITY_ONLY,
    AgentAction,
    BaseEvent,
    ExecutionEventType,
    ToolAction,
)
from domyn.agents.integrations.hermes_platform_gateway.adapter import (
    DomynPlatformAdapter,
    _build_session_key,
)

_CHANNEL = "chan-1"
_CONV = "conv-1"
_SESSION_KEY = _build_session_key(_CHANNEL, _CONV)

# Abridged form of the notice hermes appends the first time a message lands on a
# busy session — the payload that used to close the turn.
_BUSY_ACK = "⏩ Steered into current run. Your message arrives after the next tool call."


def _make_adapter():
    mock_client = MagicMock()
    mock_client.send_event = AsyncMock()

    adapter = DomynPlatformAdapter(
        config=MagicMock(),
        channel_id=_CHANNEL,
        relay_client_factory=lambda _cb: mock_client,
        tool_rename_mapping={},
    )
    adapter._turn_by_session[_SESSION_KEY] = BaseEvent(
        event_type=ExecutionEventType.AGENT_START,
        author="u1",
        conversation_id=_CONV,
        interaction_id="int-1",
        turn_id="turn-1",
        action=AgentAction(name="invoke", parameters={"input": "hello"}),
    )
    adapter._client = mock_client
    return adapter, mock_client


def _register_pending(adapter, fut, loop, call_id: str = "call-1") -> None:
    adapter._pending_calls[call_id] = (fut, loop)
    adapter._session_by_call[call_id] = _SESSION_KEY


def _tool_result(call_id: str, *, visibility: bool = False, legacy: bool = False) -> BaseEvent:
    """A TOOL_END as the platform (or our own visibility path) sends it back."""
    return BaseEvent(
        event_type=ExecutionEventType.TOOL_END,
        author="u1",
        conversation_id=_CONV,
        interaction_id="int-1",
        turn_id="turn-1",
        metadata={METADATA_VISIBILITY_ONLY: True} if visibility else {},
        action=ToolAction(
            name="read_file",
            parameters={},
            call_id=f"visibility-{call_id}" if legacy else call_id,
            observation="done",
        ),
    )


def _sent(mock_client) -> BaseEvent:
    return mock_client.send_event.await_args.args[0]


# ---------------------------------------------------------------------------
# send() — a pending tool call downgrades AGENT_END → partial RESPONSE
# ---------------------------------------------------------------------------


def test_send_keeps_turn_open_while_tool_call_pending():
    """The busy ack must not be the turn terminator."""

    async def scenario():
        adapter, mock_client = _make_adapter()
        loop = asyncio.get_running_loop()
        _register_pending(adapter, loop.create_future(), loop)
        adapter._start_heartbeat(_SESSION_KEY)

        await adapter.send(chat_id=_CONV, content=_BUSY_ACK, reply_to="msg-2")

        event = _sent(mock_client)
        assert event.event_type == ExecutionEventType.RESPONSE
        assert event.is_partial is True
        assert _SESSION_KEY in adapter._turn_by_session
        # The heartbeat is the platform's liveness signal for the turn; stopping
        # it here exposed the session to the SSE delegate_timeout.
        assert _SESSION_KEY in adapter._heartbeat_by_session
        adapter._stop_heartbeat(_SESSION_KEY)

    asyncio.run(scenario())


def test_send_closes_turn_when_nothing_pending():
    """Unchanged behaviour: a genuine final answer still ends the turn."""
    adapter, mock_client = _make_adapter()

    asyncio.run(adapter.send(chat_id=_CONV, content="Done.", reply_to="msg-1"))

    event = _sent(mock_client)
    assert event.event_type == ExecutionEventType.AGENT_END
    assert event.is_partial is False
    assert _SESSION_KEY not in adapter._turn_by_session


def test_pending_call_for_another_session_does_not_downgrade():
    """The guard is per-session: another conversation's in-flight call must not
    hold this turn open."""
    adapter, mock_client = _make_adapter()
    adapter._session_by_call["call-9"] = _build_session_key(_CHANNEL, "conv-other")

    asyncio.run(adapter.send(chat_id=_CONV, content="Done.", reply_to="msg-1"))

    assert _sent(mock_client).event_type == ExecutionEventType.AGENT_END
    assert _SESSION_KEY not in adapter._turn_by_session


def test_interim_narrative_is_unaffected_by_pending_calls():
    """A mid-turn narrative never reaches the wire — hermes surfaces it via
    TOOL_START.thought. The guard must leave that early return untouched."""

    async def scenario():
        adapter, mock_client = _make_adapter()
        loop = asyncio.get_running_loop()
        _register_pending(adapter, loop.create_future(), loop)

        result = await adapter.send(chat_id=_CONV, content="I'll check that…")

        assert result.success is True
        assert result.message_id == "interim-noop"
        mock_client.send_event.assert_not_awaited()
        assert _SESSION_KEY in adapter._turn_by_session

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# The regression that made the user's answer disappear
# ---------------------------------------------------------------------------


def test_answer_after_busy_ack_is_deliverable():
    """End-to-end shape of the bug: ack, tool result, then the run's own answer,
    which must go out as the terminal AGENT_END instead of being rejected."""

    async def scenario():
        adapter, mock_client = _make_adapter()
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        _register_pending(adapter, fut, loop)

        # A follow-up lands on the busy session: ack, turn kept open.
        await adapter.send(chat_id=_CONV, content=_BUSY_ACK, reply_to="msg-2")
        assert _sent(mock_client).event_type == ExecutionEventType.RESPONSE

        # The platform answers the tool call — still matchable, because the turn
        # was never closed.
        adapter._resolve_tool_call(_tool_result("call-1"))
        await asyncio.sleep(0)
        assert await fut == "done"
        assert not adapter._has_pending_calls(_SESSION_KEY)

        # The run's real answer now closes the turn, and reaches the user.
        result = await adapter.send(chat_id=_CONV, content="Here you go.", reply_to="msg-2")

        assert result.success is True
        assert _sent(mock_client).event_type == ExecutionEventType.AGENT_END
        assert _SESSION_KEY not in adapter._turn_by_session

    asyncio.run(scenario())


def test_send_fails_when_turn_already_gone():
    """Guards the failure mode the fix removes: with no live turn, send() is
    rejected outright — this is what "no prior AGENT_START" looked like."""
    adapter, mock_client = _make_adapter()
    adapter._turn_by_session.pop(_SESSION_KEY)

    result = asyncio.run(adapter.send(chat_id=_CONV, content="Here you go.", reply_to="msg-2"))

    assert result.success is False
    assert result.error == "no prior AGENT_START"
    mock_client.send_event.assert_not_awaited()


# ---------------------------------------------------------------------------
# call_tool bookkeeping — no exit may leave _session_by_call behind
# ---------------------------------------------------------------------------


def test_timeout_clears_both_call_maps():
    """A stale _session_by_call entry would hold the turn open for the pod's
    lifetime, since send() now reads it."""

    async def scenario():
        adapter, _ = _make_adapter()
        out = await adapter.call_tool(
            session_key=_SESSION_KEY, tool_name="read_file", args={}, timeout=0.01
        )
        assert "timed out" in json.loads(out)["error"]
        assert adapter._pending_calls == {}
        assert adapter._session_by_call == {}
        assert not adapter._has_pending_calls(_SESSION_KEY)

    asyncio.run(scenario())


def test_future_error_clears_both_call_maps():
    """The "TOOL future raised" exit used to drop neither map."""

    async def scenario():
        adapter, _ = _make_adapter()
        task = asyncio.create_task(
            adapter.call_tool(session_key=_SESSION_KEY, tool_name="read_file", args={}, timeout=5)
        )
        while not adapter._pending_calls:
            await asyncio.sleep(0)
        fut, _loop = next(iter(adapter._pending_calls.values()))
        fut.set_exception(RuntimeError("relay disconnected"))

        assert "relay disconnected" in json.loads(await task)["error"]
        assert adapter._pending_calls == {}
        assert adapter._session_by_call == {}

    asyncio.run(scenario())


def test_cancellation_during_tool_start_clears_both_call_maps():
    """A user stop cancels call_tool mid-send: CancelledError must not skip cleanup,
    or the maps would hold the turn open for the pod's lifetime."""

    async def scenario():
        adapter, mock_client = _make_adapter()
        sending = asyncio.Event()

        async def hang(_event):
            sending.set()
            await asyncio.Event().wait()

        mock_client.send_event = AsyncMock(side_effect=hang)
        task = asyncio.create_task(
            adapter.call_tool(session_key=_SESSION_KEY, tool_name="read_file", args={}, timeout=5)
        )
        await sending.wait()
        assert adapter._session_by_call  # registered before the TOOL_START send

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert adapter._pending_calls == {}
        assert adapter._session_by_call == {}

    asyncio.run(scenario())


def test_tool_start_send_failure_clears_both_call_maps():
    async def scenario():
        adapter, mock_client = _make_adapter()
        mock_client.send_event = AsyncMock(side_effect=RuntimeError("relay down"))

        out = await adapter.call_tool(
            session_key=_SESSION_KEY, tool_name="read_file", args={}, timeout=5
        )

        assert "send failed" in json.loads(out)["error"]
        assert adapter._pending_calls == {}
        assert adapter._session_by_call == {}

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# _resolve_tool_call — our own visibility frames are not tool results
# ---------------------------------------------------------------------------


def test_visibility_only_event_is_ignored(caplog):
    """Echoed visibility frames never had a pending future; resolving them only
    logged a warning that buried the real ones."""
    adapter, _ = _make_adapter()

    with caplog.at_level("WARNING"):
        adapter._resolve_tool_call(_tool_result("call-1", visibility=True))

    assert "has no pending future" not in caplog.text


def test_legacy_visibility_call_id_is_ignored(caplog):
    """Frames from a runtime old enough not to stamp the metadata flag are
    recognised by the call_id prefix, as the runner does."""
    adapter, _ = _make_adapter()

    with caplog.at_level("WARNING"):
        adapter._resolve_tool_call(_tool_result("call-1", legacy=True))

    assert "has no pending future" not in caplog.text


def test_visibility_event_does_not_consume_a_real_pending_call():
    """A visibility frame must not pop bookkeeping that belongs to a real call
    of the same id."""

    async def scenario():
        adapter, _ = _make_adapter()
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        _register_pending(adapter, fut, loop)

        adapter._resolve_tool_call(_tool_result("call-1", visibility=True))

        assert adapter._pending_calls  # still awaiting the real TOOL_END
        assert adapter._has_pending_calls(_SESSION_KEY)
        assert not fut.done()

    asyncio.run(scenario())


def test_real_tool_end_without_pending_still_warns(caplog):
    """The signal we care about — a result for a call this process never made,
    which is how a second gateway on the same relay channel shows up."""
    adapter, _ = _make_adapter()

    with caplog.at_level("WARNING"):
        adapter._resolve_tool_call(_tool_result("call-unknown"))

    assert "has no pending future" in caplog.text
