"""Tests for the turn-lifetime keep-alive heartbeat.

The heartbeat must span the whole ``_turn_by_session`` entry — not just a
hermes ``_process_message_background`` task. A top-level ``delegate_task``
always dispatches its subagents detached: the parent processing task (and with
it hermes' ``_keep_typing``) exits while the subagents run, but the FE keeps
its SSE stream open because the adapter kept the turn alive. The heartbeat is
the only liveness signal on the wire during that window.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from domyn.agents.core import AgentAction, BaseEvent, ExecutionEventType
from domyn.agents.integrations.hermes_platform_gateway.adapter import (
    DomynPlatformAdapter,
    _build_session_key,
)

_SESSION_KEY = _build_session_key("chan-1", "conv-1")


def _make_adapter(channel_id: str = "chan-1", with_turn: bool = True):
    mock_client = MagicMock()
    mock_client.send_event = AsyncMock()

    adapter = DomynPlatformAdapter(
        config=MagicMock(),
        channel_id=channel_id,
        relay_client_factory=lambda _cb: mock_client,
        tool_rename_mapping={},
    )
    adapter._HEARTBEAT_INTERVAL = 0.01
    if with_turn:
        turn = BaseEvent(
            event_type=ExecutionEventType.AGENT_START,
            author="u1",
            conversation_id="conv-1",
            interaction_id="int-1",
            turn_id="turn-1",
            action=AgentAction(name="invoke", parameters={"input": "hello"}),
        )
        adapter._turn_by_session[_build_session_key(channel_id, "conv-1")] = turn
    adapter._client = mock_client
    return adapter, mock_client


def _heartbeats(mock_client) -> list[BaseEvent]:
    return [
        call.args[0]
        for call in mock_client.send_event.await_args_list
        if call.args[0].event_type == ExecutionEventType.TOOL_UPDATE
    ]


async def _wait_for_heartbeat(mock_client, minimum: int = 1, timeout: float = 1.0) -> None:
    async with asyncio.timeout(timeout):
        while len(_heartbeats(mock_client)) < minimum:
            await asyncio.sleep(0.005)


# ---------------------------------------------------------------------------
# Start / stop lifecycle
# ---------------------------------------------------------------------------


def test_agent_start_starts_heartbeat():
    adapter, _ = _make_adapter(with_turn=False)

    async def run():
        start = BaseEvent(
            event_type=ExecutionEventType.AGENT_START,
            author="u1",
            conversation_id="conv-1",
            interaction_id="int-1",
            turn_id="turn-1",
            action=AgentAction(name="invoke", parameters={"input": "hello"}),
        )
        await adapter._handle_agent_start(start)
        assert _SESSION_KEY in adapter._turn_by_session
        task = adapter._heartbeat_by_session.get(_SESSION_KEY)
        assert task is not None and not task.done()
        adapter._stop_heartbeat(_SESSION_KEY)

    asyncio.run(run())


def test_start_heartbeat_is_idempotent():
    adapter, _ = _make_adapter()

    async def run():
        adapter._start_heartbeat(_SESSION_KEY)
        task = adapter._heartbeat_by_session[_SESSION_KEY]
        adapter._start_heartbeat(_SESSION_KEY)
        assert adapter._heartbeat_by_session[_SESSION_KEY] is task
        adapter._stop_heartbeat(_SESSION_KEY)

    asyncio.run(run())


def test_heartbeat_emits_tool_update_while_turn_open():
    adapter, mock_client = _make_adapter()

    async def run():
        adapter._start_heartbeat(_SESSION_KEY)
        await _wait_for_heartbeat(mock_client)
        adapter._stop_heartbeat(_SESSION_KEY)

    asyncio.run(run())
    beat = _heartbeats(mock_client)[0]
    assert beat.is_partial is True
    assert beat.turn_id == "turn-1"
    assert beat.conversation_id == "conv-1"
    assert "Still running 'invoke'" in beat.content[0].metadata["message"]


def test_heartbeat_self_terminates_when_turn_missing():
    adapter, mock_client = _make_adapter(with_turn=False)

    async def run():
        adapter._start_heartbeat(_SESSION_KEY)
        task = adapter._heartbeat_by_session[_SESSION_KEY]
        async with asyncio.timeout(1.0):
            await task
        # The loop removed its own registry entry on the way out.
        assert _SESSION_KEY not in adapter._heartbeat_by_session

    asyncio.run(run())
    assert _heartbeats(mock_client) == []


# ---------------------------------------------------------------------------
# The subagent gap — heartbeat survives the background-delegation window
# ---------------------------------------------------------------------------


def test_heartbeat_survives_background_wrap_up():
    """After the wrap-up partial RESPONSE the turn stays alive, so the
    heartbeat must keep ticking (this is exactly the window where hermes'
    _keep_typing is already dead while subagents run)."""
    adapter, mock_client = _make_adapter()

    async def run():
        adapter._start_heartbeat(_SESSION_KEY)
        adapter.mark_background_delegation("conv-1")
        await adapter.send(chat_id="conv-1", content="Started 2 subagents.", reply_to="msg-1")
        assert _SESSION_KEY in adapter._turn_by_session  # turn kept alive
        task = adapter._heartbeat_by_session.get(_SESSION_KEY)
        assert task is not None and not task.done()

        mock_client.send_event.reset_mock()
        await _wait_for_heartbeat(mock_client)  # still ticking after wrap-up
        adapter._stop_heartbeat(_SESSION_KEY)

    asyncio.run(run())


def test_final_send_stops_heartbeat():
    adapter, mock_client = _make_adapter()

    async def run():
        adapter._start_heartbeat(_SESSION_KEY)
        await adapter.send(chat_id="conv-1", content="Done.", reply_to="msg-1")
        assert _SESSION_KEY not in adapter._turn_by_session
        assert _SESSION_KEY not in adapter._heartbeat_by_session

    asyncio.run(run())
    final = mock_client.send_event.await_args.args[0]
    assert final.event_type == ExecutionEventType.AGENT_END


def test_handle_cancel_stops_heartbeat():
    adapter, _ = _make_adapter()

    async def run():
        adapter._start_heartbeat(_SESSION_KEY)
        cancel = BaseEvent(
            event_type=ExecutionEventType.AGENT_END,
            author="u1",
            conversation_id="conv-1",
            error_code="user_cancelled",
        )
        await adapter._handle_cancel(cancel)
        assert _SESSION_KEY not in adapter._heartbeat_by_session

    asyncio.run(run())


def test_fail_pending_clears_all_heartbeats():
    adapter, _ = _make_adapter()

    async def run():
        adapter._start_heartbeat(_SESSION_KEY)
        task = adapter._heartbeat_by_session[_SESSION_KEY]
        adapter._fail_pending("disconnect")
        assert adapter._heartbeat_by_session == {}
        await asyncio.sleep(0)  # let the cancellation propagate
        assert task.cancelled()

    asyncio.run(run())
