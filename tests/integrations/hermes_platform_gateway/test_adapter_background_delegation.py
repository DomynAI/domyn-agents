"""Tests for the background-delegation "not final" behaviour.

A top-level ``delegate_task`` dispatches its subagents in the background and
returns ``{"status": "dispatched", "mode": "background", ...}`` immediately; the
consolidated result re-enters later as a synthetic ``internal`` message. The
adapter counts in-flight dispatches per session: while any are pending it emits
outbound sends as non-terminal ``RESPONSE``s (``is_partial=True``) and keeps the
turn alive; each completion re-entry decrements the counter in
``handle_message``, and the send that follows the last one closes the turn with
the real ``AGENT_END``.
"""

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType
from gateway.session import SessionSource

from domyn.agents.core import AgentAction, BaseEvent, ExecutionEventType
from domyn.agents.integrations.hermes_platform_gateway.adapter import (
    DomynPlatformAdapter,
    _build_session_key,
)

_PKG = "domyn.agents.integrations.hermes_platform_gateway"


def _make_adapter(channel_id: str = "chan-1"):
    mock_client = MagicMock()
    mock_client.send_event = AsyncMock()

    adapter = DomynPlatformAdapter(
        config=MagicMock(),
        channel_id=channel_id,
        relay_client_factory=lambda _cb: mock_client,
        tool_rename_mapping={},
    )
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


def _internal_completion(chat_id: str = "conv-1") -> MessageEvent:
    """The synthetic message hermes' async-delegation watcher re-injects when a
    background delegation completes (``_inject_watch_notification``)."""
    source = SessionSource(
        platform=Platform("domyn"),
        chat_id=chat_id,
        chat_name=chat_id,
        chat_type="dm",
        user_id="u1",
        user_name="u1",
    )
    return MessageEvent(
        text="[ASYNC DELEGATION BATCH COMPLETE — deleg_test]",
        message_type=MessageType.TEXT,
        source=source,
        message_id=None,
        timestamp=datetime.now(),
        internal=True,
    )


def _reenter(adapter: DomynPlatformAdapter, event: MessageEvent) -> None:
    """Run the adapter's handle_message override with the gateway base stubbed,
    so only the adapter-side bookkeeping (the counter decrement) executes."""
    with patch.object(BasePlatformAdapter, "handle_message", AsyncMock()) as sup:
        asyncio.run(adapter.handle_message(event))
        sup.assert_awaited_once_with(event)


# ---------------------------------------------------------------------------
# mark_background_delegation
# ---------------------------------------------------------------------------


def test_mark_background_delegation_flags_session():
    adapter, _ = _make_adapter()
    adapter.mark_background_delegation("conv-1")
    assert _build_session_key("chan-1", "conv-1") in adapter._bg_delegation_by_session


def test_mark_background_delegation_no_op_for_empty_chat_id():
    adapter, _ = _make_adapter()
    adapter.mark_background_delegation("")
    assert adapter._bg_delegation_by_session == {}


def test_mark_background_delegation_counts_dispatches():
    """Each background dispatch adds one to the session's in-flight count."""
    adapter, _ = _make_adapter()
    session_key = _build_session_key("chan-1", "conv-1")

    adapter.mark_background_delegation("conv-1")
    adapter.mark_background_delegation("conv-1")

    assert adapter._bg_delegation_by_session[session_key] == 2


def test_fail_pending_clears_background_delegation_flag():
    """A disconnect must not leave a stale flag that would downgrade the next
    turn's final answer to a non-terminal RESPONSE."""
    adapter, _ = _make_adapter()
    adapter.mark_background_delegation("conv-1")
    assert adapter._bg_delegation_by_session  # flagged

    adapter._fail_pending("disconnect")

    assert adapter._bg_delegation_by_session == {}


# ---------------------------------------------------------------------------
# send() — background delegation downgrades AGENT_END → partial RESPONSE
# ---------------------------------------------------------------------------


def test_send_emits_partial_response_when_background_pending():
    adapter, mock_client = _make_adapter()
    session_key = _build_session_key("chan-1", "conv-1")
    adapter.mark_background_delegation("conv-1")

    asyncio.run(adapter.send(chat_id="conv-1", content="Working on it…", reply_to="msg-1"))

    event: BaseEvent = mock_client.send_event.await_args.args[0]
    assert event.event_type == ExecutionEventType.RESPONSE
    assert event.is_partial is True
    assert event.content[0].text == "Working on it…"
    # Turn is kept alive so the background completion can deliver the real end.
    assert session_key in adapter._turn_by_session
    # The dispatch is still in flight: only its completion re-entry (a synthetic
    # internal message through handle_message) decrements the counter — a send
    # must never consume it, or a completion drained as a queued follow-up at
    # end of turn would leave the turn open forever.
    assert adapter._bg_delegation_by_session[session_key] == 1


def test_send_final_agent_end_when_no_background_pending():
    adapter, mock_client = _make_adapter()
    session_key = _build_session_key("chan-1", "conv-1")

    asyncio.run(adapter.send(chat_id="conv-1", content="Done.", reply_to="msg-1"))

    event: BaseEvent = mock_client.send_event.await_args.args[0]
    assert event.event_type == ExecutionEventType.AGENT_END
    assert event.is_partial is False
    assert session_key not in adapter._turn_by_session


def test_background_completion_reenters_as_final_after_wrap_up():
    """The wrap-up is partial and keeps the turn; the completion re-entry
    (synthetic internal message) decrements the counter, so the send that
    follows it goes out as the terminal AGENT_END."""
    adapter, mock_client = _make_adapter()
    session_key = _build_session_key("chan-1", "conv-1")
    adapter.mark_background_delegation("conv-1")

    # Wrap-up: non-final.
    asyncio.run(adapter.send(chat_id="conv-1", content="Started the subagent.", reply_to="msg-1"))
    assert session_key in adapter._turn_by_session

    # The completion re-enters through handle_message, then its result is sent
    # against the same live turn: final AGENT_END.
    _reenter(adapter, _internal_completion())
    asyncio.run(adapter.send(chat_id="conv-1", content="Here are the results.", reply_to="msg-2"))
    event: BaseEvent = mock_client.send_event.await_args.args[0]
    assert event.event_type == ExecutionEventType.AGENT_END
    assert event.is_partial is False
    assert session_key not in adapter._turn_by_session


def test_two_background_delegations_close_only_after_last():
    """With two dispatches in flight, the first completion's report must stay
    non-terminal (the second subagent is still running); only the send after
    the last completion closes the turn."""
    adapter, mock_client = _make_adapter()
    session_key = _build_session_key("chan-1", "conv-1")
    adapter.mark_background_delegation("conv-1")
    adapter.mark_background_delegation("conv-1")

    # Wrap-up: non-final, both in flight.
    asyncio.run(adapter.send(chat_id="conv-1", content="Both dispatched.", reply_to="msg-1"))

    # First completion: its report is still non-terminal and keeps the turn.
    _reenter(adapter, _internal_completion())
    asyncio.run(adapter.send(chat_id="conv-1", content="First done.", reply_to="msg-2"))
    event: BaseEvent = mock_client.send_event.await_args.args[0]
    assert event.event_type == ExecutionEventType.RESPONSE
    assert event.is_partial is True
    assert session_key in adapter._turn_by_session

    # Last completion: the follow-up send is the real final word.
    _reenter(adapter, _internal_completion())
    asyncio.run(adapter.send(chat_id="conv-1", content="Second done.", reply_to="msg-3"))
    event = mock_client.send_event.await_args.args[0]
    assert event.event_type == ExecutionEventType.AGENT_END
    assert event.is_partial is False
    assert session_key not in adapter._turn_by_session


# ---------------------------------------------------------------------------
# handle_message — decrement edges
# ---------------------------------------------------------------------------


def test_non_internal_message_does_not_decrement():
    """A regular user message must never consume an in-flight delegation."""
    adapter, _ = _make_adapter()
    session_key = _build_session_key("chan-1", "conv-1")
    adapter.mark_background_delegation("conv-1")

    user_msg = _internal_completion()
    user_msg.internal = False
    _reenter(adapter, user_msg)

    assert adapter._bg_delegation_by_session[session_key] == 1


def test_internal_completion_without_pending_is_noop():
    """Internal re-injections not tied to a delegation (e.g. process watch
    notifications) must pass through without corrupting the counter."""
    adapter, _ = _make_adapter()

    _reenter(adapter, _internal_completion())

    assert adapter._bg_delegation_by_session == {}


def test_internal_completion_with_empty_chat_id_is_noop():
    adapter, _ = _make_adapter()
    session_key = _build_session_key("chan-1", "conv-1")
    adapter.mark_background_delegation("conv-1")

    _reenter(adapter, _internal_completion(chat_id=""))

    assert adapter._bg_delegation_by_session[session_key] == 1


# ---------------------------------------------------------------------------
# post_tool_call hook — detection of the background dispatch
# ---------------------------------------------------------------------------


def _register_post_tool_hook(monkeypatch):
    """Run register() with a stubbed adapter and return (hook, mock_adapter)."""
    monkeypatch.setenv("DOMYN_API_KEY", "key-1")
    monkeypatch.setenv("DOMYN_BASE_URL", "api.example.com")
    monkeypatch.setenv("DOMYN_SPACE_ID", "space-1")
    monkeypatch.setenv("DOMYN_CHANNEL_ID", "chan-1")

    mock_adapter = MagicMock()
    mock_adapter._chat_id_by_task = {"task-1": "chat-1"}
    mock_adapter.pop_visibility_call.return_value = None  # early-return after detection

    with (
        patch(f"{_PKG}.client.fetch_tools", return_value=[]),
        patch(f"{_PKG}._start_refresh_loop"),
        patch(f"{_PKG}.adapter.DomynPlatformAdapter", return_value=mock_adapter),
        patch(f"{_PKG}.relay_client.DomynRelayClient"),
    ):
        ctx = MagicMock()
        from domyn.agents.integrations.hermes_platform_gateway import register

        register(ctx)
        factory = ctx.register_platform.call_args.kwargs["adapter_factory"]
        factory(MagicMock())

    hook = next(
        c.args[1] for c in ctx.register_hook.call_args_list if c.args[0] == "post_tool_call"
    )
    return hook, mock_adapter


def test_post_tool_call_flags_background_delegation(monkeypatch):
    hook, mock_adapter = _register_post_tool_hook(monkeypatch)
    result = json.dumps({"status": "dispatched", "mode": "background", "count": 2})

    hook(tool_name="delegate_task", task_id="task-1", result=result, error=None)

    mock_adapter.mark_background_delegation.assert_called_once_with("chat-1")


def test_post_tool_call_ignores_synchronous_delegate_result(monkeypatch):
    hook, mock_adapter = _register_post_tool_hook(monkeypatch)
    result = json.dumps({"results": [{"status": "completed", "summary": "done"}]})

    hook(tool_name="delegate_task", task_id="task-1", result=result, error=None)

    mock_adapter.mark_background_delegation.assert_not_called()


def test_post_tool_call_ignores_non_delegate_tool(monkeypatch):
    hook, mock_adapter = _register_post_tool_hook(monkeypatch)
    result = json.dumps({"status": "dispatched", "mode": "background", "count": 1})

    hook(tool_name="read_file", task_id="task-1", result=result, error=None)

    mock_adapter.mark_background_delegation.assert_not_called()
