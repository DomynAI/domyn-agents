"""Unit tests for DomynPlatformAdapter LLM_START / LLM_END emission."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from domyn.agents.core import AgentAction, BaseEvent, ExecutionEventType, Part
from domyn.agents.integrations.hermes_platform_gateway.adapter import (
    DomynPlatformAdapter,
    _build_session_key,
)


def _make_adapter(channel_id: str = "chan-1"):  # type: ignore[return]
    mock_client = MagicMock()
    mock_client.send_event = AsyncMock()

    adapter = DomynPlatformAdapter(
        config=MagicMock(),
        channel_id=channel_id,
        relay_client_factory=lambda _cb: mock_client,
        tool_rename_mapping={},
    )
    # Inject a fake active turn so emit methods have correlation IDs to copy.
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
    return adapter, mock_client  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Stash helpers
# ---------------------------------------------------------------------------


def test_record_and_pop_llm_call_roundtrip():
    adapter, _ = _make_adapter()
    adapter.record_llm_call(session_id="s1", chat_id="conv-1", call_id="llm-abc")
    result = adapter.pop_llm_call("s1")
    assert result == ("conv-1", "llm-abc")
    assert adapter.pop_llm_call("s1") is None  # cleared after pop


def test_record_llm_call_no_op_for_empty_session_id():
    adapter, _ = _make_adapter()
    adapter.record_llm_call(session_id="", chat_id="conv-1", call_id="llm-abc")
    assert adapter.pop_llm_call("") is None


def test_pop_llm_call_returns_none_for_unknown_session():
    adapter, _ = _make_adapter()
    assert adapter.pop_llm_call("unknown") is None


def test_fail_pending_clears_llm_by_session():
    adapter, _ = _make_adapter()
    adapter.record_llm_call(session_id="s1", chat_id="conv-1", call_id="llm-abc")
    adapter._fail_pending("disconnect")
    assert adapter.pop_llm_call("s1") is None


# ---------------------------------------------------------------------------
# record_hermes_session
# ---------------------------------------------------------------------------


def test_record_hermes_session_roundtrip():
    adapter, _ = _make_adapter()
    adapter.record_hermes_session(session_id="s1", chat_id="conv-1")
    assert adapter._chat_id_by_hermes_session["s1"] == "conv-1"


def test_record_hermes_session_no_op_for_empty_session_id():
    adapter, _ = _make_adapter()
    adapter.record_hermes_session(session_id="", chat_id="conv-1")
    assert adapter._chat_id_by_hermes_session == {}


def test_record_hermes_session_no_op_for_empty_chat_id():
    adapter, _ = _make_adapter()
    adapter.record_hermes_session(session_id="s1", chat_id="")
    assert adapter._chat_id_by_hermes_session == {}


def test_fail_pending_clears_chat_id_by_hermes_session():
    adapter, _ = _make_adapter()
    adapter.record_hermes_session(session_id="s1", chat_id="conv-1")
    adapter._fail_pending("disconnect")
    assert adapter._chat_id_by_hermes_session == {}


# ---------------------------------------------------------------------------
# emit_llm_start
# ---------------------------------------------------------------------------


def test_emit_llm_start_sends_correct_event():
    adapter, mock_client = _make_adapter()
    asyncio.run(adapter.emit_llm_start(chat_id="conv-1", model="gpt-4", call_id="llm-xyz"))

    mock_client.send_event.assert_awaited_once()
    event: BaseEvent = mock_client.send_event.await_args.args[0]
    assert event.event_type == ExecutionEventType.LLM_START
    assert event.conversation_id == "conv-1"
    assert event.author == "u1"
    assert event.metadata == {"model": "gpt-4", "call_id": "llm-xyz"}
    assert event.content == []


def test_emit_llm_start_no_op_when_no_turn():
    adapter, mock_client = _make_adapter()
    asyncio.run(adapter.emit_llm_start(chat_id="unknown-conv", model="gpt-4", call_id="llm-xyz"))
    mock_client.send_event.assert_not_awaited()


def test_emit_llm_start_swallows_send_exception():
    adapter, mock_client = _make_adapter()
    mock_client.send_event.side_effect = RuntimeError("boom")
    # Must not propagate
    asyncio.run(adapter.emit_llm_start(chat_id="conv-1", model="gpt-4", call_id="llm-xyz"))


# ---------------------------------------------------------------------------
# emit_llm_end
# ---------------------------------------------------------------------------


def test_emit_llm_end_sends_correct_event():
    adapter, mock_client = _make_adapter()
    asyncio.run(
        adapter.emit_llm_end(
            chat_id="conv-1",
            model="gpt-4",
            call_id="llm-xyz",
            assistant_response="Hello!",
        )
    )

    mock_client.send_event.assert_awaited_once()
    event: BaseEvent = mock_client.send_event.await_args.args[0]
    assert event.event_type == ExecutionEventType.LLM_END
    assert event.conversation_id == "conv-1"
    assert event.metadata == {"model": "gpt-4", "call_id": "llm-xyz"}
    assert len(event.content) == 1
    assert event.content[0].text == "Hello!"


def test_emit_llm_end_empty_content_when_no_response():
    adapter, mock_client = _make_adapter()
    asyncio.run(
        adapter.emit_llm_end(
            chat_id="conv-1",
            model="gpt-4",
            call_id="llm-xyz",
            assistant_response="",
        )
    )
    event: BaseEvent = mock_client.send_event.await_args.args[0]
    assert event.content == []


def test_emit_llm_end_no_op_when_no_turn():
    adapter, mock_client = _make_adapter()
    asyncio.run(
        adapter.emit_llm_end(
            chat_id="unknown-conv",
            model="gpt-4",
            call_id="llm-xyz",
            assistant_response="done",
        )
    )
    mock_client.send_event.assert_not_awaited()


def test_emit_llm_end_swallows_send_exception():
    adapter, mock_client = _make_adapter()
    mock_client.send_event.side_effect = RuntimeError("boom")
    asyncio.run(
        adapter.emit_llm_end(
            chat_id="conv-1",
            model="gpt-4",
            call_id="llm-xyz",
            assistant_response="done",
        )
    )


# ---------------------------------------------------------------------------
# _turn_by_session lifecycle
# ---------------------------------------------------------------------------


def _make_send_result_args(chat_id: str = "conv-1") -> dict:
    return {"chat_id": chat_id, "content": "done", "reply_to": "msg-1"}


def test_turn_removed_from_session_on_agent_end():
    """send() with reply_to (AGENT_END path) must pop the turn entry."""
    adapter, _ = _make_adapter()
    session_key = _build_session_key("chan-1", "conv-1")
    assert session_key in adapter._turn_by_session

    asyncio.run(adapter.send(**_make_send_result_args()))

    assert session_key not in adapter._turn_by_session


def test_turn_retained_for_partial_response():
    """send() without reply_to (RESPONSE partial) must NOT remove the turn."""
    adapter, _ = _make_adapter()
    session_key = _build_session_key("chan-1", "conv-1")

    asyncio.run(adapter.send(chat_id="conv-1", content="thinking…", reply_to=None))

    assert session_key in adapter._turn_by_session


def test_fail_pending_clears_turn_by_session():
    """_fail_pending (disconnect) must wipe _turn_by_session."""
    adapter, _ = _make_adapter()
    assert adapter._turn_by_session  # has the injected entry

    adapter._fail_pending("disconnect")

    assert adapter._turn_by_session == {}


def test_second_conversation_after_first_ends():
    """After conv-A ends, starting conv-B must work without stale interference."""
    adapter, _mock_client = _make_adapter()  # starts with conv-1 active

    # End conv-1
    asyncio.run(adapter.send(**_make_send_result_args("conv-1")))
    assert _build_session_key("chan-1", "conv-1") not in adapter._turn_by_session

    # Start conv-2
    agent_start_b = BaseEvent(
        event_type=ExecutionEventType.AGENT_START,
        author="u2",
        conversation_id="conv-2",
        interaction_id="int-2",
        turn_id="turn-2",
        action=AgentAction(name="invoke", parameters={"input": "hi"}),
    )
    asyncio.run(adapter._handle_agent_start(agent_start_b))

    # Only conv-2 must be active
    assert _build_session_key("chan-1", "conv-2") in adapter._turn_by_session
    assert _build_session_key("chan-1", "conv-1") not in adapter._turn_by_session
