"""Tests for authoritative-send routing.

A background-delegation completion (and any proactive/internal final response)
re-enters as a synthetic message with no message_id, so it reaches send() with
reply_to=None. The base adapter routes it through _send_with_retry, which flags
the session as authoritative; send() must then emit a real AGENT_END rather than
suppressing it as a hermes interim narrative.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from domyn.agents.core import AgentAction, BaseEvent, ExecutionEventType
from domyn.agents.integrations.hermes_platform_gateway.adapter import (
    DomynPlatformAdapter,
    _build_session_key,
)


def _make_adapter(channel_id: str = "chan-1", conv_ids: tuple[str, ...] = ("conv-1",)):
    mock_client = MagicMock()
    mock_client.send_event = AsyncMock()

    adapter = DomynPlatformAdapter(
        config=MagicMock(),
        channel_id=channel_id,
        relay_client_factory=lambda _cb: mock_client,
        tool_rename_mapping={},
    )
    for conv_id in conv_ids:
        turn = BaseEvent(
            event_type=ExecutionEventType.AGENT_START,
            author="u1",
            conversation_id=conv_id,
            interaction_id=f"int-{conv_id}",
            turn_id=f"turn-{conv_id}",
            action=AgentAction(name="invoke", parameters={"input": "hi"}),
        )
        adapter._turn_by_session[_build_session_key(channel_id, conv_id)] = turn
    adapter._client = mock_client
    return adapter, mock_client


# ---------------------------------------------------------------------------
# send() with reply_to=None
# ---------------------------------------------------------------------------


def test_authoritative_reply_to_none_emits_agent_end():
    adapter, mock_client = _make_adapter()
    session_key = _build_session_key("chan-1", "conv-1")
    adapter._authoritative_send_sessions.add(session_key)

    result = asyncio.run(adapter.send(chat_id="conv-1", content="Subagent result.", reply_to=None))

    mock_client.send_event.assert_awaited_once()
    event: BaseEvent = mock_client.send_event.await_args.args[0]
    assert event.event_type == ExecutionEventType.AGENT_END
    assert event.is_partial is False
    assert event.content[0].text == "Subagent result."
    assert result.message_id != "interim-noop"
    assert session_key not in adapter._turn_by_session  # terminal → popped


def test_non_authoritative_reply_to_none_is_interim_noop():
    adapter, mock_client = _make_adapter()

    result = asyncio.run(adapter.send(chat_id="conv-1", content="thinking…", reply_to=None))

    assert result.message_id == "interim-noop"
    mock_client.send_event.assert_not_awaited()
    # Turn is preserved (interim narrative doesn't end the turn).
    assert _build_session_key("chan-1", "conv-1") in adapter._turn_by_session


def test_authoritative_flag_does_not_leak_across_sessions():
    adapter, mock_client = _make_adapter(conv_ids=("conv-1", "conv-2"))
    # Only conv-1 is mid-authoritative-send.
    adapter._authoritative_send_sessions.add(_build_session_key("chan-1", "conv-1"))

    # conv-2 reply_to=None must still be treated as an interim narrative.
    result = asyncio.run(adapter.send(chat_id="conv-2", content="mid…", reply_to=None))

    assert result.message_id == "interim-noop"
    mock_client.send_event.assert_not_awaited()


# ---------------------------------------------------------------------------
# _send_with_retry override
# ---------------------------------------------------------------------------


def test_send_with_retry_flags_session_and_emits_agent_end():
    """_send_with_retry marks the session authoritative for the duration, so a
    reply_to=None final response is emitted as AGENT_END, and the flag is
    cleared afterwards."""
    adapter, mock_client = _make_adapter()
    session_key = _build_session_key("chan-1", "conv-1")

    asyncio.run(adapter._send_with_retry(chat_id="conv-1", content="done", reply_to=None))

    event: BaseEvent = mock_client.send_event.await_args.args[0]
    assert event.event_type == ExecutionEventType.AGENT_END
    assert event.is_partial is False
    # Flag cleared after the send completes.
    assert session_key not in adapter._authoritative_send_sessions


def test_fail_pending_clears_authoritative_sessions():
    adapter, _ = _make_adapter()
    adapter._authoritative_send_sessions.add(_build_session_key("chan-1", "conv-1"))

    adapter._fail_pending("disconnect")

    assert adapter._authoritative_send_sessions == set()


# ---------------------------------------------------------------------------
# Retry idempotency
#
# The base adapter retries a retryable delivery by calling send() again. Each call
# builds a fresh BaseEvent, so without a pinned id a write that reached the relay
# before failing would land twice with two different event_ids — same turn, so
# neither the transport's event_id dedup nor its turn filter can collapse it.
# ---------------------------------------------------------------------------


def test_retry_of_one_logical_send_reuses_the_same_event_id():
    adapter, mock_client = _make_adapter()
    session_key = _build_session_key("chan-1", "conv-1")

    # First write fails as retryable (as if the socket died mid-send); second succeeds.
    mock_client.send_event = AsyncMock(side_effect=[RuntimeError("socket gone"), None])

    result = asyncio.run(adapter._send_with_retry(chat_id="conv-1", content="done", reply_to=None))

    assert result.success is True
    assert mock_client.send_event.await_count == 2
    first, second = (call.args[0] for call in mock_client.send_event.await_args_list)
    assert first.event_id == second.event_id
    # The pin is released once the logical send finishes.
    assert session_key not in adapter._send_event_id_by_session


def _reseed_turn(adapter, channel_id: str = "chan-1", conv_id: str = "conv-1") -> None:
    """Restore the session's turn — a terminal send pops it (see send())."""
    adapter._turn_by_session[_build_session_key(channel_id, conv_id)] = BaseEvent(
        event_type=ExecutionEventType.AGENT_START,
        author="u1",
        conversation_id=conv_id,
        interaction_id=f"int-{conv_id}",
        turn_id=f"turn-{conv_id}",
        action=AgentAction(name="invoke", parameters={"input": "hi"}),
    )


def test_separate_logical_sends_get_distinct_event_ids():
    adapter, mock_client = _make_adapter()

    asyncio.run(adapter._send_with_retry(chat_id="conv-1", content="one", reply_to="m1"))
    _reseed_turn(adapter)
    asyncio.run(adapter._send_with_retry(chat_id="conv-1", content="two", reply_to="m2"))

    first, second = (call.args[0] for call in mock_client.send_event.await_args_list)
    assert first.event_id != second.event_id


def test_interim_send_outside_retry_gets_a_fresh_event_id():
    """A direct send() (not routed through _send_with_retry) is unaffected by the pin."""
    adapter, mock_client = _make_adapter()
    adapter._authoritative_send_sessions.add(_build_session_key("chan-1", "conv-1"))

    asyncio.run(adapter.send(chat_id="conv-1", content="a", reply_to=None))
    _reseed_turn(adapter)
    asyncio.run(adapter.send(chat_id="conv-1", content="b", reply_to=None))

    first, second = (call.args[0] for call in mock_client.send_event.await_args_list)
    assert first.event_id != second.event_id
