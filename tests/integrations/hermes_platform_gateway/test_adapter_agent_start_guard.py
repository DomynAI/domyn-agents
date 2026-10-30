"""Tests for the re-delivered-AGENT_START guard in ``_handle_agent_start``.

Dispatching the same AGENT_START twice runs the whole turn a second time and emits
every one of its events again. ``DomynRelayClient._seen`` already drops replays
inside the relay's retention window; this guard is the backstop for anything outside
it.

The guard keys on ``event_id`` only — NOT on ``turn_id`` and not on "a turn is
already live" — because a clarify answer and a queued follow-up message are both
legitimate second AGENT_STARTs for the same session and must still be dispatched.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from domyn.agents.core import AgentAction, BaseEvent, ExecutionEventType, Part
from domyn.agents.integrations.hermes_platform_gateway.adapter import (
    DomynPlatformAdapter,
    _build_session_key,
)


def _make_adapter(channel_id: str = "chan-1"):
    mock_client = MagicMock()
    mock_client.send_event = AsyncMock()
    adapter = DomynPlatformAdapter(
        config=MagicMock(),
        channel_id=channel_id,
        relay_client_factory=lambda _cb: mock_client,
        tool_rename_mapping={},
    )
    adapter._client = mock_client

    dispatched: list[object] = []

    async def _capture(msg):
        dispatched.append(msg)

    adapter.handle_message = _capture
    return adapter, dispatched


def _agent_start(event_id: str, *, conv_id: str = "conv-1", text: str = "hello") -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.AGENT_START,
        author="u1",
        event_id=event_id,
        conversation_id=conv_id,
        interaction_id="int-1",
        turn_id="turn-1",
        content=[Part(text=text)],
        action=AgentAction(name="invoke", parameters={"input": text}),
    )


def test_redelivered_agent_start_is_not_dispatched_twice():
    adapter, dispatched = _make_adapter()
    event = _agent_start("event_abc")

    asyncio.run(adapter._handle_agent_start(event))
    asyncio.run(adapter._handle_agent_start(_agent_start("event_abc")))

    assert len(dispatched) == 1


def test_second_agent_start_with_a_new_id_is_dispatched():
    """A clarify answer / queued follow-up is a legitimate second AGENT_START."""
    adapter, dispatched = _make_adapter()

    asyncio.run(adapter._handle_agent_start(_agent_start("event_1")))
    asyncio.run(adapter._handle_agent_start(_agent_start("event_2", text="and now this")))

    assert len(dispatched) == 2
    turn = adapter._turn_by_session[_build_session_key("chan-1", "conv-1")]
    assert turn.event_id == "event_2"


def test_redelivery_does_not_consume_the_pending_cancel_note():
    """The guard runs before the one-shot _CANCEL_NOTE so a dropped replay can't eat it."""
    adapter, dispatched = _make_adapter()
    event = _agent_start("event_abc")
    asyncio.run(adapter._handle_agent_start(event))
    dispatched.clear()

    # Simulate a cancel having armed the note, then a replay of the live AGENT_START.
    adapter._cancelled_conversations.add("conv-1")
    asyncio.run(adapter._handle_agent_start(_agent_start("event_abc")))

    assert dispatched == []
    assert "conv-1" in adapter._cancelled_conversations

    # The next real question still gets the note prepended.
    asyncio.run(adapter._handle_agent_start(_agent_start("event_next", text="next question")))
    assert len(dispatched) == 1
    assert adapter._CANCEL_NOTE in dispatched[0].text
    assert "conv-1" not in adapter._cancelled_conversations


def test_redelivery_after_the_turn_closed_is_dispatched():
    """Once the turn is popped (final send) there is no live turn to match against."""
    adapter, dispatched = _make_adapter()
    asyncio.run(adapter._handle_agent_start(_agent_start("event_abc")))
    adapter._turn_by_session.pop(_build_session_key("chan-1", "conv-1"))
    dispatched.clear()

    asyncio.run(adapter._handle_agent_start(_agent_start("event_abc")))

    assert len(dispatched) == 1
