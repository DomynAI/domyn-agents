"""Tests for AgentServer dispatch logic.

Covered:
1. AGENT_START is passed to adapter.invoke() with emit/emit_priority.
2. TOOL_END is routed to adapter.on_tool_response().
3. TOOL_ERROR is routed to adapter.on_tool_response().
4. AGENT_END (cancel) is routed to adapter.on_cancel().
5. Auth rejection suppresses invoke.
6. Unknown event types are ignored.
"""

import asyncio
import uuid
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from domyn.agents.core import BaseEvent, ExecutionEventType, ToolAction
from domyn.agents.server import AgentServer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _event(event_type: ExecutionEventType, conversation_id: str = "conv-1") -> BaseEvent:
    return BaseEvent(
        event_type=event_type,
        author="platform",
        event_id=str(uuid.uuid4()),
        conversation_id=conversation_id,
    )


class _MockChannel:
    """Minimal InputChannel that yields a preset list of events then stops."""

    def __init__(self, events: list[BaseEvent]) -> None:
        self._events = events
        self.sent: list[BaseEvent] = []
        self.priority_sent: list[BaseEvent] = []

    async def send_event(self, conv_id: str, event: BaseEvent) -> None:
        self.sent.append(event)

    async def send_event_priority(self, conv_id: str, event: BaseEvent) -> None:
        self.priority_sent.append(event)

    async def receive(self) -> AsyncGenerator[tuple[str, BaseEvent], None]:  # type: ignore[override]
        for ev in self._events:
            yield ev.conversation_id or "conv-1", ev

    async def close(self) -> None:
        pass


class _MockAdapter:
    def __init__(self) -> None:
        self.invoked: list[BaseEvent] = []
        self.tool_responses: list[BaseEvent] = []
        self.cancelled: list[BaseEvent] = []
        self.closed = False

    async def invoke(self, event: BaseEvent, emit: Any, emit_priority: Any) -> None:
        self.invoked.append(event)

    def on_tool_response(self, event: BaseEvent) -> None:
        self.tool_responses.append(event)

    def on_cancel(self, event: BaseEvent) -> None:
        self.cancelled.append(event)

    async def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_start_dispatched_to_invoke() -> None:
    channel = _MockChannel([_event(ExecutionEventType.AGENT_START)])
    adapter = _MockAdapter()
    server = AgentServer(channel=channel, adapter=adapter)  # type: ignore[arg-type]
    await server.serve()
    await asyncio.sleep(0.05)  # let the fire-and-forget task run
    assert len(adapter.invoked) == 1
    assert adapter.closed


@pytest.mark.asyncio
async def test_tool_end_dispatched_to_on_tool_response() -> None:
    ev = _event(ExecutionEventType.TOOL_END)
    ev = ev.model_copy(
        update={"action": ToolAction(name="t", call_id="cid", parameters={}, observation="obs")}
    )
    channel = _MockChannel([ev])
    adapter = _MockAdapter()
    server = AgentServer(channel=channel, adapter=adapter)  # type: ignore[arg-type]
    await server.serve()
    assert len(adapter.tool_responses) == 1
    assert adapter.tool_responses[0].event_type == ExecutionEventType.TOOL_END


@pytest.mark.asyncio
async def test_tool_error_dispatched_to_on_tool_response() -> None:
    ev = _event(ExecutionEventType.TOOL_ERROR)
    channel = _MockChannel([ev])
    adapter = _MockAdapter()
    server = AgentServer(channel=channel, adapter=adapter)  # type: ignore[arg-type]
    await server.serve()
    assert len(adapter.tool_responses) == 1
    assert adapter.tool_responses[0].event_type == ExecutionEventType.TOOL_ERROR


@pytest.mark.asyncio
async def test_agent_end_dispatched_to_on_cancel() -> None:
    ev = _event(ExecutionEventType.AGENT_END)
    channel = _MockChannel([ev])
    adapter = _MockAdapter()
    server = AgentServer(channel=channel, adapter=adapter)  # type: ignore[arg-type]
    await server.serve()
    assert len(adapter.cancelled) == 1


@pytest.mark.asyncio
async def test_auth_rejection_suppresses_invoke() -> None:
    class _RejectAll:
        async def authenticate(self, _: Any) -> bool:
            return False

    channel = _MockChannel([_event(ExecutionEventType.AGENT_START)])
    adapter = _MockAdapter()
    server = AgentServer(channel=channel, adapter=adapter, auth=_RejectAll())  # type: ignore[arg-type]
    await server.serve()
    assert adapter.invoked == []


@pytest.mark.asyncio
async def test_auth_allows_invoke() -> None:
    class _AllowAll:
        async def authenticate(self, _: Any) -> bool:
            return True

    channel = _MockChannel([_event(ExecutionEventType.AGENT_START)])
    adapter = _MockAdapter()
    server = AgentServer(channel=channel, adapter=adapter, auth=_AllowAll())  # type: ignore[arg-type]
    await server.serve()
    await asyncio.sleep(0.05)
    assert len(adapter.invoked) == 1


@pytest.mark.asyncio
async def test_unknown_event_type_ignored() -> None:
    ev = _event(ExecutionEventType.RESPONSE)
    channel = _MockChannel([ev])
    adapter = _MockAdapter()
    server = AgentServer(channel=channel, adapter=adapter)  # type: ignore[arg-type]
    await server.serve()
    assert adapter.invoked == []
    assert adapter.tool_responses == []
    assert adapter.cancelled == []


# ---------------------------------------------------------------------------
# TOOL_FEEDBACK — resuming after a human approval
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_feedback_starts_a_turn() -> None:
    """A human approval must re-enter the runner through invoke().

    The runner returns as soon as it suspends on TOOL_NEEDS_FEEDBACK, so there is
    no live turn to hand the answer to. This used to be dropped, which made the
    approval gate unusable on every channel: the agent asked, the human answered,
    and nothing happened.
    """
    channel = _MockChannel([_event(ExecutionEventType.TOOL_FEEDBACK)])
    adapter = _MockAdapter()
    server = AgentServer(channel=channel, adapter=adapter)  # type: ignore[arg-type]
    await server.serve()
    await asyncio.sleep(0.05)
    assert len(adapter.invoked) == 1
    assert adapter.invoked[0].event_type == ExecutionEventType.TOOL_FEEDBACK


@pytest.mark.asyncio
async def test_tool_feedback_is_authenticated() -> None:
    """An unauthenticated approval must be refused.

    Approval decides whether a gated tool runs; accepting one without auth would
    let a third party authorise actions on the user's behalf.
    """
    auth = MagicMock()
    auth.authenticate = AsyncMock(return_value=False)
    channel = _MockChannel([_event(ExecutionEventType.TOOL_FEEDBACK)])
    adapter = _MockAdapter()
    server = AgentServer(channel=channel, adapter=adapter, auth=auth)  # type: ignore[arg-type]
    await server.serve()
    await asyncio.sleep(0.05)
    assert adapter.invoked == []
    auth.authenticate.assert_awaited_once()


@pytest.mark.asyncio
async def test_tool_feedback_passes_when_authenticated() -> None:
    auth = MagicMock()
    auth.authenticate = AsyncMock(return_value=True)
    channel = _MockChannel([_event(ExecutionEventType.TOOL_FEEDBACK)])
    adapter = _MockAdapter()
    server = AgentServer(channel=channel, adapter=adapter, auth=auth)  # type: ignore[arg-type]
    await server.serve()
    await asyncio.sleep(0.05)
    assert len(adapter.invoked) == 1


@pytest.mark.asyncio
async def test_auth_exception_rejects_instead_of_crashing() -> None:
    """A broken authenticator must fail closed, not take the server down."""
    auth = MagicMock()
    auth.authenticate = AsyncMock(side_effect=RuntimeError("auth backend down"))
    channel = _MockChannel([_event(ExecutionEventType.TOOL_FEEDBACK)])
    adapter = _MockAdapter()
    server = AgentServer(channel=channel, adapter=adapter, auth=auth)  # type: ignore[arg-type]
    await server.serve()
    await asyncio.sleep(0.05)
    assert adapter.invoked == []


@pytest.mark.asyncio
async def test_full_approval_round_trip_through_the_server() -> None:
    """AGENT_START then TOOL_FEEDBACK: two invocations, in order."""
    start = _event(ExecutionEventType.AGENT_START)
    feedback = _event(ExecutionEventType.TOOL_FEEDBACK)
    channel = _MockChannel([start, feedback])
    adapter = _MockAdapter()
    server = AgentServer(channel=channel, adapter=adapter)  # type: ignore[arg-type]
    await server.serve()
    await asyncio.sleep(0.05)
    assert [e.event_type for e in adapter.invoked] == [
        ExecutionEventType.AGENT_START,
        ExecutionEventType.TOOL_FEEDBACK,
    ]


@pytest.mark.asyncio
async def test_normalizer_failure_drops_only_that_event() -> None:
    normalizer = MagicMock()
    normalizer.normalize = MagicMock(side_effect=ValueError("bad payload"))
    channel = _MockChannel([_event(ExecutionEventType.TOOL_FEEDBACK)])
    adapter = _MockAdapter()
    server = AgentServer(channel=channel, adapter=adapter, normalizer=normalizer)  # type: ignore[arg-type]
    await server.serve()
    await asyncio.sleep(0.05)
    assert adapter.invoked == []
    assert adapter.closed
