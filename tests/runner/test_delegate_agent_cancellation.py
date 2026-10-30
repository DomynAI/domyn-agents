"""Tests for DelegateAgent cancellation behaviour.

Verifies that when CancelledError propagates through DelegateAgent.execute(),
the transport's cancel() method is called with the correct conversation_id
before the error is re-raised.
"""

import asyncio
from collections.abc import AsyncGenerator

import pytest

from domyn.agents import BaseEvent, ExecutionEventType, Part
from domyn.agents.agents import DelegateAgent
from domyn.agents.core import InteractionContext
from domyn.agents.runner import RunnerTransport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user_input(text: str = "hello") -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text=text)],
    )


def _make_context(conversation_id: str = "conv-cancel-test") -> InteractionContext:
    return InteractionContext(
        conversation_id=conversation_id,
        user_id="user-1",
        agent_input=_user_input(),
    )


class _TrackingTransport(RunnerTransport):
    """Transport that yields a fixed sequence of events and tracks cancel() calls."""

    def __init__(self, events: list[BaseEvent]) -> None:
        self._events = events
        self.cancel_called_with: list[str] = []

    async def execute(
        self, agent_name: str, context: InteractionContext
    ) -> AsyncGenerator[BaseEvent, None]:
        for event in self._events:
            yield event

    async def send_result(self, event: BaseEvent, conversation_id: str | None = None) -> None:
        pass

    async def cancel(self, conversation_id: str) -> None:
        self.cancel_called_with.append(conversation_id)


class _HangingTransport(RunnerTransport):
    """Transport that suspends indefinitely, allowing CancelledError injection."""

    def __init__(self) -> None:
        self.cancel_called_with: list[str] = []

    async def execute(
        self, agent_name: str, context: InteractionContext
    ) -> AsyncGenerator[BaseEvent, None]:
        await asyncio.sleep(9999)  # hangs until cancelled
        yield BaseEvent(  # never reached
            event_type=ExecutionEventType.RESPONSE,
            author=agent_name,
            content=[Part(text="unreachable")],
        )

    async def send_result(self, event: BaseEvent, conversation_id: str | None = None) -> None:
        pass

    async def cancel(self, conversation_id: str) -> None:
        self.cancel_called_with.append(conversation_id)


def _make_agent(transport: RunnerTransport) -> DelegateAgent:
    return DelegateAgent(name="Remote", description="remote delegate", transport=transport)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_not_called_on_normal_completion():
    """transport.cancel() must NOT be called when the agent finishes normally."""
    transport = _TrackingTransport(
        [
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="Remote",
                content=[Part(text="done")],
                is_partial=False,
            )
        ]
    )
    agent = _make_agent(transport)
    context = _make_context()

    events = [ev async for ev, _ in agent.execute(context)]

    assert len(events) >= 1
    assert transport.cancel_called_with == []


@pytest.mark.asyncio
async def test_cancel_called_with_conversation_id_on_cancellation():
    """transport.cancel() is called with the right conversation_id when CancelledError fires."""
    transport = _HangingTransport()
    agent = _make_agent(transport)
    conv_id = "conv-abc-123"
    context = _make_context(conversation_id=conv_id)

    async def _run():
        async for _ in agent.execute(context):
            pass

    task = asyncio.create_task(_run())
    await asyncio.sleep(0)  # let the task start and reach the sleep
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    # Give the fire-and-forget cancel task a chance to run
    await asyncio.sleep(0)

    assert transport.cancel_called_with == [conv_id]


@pytest.mark.asyncio
async def test_cancelled_error_is_reraised():
    """CancelledError must propagate out of execute() after cancel() is called."""
    transport = _HangingTransport()
    agent = _make_agent(transport)
    context = _make_context()

    async def _run():
        async for _ in agent.execute(context):
            pass

    task = asyncio.create_task(_run())
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_cancel_called_once_even_if_transport_yields_nothing():
    """cancel() fires exactly once regardless of how many events were yielded."""
    transport = _HangingTransport()
    agent = _make_agent(transport)
    context = _make_context(conversation_id="conv-zero-events")

    task = asyncio.create_task(
        asyncio.wait_for(
            _collect(agent.execute(context)),
            timeout=0.05,
        )
    )

    with pytest.raises((asyncio.TimeoutError, asyncio.CancelledError)):
        await task

    await asyncio.sleep(0)

    assert len(transport.cancel_called_with) == 1
    assert transport.cancel_called_with[0] == "conv-zero-events"


async def _collect(gen) -> list:
    return [ev async for ev, _ in gen]
