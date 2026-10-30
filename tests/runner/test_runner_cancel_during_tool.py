"""Tests that a cancellation reaches the delegate while a local tool is running.

``DelegateAgent.execute`` signals the remote runtime from its own ``finally``, so it
only fires if the generator is finalized. While the runner executes a tool for the
delegate it has stopped reading that generator (it pauses the transport first), so a
cancellation landing there unwinds the runner's frame with the generator suspended at
its last yield: without an explicit hand-off it is abandoned, its ``finally`` deferred
to garbage collection, and the remote keeps waiting for a result nobody will send.

The agent-level half of the contract is pinned in ``test_delegate_agent_cancellation.py``.
"""

import asyncio
import uuid
from collections.abc import AsyncGenerator

import pytest

from domyn.agents import Runner
from domyn.agents.agents import DelegateAgent
from domyn.agents.core import BaseEvent, ExecutionEventType, InteractionContext, Part, ToolAction
from domyn.agents.runner import RunnerTransport
from domyn.agents.tools import FunctionTool


class _RecordingTransport(RunnerTransport):
    """Replays a fixed sequence from the remote and records cancel() calls."""

    def __init__(self, events: list[BaseEvent]) -> None:
        self._events = events
        self.cancel_called_with: list[str] = []
        self.paused = 0

    async def execute(
        self, agent_name: str, context: InteractionContext
    ) -> AsyncGenerator[BaseEvent, None]:
        for event in self._events:
            yield event
        await asyncio.sleep(9999)  # the remote stays live until told otherwise

    async def send_result(self, event: BaseEvent, conversation_id: str | None = None) -> None:
        pass

    async def pause_receiving(self) -> None:
        self.paused += 1

    async def cancel(self, conversation_id: str) -> None:
        self.cancel_called_with.append(conversation_id)


def _tool_start(name: str) -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.TOOL_START,
        author="ext",
        action=ToolAction(name=name, parameters={"query": "cats"}, call_id=str(uuid.uuid4())),
    )


def _response() -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.RESPONSE,
        author="ext",
        content=[Part(text="done")],
        is_partial=False,
    )


def _make_agent(
    events: list[BaseEvent], tool_entered: asyncio.Event | None = None
) -> tuple[DelegateAgent, _RecordingTransport]:
    async def web_search(query: str) -> str:
        """Search the web."""
        if tool_entered is not None:
            tool_entered.set()
            await asyncio.sleep(9999)  # hangs until cancelled
        return f"result for {query}"

    transport = _RecordingTransport(events)
    agent = DelegateAgent(
        name="RemoteRoot",
        description="root delegate",
        transport=transport,
        tools=[FunctionTool(func=web_search)],
    )
    return agent, transport


def _run(agent: DelegateAgent, conversation_id: str) -> AsyncGenerator[BaseEvent, None]:
    return Runner().run(
        application_name="app",
        user_id="u1",
        root=agent,
        user_input=BaseEvent(
            event_type=ExecutionEventType.USER_INPUT,
            author="user",
            content=[Part(text="search for me")],
        ),
        conversation_id=conversation_id,
    )


async def _wait_for(predicate, timeout: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate() and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_cancel_reaches_the_delegate_while_a_tool_is_running():
    """The remote must be told, even though the runner was inside the tool call."""
    tool_entered = asyncio.Event()
    agent, transport = _make_agent([_tool_start("web_search")], tool_entered)
    conv_id = "conv-cancel-in-tool"

    async def _consume() -> None:
        async for _ in _run(agent, conv_id):
            pass

    task = asyncio.create_task(_consume())
    await asyncio.wait_for(tool_entered.wait(), timeout=5)
    assert transport.paused == 1, "the runner is expected to stop reading the delegate here"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await _wait_for(lambda: transport.cancel_called_with)
    assert transport.cancel_called_with == [conv_id]


@pytest.mark.asyncio
async def test_no_cancel_signal_on_a_normal_run():
    """Closing the delegate on the way out must not read as a cancellation."""
    agent, transport = _make_agent([_tool_start("web_search"), _response()])

    events = [event async for event in _run(agent, "conv-normal")]

    assert ExecutionEventType.TOOL_END in [event.event_type for event in events]
    await _wait_for(lambda: transport.cancel_called_with, timeout=0.1)
    assert transport.cancel_called_with == []
