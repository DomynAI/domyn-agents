"""Tests for how the runner treats a TOOL_START coming back from a DelegateAgent.

A delegated runtime (the hermes sandbox) sends TOOL_START for two very different reasons: to ask
the local runner to run a tool it doesn't have, and to *report* that it already ran one of its own
built-ins. The two events are built identically, so the reporting one carries
``METADATA_VISIBILITY_ONLY``. Getting this wrong is expensive and silent: the local agent holds
tools of its own, and a name shared with a sandbox built-in would make the runner execute a call
nobody made — or stall the turn on an approval for a call that doesn't exist.

The producing half of the contract is pinned in
``tests/integrations/hermes_platform_gateway/test_adapter_visibility_events.py``.
"""

import uuid
from collections.abc import AsyncGenerator

import pytest

from domyn.agents import Runner
from domyn.agents.agents import DelegateAgent
from domyn.agents.core import (
    METADATA_VISIBILITY_ONLY,
    BaseEvent,
    ExecutionEventType,
    InteractionContext,
    Part,
    ToolAction,
)
from domyn.agents.runner import RunnerTransport
from domyn.agents.tools import FunctionTool


class _MockTransport(RunnerTransport):
    """In-process transport that replays a fixed sequence of events from the remote."""

    def __init__(self, events: list[BaseEvent]) -> None:
        self._events = events

    async def execute(
        self, agent_name: str, context: InteractionContext
    ) -> AsyncGenerator[BaseEvent, None]:
        for event in self._events:
            yield event

    async def send_result(self, event: BaseEvent, conversation_id: str | None = None) -> None:
        pass


def _make_agent_with_recorded_tool(events: list[BaseEvent]) -> tuple[DelegateAgent, list[str]]:
    """A DelegateAgent holding a local `web_search` tool that records its invocations."""
    calls: list[str] = []

    def web_search(query: str) -> str:
        """Search the web."""
        calls.append(query)
        return f"result for {query}"

    agent = DelegateAgent(
        name="RemoteRoot",
        description="root delegate",
        transport=_MockTransport(events),
        tools=[FunctionTool(func=web_search)],
    )
    return agent, calls


def _tool_start(call_id: str, metadata: dict | None = None) -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.TOOL_START,
        author="ext",
        metadata=metadata or {},
        action=ToolAction(name="web_search", parameters={"query": "cats"}, call_id=call_id),
    )


def _response() -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.RESPONSE,
        author="ext",
        content=[Part(text="done")],
        is_partial=False,
    )


async def _run(agent: DelegateAgent, conversation_id: str) -> list[BaseEvent]:
    return [
        event
        async for event in Runner().run(
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
    ]


@pytest.mark.asyncio
async def test_visibility_flag_forwards_without_executing():
    """The flag alone must stop execution — the call_id here is an ordinary uuid."""
    agent, calls = _make_agent_with_recorded_tool(
        [_tool_start(str(uuid.uuid4()), {METADATA_VISIBILITY_ONLY: True}), _response()]
    )

    events = await _run(agent, "conv-flag")

    assert calls == [], "the local tool ran for an event that only reported a remote execution"
    # Still forwarded: the event is the sandbox's own tool call, and consumers show it.
    assert ExecutionEventType.TOOL_START in [event.event_type for event in events]


@pytest.mark.asyncio
async def test_legacy_visibility_call_id_forwards_without_executing():
    """Older delegated runtimes stamp no flag and only tag the call_id.

    The prefix is the compatibility path for them; the metadata flag is what a
    current runtime sends.
    """
    agent, calls = _make_agent_with_recorded_tool(
        [_tool_start(f"visibility-{uuid.uuid4()}"), _response()]
    )

    events = await _run(agent, "conv-legacy")

    assert calls == []
    assert ExecutionEventType.TOOL_START in [event.event_type for event in events]


@pytest.mark.asyncio
async def test_unmarked_tool_start_is_executed_locally():
    """The counterpart: without either signal the runner must still run the tool.

    Without this the two tests above would also pass under a guard that never executes anything.
    """
    agent, calls = _make_agent_with_recorded_tool([_tool_start(str(uuid.uuid4())), _response()])

    events = await _run(agent, "conv-real")

    assert calls == ["cats"]
    assert ExecutionEventType.TOOL_END in [event.event_type for event in events]
