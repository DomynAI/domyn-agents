"""A DelegateAgent over WebSocket whose external agent calls a tool this runner holds.

``WebSocketTransport.send_result`` queues the result per conversation and refuses one
without a ``conversation_id``. Every other transport in the tests ignores the argument,
so the runner calling it without one went unnoticed. These tests keep the real
``send_result`` and replace only how events arrive.
"""

from collections.abc import AsyncGenerator
from typing import Any

import pytest

from domyn.agents import Runner
from domyn.agents.agents import DelegateAgent
from domyn.agents.approvals import ApprovalPolicy
from domyn.agents.core import (
    ApprovalType,
    BaseEvent,
    ExecutionEventType,
    FeedbackAction,
    FeedbackParameters,
    InteractionContext,
    Part,
    ToolAction,
)
from domyn.agents.runner.websocket_transport import WebSocketTransport
from domyn.agents.tools import FunctionTool


class _ScriptedWebSocketTransport(WebSocketTransport):
    """Replays the remote's events instead of connecting; ``send_result`` is the real one."""

    # Inherited, the tag would register this class as the `websocket` transport for every test.
    type = None

    def __init__(self, *turns: list[BaseEvent]) -> None:
        super().__init__(ws_url="ws://unused")
        self._turns = list(turns)

    async def execute(
        self, agent_name: str, context: InteractionContext, run_config: dict[str, Any] | None = None
    ) -> AsyncGenerator[BaseEvent, None]:
        for event in self._turns.pop(0):
            yield event


def _tool_start() -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.TOOL_START,
        author="Remote",
        action=ToolAction(name="add_numbers", parameters={"a": 3, "b": 4}, call_id="c1"),
    )


def _answer() -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.RESPONSE,
        author="Remote",
        content=[Part(text="7")],
        is_partial=False,
    )


def _agent(
    transport: WebSocketTransport, approval_type: ApprovalType | None = None
) -> DelegateAgent:
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    return DelegateAgent(
        name="Remote",
        description="external agent",
        transport=transport,
        tools=[FunctionTool(func=add_numbers, approval_type=approval_type)],
    )


async def _run(runner: Runner, agent: DelegateAgent, user_input: BaseEvent) -> list[BaseEvent]:
    return [
        event
        async for event in runner.run(
            application_name="app",
            user_id="u1",
            root=agent,
            user_input=user_input,
            conversation_id="conv-ws",
        )
    ]


def _user_input() -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text="3 + 4?")]
    )


async def _queued(transport: WebSocketTransport) -> BaseEvent:
    return await transport._pending_queues["conv-ws"].get()


@pytest.mark.asyncio
async def test_the_result_of_a_tool_the_runner_ran_is_queued_for_the_conversation() -> None:
    transport = _ScriptedWebSocketTransport([_tool_start(), _answer()])

    events = await _run(Runner(), _agent(transport), _user_input())

    assert ExecutionEventType.ERROR_OCCURRED not in [e.event_type for e in events]
    assert events[-1].event_type == ExecutionEventType.AGENT_END
    result = await _queued(transport)
    assert result.event_type == ExecutionEventType.TOOL_END
    assert result.action.observation == 7
    assert result.conversation_id == "conv-ws"


@pytest.mark.asyncio
async def test_a_call_the_policy_refuses_sends_the_refusal_back() -> None:
    transport = _ScriptedWebSocketTransport([_tool_start(), _answer()])
    runner = Runner(approval_policy=ApprovalPolicy(deny_rules=["*"]))

    await _run(runner, _agent(transport, ApprovalType.USER_CONFIRM), _user_input())

    refusal = await _queued(transport)
    assert refusal.event_type == ExecutionEventType.TOOL_END
    assert refusal.action.call_id == "c1"


@pytest.mark.asyncio
async def test_a_call_approved_by_the_user_sends_its_result_back() -> None:
    transport = _ScriptedWebSocketTransport([_tool_start()], [_answer()])
    runner = Runner()
    agent = _agent(transport, ApprovalType.USER_CONFIRM)

    paused = await _run(runner, agent, _user_input())
    assert paused[-1].event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK

    approval = BaseEvent(
        event_type=ExecutionEventType.TOOL_FEEDBACK,
        author="user",
        action=FeedbackAction(
            name="add_numbers",
            parameters=[
                FeedbackParameters(name="a", value=3, type="int"),
                FeedbackParameters(name="b", value=4, type="int"),
            ],
            approval_type=ApprovalType.USER_CONFIRM,
            approve=True,
        ),
    )
    resumed = await _run(runner, agent, approval)

    assert resumed[-1].event_type == ExecutionEventType.AGENT_END
    result = await _queued(transport)
    assert result.event_type == ExecutionEventType.TOOL_END
    assert result.action.observation == 7
