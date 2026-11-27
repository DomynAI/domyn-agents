"""A turn started by a channel's AGENT_START (Telegram, Kafka…), which names no agent.

Resumed after an approval, the model must read the task, its call and the call's result,
in that order: the start used to be hidden from the history (its author is no agent),
then read back after the result, and the runner failed on its missing action.
"""

from collections.abc import AsyncGenerator
from typing import Any

import pytest

from domyn.agents import Runner
from domyn.agents.agents import Agent
from domyn.agents.core import ApprovalType, BaseEvent, ExecutionEventType, Part
from domyn.agents.llm import BaseLLMProvider, LLMResponse
from domyn.agents.planner_strategy import ToolUseWithThoughtPlannerStrategy
from domyn.agents.tools import DelegateTool


class _BlockThenAnswer(BaseLLMProvider):
    type: str = "mock"
    name: str = "mock"
    model_name: str = "mock"
    supported_structured_outputs: None = None

    def __init__(self) -> None:
        super().__init__(model_name="mock")
        object.__setattr__(self, "requests", [])

    async def generate(self, messages: Any, **kwargs: Any) -> LLMResponse:
        self.requests.append(list(messages))
        if any(m.role == "tool" for m in messages):
            return LLMResponse(content="Blocked.", generated_tokens=1, execution_time=0.0)
        return LLMResponse(
            content="",
            generated_tokens=1,
            execution_time=0.0,
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "block_card", "arguments": '{"card_id": "K-1"}'},
                }
            ],
        )

    async def generate_streaming(
        self, messages: Any, **kwargs: Any
    ) -> AsyncGenerator[LLMResponse, None]:
        yield await self.generate(messages, **kwargs)


@pytest.mark.asyncio
async def test_a_resumed_turn_reads_its_task_before_the_result() -> None:
    llm = _BlockThenAnswer()
    tool = DelegateTool(
        tool_name="block_card",
        tool_description="Block a card.",
        tool_parameters={"card_id": {"type": "string"}},
        required_params=["card_id"],
        approval_type=ApprovalType.USER_CONFIRM,
        wrapped=lambda card_id: "blocked",
    )
    agent = Agent(
        name="Analyst",
        description="test",
        instruction="Block cards.",
        llm_provider=llm,
        planner=ToolUseWithThoughtPlannerStrategy(),
        tools=[tool],
    )
    runner = Runner()
    start = BaseEvent(
        event_type=ExecutionEventType.AGENT_START,
        author="fraud.alerts",
        content=[Part(text='{"alert_id": "A-1"}')],
        conversation_id="A-1",
    )

    request = None
    async for event in runner.run(
        application_name="t", user_id="u", conversation_id="A-1", root=agent, user_input=start
    ):
        if event.event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK:
            request = event
    assert request is not None

    answer = BaseEvent(
        event_type=ExecutionEventType.TOOL_FEEDBACK,
        author="analyst",
        action=request.action.model_copy(update={"approve": True}),
    )
    events = [
        event
        async for event in runner.run(
            application_name="t", user_id="u", conversation_id="A-1", root=agent, user_input=answer
        )
    ]

    assert not [e for e in events if e.event_type == ExecutionEventType.ERROR_OCCURRED]
    resumed = [m for m in llm.requests[-1] if m.role != "system"]
    assert [m.role for m in resumed] == ["user", "assistant", "tool"]
    assert "A-1" in resumed[0].content
