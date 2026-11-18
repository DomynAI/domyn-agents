"""The two tool-use strategies run one loop (``ToolUsePlannerStrategy.execute``).

It used to be copied, and the copies had drifted apart where no test looked; these
pin what each one did differently and now both do the same way.
"""

# No `from __future__ import annotations`: builds real agents.
import asyncio
from typing import Any

import pytest
from pydantic import Field

from domyn.agents.agents import Agent
from domyn.agents.core import BaseEvent, ExecutionEventType, Part
from domyn.agents.llm.base import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
    LLMToolCall,
    ToolCallFunction,
)
from domyn.agents.planner_strategy import PlannerStrategyFactory
from domyn.agents.planner_strategy.tool_use import MAX_TOOL_CALL_ARGS_RETRIES
from domyn.agents.runner import Runner

PLANNERS = ["tool_use", "tool_use_thought"]


class _Replies(BaseLLMProvider):
    """Answers with ``replies`` in turn, the last one for ever."""

    type: str = "replies"
    name: str = "replies"
    model_name: str = "replies"
    supported_structured_outputs: list[Any] | None = None
    replies: list[LLMResponse] = Field(default_factory=list)
    asked: int = 0

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        self.asked += 1
        return self.replies[min(self.asked, len(self.replies)) - 1]

    async def generate_streaming(self, messages: list[LLMMessage], **kwargs: Any) -> Any:
        raise NotImplementedError
        yield  # pragma: no cover


def _says(text: str, calls: list[tuple[str, str]] | None = None) -> LLMResponse:
    return LLMResponse(
        content=text,
        generated_tokens=1,
        execution_time=0.0,
        usage={},
        tool_calls=[
            LLMToolCall(id=f"c{i}", type="function", function=ToolCallFunction(name=n, arguments=a))
            for i, (n, a) in enumerate(calls or [])
        ],
    )


async def _turn(planner: str, provider: _Replies, **options: Any) -> list[BaseEvent]:
    agent = Agent(
        name="Solo",
        description="the agent",
        instruction="answer",
        llm_provider=provider,
        planner=PlannerStrategyFactory.from_config(
            {"type": planner, **options}, save_in_registry=False
        ),
    )
    ask = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text="hi")]
    )
    return [
        e
        async for e in Runner().run(
            application_name="one_loop",
            user_id="u",
            conversation_id=f"c_{planner}_{id(provider)}",
            root=agent,
            user_input=ask,
        )
    ]


class TestAModelThatSaysNothing:
    """An answer with no text and no call is asked again, a few times. ``tool_use``
    asked again with no limit, so such a model held the turn for ever; the thought
    strategy did not ask again at all."""

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_it_is_asked_again_and_the_turn_still_ends(self, planner: str) -> None:
        provider = _Replies(replies=[_says("")])

        await asyncio.wait_for(_turn(planner, provider, max_iterations=2), timeout=10)

        # Each of the two iterations: the first answer and its retries.
        assert provider.asked == 2 * (1 + MAX_TOOL_CALL_ARGS_RETRIES)

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_an_answer_after_the_nudge_is_the_answer(self, planner: str) -> None:
        provider = _Replies(replies=[_says(""), _says("Hello.")])

        events = await _turn(planner, provider)

        answers = [e for e in events if e.event_type == ExecutionEventType.RESPONSE]
        assert answers and "Hello." in answers[-1].content[0].text
        assert provider.asked == 2


class TestAnAnswerOfBlanks:
    """Only whitespace, from an agent that ends with STOP: the thought strategy read
    names that exist only when the response made calls, and failed with a
    NameError."""

    @pytest.mark.parametrize("use_stop", [False, True])
    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_it_ends_the_turn_without_an_error(self, planner: str, use_stop: bool) -> None:
        events = await _turn(planner, _Replies(replies=[_says("   ")]), use_stop=use_stop)

        assert not [e for e in events if e.event_type == ExecutionEventType.ERROR_OCCURRED]
        assert events[-1].event_type == ExecutionEventType.AGENT_END


class TestStop:
    """The AGENT_END a STOP call produces carries that call's id, in both."""

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_it_carries_the_call_id(self, planner: str) -> None:
        provider = _Replies(replies=[_says("", [("STOP", '{"info": "done"}')])])

        events = await _turn(planner, provider, use_stop=True)

        [end] = [
            e for e in events if e.event_type == ExecutionEventType.AGENT_END and e.author == "Solo"
        ]
        assert end.action.call_id == "c0"
