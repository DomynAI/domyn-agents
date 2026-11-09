"""LLM_END carries what the call cost, for both tool-use planners."""

# No `from __future__ import annotations`: builds real tools.
import json
from typing import Any

import pytest
from pydantic import PrivateAttr

from domyn.agents import Agent, BaseEvent, ExecutionEventType, Part, Runner
from domyn.agents.core import TOKEN_USAGE_METADATA_KEY
from domyn.agents.llm import BaseLLMProvider, LLMResponse, LLMToolCall, ToolCallFunction
from domyn.agents.planner_strategy import PlannerStrategyFactory
from domyn.agents.tools import FunctionTool

PLANNERS = ["tool_use", "tool_use_thought"]
USAGE = {
    "prompt_tokens": 100,
    "completion_tokens": 20,
    "total_tokens": 120,
    "prompt_tokens_details": {"cached_tokens": 64},
    "completion_tokens_details": {"reasoning_tokens": 5},
}
BROKEN = '{"path": "a.txt", "content": "he said "hi"", "reasoning": "x"}'


class _Provider(BaseLLMProvider):
    """Answers with `responses` in order, the last one repeated."""

    type: str = "mock"
    name: str = "mock"
    model_name: str = "mock"
    supported_structured_outputs: None = None
    responses: list[LLMResponse]
    _calls: int = PrivateAttr(default=0)

    def _next(self) -> LLMResponse:
        response = self.responses[min(self._calls, len(self.responses) - 1)]
        self._calls += 1
        return response

    async def generate(self, messages: Any, **kwargs: Any) -> LLMResponse:
        return self._next()

    async def generate_streaming(self, messages: Any, **kwargs: Any) -> Any:
        """Like the real providers: text first, usage on a closing chunk."""
        response = self._next()
        yield LLMResponse(content=response.content, generated_tokens=0, execution_time=0.0)
        yield LLMResponse(
            content="",
            usage=response.usage,
            generated_tokens=0,
            execution_time=0.0,
            tool_calls=response.tool_calls,
        )


def _answer(usage: dict[str, Any], content: str = "Done.") -> LLMResponse:
    return LLMResponse(content=content, usage=usage, generated_tokens=1, execution_time=0.0)


def _agent(provider: BaseLLMProvider, planner: str, stream: bool) -> Agent:
    def write_file(path: str, content: str) -> str:
        """Write a file.

        Args:
            path: Where.
            content: What.
        """
        return "ok"

    return Agent(
        name="A",
        description="d",
        instruction="i",
        llm_provider=provider,
        planner=PlannerStrategyFactory.from_config({"type": planner}, save_in_registry=False),
        tools=[FunctionTool(func=write_file)],
        stream=stream,
    )


async def _llm_ends(provider: BaseLLMProvider, planner: str, stream: bool) -> list[BaseEvent]:
    events = []
    async for event in Runner().run(
        application_name="app",
        user_id="u",
        root=_agent(provider, planner, stream),
        user_input=BaseEvent(
            event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text="go")]
        ),
        conversation_id="c1",
    ):
        if event.event_type == ExecutionEventType.LLM_END:
            events.append(event)
    return events


@pytest.mark.parametrize("planner", PLANNERS)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_llm_end_carries_the_breakdown(planner: str, stream: bool) -> None:
    ends = await _llm_ends(_Provider(responses=[_answer(USAGE)]), planner, stream)
    assert len(ends) == 1
    assert ends[0].metadata[TOKEN_USAGE_METADATA_KEY] == {
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
        "cache_read_tokens": 64,
        "reasoning_tokens": 5,
        "requests": 1,
    }


@pytest.mark.parametrize("planner", PLANNERS)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_no_usage_reported_means_no_key(planner: str, stream: bool) -> None:
    ends = await _llm_ends(_Provider(responses=[_answer({})]), planner, stream)
    assert ends and all(TOKEN_USAGE_METADATA_KEY not in e.metadata for e in ends)


@pytest.mark.parametrize("planner", PLANNERS)
@pytest.mark.asyncio
async def test_a_retry_is_counted_in_the_same_llm_end(planner: str) -> None:
    """A malformed tool call is asked again before LLM_END: both calls were paid."""
    broken = LLMResponse(
        content="",
        usage={"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
        generated_tokens=1,
        execution_time=0.0,
        tool_calls=[
            LLMToolCall(id="c1", function=ToolCallFunction(name="write_file", arguments=BROKEN))
        ],
    )
    ends = await _llm_ends(_Provider(responses=[broken, _answer(USAGE)]), planner, False)
    assert len(ends) == 1
    assert ends[0].metadata[TOKEN_USAGE_METADATA_KEY] == {
        "input_tokens": 150,
        "output_tokens": 30,
        "total_tokens": 180,
        "cache_read_tokens": 64,
        "reasoning_tokens": 5,
        "requests": 2,
    }


@pytest.mark.parametrize("planner", PLANNERS)
@pytest.mark.asyncio
async def test_each_round_trip_reports_its_own(planner: str) -> None:
    """A tool call, then the answer: two LLM_ENDs, each with its own call's count."""
    call = LLMResponse(
        content="",
        usage={"prompt_tokens": 40, "completion_tokens": 8},
        generated_tokens=1,
        execution_time=0.0,
        tool_calls=[
            LLMToolCall(
                id="c1",
                function=ToolCallFunction(
                    name="write_file",
                    arguments=json.dumps({"path": "a", "content": "b", "reasoning": "r"}),
                ),
            )
        ],
    )
    ends = await _llm_ends(_Provider(responses=[call, _answer(USAGE)]), planner, False)
    assert [e.metadata[TOKEN_USAGE_METADATA_KEY]["input_tokens"] for e in ends] == [40, 100]
    assert all(e.metadata[TOKEN_USAGE_METADATA_KEY]["requests"] == 1 for e in ends)
