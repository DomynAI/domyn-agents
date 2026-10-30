"""react, asked for by its name alone, answers: ``ReactPlannerStrategy()`` and
``{"type": "react"}`` need no context manager.
"""

from collections.abc import AsyncGenerator, Callable
from typing import Any

import pytest

from domyn.agents import Agent, BaseEvent, ExecutionEventType, Part, Runner
from domyn.agents.llm import BaseLLMProvider, LLMResponse
from domyn.agents.planner_strategy import ReactPlannerStrategy

_ANSWER = (
    "<thought>Nothing to look up.</thought>"
    '<action>{"name": "RESPOND", "arguments": {"text": "hello"}}</action>'
)


class _TextProtocolModel(BaseLLMProvider):
    """Answers in react's text protocol, the only one react reads."""

    type: str = "fake"
    name: str = "fake"
    model_name: str = "fake"
    supported_structured_outputs: None = None

    async def generate(self, messages: Any, **kwargs: Any) -> LLMResponse:
        return LLMResponse(content=_ANSWER, generated_tokens=1, execution_time=0.0)

    async def generate_streaming(
        self, messages: Any, **kwargs: Any
    ) -> AsyncGenerator[LLMResponse, None]:
        yield await self.generate(messages, **kwargs)


def _in_code() -> Agent:
    return Agent(
        name="A",
        description="d",
        instruction="i",
        llm_provider=_TextProtocolModel(),
        planner=ReactPlannerStrategy(),
    )


def _from_config() -> Agent:
    return Agent.from_config(
        {
            "name": "A",
            "description": "d",
            "instruction": "i",
            "llm_provider": _TextProtocolModel(),
            "planner_strategy": {"type": "react"},
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("build", [_in_code, _from_config], ids=["in_code", "from_config"])
async def test_react_named_alone_answers(build: Callable[[], Agent]) -> None:
    answers: list[str] = []
    async for event in Runner().run(
        application_name="a",
        user_id="u",
        root=build(),
        user_input=BaseEvent(
            event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text="hi")]
        ),
        conversation_id="c1",
    ):
        if event.event_type == ExecutionEventType.RESPONSE:
            answers.extend(p.text for p in event.content or [] if p.text)

    assert answers == ["hello"]
