from collections.abc import AsyncGenerator
from typing import Any

import pytest

from domyn.agents import BaseEvent, ExecutionEventType
from domyn.agents.llm import BaseLLMProvider, LLMMessage, LLMResponse
from domyn.agents.planner_strategy.context_management.summary import EventsSummarizer

STANDARD_RESPONSE = "This is a new mock summary of the events."
CACHED_RESPONSE = "This is the cached summary of the events."


class MockLLMProvider(BaseLLMProvider):
    response: str = STANDARD_RESPONSE
    type: str = "mock"
    name: str = "mock_provider"
    model_name: str = "mock-model"

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        # Mock response for testing purposes
        return LLMResponse(content=self.response, generated_tokens=0, execution_time=0)

    async def generate_streaming(
        self, messages: list[LLMMessage], **kwargs: Any
    ) -> AsyncGenerator[LLMResponse, None]:
        """Asynchronously generate content from messages."""


test_data = [
    (
        [
            BaseEvent(
                event_type=ExecutionEventType.USER_INPUT, author="user", content=[], event_id="1"
            ),
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE, author="agent", content=[], event_id="2"
            ),
        ],
        {"summaries": {str(("1", "2")): CACHED_RESPONSE}},
        CACHED_RESPONSE,
    ),
    (
        [
            BaseEvent(
                event_type=ExecutionEventType.USER_INPUT, author="user", content=[], event_id="1"
            ),
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE, author="agent", content=[], event_id="3"
            ),
        ],
        {"summaries": {str(("1", "2")): CACHED_RESPONSE}},
        STANDARD_RESPONSE,
    ),
]


@pytest.mark.parametrize("events, state, expected_summary", test_data)
@pytest.mark.asyncio
async def test_events_summarizer(
    events: list[BaseEvent], state: dict[str, Any], expected_summary: str
):
    summarizer = EventsSummarizer(llm_provider=MockLLMProvider())
    summary = await summarizer.get_history_summary(events, cache=state)
    assert summary == expected_summary
