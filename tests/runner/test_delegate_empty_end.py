"""A delegate that ends without a final answer must not have one invented for it.

A delegate that streams its answer as tokens ends with a bare AGENT_END. The root
``DelegateAgent`` still turns that into the final RESPONSE (the runner closes the turn only after
one), but the RESPONSE carries no text and no made-up thought such as "finished without providing
a content", which the platform would show next to the real answer.
"""

from collections.abc import AsyncGenerator

import pytest

from domyn.agents import BaseEvent, ExecutionEventType, Part, Runner
from domyn.agents.agents import DelegateAgent
from domyn.agents.core import InteractionContext
from domyn.agents.runner import RunnerTransport

_NAME = "Main agent"


def _event(event_type: ExecutionEventType, **fields: object) -> BaseEvent:
    return BaseEvent(event_type=event_type, author=_NAME, **fields)  # type: ignore[arg-type]


class _Transport(RunnerTransport):
    def __init__(self, events: list[BaseEvent]) -> None:
        self._events = events

    async def execute(
        self, agent_name: str, context: InteractionContext
    ) -> AsyncGenerator[BaseEvent, None]:
        for event in self._events:
            yield event

    async def send_result(self, event: BaseEvent) -> None:
        pass


async def _run(events: list[BaseEvent]) -> list[BaseEvent]:
    agent = DelegateAgent(name=_NAME, description="d", transport=_Transport(events))
    user_input = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text="hi")]
    )
    return [
        e
        async for e in Runner().run(
            application_name="app",
            user_id="u",
            root=agent,
            user_input=user_input,
            conversation_id="c",
        )
    ]


def _tokens() -> list[BaseEvent]:
    return [
        _event(ExecutionEventType.RESPONSE_TOKEN, content=[Part(text=t)], is_partial=True)
        for t in ("Hel", "lo")
    ]


@pytest.mark.asyncio
async def test_streamed_answer_then_bare_end_adds_no_text_or_thought() -> None:
    events = await _run([*_tokens(), _event(ExecutionEventType.AGENT_END)])

    final = [e for e in events if e.event_type == ExecutionEventType.RESPONSE]
    assert len(final) == 1
    assert not final[0].is_partial
    assert final[0].content == []
    assert [p for e in events for p in (e.content or []) if p.thought] == []


@pytest.mark.asyncio
async def test_the_turn_is_still_closed() -> None:
    events = await _run([*_tokens(), _event(ExecutionEventType.AGENT_END)])

    assert [e.event_type for e in events][-2:] == [
        ExecutionEventType.RESPONSE,
        ExecutionEventType.AGENT_END,
    ]


@pytest.mark.asyncio
async def test_a_failure_still_reaches_the_platform() -> None:
    events = await _run(
        [_event(ExecutionEventType.AGENT_END, error_code="Boom", error_message="it failed")]
    )

    (final,) = [e for e in events if e.event_type == ExecutionEventType.RESPONSE]
    assert (final.error_code, final.error_message) == ("Boom", "it failed")


@pytest.mark.asyncio
async def test_an_answer_the_delegate_sends_is_passed_on_untouched() -> None:
    events = await _run(
        [
            _event(ExecutionEventType.AGENT_END, content=[Part(text="Hello")]),
        ]
    )

    (final,) = [e for e in events if e.event_type == ExecutionEventType.RESPONSE]
    assert [p.text for p in final.content] == ["Hello"]
