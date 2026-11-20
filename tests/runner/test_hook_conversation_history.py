"""An event hook asking for `conversation_history` sees the earlier turns, not the current one."""

from collections.abc import AsyncGenerator
from typing import Any

import pytest

from domyn.agents import Runner
from domyn.agents.agents import Agent
from domyn.agents.core import BaseEvent, ExecutionEventType, Part
from domyn.agents.hooks import FunctionHook, HookPhase, HookType
from domyn.agents.hooks.base import InputFields
from domyn.agents.llm import BaseLLMProvider


def _texts(events: list[BaseEvent]) -> list[tuple[ExecutionEventType, str]]:
    return [(e.event_type, e.content[0].text or "") for e in events if e.content]


@pytest.mark.asyncio
async def test_user_input_hook_receives_the_earlier_turns(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    seen: list[tuple[BaseEvent, list[BaseEvent]]] = []

    def check_input(current_event: BaseEvent, conversation_history: list[BaseEvent]) -> None:
        seen.append((current_event, conversation_history))

    hook = FunctionHook(
        name="check_input",
        fun=check_input,
        trigger=HookPhase.ON_AGENT_EVENT,
        type=HookType.LISTENER,
        on_event_triggers=[ExecutionEventType.USER_INPUT],
        input_fields=[InputFields.CURRENT_EVENT, InputFields.CONVERSATION_HISTORY],
    )
    agent = Agent(
        name="TestAgent",
        description="test agent",
        instruction="you are a test agent",
        llm_provider=llm_provider,
        hooks=[hook],
    )
    answers = iter(["first answer", "second answer"])

    async def execute(*args: Any, **kwargs: Any) -> AsyncGenerator[BaseEvent, None]:
        yield BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="TestAgent",
            content=[Part(text=next(answers))],
        )

    agent._execute = execute  # type: ignore[method-assign]

    for text in ("first question", "second question"):
        async for _ in runner.run(
            application_name="test_app",
            user_id="test_user",
            root=agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text=text)]
            ),
            conversation_id="test_hook_conversation_history",
        ):
            pass

    assert [current.content[0].text for current, _ in seen] == ["first question", "second question"]
    assert seen[0][1] == []
    history = _texts(seen[1][1])
    assert (ExecutionEventType.USER_INPUT, "first question") in history
    assert (ExecutionEventType.RESPONSE, "first answer") in history
    assert all(text != "second question" for _, text in history)
