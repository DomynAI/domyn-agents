"""A WrappedAgent (a function or a plain class) as the sub-agent a runner hands off to.

The root agent's planner is stubbed with a fixed AGENT_CALL, so these exercise the
runner's handoff and the wrapped agent's own execution, not a model.
"""

from collections.abc import AsyncGenerator, Iterator
from typing import Any

import pytest

from domyn.agents import Runner
from domyn.agents.agents import Agent, WrappedAgent
from domyn.agents.core import (
    AgentAction,
    BaseEvent,
    ExecutionEventType,
    InteractionContext,
    Part,
    ToolAction,
)
from domyn.agents.hooks import FunctionHook, HookPhase, HookType
from domyn.agents.llm import BaseLLMProvider
from domyn.agents.registry import get_agent_registry


def _agent_end(author: str, context: InteractionContext, text: str) -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.AGENT_END,
        author=author,
        action=ToolAction(
            name="STOP", parameters={}, observation=text, input_event=context.agent_input
        ),
        content=[Part(text=text)],
    )


async def order_agent(interaction_context: InteractionContext) -> AsyncGenerator[BaseEvent, None]:
    yield _agent_end("OrderAgent", interaction_context, "order 12345")


class PaymentAgent:
    async def execute_new(
        self, interaction_context: InteractionContext
    ) -> AsyncGenerator[BaseEvent, None]:
        yield _agent_end("PaymentAgent", interaction_context, "payment ok")


def _wrapped_function(**kwargs: Any) -> WrappedAgent:
    return WrappedAgent(
        name="OrderAgent", description="orders", wrapped_object=order_agent, **kwargs
    )


def _wrapped_class() -> WrappedAgent:
    return WrappedAgent(
        name="PaymentAgent",
        description="payments",
        wrapped_object=PaymentAgent(),
        default_execute_agent_function="execute_new",
    )


def _root(provider: BaseLLMProvider, sub: WrappedAgent | str, sub_name: str) -> Agent:
    root = Agent(
        name="Root",
        description="root",
        instruction="root",
        llm_provider=provider,
        sub_agents=[sub],
    )
    planned = iter(
        [
            BaseEvent(
                event_type=ExecutionEventType.AGENT_CALL,
                author="Root",
                content=[],
                action=AgentAction(name=sub_name, parameters={"task": "check"}, thought="t"),
            ),
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE, author="Root", content=[Part(text="done")]
            ),
        ]
    )

    async def _execute(*args: Any, **kwargs: Any) -> AsyncGenerator[BaseEvent, None]:
        for event in planned:  # shared iterator: re-entering after the handoff resumes it
            yield event

    root._execute = _execute
    return root


async def _run(root: Agent) -> list[BaseEvent]:
    return [
        event
        async for event in Runner().run(
            application_name="app",
            user_id="u1",
            root=root,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text="go")]
            ),
            conversation_id="conv-wrapped",
        )
    ]


def _trace(events: list[BaseEvent]) -> list[tuple[ExecutionEventType, str]]:
    return [(e.event_type, e.author) for e in events]


@pytest.fixture
def registered() -> Iterator[WrappedAgent]:
    agent = _wrapped_function()
    get_agent_registry().register_agent(agent.name, agent)
    yield agent
    get_agent_registry()._agents.pop(agent.name, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("make", [_wrapped_function, _wrapped_class])
async def test_the_root_hands_off_to_a_wrapped_agent_and_gets_its_answer(
    llm_provider: BaseLLMProvider, make: Any
) -> None:
    sub = make()

    events = await _run(_root(llm_provider, sub, sub.name))

    assert _trace(events) == [
        (ExecutionEventType.AGENT_START, "Root"),
        (ExecutionEventType.AGENT_CALL, "Root"),
        (ExecutionEventType.AGENT_START, sub.name),
        (ExecutionEventType.AGENT_END, sub.name),
        (ExecutionEventType.RESPONSE, "Root"),
        (ExecutionEventType.AGENT_END, "Root"),
    ]


@pytest.mark.asyncio
async def test_a_wrapped_agent_named_in_sub_agents_is_found_in_the_registry(
    llm_provider: BaseLLMProvider, registered: WrappedAgent
) -> None:
    events = await _run(_root(llm_provider, registered.name, registered.name))

    assert (ExecutionEventType.AGENT_END, "OrderAgent") in _trace(events)
    assert events[-2].content[0].text == "done"


@pytest.mark.asyncio
async def test_its_on_event_hooks_see_its_events(llm_provider: BaseLLMProvider) -> None:
    seen: list[ExecutionEventType] = []

    def record(event: BaseEvent, context: InteractionContext) -> None:
        seen.append(event.event_type)

    hook = FunctionHook(
        name="record", fun=record, trigger=HookPhase.ON_AGENT_EVENT, type=HookType.LISTENER
    )
    sub = _wrapped_function(on_event_hooks=[hook])

    await _run(_root(llm_provider, sub, sub.name))

    assert seen == [ExecutionEventType.AGENT_START, ExecutionEventType.AGENT_END]


@pytest.mark.asyncio
async def test_a_function_that_ends_without_agent_end_stops_there() -> None:
    """The function's events are all there is: nothing is looked up on it afterwards."""

    async def answer(context: InteractionContext) -> AsyncGenerator[BaseEvent, None]:
        yield BaseEvent(event_type=ExecutionEventType.RESPONSE, author="Fn", content=[])

    agent = WrappedAgent(name="Fn", description="d", wrapped_object=answer)
    context = InteractionContext(
        conversation_id="c",
        user_id="u",
        agent_input=BaseEvent(event_type=ExecutionEventType.AGENT_START, author="Fn", content=[]),
    )

    yielded = [item async for item in agent.execute(context)]

    assert [(event.event_type, ctx) for event, ctx in yielded] == [
        (ExecutionEventType.RESPONSE, context)
    ]
