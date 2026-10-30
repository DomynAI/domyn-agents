import asyncio
from collections.abc import AsyncGenerator

import pytest

from domyn.agents import Runner
from domyn.agents.agents import Agent
from domyn.agents.core import (
    METADATA_EXTERNAL_PART_AGENT_VISIBLE,
    BaseEvent,
    ExecutionEventType,
    InteractionContext,
    Part,
    ToolAction,
)
from domyn.agents.llm import BaseLLMProvider
from domyn.agents.tools import FunctionTool


@pytest.mark.asyncio
async def test_push_parts_interleaves_with_slow_agent_loop(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """push_parts() should surface injected parts as a partial RESPONSE event
    while the agent loop is still busy (e.g. awaiting a slow LLM/tool call),
    without waiting for the agent loop to reach a yield point on its own."""

    slow_step_started = asyncio.Event()

    async def slow_execute(self, context: InteractionContext) -> AsyncGenerator[BaseEvent, None]:
        slow_step_started.set()
        await asyncio.sleep(0.2)
        yield BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="MainAgent",
            content=[Part(text="final answer")],
            is_partial=False,
        )

    main_agent = Agent(
        name="MainAgent",
        description="main agent description",
        instruction="main agent instruction",
        llm_provider=llm_provider,
    )
    main_agent._execute = slow_execute.__get__(main_agent, Agent)

    application_name = "TestApplication"
    user_id = "test_user"
    conversation_id = "test_push_parts_conversation"

    input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="Start the process")],
    )

    collected: list[BaseEvent] = []

    async def consume() -> None:
        async for event in runner.run(
            conversation_id=conversation_id,
            application_name=application_name,
            user_id=user_id,
            root=main_agent,
            user_input=input_event,
        ):
            collected.append(event)

    consumer_task = asyncio.create_task(consume())

    # Wait until the agent loop is actually inside the slow step, then push.
    await asyncio.wait_for(slow_step_started.wait(), timeout=1.0)
    pushed = await runner.push_parts(
        application_name=application_name,
        conversation_id=conversation_id,
        parts=[Part(text="injected notice")],
    )
    assert pushed.delivered is True

    await asyncio.wait_for(consumer_task, timeout=2.0)

    injected_events = [
        e
        for e in collected
        if e.event_type == ExecutionEventType.RESPONSE
        and e.content
        and e.content[0].text == "injected notice"
    ]
    assert len(injected_events) == 1
    assert injected_events[0].is_partial is True

    final_events = [
        e for e in collected if e.event_type == ExecutionEventType.RESPONSE and not e.is_partial
    ]
    assert len(final_events) == 1
    assert final_events[0].content[0].text == "final answer"

    # The injected event must have arrived before the (slow) final response.
    injected_index = collected.index(injected_events[0])
    final_index = collected.index(final_events[0])
    assert injected_index < final_index


@pytest.mark.asyncio
async def test_push_parts_returns_false_when_no_active_run(runner: Runner) -> None:
    pushed = await runner.push_parts(
        application_name="TestApplication",
        conversation_id="no-such-conversation",
        parts=[Part(text="hello")],
    )
    assert pushed.delivered is False


@pytest.mark.asyncio
async def test_push_parts_interleaves_with_slow_tool_call(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """Simulates: TOOL_START fires, then while the tool is still executing two
    parts are pushed externally, and only afterwards does the tool's own
    TOOL_END (and the agent's final response) arrive."""

    tool_started = asyncio.Event()

    async def slow_tool_fn() -> str:
        tool_started.set()
        await asyncio.sleep(0.2)
        return "tool result"

    slow_tool = FunctionTool(func=slow_tool_fn)

    async def agent_execute(self, context: InteractionContext) -> AsyncGenerator[BaseEvent, None]:
        yield BaseEvent(
            event_type=ExecutionEventType.TOOL_START,
            author="MainAgent",
            action=ToolAction(name=slow_tool.name, parameters={}, thought="calling slow tool"),
        )
        yield BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="MainAgent",
            content=[Part(text="final answer after tool")],
            is_partial=False,
        )

    main_agent = Agent(
        name="MainAgent",
        description="main agent description",
        instruction="main agent instruction",
        llm_provider=llm_provider,
        tools=[slow_tool],
    )
    main_agent._execute = agent_execute.__get__(main_agent, Agent)

    application_name = "TestApplication"
    user_id = "test_user"
    conversation_id = "test_push_parts_tool_conversation"

    input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="Start the process")],
    )

    collected: list[BaseEvent] = []

    async def consume() -> None:
        async for event in runner.run(
            conversation_id=conversation_id,
            application_name=application_name,
            user_id=user_id,
            root=main_agent,
            user_input=input_event,
        ):
            collected.append(event)

    consumer_task = asyncio.create_task(consume())

    # Wait until TOOL_START has fired and the tool itself is actually running.
    await asyncio.wait_for(tool_started.wait(), timeout=1.0)

    pushed_first = await runner.push_parts(
        application_name=application_name,
        conversation_id=conversation_id,
        parts=[Part(text="progress update 1")],
    )
    pushed_second = await runner.push_parts(
        application_name=application_name,
        conversation_id=conversation_id,
        parts=[Part(text="progress update 2")],
    )
    assert pushed_first.delivered is True
    assert pushed_second.delivered is True

    await asyncio.wait_for(consumer_task, timeout=2.0)

    tool_start_events = [e for e in collected if e.event_type == ExecutionEventType.TOOL_START]
    tool_end_events = [e for e in collected if e.event_type == ExecutionEventType.TOOL_END]
    injected_events = [
        e
        for e in collected
        if e.event_type == ExecutionEventType.RESPONSE
        and e.is_partial
        and e.content
        and e.content[0].text in {"progress update 1", "progress update 2"}
    ]
    final_events = [
        e for e in collected if e.event_type == ExecutionEventType.RESPONSE and not e.is_partial
    ]

    assert len(tool_start_events) == 1
    assert len(tool_end_events) == 1
    assert len(injected_events) == 2
    assert [e.content[0].text for e in injected_events] == [
        "progress update 1",
        "progress update 2",
    ]
    assert len(final_events) == 1
    assert final_events[0].content[0].text == "final answer after tool"

    # Ordering: TOOL_START -> both injected parts (pushed while the tool was
    # still running) -> TOOL_END -> final RESPONSE.
    tool_start_index = collected.index(tool_start_events[0])
    tool_end_index = collected.index(tool_end_events[0])
    final_index = collected.index(final_events[0])
    injected_indices = sorted(collected.index(e) for e in injected_events)

    assert tool_start_index < injected_indices[0]
    assert injected_indices[-1] < tool_end_index
    assert tool_end_index < final_index


@pytest.mark.asyncio
async def test_push_parts_honours_event_type_and_metadata(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """Content that did not come from the agent is pushed as EXTERNAL_PART, so that
    whether it re-enters the model's context stays a decision of the pusher
    (METADATA_EXTERNAL_PART_AGENT_VISIBLE) instead of being implied by the event
    type. A RESPONSE would be agent history unconditionally."""

    slow_step_started = asyncio.Event()

    async def slow_execute(self, context: InteractionContext) -> AsyncGenerator[BaseEvent, None]:
        slow_step_started.set()
        await asyncio.sleep(0.2)
        yield BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="MainAgent",
            content=[Part(text="final answer")],
            is_partial=False,
        )

    main_agent = Agent(
        name="MainAgent",
        description="main agent description",
        instruction="main agent instruction",
        llm_provider=llm_provider,
    )
    main_agent._execute = slow_execute.__get__(main_agent, Agent)

    application_name = "TestApplication"
    conversation_id = "test_push_parts_event_type"

    collected: list[BaseEvent] = []

    async def consume() -> None:
        async for event in runner.run(
            conversation_id=conversation_id,
            application_name=application_name,
            user_id="test_user",
            root=main_agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                author="user",
                content=[Part(text="Start the process")],
            ),
        ):
            collected.append(event)

    consumer_task = asyncio.create_task(consume())
    await asyncio.wait_for(slow_step_started.wait(), timeout=1.0)

    pushed = await runner.push_parts(
        application_name=application_name,
        conversation_id=conversation_id,
        parts=[Part(text="pushed by a tool")],
        author="weather-tool",
        event_type=ExecutionEventType.EXTERNAL_PART,
        metadata={METADATA_EXTERNAL_PART_AGENT_VISIBLE: True},
    )
    assert pushed.delivered is True

    await asyncio.wait_for(consumer_task, timeout=2.0)

    injected = [e for e in collected if e.event_type == ExecutionEventType.EXTERNAL_PART]
    assert len(injected) == 1
    assert injected[0].content[0].text == "pushed by a tool"
    assert injected[0].author == "weather-tool"
    assert injected[0].metadata[METADATA_EXTERNAL_PART_AGENT_VISIBLE] is True
    assert injected[0].is_partial is True


@pytest.mark.asyncio
async def test_push_parts_defaults_to_response_without_metadata(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """Existing callers pass neither event_type nor metadata: they must keep getting
    a partial RESPONSE authored by the entrypoint agent, with an empty metadata dict
    rather than None (the context filters call .get() on it)."""

    slow_step_started = asyncio.Event()

    async def slow_execute(self, context: InteractionContext) -> AsyncGenerator[BaseEvent, None]:
        slow_step_started.set()
        await asyncio.sleep(0.2)
        yield BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="MainAgent",
            content=[Part(text="final answer")],
            is_partial=False,
        )

    main_agent = Agent(
        name="MainAgent",
        description="main agent description",
        instruction="main agent instruction",
        llm_provider=llm_provider,
    )
    main_agent._execute = slow_execute.__get__(main_agent, Agent)

    application_name = "TestApplication"
    conversation_id = "test_push_parts_defaults"

    collected: list[BaseEvent] = []

    async def consume() -> None:
        async for event in runner.run(
            conversation_id=conversation_id,
            application_name=application_name,
            user_id="test_user",
            root=main_agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                author="user",
                content=[Part(text="Start the process")],
            ),
        ):
            collected.append(event)

    consumer_task = asyncio.create_task(consume())
    await asyncio.wait_for(slow_step_started.wait(), timeout=1.0)

    # `.delivered`, not the bare result: a PushResult is truthy whatever it says.
    assert (
        await runner.push_parts(
            application_name=application_name,
            conversation_id=conversation_id,
            parts=[Part(text="legacy push")],
        )
    ).delivered is True

    await asyncio.wait_for(consumer_task, timeout=2.0)

    injected = [
        e
        for e in collected
        if e.event_type == ExecutionEventType.RESPONSE
        and e.is_partial
        and e.content
        and e.content[0].text == "legacy push"
    ]
    assert len(injected) == 1
    assert injected[0].author == "MainAgent"
    assert injected[0].metadata == {}
