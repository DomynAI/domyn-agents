"""Regression test: non-partial events must appear exactly once in interaction_history.

The bug was that `Agent._execute()` and `Runner._add_event()` both appended
events to `context.interaction_history`, causing every non-partial event to
be duplicated.  The fix removed the append from `Agent._execute()`.
"""

from collections import Counter
from collections.abc import AsyncGenerator

import pytest

from domyn.agents import Runner
from domyn.agents.agents import Agent
from domyn.agents.core import BaseEvent, ExecutionEventType, InteractionContext, Part, ToolAction
from domyn.agents.hooks import FunctionHook, HookPhase, HookType
from domyn.agents.llm import BaseLLMProvider
from domyn.agents.tools import (
    FunctionTool,
    StreamableFunctionTool,
    StreamableToolResult,
    ToolResult,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def async_gen_from_list(values):
    _iter = iter(values)
    finished = False

    async def _gen(*args, **kwargs) -> AsyncGenerator:
        nonlocal finished
        if finished:
            raise StopAsyncIteration
        for v in _iter:
            yield v
        finished = True

    return _gen


def add(a: int, b: int) -> ToolResult:
    """Add two numbers."""
    return ToolResult(data=str(a + b))


# ── tests ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_duplicate_events_in_interaction_history(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """Non-partial events must appear exactly once in interaction_history."""
    captured_context: list[InteractionContext] = []

    async def capture_context(context: InteractionContext, agent: Agent) -> None:
        captured_context.append(context)

    post_hook = FunctionHook(
        name="capture_ctx",
        fun=capture_context,
        trigger=HookPhase.POST_AGENT_CALL,
        type=HookType.LISTENER,
    )

    agent = Agent(
        name="TestAgent",
        description="test agent",
        instruction="you are a test agent",
        llm_provider=llm_provider,
        tools=[FunctionTool(func=add)],
        hooks=[post_hook],
    )

    agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.LLM_START,
                author="TestAgent",
                content=[Part(text="calling llm")],
            ),
            BaseEvent(
                event_type=ExecutionEventType.LLM_END,
                author="TestAgent",
                content=[Part(text="llm done")],
            ),
            BaseEvent(
                event_type=ExecutionEventType.TOOL_START,
                author="TestAgent",
                action=ToolAction(
                    name="add",
                    call_id="call_1",
                    parameters={"a": 1, "b": 2},
                    thought="adding",
                ),
            ),
            BaseEvent(
                event_type=ExecutionEventType.LLM_START,
                author="TestAgent",
                content=[Part(text="calling llm")],
            ),
            BaseEvent(
                event_type=ExecutionEventType.LLM_END,
                author="TestAgent",
                content=[Part(text="llm done")],
            ),
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="TestAgent",
                content=[Part(text="result is 3")],
            ),
        ]
    )

    user_input = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="add 1 and 2")],
    )

    _ = [
        e
        async for e in runner.run(
            application_name="test_app",
            user_id="test_user",
            root=agent,
            user_input=user_input,
            conversation_id="test_no_dupes",
        )
    ]

    assert len(captured_context) == 1, "post-hook should have been called once"
    history = captured_context[0].interaction_history

    # No event object (by identity) should appear more than once
    seen_ids: set[int] = set()
    duplicates = []
    for ev in history:
        if id(ev) in seen_ids:
            duplicates.append(ev.event_type)
        seen_ids.add(id(ev))

    assert duplicates == [], f"Duplicate event objects found in interaction_history: {duplicates}"

    # Also verify by event_type counts: each scripted event type should appear
    # the expected number of times (LLM_START x2, LLM_END x2, TOOL_START x1,
    # RESPONSE x1) plus runner-generated events (AGENT_START, TOOL_END, etc.)
    type_counts = Counter(ev.event_type for ev in history)
    assert type_counts[ExecutionEventType.LLM_START] == 2
    assert type_counts[ExecutionEventType.LLM_END] == 2
    assert type_counts[ExecutionEventType.TOOL_START] == 1
    assert type_counts[ExecutionEventType.RESPONSE] == 1


@pytest.mark.asyncio
async def test_no_duplicate_db_events_on_agent_event_listener_with_streaming_tool(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """Regression: on_agent_event LISTENER hook with RESPONSE trigger + streaming tool.

    Two bugs existed:
    (1) The runner aborted after the first HOOK_START because event_type==RESPONSE
        set should_continue=False before the hook body ran. The hook was never
        entered and streaming stopped immediately.
    (2) After fixing (1), a duplicate _add_event call stored the same event object
        twice, violating the (application_name, conversation_id, turn_id, event_id)
        primary key in the DB.
    """

    observed_partials: list[BaseEvent] = []

    async def streaming_func() -> AsyncGenerator:
        yield StreamableToolResult(data=Part(text="chunk_1"))
        yield StreamableToolResult(data=Part(text="chunk_2"))
        yield ToolResult(data="done")

    def observe_response(event: BaseEvent, context) -> None:
        if event.is_partial:
            observed_partials.append(event)

    hook = FunctionHook(
        name="observe_response",
        fun=observe_response,
        trigger=HookPhase.ON_AGENT_EVENT,
        type=HookType.LISTENER,
        on_event_triggers=[ExecutionEventType.RESPONSE],
    )

    agent = Agent(
        name="TestAgent",
        description="test agent",
        instruction="you are a test agent",
        llm_provider=llm_provider,
        tools=[StreamableFunctionTool(func=streaming_func)],
        hooks=[hook],
    )

    agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.TOOL_START,
                author="TestAgent",
                action=ToolAction(
                    name="streaming_func",
                    call_id="call_1",
                    parameters={},
                    thought="streaming",
                ),
            ),
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="TestAgent",
                content=[Part(text="all done")],
            ),
        ]
    )

    user_input = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="go")],
    )

    events = [
        e
        async for e in runner.run(
            application_name="test_app",
            user_id="test_user",
            root=agent,
            user_input=user_input,
            conversation_id="test_no_dupes_streaming_listener",
        )
    ]

    # (1) The hook body must have run for both streaming chunks. Before the fix
    # should_continue was set to False right after HOOK_START, so the hook never
    # executed and observed_partials would be empty.
    assert len(observed_partials) == 2, (
        f"Hook must observe both partial streaming chunks; got {len(observed_partials)}"
    )

    # (2) Runner must complete with a final RESPONSE (not abort mid-stream).
    final_responses = [
        e for e in events if e.event_type == ExecutionEventType.RESPONSE and not e.is_partial
    ]
    assert len(final_responses) == 1, (
        f"Expected exactly one final RESPONSE; got {len(final_responses)}"
    )

    # (3) No duplicate event_ids in the conversation store (proxy for DB primary-key
    # uniqueness). The duplicate arose because _add_event was called inside the
    # early-exit block and again unconditionally in the post-loop code.
    conv_info = await runner.conversation_manager.get_conversation(
        "test_app", "test_user", "test_no_dupes_streaming_listener"
    )
    event_ids = [e.event_id for e in conv_info.events_history if e.event_id is not None]
    duplicates = [eid for eid in set(event_ids) if event_ids.count(eid) > 1]
    assert duplicates == [], f"Duplicate event_ids in conversation history: {duplicates}"


@pytest.mark.asyncio
async def test_no_duplicate_db_events_on_agent_event_updater_transforms_streaming_response(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """Regression: on_agent_event UPDATER hook that converts a partial RESPONSE from a
    streaming tool into a final RESPONSE must store that event exactly once.

    When should_continue becomes False because the hook-modified event is a non-partial
    RESPONSE, the early-exit path previously called _add_event and the post-loop code
    called it again, producing a duplicate row in the DB.
    """

    async def streaming_func() -> AsyncGenerator:
        yield StreamableToolResult(data=Part(text="chunk_1"))
        yield ToolResult(data="done")

    def intercept_first_chunk(event: BaseEvent, context) -> BaseEvent:
        # Transform the first partial RESPONSE into a final RESPONSE to trigger
        # the early-exit path (should_continue=False, break_execution=False).
        return BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author=event.author,
            content=[Part(text="intercepted by hook")],
            is_partial=False,
            turn_id=event.turn_id,
            interaction_id=event.interaction_id,
        )

    hook = FunctionHook(
        name="intercept_chunk",
        fun=intercept_first_chunk,
        trigger=HookPhase.ON_AGENT_EVENT,
        type=HookType.UPDATER,
        on_event_triggers=[ExecutionEventType.RESPONSE],
    )

    agent = Agent(
        name="TestAgent",
        description="test agent",
        instruction="you are a test agent",
        llm_provider=llm_provider,
        tools=[StreamableFunctionTool(func=streaming_func)],
        hooks=[hook],
    )

    agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.TOOL_START,
                author="TestAgent",
                action=ToolAction(
                    name="streaming_func",
                    call_id="call_1",
                    parameters={},
                    thought="streaming",
                ),
            ),
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="TestAgent",
                content=[Part(text="original — must not be yielded")],
            ),
        ]
    )

    user_input = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="go")],
    )

    events = [
        e
        async for e in runner.run(
            application_name="test_app",
            user_id="test_user",
            root=agent,
            user_input=user_input,
            conversation_id="test_no_dupes_streaming_updater",
        )
    ]

    # The hook-modified event must be emitted (not the original chunk).
    final_responses = [
        e for e in events if e.event_type == ExecutionEventType.RESPONSE and not e.is_partial
    ]
    assert len(final_responses) == 1, (
        f"Expected exactly one final RESPONSE; got {len(final_responses)}"
    )
    assert final_responses[0].content[0].text == "intercepted by hook"

    # The hook-modified RESPONSE must appear exactly once in the conversation store.
    conv_info = await runner.conversation_manager.get_conversation(
        "test_app", "test_user", "test_no_dupes_streaming_updater"
    )
    event_ids = [e.event_id for e in conv_info.events_history if e.event_id is not None]
    duplicates = [eid for eid in set(event_ids) if event_ids.count(eid) > 1]
    assert duplicates == [], f"Duplicate event_ids in conversation history: {duplicates}"
