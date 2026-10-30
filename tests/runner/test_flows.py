import asyncio
import inspect as _inspect
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta
from inspect import Parameter, Signature
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from domyn.agents.agents import Agent, ContextVisibility, DelegateAgent
from domyn.agents.conversation_manager import InMemoryConversationManager, StackAgentEntry
from domyn.agents.core import (
    METADATA_CONVERTED_TO_AGENT_END,
    METADATA_FORWARD_VERBATIM,
    AgentAction,
    ApprovalType,
    BaseAgent,
    BaseEvent,
    ExecutionEventType,
    FeedbackAction,
    FeedbackParameters,
    InteractionContext,
    Part,
    ToolAction,
)
from domyn.agents.hooks import (
    FunctionHook as FunctionAgentHook,
)
from domyn.agents.hooks import (
    FunctionHook as FunctionToolPreHook,
)
from domyn.agents.hooks import HookPhase, HookResult, HookType
from domyn.agents.llm import BaseLLMProvider, LLMMessage
from domyn.agents.runner import Runner, RunnerState, RunnerTransport
from domyn.agents.tools import FunctionTool


## utils
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
        return

    return _gen


@pytest.mark.asyncio
async def test_flow_sub_agent_ask_missing_info_bounces_to_parent(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """A sub-agent's need_feedback RESPONSE never reaches the user directly.

    The runner converts it into an AGENT_END carrying the question as the
    observation; the parent resumes and relays the question with its OWN
    need_feedback RESPONSE. The user's answer is then routed to the parent,
    which re-delegates; the sub-agent's final RESPONSE bounces back the same
    way (converted to AGENT_END, not yielded).
    """

    sub_agent = Agent(
        name="SubAgent",
        description="subagent description",
        instruction="subagent instruction",
        llm_provider=llm_provider,
    )

    main_agent = Agent(
        name="MainAgent",
        description="main agent description",
        instruction="main agent instruction",
        sub_agents=[sub_agent],
        llm_provider=llm_provider,
    )

    execute_events_main = [
        BaseEvent(
            event_type=ExecutionEventType.AGENT_CALL,
            author="main_agent",
            content=[],
            action=AgentAction(
                name="SubAgent", parameters={"agent_name": "SubAgent"}, thought="Invoking SubAgent"
            ),
        ),
        # consumed after the bounce, when the runner re-enters main's execute:
        # the parent relays the sub-agent's question to the user.
        BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="main_agent",
            content=[Part(text="SubAgent needs to know: what is your name?")],
            need_feedback=True,
        ),
    ]

    sub_agent_execute_events = [
        BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="sub_agent",
            content=[Part(text="What is your name?")],
            need_feedback=True,
        ),
        BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="sub_agent",
            content=[Part(text="Event to not reach, the sub-agent was closed by the bounce.")],
            need_feedback=False,
        ),
    ]

    main_agent._execute = async_gen_from_list(execute_events_main)
    sub_agent._execute = async_gen_from_list(sub_agent_execute_events)

    application_name = "TestApplication"
    user_id = "test_user"
    conversation_id = "test_flow_sub_agent_ask_missing_info_bounces_to_parent"

    input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="Start the process")],
        turn_id="turn_1",
    )

    runner_iterator = runner.run(
        conversation_id=conversation_id,
        application_name=application_name,
        user_id=user_id,
        root=main_agent,
        user_input=input_event,
    )
    # agent start of the main agent triggered by user input
    first_event = await anext(runner_iterator)
    assert first_event.event_type == ExecutionEventType.AGENT_START
    assert first_event.author == "MainAgent"

    # turn_id checks — first turn
    turn_id_1 = first_event.turn_id
    assert turn_id_1 == "turn_1"

    # strategy emits AGENT_CALL (calling agent -> sub-agent)
    second_event = await anext(runner_iterator)
    assert second_event.event_type == ExecutionEventType.AGENT_CALL
    assert second_event.author == "main_agent"
    assert second_event.action.name == "SubAgent"
    assert second_event.turn_id == turn_id_1

    # runner-created AGENT_START for the called sub-agent; stack is updated after this
    third_event = await anext(runner_iterator)
    assert third_event.event_type == ExecutionEventType.AGENT_START
    assert third_event.author == "SubAgent"
    assert third_event.action.name == "SubAgent"
    assert third_event.turn_id == turn_id_1

    # check stack — now SubAgent is on top
    assert (
        await runner.conversation_manager.get_last_agent_stack(
            application_name=application_name,
            user_id=user_id,
            conversation_id=conversation_id,
        )
    ).agent_name == "SubAgent"

    # The sub-agent's need_feedback RESPONSE is NOT yielded: the runner
    # converts it into an AGENT_END whose observation carries the question.
    fourth_event = await anext(runner_iterator)
    assert fourth_event.event_type == ExecutionEventType.AGENT_END
    assert fourth_event.author == "sub_agent"
    assert fourth_event.turn_id == turn_id_1
    observation = " ".join(p.text for p in fourth_event.content if p.text)
    assert "What is your name?" in observation
    assert "needs more information" in observation
    assert "What is your name?" in fourth_event.action.parameters["info"]

    # The parent resumed and relays the question to the user with its own
    # need_feedback RESPONSE — this one IS user-facing (root agent).
    fifth_event = await anext(runner_iterator)
    assert fifth_event.event_type == ExecutionEventType.RESPONSE
    assert fifth_event.need_feedback is True
    assert fifth_event.author == "main_agent"
    assert fifth_event.turn_id == turn_id_1

    with pytest.raises(StopAsyncIteration):
        await anext(runner_iterator)  # runner paused, waiting for the user's answer

    # SubAgent was popped by the bounce; MainAgent is the pending agent.
    conversation_info = await runner.conversation_manager.get_conversation(
        application_name=application_name,
        user_id=user_id,
        conversation_id=conversation_id,
    )
    assert [c.agent_name for c in conversation_info.agents_name_stack] == ["MainAgent"]
    # USER_INPUT + AGENT_START(main) + AGENT_CALL + AGENT_START(sub)
    # + RESPONSE(sub, marked) + AGENT_END(sub, synthesized) + RESPONSE(main)
    assert len(conversation_info.events_history) == 7
    # The sub-agent's original RESPONSE stays in history for auditability,
    # marked so context managers skip it in favor of the synthesized AGENT_END.
    sub_response = next(
        e
        for e in conversation_info.events_history
        if e.event_type == ExecutionEventType.RESPONSE and e.author == "sub_agent"
    )
    assert sub_response.metadata.get(METADATA_CONVERTED_TO_AGENT_END) is True

    # --- second turn: the user answers, the parent re-delegates ---

    feedback_input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="My name is Alice")],
        turn_id="turn_2",
    )

    main_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.AGENT_CALL,
                author="main_agent",
                content=[],
                action=AgentAction(
                    name="SubAgent",
                    parameters={"task": "greet the user, their name is Alice"},
                    thought="Re-delegating with the answer",
                ),
            ),
        ]
    )
    sub_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="sub_agent",
                content=[Part(text="Hello Alice!")],
                need_feedback=False,
            ),
        ]
    )

    runner_iterator = runner.run(
        conversation_id=conversation_id,
        application_name=application_name,
        user_id=user_id,
        root=main_agent,
        user_input=feedback_input_event,
    )

    events = [a async for a in runner_iterator]
    # AGENT_START(main, resumed), AGENT_CALL, AGENT_START(sub),
    # AGENT_END(sub, synthesized from its final RESPONSE)
    assert [e.event_type for e in events] == [
        ExecutionEventType.AGENT_START,
        ExecutionEventType.AGENT_CALL,
        ExecutionEventType.AGENT_START,
        ExecutionEventType.AGENT_END,
    ]
    turn_id_2 = events[0].turn_id
    assert turn_id_2 == "turn_2"
    assert all(e.turn_id == turn_id_2 for e in events)

    # The final answer bounced to the parent as AGENT_END content — no
    # sub-agent RESPONSE was ever yielded.
    assert events[3].author == "sub_agent"
    assert events[3].content[0].text == "Hello Alice!"
    assert not any(e.event_type == ExecutionEventType.RESPONSE for e in events)


@pytest.mark.asyncio
async def test_flow_agent_with_hooks(runner: Runner, llm_provider: BaseLLMProvider) -> None:
    """Test a flow where an agent with pre and post hooks is executed."""

    params = [
        Parameter("agent", Parameter.POSITIONAL_OR_KEYWORD),
        Parameter("context", Parameter.POSITIONAL_OR_KEYWORD),
    ]

    pre_func = AsyncMock(return_value=None)
    pre_func.__signature__ = Signature(params)
    pre = FunctionAgentHook(
        name="PreHook",
        description="A pre-hook that adds a message before agent execution",
        fun=pre_func,
        trigger=HookPhase.PRE_AGENT_CALL,
        type=HookType.LISTENER,
    )

    post_func = AsyncMock(side_effect=lambda agent, context: {"new_context": context})
    post_func.__signature__ = Signature(params)
    post = FunctionAgentHook(
        name="PostHook",
        description="A post-hook that updates the context after agent execution",
        fun=post_func,
        trigger=HookPhase.POST_AGENT_CALL,
        type=HookType.UPDATER,
    )

    main_agent = Agent(
        name="MainAgent",
        description="main agent description",
        instruction="main agent instruction",
        sub_agents=[],
        llm_provider=llm_provider,
        hooks=[pre, post],
    )

    agent_execution_events = [
        BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="MainAgent",
            content=[Part(text="Response partial from agent with hooks.")],
            is_partial=True,
            need_feedback=False,
        ),
        BaseEvent(event_type=ExecutionEventType.AGENT_END, author="MainAgent", content=[]),
    ]
    main_agent._execute = async_gen_from_list(agent_execution_events)

    application_name = "TestApplication"
    user_id = "test_user"
    conversation_id = "test_flow_agent_with_hooks"

    input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="Start the process")],
        turn_id="turn_1",
    )

    # create runner iterator
    runner_iterator = runner.run(
        conversation_id=conversation_id,
        application_name=application_name,
        user_id=user_id,
        root=main_agent,
        user_input=input_event,
    )

    events = [a async for a in runner_iterator]
    assert events

    assert (
        len(events) == 7
    )  # agent start, hook start, hook end, response, hook start, hook end, agent_end

    # turn_id: all events in this single turn must share the same turn_id
    turn_id = events[0].turn_id
    assert turn_id == "turn_1"
    assert all(e.turn_id == turn_id for e in events)
    assert pre_func.await_count == 1
    assert events[1].event_type == ExecutionEventType.HOOK_START and events[1].author == "MainAgent"
    assert events[2].event_type == ExecutionEventType.HOOK_END and events[2].author == "MainAgent"
    assert events[4].event_type == ExecutionEventType.HOOK_START and events[4].author == "MainAgent"
    assert events[5].event_type == ExecutionEventType.HOOK_END and events[5].author == "MainAgent"

    assert events[1].metadata["hook_name"] == "PreHook"
    assert events[2].metadata["hook_name"] == "PreHook"
    assert events[4].metadata["hook_name"] == "PostHook"
    assert events[5].metadata["hook_name"] == "PostHook"

    assert (
        events[0].event_type == ExecutionEventType.AGENT_START and events[0].author == "MainAgent"
    )
    assert events[6].event_type == ExecutionEventType.AGENT_END and events[6].author == "MainAgent"


@pytest.mark.asyncio
async def test_flow_direct_tool_call_events_have_turn_id(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """TOOL_END emitted during a direct tool call (no approval required) must carry turn_id.

    This covers the runner path where _execute_tool events previously did not get
    turn_id assigned before being yielded.
    """

    def add(a: int, b: int) -> int:
        return a + b

    main_agent = Agent(
        name="MainAgent",
        description="main agent description",
        instruction="main agent instruction",
        llm_provider=llm_provider,
        tools=[FunctionTool(func=add)],
    )

    main_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.TOOL_START,
                author="MainAgent",
                action=ToolAction(name="add", parameters={"a": 1, "b": 2}, thought="calling add"),
            ),
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="MainAgent",
                content=[Part(text="Result is 3")],
            ),
        ]
    )

    input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="add 1 and 2")],
        turn_id="turn_1",
    )

    events = [
        e
        async for e in runner.run(
            conversation_id="test_direct_tool_turn_id",
            application_name="TestApp",
            user_id="test_user",
            root=main_agent,
            user_input=input_event,
        )
    ]

    missing = [e.event_type for e in events if e.turn_id is None]
    assert not missing, f"Events missing turn_id: {missing}"
    assert all(e.turn_id == "turn_1" for e in events)

    tool_end = next((e for e in events if e.event_type == ExecutionEventType.TOOL_END), None)
    assert tool_end is not None, "Expected a TOOL_END event"
    assert tool_end.turn_id == "turn_1"


@pytest.mark.asyncio
async def test_flow_direct_tool_error_has_turn_id(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """TOOL_ERROR emitted when a direct tool call raises an exception must carry turn_id."""

    def failing_tool(x: int) -> int:
        raise ValueError("intentional failure")

    main_agent = Agent(
        name="MainAgent",
        description="main agent description",
        instruction="main agent instruction",
        llm_provider=llm_provider,
        tools=[FunctionTool(func=failing_tool)],
    )

    main_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.TOOL_START,
                author="MainAgent",
                action=ToolAction(
                    name="failing_tool", parameters={"x": 1}, thought="calling failing_tool"
                ),
            ),
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="MainAgent",
                content=[Part(text="Tool failed, sorry")],
            ),
        ]
    )

    input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="call the failing tool")],
        turn_id="turn_1",
    )

    events = [
        e
        async for e in runner.run(
            conversation_id="test_direct_tool_error_turn_id",
            application_name="TestApp",
            user_id="test_user",
            root=main_agent,
            user_input=input_event,
        )
    ]

    missing = [e.event_type for e in events if e.turn_id is None]
    assert not missing, f"Events missing turn_id: {missing}"
    assert all(e.turn_id == "turn_1" for e in events)

    tool_error = next((e for e in events if e.event_type == ExecutionEventType.TOOL_ERROR), None)
    assert tool_error is not None, "Expected a TOOL_ERROR event"
    assert tool_error.turn_id == "turn_1"


@pytest.mark.asyncio
async def test_flow_sub_agent_responses_never_reach_user(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """Sub-agent RESPONSE events are never yielded to the consumer.

    Partials are persisted but suppressed; the final RESPONSE is converted
    into an AGENT_END that returns control (and the content) to the parent.
    """

    sub_agent = Agent(
        name="SubAgent",
        description="subagent description",
        instruction="subagent instruction",
        llm_provider=llm_provider,
    )

    main_agent = Agent(
        name="MainAgent",
        description="main agent description",
        instruction="main agent instruction",
        sub_agents=[sub_agent],
        llm_provider=llm_provider,
    )

    execute_events_main = [
        BaseEvent(
            event_type=ExecutionEventType.AGENT_CALL,
            author="main_agent",
            content=[],
            action=AgentAction(
                name="SubAgent", parameters={"agent_name": "SubAgent"}, thought="Invoking SubAgent"
            ),
        ),
    ]

    sub_agent_execute_events = [
        BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="sub_agent",
            content=[Part(text="Intermediate narrative from the sub-agent.")],
            is_partial=True,
        ),
        BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="sub_agent",
            content=[Part(text="Sub-agent final answer.")],
            need_feedback=False,
        ),
    ]

    main_agent._execute = async_gen_from_list(execute_events_main)
    sub_agent._execute = async_gen_from_list(sub_agent_execute_events)

    application_name = "TestApplication"
    user_id = "test_user"
    conversation_id = "test_flow_sub_agent_responses_never_reach_user"

    user_input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="start process")],
        turn_id="turn_1",
    )

    # create runner iterator
    runner_iterator = runner.run(
        conversation_id=conversation_id,
        application_name=application_name,
        user_id=user_id,
        root=main_agent,
        user_input=user_input_event,
    )

    events = [a async for a in runner_iterator]
    # AGENT_START(user->main), AGENT_CALL(main->sub), AGENT_START(sub, runner-created),
    # AGENT_END(sub, synthesized from its final RESPONSE — the partial is suppressed)
    assert [e.event_type for e in events] == [
        ExecutionEventType.AGENT_START,
        ExecutionEventType.AGENT_CALL,
        ExecutionEventType.AGENT_START,
        ExecutionEventType.AGENT_END,
    ]
    assert events[3].author == "sub_agent"
    assert events[3].content[0].text == "Sub-agent final answer."

    # turn_id: all events in the first turn must share the same turn_id
    turn_id_1 = events[0].turn_id
    assert turn_id_1 == "turn_1"
    assert all(e.turn_id == turn_id_1 for e in events)

    # Both sub-agent RESPONSEs are persisted: the partial unmarked (regular
    # internal narrative), the converted final one marked for context managers.
    conversation_info = await runner.conversation_manager.get_conversation(
        application_name=application_name,
        user_id=user_id,
        conversation_id=conversation_id,
    )
    sub_responses = [
        e
        for e in conversation_info.events_history
        if e.event_type == ExecutionEventType.RESPONSE and e.author == "sub_agent"
    ]
    assert len(sub_responses) == 2
    partial, final = sub_responses
    assert partial.is_partial and not partial.metadata.get(METADATA_CONVERTED_TO_AGENT_END)
    assert final.metadata.get(METADATA_CONVERTED_TO_AGENT_END) is True


@pytest.mark.asyncio
async def test_flow_sub_agent_response_tokens_not_yielded(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """Streaming tokens from a sub-agent are internal: they must not reach the
    consumer. The content arrives grouped in the converted AGENT_END instead."""

    sub_agent = Agent(
        name="SubAgent",
        description="subagent description",
        instruction="subagent instruction",
        llm_provider=llm_provider,
    )

    main_agent = Agent(
        name="MainAgent",
        description="main agent description",
        instruction="main agent instruction",
        sub_agents=[sub_agent],
        llm_provider=llm_provider,
    )

    main_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.AGENT_CALL,
                author="main_agent",
                content=[],
                action=AgentAction(
                    name="SubAgent",
                    parameters={"agent_name": "SubAgent"},
                    thought="Invoking SubAgent",
                ),
            ),
        ]
    )
    sub_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE_TOKEN,
                author="sub_agent",
                content=[Part(text="Hello ")],
                is_partial=True,
            ),
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE_TOKEN,
                author="sub_agent",
                content=[Part(text="world")],
                is_partial=True,
            ),
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="sub_agent",
                content=[Part(text="Hello world")],
                need_feedback=False,
            ),
        ]
    )

    input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="start")],
        turn_id="turn_1",
    )

    events = [
        e
        async for e in runner.run(
            conversation_id="test_flow_sub_agent_response_tokens_not_yielded",
            application_name="TestApplication",
            user_id="test_user",
            root=main_agent,
            user_input=input_event,
        )
    ]

    assert not any(e.event_type == ExecutionEventType.RESPONSE_TOKEN for e in events)
    assert not any(e.event_type == ExecutionEventType.RESPONSE for e in events)
    agent_end = next(
        e
        for e in events
        if e.event_type == ExecutionEventType.AGENT_END and e.author == "sub_agent"
    )
    assert agent_end.content[0].text == "Hello world"


@pytest.mark.asyncio
async def test_flow_sub_agent_bounce_preserves_forward_verbatim_metadata(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """FORWARD_VERBATIM metadata on a sub-agent's final RESPONSE must survive
    the conversion to AGENT_END so the parent can re-forward verbatim."""

    sub_agent = Agent(
        name="SubAgent",
        description="subagent description",
        instruction="subagent instruction",
        llm_provider=llm_provider,
    )

    main_agent = Agent(
        name="MainAgent",
        description="main agent description",
        instruction="main agent instruction",
        sub_agents=[sub_agent],
        llm_provider=llm_provider,
    )

    main_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.AGENT_CALL,
                author="main_agent",
                content=[],
                action=AgentAction(
                    name="SubAgent",
                    parameters={"agent_name": "SubAgent"},
                    thought="Invoking SubAgent",
                ),
            ),
        ]
    )
    sub_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="sub_agent",
                content=[Part(text="verbatim payload")],
                metadata={METADATA_FORWARD_VERBATIM: True},
            ),
        ]
    )

    input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="start")],
        turn_id="turn_1",
    )

    events = [
        e
        async for e in runner.run(
            conversation_id="test_flow_sub_agent_bounce_preserves_fv",
            application_name="TestApplication",
            user_id="test_user",
            root=main_agent,
            user_input=input_event,
        )
    ]

    agent_end = next(
        e
        for e in events
        if e.event_type == ExecutionEventType.AGENT_END and e.author == "sub_agent"
    )
    assert agent_end.metadata.get(METADATA_FORWARD_VERBATIM) is True


@pytest.mark.asyncio
async def test_flow_nested_sub_agent_responses_bounce_level_by_level(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """A grandchild's final RESPONSE bounces to its caller, which in turn
    bounces its own final RESPONSE to the root — one stack level at a time."""

    sub_sub_agent = Agent(
        name="SubSubAgent",
        description="grandchild description",
        instruction="grandchild instruction",
        llm_provider=llm_provider,
    )
    sub_agent = Agent(
        name="SubAgent",
        description="subagent description",
        instruction="subagent instruction",
        sub_agents=[sub_sub_agent],
        llm_provider=llm_provider,
    )
    main_agent = Agent(
        name="MainAgent",
        description="main agent description",
        instruction="main agent instruction",
        sub_agents=[sub_agent],
        llm_provider=llm_provider,
    )

    main_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.AGENT_CALL,
                author="main_agent",
                content=[],
                action=AgentAction(
                    name="SubAgent",
                    parameters={"agent_name": "SubAgent"},
                    thought="Invoking SubAgent",
                ),
            ),
        ]
    )
    sub_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.AGENT_CALL,
                author="sub_agent",
                content=[],
                action=AgentAction(
                    name="SubSubAgent",
                    parameters={"agent_name": "SubSubAgent"},
                    thought="Invoking SubSubAgent",
                ),
            ),
            # consumed when SubAgent resumes after the grandchild's bounce
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="sub_agent",
                content=[Part(text="middle answer")],
                need_feedback=False,
            ),
        ]
    )
    sub_sub_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="sub_sub_agent",
                content=[Part(text="deep answer")],
                need_feedback=False,
            ),
        ]
    )

    input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="start")],
        turn_id="turn_1",
    )

    events = [
        e
        async for e in runner.run(
            conversation_id="test_flow_nested_sub_agent_bounce",
            application_name="TestApplication",
            user_id="test_user",
            root=main_agent,
            user_input=input_event,
        )
    ]

    assert [e.event_type for e in events] == [
        ExecutionEventType.AGENT_START,  # MainAgent
        ExecutionEventType.AGENT_CALL,  # main -> Sub
        ExecutionEventType.AGENT_START,  # SubAgent
        ExecutionEventType.AGENT_CALL,  # sub -> SubSub
        ExecutionEventType.AGENT_START,  # SubSubAgent
        ExecutionEventType.AGENT_END,  # SubSub bounce (deep answer)
        ExecutionEventType.AGENT_END,  # Sub bounce (middle answer)
    ]
    assert events[5].author == "sub_sub_agent"
    assert events[5].content[0].text == "deep answer"
    assert events[6].author == "sub_agent"
    assert events[6].content[0].text == "middle answer"
    assert not any(e.event_type == ExecutionEventType.RESPONSE for e in events)


@pytest.mark.asyncio
async def test_gating_a_tool_call_emits_its_thought_once(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """One thought in, one thought out, however many events the gate produces.

    A consumer that draws a row per thought-bearing event — the platform's reasoning
    panel does — prints the pair as the same line twice. The count is what this asserts,
    not which event carries it: the TOOL_START is the one that does, because it is also
    what an ungated call emits, so the panel reads the same either way.
    """

    def add_tool(a: int, b: int) -> int:
        return a + b

    main_agent = Agent(
        name="MainAgent",
        description="main agent description",
        instruction="main agent instruction",
        llm_provider=llm_provider,
        tools=[FunctionTool(func=add_tool, approval_type=ApprovalType.USER_CONFIRM)],
    )
    main_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.TOOL_START,
                author="main_agent",
                action=ToolAction(
                    name="add_tool", parameters={"a": 1, "b": 2}, thought="Invoking add_tool"
                ),
            ),
        ]
    )

    events = [
        e
        async for e in runner.run(
            conversation_id="test_gating_a_tool_call_emits_its_thought_once",
            application_name="TestApplication",
            user_id="test_user",
            root=main_agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                author="user",
                content=[Part(text="start process")],
                turn_id="turn_1",
            ),
        )
    ]

    thoughts = [
        (e.event_type, e.action.thought) for e in events if getattr(e.action, "thought", None)
    ]
    assert thoughts == [(ExecutionEventType.TOOL_START, "Invoking add_tool")]


@pytest.mark.asyncio
async def test_flow_with_feedback_tool(runner: Runner, llm_provider: BaseLLMProvider) -> None:
    """Test a flow"""

    def add_tool(a: int, b: int) -> int:
        return a + b

    main_agent = Agent(
        name="MainAgent",
        description="main agent description",
        instruction="main agent instruction",
        llm_provider=llm_provider,
        tools=[FunctionTool(func=add_tool, approval_type=ApprovalType.USER_CONFIRM)],
    )

    execute_events_main = [
        BaseEvent(
            event_type=ExecutionEventType.TOOL_START,
            author="main_agent",
            action=ToolAction(
                name="add_tool", parameters={"a": 1, "b": 2}, thought="Invoking add_tool"
            ),
        ),
        BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="main_agent",
            content=[Part(text="Result is 3")],
        ),
    ]

    main_agent._execute = async_gen_from_list(execute_events_main)

    application_name = "TestApplication"
    user_id = "test_user"
    conversation_id = "test_flow_with_feedback_tool"

    user_input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="start process")],
        turn_id="turn_1",
    )

    # create runner iterator
    runner_iterator = runner.run(
        conversation_id=conversation_id,
        application_name=application_name,
        user_id=user_id,
        root=main_agent,
        user_input=user_input_event,
    )

    events = [a async for a in runner_iterator]
    assert len(events) == 3  # agent_start main, tool_start, tool_needs_feedback

    assert events[1].event_type == ExecutionEventType.TOOL_START
    assert events[1].action.thought == "Invoking add_tool"

    assert events[2].event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK
    assert events[2].author == "MainAgent"
    assert events[2].action.name == "add_tool"
    assert isinstance(events[2].action, FeedbackAction)
    # the field validator transforms int to 'int' in FeedbackParameters
    assert events[2].action.parameters == [
        FeedbackParameters(name="a", value=1, type=int),
        FeedbackParameters(name="b", value=2, type="int"),
    ]
    # Carries no thought of its own: the TOOL_START above has it, and repeating it here
    # is the same line twice for a consumer that renders one row per thought.
    assert events[2].action.thought is None

    # turn_id: first turn
    turn_id_1 = events[0].turn_id
    assert turn_id_1 == "turn_1"
    assert all(e.turn_id == turn_id_1 for e in events)

    # provide feedback
    user_input_event = BaseEvent(
        event_type=ExecutionEventType.TOOL_FEEDBACK,
        author="user",
        action=FeedbackAction(
            name="add_tool",
            parameters=[
                FeedbackParameters(name="a", value=1, type=int),
                FeedbackParameters(name="b", value=2, type="int"),
            ],
            approval_type=ApprovalType.USER_CONFIRM,
            approve=True,
        ),
    )

    # second call
    runner_iterator = runner.run(
        conversation_id=conversation_id,
        application_name=application_name,
        user_id=user_id,
        root=main_agent,
        user_input=user_input_event,
    )
    events = [a async for a in runner_iterator]
    assert len(events) == 3  # TOOL_END, RESPONSE, AGENT_END
    assert events[0].event_type == ExecutionEventType.TOOL_END
    assert events[0].action.observation == 3

    # turn_id: TOOL_FEEDBACK resumes the same turn → same turn_id as first run
    assert all(e.turn_id == turn_id_1 for e in events)


@pytest.mark.asyncio
async def test_flow_user_confirm_executes_tool_start_params_not_feedback_echo(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """USER_CONFIRM ignores the values echoed back and executes the TOOL_START's.

    A USER_CONFIRM card is read-only: the user can only approve or refuse, so the
    parameters that come back are a round-trip of what was *displayed*. Hosts are
    allowed to rewrite a displayed value for readability (a host may beautify
    opaque commands and queries before forwarding the card), and that must never
    change what the tool runs.
    """

    def run_query(query: str) -> str:
        return f"ran:{query}"

    main_agent = Agent(
        name="MainAgent",
        description="main agent description",
        instruction="main agent instruction",
        llm_provider=llm_provider,
        tools=[FunctionTool(func=run_query, approval_type=ApprovalType.USER_CONFIRM)],
    )

    main_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.TOOL_START,
                author="main_agent",
                action=ToolAction(
                    name="run_query",
                    parameters={"query": "SELECT 1"},
                    thought="Invoking run_query",
                ),
            ),
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="main_agent",
                content=[Part(text="done")],
            ),
        ]
    )

    application_name = "TestApplication"
    user_id = "test_user"
    conversation_id = "test_flow_user_confirm_display_rewrite"

    events = [
        e
        async for e in runner.run(
            conversation_id=conversation_id,
            application_name=application_name,
            user_id=user_id,
            root=main_agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                author="user",
                content=[Part(text="start process")],
                turn_id="turn_1",
            ),
        )
    ]
    assert events[-1].event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK

    # The feedback echoes a *rewritten* value, as a display layer would.
    events = [
        e
        async for e in runner.run(
            conversation_id=conversation_id,
            application_name=application_name,
            user_id=user_id,
            root=main_agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.TOOL_FEEDBACK,
                author="user",
                action=FeedbackAction(
                    name="run_query",
                    parameters=[
                        FeedbackParameters(
                            name="query",
                            value="Reads a single constant row, changing nothing",
                            type="str",
                        )
                    ],
                    approval_type=ApprovalType.USER_CONFIRM,
                    approve=True,
                ),
            ),
        )
    ]

    assert events[0].event_type == ExecutionEventType.TOOL_END
    assert events[0].action.observation == "ran:SELECT 1"
    assert events[0].action.parameters == {"query": "SELECT 1"}
    # Parameters match the TOOL_START, so no "user updated tool parameters" nudge.
    assert events[0].action.observation is not None
    assert "update tool parameters" not in str(events[0].action.observation)


@pytest.mark.asyncio
async def test_flow_with_feedback_tool_params_update(runner: Runner, llm_provider: BaseLLMProvider):
    """Test a flow"""

    def add_tool(a: int, b: int) -> int:
        return a + b

    main_agent = Agent(
        name="MainAgent",
        description="main agent description",
        instruction="main agent instruction",
        llm_provider=llm_provider,
        tools=[FunctionTool(func=add_tool, approval_type=ApprovalType.PARAMS_UPDATE)],
    )

    execute_events_main = [
        BaseEvent(
            event_type=ExecutionEventType.LLM_START,
            author="main_agent",
            content=[],
            action=None,
        ),
        BaseEvent(
            event_type=ExecutionEventType.LLM_END,
            author="main_agent",
            content=[],
            action=None,
        ),
        BaseEvent(
            event_type=ExecutionEventType.TOOL_START,
            author="main_agent",
            content=[],
            action=ToolAction(
                name="add_tool", parameters={"a": 1, "b": 2}, thought="Invoking add_tool"
            ),
        ),
        BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="main_agent",
            content=[Part(text="Result is 4")],
        ),
    ]

    main_agent._execute = async_gen_from_list(execute_events_main)

    application_name = "TestApplication"
    user_id = "test_user"
    conversation_id = "test_flow_with_feedback_tool"

    user_input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="start process")],
        turn_id="turn_1",
    )

    # create runner iterator
    runner_iterator = runner.run(
        conversation_id=conversation_id,
        application_name=application_name,
        user_id=user_id,
        root=main_agent,
        user_input=user_input_event,
    )

    events = [a async for a in runner_iterator]
    # agent_start main, llm_start, llm_end, tool_start, tool_needs_feedback
    assert len(events) == 5

    assert events[-2].event_type == ExecutionEventType.TOOL_START
    assert events[-2].action.thought == "Invoking add_tool"

    assert events[-1].event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK
    assert events[-1].author == "MainAgent"
    assert events[-1].action.name == "add_tool"
    assert isinstance(events[-1].action, FeedbackAction)
    assert events[-1].action.parameters == [
        FeedbackParameters(name="a", value=1, type="int"),
        FeedbackParameters(name="b", value=2, type="int"),
    ]

    # turn_id: first turn
    turn_id_1 = events[0].turn_id
    assert turn_id_1 == "turn_1"
    assert all(e.turn_id == turn_id_1 for e in events)

    # provide feedback
    user_input_event = BaseEvent(
        event_type=ExecutionEventType.TOOL_FEEDBACK,
        author="user",
        action=FeedbackAction(
            name="add_tool",
            parameters=[
                FeedbackParameters(name="a", value=1, type="int"),
                FeedbackParameters(name="b", value=3, type="int"),
            ],
            approval_type=ApprovalType.PARAMS_UPDATE,
        ),
    )

    # second call
    runner_iterator = runner.run(
        conversation_id=conversation_id,
        application_name=application_name,
        user_id=user_id,
        root=main_agent,
        user_input=user_input_event,
    )
    events = [a async for a in runner_iterator]
    assert len(events) == 3  # TOOL_END, RESPONSE, AGENT_END
    assert events[0].event_type == ExecutionEventType.TOOL_END
    assert events[0].action.observation == 4

    # turn_id: TOOL_FEEDBACK resumes the same turn → same turn_id as first run
    assert all(e.turn_id == turn_id_1 for e in events)


@pytest.mark.asyncio
async def test_flow_with_feedback_tool_user_dont_confirm(
    runner: Runner, llm_provider: BaseLLMProvider
):
    """Test a flow"""

    def add_tool(a: int, b: int) -> int:
        return a + b

    main_agent = Agent(
        name="MainAgent",
        description="main agent description",
        instruction="main agent instruction",
        llm_provider=llm_provider,
        tools=[FunctionTool(func=add_tool, approval_type=ApprovalType.USER_CONFIRM)],
    )

    execute_events_main = [
        BaseEvent(
            event_type=ExecutionEventType.TOOL_START,
            author="main_agent",
            content=[],
            action=ToolAction(
                name="add_tool", parameters={"a": 1, "b": 2}, thought="Invoking add_tool"
            ),
        ),
        BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="main_agent",
            content=[Part(text="User Refused the tool execution")],
        ),
    ]

    main_agent._execute = async_gen_from_list(execute_events_main)

    application_name = "TestApplication"
    user_id = "test_user"
    conversation_id = "test_flow_with_feedback_tool"

    user_input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="start process")],
        turn_id="turn_1",
    )

    # create runner iterator
    runner_iterator = runner.run(
        conversation_id=conversation_id,
        application_name=application_name,
        user_id=user_id,
        root=main_agent,
        user_input=user_input_event,
    )

    events = [a async for a in runner_iterator]
    assert len(events) == 3  # agent_start main, tool_start, tool_needs_feedback

    assert events[1].event_type == ExecutionEventType.TOOL_START
    assert events[1].action.thought == "Invoking add_tool"

    assert events[2].event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK
    assert events[2].author == "MainAgent"
    assert events[2].action.name == "add_tool"
    assert isinstance(events[2].action, FeedbackAction)
    assert events[2].action.parameters == [
        FeedbackParameters(name="a", value=1, type="int"),
        FeedbackParameters(name="b", value=2, type="int"),
    ]

    # turn_id: first turn
    turn_id_1 = events[0].turn_id
    assert turn_id_1 == "turn_1"
    assert all(e.turn_id == turn_id_1 for e in events)

    # provide feedback
    user_input_event = BaseEvent(
        event_type=ExecutionEventType.TOOL_FEEDBACK,
        author="user",
        action=FeedbackAction(
            name="add_tool",
            parameters=[
                FeedbackParameters(name="a", value=1, type="int"),
                FeedbackParameters(name="b", value=2, type="int"),
            ],
            approval_type=ApprovalType.USER_CONFIRM,
            approve=False,
        ),
    )

    # second call
    runner_iterator = runner.run(
        conversation_id=conversation_id,
        application_name=application_name,
        user_id=user_id,
        root=main_agent,
        user_input=user_input_event,
    )
    events = [a async for a in runner_iterator]
    assert len(events) == 3  # TOOL_END, RESPONSE, AGENT_END
    assert events[0].event_type == ExecutionEventType.TOOL_END
    assert "refused" in str(events[0].action.observation).lower()
    assert events[1].event_type == ExecutionEventType.RESPONSE

    # turn_id: TOOL_FEEDBACK resumes the same turn → same turn_id as first run
    assert all(e.turn_id == turn_id_1 for e in events)


@pytest.mark.asyncio
async def test_flow_pre_hook_break_execution(runner: Runner, llm_provider: BaseLLMProvider):
    """Test that a SubAgent's pre-hook with break_execution=True stops the sub-agent before it runs.

    MainAgent's _execute emits AGENT_START (MainAgent -> SubAgent).
    SubAgent has a breaking pre-hook that fires before its _execute is invoked.

    Expected event sequence:
    1. AGENT_START  (user -> MainAgent)
    2. HOOK_START   (SubAgent pre-hook)
    3. HOOK_END     (SubAgent pre-hook)
    4. RESPONSE     (break_loop_response emitted by run_all_agent_hooks)
    5. AGENT_START  (MainAgent -> SubAgent, yielded at the bottom of the runner loop)
    SubAgent's _execute must NOT be iterated.
    """

    call_count = {"executed": 0}

    async def breaking_pre_hook(context, agent) -> HookResult:
        return HookResult(
            success=True,
            emitted_content=[
                Part(text="Execution stopped by pre-hook"),
                Part(metadata={"some": "metadata"}),
            ],
            error=None,
            break_execution=True,
            name="breaking_pre_hook",
            new_context=None,
        )

    breaking_pre_hook.__signature__ = __import__("inspect").Signature(
        [
            __import__("inspect").Parameter(
                "context", __import__("inspect").Parameter.POSITIONAL_OR_KEYWORD
            ),
            __import__("inspect").Parameter(
                "agent", __import__("inspect").Parameter.POSITIONAL_OR_KEYWORD
            ),
        ]
    )

    pre = FunctionAgentHook(
        name="BreakingPreHook",
        fun=breaking_pre_hook,
        trigger=HookPhase.PRE_AGENT_CALL,
        type=HookType.LISTENER,
    )

    sub_agent = Agent(
        name="SubAgent",
        description="subagent description",
        instruction="subagent instruction",
        llm_provider=llm_provider,
        hooks=[pre],
    )

    main_agent = Agent(
        name="MainAgent",
        description="main agent description",
        instruction="main agent instruction",
        llm_provider=llm_provider,
        sub_agents=[sub_agent],
    )

    main_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.AGENT_CALL,
                author="MainAgent",
                content=[],
                action=AgentAction(
                    name=sub_agent.name,
                    parameters={"agent_name": sub_agent.name},
                    thought="Invoking SubAgent",
                ),
            ),
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author=sub_agent.name,
                content=[Part(text="Agent final response. This should not be reached.")],
                is_partial=False,
                need_feedback=False,
            ),
            BaseEvent(event_type=ExecutionEventType.AGENT_END, author=sub_agent.name, content=[]),
        ]
    )
    sub_agent._execute = async_gen_from_list([])  # should never be iterated

    application_name = "TestApplication"
    user_id = "test_user"
    conversation_id = "test_flow_pre_hook_break_execution"

    input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="Start the process")],
        turn_id="turn_1",
    )

    runner_iterator = runner.run(
        conversation_id=conversation_id,
        application_name=application_name,
        user_id=user_id,
        root=main_agent,
        user_input=input_event,
    )

    events = [a async for a in runner_iterator]

    # AGENT_START (user->MainAgent), AGENT_CALL (MainAgent->SubAgent), AGENT_START (SubAgent, runner-created),
    # HOOK_START (SubAgent pre-hook), HOOK_END, RESPONSE (break)
    assert len(events) == 6, (
        f"Expected 6 events, got {len(events)}: {[e.event_type for e in events]}"
    )
    assert events[0].event_type == ExecutionEventType.AGENT_START
    assert events[0].author == "MainAgent"
    # strategy emits AGENT_CALL
    assert events[1].event_type == ExecutionEventType.AGENT_CALL
    assert events[1].author == "MainAgent"
    assert events[1].action.name == "SubAgent"
    # runner creates AGENT_START for the called sub-agent
    assert events[2].event_type == ExecutionEventType.AGENT_START
    assert events[2].author == "SubAgent"
    assert events[2].action.name == "SubAgent"
    # SubAgent's pre-hook fires
    assert events[3].event_type == ExecutionEventType.HOOK_START
    assert events[3].metadata["hook_name"] == "BreakingPreHook"
    assert events[4].event_type == ExecutionEventType.HOOK_END
    assert events[4].metadata["hook_name"] == "breaking_pre_hook"
    assert events[4].content == [
        Part(text="Execution stopped by pre-hook"),
        Part(metadata={"some": "metadata"}),
    ]
    # The break_loop_response is a RESPONSE event with the hook's message
    assert events[5].event_type == ExecutionEventType.RESPONSE
    assert events[5].content[0].text == "Execution stopped by pre-hook"
    # SubAgent's _execute must NOT have been iterated
    assert call_count["executed"] == 0

    # turn_id: all events in this single turn must share the same turn_id
    turn_id = events[0].turn_id
    assert turn_id == "turn_1"
    assert all(e.turn_id == turn_id for e in events)


@pytest.mark.asyncio
async def test_flow_post_hook_break_execution(runner: Runner, llm_provider: BaseLLMProvider):
    """Test that a post-hook with break_execution=True stops execution after the agent responds.

    Expected event sequence:
    1. AGENT_START  (user -> MainAgent)
    2. RESPONSE     (agent final response)
    3. HOOK_START   (post-hook, inside try/finally)
    4. HOOK_END     (post-hook, inside try/finally)
    5. RESPONSE     (break_loop_response from try/finally block)
    6. HOOK_START   (post-hook, second run_all_agent_hooks call)
    7. HOOK_END     (post-hook, second run_all_agent_hooks call)
    8. RESPONSE     (break_loop_response from second block)
    9. AGENT_END
    """
    from domyn.agents.hooks import HookResult

    async def breaking_post_hook(context, agent) -> HookResult:
        return HookResult(
            success=True,
            emitted_content=[
                Part(text="Execution stopped by post-hook"),
                Part(metadata={"some": "metadata"}),
            ],
            error=None,
            break_execution=True,
            name="breaking_post_hook",
            new_context=None,
        )

    breaking_post_hook.__signature__ = __import__("inspect").Signature(
        [
            __import__("inspect").Parameter(
                "context", __import__("inspect").Parameter.POSITIONAL_OR_KEYWORD
            ),
            __import__("inspect").Parameter(
                "agent", __import__("inspect").Parameter.POSITIONAL_OR_KEYWORD
            ),
        ]
    )

    post = FunctionAgentHook(
        name="BreakingPostHook",
        fun=breaking_post_hook,
        trigger=HookPhase.POST_AGENT_CALL,
        type=HookType.LISTENER,
    )

    sub_agent = Agent(
        name="SubAgent",
        description="subagent description",
        instruction="subagent instruction",
        llm_provider=llm_provider,
    )

    main_agent = Agent(
        name="MainAgent",
        description="main agent description",
        instruction="main agent instruction",
        llm_provider=llm_provider,
        hooks=[post],
        sub_agents=[sub_agent],
    )

    agent_execution_events = [
        BaseEvent(
            event_type=ExecutionEventType.AGENT_CALL,
            author="MainAgent",
            content=[],
            action=AgentAction(
                name=sub_agent.name,
                parameters={"agent_name": sub_agent.name},
                thought="Invoking SubAgent",
            ),
        ),
        BaseEvent(
            event_type=ExecutionEventType.AGENT_END, author="SubAgent", content=[], action=None
        ),
        BaseEvent(
            event_type=ExecutionEventType.AGENT_CALL,
            author="SubAgent",
            content=[],
            action=AgentAction(
                name=main_agent.name,
                parameters={"agent_name": main_agent.name},
                thought="Invoking SubAgent",
            ),
        ),
        BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="MainAgent",
            content=[Part(text="Agent final response. This should not be reached.")],
            is_partial=False,
            need_feedback=False,
        ),
        BaseEvent(event_type=ExecutionEventType.AGENT_END, author="MainAgent", content=[]),
    ]
    main_agent._execute = async_gen_from_list(agent_execution_events)

    application_name = "TestApplication"
    user_id = "test_user"
    conversation_id = "test_flow_post_hook_break_execution"

    input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="Start the process")],
        turn_id="turn_1",
    )

    runner_iterator = runner.run(
        conversation_id=conversation_id,
        application_name=application_name,
        user_id=user_id,
        root=main_agent,
        user_input=input_event,
    )

    events = [a async for a in runner_iterator]

    assert len(events) == 4
    assert events[0].event_type == ExecutionEventType.AGENT_START
    assert events[1].event_type == ExecutionEventType.HOOK_START
    assert events[2].event_type == ExecutionEventType.HOOK_END
    assert events[2].content == [
        Part(text="Execution stopped by post-hook"),
        Part(metadata={"some": "metadata"}),
    ]
    assert events[3].event_type == ExecutionEventType.RESPONSE
    assert events[3].content[0].text == "Execution stopped by post-hook"

    # turn_id: all events in this single turn must share the same turn_id
    turn_id = events[0].turn_id
    assert turn_id == "turn_1"
    assert all(e.turn_id == turn_id for e in events)


@pytest.mark.asyncio
async def test_flow_break_loop_response_guardrail_reason(
    runner: Runner, llm_provider: BaseLLMProvider
):
    """Test break_loop_response_event prioritization: guardrail reason (Priority 1).

    When a hook returns event_content with a Part that has a guardrail_result where
    passed=False and reason is set, the RESPONSE event's text must be the guardrail reason,
    even if a Part with text also exists later in the list.

    Expected event sequence:
    1. AGENT_START  (user -> MainAgent)
    2. HOOK_START   (pre-hook)
    3. HOOK_END     (pre-hook)
    4. RESPONSE     (text == guardrail reason, not the plain text part)
    5. AGENT_START  (MainAgent -> SubAgent, yielded at the bottom of the runner loop)
    """
    from domyn.agents.core import GuardrailResult
    from domyn.agents.hooks import HookResult

    guardrail_reason = "Content blocked by safety guardrail"
    plain_text = "This plain text should NOT be used"

    async def guardrail_pre_hook(context, agent) -> HookResult:
        return HookResult(
            success=True,
            emitted_content=[
                Part(
                    guardrail_result=GuardrailResult(
                        name="safety_check",
                        score=0.95,
                        reason=guardrail_reason,
                        passed=False,
                        error_occurred=False,
                    )
                ),
                Part(text=plain_text),  # should be ignored because guardrail reason takes priority
            ],
            error=None,
            break_execution=True,
            name="guardrail_pre_hook",
            new_context=None,
        )

    guardrail_pre_hook.__signature__ = __import__("inspect").Signature(
        [
            __import__("inspect").Parameter(
                "context", __import__("inspect").Parameter.POSITIONAL_OR_KEYWORD
            ),
            __import__("inspect").Parameter(
                "agent", __import__("inspect").Parameter.POSITIONAL_OR_KEYWORD
            ),
        ]
    )

    pre = FunctionAgentHook(
        name="GuardrailPreHook",
        fun=guardrail_pre_hook,
        trigger=HookPhase.PRE_AGENT_CALL,
        type=HookType.LISTENER,
    )

    sub_agent = Agent(
        name="SubAgent",
        description="subagent description",
        instruction="subagent instruction",
        llm_provider=llm_provider,
        hooks=[pre],
    )

    main_agent = Agent(
        name="MainAgent",
        description="main agent description",
        instruction="main agent instruction",
        llm_provider=llm_provider,
        sub_agents=[sub_agent],
    )

    main_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.AGENT_CALL,
                author="MainAgent",
                content=[],
                action=AgentAction(
                    name=sub_agent.name,
                    parameters={"agent_name": sub_agent.name},
                    thought="Invoking SubAgent",
                ),
            ),
        ]
    )
    sub_agent._execute = async_gen_from_list([])  # should never be iterated

    input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="Start the process")],
        turn_id="turn_1",
    )

    runner_iterator = runner.run(
        conversation_id="test_break_loop_guardrail_reason",
        application_name="TestApplication",
        user_id="test_user",
        root=main_agent,
        user_input=input_event,
    )

    events = [a async for a in runner_iterator]

    # AGENT_START (user->MainAgent), AGENT_CALL (MainAgent->SubAgent), AGENT_START (SubAgent, runner-created),
    # HOOK_START (SubAgent pre-hook), HOOK_END, RESPONSE (guardrail reason)
    assert len(events) == 6, (
        f"Expected 6 events, got {len(events)}: {[e.event_type for e in events]}"
    )
    assert events[0].event_type == ExecutionEventType.AGENT_START
    assert events[0].author == "MainAgent"
    assert events[1].event_type == ExecutionEventType.AGENT_CALL
    assert events[1].author == "MainAgent"
    assert events[1].action.name == "SubAgent"
    assert events[2].event_type == ExecutionEventType.AGENT_START
    assert events[2].author == "SubAgent"
    assert events[3].event_type == ExecutionEventType.HOOK_START
    assert events[3].metadata["hook_name"] == "GuardrailPreHook"
    assert events[4].event_type == ExecutionEventType.HOOK_END
    # Priority 1: RESPONSE text must be the guardrail reason, NOT the plain text part
    assert events[5].event_type == ExecutionEventType.RESPONSE
    assert events[5].content[0].text == guardrail_reason
    assert events[5].content[0].text != plain_text

    # turn_id: all events in this single turn must share the same turn_id
    turn_id = events[0].turn_id
    assert turn_id == "turn_1"
    assert all(e.turn_id == turn_id for e in events)


@pytest.mark.asyncio
async def test_flow_break_loop_response_guardrail_passed_falls_through_to_text(
    runner: Runner, llm_provider: BaseLLMProvider
):
    """Test break_loop_response_event prioritization: guardrail with passed=True is skipped (Priority 2).

    When a hook returns a guardrail_result with passed=True, that guardrail is NOT a failure
    and its reason must NOT be used.  The next Part with plain text is used instead.

    Expected event sequence:
    1. AGENT_START  (user -> MainAgent)
    2. HOOK_START   (pre-hook)
    3. HOOK_END     (pre-hook)
    4. RESPONSE     (text == plain-text part, not the guardrail reason)
    5. AGENT_START  (MainAgent -> SubAgent, yielded at the bottom of the runner loop)
    """
    from domyn.agents.core import GuardrailResult
    from domyn.agents.hooks import HookResult

    guardrail_reason = "This guardrail passed, should be ignored"
    plain_text = "This plain text SHOULD be used because guardrail passed"

    async def passing_guardrail_hook(context, agent) -> HookResult:
        return HookResult(
            success=True,
            emitted_content=[
                Part(
                    guardrail_result=GuardrailResult(
                        name="safety_check",
                        score=0.1,
                        reason=guardrail_reason,
                        passed=True,  # passed=True → should NOT trigger guardrail priority
                        error_occurred=False,
                    )
                ),
                Part(text=plain_text),
            ],
            error=None,
            break_execution=True,
            name="passing_guardrail_hook",
            new_context=None,
        )

    passing_guardrail_hook.__signature__ = __import__("inspect").Signature(
        [
            __import__("inspect").Parameter(
                "context", __import__("inspect").Parameter.POSITIONAL_OR_KEYWORD
            ),
            __import__("inspect").Parameter(
                "agent", __import__("inspect").Parameter.POSITIONAL_OR_KEYWORD
            ),
        ]
    )

    pre = FunctionAgentHook(
        name="PassingGuardrailHook",
        fun=passing_guardrail_hook,
        trigger=HookPhase.PRE_AGENT_CALL,
        type=HookType.LISTENER,
    )

    sub_agent = Agent(
        name="SubAgent",
        description="subagent description",
        instruction="subagent instruction",
        llm_provider=llm_provider,
        hooks=[pre],
    )

    main_agent = Agent(
        name="MainAgent",
        description="main agent description",
        instruction="main agent instruction",
        llm_provider=llm_provider,
        sub_agents=[sub_agent],
    )

    main_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.AGENT_CALL,
                author="MainAgent",
                content=[],
                action=AgentAction(
                    name=sub_agent.name,
                    parameters={"agent_name": sub_agent.name},
                    thought="Invoking SubAgent",
                ),
            ),
        ]
    )
    sub_agent._execute = async_gen_from_list([])

    input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="Start the process")],
        turn_id="turn_1",
    )

    runner_iterator = runner.run(
        conversation_id="test_break_loop_guardrail_passed_falls_through",
        application_name="TestApplication",
        user_id="test_user",
        root=main_agent,
        user_input=input_event,
    )

    events = [a async for a in runner_iterator]

    assert len(events) == 6, (
        f"Expected 6 events, got {len(events)}: {[e.event_type for e in events]}"
    )
    assert events[5].event_type == ExecutionEventType.RESPONSE
    # Priority 2: plain text is used because the guardrail passed
    assert events[5].content[0].text == plain_text
    assert events[5].content[0].text != guardrail_reason

    # turn_id: all events in this single turn must share the same turn_id
    turn_id = events[0].turn_id
    assert turn_id == "turn_1"
    assert all(e.turn_id == turn_id for e in events)


@pytest.mark.asyncio
async def test_flow_break_loop_response_guardrail_no_reason_falls_through_to_text(
    runner: Runner, llm_provider: BaseLLMProvider
):
    """Test break_loop_response_event prioritization: guardrail failed but no reason → falls to text (Priority 2).

    When a hook returns a guardrail_result with passed=False but reason=None,
    the logic must skip to Priority 2 and use the plain-text Part instead.
    """
    from domyn.agents.core import GuardrailResult
    from domyn.agents.hooks import HookResult

    plain_text = "Fallback text because guardrail has no reason"

    async def no_reason_guardrail_hook(context, agent) -> HookResult:
        return HookResult(
            success=True,
            emitted_content=[
                Part(
                    guardrail_result=GuardrailResult(
                        name="safety_check",
                        score=0.9,
                        reason=None,  # no reason → skip Priority 1
                        passed=False,
                        error_occurred=False,
                    )
                ),
                Part(text=plain_text),
            ],
            error=None,
            break_execution=True,
            name="no_reason_guardrail_hook",
            new_context=None,
        )

    no_reason_guardrail_hook.__signature__ = __import__("inspect").Signature(
        [
            __import__("inspect").Parameter(
                "context", __import__("inspect").Parameter.POSITIONAL_OR_KEYWORD
            ),
            __import__("inspect").Parameter(
                "agent", __import__("inspect").Parameter.POSITIONAL_OR_KEYWORD
            ),
        ]
    )

    pre = FunctionAgentHook(
        name="NoReasonGuardrailHook",
        fun=no_reason_guardrail_hook,
        trigger=HookPhase.PRE_AGENT_CALL,
        type=HookType.LISTENER,
    )

    sub_agent = Agent(
        name="SubAgent",
        description="subagent description",
        instruction="subagent instruction",
        llm_provider=llm_provider,
        hooks=[pre],
    )

    main_agent = Agent(
        name="MainAgent",
        description="main agent description",
        instruction="main agent instruction",
        llm_provider=llm_provider,
        sub_agents=[sub_agent],
    )

    main_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.AGENT_CALL,
                author="MainAgent",
                content=[],
                action=AgentAction(
                    name=sub_agent.name,
                    parameters={"agent_name": sub_agent.name},
                    thought="Invoking SubAgent",
                ),
            ),
        ]
    )
    sub_agent._execute = async_gen_from_list([])

    input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="Start the process")],
        turn_id="turn_1",
    )

    runner_iterator = runner.run(
        conversation_id="test_break_loop_guardrail_no_reason",
        application_name="TestApplication",
        user_id="test_user",
        root=main_agent,
        user_input=input_event,
    )

    events = [a async for a in runner_iterator]

    assert len(events) == 6, (
        f"Expected 6 events, got {len(events)}: {[e.event_type for e in events]}"
    )
    assert events[5].event_type == ExecutionEventType.RESPONSE
    # Priority 2: plain text is used because guardrail reason is None
    assert events[5].content[0].text == plain_text

    # turn_id: all events in this single turn must share the same turn_id
    turn_id = events[0].turn_id
    assert turn_id == "turn_1"
    assert all(e.turn_id == turn_id for e in events)


@pytest.mark.asyncio
async def test_flow_break_loop_response_default_message(
    runner: Runner, llm_provider: BaseLLMProvider
):
    """Test break_loop_response_event prioritization: default message (Priority 3).

    When a hook breaks execution with event_content that contains no guardrail reason
    and no plain text, the RESPONSE event must fall back to the default message
    f"Execution stopped by hook {hook_result.name}".
    """
    from domyn.agents.hooks import HookResult

    hook_name = "no_content_hook"

    async def no_content_hook(context, agent) -> HookResult:
        return HookResult(
            success=True,
            emitted_content=[
                Part(metadata={"some": "metadata"}),  # no text, no guardrail_result
            ],
            error=None,
            break_execution=True,
            name=hook_name,
            new_context=None,
        )

    no_content_hook.__signature__ = __import__("inspect").Signature(
        [
            __import__("inspect").Parameter(
                "context", __import__("inspect").Parameter.POSITIONAL_OR_KEYWORD
            ),
            __import__("inspect").Parameter(
                "agent", __import__("inspect").Parameter.POSITIONAL_OR_KEYWORD
            ),
        ]
    )

    pre = FunctionAgentHook(
        name="NoContentHook",
        fun=no_content_hook,
        trigger=HookPhase.PRE_AGENT_CALL,
        type=HookType.LISTENER,
    )

    sub_agent = Agent(
        name="SubAgent",
        description="subagent description",
        instruction="subagent instruction",
        llm_provider=llm_provider,
        hooks=[pre],
    )

    main_agent = Agent(
        name="MainAgent",
        description="main agent description",
        instruction="main agent instruction",
        llm_provider=llm_provider,
        sub_agents=[sub_agent],
    )

    main_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.AGENT_CALL,
                author="MainAgent",
                content=[],
                action=AgentAction(
                    name=sub_agent.name,
                    parameters={"agent_name": sub_agent.name},
                    thought="Invoking SubAgent",
                ),
            ),
        ]
    )
    sub_agent._execute = async_gen_from_list([])

    input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="Start the process")],
        turn_id="turn_1",
    )

    runner_iterator = runner.run(
        conversation_id="test_break_loop_default_message",
        application_name="TestApplication",
        user_id="test_user",
        root=main_agent,
        user_input=input_event,
    )

    events = [a async for a in runner_iterator]

    assert len(events) == 6, (
        f"Expected 6 events, got {len(events)}: {[e.event_type for e in events]}"
    )
    assert events[5].event_type == ExecutionEventType.RESPONSE
    # Priority 3: default message because there is no text and no guardrail reason
    assert events[5].content[0].text == f"Execution stopped by hook {hook_name}"

    # turn_id: all events in this single turn must share the same turn_id
    turn_id = events[0].turn_id
    assert turn_id == "turn_1"
    assert all(e.turn_id == turn_id for e in events)


@pytest.mark.asyncio
async def test_flow_nested_subagents_stop(runner: Runner, llm_provider: BaseLLMProvider):
    """Test nested sub-agents where Main -> A -> B; B ends, then A ends, then Main ends.
    Verifies authors and that AGENT_END unwinds to the caller at each step."""

    # Define agents
    sub_agent_b = Agent(
        name="SubAgentB",
        description="subagent B",
        instruction="instruction B",
        llm_provider=llm_provider,
    )

    sub_agent_a = Agent(
        name="SubAgentA",
        description="subagent A",
        instruction="instruction A",
        sub_agents=[sub_agent_b],
        llm_provider=llm_provider,
    )

    main_agent = Agent(
        name="MainAgent",
        description="main",
        instruction="main instruction",
        sub_agents=[sub_agent_a],
        llm_provider=llm_provider,
    )

    main_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.AGENT_CALL,
                author=main_agent.name,
                content=[],
                action=AgentAction(
                    name=sub_agent_a.name,
                    parameters={"agent_name": sub_agent_a.name},
                    thought="Invoking SubAgentA",
                ),
            ),
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author=main_agent.name,
                content=[Part(text="Main doing something")],
                action=AgentAction(name="RESPONSE", parameters={}, thought="Main response"),
            ),
        ]
    )

    sub_agent_a._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.AGENT_CALL,
                author=sub_agent_a.name,
                content=[],
                action=AgentAction(
                    name=sub_agent_b.name,
                    parameters={"agent_name": sub_agent_b.name},
                    thought="Invoking SubAgentB",
                ),
            ),
            BaseEvent(
                event_type=ExecutionEventType.AGENT_END,
                author=sub_agent_a.name,
                content=[Part(thought="A finished")],
            ),
        ]
    )

    sub_agent_b._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.AGENT_END,
                author=sub_agent_b.name,
                content=[Part(thought="B finished")],
            )
        ]
    )

    application_name = "TestApplication"
    user_id = "test_user"
    conversation_id = "test_flow_nested_subagents_stop"

    user_input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="start nested call")],
        turn_id="turn_1",
    )

    # Run
    runner_iterator = runner.run(
        conversation_id=conversation_id,
        application_name=application_name,
        user_id=user_id,
        root=main_agent,
        user_input=user_input_event,
    )

    # 1) Runner emits start for user -> MainAgent
    e1 = await anext(runner_iterator)
    assert e1.event_type == ExecutionEventType.AGENT_START and e1.author == "MainAgent"

    # turn_id is stable for all events in this single turn
    turn_id = e1.turn_id
    assert turn_id == "turn_1"

    # stack should have MainAgent
    top = await runner.conversation_manager.get_last_agent_stack(
        application_name=application_name,
        user_id=user_id,
        conversation_id=conversation_id,
    )
    assert top == StackAgentEntry(
        agent_name="MainAgent", interaction_id=e1.interaction_id, turn_id=turn_id
    )

    # 2) Main calls A (strategy-emitted AGENT_CALL)
    e2 = await anext(runner_iterator)
    assert e2.event_type == ExecutionEventType.AGENT_CALL and e2.author == "MainAgent"
    assert e2.action.name == "SubAgentA"
    assert e2.turn_id == turn_id

    # 3) Runner-created AGENT_START for SubAgentA; stack is updated after this
    e3 = await anext(runner_iterator)
    assert e3.event_type == ExecutionEventType.AGENT_START and e3.author == "SubAgentA"
    assert e3.action.name == "SubAgentA"
    assert e3.turn_id == turn_id
    top = await runner.conversation_manager.get_last_agent_stack(
        application_name=application_name,
        user_id=user_id,
        conversation_id=conversation_id,
    )
    assert top.agent_name == "SubAgentA"

    # 4) A calls B (strategy-emitted AGENT_CALL from SubAgentA)
    e4 = await anext(runner_iterator)
    assert e4.event_type == ExecutionEventType.AGENT_CALL and e4.author == "SubAgentA"
    assert e4.action.name == "SubAgentB"
    assert e4.turn_id == turn_id

    # 5) Runner-created AGENT_START for SubAgentB; stack updated
    e5 = await anext(runner_iterator)
    assert e5.event_type == ExecutionEventType.AGENT_START and e5.author == "SubAgentB"
    assert e5.action.name == "SubAgentB"
    assert e5.turn_id == turn_id
    top = await runner.conversation_manager.get_last_agent_stack(
        application_name=application_name,
        user_id=user_id,
        conversation_id=conversation_id,
    )
    assert top.agent_name == "SubAgentB"

    # 6) B ends -> returns to A
    e6 = await anext(runner_iterator)
    assert e6.event_type == ExecutionEventType.AGENT_END and e6.author == "SubAgentB"
    assert e6.turn_id == turn_id
    top = await runner.conversation_manager.get_last_agent_stack(
        application_name=application_name,
        user_id=user_id,
        conversation_id=conversation_id,
    )
    assert top == StackAgentEntry(
        agent_name="SubAgentA", interaction_id=e3.interaction_id, turn_id=turn_id
    )

    # 7) A ends -> returns to Main
    e7 = await anext(runner_iterator)
    assert e7.event_type == ExecutionEventType.AGENT_END and e7.author == "SubAgentA"
    assert e7.turn_id == turn_id
    top = await runner.conversation_manager.get_last_agent_stack(
        application_name=application_name,
        user_id=user_id,
        conversation_id=conversation_id,
    )
    assert top.agent_name == "MainAgent"

    # 8) Main ends -> stack emptied
    e8 = await anext(runner_iterator)
    assert e8.event_type == ExecutionEventType.RESPONSE and e8.author == "MainAgent"
    assert e8.turn_id == turn_id
    assert (
        await runner.conversation_manager.get_last_agent_stack(
            application_name=application_name,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        is None
    )
    e9 = await anext(runner_iterator)
    assert e9.event_type == ExecutionEventType.AGENT_END and e9.author == "MainAgent"
    assert e9.turn_id == turn_id
    with pytest.raises(StopAsyncIteration):
        await anext(runner_iterator)


@pytest.mark.asyncio
async def test_flow_streamable_listener_hook(runner: Runner, llm_provider: BaseLLMProvider):
    """Test a flow where an agent has a streamable listener pre-hook.

    The hook yields intermediate StreamableHookResult chunks followed by a final
    HookResult (no new_context, as it is a listener).

    Expected event sequence:
    1. AGENT_START  (user -> MainAgent)
    2. HOOK_START   (streamable listener pre-hook)
    3. HOOK_UPDATE  (first streaming chunk)
    4. HOOK_UPDATE  (second streaming chunk)
    5. HOOK_END     (final HookResult from the hook)
    6. RESPONSE     (agent final response)
    7. AGENT_END
    """
    from collections.abc import AsyncGenerator

    from domyn.agents.hooks import (
        HookResult,
    )

    chunk_texts = ["chunk one", "chunk two"]
    listener_called_with: dict = {}

    async def streamable_listener(context: InteractionContext, agent) -> AsyncGenerator:
        listener_called_with["conversation_id"] = context.conversation_id
        for text in chunk_texts:
            yield HookResult(
                new_context=None,
                emitted_content=[Part(text=text)],
                success=True,
                error=None,
                name=None,
                is_update=True,
            )
        # No final HookResult yielded: drain() will synthesise a successful one

    streamable_listener.__signature__ = __import__("inspect").Signature(
        [
            __import__("inspect").Parameter(
                "context", __import__("inspect").Parameter.POSITIONAL_OR_KEYWORD
            ),
            __import__("inspect").Parameter(
                "agent", __import__("inspect").Parameter.POSITIONAL_OR_KEYWORD
            ),
        ]
    )

    pre = FunctionAgentHook(
        name="StreamableListenerHook",
        fun=streamable_listener,
        trigger=HookPhase.PRE_AGENT_CALL,
        type=HookType.LISTENER,
    )

    main_agent = Agent(
        name="MainAgent",
        description="main agent description",
        instruction="main agent instruction",
        llm_provider=llm_provider,
        hooks=[pre],
    )

    agent_execution_events = [
        BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="MainAgent",
            content=[Part(text="Final response after streamable listener hook.")],
            is_partial=False,
            need_feedback=False,
        ),
        BaseEvent(event_type=ExecutionEventType.AGENT_END, author="MainAgent", content=[]),
    ]
    main_agent._execute = async_gen_from_list(agent_execution_events)

    application_name = "TestApplication"
    user_id = "test_user"
    conversation_id = "test_flow_streamable_listener_hook"

    input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="Start the process")],
    )

    runner_iterator = runner.run(
        conversation_id=conversation_id,
        application_name=application_name,
        user_id=user_id,
        root=main_agent,
        user_input=input_event,
    )

    events = [a async for a in runner_iterator]

    # AGENT_START, HOOK_START, HOOK_UPDATE x2, HOOK_END, RESPONSE, AGENT_END
    assert len(events) == 7, (
        f"Expected 7 events, got {len(events)}: {[e.event_type for e in events]}"
    )
    assert events[0].event_type == ExecutionEventType.AGENT_START
    assert events[0].author == "MainAgent"

    assert events[1].event_type == ExecutionEventType.HOOK_START
    assert events[1].metadata["hook_name"] == "StreamableListenerHook"
    assert events[1].author == "MainAgent"

    assert events[2].event_type == ExecutionEventType.HOOK_UPDATE
    assert events[2].content[0].text == "chunk one"
    assert events[2].author == "MainAgent"

    assert events[3].event_type == ExecutionEventType.HOOK_UPDATE
    assert events[3].content[0].text == "chunk two"
    assert events[3].author == "MainAgent"

    # Synthesised final HOOK_END from drain()
    assert events[4].event_type == ExecutionEventType.HOOK_END
    assert events[4].metadata["hook_name"] == "StreamableListenerHook"
    assert events[4].author == "MainAgent"

    assert events[5].event_type == ExecutionEventType.RESPONSE
    assert events[5].content[0].text == "Final response after streamable listener hook."

    assert events[6].event_type == ExecutionEventType.AGENT_END

    # Verify the hook received the correct context
    assert listener_called_with["conversation_id"] == conversation_id


@pytest.mark.asyncio
async def test_flow_streamable_updater_hook_updates_context(
    runner: Runner, llm_provider: BaseLLMProvider
):
    """Test a flow where an agent has a streamable updater pre-hook.

    The hook yields intermediate StreamableHookResult chunks and then a final
    HookResult carrying a new InteractionContext with an updated ``state``.

    Expected event sequence:
    1. AGENT_START  (user -> MainAgent)
    2. HOOK_START   (streamable updater pre-hook)
    3. HOOK_UPDATE  (first streaming chunk)
    4. HOOK_UPDATE  (second streaming chunk)
    5. HOOK_END     (final HookResult -context updated)
    6. RESPONSE     (agent final response)
    7. AGENT_END

    The key assertion is that the context passed to the agent's _execute reflects
    the updated state set by the updater hook.
    """
    import copy
    from collections.abc import AsyncGenerator

    from domyn.agents.hooks import HookResult

    captured_context: dict = {}
    chunk_texts = ["processing…", "almost done…"]
    updated_state = {"hook_key": "hook_value", "processed": True}

    async def streamable_updater(context: InteractionContext, agent: BaseAgent) -> AsyncGenerator:
        for text in chunk_texts:
            yield HookResult(
                new_context=None,
                emitted_content=[Part(text=text)],
                success=True,
                error=None,
                name=None,
                is_update=True,
            )
        # Final item: HookResult with a new_context that carries updated state
        new_ctx = copy.deepcopy(context)
        new_ctx.state.update(updated_state)
        yield HookResult(
            new_context=new_ctx,
            emitted_content=[Part(text="Context updated by streamable updater.")],
            success=True,
            error=None,
        )

    streamable_updater.__signature__ = __import__("inspect").Signature(
        [
            __import__("inspect").Parameter(
                "context", __import__("inspect").Parameter.POSITIONAL_OR_KEYWORD
            ),
            __import__("inspect").Parameter(
                "agent", __import__("inspect").Parameter.POSITIONAL_OR_KEYWORD
            ),
        ]
    )

    pre = FunctionAgentHook(
        name="StreamableUpdaterHook",
        fun=streamable_updater,
        trigger=HookPhase.PRE_AGENT_CALL,
        type=HookType.UPDATER,
    )

    # We intercept the context inside _execute to verify the hook updated it
    original_execute_called = False

    async def capturing_execute(context: InteractionContext):
        nonlocal original_execute_called
        original_execute_called = True
        captured_context["state"] = dict(context.state)
        yield (
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="MainAgent",
                content=[Part(text="Final response after streamable updater hook.")],
                is_partial=False,
                need_feedback=False,
            ),
            context,
        )

    main_agent = Agent(
        name="MainAgent",
        description="main agent description",
        instruction="main agent instruction",
        llm_provider=llm_provider,
        hooks=[pre],
    )

    # Override _execute to use async_gen_from_list pattern compatible with test harness
    agent_execution_events = [
        BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="MainAgent",
            content=[Part(text="Final response after streamable updater hook.")],
            is_partial=False,
            need_feedback=False,
        ),
        BaseEvent(event_type=ExecutionEventType.AGENT_END, author="MainAgent", content=[]),
    ]
    main_agent._execute = async_gen_from_list(agent_execution_events)

    application_name = "TestApplication"
    user_id = "test_user"
    conversation_id = "test_flow_streamable_updater_hook"

    input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="Start the process")],
    )

    runner_iterator = runner.run(
        conversation_id=conversation_id,
        application_name=application_name,
        user_id=user_id,
        root=main_agent,
        user_input=input_event,
    )

    events = [a async for a in runner_iterator]

    # AGENT_START, HOOK_START, HOOK_UPDATE x2, HOOK_END, RESPONSE, AGENT_END
    assert len(events) == 7, (
        f"Expected 7 events, got {len(events)}: {[e.event_type for e in events]}"
    )
    assert events[0].event_type == ExecutionEventType.AGENT_START
    assert events[0].author == "MainAgent"

    assert events[1].event_type == ExecutionEventType.HOOK_START
    assert events[1].metadata["hook_name"] == "StreamableUpdaterHook"
    assert events[1].author == "MainAgent"

    assert events[2].event_type == ExecutionEventType.HOOK_UPDATE
    assert events[2].content[0].text == "processing…"
    assert events[2].author == "MainAgent"

    assert events[3].event_type == ExecutionEventType.HOOK_UPDATE
    assert events[3].content[0].text == "almost done…"
    assert events[3].author == "MainAgent"

    # Final HOOK_END carries the event_content set in the HookResult
    assert events[4].event_type == ExecutionEventType.HOOK_END
    assert events[4].metadata["hook_name"] == "StreamableUpdaterHook"
    assert events[4].content[0].text == "Context updated by streamable updater."
    assert events[4].author == "MainAgent"

    assert events[5].event_type == ExecutionEventType.RESPONSE
    assert events[5].content[0].text == "Final response after streamable updater hook."

    assert events[6].event_type == ExecutionEventType.AGENT_END

    # Verify the context was actually updated by the hook: the conversation state
    # persisted by the runner should contain the keys set by the updater hook.
    conversation_info = await runner.conversation_manager.get_conversation(
        application_name=application_name,
        user_id=user_id,
        conversation_id=conversation_id,
    )
    for key, value in updated_state.items():
        assert conversation_info.state.get(key) == value, (
            f"Expected state['{key}'] == {value!r}, got {conversation_info.state.get(key)!r}"
        )


@pytest.mark.asyncio
async def test_flow_break_loop_response_error_occurred_guardrail_skipped(
    runner: Runner, llm_provider: BaseLLMProvider
):
    """Test break_loop_response_event: error_occurred=True guardrail is skipped, second non-error guardrail reason is used.

    When event_content contains two GuardrailResult parts where:
    - First part: passed=False, error_occurred=True, reason set  → must be SKIPPED (error_occurred=True)
    - Second part: passed=False, error_occurred=False, reason set → must be USED as the response text

    The logic in break_loop_response_event only picks a guardrail reason when
    ``not part.guardrail_result.error_occurred`` is True, so the first part must be
    skipped in favour of the second.

    Expected event sequence:
    1. AGENT_START  (user -> MainAgent)
    2. HOOK_START   (pre-hook)
    3. HOOK_END     (pre-hook)
    4. RESPONSE     (text == second guardrail reason)
    5. AGENT_START  (MainAgent -> SubAgent, yielded at the bottom of the runner loop)
    """
    from domyn.agents.core import GuardrailResult
    from domyn.agents.hooks import HookResult

    error_guardrail_reason = "This guardrail errored - must NOT appear as response"
    valid_guardrail_reason = "Content blocked by second guardrail - MUST appear as response"

    async def mixed_guardrail_hook(context, agent) -> HookResult:
        return HookResult(
            success=True,
            emitted_content=[
                # First part: failed but with error_occurred=True → must be skipped
                Part(
                    guardrail_result=GuardrailResult(
                        name="errored_check",
                        score=0.8,
                        reason=error_guardrail_reason,
                        passed=False,
                        error_occurred=True,  # disqualifies this part from being Priority 1
                    )
                ),
                # Second part: failed with no error → its reason must be used
                Part(
                    guardrail_result=GuardrailResult(
                        name="safety_check",
                        score=0.95,
                        reason=valid_guardrail_reason,
                        passed=False,
                        error_occurred=False,
                    )
                ),
            ],
            error=None,
            break_execution=True,
            name="mixed_guardrail_hook",
            new_context=None,
        )

    mixed_guardrail_hook.__signature__ = __import__("inspect").Signature(
        [
            __import__("inspect").Parameter(
                "context", __import__("inspect").Parameter.POSITIONAL_OR_KEYWORD
            ),
            __import__("inspect").Parameter(
                "agent", __import__("inspect").Parameter.POSITIONAL_OR_KEYWORD
            ),
        ]
    )

    pre = FunctionAgentHook(
        name="MixedGuardrailHook",
        fun=mixed_guardrail_hook,
        trigger=HookPhase.PRE_AGENT_CALL,
        type=HookType.LISTENER,
    )

    sub_agent = Agent(
        name="SubAgent",
        description="subagent description",
        instruction="subagent instruction",
        llm_provider=llm_provider,
        hooks=[pre],
    )

    main_agent = Agent(
        name="MainAgent",
        description="main agent description",
        instruction="main agent instruction",
        llm_provider=llm_provider,
        sub_agents=[sub_agent],
    )

    main_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.AGENT_CALL,
                author="MainAgent",
                content=[],
                action=AgentAction(
                    name=sub_agent.name,
                    parameters={"agent_name": sub_agent.name},
                    thought="Invoking SubAgent",
                ),
            ),
        ]
    )
    sub_agent._execute = async_gen_from_list([])  # should never be iterated

    input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="Start the process")],
        turn_id="turn_1",
    )

    runner_iterator = runner.run(
        conversation_id="test_break_loop_error_occurred_skipped",
        application_name="TestApplication",
        user_id="test_user",
        root=main_agent,
        user_input=input_event,
    )

    events = [a async for a in runner_iterator]

    # AGENT_START (user->MainAgent), AGENT_CALL (MainAgent->SubAgent), AGENT_START (SubAgent, runner-created),
    # HOOK_START, HOOK_END, RESPONSE
    assert len(events) == 6, (
        f"Expected 6 events, got {len(events)}: {[e.event_type for e in events]}"
    )
    assert events[0].event_type == ExecutionEventType.AGENT_START
    assert events[0].author == "MainAgent"
    assert events[1].event_type == ExecutionEventType.AGENT_CALL
    assert events[1].author == "MainAgent"
    assert events[1].action.name == "SubAgent"
    assert events[2].event_type == ExecutionEventType.AGENT_START
    assert events[2].author == "SubAgent"
    assert events[3].event_type == ExecutionEventType.HOOK_START
    assert events[3].metadata["hook_name"] == "MixedGuardrailHook"
    assert events[4].event_type == ExecutionEventType.HOOK_END
    # The errored guardrail (error_occurred=True) must be skipped;
    # the second guardrail's reason must be used as the response text.
    assert events[5].event_type == ExecutionEventType.RESPONSE
    assert events[5].content[0].text == valid_guardrail_reason, (
        f"Expected '{valid_guardrail_reason}', got '{events[5].content[0].text}'"
    )
    assert events[5].content[0].text != error_guardrail_reason

    # turn_id: all events in this single turn must share the same turn_id
    turn_id = events[0].turn_id
    assert turn_id == "turn_1"
    assert all(e.turn_id == turn_id for e in events)


@pytest.mark.asyncio
async def test_flow_on_agent_event_hook_returns_final_response_ends_execution(
    runner: Runner, llm_provider: BaseLLMProvider
):
    """on_agent_event updater hook returning a RESPONSE with final=True must end execution.

    The agent emits a TOOL_START followed by TOOL_END and a final RESPONSE.  The
    hook intercepts TOOL_START and replaces it with a final RESPONSE (propagating
    the received event's turn_id).  The runner must stop immediately — neither
    TOOL_END nor the agent's original RESPONSE should be emitted.

    Note: hooks only fire for event types listed in on_event_triggers.  TOOL_START
    and USER_INPUT are both in that list, so both trigger hook execution.
    Hooks do NOT fire on the AGENT_START derived from the initial USER_INPUT —
    the hook has already had a chance to act on USER_INPUT.

    Expected event sequence:
    1. HOOK_START   (hook fires on USER_INPUT, returns None → pass-through)
    2. HOOK_END
    3. AGENT_START  (user -> MainAgent, no hook fired here)
    4. HOOK_START   (hook fires on TOOL_START → replaces with final RESPONSE)
    5. HOOK_END
    6. RESPONSE     (final, the hook-modified event, carrying turn_id from TOOL_START)
    7. HOOK_START   (hook fires on AGENT_END, returns None → pass-through)
    8. HOOK_END
    9. AGENT_END
    """
    hook_intercept_text = "Execution cut short by on_agent_event hook"

    def on_event_returns_final_response(event: BaseEvent, context, **kwargs) -> BaseEvent | None:
        if event.event_type == ExecutionEventType.TOOL_START:
            return BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="MainAgent",
                content=[Part(text=hook_intercept_text)],
                is_partial=False,
                need_feedback=False,
                turn_id=event.turn_id,
            )
        return None

    on_event_hook = FunctionAgentHook(
        name="FinalResponseOnEventHook",
        fun=on_event_returns_final_response,
        trigger=HookPhase.ON_AGENT_EVENT,
        type=HookType.UPDATER,
    )

    main_agent = Agent(
        name="MainAgent",
        description="main agent description",
        instruction="main agent instruction",
        llm_provider=llm_provider,
        hooks=[on_event_hook],
        tools=[FunctionTool(func=lambda: None)],
    )

    # Agent would call a tool then respond — but the hook short-circuits on TOOL_START,
    # so TOOL_END and the original RESPONSE must never be yielded.
    agent_execution_events = [
        BaseEvent(
            event_type=ExecutionEventType.TOOL_START,
            author="MainAgent",
            action=ToolAction(name="lambda", parameters={}, thought="calling tool"),
        ),
        BaseEvent(
            event_type=ExecutionEventType.TOOL_END,
            author="MainAgent",
            action=ToolAction(name="lambda", parameters={}, thought="tool done"),
            content=[Part(text="TOOL_END — must NOT be yielded")],
        ),
        BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="MainAgent",
            content=[Part(text="Agent original final — must NOT be yielded")],
            is_partial=False,
            need_feedback=False,
        ),
    ]
    main_agent._execute = async_gen_from_list(agent_execution_events)

    input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="Start the process")],
        turn_id="turn_1",
    )

    events = [
        e
        async for e in runner.run(
            conversation_id="test_on_agent_event_final_response_ends_execution",
            application_name="TestApplication",
            user_id="test_user",
            root=main_agent,
            user_input=input_event,
        )
    ]

    event_types = [e.event_type for e in events]
    assert event_types == [
        ExecutionEventType.HOOK_START,  # hook fires on USER_INPUT (pass-through)
        ExecutionEventType.HOOK_END,
        ExecutionEventType.AGENT_START,  # no hook fired here (derived from USER_INPUT)
        ExecutionEventType.HOOK_START,  # hook fires on TOOL_START → final RESPONSE
        ExecutionEventType.HOOK_END,
        ExecutionEventType.RESPONSE,
        ExecutionEventType.HOOK_START,  # hook fires on AGENT_END (pass-through)
        ExecutionEventType.HOOK_END,
        ExecutionEventType.AGENT_END,
    ], f"Unexpected event sequence: {event_types}"

    response_event = events[5]
    assert response_event.is_partial is False
    assert response_event.content[0].text == hook_intercept_text

    # TOOL_END and the agent's original RESPONSE must not appear
    assert not any(e.event_type == ExecutionEventType.TOOL_END for e in events)
    assert not any(
        e.event_type == ExecutionEventType.RESPONSE
        and e.content
        and "must NOT be yielded" in (e.content[0].text or "")
        for e in events
    )

    # turn_id: all events carry the same turn_id (hook propagated it from TOOL_START)
    turn_id = events[2].turn_id  # AGENT_START
    assert turn_id == "turn_1"
    assert all(e.turn_id == turn_id for e in events)


@pytest.mark.asyncio
async def test_streaming_agent_forward_verbatim_response_yielded(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """When a streaming agent emits a RESPONSE with METADATA_FORWARD_VERBATIM
    (forward-verbatim), the runner must yield it even though stream=True.

    Regression guard for the interaction between streaming-agent RESPONSE
    suppression (no final RESPONSE for streaming agents) and FORWARD_VERBATIM
    (synthetic non-streamed RESPONSE that carries the actual user-visible content).
    """
    main_agent = Agent(
        name="StreamAgent",
        description="streaming agent",
        instruction="instruction",
        llm_provider=llm_provider,
        stream=True,
    )

    agent_execution_events = [
        BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="StreamAgent",
            content=[Part(text="forwarded verbatim content")],
            is_partial=False,
            metadata={METADATA_FORWARD_VERBATIM: True},
        ),
    ]
    main_agent._execute = async_gen_from_list(agent_execution_events)

    input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="trigger forward verbatim")],
        turn_id="turn_fwd",
    )

    runner_iterator = runner.run(
        conversation_id="test_streaming_forward_verbatim",
        application_name="TestApp",
        user_id="test_user",
        root=main_agent,
        user_input=input_event,
    )

    events = [e async for e in runner_iterator]
    response_events = [e for e in events if e.event_type == ExecutionEventType.RESPONSE]

    assert len(response_events) == 1, (
        f"Expected 1 RESPONSE (forward_verbatim), got {len(response_events)}: "
        f"{[e.metadata for e in response_events]}"
    )
    assert response_events[0].content[0].text == "forwarded verbatim content"
    assert response_events[0].metadata[METADATA_FORWARD_VERBATIM] is True


@pytest.mark.asyncio
async def test_streaming_agent_normal_response_suppressed(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """Normal final RESPONSE from a streaming agent is suppressed.

    Ensures the forward_verbatim carve-out doesn't accidentally re-enable
    yielding of all streaming responses.
    """
    main_agent = Agent(
        name="StreamAgent",
        description="streaming agent",
        instruction="instruction",
        llm_provider=llm_provider,
        stream=True,
    )

    agent_execution_events = [
        BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="StreamAgent",
            content=[Part(text="already streamed content")],
            is_partial=False,
        ),
    ]
    main_agent._execute = async_gen_from_list(agent_execution_events)

    input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="normal streaming question")],
        turn_id="turn_norm",
    )

    runner_iterator = runner.run(
        conversation_id="test_streaming_normal_suppressed",
        application_name="TestApp",
        user_id="test_user",
        root=main_agent,
        user_input=input_event,
    )

    events = [e async for e in runner_iterator]
    response_events = [e for e in events if e.event_type == ExecutionEventType.RESPONSE]

    assert len(response_events) == 0, (
        f"Expected 0 RESPONSE (normal streaming suppressed), got {len(response_events)}"
    )
    conversation_info = await runner.conversation_manager.get_conversation(
        "TestApp", "test_user", "test_streaming_normal_suppressed"
    )
    assert conversation_info is not None
    persisted_responses = [
        e for e in conversation_info.events_history if e.event_type == ExecutionEventType.RESPONSE
    ]
    assert len(persisted_responses) == 1
    assert persisted_responses[0].content[0].text == "already streamed content"


@pytest.mark.asyncio
async def test_nostream_agent_forward_verbatim_response_yielded(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """Non-streaming agent with METADATA_FORWARD_VERBATIM yields RESPONSE.

    Baseline: forward-verbatim on stream=False path (default). Ensures the
    forward_verbatim path is yielded regardless of stream flag.
    """
    main_agent = Agent(
        name="NoStreamAgent",
        description="non-streaming agent",
        instruction="instruction",
        llm_provider=llm_provider,
        stream=False,
    )

    agent_execution_events = [
        BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="NoStreamAgent",
            content=[Part(text="forwarded verbatim content nostream")],
            is_partial=False,
            metadata={METADATA_FORWARD_VERBATIM: True},
        ),
    ]
    main_agent._execute = async_gen_from_list(agent_execution_events)

    input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="trigger forward verbatim nostream")],
        turn_id="turn_fwd_nostream",
    )

    runner_iterator = runner.run(
        conversation_id="test_nostream_forward_verbatim",
        application_name="TestApp",
        user_id="test_user",
        root=main_agent,
        user_input=input_event,
    )

    events = [e async for e in runner_iterator]
    response_events = [e for e in events if e.event_type == ExecutionEventType.RESPONSE]

    assert len(response_events) == 1, (
        f"Expected 1 RESPONSE (nostream + forward_verbatim), got {len(response_events)}"
    )
    assert response_events[0].content[0].text == "forwarded verbatim content nostream"
    assert response_events[0].metadata[METADATA_FORWARD_VERBATIM] is True


@pytest.mark.asyncio
async def test_subagent_forward_verbatim_end_returns_to_parent(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """Sub-agent ``AGENT_END`` carrying FORWARD_VERBATIM returns to parent.

    The sub-agent marks its result as forwardable, but the runner must not let
    the sub-agent decide that this is the final user-facing response.
    """
    sub_agent = Agent(
        name="SubAgent",
        description="sub-agent",
        instruction="sub-agent instruction",
        llm_provider=llm_provider,
    )
    main_agent = Agent(
        name="MainAgent",
        description="main agent",
        instruction="main agent instruction",
        sub_agents=[sub_agent],
        llm_provider=llm_provider,
    )

    main_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.AGENT_CALL,
                author="MainAgent",
                action=AgentAction(name="SubAgent", parameters={"task": "read tool result"}),
            ),
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="MainAgent",
                content=[Part(text="parent chooses the final answer")],
            ),
        ]
    )
    sub_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.AGENT_END,
                author="SubAgent",
                content=[Part(text="sub-agent verbatim answer")],
                action=ToolAction(name="STOP", parameters={"info": "unused stop info"}),
                metadata={METADATA_FORWARD_VERBATIM: True},
            )
        ]
    )

    input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="delegate")],
        turn_id="turn_subagent_fv",
    )

    events = [
        e
        async for e in runner.run(
            conversation_id="test_subagent_forward_verbatim_end",
            application_name="TestApp",
            user_id="test_user",
            root=main_agent,
            user_input=input_event,
        )
    ]

    response_events = [e for e in events if e.event_type == ExecutionEventType.RESPONSE]
    assert len(response_events) == 1
    response = response_events[0]
    assert response.author == "MainAgent"
    assert response.content[0].text == "parent chooses the final answer"
    assert response.metadata.get(METADATA_FORWARD_VERBATIM) is None

    conversation_info = await runner.conversation_manager.get_conversation(
        "TestApp", "test_user", "test_subagent_forward_verbatim_end"
    )
    assert conversation_info is not None
    persisted_responses = [
        e for e in conversation_info.events_history if e.event_type == ExecutionEventType.RESPONSE
    ]
    assert len(persisted_responses) == 1
    assert persisted_responses[0].content[0].text == "parent chooses the final answer"
    assert persisted_responses[0].metadata.get(METADATA_FORWARD_VERBATIM) is None

    response_texts = [
        part.text for event in response_events for part in event.content if part.text is not None
    ]
    assert "sub-agent verbatim answer" not in response_texts, (
        "sub-agent FV-marked text must not surface as a user-facing RESPONSE; "
        "the parent chooses the final answer"
    )
    assert "parent chooses the final answer" in response_texts

    # Sub-agent AGENT_END is yielded (lifecycle visible to consumer for
    # FE/analytics) and persisted as the forwardable candidate. The runner
    # closes the stream with the root agent's AGENT_END.
    final_agent_end = events[-1]
    assert final_agent_end.event_type == ExecutionEventType.AGENT_END
    assert final_agent_end.author == "MainAgent"
    assert final_agent_end.content == []
    assert final_agent_end.action is None

    persisted_agent_ends = [
        e for e in conversation_info.events_history if e.event_type == ExecutionEventType.AGENT_END
    ]
    persisted_authors = [e.author for e in persisted_agent_ends]
    assert "SubAgent" in persisted_authors, (
        "sub-agent AGENT_END must remain in history as the forwardable candidate"
    )
    assert "MainAgent" in persisted_authors, (
        "root AGENT_END must be persisted to close the AGENT_START/AGENT_END pairing"
    )
    sub_agent_end = next(e for e in persisted_agent_ends if e.author == "SubAgent")
    assert sub_agent_end.metadata[METADATA_FORWARD_VERBATIM] is True

    assert (
        await runner.conversation_manager.get_last_agent_stack(
            "TestApp", "test_user", "test_subagent_forward_verbatim_end"
        )
        is None
    )


@pytest.mark.asyncio
async def test_nostream_agent_normal_response_yielded(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """Non-streaming agent with normal RESPONSE (no forward_verbatim) yields.

    Completes the 2x2 matrix (stream x forward_verbatim). Ensures baseline
    RESPONSE flow is not affected by the forward_verbatim carve-out.
    """
    main_agent = Agent(
        name="NoStreamAgent",
        description="non-streaming agent",
        instruction="instruction",
        llm_provider=llm_provider,
        stream=False,
    )

    agent_execution_events = [
        BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="NoStreamAgent",
            content=[Part(text="normal response nostream")],
            is_partial=False,
        ),
    ]
    main_agent._execute = async_gen_from_list(agent_execution_events)

    input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="normal nostream question")],
        turn_id="turn_norm_nostream",
    )

    runner_iterator = runner.run(
        conversation_id="test_nostream_normal_response",
        application_name="TestApp",
        user_id="test_user",
        root=main_agent,
        user_input=input_event,
    )

    events = [e async for e in runner_iterator]
    response_events = [e for e in events if e.event_type == ExecutionEventType.RESPONSE]

    assert len(response_events) == 1, (
        f"Expected 1 RESPONSE (nostream + no forward_verbatim), got {len(response_events)}"
    )
    assert response_events[0].content[0].text == "normal response nostream"


@pytest.mark.asyncio
async def test_streaming_agent_ask_user_response_yielded(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """When a streaming agent emits a feedback RESPONSE for ASK_USER_MISSING_INFO,
    the runner must yield it even though stream=True.

    Feedback responses are emitted as a single direct yield by the planner (no
    RESPONSE_TOKEN sequence preceding them), so suppressing them under stream=True
    would drop the user-facing question entirely.
    """
    main_agent = Agent(
        name="StreamAgent",
        description="streaming agent",
        instruction="instruction",
        llm_provider=llm_provider,
        stream=True,
    )

    agent_execution_events = [
        BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="StreamAgent",
            content=[Part(text="Could you clarify which year you mean?")],
            is_partial=False,
            need_feedback=True,
            action=ToolAction(name="ASK_USER_MISSING_INFO", parameters={}),
        ),
    ]
    main_agent._execute = async_gen_from_list(agent_execution_events)

    input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="trigger ask_user")],
        turn_id="turn_ask_user_stream",
    )

    runner_iterator = runner.run(
        conversation_id="test_streaming_ask_user",
        application_name="TestApp",
        user_id="test_user",
        root=main_agent,
        user_input=input_event,
    )

    events = [e async for e in runner_iterator]
    response_events = [e for e in events if e.event_type == ExecutionEventType.RESPONSE]

    assert len(response_events) == 1, (
        f"Expected 1 RESPONSE (streaming ASK_USER), got {len(response_events)}: "
        f"all event_types={[e.event_type for e in events]}"
    )
    assert response_events[0].content[0].text == "Could you clarify which year you mean?"
    assert response_events[0].action.name == "ASK_USER_MISSING_INFO"
    assert response_events[0].need_feedback is True


@pytest.mark.asyncio
async def test_streaming_agent_custom_feedback_response_yielded(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """Any streaming RESPONSE with ``need_feedback=True`` must be yielded,
    regardless of the action name.

    Feedback is the semantic signal that the runner is about to pause for user
    input. If the response were suppressed under ``stream=True``, the runner
    would return without ever showing the user-facing question.
    """
    main_agent = Agent(
        name="StreamAgent",
        description="streaming agent",
        instruction="instruction",
        llm_provider=llm_provider,
        stream=True,
    )

    agent_execution_events = [
        BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="StreamAgent",
            content=[Part(text="Which dataset should I use?")],
            is_partial=False,
            need_feedback=True,
            action=ToolAction(name="CUSTOM_CLARIFY", parameters={}),
        ),
    ]
    main_agent._execute = async_gen_from_list(agent_execution_events)

    input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="trigger custom feedback")],
        turn_id="turn_custom_feedback_stream",
    )

    runner_iterator = runner.run(
        conversation_id="test_streaming_custom_feedback",
        application_name="TestApp",
        user_id="test_user",
        root=main_agent,
        user_input=input_event,
    )

    events = [e async for e in runner_iterator]
    response_events = [e for e in events if e.event_type == ExecutionEventType.RESPONSE]

    assert len(response_events) == 1
    assert response_events[0].content[0].text == "Which dataset should I use?"
    assert response_events[0].action.name == "CUSTOM_CLARIFY"
    assert response_events[0].need_feedback is True


@pytest.mark.asyncio
async def test_streaming_agent_forward_verbatim_action_without_metadata_suppressed(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """Streaming + RESPONSE with action.name='FORWARD_VERBATIM' is not enough.

    FORWARD_VERBATIM is a recent first-class feature: the runner contract is
    the metadata flag, not the action name. Without METADATA_FORWARD_VERBATIM,
    a normal streaming RESPONSE is suppressed because the text should already
    have been emitted as RESPONSE_TOKEN.
    """
    main_agent = Agent(
        name="StreamAgent",
        description="streaming agent",
        instruction="instruction",
        llm_provider=llm_provider,
        stream=True,
    )

    agent_execution_events = [
        BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="StreamAgent",
            content=[Part(text="forwarded text")],
            is_partial=False,
            action=ToolAction(name="FORWARD_VERBATIM", parameters={}),
        ),
    ]
    main_agent._execute = async_gen_from_list(agent_execution_events)

    input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="trigger forward_verbatim by action name")],
        turn_id="turn_fv_action_stream",
    )

    runner_iterator = runner.run(
        conversation_id="test_streaming_forward_verbatim_action",
        application_name="TestApp",
        user_id="test_user",
        root=main_agent,
        user_input=input_event,
    )

    events = [e async for e in runner_iterator]
    response_events = [e for e in events if e.event_type == ExecutionEventType.RESPONSE]

    assert len(response_events) == 0, (
        f"FORWARD_VERBATIM action name without metadata must not bypass streaming "
        f"RESPONSE suppression; got {len(response_events)}"
    )


@pytest.mark.asyncio
async def test_streaming_agent_unknown_action_response_suppressed(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """Carve-out scope guard: streaming + RESPONSE with ``need_feedback=False``
    and no FORWARD_VERBATIM metadata must still be suppressed under ``stream=True``.

    Streaming-agent RESPONSE suppression drops *all* terminal RESPONSE events on
    streaming agents because the text already streamed via ``RESPONSE_TOKEN``. The
    carve-out for metadata-forwarded and feedback responses must not broaden to
    arbitrary non-feedback responses.
    """
    main_agent = Agent(
        name="StreamAgent",
        description="streaming agent",
        instruction="instruction",
        llm_provider=llm_provider,
        stream=True,
    )

    agent_execution_events = [
        BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="StreamAgent",
            content=[Part(text="already streamed reply")],
            is_partial=False,
            action=ToolAction(name="RESPOND", parameters={}),
        ),
    ]
    main_agent._execute = async_gen_from_list(agent_execution_events)

    input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="trigger respond action stream")],
        turn_id="turn_respond_stream",
    )

    runner_iterator = runner.run(
        conversation_id="test_streaming_unknown_action_suppressed",
        application_name="TestApp",
        user_id="test_user",
        root=main_agent,
        user_input=input_event,
    )

    events = [e async for e in runner_iterator]
    response_events = [e for e in events if e.event_type == ExecutionEventType.RESPONSE]

    assert len(response_events) == 0, (
        f"Streaming RESPONSE with non-special action must be suppressed; "
        f"got {len(response_events)}: {[e.action.name for e in response_events]}"
    )


@pytest.mark.asyncio
async def test_nostream_agent_ask_user_response_yielded(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """Sanity baseline: ``stream=False`` + ASK_USER_MISSING_INFO RESPONSE is
    yielded normally.

    Non-streaming path is unaffected by the carve-out: the runner falls into
    the ``not stream`` branch first and yields. Pinned to detect a future
    refactor that accidentally hides ASK_USER on the non-streaming path
    (e.g. by adding a misplaced guard before the carve-out).
    """
    main_agent = Agent(
        name="NoStreamAgent",
        description="non-streaming agent",
        instruction="instruction",
        llm_provider=llm_provider,
        stream=False,
    )

    agent_execution_events = [
        BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="NoStreamAgent",
            content=[Part(text="Which year do you mean?")],
            is_partial=False,
            need_feedback=True,
            action=ToolAction(name="ASK_USER_MISSING_INFO", parameters={}),
        ),
    ]
    main_agent._execute = async_gen_from_list(agent_execution_events)

    input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="trigger ask_user nostream")],
        turn_id="turn_ask_user_nostream",
    )

    runner_iterator = runner.run(
        conversation_id="test_nostream_ask_user",
        application_name="TestApp",
        user_id="test_user",
        root=main_agent,
        user_input=input_event,
    )

    events = [e async for e in runner_iterator]
    response_events = [e for e in events if e.event_type == ExecutionEventType.RESPONSE]

    assert len(response_events) == 1
    assert response_events[0].content[0].text == "Which year do you mean?"
    assert response_events[0].action.name == "ASK_USER_MISSING_INFO"
    assert response_events[0].need_feedback is True


# ===========================================================================
# on_agent_event hook — full runner.run() flow (mirrors test_runner.py unit tests)
# ===========================================================================


def _make_on_event_hook(fn) -> FunctionAgentHook:
    return FunctionAgentHook(fun=fn, trigger=HookPhase.ON_AGENT_EVENT, type=HookType.UPDATER)


def _make_on_event_listener_hook(fn) -> FunctionAgentHook:
    return FunctionAgentHook(fun=fn, trigger=HookPhase.ON_AGENT_EVENT, type=HookType.LISTENER)


@pytest.mark.asyncio
async def test_flow_on_event_user_input_redaction_persists(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """USER_INPUT modified by on_agent_event hook is stored in conversation history
    with the redacted content — the original text is never seen by the agent."""

    def redact_fn(event: BaseEvent, context, **_) -> BaseEvent | None:
        if event.event_type != ExecutionEventType.USER_INPUT:
            return None
        modified = event.model_copy(deep=True)
        modified.content = [
            Part(text=p.text.replace("secret_token", "[REDACTED]")) if p.text else p
            for p in modified.content
        ]
        return modified

    hook = _make_on_event_hook(redact_fn)

    agent = Agent(
        name="MainAgent",
        description="agent",
        instruction="agent instruction",
        llm_provider=llm_provider,
        hooks=[hook],
    )
    agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="MainAgent",
                content=[Part(text="ok")],
                is_partial=False,
                need_feedback=False,
            )
        ]
    )

    conversation_id = "flow_on_event_user_input_redact"
    _ = [
        e
        async for e in runner.run(
            application_name="TestApp",
            user_id="test_user",
            root=agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                author="user",
                content=[Part(text="my secret_token is abc123")],
            ),
            conversation_id=conversation_id,
        )
    ]

    conversation = await runner.conversation_manager.get_conversation(
        "TestApp", "test_user", conversation_id
    )
    stored_inputs = [
        e for e in conversation.events_history if e.event_type == ExecutionEventType.USER_INPUT
    ]
    assert stored_inputs, "USER_INPUT must be persisted in conversation history"
    assert "[REDACTED]" in stored_inputs[0].content[0].text
    assert "secret_token" not in stored_inputs[0].content[0].text


@pytest.mark.asyncio
async def test_flow_on_event_agent_start_task_redaction_persists_across_handoff(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """AGENT_START content redacted by the parent's on_agent_event hook is what the
    sub-agent's context receives — the original task text never appears in the yielded event."""

    def redact_task_fn(event: BaseEvent, context, **_) -> BaseEvent | None:
        if event.event_type != ExecutionEventType.AGENT_START:
            return None
        modified = event.model_copy(deep=True)
        modified.content = [
            Part(text=p.text.replace("confidential", "[REDACTED]")) if p.text else p
            for p in modified.content
        ]
        return modified

    hook = _make_on_event_hook(redact_task_fn)

    sub_agent = Agent(
        name="SubAgent",
        description="sub",
        instruction="sub instruction",
        llm_provider=llm_provider,
    )
    sub_agent._execute = async_gen_from_list(
        [BaseEvent(event_type=ExecutionEventType.AGENT_END, author="SubAgent", content=[])]
    )

    main_agent = Agent(
        name="MainAgent",
        description="main",
        instruction="main instruction",
        llm_provider=llm_provider,
        sub_agents=[sub_agent],
        hooks=[hook],
    )
    main_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.AGENT_START,
                author="MainAgent",
                content=[Part(text="Perform the confidential analysis task")],
                action=AgentAction(
                    name="SubAgent", parameters={}, thought="Delegating confidential task"
                ),
            ),
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="MainAgent",
                content=[Part(text="done")],
                is_partial=False,
                need_feedback=False,
            ),
        ]
    )

    events = [
        e
        async for e in runner.run(
            application_name="TestApp",
            user_id="test_user",
            root=main_agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                author="user",
                content=[Part(text="start")],
            ),
            conversation_id="flow_on_event_agent_start_redact",
        )
    ]

    # The AGENT_START for SubAgent is yielded with the redacted content
    child_starts = [
        e
        for e in events
        if e.event_type == ExecutionEventType.AGENT_START
        and e.action
        and e.action.name == "SubAgent"
    ]
    assert child_starts, "AGENT_START for SubAgent must be yielded"
    assert "[REDACTED]" in child_starts[0].content[0].text
    assert "confidential" not in child_starts[0].content[0].text


@pytest.mark.asyncio
async def test_flow_on_event_agent_end_use_stop_redaction_returned_to_calling_agent(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """AGENT_END emitted by a sub-agent (use_stop) is redacted by that sub-agent's
    on_agent_event hook; the redacted event is what the runner yields back to the caller."""

    def redact_result_fn(event: BaseEvent, context, **_) -> BaseEvent | None:
        if event.event_type != ExecutionEventType.AGENT_END:
            return None
        modified = event.model_copy(deep=True)
        modified.content = [
            Part(text=p.text.replace("sensitive_result", "[REDACTED]")) if p.text else p
            for p in modified.content
        ]
        return modified

    hook = _make_on_event_hook(redact_result_fn)

    sub_agent = Agent(
        name="SubAgent",
        description="sub",
        instruction="sub instruction",
        llm_provider=llm_provider,
        hooks=[hook],
    )
    sub_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.AGENT_END,
                author="SubAgent",
                content=[Part(text="The sensitive_result is: top secret data")],
                action=ToolAction(name="STOP", parameters={"info": "done"}, thought="Done"),
            )
        ]
    )

    main_agent = Agent(
        name="MainAgent",
        description="main",
        instruction="main instruction",
        llm_provider=llm_provider,
        sub_agents=[sub_agent],
    )
    main_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.AGENT_CALL,
                author="MainAgent",
                content=[],
                action=AgentAction(name="SubAgent", parameters={}, thought="Delegating"),
            ),
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="MainAgent",
                content=[Part(text="done")],
                is_partial=False,
                need_feedback=False,
            ),
        ]
    )

    events = [
        e
        async for e in runner.run(
            application_name="TestApp",
            user_id="test_user",
            root=main_agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                author="user",
                content=[Part(text="start")],
            ),
            conversation_id="flow_on_event_agent_end_redact",
        )
    ]

    # The AGENT_END from SubAgent is yielded with the redacted content
    child_ends = [
        e for e in events if e.event_type == ExecutionEventType.AGENT_END and e.author == "SubAgent"
    ]
    assert child_ends, "AGENT_END from SubAgent must be yielded"
    assert "[REDACTED]" in child_ends[0].content[0].text
    assert "sensitive_result" not in child_ends[0].content[0].text


@pytest.mark.asyncio
async def test_flow_on_event_response_modified_before_returned_to_user(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """ON_AGENT_EVENT updater hook intercepts the final RESPONSE event and rewrites its
    content before it is yielded to the caller."""

    def censor_response(event: BaseEvent, context, **_) -> BaseEvent | None:
        if event.event_type != ExecutionEventType.RESPONSE:
            return None
        modified = event.model_copy(deep=True)
        modified.content = [
            Part(text=p.text.replace("secret", "[CENSORED]")) if p.text else p
            for p in modified.content
        ]
        return modified

    hook = _make_on_event_hook(censor_response)

    agent = Agent(
        name="MainAgent",
        description="main",
        instruction="main instruction",
        llm_provider=llm_provider,
        hooks=[hook],
    )
    agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="MainAgent",
                content=[Part(text="The secret answer is 42")],
                is_partial=False,
                need_feedback=False,
            )
        ]
    )

    events = [
        e
        async for e in runner.run(
            application_name="TestApp",
            user_id="test_user",
            root=agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                author="user",
                content=[Part(text="tell me the secret")],
            ),
            conversation_id="flow_on_event_response_censored",
        )
    ]

    responses = [e for e in events if e.event_type == ExecutionEventType.RESPONSE]
    assert responses, "A RESPONSE event must be yielded"
    assert "[CENSORED]" in responses[0].content[0].text
    assert "secret" not in responses[0].content[0].text


@pytest.mark.asyncio
async def test_flow_on_event_listener_break_on_user_input(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """ON_AGENT_EVENT listener hook that breaks execution on the incoming USER_INPUT.

    The hook fires before the agent loop starts. Because it sets break_execution=True
    the runner returns immediately without yielding AGENT_START or executing the agent.

    Expected event sequence:
    1. HOOK_START
    2. HOOK_END   (break_execution content emitted here)
    3. RESPONSE   (break_loop_response)
    Agent _execute must NOT be called.
    """
    executed = {"count": 0}

    def break_on_input(event: BaseEvent, context: Any, **_) -> HookResult:
        return HookResult(
            success=True,
            emitted_content=[Part(text="Blocked by on_agent_event listener on user input")],
            error=None,
            break_execution=True,
            name="break_on_input",
            new_context=None,
        )

    hook = _make_on_event_listener_hook(break_on_input)
    hook.on_event_triggers = [ExecutionEventType.USER_INPUT]

    agent = Agent(
        name="MainAgent",
        description="main",
        instruction="main instruction",
        llm_provider=llm_provider,
        hooks=[hook],
    )

    async def counting_execute(*args, **kwargs):
        executed["count"] += 1
        return
        yield  # make it an async generator

    agent._execute = counting_execute

    events = [
        e
        async for e in runner.run(
            application_name="TestApp",
            user_id="test_user",
            root=agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                author="user",
                content=[Part(text="hello")],
                turn_id="turn_break_input",
            ),
            conversation_id="flow_on_event_listener_break_user_input",
        )
    ]

    event_types = [e.event_type for e in events]
    assert ExecutionEventType.HOOK_START in event_types
    assert ExecutionEventType.HOOK_END in event_types
    # break_loop_response is a RESPONSE event
    assert ExecutionEventType.RESPONSE in event_types
    # AGENT_START must not be yielded — runner returned before reaching it
    assert ExecutionEventType.AGENT_START not in event_types
    # agent _execute must never have been invoked
    assert executed["count"] == 0
    # the break response carries the hook's message
    response = next(e for e in events if e.event_type == ExecutionEventType.RESPONSE)
    assert any(
        p.text and "Blocked by on_agent_event listener on user input" in p.text
        for p in response.content
    )


@pytest.mark.asyncio
async def test_flow_on_event_listener_break_on_agent_emitted_event(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """ON_AGENT_EVENT listener hook that breaks execution when the agent emits a RESPONSE.

    The agent emits one RESPONSE event. The listener hook fires on it and sets
    break_execution=True. The runner breaks out of the agent loop without yielding
    the original RESPONSE to the caller.

    Expected event sequence:
    1. AGENT_START  (user → MainAgent, new conversation)
    2. HOOK_START   (on_agent_event listener fires on the RESPONSE event)
    3. HOOK_END     (break_execution content)
    4. RESPONSE     (break_loop_response synthesised from the hook result)
    The original RESPONSE from the agent must NOT appear as a separate event.
    """

    def break_on_response(event: BaseEvent, context: Any, **_) -> HookResult:
        return HookResult(
            success=True,
            emitted_content=[Part(text="Blocked by on_agent_event listener on response")],
            error=None,
            break_execution=True,
            name="break_on_response",
            new_context=None,
        )

    hook = _make_on_event_listener_hook(break_on_response)
    hook.on_event_triggers = [ExecutionEventType.RESPONSE]

    agent = Agent(
        name="MainAgent",
        description="main",
        instruction="main instruction",
        llm_provider=llm_provider,
        hooks=[hook],
    )
    agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="MainAgent",
                content=[Part(text="original agent answer")],
                is_partial=False,
                need_feedback=False,
            )
        ]
    )

    events = [
        e
        async for e in runner.run(
            application_name="TestApp",
            user_id="test_user",
            root=agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                author="user",
                content=[Part(text="hello")],
                turn_id="turn_break_response",
            ),
            conversation_id="flow_on_event_listener_break_response",
        )
    ]

    event_types = [e.event_type for e in events]
    assert event_types[0] == ExecutionEventType.AGENT_START
    assert ExecutionEventType.HOOK_START in event_types
    assert ExecutionEventType.HOOK_END in event_types
    # break_loop_response RESPONSE must be present
    responses = [e for e in events if e.event_type == ExecutionEventType.RESPONSE]
    assert responses, "Expected a RESPONSE (break_loop_response) event"
    # that response must carry the hook's message, not the agent's original text
    assert any(
        p.text and "Blocked by on_agent_event listener on response" in p.text
        for p in responses[0].content
    )
    assert not any(p.text and "original agent answer" in p.text for p in responses[0].content)


@pytest.mark.asyncio
async def test_flow_on_event_agent_start_modified_agent_call_unchanged_standard_visibility(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """The runner-created AGENT_START for a STANDARD-visibility sub-agent is stripped of a
    private message by that sub-agent's on_agent_event hook, while the AGENT_CALL emitted by
    the parent is yielded unchanged.

    Two guarantees verified end-to-end via runner.run():
    1. on_agent_event hooks fire on AGENT_START (not on AGENT_CALL) — only AGENT_START loses
       the private message.
    2. STANDARD visibility — the context SubAgent receives contains no AGENT_CALL event
       authored by the parent, even though that event was stored in the full conversation
       history before the hook ran.
    """

    private = "private_message_xyz"

    def strip_private_from_start(event: BaseEvent, context, **_) -> BaseEvent | None:
        if event.event_type != ExecutionEventType.AGENT_START:
            return None
        modified = event.model_copy(deep=True)
        modified.content = [
            Part(text=p.text.replace(private, "[REMOVED]")) if p.text else p
            for p in modified.content
        ]
        return modified

    hook = _make_on_event_hook(strip_private_from_start)

    captured_contexts: list[InteractionContext] = []

    def make_sub_execute():
        async def _gen(ctx: InteractionContext, *args, **kwargs) -> AsyncGenerator:
            captured_contexts.append(ctx)
            yield BaseEvent(event_type=ExecutionEventType.AGENT_END, author="SubAgent", content=[])

        return _gen

    sub_agent = Agent(
        name="SubAgent",
        description="sub",
        instruction="sub instruction",
        llm_provider=llm_provider,
        hooks=[hook],
        context_visibility=ContextVisibility.STANDARD,
    )
    sub_agent._execute = make_sub_execute()

    main_agent = Agent(
        name="MainAgent",
        description="main",
        instruction="main instruction",
        llm_provider=llm_provider,
        sub_agents=[sub_agent],
    )
    main_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.AGENT_CALL,
                author="MainAgent",
                content=[],
                action=AgentAction(
                    name="SubAgent",
                    parameters={"task": f"Perform this {private} analysis"},
                    thought="Delegating",
                ),
            ),
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="MainAgent",
                content=[Part(text="done")],
                is_partial=False,
                need_feedback=False,
            ),
        ]
    )

    events = [
        e
        async for e in runner.run(
            application_name="TestApp",
            user_id="test_user",
            root=main_agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                author="user",
                content=[Part(text="start")],
            ),
            conversation_id="flow_on_event_start_not_call_standard_visibility",
        )
    ]

    # 1. AGENT_CALL yielded with original private message intact — hook did not touch it
    agent_calls = [e for e in events if e.event_type == ExecutionEventType.AGENT_CALL]
    assert agent_calls, "AGENT_CALL must be yielded"
    assert private in agent_calls[0].action.parameters["task"]

    # 2. AGENT_START for SubAgent yielded with private message stripped by the hook
    sub_starts = [
        e
        for e in events
        if e.event_type == ExecutionEventType.AGENT_START and e.author == "SubAgent"
    ]
    assert sub_starts, "AGENT_START for SubAgent must be yielded"
    assert private not in sub_starts[0].content[0].text
    assert "[REMOVED]" in sub_starts[0].content[0].text

    # 3. STANDARD visibility — SubAgent's context.conversation_history must not contain
    #    the parent's AGENT_CALL (authored by "MainAgent"), even though the runner stored
    #    that event in the full conversation history before applying the hook.
    assert captured_contexts, "SubAgent must have executed and captured its context"
    sub_context = captured_contexts[0]
    assert not any(
        e.event_type == ExecutionEventType.AGENT_CALL and e.author == "MainAgent"
        for e in sub_context.conversation_history
    )


@pytest.mark.asyncio
async def test_agent_hook_events_saved_to_conversation_manager(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """HOOK_START and HOOK_END for agent pre/post hooks must be persisted to events_history."""

    async def pre_hook_fn(agent, context):
        pass

    async def post_hook_fn(agent, context):
        return {"new_context": context}

    pre = FunctionAgentHook(
        name="PreHook",
        fun=pre_hook_fn,
        trigger=HookPhase.PRE_AGENT_CALL,
        type=HookType.LISTENER,
    )
    post = FunctionAgentHook(
        name="PostHook",
        fun=post_hook_fn,
        trigger=HookPhase.POST_AGENT_CALL,
        type=HookType.UPDATER,
    )

    main_agent = Agent(
        name="MainAgent",
        description="main agent description",
        instruction="main agent instruction",
        llm_provider=llm_provider,
        hooks=[pre, post],
    )
    main_agent._execute = async_gen_from_list(
        [BaseEvent(event_type=ExecutionEventType.AGENT_END, author="MainAgent", content=[])]
    )

    application_name = "TestApplication"
    user_id = "test_user"
    conversation_id = "test_agent_hook_events_saved_to_cm"

    async for _ in runner.run(
        conversation_id=conversation_id,
        application_name=application_name,
        user_id=user_id,
        root=main_agent,
        user_input=BaseEvent(
            event_type=ExecutionEventType.USER_INPUT,
            author="user",
            content=[Part(text="start")],
        ),
    ):
        pass

    conversation_info = await runner.conversation_manager.get_conversation(
        application_name=application_name,
        user_id=user_id,
        conversation_id=conversation_id,
    )
    saved_types = [e.event_type for e in conversation_info.events_history]

    assert saved_types.count(ExecutionEventType.HOOK_START) == 2
    assert saved_types.count(ExecutionEventType.HOOK_END) == 2

    hook_events = [
        e
        for e in conversation_info.events_history
        if e.event_type in (ExecutionEventType.HOOK_START, ExecutionEventType.HOOK_END)
    ]
    hook_names = [e.metadata.get("hook_name") for e in hook_events]
    assert hook_names.count("PreHook") == 2
    assert hook_names.count("PostHook") == 2


@pytest.mark.asyncio
async def test_tool_pre_hook_events_saved_to_conversation_manager(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """HOOK_START and HOOK_END for a tool updater pre-hook must be persisted to events_history."""

    def my_tool(x: int) -> int:
        return x * 2

    def pre_hook_fn(tool, tool_args):
        return {"new_tool_args": tool_args}

    pre_hook = FunctionToolPreHook(
        name="MyPreHook",
        fun=pre_hook_fn,
        trigger=HookPhase.PRE_TOOL_CALL,
        type=HookType.UPDATER,
    )

    tool = FunctionTool(func=my_tool, hooks=[pre_hook])

    main_agent = Agent(
        name="MainAgent",
        description="main agent description",
        instruction="main agent instruction",
        llm_provider=llm_provider,
        tools=[tool],
    )
    main_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.TOOL_START,
                author="MainAgent",
                action=ToolAction(name="my_tool", parameters={"x": 5}, thought="calling my_tool"),
            ),
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="MainAgent",
                content=[Part(text="done")],
            ),
        ]
    )

    application_name = "TestApplication"
    user_id = "test_user"
    conversation_id = "test_tool_pre_hook_events_saved_to_cm"

    async for _ in runner.run(
        conversation_id=conversation_id,
        application_name=application_name,
        user_id=user_id,
        root=main_agent,
        user_input=BaseEvent(
            event_type=ExecutionEventType.USER_INPUT,
            author="user",
            content=[Part(text="start")],
        ),
    ):
        pass

    conversation_info = await runner.conversation_manager.get_conversation(
        application_name=application_name,
        user_id=user_id,
        conversation_id=conversation_id,
    )
    saved_types = [e.event_type for e in conversation_info.events_history]

    assert ExecutionEventType.HOOK_START in saved_types
    assert ExecutionEventType.HOOK_END in saved_types

    hook_events = [
        e
        for e in conversation_info.events_history
        if e.event_type in (ExecutionEventType.HOOK_START, ExecutionEventType.HOOK_END)
    ]
    assert all(e.metadata.get("hook_name") == "MyPreHook" for e in hook_events)


@pytest.mark.asyncio
async def test_tool_post_hook_events_saved_to_conversation_manager(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """HOOK_START and HOOK_END for a tool listener post-hook must be persisted to events_history."""

    def my_tool(x: int) -> int:
        return x * 2

    called = {}

    async def post_hook_fn(tool, result):
        called["yes"] = True

    post_hook = FunctionToolPreHook(
        name="MyPostHook",
        fun=post_hook_fn,
        trigger=HookPhase.POST_TOOL_CALL,
        type=HookType.LISTENER,
    )

    tool = FunctionTool(func=my_tool, hooks=[post_hook])

    main_agent = Agent(
        name="MainAgent",
        description="main agent description",
        instruction="main agent instruction",
        llm_provider=llm_provider,
        tools=[tool],
    )
    main_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.TOOL_START,
                author="MainAgent",
                action=ToolAction(name="my_tool", parameters={"x": 5}, thought="calling my_tool"),
            ),
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="MainAgent",
                content=[Part(text="done")],
            ),
        ]
    )

    application_name = "TestApplication"
    user_id = "test_user"
    conversation_id = "test_tool_post_hook_events_saved_to_cm"

    async for _ in runner.run(
        conversation_id=conversation_id,
        application_name=application_name,
        user_id=user_id,
        root=main_agent,
        user_input=BaseEvent(
            event_type=ExecutionEventType.USER_INPUT,
            author="user",
            content=[Part(text="start")],
        ),
    ):
        pass

    conversation_info = await runner.conversation_manager.get_conversation(
        application_name=application_name,
        user_id=user_id,
        conversation_id=conversation_id,
    )
    saved_types = [e.event_type for e in conversation_info.events_history]

    assert ExecutionEventType.HOOK_START in saved_types
    assert ExecutionEventType.HOOK_END in saved_types

    hook_events = [
        e
        for e in conversation_info.events_history
        if e.event_type in (ExecutionEventType.HOOK_START, ExecutionEventType.HOOK_END)
    ]
    assert all(e.metadata.get("hook_name") == "MyPostHook" for e in hook_events)
    assert called.get("yes") is True


@pytest.mark.asyncio
async def test_flow_with_feedback_tool_on_standard_visibility_subagent(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """Regression: TOOL_FEEDBACK targeting a sub-agent with ContextVisibility.STANDARD
    must not crash while recovering the agent input.

    Previously, after handling TOOL_FEEDBACK the runner pulled the last USER_INPUT event
    out of the visibility-filtered conversation_history; but for a non-entrypoint,
    non-GLOBAL sub-agent those events are stripped (their interaction_id differs from the
    sub-agent's), and the lookup raised IndexError. The fix recovers the sub-agent's own
    AGENT_START instead, which always survives the visibility filter.
    """

    def add_tool(a: int, b: int) -> int:
        return a + b

    captured_sub_contexts: list[InteractionContext] = []

    sub_execute_events = [
        BaseEvent(
            event_type=ExecutionEventType.TOOL_START,
            author="SubAgent",
            action=ToolAction(
                name="add_tool", parameters={"a": 1, "b": 2}, thought="invoking add_tool"
            ),
        ),
        BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="SubAgent",
            content=[Part(text="3")],
            is_partial=False,
            need_feedback=False,
        ),
    ]
    sub_iter = iter(sub_execute_events)

    async def sub_execute(ctx: InteractionContext, *args, **kwargs) -> AsyncGenerator:
        captured_sub_contexts.append(ctx)
        for ev in sub_iter:
            yield ev
            return

    sub_agent = Agent(
        name="SubAgent",
        description="sub",
        instruction="sub instruction",
        llm_provider=llm_provider,
        tools=[FunctionTool(func=add_tool, approval_type=ApprovalType.USER_CONFIRM)],
        context_visibility=ContextVisibility.STANDARD,
    )
    sub_agent._execute = sub_execute

    main_agent = Agent(
        name="MainAgent",
        description="main",
        instruction="main instruction",
        llm_provider=llm_provider,
        sub_agents=[sub_agent],
    )
    main_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.AGENT_CALL,
                author="MainAgent",
                content=[],
                action=AgentAction(
                    name="SubAgent",
                    parameters={"task": "add 1 and 2"},
                    thought="delegating",
                ),
            ),
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="MainAgent",
                content=[Part(text="done")],
                is_partial=False,
                need_feedback=False,
            ),
        ]
    )

    application_name = "TestApp"
    user_id = "test_user"
    conversation_id = "feedback_subagent_standard_visibility"

    user_input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="please add 1 + 2")],
        turn_id="turn_1",
    )

    events_run_1 = [
        e
        async for e in runner.run(
            application_name=application_name,
            user_id=user_id,
            root=main_agent,
            user_input=user_input_event,
            conversation_id=conversation_id,
        )
    ]
    needs_feedback = [
        e for e in events_run_1 if e.event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK
    ]
    assert needs_feedback, "Sub-agent's tool should request approval"
    assert needs_feedback[0].author == "SubAgent"

    feedback_event = BaseEvent(
        event_type=ExecutionEventType.TOOL_FEEDBACK,
        author="user",
        action=FeedbackAction(
            name="add_tool",
            parameters=[
                FeedbackParameters(name="a", value=1, type=int),
                FeedbackParameters(name="b", value=2, type="int"),
            ],
            approval_type=ApprovalType.USER_CONFIRM,
            approve=True,
        ),
    )

    events_run_2 = [
        e
        async for e in runner.run(
            application_name=application_name,
            user_id=user_id,
            root=main_agent,
            user_input=feedback_event,
            conversation_id=conversation_id,
        )
    ]

    tool_ends = [e for e in events_run_2 if e.event_type == ExecutionEventType.TOOL_END]
    assert tool_ends, "Tool must execute after approval — no IndexError on input recovery"
    assert tool_ends[0].action.observation == 3

    assert len(captured_sub_contexts) >= 2, (
        "SubAgent should be invoked at least twice: once before approval, once after TOOL_FEEDBACK"
    )
    recovered = captured_sub_contexts[-1].agent_input
    assert recovered.event_type == ExecutionEventType.AGENT_START
    assert recovered.action.name == "SubAgent"

    assert all(e.turn_id == "turn_1" for e in events_run_2 if e.turn_id is not None)

    assert captured_sub_contexts[-1].is_subagent, (
        "is_subagent must remain True on TOOL_FEEDBACK resume inside a sub-agent"
    )


@pytest.mark.asyncio
async def test_flow_on_agent_event_hook_modifies_user_input_to_response_breaks_and_returns(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """on_agent_event updater hook replacing USER_INPUT with a RESPONSE event must break
    execution immediately and yield that RESPONSE to the consumer.

    No AGENT_START or agent loop execution should occur.

    Expected event sequence:
    1. HOOK_START  (hook fires on USER_INPUT → returns RESPONSE)
    2. HOOK_END
    3. RESPONSE    (the hook-modified event)
    """
    intercept_text = "Blocked at user input"

    def hook_fn(event: BaseEvent, context, **kwargs) -> BaseEvent | None:
        if event.event_type == ExecutionEventType.USER_INPUT:
            return BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="MainAgent",
                content=[Part(text=intercept_text)],
                is_partial=False,
                need_feedback=False,
                turn_id=event.turn_id,
            )
        return None

    hook = FunctionAgentHook(
        name="BlockInputHook",
        fun=hook_fn,
        trigger=HookPhase.ON_AGENT_EVENT,
        type=HookType.UPDATER,
    )

    main_agent = Agent(
        name="MainAgent",
        description="main agent",
        instruction="instruction",
        llm_provider=llm_provider,
        hooks=[hook],
    )
    main_agent._execute = async_gen_from_list([])

    input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="do something")],
        turn_id="turn_1",
    )

    events = [
        e
        async for e in runner.run(
            conversation_id="test_hook_response_on_user_input",
            application_name="TestApp",
            user_id="user1",
            root=main_agent,
            user_input=input_event,
        )
    ]

    event_types = [e.event_type for e in events]
    assert event_types == [
        ExecutionEventType.HOOK_START,
        ExecutionEventType.HOOK_END,
        ExecutionEventType.RESPONSE,
    ], f"Unexpected event sequence: {event_types}"

    response = events[2]
    assert response.content[0].text == intercept_text
    assert response.is_partial is False

    assert not any(e.event_type == ExecutionEventType.AGENT_START for e in events)
    assert not any(e.event_type == ExecutionEventType.AGENT_END for e in events)


@pytest.mark.asyncio
async def test_flow_on_agent_event_hook_modifies_subagent_start_to_response_breaks_and_returns(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """on_agent_event updater hook on a sub-agent replacing its AGENT_START with a RESPONSE
    event must break execution at handoff time and yield that RESPONSE to the consumer.

    The sub-agent must never execute. The parent's AGENT_CALL is still emitted because
    it was yielded before the sub-agent's hooks fired.

    Expected event sequence:
    1. AGENT_START  (MainAgent start — no hook on main agent)
    2. AGENT_CALL   (MainAgent delegates to SubAgent)
    3. HOOK_START   (SubAgent's AGENT_START → RESPONSE)
    4. HOOK_END
    5. RESPONSE     (the hook-modified event)
    """
    intercept_text = "Blocked at sub-agent start"

    def sub_hook_fn(event: BaseEvent, context, **kwargs) -> BaseEvent | None:
        if event.event_type == ExecutionEventType.AGENT_START:
            return BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="SubAgent",
                content=[Part(text=intercept_text)],
                is_partial=False,
                need_feedback=False,
                turn_id=event.turn_id,
            )
        return None

    sub_hook = FunctionAgentHook(
        name="BlockSubAgentStartHook",
        fun=sub_hook_fn,
        trigger=HookPhase.ON_AGENT_EVENT,
        type=HookType.UPDATER,
    )

    sub_agent = Agent(
        name="SubAgent",
        description="sub agent",
        instruction="instruction",
        llm_provider=llm_provider,
        hooks=[sub_hook],
    )
    sub_agent._execute = async_gen_from_list([])

    main_agent = Agent(
        name="MainAgent",
        description="main agent",
        instruction="instruction",
        llm_provider=llm_provider,
        sub_agents=[sub_agent],
    )
    main_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.AGENT_CALL,
                author="MainAgent",
                content=[],
                action=AgentAction(
                    name="SubAgent",
                    parameters={"task": "do something"},
                    thought="delegating",
                ),
            )
        ]
    )

    input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="do something")],
        turn_id="turn_1",
    )

    events = [
        e
        async for e in runner.run(
            conversation_id="test_hook_response_on_subagent_start",
            application_name="TestApp",
            user_id="user1",
            root=main_agent,
            user_input=input_event,
        )
    ]

    event_types = [e.event_type for e in events]
    assert event_types == [
        ExecutionEventType.AGENT_START,  # MainAgent
        ExecutionEventType.AGENT_CALL,  # MainAgent → SubAgent
        ExecutionEventType.HOOK_START,  # SubAgent AGENT_START → RESPONSE
        ExecutionEventType.HOOK_END,
        ExecutionEventType.RESPONSE,
    ], f"Unexpected event sequence: {event_types}"

    response = events[4]
    assert response.content[0].text == intercept_text
    assert response.is_partial is False

    # SubAgent's AGENT_START was replaced — it must never appear
    assert not any(
        e.event_type == ExecutionEventType.AGENT_START and e.author == "SubAgent" for e in events
    )


# ===========================================================================
# break_on_error tests
# ===========================================================================


def _make_raising_hook(
    trigger: HookPhase, hook_type: HookType, break_on_error: bool
) -> FunctionAgentHook:
    """Return a FunctionAgentHook whose callable always raises RuntimeError."""

    async def erroring_hook(context, agent):
        raise RuntimeError("intentional hook failure")

    erroring_hook.__signature__ = _inspect.Signature(
        [
            _inspect.Parameter("context", _inspect.Parameter.POSITIONAL_OR_KEYWORD),
            _inspect.Parameter("agent", _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ]
    )
    return FunctionAgentHook(
        name="ErroringHook",
        fun=erroring_hook,
        trigger=trigger,
        type=hook_type,
        break_on_error=break_on_error,
    )


@pytest.mark.asyncio
async def test_flow_pre_hook_break_on_error_true(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """A pre-hook that raises an exception with break_on_error=True must stop execution.

    Expected event sequence:
    1. AGENT_START  (user → MainAgent)
    2. HOOK_START   (pre-hook)
    3. HOOK_ERROR   (exception caught, break_on_error=True → break_execution=True)
    4. RESPONSE     (break_loop_response with guardrail content)
    The agent's _execute must never be iterated.
    """
    from domyn.agents.core import GuardrailResult

    hook = _make_raising_hook(HookPhase.PRE_AGENT_CALL, HookType.UPDATER, break_on_error=True)

    agent = Agent(
        name="MainAgent",
        description="main",
        instruction="main instruction",
        llm_provider=llm_provider,
        hooks=[hook],
    )
    executed = {"count": 0}

    async def _never_executed(*_, **__):
        executed["count"] += 1
        return
        yield  # make it an async generator

    agent._execute = _never_executed

    events = [
        e
        async for e in runner.run(
            conversation_id="test_pre_hook_break_on_error_true",
            application_name="TestApp",
            user_id="test_user",
            root=agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                author="user",
                content=[Part(text="hello")],
                turn_id="turn_1",
            ),
        )
    ]

    event_types = [e.event_type for e in events]
    assert event_types[0] == ExecutionEventType.AGENT_START
    assert ExecutionEventType.HOOK_START in event_types
    assert ExecutionEventType.HOOK_ERROR in event_types
    assert ExecutionEventType.RESPONSE in event_types
    # agent body must never have run
    assert executed["count"] == 0

    # The break RESPONSE must carry a non-empty message (not the generic fallback)
    response = next(e for e in events if e.event_type == ExecutionEventType.RESPONSE)
    assert response.content, "Break response must have content"
    all_text = " ".join(p.text for p in response.content if p.text)
    assert all_text, "Break response text must not be empty"

    # HOOK_ERROR must carry the hook name in metadata
    hook_error_event = next(e for e in events if e.event_type == ExecutionEventType.HOOK_ERROR)
    assert hook_error_event.metadata.get("hook_name") == "ErroringHook"


@pytest.mark.asyncio
async def test_flow_pre_hook_break_on_error_false(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """A pre-hook that raises with break_on_error=False must emit HOOK_ERROR but let
    execution continue so the agent runs and returns its response.

    Expected event sequence:
    1. AGENT_START  (user → MainAgent)
    2. HOOK_START   (pre-hook)
    3. HOOK_ERROR   (exception caught, break_on_error=False → no break)
    4. RESPONSE     (agent's own response — execution was NOT blocked)
    """
    hook = _make_raising_hook(HookPhase.PRE_AGENT_CALL, HookType.UPDATER, break_on_error=False)

    agent = Agent(
        name="MainAgent",
        description="main",
        instruction="main instruction",
        llm_provider=llm_provider,
        hooks=[hook],
    )
    agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="MainAgent",
                content=[Part(text="agent ran fine")],
                is_partial=False,
                need_feedback=False,
            )
        ]
    )

    events = [
        e
        async for e in runner.run(
            conversation_id="test_pre_hook_break_on_error_false",
            application_name="TestApp",
            user_id="test_user",
            root=agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                author="user",
                content=[Part(text="hello")],
                turn_id="turn_1",
            ),
        )
    ]

    event_types = [e.event_type for e in events]
    assert ExecutionEventType.HOOK_ERROR in event_types, (
        "Expected HOOK_ERROR for the raised exception"
    )

    # Execution must NOT have been broken — agent response must be present
    responses = [e for e in events if e.event_type == ExecutionEventType.RESPONSE]
    assert responses, "Agent response must still be emitted when break_on_error=False"
    assert any(p.text and "agent ran fine" in p.text for p in responses[-1].content), (
        "Final response must be the agent's own, not a break message"
    )


@pytest.mark.asyncio
async def test_flow_post_hook_break_on_error_true(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """A post-hook that raises with break_on_error=True must emit HOOK_ERROR and a break
    RESPONSE after the agent has already responded.

    Expected event sequence:
    1. AGENT_START
    2. RESPONSE     (agent's response)
    3. HOOK_START   (post-hook)
    4. HOOK_ERROR   (exception, break_on_error=True)
    5. RESPONSE     (break_loop_response)
    6. AGENT_END
    """
    hook = _make_raising_hook(HookPhase.POST_AGENT_CALL, HookType.UPDATER, break_on_error=True)

    agent = Agent(
        name="MainAgent",
        description="main",
        instruction="main instruction",
        llm_provider=llm_provider,
        hooks=[hook],
    )
    agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="MainAgent",
                content=[Part(text="agent answer")],
                is_partial=False,
                need_feedback=False,
            ),
            BaseEvent(event_type=ExecutionEventType.AGENT_END, author="MainAgent", content=[]),
        ]
    )

    events = [
        e
        async for e in runner.run(
            conversation_id="test_post_hook_break_on_error_true",
            application_name="TestApp",
            user_id="test_user",
            root=agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                author="user",
                content=[Part(text="hello")],
                turn_id="turn_1",
            ),
        )
    ]

    event_types = [e.event_type for e in events]
    assert ExecutionEventType.HOOK_ERROR in event_types
    # A break RESPONSE must appear after the HOOK_ERROR
    hook_error_idx = next(
        i for i, e in enumerate(events) if e.event_type == ExecutionEventType.HOOK_ERROR
    )
    responses_after_error = [
        e for e in events[hook_error_idx:] if e.event_type == ExecutionEventType.RESPONSE
    ]
    assert responses_after_error, "Break RESPONSE must follow the HOOK_ERROR"
    break_response = responses_after_error[0]
    assert any(p.text for p in break_response.content), "Break response must have non-empty text"


@pytest.mark.asyncio
async def test_flow_post_hook_break_on_error_false(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """A post-hook that raises with break_on_error=False must emit HOOK_ERROR but not
    break execution — the agent's AGENT_END must still appear.

    Expected event sequence:
    1. AGENT_START
    2. RESPONSE     (agent's response)
    3. HOOK_START   (post-hook)
    4. HOOK_ERROR   (exception, break_on_error=False — no break)
    5. AGENT_END
    """
    hook = _make_raising_hook(HookPhase.POST_AGENT_CALL, HookType.UPDATER, break_on_error=False)

    agent = Agent(
        name="MainAgent",
        description="main",
        instruction="main instruction",
        llm_provider=llm_provider,
        hooks=[hook],
    )
    agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="MainAgent",
                content=[Part(text="agent answer")],
                is_partial=False,
                need_feedback=False,
            ),
            BaseEvent(event_type=ExecutionEventType.AGENT_END, author="MainAgent", content=[]),
        ]
    )

    events = [
        e
        async for e in runner.run(
            conversation_id="test_post_hook_break_on_error_false",
            application_name="TestApp",
            user_id="test_user",
            root=agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                author="user",
                content=[Part(text="hello")],
                turn_id="turn_1",
            ),
        )
    ]

    event_types = [e.event_type for e in events]
    assert ExecutionEventType.HOOK_ERROR in event_types
    # No extra RESPONSE injected by break machinery
    responses = [e for e in events if e.event_type == ExecutionEventType.RESPONSE]
    assert len(responses) == 1, "Only the agent's own RESPONSE should appear"
    assert any(p.text and "agent answer" in p.text for p in responses[0].content)
    # Execution completed normally


def _make_success_false_hook(
    trigger: HookPhase, hook_type: HookType, break_on_error: bool
) -> FunctionAgentHook:
    """Return a FunctionAgentHook whose callable returns HookResult(success=False)."""

    async def failing_hook(context, agent):
        return HookResult(success=False, name="FailingHook")

    failing_hook.__signature__ = _inspect.Signature(
        [
            _inspect.Parameter("context", _inspect.Parameter.POSITIONAL_OR_KEYWORD),
            _inspect.Parameter("agent", _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ]
    )
    return FunctionAgentHook(
        name="FailingHook",
        fun=failing_hook,
        trigger=trigger,
        type=hook_type,
        break_on_error=break_on_error,
    )


@pytest.mark.asyncio
async def test_flow_pre_hook_success_false_break_on_error_true(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """A pre-hook that returns success=False with break_on_error=True must stop execution.

    The runner emits HOOK_ERROR for any HookResult with success=False.
    With break_on_error=True the hook also sets break_execution=True, so a break
    RESPONSE must follow and the agent's _execute must never be called.

    Expected event sequence:
    1. AGENT_START  (user → MainAgent)
    2. HOOK_START   (pre-hook)
    3. HOOK_ERROR   (success=False → runner maps to HOOK_ERROR)
    4. RESPONSE     (break_loop_response with guardrail content)
    """
    hook = _make_success_false_hook(
        HookPhase.PRE_AGENT_CALL, HookType.LISTENER, break_on_error=True
    )

    agent = Agent(
        name="MainAgent",
        description="main",
        instruction="main instruction",
        llm_provider=llm_provider,
        hooks=[hook],
    )
    execute_called = {"count": 0}

    async def _execute_spy(*args, **kwargs):
        execute_called["count"] += 1
        return
        yield  # make it an async generator

    agent._execute = _execute_spy

    events = [
        e
        async for e in runner.run(
            conversation_id="test_pre_hook_success_false_break_on_error",
            application_name="TestApp",
            user_id="test_user",
            root=agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                author="user",
                content=[Part(text="hello")],
                turn_id="turn_1",
            ),
        )
    ]

    event_types = [e.event_type for e in events]
    # success=False → runner emits HOOK_ERROR (same as an exception path)
    assert ExecutionEventType.HOOK_ERROR in event_types
    # A break RESPONSE must appear after the HOOK_ERROR
    hook_error_idx = next(
        i for i, e in enumerate(events) if e.event_type == ExecutionEventType.HOOK_ERROR
    )
    responses_after = [
        e for e in events[hook_error_idx:] if e.event_type == ExecutionEventType.RESPONSE
    ]
    assert responses_after, "Break RESPONSE must be emitted after HOOK_ERROR"
    assert any(p.text for p in responses_after[0].content), (
        "Break response must have non-empty text"
    )
    # The agent's _execute must not have been called
    assert execute_called["count"] == 0, "Agent _execute must not run when pre-hook breaks"


@pytest.mark.asyncio
async def test_flow_post_hook_success_false_break_on_error_true(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """A post-hook that returns success=False with break_on_error=True must inject a break RESPONSE.

    The runner emits HOOK_ERROR for any HookResult with success=False.
    With break_on_error=True the hook also sets break_execution=True, so a break
    RESPONSE must follow the HOOK_ERROR.

    Expected event sequence:
    1. AGENT_START
    2. RESPONSE     (agent's own response)
    3. HOOK_START   (post-hook)
    4. HOOK_ERROR   (success=False → runner maps to HOOK_ERROR)
    5. RESPONSE     (break_loop_response with guardrail content)
    6. AGENT_END
    """
    hook = _make_success_false_hook(
        HookPhase.POST_AGENT_CALL, HookType.LISTENER, break_on_error=True
    )

    agent = Agent(
        name="MainAgent",
        description="main",
        instruction="main instruction",
        llm_provider=llm_provider,
        hooks=[hook],
    )
    agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="MainAgent",
                content=[Part(text="agent answer")],
                is_partial=False,
                need_feedback=False,
            ),
            BaseEvent(event_type=ExecutionEventType.AGENT_END, author="MainAgent", content=[]),
        ]
    )

    events = [
        e
        async for e in runner.run(
            conversation_id="test_post_hook_success_false_break_on_error",
            application_name="TestApp",
            user_id="test_user",
            root=agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                author="user",
                content=[Part(text="hello")],
                turn_id="turn_1",
            ),
        )
    ]

    event_types = [e.event_type for e in events]
    assert ExecutionEventType.HOOK_ERROR in event_types
    # A break RESPONSE must appear after the HOOK_ERROR
    hook_error_idx = next(
        i for i, e in enumerate(events) if e.event_type == ExecutionEventType.HOOK_ERROR
    )
    responses_after = [
        e for e in events[hook_error_idx:] if e.event_type == ExecutionEventType.RESPONSE
    ]
    assert responses_after, "Break RESPONSE must be emitted after HOOK_ERROR"
    assert any(p.text for p in responses_after[0].content), (
        "Break response must have non-empty text"
    )
    # The agent's own RESPONSE must also appear (before the hook error)
    responses_before = [
        e for e in events[:hook_error_idx] if e.event_type == ExecutionEventType.RESPONSE
    ]
    assert any(
        any(p.text and "agent answer" in p.text for p in e.content) for e in responses_before
    ), "Agent's RESPONSE must appear before the hook breaks"
    assert ExecutionEventType.AGENT_END in event_types


@pytest.mark.asyncio
async def test_flow_on_agent_event_hook_success_false_break_on_error_true(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """An on_agent_event hook that returns success=False with break_on_error=True must stop execution.

    The hook fires on the USER_INPUT event (before the agent loop starts). Because
    break_on_error=True, the framework sets break_execution=True on the result, so the
    runner must emit a break RESPONSE and never call the agent's _execute.

    The runner maps success=False → HOOK_ERROR (same as the exception path).

    Expected event sequence:
    1. HOOK_START
    2. HOOK_ERROR   (success=False → runner maps to HOOK_ERROR)
    3. RESPONSE     (break_loop_response with guardrail content)
    Agent _execute must NOT be called.
    """
    executed = {"count": 0}

    def failing_on_event_hook(event: BaseEvent, context: Any, **_) -> HookResult:
        return HookResult(success=False, name="FailingOnEventHook")

    hook = _make_on_event_listener_hook(failing_on_event_hook)
    hook.on_event_triggers = [ExecutionEventType.USER_INPUT]
    hook.break_on_error = True

    agent = Agent(
        name="MainAgent",
        description="main",
        instruction="main instruction",
        llm_provider=llm_provider,
        hooks=[hook],
    )

    async def counting_execute(*args, **kwargs):
        executed["count"] += 1
        return
        yield  # make it an async generator

    agent._execute = counting_execute

    events = [
        e
        async for e in runner.run(
            application_name="TestApp",
            user_id="test_user",
            root=agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                author="user",
                content=[Part(text="hello")],
                turn_id="turn_on_event_success_false",
            ),
            conversation_id="flow_on_agent_event_success_false_break_on_error",
        )
    ]

    event_types = [e.event_type for e in events]
    assert ExecutionEventType.HOOK_START in event_types
    # success=False → runner emits HOOK_ERROR
    assert ExecutionEventType.HOOK_ERROR in event_types
    # A break RESPONSE must follow the HOOK_ERROR
    hook_error_idx = next(
        i for i, e in enumerate(events) if e.event_type == ExecutionEventType.HOOK_ERROR
    )
    responses_after = [
        e for e in events[hook_error_idx:] if e.event_type == ExecutionEventType.RESPONSE
    ]
    assert responses_after, "Break RESPONSE must be emitted after HOOK_ERROR"
    assert any(p.text for p in responses_after[0].content), (
        "Break response must have non-empty text"
    )
    # AGENT_START must not appear — runner exited before the agent loop
    assert ExecutionEventType.AGENT_START not in event_types
    assert executed["count"] == 0, "Agent _execute must not run when on_agent_event hook breaks"


# ---------------------------------------------------------------------------
# DelegateAgent HITL (TOOL_NEEDS_FEEDBACK / TOOL_FEEDBACK) flow tests
# ---------------------------------------------------------------------------


class _HitlTransport(RunnerTransport):
    """Two-phase transport for HITL tests.

    First execute() call yields *initial_events* (e.g. TOOL_START) then the
    runner closes the iterator and pauses waiting for user approval.

    send_result() queues the TOOL_END result (mirrors real transport
    _pending_outbound semantics).

    Second execute() call detects the queued result and yields
    *post_result_events* (e.g. the remote agent's final RESPONSE), mirroring
    how the real WS/SSE transports skip AGENT_START and send TOOL_END directly
    when _pending_outbound is pre-set.

    sent_results records every event passed to send_result() so tests can
    assert what the runner sent back to the remote agent.
    """

    def __init__(
        self,
        initial_events: list[BaseEvent],
        post_result_events: list[BaseEvent],
    ) -> None:
        self._initial_events = list(initial_events)
        self._post_result_events = list(post_result_events)
        self._pending_result: BaseEvent | None = None
        self.sent_results: list[BaseEvent] = []

    async def execute(
        self, agent_name: str, context: InteractionContext
    ) -> AsyncGenerator[BaseEvent, None]:
        if self._pending_result is not None:
            # Resume mode: send_result() was called before this execute() —
            # skip initial events and yield post-result events directly.
            self._pending_result = None
            for event in self._post_result_events:
                yield event
        else:
            for event in self._initial_events:
                yield event

    async def send_result(self, event: BaseEvent, conversation_id: str | None = None) -> None:
        self.sent_results.append(event)
        self._pending_result = event


def _make_delegate_agent(
    transport: _HitlTransport,
    approval_type: ApprovalType,
) -> DelegateAgent:
    def add_tool(a: int, b: int) -> int:
        return a + b

    return DelegateAgent(
        name="RemoteAgent",
        description="remote agent",
        transport=transport,
        tools=[FunctionTool(func=add_tool, approval_type=approval_type)],
    )


@pytest.mark.asyncio
async def test_flow_delegate_agent_tool_needs_feedback(runner: Runner) -> None:
    """TOOL_START from the remote agent triggers TOOL_NEEDS_FEEDBACK when the
    intercepted local tool has an approval_type.  TOOL_START (carrying its
    thought) must be forwarded to the consumer immediately before
    TOOL_NEEDS_FEEDBACK, matching regular-agent behavior."""
    transport = _HitlTransport(
        initial_events=[
            BaseEvent(
                event_type=ExecutionEventType.TOOL_START,
                author="RemoteAgent",
                action=ToolAction(
                    name="add_tool", parameters={"a": 1, "b": 2}, thought="calling add_tool"
                ),
            )
        ],
        post_result_events=[],
    )
    agent = _make_delegate_agent(transport, ApprovalType.USER_CONFIRM)

    events = [
        e
        async for e in runner.run(
            application_name="app",
            user_id="user",
            root=agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                author="user",
                content=[Part(text="go")],
                turn_id="turn_1",
            ),
            conversation_id="test_delegate_hitl_needs_feedback",
        )
    ]

    event_types = [e.event_type for e in events]
    assert event_types[-2:] == [
        ExecutionEventType.TOOL_START,
        ExecutionEventType.TOOL_NEEDS_FEEDBACK,
    ], "TOOL_START must be forwarded immediately before TOOL_NEEDS_FEEDBACK"

    tool_start = events[-2]
    assert tool_start.author == "RemoteAgent"
    assert tool_start.action.thought == "calling add_tool"

    tnf = events[-1]
    assert tnf.author == "RemoteAgent"
    assert isinstance(tnf.action, FeedbackAction)
    assert tnf.action.name == "add_tool"
    assert tnf.action.parameters == [
        FeedbackParameters(name="a", value=1, type="int"),
        FeedbackParameters(name="b", value=2, type="int"),
    ]
    # The forwarded TOOL_START above carries the thought; this event must not repeat it.
    assert tnf.action.thought is None


@pytest.mark.asyncio
async def test_flow_delegate_agent_tool_feedback_approve(runner: Runner) -> None:
    """Full HITL round-trip for a DelegateAgent local tool.

    Run 1: remote TOOL_START → TOOL_NEEDS_FEEDBACK emitted, runner pauses.
    Run 2: TOOL_FEEDBACK (approve=True) → tool executes, TOOL_END yielded to
           consumer and sent back to remote agent via transport.send_result().
           Remote agent continues and emits a final RESPONSE."""
    transport = _HitlTransport(
        initial_events=[
            BaseEvent(
                event_type=ExecutionEventType.TOOL_START,
                author="RemoteAgent",
                action=ToolAction(
                    name="add_tool", parameters={"a": 1, "b": 2}, thought="calling add_tool"
                ),
            )
        ],
        post_result_events=[
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="RemoteAgent",
                content=[Part(text="The result is 3")],
                is_partial=False,
            )
        ],
    )
    agent = _make_delegate_agent(transport, ApprovalType.USER_CONFIRM)
    conv_id = "test_delegate_hitl_approve"
    app, user = "app", "user"

    # Run 1 — establish TOOL_NEEDS_FEEDBACK
    turn_id = "turn_1"
    run1 = [
        e
        async for e in runner.run(
            application_name=app,
            user_id=user,
            root=agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                author="user",
                content=[Part(text="go")],
                turn_id=turn_id,
            ),
            conversation_id=conv_id,
        )
    ]
    assert run1[-1].event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK

    # Run 2 — approve
    run2 = [
        e
        async for e in runner.run(
            application_name=app,
            user_id=user,
            root=agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.TOOL_FEEDBACK,
                author="user",
                action=FeedbackAction(
                    name="add_tool",
                    parameters=[
                        FeedbackParameters(name="a", value=1, type="int"),
                        FeedbackParameters(name="b", value=2, type="int"),
                    ],
                    approval_type=ApprovalType.USER_CONFIRM,
                    approve=True,
                ),
            ),
            conversation_id=conv_id,
        )
    ]

    event_types2 = [e.event_type for e in run2]
    assert ExecutionEventType.TOOL_END in event_types2

    tool_end = next(e for e in run2 if e.event_type == ExecutionEventType.TOOL_END)
    assert tool_end.action.observation == 3

    # transport must have received TOOL_END so the remote agent can resume
    assert len(transport.sent_results) == 1
    assert transport.sent_results[0].event_type == ExecutionEventType.TOOL_END
    assert transport.sent_results[0].action.observation == 3

    # Remote agent continued after receiving the result
    assert ExecutionEventType.RESPONSE in event_types2

    # turn_id is preserved across the feedback round-trip
    assert all(e.turn_id == turn_id for e in run2)


@pytest.mark.asyncio
async def test_flow_delegate_agent_approval_with_the_remote_runners_own_ids(
    runner: Runner,
) -> None:
    """A remote runner stamps its own interaction_id and clock; its approval still resumes.

    Regression: the request was stored under the remote's interaction_id, so the reply
    found nothing to answer; and the remote's clock placed it after what came next.
    """
    remote_ids = {
        "interaction_id": "remote-interaction",
        "timestamp": datetime.now() + timedelta(hours=2),
    }
    transport = _HitlTransport(
        initial_events=[
            BaseEvent(
                event_type=ExecutionEventType.TOOL_START,
                author="RemoteAgent",
                action=ToolAction(name="add_tool", parameters={"a": 1, "b": 2}),
                **remote_ids,
            )
        ],
        post_result_events=[
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="RemoteAgent",
                content=[Part(text="The result is 3")],
                is_partial=False,
                **remote_ids,
            )
        ],
    )
    agent = _make_delegate_agent(transport, ApprovalType.USER_CONFIRM)

    async def turn(user_input: BaseEvent) -> list[BaseEvent]:
        return [
            e
            async for e in runner.run(
                application_name="app",
                user_id="user",
                root=agent,
                user_input=user_input,
                conversation_id="test_delegate_hitl_remote_ids",
            )
        ]

    asked = await turn(
        BaseEvent(
            event_type=ExecutionEventType.USER_INPUT,
            author="user",
            content=[Part(text="go")],
            turn_id="turn_1",
        )
    )
    tool_start, request = asked[-2:]
    assert request.event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK
    assert tool_start.interaction_id == request.interaction_id
    assert tool_start.timestamp <= request.timestamp

    resumed = await turn(
        BaseEvent(
            event_type=ExecutionEventType.TOOL_FEEDBACK,
            author="user",
            action=FeedbackAction(
                name="add_tool",
                parameters=[
                    FeedbackParameters(name="a", value=1, type="int"),
                    FeedbackParameters(name="b", value=2, type="int"),
                ],
                approval_type=ApprovalType.USER_CONFIRM,
                approve=True,
            ),
        )
    )
    assert [e.action.observation for e in transport.sent_results] == [3]
    assert ExecutionEventType.RESPONSE in [e.event_type for e in resumed]


@pytest.mark.asyncio
async def test_flow_delegate_agent_tool_feedback_params_update(runner: Runner) -> None:
    """TOOL_FEEDBACK with updated parameters executes the tool with the new values.
    The TOOL_END observation must contain the params-changed notice and reflect
    the updated result.  The remote agent receives the enriched TOOL_END."""
    transport = _HitlTransport(
        initial_events=[
            BaseEvent(
                event_type=ExecutionEventType.TOOL_START,
                author="RemoteAgent",
                action=ToolAction(
                    name="add_tool", parameters={"a": 1, "b": 2}, thought="calling add_tool"
                ),
            )
        ],
        post_result_events=[
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="RemoteAgent",
                content=[Part(text="The result is 4")],
                is_partial=False,
            )
        ],
    )
    agent = _make_delegate_agent(transport, ApprovalType.PARAMS_UPDATE)
    conv_id = "test_delegate_hitl_params_update"
    app, user = "app", "user"

    # Run 1
    run1 = [
        e
        async for e in runner.run(
            application_name=app,
            user_id=user,
            root=agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                author="user",
                content=[Part(text="go")],
                turn_id="turn_1",
            ),
            conversation_id=conv_id,
        )
    ]
    assert run1[-1].event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK

    # Run 2 — approve with b=3 instead of b=2
    run2 = [
        e
        async for e in runner.run(
            application_name=app,
            user_id=user,
            root=agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.TOOL_FEEDBACK,
                author="user",
                action=FeedbackAction(
                    name="add_tool",
                    parameters=[
                        FeedbackParameters(name="a", value=1, type="int"),
                        FeedbackParameters(name="b", value=3, type="int"),
                    ],
                    approval_type=ApprovalType.PARAMS_UPDATE,
                    approve=True,
                ),
            ),
            conversation_id=conv_id,
        )
    ]

    tool_end = next(e for e in run2 if e.event_type == ExecutionEventType.TOOL_END)
    # Tool ran with updated b=3 → 1 + 3 = 4; observation has result + params-changed notice
    assert str(tool_end.action.observation).startswith("4")
    assert "IMPORTANT" in str(tool_end.action.observation)

    # Transport receives enriched TOOL_END with param-change notice
    assert len(transport.sent_results) == 1
    sent = transport.sent_results[0]
    assert sent.event_type == ExecutionEventType.TOOL_END
    assert "IMPORTANT" in str(sent.action.observation), (
        "params-changed notice must appear in the observation sent to the remote agent"
    )


@pytest.mark.asyncio
async def test_flow_delegate_agent_tool_feedback_reject(runner: Runner) -> None:
    """TOOL_FEEDBACK with approve=False yields TOOL_END with a refusal observation
    and sends it back via transport so the remote agent can handle the rejection."""
    transport = _HitlTransport(
        initial_events=[
            BaseEvent(
                event_type=ExecutionEventType.TOOL_START,
                author="RemoteAgent",
                action=ToolAction(
                    name="add_tool", parameters={"a": 1, "b": 2}, thought="calling add_tool"
                ),
            )
        ],
        post_result_events=[
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="RemoteAgent",
                content=[Part(text="User refused the tool")],
                is_partial=False,
            )
        ],
    )
    agent = _make_delegate_agent(transport, ApprovalType.USER_CONFIRM)
    conv_id = "test_delegate_hitl_reject"
    app, user = "app", "user"

    # Run 1
    run1 = [
        e
        async for e in runner.run(
            application_name=app,
            user_id=user,
            root=agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                author="user",
                content=[Part(text="go")],
                turn_id="turn_1",
            ),
            conversation_id=conv_id,
        )
    ]
    assert run1[-1].event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK

    # Run 2 — reject
    run2 = [
        e
        async for e in runner.run(
            application_name=app,
            user_id=user,
            root=agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.TOOL_FEEDBACK,
                author="user",
                action=FeedbackAction(
                    name="add_tool",
                    parameters=[
                        FeedbackParameters(name="a", value=1, type="int"),
                        FeedbackParameters(name="b", value=2, type="int"),
                    ],
                    approval_type=ApprovalType.USER_CONFIRM,
                    approve=False,
                ),
            ),
            conversation_id=conv_id,
        )
    ]

    # The refused TOOL_END is yielded to the consumer, saved to history, and sent via transport
    assert ExecutionEventType.TOOL_END in [e.event_type for e in run2]
    tool_end = next(e for e in run2 if e.event_type == ExecutionEventType.TOOL_END)
    assert "refused" in str(tool_end.action.observation).lower()

    # Transport receives the refusal TOOL_END so the remote agent can react
    assert len(transport.sent_results) == 1
    sent = transport.sent_results[0]
    assert sent.event_type == ExecutionEventType.TOOL_END
    assert "refused" in str(sent.action.observation).lower()

    # Remote agent continued and emitted a response after the rejection
    assert ExecutionEventType.RESPONSE in [e.event_type for e in run2]


@pytest.mark.asyncio
async def test_flow_delegate_agent_own_approval_reply_reaches_it(runner: Runner) -> None:
    """A reply to the delegate's own approval request is forwarded to it."""

    class _Recording(RunnerTransport):
        def __init__(self) -> None:
            self.inputs: list[BaseEvent] = []

        async def execute(
            self, agent_name: str, context: InteractionContext
        ) -> AsyncGenerator[BaseEvent, None]:
            self.inputs.append(context.agent_input)
            if len(self.inputs) == 1:
                # The delegate's own tool, then its own approval request.
                yield BaseEvent(
                    event_type=ExecutionEventType.TOOL_START,
                    author="RemoteAgent",
                    action=ToolAction(name="terminal", parameters={"command": "rm -rf build"}),
                )
                yield BaseEvent(
                    event_type=ExecutionEventType.TOOL_NEEDS_FEEDBACK,
                    author="RemoteAgent",
                    action=FeedbackAction(
                        name="exec_approval",
                        approval_type=ApprovalType.USER_CONFIRM,
                        call_id="confirm-1",
                    ),
                )
            else:
                yield BaseEvent(
                    event_type=ExecutionEventType.RESPONSE,
                    author="RemoteAgent",
                    content=[Part(text="done")],
                    is_partial=False,
                )

        async def send_result(self, event: BaseEvent, conversation_id: str | None = None) -> None:
            pass

    transport = _Recording()
    agent = DelegateAgent(name="RemoteAgent", description="remote", transport=transport)

    async def send(event: BaseEvent) -> list[BaseEvent]:
        return [
            e
            async for e in runner.run(
                application_name="app",
                user_id="user",
                root=agent,
                user_input=event,
                conversation_id="test_delegate_own_approval",
            )
        ]

    first = await send(
        BaseEvent(
            event_type=ExecutionEventType.USER_INPUT,
            author="user",
            content=[Part(text="clean up")],
            turn_id="turn_1",
        )
    )
    asked = first[-1]
    assert asked.event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK
    asked.action.approve = True
    second = await send(
        BaseEvent(
            event_type=ExecutionEventType.TOOL_FEEDBACK,
            author="user",
            action=asked.action,
            turn_id="turn_1",
        )
    )
    assert transport.inputs[-1].event_type == ExecutionEventType.TOOL_FEEDBACK
    assert transport.inputs[-1].action.name == "exec_approval"
    assert ExecutionEventType.RESPONSE in [e.event_type for e in second]


# ---------------------------------------------------------------------------
# Reinstantiation tests — verify stateless HITL resume across object rebuilds
# ---------------------------------------------------------------------------


def _make_hitl_transport_and_agent(
    approval_type: ApprovalType,
) -> tuple[_HitlTransport, DelegateAgent]:
    """Return a fresh transport+agent pair that emit TOOL_START then RESPONSE."""
    transport = _HitlTransport(
        initial_events=[
            BaseEvent(
                event_type=ExecutionEventType.TOOL_START,
                author="RemoteAgent",
                action=ToolAction(
                    name="add_tool", parameters={"a": 1, "b": 2}, thought="calling add_tool"
                ),
            )
        ],
        post_result_events=[
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="RemoteAgent",
                content=[Part(text="The result is 3")],
                is_partial=False,
            )
        ],
    )
    agent = _make_delegate_agent(transport, approval_type)
    return transport, agent


def _make_tool_feedback_event() -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.TOOL_FEEDBACK,
        author="user",
        action=FeedbackAction(
            name="add_tool",
            parameters=[
                FeedbackParameters(name="a", value=1, type="int"),
                FeedbackParameters(name="b", value=2, type="int"),
            ],
            approval_type=ApprovalType.USER_CONFIRM,
            approve=True,
        ),
    )


@pytest.mark.asyncio
async def test_flow_delegate_hitl_rebuilt_agent_and_transport() -> None:
    """HITL round-trip where DelegateAgent and transport are rebuilt between
    run 1 and run 2, simulating per-request agent reconstruction.

    The same Runner (shared conversation_manager) is used, but each run
    receives a brand-new DelegateAgent + transport instance.  The stateless
    resume must still deliver TOOL_END to the new transport and receive the
    remote RESPONSE."""
    conv_mgr = InMemoryConversationManager()
    runner = Runner(conversation_manager=conv_mgr)
    conv_id = "test_hitl_rebuilt_agent_transport"
    app, user = "app", "user"

    _, agent1 = _make_hitl_transport_and_agent(ApprovalType.USER_CONFIRM)
    run1 = [
        e
        async for e in runner.run(
            application_name=app,
            user_id=user,
            root=agent1,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                author="user",
                content=[Part(text="go")],
                turn_id="turn_1",
            ),
            conversation_id=conv_id,
        )
    ]
    assert run1[-1].event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK

    # Run 2 uses a completely new transport + agent
    transport2, agent2 = _make_hitl_transport_and_agent(ApprovalType.USER_CONFIRM)
    run2 = [
        e
        async for e in runner.run(
            application_name=app,
            user_id=user,
            root=agent2,
            user_input=_make_tool_feedback_event(),
            conversation_id=conv_id,
        )
    ]

    event_types2 = [e.event_type for e in run2]
    assert ExecutionEventType.TOOL_END in event_types2
    tool_end = next(e for e in run2 if e.event_type == ExecutionEventType.TOOL_END)
    assert tool_end.action.observation == 3

    assert len(transport2.sent_results) == 1
    assert transport2.sent_results[0].event_type == ExecutionEventType.TOOL_END
    assert transport2.sent_results[0].action.observation == 3

    assert ExecutionEventType.RESPONSE in event_types2


@pytest.mark.asyncio
async def test_flow_delegate_hitl_rebuilt_runner() -> None:
    """HITL round-trip where the Runner itself is rebuilt between run 1 and
    run 2, simulating a server restart.

    A shared InMemoryConversationManager carries the persisted event history
    so the second runner can reconstruct the conversation.  The stateless
    resume must still work with a fresh runner that has no _paused_iterators."""
    conv_mgr = InMemoryConversationManager()
    runner1 = Runner(conversation_manager=conv_mgr)
    conv_id = "test_hitl_rebuilt_runner"
    app, user = "app", "user"

    _, agent1 = _make_hitl_transport_and_agent(ApprovalType.USER_CONFIRM)
    run1 = [
        e
        async for e in runner1.run(
            application_name=app,
            user_id=user,
            root=agent1,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                author="user",
                content=[Part(text="go")],
                turn_id="turn_1",
            ),
            conversation_id=conv_id,
        )
    ]
    assert run1[-1].event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK

    # Simulate runner restart: brand-new Runner sharing the same conv_mgr
    runner2 = Runner(conversation_manager=conv_mgr)
    transport2, agent2 = _make_hitl_transport_and_agent(ApprovalType.USER_CONFIRM)
    run2 = [
        e
        async for e in runner2.run(
            application_name=app,
            user_id=user,
            root=agent2,
            user_input=_make_tool_feedback_event(),
            conversation_id=conv_id,
        )
    ]

    event_types2 = [e.event_type for e in run2]
    assert ExecutionEventType.TOOL_END in event_types2
    tool_end = next(e for e in run2 if e.event_type == ExecutionEventType.TOOL_END)
    assert tool_end.action.observation == 3

    assert len(transport2.sent_results) == 1
    assert transport2.sent_results[0].action.observation == 3

    assert ExecutionEventType.RESPONSE in event_types2


@pytest.mark.asyncio
async def test_flow_delegate_hitl_full_rebuild() -> None:
    """HITL round-trip with runner, agent, and transport all rebuilt between
    run 1 and run 2 — the most extreme reinstantiation scenario.

    Only the conversation_manager is shared (as it would be in a real system
    backed by a persistent store).  Verifies that no in-process object identity
    is required for the stateless resume to succeed."""
    conv_mgr = InMemoryConversationManager()
    conv_id = "test_hitl_full_rebuild"
    app, user = "app", "user"

    runner1 = Runner(conversation_manager=conv_mgr)
    _, agent1 = _make_hitl_transport_and_agent(ApprovalType.USER_CONFIRM)
    run1 = [
        e
        async for e in runner1.run(
            application_name=app,
            user_id=user,
            root=agent1,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                author="user",
                content=[Part(text="go")],
                turn_id="turn_1",
            ),
            conversation_id=conv_id,
        )
    ]
    assert run1[-1].event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK

    # Everything rebuilt: new runner, new agent, new transport
    runner2 = Runner(conversation_manager=conv_mgr)
    transport2, agent2 = _make_hitl_transport_and_agent(ApprovalType.USER_CONFIRM)
    run2 = [
        e
        async for e in runner2.run(
            application_name=app,
            user_id=user,
            root=agent2,
            user_input=_make_tool_feedback_event(),
            conversation_id=conv_id,
        )
    ]

    event_types2 = [e.event_type for e in run2]
    assert ExecutionEventType.TOOL_END in event_types2
    tool_end = next(e for e in run2 if e.event_type == ExecutionEventType.TOOL_END)
    assert tool_end.action.observation == 3

    assert len(transport2.sent_results) == 1
    assert transport2.sent_results[0].action.observation == 3

    assert ExecutionEventType.RESPONSE in event_types2


@pytest.mark.asyncio
async def test_flow_delegate_hitl_call_id_preserved() -> None:
    """The TOOL_END sent to the remote agent after HITL must carry the original
    call_id from the TOOL_START, not the fresh UUID generated by the client's
    TOOL_FEEDBACK event.

    Remote runners (e.g. LangGraph Runtime) key pending platform-tool futures
    by call_id.  A mismatched call_id silently drops the response and leaves
    the remote agent hanging."""
    captured_tool_ends: list[BaseEvent] = []

    class _CapturingTransport(_HitlTransport):
        async def send_result(self, event: BaseEvent, conversation_id: str | None = None) -> None:
            captured_tool_ends.append(event)
            await super().send_result(event)

    original_call_id = "fixed-call-id-from-tool-start"

    transport = _CapturingTransport(
        initial_events=[
            BaseEvent(
                event_type=ExecutionEventType.TOOL_START,
                author="RemoteAgent",
                action=ToolAction(
                    name="add_tool",
                    parameters={"a": 1, "b": 2},
                    call_id=original_call_id,
                ),
            )
        ],
        post_result_events=[
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="RemoteAgent",
                content=[Part(text="done")],
                is_partial=False,
            )
        ],
    )
    agent = _make_delegate_agent(transport, ApprovalType.USER_CONFIRM)
    conv_mgr = InMemoryConversationManager()
    runner = Runner(conversation_manager=conv_mgr)
    conv_id = "test_hitl_call_id"
    app, user = "app", "user"

    run1 = [
        e
        async for e in runner.run(
            application_name=app,
            user_id=user,
            root=agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                author="user",
                content=[Part(text="go")],
                turn_id="turn_1",
            ),
            conversation_id=conv_id,
        )
    ]
    assert run1[-1].event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK

    # Client sends TOOL_FEEDBACK WITHOUT including the original call_id
    # (fresh UUID from default_factory) — this is the normal client behaviour.
    feedback = _make_tool_feedback_event()
    assert feedback.action.call_id != original_call_id, (
        "Precondition: fresh FeedbackAction must have a different call_id"
    )

    # Submit the feedback — the runner should restore the original call_id.
    [
        e
        async for e in runner.run(
            application_name=app,
            user_id=user,
            root=agent,
            user_input=feedback,
            conversation_id=conv_id,
        )
    ]

    assert len(captured_tool_ends) == 1
    sent = captured_tool_ends[0]
    assert sent.action.call_id == original_call_id, (
        f"TOOL_END must carry the original call_id '{original_call_id}', "
        f"got '{sent.action.call_id}' instead"
    )
