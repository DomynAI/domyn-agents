"""Tests that verify the agent stack is emptied when a hook breaks execution early.

Without the fix, a stale stack entry would cause the next user-input turn to be
treated as a resumed conversation (is_new_conversation=False), suppressing the
AGENT_START event from being persisted in history. In the sub-agent break cases
the stale entry also points to the sub-agent, causing the wrong agent to be
resumed on the next turn.
"""

import inspect
from collections.abc import AsyncGenerator

import pytest

from domyn.agents import Runner
from domyn.agents.agents import Agent
from domyn.agents.core import (
    AgentAction,
    BaseEvent,
    ExecutionEventType,
    Part,
)
from domyn.agents.hooks import FunctionHook, HookPhase, HookResult, HookType
from domyn.agents.llm import BaseLLMProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_APP = "TestApp"
_USER = "test_user"


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


def _user_input(text: str = "hello", turn_id: str = "turn_1") -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text=text)],
        turn_id=turn_id,
    )


def _breaking_agent_hook(trigger: HookPhase) -> FunctionHook:
    """Pre/post agent hook that always breaks execution."""

    async def _fn(context, agent) -> HookResult:
        return HookResult(
            success=True,
            emitted_content=[Part(text="blocked by hook")],
            break_execution=True,
            name="_breaking_agent_hook",
        )

    _fn.__signature__ = inspect.Signature(
        [
            inspect.Parameter("context", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("agent", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ]
    )
    return FunctionHook(
        name="BreakingHook",
        fun=_fn,
        trigger=trigger,
        type=HookType.LISTENER,
    )


def _breaking_on_event_hook(on_triggers: list[ExecutionEventType] | None = None) -> FunctionHook:
    """on_agent_event hook that breaks execution when it fires."""

    async def _fn(event: BaseEvent, context, **_) -> HookResult:
        return HookResult(
            success=True,
            emitted_content=[Part(text="blocked by on_event hook")],
            break_execution=True,
            name="_breaking_on_event_hook",
        )

    return FunctionHook(
        fun=_fn,
        trigger=HookPhase.ON_AGENT_EVENT,
        type=HookType.UPDATER,
        on_event_triggers=on_triggers or [ExecutionEventType.USER_INPUT],
    )


async def _drain(runner_gen) -> list[BaseEvent]:
    return [e async for e in runner_gen]


async def _stack_is_empty(runner: Runner, conv_id: str) -> bool:
    entry = await runner.conversation_manager.get_last_agent_stack(_APP, _USER, conv_id)
    return entry is None


# ---------------------------------------------------------------------------
# Case 1: on_agent_event hook breaks on USER_INPUT (run() early-return path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_input_on_event_hook_break_empties_stack(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """When an on_agent_event hook breaks on USER_INPUT, the stack must be empty
    afterwards so the next turn is treated as a new conversation."""

    agent = Agent(
        name="Agent",
        description="desc",
        instruction="instr",
        llm_provider=llm_provider,
        hooks=[_breaking_on_event_hook([ExecutionEventType.USER_INPUT])],
    )
    agent._execute = async_gen_from_list([])

    conv_id = "test_user_input_hook_break_stack"

    await _drain(
        runner.run(_APP, _USER, root=agent, user_input=_user_input(), conversation_id=conv_id)
    )

    assert await _stack_is_empty(runner, conv_id), (
        "Stack must be empty after hook break on USER_INPUT"
    )


@pytest.mark.asyncio
async def test_user_input_on_event_hook_break_second_turn_starts_fresh(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """After an on_agent_event hook breaks on USER_INPUT, the next user-input turn
    must yield and persist an AGENT_START (is_new_conversation=True path)."""

    agent = Agent(
        name="Agent",
        description="desc",
        instruction="instr",
        llm_provider=llm_provider,
        hooks=[_breaking_on_event_hook([ExecutionEventType.USER_INPUT])],
    )
    agent._execute = async_gen_from_list([])

    conv_id = "test_user_input_hook_break_second_turn"

    # First turn: hook breaks on USER_INPUT
    await _drain(
        runner.run(
            _APP,
            _USER,
            root=agent,
            user_input=_user_input(turn_id="turn_1"),
            conversation_id=conv_id,
        )
    )

    # Remove the breaking hook so the second turn can proceed normally and emit AGENT_START
    agent.hooks = []
    agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="Agent",
                content=[Part(text="ok")],
                is_partial=False,
                need_feedback=False,
            )
        ]
    )

    events = await _drain(
        runner.run(
            _APP,
            _USER,
            root=agent,
            user_input=_user_input(turn_id="turn_2"),
            conversation_id=conv_id,
        )
    )

    event_types = [e.event_type for e in events]
    assert ExecutionEventType.AGENT_START in event_types, (
        f"AGENT_START must be emitted on the second turn; got {event_types}"
    )

    # Verify AGENT_START was persisted to the interaction history
    conv_info = await runner.conversation_manager.get_conversation(_APP, _USER, conv_id)
    history_types = [e.event_type for e in conv_info.events_history]
    assert ExecutionEventType.AGENT_START in history_types, (
        f"AGENT_START must be saved to history on the second turn; history has {history_types}"
    )


# ---------------------------------------------------------------------------
# Case 2: AGENT_CALL post-hook on the calling (parent) agent breaks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_call_post_hook_break_empties_stack(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """When the calling agent's post-hook breaks during AGENT_CALL processing,
    the stack must be emptied so the next turn resumes from the entrypoint."""

    sub_agent = Agent(
        name="SubAgent", description="desc", instruction="instr", llm_provider=llm_provider
    )
    sub_agent._execute = async_gen_from_list([])

    main_agent = Agent(
        name="MainAgent",
        description="desc",
        instruction="instr",
        llm_provider=llm_provider,
        sub_agents=[sub_agent],
        hooks=[_breaking_agent_hook(HookPhase.POST_AGENT_CALL)],
    )
    main_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.AGENT_CALL,
                author="MainAgent",
                content=[],
                action=AgentAction(
                    name="SubAgent",
                    parameters={"agent_name": "SubAgent"},
                    thought="delegate",
                ),
            )
        ]
    )

    conv_id = "test_agent_call_post_hook_break_stack"

    await _drain(
        runner.run(_APP, _USER, root=main_agent, user_input=_user_input(), conversation_id=conv_id)
    )

    assert await _stack_is_empty(runner, conv_id), (
        "Stack must be empty after AGENT_CALL post-hook breaks execution"
    )


@pytest.mark.asyncio
async def test_agent_call_post_hook_break_second_turn_resumes_entrypoint(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """After an AGENT_CALL post-hook break, the next turn must start from
    MainAgent (entrypoint), not SubAgent."""

    sub_agent = Agent(
        name="SubAgent", description="desc", instruction="instr", llm_provider=llm_provider
    )
    sub_agent._execute = async_gen_from_list([])

    main_agent = Agent(
        name="MainAgent",
        description="desc",
        instruction="instr",
        llm_provider=llm_provider,
        sub_agents=[sub_agent],
        hooks=[_breaking_agent_hook(HookPhase.POST_AGENT_CALL)],
    )
    main_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.AGENT_CALL,
                author="MainAgent",
                content=[],
                action=AgentAction(
                    name="SubAgent",
                    parameters={"agent_name": "SubAgent"},
                    thought="delegate",
                ),
            )
        ]
    )

    conv_id = "test_agent_call_post_hook_break_second_turn"

    # First turn: post-hook breaks during AGENT_CALL
    await _drain(
        runner.run(
            _APP,
            _USER,
            root=main_agent,
            user_input=_user_input(turn_id="turn_1"),
            conversation_id=conv_id,
        )
    )

    # Second turn: remove breaking hook, run normally
    main_agent.hooks = []
    main_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="MainAgent",
                content=[Part(text="done")],
                is_partial=False,
                need_feedback=False,
            )
        ]
    )

    events = await _drain(
        runner.run(
            _APP,
            _USER,
            root=main_agent,
            user_input=_user_input(turn_id="turn_2"),
            conversation_id=conv_id,
        )
    )

    agent_start_events = [e for e in events if e.event_type == ExecutionEventType.AGENT_START]
    assert agent_start_events, "AGENT_START must be emitted on second turn"
    # The first AGENT_START must be for MainAgent, not SubAgent
    assert agent_start_events[0].author == "MainAgent", (
        f"Second turn must start from MainAgent (entrypoint), got {agent_start_events[0].author}"
    )


# ---------------------------------------------------------------------------
# Case 3: Sub-agent's pre-hook breaks (should_continue=False via while condition)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sub_agent_pre_hook_break_empties_stack(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """When the called sub-agent's pre-hook breaks execution, the sub-agent is
    already on the stack. The stack must be emptied so the next turn does not
    resume from the sub-agent."""

    sub_agent = Agent(
        name="SubAgent",
        description="desc",
        instruction="instr",
        llm_provider=llm_provider,
        hooks=[_breaking_agent_hook(HookPhase.PRE_AGENT_CALL)],
    )
    sub_agent._execute = async_gen_from_list([])

    main_agent = Agent(
        name="MainAgent",
        description="desc",
        instruction="instr",
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
                    parameters={"agent_name": "SubAgent"},
                    thought="delegate",
                ),
            )
        ]
    )

    conv_id = "test_sub_agent_pre_hook_break_stack"

    await _drain(
        runner.run(_APP, _USER, root=main_agent, user_input=_user_input(), conversation_id=conv_id)
    )

    assert await _stack_is_empty(runner, conv_id), (
        "Stack must be empty after sub-agent pre-hook breaks execution"
    )


@pytest.mark.asyncio
async def test_sub_agent_pre_hook_break_second_turn_resumes_entrypoint(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """After a sub-agent pre-hook break, the next turn must resume from MainAgent
    (the entrypoint), not SubAgent."""

    sub_agent = Agent(
        name="SubAgent",
        description="desc",
        instruction="instr",
        llm_provider=llm_provider,
        hooks=[_breaking_agent_hook(HookPhase.PRE_AGENT_CALL)],
    )
    sub_agent._execute = async_gen_from_list([])

    main_agent = Agent(
        name="MainAgent",
        description="desc",
        instruction="instr",
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
                    parameters={"agent_name": "SubAgent"},
                    thought="delegate",
                ),
            )
        ]
    )

    conv_id = "test_sub_agent_pre_hook_break_second_turn"

    # First turn: sub-agent pre-hook breaks
    await _drain(
        runner.run(
            _APP,
            _USER,
            root=main_agent,
            user_input=_user_input(turn_id="turn_1"),
            conversation_id=conv_id,
        )
    )

    # Second turn: remove breaking hook, let MainAgent respond normally
    sub_agent.hooks = []
    main_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="MainAgent",
                content=[Part(text="done")],
                is_partial=False,
                need_feedback=False,
            )
        ]
    )

    events = await _drain(
        runner.run(
            _APP,
            _USER,
            root=main_agent,
            user_input=_user_input(turn_id="turn_2"),
            conversation_id=conv_id,
        )
    )

    agent_start_events = [e for e in events if e.event_type == ExecutionEventType.AGENT_START]
    assert agent_start_events, "AGENT_START must be emitted on second turn"
    assert agent_start_events[0].author == "MainAgent", (
        f"Second turn must resume from MainAgent, not SubAgent; got {agent_start_events[0].author}"
    )


# ---------------------------------------------------------------------------
# Case 5: AGENT_END post-hook on the sub-agent breaks (sub-agent not popped)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_end_post_hook_break_empties_stack(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """When the sub-agent's post-hook breaks during AGENT_END processing, the
    sub-agent is never popped from the stack. The stack must still be emptied."""

    sub_agent = Agent(
        name="SubAgent",
        description="desc",
        instruction="instr",
        llm_provider=llm_provider,
        hooks=[_breaking_agent_hook(HookPhase.POST_AGENT_CALL)],
    )
    sub_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.AGENT_END,
                author="SubAgent",
                content=[],
            )
        ]
    )

    main_agent = Agent(
        name="MainAgent",
        description="desc",
        instruction="instr",
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
                    parameters={"agent_name": "SubAgent"},
                    thought="delegate",
                ),
            )
        ]
    )

    conv_id = "test_agent_end_post_hook_break_stack"

    await _drain(
        runner.run(_APP, _USER, root=main_agent, user_input=_user_input(), conversation_id=conv_id)
    )

    assert await _stack_is_empty(runner, conv_id), (
        "Stack must be empty after sub-agent AGENT_END post-hook breaks execution"
    )


@pytest.mark.asyncio
async def test_agent_end_post_hook_break_second_turn_resumes_entrypoint(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """After a sub-agent AGENT_END post-hook break, the next turn must start from
    MainAgent (entrypoint), not SubAgent."""

    sub_agent = Agent(
        name="SubAgent",
        description="desc",
        instruction="instr",
        llm_provider=llm_provider,
        hooks=[_breaking_agent_hook(HookPhase.POST_AGENT_CALL)],
    )
    sub_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.AGENT_END,
                author="SubAgent",
                content=[],
            )
        ]
    )

    main_agent = Agent(
        name="MainAgent",
        description="desc",
        instruction="instr",
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
                    parameters={"agent_name": "SubAgent"},
                    thought="delegate",
                ),
            )
        ]
    )

    conv_id = "test_agent_end_post_hook_break_second_turn"

    # First turn: AGENT_END post-hook breaks
    await _drain(
        runner.run(
            _APP,
            _USER,
            root=main_agent,
            user_input=_user_input(turn_id="turn_1"),
            conversation_id=conv_id,
        )
    )

    # Second turn: remove breaking hook, MainAgent responds normally
    sub_agent.hooks = []
    main_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="MainAgent",
                content=[Part(text="done")],
                is_partial=False,
                need_feedback=False,
            )
        ]
    )

    events = await _drain(
        runner.run(
            _APP,
            _USER,
            root=main_agent,
            user_input=_user_input(turn_id="turn_2"),
            conversation_id=conv_id,
        )
    )

    agent_start_events = [e for e in events if e.event_type == ExecutionEventType.AGENT_START]
    assert agent_start_events, "AGENT_START must be emitted on second turn"
    assert agent_start_events[0].author == "MainAgent", (
        f"Second turn must resume from MainAgent; got {agent_start_events[0].author}"
    )
