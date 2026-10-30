"""Tests for the per-hook `silent` flag.

A silent hook must still run normally (side effects, HookResult updates all
apply), but its own HOOK_START/HOOK_UPDATE/HOOK_END/HOOK_ERROR events must not
be persisted to conversation history nor forwarded to interaction_history.
The live outward event stream (what Runner.run()/`_execute_tool` yields) is a
separate channel and is unaffected by `silent`.
"""

from collections.abc import AsyncGenerator

import pytest

from domyn.agents.agents import Agent
from domyn.agents.conversation_manager import ConversationInfo
from domyn.agents.core import BaseEvent, ExecutionEventType, InteractionContext, Part, ToolAction
from domyn.agents.hooks import FunctionHook, HookPhase, HookResult, HookType
from domyn.agents.llm import BaseLLMProvider, LLMMessage
from domyn.agents.runner import Runner, RunnerState
from domyn.agents.tools import FunctionTool, ToolResult

_APP = "test_app"
_USER = "test_user"
_CONV = "test_convo"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _make_state(agent: Agent) -> RunnerState:
    return RunnerState(
        application_name=_APP,
        user_id=_USER,
        conversation_id=_CONV,
        conversation_info=ConversationInfo(
            application_name=_APP,
            user_id=_USER,
            conversation_id=_CONV,
            events_history=[],
        ),
        current_agent=agent,
        available_agents=[agent],
        entrypoint_agent=agent,
    )


def _make_context() -> InteractionContext:
    return InteractionContext(
        conversation_id=_CONV,
        user_id=_USER,
        agent_input=BaseEvent(
            event_type=ExecutionEventType.USER_INPUT,
            content=[Part(text="test")],
            author="user",
        ),
        state={},
        run_config={},
    )


def _make_tool_event(tool_name: str, parameters: dict) -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.TOOL_START,
        content=[],
        author="test_agent",
        action=ToolAction(name=tool_name, parameters=parameters, thought="executing tool"),
    )


async def _collect(async_gen) -> list:
    return [item async for item in async_gen]


def _hook_names(events: list[BaseEvent]) -> list[str]:
    return [e.metadata.get("hook_name") for e in events if "hook_name" in e.metadata]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> Runner:
    return Runner()


@pytest.fixture
def mock_llm() -> BaseLLMProvider:
    class MockLLM(BaseLLMProvider):
        type: str = "mock"
        name: str = "mock-llm"
        model_name: str = "mock-model"
        supported_structured_outputs: None = None

        async def generate(self, messages: LLMMessage, **kwargs):
            pass

        async def generate_streaming(self, messages: LLMMessage, **kwargs):
            yield

    return MockLLM()


@pytest.fixture
def agent(mock_llm) -> Agent:
    return Agent(
        name="test_agent",
        description="test agent",
        instruction="test instruction",
        llm_provider=mock_llm,
    )


# ===========================================================================
# Tool pre-hooks (updater path — direct `hook` loop variable)
# ===========================================================================


class TestSilentToolPreHookUpdater:
    @pytest.mark.asyncio
    async def test_silent_updater_pre_hook_runs_but_events_not_persisted_or_forwarded(
        self, runner: Runner, agent: Agent
    ) -> None:
        captured = {}

        def modify_args(tool: FunctionTool, tool_args: dict) -> dict:
            tool_args["a"] = tool_args["a"] + 100
            return {"new_tool_args": tool_args}

        def add(a: int, b: int) -> int:
            captured["a"] = a
            captured["b"] = b
            return a + b

        silent_hook = FunctionHook(
            trigger=HookPhase.PRE_TOOL_CALL,
            type=HookType.UPDATER,
            name="silent_add_100",
            fun=modify_args,
            silent=True,
        )
        tool = FunctionTool(func=add, hooks=[silent_hook])
        state = _make_state(agent)
        context = _make_context()
        event = _make_tool_event("add", {"a": 1, "b": 2})

        await runner.conversation_manager.create_conversation(_APP, _USER, _CONV)

        events = await _collect(runner._execute_tool(tool, state, context, event))

        # The hook still ran: tool_args were modified as expected.
        assert captured["a"] == 101
        assert captured["b"] == 2

        # It IS still present in the live outward stream (separate channel).
        assert "silent_add_100" in _hook_names(events)

        # It must NOT be persisted to conversation history...
        conv_info = await runner.conversation_manager.get_conversation(_APP, _USER, _CONV)
        assert "silent_add_100" not in _hook_names(conv_info.events_history)

        # ...nor forwarded to interaction_history.
        assert "silent_add_100" not in _hook_names(context.interaction_history)


# ===========================================================================
# Tool pre-hooks (listener path — merged via `_drain_listener_queue`)
# ===========================================================================


class TestSilentToolPreHookListener:
    @pytest.mark.asyncio
    async def test_silent_listener_pre_hook_runs_but_events_not_persisted_or_forwarded(
        self, runner: Runner, agent: Agent
    ) -> None:
        observed = {}

        def observe(tool: FunctionTool, tool_args: dict) -> dict:
            observed["a"] = tool_args.get("a")
            return {"emitted_content": "observed"}

        def add(a: int, b: int) -> int:
            return a + b

        silent_hook = FunctionHook(
            trigger=HookPhase.PRE_TOOL_CALL,
            type=HookType.LISTENER,
            name="silent_observer",
            fun=observe,
            silent=True,
        )
        normal_hook = FunctionHook(
            trigger=HookPhase.PRE_TOOL_CALL,
            type=HookType.LISTENER,
            name="normal_observer",
            fun=lambda tool, tool_args: None,
        )
        tool = FunctionTool(func=add, hooks=[silent_hook, normal_hook])
        state = _make_state(agent)
        context = _make_context()
        event = _make_tool_event("add", {"a": 5, "b": 10})

        await runner.conversation_manager.create_conversation(_APP, _USER, _CONV)

        events = await _collect(runner._execute_tool(tool, state, context, event))

        # The silent hook still ran.
        assert observed["a"] == 5

        # Both hooks' events are still present in the live outward stream.
        assert "silent_observer" in _hook_names(events)
        assert "normal_observer" in _hook_names(events)

        conv_info = await runner.conversation_manager.get_conversation(_APP, _USER, _CONV)
        persisted_names = _hook_names(conv_info.events_history)
        forwarded_names = _hook_names(context.interaction_history)

        # The silent hook's events are absent from both persisted/forwarded history...
        assert "silent_observer" not in persisted_names
        assert "silent_observer" not in forwarded_names

        # ...while the normal hook's events still appear in both.
        assert "normal_observer" in persisted_names
        assert "normal_observer" in forwarded_names


# ===========================================================================
# Agent pre-hooks (via full Runner.run())
# ===========================================================================


class TestSilentAgentPreHook:
    @pytest.mark.asyncio
    async def test_silent_agent_pre_hook_runs_but_events_not_persisted_or_forwarded(
        self, runner: Runner, agent: Agent
    ) -> None:
        called = {"silent": False, "normal": False}
        captured_context: list[InteractionContext] = []

        async def silent_fn(context: InteractionContext, agent: Agent) -> None:
            called["silent"] = True
            captured_context.append(context)

        async def normal_fn(context: InteractionContext, agent: Agent) -> None:
            called["normal"] = True

        silent_hook = FunctionHook(
            name="silent_pre_agent",
            fun=silent_fn,
            trigger=HookPhase.PRE_AGENT_CALL,
            type=HookType.LISTENER,
            silent=True,
        )
        normal_hook = FunctionHook(
            name="normal_pre_agent",
            fun=normal_fn,
            trigger=HookPhase.PRE_AGENT_CALL,
            type=HookType.LISTENER,
        )
        agent.hooks = [silent_hook, normal_hook]
        agent._execute = async_gen_from_list(
            [
                BaseEvent(
                    event_type=ExecutionEventType.RESPONSE,
                    author=agent.name,
                    content=[Part(text="done")],
                    is_partial=False,
                    need_feedback=False,
                )
            ]
        )

        user_input = BaseEvent(
            event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text="hi")]
        )

        events = await _collect(
            runner.run(
                _APP,
                _USER,
                root=agent,
                user_input=user_input,
                conversation_id=_CONV,
            )
        )

        # Both hooks ran.
        assert called["silent"] is True
        assert called["normal"] is True
        assert len(captured_context) == 1

        # Both hooks' events are present in the live outward stream.
        assert "silent_pre_agent" in _hook_names(events)
        assert "normal_pre_agent" in _hook_names(events)

        conv_info = await runner.conversation_manager.get_conversation(_APP, _USER, _CONV)
        persisted_names = _hook_names(conv_info.events_history)
        forwarded_names = _hook_names(captured_context[0].interaction_history)

        assert "silent_pre_agent" not in persisted_names
        assert "silent_pre_agent" not in forwarded_names
        assert "normal_pre_agent" in persisted_names
        assert "normal_pre_agent" in forwarded_names


# ===========================================================================
# on_agent_event hooks (via full Runner.run())
# ===========================================================================


class TestSilentOnEventHook:
    @pytest.mark.asyncio
    async def test_silent_on_event_hook_runs_but_events_not_persisted_or_forwarded(
        self, runner: Runner, agent: Agent
    ) -> None:
        observed: list[str] = []

        async def observe_response(event: BaseEvent, context, **_) -> HookResult:
            observed.append(event.event_type)
            return HookResult(success=True, emitted_content=None, name="silent_on_event")

        silent_hook = FunctionHook(
            fun=observe_response,
            name="silent_on_event",
            trigger=HookPhase.ON_AGENT_EVENT,
            type=HookType.UPDATER,
            on_event_triggers=[ExecutionEventType.RESPONSE],
            silent=True,
        )
        agent.hooks = [silent_hook]
        agent._execute = async_gen_from_list(
            [
                BaseEvent(
                    event_type=ExecutionEventType.RESPONSE,
                    author=agent.name,
                    content=[Part(text="done")],
                    is_partial=False,
                    need_feedback=False,
                )
            ]
        )

        user_input = BaseEvent(
            event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text="hi")]
        )

        events = await _collect(
            runner.run(
                _APP,
                _USER,
                root=agent,
                user_input=user_input,
                conversation_id=_CONV,
            )
        )

        # The hook ran.
        assert observed == [ExecutionEventType.RESPONSE]

        # Still present in the live outward stream.
        assert "silent_on_event" in _hook_names(events)

        conv_info = await runner.conversation_manager.get_conversation(_APP, _USER, _CONV)
        assert "silent_on_event" not in _hook_names(conv_info.events_history)
