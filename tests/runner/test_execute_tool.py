"""Tests for Runner._execute_tool covering pre-hooks, tool execution, and post-hooks."""

import asyncio
import json
from collections.abc import AsyncGenerator, Callable
from copy import deepcopy
from typing import Any

import httpx
import pytest

from domyn.agents.agents import Agent
from domyn.agents.conversation_manager import ConversationInfo
from domyn.agents.core import BaseEvent, ExecutionEventType, InteractionContext, Part, ToolAction
from domyn.agents.hooks import FunctionHook
from domyn.agents.hooks.base import HookPhase, HookRestAPIData, HookResult, HookType, InputFields
from domyn.agents.runner import Runner, RunnerState
from domyn.agents.skills import LoadedSkills, SkillContent
from domyn.agents.tools import FunctionTool, StreamableFunctionTool, ToolResult
from domyn.agents.tools.read_tool_result import ToolResults

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(
    func: Callable, pre_hooks: list | None = None, post_hooks: list | None = None
) -> FunctionTool:
    """Create a FunctionTool with optional hooks attached."""
    hooks = []
    if pre_hooks:
        hooks.extend(pre_hooks)
    if post_hooks:
        hooks.extend(post_hooks)
    return FunctionTool(func=func, hooks=hooks)


def _make_streamable_tool(
    func: Callable, pre_hooks: list | None = None, post_hooks: list | None = None
) -> StreamableFunctionTool:
    """Create a StreamableFunctionTool with optional hooks attached."""
    hooks = []
    if pre_hooks:
        hooks.extend(pre_hooks)
    if post_hooks:
        hooks.extend(post_hooks)
    return StreamableFunctionTool(func=func, hooks=hooks)


def _make_state(agent: Agent) -> RunnerState:
    return RunnerState(
        application_name="test_app",
        user_id="test_user",
        conversation_id="test_convo",
        conversation_info=ConversationInfo(
            application_name="test_app",
            user_id="test_user",
            conversation_id="test_convo",
            events_history=[],
        ),
        current_agent=agent,
        available_agents=[agent],
        entrypoint_agent=agent,
    )


def _make_context(state: dict | None = None, run_config: dict | None = None) -> InteractionContext:
    return InteractionContext(
        conversation_id="test_convo",
        user_id="test_user",
        agent_input=BaseEvent(
            event_type=ExecutionEventType.USER_INPUT,
            content=[Part(text="test")],
            author="user",
        ),
        state=state or {},
        run_config=run_config or {},
    )


def _make_event(tool_name: str, parameters: dict) -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.TOOL_START,
        content=[],
        author="test_agent",
        action=ToolAction(name=tool_name, parameters=parameters, thought="executing tool"),
    )


async def _collect_events(async_gen) -> list[BaseEvent]:
    events = []
    async for ev in async_gen:
        events.append(ev)
    return events


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> Runner:
    return Runner()


@pytest.fixture
def mock_llm():
    from domyn.agents.llm import BaseLLMProvider, LLMMessage

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
# 1. Basic tool execution (no hooks)
# ===========================================================================


class TestExecuteToolBasic:
    """Test _execute_tool with no hooks, just the tool.execute path."""

    @pytest.mark.asyncio
    async def test_sync_tool_returns_tool_end(self, runner: Runner, agent: Agent) -> None:
        """A simple sync function tool yields a TOOL_END event with the result."""

        def add(a: int, b: int) -> int:
            return a + b

        tool = _make_tool(add)
        state = _make_state(agent)
        context = _make_context()
        event = _make_event("add", {"a": 2, "b": 3})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))

        tool_end: list[BaseEvent] = [
            e for e in events if e.event_type == ExecutionEventType.TOOL_END
        ]
        assert len(tool_end) == 1
        assert tool_end[0].action.observation == 5

    @pytest.mark.asyncio
    async def test_tool_args_include_run_config_and_state(
        self, runner: Runner, agent: Agent
    ) -> None:
        """Tool receives run_config and state from context plus action parameters."""
        captured_args: dict[str, Any] = {}

        def capture_tool(x: str, run_config: dict | None = None, state: dict | None = None) -> str:
            captured_args["x"] = x
            captured_args["run_config"] = run_config
            captured_args["state"] = state
            return f"got {x}"

        tool = _make_tool(capture_tool)
        state = _make_state(agent)
        context = _make_context(state={"counter": 1}, run_config={"key": "val"})
        event = _make_event("capture_tool", {"x": "hello"})

        await _collect_events(runner._execute_tool(tool, state, context, event))

        assert captured_args["x"] == "hello"
        assert captured_args["run_config"] == {"key": "val"}
        assert captured_args["state"] == {"counter": 1}

    @pytest.mark.asyncio
    async def test_tool_returning_tool_result_directly(self, runner: Runner, agent: Agent) -> None:
        """Tool can return a ToolResult directly."""

        def my_tool(x: int) -> ToolResult:
            return ToolResult(success=True, data=x * 10, metadata={"src": "direct"})

        tool = _make_tool(my_tool)
        state = _make_state(agent)
        context = _make_context()
        event = _make_event("my_tool", {"x": 4})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))
        tool_end: list[BaseEvent] = [
            e for e in events if e.event_type == ExecutionEventType.TOOL_END
        ]
        assert tool_end[0].action.observation == 40
        assert tool_end[0].metadata == {"src": "direct"}

    @pytest.mark.asyncio
    async def test_tool_state_update_propagates_to_context(
        self, runner: Runner, agent: Agent
    ) -> None:
        """When tool returns state in ToolResult, context.state is updated in place."""

        def stateful(val: str) -> ToolResult:
            return ToolResult(success=True, data="ok", state={"new_key": val})

        tool = _make_tool(stateful)
        state = _make_state(agent)
        context = _make_context(state={"old_key": "old"})
        event = _make_event("stateful", {"val": "new_value"})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))
        assert events[0].event_type == ExecutionEventType.TOOL_END
        assert events[0].action.observation == "ok"
        # context.state should be replaced entirely
        assert context.state == {"new_key": "new_value"}

    @pytest.mark.asyncio
    async def test_tool_state_update_overwrites_context(self, runner: Runner, agent: Agent) -> None:
        """When tool returns state in ToolResult, context.state is updated"""

        def stateful(val: str, state: dict) -> ToolResult:
            state.update({"new_key": val})
            return ToolResult(success=True, data="ok", state=state)

        tool = _make_tool(stateful)
        state = _make_state(agent)
        context = _make_context(state={"old_key": "old"})
        event = _make_event("stateful", {"val": "new_value"})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))
        assert events[0].event_type == ExecutionEventType.TOOL_END
        assert events[0].action.observation == "ok"
        # context.state should be replaced entirely
        assert context.state == {"new_key": "new_value", "old_key": "old"}

    @pytest.mark.asyncio
    async def test_tool_state_update_without_returning_state_toolresult(
        self, runner: Runner, agent: Agent
    ) -> None:
        """When tool returns state in ToolResult, context.state is updated"""

        def stateful(val: str, state: dict) -> ToolResult:
            state.update({"new_key": val})
            return ToolResult(success=True, data="ok")

        tool = _make_tool(stateful)
        state = _make_state(agent)
        context = _make_context(state={"old_key": "old"})
        event = _make_event("stateful", {"val": "new_value"})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))
        assert events[0].event_type == ExecutionEventType.TOOL_END
        assert events[0].action.observation == "ok"
        # context.state should be replaced entirely
        assert context.state == {"old_key": "old"}

    @pytest.mark.asyncio
    async def test_tool_state_update_inplace_context(self, runner: Runner, agent: Agent) -> None:
        """When tool updates the state in place and returns it in a dict"""

        def stateful(val: str, state: dict) -> dict:
            state.update({"old_key": "overwritten"})
            return {"result": "ok", "state": state}

        tool = _make_tool(stateful)
        state = _make_state(agent)
        context = _make_context(state={"old_key": "old"})
        event = _make_event("stateful", {"val": "new_value"})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))
        assert events[0].event_type == ExecutionEventType.TOOL_END
        assert events[0].action.observation == "ok"
        assert context.state == {"old_key": "overwritten"}

    @pytest.mark.asyncio
    async def test_tool_state_update_without_returning_state(
        self, runner: Runner, agent: Agent
    ) -> None:
        """When tool updates the state in place but doesn't return it"""

        def stateful(val: str, state: dict) -> dict:
            state.update({"old_key": "overwritten"})
            return {"result": "ok"}

        tool = _make_tool(stateful)
        state = _make_state(agent)
        context = _make_context(state={"old_key": "old"})
        event = _make_event("stateful", {"val": "new_value"})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))
        assert events[0].event_type == ExecutionEventType.TOOL_END
        assert events[0].action.observation == "ok"
        assert context.state == {"old_key": "old"}

    @pytest.mark.asyncio
    async def test_tool_exception_yields_tool_error(self, runner: Runner, agent: Agent) -> None:
        """If tool.execute raises, a TOOL_ERROR event is yielded."""

        def failing_tool(x: int, state: dict) -> str:
            state.update({"counter": state.get("counter", 0) + 1})
            raise ValueError("boom")

        tool = _make_tool(failing_tool)
        state = _make_state(agent)
        context = _make_context(state={"counter": 1})
        event = _make_event("failing_tool", {"x": 1})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))
        errors = [e for e in events if e.event_type == ExecutionEventType.TOOL_ERROR]
        assert len(errors) == 1
        assert context.state == {"counter": 1}  # state should not be modified on error


# ===========================================================================
# 2. Pre-hooks
# ===========================================================================


class TestExecuteToolPreHooks:
    """Test that pre-hooks run before tool execution and can modify tool_args."""

    @pytest.mark.asyncio
    async def test_updater_pre_hook_modifies_tool_args(self, runner: Runner, agent: Agent) -> None:
        """An updater pre-hook can change tool_args before execution."""
        captured = {}

        def my_pre_hook(tool: FunctionTool, tool_args: dict) -> dict:
            tool_args["a"] = tool_args["a"] + 100
            return {"new_tool_args": tool_args}

        def add(a: int, b: int) -> int:
            captured["a"] = a
            captured["b"] = b
            return a + b

        pre_hook = FunctionHook(
            trigger=HookPhase.PRE_TOOL_CALL,
            type=HookType.UPDATER,
            name="add_100",
            fun=my_pre_hook,
        )
        tool = _make_tool(add, pre_hooks=[pre_hook])
        state = _make_state(agent)
        context = _make_context()
        event = _make_event("add", {"a": 1, "b": 2})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))
        assert len(events) == 3
        assert events[0].event_type == ExecutionEventType.HOOK_START
        assert events[1].event_type == ExecutionEventType.HOOK_END
        assert events[2].event_type == ExecutionEventType.TOOL_END
        # pre-hook added 100 to a
        assert captured["a"] == 101
        assert captured["b"] == 2
        tool_end = [e for e in events if e.event_type == ExecutionEventType.TOOL_END]
        assert tool_end[0].action.observation == 103
        assert context.state == {}

    @pytest.mark.asyncio
    async def test_updater_pre_hook_does_not_return_tool_args(
        self, runner: Runner, agent: Agent
    ) -> None:
        """An updater pre-hook must return tool args."""
        captured = {}

        def my_pre_hook(tool: FunctionTool, tool_args: dict) -> dict:
            tool_args["a"] = tool_args["a"] + 100
            return {}

        def add(a: int, b: int) -> int:
            captured["a"] = a
            captured["b"] = b
            return a + b

        pre_hook = FunctionHook(
            trigger=HookPhase.PRE_TOOL_CALL,
            type=HookType.UPDATER,
            name="add_100",
            fun=my_pre_hook,
        )
        tool = _make_tool(add, pre_hooks=[pre_hook])
        state = _make_state(agent)
        context = _make_context()
        event = _make_event("add", {"a": 1, "b": 2})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))
        assert len(events) == 3
        assert events[0].event_type == ExecutionEventType.HOOK_START
        assert events[1].event_type == ExecutionEventType.HOOK_ERROR
        assert events[2].event_type == ExecutionEventType.TOOL_END

    @pytest.mark.asyncio
    async def test_chained_updater_pre_hooks(self, runner: Runner, agent: Agent) -> None:
        """Multiple updater pre-hooks run sequentially, each seeing the modified args."""

        def hook_a(tool: FunctionTool, tool_args: dict) -> dict:
            tool_args["x"] = tool_args["x"] + "_a"
            return {"new_tool_args": tool_args}

        def hook_b(tool: FunctionTool, tool_args: dict) -> dict:
            tool_args["x"] = tool_args["x"] + "_b"
            return {"new_tool_args": tool_args}

        captured = {}

        def echo(x: str) -> str:
            captured["x"] = x
            return x

        hooks = [
            FunctionHook(
                trigger=HookPhase.PRE_TOOL_CALL,
                type=HookType.UPDATER,
                name="hook_a",
                fun=hook_a,
            ),
            FunctionHook(
                trigger=HookPhase.PRE_TOOL_CALL,
                type=HookType.UPDATER,
                name="hook_b",
                fun=hook_b,
            ),
        ]
        tool = _make_tool(echo, pre_hooks=hooks)
        state = _make_state(agent)
        context = _make_context()
        event = _make_event("echo", {"x": "start"})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))
        assert len(events) == 5
        assert captured["x"] == "start_a_b"

    @pytest.mark.asyncio
    async def test_updater_pre_hook_modifies_state(self, runner: Runner, agent: Agent) -> None:
        """When pre-hook returns new_tool_args with 'state', context.state is updated."""

        def state_hook(tool: FunctionTool, tool_args: dict) -> dict:
            tool_args["state"] = {"injected": True}
            return {"new_tool_args": tool_args}

        def noop(x: int) -> str:
            return "ok"

        pre_hook = FunctionHook(
            trigger=HookPhase.PRE_TOOL_CALL,
            type=HookType.UPDATER,
            name="state_hook",
            fun=state_hook,
        )
        tool = _make_tool(noop, pre_hooks=[pre_hook])
        state = _make_state(agent)
        context = _make_context(state={"original": True})
        event = _make_event("noop", {"x": 1})

        await _collect_events(runner._execute_tool(tool, state, context, event))

        assert context.state == {"injected": True}

    @pytest.mark.asyncio
    async def test_updater_pre_hook_modifies_state_in_place(
        self, runner: Runner, agent: Agent
    ) -> None:
        """When pre-hook returns new_tool_args with 'state', context.state is updated."""

        def state_hook(tool: FunctionTool, tool_args: dict) -> dict:
            state = tool_args["state"]
            state["injected"] = True
            return {"new_tool_args": tool_args}

        def noop(x: int) -> str:
            return "ok"

        pre_hook = FunctionHook(
            trigger=HookPhase.PRE_TOOL_CALL,
            type=HookType.UPDATER,
            name="state_hook",
            fun=state_hook,
        )
        tool = _make_tool(noop, pre_hooks=[pre_hook])
        state = _make_state(agent)
        context = _make_context(state={"original": True})
        event = _make_event("noop", {"x": 1})

        await _collect_events(runner._execute_tool(tool, state, context, event))

        assert context.state == {"original": True, "injected": True}

    @pytest.mark.asyncio
    async def test_updater_pre_hook_modifies_state_in_place_without_returning(
        self, runner: Runner, agent: Agent
    ) -> None:
        """When pre-hook modifies state in place without returning state in new_tool_args"""

        def state_hook(tool: FunctionTool, tool_args: dict) -> dict:
            state = tool_args["state"]
            state["injected"] = True
            del tool_args["state"]
            return {"new_tool_args": tool_args}

        def noop(x: int) -> str:
            return "ok"

        pre_hook = FunctionHook(
            trigger=HookPhase.PRE_TOOL_CALL,
            type=HookType.UPDATER,
            name="state_hook",
            fun=state_hook,
        )
        tool = _make_tool(noop, pre_hooks=[pre_hook])
        state = _make_state(agent)
        context = _make_context(state={"original": True})
        event = _make_event("noop", {"x": 1})

        await _collect_events(runner._execute_tool(tool, state, context, event))

        assert context.state == {"original": True}

    @pytest.mark.asyncio
    async def test_updater_pre_hook_set_state_none(self, runner: Runner, agent: Agent) -> None:
        """When pre-hook sets state to None, context.state remains unchanged."""

        def state_hook(tool: FunctionTool, tool_args: dict) -> dict:
            tool_args["state"] = None
            return {"new_tool_args": tool_args}

        def noop(x: int) -> str:
            return "ok"

        pre_hook = FunctionHook(
            trigger=HookPhase.PRE_TOOL_CALL,
            type=HookType.UPDATER,
            name="state_hook",
            fun=state_hook,
        )
        tool = _make_tool(noop, pre_hooks=[pre_hook])
        state = _make_state(agent)
        context = _make_context(state={"original": True})
        event = _make_event("noop", {"x": 1})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))
        assert events[0].event_type == ExecutionEventType.HOOK_START
        assert events[1].event_type == ExecutionEventType.HOOK_ERROR
        assert events[2].event_type == ExecutionEventType.TOOL_END
        assert context.state == {"original": True}

    @pytest.mark.asyncio
    async def test_updater_pre_hook_set_state_as_list(self, runner: Runner, agent: Agent) -> None:
        """When pre-hook sets state to a list, context.state remains unchanged."""

        def state_hook(tool: FunctionTool, tool_args: dict) -> dict:
            tool_args["state"] = ["This", "is", "a", "list"]
            return {"new_tool_args": tool_args}

        def noop(x: int) -> str:
            return "ok"

        pre_hook = FunctionHook(
            trigger=HookPhase.PRE_TOOL_CALL,
            type=HookType.UPDATER,
            name="state_hook",
            fun=state_hook,
        )
        tool = _make_tool(noop, pre_hooks=[pre_hook])
        state = _make_state(agent)
        context = _make_context(state={"original": True})
        event = _make_event("noop", {"x": 1})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))
        assert events[0].event_type == ExecutionEventType.HOOK_START
        assert events[1].event_type == ExecutionEventType.HOOK_ERROR
        assert events[2].event_type == ExecutionEventType.TOOL_END
        assert context.state == {"original": True}

    @pytest.mark.asyncio
    async def test_updater_pre_hook_modifies_state_in_context_toolprehookresult(
        self, runner: Runner, agent: Agent
    ) -> None:
        """When pre-hook returns new_tool_args with 'state', context.state is updated."""

        def state_hook(tool: FunctionTool, tool_args: dict) -> HookResult:
            tool_args["state"] = {"injected": True}
            return HookResult(
                success=True, error=None, emitted_content=None, new_tool_args=tool_args
            )

        def noop(x: int) -> str:
            return "ok"

        pre_hook = FunctionHook(
            trigger=HookPhase.PRE_TOOL_CALL,
            type=HookType.UPDATER,
            name="state_hook",
            fun=state_hook,
        )
        tool = _make_tool(noop, pre_hooks=[pre_hook])
        state = _make_state(agent)
        context = _make_context(state={"original": True})
        event = _make_event("noop", {"x": 1})

        await _collect_events(runner._execute_tool(tool, state, context, event))

        assert context.state == {"injected": True}
        assert "original" not in context.state

    @pytest.mark.asyncio
    async def test_listener_pre_hook_does_not_modify_args(
        self, runner: Runner, agent: Agent
    ) -> None:
        """Listener pre-hooks observe but cannot modify tool args."""
        observed = {}

        def listener(tool: FunctionTool, tool_args: dict) -> dict:
            observed["a"] = tool_args.get("a")
            # returning None => listener
            return {"emitted_content": "test_content"}

        captured = {}

        def add(a: int, b: int) -> int:
            captured["a"] = a
            return a + b

        pre_hook = FunctionHook(
            trigger=HookPhase.PRE_TOOL_CALL, name="observe", fun=listener, type=HookType.LISTENER
        )
        tool = _make_tool(add, pre_hooks=[pre_hook])
        state = _make_state(agent)
        context = _make_context()
        event = _make_event("add", {"a": 5, "b": 10})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))

        # listener saw the original value
        assert observed["a"] == 5
        # tool received unmodified args
        assert captured["a"] == 5
        assert events[0].event_type == ExecutionEventType.HOOK_START
        assert events[1].event_type == ExecutionEventType.HOOK_END
        assert events[1].content[0].text == "test_content"
        assert events[2].event_type == ExecutionEventType.TOOL_END

    @pytest.mark.asyncio
    async def test_listener_pre_hook_modify_args(self, runner: Runner, agent: Agent) -> None:
        """Listener pre-hooks observe but cannot modify tool args."""

        def listener(tool: FunctionTool, tool_args: dict) -> dict:
            return {"new_tool_args": tool_args}

        def add(a: int, b: int) -> int:
            return a + b

        pre_hook = FunctionHook(
            trigger=HookPhase.PRE_TOOL_CALL,
            type=HookType.LISTENER,
            name="observe",
            fun=listener,
        )
        tool = _make_tool(add, pre_hooks=[pre_hook])
        state = _make_state(agent)
        context = _make_context()
        event = _make_event("add", {"a": 5, "b": 10})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))

        assert events[0].event_type == ExecutionEventType.HOOK_START
        assert events[1].event_type == ExecutionEventType.HOOK_ERROR
        assert events[2].event_type == ExecutionEventType.TOOL_END

    @pytest.mark.asyncio
    async def test_pre_hook_events_emitted(self, runner: Runner, agent: Agent) -> None:
        """Pre-hooks emit HOOK_START and HOOK_END events."""

        def my_hook(tool: FunctionTool, tool_args: dict) -> dict:
            return {"new_tool_args": tool_args}

        pre_hook = FunctionHook(
            trigger=HookPhase.PRE_TOOL_CALL,
            type=HookType.UPDATER,
            name="logging_hook",
            fun=my_hook,
        )

        def noop(x: int) -> str:
            return "ok"

        tool = _make_tool(noop, pre_hooks=[pre_hook])
        state = _make_state(agent)
        context = _make_context()
        event = _make_event("noop", {"x": 1})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))

        hook_starts = [e for e in events if e.event_type == ExecutionEventType.HOOK_START]
        hook_ends = [e for e in events if e.event_type == ExecutionEventType.HOOK_END]
        assert len(hook_starts) >= 1
        assert len(hook_ends) >= 1
        assert hook_starts[0].metadata["hook_name"] == "logging_hook"


class TestARestPreHook:
    """A REST pre-hook sends the call's ``tool_args`` in its request body. The runner
    used to put live objects there, ``loaded_skills`` and, for every agent on a
    tool-use planner, ``tool_results``: the body failed to serialize, the request was
    never sent, and the tool ran as if the hook did not exist, a ``HOOK_ERROR`` the
    only trace."""

    @pytest.mark.asyncio
    async def test_is_sent_the_call_and_the_tool_still_gets_what_only_it_needs(
        self, runner: Runner, agent: Agent
    ) -> None:
        sent: list[dict[str, Any]] = []

        def hook_server(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            sent.append(body)
            # JSON is all a REST hook can send back, so the arguments it returns
            # carry nothing the runner put there that JSON cannot.
            tool_args = body["tool_args"] | {"query": body["tool_args"]["query"].upper()}
            return httpx.Response(200, json={"new_tool_args": tool_args})

        received: dict[str, Any] = {}

        def lookup(query: str, tool_results: Any = None, loaded_skills: Any = None) -> str:
            received.update(query=query, tool_results=tool_results, loaded_skills=loaded_skills)
            return "found"

        hook = FunctionHook(
            trigger=HookPhase.PRE_TOOL_CALL,
            type=HookType.UPDATER,
            name="rest_pre_hook",
            fun=None,
            rest_api=HookRestAPIData(url="http://hooks.example/pre", method="POST"),
            input_fields=[InputFields.TOOL_NAME, InputFields.TOOL_ARGS],
            httpx_client=httpx.AsyncClient(transport=httpx.MockTransport(hook_server)),
        )
        state = _make_state(agent)
        skills = LoadedSkills(bodies={"style": SkillContent(name="style", body="Be terse.")})
        state.skills[agent.name] = skills

        events = await _collect_events(
            runner._execute_tool(
                _make_tool(lookup, pre_hooks=[hook]),
                state,
                _make_context(),
                _make_event("lookup", {"query": "jira"}),
            )
        )

        assert [e.event_type for e in events] == [
            ExecutionEventType.HOOK_START,
            ExecutionEventType.HOOK_END,
            ExecutionEventType.TOOL_END,
        ]
        assert len(sent) == 1
        assert sent[0]["tool_args"]["query"] == "jira"
        assert not {"tool_results", "loaded_skills"} & sent[0]["tool_args"].keys()
        assert received["query"] == "JIRA"
        assert isinstance(received["tool_results"], ToolResults)
        assert received["loaded_skills"] == skills


# ===========================================================================
# 3. Post-hooks
# ===========================================================================


class TestExecuteToolPostHooks:
    @pytest.mark.asyncio
    async def test_updater_post_hook_modifies_result(self, runner: Runner, agent: Agent) -> None:
        """An updater post-hook can change the tool_result."""

        def my_post_hook(tool: FunctionTool, result: ToolResult) -> dict:
            return {
                "new_result": ToolResult(
                    success=True,
                    data=f"modified:{result.data}",
                    state=result.state,
                )
            }

        def greet(name: str) -> str:
            return f"hello {name}"

        post_hook = FunctionHook(
            trigger=HookPhase.POST_TOOL_CALL,
            type=HookType.UPDATER,
            name="modify_result",
            fun=my_post_hook,
        )
        tool = _make_tool(greet, post_hooks=[post_hook])
        state = _make_state(agent)
        context = _make_context()
        event = _make_event("greet", {"name": "world"})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))

        assert events[0].event_type == ExecutionEventType.HOOK_START
        assert events[1].event_type == ExecutionEventType.HOOK_END
        assert events[2].event_type == ExecutionEventType.TOOL_END
        assert context.state == {}
        assert events[2].action.observation == "modified:hello world"

    @pytest.mark.asyncio
    async def test_chained_updater_post_hooks(self, runner: Runner, agent: Agent) -> None:
        """Multiple post-hooks run sequentially, each seeing the modified result."""

        def hook_a(tool: FunctionTool, result: ToolResult) -> dict:
            return {"new_result": ToolResult(success=True, data=f"{result.data}_A")}

        def hook_b(tool: FunctionTool, result: ToolResult) -> dict:
            return {"new_result": ToolResult(success=True, data=f"{result.data}_B")}

        def echo(x: str) -> str:
            return x

        hooks = [
            FunctionHook(
                trigger=HookPhase.POST_TOOL_CALL,
                type=HookType.UPDATER,
                name="hook_a",
                fun=hook_a,
            ),
            FunctionHook(
                trigger=HookPhase.POST_TOOL_CALL,
                type=HookType.UPDATER,
                name="hook_b",
                fun=hook_b,
            ),
        ]
        tool = _make_tool(echo, post_hooks=hooks)
        state = _make_state(agent)
        context = _make_context()
        event = _make_event("echo", {"x": "start"})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))
        assert len(events) == 5
        assert events[-1].action.observation == "start_A_B"

    @pytest.mark.asyncio
    async def test_updater_post_hook_replaces_state(self, runner: Runner, agent: Agent) -> None:
        """When post-hook returns new_result with state"""

        def state_post_hook(tool: FunctionTool, result: ToolResult) -> dict:
            return {
                "new_result": ToolResult(
                    success=True,
                    data=result.data,
                    state={"post_hook_state": True},
                )
            }

        def noop(x: int) -> str:
            return "ok"

        post_hook = FunctionHook(
            trigger=HookPhase.POST_TOOL_CALL,
            type=HookType.UPDATER,
            name="state_post_hook",
            fun=state_post_hook,
        )
        tool = _make_tool(noop, post_hooks=[post_hook])
        state = _make_state(agent)
        context = _make_context(state={"original": True})
        event = _make_event("noop", {"x": 1})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))

        assert context.state == {"post_hook_state": True}
        assert events[0].event_type == ExecutionEventType.HOOK_START
        assert events[1].event_type == ExecutionEventType.HOOK_END
        assert events[2].event_type == ExecutionEventType.TOOL_END

    @pytest.mark.asyncio
    async def test_updater_post_hook_returns_new_state(self, runner: Runner, agent: Agent) -> None:
        """When post-hook returns new_result with state"""

        def state_post_hook(tool: FunctionTool, result: ToolResult) -> dict:
            result.state.update({"new_key": "new_value"})
            return {
                "new_result": ToolResult(
                    success=True,
                    data=result.data,
                    state=result.state,
                )
            }

        def noop(x: int) -> str:
            return "ok"

        post_hook = FunctionHook(
            trigger=HookPhase.POST_TOOL_CALL,
            type=HookType.UPDATER,
            name="state_post_hook",
            fun=state_post_hook,
        )
        tool = _make_tool(noop, post_hooks=[post_hook])
        state = _make_state(agent)
        context = _make_context(state={"original": True})
        event = _make_event("noop", {"x": 1})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))

        assert context.state == {"original": True, "new_key": "new_value"}
        assert events[0].event_type == ExecutionEventType.HOOK_START
        assert events[1].event_type == ExecutionEventType.HOOK_END
        assert events[2].event_type == ExecutionEventType.TOOL_END

    @pytest.mark.asyncio
    async def test_updater_post_hook_updates_state_without_returning(
        self, runner: Runner, agent: Agent
    ) -> None:
        """When post-hook returns new_result with state"""

        def state_post_hook(tool: FunctionTool, result: ToolResult) -> dict:
            result.state.update({"new_key": "new_value"})
            return {
                "new_result": ToolResult(
                    success=True,
                    data=result.data,
                    state=None,
                )
            }

        def noop(x: int) -> str:
            return "ok"

        post_hook = FunctionHook(
            trigger=HookPhase.POST_TOOL_CALL,
            type=HookType.UPDATER,
            name="state_post_hook",
            fun=state_post_hook,
        )
        tool = _make_tool(noop, post_hooks=[post_hook])
        state = _make_state(agent)
        context = _make_context(state={"original": True})
        event = _make_event("noop", {"x": 1})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))

        assert context.state == {"original": True}
        assert events[0].event_type == ExecutionEventType.HOOK_START
        assert events[1].event_type == ExecutionEventType.HOOK_END
        assert events[2].event_type == ExecutionEventType.TOOL_END

    @pytest.mark.asyncio
    async def test_updater_post_hook_updates_state_returning_same_tool_result(
        self, runner: Runner, agent: Agent
    ) -> None:
        """When post-hook returns new_result with state"""

        def state_post_hook(tool: FunctionTool, result: ToolResult) -> dict:
            result.state.update({"new_key": "new_value"})
            return {"new_result": result}

        def noop(x: int) -> str:
            return "ok"

        post_hook = FunctionHook(
            trigger=HookPhase.POST_TOOL_CALL,
            type=HookType.UPDATER,
            name="state_post_hook",
            fun=state_post_hook,
        )
        tool = _make_tool(noop, post_hooks=[post_hook])
        state = _make_state(agent)
        context = _make_context(state={"original": True})
        event = _make_event("noop", {"x": 1})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))

        assert context.state == {"original": True, "new_key": "new_value"}
        assert events[0].event_type == ExecutionEventType.HOOK_START
        assert events[1].event_type == ExecutionEventType.HOOK_END
        assert events[2].event_type == ExecutionEventType.TOOL_END

    @pytest.mark.asyncio
    async def test_updater_post_hook_updates_state_with_none(
        self, runner: Runner, agent: Agent
    ) -> None:
        """When post-hook returns new_result with state"""

        def state_post_hook(tool: FunctionTool, result: ToolResult) -> dict:
            result.state = None
            return {"new_result": result}

        def noop(x: int) -> str:
            return "ok"

        post_hook = FunctionHook(
            trigger=HookPhase.POST_TOOL_CALL,
            type=HookType.UPDATER,
            name="state_post_hook",
            fun=state_post_hook,
        )
        tool = _make_tool(noop, post_hooks=[post_hook])
        state = _make_state(agent)
        context = _make_context(state={"original": True})
        event = _make_event("noop", {"x": 1})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))

        assert context.state == {"original": True}
        assert events[0].event_type == ExecutionEventType.HOOK_START
        assert events[1].event_type == ExecutionEventType.HOOK_END
        assert events[2].event_type == ExecutionEventType.TOOL_END

    @pytest.mark.asyncio
    async def test_updater_post_hook_clears_state(self, runner: Runner, agent: Agent) -> None:
        """When post-hook returns new_result with state"""

        def state_post_hook(tool: FunctionTool, result: ToolResult) -> dict:
            result.state = {}
            return {"new_result": result}

        def noop(x: int) -> str:
            return "ok"

        post_hook = FunctionHook(
            trigger=HookPhase.POST_TOOL_CALL,
            type=HookType.UPDATER,
            name="state_post_hook",
            fun=state_post_hook,
        )
        tool = _make_tool(noop, post_hooks=[post_hook])
        state = _make_state(agent)
        context = _make_context(state={"original": True})
        event = _make_event("noop", {"x": 1})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))

        assert context.state == {}
        assert events[0].event_type == ExecutionEventType.HOOK_START
        assert events[1].event_type == ExecutionEventType.HOOK_END
        assert events[2].event_type == ExecutionEventType.TOOL_END

    @pytest.mark.asyncio
    async def test_updater_post_hook_fails(self, runner: Runner, agent: Agent) -> None:
        """When post-hook returns new_result with state"""

        def state_post_hook(tool: FunctionTool, result: ToolResult) -> dict:
            result.state = {}
            return {}

        def noop(x: int) -> str:
            return "ok"

        post_hook = FunctionHook(
            trigger=HookPhase.POST_TOOL_CALL,
            type=HookType.UPDATER,
            name="state_post_hook",
            fun=state_post_hook,
        )
        tool = _make_tool(noop, post_hooks=[post_hook])
        state = _make_state(agent)
        context = _make_context(state={"original": True})
        event = _make_event("noop", {"x": 1})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))

        assert context.state == {"original": True}
        assert events[0].event_type == ExecutionEventType.HOOK_START
        assert events[1].event_type == ExecutionEventType.HOOK_ERROR
        assert events[2].event_type == ExecutionEventType.TOOL_END

    @pytest.mark.asyncio
    async def test_listener_post_hook_does_not_modify_result(
        self, runner: Runner, agent: Agent
    ) -> None:
        """Listener post-hooks observe but cannot modify the result."""
        observed = {}

        def listener(tool: FunctionTool, result: ToolResult) -> None:
            result.state.update({"updated": True})
            observed["data"] = result.data
            return None

        def greet(name: str) -> str:
            return f"hello {name}"

        post_hook = FunctionHook(
            trigger=HookPhase.POST_TOOL_CALL,
            type=HookType.LISTENER,
            name="observe_result",
            fun=listener,
        )
        tool = _make_tool(greet, post_hooks=[post_hook])
        state = _make_state(agent)
        context = _make_context(state={"original": True})
        event = _make_event("greet", {"name": "world"})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))

        assert observed["data"] == "hello world"
        assert context.state == {"original": True}
        assert events[0].event_type == ExecutionEventType.HOOK_START
        assert events[1].event_type == ExecutionEventType.HOOK_END
        assert events[2].event_type == ExecutionEventType.TOOL_END
        assert events[2].action.observation == "hello world"

    @pytest.mark.asyncio
    async def test_listener_updater_post_hooks(self, runner: Runner, agent: Agent) -> None:
        """Listener post-hooks observe but cannot modify the result."""
        observed = {}

        def updater(tool: FunctionTool, result: ToolResult) -> dict:
            result.state.update({"updated": True})
            result.data = result.data + " updated by updater"
            return {"new_result": result}

        def listener(tool: FunctionTool, result: ToolResult) -> None:
            observed["data"] = result.data
            return None

        def greet(name: str) -> str:
            return f"hello {name}"

        listener_post_hook = FunctionHook(
            trigger=HookPhase.POST_TOOL_CALL,
            type=HookType.LISTENER,
            name="observe_result",
            fun=listener,
        )
        updater_post_hook = FunctionHook(
            trigger=HookPhase.POST_TOOL_CALL,
            type=HookType.UPDATER,
            name="update_result",
            fun=updater,
        )
        tool = _make_tool(greet, post_hooks=[listener_post_hook, updater_post_hook])
        state = _make_state(agent)
        context = _make_context(state={"original": True})
        event = _make_event("greet", {"name": "world"})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))

        assert observed["data"] == "hello world updated by updater"
        assert context.state == {"original": True, "updated": True}
        assert events[0].event_type == ExecutionEventType.HOOK_START
        assert events[1].event_type == ExecutionEventType.HOOK_END
        assert events[2].event_type == ExecutionEventType.HOOK_START
        assert events[3].event_type == ExecutionEventType.HOOK_END
        assert events[4].event_type == ExecutionEventType.TOOL_END
        assert events[4].action.observation == "hello world updated by updater"


# ===========================================================================
# 4. Combined pre + post hooks
# ===========================================================================


class TestExecuteToolCombinedHooks:
    """Test full pipeline: pre-hooks -> execute -> post-hooks."""

    @pytest.mark.asyncio
    async def test_pre_and_post_hooks_full_pipeline(self, runner: Runner, agent: Agent) -> None:
        """Pre-hook modifies args, tool runs, post-hook modifies result and state updates."""
        execution_order = []

        def pre_hook(tool: FunctionTool, tool_args: dict) -> dict:
            tool_args["state"].update({"pre_hook": True})
            execution_order.append("pre")
            tool_args["x"] = tool_args["x"] * 2
            return {"new_tool_args": tool_args}

        def post_hook(tool: FunctionTool, result: ToolResult) -> dict:
            result.state.update({"post_hook": True})
            result.data = result.data + 1000
            execution_order.append("post")
            return {
                "new_result": ToolResult(
                    success=True,
                    data=result.data,
                    state=result.state,
                )
            }

        captured_tool_state = {}

        def compute(x: int, state: dict) -> int:
            nonlocal captured_tool_state
            captured_tool_state = deepcopy(state)
            execution_order.append("tool")
            return x + 1

        pre = FunctionHook(
            trigger=HookPhase.PRE_TOOL_CALL,
            type=HookType.UPDATER,
            name="double_x",
            fun=pre_hook,
        )
        post = FunctionHook(
            trigger=HookPhase.POST_TOOL_CALL,
            type=HookType.UPDATER,
            name="add_1000",
            fun=post_hook,
        )
        tool = _make_tool(compute, pre_hooks=[pre], post_hooks=[post])
        state = _make_state(agent)
        context = _make_context(state={"initial": True})
        event = _make_event("compute", {"x": 5})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))

        assert events[0].event_type == ExecutionEventType.HOOK_START
        assert events[1].event_type == ExecutionEventType.HOOK_END
        assert events[2].event_type == ExecutionEventType.HOOK_START
        assert events[3].event_type == ExecutionEventType.HOOK_END
        assert events[4].event_type == ExecutionEventType.TOOL_END
        assert captured_tool_state == {"initial": True, "pre_hook": True}
        assert context.state == {"initial": True, "pre_hook": True, "post_hook": True}
        # x=5 -> pre doubles to 10 -> tool returns 11 -> post adds 1000 = 1011
        assert events[4].action.observation == 1011
        assert execution_order == ["pre", "tool", "post"]

    @pytest.mark.asyncio
    async def test_state_flows_through_pre_tool_post(self, runner: Runner, agent: Agent) -> None:
        """State changes propagate: pre-hook sets state -> tool sees it -> post-hook modifies."""

        def pre_hook(tool: FunctionTool, tool_args: dict) -> dict:
            tool_args["state"] = {"step": "pre"}
            return {"new_tool_args": tool_args}

        def post_hook(tool: FunctionTool, result: ToolResult) -> dict:
            return {
                "new_result": ToolResult(
                    success=True,
                    data=result.data,
                    state={"step": "post", "tool_saw": result.data},
                )
            }

        def observe_state(x: int, state: dict) -> str:
            return f"state_step={state.get('step', 'none')}"

        pre = FunctionHook(
            trigger=HookPhase.PRE_TOOL_CALL,
            type=HookType.UPDATER,
            name="set_state",
            fun=pre_hook,
        )
        post = FunctionHook(
            trigger=HookPhase.POST_TOOL_CALL,
            type=HookType.UPDATER,
            name="update_state",
            fun=post_hook,
        )
        tool = _make_tool(observe_state, pre_hooks=[pre], post_hooks=[post])
        state = _make_state(agent)
        context = _make_context(state={"step": "initial"})
        event = _make_event("observe_state", {"x": 1})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))

        # After pre-hook, context.state == {"step": "pre"}
        # Tool sees state={"step": "pre"} and returns that as data
        tool_end = [e for e in events if e.event_type == ExecutionEventType.TOOL_END]
        assert tool_end[0].action.observation == "state_step=pre"
        # After post-hook, context.state updated
        assert context.state == {"step": "post", "tool_saw": "state_step=pre"}

    @pytest.mark.asyncio
    async def test_mixed_updater_listener_hooks(self, runner: Runner, agent: Agent) -> None:
        """Updater + listener pre-hooks, updater + listener post-hooks together."""
        execution_trace = []
        state_trace = {}

        def updater_pre(tool: FunctionTool, tool_args: dict) -> dict:
            state_trace["updater_pre"] = deepcopy(tool_args.get("state"))
            execution_trace.append("updater_pre")
            tool_args["x"] = tool_args["x"] + 10
            tool_args["state"] = None
            return {"new_tool_args": tool_args}

        def listener_pre(tool: FunctionTool, tool_args: dict) -> None:
            state_trace["listener_pre"] = deepcopy(tool_args.get("state"))
            tool_args["state"].update({"pre_listener": True})  # This should not affect the state
            execution_trace.append("listener_pre")
            return None

        def updater_post(tool: FunctionTool, result: ToolResult) -> dict:
            state_trace["updater_post"] = deepcopy(result.state)
            execution_trace.append("updater_post")
            result.state.update({"post_updater": True})
            return {
                "new_result": ToolResult(
                    success=True, data=result.data + "_modified", state=result.state
                )
            }

        def listener_post(tool: FunctionTool, result: ToolResult) -> None:
            state_trace["listener_post"] = deepcopy(result.state)
            execution_trace.append("listener_post")
            result.state.update({"post_listener": True})  # This should not affect the state
            return None

        def my_tool(x: int, state: dict) -> str:
            state_trace["tool"] = deepcopy(state)
            state.update({"tool": True})
            execution_trace.append("tool")
            return str(x)

        hooks_pre = [
            FunctionHook(
                trigger=HookPhase.PRE_TOOL_CALL,
                type=HookType.UPDATER,
                name="updater_pre",
                fun=updater_pre,
            ),
            FunctionHook(
                trigger=HookPhase.PRE_TOOL_CALL,
                type=HookType.LISTENER,
                name="listener_pre",
                fun=listener_pre,
            ),
        ]
        hooks_post = [
            FunctionHook(
                trigger=HookPhase.POST_TOOL_CALL,
                type=HookType.UPDATER,
                name="updater_post",
                fun=updater_post,
            ),
            FunctionHook(
                trigger=HookPhase.POST_TOOL_CALL,
                type=HookType.LISTENER,
                name="listener_post",
                fun=listener_post,
            ),
        ]
        tool = _make_tool(my_tool, pre_hooks=hooks_pre, post_hooks=hooks_post)
        state = _make_state(agent)
        context = _make_context(state={"initial": True})
        event = _make_event("my_tool", {"x": 5})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))

        assert execution_trace == [
            "updater_pre",
            "listener_pre",
            "tool",
            "updater_post",
            "listener_post",
        ]

        # x=5 -> updater_pre adds 10 -> tool("15") -> updater_post modifies
        tool_end = [e for e in events if e.event_type == ExecutionEventType.TOOL_END]
        assert len(events) == 9
        assert events[1].event_type == ExecutionEventType.HOOK_ERROR
        assert events[-1].event_type == ExecutionEventType.TOOL_END
        assert tool_end[0].action.observation == "5_modified"
        # check state flow
        assert (
            state_trace["updater_pre"] == {"initial": True}
        )  # the updater tool pre hooks fails (can't set the state to None), so the listener pre hook and the tool see the original state
        assert state_trace["listener_pre"] == {"initial": True}
        assert state_trace["tool"] == {
            "initial": True
        }  # listener pre hook can't modify the state, so the tool sees the original state
        assert state_trace["updater_post"] == {"initial": True}  # the tool did not modify the state
        assert state_trace["listener_post"] == {
            "initial": True,
            "post_updater": True,
        }  # the listener post hook sees the state modified by the updater post hook


# ===========================================================================
# 5. Streamable tool execution
# ===========================================================================


class TestExecuteStreamableTool:
    """Test _execute_tool with StreamableFunctionTool."""

    @pytest.mark.asyncio
    async def test_streamable_tool_yields_partial_and_final(
        self, runner: Runner, agent: Agent
    ) -> None:
        """Streamable tool yields RESPONSE partials and a final TOOL_END."""
        from domyn.agents.tools import StreamableToolResult

        async def streaming_tool(x: int) -> AsyncGenerator[StreamableToolResult | ToolResult, None]:
            yield StreamableToolResult(data=Part(text="chunk1"))
            yield StreamableToolResult(data=Part(text="chunk2"))
            yield ToolResult(success=True, data="final")

        tool = _make_streamable_tool(streaming_tool)
        state = _make_state(agent)
        context = _make_context()
        event = _make_event("streaming_tool", {"x": 1})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))

        partials = [
            e for e in events if e.event_type == ExecutionEventType.RESPONSE and e.is_partial
        ]
        tool_end = [e for e in events if e.event_type == ExecutionEventType.TOOL_END]
        assert len(partials) == 2
        assert len(tool_end) == 1
        assert tool_end[0].action.observation == "final"

    @pytest.mark.asyncio
    async def test_streamable_tool_with_hooks(self, runner: Runner, agent: Agent) -> None:
        """Streamable tool works with pre/post hooks."""
        from domyn.agents.tools import StreamableToolResult

        def pre_hook(tool: FunctionTool, tool_args: dict) -> dict:
            tool_args["x"] = tool_args["x"] * 3
            tool_args["state"] = {"pre_hook": True}
            return {"new_tool_args": tool_args}

        def post_hook(tool: FunctionTool, result: ToolResult) -> dict:
            result.state.update({"post_hook": True})
            return {
                "new_result": ToolResult(
                    success=True, data=f"post:{result.data}", state=result.state
                )
            }

        async def streaming_tool(x: int) -> AsyncGenerator[StreamableToolResult | ToolResult, None]:
            yield StreamableToolResult(data=Part(text=f"streaming {x}"))
            yield ToolResult(success=True, data=f"done:{x}")

        pre = FunctionHook(
            trigger=HookPhase.PRE_TOOL_CALL, type=HookType.UPDATER, name="triple", fun=pre_hook
        )
        post = FunctionHook(
            trigger=HookPhase.POST_TOOL_CALL, type=HookType.UPDATER, name="prefix", fun=post_hook
        )
        tool = _make_streamable_tool(streaming_tool, pre_hooks=[pre], post_hooks=[post])
        state = _make_state(agent)
        context = _make_context(state={"initial": True})
        event = _make_event("streaming_tool", {"x": 2})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))

        tool_end = [e for e in events if e.event_type == ExecutionEventType.TOOL_END]
        # x=2 * 3=6, tool yields "done:6", post prepends "post:"
        assert tool_end[0].action.observation == "post:done:6"
        assert context.state == {"pre_hook": True, "post_hook": True}

    @pytest.mark.asyncio
    async def test_subagent_suppresses_partials_and_uses_streamed_observation(
        self, runner: Runner, agent: Agent
    ) -> None:
        """For a subagent, streamable chunks emit no partial RESPONSE; the joined
        streamed text becomes the TOOL_END observation."""
        from domyn.agents.tools import StreamableToolResult

        async def streaming_tool(x: int) -> AsyncGenerator[StreamableToolResult | ToolResult, None]:
            yield StreamableToolResult(data=Part(text="chunk1"))
            yield StreamableToolResult(data=[Part(text="chunk2"), Part(text="chunk3")])
            yield ToolResult(success=True, data="final")

        tool = _make_streamable_tool(streaming_tool)
        state = _make_state(agent)
        context = _make_context()
        context.is_subagent = True
        event = _make_event("streaming_tool", {"x": 1})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))

        partials = [
            e for e in events if e.event_type == ExecutionEventType.RESPONSE and e.is_partial
        ]
        tool_end = [e for e in events if e.event_type == ExecutionEventType.TOOL_END]
        assert partials == []  # no partial RESPONSE events for subagents
        assert len(tool_end) == 1
        # The streamed parts (serialized, in order) become the observation; data is
        # ignored. Assert on the serialized parts rather than the exact NOTE prose so
        # the test stays robust to wording tweaks of the sub-agent guidance.
        observation = tool_end[0].action.observation
        for part in [Part(text="chunk1"), Part(text="chunk2"), Part(text="chunk3")]:
            assert part.model_dump_json(exclude_none=True) in observation
        assert "final" not in observation  # tool_result.data is not used as the observation
        # Nothing leaks to the user through the event content for subagents.
        assert tool_end[0].content == []

    @pytest.mark.asyncio
    async def test_subagent_folds_streamed_parts_and_emitted_content_into_observation(
        self, runner: Runner, agent: Agent
    ) -> None:
        """For a subagent, both the suppressed streamed parts and the tool's
        emitted_content are folded into the observation; the event content stays empty."""
        from domyn.agents.tools import StreamableToolResult

        async def streaming_tool(x: int) -> AsyncGenerator[StreamableToolResult | ToolResult, None]:
            yield StreamableToolResult(data=Part(text="streamed-0"))
            yield ToolResult(success=True, data="final", emitted_content=[Part(text="emitted-0")])

        tool = _make_streamable_tool(streaming_tool)
        state = _make_state(agent)
        context = _make_context()
        context.is_subagent = True
        event = _make_event("streaming_tool", {"x": 1})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))

        tool_end = [e for e in events if e.event_type == ExecutionEventType.TOOL_END]
        assert len(tool_end) == 1
        # Both the streamed part and the emitted_content appear in the observation...
        observation = tool_end[0].action.observation
        for part in [Part(text="streamed-0"), Part(text="emitted-0")]:
            assert part.model_dump_json(exclude_none=True) in observation
        # ...and nothing is carried on the event content.
        assert tool_end[0].content == []

    @pytest.mark.asyncio
    async def test_subagent_without_streamed_chunks_keeps_data_observation(
        self, runner: Runner, agent: Agent
    ) -> None:
        """A subagent tool that streams nothing keeps tool_result.data as observation."""
        from domyn.agents.tools import StreamableToolResult

        async def streaming_tool(x: int) -> AsyncGenerator[StreamableToolResult | ToolResult, None]:
            yield ToolResult(success=True, data="final")

        tool = _make_streamable_tool(streaming_tool)
        state = _make_state(agent)
        context = _make_context()
        context.is_subagent = True
        event = _make_event("streaming_tool", {"x": 1})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))

        tool_end = [e for e in events if e.event_type == ExecutionEventType.TOOL_END]
        assert tool_end[0].action.observation == "final"

    @pytest.mark.asyncio
    async def test_use_stop_agent_suppresses_partials_even_when_not_subagent(
        self, runner: Runner, agent: Agent
    ) -> None:
        """An entrypoint agent whose planner runs with use_stop=True also suppresses
        partials: chunks become the observation and the event content stays empty."""
        from domyn.agents.tools import StreamableToolResult

        async def streaming_tool(x: int) -> AsyncGenerator[StreamableToolResult | ToolResult, None]:
            yield StreamableToolResult(data=Part(text="chunk1"))
            yield ToolResult(success=True, data="final", emitted_content=[Part(text="emitted-0")])

        agent.planner.use_stop = True
        tool = _make_streamable_tool(streaming_tool)
        state = _make_state(agent)
        context = _make_context()
        # Not a subagent: suppression is driven solely by use_stop.
        assert context.is_subagent is False
        event = _make_event("streaming_tool", {"x": 1})

        events = await _collect_events(runner._execute_tool(tool, state, context, event))

        partials = [
            e for e in events if e.event_type == ExecutionEventType.RESPONSE and e.is_partial
        ]
        tool_end = [e for e in events if e.event_type == ExecutionEventType.TOOL_END]
        assert partials == []
        assert len(tool_end) == 1
        observation = tool_end[0].action.observation
        for part in [Part(text="chunk1"), Part(text="emitted-0")]:
            assert part.model_dump_json(exclude_none=True) in observation
        assert tool_end[0].content == []


class TestTheCeilingOnAToolCall:
    """A tool that never returns used to hold the turn for as long as it liked.

    Nothing here is about how long work may take: the ceiling sits above every
    timeout the package configures for itself, so a tool with its own limit
    still fails with its own message. This catches the call with no limit.
    """

    @pytest.mark.asyncio
    async def test_a_wedged_tool_becomes_an_observation_not_a_hang(self, agent: Agent) -> None:
        """The model is told, and told enough to do something other than retry."""

        async def never_returns() -> str:
            await asyncio.sleep(30)
            return "unreachable"

        runner = Runner(tool_timeout_s=0.05)
        events = await _collect_events(
            runner._execute_tool(
                _make_tool(never_returns),
                _make_state(agent),
                _make_context(),
                _make_event("never_returns", {}),
            )
        )

        errors = [e for e in events if e.event_type == ExecutionEventType.TOOL_ERROR]
        assert len(errors) == 1
        assert errors[0].error_code == "tool_timeout"
        assert "never_returns" in errors[0].action.observation

    @pytest.mark.asyncio
    async def test_a_tool_that_finishes_in_time_is_untouched(self, agent: Agent) -> None:
        runner = Runner(tool_timeout_s=5)
        events = await _collect_events(
            runner._execute_tool(
                _make_tool(lambda a: a * 2),
                _make_state(agent),
                _make_context(),
                _make_event("<lambda>", {"a": 21}),
            )
        )

        done = next(e for e in events if e.event_type == ExecutionEventType.TOOL_END)
        assert done.action.observation == 42

    @pytest.mark.asyncio
    async def test_a_tool_sets_its_own(self, agent: Agent) -> None:
        """Some work is legitimately slower than any number chosen for everyone."""

        async def slow() -> str:
            await asyncio.sleep(0.15)
            return "done"

        tool = _make_tool(slow)
        tool.timeout_s = 5
        runner = Runner(tool_timeout_s=0.05)
        events = await _collect_events(
            runner._execute_tool(tool, _make_state(agent), _make_context(), _make_event("slow", {}))
        )

        assert [e.event_type for e in events] == [ExecutionEventType.TOOL_END]

    @pytest.mark.asyncio
    async def test_zero_means_no_ceiling(self, agent: Agent) -> None:
        async def slow() -> str:
            await asyncio.sleep(0.1)
            return "done"

        events = await _collect_events(
            Runner(tool_timeout_s=0)._execute_tool(
                _make_tool(slow), _make_state(agent), _make_context(), _make_event("slow", {})
            )
        )

        assert [e.event_type for e in events] == [ExecutionEventType.TOOL_END]

    @pytest.mark.asyncio
    async def test_a_streaming_tool_is_measured_between_chunks(self, agent: Agent) -> None:
        """Otherwise a total would cut off precisely the tools that report what
        they are doing — the ones there is least reason to give up on."""
        from domyn.agents.tools.base import UpdateToolResult

        async def reporting() -> AsyncGenerator[Any, None]:
            for done in range(4):
                await asyncio.sleep(0.04)
                yield UpdateToolResult(data={"progress": done})
            yield ToolResult(success=True, data="finished")

        runner = Runner(tool_timeout_s=0.1)
        events = await _collect_events(
            runner._execute_tool(
                _make_streamable_tool(reporting),
                _make_state(agent),
                _make_context(),
                _make_event("reporting", {}),
            )
        )

        # 0.16s in total, over a 0.1s ceiling, and it still finishes.
        assert [e for e in events if e.event_type == ExecutionEventType.TOOL_END]
        assert len([e for e in events if e.event_type == ExecutionEventType.TOOL_UPDATE]) == 4

    @pytest.mark.asyncio
    async def test_a_streaming_tool_that_goes_quiet_is_given_up_on(self, agent: Agent) -> None:
        async def stalls() -> AsyncGenerator[Any, None]:
            from domyn.agents.tools.base import UpdateToolResult

            yield UpdateToolResult(data={"progress": 0})
            await asyncio.sleep(30)
            yield ToolResult(success=True, data="unreachable")

        runner = Runner(tool_timeout_s=0.05)
        events = await _collect_events(
            runner._execute_tool(
                _make_streamable_tool(stalls),
                _make_state(agent),
                _make_context(),
                _make_event("stalls", {}),
            )
        )

        assert events[-1].event_type == ExecutionEventType.TOOL_ERROR
        assert events[-1].error_code == "tool_timeout"

    @pytest.mark.asyncio
    async def test_a_delegate_is_exempt(self, agent: Agent) -> None:
        """A delegate is a whole nested turn, bounded by its own transport, and a
        sub-agent working as intended must not trip a ceiling meant for calls."""
        from domyn.agents.tools import DelegateTool

        runner = Runner(tool_timeout_s=30)
        delegate = DelegateTool(tool_name="helper")

        assert runner._tool_ceiling(delegate) == 0
        delegate.timeout_s = 5
        assert runner._tool_ceiling(delegate) == 5
