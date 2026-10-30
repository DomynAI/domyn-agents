import random
from collections.abc import AsyncGenerator
from unittest.mock import MagicMock

import pytest

from domyn.agents import Runner
from domyn.agents.agents import Agent
from domyn.agents.core import BaseEvent, ExecutionEventType, InteractionContext, Part, ToolAction
from domyn.agents.hooks import HookPhase, HookType
from domyn.agents.llm import BaseLLMProvider
from domyn.agents.planner_strategy import BasePlannerStrategy
from domyn.agents.tools import StreamableFunctionTool, StreamableToolResult


@pytest.fixture
def agent():
    return Agent(
        name="BaseAgentTest",
        description="A test agent.",
        instruction="You are a test agent.",
        llm_provider=MagicMock(BaseLLMProvider),
        planner=MagicMock(BasePlannerStrategy),
    )


@pytest.mark.asyncio
async def test_streammable_tool_multiple_response(agent: Agent):
    async def sample_streaming_tool(text: str) -> AsyncGenerator[StreamableToolResult, None]:
        """A sample streaming tool that yields multiple parts."""
        for i in range(3):
            yield StreamableToolResult(data=Part(text=f"{text} part {i + 1}"))
        yield StreamableToolResult(data=Part(text="observation"))

    tool = StreamableFunctionTool(func=sample_streaming_tool)

    agent.tools.append(tool)

    tool_action = ToolAction(name=tool.name, parameters={"text": "Hello"})

    async def execute_mock(*args, **kwargs):
        yield BaseEvent(
            event_type=ExecutionEventType.TOOL_START,
            action=tool_action,
            author="agent",
            content=[Part(text="Starting tool execution...")],
        )

    agent.planner.execute = execute_mock

    runner = Runner()
    execute = runner.run(
        application_name="TestApp",
        user_id="test",
        root=agent,
        user_input=BaseEvent(
            event_type=ExecutionEventType.USER_INPUT,
            content=[Part(text="Test user input")],
            author="user",
        ),
    )

    all_events = [event async for event in execute]
    assert len(all_events) == 7  # AGENT_START + TOOL_START + 4 tool results + TOOL_END
    assert all_events[1].event_type == ExecutionEventType.TOOL_START
    assert all_events[6].event_type == ExecutionEventType.TOOL_END
    assert all_events[6].action == ToolAction(
        name=tool.name,
        parameters={"text": "Hello"},
        observation={"text": "observation"},
        call_id=tool_action.call_id,
    )


@pytest.mark.asyncio
async def test_streamable_tool_multiple_response_with_list_of_parts(agent: Agent):
    async def sample_streaming_tool(text: str) -> AsyncGenerator[StreamableToolResult, None]:
        """A sample streaming tool that yields multiple parts."""
        for i in range(3):
            yield StreamableToolResult(data=Part(text=f"{text} part {i + 1}"))
        yield StreamableToolResult(data=[Part(text="observation"), Part(text="observation part 2")])

    tool = StreamableFunctionTool(func=sample_streaming_tool)

    agent.tools.append(tool)

    tool_action = ToolAction(name=tool.name, parameters={"text": "Hello"})

    async def execute_mock(*args, **kwargs):
        yield BaseEvent(
            event_type=ExecutionEventType.TOOL_START,
            action=tool_action,
            author="agent",
            content=[Part(text="Starting tool execution...")],
        )

    agent.planner.execute = execute_mock

    runner = Runner()
    execute = runner.run(
        application_name="TestApp",
        user_id="test",
        root=agent,
        user_input=BaseEvent(
            event_type=ExecutionEventType.USER_INPUT,
            content=[Part(text="Test user input")],
            author="user",
        ),
    )

    all_events = [event async for event in execute]
    assert len(all_events) == 7  # AGENT_START + TOOL_START + 4 tool results + TOOL_END
    assert all_events[1].event_type == ExecutionEventType.TOOL_START
    assert all_events[6].event_type == ExecutionEventType.TOOL_END
    assert all_events[6].action == ToolAction(
        name=tool.name,
        parameters={"text": "Hello"},
        observation=[{"text": "observation"}, {"text": "observation part 2"}],
        call_id=tool_action.call_id,
    )


@pytest.mark.asyncio
async def test_streamable_tool_multiple_response_string_as_observation(agent: Agent):
    observation = random.randint(1, 1000)

    async def sample_streaming_tool(text: str) -> AsyncGenerator[StreamableToolResult, None]:
        """A sample streaming tool that yields multiple parts."""
        for i in range(3):
            yield StreamableToolResult(data=Part(text=f"{text} part {i + 1}"))
        yield observation

    tool = StreamableFunctionTool(func=sample_streaming_tool)

    agent.tools.append(tool)

    tool_action = ToolAction(name=tool.name, parameters={"text": "Hello"})

    async def execute_mock(*args, **kwargs):
        yield BaseEvent(
            event_type=ExecutionEventType.TOOL_START,
            action=tool_action,
            author="agent",
            content=[Part(text="Starting tool execution...")],
        )

    agent.planner.execute = execute_mock

    runner = Runner()
    execute = runner.run(
        application_name="TestApp",
        user_id="test",
        root=agent,
        user_input=BaseEvent(
            event_type=ExecutionEventType.USER_INPUT,
            content=[Part(text="Test user input")],
            author="user",
        ),
    )

    all_events = [event async for event in execute]
    assert len(all_events) == 6  # AGENT_START + TOOL_START + 3 tool results streaming + TOOL_END
    assert all_events[1].event_type == ExecutionEventType.TOOL_START
    assert all_events[5].event_type == ExecutionEventType.TOOL_END
    assert all_events[5].action == ToolAction(
        name=tool.name,
        parameters={"text": "Hello"},
        observation=str(observation),
        call_id=tool_action.call_id,
    )


@pytest.mark.asyncio
async def test_streamable_tool_state_propagation():
    """Test that state mutations are preserved (not lost to deepcopy)."""

    async def tool_that_modifies_state(state: dict) -> AsyncGenerator[str, None]:
        """Tool that modifies state."""
        state["modified"] = True
        state["counter"] = state.get("counter", 0) + 1
        yield "State modified"

    tool = StreamableFunctionTool(func=tool_that_modifies_state)

    # Initial state
    state = {"counter": 0}

    # Execute tool
    results = []
    async for result in tool.execute(state=state):
        if hasattr(result, "data"):
            results.append(result)

    # State should be modified (not lost to deepcopy)
    assert state["modified"] is True
    assert state["counter"] == 1


@pytest.mark.asyncio
async def test_streamable_tool_no_copy_keys_respected():
    """Test that NO_COPY_KEYS (state, agent, tool, result) are not deepcopied."""

    # Create objects that we can track by identity
    agent_obj = MagicMock(name="test_agent")
    tool_obj = MagicMock(name="test_tool")
    state_obj = {"key": "value"}
    result_obj = MagicMock(name="test_result")

    # Track if objects are the same instance (not copied)
    received_ids = {}

    async def tool_that_checks_identity(
        state: dict, agent, tool, result
    ) -> AsyncGenerator[str, None]:
        """Tool that checks if objects are same instance."""
        received_ids["state"] = id(state)
        received_ids["agent"] = id(agent)
        received_ids["tool"] = id(tool)
        received_ids["result"] = id(result)
        yield "checked"

    streamable_tool = StreamableFunctionTool(func=tool_that_checks_identity)

    # Original IDs
    original_ids = {
        "state": id(state_obj),
        "agent": id(agent_obj),
        "tool": id(tool_obj),
        "result": id(result_obj),
    }

    async for _ in streamable_tool.execute(
        state=state_obj,
        agent=agent_obj,
        tool=tool_obj,
        result=result_obj,
    ):
        pass

    # Objects should be the same instances (not deepcopied)
    assert received_ids["state"] == original_ids["state"]
    assert received_ids["agent"] == original_ids["agent"]
    assert received_ids["tool"] == original_ids["tool"]
    assert received_ids["result"] == original_ids["result"]
