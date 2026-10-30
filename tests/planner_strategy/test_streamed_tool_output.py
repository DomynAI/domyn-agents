"""What a streamable tool showed the user reaches the model inside its result.

The runner records each chunk as a partial RESPONSE carrying the tool's action.
Rendered as assistant turns, they were moved after the tool result and the
request ended on an assistant message, which DeepSeek answered with nothing.
"""

import uuid
from unittest.mock import MagicMock

import pytest

from domyn.agents.core import BaseEvent, ExecutionEventType, InteractionContext, Part, ToolAction
from domyn.agents.planner_strategy import (
    ToolUseContextManager,
    ToolUseWithThoughtContextManager,
)
from domyn.agents.planner_strategy.utils import reorder_messages_around_tool_calls


def _agent():
    agent = MagicMock()
    agent.name = "test_agent"
    agent.description = "A test agent"
    agent.instruction = "You are helpful."
    agent.tools = []
    agent.sub_agents = []
    agent.memory_manager = None
    agent.get_tool_by_name = lambda name: None
    agent.stream = False
    return agent


def _context(history):
    ctx = MagicMock(spec=InteractionContext)
    ctx.conversation_id = "conv_1"
    ctx.user_id = "user_1"
    ctx.interaction_id = "int_1"
    ctx.conversation_history = history
    ctx.interaction_history = []
    ctx.handoff_targets_description = []
    ctx.state = {}
    ctx.run_config = {}
    ctx.agent_input = None
    return ctx


def _event(event_type, *, action=None, text=None, is_partial=False):
    return BaseEvent(
        event_type=event_type,
        author="test_agent",
        event_id=str(uuid.uuid4()),
        action=action,
        content=[Part(text=text)] if text else [],
        is_partial=is_partial,
    )


def _history(*, observation="done"):
    call = ToolAction(call_id="call_1", name="report", parameters={})
    end = call.model_copy(update={"observation": observation})
    return [
        _event(ExecutionEventType.USER_INPUT, text="Write the report"),
        _event(ExecutionEventType.TOOL_START, action=call),
        _event(ExecutionEventType.RESPONSE, action=call, text="# Report\n", is_partial=True),
        _event(ExecutionEventType.RESPONSE, action=call, text="All good.", is_partial=True),
        _event(ExecutionEventType.TOOL_END, action=end),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("manager", [ToolUseContextManager, ToolUseWithThoughtContextManager])
async def test_the_request_ends_on_the_tool_result(manager):
    messages = await manager().get_context(context=_context(_history()), agent=_agent())
    messages = reorder_messages_around_tool_calls(messages)

    assert [m.role for m in messages] == ["system", "user", "assistant", "tool"]
    assert messages[-1].content.endswith(
        "done\n\nShown to the user while the tool ran:\n# Report\nAll good."
    )


@pytest.mark.asyncio
async def test_a_tool_with_an_empty_result_still_says_what_it_showed():
    messages = await ToolUseContextManager().get_context(
        context=_context(_history(observation="")), agent=_agent()
    )

    tool = messages[-1]
    assert tool.role == "tool"
    assert tool.content.endswith("Shown to the user while the tool ran:\n# Report\nAll good.")


@pytest.mark.asyncio
async def test_a_partial_response_without_a_tool_is_still_an_assistant_turn():
    history = [
        _event(ExecutionEventType.USER_INPUT, text="Hi"),
        _event(ExecutionEventType.RESPONSE, text="Let me look.", is_partial=True),
    ]
    messages = await ToolUseContextManager().get_context(context=_context(history), agent=_agent())

    assert messages[-1].role == "assistant"
    assert messages[-1].content == "Let me look."
