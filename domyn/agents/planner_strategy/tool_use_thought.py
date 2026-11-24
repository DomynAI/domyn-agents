"""ReAct-style tool-use planner strategy.

Extends ``ToolUsePlannerStrategy`` with explicit thought emission: each
action's ``thought`` is the call's ``reasoning`` argument, or, when the first
call has none, the response's ``reasoning_content``, ``thinking_blocks`` or
text written beside the calls. The thought is shown to the user; it is not sent
back to the model as text, since the argument already carries it.

The companion ``ToolUseWithThoughtContextManager`` instructs the model to put
its reasoning in that argument and never to write text beside a call.

When content appears **without** tool calls it is the response to the user
(streamable), not a thought.

The loop is ``ToolUsePlannerStrategy.execute``: this strategy only switches on
``carries_thought`` and brings its context manager. It used to be a copy of that
loop, and a fix made to one was found missing from the other more than once.
"""

from typing import ClassVar

from domyn.agents.planner_strategy.context_management.tool_use import (
    ToolUseWithThoughtContextManager,
)
from domyn.agents.planner_strategy.tool_use import (
    ToolUsePlannerStrategy,
    _extract_thought,
)

__all__ = ["ToolUseWithThoughtPlannerStrategy", "_extract_thought"]


class ToolUseWithThoughtPlannerStrategy(ToolUsePlannerStrategy):
    """Native tool-calling strategy with explicit thought emission (ReAct-style).

    Uses ``ToolUseWithThoughtContextManager`` by default, which appends a reasoning
    instruction to the system prompt telling the LLM to put its reasoning in each
    tool call's ``reasoning`` argument. That argument — or, when it is empty on
    the first call, ``reasoning_content`` / ``thinking_blocks`` for models with
    native extended thinking, then any text written beside the call — is the
    call's ``thought``.

    Thought extraction priority (first non-empty wins):
    0. the call's ``reasoning`` argument
    1. ``reasoning_content``  — DeepSeek R1, Kimi, etc.
    2. ``thinking_blocks``    — Anthropic extended thinking
    3. ``response.content``   — text the model wrote beside the call
    """

    type: str = "tool_use_thought"
    name: str = "tool_use_thought"
    context_manager: ToolUseWithThoughtContextManager = ToolUseWithThoughtContextManager()
    carries_thought: ClassVar[bool] = True
