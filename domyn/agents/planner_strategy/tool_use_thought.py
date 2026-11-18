"""ReAct-style tool-use planner strategy.

Extends ``ToolUsePlannerStrategy`` with explicit thought emission:
the model's reasoning (from ``reasoning_content``, ``thinking_blocks``,
or preamble ``content`` when accompanied by tool calls) is extracted and
surfaced as ``Part(thought=...)`` inside the ``LLM_END`` event, and propagated
into every action's ``thought`` field so that ``ToolUseContextManager``
correctly reconstructs the history.

When the model has no native thinking capability, the companion
``ToolUseWithThoughtContextManager`` instructs it to emit plain-text reasoning
as ``content`` before tool calls; that text is captured as the thought.

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
    instruction to the system prompt so that the LLM emits plain-text ``content``
    before every tool call.  That content — or ``reasoning_content`` /
    ``thinking_blocks`` for models with native extended thinking — is then
    extracted and emitted as ``Part(thought=...)`` inside the ``LLM_END`` event.

    Thought extraction priority (first non-empty wins):
    1. ``reasoning_content``  — DeepSeek R1, Kimi, etc.
    2. ``thinking_blocks``    — Anthropic extended thinking
    3. ``response.content``   — preamble text solicited by the system prompt
    """

    type: str = "tool_use_thought"
    name: str = "tool_use_thought"
    context_manager: ToolUseWithThoughtContextManager = ToolUseWithThoughtContextManager()
    carries_thought: ClassVar[bool] = True
