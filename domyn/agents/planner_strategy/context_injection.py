"""Shared per-turn context injection for every planner strategy.

Memory recall used to be copy-pasted, identically, into all three planners
(``react``, ``tool_use``, ``tool_use_thought``). Every new kind of injected
context would have become a fourth, fifth and sixth copy — and any planner whose
copy was forgotten would silently lose the feature. One helper, called from each
planner, means adding a source of context is a single edit and no planner is left
behind.

Injected material is added as a **message**, not folded into the system prompt,
because the system prompt is rebuilt on every planner iteration and mutating it
makes the prompt prefix unstable within a single turn.

Standing instructions (``SOUL.md`` / ``AGENTS.md``, see
``domyn.agents.instructions``) deliberately do *not* come through here, although
the runner loads them per turn just as it loads skills: the agent merges them into
its instruction and the planner passes that to ``get_context`` as
``instruction=``, so they reach the system prompt.

The tool-use planners build these messages *before* deciding whether to compact
(:func:`turn_context_messages`) and insert them after
(:func:`insert_turn_context`): they are part of the request, so the threshold
counts them, and they are not history, so no compaction touches them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from domyn.agents.llm import LLMMessage
from domyn.agents.logger import get_logger

if TYPE_CHECKING:
    from domyn.agents.core import BaseAgent, InteractionContext

# Position 1: right after the system message, before the conversation. Keeps the
# system prefix intact (stable for provider-side prefix caching) while placing the
# context ahead of the history it is meant to inform.
_INJECT_AT = 1


async def turn_context_messages(
    agent: BaseAgent,
    context: InteractionContext,
    skills_payload: str = "",
) -> list[LLMMessage]:
    """The messages this turn adds after the system prompt: skills, then memory.

    Empty with no memory manager and an empty ``skills_payload``, so agents that
    use neither get exactly the history they had.
    """
    injected: list[LLMMessage] = []
    if skills_payload:
        injected.append(LLMMessage(role="assistant", content=skills_payload))
        get_logger().debug(f"skills: injected {len(skills_payload)} chars of skill context")

    memory_manager = getattr(agent, "memory_manager", None)
    if memory_manager is not None:
        agent_input = getattr(context, "agent_input", None)
        question = ""
        if agent_input is not None and getattr(agent_input, "content", None):
            first = agent_input.content[0]
            question = first.text or ""

        recalled = await memory_manager.recall_memories(
            user_id=context.user_id,
            agent=agent,
            recency_limit=memory_manager.recency_limit,
            question=question,
            run_config=context.run_config,
        )
        if recalled:
            injected.append(
                LLMMessage(
                    role="assistant",
                    content="\n".join(memory.memory for memory in recalled),
                )
            )
    return injected


def insert_turn_context(messages: list[LLMMessage], injected: list[LLMMessage]) -> list[LLMMessage]:
    """Insert ``injected`` into ``messages`` in place, and return the same list."""
    messages[_INJECT_AT:_INJECT_AT] = injected
    return messages


async def inject_turn_context(
    messages: list[LLMMessage],
    agent: BaseAgent,
    context: InteractionContext,
    skills_payload: str = "",
) -> list[LLMMessage]:
    """Build this turn's context and insert it into ``messages``, in place.

    For a planner with nothing to measure in between; returns the same list.
    """
    return insert_turn_context(
        messages, await turn_context_messages(agent, context, skills_payload)
    )
