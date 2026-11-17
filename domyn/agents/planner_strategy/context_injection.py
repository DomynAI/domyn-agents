"""Shared per-turn context injection for every planner strategy.

Memory recall used to be copy-pasted, identically, into all three planners
(``react``, ``tool_use``, ``tool_use_thought``). Every new kind of injected
context would have become a fourth, fifth and sixth copy — and any planner whose
copy was forgotten would silently lose the feature. One helper, called from each
planner, means adding a source of context is a single edit and no planner is left
behind.

Memory is added as a **message**, not folded into the system prompt: the page
recalled is picked by the user's words, so it can differ from one turn to the next.

The skills block is the opposite case, and goes at the **end of the system prompt**
(:func:`with_skills`). It is resolved once per agent entry and passed unchanged to
every iteration, and with the default selector it is the index alone, which changes
only when a skill is added, removed or edited: as stable as the rest of the system
prompt, so the prefix a provider caches is unchanged. Delivered as an assistant
message it read as something the model had said itself, and models acted on it far
less than on their instructions.

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


def with_skills(messages: list[LLMMessage], skills_payload: str) -> list[LLMMessage]:
    """``messages`` with ``skills_payload`` at the end of the system prompt, in place.

    The system message is replaced, not mutated: it is a fresh object on every
    iteration today, but a context manager that cached it would otherwise get the
    block appended again on each call. With no system message, one is added.
    """
    if not skills_payload:
        return messages
    if messages and messages[0].role == "system":
        system = messages[0]
        content = system.content if isinstance(system.content, str) else ""
        joined = f"{content}\n\n{skills_payload}" if content else skills_payload
        messages[0] = system.model_copy(update={"content": joined})
    else:
        messages.insert(0, LLMMessage(role="system", content=skills_payload))
    get_logger().debug(f"skills: {len(skills_payload)} chars at the end of the system prompt")
    return messages


async def turn_context_messages(
    agent: BaseAgent,
    context: InteractionContext,
) -> list[LLMMessage]:
    """The messages this turn adds after the system prompt: the recalled memory.

    Empty with no memory manager, so agents without one get exactly the history
    they had.
    """
    injected: list[LLMMessage] = []
    memory_manager = getattr(agent, "memory_manager", None)
    if memory_manager is not None:
        block = await memory_manager.recall_block(agent, context)
        if block:
            injected.append(LLMMessage(role="assistant", content=block, part="memory"))
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
    """Put the skills in the system prompt and insert this turn's context, in place.

    For a planner with nothing to measure in between; returns the same list.
    """
    return insert_turn_context(
        with_skills(messages, skills_payload), await turn_context_messages(agent, context)
    )
