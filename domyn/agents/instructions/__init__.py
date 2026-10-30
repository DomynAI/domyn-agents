"""Standing instructions — ``SOUL.md`` and ``AGENTS.md``.

Two instructions, and the distinction between them is the whole design rather
than a naming preference:

``SOUL.md``
    Who the agent is: tone, standing preferences, what it refuses to do. Follows
    the *user*, not the project. An agent has one: the first ``SOUL.md`` in path
    order, so the user's own wins over one a project ships in ``./.domyn`` or
    ``./.agents`` (read only when ``domyn setup`` turned that on).

``AGENTS.md``
    What the agent is working on. Every one found is included, in path order:
    the user's first, the project's last, where a model weighs them most.

They work like skills. An agent declares an :class:`InstructionRuntime`
(``Agent.standing_instructions``) around an :class:`InstructionsSource` — files
on disk, or text held in memory — and the runner loads it **once per turn**
(``Runner._load_turn_instructions``), so an edited file is picked up on the next
message and a ``/clear`` needs nothing special. Loading never takes a turn down:
a missing file is the normal case, an unreadable one is a warning, an oversized
one is truncated, and a source that fails gives that turn no instructions.

What was loaded is merged into the agent's instruction **at render time**
(:meth:`InstructionRuntime.merge`) and reaches the *system prompt* —
not the injected-message seam skills and memory use — for two reasons:

* Compaction never touches the system prompt, so standing instructions cannot
  be summarized away mid-session.
* The content is constant between edits, so it belongs in the part of the prompt
  that providers can cache.

``Agent.instruction`` itself is never rewritten, which is what makes
:func:`attach_instructions` idempotent.
"""

from collections.abc import Sequence
from typing import Any

from domyn.agents.instructions.models import (
    AgentInstructions,
    Instruction,
    InstructionKind,
)
from domyn.agents.instructions.runtime import MAX_INSTRUCTION_BYTES, InstructionRuntime
from domyn.agents.instructions.sources import (
    InMemoryInstructionsSource,
    InstructionsSource,
    LocalDirInstructionsSource,
)
from domyn.agents.logger import get_logger


def attach_instructions(agent: Any, runtime: InstructionRuntime) -> Any:
    """Give ``agent`` standing instructions. Pure wiring, no I/O.

    Idempotent: it replaces whatever runtime was there, and the instruction text
    is merged at render time rather than appended here, so calling it twice is
    the same as calling it once.

    Only the agent it is given, never its sub-agents: a standing preamble
    repeated down a tree is paid again at every handoff. An agent that declares
    no ``standing_instructions`` field (``WrappedAgent``, ``DelegateAgent``,
    anything behind an adapter) is returned untouched — its prompt is not ours.
    """
    if hasattr(agent, "standing_instructions"):
        agent.standing_instructions = runtime
    return agent


async def load_declared(agent: Any) -> AgentInstructions | None:
    """What ``agent``'s runtime would load right now, for display only.

    ``None`` when the agent declares no runtime; empty when it cannot be read.
    The runner loads again at the top of every turn, and that is the load the
    model sees — this one answers "which files are in play" for a status line or
    ``/config``, so it must never raise.
    """
    runtime = getattr(agent, "standing_instructions", None)
    if not isinstance(runtime, InstructionRuntime):
        return None
    try:
        return await runtime.load()
    except Exception as exc:
        get_logger().warning(f"instructions: cannot read them for display: {type(exc).__name__}")
        get_logger().debug(f"instructions display load detail: {exc}")
        return AgentInstructions()


__all__: Sequence[str] = (
    "MAX_INSTRUCTION_BYTES",
    "AgentInstructions",
    "InMemoryInstructionsSource",
    "Instruction",
    "InstructionKind",
    "InstructionRuntime",
    "InstructionsSource",
    "LocalDirInstructionsSource",
    "attach_instructions",
    "load_declared",
)
