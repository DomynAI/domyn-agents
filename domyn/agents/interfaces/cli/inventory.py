"""Which skills and standing instructions the session's agent has, right now.

The startup line under the banner names them once, and it is scrollback: it cannot
be redrawn. Both are read from disk on every turn, though, so a skill added to
``./.domyn/skills`` or an ``AGENTS.md`` created — by the user, or by the agent itself —
changes what the model sees without changing that line. The REPL takes an inventory when
the session starts and at the end of every turn, and says what changed, if anything
(`describe_change`). It counts what the startup line counts: every skill in the
index, available or not, and the instruction files by name.
"""

from dataclasses import dataclass
from typing import Any

from domyn.agents import instructions
from domyn.agents.logger import get_logger
from domyn.agents.skills import SkillQuery, SkillsRuntime


@dataclass(frozen=True)
class Inventory:
    skills: tuple[str, ...]
    """Every skill in the index, available or not, by name."""
    instructions: tuple[str, ...]
    """The instruction files loaded, by name, in the order they are merged."""


async def take_inventory(agent: Any) -> Inventory:
    """Read the agent's skills and instructions as the next turn would. Never raises:
    a source that cannot be read counts as empty, as it does for the turn."""
    skills: tuple[str, ...] = ()
    runtime = getattr(agent, "skills", None)
    if isinstance(runtime, SkillsRuntime):
        try:
            index = await runtime.index(SkillQuery(agent_name=agent.name))
            skills = tuple(skill.name for skill in index.skills)
        except Exception as exc:
            get_logger().warning(f"skills: cannot read them for display: {type(exc).__name__}")
            get_logger().debug(f"skills display load detail: {exc}")
    loaded = await instructions.load_declared(agent)
    return Inventory(skills=skills, instructions=loaded.names if loaded else ())


def _plural(n: int) -> str:
    return f"{n} skill{'s' if n != 1 else ''}"


def describe_change(before: Inventory, after: Inventory) -> str | None:
    """One line saying what changed between two inventories, or ``None``.

    Names what was added and removed rather than only the new count: "9 skills" after
    "9 skills" can hide one skill replaced by another. A skill whose file changed but
    whose name did not is not a change here; its new text is what the next turn reads.
    """
    parts: list[str] = []
    if set(before.skills) != set(after.skills):
        added = sorted(set(after.skills) - set(before.skills))
        removed = sorted(set(before.skills) - set(after.skills))
        detail = [f"+{name}" for name in added] + [f"-{name}" for name in removed]
        parts.append(f"{_plural(len(after.skills))} ({', '.join(detail)})")
    if before.instructions != after.instructions:
        parts.append(f"instructions: {', '.join(after.instructions) or 'none'}")
    return f"  Now: {'  ·  '.join(parts)}" if parts else None
