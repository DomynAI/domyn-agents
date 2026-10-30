"""What a load of standing instructions produces.

Kept apart from the sources and the runtime so both can import it without
importing each other — the same split as ``skills/models.py``.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from pathlib import PurePath
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


class InstructionKind(Enum):
    """What an instruction is, which is also where it sorts.

    ``SOUL`` is who the agent is (``SOUL.md``), ``AGENT`` what it is working on
    (``AGENTS.md``), and ``TASK`` a one-off brief handed over as a file of any
    other name (``brief.md`` listed in ``standing_instructions.source.paths``).

    **The declaration order is the precedence**: identity first, the task last,
    so the instructions closest to the task land where a model weighs them most.
    Reordering these members reorders the prompt. Each value is also the name of
    its slot on :class:`AgentInstructions`.
    """

    SOUL = "soul"
    AGENT = "agent"
    TASK = "task"


class Instruction(BaseModel):
    """One standing instruction, wherever it came from."""

    model_config = ConfigDict(frozen=True)

    text: str
    instruction_kind: InstructionKind
    path: str | None = None
    """Where it was read from. ``None`` for an instruction with no file behind it
    (an in-memory source); a file source always sets it, because the path in the
    rendered text is the only route from the model back to the file to edit."""
    truncated: bool = False

    @property
    def name(self) -> str:
        """The file's name, or the kind for an instruction with no file behind it."""
        return PurePath(self.path).name if self.path else self.instruction_kind.value

    def render(self) -> str:
        """The block handed to the model, carrying its own provenance.

        The absolute path is in the text on purpose: an agent asked "why do you
        keep doing that" can answer with the file to edit, and the file tools
        cannot open ``~/.domyn``.
        """
        header = f"[{self.name} — {self.path}]" if self.path else f"[{self.name}]"
        if self.truncated:
            header += " (truncated: instruction budget exceeded)"
        return f"{header}\n{self.text}"


class AgentInstructions(BaseModel):
    """Everything one load found for one agent, plus what had to be left out.

    One slot per :class:`InstructionKind`, named after its value. Each holds any
    number of instructions — one per file found, so several declared paths
    contribute one each — in the order they were found. Any may be empty: a
    missing ``SOUL.md`` or ``AGENTS.md`` is the normal case, not an error. An empty value is falsy, so "switched off" and "found nothing"
    read the same to every caller.
    """

    model_config = ConfigDict(frozen=True)

    soul: tuple[Instruction, ...] = ()
    agent: tuple[Instruction, ...] = ()
    task: tuple[Instruction, ...] = ()
    dropped: tuple[str, ...] = ()
    skipped_sections: tuple[str, ...] = ()
    """Headings of conditional ``##`` sections a decision model judged not to
    apply this turn (``instructions/conditional.py``). Separate from ``dropped``
    because the two are different claims: that one is "it did not fit", this one
    is "it did not apply". Both are surfaced rather than silent — an instruction
    that vanished with no way to find out is indistinguishable from one the
    model chose to ignore."""
    """Instructions found but not included at all, because the budget was already
    spent. Recorded rather than forgotten: a standing instruction that vanished
    without a trace is a rule the agent stops following that nobody can see was
    removed."""
    superseded: tuple[str, ...] = ()
    """``SOUL.md`` files found after the first one, and not read: an agent has one
    identity, so the first in path order is the soul. Recorded for the same reason
    as ``dropped``, but not a cut: nothing was lost to the budget."""

    @model_validator(mode="after")
    def validate_instructions(self) -> AgentInstructions:
        for kind in InstructionKind:
            if any(one.instruction_kind != kind for one in getattr(self, kind.value)):
                raise ValueError(
                    f"{kind.value} instructions must have instruction_kind={kind.name}"
                )
        return self

    @classmethod
    def from_parts(
        cls,
        parts: Iterable[Instruction],
        dropped: tuple[str, ...] = (),
        superseded: tuple[str, ...] = (),
    ) -> AgentInstructions:
        """Sort ``parts`` into their slots by kind, keeping their order within one."""
        grouped: dict[str, list[Instruction]] = {kind.value: [] for kind in InstructionKind}
        for one in parts:
            grouped[one.instruction_kind.value].append(one)
        # `Any` because the splat is checked against every field it could reach,
        # and that now includes `skipped_sections`, a tuple of `str`. The keys
        # are kind values, so it only ever reaches the per-kind slots.
        slots: dict[str, Any] = {slot: tuple(found) for slot, found in grouped.items()}
        return cls(**slots, dropped=dropped, superseded=superseded)

    def __bool__(self) -> bool:
        return bool(self.parts)

    @property
    def parts(self) -> tuple[Instruction, ...]:
        """Every instruction, in precedence order: by kind, then as found."""
        return tuple(one for kind in InstructionKind for one in getattr(self, kind.value))

    @property
    def truncated(self) -> bool:
        return bool(self.dropped) or any(one.truncated for one in self.parts)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(one.name for one in self.parts)

    @property
    def chars(self) -> int:
        return sum(len(one.text) for one in self.parts)

    @property
    def text(self) -> str:
        """The whole rendered block, or ``""`` when nothing was found."""
        return "\n\n".join(one.render() for one in self.parts)
