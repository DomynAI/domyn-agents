"""Instruction sources: where an agent's standing instructions come from.

Same shape as ``skills/sources.py``: a source does the I/O and self-registers by
its ``type`` tag, so one can be named from YAML; the runtime around it owns the
budget. A new kind of source — a remote one serving a different soul per user —
is a subclass, not a change here.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, field_validator

from domyn.agents.instructions.discovery import (
    AGENTS_FILENAMES,
    AUTODISCOVERY_DIRS,
    SOUL_FILENAME,
    STATE_DIRS,
    find_instruction_files,
    is_regular_file,
)
from domyn.agents.instructions.models import AgentInstructions, Instruction, InstructionKind
from domyn.agents.logger import get_logger
from domyn.agents.skills.layout import listed_path_is_linked


class InstructionsSource(BaseModel, ABC):
    """Base class for instruction sources."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _registry: ClassVar[dict[str, type[InstructionsSource]]] = {}

    type: str

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        # After the model is built, so `model_fields` is populated — see
        # SkillSource.__pydantic_init_subclass__ for why it is read here.
        super().__pydantic_init_subclass__(**kwargs)
        field = cls.model_fields.get("type")
        tag = getattr(field, "default", None)
        if isinstance(tag, str) and tag:
            InstructionsSource._registry[tag] = cls

    @abstractmethod
    async def load(self) -> AgentInstructions:
        """Read the instructions. Called by the runner once per turn.

        Never raises for a missing instruction: absent is the normal case. A
        source that cannot be reached may raise; the runner degrades that to no
        instructions for the turn rather than taking it down.
        """

    def describe(self) -> str:
        return self.type


class InMemoryInstructionsSource(InstructionsSource):
    """Instructions provided programmatically: no filesystem involved."""

    type: str = "memory"
    soul: str = ""
    agent: str = ""
    task: str = ""

    @field_validator("soul", "agent", "task", mode="before")
    @classmethod
    def _strip(cls, value: str | None) -> str:
        """Stripped once, at construction. A blank value is no instruction: its
        header would tell the model there are instructions and show it none."""
        return (value or "").strip()

    async def load(self) -> AgentInstructions:
        return AgentInstructions.from_parts(
            Instruction(text=text, instruction_kind=kind)
            for kind in InstructionKind
            if (text := getattr(self, kind.value))
        )


def _read(path: Path) -> str | None:
    """Stripped file contents, or ``None`` when it could not be read or is empty."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as exc:
        get_logger().warning(
            f"instructions: could not read {path}, ignoring it: {type(exc).__name__}"
        )
        get_logger().debug(f"instructions read detail: {exc}")
        return None
    if not text:
        # A file someone created and has not written yet. Rendering its header
        # would tell the model there are instructions and then show it none.
        get_logger().debug(f"instructions: {path} is empty, skipping")
        return None
    return text


class LocalDirInstructionsSource(InstructionsSource):
    """``SOUL.md`` / ``AGENTS.md`` read from disk.

    Each entry of ``paths`` is a **directory**, in which the known names are
    looked for, or a **file**, taken as given whatever it is called (a name other
    than the two known ones makes it a ``TASK``). Every file found is included,
    once even when reached twice, **except souls**: an agent has one identity, so
    the first ``SOUL.md`` in path order is the soul and any later one is recorded
    in ``superseded`` and not read. They are ordered by kind and never by the
    order the paths were given, so the soul still comes first whichever way round
    two paths were typed; within a kind, path order is kept.

    A directory contributes only the files directly in it, except the two
    autodiscovery entries ``.domyn`` and ``.agents`` (`AUTODISCOVERY_DIRS`, as
    written): those are walked on every load, breadth first, with domyn's own
    state folders skipped (`find_instruction_files`, `STATE_DIRS`), so their
    souls come shallowest first and their ``AGENTS.md`` files from the most
    general to the most specific.

    A relative entry is resolved against the working directory on every load:
    that is how ``domyn setup`` lists ``./.domyn`` and ``./.agents``. Nothing walks
    up to a git root. No link is followed — the entry itself, every folder on the
    way to a relative one, and the files in a directory — since a relative entry
    is chosen by whatever project the CLI starts in, and ``AGENTS.md -> ~/.ssh/id_rsa``
    would send that file to the model on every turn.
    """

    type: str = "local_dir"
    paths: list[Path] = []

    soul_filename: ClassVar[str] = SOUL_FILENAME
    agents_filenames: ClassVar[tuple[str, ...]] = AGENTS_FILENAMES

    async def load(self) -> AgentInstructions:
        # Off the event loop: the runner loads every agent's sources at once.
        return await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> AgentInstructions:
        found: list[Instruction] = []
        seen: set[Path] = set()
        soul: Path | None = None
        superseded: list[str] = []
        for source in self.paths:
            for path in self._candidates(source.expanduser()):
                if path in seen:
                    continue
                seen.add(path)
                kind = self._kind_of(path)
                if kind is InstructionKind.SOUL and soul is not None:
                    get_logger().debug(f"instructions: {path} not used, {soul} is the soul")
                    superseded.append(str(path))
                    continue
                text = _read(path)
                if text is None:
                    # An empty SOUL.md is no soul: the next one found takes its place.
                    continue
                if kind is InstructionKind.SOUL:
                    soul = path
                found.append(Instruction(text=text, instruction_kind=kind, path=str(path)))
        return AgentInstructions.from_parts(found, superseded=tuple(superseded))

    def _kind_of(self, path: Path) -> InstructionKind:
        """By name: ``SOUL.md`` is the soul, ``AGENTS.md`` / ``AGENT.md`` the agent's
        brief, and any other file — one named on the command line, since a
        directory only ever contributes the known names — a task."""
        name = path.name.upper()
        if name == self.soul_filename.upper():
            return InstructionKind.SOUL
        if name in {one.upper() for one in self.agents_filenames}:
            return InstructionKind.AGENT
        return InstructionKind.TASK

    def _candidates(self, source: Path) -> list[Path]:
        """The files one configured path contributes, without reading them."""
        if listed_path_is_linked(source):
            get_logger().warning(f"instructions: {source} goes through a link, not followed")
            return []
        if source.is_file():
            return [source.resolve()]
        if not source.is_dir():
            get_logger().debug(f"instructions: nothing at {source}")
            return []
        if source in AUTODISCOVERY_DIRS:
            walked = find_instruction_files(source, skip=STATE_DIRS)
            # Every soul, so the first-soul rule in `_load_sync` records the others.
            souls = [*([walked.soul] if walked.soul else []), *walked.superseded]
            return [one.resolve() for one in [*souls, *walked.agents]]
        found: list[Path] = []
        soul = source / self.soul_filename
        if is_regular_file(soul):
            found.append(soul.resolve())
        for candidate in self.agents_filenames:
            agents = source / candidate
            if is_regular_file(agents):
                found.append(agents.resolve())
                break
        for name in (self.soul_filename, *self.agents_filenames):
            if (source / name).is_symlink():
                get_logger().warning(f"instructions: {source / name} is a link, not followed")
        return found

    def describe(self) -> str:
        return f"{self.type}({', '.join(str(path) for path in self.paths)})"
