"""Instruction sources: where an agent's standing instructions come from.

Same shape as ``skills/sources.py``: a source does the I/O and self-registers by
its ``type`` tag, so one can be named from YAML; the runtime around it owns the
budget. A new kind of source — a remote one serving a different soul per user —
is a subclass, not a change here.
"""

from __future__ import annotations

import asyncio
import posixpath
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from domyn.agents.instructions.discovery import (
    AGENTS_FILENAMES,
    SOUL_FILENAME,
    find_instruction_files,
    is_regular_file,
)
from domyn.agents.instructions.models import AgentInstructions, Instruction, InstructionKind
from domyn.agents.logger import get_logger
from domyn.agents.resources.base import Resource, ResourceError, relative_path
from domyn.agents.skills.layout import listed_path_is_linked
from domyn.agents.skills.sources import ResourceRef, bind_resources, resource_refs


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

    A directory contributes only the files directly in it, except one also listed in
    ``walk``: that one is walked on every load, breadth first
    (`find_instruction_files`), so its souls come shallowest first and its
    ``AGENTS.md`` files from the most general to the most specific. What a project's ``.domyn`` and ``.agents`` need; said, never inferred
    from how a path is spelled, so a copy of the folder elsewhere is walked the same.

    A relative entry is resolved against the working directory on every load:
    that is how ``domyn setup`` lists ``./.domyn`` and ``./.agents``. Nothing walks
    up to a git root. No link is followed — the entry itself, every folder on the
    way to a relative one, and the files in a directory — since a relative entry
    is chosen by whatever project the CLI starts in, and ``AGENTS.md -> ~/.ssh/id_rsa``
    would send that file to the model on every turn.
    """

    type: str = "local_dir"
    paths: list[Path] = []
    walk: list[Path] = []
    """Entries of ``paths`` walked with their subfolders, rather than read flat."""

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
        if source in {one.expanduser() for one in self.walk}:
            walked = find_instruction_files(source)
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


class ResourceInstructionsSource(InstructionsSource):
    """``SOUL.md`` and ``AGENTS.md`` read from declared resources, under ``path`` of
    every root, in order.

    Bound when the configuration loads (:meth:`bind`), like the skills source of the
    same type. A resource on this machine is read exactly as a ``local_dir`` source
    reads its paths (a root that is a file is taken as given, a folder read flat
    unless its entry says ``walk: true``); any other through its file-store view.
    Across resources the rule is the one within a ``local_dir`` source: the first
    soul is the soul, every ``AGENTS.md`` is read.
    """

    type: str = "resource"
    resource: list[ResourceRef] = Field(
        description="Resource ids, or {id, path, walk} entries, in order (earlier wins)"
    )
    path: str = Field(
        default=".", description="The folder read under every root of an entry naming none"
    )

    _bound: list[Resource] = PrivateAttr(default_factory=list)

    @field_validator("resource", mode="before")
    @classmethod
    def _refs(cls, value: Any) -> list[ResourceRef]:
        return resource_refs(value)

    @field_validator("path")
    @classmethod
    def _relative(cls, value: str) -> str:
        return relative_path(value) or "."

    @property
    def ids(self) -> list[str]:
        return [ref.id for ref in self.resource]

    def bind(self, declared: Mapping[str, Resource]) -> ResourceInstructionsSource:
        self._bound = bind_resources(self.ids, declared, "the standing instructions source")
        return self

    @property
    def bound(self) -> list[Resource]:
        return list(self._bound)

    def _reads(self) -> list[tuple[Resource, str, bool]]:
        return [
            (one, ref.subpath(self.path), ref.walk)
            for one, ref in zip(self._bound, self.resource, strict=True)
        ]

    def local_paths(self) -> list[Path] | None:
        """The paths read, when every bound resource is on this machine."""
        paths: list[Path] = []
        for one, subpath, _ in self._reads():
            roots = one.local_roots()
            if roots is None:
                return None
            paths.extend(root / subpath for root in roots)
        return paths

    def walked_paths(self) -> list[Path]:
        """Those of `local_paths` read with their subfolders."""
        return [
            root / subpath
            for one, subpath, walk in self._reads()
            if walk
            for root in one.local_roots() or ()
        ]

    async def load(self) -> AgentInstructions:
        if not self._bound:
            raise ResourceError(
                f"The instructions source names resource {', '.join(self.ids)} but was "
                "never bound to it: load the configuration through ConfigurationV1, or call "
                "bind()."
            )
        local = self.local_paths()
        if local is not None:
            # One source over every path, so the first-soul rule and the de-duplication
            # of a file reached twice are the local source's own.
            return await LocalDirInstructionsSource(paths=local, walk=self.walked_paths()).load()
        parts: list[Instruction] = []
        superseded: list[str] = []
        soul: str | None = None
        for one, subpath, walk in self._reads():
            roots = one.local_roots()
            if roots is not None:
                under = [root / subpath for root in roots]
                loaded = await LocalDirInstructionsSource(
                    paths=under, walk=under if walk else []
                ).load()
                found = list(loaded.parts)
                superseded.extend(loaded.superseded)
            else:
                found = await self._load_remote(one, subpath, walk)
            for instruction in found:
                if instruction.instruction_kind is InstructionKind.SOUL:
                    if soul is not None:
                        superseded.append(instruction.path or instruction.name)
                        continue
                    soul = instruction.path or instruction.name
                parts.append(instruction)
        return AgentInstructions.from_parts(parts, superseded=tuple(superseded))

    async def _load_remote(self, resource: Resource, subpath: str, walk: bool) -> list[Instruction]:
        top = "" if subpath == "." else subpath
        folders = await _remote_folders(resource, top) if walk else [top]
        found: list[Instruction] = []
        for folder in folders:
            found.extend(await self._read_remote_folder(resource, folder))
        return found

    async def _read_remote_folder(self, resource: Resource, folder: str) -> list[Instruction]:
        found: list[Instruction] = []
        names = [
            (LocalDirInstructionsSource.soul_filename, InstructionKind.SOUL),
            *((name, InstructionKind.AGENT) for name in AGENTS_FILENAMES),
        ]
        has_agents = False
        for name, kind in names:
            if kind is InstructionKind.AGENT and has_agents:
                break
            path = posixpath.join(folder, name) if folder else name
            try:
                text = (await resource.read_text(path)).strip()
            except FileNotFoundError:
                continue
            except Exception as exc:  # a remote store fails in its own words
                get_logger().warning(
                    f"instructions: could not read {path} in resource {resource.id!r}: "
                    f"{type(exc).__name__}"
                )
                get_logger().debug(f"instructions read detail: {exc}")
                continue
            if not text:
                continue
            has_agents = has_agents or kind is InstructionKind.AGENT
            found.append(
                Instruction(text=text, instruction_kind=kind, path=f"{resource.id}:{path}")
            )
        return found

    def describe(self) -> str:
        shown = []
        for ref in self.resource:
            subpath = ref.subpath(self.path)
            name = ref.id if subpath == "." else f"{ref.id}/{subpath}"
            shown.append(f"{name}, walked" if ref.walk else name)
        return f"{self.type}({', '.join(shown)})"


# A walk of a resource that is not on this machine lists one folder per request, so
# it is bounded like the skills walk.
REMOTE_INSTRUCTION_DEPTH = 4


async def _remote_folders(resource: Resource, top: str) -> list[str]:
    """``top`` and the folders below it, breadth first and sorted by name within a
    level, hidden ones skipped."""
    found = [top]
    level = [top]
    for _ in range(REMOTE_INSTRUCTION_DEPTH):
        below: list[str] = []
        for folder in level:
            for entry in await resource.list(folder):
                name = posixpath.basename(entry.path)
                if not entry.is_dir or name.startswith("."):
                    continue
                below.append(entry.path)
        level = sorted(below)
        found.extend(level)
    return found
