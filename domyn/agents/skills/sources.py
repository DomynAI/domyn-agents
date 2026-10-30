"""Skill sources: where skills come from.

The interface is deliberately split into :meth:`SkillSource.index` and
:meth:`SkillSource.fetch` rather than a single ``discover() -> list[Skill]``.
That split is what makes three things possible at once:

* progressive disclosure — the index is cheap because bodies are never loaded;
* remote sources that filter per user, or return only *part* of a skill;
* a local source that reads frontmatter only, staying fast with many skills.

One source is active per agent. Combining several is not a special case in the
interface: a composite source is itself a ``SkillSource``, so it can be added
later without touching anything here.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from domyn.agents.logger import get_logger
from domyn.agents.skills.external_files import resolve_external_files
from domyn.agents.skills.layout import listed_path_is_linked, own_skill_file, walk_skill_tree
from domyn.agents.skills.models import (
    LoadedSkills,
    SkillContent,
    SkillDescriptor,
    SkillIndex,
    SkillQuery,
    SkillRef,
)
from domyn.agents.skills.parser import extract_section, parse_skill_file, parse_skill_text


class SkillNotFoundError(LookupError):
    """A skill was requested that the source cannot provide."""


class SkillSource(BaseModel, ABC):
    """Base class for skill sources.

    Subclasses self-register by their ``type`` tag, matching the pattern already
    used by memory managers and tools, so a source can be resolved from
    configuration by name.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _registry: ClassVar[dict[str, type[SkillSource]]] = {}

    type: str

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        # Pydantic calls this *after* the model is built, so `model_fields` is
        # populated. Reading the default there works regardless of whether the
        # annotation is a plain str, a Literal, or a string (PEP 563).
        super().__pydantic_init_subclass__(**kwargs)
        field = cls.model_fields.get("type")
        tag = getattr(field, "default", None)
        if isinstance(tag, str) and tag:
            SkillSource._registry[tag] = cls

    @abstractmethod
    async def index(self, query: SkillQuery) -> SkillIndex:
        """Return the L0 index. Must be cheap: no bodies are loaded."""

    @abstractmethod
    async def fetch(self, ref: SkillRef, query: SkillQuery) -> SkillContent:
        """Return the full body (L1) or one section (L2)."""

    async def load(self, query: SkillQuery) -> LoadedSkills:
        """Everything this source can hand over up front. Index only, by default.

        This is where a source does its I/O: every source in this package except
        the ``RemoteSkillSource`` placeholder implements it and derives ``index``
        and ``fetch`` from it, so there is one
        answer to "what did this cost" rather than one per method.

        The default exists for a source that cannot do better — an index-only
        third-party one — and for those it is correct rather than merely
        tolerated: it invents no bodies, and expansion pays ``fetch`` for each one
        actually needed.
        """
        return LoadedSkills(index=await self.index(query))

    async def search(
        self, text: str, limit: int, query: SkillQuery
    ) -> list[SkillDescriptor] | None:
        """The skills best matching ``text``, best first; ``None`` to leave it to the runtime.

        For a source that can search better where its skills live — a remote
        catalogue with its own index — than a model can over a catalogue handed to
        it. Every in-tree source returns ``None``: the runtime then searches the
        catalogue it loaded (:class:`~domyn.agents.skills.search.SkillSearch`).
        Up to ``limit`` hits; one more says the query can be narrowed.
        """
        return None

    def describe(self) -> str:
        return self.type


def evaluate_requirements(
    descriptor: SkillDescriptor, *, available_tools: list[str] | None = None
) -> SkillDescriptor:
    """Mark a descriptor unavailable when its requirements are unmet.

    The skill is never removed: an unavailable skill that stays visible with a
    reason is debuggable, whereas one that silently disappears is not.

    ``external_files`` are checked only for a skill with a ``path``: they are relative
    to its ``SKILL.md``, and a skill held in memory or served remotely has none.
    """
    requires = descriptor.requires
    if requires.is_empty():
        return descriptor

    missing: list[str] = []

    for binary in requires.bins:
        if shutil.which(binary) is None:
            missing.append(f"binary '{binary}'")
    for variable in requires.env:
        if not os.environ.get(variable):
            missing.append(f"env '{variable}'")
    if requires.os and sys.platform not in requires.os:
        missing.append(f"platform (needs {', '.join(requires.os)})")
    if requires.tools:
        present = set(available_tools or [])
        for tool in requires.tools:
            if tool not in present:
                missing.append(f"tool '{tool}'")
    if descriptor.path is not None:
        # As written, relative to the skill file: the reason reaches the model and the
        # user, and an absolute path would carry the home folder with it.
        for item in resolve_external_files(descriptor.path, requires.external_files):
            if item.problem:
                missing.append(f"file '{item.entry}' ({item.problem})")

    if not missing:
        return descriptor

    return descriptor.model_copy(
        update={"available": False, "unavailable_reason": ", ".join(missing)}
    )


class InMemorySkillSource(SkillSource):
    """Skills provided programmatically.

    This is what keeps the standalone-SDK mode honest: skills usable with no
    filesystem, no server and no configuration file.
    """

    type: str = "memory"
    skills: dict[str, str] = {}

    async def load(self, query: SkillQuery) -> LoadedSkills:
        descriptors: list[SkillDescriptor] = []
        bodies: dict[str, SkillContent] = {}
        for name, text in self.skills.items():
            descriptor, content = parse_skill_text(text, fallback_name=name, source=self.type)
            descriptors.append(
                evaluate_requirements(descriptor, available_tools=query.available_tools)
            )
            # Filed under the *parsed* name, not the mapping key: the two differ
            # whenever frontmatter declares a `name:`, and the index reports the
            # parsed one. Keying on the mapping key would make them disagree.
            # setdefault, so two entries parsing to one name resolve to the first
            # of them — which is what scanning in order used to do.
            bodies.setdefault(descriptor.name, content)
        return LoadedSkills(index=SkillIndex(skills=descriptors), bodies=bodies)

    async def index(self, query: SkillQuery) -> SkillIndex:
        return (await self.load(query)).index

    async def fetch(self, ref: SkillRef, query: SkillQuery) -> SkillContent:
        content = (await self.load(query)).body(ref.name)
        if content is None:
            raise SkillNotFoundError(ref.name)
        return narrow_to_section(content, ref)


class LocalDirSkillSource(SkillSource):
    """Skills read from one or more directories on disk.

    Supports both layouts found in the wild: ``<dir>/<skill>/SKILL.md`` and a flat
    ``<dir>/<skill>.md``. When the same name appears more than once, the earlier
    directory in ``paths`` wins, so callers express precedence by ordering rather
    than by a separate rule.
    """

    type: str = "local_dir"
    paths: list[Path] = []

    async def load(self, query: SkillQuery) -> LoadedSkills:
        """One pass over every configured directory: descriptors and bodies.

        ``parse_skill_file`` returns both halves already. Discarding the body here
        only to re-walk every directory and re-parse every file for it in
        ``fetch`` cost a full walk per expanded skill; keeping it costs nothing.

        Both blocking halves — the walk and the per-file parse — run in threads.
        This is called once per turn for every agent in the tree at once, so a
        directory read left on the event loop stalls every other agent's load with
        it, and any in-flight remote source besides. The merge stays here: dict
        insertion order *is* the precedence rule, and ``gather`` returns results in
        submission order, never completion order.
        """
        logger = get_logger()

        per_directory: list[list[Path]] = await asyncio.gather(
            *(
                # ``~`` expanded here and not on ``paths``, so what the user wrote is
                # what ``describe()`` and ``/config`` show. Nothing else expands it:
                # the walk would look for a folder literally named ``~`` under the
                # cwd and find nothing.
                asyncio.to_thread(_iter_skill_files, directory.expanduser())
                for directory in self.paths
            )
        )
        # Flattened in ``self.paths`` order: precedence is expressed by ordering, so
        # the flattening is what keeps "earlier directory wins" true below.
        skill_files = [path for directory_files in per_directory for path in directory_files]

        parsed: list[tuple[SkillDescriptor, SkillContent] | BaseException] = await asyncio.gather(
            *(
                asyncio.to_thread(
                    _load_skill_file,
                    path,
                    source=self.type,
                    available_tools=query.available_tools,
                )
                for path in skill_files
            ),
            return_exceptions=True,
        )

        descriptors: dict[str, SkillDescriptor] = {}
        bodies: dict[str, SkillContent] = {}
        for path, result in zip(skill_files, parsed, strict=True):
            if isinstance(result, BaseException):
                if not isinstance(result, OSError | ValueError):
                    # ``return_exceptions=True`` hands back *every* failure as a
                    # value, a cancelled turn's CancelledError included. Only the
                    # two a malformed file raises may be swallowed; anything else
                    # has to keep leaving load() the way it did when the parse was
                    # inline, rather than being logged as a bad skill file.
                    raise result
                # One malformed skill must not blind the agent to the others.
                # This now covers fetch() too, which used to let a neighbour's
                # SkillParseError escape and take the whole payload with it.
                logger.warning(f"skipping unreadable skill at {path}: {type(result).__name__}")
                logger.debug(f"skill parse detail for {path}: {result}")
                continue
            descriptor, content = result
            if descriptor.name in descriptors:
                continue
            descriptors[descriptor.name] = descriptor
            bodies[descriptor.name] = content

        return LoadedSkills(index=SkillIndex(skills=list(descriptors.values())), bodies=bodies)

    async def index(self, query: SkillQuery) -> SkillIndex:
        return (await self.load(query)).index

    async def fetch(self, ref: SkillRef, query: SkillQuery) -> SkillContent:
        content = (await self.load(query)).body(ref.name)
        if content is None:
            raise SkillNotFoundError(ref.name)
        return narrow_to_section(content, ref)

    def describe(self) -> str:
        return f"{self.type}({', '.join(str(p) for p in self.paths)})"


def _load_skill_file(
    path: Path, *, source: str, available_tools: list[str] | None
) -> tuple[SkillDescriptor, SkillContent]:
    """Parse one skill file and resolve its requirements, in a single thread hop.

    Both halves in the same function because ``evaluate_requirements`` blocks too:
    ``shutil.which`` stats every entry on ``PATH``, and every ``external_files`` entry
    is resolved and stat'ed. Leaving it behind on the loop
    would put back a good part of what offloading the parse just removed, and
    costs a second hop per file to avoid.

    A duplicate name therefore has its requirements evaluated even though the merge
    is about to discard it. That is wasted ``which`` calls, not a behaviour change:
    the function returns a copy and reads nothing the winner touches.

    ``parse_skill_file`` is read from this module's globals on every call rather
    than bound anywhere, and ``path`` is passed positionally, because the loader
    test counts parses by monkeypatching that module attribute.
    """
    descriptor, content = parse_skill_file(path, source=source)
    return evaluate_requirements(descriptor, available_tools=available_tools), content


def _iter_skill_files(directory: Path) -> list[Path]:
    """Skill files inside a directory, in a stable order.

    A path that is itself a ``.md`` file is a single (loose) skill. A directory holding
    its own ``SKILL.md`` is a single skill and yields just that file; otherwise every
    skill under it, group folders included (`skills.layout.walk_skill_tree`, the walk
    ``domyn setup`` offers from, so the two agree on what a folder holds).

    A listed path that is a symlink yields nothing, and neither does a symlink inside
    it: ``./.domyn/skills`` and ``./.agents/skills`` are read from whatever folder
    ``domyn run`` starts in, and a link there must not decide what is loaded. For a
    relative path a linked folder on the way to it counts too (``.agents -> ~``); above
    an absolute path only the last component counts, so a symlinked parent (``/tmp`` on
    macOS) is fine (`listed_path_is_linked`).

    Sorted so the index is deterministic across runs; an index that reorders
    itself makes prompt prefixes unstable for no reason.
    """
    if listed_path_is_linked(directory):
        return []
    # A listed path that is one loose skill file (`domyn setup` used to list those singly).
    if directory.is_file():
        return [directory] if directory.suffix.lower() == ".md" else []
    # A directory that is itself a skill: what a config lists when it names one
    # skill rather than a folder of them (the layout `domyn setup` installs).
    own = own_skill_file(directory) if directory.is_dir() else None
    if own is not None:
        return [own]
    return [found.skill_file for found in walk_skill_tree(directory)]


def narrow_to_section(content: SkillContent, ref: SkillRef) -> SkillContent:
    """Narrow content to a section when one was requested.

    Shared with the runtime's expansion, which serves a section request straight
    from a preloaded body — the same slice, without a second read.
    """
    if ref.section is None:
        return content
    section_body = extract_section(content.body, ref.section)
    if section_body is None:
        raise SkillNotFoundError(f"{ref.name}#{ref.section}")
    return SkillContent(
        name=content.name,
        body=section_body,
        version=content.version,
        section=ref.section,
        partial=True,
        path=content.path,
    )
