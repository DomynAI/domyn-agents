"""A resource: a place files live, declared once in a configuration and named by id.

A resource is a **declaration, not behaviour**. Loading a configuration does nothing
with one beyond checking it can be used: it says "this place exists, here is its id,
here is how to reach it". Whatever needs a place — a tool bundle, a skills or
instructions source, a platform's sandbox — names the resource by id, and that is
the only thing that makes it do anything.

Two views, because consumers need two different things:

* the **file-store view** (:meth:`Resource.list`, :meth:`~Resource.read`,
  :meth:`~Resource.write`, :meth:`~Resource.stat`), what skills, instructions and
  file tools need, which every type implements;
* the **mount view** (:meth:`Resource.materialize`), what something running a
  process needs: a local directory. Only a type that can give one implements it;
  an execution bundle bound to a resource without one is refused when the
  configuration loads.

Types self-register by their ``type`` tag, as skill and instruction sources do, so a
platform adds its own (``platform_fs``) by subclassing, with no change here.
"""

from __future__ import annotations

import builtins
import posixpath
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field


class ResourceError(ValueError):
    """A resource is declared, referenced or used wrongly.

    A ``ValueError`` so the loader's callers, which already report a bad
    configuration by catching one, report these too.
    """


@dataclass(frozen=True)
class ResourceEntry:
    """One file or folder of a resource, as its file-store view lists it."""

    path: str
    """Relative to the resource, ``/``-separated, never starting with ``/``."""
    is_dir: bool
    size: int | None = None


@dataclass(frozen=True)
class MountSpec:
    """Where a resource is on this machine, for something that runs a process in it."""

    paths: tuple[Path, ...]
    """Absolute folders, the primary one first."""
    read_only: bool = False
    extra: dict[str, Any] = field(default_factory=dict)
    """Whatever a platform type needs to describe its mount; empty for ``local_dir``."""


def relative_path(path: str) -> str:
    """``path`` as a resource-relative, ``/``-separated path, or a :class:`ResourceError`.

    The fence every type applies before it touches anything: absolute paths, a
    ``~`` and any ``..`` that climbs out are refused, so a path can only name
    something inside the resource. ``""`` and ``"."`` are the resource itself.
    """
    raw = str(path).replace("\\", "/")
    if raw.startswith(("/", "~")):
        raise ResourceError(f"'{path}' is not relative to the resource. Use a relative path.")
    normalized = posixpath.normpath(raw) if raw else "."
    if normalized == ".." or normalized.startswith("../"):
        raise ResourceError(f"'{path}' leads outside the resource.")
    return "" if normalized == "." else normalized


class Resource(BaseModel, ABC):
    """Base class of every resource type."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    _registry: ClassVar[dict[str, type[Resource]]] = {}

    id: str = Field(
        min_length=1,
        description=(
            "Unique within the configuration, and never derived from an agent's name: "
            "renaming an agent must not move its files."
        ),
    )
    type: str
    read_only: bool = False
    """Enforced by the resource, not by the bundles bound to it: its file-store view
    refuses to write, and its mount view says so (``MountSpec.read_only``) to
    whatever mounts it. Commands run in it are not refused: a ``local_dir`` cannot
    keep a process from writing to a folder, a type mounted read-only can."""

    mountable: ClassVar[bool] = False
    """Whether :meth:`materialize` gives a local directory. Read at load time, where
    an execution bundle bound to a resource without one is refused."""

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        # After the model is built, so `model_fields` is populated — see
        # SkillSource.__pydantic_init_subclass__ for why it is read here.
        super().__pydantic_init_subclass__(**kwargs)
        tag = getattr(cls.model_fields.get("type"), "default", None)
        if isinstance(tag, str) and tag:
            Resource._registry[tag] = cls

    # -- load time ---------------------------------------------------------

    def check(self) -> builtins.list[str]:
        """What is wrong with this resource as declared, before any use.

        Raises :class:`ResourceError` for what makes it unusable; returns warnings for what may still be fine (a folder that does
        not exist yet). Cheap: no network, no listing.
        """
        return []

    def local_roots(self) -> tuple[Path, ...] | None:
        """The folders on this machine the resource is, primary first; ``None`` when
        it is not on this machine. What a local consumer reads directly, with the
        link and containment checks it already has."""
        return None

    def describe(self) -> str:
        return f"{self.type} {self.id!r}"

    # -- file-store view ---------------------------------------------------

    @abstractmethod
    async def list(self, path: str = "") -> builtins.list[ResourceEntry]:
        """What a folder holds, sorted by name; empty when it does not exist."""

    @abstractmethod
    async def read(self, path: str) -> bytes:
        """A file's contents. ``FileNotFoundError`` when there is none."""

    @abstractmethod
    async def write(self, path: str, data: bytes) -> None:
        """Create or replace a file. Refused on a read-only resource."""

    @abstractmethod
    async def stat(self, path: str) -> ResourceEntry | None:
        """The entry at ``path``, or ``None`` when there is nothing there."""

    async def read_text(self, path: str) -> str:
        return (await self.read(path)).decode("utf-8", errors="replace")

    def _refuse_read_only(self, path: str) -> None:
        if self.read_only:
            raise ResourceError(f"Resource {self.id!r} is read-only: '{path}' was not written.")

    # -- mount view --------------------------------------------------------

    async def materialize(self) -> MountSpec:
        """The resource as local folders, for a consumer that runs a process in it."""
        raise ResourceError(
            f"Resource {self.id!r} ({self.type}) cannot be used as a local folder: "
            "commands cannot run in it."
        )


def resolve_resource(value: Any) -> Resource:
    """A resource from its configuration mapping, or the instance itself.

    Raises on an unknown ``type`` rather than guessing, like every other typed
    entry of a configuration.
    """
    if isinstance(value, Resource):
        return value
    if not isinstance(value, dict):
        raise ResourceError(f"A resource must be a mapping with 'id' and 'type', got {value!r}")
    tag = value.get("type")
    if not tag:
        raise ResourceError(f"Resource {value.get('id')!r} has no 'type'")
    subclass = Resource._registry.get(tag)
    if subclass is None:
        known = ", ".join(sorted(Resource._registry)) or "none registered"
        raise ResourceError(f"Resource {value.get('id')!r}: unknown type '{tag}' (known: {known})")
    return subclass.model_validate(value)
