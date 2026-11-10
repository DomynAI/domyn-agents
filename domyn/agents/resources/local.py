"""``local_dir``: folders on this machine, as an ordered list of roots.

The first root is the primary one: a ``files`` bundle's workspace, and where the
file-store view writes. The others are read too, earlier wins — what
``Workspace.extra_roots`` does for skill folders. A root is not a restriction: a
consumer reads its own subpath under every root (``skills`` for skills, ``SOUL.md``
and ``AGENTS.md`` for instructions).

A relative root means the folder the process runs in, resolved on every use, as a
skills path always has; ``~`` is the user's home.
"""

from __future__ import annotations

import asyncio
import builtins
from pathlib import Path

from pydantic import Field, field_validator

from domyn.agents.resources.base import (
    MountSpec,
    Resource,
    ResourceEntry,
    ResourceError,
    relative_path,
)


class LocalDirResource(Resource):
    """Folders on this machine, the first one primary."""

    type: str = "local_dir"
    paths: builtins.list[str] = Field(min_length=1)

    mountable = True

    @field_validator("paths", mode="before")
    @classmethod
    def _as_list(cls, value: object) -> object:
        # One root may be written on its own: `paths: .`
        return [value] if isinstance(value, str | Path) else value

    @field_validator("paths")
    @classmethod
    def _as_strings(cls, value: builtins.list[str]) -> builtins.list[str]:
        if any(not str(one).strip() for one in value):
            raise ValueError("an empty entry in 'paths' would mean the working folder by accident")
        return [str(one) for one in value]

    def local_roots(self) -> tuple[Path, ...]:
        """As written, ``~`` expanded. Kept relative when written so: the skills and
        instructions readers check a relative path's every folder for links, which
        an absolute one would skip."""
        return tuple(Path(one).expanduser() for one in self.paths)

    def check(self) -> builtins.list[str]:
        """A root that does not exist, when it is absolute. A relative one names a folder
        of wherever the process runs (a project's ``.domyn``), which most places lack."""
        return [
            f"resource {self.id!r}: {root} does not exist"
            for root in self.local_roots()
            if root.is_absolute() and not root.exists()
        ]

    def describe(self) -> str:
        return f"{self.type} {self.id!r} ({', '.join(self.paths)})"

    # -- file-store view ---------------------------------------------------

    def _candidate(self, root: Path, relative: str) -> Path | None:
        """Where ``relative`` is under one root, or ``None`` when that root cannot
        hold it. A link on the way is refused, never followed: it could lead
        anywhere on the disk."""
        if root.is_file() or root.is_symlink():
            # A root that is one file holds that file, by its name.
            if root.is_symlink():
                return None
            return root if relative == root.name else None
        target = root
        for part in relative.split("/") if relative else ():
            target = target / part
            if target.is_symlink():
                raise ResourceError(
                    f"'{relative}' goes through a link in resource {self.id!r}, not followed."
                )
        return target

    def _read_sync(self, path: str) -> bytes:
        relative = relative_path(path)
        for root in self.local_roots():
            candidate = self._candidate(root, relative)
            if candidate is not None and candidate.is_file():
                return candidate.read_bytes()
        raise FileNotFoundError(f"'{path}' is not in resource {self.id!r}")

    def _write_sync(self, path: str, data: bytes) -> None:
        self._refuse_read_only(path)
        relative = relative_path(path)
        if not relative:
            raise ResourceError("A file needs a name: the path is the resource itself.")
        primary = self.local_roots()[0]
        if not primary.is_dir() or primary.is_symlink():
            raise ResourceError(
                f"Resource {self.id!r}: {primary} is not a folder, nothing can be written there."
            )
        target = self._candidate(primary, relative)
        assert target is not None  # a folder root always has a candidate
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def _list_sync(self, path: str) -> builtins.list[ResourceEntry]:
        relative = relative_path(path)
        found: dict[str, ResourceEntry] = {}
        for root in self.local_roots():
            if not relative and root.is_file() and not root.is_symlink():
                found.setdefault(root.name, ResourceEntry(root.name, False, root.stat().st_size))
                continue
            folder = self._candidate(root, relative)
            if folder is None or not folder.is_dir():
                continue
            for child in sorted(folder.iterdir(), key=lambda one: one.name):
                if child.is_symlink():
                    continue
                name = f"{relative}/{child.name}" if relative else child.name
                is_dir = child.is_dir()
                found.setdefault(
                    child.name,
                    ResourceEntry(name, is_dir, None if is_dir else child.stat().st_size),
                )
        return [found[name] for name in sorted(found)]

    def _stat_sync(self, path: str) -> ResourceEntry | None:
        relative = relative_path(path)
        for root in self.local_roots():
            candidate = self._candidate(root, relative)
            if candidate is None or not candidate.exists():
                continue
            is_dir = candidate.is_dir()
            return ResourceEntry(relative, is_dir, None if is_dir else candidate.stat().st_size)
        return None

    async def list(self, path: str = "") -> builtins.list[ResourceEntry]:
        return await asyncio.to_thread(self._list_sync, path)

    async def read(self, path: str) -> bytes:
        return await asyncio.to_thread(self._read_sync, path)

    async def write(self, path: str, data: bytes) -> None:
        await asyncio.to_thread(self._write_sync, path, data)

    async def stat(self, path: str) -> ResourceEntry | None:
        return await asyncio.to_thread(self._stat_sync, path)

    # -- mount view --------------------------------------------------------

    async def materialize(self) -> MountSpec:
        """The folders themselves: a local folder is its own mount, nothing is copied."""
        return MountSpec(
            paths=tuple(root.resolve() for root in self.local_roots()), read_only=self.read_only
        )
