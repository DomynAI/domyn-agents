"""A resource that is not on this machine, held in memory.

Stands in for a platform's own type (``platform_fs``): what the library must handle
for any resource without local folders — read through the file-store view, refused
as a place to run commands — is tested against it.
"""

from __future__ import annotations

import builtins
import posixpath
from typing import Any

from pydantic import Field, PrivateAttr

from domyn.agents.resources import Resource, ResourceEntry, relative_path


class InMemoryResource(Resource):
    type: str = "test_memory"
    files: dict[str, str] = Field(default_factory=dict)

    _data: dict[str, bytes] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context: Any) -> None:
        self._data = {relative_path(k): v.encode() for k, v in self.files.items()}

    def _folders(self) -> set[str]:
        found: set[str] = set()
        for name in self._data:
            parent = posixpath.dirname(name)
            while parent:
                found.add(parent)
                parent = posixpath.dirname(parent)
        return found

    async def list(self, path: str = "") -> builtins.list[ResourceEntry]:
        base = relative_path(path)
        children: dict[str, ResourceEntry] = {}
        for name in [*self._data, *self._folders()]:
            parent = posixpath.dirname(name)
            if parent == base:
                children[name] = ResourceEntry(name, name not in self._data)
        return [children[k] for k in sorted(children)]

    async def read(self, path: str) -> bytes:
        try:
            return self._data[relative_path(path)]
        except KeyError:
            raise FileNotFoundError(path) from None

    async def write(self, path: str, data: bytes) -> None:
        self._refuse_read_only(path)
        self._data[relative_path(path)] = data

    async def stat(self, path: str) -> ResourceEntry | None:
        name = relative_path(path)
        if name in self._data:
            return ResourceEntry(name, False, len(self._data[name]))
        return ResourceEntry(name, True) if name in self._folders() or not name else None
