"""Where the file tools read and write: this machine, or another one.

The bodies of the file tools (``files.build_tools``) do the same work wherever the files
are: windows of lines, exact edits, documents converted, the workspace's boundary. Only
the operations that touch a file go through a ``Disk``. ``LocalDisk`` is pathlib, as the
tools always were; a backend whose files are on another machine gives one of its own
(the SSH backend's runs a few standard commands there, ``backends/ssh/disk.py``).

Paths are ``Path`` objects either way. On another machine they are that machine's paths,
and nothing here touches them through pathlib's own filesystem methods.

Imported by the container's tool host, like ``files``: nothing that reaches the agent
machinery may be imported here.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

from domyn.agents.core import documents
from domyn.agents.tools.builtin.workspace import Workspace


class DiskError(RuntimeError):
    """The disk itself could not be used (the server holding it did not answer), as
    opposed to a file on it that could not be read."""


Hit = tuple[Path, int, str]
"""A matching line: the file, its line number from 1, the line."""


class Disk(ABC):
    """The operations the file tools make on files, wherever the files are."""

    @abstractmethod
    def resolve(
        self, workspace: Workspace, candidate: str | os.PathLike[str], *, must_exist: bool = False
    ) -> Path:
        """``Workspace.resolve``, on this disk: a path the model gave, made real and
        checked against the workspace, or ``WorkspaceError``."""

    @abstractmethod
    def real(self, path: Path) -> Path:
        """``path`` with its links and ``..`` resolved. ``OSError`` when it cannot be."""

    @abstractmethod
    def exists(self, path: Path) -> bool: ...

    @abstractmethod
    def is_dir(self, path: Path) -> bool: ...

    @abstractmethod
    def is_file(self, path: Path) -> bool: ...

    @abstractmethod
    def size(self, path: Path) -> int: ...

    @abstractmethod
    def links(self, path: Path) -> int:
        """How many names the file has on disk (its hard links)."""

    @abstractmethod
    def lines(self, path: Path) -> AbstractContextManager[Iterable[str]]:
        """The file's lines as UTF-8 text, newlines translated, read as they are
        iterated: never whole. ``UnicodeDecodeError`` when it is not UTF-8."""

    @abstractmethod
    def read_bytes(self, path: Path) -> bytes: ...

    @abstractmethod
    def read_text(self, path: Path) -> str:
        """The whole file as UTF-8, newlines translated. ``UnicodeDecodeError`` when
        it is not UTF-8."""

    @abstractmethod
    def write_text(self, path: Path, text: str) -> None:
        """Replace the file's contents, creating it and the folders above it."""

    @abstractmethod
    def append_text(self, path: Path, text: str) -> None:
        """Add to the end of the file, creating it and the folders above it."""

    @abstractmethod
    def iterdir(self, path: Path) -> list[Path]: ...

    @abstractmethod
    def glob(self, base: Path, pattern: str) -> list[Path]:
        """What ``base.glob(pattern)`` names, unresolved and unchecked: the caller
        checks every match against the workspace."""

    @abstractmethod
    def search(
        self, files: Sequence[Path], regex: re.Pattern[str], limit: int
    ) -> tuple[list[Hit], int]:
        """The lines of ``files`` that ``regex`` matches, in order, stopping after
        ``limit`` of them, and how many files were read as text. Files that are not
        UTF-8 text, or cannot be read, are skipped and not counted."""

    def extract_text(self, path: Path) -> documents.Text:
        """``documents.extract_text`` for the file at ``path``."""
        return documents.extract_text_bytes(self.read_bytes(path), path)


class LocalDisk(Disk):
    """The files of this machine, through pathlib."""

    def resolve(
        self, workspace: Workspace, candidate: str | os.PathLike[str], *, must_exist: bool = False
    ) -> Path:
        return workspace.resolve(candidate, must_exist=must_exist)

    def real(self, path: Path) -> Path:
        return path.resolve()

    def exists(self, path: Path) -> bool:
        return path.exists()

    def is_dir(self, path: Path) -> bool:
        return path.is_dir()

    def is_file(self, path: Path) -> bool:
        return path.is_file()

    def size(self, path: Path) -> int:
        return path.stat().st_size

    def links(self, path: Path) -> int:
        return path.stat().st_nlink

    @contextmanager
    def lines(self, path: Path) -> Iterator[Iterable[str]]:
        with path.open(encoding="utf-8") as handle:
            yield handle

    def read_bytes(self, path: Path) -> bytes:
        return path.read_bytes()

    def read_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def append_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text)

    def iterdir(self, path: Path) -> list[Path]:
        return list(path.iterdir())

    def glob(self, base: Path, pattern: str) -> list[Path]:
        return list(base.glob(pattern))

    def search(
        self, files: Sequence[Path], regex: re.Pattern[str], limit: int
    ) -> tuple[list[Hit], int]:
        hits: list[Hit] = []
        searched = 0
        for candidate in files:
            try:
                text = candidate.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            searched += 1
            for number, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    hits.append((candidate, number, line))
                    if len(hits) >= limit:
                        return hits, searched
        return hits, searched

    def extract_text(self, path: Path) -> documents.Text:
        # By its path: a document is handed to its converter without its bytes
        # being held beside its text.
        return documents.extract_text(path)


LOCAL_DISK = LocalDisk()
