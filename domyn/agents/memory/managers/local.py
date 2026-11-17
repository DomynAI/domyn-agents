"""A memory kept in a file the user can read and edit.

The file is markdown: a heading and a bullet per entry, with what the program needs
to know about an entry in a comment at the end of the line, where an editor does not
see it:

    # Memory

    - The user prefers metric units. <!-- id=a1b2c3d4 at=2026-10-12T10:00:00+00:00 user=u1 by=Assistant -->

A bullet the user adds by hand is an entry too: it belongs to no user and no agent, so
every one of them sees it; its id comes from its text and its date from the file.
Entries are one line each. A new entry is appended, as one write, so nothing else in
the file is touched and the entries other processes append are kept. Removing entries
deletes their lines and nothing else, writing the result to a temporary file beside
the old one and renaming it, so an interrupted write leaves the old one.

Files written by earlier versions, one JSON object per line, are still read; new
entries are appended to them as markdown, and the next removal rewrites the whole file
as markdown; nothing in them is lost. The default file name
keeps its old ``.jsonl`` so that such a file is found.

Which file is which memory: ``file_path`` is the memory the agent writes, the one
file a manager stores in, and a different one is a different memory.
``read_only_sources`` are extra files read beside it, never part of it: their entries
are recalled, searched and count as already remembered, so a fact found there is not
stored again, but ``remember`` writes to ``file_path`` alone, and forgetting or
clearing never touches them. The CLI's "global" memory is
``~/.domyn/memory.md``, shared by every configuration that writes it; a project's own
is ``memory.md`` in the project's folder. A project can read both and write one: the
global memory, so what it learns there is known in other projects too, or its own,
so it learns from the global memory without adding to it.
"""

import asyncio
import contextlib
import hashlib
import os
import re
import secrets
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, unquote

from pydantic import Field, PrivateAttr

from domyn.agents.core import BaseAgent
from domyn.agents.memory.managers.base import BaseMemoryManager, Memory

_HEADING = "# Memory"
# The comment holding an entry's data is the last on its line: it cannot hold a
# `<!--` itself, so a text that has one ("uses <!-- for comments") is kept whole.
_LINE = re.compile(r"^- (?P<text>.*?)(?: <!-- (?P<meta>(?:(?!<!--).)*?) -->)?\s*$")
_META = re.compile(r"(\w+)=(\S+)")


def _one_line(text: str) -> str:
    # `-->` would end the comment that follows it on the line.
    return " ".join(text.split()).replace("-->", "-- >")


def _aware(stamp: datetime) -> datetime:
    """A time with a zone: the old files hold local times without one."""
    return stamp if stamp.tzinfo else stamp.astimezone(UTC)


def _parse(line: str, modified: datetime) -> Memory | None:
    """One line of the file as a memory, or ``None`` when it is not one."""
    stripped = line.strip()
    if stripped.startswith("{"):
        try:
            legacy = Memory.model_validate_json(stripped)
        except ValueError:
            return None
        return legacy.model_copy(update={"timestamp": _aware(legacy.timestamp)})
    match = _LINE.match(stripped)
    if match is None:
        return None
    text = match["text"].strip()
    if not text:
        return None
    meta = dict(_META.findall(match["meta"] or ""))
    try:
        at = datetime.fromisoformat(meta["at"]) if "at" in meta else modified
    except ValueError:
        at = modified
    return Memory(
        id=meta.get("id") or hashlib.sha1(text.encode()).hexdigest()[:8],
        timestamp=_aware(at),
        memory=text,
        agent_name=unquote(meta["by"]) if "by" in meta else None,
        user_id=unquote(meta["user"]) if "user" in meta else None,
    )


def _format(memory: Memory) -> str:
    user = f" user={quote(memory.user_id, safe='')}" if memory.user_id else ""
    by = f" by={quote(memory.agent_name, safe='')}" if memory.agent_name else ""
    return (
        f"- {_one_line(memory.memory)} "
        f"<!-- id={memory.id} at={memory.timestamp.isoformat()}{user}{by} -->"
    )


class FilesystemMemoryManager(BaseMemoryManager):
    """Memories in a markdown file, safe across tasks in one event loop."""

    type: Literal["filesystem"] = "filesystem"
    file_path: str = Field(default="local_memory.jsonl")
    read_only_sources: list[str] = Field(
        default_factory=list,
        description=(
            "Extra memory files, read only: their entries are recalled and searched "
            "with those of file_path, but they are not part of the memory the agent "
            "writes, so nothing is ever stored, forgotten or cleared in them"
        ),
    )
    _lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    def _read_only_paths(self) -> list[Path]:
        """The files of ``read_only_sources``, each once, ``file_path`` itself left out."""
        own = Path(self.file_path).expanduser()
        seen = {own.resolve()}
        paths: list[Path] = []
        for entry in self.read_only_sources:
            path = Path(entry).expanduser()
            if path.resolve() not in seen:
                seen.add(path.resolve())
                paths.append(path)
        return paths

    @staticmethod
    def _read_file(path: Path) -> list[Memory]:
        """Every entry of one file, in file order. A missing file has none."""
        try:
            text = path.read_text(encoding="utf-8")
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        except FileNotFoundError:
            return []
        entries = (_parse(line, modified) for line in text.splitlines())
        return [entry for entry in entries if entry is not None]

    def _read(self) -> list[Memory]:
        """Every entry, oldest first: those of ``file_path`` and, marked with the file
        they came from, those of ``read_only_sources``."""
        entries = self._read_file(Path(self.file_path).expanduser())
        others = self._read_only_paths()
        for path in others:
            entries += [
                memory.model_copy(update={"read_only_source": str(path)})
                for memory in self._read_file(path)
            ]
        if others:
            # Stable: within one file, and at equal times, the file's order is kept.
            entries.sort(key=lambda memory: memory.timestamp)
        return entries

    def _remove(self, goes: Callable[[Memory], bool]) -> int:
        """Delete the lines of the entries ``goes`` picks, and return how many.

        Every other line stays as it is: headings, notes, bullets of another kind,
        lines that are not entries at all. Entries in the JSON of earlier versions
        that stay are written as markdown, with the heading the file then lacks. The
        result goes to a temporary file beside the old one and is renamed over it,
        so an interrupted write leaves the old file; a file with nothing to remove
        is not written.
        """
        path = Path(self.file_path).expanduser()
        try:
            text = path.read_text(encoding="utf-8")
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        except FileNotFoundError:
            return 0
        kept: list[str] = []
        removed = 0
        converted = False
        for line in text.splitlines():
            entry = _parse(line, modified)
            if entry is not None and goes(entry):
                removed += 1
            elif entry is not None and line.lstrip().startswith("{"):
                kept.append(_format(entry))
                converted = True
            else:
                kept.append(line)
        if not removed:
            return 0
        if converted and _HEADING not in (line.strip() for line in kept):
            kept = [_HEADING, "", *kept]
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".memory-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as out:
                out.write("\n".join(kept) + "\n" if kept else "")
            os.replace(temporary, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise
        return removed

    def stored(self) -> list[Memory]:
        """The entries of ``file_path`` alone (read-only sources left out), in file order."""
        return self._read_file(Path(self.file_path).expanduser())

    def remove_where(self, goes: Callable[[Memory], bool]) -> int:
        """Delete the entries of ``file_path`` that ``goes`` picks; return how many.
        Everything else in the file stays as written (see `_remove`)."""
        return self._remove(goes)

    def _append(self, memory: Memory) -> None:
        """Add one entry at the end of the file, touching nothing else in it.

        One ``write`` to a file opened for appending, so the lines other processes
        append are never lost (two sessions sharing the global memory, an editor
        saving it), and nothing the user wrote by hand is rewritten. A new or empty
        file gets the heading first; a file not ending in a newline gets one.
        """
        path = Path(self.file_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            size = os.fstat(fd).st_size
            if size == 0:
                lead = f"{_HEADING}\n\n"
            else:
                lead = "" if os.pread(fd, 1, size - 1) == b"\n" else "\n"
            os.write(fd, f"{lead}{_format(memory)}\n".encode())
        finally:
            os.close(fd)

    @staticmethod
    def _seen_by(memories: list[Memory], user_id: str, agent: BaseAgent | None) -> list[Memory]:
        """What a user and an agent see: their own, the sub-agents', what nobody wrote,
        and everything of the user in a file only read.

        A file of ``read_only_sources`` is another configuration's memory, written by agents
        of its own: filtered by agent name, it would show nothing to an agent named
        otherwise, which is the whole point of reading it.
        """
        mine = [m for m in memories if m.user_id is None or m.user_id == user_id]
        if agent is None:
            return mine
        names = {agent.name, *(a.name for a in agent.sub_agents)}
        return [
            m
            for m in mine
            if m.read_only_source is not None or m.agent_name is None or m.agent_name in names
        ]

    async def write_single_memory(
        self, user_id: str, agent: BaseAgent, memory_content: str, **kwargs: Any
    ) -> None:
        text = _one_line(memory_content)
        if not text:
            return
        memory = Memory(
            id=secrets.token_hex(4),
            timestamp=datetime.now(UTC),
            memory=text,
            agent_name=agent.name,
            user_id=user_id,
        )
        async with self._lock:
            await asyncio.to_thread(self._append, memory)

    async def recall_memories(
        self,
        user_id: str,
        agent: BaseAgent | None = None,
        recency_limit: int = 5,
        question: str = "",
        **kwargs: Any,
    ) -> list[Memory]:
        """The most recent memories of a user, and of an agent and its sub-agents."""
        async with self._lock:
            entries = await asyncio.to_thread(self._read)
        return list(reversed(self._seen_by(entries, user_id, agent)))[:recency_limit]

    async def get_memory(self, memory_id: str) -> Memory | None:
        """Retrieve a single memory by ID (safe across tasks in one event loop); one
        in ``file_path`` before one with the same id in a file only read."""
        async with self._lock:
            entries = await asyncio.to_thread(self._read)
        matching = sorted(
            (m for m in entries if m.id == memory_id), key=lambda m: m.read_only_source is not None
        )
        return matching[0] if matching else None

    async def forget(
        self, agent: BaseAgent, memory_id: str, run_config: dict[str, Any] | None = None
    ) -> str:
        """Remove one remembered fact, by the id shown with it; one in a file only read
        is refused, saying where it is."""
        if (run_config or {}).get("user_id"):
            try:
                found = await self.get_memory(memory_id.strip())
            except OSError:
                found = None  # the removal below reports it
            if found is not None and found.read_only_source is not None:
                return (
                    f"Not forgotten: {found.id} is in {found.read_only_source}, a memory "
                    "this agent reads but never changes."
                )
        return await super().forget(agent, memory_id, run_config)

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a single memory by ID (safe across tasks in one event loop), from
        ``file_path`` only: the files of ``read_only_sources`` are never written."""
        async with self._lock:
            removed = await asyncio.to_thread(self._remove, lambda m: m.id == memory_id)
        return removed > 0

    async def reset_memory(self, user_id: str, agent_name: str | None = None) -> None:
        """Clear a user's memories in ``file_path``, optionally only those of one agent."""

        def goes(memory: Memory) -> bool:
            return memory.user_id == user_id and (
                agent_name is None or memory.agent_name == agent_name
            )

        async with self._lock:
            await asyncio.to_thread(self._remove, goes)

    async def complete_reset(self) -> None:
        """Clear every memory in ``file_path``, of every user and agent; what is not one
        stays, and the files of ``read_only_sources`` are not touched."""
        async with self._lock:
            await asyncio.to_thread(self._remove, lambda m: True)
