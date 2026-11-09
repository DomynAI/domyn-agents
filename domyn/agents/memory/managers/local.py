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

Which file is which memory: a different ``file_path`` is a different memory, and an
agent has one. The CLI's "global" memory is ``~/.domyn/memory.md``, shared by every
configuration that names it; a project's own is ``memory.md`` in the project's folder,
read by that project alone. Neither falls back to the other.
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
    _lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    def _read(self) -> list[Memory]:
        """Every entry, oldest first. A missing file has none."""
        path = Path(self.file_path).expanduser()
        try:
            text = path.read_text(encoding="utf-8")
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        except FileNotFoundError:
            return []
        entries = (_parse(line, modified) for line in text.splitlines())
        return [entry for entry in entries if entry is not None]

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
        """What a user and an agent see: their own, the sub-agents', and what nobody wrote."""
        mine = [m for m in memories if m.user_id is None or m.user_id == user_id]
        if agent is None:
            return mine
        names = {agent.name, *(a.name for a in agent.sub_agents)}
        return [m for m in mine if m.agent_name is None or m.agent_name in names]

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
        """Retrieve a single memory by ID (safe across tasks in one event loop)."""
        async with self._lock:
            entries = await asyncio.to_thread(self._read)
        return next((m for m in entries if m.id == memory_id), None)

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a single memory by ID (safe across tasks in one event loop)."""
        async with self._lock:
            removed = await asyncio.to_thread(self._remove, lambda m: m.id == memory_id)
        return removed > 0

    async def reset_memory(self, user_id: str, agent_name: str | None = None) -> None:
        """Clear a user's memories, optionally only those of one agent."""

        def goes(memory: Memory) -> bool:
            return memory.user_id == user_id and (
                agent_name is None or memory.agent_name == agent_name
            )

        async with self._lock:
            await asyncio.to_thread(self._remove, goes)

    async def complete_reset(self) -> None:
        """Clear all memories for all users and all agents; what is not one stays."""
        async with self._lock:
            await asyncio.to_thread(self._remove, lambda m: True)
