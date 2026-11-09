"""A memory kept in a file the user can read and edit.

The file is markdown: a heading and a bullet per entry, with what the program needs
to know about an entry in a comment at the end of the line, where an editor does not
see it:

    # Memory

    - The user prefers metric units. <!-- id=a1b2c3d4 at=2026-10-12T10:00:00+00:00 user=u1 by=Assistant -->

A bullet the user adds by hand is an entry too: it belongs to no user and no agent, so
every one of them sees it; its id comes from its text and its date from the file.
Entries are one line each. A change writes the whole file to a temporary file beside
it and renames it, so an interrupted write leaves the old one.

Files written by earlier versions, one JSON object per line, are still read, and the
next change rewrites them as markdown; nothing in them is lost. The default file name
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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, unquote

from pydantic import Field, PrivateAttr

from domyn.agents.core import BaseAgent
from domyn.agents.memory.managers.base import BaseMemoryManager, Memory, MemoryPage

_HEADING = "# Memory"
_LINE = re.compile(r"^- (?P<text>.*?)(?: <!-- (?P<meta>.*?) -->)?\s*$")
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

    def _write(self, entries: list[Memory]) -> None:
        path = Path(self.file_path).expanduser()
        body = "\n".join([_HEADING, "", *(_format(entry) for entry in entries)]) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".memory-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as out:
                out.write(body)
            os.replace(temporary, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise

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
            entries = await asyncio.to_thread(self._read)
            await asyncio.to_thread(self._write, [*entries, memory])

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

    async def recall_page(
        self,
        user_id: str,
        agent: BaseAgent | None = None,
        *,
        query: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
        **kwargs: Any,
    ) -> MemoryPage:
        """A page of what the user and agent see, the most recent first or the best
        match for ``query``.

        The file is read on every call and searched as it is (no index is kept), so
        an edit by hand is seen at once. The search runs in a thread, outside the
        lock: it only reads what was read.
        """
        async with self._lock:
            entries = await asyncio.to_thread(self._read)
        seen = list(reversed(self._seen_by(entries, user_id, agent)))
        if query:
            seen = await asyncio.to_thread(self.search.search, seen, query)
        return self.cut_page(seen, limit, cursor)

    async def get_memory(self, memory_id: str) -> Memory | None:
        """Retrieve a single memory by ID (safe across tasks in one event loop)."""
        async with self._lock:
            entries = await asyncio.to_thread(self._read)
        return next((m for m in entries if m.id == memory_id), None)

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a single memory by ID (safe across tasks in one event loop)."""
        async with self._lock:
            entries = await asyncio.to_thread(self._read)
            kept = [m for m in entries if m.id != memory_id]
            if len(kept) == len(entries):
                return False
            await asyncio.to_thread(self._write, kept)
        return True

    async def reset_memory(self, user_id: str, agent_name: str | None = None) -> None:
        """Clear a user's memories, optionally only those of one agent."""
        async with self._lock:
            entries = await asyncio.to_thread(self._read)
            if not entries:
                return

            def goes(memory: Memory) -> bool:
                return memory.user_id == user_id and (
                    agent_name is None or memory.agent_name == agent_name
                )

            await asyncio.to_thread(self._write, [m for m in entries if not goes(m)])

    async def complete_reset(self) -> None:
        """Clear all memories for all users and all agents."""
        async with self._lock:
            if not await asyncio.to_thread(self._read):
                return
            await asyncio.to_thread(self._write, [])
