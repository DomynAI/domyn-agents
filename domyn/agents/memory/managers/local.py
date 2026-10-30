import asyncio
import datetime
import os
import tempfile
from typing import Any, Literal

from pydantic import Field, PrivateAttr

from domyn.agents.core import BaseAgent
from domyn.agents.memory.managers.base import BaseMemoryManager, Memory


class FilesystemMemoryManager(BaseMemoryManager):
    """Filesystem-based memory manager, safe across tasks in one event loop."""

    type: Literal["filesystem"] = "filesystem"
    file_path: str = Field(default="local_memory.jsonl")
    _lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    async def write_single_memory(
        self,
        user_id: str,
        agent: BaseAgent,
        memory_content: str,
        **kwargs: Any,
    ) -> None:
        memory = Memory(
            timestamp=datetime.datetime.now(),
            user_id=user_id,
            memory=memory_content,
            agent_name=agent.name,
        )
        # TODO: deduplicate. Every write appends, so repeats crowd recall's recency_limit window.
        async with self._lock:
            with open(self.file_path, "a") as f:
                f.write(memory.model_dump_json() + "\n")

    async def recall_memories(
        self,
        user_id: str,
        agent: BaseAgent | None = None,
        recency_limit: int = 5,
        question: str = "",
        **kwargs: Any,
    ) -> list[Memory]:
        """Retrieve memories from the filesystem (safe across tasks in one event loop)."""
        async with self._lock:
            try:
                with open(self.file_path) as f:
                    data = f.read()

                # Parse memories, skipping malformed lines
                memories = []
                for line in data.splitlines():
                    line = line.strip()
                    if line:
                        try:
                            memory = Memory.model_validate_json(line)
                            memories.append(memory)
                        except Exception:
                            continue  # Skip malformed lines

                filtered_memories = [m for m in memories if m.user_id == user_id]
                if agent is not None:
                    visible_agent_names = [agent.name] + [a.name for a in agent.sub_agents]
                    filtered_memories = [
                        m for m in filtered_memories if m.agent_name in visible_agent_names
                    ]

                filtered_memories.sort(key=lambda m: m.timestamp, reverse=True)
                filtered_memories = filtered_memories[:recency_limit]
                return filtered_memories
            except FileNotFoundError:
                return []

    async def get_memory(self, memory_id: str) -> Memory | None:
        """Retrieve a single memory by ID (safe across tasks in one event loop)."""
        async with self._lock:
            try:
                with open(self.file_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                memory = Memory.model_validate_json(line)
                                if memory.id == memory_id:
                                    return memory
                            except Exception:
                                continue  # Skip malformed lines
                return None  # Memory not found
            except FileNotFoundError:
                return None

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a single memory by ID (safe across tasks in one event loop)."""
        async with self._lock:
            try:
                deleted = False
                with (
                    tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file,
                    open(self.file_path) as original_file,
                ):
                    for line in original_file:
                        line = line.strip()
                        if line:
                            try:
                                memory = Memory.model_validate_json(line)
                                if memory.id == memory_id:
                                    deleted = True
                                    continue  # Delete this memory by skipping the write
                                temp_file.write(line + "\n")
                            except Exception:
                                # Keep malformed lines as-is
                                temp_file.write(line + "\n")

                if deleted:
                    os.replace(temp_file.name, self.file_path)
                else:
                    # Clean up temp file if nothing was deleted
                    os.unlink(temp_file.name)

                return deleted
            except FileNotFoundError:
                return False

    async def reset_memory(self, user_id: str, agent_name: str | None = None) -> None:
        """Clear memories by user_id and optionally agent_name (safe across tasks in one loop)."""
        async with self._lock:
            try:
                with (
                    tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file,
                    open(self.file_path) as original_file,
                ):
                    for line in original_file:
                        line = line.strip()
                        if line:
                            try:
                                memory = Memory.model_validate_json(line)
                                if agent_name is not None:
                                    if (
                                        memory.user_id == user_id
                                        and memory.agent_name == agent_name
                                    ):
                                        continue  # Delete this memory by skipping the write
                                else:
                                    if memory.user_id == user_id:
                                        continue  # Delete this memory by skipping the write
                                temp_file.write(line + "\n")
                            except Exception:
                                # Keep malformed lines as-is
                                temp_file.write(line + "\n")

                os.replace(temp_file.name, self.file_path)
            except FileNotFoundError:
                pass

    async def complete_reset(self) -> None:
        """Clear all memories for all users and all agents (safe across tasks in one event loop)."""
        async with self._lock:
            try:
                with open(self.file_path, "w"):
                    pass
            except FileNotFoundError:
                pass
