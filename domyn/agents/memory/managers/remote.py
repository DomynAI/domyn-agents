import datetime
from typing import Any

import httpx

from domyn.agents.core import BaseAgent
from domyn.agents.memory.managers.base import BaseMemoryManager, Memory
from domyn.agents.tracing import add_trace_headers


class RemoteMemoryManager(BaseMemoryManager):
    """Remote memory manager implementation."""

    type: str = "remote"

    read_url: str | None = None  # For REST reading
    write_url: str | None = None  # For REST writing

    async def write_single_memory(
        self,
        user_id: str,
        agent: BaseAgent,
        memory_content: str,
        **kwargs: Any,
    ) -> None:
        if not self.write_url:
            raise ValueError("RemoteMemoryManager.write_url is not configured")
        memory = Memory(
            timestamp=datetime.datetime.now(),
            user_id=user_id,
            memory=memory_content,
            agent_name=agent.name,
        )

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.write_url,
                json=memory.model_dump(),
                headers=add_trace_headers({}),
            )
            response.raise_for_status()

    async def recall_memories(
        self,
        user_id: str,
        agent: BaseAgent | None = None,
        recency_limit: int = 5,
        question: str = "",
        **kwargs: Any,
    ) -> list[Memory]:
        """Not implemented: always returns an empty list."""
        return []

    async def get_memory(self, memory_id: str) -> Memory | None:
        """Not implemented: always returns None."""
        return None

    async def delete_memory(self, memory_id: str) -> bool:
        """Not implemented: always returns False."""
        return False
