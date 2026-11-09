"""Memory managers module - auto-registers managers in BaseMemoryManager._registry."""

from domyn.agents.memory.managers.base import (
    MEMORY_HOOK_NAME,
    BaseMemoryManager,
    Memory,
    MemoryExecutionMode,
    MemoryPage,
)
from domyn.agents.memory.managers.local import FilesystemMemoryManager
from domyn.agents.memory.managers.remote import RemoteMemoryManager

__all__ = [
    "MEMORY_HOOK_NAME",
    "BaseMemoryManager",
    "FilesystemMemoryManager",
    "Memory",
    "MemoryExecutionMode",
    "MemoryPage",
    "RemoteMemoryManager",
]
