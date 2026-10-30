"""Memory managers module - auto-registers managers in BaseMemoryManager._registry."""

from domyn.agents.memory.managers.base import (
    BaseMemoryManager,
    Memory,
    MemoryExecutionMode,
)
from domyn.agents.memory.managers.local import FilesystemMemoryManager
from domyn.agents.memory.managers.remote import RemoteMemoryManager

__all__ = [
    "BaseMemoryManager",
    "FilesystemMemoryManager",
    "Memory",
    "MemoryExecutionMode",
    "RemoteMemoryManager",
]
