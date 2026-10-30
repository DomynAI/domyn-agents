"""Tests for memory manager registry and deserialization."""

import pytest

from domyn.agents.memory.managers import BaseMemoryManager, FilesystemMemoryManager


class TestMemoryManagerRegistry:
    """Test memory manager registry and from_config pattern."""

    def test_filesystem_manager_registered(self):
        """Test that FilesystemMemoryManager is auto-registered in registry."""
        assert "filesystem" in BaseMemoryManager._registry
        assert BaseMemoryManager._registry["filesystem"] == FilesystemMemoryManager

    def test_from_config_pattern(self):
        """Test registry lookup and from_config() pattern (like BaseTool)."""
        config = {
            "type": "filesystem",
            "file_path": "/tmp/test.jsonl",
            "execution_mode": "automatic",
        }

        # Pattern: registry lookup + from_config()
        manager_class = BaseMemoryManager._registry.get("filesystem")
        manager = manager_class.from_config(config)

        assert isinstance(manager, FilesystemMemoryManager)
        assert manager.file_path == "/tmp/test.jsonl"

    def test_unknown_type_returns_none(self):
        """Test that registry returns None for unknown memory manager types."""
        manager_class = BaseMemoryManager._registry.get("unknown_type")
        assert manager_class is None
