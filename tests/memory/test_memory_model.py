"""Tests for the Memory model."""

import uuid
from datetime import datetime

import pytest

from domyn.agents.memory.managers import Memory


class TestMemory:
    """Test cases for Memory model."""

    def test_memory_creation_with_defaults(self):
        """Test creating a memory with default values."""
        memory = Memory(timestamp=datetime.now(), user_id="user123", memory="This is a test memory")

        assert memory.agent_name is None
        assert isinstance(memory.id, str)
        assert len(memory.id) == 36  # UUID4 length

    def test_memory_json_serialization(self):
        """Test memory JSON serialization."""
        timestamp = datetime.now()
        memory = Memory(
            timestamp=timestamp,
            user_id="user123",
            memory="Test serialization",
            agent_name="serializer",
        )

        json_str = memory.model_dump_json()
        assert "user123" in json_str
        assert "Test serialization" in json_str
        assert "serializer" in json_str

    def test_memory_json_deserialization(self):
        """Test memory JSON deserialization."""
        timestamp = datetime.now()
        original_memory = Memory(
            timestamp=timestamp, user_id="user123", memory="Test deserialization"
        )

        json_str = original_memory.model_dump_json()
        reconstructed_memory = Memory.model_validate_json(json_str)

        assert reconstructed_memory.id == original_memory.id
        assert reconstructed_memory.user_id == original_memory.user_id
        assert reconstructed_memory.memory == original_memory.memory
        assert reconstructed_memory.agent_name == original_memory.agent_name

    def test_memory_unique_ids(self):
        """Test that different memories get unique IDs."""
        timestamp = datetime.now()

        memory1 = Memory(timestamp=timestamp, user_id="user1", memory="Memory 1")

        memory2 = Memory(timestamp=timestamp, user_id="user1", memory="Memory 2")

        assert memory1.id != memory2.id

    def test_memory_validation_errors(self):
        """Test memory validation errors."""
        timestamp = datetime.now()

        # Missing required fields should raise validation error
        with pytest.raises(ValueError):
            Memory(timestamp=timestamp, user_id="user1")  # Missing memory

        # This should not raise - user_id is now optional
        memory = Memory(timestamp=timestamp, memory="test")
        assert memory.memory == "test"
        assert memory.user_id is None

        with pytest.raises(ValueError):
            Memory(user_id="user1", memory="test")  # Missing timestamp
