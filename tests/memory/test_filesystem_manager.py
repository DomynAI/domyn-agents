"""Tests for FilesystemMemoryManager."""

import asyncio
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from domyn.agents.core import BaseAgent
from domyn.agents.memory.managers import FilesystemMemoryManager, Memory


class TestFilesystemMemoryManager:
    """Test cases for FilesystemMemoryManager."""

    @staticmethod
    def _build_agent(name: str | None) -> BaseAgent:
        agent = Mock(spec=BaseAgent)
        agent.name = name
        agent.sub_agents = []
        return agent

    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def memory_manager(self, temp_file):
        """Create a FilesystemMemoryManager with temp file."""
        return FilesystemMemoryManager(file_path=temp_file)

    @pytest.fixture
    def sample_memory(self):
        """Create a sample memory for testing."""
        return Memory(
            timestamp=datetime.now(),
            user_id="user123",
            memory="This is a test memory",
            agent_name="test_agent",
        )

    @pytest.fixture
    def sample_memories(self):
        """Create multiple sample memories for testing."""
        base_time = datetime.now()
        return [
            Memory(
                timestamp=base_time,
                user_id="user1",
                memory="Memory 1 for user1",
                agent_name="agent1",
            ),
            Memory(
                timestamp=base_time,
                user_id="user1",
                memory="Memory 2 for user1",
                agent_name="agent2",
            ),
            Memory(
                timestamp=base_time,
                user_id="user2",
                memory="Memory 1 for user2",
                agent_name="agent1",
            ),
            Memory(
                timestamp=base_time, user_id="user2", memory="Memory 2 for user2", agent_name=None
            ),
        ]

    @pytest.mark.asyncio
    async def test_update_memory(self, memory_manager, sample_memory, temp_file):
        """Test storing a memory."""
        agent = self._build_agent(sample_memory.agent_name)
        await memory_manager.write_single_memory(
            user_id=sample_memory.user_id,
            agent=agent,
            memory_content=sample_memory.memory,
        )

        # Verify file was created and contains the memory
        assert os.path.exists(temp_file)

        with open(temp_file) as f:
            content = f.read().strip()
            assert sample_memory.user_id in content
            assert sample_memory.memory in content

    @pytest.mark.asyncio
    async def test_get_memory_existing(self, memory_manager, sample_memory):
        """Test retrieving an existing memory by ID."""
        # Store the memory first
        agent = self._build_agent(sample_memory.agent_name)
        await memory_manager.write_single_memory(
            user_id=sample_memory.user_id,
            agent=agent,
            memory_content=sample_memory.memory,
        )

        # Get all memories to find the actual ID that was created
        memories = await memory_manager.recall_memories(sample_memory.user_id, recency_limit=10)
        assert len(memories) == 1
        stored_memory = memories[0]

        # Retrieve it by the actual ID
        retrieved = await memory_manager.get_memory(stored_memory.id)

        assert retrieved is not None
        assert retrieved.id == stored_memory.id
        assert retrieved.user_id == sample_memory.user_id
        assert retrieved.memory == sample_memory.memory
        assert retrieved.agent_name == sample_memory.agent_name

    @pytest.mark.asyncio
    async def test_get_memory_nonexistent(self, memory_manager):
        """Test retrieving a nonexistent memory."""
        result = await memory_manager.get_memory("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_memory_empty_file(self, memory_manager):
        """Test retrieving from empty file."""
        result = await memory_manager.get_memory("any-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_recall_memories_by_user(self, memory_manager, sample_memories):
        """Test recalling memories for a specific user."""
        # Store all memories
        for memory in sample_memories:
            agent = self._build_agent(memory.agent_name)
            await memory_manager.write_single_memory(
                user_id=memory.user_id, agent=agent, memory_content=memory.memory
            )

        # Recall memories for user1
        user1_memories = await memory_manager.recall_memories("user1")

        assert len(user1_memories) == 2
        for memory in user1_memories:
            assert memory.user_id == "user1"

    @pytest.mark.asyncio
    async def test_recall_memories_by_user_and_agent(self, memory_manager, sample_memories):
        """Test recalling memories filtered by user and agent."""
        # Store all memories
        for memory in sample_memories:
            agent = self._build_agent(memory.agent_name)
            await memory_manager.write_single_memory(
                user_id=memory.user_id, agent=agent, memory_content=memory.memory
            )

        # We need to create a mock agent to test the agent filtering
        mock_agent = Mock(spec=BaseAgent)
        mock_agent.name = "agent1"
        mock_agent.sub_agents = []

        # Recall memories for user1 and agent1
        filtered_memories = await memory_manager.recall_memories("user1", agent=mock_agent)

        assert len(filtered_memories) == 1
        assert filtered_memories[0].user_id == "user1"
        assert filtered_memories[0].agent_name == "agent1"

    @pytest.mark.asyncio
    async def test_recall_memories_with_limit(self, memory_manager, sample_memories):
        """Test recalling memories with a limit."""
        # Store all memories
        for memory in sample_memories:
            agent = self._build_agent(memory.agent_name)
            await memory_manager.write_single_memory(
                user_id=memory.user_id, agent=agent, memory_content=memory.memory
            )

        # Recall with limit
        limited_memories = await memory_manager.recall_memories("user1", recency_limit=1)

        assert len(limited_memories) == 1

    @pytest.mark.asyncio
    async def test_recall_memories_empty_file(self, memory_manager):
        """Test recalling from empty file."""
        memories = await memory_manager.recall_memories("any-user")
        assert memories == []

    @pytest.mark.asyncio
    async def test_delete_memory_existing(self, memory_manager, sample_memory):
        """Test deleting an existing memory."""
        # Store the memory
        agent = self._build_agent(sample_memory.agent_name)
        await memory_manager.write_single_memory(
            user_id=sample_memory.user_id,
            agent=agent,
            memory_content=sample_memory.memory,
        )

        # Get all memories to find the actual ID that was created
        memories = await memory_manager.recall_memories(sample_memory.user_id, recency_limit=10)
        assert len(memories) == 1
        stored_memory = memories[0]

        # Verify it exists
        retrieved = await memory_manager.get_memory(stored_memory.id)
        assert retrieved is not None

        # Delete it
        deleted = await memory_manager.delete_memory(stored_memory.id)
        assert deleted is True

        # Verify it's gone
        retrieved = await memory_manager.get_memory(stored_memory.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_memory_nonexistent(self, memory_manager):
        """Test deleting a nonexistent memory."""
        deleted = await memory_manager.delete_memory("nonexistent-id")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_delete_memory_empty_file(self, memory_manager):
        """Test deleting from empty file."""
        deleted = await memory_manager.delete_memory("any-id")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_reset_memory_by_user(self, memory_manager, sample_memories):
        """Test resetting all memories for a user."""
        # Store all memories
        for memory in sample_memories:
            agent = self._build_agent(memory.agent_name)
            await memory_manager.write_single_memory(
                user_id=memory.user_id, agent=agent, memory_content=memory.memory
            )

        # Reset user1 memories
        await memory_manager.reset_memory("user1")

        # Verify user1 memories are gone
        user1_memories = await memory_manager.recall_memories("user1")
        assert len(user1_memories) == 0

        # Verify user2 memories still exist
        user2_memories = await memory_manager.recall_memories("user2")
        assert len(user2_memories) == 2

    @pytest.mark.asyncio
    async def test_reset_memory_by_user_and_agent(self, memory_manager, sample_memories):
        """Test resetting memories for specific user and agent."""
        # Store all memories
        for memory in sample_memories:
            agent = self._build_agent(memory.agent_name)
            await memory_manager.write_single_memory(
                user_id=memory.user_id, agent=agent, memory_content=memory.memory
            )

        # Reset user1 + agent1 memories only
        await memory_manager.reset_memory("user1", "agent1")

        # Verify only agent1 memories for user1 are gone
        user1_memories = await memory_manager.recall_memories("user1")
        assert len(user1_memories) == 1
        assert user1_memories[0].agent_name == "agent2"

        # Verify user2 memories are untouched
        user2_memories = await memory_manager.recall_memories("user2")
        assert len(user2_memories) == 2

    @pytest.mark.asyncio
    async def test_multiple_operations_integration(self, memory_manager):
        """Integration test with multiple operations."""
        # Create memories
        memory1 = Memory(timestamp=datetime.now(), user_id="test_user", memory="Memory 1")
        memory2 = Memory(timestamp=datetime.now(), user_id="test_user", memory="Memory 2")
        agent1 = self._build_agent(memory1.agent_name)
        agent2 = self._build_agent(memory2.agent_name)

        # Store memories
        await memory_manager.write_single_memory(
            user_id=memory1.user_id, agent=agent1, memory_content=memory1.memory
        )
        await memory_manager.write_single_memory(
            user_id=memory2.user_id, agent=agent2, memory_content=memory2.memory
        )

        # Recall all memories
        all_memories = await memory_manager.recall_memories("test_user")
        assert len(all_memories) == 2

        # Get specific memory - use the actual ID from stored memories
        retrieved = await memory_manager.get_memory(all_memories[0].id)
        assert retrieved is not None
        assert retrieved.id == all_memories[0].id

        # Delete one memory
        deleted = await memory_manager.delete_memory(all_memories[0].id)
        assert deleted is True

        # Verify only one remains
        remaining_memories = await memory_manager.recall_memories("test_user")
        assert len(remaining_memories) == 1
        assert remaining_memories[0].id == all_memories[1].id

        # Reset all memories
        await memory_manager.reset_memory("test_user")

        # Verify all gone
        final_memories = await memory_manager.recall_memories("test_user")
        assert len(final_memories) == 0

    @pytest.mark.asyncio
    async def test_malformed_lines_handling(self, memory_manager, temp_file):
        """Test handling of malformed JSON lines in file."""
        # Write a malformed line to the file
        with open(temp_file, "w") as f:
            f.write('{"invalid": "json"}\n')
            f.write("invalid line\n")

        # Should not crash on malformed lines
        memories = await memory_manager.recall_memories("any_user")
        assert memories == []

        memory_by_id = await memory_manager.get_memory("any_id")
        assert memory_by_id is None

    @pytest.mark.asyncio
    async def test_complete_reset_with_data(self, memory_manager, sample_memories):
        """Test complete reset with existing data."""
        # Store all memories
        for memory in sample_memories:
            agent = self._build_agent(memory.agent_name)
            await memory_manager.write_single_memory(
                user_id=memory.user_id, agent=agent, memory_content=memory.memory
            )

        # Verify data exists
        total_before = 0
        for i in range(3):
            user_memories = await memory_manager.recall_memories(f"user{i}", recency_limit=100)
            total_before += len(user_memories)
        assert total_before > 0

        # Complete reset
        await memory_manager.complete_reset()

        # Verify all data is gone
        total_after = 0
        for i in range(3):
            user_memories = await memory_manager.recall_memories(f"user{i}", recency_limit=100)
            total_after += len(user_memories)
        assert total_after == 0

    @pytest.mark.asyncio
    async def test_complete_reset_empty_file(self, memory_manager):
        """Test complete reset on empty/nonexistent file."""
        # Should not raise any exception
        await memory_manager.complete_reset()

        # Verify no memories exist
        memories = await memory_manager.recall_memories("any_user")
        assert memories == []

    @pytest.mark.asyncio
    async def test_complete_reset_integration(self, memory_manager):
        """Integration test: add data, reset, add more data."""
        # Add initial data
        memory1 = Memory(timestamp=datetime.now(), user_id="user1", memory="Initial memory")
        memory2 = Memory(timestamp=datetime.now(), user_id="user2", memory="Another memory")
        agent1 = self._build_agent(memory1.agent_name)
        agent2 = self._build_agent(memory2.agent_name)

        await memory_manager.write_single_memory(
            user_id=memory1.user_id, agent=agent1, memory_content=memory1.memory
        )
        await memory_manager.write_single_memory(
            user_id=memory2.user_id, agent=agent2, memory_content=memory2.memory
        )

        # Verify data exists
        memories = await memory_manager.recall_memories("user1", recency_limit=100)
        assert len(memories) == 1

        # Complete reset
        await memory_manager.complete_reset()

        # Verify all gone
        memories = await memory_manager.recall_memories("user1", recency_limit=100)
        assert len(memories) == 0
        memories = await memory_manager.recall_memories("user2", recency_limit=100)
        assert len(memories) == 0

        # Add new data after reset
        memory3 = Memory(timestamp=datetime.now(), user_id="user1", memory="Post-reset memory")
        agent3 = self._build_agent(memory3.agent_name)
        await memory_manager.write_single_memory(
            user_id=memory3.user_id, agent=agent3, memory_content=memory3.memory
        )

        # Verify new data works
        memories = await memory_manager.recall_memories("user1", recency_limit=100)
        assert len(memories) == 1
        assert memories[0].memory == "Post-reset memory"
