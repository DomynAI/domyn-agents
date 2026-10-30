"""Tests for InMemoryConversationManager."""

from datetime import datetime, timedelta

import pytest

from domyn.agents import BaseEvent, ExecutionEventType
from domyn.agents.conversation_manager import (
    ConversationInfo,
    InMemoryConversationManager,
    StackAgentEntry,
)


@pytest.fixture
def manager():
    """Create a fresh InMemoryConversationManager for each test."""
    return InMemoryConversationManager()


@pytest.fixture
def sample_event():
    """Create a sample event for testing."""
    return BaseEvent(event_type=ExecutionEventType.USER_INPUT, author="user_1")


class TestInMemoryConversationManager:
    """Test suite for InMemoryConversationManager."""

    @pytest.mark.asyncio
    async def test_get_user_conversations(self, manager: InMemoryConversationManager):
        """Test retrieving user conversations."""
        app_name = "test_app"
        user_id = "user_1"
        initial_state = {"key": "value"}

        # Create conversation
        created_conversation_a = await manager.create_conversation(
            application_name=app_name, user_id=user_id, initial_state=initial_state
        )

        # Create conversation
        created_conversation_b = await manager.create_conversation(
            application_name=app_name, user_id=user_id, initial_state=initial_state
        )

        # Retrieve conversation
        retrieved_conversation = await manager.get_user_conversations(app_name, user_id)

        assert retrieved_conversation is not None
        assert len(retrieved_conversation) == 2
        assert retrieved_conversation[0].application_name == app_name
        assert retrieved_conversation[0].user_id == user_id
        assert retrieved_conversation[0].conversation_id == created_conversation_a.conversation_id
        assert retrieved_conversation[0].state == initial_state
        assert retrieved_conversation[1].application_name == app_name
        assert retrieved_conversation[1].user_id == user_id
        assert retrieved_conversation[1].conversation_id == created_conversation_b.conversation_id
        assert retrieved_conversation[1].state == initial_state

    @pytest.mark.asyncio
    async def test_get_user_conversations_empty(self, manager: InMemoryConversationManager):
        """Test retrieving user conversations."""
        app_name = "test_app"
        user_id = "user_1"

        # Retrieve conversation
        retrieved_conversation = await manager.get_user_conversations(app_name, user_id)

        assert len(retrieved_conversation) == 0

    @pytest.mark.asyncio
    async def test_get_conversation(self, manager: InMemoryConversationManager):
        """Test retrieving a conversation."""
        app_name = "test_app"
        user_id = "user_1"
        initial_state = {"key": "value"}

        # Create conversation
        created_conversation = await manager.create_conversation(
            application_name=app_name, user_id=user_id, initial_state=initial_state
        )

        # add stack entry
        assert await manager.add_agent_to_stack(
            application_name=app_name,
            user_id=user_id,
            conversation_id=created_conversation.conversation_id,
            agent_name="test_agent",
            interaction_id="test_interaction_id",
        )

        # Retrieve conversation
        retrieved_conversation = await manager.get_conversation(
            app_name, user_id, created_conversation.conversation_id
        )

        assert retrieved_conversation is not None
        assert retrieved_conversation.application_name == app_name
        assert retrieved_conversation.user_id == user_id
        assert retrieved_conversation.conversation_id == created_conversation.conversation_id
        assert retrieved_conversation.state == initial_state
        assert len(retrieved_conversation.agents_name_stack) == 1
        assert isinstance(retrieved_conversation.agents_name_stack[0], StackAgentEntry)
        assert retrieved_conversation.agents_name_stack[0].agent_name == "test_agent"

    @pytest.mark.asyncio
    async def test_add_conversation_event(self, manager, sample_event):
        """Test adding an event to conversation history."""
        app_name = "test_app"
        user_id = "user_1"

        # Create conversation
        conversation = await manager.create_conversation(app_name, user_id)

        # Add event
        success = await manager.add_conversation_event(
            app_name, user_id, conversation.conversation_id, sample_event
        )
        assert success

        # Verify event was added
        updated_conversation = await manager.get_conversation(
            app_name, user_id, conversation.conversation_id
        )
        assert len(updated_conversation.events_history) == 1
        assert updated_conversation.events_history[0].event_type == sample_event.event_type
        assert updated_conversation.events_history[0].author == sample_event.author

    @pytest.mark.asyncio
    async def test_update_conversation_state(self, manager: InMemoryConversationManager):
        """Test updating conversation state."""
        app_name = "test_app"
        user_id = "user_1"
        initial_state = {"key": "value"}

        # Create conversation
        conversation = await manager.create_conversation(
            app_name, user_id, initial_state=initial_state
        )

        # Update state
        new_state = {"new_key": "new_value", "key": "updated_value"}
        success = await manager.update_conversation_state(
            app_name, user_id, conversation.conversation_id, new_state
        )
        assert success

        # Retrieve and verify updated state
        updated_conversation = await manager.get_conversation(
            app_name, user_id, conversation.conversation_id
        )
        expected_state = {"key": "updated_value", "new_key": "new_value"}
        assert updated_conversation.state == expected_state

    @pytest.mark.asyncio
    async def test_delete_user_conversation(self, manager: InMemoryConversationManager):
        """Test deleting a user's conversations."""
        app_name = "test_app"
        user_id = "user_1"

        # Create conversation
        await manager.create_conversation(app_name, user_id)
        await manager.create_conversation(app_name, user_id)

        # Verify it exists
        exists = await manager.get_user_conversations(app_name, user_id)
        assert len(exists) == 2

        # Delete user conversations
        deleted_count = await manager.delete_user_conversations(app_name, user_id)
        assert deleted_count == 2

        # Verify it no longer exists
        exists = await manager.get_user_conversations(app_name, user_id)
        assert len(exists) == 0
