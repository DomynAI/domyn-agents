"""What every conversation manager must do, run against each implementation.

Behaviour specific to one store (sqlite's persistence and timestamps, the in-memory
listing) stays in that store's own test file.
"""

from datetime import datetime
from pathlib import Path

import pytest
import pytest_asyncio

from domyn.agents import BaseEvent, ExecutionEventType
from domyn.agents.conversation_manager import (
    BaseConversationManager,
    InMemoryConversationManager,
    SqliteConversationManager,
)


@pytest_asyncio.fixture(params=["in_memory", "sqlite"])
async def manager(request: pytest.FixtureRequest, tmp_path: Path) -> BaseConversationManager:
    if request.param == "in_memory":
        return InMemoryConversationManager()
    manager = SqliteConversationManager(path=str(tmp_path / "conversations.db"))
    await manager._create_table()
    return manager


@pytest.fixture
def sample_event() -> BaseEvent:
    return BaseEvent(event_type=ExecutionEventType.USER_INPUT, author="user_1")


class TestEveryConversationManager:
    @pytest.mark.asyncio
    async def test_create_conversation(self, manager: BaseConversationManager):
        """Test creating a new conversation."""
        app_name = "test_app"
        user_id = "user_1"
        initial_state = {"key": "value"}

        conversation = await manager.create_conversation(
            application_name=app_name, user_id=user_id, initial_state=initial_state
        )

        assert conversation.application_name == app_name
        assert conversation.user_id == user_id
        assert conversation.state == initial_state
        assert conversation.conversation_id is not None
        assert conversation.conversation_id.startswith("conversation_")
        assert conversation.agents_name_stack == []
        assert conversation.events_history == []
        assert isinstance(conversation.created_at, datetime)
        assert isinstance(conversation.updated_at, datetime)

    @pytest.mark.asyncio
    async def test_create_conversation_with_custom_id(self, manager: BaseConversationManager):
        """Test creating a conversation with a custom ID."""
        app_name = "test_app"
        user_id = "user_1"
        custom_id = "custom_conversation_id"

        conversation = await manager.create_conversation(
            application_name=app_name, user_id=user_id, conversation_id=custom_id
        )

        assert conversation.conversation_id == custom_id

    @pytest.mark.asyncio
    async def test_check_conversation_exists(self, manager: BaseConversationManager):
        """Test checking if a conversation exists."""
        app_name = "test_app"
        user_id = "user_1"

        # Conversation doesn't exist initially
        exists = await manager.check_conversation_exists(app_name, user_id, "nonexistent")
        assert not exists

        # Create conversation
        conversation = await manager.create_conversation(app_name, user_id)

        # Now it should exist
        exists = await manager.check_conversation_exists(
            app_name, user_id, conversation.conversation_id
        )
        assert exists

    @pytest.mark.asyncio
    async def test_get_nonexistent_conversation(self, manager: BaseConversationManager):
        """Test retrieving a nonexistent conversation."""
        conversation = await manager.get_conversation("app", "user", "nonexistent")
        assert conversation is None

    @pytest.mark.asyncio
    async def test_add_conversation_event_with_no_conversation(
        self, manager: BaseConversationManager, sample_event: BaseEvent
    ):
        """Adding an event to a conversation that does not exist fails."""
        success = await manager.add_conversation_event(
            "test_app", "user_1", "nonexistent_conversation", sample_event
        )
        assert not success

    @pytest.mark.asyncio
    async def test_delete_conversation(self, manager: BaseConversationManager):
        """Test deleting a conversation."""
        app_name = "test_app"
        user_id = "user_1"

        # Create conversation
        conversation = await manager.create_conversation(app_name, user_id)

        # Verify it exists
        exists = await manager.check_conversation_exists(
            app_name, user_id, conversation.conversation_id
        )
        assert exists

        # Delete conversation
        success = await manager.delete_conversation(app_name, user_id, conversation.conversation_id)
        assert success

        # Verify it no longer exists
        exists = await manager.check_conversation_exists(
            app_name, user_id, conversation.conversation_id
        )
        assert not exists
