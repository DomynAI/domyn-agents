"""Tests for SqliteConversationManager."""

import json
import random
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

import pytest
import pytest_asyncio

from domyn.agents.conversation_manager import (
    SqliteConversationManager,
    StackAgentEntry,
)
from domyn.agents.conversation_manager.sqlite import AsyncSQLite
from domyn.agents.core import (
    Action,
    AgentAction,
    BaseAction,
    BaseEvent,
    ExecutionEventType,
    Part,
    StopAction,
    ToolAction,
)


@pytest_asyncio.fixture
async def manager():
    """Create a fresh SqliteConversationManager with a temporary database for each test."""
    # Create a temporary file for the database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
        temp_db_path = temp_db.name

    manager = SqliteConversationManager(path=temp_db_path)

    # Initialize the database tables
    await manager._create_table()

    yield manager

    # Cleanup: remove the temporary database file
    Path(temp_db_path).unlink(missing_ok=True)


@pytest.fixture
def sample_action() -> BaseAction:
    """Create a sample action for testing."""

    match random.choice(["ToolAction"]):
        case "ToolAction":
            return ToolAction(
                name="test_tool",
                parameters={"param1": "value1"},
                thought="Thinking about using the tool",
                observation={"result": "success"},
                call_id="call_12345",
            )
        case "StopAction":
            return StopAction(
                name="stop",
                message="Agent has stopped",
                thought="Stopping the agent",
                parameters={},
            )
        case "AgentAction":
            return AgentAction(
                name="test_agent",
                parameters={"param1": "value1"},
                thought="Thinking about the agent's action",
                observation={"result": "success"},
                message="Agent action executed successfully",
            )


@pytest.fixture
def sample_event(sample_action) -> BaseEvent:
    return BaseEvent(
        event_type=random.choice(list(ExecutionEventType)),
        author=random.choice(["user", "agent", "system"]),
        content=[Part(type="text", text="Hello, world!")],
        is_final=random.choice([True, False]),
        is_partial=random.choice([True, False]),
        interaction_id=f"interaction_{random.randint(1, 1000)}",
        action=sample_action,
    )


class TestSqliteConversationManager:
    """Test suite for SqliteConversationManager."""

    @pytest.mark.asyncio
    async def test_get_conversation(self, manager: SqliteConversationManager):
        """Test retrieving a conversation."""
        app_name = "test_app"
        user_id = "user_1"
        initial_state = {"key": "value"}

        # Create conversation
        created_conversation = await manager.create_conversation(
            application_name=app_name, user_id=user_id, initial_state=initial_state
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

    @pytest.mark.asyncio
    async def test_get_user_conversations(self, manager: SqliteConversationManager):
        """Test retrieving a user's conversations."""
        app_name = "test_app"
        user_id = "user_1"
        initial_state_a = {"key": "value"}
        initial_state_b = {"another_key": "another_value"}

        # Create conversation
        created_conversation_a = await manager.create_conversation(
            application_name=app_name, user_id=user_id, initial_state=initial_state_a
        )

        created_conversation_b = await manager.create_conversation(
            application_name=app_name, user_id=user_id, initial_state=initial_state_b
        )

        # Retrieve conversation
        retrieved_conversation_a = await manager.get_conversation(
            app_name, user_id, created_conversation_a.conversation_id
        )

        retrieved_conversation_b = await manager.get_conversation(
            app_name, user_id, created_conversation_b.conversation_id
        )

        assert retrieved_conversation_a is not None
        assert retrieved_conversation_a.application_name == app_name
        assert retrieved_conversation_a.user_id == user_id
        assert retrieved_conversation_a.conversation_id == created_conversation_a.conversation_id

        assert retrieved_conversation_b is not None
        assert retrieved_conversation_b.application_name == app_name
        assert retrieved_conversation_b.user_id == user_id
        assert retrieved_conversation_b.conversation_id == created_conversation_b.conversation_id
        assert retrieved_conversation_b.state == initial_state_b

        assert retrieved_conversation_a != retrieved_conversation_b
        assert retrieved_conversation_a.conversation_id != retrieved_conversation_b.conversation_id

    @pytest.mark.asyncio
    async def test_get_user_conversations_empty(self, manager: SqliteConversationManager):
        """Test retrieving a user's conversations."""
        app_name = "test_app"
        user_id = "user_1"

        # Retrieve conversation
        retrieved_conversation_a = await manager.get_user_conversations(app_name, user_id)

        assert len(retrieved_conversation_a) == 0

    @pytest.mark.asyncio
    async def test_add_conversation_event(self, manager: SqliteConversationManager, sample_event):
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

        # The event might not be exactly equal due to JSON serialization/deserialization
        # but the important data should be preserved
        assert updated_conversation.events_history[0].event_type == sample_event.event_type
        assert updated_conversation.events_history[0].author == sample_event.author
        assert updated_conversation.events_history[0] == sample_event
        assert updated_conversation.events_history[0].action == sample_event.action

    @pytest.mark.asyncio
    async def test_add_conversation_event_with_metadata(
        self, manager: SqliteConversationManager, sample_event
    ):
        """Test adding an event to conversation history."""
        app_name = "test_app"
        user_id = "user_1"

        # Create conversation
        conversation = await manager.create_conversation(app_name, user_id)

        # add metadata
        sample_event.metadata = {"key1": "value1", "key2": 2}

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

        # The event might not be exactly equal due to JSON serialization/deserialization
        # but the important data should be preserved
        assert updated_conversation.events_history[0].event_type == sample_event.event_type
        assert updated_conversation.events_history[0].author == sample_event.author
        assert updated_conversation.events_history[0] == sample_event
        assert updated_conversation.events_history[0].action == sample_event.action
        assert updated_conversation.events_history[0].metadata == sample_event.metadata

    @pytest.mark.asyncio
    async def test_update_conversation_state(self, manager: SqliteConversationManager):
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
        assert updated_conversation.state == new_state

    @pytest.mark.asyncio
    async def test_delete_user_conversations(self, manager: SqliteConversationManager):
        """Test deleting a user's conversations."""
        app_name = "test_app"
        user_id = "user_1"

        # Create conversation
        await manager.create_conversation(app_name, user_id)
        await manager.create_conversation(app_name, user_id)

        # Verify it exists
        exists = await manager.get_user_conversations(app_name, user_id)
        assert len(exists) == 2

        # Delete conversation
        success = await manager.delete_user_conversations(app_name, user_id)
        assert success == 2

        # Verify it no longer exists
        exists = await manager.get_user_conversations(app_name, user_id)
        assert len(exists) == 0

    @pytest.mark.asyncio
    async def test_add_agent_to_stack(self, manager: SqliteConversationManager):
        """Test adding an agent to the conversation stack."""
        app_name = "test_app"
        user_id = "user_1"
        agent_name = "test_agent"
        interaction_id = "interaction_123"

        # Create conversation
        conversation = await manager.create_conversation(app_name, user_id)

        # Add agent to stack
        success = await manager.add_agent_to_stack(
            app_name, user_id, conversation.conversation_id, agent_name, interaction_id
        )
        assert success

        # Verify agent was added
        updated_conversation = await manager.get_conversation(
            app_name, user_id, conversation.conversation_id
        )
        assert len(updated_conversation.agents_name_stack) == 1
        assert isinstance(updated_conversation.agents_name_stack[0], StackAgentEntry)
        assert updated_conversation.agents_name_stack[0].agent_name == agent_name
        assert updated_conversation.agents_name_stack[0].interaction_id == interaction_id

    @pytest.mark.asyncio
    async def test_last_agent_stack(self, manager: SqliteConversationManager):
        """Test retrieving the last agent from the conversation stack."""
        app_name = "test_app"
        user_id = "user_1"
        agent_name = "test_agent"
        interaction_id = "interaction_123"

        # Create conversation
        conversation = await manager.create_conversation(app_name, user_id)

        # Add agent to stack
        success = await manager.add_agent_to_stack(
            app_name, user_id, conversation.conversation_id, agent_name, interaction_id
        )
        assert success

        # Verify agent was added
        last_agent_stack = await manager.get_last_agent_stack(
            app_name, user_id, conversation.conversation_id
        )
        assert last_agent_stack is not None
        assert last_agent_stack.agent_name == agent_name
        assert last_agent_stack.interaction_id == interaction_id

        # verify stack size remains unchanged
        conversation_info = await manager.get_conversation(
            app_name, user_id, conversation.conversation_id
        )
        assert len(conversation_info.agents_name_stack) == 1

    @pytest.mark.asyncio
    async def test_pop_agent_from_stack(self, manager: SqliteConversationManager):
        """Test popping an agent from the conversation stack."""
        app_name = "test_app"
        user_id = "user_1"
        agent_name = "test_agent"
        interaction_id = "interaction_123"

        # Create conversation
        conversation = await manager.create_conversation(app_name, user_id)

        # Add agent to stack first
        await manager.add_agent_to_stack(
            app_name, user_id, conversation.conversation_id, agent_name, interaction_id
        )

        # Pop agent from stack
        popped_agent = await manager.pop_agent_from_stack(
            app_name, user_id, conversation.conversation_id
        )

        assert popped_agent is not None
        assert popped_agent.agent_name == agent_name
        assert popped_agent.interaction_id == interaction_id

        # Verify stack is now empty
        updated_conversation = await manager.get_conversation(
            app_name, user_id, conversation.conversation_id
        )
        assert len(updated_conversation.agents_name_stack) == 0

    @pytest.mark.asyncio
    async def test_empty_agent_stack(self, manager: SqliteConversationManager):
        """Test emptying agents stack"""
        app_name = "test_app"
        user_id = "user_1"
        agent_name = "test_agent"
        interaction_id = "interaction_123"

        # Create conversation
        conversation = await manager.create_conversation(app_name, user_id)

        # Add agent to stack
        success = await manager.add_agent_to_stack(
            app_name, user_id, conversation.conversation_id, agent_name, interaction_id
        )
        assert success

        resp = await manager.empty_agents_stack(app_name, user_id, conversation.conversation_id)
        assert resp

        # verify stack size remains unchanged
        conversation_info = await manager.get_conversation(
            app_name, user_id, conversation.conversation_id
        )
        assert len(conversation_info.agents_name_stack) == 0

    @pytest.mark.asyncio
    async def test_persistence_across_instances(self):
        """Test that data persists across different manager instances."""
        # Create a temporary database file
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
            temp_db_path = temp_db.name

        try:
            app_name = "test_app"
            user_id = "user_1"

            # Create first manager instance and add data
            manager1 = SqliteConversationManager(path=temp_db_path)
            conversation = await manager1.create_conversation(
                app_name, user_id, initial_state={"key": "value"}
            )

            # Create second manager instance (simulating app restart)
            manager2 = SqliteConversationManager(path=temp_db_path)

            # Verify data persists
            retrieved_conversation = await manager2.get_conversation(
                app_name, user_id, conversation.conversation_id
            )

            assert retrieved_conversation is not None
            assert retrieved_conversation.conversation_id == conversation.conversation_id
            assert retrieved_conversation.state == {"key": "value"}

        finally:
            # Cleanup
            Path(temp_db_path).unlink(missing_ok=True)


class TestTimestampsAreWrittenExplicitly:
    """Python 3.12 deprecated the implicit datetime adapter, and every write
    here went through it. Writing the text ourselves has to leave the stored
    format alone: it is a text column, and it is compared with `<`.
    """

    @pytest.mark.asyncio
    async def test_no_deprecation_is_raised_on_a_write(
        self, manager: SqliteConversationManager
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            await manager.create_conversation("app", "user")

    @pytest.mark.asyncio
    async def test_the_stored_format_did_not_change(
        self, manager: SqliteConversationManager
    ) -> None:
        """Byte for byte what the old adapter produced, so a database written by
        an earlier version still parses and still sorts against new rows."""
        import datetime as dt
        import sqlite3

        info = await manager.create_conversation("app", "user")

        async with AsyncSQLite(Path(manager.path)) as db:
            await db.cursor.execute(
                "SELECT created_at FROM conversation WHERE conversation_id = ?",
                (info.conversation_id,),
            )
            stored = (await db.cursor.fetchone())["created_at"]

        assert isinstance(stored, str)
        parsed = dt.datetime.fromisoformat(stored)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            adapter = sqlite3.adapters[(dt.datetime, sqlite3.PrepareProtocol)]
            assert adapter(parsed) == stored

    @pytest.mark.asyncio
    async def test_a_row_written_the_old_way_is_still_read(
        self, manager: SqliteConversationManager
    ) -> None:
        """The upgrade case: rows already on disk were written by the adapter."""
        import datetime as dt

        written = dt.datetime(2026, 1, 2, 3, 4, 5, 678901)
        async with AsyncSQLite(Path(manager.path)) as db:
            await db.cursor.execute(
                "INSERT INTO conversation VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "app",
                    "user",
                    "conv_old",
                    json.dumps([]),
                    json.dumps({}),
                    json.dumps([]),
                    written.isoformat(sep=" "),
                    written.isoformat(sep=" "),
                ),
            )
            await db.connection.commit()

        found = await manager.get_conversation("app", "user", "conv_old")

        assert found is not None
        assert found.created_at == written

    @pytest.mark.asyncio
    async def test_old_and_new_rows_order_against_each_other(
        self, manager: SqliteConversationManager
    ) -> None:
        """The cleanup query compares timestamps with `<` on a text column, so
        two formats in one table would sort on the separator character."""
        import datetime as dt

        old = dt.datetime.now() - dt.timedelta(days=90)
        async with AsyncSQLite(Path(manager.path)) as db:
            await db.cursor.execute(
                "INSERT INTO conversation VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "app",
                    "user",
                    "conv_ancient",
                    json.dumps([]),
                    json.dumps({}),
                    json.dumps([]),
                    old.isoformat(sep=" "),
                    old.isoformat(sep=" "),
                ),
            )
            await db.connection.commit()
        fresh = await manager.create_conversation("app", "user")

        removed = await manager.cleanup_old_conversations(
            older_than_days=30, application_name="app"
        )

        assert removed == 1
        assert await manager.get_conversation("app", "user", "conv_ancient") is None
        assert await manager.get_conversation("app", "user", fresh.conversation_id) is not None
