"""Sqlite implementation of conversation manager."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite

from domyn.agents.conversation_manager import BaseConversationManager, ConversationInfo
from domyn.agents.conversation_manager.base import StackAgentEntry
from domyn.agents.core import BaseEvent


def _stamp(moment: datetime | None = None) -> str:
    """A datetime as SQLite stores it here: text, space-separated.

    Written explicitly because Python 3.12 deprecated the implicit adapter that
    used to do it, and every write raised a DeprecationWarning. The format is
    byte for byte the one that adapter produced, so existing databases keep
    parsing — and, since the column is text, keep ordering: `cleanup_old_
    conversations` compares these values with `<`, and a row written in one
    format next to rows written in another compares on the separator character.
    """
    return (moment or datetime.now()).isoformat(sep=" ")


class AsyncSQLite:
    """
    A minimal aiosqlite context handler that removes boilerplate code
    and supports async operations.
    """

    def __init__(self, path: Path):
        self.path = path
        self.connection: aiosqlite.Connection | None = None
        self.cursor: aiosqlite.Cursor | None = None

    async def __aenter__(self):
        self.connection = await aiosqlite.connect(self.path)
        self.connection.row_factory = aiosqlite.Row
        self.cursor = await self.connection.cursor()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.connection.close()


class SqliteConversationManager(BaseConversationManager):
    """
    SQLite implementation of conversation manager.

    Stores all conversation data in a SQLite database. Data will persist across application restarts.
    Suitable for production environments where persistent storage is required.
    """

    def __init__(self, path: str = "./conversation.db") -> None:
        """Initialize the SQLite conversation manager."""
        self.path = path

    async def _create_table(self):
        async with AsyncSQLite(self.path) as db:
            await db.cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation (
                    application_name TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    agents_name_stack TEXT,
                    state TEXT,
                    events_history TEXT,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL,
                    PRIMARY KEY (application_name, user_id, conversation_id)
                )
            """)
            await db.connection.commit()

    async def create_conversation(
        self,
        application_name: str,
        user_id: str,
        conversation_id: str | None = None,
        initial_state: dict[str, Any] | None = None,
    ) -> ConversationInfo:
        """
        Create a new conversation for a user session.

        Args:
            application_name: Name to identify the application
            user_id: ID of the user
            conversation_id: Optional conversation ID; auto-generated if not provided
            initial_state: Optional initial state for the conversation

        Returns:
            ConversationInfo: Information about the created conversation
        """
        await self._create_table()
        conversation_info = ConversationInfo(
            conversation_id=conversation_id or self._generate_conversation_id(),
            application_name=application_name,
            user_id=user_id,
            state=initial_state or {},
        )

        async with AsyncSQLite(Path(self.path)) as db:
            await db.cursor.execute(
                """
                INSERT INTO conversation (application_name, user_id, conversation_id, agents_name_stack, state, events_history, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    application_name,
                    user_id,
                    conversation_info.conversation_id,
                    json.dumps([]),
                    json.dumps(conversation_info.state),
                    json.dumps([]),
                    _stamp(),
                    _stamp(),
                ),
            )
            await db.connection.commit()

        return conversation_info

    async def check_conversation_exists(
        self, application_name: str, user_id: str, conversation_id: str
    ) -> bool:
        """
        Check if a conversation exists.

        Args:
            application_name: Name to identify the application
            user_id: ID of the user
            conversation_id: ID of the conversation to check

        Returns:
            True if the conversation exists, False otherwise
        """
        await self._create_table()

        async with AsyncSQLite(Path(self.path)) as db:
            await db.cursor.execute(
                """
                SELECT 1 FROM conversation
                WHERE application_name = ? AND user_id = ? AND conversation_id = ?
            """,
                (application_name, user_id, conversation_id),
            )
            row = await db.cursor.fetchone()
            return row is not None

    async def get_conversation(
        self, application_name: str, user_id: str, conversation_id: str
    ) -> ConversationInfo | None:
        """
        Get a conversation by its ID.

        Args:
            application_name: Name to identify the application
            user_id: ID of the user
            conversation_id: ID of the conversation to retrieve

        Returns:
            ConversationInfo if found, None otherwise
        """
        await self._create_table()

        async with AsyncSQLite(Path(self.path)) as db:
            await db.cursor.execute(
                """
                SELECT * FROM conversation
                WHERE application_name = ? AND user_id = ? AND conversation_id = ?
            """,
                (application_name, user_id, conversation_id),
            )
            row = await db.cursor.fetchone()
            if row:
                return ConversationInfo(
                    application_name=row["application_name"],
                    user_id=row["user_id"],
                    conversation_id=row["conversation_id"],
                    agents_name_stack=json.loads(row["agents_name_stack"]),
                    state=json.loads(row["state"]),
                    events_history=[
                        BaseEvent.model_validate_json(e) for e in json.loads(row["events_history"])
                    ],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                )
            return None

    async def get_user_conversations(
        self, application_name: str, user_id: str
    ) -> list[ConversationInfo]:
        """
        Get all conversations for a specific user.

        Args:
            application_name: Name to identify the application
            user_id: ID of the user

        Returns:
            List of ConversationInfo objects
        """
        await self._create_table()
        async with AsyncSQLite(Path(self.path)) as db:
            await db.cursor.execute(
                """
                SELECT * FROM conversation
                WHERE application_name = ? AND user_id = ?
            """,
                (application_name, user_id),
            )
            rows = await db.cursor.fetchall()
            return (
                [
                    ConversationInfo(
                        application_name=row["application_name"],
                        user_id=row["user_id"],
                        conversation_id=row["conversation_id"],
                        agents_name_stack=json.loads(row["agents_name_stack"]),
                        state=json.loads(row["state"]),
                        events_history=[
                            BaseEvent.model_validate_json(e)
                            for e in json.loads(row["events_history"])
                        ],
                        created_at=datetime.fromisoformat(row["created_at"]),
                        updated_at=datetime.fromisoformat(row["updated_at"]),
                    )
                    for row in rows
                ]
                if rows
                else []
            )
        return []

    async def update_conversation_state(
        self, application_name: str, user_id: str, conversation_id: str, state: dict[str, Any]
    ) -> bool:
        """
        Replace the state of a conversation.

        Args:
            application_name: Name to identify the application
            user_id: ID of the user
            conversation_id: ID of the conversation to update
            state: The new state. It replaces the stored state, so any key
                omitted from it is removed.

        Returns:
            True if update was successful, False otherwise
        """
        await self._create_table()
        async with AsyncSQLite(Path(self.path)) as db:
            await db.cursor.execute(
                """
                UPDATE conversation
                SET state = ?, updated_at = ?
                WHERE application_name = ? AND user_id = ? AND conversation_id = ?
            """,
                (json.dumps(state), _stamp(), application_name, user_id, conversation_id),
            )
            await db.connection.commit()
            return db.cursor.rowcount > 0

    async def delete_conversation(
        self, application_name: str, user_id: str, conversation_id: str
    ) -> bool:
        """
        Delete a conversation.

        Args:
            application_name: Name to identify the application
            user_id: ID of the user
            conversation_id: ID of the conversation to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        await self._create_table()
        async with AsyncSQLite(Path(self.path)) as db:
            await db.cursor.execute(
                """
                DELETE FROM conversation
                WHERE application_name = ? AND user_id = ? AND conversation_id = ?
            """,
                (application_name, user_id, conversation_id),
            )
            await db.connection.commit()
            return db.cursor.rowcount > 0

    async def delete_user_conversations(self, application_name: str, user_id: str) -> int:
        """
        Delete all conversations for a user.

        Args:
            user_id: ID of the user
            application_name: Name to identify the application

        Returns:
            Number of conversations deleted
        """
        await self._create_table()
        async with AsyncSQLite(Path(self.path)) as db:
            await db.cursor.execute(
                """
                DELETE FROM conversation
                WHERE application_name = ? AND user_id = ?
            """,
                (application_name, user_id),
            )
            await db.connection.commit()
            return db.cursor.rowcount

    async def cleanup_old_conversations(
        self, older_than_days: int = 30, application_name: str | None = None
    ) -> int:
        """
        Clean up old conversations that haven't been updated in a while.

        Args:
            older_than_days: Delete conversations older than this many days
            application_name: Optional filter by application name

        Returns:
            Number of conversations deleted
        """
        await self._create_table()
        async with AsyncSQLite(Path(self.path)) as db:
            cutoff_date = _stamp(datetime.now() - timedelta(days=older_than_days))
            await db.cursor.execute(
                """
                DELETE FROM conversation
                WHERE updated_at < ? AND application_name = ?
            """,
                (cutoff_date, application_name),
            )
            await db.connection.commit()
            return db.cursor.rowcount

    async def add_conversation_event(
        self, application_name: str, user_id: str, conversation_id: str, event: BaseEvent
    ) -> bool:
        """
        Add an event to the conversation's event history.

        Args:
            application_name: Name to identify the application
            user_id: ID of the user
            conversation_id: ID of the conversation
            event: The event to add

        Returns:
            True if the event was added successfully, False otherwise
        """
        conversation_info = await self.get_conversation(application_name, user_id, conversation_id)
        if conversation_info is None:
            return False

        conversation_info.events_history.append(event)

        async with AsyncSQLite(Path(self.path)) as db:
            await db.cursor.execute(
                """
                UPDATE conversation
                SET events_history = ?, updated_at = ?
                WHERE application_name = ? AND user_id = ? AND conversation_id = ?
            """,
                (
                    json.dumps([e.model_dump_json() for e in conversation_info.events_history]),
                    _stamp(),
                    application_name,
                    user_id,
                    conversation_id,
                ),
            )
            await db.connection.commit()
            return db.cursor.rowcount > 0

    # stack operations

    async def add_agent_to_stack(
        self,
        application_name: str,
        user_id: str,
        conversation_id: str,
        agent_name: str,
        interaction_id: str | None = None,
        turn_id: str | None = None,
    ) -> bool:
        conversation_info = await self.get_conversation(application_name, user_id, conversation_id)
        if conversation_info is None:
            return False

        conversation_info.agents_name_stack.append(
            StackAgentEntry(agent_name=agent_name, interaction_id=interaction_id, turn_id=turn_id)
        )

        async with AsyncSQLite(Path(self.path)) as db:
            await db.cursor.execute(
                """
                UPDATE conversation
                SET agents_name_stack = ?, updated_at = ?
                WHERE application_name = ? AND user_id = ? AND conversation_id = ?
            """,
                (
                    json.dumps([e.model_dump() for e in conversation_info.agents_name_stack]),
                    _stamp(),
                    application_name,
                    user_id,
                    conversation_id,
                ),
            )
            await db.connection.commit()
            return db.cursor.rowcount > 0

    async def pop_agent_from_stack(
        self, application_name: str, user_id: str, conversation_id: str
    ) -> StackAgentEntry | None:
        conversation_info = await self.get_conversation(application_name, user_id, conversation_id)
        if conversation_info is None:
            return None

        if conversation_info and conversation_info.agents_name_stack:
            entry = conversation_info.agents_name_stack.pop(-1)
            async with AsyncSQLite(Path(self.path)) as db:
                await db.cursor.execute(
                    """
                    UPDATE conversation
                    SET agents_name_stack = ?, updated_at = ?
                    WHERE application_name = ? AND user_id = ? AND conversation_id = ?
                """,
                    (
                        json.dumps([e.model_dump() for e in conversation_info.agents_name_stack]),
                        _stamp(),
                        application_name,
                        user_id,
                        conversation_id,
                    ),
                )
                await db.connection.commit()
                return entry
        return None

    async def empty_agents_stack(
        self, application_name: str, user_id: str, conversation_id: str
    ) -> bool:
        conversation_info = await self.get_conversation(application_name, user_id, conversation_id)
        if conversation_info is None:
            return False

        if conversation_info and conversation_info.agents_name_stack:
            async with AsyncSQLite(Path(self.path)) as db:
                await db.cursor.execute(
                    """
                    UPDATE conversation
                    SET agents_name_stack = ?, updated_at = ?
                    WHERE application_name = ? AND user_id = ? AND conversation_id = ?
                """,
                    (json.dumps([]), _stamp(), application_name, user_id, conversation_id),
                )
                await db.connection.commit()
                return True
        return False

    async def get_last_agent_stack(
        self, application_name: str, user_id: str, conversation_id: str
    ) -> StackAgentEntry | None:
        conversation_info = await self.get_conversation(application_name, user_id, conversation_id)
        if conversation_info is None:
            return None

        if conversation_info is not None and conversation_info.agents_name_stack:
            return conversation_info.agents_name_stack[-1]
        return None


__all__ = ["SqliteConversationManager"]
