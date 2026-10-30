"""In-memory conversation manager."""

from datetime import datetime, timedelta
from typing import Any

from domyn.agents.conversation_manager import BaseConversationManager, ConversationInfo
from domyn.agents.conversation_manager.base import StackAgentEntry
from domyn.agents.core import BaseEvent


class InMemoryConversationManager(BaseConversationManager):
    """
    In-memory implementation of conversation manager.

    Stores all conversation data in memory. Data will be lost when the application restarts.
    Suitable for development, testing, or applications that don't require persistent storage.
    """

    def __init__(self) -> None:
        """Initialize the in-memory conversation manager."""

        # structure to hold conversations
        # application_name -> dict[user_id, dict[conversation_id, ConversationInfo]]
        self._conversations: dict[str, dict[str, dict[str, ConversationInfo]]] = {}

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
        conversation_info = ConversationInfo(
            conversation_id=conversation_id or self._generate_conversation_id(),
            application_name=application_name,
            user_id=user_id,
            state=initial_state or {},
        )

        # create in memory the conversation_info object
        if application_name not in self._conversations:
            self._conversations[application_name] = {}
        if user_id not in self._conversations[application_name]:
            self._conversations[application_name][user_id] = {}

        self._conversations[application_name][user_id][conversation_info.conversation_id] = (
            conversation_info
        )
        return conversation_info.copy()

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
        return (
            await self.get_conversation(
                application_name=application_name, user_id=user_id, conversation_id=conversation_id
            )
            is not None
        )

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
        _c = self._conversations.get(application_name, {}).get(user_id, {}).get(conversation_id)
        return _c.model_copy() if _c else None

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
        user_conversations = self._conversations.get(application_name, {}).get(user_id, {})
        if user_conversations:
            return sorted(
                [c.copy() for c in user_conversations.values()], key=lambda c: c.updated_at
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
        _c = self._conversations.get(application_name, {}).get(user_id, {}).get(conversation_id)
        if _c is None:
            return False

        _c.update_state(state)
        return True

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
        _c = self._conversations.get(application_name, {}).get(user_id, {}).get(conversation_id)
        if _c is not None:
            del self._conversations[application_name][user_id][conversation_id]
            return True
        return False

    async def delete_user_conversations(self, application_name: str, user_id: str) -> int:
        """
        Delete all conversations for a user.

        Args:
            user_id: ID of the user
            application_name: Name to identify the application

        Returns:
            Number of conversations deleted
        """
        deleted_count = 0

        app_conversations = self._conversations.get(application_name, {})
        if user_id in app_conversations:
            deleted_count = len(app_conversations[user_id])
            del app_conversations[user_id]

        return deleted_count

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
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        deleted_count = 0

        if application_name in self._conversations:
            for user_conversations in self._conversations[application_name].values():
                for conversation in user_conversations.values():
                    if conversation.updated_at < cutoff_date:
                        del self._conversations[application_name][user_conversations][
                            conversation.conversation_id
                        ]
                        deleted_count += 1
        return deleted_count

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
        _c = self._conversations.get(application_name, {}).get(user_id, {}).get(conversation_id)
        if _c is not None:
            _c.events_history.append(event)
            return True
        return False

    async def add_agent_to_stack(
        self,
        application_name: str,
        user_id: str,
        conversation_id: str,
        agent_name: str,
        interaction_id: str | None = None,
        turn_id: str | None = None,
    ) -> bool:
        _c = await self.get_conversation(
            application_name=application_name, user_id=user_id, conversation_id=conversation_id
        )
        if _c is not None:
            _c.agents_name_stack.append(
                StackAgentEntry(
                    agent_name=agent_name, interaction_id=interaction_id, turn_id=turn_id
                )
            )
            return True
        return False

    async def pop_agent_from_stack(
        self, application_name: str, user_id: str, conversation_id: str
    ) -> StackAgentEntry | None:
        _c = await self.get_conversation(
            application_name=application_name, user_id=user_id, conversation_id=conversation_id
        )
        if _c is not None and _c.agents_name_stack:
            return _c.agents_name_stack.pop(-1)
        return None

    async def empty_agents_stack(
        self, application_name: str, user_id: str, conversation_id: str
    ) -> bool:
        _c = await self.get_conversation(
            application_name=application_name, user_id=user_id, conversation_id=conversation_id
        )
        if _c is not None:
            _c.agents_name_stack.clear()
            return True
        return False

    async def get_last_agent_stack(
        self, application_name: str, user_id: str, conversation_id: str
    ) -> StackAgentEntry | None:
        _c = await self.get_conversation(
            application_name=application_name, user_id=user_id, conversation_id=conversation_id
        )
        if _c is not None and _c.agents_name_stack:
            return _c.agents_name_stack[-1]
        return None
