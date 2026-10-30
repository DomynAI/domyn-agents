"""Base conversation manager."""

import copy
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from domyn.agents.core import BaseEvent


class StackAgentEntry(BaseModel):
    """Information about a single agent in the stack."""

    agent_name: str
    interaction_id: str | None = Field(default=None, description="Interaction ID")
    turn_id: str | None = Field(default=None, description="Turn ID")


class ConversationInfo(BaseModel):
    """Information about a single Conversation"""

    application_name: str
    user_id: str
    conversation_id: str
    agents_name_stack: list[StackAgentEntry] = Field(default_factory=list)
    # State kept for the whole conversation
    state: dict[str, Any] = Field(default_factory=dict)
    events_history: list[BaseEvent] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def update_state(self, new_state: dict[str, Any]) -> None:
        """Replace the conversation state with ``new_state`` and stamp the time.

        Replaces rather than merges, so a key the caller dropped is actually gone
        — see ``BaseConversationManager.update_conversation_state`` for why that
        is the contract.

        The copy is *deep* so what is stored is a snapshot. Callers hold
        ``context.state`` and keep mutating it long after this returns; assigning
        the argument directly would leave an in-memory store aliased to it,
        publishing later edits with no write behind them and drifting from the
        persistent backends, which can only ever see what was passed here.
        """
        self.state = copy.deepcopy(new_state)
        self.updated_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert ConversationInfo to dictionary."""
        return {
            "application_name": self.application_name,
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "state": self.state,
            "events_history": [event.to_dict() for event in self.events_history],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationInfo":
        """Create ConversationInfo from dictionary."""
        return cls(
            conversation_id=data["conversation_id"],
            application_name=data["application_name"],
            user_id=data["user_id"],
            agents_name_stack=[
                StackAgentEntry(**entry) for entry in data.get("agents_name_stack", [])
            ],
            state=data.get("state", {}),
            events_history=[
                BaseEvent(**event_data) for event_data in data.get("events_history", [])
            ],
            created_at=datetime.fromisoformat(data["created_at"])
            if isinstance(data.get("created_at"), str)
            else data.get("created_at", datetime.now()),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if isinstance(data.get("updated_at"), str)
            else data.get("updated_at", datetime.now()),
        )

    def copy(self) -> "ConversationInfo":
        """Create a copy of the ConversationInfo instance."""
        return ConversationInfo(
            application_name=self.application_name,
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            state=self.state.copy(),
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


class BaseConversationManager(ABC):
    """
    Base class for managing conversations between users and agents.

    A conversation represents a continuous interaction session between
    a user and the application. It maintains state across multiple interactions.
    """

    @abstractmethod
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
            conversation_id: Optional Conversation ID. If not provided, will be auto-generated
            initial_state: Optional initial state for the conversation

        Returns:
            ConversationInfo: Information about the created conversation
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    async def update_conversation_state(
        self, application_name: str, user_id: str, conversation_id: str, state: dict[str, Any]
    ) -> bool:
        """
        Replace the state of a conversation.

        Implementations must assign, not merge. Callers pass the whole of
        ``context.state``, so a key left out of it is being removed, and cleanup
        that pops a key relies on the removal reaching storage.

        Args:
            application_name: Name to identify the application
            user_id: ID of the user
            conversation_id: ID of the conversation to update
            state: The new state. It replaces the stored state, so any key
                omitted from it is removed.

        Returns:
            True if update was successful, False otherwise
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    async def delete_user_conversations(self, application_name: str, user_id: str) -> int:
        """
        Delete all conversations for a user.

        Args:
            application_name: Name to identify the application
            user_id: ID of the user

        Returns:
            Number of conversations deleted
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    async def add_agent_to_stack(
        self,
        application_name: str,
        user_id: str,
        conversation_id: str,
        agent_name: str,
        interaction_id: str | None = None,
        turn_id: str | None = None,
    ) -> bool:
        """
        Add an agent to the conversation's stack.

        Args:
            application_name: Name to identify the application
            user_id: ID of the user
            conversation_id: ID of the conversation
            agent_name: Name of the agent to add

        Returns:
            True if the agent was added successfully, False otherwise
        """
        ...

    @abstractmethod
    async def pop_agent_from_stack(
        self, application_name: str, user_id: str, conversation_id: str
    ) -> StackAgentEntry | None:
        """
        Remove and return the next agent from the conversation's stack.

        Args:
            application_name: Name to identify the application
            user_id: ID of the user
            conversation_id: ID of the conversation

        Returns:
            The name of the agent if one was removed, None otherwise
        """
        ...

    @abstractmethod
    async def empty_agents_stack(
        self, application_name: str, user_id: str, conversation_id: str
    ) -> bool:
        """
        Empty the conversation's agent stack.

        Args:
            application_name: Name to identify the application
            user_id: ID of the user
            conversation_id: ID of the conversation

        Returns:
            True if the queue was emptied successfully, False otherwise
        """
        ...

    @abstractmethod
    async def get_last_agent_stack(
        self, application_name: str, user_id: str, conversation_id: str
    ) -> StackAgentEntry | None:
        """
        Get the last agent added to the conversation's stack.

        Args:
            application_name: Name to identify the application
            user_id: ID of the user
            conversation_id: ID of the conversation

        Returns:
            Last agent
        """
        ...

    def _generate_conversation_id(self) -> str:
        return f"conversation_{uuid.uuid4().hex}"
