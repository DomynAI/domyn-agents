import re
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from domyn.agents.core import BaseAgent, ExecutionEventType, InteractionContext
from domyn.agents.llm import LLMMessage
from domyn.agents.logger import get_logger


class MemoryExecutionMode(Enum):
    AUTOMATIC = "automatic"
    AGENTIC = "agentic"


class Memory(BaseModel):
    """Base memory model.

    Contains only framework-generic fields. Platform-specific metadata
    (e.g. tenant IDs, configuration IDs) should be added by subclassing
    this model in the service layer.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime
    memory: str
    agent_name: str | None = None
    user_id: str | None = None


class BaseMemoryManager(BaseModel, ABC):
    """Abstract base class for memory managers."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # register subclasses
    _registry: ClassVar[dict[str, type["BaseMemoryManager"]]] = {}
    type: str = Field(..., description="Type identifier for the memory manager")

    execution_mode: MemoryExecutionMode = Field(
        default=MemoryExecutionMode.AUTOMATIC, description="Execution mode for memory management"
    )

    # ── Configurable prompt and parsing tokens for build_memory_content ──
    memory_extraction_prompt: str = Field(
        default=(
            "You are an assistant that extracts and summarizes user preferences and important "
            "information about the user based on their inputs. Provide a concise summary that "
            "can be used as memory for future interactions. Store information on the user, their "
            "preferences, such as name, location, etc, or anything else that might be relevant "
            "in the future. Don't store comments like 'This information is relevant', just store "
            "the information. Always write the memory content in English, regardless of the "
            "conversation language. Provide a yes/no telling whether you have to store relevant "
            "user info or preferences, and then the information to store. Reply in the following "
            "structure:\nStoreMemory: <yes/no>\n MemoryContent: <summary>"
        ),
        description="System prompt for memory extraction. Must produce output matching the parsing tokens.",
    )
    memory_store_token: str = Field(
        default="StoreMemory",
        description="Token prefix for the store decision (e.g. 'StoreMemory')",
    )
    memory_store_positive: str = Field(
        default="yes", description="Positive value for the store decision (e.g. 'yes')"
    )
    memory_content_token: str = Field(
        default="MemoryContent",
        description="Token prefix for the memory content (e.g. 'MemoryContent')",
    )
    recency_limit: int = Field(
        default=5, description="Maximum number of memories to recall per interaction"
    )

    # auto-register sub types
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        tag = getattr(cls, "type", None)
        if tag is not None and hasattr(cls, "type") and cls.type != "BaseMemoryManager":
            # Extract the literal value if it's a Literal type
            if hasattr(cls.__annotations__.get("type", None), "__args__"):
                tag = cls.__annotations__["type"].__args__[0]
            BaseMemoryManager._registry[tag] = cls

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BaseMemoryManager":
        """
        Create a memory manager instance from configuration dictionary.

        Uses model_validate to leverage Pydantic's field-aware parsing
        (aliases, coercion, nested models). Unknown keys are ignored.
        Subclasses can override this method to provide custom configuration logic.
        """
        return cls.model_validate({k: v for k, v in config.items() if k != "type"})

    async def remember(
        self, agent: BaseAgent, memory_content: str, run_config: dict[str, Any] | None = None
    ) -> str:
        logger = get_logger()
        """
        Store information about the user for future interactions.

        Args:
            agent (BaseAgent): The agent instance that created this memory.
            memory_content (str): The content of the memory to store. Should be a complete,
                self-contained fact (e.g. 'The user lives in Italy' not just 'Italy').

        Returns:
            str: Confirmation message.
        """
        user_id = run_config.get("user_id") if run_config else None
        if not user_id:
            logger.warning("remember called without user_id in run_config — memory not stored")
            return "Memory not stored: no user_id available."
        await self.write_single_memory(
            user_id=user_id,
            agent=agent,
            memory_content=memory_content,
            run_config=run_config,
        )
        return "Memory stored."

    @abstractmethod
    async def write_single_memory(
        self,
        user_id: str,
        agent: BaseAgent,
        memory_content: str,
        **kwargs: Any,
    ) -> None:
        """Write a single memory entry.

        Args:
            user_id: The user this memory belongs to.
            memory_content: The memory text to store.
            agent: The agent instance that created this memory.
            **kwargs: Platform-specific metadata (e.g. run_config, tenant IDs).
                      Subclasses extract what they need.
        """
        ...

    async def build_memory_content(
        self,
        agent: BaseAgent,
        context: InteractionContext,
    ) -> str | None:
        """Extract user preferences from a conversation using the LLM.

        The extraction prompt and parsing tokens are configurable via class
        attributes (memory_extraction_prompt, memory_store_token, etc.).
        """
        conversation_parts = "\n".join(
            part.text or ""
            for e in context.conversation_history
            for part in e.content
            if e.event_type in [ExecutionEventType.USER_INPUT, ExecutionEventType.RESPONSE]
        )

        llm_response = await agent.llm_provider.generate(
            messages=[
                LLMMessage(role="system", content=self.memory_extraction_prompt),
                LLMMessage(role="user", content=f"Conversation:\n{conversation_parts}"),
            ],
            temperature=0.3,
            max_tokens=500,
        )

        content = llm_response.content
        store_pattern = rf"{re.escape(self.memory_store_token)}:\s*(\S+)"
        store_match = re.search(store_pattern, content, re.IGNORECASE)
        if (
            not store_match
            or store_match.group(1).strip().lower() != self.memory_store_positive.lower()
        ):
            return None
        content_pattern = rf"{re.escape(self.memory_content_token)}:\s*(.+)"
        content_match = re.search(content_pattern, content, re.DOTALL | re.IGNORECASE)
        if not content_match:
            return None
        return content_match.group(1).strip() or None

    async def update_memory(self, agent: BaseAgent, context: InteractionContext) -> None:
        """Update/store a memory. Errors are logged, never propagated."""
        try:
            memory_content = await self.build_memory_content(agent, context)
            if memory_content:
                await self.write_single_memory(
                    user_id=context.user_id,
                    agent_name=agent.name,
                    memory_content=memory_content,
                    agent=agent,
                    run_config=context.run_config,
                )
        except Exception as e:
            get_logger().warning(f"update_memory failed, skipping: {type(e).__name__}")
            get_logger().debug(f"update_memory detail: {e}", exc_info=True)

    @abstractmethod
    async def recall_memories(
        self,
        user_id: str,
        agent: BaseAgent | None = None,
        recency_limit: int = 5,
        question: str = "",
        **kwargs: Any,
    ) -> list[Memory]:
        """Retrieve memories for a user.

        Args:
            user_id: The user to retrieve memories for.
            agent: If provided, filter to memories from this agent and its sub-agents.
            recency_limit: Maximum number of memories to return.
            question: Current user query, used for semantic retrieval by backends that support it.
            **kwargs: Platform-specific filters (e.g. run_config, workspace scoping).
                      Subclasses extract what they need.
        """
        ...

    @abstractmethod
    async def get_memory(self, memory_id: str) -> Memory | None:
        """Retrieve a single memory by ID. Returns None if not found."""
        ...

    @abstractmethod
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a single memory by ID. Returns True if deleted, False if not found."""
        ...

    @abstractmethod
    async def reset_memory(self, user_id: str, agent_name: str | None = None) -> None:
        """Clear all memories for a user, optionally filtered by agent name."""
        ...

    @abstractmethod
    async def complete_reset(self) -> None:
        """Clear all memories for all users and all agents."""
        ...
