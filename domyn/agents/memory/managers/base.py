import asyncio
import re
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from domyn.agents.core import BaseAgent, ExecutionEventType, InteractionContext
from domyn.agents.decisions import runtime_of
from domyn.agents.decisions.memory import Verdict, decide, last_exchange
from domyn.agents.llm import LLMMessage
from domyn.agents.logger import get_logger
from domyn.agents.memory.render import MAX_ENTRY_CHARS, entry_line, render_block
from domyn.agents.memory.search import MemorySearch

# What a manager is asked for when it has no pages of its own: everything it has.
_READ_AT_MOST = 100_000
# Room in a page for its heading and its closing line.
_FRAME_CHARS = 400
# Turns whose recall is remembered, so the next request of a turn reuses it.
_CACHED_TURNS = 64


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


class MemoryPage(BaseModel):
    """One page of memories, and where the next one starts."""

    items: list[Memory] = Field(default_factory=list)
    next_cursor: str | None = None
    """Opaque: only the manager that issued it reads it. ``None``: there is no more."""
    total: int | None = None
    """How many there are in all, when the manager knows."""


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
    memory_writing_prompt: str = Field(
        default=(
            "You write memories about a user from a conversation. It has already been "
            "decided that the latest exchange reveals something about the user worth "
            "remembering in future conversations: who they are, a lasting preference, "
            "their situation or an ongoing goal. Write it as a concise, self-contained "
            "fact about the user (e.g. 'The user lives in Milan', not 'Milan'), in "
            "English regardless of the conversation language. Reply with the fact only, "
            "no preamble. If there is nothing worth remembering after all, reply NONE."
        ),
        description=(
            "System prompt used when a decision model has already said yes: the LLM "
            "only writes the memory."
        ),
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
        default=5,
        description="Maximum number of memories a direct call to recall_memories returns",
    )
    page_size: int = Field(
        default=20,
        gt=0,
        description="Memories per page, in the recall given to the model and in `recall`",
    )
    max_chars: int = Field(
        default=8000,
        gt=0,
        description="Characters of memories given to the model at the start of a turn",
    )
    search: MemorySearch = Field(
        default_factory=MemorySearch,
        description="How a query is matched against the memories (see memory/search.py)",
    )
    recall_timeout_s: float = Field(
        default=2.0,
        gt=0,
        description="How long the recall at the start of a turn may take before it is skipped",
    )

    _turns: OrderedDict[tuple[str | None, str | None, str], str] = PrivateAttr(
        default_factory=OrderedDict
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
        fact = " ".join(memory_content.split())
        if not fact:
            return "Nothing to remember: the text is empty."
        if len(fact) > MAX_ENTRY_CHARS:
            return (
                f"Too long to remember ({len(fact)} characters, at most {MAX_ENTRY_CHARS}): "
                "write one short, self-contained fact."
            )
        if await self._is_remembered(user_id, agent, fact):
            return "Already remembered."
        await self.write_single_memory(
            user_id=user_id,
            agent=agent,
            memory_content=fact,
            run_config=run_config,
        )
        return "Memory stored."

    async def _is_remembered(self, user_id: str, agent: BaseAgent, fact: str) -> bool:
        """Whether the same fact, give or take case and punctuation, is already there."""

        def normalized(text: str) -> str:
            return re.sub(r"[\W_]+$", "", " ".join(text.split()).casefold())

        try:
            page = await self.recall_page(user_id, agent, limit=100)
        except Exception as exc:
            get_logger().debug(f"memory: could not check for a repeat: {type(exc).__name__}")
            return False
        return normalized(fact) in {normalized(m.memory) for m in page.items}

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
        # The turn that just ended is in `interaction_history` when the post-agent
        # hook runs, not yet in `conversation_history`: reading only the latter gave
        # the LLM an empty conversation on a first turn, and it stored nothing.
        seen: set[str] = set()
        texts: list[str] = []
        for e in [*context.conversation_history, *context.interaction_history]:
            if e.event_type not in (ExecutionEventType.USER_INPUT, ExecutionEventType.RESPONSE):
                continue
            if e.event_id is not None:
                if e.event_id in seen:
                    continue
                seen.add(e.event_id)
            texts.extend(part.text or "" for part in e.content)
        conversation_parts = "\n".join(texts)

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

    async def write_memory_content(
        self,
        agent: BaseAgent,
        context: InteractionContext,
        exchange: tuple[str, str],
    ) -> str | None:
        """Have the LLM write the memory a decision model said the turn holds.

        Unlike :meth:`build_memory_content` it is not asked whether to store: that
        was decided. It may still answer ``NONE``, which stores nothing.
        """
        conversation = "\n".join(
            part.text or ""
            for e in context.conversation_history
            for part in e.content
            if e.event_type in [ExecutionEventType.USER_INPUT, ExecutionEventType.RESPONSE]
        )
        user, assistant = exchange
        llm_response = await agent.llm_provider.generate(
            messages=[
                LLMMessage(role="system", content=self.memory_writing_prompt),
                LLMMessage(
                    role="user",
                    content=(
                        f"Conversation:\n{conversation}\n\n"
                        f"Latest exchange:\nUser: {user}\nAssistant: {assistant}"
                    ),
                ),
            ],
            temperature=0.3,
            max_tokens=500,
        )
        content = (llm_response.content or "").strip()
        content = re.sub(rf"^{re.escape(self.memory_content_token)}:\s*", "", content, flags=re.I)
        if not content or content.rstrip(".").upper() == "NONE":
            return None
        return content

    async def _extract_memory_from_turn(
        self, agent: BaseAgent, context: InteractionContext
    ) -> str | None:
        """The memory to store for this turn, or ``None``.

        With a decision model whose ``memory`` use case is active, the model says
        whether and the LLM only writes; without one, or when its answer is not
        confident either way, the LLM decides and writes in one call as before.
        """
        runtime = runtime_of(agent)
        if runtime is None:
            return await self.build_memory_content(agent, context)
        exchange = last_exchange(context)
        verdict = await decide(runtime, user=exchange[0], assistant=exchange[1])
        if verdict is Verdict.SKIP:
            return None
        if verdict is Verdict.SAVE:
            return await self.write_memory_content(agent, context, exchange)
        return await self.build_memory_content(agent, context)

    async def update_memory(self, agent: BaseAgent, context: InteractionContext) -> None:
        """Update/store a memory. Errors are logged, never propagated."""
        try:
            memory_content = await self._extract_memory_from_turn(agent, context)
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

    async def recall_page(
        self,
        user_id: str,
        agent: BaseAgent | None = None,
        *,
        query: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
        **kwargs: Any,
    ) -> MemoryPage:
        """One page of memories, the most recent first or the best match for ``query``.

        The default is built on :meth:`recall_memories`, so every manager has pages
        and a search without changes: it reads what the manager has, ranks it with
        :attr:`search` and cuts it. A
        backend with pages of its own (a server) overrides this, and its ``cursor``
        is whatever it likes.
        """
        memories = await self.recall_memories(
            user_id=user_id,
            agent=agent,
            recency_limit=_READ_AT_MOST,
            question=query or "",
            **kwargs,
        )
        if query:
            # Off the event loop: on a large memory a search is CPU for a while.
            memories = await asyncio.to_thread(self.search.search, memories, query)
        return self.cut_page(memories, limit or self.page_size, cursor)

    @staticmethod
    def cut_page(memories: list[Memory], limit: int, cursor: str | None) -> MemoryPage:
        """A page of an offset cursor over a list already in order."""
        try:
            start = int(cursor) if cursor else 0
        except ValueError as exc:
            raise ValueError("not a cursor of this memory") from exc
        if start < 0:
            raise ValueError("not a cursor of this memory")
        window = memories[start : start + limit]
        more = start + len(window) < len(memories)
        return MemoryPage(
            items=window,
            next_cursor=str(start + len(window)) if more else None,
            total=len(memories),
        )

    async def _fitting_page(
        self,
        user_id: str,
        agent: BaseAgent | None,
        *,
        query: str | None,
        cursor: str | None,
        budget: int,
        **kwargs: Any,
    ) -> MemoryPage:
        """A page that fits in ``budget`` characters, with a cursor that follows it.

        When the page is too long the entries that fit are asked for again with that
        many as the limit, so the cursor of the second answer starts after the last
        one shown and not after the last one fetched.
        """
        page = await self.recall_page(user_id, agent, query=query, cursor=cursor, **kwargs)
        room = budget - _FRAME_CHARS
        used = 0
        fit = 0
        for memory in page.items:
            used += len(entry_line(memory)) + 1
            if used > room and fit > 0:
                break
            fit += 1
        if 0 < fit < len(page.items):
            return await self.recall_page(
                user_id, agent, query=query, limit=fit, cursor=cursor, **kwargs
            )
        return page

    async def recall_block(self, agent: BaseAgent, context: "InteractionContext") -> str:
        """The markdown the model is given at the start of the turn; empty for none.

        Asked once per turn, under a timeout: a second request of the same turn gets
        the same text, and a manager that is slow or fails means no memory this time,
        never a failed turn.
        """
        key = (
            getattr(context, "conversation_id", None),
            getattr(context, "turn_id", None) or getattr(context, "interaction_id", None),
            agent.name,
        )
        cacheable = key[1] is not None
        if cacheable and key in self._turns:
            self._turns.move_to_end(key)
            return self._turns[key]

        question = ""
        agent_input = getattr(context, "agent_input", None)
        if agent_input is not None and getattr(agent_input, "content", None):
            question = (agent_input.content[0].text or "").strip()
        block = ""
        try:
            async with asyncio.timeout(self.recall_timeout_s):
                # Passed on: a manager may filter on what the caller configured.
                extra = {"run_config": context.run_config}
                page = await self._fitting_page(
                    context.user_id,
                    agent,
                    query=question or None,
                    cursor=None,
                    budget=self.max_chars,
                    **extra,
                )
                if question and not page.items:
                    page = await self._fitting_page(
                        context.user_id,
                        agent,
                        query=None,
                        cursor=None,
                        budget=self.max_chars,
                        **extra,
                    )
            block = render_block(
                page.items, total=page.total, next_cursor=page.next_cursor, query=question or None
            )
        except Exception as exc:
            get_logger().warning(f"memory: recall skipped: {type(exc).__name__}")
            get_logger().debug(f"memory: recall failure detail: {exc}")
        if cacheable:
            self._turns[key] = block
            while len(self._turns) > _CACHED_TURNS:
                self._turns.popitem(last=False)
        return block

    async def recall(
        self,
        agent: BaseAgent,
        query: str = "",
        cursor: str = "",
        run_config: dict[str, Any] | None = None,
    ) -> str:
        """Read what is remembered about the user, a page at a time.

        Without arguments: the most recent. With a ``query``: the best matches. With
        the ``cursor`` of a previous answer: the next page.
        """
        user_id = run_config.get("user_id") if run_config else None
        if not user_id:
            return "Memory is not available: no user_id."
        asked = query.strip() or None
        try:
            page = await self._fitting_page(
                user_id, agent, query=asked, cursor=cursor or None, budget=20_000
            )
        except Exception as exc:
            get_logger().warning(f"memory: recall failed: {type(exc).__name__}")
            get_logger().debug(f"memory: recall failure detail: {exc}")
            return "Memory is not available right now."
        if not page.items:
            return f'No memory matches "{asked}".' if asked else "Nothing is remembered yet."
        return render_block(page.items, total=page.total, next_cursor=page.next_cursor, query=asked)

    async def forget(
        self, agent: BaseAgent, memory_id: str, run_config: dict[str, Any] | None = None
    ) -> str:
        """Remove one remembered fact, by the id shown with it."""
        if not (run_config or {}).get("user_id"):
            return "Memory is not available: no user_id."
        try:
            done = await self.delete_memory(memory_id.strip())
        except Exception as exc:
            get_logger().warning(f"memory: forget failed: {type(exc).__name__}")
            get_logger().debug(f"memory: forget failure detail: {exc}")
            return "Memory is not available right now: nothing was forgotten."
        return "Forgotten." if done else f"No memory has the id {memory_id!r}."

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
