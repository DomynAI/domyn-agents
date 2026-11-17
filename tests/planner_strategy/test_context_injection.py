"""Tests for the shared per-turn context injection helper.

The point of the helper is that no planner can silently miss a source of injected
context. These tests assert the contract every planner relies on.
"""

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import Field

from domyn.agents import BaseEvent, ExecutionEventType, Part
from domyn.agents.core import InteractionContext
from domyn.agents.llm import LLMMessage
from domyn.agents.memory.managers.base import BaseMemoryManager, Memory
from domyn.agents.planner_strategy.context_injection import inject_turn_context, with_skills


class _Memory(BaseMemoryManager):
    """A real manager over a fixed list, recording how it is asked."""

    type: str = "injection-test"
    texts: list[str] = Field(default_factory=list)
    calls: list[dict[str, Any]] = Field(default_factory=list)

    def __init__(self, memories: list[str]) -> None:
        super().__init__(texts=memories, calls=[])

    async def recall_memories(self, user_id, agent=None, recency_limit=5, question="", **kwargs):
        self.calls.append({"user_id": user_id, "question": question, **kwargs})
        stamp = datetime(2026, 1, 1, tzinfo=UTC)
        return [Memory(id=f"m{i}", memory=t, timestamp=stamp) for i, t in enumerate(self.texts)]

    async def write_single_memory(self, user_id, agent, memory_content, **kwargs): ...

    async def get_memory(self, memory_id): ...

    async def delete_memory(self, memory_id):
        return False

    async def reset_memory(self, user_id, agent_name=None): ...

    async def complete_reset(self): ...


class _Agent:
    name = "A"

    def __init__(self, memory: _Memory | None = None) -> None:
        self.memory_manager = memory


def _context(**kwargs: Any) -> InteractionContext:
    return InteractionContext(
        conversation_id="c1",
        user_id="u1",
        agent_input=BaseEvent(
            event_type=ExecutionEventType.USER_INPUT,
            author="user",
            content=[Part(text="what is the policy?")],
        ),
        **kwargs,
    )


def _messages() -> list[LLMMessage]:
    return [
        LLMMessage(role="system", content="you are an agent"),
        LLMMessage(role="user", content="what is the policy?"),
    ]


class TestNoOpByDefault:
    @pytest.mark.asyncio
    async def test_agent_without_memory_or_skills_is_untouched(self) -> None:
        """Agents using neither feature must behave exactly as before."""
        messages = _messages()
        before = [m.content for m in messages]
        await inject_turn_context(messages, _Agent(), _context())
        assert [m.content for m in messages] == before


class TestMemory:
    @pytest.mark.asyncio
    async def test_recalled_memory_is_injected_after_the_system_message(self) -> None:
        messages = _messages()
        await inject_turn_context(
            messages, _Agent(_Memory(["user prefers metric units"])), _context()
        )
        assert messages[0].role == "system"
        assert "metric units" in (messages[1].content or "")

    @pytest.mark.asyncio
    async def test_empty_recall_injects_nothing(self) -> None:
        messages = _messages()
        await inject_turn_context(messages, _Agent(_Memory([])), _context())
        assert len(messages) == 2

    @pytest.mark.asyncio
    async def test_question_and_run_config_are_forwarded(self) -> None:
        memory = _Memory(["m"])
        await inject_turn_context(_messages(), _Agent(memory), _context())
        assert memory.calls[0]["question"] == "what is the policy?"
        assert "run_config" in memory.calls[0]
        assert memory.calls[0]["user_id"] == "u1"


class TestSkills:
    @pytest.mark.asyncio
    async def test_payload_goes_at_the_end_of_the_system_prompt(self) -> None:
        """Not as a message: an assistant message read as the model's own words."""
        messages = _messages()
        system = messages[0].content
        await inject_turn_context(
            messages, _Agent(), _context(), "Available skills:\n- sql-style: conventions"
        )
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[0].content == f"{system}\n\nAvailable skills:\n- sql-style: conventions"

    @pytest.mark.asyncio
    async def test_empty_payload_injects_nothing(self) -> None:
        messages = _messages()
        system = messages[0].content
        await inject_turn_context(messages, _Agent(), _context(), "")
        assert len(messages) == 2
        assert messages[0].content == system

    @pytest.mark.asyncio
    async def test_memory_is_a_message_and_skills_are_in_the_system_prompt(self) -> None:
        messages = _messages()
        await inject_turn_context(
            messages, _Agent(_Memory(["MEMORY BLOCK"])), _context(), "SKILLS BLOCK"
        )
        assert messages[0].role == "system"
        assert (messages[0].content or "").endswith("SKILLS BLOCK")
        assert "MEMORY BLOCK" in (messages[1].content or "")
        assert len(messages) == 3

    def test_a_cached_system_message_is_not_changed(self) -> None:
        """The system message is replaced, so the block is never appended twice."""
        system = LLMMessage(role="system", content="You are helpful.")
        messages = [system]
        with_skills(messages, "SKILLS")
        with_skills([system], "SKILLS")
        assert system.content == "You are helpful."
        assert messages[0].content == "You are helpful.\n\nSKILLS"

    def test_without_a_system_message_one_is_added(self) -> None:
        messages = [LLMMessage(role="user", content="hi")]
        with_skills(messages, "SKILLS")
        assert [(m.role, m.content) for m in messages] == [("system", "SKILLS"), ("user", "hi")]


class TestAllPlannersUseIt:
    """Every planner must call the helper, or it silently loses these features.

    This is the regression guard for the bug the helper exists to prevent: a
    planner keeping its own copy, or losing the call in a refactor.
    """

    @pytest.mark.parametrize("module", ["react", "tool_use", "tool_use_thought"])
    def test_planner_takes_the_payload_and_never_resolves_it(self, module: str) -> None:
        from pathlib import Path

        source = Path(f"domyn/agents/planner_strategy/{module}.py").read_text(encoding="utf-8")
        assert "skills_payload" in source, f"{module} never takes a skills payload"
        assert "resolve_skills_payload" not in source, f"{module} resolves skills itself"

    @pytest.mark.parametrize("module", ["react", "tool_use", "tool_use_thought"])
    def test_planner_has_no_private_copy(self, module: str) -> None:
        from pathlib import Path

        source = Path(f"domyn/agents/planner_strategy/{module}.py").read_text(encoding="utf-8")
        assert "recall_memories(" not in source, f"{module} still recalls memory on its own"
