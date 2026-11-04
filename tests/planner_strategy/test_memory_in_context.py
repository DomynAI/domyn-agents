"""The recalled memories are counted apart in the context measurement.

What matters: the total does not change (they were always in it, inside the
messages), the count is there in both tool-use planners (the thought planner's loop
is a copy of its parent's, so a change applied to one only is a shipping bug), and
nothing shows for an agent without memory.
"""

import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from rich.console import Console

from domyn.agents.core import ExecutionEventType
from domyn.agents.interfaces.cli import context_view
from domyn.agents.interfaces.cli.theme import Theme
from domyn.agents.llm.base import LLMMessage, LLMResponse
from domyn.agents.memory.managers import FilesystemMemoryManager
from domyn.agents.planner_strategy.context_management.budget import (
    CONTEXT_USAGE_METADATA_KEY,
    ContextUsage,
    measure,
)
from domyn.agents.planner_strategy.context_management.compaction import Compactor
from domyn.agents.planner_strategy.tool_use import ToolUsePlannerStrategy
from domyn.agents.planner_strategy.tool_use_thought import ToolUseWithThoughtPlannerStrategy
from tests.planner_strategy.test_tool_use import make_context, make_mock_agent

PLANNERS = [ToolUsePlannerStrategy, ToolUseWithThoughtPlannerStrategy]


def _memory_message(entries: int = 3) -> LLMMessage:
    lines = ["## Remembered about the user", "", "Facts.", ""]
    lines += [f"- fact number {i} *(2026-10-12 · id `a{i}`)*" for i in range(entries)]
    return LLMMessage(role="assistant", content="\n".join(lines), part="memory")


class TestTheMeasurement:
    def test_the_total_is_the_same_with_or_without_the_split(self) -> None:
        plain = ContextUsage(message_tokens=1000, tool_definition_tokens=300)
        split = ContextUsage(
            message_tokens=1000, tool_definition_tokens=300, memory_tokens=200, memory_entries=3
        )
        assert split.total_tokens == plain.total_tokens == 1300

    def test_measure_carries_the_split(self) -> None:
        usage = measure([LLMMessage(role="user", content="hi")], memory_tokens=7, memory_entries=2)
        assert (usage.memory_tokens, usage.memory_entries) == (7, 2)

    def test_it_defaults_to_nothing(self) -> None:
        usage = measure([LLMMessage(role="user", content="hi")])
        assert usage.memory_tokens == 0 and usage.memory_entries == 0

    def test_a_negative_count_is_refused(self) -> None:
        with pytest.raises(ValueError):
            ContextUsage(message_tokens=1, memory_tokens=-1)


class TestUsageMetadata:
    def metadata(self, turn_context):
        compactor = Compactor()
        conversation = [LLMMessage(role="user", content="hello there")]
        messages = [*conversation, *(turn_context or [])]
        agent = MagicMock(spec=[])
        return compactor.usage_metadata(messages, None, agent, None, turn_context)[
            CONTEXT_USAGE_METADATA_KEY
        ]

    def test_only_the_tagged_message_is_counted_as_memory(self) -> None:
        skills = LLMMessage(role="assistant", content="Available skills:\n- sql: how")
        usage = self.metadata([skills, _memory_message(3)])
        assert usage["memory_entries"] == 3
        assert 0 < usage["memory_tokens"] < usage["message_tokens"]

    def test_the_total_does_not_depend_on_whether_it_is_told(self) -> None:
        turn_context = [_memory_message()]
        told = self.metadata(turn_context)
        untold = Compactor().usage_metadata(
            [LLMMessage(role="user", content="hello there"), *turn_context],
            None,
            MagicMock(spec=[]),
        )[CONTEXT_USAGE_METADATA_KEY]
        assert told["message_tokens"] == untold["message_tokens"]
        assert untold["memory_tokens"] == 0

    def test_without_memory_nothing_is_counted(self) -> None:
        usage = self.metadata([LLMMessage(role="assistant", content="Available skills:\n- a: b")])
        assert usage["memory_tokens"] == 0 and usage["memory_entries"] == 0


@pytest.mark.parametrize("planner_cls", PLANNERS)
class TestBothPlanners:
    async def usage_seen(self, planner_cls, tmp_path, *, with_memory: bool) -> dict:
        agent = make_mock_agent()
        agent.planner = MagicMock()
        agent.planner.use_stop = False
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            return_value=LLMResponse(content="done", generated_tokens=1, execution_time=0.1)
        )
        if with_memory:
            manager = FilesystemMemoryManager(file_path=str(tmp_path / "memory.md"))

            class _A:
                name = agent.name
                sub_agents: list = []  # noqa: RUF012

            for i in range(4):
                await manager.write_single_memory("user_1", _A(), f"fact number {i}")
            agent.memory_manager = manager
        else:
            agent.memory_manager = None
        context = make_context()
        usage = None
        async for event in planner_cls().execute(agent, context):
            if event.event_type is ExecutionEventType.LLM_START:
                usage = (event.metadata or {}).get(CONTEXT_USAGE_METADATA_KEY)
        assert usage is not None
        return usage

    @pytest.mark.asyncio
    async def test_the_memories_are_counted_apart(self, planner_cls, tmp_path) -> None:
        usage = await self.usage_seen(planner_cls, tmp_path, with_memory=True)
        assert usage["memory_entries"] == 4
        assert 0 < usage["memory_tokens"] <= usage["message_tokens"]

    @pytest.mark.asyncio
    async def test_an_agent_without_memory_shows_none(self, planner_cls, tmp_path) -> None:
        usage = await self.usage_seen(planner_cls, tmp_path, with_memory=False)
        assert usage["memory_tokens"] == 0 and usage["memory_entries"] == 0


class TestTheView:
    def show(self, usage: ContextUsage) -> str:
        buffer = io.StringIO()
        console = Console(file=buffer, width=100, force_terminal=False)
        context_view.render(console, Theme.build(allow_animation=False), usage)
        return buffer.getvalue()

    def test_memory_is_a_row_of_its_own_and_the_conversation_row_gives_it_up(self) -> None:
        text = self.show(
            ContextUsage(
                message_tokens=5000,
                tool_definition_tokens=1000,
                memory_tokens=800,
                memory_entries=14,
            )
        )
        assert "memory (14)" in text and " 800 " in text
        assert "4.2k" in text  # conversation: 5000 - 800
        assert " 6k" in text  # the total is unchanged

    def test_no_memory_no_row(self) -> None:
        text = self.show(ContextUsage(message_tokens=5000, tool_definition_tokens=1000))
        assert "memory" not in text

    def test_a_count_larger_than_the_messages_cannot_make_the_conversation_negative(self) -> None:
        text = self.show(ContextUsage(message_tokens=100, memory_tokens=500))
        assert "-" not in text.split("conversation")[1].splitlines()[0]
