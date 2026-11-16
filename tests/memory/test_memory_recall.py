"""Pages, the recall at the start of a turn, and the tools that read memory back.

What can go wrong: a page that skips what did not fit, a recall asked for on every
request of a turn, a slow or failing manager failing the turn, an agent that cannot
read back or drop what it remembered, and a repeat stored twice.
"""

import asyncio
import json
import os
import tempfile
from types import SimpleNamespace
from typing import ClassVar

import pytest

from domyn.agents.agents.agent import Agent
from domyn.agents.llm.base import BaseLLMProvider, LLMMessage, LLMResponse
from domyn.agents.memory.managers import (
    BaseMemoryManager,
    FilesystemMemoryManager,
    MemoryExecutionMode,
)

RUN = {"user_id": "u"}


class _A:
    name = "A"
    sub_agents: ClassVar[list] = []


def context(question="", turn_id="t1", user="u"):
    content = [SimpleNamespace(text=question)] if question else []
    return SimpleNamespace(
        conversation_id="c1",
        turn_id=turn_id,
        interaction_id="i1",
        user_id=user,
        run_config={},
        agent_input=SimpleNamespace(content=content),
    )


@pytest.fixture
def md(tmp_path):
    return FilesystemMemoryManager(file_path=str(tmp_path / "m.md"))


class Counting(FilesystemMemoryManager):
    """A real manager that counts its reads."""

    reads: int = 0

    async def recall_page(self, *args, **kwargs):
        object.__setattr__(self, "reads", self.reads + 1)
        return await super().recall_page(*args, **kwargs)


async def fill(manager, texts):
    for text in texts:
        await manager.write_single_memory("u", _A(), text)


class TestPagesForEveryManager:
    @pytest.mark.asyncio
    async def test_the_filesystem_manager_gets_pages_and_search_without_changes(self, tmp_path):
        manager = FilesystemMemoryManager(file_path=str(tmp_path / "m.jsonl"))
        await fill(manager, [f"tea number {i}" if i % 2 else f"coffee {i}" for i in range(6)])
        page = await manager.recall_page("u", limit=4)
        assert len(page.items) == 4 and page.next_cursor == "4" and page.total == 6
        rest = await manager.recall_page("u", limit=4, cursor=page.next_cursor)
        assert len(rest.items) == 2 and rest.next_cursor is None
        found = await manager.recall_page("u", query="tea")
        assert {m.memory for m in found.items} == {"tea number 1", "tea number 3", "tea number 5"}

    @pytest.mark.asyncio
    async def test_a_subclass_with_only_the_old_methods_still_works(self):
        class Old(BaseMemoryManager):
            type: str = "old-style"

            async def write_single_memory(self, user_id, agent, memory_content, **kwargs): ...
            async def recall_memories(
                self, user_id, agent=None, recency_limit=5, question="", **kw
            ):
                return []

            async def get_memory(self, memory_id): ...
            async def delete_memory(self, memory_id):
                return False

            async def reset_memory(self, user_id, agent_name=None): ...
            async def complete_reset(self): ...

        assert (await Old().recall_page("u")).items == []


class TestAPageCutToItsBudget:
    """What fits in the characters, read once, with a cursor right after it."""

    @pytest.mark.asyncio
    async def test_the_default_pages_are_read_once_and_cut(self, tmp_path):
        class ReadCounting(FilesystemMemoryManager):
            reads: int = 0

            async def recall_memories(self, *args, **kwargs):
                object.__setattr__(self, "reads", self.reads + 1)
                return await super().recall_memories(*args, **kwargs)

        manager = ReadCounting(file_path=str(tmp_path / "m.md"))
        await fill(manager, [f"fact {i} " + "x" * 200 for i in range(10)])
        page = await manager._fitting_page("u", _A(), query=None, cursor=None, budget=1_000)
        assert manager.reads == 1
        assert 0 < len(page.items) < 10 and page.total == 10
        assert page.next_cursor == str(len(page.items))
        rest = await manager.recall_page("u", cursor=page.next_cursor)
        shown = [m.memory for m in page.items] + [m.memory for m in rest.items]
        assert shown == [m.memory for m in (await manager.recall_page("u")).items]

    @pytest.mark.asyncio
    async def test_a_manager_with_pages_of_its_own_is_asked_for_its_cursor(self, tmp_path):
        class OwnPages(FilesystemMemoryManager):
            asked: ClassVar[list] = []

            async def recall_page(self, user_id, agent=None, *, limit=None, cursor=None, **kw):
                self.asked.append(limit)
                page = await super().recall_page(user_id, agent, limit=limit, cursor=cursor, **kw)
                return page.model_copy(update={"next_cursor": f"server-{page.next_cursor}"})

        manager = OwnPages(file_path=str(tmp_path / "m.md"))
        await fill(manager, [f"fact {i} " + "x" * 200 for i in range(10)])
        page = await manager._fitting_page("u", _A(), query=None, cursor=None, budget=1_000)
        assert manager.asked == [None, len(page.items)]
        assert page.next_cursor == f"server-{len(page.items)}"


class TestTheRecallAtTheStartOfATurn:
    @pytest.mark.asyncio
    async def test_it_is_markdown_with_a_heading_and_ids(self, md):
        await fill(md, ["likes tea"])
        block = await md.recall_block(_A(), context())
        assert block.startswith("## Remembered about the user")
        assert "- likes tea" in block and "id `" in block

    @pytest.mark.asyncio
    async def test_nothing_remembered_is_nothing_injected(self, md):
        assert await md.recall_block(_A(), context()) == ""

    @pytest.mark.asyncio
    async def test_the_same_turn_asks_the_manager_once(self, tmp_path):
        manager = Counting(file_path=str(tmp_path / "m.md"))
        await fill(manager, ["likes tea"])
        manager.reads = 0
        first = await manager.recall_block(_A(), context())
        again = await manager.recall_block(_A(), context())
        assert first == again and manager.reads == 1

    @pytest.mark.asyncio
    async def test_the_next_turn_sees_what_was_added(self, md):
        assert await md.recall_block(_A(), context(turn_id="t1")) == ""
        await fill(md, ["new fact"])
        assert "new fact" in await md.recall_block(_A(), context(turn_id="t2"))

    @pytest.mark.asyncio
    async def test_the_question_picks_what_is_shown_and_none_matching_falls_back_to_recent(
        self, md
    ):
        await fill(md, ["likes green tea", "owns a bicycle"])
        assert "likes green tea" in await md.recall_block(_A(), context("what tea do I like?"))
        assert "owns a bicycle" in await md.recall_block(
            _A(), context("zzz unrelated", turn_id="t2")
        )

    @pytest.mark.asyncio
    async def test_the_cache_is_bounded(self, md):
        for i in range(100):
            await md.recall_block(_A(), context(turn_id=f"t{i}"))
        assert len(md._turns) <= 64

    @pytest.mark.asyncio
    async def test_a_page_over_the_budget_is_narrowed_and_its_cursor_follows_the_last_one_shown(
        self, tmp_path
    ):
        manager = FilesystemMemoryManager(file_path=str(tmp_path / "m.md"), max_chars=1000)
        await fill(manager, [f"fact {i:02d} " + "x" * 80 for i in range(20)])
        block = await manager.recall_block(_A(), context())
        shown = [line for line in block.splitlines() if line.startswith("- ")]
        assert 0 < len(shown) < 20
        cursor = block.split('cursor="')[1].split('"')[0]
        assert cursor == str(len(shown))
        text = await manager.recall(_A(), cursor=cursor, run_config=RUN)
        assert f"fact {19 - len(shown):02d}" in text

    @pytest.mark.asyncio
    async def test_only_the_characters_bound_a_page(self, tmp_path):
        """No count: short memories fill the budget, however many there are."""
        manager = FilesystemMemoryManager(file_path=str(tmp_path / "m.md"), max_chars=8000)
        await fill(manager, [f"fact {i:02d}" for i in range(50)])
        block = await manager.recall_block(_A(), context())
        assert len([line for line in block.splitlines() if line.startswith("- ")]) == 50
        assert 'cursor="' not in block

    @pytest.mark.asyncio
    async def test_reading_every_page_shows_every_entry_once(self, tmp_path):
        manager = FilesystemMemoryManager(file_path=str(tmp_path / "m.md"), max_chars=800)
        await fill(manager, [f"fact {i:02d} " + "x" * 80 for i in range(15)])
        seen, cursor = [], ""
        for _ in range(30):
            text = await manager.recall(_A(), cursor=cursor, run_config=RUN)
            seen += [line.split(" ")[2] for line in text.splitlines() if line.startswith("- fact")]
            if 'cursor="' not in text:
                break
            cursor = text.split('cursor="')[1].split('"')[0]
        assert sorted(seen) == [f"{i:02d}" for i in range(15)]


class TestFailuresDoNotFailTheTurn:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error", [ValueError("bad"), NotImplementedError("later"), OSError("disk")]
    )
    async def test_a_failing_manager_means_no_memory(self, md, error):
        async def broken(*args, **kwargs):
            raise error

        object.__setattr__(md, "recall_page", broken)
        assert await md.recall_block(_A(), context()) == ""

    @pytest.mark.asyncio
    async def test_a_slow_manager_is_given_up_on(self, tmp_path):
        manager = FilesystemMemoryManager(file_path=str(tmp_path / "m.md"), recall_timeout_s=0.05)

        async def slow(*args, **kwargs):
            await asyncio.sleep(5)

        object.__setattr__(manager, "recall_page", slow)
        started = asyncio.get_running_loop().time()
        assert await manager.recall_block(_A(), context()) == ""
        assert asyncio.get_running_loop().time() - started < 1.0

    @pytest.mark.asyncio
    async def test_the_run_config_reaches_the_manager(self, md):
        seen = {}

        async def spy(user_id, agent=None, **kwargs):
            seen.update(kwargs)
            return await FilesystemMemoryManager.recall_page(md, user_id, agent, **kwargs)

        object.__setattr__(md, "recall_page", spy)
        ctx = context()
        ctx.run_config = {"workspace": "w"}
        await md.recall_block(_A(), ctx)
        assert seen["run_config"] == {"workspace": "w"}


class TestRemember:
    @pytest.mark.asyncio
    async def test_a_fact_is_stored(self, md):
        assert await md.remember(_A(), "  likes   tea ", RUN) == "Memory stored."
        assert [m.memory for m in (await md.recall_page("u")).items] == ["likes tea"]

    @pytest.mark.asyncio
    async def test_the_same_fact_is_not_stored_twice(self, md):
        await md.remember(_A(), "Likes tea.", RUN)
        assert await md.remember(_A(), "likes   TEA", RUN) == "Already remembered."
        assert (await md.recall_page("u")).total == 1

    @pytest.mark.asyncio
    async def test_empty_and_long_texts_are_refused_with_a_reason(self, md):
        assert "empty" in await md.remember(_A(), "   ", RUN)
        assert "Too long" in await md.remember(_A(), "x" * 600, RUN)
        assert (await md.recall_page("u")).total == 0

    @pytest.mark.asyncio
    async def test_without_a_user_nothing_is_stored(self, md):
        assert "no user_id" in await md.remember(_A(), "fact", {})


class TestRecallAndForgetTools:
    @pytest.mark.asyncio
    async def test_recall_says_what_it_found_or_that_it_found_nothing(self, md):
        assert await md.recall(_A(), run_config=RUN) == "Nothing is remembered yet."
        await fill(md, ["likes tea"])
        assert 'No memory matches "zebra"' in await md.recall(_A(), "zebra", run_config=RUN)
        assert "likes tea" in await md.recall(_A(), "tea", run_config=RUN)

    @pytest.mark.asyncio
    async def test_a_cursor_that_is_not_ours_does_not_raise(self, md):
        await fill(md, ["x"])
        assert "not available" in await md.recall(_A(), cursor="garbage", run_config=RUN)

    @pytest.mark.asyncio
    async def test_without_a_user_it_says_so(self, md):
        assert "no user_id" in await md.recall(_A())
        assert "no user_id" in await md.forget(_A(), "x")

    @pytest.mark.asyncio
    async def test_forget_by_id(self, md):
        await fill(md, ["a", "b"])
        target = (await md.recall_page("u")).items[0]
        assert await md.forget(_A(), f" {target.id} ", RUN) == "Forgotten."
        assert "No memory has the id" in await md.forget(_A(), "nope", RUN)
        assert (await md.recall_page("u")).total == 1


class _LLM(BaseLLMProvider):
    type: str = "mock"
    name: str = "mock"
    model_name: str = "mock"
    supported_structured_outputs: None = None

    async def generate(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        return LLMResponse(content="x", usage={}, generated_tokens=1, execution_time=0.0)

    async def generate_streaming(self, messages, **kwargs):
        yield


class TestTheAgentGetsTheTools:
    def names(self, manager):
        agent = Agent(
            name="A", description="d", instruction="i", llm_provider=_LLM(), memory_manager=manager
        )
        return {t.name for t in agent.tools}

    def test_an_agentic_agent_can_remember_recall_and_forget(self, md):
        assert {"remember", "recall", "forget"} <= self.names(
            md.model_copy(update={"execution_mode": MemoryExecutionMode.AGENTIC})
        )

    def test_an_automatic_agent_keeps_the_tools_it_had(self, md):
        names = self.names(md.model_copy(update={"execution_mode": MemoryExecutionMode.AUTOMATIC}))
        assert not {"remember", "recall", "forget"} & names

    @pytest.mark.parametrize("mode", ["agentic"])
    def test_it_is_configured_from_a_plain_dict(self, tmp_path, mode):
        agent = Agent(
            name="A",
            description="d",
            instruction="i",
            llm_provider=_LLM(),
            memory_manager={
                "type": "filesystem",
                "file_path": str(tmp_path / "m.md"),
                "execution_mode": mode,
            },
        )
        assert isinstance(agent.memory_manager, FilesystemMemoryManager)
        assert "recall" in {t.name for t in agent.tools}
