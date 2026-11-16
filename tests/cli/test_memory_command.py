"""``/memory``: see, add to, forget from and clear what the agent remembers.

Through the REPL, because the wiring is where it breaks: a command that cannot reach
the agent's manager shows an empty list and looks like good news.
"""

import io
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

import pytest
from rich.console import Console

from domyn.agents.agents.agent import Agent
from domyn.agents.interfaces.cli import commands, memory_view
from domyn.agents.interfaces.cli import select as select_mod
from domyn.agents.interfaces.cli.repl import USER_ID, Repl
from domyn.agents.interfaces.cli.theme import Theme
from domyn.agents.llm.base import BaseLLMProvider, LLMMessage, LLMResponse
from domyn.agents.memory.managers import FilesystemMemoryManager, MemoryExecutionMode
from domyn.agents.runner import Runner


class _LLM(BaseLLMProvider):
    type: str = "mock"
    name: str = "mock"
    model_name: str = "mock"
    supported_structured_outputs: None = None

    async def generate(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        return LLMResponse(content="x", usage={}, generated_tokens=1, execution_time=0.0)

    async def generate_streaming(self, messages, **kwargs):
        yield


def _repl(tmp_path, *, with_memory=True, read_only_sources=()):
    manager = (
        FilesystemMemoryManager(
            file_path=str(tmp_path / "memory.md"),
            execution_mode=MemoryExecutionMode.AGENTIC,
            read_only_sources=[str(path) for path in read_only_sources],
        )
        if with_memory
        else None
    )
    agent = Agent(
        name="Assistant",
        description="d",
        instruction="i",
        llm_provider=_LLM(),
        memory_manager=manager,
    )
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=False, markup=False)
    repl = Repl(
        agent,
        console,
        Theme.build(allow_animation=False),
        runner=Runner(),
        enable_enhanced_input=False,
    )
    return repl, buffer, manager


async def _fill(manager, n):
    class _A:
        name = "Assistant"
        sub_agents: ClassVar[list] = []

    for i in range(n):
        await manager.write_single_memory(USER_ID, _A(), f"fact number {i:02d}")


class TestTheCommandExists:
    def test_it_resolves_and_is_offered(self):
        resolved = commands.resolve("/memory")
        assert resolved is not None and resolved[0].name == "/memory"
        assert "/memory" in commands.completion_names()
        assert any(row[0].startswith("/memory") for row in commands.help_lines())


class TestWithoutMemory:
    @pytest.mark.asyncio
    async def test_it_says_memory_is_off_and_how_to_turn_it_on(self, tmp_path):
        repl, buffer, _ = _repl(tmp_path, with_memory=False)
        await repl._handle_memory("")
        assert "Memory is off" in buffer.getvalue() and "domyn setup" in buffer.getvalue()


class TestListing:
    @pytest.mark.asyncio
    async def test_nothing_remembered_says_so_and_names_the_file(self, tmp_path):
        repl, buffer, _ = _repl(tmp_path)
        await repl._handle_memory("")
        assert "Nothing is remembered yet" in buffer.getvalue()
        assert str(tmp_path / "memory.md") in buffer.getvalue()

    @pytest.mark.asyncio
    async def test_a_page_shows_id_date_and_text_newest_first(self, tmp_path):
        repl, buffer, manager = _repl(tmp_path)
        await _fill(manager, 3)
        await repl._handle_memory("list")
        text = buffer.getvalue()
        assert text.index("fact number 02") < text.index("fact number 00")
        assert datetime.now(UTC).date().isoformat() in text
        assert "page 1/1" in text

    @pytest.mark.asyncio
    async def test_pages_are_numbered_and_the_next_is_named(self, tmp_path):
        repl, buffer, manager = _repl(tmp_path)
        await _fill(manager, memory_view.PAGE_SIZE + 5)
        await repl._handle_memory("")
        assert "page 1/2" in buffer.getvalue() and "/memory list 2" in buffer.getvalue()
        assert "fact number 00" not in buffer.getvalue()
        await repl._handle_memory("list 2")
        assert "page 2/2" in buffer.getvalue() and "fact number 00" in buffer.getvalue()

    @pytest.mark.asyncio
    async def test_a_page_past_the_end_shows_the_last_one(self, tmp_path):
        repl, buffer, manager = _repl(tmp_path)
        await _fill(manager, memory_view.PAGE_SIZE + 5)
        await repl._handle_memory("list 9")
        assert "page 2/2" in buffer.getvalue()

    @pytest.mark.asyncio
    async def test_an_escape_sequence_in_a_memory_does_not_reach_the_terminal(self, tmp_path):
        repl, buffer, manager = _repl(tmp_path)

        class _A:
            name = "Assistant"
            sub_agents: ClassVar[list] = []

        await manager.write_single_memory(USER_ID, _A(), "red \x1b[31mtext\x1b[0m")
        await repl._handle_memory("")
        assert "\x1b" not in buffer.getvalue()


class TestAddingAndForgetting:
    @pytest.mark.asyncio
    async def test_add_stores_and_says_so(self, tmp_path):
        repl, buffer, manager = _repl(tmp_path)
        await repl._handle_memory("add likes tea")
        assert "Memory stored." in buffer.getvalue()
        assert (await manager.recall_page(USER_ID)).items[0].memory == "likes tea"

    @pytest.mark.asyncio
    async def test_add_without_text_says_usage(self, tmp_path):
        repl, buffer, _ = _repl(tmp_path)
        await repl._handle_memory("add")
        assert "Usage: /memory add" in buffer.getvalue()

    @pytest.mark.asyncio
    async def test_forget_removes_by_the_id_shown(self, tmp_path):
        repl, buffer, manager = _repl(tmp_path)
        await _fill(manager, 2)
        target = (await manager.recall_page(USER_ID)).items[0]
        await repl._handle_memory(f"forget {target.id}")
        assert "Forgotten." in buffer.getvalue()
        assert (await manager.recall_page(USER_ID)).total == 1

    @pytest.mark.asyncio
    async def test_forget_an_unknown_id_says_so(self, tmp_path):
        repl, buffer, _ = _repl(tmp_path)
        await repl._handle_memory("forget nope")
        assert "No memory has the id" in buffer.getvalue()


class TestClearing:
    @pytest.fixture(autouse=True)
    def numbered_prompts(self, monkeypatch):
        monkeypatch.setattr(select_mod, "_interactive", lambda theme: False)

    @pytest.mark.asyncio
    async def test_it_asks_first_and_no_is_the_default(self, tmp_path):
        repl, buffer, manager = _repl(tmp_path)
        await _fill(manager, 2)
        with patch("builtins.input", side_effect=[""]):
            await repl._handle_memory("clear")
        assert "Nothing was removed." in buffer.getvalue()
        assert (await manager.recall_page(USER_ID)).total == 2

    @pytest.mark.asyncio
    async def test_yes_clears(self, tmp_path):
        repl, buffer, manager = _repl(tmp_path)
        await _fill(manager, 2)
        with patch("builtins.input", side_effect=["yes"]):
            await repl._handle_memory("clear")
        assert "Memory cleared." in buffer.getvalue()
        # The file is emptied, the conversation is not: the user is told so.
        assert "This conversation still holds what was said in it" in buffer.getvalue()
        assert (await manager.recall_page(USER_ID)).total == 0

    @pytest.mark.asyncio
    async def test_it_says_hand_written_entries_were_kept(self, tmp_path):
        repl, buffer, manager = _repl(tmp_path)
        await _fill(manager, 2)
        with open(tmp_path / "memory.md", "a") as f:
            f.write("- written by hand\n")
        with patch("builtins.input", side_effect=["yes"]):
            await repl._handle_memory("clear")
        assert "except 1 entry written by hand" in buffer.getvalue()
        assert "/memory clear all" in buffer.getvalue()
        assert [m.memory for m in (await manager.recall_page(USER_ID)).items] == ["written by hand"]

    @pytest.mark.asyncio
    async def test_all_empties_the_file_after_asking(self, tmp_path):
        repl, buffer, manager = _repl(tmp_path)
        await _fill(manager, 2)
        with open(tmp_path / "memory.md", "a") as f:
            f.write("- written by hand\n")
        await manager.write_single_memory("someone-else", repl._agent, "not mine")
        with patch("builtins.input", side_effect=["yes"]):
            await repl._handle_memory("clear all")
        assert str(tmp_path / "memory.md") in buffer.getvalue().replace("\n", "")
        assert "every entry in the file" in buffer.getvalue()
        assert "not mine" not in (tmp_path / "memory.md").read_text()
        assert "written by hand" not in (tmp_path / "memory.md").read_text()

    @pytest.mark.asyncio
    async def test_all_answered_no_removes_nothing(self, tmp_path):
        repl, buffer, manager = _repl(tmp_path)
        await _fill(manager, 2)
        with patch("builtins.input", side_effect=[""]):
            await repl._handle_memory("clear all")
        assert "Nothing was removed." in buffer.getvalue()
        assert (await manager.recall_page(USER_ID)).total == 2

    @pytest.mark.asyncio
    async def test_another_argument_is_refused_without_asking(self, tmp_path):
        repl, buffer, manager = _repl(tmp_path)
        await _fill(manager, 2)
        await repl._handle_memory("clear everything")
        assert "Usage: /memory clear [all]" in buffer.getvalue()
        assert (await manager.recall_page(USER_ID)).total == 2

    @pytest.mark.asyncio
    async def test_all_is_refused_for_a_memory_that_is_not_a_file(self, tmp_path):
        class _NoFile:
            async def complete_reset(self):
                raise AssertionError("a shared store must not be emptied")

        repl, buffer, _ = _repl(tmp_path)
        repl._agent = SimpleNamespace(memory_manager=_NoFile())
        await repl._handle_memory("clear all")
        assert "not one" in buffer.getvalue()


class TestUnknown:
    @pytest.mark.asyncio
    async def test_an_unknown_verb_names_the_ones_there_are(self, tmp_path):
        repl, buffer, _ = _repl(tmp_path)
        await repl._handle_memory("frobnicate")
        assert "Unknown: /memory frobnicate" in buffer.getvalue()


class TestTheUserIsPassedToTheRunner:
    @pytest.mark.asyncio
    async def test_remember_can_find_who_it_writes_for(self, tmp_path):
        """The reason `remember` wrote nothing in the CLI: no `run_config`."""
        repl, _, _manager = _repl(tmp_path)
        seen: dict = {}

        async def spy(**kwargs):
            seen.update(kwargs)
            return
            yield  # pragma: no cover

        repl._runner.run = spy  # type: ignore[method-assign]
        from domyn.agents.core import BaseEvent, ExecutionEventType, Part

        await repl._drive(
            BaseEvent(
                event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text="hi")]
            )
        )
        assert seen["run_config"] == {"user_id": USER_ID}


class TestAMemoryFileOnlyRead:
    """A project reading the global memory beside its own (`read_only_sources`)."""

    @pytest.fixture
    def other(self, tmp_path):
        path = tmp_path / "global.md"
        path.write_text(
            f"# Memory\n\n- lives in Milan <!-- id=g1 at=2026-01-01T00:00:00+00:00 "
            f"user={USER_ID} by=Elsewhere -->\n",
            encoding="utf-8",
        )
        return path

    @pytest.fixture(autouse=True)
    def numbered_prompts(self, monkeypatch):
        monkeypatch.setattr(select_mod, "_interactive", lambda theme: False)

    @pytest.mark.asyncio
    async def test_the_list_marks_its_entries_and_names_the_file(self, tmp_path, other):
        repl, buffer, manager = _repl(tmp_path, read_only_sources=[other])
        await _fill(manager, 1)
        await repl._handle_memory("list")
        printed = buffer.getvalue()
        assert "lives in Milan  (read only)" in printed
        assert "fact number 00  (read only)" not in printed
        assert f"reads {other} (read only)" in printed.replace("\n", "")

    @pytest.mark.asyncio
    async def test_forget_says_where_the_entry_is_and_leaves_it(self, tmp_path, other):
        repl, buffer, _ = _repl(tmp_path, read_only_sources=[other])
        before = other.read_text()
        await repl._handle_memory("forget g1")
        assert "Not forgotten" in buffer.getvalue()
        assert other.read_text() == before

    @pytest.mark.asyncio
    async def test_clear_does_not_count_them_as_left_behind(self, tmp_path, other):
        repl, buffer, manager = _repl(tmp_path, read_only_sources=[other])
        await _fill(manager, 2)
        with patch("builtins.input", side_effect=["yes"]):
            await repl._handle_memory("clear")
        assert "Memory cleared." in buffer.getvalue()
        assert "written by hand" not in buffer.getvalue()
        assert [m.memory for m in (await manager.recall_page(USER_ID)).items] == ["lives in Milan"]
