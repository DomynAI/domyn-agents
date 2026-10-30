"""The skills and instruction files in play, said again whenever a turn changed them.

The line under the banner is printed once and cannot be redrawn, while both are read
from disk every turn. So the REPL takes an inventory at the end of each turn and
names what changed — a skill or file added or removed by the user, or by the agent.
"""

import io
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from rich.console import Console

from domyn.agents import Agent
from domyn.agents.instructions import InstructionRuntime, LocalDirInstructionsSource
from domyn.agents.interfaces.cli import Theme
from domyn.agents.interfaces.cli.inventory import Inventory, describe_change, take_inventory
from domyn.agents.interfaces.cli.repl import Repl
from domyn.agents.llm import BaseLLMProvider, LLMMessage, LLMResponse
from domyn.agents.planner_strategy import ToolUsePlannerStrategy
from domyn.agents.skills import LocalDirSkillSource, SkillsRuntime


def _skill(folder: Path, name: str) -> None:
    (folder / name).mkdir(parents=True)
    (folder / name / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name} skill\n---\nbody\n", encoding="utf-8"
    )


class TestDescribingAChange:
    def test_nothing_changed_says_nothing(self) -> None:
        same = Inventory(skills=("a", "b"), instructions=("SOUL.md",))

        assert describe_change(same, same) is None

    def test_the_same_skills_in_another_order_are_no_change(self) -> None:
        before = Inventory(skills=("a", "b"), instructions=())
        after = Inventory(skills=("b", "a"), instructions=())

        assert describe_change(before, after) is None

    def test_added_and_removed_skills_are_named_with_the_new_count(self) -> None:
        before = Inventory(skills=("a", "old"), instructions=())
        after = Inventory(skills=("a", "new", "newer"), instructions=())

        assert describe_change(before, after) == "  Now: 3 skills (+new, +newer, -old)"

    def test_one_replaced_by_another_is_a_change_though_the_count_is_not(self) -> None:
        before = Inventory(skills=("a",), instructions=())
        after = Inventory(skills=("b",), instructions=())

        assert describe_change(before, after) == "  Now: 1 skill (+b, -a)"

    def test_instruction_files_are_listed_as_they_now_are(self) -> None:
        before = Inventory(skills=(), instructions=("SOUL.md",))

        assert (
            describe_change(before, Inventory(skills=(), instructions=("SOUL.md", "AGENTS.md")))
            == "  Now: instructions: SOUL.md, AGENTS.md"
        )
        assert (
            describe_change(before, Inventory(skills=(), instructions=()))
            == "  Now: instructions: none"
        )

    def test_both_on_one_line(self) -> None:
        before = Inventory(skills=(), instructions=())
        after = Inventory(skills=("a",), instructions=("SOUL.md",))

        assert describe_change(before, after) == "  Now: 1 skill (+a)  ·  instructions: SOUL.md"


class _ChangesThingsProvider(BaseLLMProvider):
    """Answers, and changes the files on disk while doing so — standing in for an
    agent that writes a skill or an AGENTS.md during its turn."""

    type: str = "scripted"
    model_name: str = "scripted"
    supported_structured_outputs: list[Any] | None = None
    side_effect: Callable[[], None] | None = None

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        if self.side_effect is not None:
            self.side_effect()
        return LLMResponse(content="done", generated_tokens=1, execution_time=0.0, usage={})

    async def generate_streaming(self, messages: list[LLMMessage], **kwargs: Any) -> Any:
        raise NotImplementedError
        yield  # pragma: no cover


class _Editor:
    """Plays back a script; ``before[i]`` runs just before line ``i`` is returned."""

    def __init__(self, *lines: str, before: dict[int, Callable[[], None]] | None = None) -> None:
        self._lines = list(lines)
        self._before = before or {}
        self._asked = 0

    async def prompt_async(self, *args: Any, **kwargs: Any) -> str:
        if not self._lines:
            raise EOFError
        if (hook := self._before.get(self._asked)) is not None:
            hook()
        self._asked += 1
        return self._lines.pop(0)


def _session(
    tmp_path: Path, *script: str, side_effect: Callable[[], None] | None = None
) -> tuple[Repl, io.StringIO]:
    agent = Agent(
        name="Assistant",
        description="test agent",
        instruction="Be brief.",
        llm_provider=_ChangesThingsProvider(side_effect=side_effect),
        planner=ToolUsePlannerStrategy(),
    )
    agent.skills = SkillsRuntime(source=LocalDirSkillSource(paths=[tmp_path / "skills"]))
    agent.standing_instructions = InstructionRuntime(
        source=LocalDirInstructionsSource(paths=[tmp_path / "home"])
    )
    buffer = io.StringIO()
    repl = Repl(
        agent,
        Console(file=buffer, width=200, force_terminal=False),
        Theme.build(allow_animation=False),
        enable_enhanced_input=False,
    )
    repl._input._session = _Editor(*script)
    return repl, buffer


@pytest.fixture
def folders(tmp_path: Path) -> Path:
    (tmp_path / "skills").mkdir()
    (tmp_path / "home").mkdir()
    _skill(tmp_path / "skills", "first")
    (tmp_path / "home" / "SOUL.md").write_text("Be kind.\n", encoding="utf-8")
    return tmp_path


class TestAtTheEndOfATurn:
    @pytest.mark.asyncio
    async def test_a_skill_written_during_the_turn_is_announced(self, folders: Path) -> None:
        repl, out = _session(folders, "hi", side_effect=lambda: _skill(folders / "skills", "late"))

        await repl.run()

        assert "Now: 2 skills (+late)" in out.getvalue()

    @pytest.mark.asyncio
    async def test_an_instruction_file_written_during_the_turn_is_announced(
        self, folders: Path
    ) -> None:
        def _write() -> None:
            (folders / "home" / "AGENTS.md").write_text("Work on X.\n", encoding="utf-8")

        repl, out = _session(folders, "hi", side_effect=_write)

        await repl.run()

        assert "Now: instructions: SOUL.md, AGENTS.md" in out.getvalue()

    @pytest.mark.asyncio
    async def test_a_turn_that_changed_nothing_says_nothing(self, folders: Path) -> None:
        repl, out = _session(folders, "hi", "again")

        await repl.run()

        assert "Now:" not in out.getvalue()

    @pytest.mark.asyncio
    async def test_a_change_is_said_once_not_after_every_turn(self, folders: Path) -> None:
        calls: list[int] = []

        def _once() -> None:
            calls.append(1)
            if len(calls) == 1:
                _skill(folders / "skills", "late")

        repl, out = _session(folders, "hi", "again", side_effect=_once)

        await repl.run()

        assert out.getvalue().count("Now: 2 skills (+late)") == 1

    @pytest.mark.asyncio
    async def test_a_change_made_between_turns_is_said_after_the_next(self, folders: Path) -> None:
        # Made by the user in another window while typing the second message: it
        # is in play for that turn, and said when it ends.
        repl, out = _session(folders)
        repl._input._session = _Editor(
            "hi", "again", before={1: lambda: _skill(folders / "skills", "from-outside")}
        )

        await repl.run()

        first, second = out.getvalue().split("Completed in", 1)
        assert "from-outside" not in first
        assert "Now: 2 skills (+from-outside)" in second


class TestTheCommandsReadTheFilesNow:
    """`/skills` and `/config` read from disk when used, not at startup."""

    @pytest.mark.asyncio
    async def test_skills_lists_a_skill_added_after_the_session_started(
        self, folders: Path
    ) -> None:
        repl, out = _session(folders)
        repl._input._session = _Editor(
            "/skills", "/skills", before={1: lambda: _skill(folders / "skills", "added")}
        )

        await repl.run()

        first, second = out.getvalue().split("/skill <name> to read one", 1)
        assert "added" not in first
        assert "added" in second

    @pytest.mark.asyncio
    async def test_skill_shows_the_file_as_it_is_now(self, folders: Path) -> None:
        def _edit() -> None:
            (folders / "skills" / "first" / "SKILL.md").write_text(
                "---\nname: first\ndescription: d\n---\nEDITED BODY\n", encoding="utf-8"
            )

        repl, out = _session(folders)
        repl._input._session = _Editor("/skill first", "/skill first", before={1: _edit})

        await repl.run()

        # The first /skill shows the body as it was, the second the edit.
        before_edit, after_edit = out.getvalue().split("EDITED BODY", 1)
        assert "body" in before_edit
        assert "EDITED BODY" not in after_edit


def _write_config(path: Path, skill_paths: list[str], instruction: str = "Be brief.") -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "entrypoint": "Assistant",
                "llm_providers": [{"name": "p", "type": "openai", "model_name": "m"}],
                "planner_strategies": [{"name": "plain", "type": "tool_use"}],
                "agents": [
                    {
                        "name": "Assistant",
                        "description": "test agent",
                        "instruction": instruction,
                        "llm_provider": "p",
                        "planner_strategy": "plain",
                        "skills": {"source": {"type": "local_dir", "paths": skill_paths}},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


class TestTheConfigIsReadEveryTurn:
    """``config.yaml`` is compared with what the agent was built from before and after
    every turn, and the agent rebuilt when it changed: any change is in use from the
    next message, with the session's command-line options applied again."""

    @pytest.fixture
    def config(self, folders: Path) -> Path:
        path = folders / "config.yaml"
        _write_config(path, [str(folders / "skills")])
        return path

    @staticmethod
    def _wire(
        repl: Repl, config: Path, seen: list[list[str]] | None = None, options: Any = None
    ) -> list[int]:
        """Give ``repl`` the real reload, with a scripted model in place of the
        configured one; returns a list counting the reloads."""
        from domyn.agents.cli.commands.run.command import SessionOptions, load_session_agent

        calls: list[int] = []

        async def _reload() -> Any:
            calls.append(1)
            agent, raw = await load_session_agent(config, options or SessionOptions())
            agent.llm_provider = _Recording(seen=seen if seen is not None else [])
            return agent, raw

        repl._config_file = config
        repl._reload = _reload
        repl._config_content = config.read_bytes()
        repl._agent.llm_provider = _Recording(seen=seen if seen is not None else [])
        return calls

    @staticmethod
    def _add_listed_skill(folders: Path, config: Path) -> None:
        _skill(folders / "more", "listed-later")
        _write_config(config, [str(folders / "skills"), str(folders / "more")])

    @pytest.mark.asyncio
    async def test_a_skill_listed_during_a_turn_is_used_from_the_next_message(
        self, folders: Path, config: Path
    ) -> None:
        seen: list[list[str]] = []
        repl, out = _session(folders)
        self._wire(repl, config, seen)
        repl._input._session = _Editor(
            "hi", "again", before={1: lambda: self._add_listed_skill(folders, config)}
        )

        await repl.run()

        assert not any("listed-later" in text for text in seen[0])
        assert any("listed-later" in text for text in seen[-1])
        assert "Reloaded" in out.getvalue()
        assert "Now: 2 skills (+listed-later)" in out.getvalue()

    @pytest.mark.asyncio
    async def test_skills_lists_it_right_away(self, folders: Path, config: Path) -> None:
        repl, out = _session(folders)
        self._wire(repl, config)
        repl._input._session = _Editor(
            "/skills", "/skills", before={1: lambda: self._add_listed_skill(folders, config)}
        )

        await repl.run()

        first, second = out.getvalue().split("/skill <name> to read one", 1)
        assert "listed-later" not in first
        assert "listed-later" in second

    @pytest.mark.asyncio
    async def test_any_field_is_picked_up_not_only_skills(
        self, folders: Path, config: Path
    ) -> None:
        seen: list[list[str]] = []
        repl, _ = _session(folders)
        self._wire(repl, config, seen)
        repl._input._session = _Editor(
            "hi",
            "again",
            before={
                1: lambda: _write_config(
                    config, [str(folders / "skills")], instruction="Answer like a PIRATE."
                )
            },
        )

        await repl.run()

        assert not any("PIRATE" in text for text in seen[0])
        assert any("PIRATE" in text for text in seen[-1])

    @pytest.mark.asyncio
    async def test_an_unchanged_file_is_not_rebuilt(self, folders: Path, config: Path) -> None:
        repl, _ = _session(folders)
        calls = self._wire(repl, config)
        repl._input._session = _Editor("hi", "again", "/skills", "/config", "/agent")

        await repl.run()

        assert calls == []

    @pytest.mark.asyncio
    async def test_a_broken_file_keeps_the_agent_and_says_so_once(
        self, folders: Path, config: Path
    ) -> None:
        repl, out = _session(folders)
        self._wire(repl, config)
        before = repl._agent
        repl._input._session = _Editor(
            "hi",
            "again",
            "and again",
            before={1: lambda: config.write_text("agents: [unclosed\n", encoding="utf-8")},
        )

        await repl.run()

        assert repl._agent is before
        assert out.getvalue().count("changed but cannot be loaded") == 1
        assert out.getvalue().count("done") == 3  # every turn still ran

    @pytest.mark.asyncio
    async def test_the_command_line_options_are_kept(self, folders: Path, config: Path) -> None:
        from domyn.agents.cli.commands.run.command import SessionOptions
        from domyn.agents.skills import AllSkillsSelector

        repl, _ = _session(folders)
        self._wire(repl, config, options=SessionOptions(expand_skills=True))
        repl._input._session = _Editor(
            "hi", "again", before={1: lambda: self._add_listed_skill(folders, config)}
        )

        await repl.run()

        assert repl._agent.skills.source.paths == [folders / "skills", folders / "more"]
        assert isinstance(repl._agent.skills.selector, AllSkillsSelector)

    @pytest.mark.asyncio
    async def test_a_session_without_a_config_file_is_left_alone(self, folders: Path) -> None:
        repl, _ = _session(folders, "hi")
        before = repl._agent

        await repl.run()

        assert repl._agent is before


class TestAutodiscoverCommand:
    """``/skills-autodiscover [on|off]`` flips the switch ``domyn setup`` asks about, in the
    session's config file, and the session uses the result from the next message."""

    @pytest.fixture
    def config(self, folders: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        from domyn.agents.interfaces.cli import config as cli_config
        from domyn.agents.interfaces.cli import repl as cli_repl
        from domyn.agents.interfaces.cli import setup as cli_setup

        # Absolute stand-ins for `.domyn/skills` and `.agents/skills`, read in this order.
        local = (folders / "domyn-local", folders / "agents-local")
        _skill(local[0], "mine")
        _skill(local[1], "shared")
        for module in (cli_config, cli_setup, cli_repl):
            monkeypatch.setattr(module, "LOCAL_SKILLS_DIRS", local)
        path = folders / "config.yaml"
        _write_config(path, [str(folders / "skills")])
        monkeypatch.setattr(cli_repl, "config_path", lambda: path)
        return path

    @staticmethod
    def _paths(config: Path) -> list[str]:
        raw = yaml.safe_load(config.read_text(encoding="utf-8"))
        return list(raw["agents"][0].get("skills", {}).get("source", {}).get("paths", []))

    @pytest.mark.asyncio
    async def test_on_lists_both_folders_first_and_loads_their_skills_now(
        self, folders: Path, config: Path
    ) -> None:
        repl, out = _session(folders)
        TestTheConfigIsReadEveryTurn._wire(repl, config)
        repl._input._session = _Editor(
            "/skills-autodiscover", "/skills-autodiscover on", "/skills-autodiscover"
        )

        await repl.run()

        assert self._paths(config) == [
            str(folders / "domyn-local"),
            str(folders / "agents-local"),
            str(folders / "skills"),
        ]
        text = out.getvalue()
        assert "Skill autodiscovery is off" in text
        assert "Skill autodiscovery on" in text
        assert "Now: 3 skills (+mine, +shared)" in text
        assert "Skill autodiscovery is on" in text
        assert "Reloaded" not in text  # said by the command itself, once

    @pytest.mark.asyncio
    async def test_off_removes_both_entries_and_keeps_the_rest(
        self, folders: Path, config: Path
    ) -> None:
        _write_config(
            config,
            [str(folders / "domyn-local"), str(folders / "agents-local"), str(folders / "skills")],
        )
        repl, out = _session(folders)
        TestTheConfigIsReadEveryTurn._wire(repl, config)
        repl._input._session = _Editor("/skills-autodiscover off", "/skills-autodiscover off")

        await repl.run()

        assert self._paths(config) == [str(folders / "skills")]
        # Collapsed: two absolute folder names make the line long enough to wrap.
        text = " ".join(out.getvalue().split())
        assert "no longer loaded" in text
        assert "already off" in text

    @pytest.mark.asyncio
    async def test_anything_but_on_or_off_changes_nothing(
        self, folders: Path, config: Path
    ) -> None:
        before = config.read_bytes()
        repl, out = _session(folders)
        TestTheConfigIsReadEveryTurn._wire(repl, config)
        repl._input._session = _Editor("/skills-autodiscover yes")

        await repl.run()

        assert config.read_bytes() == before
        assert "Usage: /skills-autodiscover [on|off]" in out.getvalue()

    @pytest.mark.asyncio
    async def test_a_file_run_by_name_is_not_rewritten(
        self, folders: Path, config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from domyn.agents.interfaces.cli import repl as cli_repl

        monkeypatch.setattr(cli_repl, "config_path", lambda: folders / "other.yaml")
        before = config.read_bytes()
        repl, out = _session(folders)
        TestTheConfigIsReadEveryTurn._wire(repl, config)
        repl._input._session = _Editor("/skills-autodiscover on")

        await repl.run()

        assert config.read_bytes() == before
        assert "only applies to the default configuration" in out.getvalue()


class _Recording(_ChangesThingsProvider):
    """Records every request's message texts in ``seen`` — typed ``Any`` so pydantic
    keeps the caller's list rather than validating a copy of it."""

    seen: Any = None

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        self.seen.append([str(m.content) for m in messages])
        return await super().generate(messages, **kwargs)
