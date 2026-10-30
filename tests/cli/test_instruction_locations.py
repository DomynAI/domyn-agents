"""Where the CLI reads SOUL.md and AGENTS.md from, and what it says about it.

Two decisions are under test here rather than one. **Where they come from**
follows the skills' rule: each agent reads what its own definition declares and
nothing else — ``~/.domyn`` is not an instructions folder — with ``./.domyn`` and
``./.agents`` among the paths when setup turned autodiscovery on. And **when they
reach the agent**: the runtime is attached before the session starts, so the REPL
is handed an agent that is already complete; the files themselves are read by the
runner per turn.
"""

# No `from __future__ import annotations`: builds real agents.
import io
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer
from rich.console import Console

from domyn.agents import Agent
from domyn.agents.cli.commands.run import command
from domyn.agents.cli.commands.run.command import apply_session_instructions
from domyn.agents.instructions import (
    AgentInstructions,
    InMemoryInstructionsSource,
    InstructionRuntime,
    LocalDirInstructionsSource,
)
from domyn.agents.interfaces.cli import Theme
from domyn.agents.interfaces.cli import config as cli_config
from domyn.agents.interfaces.cli.config import (
    describe_configuration_v1,
    resolve_instruction_dirs,
)


@pytest.fixture
def elsewhere(tmp_path: Path) -> Path:
    directory = tmp_path / "somewhere-else"
    directory.mkdir()
    (directory / "SOUL.md").write_text("Answer in Italian.", encoding="utf-8")
    return directory


@pytest.fixture
def local_dirs(monkeypatch: pytest.MonkeyPatch) -> tuple[Path, ...]:
    """The real autodiscovery entries, which `conftest.py` points where nothing
    exists for every other test."""
    dirs = (Path(".domyn"), Path(".agents"))
    monkeypatch.setattr(cli_config, "AUTODISCOVERY_DIRS", dirs)
    return dirs


def _write(directory: Path, name: str, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(text, encoding="utf-8")
    return directory / name


def _raw() -> dict[str, Any]:
    """A minimal ConfigurationV1 dict, the shape the CLI passes around."""
    return {
        "version": "1.0",
        "entrypoint": "Assistant",
        "llm_providers": [{"name": "p", "type": "vllm", "model_name": "m", "url": "http://x"}],
        "agents": [{"name": "Assistant", "llm_provider": "p"}],
        "metadata": {},
    }


async def _loaded(declared: list[Path]) -> AgentInstructions:
    source = LocalDirInstructionsSource(paths=resolve_instruction_dirs(declared))
    return await InstructionRuntime(source=source).load()


def _declaring(agent: Agent, *paths: Path) -> Agent:
    agent.standing_instructions = InstructionRuntime(
        source=LocalDirInstructionsSource(paths=list(paths))
    )
    return agent


class TestWhereTheyComeFrom:
    @pytest.mark.asyncio
    async def test_nothing_is_read_from_the_home_directory(self) -> None:
        """``~/.domyn`` is not an instructions folder any more."""
        _write(cli_config.GLOBAL_DIR, "SOUL.md", "Answer in English.")

        assert not await _loaded([])

    @pytest.mark.asyncio
    async def test_several_paths_can_be_declared(self, elsewhere: Path, tmp_path: Path) -> None:
        _write(tmp_path / "other", "AGENTS.md", "Project notes.")

        loaded = await _loaded([elsewhere, tmp_path / "other"])

        assert loaded.names == ("SOUL.md", "AGENTS.md")

    @pytest.mark.asyncio
    async def test_nothing_is_read_from_the_current_directory_unless_listed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cwd = tmp_path / "somewhere"
        _write(cwd / ".domyn", "SOUL.md", "not mine")
        monkeypatch.chdir(cwd)

        assert not await _loaded([])

    @pytest.mark.asyncio
    async def test_an_autodiscovery_entry_reads_the_folder_domyn_runs_in(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_dirs: tuple[Path, ...]
    ) -> None:
        cwd = tmp_path / "project"
        _write(cwd / ".domyn", "AGENTS.md", "Run make check.")
        monkeypatch.chdir(cwd)

        assert "Run make check." in (await _loaded(list(local_dirs))).text

    def test_started_from_home_the_home_directory_is_not_autodiscovered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_dirs: tuple[Path, ...]
    ) -> None:
        """From ``$HOME``, ``./.domyn`` *is* ``~/.domyn``."""
        home = tmp_path / "home"
        monkeypatch.setattr(cli_config, "GLOBAL_DIR", home / ".domyn")
        _write(home / ".domyn", "SOUL.md", "Answer in English.")
        monkeypatch.chdir(home)

        assert resolve_instruction_dirs(list(local_dirs)) == [Path(".agents")]

    @pytest.mark.asyncio
    async def test_the_users_soul_wins_and_the_projects_brief_lands_last(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        local_dirs: tuple[Path, ...],
    ) -> None:
        user = tmp_path / "user"
        _write(user, "SOUL.md", "user soul")
        _write(user, "AGENTS.md", "user brief")
        cwd = tmp_path / "project"
        project_soul = _write(cwd / ".domyn", "SOUL.md", "project soul")
        _write(cwd / ".domyn", "AGENTS.md", "domyn brief")
        _write(cwd / ".agents", "AGENTS.md", "agents brief")
        monkeypatch.chdir(cwd)

        loaded = await _loaded([user / "SOUL.md", user / "AGENTS.md", *local_dirs])

        assert "project soul" not in loaded.text
        assert loaded.superseded == (str(project_soul.resolve()),)
        text = loaded.text
        assert (
            text.index("user soul")
            < text.index("user brief")
            < text.index("domyn brief")
            < text.index("agents brief")
        )

    @pytest.mark.asyncio
    async def test_without_a_soul_of_the_users_the_domyn_folder_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_dirs: tuple[Path, ...]
    ) -> None:
        cwd = tmp_path / "project"
        _write(cwd / ".domyn", "SOUL.md", "domyn soul")
        _write(cwd / ".agents", "SOUL.md", "agents soul")
        monkeypatch.chdir(cwd)

        loaded = await _loaded(list(local_dirs))

        assert "domyn soul" in loaded.text
        assert "agents soul" not in loaded.text


class TestWhichRuntimeTheAgentGets:
    def test_an_agent_without_one_gets_none(self, agent: Agent) -> None:
        apply_session_instructions(agent)

        assert agent.standing_instructions is None

    def test_declared_paths_are_kept(self, agent: Agent, tmp_path: Path) -> None:
        declared = tmp_path / "declared"
        declared.mkdir()
        _declaring(agent, declared)

        apply_session_instructions(agent)

        assert agent.standing_instructions.source.paths == [declared]

    def test_the_declared_runtime_is_copied_not_edited(self, agent: Agent, elsewhere: Path) -> None:
        """It may be shared with another agent built from the same config."""
        declared = _declaring(agent, elsewhere).standing_instructions

        apply_session_instructions(agent)

        assert agent.standing_instructions is not declared
        assert declared.source.paths == [elsewhere]

    def test_its_budget_survives(self, agent: Agent, elsewhere: Path) -> None:
        _declaring(agent, elsewhere).standing_instructions.max_bytes = 10

        apply_session_instructions(agent)

        assert agent.standing_instructions.max_bytes == 10

    def test_a_source_that_is_not_a_directory_is_kept(self, agent: Agent) -> None:
        """Its author's."""
        declared = InstructionRuntime(source=InMemoryInstructionsSource(soul="Be brief."))
        agent.standing_instructions = declared

        apply_session_instructions(agent)

        assert agent.standing_instructions is declared

    def test_a_sub_agent_is_never_touched(
        self, agent: Agent, child_agent: Agent, tmp_path: Path
    ) -> None:
        """Like skills: only the root is touched. A sub-agent reads what its own
        definition declares, or nothing."""
        declared = tmp_path / "child"
        declared.mkdir()
        _declaring(child_agent, declared)
        bare = child_agent.model_copy(update={"name": "Bare", "standing_instructions": None})
        agent.sub_agents = [child_agent, bare]

        apply_session_instructions(agent)

        assert child_agent.standing_instructions.source.paths == [declared]
        assert bare.standing_instructions is None

    def test_the_agents_instruction_is_not_touched(self, agent: Agent, elsewhere: Path) -> None:
        apply_session_instructions(_declaring(agent, elsewhere))

        assert agent.instruction == "Help with tests"


class TestWhenTheyReachTheAgent:
    """The runtime is attached before the session; the text is read per turn."""

    def test_run_command_attaches_them(self, agent: Agent, elsewhere: Path) -> None:
        """Nothing else would fail if this line were dropped — the session would
        simply run without the instructions."""
        _declaring(agent, elsewhere)
        with (
            patch.object(command, "resolve_agent_from_file", new=AsyncMock()) as resolve,
            patch.object(command, "start_session", new=AsyncMock(return_value=0)),
            patch.object(command, "console", MagicMock()),
        ):
            resolve.return_value = (agent, _raw())
            with pytest.raises(typer.Exit):
                command.run_command(
                    file=None,
                    prompt=None,
                    agent_name=None,
                    expand_skills=False,
                    verbose=False,
                    no_banner=True,
                )

        assert isinstance(agent.standing_instructions, InstructionRuntime)
        assert agent.standing_instructions.source.paths == [elsewhere]

    @pytest.mark.asyncio
    async def test_the_status_line_names_the_files(
        self, agent: Agent, elsewhere: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The only moment these files are announced."""
        buffer = io.StringIO()
        monkeypatch.setattr(
            command, "console", Console(file=buffer, width=200, force_terminal=False)
        )
        apply_session_instructions(_declaring(agent, elsewhere))

        with patch.object(command, "Repl", new=MagicMock(return_value=MagicMock())) as repl:
            repl.return_value.run = MagicMock(return_value=_done(0))
            await command.start_session(
                agent,
                Theme.build(allow_animation=False),
                initial_prompt=None,
                verbose=False,
                show_status=True,
            )

        assert "SOUL.md" in buffer.getvalue()

    @pytest.mark.asyncio
    async def test_a_session_without_them_says_nothing_extra(
        self, agent: Agent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        buffer = io.StringIO()
        monkeypatch.setattr(
            command, "console", Console(file=buffer, width=200, force_terminal=False)
        )
        apply_session_instructions(agent)

        with patch.object(command, "Repl", new=MagicMock(return_value=MagicMock())) as repl:
            repl.return_value.run = MagicMock(return_value=_done(0))
            await command.start_session(
                agent,
                Theme.build(allow_animation=False),
                initial_prompt=None,
                verbose=False,
                show_status=True,
            )

        assert ".md" not in buffer.getvalue()


async def _done(value: int) -> int:
    return value


class TestTheySurviveAReload:
    """``/model`` rebuilds the agent from disk mid-session, without the runtime the
    session resolved — so the runtime has to be carried across."""

    @staticmethod
    async def _reload(agent: Agent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Agent:
        from domyn.agents.interfaces.cli import config as cfg
        from domyn.agents.interfaces.cli import repl as repl_mod
        from domyn.agents.interfaces.cli.repl import Repl

        monkeypatch.setattr(cfg, "GLOBAL_DIR", tmp_path / "home" / ".domyn")
        monkeypatch.setattr(cfg, "PROJECT_DIR", tmp_path / "project" / ".domyn")

        usable = _raw()
        monkeypatch.setattr(repl_mod, "run_setup", lambda *a, **k: usable)

        # The agent the reload produces: straight off disk, no runtime.
        reloaded = agent.model_copy(deep=True)
        reloaded.standing_instructions = None

        async def _load(*args: object, **kwargs: object) -> Agent:
            return reloaded

        monkeypatch.setattr(repl_mod, "load_application_from_file", _load)

        repl = Repl(
            agent,
            Console(file=io.StringIO(), width=100, force_terminal=False),
            Theme.build(allow_animation=False),
            config=usable,
            config_file=cfg.config_path(),
            enable_enhanced_input=False,
        )
        await repl._change_model()
        return reloaded

    @pytest.mark.asyncio
    async def test_model_carries_the_runtime_across(
        self, agent: Agent, elsewhere: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        apply_session_instructions(_declaring(agent, elsewhere))
        runtime = agent.standing_instructions

        reloaded = await self._reload(agent, tmp_path, monkeypatch)

        assert reloaded.standing_instructions is runtime
        assert reloaded.instruction == "Help with tests"

    @pytest.mark.asyncio
    async def test_a_session_without_them_reloads_without_them(
        self, agent: Agent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reloaded = await self._reload(agent, tmp_path, monkeypatch)

        assert reloaded.standing_instructions is None
