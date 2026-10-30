"""What the run command and the skills commands do about the skills `domyn setup` installs.

Setup puts them in a directory nobody scans and names each one in the config, so
everything that wants them has to come through the config: the session (which
builds the agent from it), `domyn skills`, and the hint for a config that predates
the choice.
"""

import io
from pathlib import Path
from typing import Any

import pytest
import yaml
from rich.console import Console
from typer.testing import CliRunner

import domyn.agents.cli.commands.run.command as run_module
from domyn.agents import Agent
from domyn.agents.cli.cli import app
from domyn.agents.cli.commands.run.command import _lacks_chosen_skills
from domyn.agents.interfaces.cli import Theme
from domyn.agents.interfaces.cli import config as cfg
from domyn.agents.skills.catalog import install_skills


def _config(skills: dict[str, Any] | None) -> dict[str, Any]:
    return cfg.configuration_v1_dict_from(
        cfg.ProviderConfig(type="openai", model_name="m"),
        cfg.AgentConfig(name="A", instruction="i", skills=skills),
        cfg.CompactionConfig(),
    )


class TestTheHintForAConfigThatNeverChose:
    def test_a_saved_config_with_no_skills_gets_it(self) -> None:
        assert _lacks_chosen_skills(_config(None), cfg.config_path()) is True

    def test_a_saved_config_with_skills_does_not(self) -> None:
        skills = {"source": {"type": "local_dir", "paths": ["/x"]}}

        assert _lacks_chosen_skills(_config(skills), cfg.config_path()) is False

    def test_an_agent_file_named_on_the_command_line_does_not(self, tmp_path: Path) -> None:
        """It declares its own skills, or none, and is not setup's to ask about."""
        assert _lacks_chosen_skills(_config(None), tmp_path / "agent.yaml") is False

    def test_an_agent_that_did_not_come_from_a_config_does_not(self) -> None:
        assert _lacks_chosen_skills(None, None) is False
        assert _lacks_chosen_skills({}, cfg.config_path()) is False

    @pytest.mark.asyncio
    async def test_the_session_says_it_once_under_the_status_line(
        self, agent: Agent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Repl:
            def __init__(self, *args: Any, **kwargs: Any) -> None: ...

            async def run(self, initial_prompt: str | None) -> int:
                return 0

        buffer = io.StringIO()
        monkeypatch.setattr(run_module, "console", Console(file=buffer, width=100))
        monkeypatch.setattr(run_module, "Repl", _Repl)

        await run_module.start_session(
            agent,
            Theme.build(allow_animation=False),
            initial_prompt=None,
            verbose=False,
            show_status=True,
            config=_config(None),
            config_file=cfg.config_path(),
        )

        assert buffer.getvalue().count("domyn setup") == 1


class TestTheFlagIsGone:
    @staticmethod
    def _options(command: str) -> set[str]:
        import typer

        cmd = typer.main.get_command(app).commands[command]  # type: ignore[attr-defined]
        return {opt for param in cmd.params for opt in param.opts + param.secondary_opts}

    def test_run_has_no_autodiscovery_option_any_more(self) -> None:
        """`domyn run` does not scan `./skills`; setup finds local skills, so there is
        nothing for the option to switch."""
        options = self._options("run")

        assert not {o for o in options if "autodiscovery" in o or "default-skills" in o}

    def test_skills_come_only_from_the_config(self) -> None:
        """No extra folder from the command line: `domyn setup` lists the skills, and
        `domyn skills` shows what a session would load, nothing more."""
        assert "--skills" not in self._options("run")
        assert "--dir" not in self._options("skills")


class TestDomynSkills:
    @pytest.fixture
    def installed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
        """A saved config naming two installed skills, in a project with no ./skills."""
        paths = install_skills(["charts", "brainstorming"], tmp_path / "installed")
        config = _config({"source": {"type": "local_dir", "paths": [str(p) for p in paths]}})
        saved = tmp_path / "config.yaml"
        saved.write_text(yaml.safe_dump(config), encoding="utf-8")
        monkeypatch.setattr(cfg, "config_path", lambda *, prefer_project=True: saved)
        monkeypatch.chdir(tmp_path)
        return paths

    def test_it_lists_what_the_saved_config_installed(
        self, installed: list[Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        from domyn.agents.cli.commands.skills.command import _configured_skill_dirs

        assert _configured_skill_dirs() == installed

    def test_with_no_config_there_is_nothing_to_add(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from domyn.agents.cli.commands.skills.command import _configured_skill_dirs

        monkeypatch.setattr(cfg, "config_path", lambda *, prefer_project=True: tmp_path / "no.yaml")

        assert _configured_skill_dirs() == []

    def test_the_command_shows_each_installed_skill(self, installed: list[Path]) -> None:
        result = CliRunner().invoke(app, ["skills"])

        assert result.exit_code == 0, result.output
        assert "charts" in result.output
        assert "brainstorming" in result.output
