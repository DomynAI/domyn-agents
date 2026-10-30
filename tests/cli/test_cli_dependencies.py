"""Tests for domyn/agents/cli/cli_dependencies.py, and for the commands that call it."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from domyn.agents import Agent
from domyn.agents.cli.cli_dependencies import cli_requirements, install_cli_requirements_here

_MODULE = "domyn.agents.cli.cli_dependencies"


class TestCliRequirements:
    def test_reads_the_cli_extra_from_the_installed_metadata(self):
        assert cli_requirements() == ["prompt-toolkit>=3.0.50"]

    def test_keeps_only_the_cli_extra_and_drops_its_marker(self):
        metadata = [
            "typer>=0.9",
            'prompt-toolkit>=3.0.50; extra == "cli"',
            "opentelemetry-api>=1.28.0; extra == 'tracing'",
        ]
        with patch("importlib.metadata.requires", return_value=metadata):
            assert cli_requirements() == ["prompt-toolkit>=3.0.50"]


class TestInstallCliRequirementsHere:
    """``uv venv && uv run domyn init && uv run domyn run`` runs the session in the
    checkout's environment, synced without extras: the line editor must be put there."""

    @staticmethod
    def _virtualenv(fake_sys: MagicMock) -> None:
        fake_sys.prefix, fake_sys.base_prefix = "/venv", "/usr"
        fake_sys.executable = "/venv/bin/python"

    def test_installs_into_the_running_interpreter(self):
        with (
            patch(f"{_MODULE}.find_spec", side_effect=[None, object()]),
            patch(f"{_MODULE}.sys") as fake_sys,
            patch(f"{_MODULE}.subprocess.run") as run,
        ):
            self._virtualenv(fake_sys)
            assert install_cli_requirements_here() is True
        run.assert_called_once()
        assert run.call_args.args[0] == [
            "uv",
            "pip",
            "install",
            "--python",
            "/venv/bin/python",
            "prompt-toolkit>=3.0.50",
        ]

    def test_falls_back_to_pip_without_uv(self):
        with (
            patch(f"{_MODULE}.find_spec", side_effect=[None, object()]),
            patch(f"{_MODULE}.sys") as fake_sys,
            patch(f"{_MODULE}.subprocess.run", side_effect=[FileNotFoundError, None]) as run,
        ):
            self._virtualenv(fake_sys)
            assert install_cli_requirements_here() is True
        assert run.call_args.args[0] == [
            "/venv/bin/python",
            "-m",
            "pip",
            "install",
            "prompt-toolkit>=3.0.50",
        ]

    def test_reports_a_failed_install(self, capsys: pytest.CaptureFixture[str]):
        failure = subprocess.CalledProcessError(1, "install", stderr="no network")
        with (
            patch(f"{_MODULE}.find_spec", return_value=None),
            patch(f"{_MODULE}.sys") as fake_sys,
            patch(f"{_MODULE}.subprocess.run", side_effect=failure),
        ):
            self._virtualenv(fake_sys)
            assert install_cli_requirements_here() is False
        assert "could not install" in capsys.readouterr().err

    def test_does_nothing_when_already_installed(self):
        with patch(f"{_MODULE}.subprocess.run") as run:
            # prompt_toolkit is in the dev environment
            assert install_cli_requirements_here() is True
        run.assert_not_called()

    def test_never_installs_into_a_system_interpreter(self):
        with (
            patch(f"{_MODULE}.find_spec", return_value=None),
            patch(f"{_MODULE}.sys") as fake_sys,
            patch(f"{_MODULE}.subprocess.run") as run,
        ):
            fake_sys.prefix = fake_sys.base_prefix = "/usr"
            assert install_cli_requirements_here() is False
        run.assert_not_called()


class TestDomynRunInstallsTheLineEditor:
    """A fresh ``uv run domyn run`` from a checkout has no line editor until it is
    installed, and the setup wizard is the first thing to read a line."""

    @staticmethod
    def _run(monkeypatch: pytest.MonkeyPatch, *, terminal: bool) -> list[str]:
        """What ran, in order: the install, and resolving the agent (or the wizard)."""
        import typer

        from domyn.agents.cli.commands.run import command

        calls: list[str] = []

        async def _resolve(*args: object, **kwargs: object) -> tuple[Agent, dict]:
            calls.append("resolve")
            raise typer.Exit(1)

        def _install() -> bool:
            calls.append("install")
            return True

        monkeypatch.setattr(command, "interactive_terminal", lambda: terminal)
        monkeypatch.setattr(command, "install_cli_requirements_here", _install)
        monkeypatch.setattr(command, "resolve_agent_from_file", _resolve)

        with pytest.raises(typer.Exit):
            command.run_command(
                file=None,
                prompt=None,
                agent_name=None,
                expand_skills=False,
                skill_sections=False,
                verbose=False,
                no_banner=True,
            )
        return calls

    def test_installs_before_the_agent_or_the_wizard(self, monkeypatch: pytest.MonkeyPatch):
        assert self._run(monkeypatch, terminal=True) == ["install", "resolve"]

    def test_a_pipe_installs_nothing(self, monkeypatch: pytest.MonkeyPatch):
        assert self._run(monkeypatch, terminal=False) == ["resolve"]
