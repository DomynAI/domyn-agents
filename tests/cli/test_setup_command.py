"""Tests for `domyn setup` (domyn/agents/cli/commands/setup/command.py)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
import yaml

from domyn.agents.cli.commands.setup import setup_command
from domyn.agents.interfaces.cli import config as cfg
from domyn.agents.interfaces.cli import select as select_mod


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cfg, "GLOBAL_DIR", tmp_path / "home" / ".domyn")
    monkeypatch.setattr(cfg, "PROJECT_DIR", tmp_path / "project" / ".domyn")
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


@pytest.fixture(autouse=True)
def force_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(select_mod, "_interactive", lambda theme: False)


class TestShow:
    def test_no_configuration_yet_is_reported_gracefully(self) -> None:
        with patch("domyn.agents.cli.commands.setup.command.console", MagicMock()) as console:
            setup_command(show=True)

        printed = " ".join(str(c.args[0]) for c in console.print.call_args_list if c.args)
        assert "No configuration yet" in printed

    def test_shows_an_existing_configurationv1_file(self) -> None:
        with (
            patch(
                "builtins.input",
                side_effect=[
                    "openai",
                    "gpt-4o-mini",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ],
            ),
            patch("domyn.agents.cli.commands.setup.command.console", MagicMock()),
        ):
            setup_command(show=False)

        with patch("domyn.agents.cli.commands.setup.command.console", MagicMock()) as console:
            setup_command(show=True)

        printed = " ".join(str(c.args[0]) for c in console.print.call_args_list if c.args)
        assert "gpt-4o-mini" in printed

    def test_an_old_shaped_file_shows_as_an_empty_configuration(self) -> None:
        """No migration any more: an old CliConfig-shaped file (`provider:`/
        `agent:` keys, no `entrypoint:`) isn't recognized as ConfigurationV1,
        so it renders as defaults rather than being converted or crashing."""
        cfg.config_path().parent.mkdir(parents=True, exist_ok=True)
        cfg.config_path().write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "provider": {"type": "openai", "model_name": "legacy-model"},
                    "agent": {"name": "LegacyAgent", "description": "d", "instruction": "i"},
                }
            ),
            encoding="utf-8",
        )

        with patch("domyn.agents.cli.commands.setup.command.console", MagicMock()) as console:
            setup_command(show=True)  # must not raise

        printed = " ".join(str(c.args[0]) for c in console.print.call_args_list if c.args)
        assert "legacy-model" not in printed

        on_disk = yaml.safe_load(cfg.config_path().read_text())
        assert "provider" in on_disk  # left untouched on disk


class TestRun:
    def test_writes_a_configurationv1_file(self) -> None:
        with (
            patch(
                "builtins.input",
                side_effect=[
                    "openai",
                    "gpt-4o-mini",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ],
            ),
            patch("domyn.agents.cli.commands.setup.command.console", MagicMock()),
        ):
            setup_command(show=False)

        on_disk = yaml.safe_load(cfg.config_path().read_text())
        assert on_disk["entrypoint"] == "Assistant"

    def test_aborting_exits_nonzero(self) -> None:
        with (
            patch("builtins.input", side_effect=EOFError),
            patch("domyn.agents.cli.commands.setup.command.console", MagicMock()),
            pytest.raises(typer.Exit) as exc_info,
        ):
            setup_command(show=False)

        assert exc_info.value.exit_code == 1
