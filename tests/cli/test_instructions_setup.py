"""`domyn setup` asking where the agent's SOUL.md and AGENTS.md are.

A file is saved as given; a folder is walked and every file found is saved by
absolute path into the agent's ``standing_instructions``. The agent's name and
instruction are no longer asked: a re-run keeps whatever the config holds.
"""

import io
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml
from rich.console import Console

from domyn.agents.interfaces.cli import Theme, setup
from domyn.agents.interfaces.cli import config as cfg
from domyn.agents.interfaces.cli import select as select_mod

FIELDS = ["openai", "m", "", "", "", ""]
"""The answers up to the context window; the instructions question follows."""

# Captured before the CLI conftest replaces it for every other wizard test.
_REAL_STEP = setup._choose_instructions


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "GLOBAL_DIR", tmp_path / "home" / ".domyn")
    monkeypatch.setattr(cfg, "PROJECT_DIR", tmp_path / "nowhere" / ".domyn")
    monkeypatch.setattr(select_mod, "_interactive", lambda theme: False)
    monkeypatch.setattr(setup, "_choose_instructions", _REAL_STEP)
    for name in ("OPENAI_API_KEY", "LLM_API_KEY"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def theme() -> Theme:
    return Theme.build(allow_animation=False)


@pytest.fixture
def out() -> io.StringIO:
    return io.StringIO()


@pytest.fixture
def console(out: io.StringIO) -> Console:
    return Console(file=out, width=300, force_terminal=False)


def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _run(
    console: Console, theme: Theme, answer: str, existing: dict[str, Any] | None = None
) -> dict[str, Any]:
    """A run answering ``answer`` to the instructions question; ``none`` for the built-ins."""
    fields = [""] * 6 if existing else FIELDS
    with patch("builtins.input", side_effect=[*fields, answer, "none"]):
        result = setup.run_setup(console, theme, existing, path=Path("config.yaml"))
    assert result is not None
    return result


def _paths(config: dict[str, Any]) -> list[str] | None:
    declared = config["agents"][0].get("standing_instructions")
    return None if declared is None else declared["source"]["paths"]


class TestAFolder:
    def test_its_first_soul_and_every_agents_file_are_saved(
        self, console: Console, theme: Theme, tmp_path: Path
    ) -> None:
        notes = tmp_path / "notes"
        soul = _write(notes / "SOUL.md")
        _write(notes / "b" / "SOUL.md")
        top = _write(notes / "a" / "AGENTS.md")
        deep = _write(notes / "b" / "c" / "AGENTS.md")

        result = _run(console, theme, str(notes))

        assert _paths(result) == [str(soul), str(top), str(deep)]
        assert result["agents"][0]["standing_instructions"]["source"]["type"] == "local_dir"

    def test_what_is_found_is_named(
        self, console: Console, theme: Theme, tmp_path: Path, out: io.StringIO
    ) -> None:
        soul = _write(tmp_path / "notes" / "SOUL.md")

        _run(console, theme, str(tmp_path / "notes"))

        assert str(soul) in out.getvalue()

    def test_a_relative_answer_is_saved_absolute(self, console: Console, theme: Theme) -> None:
        soul = _write(Path("notes") / "SOUL.md")

        result = _run(console, theme, "notes")

        assert _paths(result) == [str(soul.absolute())]

    def test_the_saved_files_are_what_the_agent_loads(
        self, console: Console, theme: Theme, tmp_path: Path
    ) -> None:
        _write(tmp_path / "notes" / "SOUL.md", "Answer in Italian.")
        _write(tmp_path / "notes" / "x" / "AGENTS.md", "Project notes.")
        result = _run(console, theme, str(tmp_path / "notes"))

        _, agent, _ = cfg.current_settings_from(yaml.safe_load(Path("config.yaml").read_text()))

        assert agent.standing_instructions == result["agents"][0]["standing_instructions"]


class TestAFile:
    def test_any_name_is_saved_as_given(
        self, console: Console, theme: Theme, tmp_path: Path
    ) -> None:
        brief = _write(tmp_path / "brief.md")

        assert _paths(_run(console, theme, str(brief))) == [str(brief)]


class TestNothingSaved:
    def test_a_folder_with_nothing_in_it(
        self, console: Console, theme: Theme, tmp_path: Path, out: io.StringIO
    ) -> None:
        (tmp_path / "empty").mkdir()

        result = _run(console, theme, str(tmp_path / "empty"))

        assert _paths(result) is None
        assert "No SOUL.md or AGENTS.md found" in out.getvalue()

    def test_a_path_that_is_not_there(
        self, console: Console, theme: Theme, tmp_path: Path, out: io.StringIO
    ) -> None:
        result = _run(console, theme, str(tmp_path / "missing"))

        assert _paths(result) is None
        assert "does not exist" in out.getvalue()

    def test_a_link(self, console: Console, theme: Theme, tmp_path: Path, out: io.StringIO) -> None:
        _write(tmp_path / "real" / "SOUL.md")
        (tmp_path / "link").symlink_to(tmp_path / "real")

        result = _run(console, theme, str(tmp_path / "link"))

        assert _paths(result) is None
        assert "is a link" in out.getvalue()

    def test_an_empty_answer(self, console: Console, theme: Theme) -> None:
        assert _paths(_run(console, theme, "")) is None


class TestAReRun:
    def test_an_empty_answer_removes_what_was_there(
        self, console: Console, theme: Theme, tmp_path: Path, out: io.StringIO
    ) -> None:
        soul = _write(tmp_path / "notes" / "SOUL.md")
        first = _run(console, theme, str(tmp_path / "notes"))

        second = _run(console, theme, "", first)

        assert _paths(second) is None
        assert f"Instructions currently: {soul}" in out.getvalue()

    def test_a_source_setup_did_not_write_is_kept_without_asking(
        self, console: Console, theme: Theme
    ) -> None:
        declared = {"source": {"type": "memory", "soul": "Be brief."}}
        existing = cfg.configuration_v1_dict_from(
            cfg.ProviderConfig(type="openai", model_name="m"),
            cfg.AgentConfig(standing_instructions=declared),
            cfg.CompactionConfig(),
        )

        # No answer for the instructions question: it is not asked.
        with patch("builtins.input", side_effect=[*[""] * 6, "none"]):
            result = setup.run_setup(console, theme, existing, path=Path("config.yaml"))

        assert result is not None
        assert result["agents"][0]["standing_instructions"] == declared

    def test_the_agents_name_and_instruction_are_kept(self, console: Console, theme: Theme) -> None:
        """Setup no longer asks for them, so what the config holds stays."""
        existing = cfg.configuration_v1_dict_from(
            cfg.ProviderConfig(type="openai", model_name="m"),
            cfg.AgentConfig(name="Mine", instruction="Be terse."),
            cfg.CompactionConfig(),
        )

        result = _run(console, theme, "", existing)

        assert result["entrypoint"] == "Mine"
        assert result["agents"][0]["instruction"] == "Be terse."

    def test_a_first_setup_gets_the_defaults(self, console: Console, theme: Theme) -> None:
        result = _run(console, theme, "")

        assert result["agents"][0]["name"] == cfg.AgentConfig().name
        assert result["agents"][0]["instruction"] == cfg.AgentConfig().instruction
