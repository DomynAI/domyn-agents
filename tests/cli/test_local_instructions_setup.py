"""`domyn setup` turning on ``./.domyn`` / ``./.agents`` instructions autodiscovery.

One question, asked after the instructions path. On yes, ``.domyn`` and ``.agents`` are
listed as written, **after** the user's own paths, and walked by the loader, subfolders
included: the first ``SOUL.md`` wins, so the
user's beats a project's, and every ``AGENTS.md`` is included with the project's last.
Relative, they are read every turn from the folder ``domyn run`` starts in.
"""

import io
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from domyn.agents.instructions import InstructionRuntime, LocalDirInstructionsSource
from domyn.agents.interfaces.cli import Theme, setup
from domyn.agents.interfaces.cli import config as cfg
from domyn.agents.interfaces.cli import select as select_mod

AUTO = [".domyn", ".agents"]
"""What a yes writes, in precedence order."""

FIELDS = ["openai", "m", "", "", "", ""]
"""The answers up to the context window; the instructions path and this question follow."""

# Captured before the CLI conftest replaces them for every other wizard test.
_REAL_PATH_STEP = setup._choose_instructions
_REAL_LOCAL_STEP = setup._choose_local_instructions


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "GLOBAL_DIR", tmp_path / "home" / ".domyn")
    monkeypatch.setattr(cfg, "PROJECT_DIR", tmp_path / "nowhere" / ".domyn")
    monkeypatch.setattr(select_mod, "_interactive", lambda theme: False)
    monkeypatch.setattr(setup, "_choose_instructions", _REAL_PATH_STEP)
    monkeypatch.setattr(setup, "_choose_local_instructions", _REAL_LOCAL_STEP)
    for module in (cfg, setup):
        monkeypatch.setattr(module, "AUTODISCOVERY_DIRS", tuple(Path(p) for p in AUTO))
    for name in ("OPENAI_API_KEY", "LLM_API_KEY"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def theme() -> Theme:
    return Theme.build(allow_animation=False)


@pytest.fixture
def console() -> MagicMock:
    return MagicMock()


def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _existing(paths: list[str]) -> dict[str, Any]:
    return cfg.configuration_v1_dict_from(
        cfg.ProviderConfig(type="openai", model_name="m"),
        cfg.AgentConfig(standing_instructions={"source": {"type": "local_dir", "paths": paths}}),
        cfg.CompactionConfig(),
    )


def _run(
    console: Any, theme: Theme, path: str, local: str, existing: dict[str, Any] | None = None
) -> dict[str, Any]:
    """A run answering ``path`` to the instructions path, ``local`` to this question, and
    ``none`` for the built-in skills."""
    fields = [""] * 6 if existing else FIELDS
    with patch("builtins.input", side_effect=[*fields, path, local, "none"]):
        result = setup.run_setup(console, theme, existing, path=Path("config.yaml"))
    assert result is not None
    return result


def _paths(config: dict[str, Any]) -> list[str]:
    declared = config["agents"][0].get("standing_instructions")
    return [] if declared is None else declared["source"]["paths"]


class TestTheQuestion:
    def test_yes_is_the_default_on_a_first_setup(self, console: MagicMock, theme: Theme) -> None:
        assert _paths(_run(console, theme, "", "")) == AUTO

    def test_answering_no_lists_nothing(self, console: MagicMock, theme: Theme) -> None:
        assert _paths(_run(console, theme, "", "no")) == []

    def test_the_folders_come_after_the_users_own(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        soul = _write(tmp_path / "notes" / "SOUL.md")

        result = _run(console, theme, str(tmp_path / "notes"), "yes")

        assert _paths(result) == [str(soul), *AUTO]

    def test_it_says_where_they_are_read_from(self, theme: Theme) -> None:
        buffer = io.StringIO()

        _run(Console(file=buffer, width=200), theme, "", "yes")

        assert "./.domyn and ./.agents are loaded from the folder domyn runs in" in (
            buffer.getvalue()
        )

    def test_ctrl_c_at_the_question_saves_nothing(self, console: MagicMock, theme: Theme) -> None:
        with patch("builtins.input", side_effect=[*FIELDS, "", EOFError]):
            assert setup.run_setup(console, theme, path=Path("config.yaml")) is None
        assert not Path("config.yaml").exists()


class TestAReRun:
    def test_an_empty_answer_keeps_it_on(self, console: MagicMock, theme: Theme) -> None:
        first = _run(console, theme, "", "yes")

        assert _paths(_run(console, theme, "", "", first)) == AUTO

    def test_an_empty_answer_keeps_it_off(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        brief = str(_write(tmp_path / "brief.md"))

        second = _run(console, theme, brief, "", _existing([brief]))

        assert _paths(second) == [brief]

    def test_answering_no_turns_it_off_and_keeps_the_users_paths(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        brief = str(_write(tmp_path / "brief.md"))

        second = _run(console, theme, brief, "no", _existing([brief, *AUTO]))

        assert _paths(second) == [brief]

    def test_an_empty_path_answer_keeps_the_folders(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        """Empty removes the user's paths, which are this question's business no more
        than the folders are the path question's."""
        brief = str(_write(tmp_path / "brief.md"))

        second = _run(console, theme, "", "", _existing([brief, *AUTO]))

        assert _paths(second) == AUTO

    def test_they_are_listed_once_however_often_setup_runs(
        self, console: MagicMock, theme: Theme
    ) -> None:
        config = _run(console, theme, "", "yes")
        for _ in range(2):
            config = _run(console, theme, "", "yes", config)

        assert _paths(config) == AUTO

    def test_a_source_setup_did_not_write_is_kept_without_asking(
        self, console: MagicMock, theme: Theme
    ) -> None:
        declared = {"source": {"type": "memory", "soul": "Be brief."}}

        assert setup._choose_local_instructions(console, theme, declared, first_setup=False) == (
            declared
        )


class TestWhatTheAgentLoads:
    @pytest.mark.asyncio
    async def test_the_users_soul_wins_and_the_projects_brief_comes_last(
        self, console: MagicMock, theme: Theme, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write(tmp_path / "notes" / "SOUL.md", "user soul")
        _write(tmp_path / "notes" / "AGENTS.md", "user brief")
        result = _run(console, theme, str(tmp_path / "notes"), "yes")
        project = tmp_path / "project"
        _write(project / ".domyn" / "instructions" / "SOUL.md", "project soul")
        _write(project / ".domyn" / "AGENTS.md", "domyn brief")
        _write(project / ".domyn" / "instructions" / "extra" / "AGENTS.md", "domyn deep brief")
        _write(project / ".domyn" / "conversations" / "c1" / "AGENTS.md", "scratch")
        _write(project / ".agents" / "AGENTS.md", "agents brief")
        monkeypatch.chdir(project)

        source = LocalDirInstructionsSource(paths=[Path(p) for p in _paths(result)])
        loaded = await InstructionRuntime(source=source).load()

        assert [one.text for one in loaded.soul] == ["user soul"]
        assert [one.text for one in loaded.agent] == [
            "user brief",
            "domyn brief",
            "domyn deep brief",
            "agents brief",
        ]
