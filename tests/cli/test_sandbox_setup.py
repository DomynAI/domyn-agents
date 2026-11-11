"""`domyn setup` asking where shell commands run: sandboxed in Docker, or on this machine.

A sub-question of the tools step, asked only when `shell` is among the tools. The
answer is written as the shell bundle's `sandbox`. Sandboxed is the starting choice
unless Docker is not installed; a re-run starts on what the config says.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from domyn.agents.interfaces.cli import Theme, setup
from domyn.agents.interfaces.cli import config as cfg
from domyn.agents.interfaces.cli import select as select_mod
from domyn.agents.tools.builtin import BUNDLES

FIELDS = ["openai", "m", "", "", "", ""]

# Captured before the CLI conftest replaces them for every other wizard test.
_REAL_TOOLS_STEP = setup._choose_tools
_REAL_SANDBOX_STEP = setup._choose_sandbox


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "GLOBAL_DIR", tmp_path / "home" / ".domyn")
    monkeypatch.setattr(select_mod, "_interactive", lambda theme: False)
    monkeypatch.setattr(setup, "_choose_tools", _REAL_TOOLS_STEP)
    monkeypatch.setattr(setup, "_choose_sandbox", _REAL_SANDBOX_STEP)
    monkeypatch.setattr(setup, "_docker_installed", lambda binary="docker": True)
    monkeypatch.setattr(setup, "_docker_answers", lambda binary="docker": True)
    for name in ("OPENAI_API_KEY", "LLM_API_KEY"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


@pytest.fixture
def theme() -> Theme:
    return Theme.build(allow_animation=False)


@pytest.fixture
def console() -> MagicMock:
    return MagicMock()


def _run(
    console: MagicMock,
    theme: Theme,
    answers: list[Any],
    existing: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any] | None:
    with patch("builtins.input", side_effect=answers):
        return setup.run_setup(console, theme, existing, **kwargs)


def _sandbox_of(config: dict[str, Any]) -> str | None:
    entries = config.get("tool_bundles") or []
    return entries[0].get("sandbox") if entries else None


def _printed(console: MagicMock) -> str:
    return " ".join(str(call.args[0]) for call in console.print.call_args_list if call.args)


def _shell_only() -> str:
    return str(list(BUNDLES).index("shell") + 1)


class TestAFirstSetup:
    def test_sandboxed_is_the_default(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        result = _run(console, theme, [*FIELDS, "", "", ""], path=tmp_path / "c.yaml")

        assert result is not None
        assert _sandbox_of(result) == "docker"

    def test_local_can_be_chosen(self, console: MagicMock, theme: Theme, tmp_path: Path) -> None:
        result = _run(console, theme, [*FIELDS, "", "2", ""], path=tmp_path / "c.yaml")

        assert result is not None
        assert _sandbox_of(result) == "local"

    def test_without_docker_installed_local_is_the_default(
        self, console: MagicMock, theme: Theme, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(setup, "_docker_installed", lambda binary="docker": False)

        result = _run(console, theme, [*FIELDS, "", "", ""], path=tmp_path / "c.yaml")

        assert result is not None
        assert _sandbox_of(result) == "local"

    def test_choosing_docker_when_it_is_not_installed_warns_and_saves(
        self, console: MagicMock, theme: Theme, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(setup, "_docker_installed", lambda binary="docker": False)

        result = _run(console, theme, [*FIELDS, "", "1", ""], path=tmp_path / "c.yaml")

        assert result is not None
        assert _sandbox_of(result) == "docker"
        assert "Docker was not found" in _printed(console)

    def test_choosing_docker_while_it_is_down_warns_and_saves(
        self, console: MagicMock, theme: Theme, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(setup, "_docker_answers", lambda binary="docker": False)

        result = _run(console, theme, [*FIELDS, "", "", ""], path=tmp_path / "c.yaml")

        assert result is not None
        assert _sandbox_of(result) == "docker"
        assert "installed but not running" in _printed(console)

    def test_it_is_not_asked_without_the_shell(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        # One answer fewer: nothing to say about where commands run.
        result = _run(
            console,
            theme,
            [*FIELDS, str(list(BUNDLES).index("files") + 1), ""],
            path=tmp_path / "c.yaml",
        )

        assert result is not None
        assert "Where should shell commands run?" not in _printed(console)
        assert _sandbox_of(result) in (None, "local")


class TestARerun:
    def test_starts_on_what_the_config_says(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        target = tmp_path / "c.yaml"
        first = _run(console, theme, [*FIELDS, "", "2", ""], path=target)
        assert first is not None and _sandbox_of(first) == "local"

        # Enter on every question keeps a local config local, though a first setup
        # would have started on sandboxed.
        second = _run(console, theme, [""] * 10, first, path=target)
        assert second is not None
        assert _sandbox_of(second) == "local"

    def test_a_sandboxed_config_stays_sandboxed(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        target = tmp_path / "c.yaml"
        first = _run(console, theme, [*FIELDS, "", "1", ""], path=target)
        assert first is not None and _sandbox_of(first) == "docker"

        second = _run(console, theme, [""] * 10, first, path=target)
        assert second is not None
        assert _sandbox_of(second) == "docker"

    def test_the_docker_options_survive_a_rerun(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        target = tmp_path / "c.yaml"
        first = _run(console, theme, [*FIELDS, "", "1", ""], path=target)
        assert first is not None
        first["tool_bundles"][0]["docker"] = {"allow": ["example.org"]}

        second = _run(console, theme, [""] * 10, first, path=target)

        assert second is not None
        assert second["tool_bundles"][0]["docker"] == {"allow": ["example.org"]}
