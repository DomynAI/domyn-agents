"""`domyn setup` asking which built-in tool bundles the agent mounts.

Every bundle starts ticked on a first setup, the browser included, the current choice
on a re-run, and what is unticked is not mounted — including all of them, which must stay that way on the
next run rather than read back as the default.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from domyn.agents.interfaces.cli import Theme, setup
from domyn.agents.interfaces.cli import config as cfg
from domyn.agents.interfaces.cli import select as select_mod
from domyn.agents.tools.builtin import BUNDLES

FIELDS = ["openai", "m", "", "", "", ""]
"""The answers up to the max tokens and context window; the tools step follows, then the skills."""

# Captured before the CLI conftest replaces it for every other wizard test.
_REAL_TOOLS_STEP = setup._choose_tools


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "GLOBAL_DIR", tmp_path / "home" / ".domyn")
    monkeypatch.setattr(cfg, "PROJECT_DIR", tmp_path / "nowhere" / ".domyn")
    monkeypatch.setattr(select_mod, "_interactive", lambda theme: False)
    monkeypatch.setattr(setup, "_choose_tools", _REAL_TOOLS_STEP)
    for name in ("OPENAI_API_KEY", "LLM_API_KEY"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


@pytest.fixture
def theme() -> Theme:
    return Theme.build(allow_animation=False)


@pytest.fixture
def console() -> MagicMock:
    return MagicMock()


def _number(name: str) -> str:
    return str(list(BUNDLES).index(name) + 1)


def _mounted(config: dict[str, Any]) -> list[str] | None:
    """The bundles the saved config mounts, or ``None`` when it names no bundle."""
    agent = config["agents"][0]
    if not agent.get("tool_bundles"):
        return None
    entries = {b["name"]: b for b in config.get("tool_bundles") or []}
    return list(entries[agent["tool_bundles"][0]]["bundles"])


def _run(
    console: MagicMock,
    theme: Theme,
    answers: list[Any],
    existing: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any] | None:
    with patch("builtins.input", side_effect=answers):
        return setup.run_setup(console, theme, existing, **kwargs)


class TestAFirstSetup:
    def test_an_empty_answer_mounts_every_bundle(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        result = _run(console, theme, [*FIELDS, "", ""], path=tmp_path / "config.yaml")

        assert result is not None
        assert result["agents"][0]["tool_bundles"] == ["cli_tools"]
        assert _mounted(result) == list(BUNDLES)
        assert "browser" in (_mounted(result) or [])

    def test_an_unticked_bundle_is_not_mounted(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        result = _run(console, theme, [*FIELDS, _number("files"), ""], path=tmp_path / "c.yaml")

        assert result is not None
        assert _mounted(result) == ["files"]

    def test_none_mounts_nothing(self, console: MagicMock, theme: Theme, tmp_path: Path) -> None:
        result = _run(console, theme, [*FIELDS, "none", ""], path=tmp_path / "config.yaml")

        assert result is not None
        assert _mounted(result) is None
        assert "tool_bundles" not in result

    def test_aborting_the_picker_saves_nothing(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        target = tmp_path / "config.yaml"
        assert _run(console, theme, [*FIELDS, EOFError], path=target) is None
        assert not target.exists()


class TestARerun:
    def test_starts_with_the_current_choice_ticked(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        target = tmp_path / "config.yaml"
        first = _run(console, theme, [*FIELDS, _number("shell"), ""], path=target)
        assert first is not None

        second = _run(console, theme, [""] * 8, first, path=target)

        assert second is not None
        assert _mounted(second) == ["shell"]

    def test_unticking_everything_stays_unticked(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        """No `tool_bundles` on the agent reads back as no tools, not as the default."""
        target = tmp_path / "config.yaml"
        first = _run(console, theme, [*FIELDS, "none", ""], path=target)
        assert first is not None

        second = _run(console, theme, [""] * 8, yaml.safe_load(target.read_text()), path=target)

        assert second is not None
        assert _mounted(second) is None

    def test_a_bundle_that_is_not_built_in_is_kept(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        target = tmp_path / "config.yaml"
        first = _run(console, theme, [*FIELDS, "", ""], path=target)
        assert first is not None
        first["tool_bundles"][0]["bundles"].append("handmade")

        second = _run(console, theme, [*[""] * 6, _number("files"), ""], first, path=target)

        assert second is not None
        assert _mounted(second) == ["files", "handmade"]

    def test_the_rest_of_the_bundle_entry_survives(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        target = tmp_path / "config.yaml"
        first = _run(console, theme, [*FIELDS, "", ""], path=target)
        assert first is not None
        first["tool_bundles"][0].update(
            {"allowed_env": ["GITHUB_TOKEN"], "isolate_environment": False}
        )

        second = _run(console, theme, [""] * 8, first, path=target)

        assert second is not None
        entry = second["tool_bundles"][0]
        assert entry["allowed_env"] == ["GITHUB_TOKEN"]
        assert entry["isolate_environment"] is False

    def test_a_provider_only_run_neither_asks_nor_changes_the_tools(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        target = tmp_path / "config.yaml"
        first = _run(console, theme, [*FIELDS, _number("files"), ""], path=target)
        assert first is not None

        second = _run(
            console,
            theme,
            ["anthropic", "m2", "", "", "", ""],
            first,
            path=target,
            provider_only=True,
        )

        assert second is not None
        assert _mounted(second) == ["files"]


class TestTheBrowser:
    """Browsing is a row of the checklist, ticked on a first setup like the others."""

    def test_unticking_it_on_a_first_setup_leaves_it_off(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        others = ",".join(_number(name) for name in ("files", "shell"))
        result = _run(console, theme, [*FIELDS, others, ""], path=tmp_path / "c.yaml")

        assert result is not None
        assert _mounted(result) == ["files", "shell"]

    def test_unticking_it_turns_it_off_again(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        target = tmp_path / "c.yaml"
        on = _run(console, theme, [*FIELDS, "", ""], path=target)
        assert on is not None
        assert "browser" in (_mounted(on) or [])
        others = ",".join(_number(name) for name in ("files", "shell"))

        off = _run(console, theme, [*[""] * 6, others, ""], on, path=target)

        assert off is not None
        assert _mounted(off) == ["files", "shell"]

    def test_a_missing_agent_browser_is_said_but_not_refused(
        self, console: MagicMock, theme: Theme, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(setup.shutil, "which", lambda name: None)

        result = _run(console, theme, [*FIELDS, "", ""], path=tmp_path / "c.yaml")

        assert result is not None
        assert "browser" in (_mounted(result) or [])
        printed = " ".join(str(c.args[0]) for c in console.print.call_args_list if c.args)
        assert "npm install -g agent-browser" in printed

    def test_no_hint_when_it_is_not_chosen(
        self, console: MagicMock, theme: Theme, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(setup.shutil, "which", lambda name: None)
        others = ",".join(_number(name) for name in ("files", "shell"))

        _run(console, theme, [*FIELDS, others, ""], path=tmp_path / "c.yaml")

        printed = " ".join(str(c.args[0]) for c in console.print.call_args_list if c.args)
        assert "not found on PATH" not in printed
