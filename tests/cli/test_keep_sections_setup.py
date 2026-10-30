"""`domyn setup` asking, on a re-run, whether to keep each section as the config has it.

Four sections — model, instructions, tools, skills — each shown first and kept on a yes
(the default, so Enter keeps), which skips its questions. Not asked on a first setup, on
a ``provider_only`` run (``/model``), or for a model the config leaves incomplete.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from domyn.agents.interfaces.cli import Theme, setup
from domyn.agents.interfaces.cli import config as cfg
from domyn.agents.interfaces.cli import select as select_mod

FIELDS = ["openai", "m2", "", "", "", ""]
"""The model section's answers: provider, model, url, api key, max tokens, context window."""

SKILLS = {"source": {"type": "local_dir", "paths": ["/somewhere/skills/one"]}}
INSTRUCTIONS = {"source": {"type": "local_dir", "paths": ["/somewhere/SOUL.md", ".domyn"]}}

# Captured before the CLI conftest replaces it for every other wizard test.
_REAL_KEEP = setup._keep


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "GLOBAL_DIR", tmp_path / "home" / ".domyn")
    monkeypatch.setattr(cfg, "PROJECT_DIR", tmp_path / "nowhere" / ".domyn")
    monkeypatch.setattr(select_mod, "_interactive", lambda theme: False)
    for name in ("OPENAI_API_KEY", "LLM_API_KEY"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


@pytest.fixture(autouse=True)
def keep(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """The real question, recorded."""
    spy = MagicMock(side_effect=_REAL_KEEP)
    monkeypatch.setattr(setup, "_keep", spy)
    return spy


@pytest.fixture(autouse=True)
def steps(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """The sections' own steps, recorded, each keeping the config as it is."""
    recorded = {}
    for name in ("_choose_instructions", "_choose_tools", "_choose_builtin_skills"):
        recorded[name] = MagicMock(side_effect=lambda console, theme, current, **_: current)
        monkeypatch.setattr(setup, name, recorded[name])
    return recorded


@pytest.fixture
def theme() -> Theme:
    return Theme.build(allow_animation=False)


@pytest.fixture
def console() -> MagicMock:
    return MagicMock()


def _existing(model_name: str = "m1") -> dict[str, Any]:
    return cfg.configuration_v1_dict_from(
        cfg.ProviderConfig(type="openai", model_name=model_name, max_tokens=8000),
        cfg.AgentConfig(
            instruction="Be terse.",
            tools=["files"],
            skills=SKILLS,
            standing_instructions=INSTRUCTIONS,
        ),
        cfg.CompactionConfig(),
    )


def _run(
    console: MagicMock,
    theme: Theme,
    answers: list[Any],
    existing: dict[str, Any] | None,
    target: Path,
    **kwargs: Any,
) -> dict[str, Any] | None:
    with patch("builtins.input", side_effect=answers):
        return setup.run_setup(console, theme, existing, path=target, **kwargs)


def _asked(keep: MagicMock) -> list[str]:
    return [call.args[2] for call in keep.call_args_list]


class TestNotAsked:
    def test_on_a_first_setup(
        self, console: MagicMock, theme: Theme, tmp_path: Path, keep: MagicMock
    ) -> None:
        result = _run(console, theme, FIELDS, None, tmp_path / "config.yaml")

        assert result is not None
        assert keep.call_count == 0

    def test_on_a_provider_only_run(
        self,
        console: MagicMock,
        theme: Theme,
        tmp_path: Path,
        keep: MagicMock,
        steps: dict[str, MagicMock],
    ) -> None:
        result = _run(
            console, theme, FIELDS, _existing(), tmp_path / "config.yaml", provider_only=True
        )

        assert result is not None
        assert keep.call_count == 0
        assert result["llm_providers"][0]["model_name"] == "m2"

    def test_for_an_incomplete_model(
        self, console: MagicMock, theme: Theme, tmp_path: Path, keep: MagicMock
    ) -> None:
        result = _run(
            console, theme, [*FIELDS, "", "", ""], _existing(model_name=""), tmp_path / "c.yaml"
        )

        assert result is not None
        assert _asked(keep) == ["instructions", "tools", "skills"]
        assert result["llm_providers"][0]["model_name"] == "m2"


class TestARerun:
    def test_enter_on_every_section_keeps_the_config_as_it_was(
        self,
        console: MagicMock,
        theme: Theme,
        tmp_path: Path,
        keep: MagicMock,
        steps: dict[str, MagicMock],
    ) -> None:
        existing = _existing()
        result = _run(console, theme, ["", "", "", ""], existing, tmp_path / "config.yaml")

        assert result == existing
        assert _asked(keep) == ["model", "instructions", "tools", "skills"]
        assert all(step.call_count == 0 for step in steps.values())

    def test_no_on_the_model_asks_only_for_the_model(
        self,
        console: MagicMock,
        theme: Theme,
        tmp_path: Path,
        steps: dict[str, MagicMock],
    ) -> None:
        existing = _existing()
        result = _run(console, theme, ["n", *FIELDS, "", "", ""], existing, tmp_path / "c.yaml")

        assert result is not None
        assert result["llm_providers"][0]["model_name"] == "m2"
        assert result["agents"] == existing["agents"]
        assert result["tool_bundles"] == existing["tool_bundles"]
        assert all(step.call_count == 0 for step in steps.values())

    def test_no_on_the_tools_runs_only_the_tools_step(
        self,
        console: MagicMock,
        theme: Theme,
        tmp_path: Path,
        steps: dict[str, MagicMock],
    ) -> None:
        result = _run(console, theme, ["", "", "n", ""], _existing(), tmp_path / "config.yaml")

        assert result is not None
        assert steps["_choose_tools"].call_count == 1
        assert steps["_choose_instructions"].call_count == 0
        assert steps["_choose_builtin_skills"].call_count == 0

    def test_shows_what_each_section_holds(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        _run(console, theme, ["", "", "", ""], _existing(), tmp_path / "config.yaml")

        printed = [str(c.args[0]) for c in console.print.call_args_list if c.args]
        for line in (
            "model: m1",
            "Be terse.",
            "max tokens: 8000",
            "/somewhere/SOUL.md",
            ".domyn",
            "files",
            "/somewhere/skills/one",
        ):
            assert f"    {line}" in printed
        assert "  Extra instructions for the agent:" in printed

    def test_aborting_a_keep_question_saves_nothing(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        target = tmp_path / "config.yaml"

        assert _run(console, theme, [EOFError], _existing(), target) is None
        assert not target.exists()
