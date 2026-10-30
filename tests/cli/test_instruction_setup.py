"""`domyn setup` asking for the agent's own ``instruction``, the start of its system prompt.

Asked first in the instructions section, before the ``SOUL.md`` / ``AGENTS.md`` path;
Enter keeps the current one. Not asked by ``/model``, nor when the section is kept.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from domyn.agents.interfaces.cli import Theme, setup
from domyn.agents.interfaces.cli import config as cfg
from domyn.agents.interfaces.cli import select as select_mod

FIELDS = ["openai", "m", "", "", "", ""]
"""The model section's answers: provider, model, url, api key, max tokens, context window."""

NAME = ""
"""The agent's name, asked just before the instruction: Enter keeps it."""

# Captured before the CLI conftest wraps it for every other wizard test.
_REAL_ASK = setup._ask


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "GLOBAL_DIR", tmp_path / "home" / ".domyn")
    monkeypatch.setattr(cfg, "PROJECT_DIR", tmp_path / "nowhere" / ".domyn")
    monkeypatch.setattr(select_mod, "_interactive", lambda theme: False)
    monkeypatch.setattr(setup, "_ask", _REAL_ASK)
    monkeypatch.setattr(
        setup, "_choose_builtin_skills", lambda console, theme, current, **_: current
    )
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
    existing: dict[str, Any] | None,
    target: Path,
    **kwargs: Any,
) -> dict[str, Any] | None:
    with patch("builtins.input", side_effect=answers):
        return setup.run_setup(console, theme, existing, path=target, **kwargs)


def _existing(instruction: str) -> dict[str, Any]:
    return cfg.configuration_v1_dict_from(
        cfg.ProviderConfig(type="openai", model_name="m", max_tokens=8000),
        cfg.AgentConfig(instruction=instruction),
        cfg.CompactionConfig(),
    )


class TestTheSystemInstruction:
    def test_is_saved_as_typed(self, console: MagicMock, theme: Theme, tmp_path: Path) -> None:
        result = _run(
            console, theme, [*FIELDS, NAME, "Answer in Italian."], None, tmp_path / "config.yaml"
        )

        assert result is not None
        assert result["agents"][0]["instruction"] == "Answer in Italian."

    def test_enter_keeps_the_default_on_a_first_setup(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        result = _run(console, theme, [*FIELDS, NAME, ""], None, tmp_path / "config.yaml")

        assert result is not None
        assert result["agents"][0]["instruction"] == cfg.AgentConfig().instruction

    def test_enter_keeps_the_current_one_on_a_rerun(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        result = _run(
            console, theme, [*FIELDS, NAME, ""], _existing("Be terse."), tmp_path / "config.yaml"
        )

        assert result is not None
        assert result["agents"][0]["instruction"] == "Be terse."

    def test_is_not_asked_by_a_provider_only_run(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        result = _run(
            console,
            theme,
            FIELDS,
            _existing("Be terse."),
            tmp_path / "config.yaml",
            provider_only=True,
        )

        assert result is not None
        assert result["agents"][0]["instruction"] == "Be terse."
