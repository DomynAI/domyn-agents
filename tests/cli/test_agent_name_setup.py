"""`domyn setup` asking for the agent's name, the config's entrypoint.

Asked in the instructions section, just before the extra instructions; Enter keeps the
current one, and an erased field is ``AgentConfig``'s default. Not asked by ``/model``,
nor when the section is kept.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from domyn.agents.agents.agent import MAX_AGENT_NAME_LENGTH
from domyn.agents.interfaces.cli import Theme, setup
from domyn.agents.interfaces.cli import config as cfg
from domyn.agents.interfaces.cli import select as select_mod

FIELDS = ["openai", "m", "", "", "", ""]
"""The model section's answers: provider, model, url, api key, max tokens, context window."""

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


def _existing(name: str) -> dict[str, Any]:
    config = cfg.configuration_v1_dict_from(
        cfg.ProviderConfig(type="openai", model_name="m", max_tokens=8000),
        cfg.AgentConfig(name=name),
        cfg.CompactionConfig(),
    )
    config["agents"][0]["memory"] = {"type": "filesystem"}
    config["agents"].append({"name": "Helper", "description": "written by hand"})
    return config


class TestTheAgentName:
    def test_is_saved_as_typed(self, console: MagicMock, theme: Theme, tmp_path: Path) -> None:
        result = _run(console, theme, [*FIELDS, "Coder", ""], None, tmp_path / "config.yaml")

        assert result is not None
        assert result["entrypoint"] == "Coder"
        assert result["agents"][0]["name"] == "Coder"

    def test_enter_keeps_the_default_on_a_first_setup(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        result = _run(console, theme, [*FIELDS, "", ""], None, tmp_path / "config.yaml")

        assert result is not None
        assert result["entrypoint"] == cfg.AgentConfig().name

    def test_enter_keeps_the_current_one_on_a_rerun(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        result = _run(
            console, theme, [*FIELDS, "", ""], _existing("Coder"), tmp_path / "config.yaml"
        )

        assert result is not None
        assert result["entrypoint"] == "Coder"

    def test_a_rename_keeps_the_agents_hand_written_keys_and_adds_no_agent(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        result = _run(
            console, theme, [*FIELDS, "Coder", ""], _existing("Assistant"), tmp_path / "config.yaml"
        )

        assert result is not None
        assert result["entrypoint"] == "Coder"
        assert [a["name"] for a in result["agents"]] == ["Coder", "Helper"]
        assert result["agents"][0]["memory"] == {"type": "filesystem"}

    def test_another_agents_name_is_refused_and_asked_again(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        result = _run(
            console,
            theme,
            [*FIELDS, "Helper", "Coder", ""],
            _existing("Assistant"),
            tmp_path / "config.yaml",
        )

        assert result is not None
        assert [a["name"] for a in result["agents"]] == ["Coder", "Helper"]

    def test_a_name_longer_than_an_agent_accepts_is_refused_and_asked_again(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        too_long = "x" * (MAX_AGENT_NAME_LENGTH + 1)
        longest = "x" * MAX_AGENT_NAME_LENGTH

        result = _run(
            console, theme, [*FIELDS, too_long, longest, ""], None, tmp_path / "config.yaml"
        )

        assert result is not None
        assert result["entrypoint"] == longest

    def test_is_not_asked_by_a_provider_only_run(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        result = _run(
            console,
            theme,
            FIELDS,
            _existing("Coder"),
            tmp_path / "config.yaml",
            provider_only=True,
        )

        assert result is not None
        assert result["entrypoint"] == "Coder"
