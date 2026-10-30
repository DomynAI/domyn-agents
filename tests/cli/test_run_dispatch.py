"""Tests for `resolve_agent_from_file` against the saved default config path.

The default config is a real ConfigurationV1 file, loaded through the same
`resolve_agent_from_file` a `.yaml` file handed to `domyn run` explicitly
would use — `allow_setup=True` is what makes the default-config call
different. These tests cover what that adds on top of a plain
`load_application_from_file` call: running the wizard when nothing usable
exists yet, and threading skill directories through as extra workspace roots
rather than freezing them into the YAML.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from domyn.agents.interfaces.cli import Theme
from domyn.agents.interfaces.cli import config as cfg
from domyn.agents.interfaces.cli import select as select_mod

from domyn.agents.cli.commands.run.command import resolve_agent_from_file  # isort: skip


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


@pytest.fixture
def theme() -> Theme:
    return Theme.build(allow_animation=False)


@pytest.mark.asyncio
class TestNoConfigYet:
    async def test_the_wizard_runs_and_the_agent_loads(self, theme: Theme) -> None:
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
            patch("domyn.agents.cli.commands.run.command.console", MagicMock()),
        ):
            resolved = await resolve_agent_from_file(cfg.config_path(), theme, allow_setup=True)

        assert resolved is not None
        agent, raw = resolved
        assert agent.name == "Assistant"
        assert raw["entrypoint"] == "Assistant"

    async def test_aborting_the_wizard_returns_none(self, theme: Theme) -> None:
        with (
            patch("builtins.input", side_effect=EOFError),
            patch("domyn.agents.cli.commands.run.command.console", MagicMock()),
        ):
            resolved = await resolve_agent_from_file(cfg.config_path(), theme, allow_setup=True)

        assert resolved is None


@pytest.mark.asyncio
class TestAnExistingConfig:
    async def test_a_usable_config_loads_without_prompting(self, theme: Theme) -> None:
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
            patch("domyn.agents.cli.commands.run.command.console", MagicMock()),
        ):
            await resolve_agent_from_file(cfg.config_path(), theme, allow_setup=True)

        with (
            patch("builtins.input", side_effect=AssertionError("should not prompt")),
            patch("domyn.agents.cli.commands.run.command.console", MagicMock()),
        ):
            resolved = await resolve_agent_from_file(cfg.config_path(), theme, allow_setup=True)

        assert resolved is not None
        agent, _ = resolved
        assert agent.name == "Assistant"

    async def test_an_incomplete_config_re_runs_the_wizard(self, theme: Theme) -> None:
        cfg.config_path().parent.mkdir(parents=True, exist_ok=True)
        cfg.config_path().write_text(
            yaml.safe_dump(
                {
                    "version": "1.0",
                    "entrypoint": "Assistant",
                    "llm_providers": [{"name": "default_provider", "type": "openai"}],
                    "agents": [
                        {
                            "name": "Assistant",
                            "description": "d",
                            "instruction": "i",
                            "llm_provider": "default_provider",
                            "planner_strategy": "tool_use_thought",
                        }
                    ],
                    "planner_strategies": [
                        {"name": "tool_use_thought", "type": "tool_use_thought"}
                    ],
                }
            ),
            encoding="utf-8",
        )

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
            patch("domyn.agents.cli.commands.run.command.console", MagicMock()),
        ):
            resolved = await resolve_agent_from_file(cfg.config_path(), theme, allow_setup=True)

        assert resolved is not None
        agent, raw = resolved
        assert raw["llm_providers"][0]["model_name"] == "gpt-4o-mini"
        assert agent.llm_provider.model_name == "gpt-4o-mini"


@pytest.mark.asyncio
class TestAnOldShapedFile:
    """Old CliConfig-shaped files (`provider:`/`agent:` keys, no `entrypoint:`)
    are no longer migrated — they're simply not a ConfigurationV1 file, so
    `is_configuration_v1_usable` reports them unusable and the wizard runs,
    the same as any other incomplete or missing config."""

    async def test_it_is_treated_as_unusable_and_the_wizard_runs(self, theme: Theme) -> None:
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
            patch("domyn.agents.cli.commands.run.command.console", MagicMock()),
        ):
            resolved = await resolve_agent_from_file(cfg.config_path(), theme, allow_setup=True)

        assert resolved is not None
        agent, raw = resolved
        assert agent.name == "Assistant"
        assert raw["entrypoint"] == "Assistant"


@pytest.mark.asyncio
class TestAnExplicitlyNamedFile:
    """`allow_setup=False` is the only thing that distinguishes an explicit
    `domyn run some.yaml` from the default-config path: an agent's `skills:`
    directories must apply exactly the same way, and an unusable file must be reported
    rather than turned into a wizard prompt."""

    def _write(self, path: Path, extra: dict) -> None:
        path.write_text(
            yaml.safe_dump(
                {
                    "version": "1.0",
                    "entrypoint": "Assistant",
                    "llm_providers": [
                        {
                            "name": "default_provider",
                            "type": "openai",
                            "model_name": "gpt-4o-mini",
                        }
                    ],
                    "planner_strategies": [
                        {"name": "tool_use_thought", "type": "tool_use_thought"}
                    ],
                    "agents": [
                        {
                            "name": "Assistant",
                            "description": "d",
                            "instruction": "i",
                            "llm_provider": "default_provider",
                            "planner_strategy": "tool_use_thought",
                            "tool_bundles": ["fs"],
                        }
                    ],
                    "tool_bundles": [{"name": "fs", "bundles": ["files"]}],
                    **extra,
                }
            ),
            encoding="utf-8",
        )

    async def test_an_agents_skill_dirs_reach_the_workspace(
        self, tmp_path: Path, theme: Theme
    ) -> None:
        config_file = tmp_path / "agent.yaml"
        workspace_dir = tmp_path / "workspace"
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        self._write(
            config_file,
            {
                "tool_bundles": [
                    {"name": "fs", "bundles": ["files"], "workspace": str(workspace_dir)}
                ],
                # Replaces the whole default `agents` list written by `_write`.
                "agents": [
                    {
                        "name": "Assistant",
                        "description": "d",
                        "instruction": "i",
                        "llm_provider": "default_provider",
                        "planner_strategy": "tool_use_thought",
                        "tool_bundles": ["fs"],
                        "skills": {
                            "source": {"type": "local_dir", "paths": [str(skills_dir)]},
                        },
                    }
                ],
            },
        )

        resolved = await resolve_agent_from_file(config_file, theme, allow_setup=False)

        assert resolved is not None
        agent, _ = resolved
        assert skills_dir.resolve() in agent.builtin_tools[0].workspace.extra_roots
        assert agent.skills is not None
        assert agent.skills.source.paths == [skills_dir]  # type: ignore[attr-defined]

    async def test_planner_strategy_compactor_reaches_the_planner(
        self, tmp_path: Path, theme: Theme
    ) -> None:
        config_file = tmp_path / "agent.yaml"
        self._write(
            config_file,
            {
                "planner_strategies": [
                    {
                        "name": "tool_use_thought",
                        "type": "tool_use_thought",
                        "compactor": {"strategy": "prune_only"},
                    }
                ],
            },
        )

        resolved = await resolve_agent_from_file(config_file, theme, allow_setup=False)

        assert resolved is not None
        agent, _ = resolved
        assert agent.planner.compactor.strategy_name == "prune_only"

    async def test_an_incomplete_file_is_reported_without_prompting(
        self, tmp_path: Path, theme: Theme
    ) -> None:
        """No wizard for a file the user named explicitly — it's a mistake to
        report, not something to interactively build."""
        config_file = tmp_path / "agent.yaml"
        config_file.write_text(
            yaml.safe_dump(
                {
                    "version": "1.0",
                    "entrypoint": "Assistant",
                    "llm_providers": [{"name": "default_provider", "type": "openai"}],
                    "agents": [
                        {
                            "name": "Assistant",
                            "description": "d",
                            "instruction": "i",
                            "llm_provider": "default_provider",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        with patch("builtins.input", side_effect=AssertionError("should not prompt")):
            resolved = await resolve_agent_from_file(config_file, theme, allow_setup=False)

        assert resolved is None

    async def test_a_missing_file_is_reported_without_prompting(
        self, tmp_path: Path, theme: Theme
    ) -> None:
        with patch("builtins.input", side_effect=AssertionError("should not prompt")):
            resolved = await resolve_agent_from_file(
                tmp_path / "missing.yaml", theme, allow_setup=False
            )

        assert resolved is None
