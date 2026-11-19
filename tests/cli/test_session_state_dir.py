"""`domyn run` keeps the tools' state in ``home_dir()``, beside ``conversations.db``:
the project's folder, else ``~/.domyn``. Every command sets ``DOMYN_PROJECT_DIR`` to it
(`use_project`), and the built-in tools read it when they are built."""

import asyncio
from pathlib import Path

import pytest
from rich.console import Console

from domyn.agents.cli.project import select_project
from domyn.agents.interfaces.cli import config as cli_config
from domyn.agents.interfaces.cli.theme import Theme
from domyn.agents.loader import load_application_from_file

CONFIG = (
    "version: '1.0'\nentrypoint: A\n"
    "llm_providers: [{name: p, type: vllm, model_name: m, url: 'http://x'}]\n"
    "resources: [{id: work, type: local_dir, paths: [WORKSPACE]}]\n"
    "tool_bundles: [{name: t, bundles: [shell]}]\n"
    "agents: [{name: A, description: d, instruction: i, llm_provider: p,"
    " tool_bundles: [{use: t, workspace: work}]}]\n"
)


def _tools_state_dir(config: Path) -> str:
    agent = asyncio.run(load_application_from_file(str(config)))
    (built,) = agent.builtin_tools
    assert built.backend is not None
    return built.backend.state_dir


@pytest.fixture
def config(tmp_path: Path) -> Path:
    workspace = tmp_path / "work"
    workspace.mkdir()
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG.replace("WORKSPACE", str(workspace)), encoding="utf-8")
    return path


def test_without_a_project_it_is_the_global_folder(config: Path) -> None:
    select_project(None, Console(), Theme.build(allow_animation=False))

    assert _tools_state_dir(config) == str(cli_config.GLOBAL_DIR)


def test_in_a_project_it_is_the_projects_folder(config: Path, tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / cli_config.CONFIG_FILENAME).write_text("{}", encoding="utf-8")
    cli_config.use_project(str(project))

    assert _tools_state_dir(config) == str(project.resolve())
