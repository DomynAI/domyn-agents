"""`domyn run --resume <id>` with no `--project` finds the project holding the id."""

from pathlib import Path

import pytest

from domyn.agents.conversation_manager import SqliteConversationManager
from domyn.agents.interfaces.cli import config, conversations
from domyn.agents.interfaces.cli.repl import APPLICATION_NAME, USER_ID


async def _store(folder: Path, conversation_id: str) -> None:
    folder.mkdir(parents=True)
    (folder / config.CONFIG_FILENAME).write_text("version: '1.0'\n")
    manager = SqliteConversationManager(path=str(folder / config.PrivateFiles.CONVERSATIONS))
    await manager.create_conversation(APPLICATION_NAME, USER_ID, conversation_id)


@pytest.mark.asyncio
async def test_names_the_project_that_has_the_conversation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "GLOBAL_DIR", tmp_path)
    projects = tmp_path / config.PROJECTS_DIRNAME
    await _store(projects / "alpha", "conv_a")
    await _store(projects / "beta", "conv_b")

    assert await conversations.project_holding(APPLICATION_NAME, USER_ID, "conv_b") == "beta"
    assert await conversations.project_holding(APPLICATION_NAME, USER_ID, "conv_none") is None


@pytest.mark.asyncio
async def test_a_project_without_history_is_skipped_and_not_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "GLOBAL_DIR", tmp_path)
    folder = tmp_path / config.PROJECTS_DIRNAME / "fresh"
    folder.mkdir(parents=True)
    (folder / config.CONFIG_FILENAME).write_text("version: '1.0'\n")

    assert await conversations.project_holding(APPLICATION_NAME, USER_ID, "conv_a") is None
    assert not (folder / config.PrivateFiles.CONVERSATIONS).exists()
