"""State kept outside the folder the CLI is started in: venvs, process logs and
browser files go to the state folder (``~/.domyn`` or a project), never ``./.domyn``."""

from pathlib import Path

import pytest

from domyn.agents.tools.builtin import (
    BrowserOptions,
    BrowserSessions,
    LocalBackend,
    Workspace,
    processes,
)
from domyn.agents.tools.builtin.session import SessionEnvironment, set_state_dir


@pytest.fixture(autouse=True)
def _no_default() -> None:
    set_state_dir(None)
    yield
    set_state_dir(None)


def test_a_conversation_lives_in_the_state_folder(tmp_path: Path) -> None:
    work, state = tmp_path / "work", tmp_path / "state"
    work.mkdir()
    paths = SessionEnvironment(work, state).paths("abc")

    assert paths.home == state / "conversations" / "abc"
    assert not (work / ".domyn").exists()


def test_without_one_it_stays_under_the_workspace(tmp_path: Path) -> None:
    assert SessionEnvironment(tmp_path).paths("abc").home == tmp_path / ".domyn/conversations/abc"


def test_the_default_state_folder_reaches_the_backend(tmp_path: Path) -> None:
    set_state_dir(tmp_path / "home")
    backend = LocalBackend(Workspace.at(tmp_path))

    assert backend.state_dir == str((tmp_path / "home").resolve())
    assert BrowserSessions(backend, BrowserOptions()).browser_dir == f"{backend.state_dir}/browser"
    home = processes.processes_home(backend.state_dir)
    assert home == f"{backend.state_dir}/processes"
    assert processes.new_record("ls", "c", home).dir.startswith(home)


def test_an_unplain_path_falls_back_to_the_relative_folder() -> None:
    assert processes.processes_home("/Users/a b/.domyn") == processes.HOME
    assert processes.processes_home('/x"; rm -rf /; "') == processes.HOME


def test_a_stored_home_that_is_not_plain_is_refused() -> None:
    with pytest.raises(ValueError):
        processes.ProcessRecord(handle="deadbeef", command="ls", home='/x"; rm -rf ~; "')
