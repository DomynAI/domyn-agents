"""Finding Git Bash, the shell commands run in on Windows.

Written against fake install folders so they run on every platform: what is
checked is which file is chosen, not that it starts.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from domyn.agents.tools.builtin.git_bash import (
    ENV_VAR,
    GitBashNotFoundError,
    find_git_bash,
)


def _file(path: Path, *, executable: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _git_install(root: Path) -> Path:
    """A Git for Windows layout: cmd/git.exe and bin/bash.exe."""
    _file(root / "cmd" / "git.exe", executable=True)
    _file(root / "cmd" / "git", executable=True)  # what `which` finds off Windows
    return _file(root / "bin" / "bash.exe", executable=True)


class TestTheConfiguredPath:
    def test_wins_over_everything_else(self, tmp_path: Path) -> None:
        mine = _file(tmp_path / "elsewhere" / "bash.exe")
        _git_install(tmp_path / "ProgramFiles" / "Git")
        env = {ENV_VAR: str(mine), "ProgramFiles": str(tmp_path / "ProgramFiles"), "PATH": ""}
        assert find_git_bash(env) == mine

    def test_a_stale_one_is_reported_not_replaced(self, tmp_path: Path) -> None:
        _git_install(tmp_path / "ProgramFiles" / "Git")
        env = {
            ENV_VAR: str(tmp_path / "gone" / "bash.exe"),
            "ProgramFiles": str(tmp_path / "ProgramFiles"),
            "PATH": "",
        }
        with pytest.raises(GitBashNotFoundError, match=ENV_VAR):
            find_git_bash(env)

    def test_the_wsl_launcher_is_refused(self, tmp_path: Path) -> None:
        wsl = _file(tmp_path / "Windows" / "System32" / "bash.exe")
        with pytest.raises(GitBashNotFoundError):
            find_git_bash({ENV_VAR: str(wsl), "PATH": ""})


class TestFoundBesideGit:
    def test_from_the_cmd_folder(self, tmp_path: Path) -> None:
        bash = _git_install(tmp_path / "Tools" / "Git")
        env = {"PATH": str(tmp_path / "Tools" / "Git" / "cmd")}
        assert find_git_bash(env).resolve() == bash.resolve()

    def test_from_mingw64_bin(self, tmp_path: Path) -> None:
        root = tmp_path / "Git"
        bash = _file(root / "bin" / "bash.exe")
        _file(root / "mingw64" / "bin" / "git", executable=True)
        _file(root / "mingw64" / "bin" / "git.exe", executable=True)
        env = {"PATH": str(root / "mingw64" / "bin")}
        assert find_git_bash(env).resolve() == bash.resolve()


class TestTheStandardFolders:
    def test_program_files(self, tmp_path: Path) -> None:
        bash = _git_install(tmp_path / "ProgramFiles" / "Git")
        env = {"ProgramFiles": str(tmp_path / "ProgramFiles"), "PATH": ""}
        assert find_git_bash(env) == bash

    def test_a_per_user_install(self, tmp_path: Path) -> None:
        bash = _git_install(tmp_path / "Local" / "Programs" / "Git")
        env = {"LOCALAPPDATA": str(tmp_path / "Local"), "PATH": ""}
        assert find_git_bash(env) == bash


class TestNothingFound:
    def test_says_how_to_install_it(self, tmp_path: Path) -> None:
        with pytest.raises(GitBashNotFoundError, match="Git for Windows"):
            find_git_bash({"PATH": str(tmp_path)})

    def test_a_bash_on_path_alone_is_not_used(self, tmp_path: Path) -> None:
        # System32\bash.exe is WSL's launcher: never taken from PATH.
        system32 = tmp_path / "Windows" / "System32"
        _file(system32 / "bash.exe", executable=True)
        _file(system32 / "bash", executable=True)
        with pytest.raises(GitBashNotFoundError):
            find_git_bash({"PATH": str(system32)})


@pytest.mark.skipif(os.name != "nt", reason="Windows only: the runner's own Git Bash")
def test_the_runner_has_one() -> None:
    assert find_git_bash().name.lower() == "bash.exe"
