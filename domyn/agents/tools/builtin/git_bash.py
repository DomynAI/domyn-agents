r"""The shell commands run in on Windows: Git Bash, from Git for Windows.

Everything that runs a command — ``run_command``, the background processes, the
browser tools — hands the backend a line of POSIX shell, and the approval gate
splits that line into the programs it runs with a POSIX parser
(``shell.py``). On Windows the line must therefore be run by a POSIX shell too:
``cmd.exe`` reads ``&``, ``^``, ``%VAR%`` and quotes differently, so a grant the
parser judged to cover ``git status`` could cover something else entirely when
``cmd.exe`` reads it. Git for Windows ships ``bash.exe``, and nearly every
developer machine has it; without it no command runs and the error says how to
install it.

Found, in order:

1. ``DOMYN_GIT_BASH_PATH``, a path to ``bash.exe``, for an install somewhere
   unusual. Honoured only when the file exists, so a stale value is reported
   rather than silently replaced.
2. Beside the ``git`` on ``PATH``: ``<Git>\cmd\git.exe`` (or ``<Git>\bin``,
   ``<Git>\mingw64\bin``) has ``<Git>\bin\bash.exe``.
3. The standard install folders: ``%ProgramFiles%\Git``,
   ``%ProgramFiles(x86)%\Git``, ``%LOCALAPPDATA%\Programs\Git`` (a per-user
   install), ``%ProgramW6432%\Git``.

Never ``shutil.which("bash")``: ``C:\Windows\System32\bash.exe`` is WSL's
launcher, which runs the line in a Linux distribution, with another file system
and another environment — or fails with a message about installing one.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ENV_VAR = "DOMYN_GIT_BASH_PATH"

INSTALL_HINT = (
    "domyn runs commands with Git Bash on Windows. Install Git for Windows "
    "(https://git-scm.com/downloads/win, or `winget install --id Git.Git -e`), "
    f"or set {ENV_VAR} to the full path of its bash.exe."
)


class GitBashNotFoundError(RuntimeError):
    """No usable Git Bash on this machine."""


def _beside_git(git: str) -> list[Path]:
    """Where ``bash.exe`` sits relative to a ``git.exe``."""
    folder = Path(git).resolve().parent
    # <Git>\cmd\git.exe, <Git>\bin\git.exe, <Git>\mingw64\bin\git.exe
    roots = [folder.parent, folder.parent.parent]
    return [root / "bin" / "bash.exe" for root in roots]


def _install_folders(environ: os._Environ[str] | dict[str, str]) -> list[Path]:
    folders = []
    for name in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
        base = environ.get(name)
        if base:
            folders.append(Path(base) / "Git")
    local = environ.get("LOCALAPPDATA")
    if local:
        folders.append(Path(local) / "Programs" / "Git")
    return [folder / "bin" / "bash.exe" for folder in folders]


def _is_wsl_launcher(path: Path) -> bool:
    r"""``System32\bash.exe`` (and its ``SysWOW64``/``Sysnative`` views) is WSL."""
    parts = {part.lower() for part in path.parts}
    return bool(parts & {"system32", "syswow64", "sysnative"})


def find_git_bash(environ: os._Environ[str] | dict[str, str] | None = None) -> Path:
    """The ``bash.exe`` of Git for Windows, or ``GitBashNotFoundError``."""
    env = os.environ if environ is None else environ

    configured = env.get(ENV_VAR)
    if configured:
        path = Path(configured)
        if path.is_file() and not _is_wsl_launcher(path):
            return path
        raise GitBashNotFoundError(
            f"{ENV_VAR} is set to {configured}, which is not a Git Bash bash.exe. {INSTALL_HINT}"
        )

    candidates: list[Path] = []
    git = shutil.which("git", path=env.get("PATH"))
    if git:
        candidates.extend(_beside_git(git))
    candidates.extend(_install_folders(env))

    for candidate in candidates:
        if candidate.is_file() and not _is_wsl_launcher(candidate):
            return candidate
    raise GitBashNotFoundError(f"Git Bash was not found. {INSTALL_HINT}")
