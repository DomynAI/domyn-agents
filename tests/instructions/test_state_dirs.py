"""`STATE_DIRS` names every folder domyn itself writes directly under ``.domyn``.

The autodiscovery walk over ``./.domyn`` skips them: what is in there is the agent's
own scratch (a conversation's files, a browser page, a process log) or skill folders,
and an ``AGENTS.md`` among them must never reach the system prompt. A built-in that
starts writing somewhere new under ``.domyn`` fails here until it is listed.
"""

import inspect
from pathlib import Path, PurePath

from domyn.agents.instructions.discovery import STATE_DIRS
from domyn.agents.interfaces.cli.config import LOCAL_SKILLS_DIRS
from domyn.agents.tools.builtin import BrowserSessions, processes
from domyn.agents.tools.builtin.session import HOME_DIRNAME, SessionPaths


def test_conversation_folders(tmp_path: Path) -> None:
    home = SessionPaths(tmp_path, "c").home.relative_to(tmp_path)

    assert home.parts[0] == HOME_DIRNAME
    assert home.parts[1] in STATE_DIRS


def test_the_browser_folder() -> None:
    default = inspect.signature(BrowserSessions.__init__).parameters["browser_dir"].default

    assert PurePath(default).parts[0] == HOME_DIRNAME
    assert PurePath(default).parts[1] in STATE_DIRS


def test_the_process_folder() -> None:
    assert PurePath(processes.HOME).parts[0] == HOME_DIRNAME
    assert PurePath(processes.HOME).parts[1] in STATE_DIRS


def test_the_skills_folders() -> None:
    """``.domyn/skills`` and ``.agents/skills``: skill folders may hold an ``AGENTS.md`` of
    their own, written for the skill and not for the agent."""
    assert {path.parts[-1] for path in LOCAL_SKILLS_DIRS} <= STATE_DIRS
