"""Where the container keeps a session's working state on the host.

The background commands' logs, the browser's saved pages and each conversation's
folder (its recorded installs, its output). In the CLI that is the folder ``DOMYN_PROJECT_DIR`` names,
``~/.domyn`` or a project; without one it is ``.domyn`` in the workspace, as it is
for the local backend. A command in the container sees these paths at the same
place the host has them.
"""

import os
from pathlib import Path
from typing import Self

from domyn.agents.tools.builtin.backends.base import BackendUnavailableError
from domyn.agents.tools.builtin.session import (
    HOME_DIRNAME,
    SessionEnvironment,
    StateFolders,
    ignore_from_version_control,
    linked_below,
    state_dir_from_environment,
)


class StateFolder:
    def __init__(self, workspace_root: Path, explicit: Path | None) -> None:
        self._workspace_root = workspace_root
        self._explicit = explicit

    @classmethod
    def resolve(cls, workspace_root: Path, state_dir: str | os.PathLike[str] | None) -> Self:
        """The folder named, else the one ``DOMYN_PROJECT_DIR`` names, else the workspace's."""
        if state_dir is None:
            state_dir = state_dir_from_environment()
        explicit = Path(state_dir).expanduser().absolute() if state_dir is not None else None
        return cls(workspace_root, explicit)

    @property
    def as_seen_by_commands(self) -> str:
        """Relative when none was named: commands run in the workspace, so it is
        ``<workspace>/.domyn``."""
        return str(self._explicit) if self._explicit is not None else HOME_DIRNAME

    @property
    def root(self) -> Path:
        return self._explicit or self._workspace_root / HOME_DIRNAME

    @property
    def project_dir(self) -> Path | None:
        """The folder ``DOMYN_PROJECT_DIR`` names (``~/.domyn`` or a project), when the
        session has one. Without it the state is in the workspace and there is no config."""
        return self._explicit

    @property
    def config(self) -> Path | None:
        """The session's ``config.yaml``, which is in that folder. A container may read
        it and never write it: it names code the host runs and which backend to use."""
        return self._explicit / "config.yaml" if self._explicit is not None else None

    @property
    def processes(self) -> Path:
        return self.root / StateFolders.PROCESSES

    @property
    def browser(self) -> Path:
        return self.root / StateFolders.BROWSER

    def conversation_home(self, conversation: str) -> Path:
        environment = SessionEnvironment(self._workspace_root, state_dir=self._explicit)
        return environment.paths(conversation).home

    def prepare(self, conversation: str) -> None:
        """Make the places a container writes in, without following a link.

        A cloned repository can commit ``.domyn`` or any folder in it as a link, and
        a project folder given by path may be someone else's export; a bind mount
        follows a link. The state folder itself may be one (``~/.domyn`` kept with
        the dotfiles), what is below it may not.
        """
        start = self._explicit or self._workspace_root
        for target in (self.processes, self.browser, self.conversation_home(conversation)):
            if (link := linked_below(start, target)) is not None:
                raise BackendUnavailableError(
                    f"{link} is a link: the container will not mount through it"
                )
            target.mkdir(parents=True, exist_ok=True)
        if self._explicit is None:
            # In the workspace, where `git status` would otherwise list it.
            ignore_from_version_control(self.conversation_home(conversation).parent)
