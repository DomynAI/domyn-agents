"""``backend: auto``: a Docker container when one can be used, this machine when not.

Unlike an explicit ``docker``, which refuses to start without a container, this one
starts either way and *says* which it got: the reason is ``fallback``, and the
front-end prints it, so nobody believes they are isolated when they are not.
"""

import asyncio
import os
from collections.abc import Iterable
from pathlib import Path

from domyn.agents.logger import get_logger
from domyn.agents.tools.builtin.backends.base import (
    BackendUnavailableError,
    CommandResult,
    ToolHost,
)
from domyn.agents.tools.builtin.backends.docker import DockerBackend, DockerBackendOptions
from domyn.agents.tools.builtin.backends.docker.cli import DockerCli
from domyn.agents.tools.builtin.backends.local import LocalBackend
from domyn.agents.tools.builtin.session import HOME_DIRNAME, state_dir_from_environment
from domyn.agents.tools.builtin.workspace import Workspace


class AutoBackend(DockerBackend):
    """A Docker backend that runs on this machine instead when Docker cannot be used.

    The decision is taken at the front-end's ``check`` (or ``prepare``, when the image
    will not pull), and not again after the first command: a Docker that goes down
    mid-session fails the command with a message instead of moving the conversation to
    the host half way (its files and installs would be in the other place). A rebuilt
    backend (config reload, ``/model``) decides afresh. After a fallback every call goes
    to a ``LocalBackend``, and ``name``, ``is_isolated`` and ``destroys_on_close`` say so.
    """

    name = "auto"
    is_isolated = False  # until it has decided
    destroys_on_close = False

    def __init__(
        self,
        workspace: Workspace,
        *,
        isolate_environment: bool = True,
        allowed_env: Iterable[str] = (),
        docker: DockerBackendOptions | None = None,
        state_dir: str | os.PathLike[str] | None = None,
        cli: DockerCli | None = None,
    ) -> None:
        # Resolved once, here, and handed to both: whichever is used keeps a
        # conversation's state in the same place.
        if state_dir is None:
            state_dir = state_dir_from_environment()
        self._state_dir = Path(state_dir).expanduser().absolute() if state_dir is not None else None
        allowed = list(allowed_env)
        super().__init__(
            workspace, options=docker, allowed_env=allowed, state_dir=self._state_dir, cli=cli
        )
        self._local = LocalBackend(
            workspace,
            isolate_environment=isolate_environment,
            allowed_env=allowed,
            state_dir=self._state_dir,
        )
        self._deciding = asyncio.Lock()
        self._decided = False
        self._on_this_machine = False
        self.fallback: BackendUnavailableError | None = None
        """Why Docker was not used, when it was not."""

    @property
    def state_dir(self) -> str:
        return str(self._state_dir) if self._state_dir is not None else HOME_DIRNAME

    async def decide(self) -> None:
        async with self._deciding:
            if self._decided:
                return
            try:
                await super().check()
            except BackendUnavailableError as exc:
                self._fall_back(exc)
                return
            self._decided = True
            self.name, self.is_isolated, self.destroys_on_close = "docker", True, True

    def _fall_back(self, problem: BackendUnavailableError) -> None:
        self.fallback = problem
        get_logger().info(f"docker unavailable, commands run locally: {problem.problem}")
        if self.on_notice is not None:
            self.on_notice(f"{problem.problem} Commands run on this machine, not in a container.")
        self._decided = self._on_this_machine = True
        self.name, self.is_isolated, self.destroys_on_close = "local", False, False

    async def check(self) -> None:
        await self.decide()

    async def start(self) -> None:
        await self.decide()
        await (self._local if self._on_this_machine else super()).start()

    async def prepare(self) -> None:
        """Get Docker ready; when it cannot be (the image will not pull), run on this
        machine instead, like when Docker is not there at all. Nothing has run yet: the
        front-end prepares before the first command."""
        await self.decide()
        if self._on_this_machine:
            return
        try:
            await super().prepare()
        except BackendUnavailableError as exc:
            self._fall_back(exc)

    async def warm_up(self, conversation_id: str) -> None:
        await self.decide()
        if not self._on_this_machine:
            await super().warm_up(conversation_id)

    async def run(
        self,
        command: str,
        *,
        timeout: float,
        conversation_id: str = "",
        stdin: str | None = None,
    ) -> CommandResult:
        await self.decide()
        backend = self._local if self._on_this_machine else super()
        return await backend.run(
            command, timeout=timeout, conversation_id=conversation_id, stdin=stdin
        )

    @property
    def tool_host(self) -> ToolHost | None:
        return None if self._on_this_machine else super().tool_host

    async def end_conversation(self, conversation_id: str) -> None:
        if self._decided and not self._on_this_machine:
            await super().end_conversation(conversation_id)

    async def explain_failure(self, since: float) -> str:
        if self._decided and not self._on_this_machine:
            return await super().explain_failure(since)
        return ""

    async def close(self) -> None:
        if not self._decided:
            return
        await (self._local if self._on_this_machine else super()).close()
