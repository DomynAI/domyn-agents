"""``sandbox: auto``: a Docker sandbox when one can be used, this machine when not.

Unlike an explicit ``docker``, which refuses to start without a sandbox, this one
starts either way and *says* which it got: the reason is ``fallback``, and the
front-end prints it, so nobody believes they are sandboxed when they are not.
"""

import asyncio
import os
from collections.abc import Iterable
from pathlib import Path

from domyn.agents.logger import get_logger
from domyn.agents.tools.builtin.backends import (
    BackendUnavailableError,
    CommandBackend,
    CommandResult,
    LocalBackend,
)
from domyn.agents.tools.builtin.docker_sandbox import DockerBackend, DockerBackendOptions
from domyn.agents.tools.builtin.sandbox import SandboxBackend
from domyn.agents.tools.builtin.session import HOME_DIRNAME, state_dir_from_environment
from domyn.agents.tools.builtin.workspace import Workspace


class AutoBackend(SandboxBackend):
    """Docker when it can be used, this machine when it cannot, decided once.

    The decision is taken at the front-end's ``check``, and not again for this object:
    a Docker that goes down mid-session fails the command with a message instead of
    moving the conversation to the host half way (its files and installs would be in
    the other place). A rebuilt backend (config reload, ``/model``) decides afresh.
    """

    name = "auto"

    def __init__(
        self,
        workspace: Workspace,
        *,
        isolate_environment: bool = True,
        allowed_env: Iterable[str] = (),
        docker: DockerBackendOptions | None = None,
        state_dir: str | os.PathLike[str] | None = None,
        sandbox: SandboxBackend | None = None,
    ) -> None:
        """``sandbox`` is the one to try; by default a Docker one made from ``docker``."""
        # Resolved once, here, and handed to whichever backend is chosen: the two must
        # keep a conversation's state in the same place.
        if state_dir is None:
            state_dir = state_dir_from_environment()
        self._state_dir = Path(state_dir).expanduser().absolute() if state_dir is not None else None
        self._sandbox = sandbox or DockerBackend(
            workspace, options=docker, allowed_env=allowed_env, state_dir=self._state_dir
        )
        self._local = LocalBackend(
            workspace,
            isolate_environment=isolate_environment,
            allowed_env=allowed_env,
            state_dir=self._state_dir,
        )
        self._chosen: CommandBackend | None = None
        self._deciding = asyncio.Lock()
        self.fallback: BackendUnavailableError | None = None
        """Why Docker was not used, when it was not."""

    @property
    def state_dir(self) -> str:
        return str(self._state_dir) if self._state_dir is not None else HOME_DIRNAME

    async def decide(self) -> CommandBackend:
        async with self._deciding:
            if self._chosen is None:
                self._chosen = await self._choose()
            return self._chosen

    async def _choose(self) -> CommandBackend:
        try:
            await self._sandbox.check()
        except BackendUnavailableError as exc:
            self.fallback = exc
            get_logger().info(f"sandbox unavailable, commands run locally: {exc.problem}")
            self._notify(f"{exc.problem} Commands run on this machine, with no sandbox.")
            chosen: CommandBackend = self._local
        else:
            self._sandbox.on_notice = self._notify
            chosen = self._sandbox
        self.name = chosen.name
        self.is_isolated = chosen.is_isolated
        self.destroys_on_close = isinstance(chosen, SandboxBackend) and chosen.destroys_on_close
        return chosen

    def _notify(self, message: str) -> None:
        if self.on_notice is not None:
            self.on_notice(message)

    async def check(self) -> None:
        await self.decide()

    async def start(self) -> None:
        await (await self.decide()).start()

    async def prepare(self) -> None:
        chosen = await self.decide()
        if isinstance(chosen, SandboxBackend):
            await chosen.prepare()

    async def warm_up(self, conversation_id: str) -> None:
        chosen = await self.decide()
        if isinstance(chosen, SandboxBackend):
            await chosen.warm_up(conversation_id)

    async def run(
        self,
        command: str,
        *,
        timeout: float,
        conversation_id: str = "",
        stdin: str | None = None,
    ) -> CommandResult:
        chosen = await self.decide()
        return await chosen.run(
            command, timeout=timeout, conversation_id=conversation_id, stdin=stdin
        )

    async def end_conversation(self, conversation_id: str) -> None:
        if isinstance(self._chosen, SandboxBackend):
            await self._chosen.end_conversation(conversation_id)

    async def explain_failure(self, since: float) -> str:
        if isinstance(self._chosen, SandboxBackend):
            return await self._chosen.explain_failure(since)
        return ""

    async def close(self) -> None:
        if self._chosen is not None:
            await self._chosen.close()
