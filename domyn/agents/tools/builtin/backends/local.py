"""Commands run on this machine, in the workspace directory."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from domyn.agents.logger import get_logger
from domyn.agents.tools.builtin.backends.base import CommandBackend, CommandResult
from domyn.agents.tools.builtin.session import (
    HOME_DIRNAME,
    SessionEnvironment,
    StateFolders,
    environment_for,
    linked_below,
    restricted_environment,
    state_dir_from_environment,
)

if TYPE_CHECKING:
    from domyn.agents.tools.builtin.workspace import Workspace


# How long to wait for a killed command to be reaped. Short: SIGKILL is not
# refusable, so anything longer means something is wrong that waiting will not
# fix.
_REAP_TIMEOUT_S = 5.0


class LocalBackend(CommandBackend):
    """Runs on this machine, in the workspace directory.

    No isolation: the workspace bounds where the *file* tools may look and
    bounds nothing about what a command can reach. Said plainly, because the
    difference between "contained" and "well-behaved" is why sandboxes exist.
    """

    name = "local"
    is_isolated = False

    def __init__(
        self,
        workspace: Workspace,
        *,
        isolate_environment: bool = True,
        allowed_env: Iterable[str] = (),
        state_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        """``state_dir`` is the folder the conversations' environments, the
        background commands' logs and the browser's files go in. When not given, the
        folder ``DOMYN_PROJECT_DIR`` names (`state_dir_from_environment`), so nothing
        is written into the workspace. Without either, ``.domyn`` in the workspace.
        Read when the backend is built, so a session keeps the folder it started with."""
        self._workspace = workspace
        self._isolate = isolate_environment
        if state_dir is None:
            state_dir = state_dir_from_environment()
        self._state_dir = Path(state_dir).expanduser().absolute() if state_dir is not None else None
        self._sessions = (
            SessionEnvironment(workspace.root, state_dir=self._state_dir)
            if isolate_environment
            else None
        )
        self._allowed_env = list(allowed_env)

    @property
    def state_dir(self) -> str:
        # Relative when none was named: commands run in the workspace root, so it
        # is `<workspace>/.domyn`, written as it always was.
        return str(self._state_dir) if self._state_dir is not None else HOME_DIRNAME

    def _linked_state_folder(self) -> Path | None:
        """A link on the way to ``processes/`` or ``browser/``, checked from the
        named state folder (which may be a link itself), or from the workspace, where
        ``.domyn`` is a folder a cloned repository can commit as a link."""
        start = self._state_dir if self._state_dir is not None else self._workspace.root
        base = self._state_dir if self._state_dir is not None else start / HOME_DIRNAME
        for name in (StateFolders.PROCESSES, StateFolders.BROWSER):
            if (link := linked_below(start, base / name)) is not None:
                return link
        return None

    @property
    def sessions(self) -> SessionEnvironment | None:
        """The per-conversation environments, so a front-end can show where
        they are. ``None`` when isolation is off."""
        return self._sessions

    async def run(
        self,
        command: str,
        *,
        timeout: float,
        conversation_id: str = "",
        stdin: str | None = None,
    ) -> CommandResult:
        logger = get_logger()
        if (link := self._linked_state_folder()) is not None:
            # Refused rather than run: the background commands' wrapper and the
            # browser write below these folders with `mkdir -p` and `>`, which follow
            # a link wherever it leads.
            return CommandResult(
                exit_code=None,
                stdout="",
                stderr="",
                backend_error=(
                    f"{link} is a link. Commands keep their logs and the browser its "
                    "files there, and links are not followed: remove it and try again."
                ),
            )
        restricted = restricted_environment(self._allowed_env)
        env = restricted
        if self._sessions is not None and conversation_id:
            paths = self._sessions.prepare(conversation_id)
            if paths is not None:
                env = environment_for(paths, base=restricted)

        logger.info(f"run_command: executing locally in {self._workspace.root}")
        logger.debug(f"run_command: {command!r}")

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=str(self._workspace.root),
                env=env,
                # DEVNULL, never inherited: unset, the child reads whatever
                # this process reads — in the CLI, the terminal the user is
                # typing into. Measured: it really does steal their keystrokes.
                # EOF instead makes an unanswerable prompt fail fast.
                stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # Its own process group, so a timeout can kill the whole tree
                # and not just the shell. See _terminate_tree.
                start_new_session=True,
            )
        except OSError as exc:
            logger.warning(f"run_command: could not start: {type(exc).__name__}")
            logger.debug(f"run_command start detail: {exc}")
            return CommandResult(
                exit_code=None,
                stdout="",
                stderr="",
                backend_error=f"could not start the command: {type(exc).__name__}: {exc}",
            )

        try:
            raw_out, raw_err = await asyncio.wait_for(
                process.communicate(input=stdin.encode() if stdin is not None else None),
                timeout=timeout,
            )
        except TimeoutError:
            await _terminate_tree(process)
            return CommandResult(exit_code=None, stdout="", stderr="", timed_out=True)

        return CommandResult(
            exit_code=process.returncode,
            stdout=raw_out.decode("utf-8", "replace"),
            stderr=raw_err.decode("utf-8", "replace"),
        )


async def _terminate_tree(process: asyncio.subprocess.Process) -> None:
    """Kill a timed-out command and everything it started.

    Killing the shell alone fails worse than a leak: ``sh -c "sleep 30 & wait"``
    forks, so SIGKILL to the shell leaves a child holding our pipes and
    ``process.wait()`` blocks until it finishes anyway. Measured — a one-second
    timeout that returned after forty-seven. It depends on the shell (a simple
    command is ``exec``-ed and has no child), which is why it passed locally and
    failed in CI.

    Killing the group takes every descendant and releases the pipes. The final
    wait is bounded too: this is not the place to assume ``wait`` cannot block.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)

    # The group kill covers the shell too, but a platform without process groups
    # falls through to here having done nothing.
    with contextlib.suppress(ProcessLookupError):
        process.kill()

    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(process.wait(), timeout=_REAP_TIMEOUT_S)
