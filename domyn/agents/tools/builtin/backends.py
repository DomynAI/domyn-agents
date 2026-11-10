"""Where a command actually runs.

One backend ships here (the local machine); the interface is meant for others,
e.g. a remote executor, implemented outside the package.

Each member is there for what a remote backend needs and a local one does not:

* **``start``/``close``.** A container has to be created and destroyed; without
  a lifecycle it would create itself lazily inside ``run`` and never be torn
  down.
* **One backend per session, not per command.** It has to survive between calls,
  or ``pip install x`` then ``python -c "import x"`` fails — which is most of
  what a shell is for.
* **A structured result, not a rendered string.** Rendering is the tool's job,
  and a remote backend's own failures ("the container is gone") must stay
  distinguishable from the command's exit code.
* **The command is a string, not an argv.** Both ends run it through a shell;
  see ``shell.py``.
* **The workspace is bound at construction.** A container mounts it once.

``is_isolated`` exists purely for later: an approval policy would key on it to
relax the gate, the way Hermes drops its dangerous-command checks behind a
container. Nothing reads it while every backend answers ``False``.

One seam did not hold: ``run`` cannot carry a backend's own routing information,
because ``FunctionTool.execute`` drops arguments a callable does not declare. An
external backend that needs routing data must currently widen ``run`` and ship its
own tool; widening it here is what would let the shipped tool serve both.

**Anything added here has a consumer outside this repository**: a defaulted
parameter is safe, a required one is a breaking change.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from domyn.agents.logger import get_logger
from domyn.agents.tools.builtin.git_bash import GitBashNotFoundError, find_git_bash
from domyn.agents.tools.builtin.session import (
    SessionEnvironment,
    environment_for,
    restricted_environment,
)

if TYPE_CHECKING:
    from domyn.agents.tools.builtin.workspace import Workspace


# How long to wait for a killed command to be reaped. Short: SIGKILL is not
# refusable, so anything longer means something is wrong that waiting will not
# fix.
_REAP_TIMEOUT_S = 5.0

# CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW, spelled as numbers: the names
# exist in `subprocess` on Windows only, and mypy checks this file everywhere.
_WINDOWS_CREATION_FLAGS = 0x00000200 | 0x08000000


class ShellUnavailableError(RuntimeError):
    """The backend's shell cannot be started here (no Git Bash on Windows).

    Reported as the command's ``backend_error``, with what to install, rather
    than raised: the agent keeps working, and the model can tell the user.
    """


class BackendUnavailableError(RuntimeError):
    """The requested backend cannot be used.

    Raised, never swallowed into a fallback: running on the host while the user
    believes they are sandboxed turns a safety feature into a hazard.
    """


@dataclass(frozen=True)
class CommandResult:
    """What a command did, before it is turned into text for the model."""

    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    backend_error: str | None = None
    """The backend itself failed, as opposed to the command failing. Kept
    separate so "the container is gone" is never reported as "your command
    exited non-zero"."""


class CommandBackend(ABC):
    """Somewhere commands can be run."""

    name: str = "backend"

    is_isolated: bool = False
    """Whether a destructive command is contained by something other than
    trust. See the module docstring: no policy consults this yet."""

    @abstractmethod
    async def run(
        self,
        command: str,
        *,
        timeout: float,
        conversation_id: str = "",
        stdin: str | None = None,
    ) -> CommandResult:
        """Run one command. Must not raise for an ordinary command failure.

        ``conversation_id`` selects the working environment. A container backend
        will use it to pick a container; the local one uses it to pick a
        virtualenv, so a package installed in one conversation does not appear
        in another — and, more importantly, not in the runtime running the CLI.

        ``stdin`` is the whole input the command will ever get, supplied up
        front. ``None`` means none — which is not the same as leaving it unset;
        see ``LocalBackend.run``.
        """

    # Not abstract, and empty on purpose: a backend with nothing to set up —
    # the local one — should not have to write two empty methods to say so.
    # Forcing them would make "nothing to do" and "forgot to implement" look
    # identical in every subclass.
    async def start(self) -> None:  # noqa: B027
        """Prepare the backend. Idempotent, and called before the first command."""

    async def close(self) -> None:  # noqa: B027
        """Release whatever ``start`` acquired. Must tolerate never having started."""


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
    ) -> None:
        self._workspace = workspace
        self._isolate = isolate_environment
        self._sessions = SessionEnvironment(workspace.root) if isolate_environment else None
        self._allowed_env = list(allowed_env)

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
        restricted = restricted_environment(self._allowed_env)
        env = restricted
        if self._sessions is not None and conversation_id:
            paths = self._sessions.prepare(conversation_id)
            if paths is not None:
                env = environment_for(paths, base=restricted)

        logger.info(f"run_command: executing locally in {self._workspace.root}")
        logger.debug(f"run_command: {command!r}")

        # DEVNULL, never inherited: unset, the child reads whatever this process
        # reads — in the CLI, the terminal the user is typing into. Measured: it
        # really does steal their keystrokes. EOF instead makes an unanswerable
        # prompt fail fast.
        streams: dict[str, Any] = {
            "cwd": str(self._workspace.root),
            "env": env,
            "stdin": asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
        }
        try:
            process = await self._spawn(command, streams)
        except ShellUnavailableError as exc:
            return CommandResult(exit_code=None, stdout="", stderr="", backend_error=str(exc))
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
            await self._terminate(process)
            return CommandResult(exit_code=None, stdout="", stderr="", timed_out=True)

        return CommandResult(
            exit_code=process.returncode,
            stdout=raw_out.decode("utf-8", "replace"),
            stderr=raw_err.decode("utf-8", "replace"),
        )

    async def _spawn(self, command: str, streams: dict[str, Any]) -> asyncio.subprocess.Process:
        """Start ``command`` in this backend's shell. The seam a subclass replaces."""
        if os.name == "nt":
            # The system shell there is cmd.exe, which reads the line differently
            # from the POSIX parser that approved it. GitBashBackend overrides this.
            raise ShellUnavailableError(
                "LocalBackend does not run commands on Windows: use "
                "build_backend('local', ...) or GitBashBackend, which run them in Git Bash."
            )
        return await asyncio.create_subprocess_shell(
            command,
            # Its own process group, so a timeout can kill the whole tree and
            # not just the shell. See _terminate_tree.
            start_new_session=True,
            **streams,
        )

    async def _terminate(self, process: asyncio.subprocess.Process) -> None:
        """End a timed-out command and everything it started."""
        await _terminate_tree(process)


class GitBashBackend(LocalBackend):
    """Runs on this Windows machine, in Git Bash, in the workspace directory.

    What ``local`` means on Windows. The line is POSIX shell — the approval
    gate split it with a POSIX parser, the background processes and the browser
    tools write it for bash — so it is run by bash, never by ``cmd.exe``, which
    reads ``&``, ``^``, ``%VAR%`` and quotes differently. See git_bash.py for
    where it is looked for.

    Looked for on the first command, not when built: an agent with no Git Bash
    still starts and still answers, and each command it tries says how to
    install it.
    """

    name = "git_bash"

    def __init__(
        self,
        workspace: Workspace,
        *,
        isolate_environment: bool = True,
        allowed_env: Iterable[str] = (),
        bash: Path | None = None,
    ) -> None:
        super().__init__(
            workspace, isolate_environment=isolate_environment, allowed_env=allowed_env
        )
        self._bash = bash

    @property
    def bash(self) -> Path:
        """The ``bash.exe`` commands run in, found on first use."""
        if self._bash is None:
            try:
                self._bash = find_git_bash()
            except GitBashNotFoundError as exc:
                raise ShellUnavailableError(str(exc)) from exc
        return self._bash

    async def _spawn(self, command: str, streams: dict[str, Any]) -> asyncio.subprocess.Process:
        # The line travels in the environment, not on the command line: a Git Bash
        # started from a Windows program re-reads its command line by rules of its
        # own, with single quotes as quotes and globbing, so every line with a '
        # in it (all of shlex.quote's) arrived rewritten. Unset before it runs.
        env = {**(streams.get("env") or os.environ), "DOMYN_COMMAND": command}
        return await asyncio.create_subprocess_exec(
            str(self.bash),
            "-c",
            'c=$DOMYN_COMMAND; unset DOMYN_COMMAND; eval "$c"',
            # Its own group, so a timeout can end the whole tree with taskkill
            # /T; and no console window flashing up for every command.
            creationflags=_WINDOWS_CREATION_FLAGS,
            **{**streams, "env": env},
        )

    async def _terminate(self, process: asyncio.subprocess.Process) -> None:
        """``taskkill /T``: Windows has no process group to signal, but ending a
        process with /T takes every process it started, which is the same tree."""
        with contextlib.suppress(OSError, TimeoutError):
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/F",
                "/T",
                "/PID",
                str(process.pid),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                creationflags=_WINDOWS_CREATION_FLAGS,
            )
            await asyncio.wait_for(killer.wait(), timeout=_REAP_TIMEOUT_S)
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=_REAP_TIMEOUT_S)


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


def build_backend(
    name: str,
    workspace: Workspace,
    *,
    isolate_environment: bool = True,
    allowed_env: Iterable[str] = (),
) -> CommandBackend:
    """Resolve a backend by name.

    An unknown name raises, unlike every other unknown config value here, which
    degrades with a warning: handing back a local backend to someone who asked
    to be sandboxed gives the illusion of a boundary, which is worse than none.

    ``local`` is this machine's shell: ``/bin/sh`` on macOS and Linux, Git Bash
    on Windows, where ``cmd.exe`` is never used. ``git_bash`` names the second
    explicitly, and exists on Windows only: elsewhere there is no Git Bash, and
    quietly running ``/bin/sh`` instead would be running something else than
    the configuration says.
    """
    if name == "git_bash" and os.name != "nt":
        raise BackendUnavailableError(
            "the 'git_bash' command backend runs on Windows only; use 'local' here. "
            "Refusing to fall back to another shell."
        )
    if name == "git_bash" or (name == "local" and os.name == "nt"):
        return GitBashBackend(
            workspace, isolate_environment=isolate_environment, allowed_env=allowed_env
        )
    if name == "local":
        return LocalBackend(
            workspace, isolate_environment=isolate_environment, allowed_env=allowed_env
        )
    raise BackendUnavailableError(
        f"unknown command backend '{name}' (available: local, git_bash on Windows). "
        "Refusing to fall back to running on this machine."
    )
