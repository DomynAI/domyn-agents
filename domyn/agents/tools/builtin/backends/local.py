"""Commands run on this machine, in the workspace directory."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import os
import signal
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from domyn.agents.logger import get_logger
from domyn.agents.tools.builtin.backends.base import CommandBackend, CommandResult
from domyn.agents.tools.builtin.git_bash import GitBashNotFoundError, find_git_bash
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

# CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW, spelled as numbers: the names
# exist in `subprocess` on Windows only, and mypy checks this file everywhere.
_WINDOWS_CREATION_FLAGS = 0x00000200 | 0x08000000


class ShellUnavailableError(RuntimeError):
    """The backend's shell cannot be started here (no Git Bash on Windows).

    Reported as the command's ``backend_error``, with what to install, rather
    than raised: the agent keeps working, and the model can tell the user.
    """


class LocalBackend(CommandBackend):
    """Runs on this machine, in the workspace directory.

    No isolation: the workspace bounds where the *file* tools may look and
    bounds nothing about what a command can reach. Said plainly, because the
    difference between "contained" and "well-behaved" is why containers exist.
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
        finally:
            self._finished(process)

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

    def _finished(self, process: asyncio.subprocess.Process) -> None:
        """Release what ``_spawn`` kept for ``process``, however its run ended."""


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
        state_dir: str | os.PathLike[str] | None = None,
        bash: Path | None = None,
    ) -> None:
        super().__init__(
            workspace,
            isolate_environment=isolate_environment,
            allowed_env=allowed_env,
            state_dir=state_dir,
        )
        self._bash = bash
        # The job each running command is in, by its pid: see _spawn.
        self._jobs: dict[int, _Job] = {}

    @property
    def state_dir(self) -> str:
        # With forward slashes (`C:/Users/me/.domyn`): every path built on it goes
        # into a bash line or into a message the model copies into one, and in bash
        # an unquoted backslash is an escape. Windows and Python read it as well.
        return Path(super().state_dir).as_posix()

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
        process = await asyncio.create_subprocess_exec(
            str(self.bash),
            "-c",
            'c=$DOMYN_COMMAND; unset DOMYN_COMMAND; eval "$c"',
            # Its own group, and no console window flashing up for every command.
            creationflags=_WINDOWS_CREATION_FLAGS,
            **{**streams, "env": env},
        )
        # In a job at once, before bash has had time to start anything: whatever it
        # starts is in the job too, which is what a timeout ends (see _terminate).
        if (job := _Job.around(process.pid)) is not None:
            self._jobs[process.pid] = job
        return process

    async def _terminate(self, process: asyncio.subprocess.Process) -> None:
        """End the command's job, then ``taskkill /T`` for anything outside it.

        The job is what reaches everything: ``taskkill /T`` follows the parent of
        each process, and Git Bash starts a command that bash forks for (any line
        with ``;``, ``&&`` or a loop, and all of them through ``eval``) from a
        process outside bash's tree. That command outlived the timeout, holding
        the pipes, and Python printed "Event loop is closed" on exit. The tree
        kill stays for a command whose job could not be made."""
        if (job := self._jobs.pop(process.pid, None)) is not None:
            job.terminate()
            job.close()
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
        # communicate, not wait: asyncio closes the transport once the process
        # has exited and both pipes have reached their end, and on Windows the
        # pipes get there after the exit. Left half way, the transport outlived
        # the event loop and Python printed "Event loop is closed" on exit.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(process.communicate(), timeout=_REAP_TIMEOUT_S)

    def _finished(self, process: asyncio.subprocess.Process) -> None:
        # Closing the handle ends nothing: the job is not killed on close, so what a
        # finished command left running on purpose (a server started with &) runs on.
        if (job := self._jobs.pop(process.pid, None)) is not None:
            job.close()


class _Job:
    """A Windows Job Object holding one command and every process it starts.

    Not killed when its handle is closed: only ``terminate`` ends what is in it.
    Everything here is Windows only; elsewhere ``around`` gives ``None``.
    """

    def __init__(self, handle: int) -> None:
        self._handle: int | None = handle

    @classmethod
    def around(cls, pid: int) -> _Job | None:
        """A new job with the process ``pid`` in it, or ``None`` when that fails
        (then the caller falls back to ending the tree by its parents)."""
        if sys.platform == "win32":
            kernel32 = _kernel32()
            job = kernel32.CreateJobObjectW(None, None)
            if not job:
                return None
            process = kernel32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid)
            assigned = bool(process) and bool(kernel32.AssignProcessToJobObject(job, process))
            if process:
                kernel32.CloseHandle(process)
            if not assigned:
                kernel32.CloseHandle(job)
                return None
            return cls(job)
        else:
            return None

    def terminate(self) -> None:
        """End every process in the job."""
        if sys.platform == "win32" and self._handle is not None:
            _kernel32().TerminateJobObject(self._handle, 1)

    def close(self) -> None:
        if sys.platform == "win32" and self._handle is not None:
            _kernel32().CloseHandle(self._handle)
        self._handle = None


# Access rights AssignProcessToJobObject needs on the process.
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100


@functools.cache
def _kernel32() -> Any:
    """kernel32 with the job functions typed, so handles are not cut to 32 bits."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined,unused-ignore]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.TerminateJobObject.restype = wintypes.BOOL
    kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    return kernel32


def this_machine(
    workspace: Workspace,
    *,
    isolate_environment: bool = True,
    allowed_env: Iterable[str] = (),
    state_dir: str | os.PathLike[str] | None = None,
) -> LocalBackend:
    """The backend for this machine's shell: ``/bin/sh`` on macOS and Linux, Git
    Bash on Windows, where ``cmd.exe`` is never used. What ``local`` resolves to,
    and what ``auto`` falls back to."""
    cls = GitBashBackend if os.name == "nt" else LocalBackend
    return cls(
        workspace,
        isolate_environment=isolate_environment,
        allowed_env=allowed_env,
        state_dir=state_dir,
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
