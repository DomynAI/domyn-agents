"""Where a command actually runs.

Two backends ship here: the local machine, and a Docker container
(``docker_backend.py``). The interface is meant for others too, e.g. a remote
executor, implemented outside the package.

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
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from domyn.agents.logger import get_logger
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


class BackendUnavailableError(RuntimeError):
    """The requested backend cannot be used.

    Raised, never swallowed into a fallback: running on the host while the user
    believes they are sandboxed turns a safety feature into a hazard.

    ``problem`` says what is wrong and ``fix`` what to do about it, when the backend
    knows: a front-end can lay them out, while ``str(error)`` has both in one line
    for everyone else (a tool error the model reads).
    """

    def __init__(self, problem: str, *, fix: str | None = None) -> None:
        super().__init__(f"{problem} {fix}" if fix else problem)
        self.problem = problem
        self.fix = fix


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

    @property
    def state_dir(self) -> str:
        """Where the tools keep their state on the machine commands run on: the
        logs of background commands (``<state_dir>/processes``) and the browser's
        files (``<state_dir>/browser``).

        Relative to the directory commands run in, ``.domyn``, unless a backend
        knows better: a remote one that answers nothing else keeps its state
        beside the workspace it mounts, as before. Put into shell lines quoted."""
        return HOME_DIRNAME

    @property
    def processes_dir(self) -> str:
        """The background commands' folders: ``<state_dir>/processes``."""
        return f"{self.state_dir}/{StateFolders.PROCESSES}"

    destroys_on_close: bool = False
    """Whether what a command started is stopped when its conversation ends or the
    session closes. A container backend answers ``True``: unlike the local one,
    a command left running in it does not outlive the session."""

    on_notice: Callable[[str], None] | None = None
    """Where a backend says what it is doing that takes a while (pulling an
    image, starting a container). The front-end sets it; ``None`` is silent."""

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

    # The three below are defaulted and empty for the same reason as the two
    # above, and because this class has consumers outside the repository: a
    # method added without a default would break each of them.
    async def check(self) -> None:  # noqa: B027
        """Raise :class:`BackendUnavailableError` if this backend cannot be used at
        all (Docker not installed, daemon not running). Fast, and downloads
        nothing: a front-end calls it at start-up to refuse to run, rather than
        discover it at the first command. Defaults to nothing."""

    async def prepare(self) -> None:  # noqa: B027
        """Do the slow, first-time part of ``start`` now, in the foreground, so the
        front-end can say what is happening. Defaults to nothing."""

    async def warm_up(self, conversation_id: str) -> None:  # noqa: B027
        """Have what ``conversation_id`` will need ready before its first command.
        Quiet, and must not raise: the first command meets the same failure."""

    async def end_conversation(self, conversation_id: str) -> None:  # noqa: B027
        """The conversation is over: release what it holds (a container). Defaults
        to nothing."""


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


def build_backend(
    name: str,
    workspace: Workspace,
    *,
    isolate_environment: bool = True,
    allowed_env: Iterable[str] = (),
    state_dir: str | os.PathLike[str] | None = None,
    docker: object | None = None,
) -> CommandBackend:
    """Resolve a backend by name.

    An unknown name raises, unlike every other unknown config value here, which
    degrades with a warning: handing back a local backend to someone who asked
    to be sandboxed gives the illusion of a boundary, which is worse than none.
    """
    if name == "local":
        return LocalBackend(
            workspace,
            isolate_environment=isolate_environment,
            allowed_env=allowed_env,
            state_dir=state_dir,
        )
    if name == "docker":
        # Imported here: it needs pydantic models and a module that imports this one.
        from domyn.agents.tools.builtin.docker_backend import DockerBackend, DockerOptions

        options = docker if isinstance(docker, DockerOptions) else DockerOptions()
        return DockerBackend(workspace, options=options, allowed_env=allowed_env)
    raise BackendUnavailableError(
        f"unknown command backend '{name}' (available: local, docker). "
        "Refusing to fall back to running on this machine."
    )
