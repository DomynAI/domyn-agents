"""Commands and files on a server reached with ``ssh``; the agent stays on this machine.

What runs where:

* **Commands** run on the server, in ``ssh.workspace``, through the server user's login
  shell and ``bash``. The time limit is enforced there (``timeout -s KILL``): killing the
  ``ssh`` client here does not stop what it started.
* **The file tools** run here, on the server's files (``RemoteDisk``): the same tools,
  each operation a short line of standard commands. Nothing is installed on the server.
* **Skills** are read here, and copied to the server for the commands that run them
  (``SkillCopies``).
* **The browser** runs here by default (``ssh.browser: local``), on this machine's
  backend: the server often has no display, no Chrome and no route to what the user
  browses.
* **The agent, the model and the approvals** stay here, as with every backend.

What the server needs: an ``sshd`` that accepts a key or an agent with no question asked
(``BatchMode``), a login shell of the ``sh`` family, and GNU ``bash``, ``timeout``,
``find``, ``realpath``, ``grep``, ``xargs`` and ``tar`` — any Linux distribution has them.
``setsid``, ``ps`` and ``pkill`` for commands that run in the background. Missing ones
are named when the backend starts.

No isolation: a command runs as the server's user, with what that user can reach. That is
the user's choice; the approval gate is the same as on this machine.
"""

import asyncio
import hashlib
import shlex
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from domyn.agents.logger import get_logger
from domyn.agents.tools.builtin.backends.base import (
    BackendUnavailableError,
    CommandBackend,
    CommandResult,
)
from domyn.agents.tools.builtin.backends.local import LocalBackend, this_machine
from domyn.agents.tools.builtin.backends.ssh.connection import (
    SSH_ERROR,
    SshConnection,
    SshUnavailableError,
)
from domyn.agents.tools.builtin.backends.ssh.disk import RemoteDisk
from domyn.agents.tools.builtin.backends.ssh.options import SshBackendOptions
from domyn.agents.tools.builtin.backends.ssh.skills import SkillCopies
from domyn.agents.tools.builtin.disk import Disk, DiskError
from domyn.agents.tools.builtin.workspace import Workspace

PROBE_SECONDS = 60
"""The first call, which opens the connection: a handshake, maybe a jump host."""
GRACE_SECONDS = 15.0
"""How much longer than a command's limit this side waits for ``ssh`` to come back."""
_TIMED_OUT = (124, 137)

_MARKER = b"domyn-probe\0"
_NEEDED = ("bash", "timeout")
_BACKGROUND = ("setsid", "nohup", "ps", "pkill")


class SshBackend(CommandBackend):
    """Runs the tools on a server reached with the system ``ssh``."""

    name = "ssh"
    is_isolated = False

    def __init__(
        self,
        workspace: Workspace,
        *,
        options: SshBackendOptions | None = None,
        allowed_env: Iterable[str] = (),
    ) -> None:
        """``workspace`` is this machine's: its extra roots are the skill folders, read
        here and copied to the server. The server's workspace is ``options.workspace``."""
        options = options or SshBackendOptions()
        if not options.workspace:
            raise SshUnavailableError(
                "the ssh backend needs the folder to work in on the server: set "
                "`ssh: workspace:` in the tools bundle, as an absolute path."
            )
        self._options = options
        self._connection = SshConnection(options)
        self._skills = SkillCopies((*workspace.extra_roots, *workspace.covered_roots))
        self._skill_roots = self._skills.roots
        self._lock = asyncio.Lock()
        self._root: str | None = None
        self._home = ""
        self._missing_for_files: list[str] = []
        self._local = this_machine(workspace, allowed_env=allowed_env)
        self.on_notice: Callable[[str], None] | None = None
        """Told what the backend has to say while it starts (a missing command)."""

    @property
    def host(self) -> str:
        return self._options.host

    @property
    def state_dir(self) -> str:
        """The state folder on the server: ``ssh.state_dir``, or ``~/.cache/domyn/<id>``
        there, which is known once the backend has started (before, the base class's
        answer, which nothing reads for real: every tool starts the backend first)."""
        if self._options.state_dir is not None:
            return self._options.state_dir
        if self._root is None:
            return super().state_dir
        key = hashlib.sha256(f"{self.host}:{self.remote_root}".encode()).hexdigest()[:12]
        return f"{self._home}/.cache/domyn/{key}"

    @property
    def browser_backend(self) -> CommandBackend:
        """Where the ``browser`` bundle runs: this machine unless ``ssh.browser`` says
        the server."""
        return self if self._options.browser == "remote" else self._local

    @property
    def local_backend(self) -> LocalBackend:
        return self._local

    @property
    def remote_root(self) -> str:
        """The server's workspace, resolved. Known once the backend has started."""
        if self._root is None:
            raise RuntimeError("the ssh backend is used before it started")
        return self._root

    def _state_path(self) -> str:
        state = self.state_dir
        return state if state.startswith("/") else f"{self.remote_root}/{state}"

    async def start(self) -> None:
        """Open the connection and look at the server, once. Raises
        ``BackendUnavailableError`` saying what is wrong and what to do."""
        async with self._lock:
            if self._root is not None:
                return
            await asyncio.to_thread(self._connection.prepare)
            root, home, missing = await self._probe()
            needed = [name for name in missing if name in _NEEDED]
            if needed:
                raise SshUnavailableError(
                    f"the server {self.host} lacks {', '.join(needed)}, which every command "
                    "runs through.",
                    fix="Install them there (GNU bash and coreutils).",
                )
            self._root, self._home = root, home
            self._missing_for_files = [name for name in missing if name.startswith("files:")]
            later = [name for name in missing if name in _BACKGROUND or name == "tar"]
            if later or self._missing_for_files:
                listed = ", ".join(
                    [*later, *(n.removeprefix("files:") for n in self._missing_for_files)]
                )
                self._say(
                    f"the server {self.host} lacks {listed}: background commands, the file "
                    "tools or the skills' scripts may not work until they are installed."
                )

    async def _probe(self) -> tuple[str, str, list[str]]:
        script = (
            "printf 'domyn-probe\\0'; "
            f"cd -- {shlex.quote(self._options.workspace)} 2>/dev/null && pwd -P | tr -d '\\n'; "
            "printf '\\0%s\\0' \"$HOME\"; "
            f"for c in {' '.join((*_NEEDED, *_BACKGROUND, 'tar'))}; do "
            'command -v "$c" >/dev/null 2>&1 || printf \'%s\\0\' "$c"; done; '
            "find / -maxdepth 0 -printf '' >/dev/null 2>&1 || printf 'files:GNU find\\0'; "
            "realpath -m -z / >/dev/null 2>&1 || printf 'files:GNU realpath\\0'; "
            "command -v xargs >/dev/null 2>&1 || printf 'files:xargs\\0'; true"
        )
        out = await self._connection.run(
            f"sh -c {shlex.quote(script)}", stdin=None, timeout=PROBE_SECONDS
        )
        if out is None:
            raise SshUnavailableError(
                f"the server {self.host} did not answer within {PROBE_SECONDS}s.",
                fix="Check that `ssh " + self.host + " true` works from this terminal.",
            )
        if out.code == SSH_ERROR or not out.stdout.startswith(_MARKER):
            if out.code != SSH_ERROR and out.stdout:
                raise SshUnavailableError(
                    f"the login shell of {self.host} prints text before any command, which "
                    "would be read as the commands' output.",
                    fix="Make it print nothing in a non-interactive session (look at "
                    "~/.bashrc, ~/.profile or ~/.zshenv on the server).",
                )
            raise SshUnavailableError(
                f"could not connect to {self.host}: {_tail(out.stderr) or 'ssh failed'}",
                fix=f"Check that `ssh {self.host} true` works from this terminal with no "
                "question asked: an unknown host key or a password stop it here, since "
                "nobody is there to answer.",
            )
        root, home, *missing = out.stdout[len(_MARKER) :].decode("utf-8", "replace").split("\0")
        if not root:
            raise SshUnavailableError(
                f"{self._options.workspace} is not a folder on {self.host}.",
                fix="Create it there, or set `ssh: workspace:` to one that exists.",
            )
        return root, home, [name for name in missing if name]

    def _say(self, message: str) -> None:
        get_logger().warning(f"ssh: {message}")
        if self.on_notice is not None:
            self.on_notice(message)

    async def run(
        self,
        command: str,
        *,
        timeout: float,
        conversation_id: str = "",
        stdin: str | None = None,
    ) -> CommandResult:
        try:
            await self.start()
        except BackendUnavailableError as exc:
            return CommandResult(exit_code=None, stdout="", stderr="", backend_error=str(exc))
        base = f"{self._state_path()}/skills"
        await self._skills.sync(self._connection, base)
        command = self._skills.rewrite(command, base)
        limit = max(1, int(timeout))
        remote = (
            f"cd {shlex.quote(self.remote_root)} && "
            f"exec timeout -s KILL {limit} bash -c {shlex.quote(command)}"
        )
        get_logger().info(f"run_command: executing on {self.host}")
        get_logger().debug(f"run_command: {command!r}")
        started = time.monotonic()
        out = await self._connection.run(
            remote,
            stdin=stdin.encode() if stdin is not None else None,
            timeout=limit + GRACE_SECONDS,
        )
        if out is None:
            return CommandResult(exit_code=None, stdout="", stderr="", timed_out=True)
        overran = time.monotonic() - started >= limit - 1
        if out.code in _TIMED_OUT and overran:
            return CommandResult(exit_code=None, stdout="", stderr="", timed_out=True)
        if out.code == SSH_ERROR:
            if not await self._connection.alive():
                return CommandResult(
                    exit_code=None,
                    stdout="",
                    stderr="",
                    backend_error=f"lost the connection to {self.host}: {_tail(out.stderr)}",
                )
            if overran:
                # `timeout -s KILL` kills its whole group, itself included, and `ssh`
                # reports a command killed by a signal as 255.
                return CommandResult(exit_code=None, stdout="", stderr="", timed_out=True)
        return CommandResult(
            exit_code=out.code,
            stdout=out.stdout.decode("utf-8", "replace"),
            stderr=out.stderr.decode("utf-8", "replace"),
        )

    def file_disk(self) -> tuple[Workspace, Disk] | None:
        if self._missing_for_files:
            listed = ", ".join(name.removeprefix("files:") for name in self._missing_for_files)
            raise DiskError(
                f"the file tools need {listed} on the server {self.host}, which it lacks. "
                "Install them there with run_command (GNU findutils and coreutils), or "
                "read and change files with run_command instead."
            )
        space = Workspace(root=Path(self.remote_root), extra_roots=tuple(self._skill_roots))
        return space, RemoteDisk(self._connection, home=self._home)

    async def close(self) -> None:
        await self._local.close()
        await self._connection.close()


def _tail(data: bytes) -> str:
    return data.decode("utf-8", "replace").strip()[-300:]
