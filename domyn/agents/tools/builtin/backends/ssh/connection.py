"""Talking to the server through the system ``ssh``.

One connection is opened and every call goes through it (``ControlMaster``): a call
then costs a round trip, not a handshake. Nothing of this machine is handed to the
server: no agent forwarding, no port or X11 forwarding, and ``ssh`` sends only the
environment its own configuration names. Host keys are checked as ``ssh`` always checks
them, and ``BatchMode`` turns any question (an unknown host key, a password) into a
failure, since nobody is there to answer it.
"""

import asyncio
import contextlib
import os
import shutil
import signal
import stat
import subprocess
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from domyn.agents.tools.builtin.backends.base import BackendUnavailableError
from domyn.agents.tools.builtin.backends.ssh.options import SshBackendOptions


class SshUnavailableError(BackendUnavailableError):
    """The server cannot be used: not reached, refused, or missing what the tools need."""


MAX_SESSIONS = 8
"""Calls open on the connection at once. ``sshd`` refuses a connection's sessions past
its ``MaxSessions``, 10 by default; the rest wait their turn."""

SSH_ERROR = 255
"""The exit code ``ssh`` uses for its own failures, and the one a command may use too."""

_TERMINATE_WAIT_SECONDS = 5


@dataclass(frozen=True)
class Output:
    code: int
    stdout: bytes
    stderr: bytes


class SshConnection:
    """The ``ssh`` command line for one server, and a limit on the calls open at once."""

    def __init__(self, options: SshBackendOptions) -> None:
        if not options.host or options.host.startswith("-"):
            raise SshUnavailableError(
                "the ssh backend needs a server: set `ssh: host:` in the tools bundle.",
            )
        self.options = options
        self._sessions = threading.BoundedSemaphore(MAX_SESSIONS)

    @property
    def binary(self) -> str:
        found = shutil.which(self.options.binary)
        if found is None:
            raise SshUnavailableError(
                f"`{self.options.binary}` was not found, and the ssh backend runs it.",
                fix="Install OpenSSH's client, or set `ssh: binary:` to where it is.",
            )
        return found

    @property
    def control_dir(self) -> Path:
        """Where the connection's socket is. Short, because a socket's path has a
        budget of about 100 bytes, which a macOS temporary folder already uses half of."""
        return Path("/tmp") / f"domyn-ssh-{os.getuid()}"

    def prepare(self) -> None:
        """Make the socket's folder, ours alone, or refuse one that is not."""
        folder = self.control_dir
        with contextlib.suppress(FileExistsError):
            folder.mkdir(mode=0o700)
        info = folder.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise SshUnavailableError(
                f"{folder} is not a folder only you can use, and the ssh backend keeps the "
                "socket of its connection there.",
                fix=f"Remove it (`rm -rf {folder}`) and start again.",
            )

    def argv(self, remote: str | None = None, *, control: str | None = None) -> list[str]:
        """``ssh`` with every option this backend sets, then the server and ``remote``,
        the command the server's login shell runs."""
        options = self.options
        args = [
            self.binary,
            "-T",
            "-S", str(self.control_dir / "%C"),
            "-o", "ControlMaster=auto",
            "-o", "ControlPersist=300",
            "-o", "BatchMode=yes",
            "-o", f"ConnectTimeout={options.connect_timeout_s}",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3",
            "-o", "ForwardAgent=no",
            "-o", "ForwardX11=no",
            "-o", "ClearAllForwardings=yes",
            "-o", "PermitLocalCommand=no",
        ]  # fmt: skip
        if options.config_file:
            args += ["-F", str(Path(options.config_file).expanduser())]
        if options.user:
            args += ["-l", options.user]
        if options.port:
            args += ["-p", str(options.port)]
        if options.identity_file:
            args += [
                "-i",
                str(Path(options.identity_file).expanduser()),
                "-o",
                "IdentitiesOnly=yes",
            ]
        if control is not None:
            args += ["-O", control]
        args += ["--", options.host]
        if remote is not None:
            args.append(remote)
        return args

    # --- from the event loop ----------------------------------------------------------

    async def run(self, remote: str, *, stdin: bytes | None, timeout: float) -> Output | None:
        """Run ``remote`` on the server. ``None`` when ``ssh`` was still there after
        ``timeout``: then it is killed, with everything it started here."""
        await asyncio.to_thread(self._sessions.acquire)
        try:
            process = await asyncio.create_subprocess_exec(
                *self.argv(remote),
                stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            try:
                out, err = await asyncio.wait_for(process.communicate(stdin), timeout=timeout)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(process.wait(), timeout=_TERMINATE_WAIT_SECONDS)
                return None
            return Output(process.returncode if process.returncode is not None else 1, out, err)
        finally:
            self._sessions.release()

    async def alive(self) -> bool:
        """Whether the shared connection is up: what tells an exit code of 255 that
        ``ssh`` failed from one the command returned."""
        try:
            process = await asyncio.create_subprocess_exec(
                *self.argv(control="check"),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            return await asyncio.wait_for(process.wait(), timeout=10) == 0
        except (OSError, TimeoutError):
            return False

    async def close(self) -> None:
        """End the shared connection, if there is one."""
        with contextlib.suppress(OSError, TimeoutError, BackendUnavailableError):
            process = await asyncio.create_subprocess_exec(
                *self.argv(control="exit"),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(process.wait(), timeout=10)

    # --- from a thread (the file tools' bodies) ---------------------------------------

    def run_sync(self, remote: str, *, stdin: bytes | None = None, timeout: float) -> Output:
        """``run``, blocking, for the file tools' bodies, which run in a thread. A call
        still there after ``timeout`` raises ``TimeoutError``."""
        with self._sessions:
            completed = subprocess.run(
                self.argv(remote),
                input=stdin,
                stdin=None if stdin is not None else subprocess.DEVNULL,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        return Output(completed.returncode, completed.stdout, completed.stderr)

    @contextlib.contextmanager
    def stream(
        self, remote: str, *, stdin: bytes | None = None
    ) -> Iterator[subprocess.Popen[bytes]]:
        """``remote`` running, its output read as it comes. Killed when the block ends,
        so a reader that has seen enough does not wait for the rest.

        ``stdin`` is written whole and closed before the output is read: the command
        must read all of it before it prints much, or both ends wait on each other."""
        with self._sessions:
            process = subprocess.Popen(
                self.argv(remote),
                stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                if stdin is not None:
                    assert process.stdin is not None
                    process.stdin.write(stdin)
                    process.stdin.close()
                yield process
            finally:
                if process.poll() is None:
                    process.kill()
                for pipe in (process.stdin, process.stdout, process.stderr):
                    if pipe is not None:
                        pipe.close()
                process.wait()
