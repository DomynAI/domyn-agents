"""Talking to the ``docker`` command line: one call, or one command in a container."""

import asyncio
import contextlib
import os
import platform
import signal
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from pathlib import Path
from typing import NamedTuple, TypeVar

from domyn.agents.logger import get_logger
from domyn.agents.tools.builtin.backends.base import BackendUnavailableError, CommandResult

# How long each kind of docker call may take before it is given up on. A call that
# times out is reported as a failure like any other, and tried again where that is safe.
DAEMON_CHECK_SECONDS = 10
"""``docker info``: is the daemon there. Short, because a front-end asks at start-up."""
INSPECT_SECONDS = 20
"""Asking the daemon about something: inspect, ps, logs, network ls."""
ACTION_SECONDS = 30
"""Removing, creating or connecting something, and the probe in a new container."""
SLOW_ACTION_SECONDS = 60
"""Starting the proxy, removing several containers, listing what is installed."""
CONTAINER_START_SECONDS = 120
"""``docker run`` of a command container: it creates the container and starts it."""
PULL_SECONDS = 1800
"""Pulling an image, which can be a large download."""
BUILD_SECONDS = 900
"""Building the sandbox image."""

# How much longer than a command's own limit the host waits. The limit is enforced
# inside the container by `timeout`; this only bounds a `docker exec` that has
# stopped answering.
EXEC_GRACE_SECONDS = 15.0
RETRY_DELAY_SECONDS = 1.0
_TERMINATE_WAIT_SECONDS = 5

# Exit codes `docker` and `timeout` use for "I could not run it" and "it ran too long".
_DOCKER_ERROR = 125
_TIMED_OUT = (124, 137)

# Everything a command needs on its PATH inside the container: the user prefixes
# `pip install --user` and `npm install -g` write to.
_PATH_PREFIX = 'export PATH="$HOME/.local/bin:$HOME/.npm-global/bin:$PATH"\n'

_T = TypeVar("_T")


class DockerOutput(NamedTuple):
    """What a ``docker`` command printed and returned."""

    code: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.code == 0

    def error(self, limit: int = 300) -> str:
        """The end of stderr, where a failing docker command says why."""
        return self.stderr.strip()[-limit:]


class TransientDockerError(BackendUnavailableError):
    """Docker itself failed to do something that normally works: worth one more try.

    A busy daemon answers a start with an error now and then (``No such container``
    for a container it had just created, a proxy slower than the wait). It says
    nothing about the image or the configuration, unlike a refusal that comes back
    twice."""


def container_is_gone(text: str) -> bool:
    lowered = text.lower()
    return "no such container" in lowered or "is not running" in lowered


async def retry_once(operation: Callable[[], Awaitable[_T]], *, describe: str) -> _T:
    """Run ``operation`` again, once, when it fails with a ``TransientDockerError``."""
    try:
        return await operation()
    except TransientDockerError as exc:
        get_logger().warning(f"sandbox: {describe} failed, trying again: {exc}")
    await asyncio.sleep(RETRY_DELAY_SECONDS)
    return await operation()


class DockerCli:
    """``docker`` as a subprocess. Calling it takes the arguments, ``docker`` left out."""

    def __init__(self, binary: str = "docker") -> None:
        self.binary = binary

    async def __call__(
        self, *args: str, timeout: float = 60.0, stdin: str | None = None
    ) -> DockerOutput:
        try:
            process = await self._spawn(args, piped_stdin=stdin is not None)
        except FileNotFoundError as exc:
            raise BackendUnavailableError(
                f"Docker was not found ('{self.binary}' is not on your PATH).",
                fix="Install Docker from https://docs.docker.com/get-docker/.",
            ) from exc
        try:
            out, err = await asyncio.wait_for(
                process.communicate(input=_encode(stdin)), timeout=timeout
            )
        except TimeoutError:
            await _terminate(process, whole_group=False)
            verb = args[0] if args else ""
            return DockerOutput(124, "", f"docker {verb} timed out after {timeout:.0f}s")
        return DockerOutput(_exit_code(process), _decode(out), _decode(err))

    async def check_daemon(self) -> None:
        """Raise ``BackendUnavailableError`` unless the daemon answers."""
        info = await self("info", "--format", "{{.ServerVersion}}", timeout=DAEMON_CHECK_SECONDS)
        if info.ok:
            return
        # Short on purpose: the client's own line carries the socket path and reads
        # as noise. It stays in the log.
        get_logger().debug(f"docker info detail: {info.error(400)}")
        if "permission denied" in info.stderr.lower():
            reason = "permission denied connecting to the Docker daemon: is your user allowed to use Docker?"
        else:
            reason = "failed to connect to the docker daemon: is the daemon running?"
        raise BackendUnavailableError(
            f"Docker is installed but not running ({reason}).", fix=self._how_to_start()
        )

    async def exec_command(
        self,
        container: str,
        command: str,
        *,
        workdir: Path,
        timeout: float,
        stdin: str | None = None,
        forward_env: Iterable[str] = (),
    ) -> CommandResult:
        """Run a shell command in a running container.

        The limit is enforced inside the container, by ``timeout``: killing this
        ``docker exec`` client would not stop what it started there.
        """
        limit = max(1, int(timeout))
        argv = _exec_arguments(container, command, workdir, limit, forward_env)
        started = time.monotonic()
        try:
            process = await self._spawn(argv, piped_stdin=stdin is not None, new_session=True)
        except OSError as exc:
            return _backend_error(f"could not run docker: {type(exc).__name__}: {exc}")
        try:
            out, err = await asyncio.wait_for(
                process.communicate(input=_encode(stdin)), timeout=limit + EXEC_GRACE_SECONDS
            )
        except TimeoutError:
            # `timeout` inside the container should have ended it, so if the client
            # is still here something is wrong with the container.
            await _terminate(process, whole_group=True)
            return CommandResult(exit_code=None, stdout="", stderr="", timed_out=True)
        return _exec_result(
            _exit_code(process),
            _decode(out),
            _decode(err),
            elapsed=time.monotonic() - started,
            limit=limit,
        )

    async def _spawn(
        self, args: Sequence[str], *, piped_stdin: bool, new_session: bool = False
    ) -> asyncio.subprocess.Process:
        return await asyncio.create_subprocess_exec(
            self.binary,
            *args,
            stdin=asyncio.subprocess.PIPE if piped_stdin else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=new_session,
        )

    def _how_to_start(self) -> str:
        name = Path(self.binary).name
        if name != "docker":
            return f"Start {name} (for podman on macOS or Windows: `podman machine start`)."
        system = platform.system()
        if system == "Darwin":
            return "Start Docker Desktop (`open -a Docker`) and wait until it says it is running."
        if system == "Linux":
            return "Start the Docker service (`sudo systemctl start docker`)."
        return "Start Docker Desktop and wait until it says it is running."


def _exec_arguments(
    container: str, command: str, workdir: Path, limit: int, forward_env: Iterable[str]
) -> list[str]:
    forwarded = [flag for name in forward_env if name in os.environ for flag in ("-e", name)]
    return [
        "exec", "-i", "-w", str(workdir), *forwarded,
        container, "timeout", "-s", "KILL", str(limit), "bash", "-c", _PATH_PREFIX + command,
    ]  # fmt: skip


def _exec_result(
    code: int, stdout: str, stderr: str, *, elapsed: float, limit: int
) -> CommandResult:
    if code in _TIMED_OUT and elapsed >= limit - 1:
        return CommandResult(exit_code=None, stdout="", stderr="", timed_out=True)
    docker_failed = code == _DOCKER_ERROR or (
        code == 1 and container_is_gone(stderr) and not stdout
    )
    if docker_failed:
        return _backend_error(stderr.strip()[-300:] or "docker failed")
    return CommandResult(exit_code=code, stdout=stdout, stderr=stderr)


def _backend_error(message: str) -> CommandResult:
    return CommandResult(exit_code=None, stdout="", stderr="", backend_error=message)


def _encode(text: str | None) -> bytes | None:
    return text.encode() if text is not None else None


def _decode(data: bytes) -> str:
    return data.decode("utf-8", "replace")


def _exit_code(process: asyncio.subprocess.Process) -> int:
    return process.returncode if process.returncode is not None else 1


async def _terminate(process: asyncio.subprocess.Process, *, whole_group: bool) -> None:
    """Kill a client that has stopped answering, and reap it. ``whole_group`` only for
    one started in a session of its own: the group of anything else is ours."""
    if whole_group:
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    with contextlib.suppress(ProcessLookupError):
        process.kill()
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(process.wait(), timeout=_TERMINATE_WAIT_SECONDS)
