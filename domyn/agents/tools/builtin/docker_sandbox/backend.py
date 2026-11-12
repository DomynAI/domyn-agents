"""Run shell commands in a Docker container; everything else stays on this machine.

The agent, its file tools and its approvals run in the terminal process as ever.
Only what ``run_command`` starts is moved into a container that is locked in: the
workspace is all it can write, almost nothing of the host is mounted, and the only
network it has is the proxy's. This is the opposite of running the whole agent in
a container, and the split the Domyn platform uses.

**The transport is ``docker exec``.** There is no relay and no daemon of ours in
the container: one long-lived container per conversation, one ``docker exec`` per
command. The approval gate runs on the host before the command string is sent, so
``/auto``, deny rules and the decision model all work unchanged: the container is
only where the command runs afterwards.

**Life cycle.** A container is made on a conversation's first command, and removed
when the conversation ends (``end_conversation``: ``/clear``, ``/resume``, exit)
and at ``close``. Containers are named after this process, so a backend rebuilt by
a config reload *adopts* the one already running instead of leaking it; one left
by a process that has died is removed by the next start. What the agent installed
is recorded in the conversation's folder and installed again in a new container
for it (``--resume``).

**The claim this supports** is that a command cannot touch the host outside the
workspace, its credentials or its processes. It does not say a command cannot send
data out through a host the network allows, and the workspace itself is writable by
design. See ``mounts``, ``network``, ``images``, ``installs`` and ``containers`` for
the parts.
"""

import asyncio
import contextlib
import json
import os
import shlex
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

from domyn.agents.logger import get_logger
from domyn.agents.tools.builtin.backends import BackendUnavailableError, CommandResult
from domyn.agents.tools.builtin.docker_sandbox.cli import (
    ACTION_SECONDS,
    INSPECT_SECONDS,
    SLOW_ACTION_SECONDS,
    DockerCli,
    container_is_gone,
    retry_once,
)
from domyn.agents.tools.builtin.docker_sandbox.containers import ContainerLauncher, DockerContainer
from domyn.agents.tools.builtin.docker_sandbox.images import ensure_image
from domyn.agents.tools.builtin.docker_sandbox.installs import InstallLog
from domyn.agents.tools.builtin.docker_sandbox.mounts import (
    build_mounts,
    refuse_unsafe_workspace,
    skill_roots,
)
from domyn.agents.tools.builtin.docker_sandbox.network import EgressNetwork
from domyn.agents.tools.builtin.docker_sandbox.options import DockerBackendOptions
from domyn.agents.tools.builtin.docker_sandbox.scope import (
    FINGERPRINT_LABEL,
    SandboxScope,
    digest,
    remove_orphans,
)
from domyn.agents.tools.builtin.docker_sandbox.state import StateFolder
from domyn.agents.tools.builtin.sandbox import SandboxBackend
from domyn.agents.tools.builtin.session import safe_id
from domyn.agents.tools.builtin.workspace import Workspace

DEFAULT_CONVERSATION = "default"
_BACKGROUND_COMMAND_RUNNING = 3  # the exit code `_has_background_command`'s script ends with


class _ExistingContainer(NamedTuple):
    """What ``docker inspect`` says about a container with a conversation's name."""

    running: bool
    fingerprint: str


class DockerBackend(SandboxBackend):
    """Commands run in a container per conversation. See the module docstring."""

    name = "docker"
    is_isolated = True
    destroys_on_close = True

    def __init__(
        self,
        workspace: Workspace,
        *,
        options: DockerBackendOptions | None = None,
        allowed_env: Iterable[str] = (),
        state_dir: str | os.PathLike[str] | None = None,
        cli: DockerCli | None = None,
    ) -> None:
        """``state_dir`` is where the background commands' logs and each conversation's
        folder live, as for ``LocalBackend``: the folder ``DOMYN_PROJECT_DIR`` names
        when none is given, else ``.domyn`` in the workspace. Read when the backend
        is built. ``cli`` replaces the ``docker`` command, for tests."""
        self._workspace = workspace
        self._options = options or DockerBackendOptions()
        self._allowed_env = [name for name in allowed_env if name]
        self._cli = cli or DockerCli(self._options.docker_binary)
        self._scope = SandboxScope.of(workspace.root)
        self._state = StateFolder.resolve(workspace.root, state_dir)
        self._network = (
            EgressNetwork(self._cli, self._scope, self._options.allowed_hosts(), self._say)
            if self._options.has_network
            else None
        )
        self._launcher = ContainerLauncher(
            self._cli, self._options, self._scope, self._network, workspace.root, self._say
        )
        self._installs = InstallLog(
            self._cli,
            self._state,
            workspace.root,
            self._say,
            can_download=self._options.has_network,
        )
        self._prepare_lock = asyncio.Lock()
        self._conversation_locks: dict[str, asyncio.Lock] = {}
        self._containers: dict[str, DockerContainer] = {}
        self._deferred_for: dict[str, str] = {}
        self._prepared_image: str | None = None
        self._announce = False

    @property
    def state_dir(self) -> str:
        return self._state.as_seen_by_commands

    def container_name(self, conversation: str) -> str:
        return self._scope.container_name(conversation)

    # --- the interface ---------------------------------------------------

    async def check(self) -> None:
        """Docker is installed and its daemon answers. The fast part of ``prepare``."""
        refuse_unsafe_workspace(self._workspace.root)
        await self._cli.check_daemon()

    async def start(self) -> None:
        await self.prepare()

    async def prepare(self) -> None:
        """Check Docker and make sure the image exists. The slow, first-time part.

        Called in the foreground at session start, so a pull or a build happens
        under a message and a spinner rather than inside someone's first command.
        """
        self._announce = True
        await self._prepare()

    async def warm_up(self, conversation_id: str) -> None:
        """Have the conversation's container ready, quietly. Never raises: a failure
        here is met again, with the same message, by the first command."""
        self._announce = False
        try:
            await self._prepare()
            await self._ensure_container(conversation_id or DEFAULT_CONVERSATION)
        except BackendUnavailableError as exc:
            get_logger().warning(f"sandbox warm-up failed: {exc}")
        except Exception as exc:
            get_logger().warning(f"sandbox warm-up failed: {type(exc).__name__}")
            get_logger().debug(f"sandbox warm-up detail: {exc}")

    async def run(
        self,
        command: str,
        *,
        timeout: float,
        conversation_id: str = "",
        stdin: str | None = None,
    ) -> CommandResult:
        self._announce = True
        conversation = conversation_id or DEFAULT_CONVERSATION
        return await self._run_in_container(
            conversation, command, timeout, stdin, retry_if_lost=True
        )

    async def end_conversation(self, conversation_id: str) -> None:
        conversation = conversation_id or DEFAULT_CONVERSATION
        # Under the conversation's lock: a container still being made for it (the
        # warm-up) finishes first, or it would be made after this and never removed.
        async with self._lock_for(conversation):
            self._containers.pop(conversation, None)
            self._deferred_for.pop(conversation, None)
            with contextlib.suppress(Exception):
                await self._cli(
                    "rm", "-f", self.container_name(conversation), timeout=ACTION_SECONDS
                )

    async def close(self) -> None:
        """Remove every container, the proxy and the network of this process.

        Found by label, not by what this object remembers: a backend rebuilt by a
        config reload adopted containers the old object made, and a warm-up in
        flight is finished first (the conversation locks) so it cannot leave one
        behind.
        """
        locks = list(self._conversation_locks.values())
        for lock in locks:
            await lock.acquire()
        try:
            self._containers.clear()
            with contextlib.suppress(Exception):
                await self._remove_scope()
        finally:
            for lock in locks:
                lock.release()

    async def explain_failure(self, since: float) -> str:
        """Say which host the proxy refused since ``since``, if one was."""
        if self._network is None:
            return ""
        blocked = await self._network.blocked_since(since)
        if not blocked:
            return ""
        return f"[sandbox] blocked connection(s) to: {', '.join(blocked)}. {self._blocked_hint()}"

    # --- preparing --------------------------------------------------------

    def _say(self, message: str) -> None:
        get_logger().info(f"sandbox: {message}")
        if self._announce and self.on_notice is not None:
            self.on_notice(message)

    @property
    def _image(self) -> str:
        if self._prepared_image is None:
            raise RuntimeError("the sandbox image is used before the sandbox is prepared")
        return self._prepared_image

    async def _prepare(self) -> None:
        async with self._prepare_lock:
            if self._prepared_image is not None:
                return
            await self.check()
            image = await retry_once(self._ensure_image, describe="preparing the sandbox image")
            await remove_orphans(self._cli)
            self._prepared_image = image

    async def _ensure_image(self) -> str:
        return await ensure_image(self._cli, self._options.image, self._say)

    def _blocked_hint(self) -> str:
        if self._options.allowed_network == "public":
            return "The sandbox only allows HTTPS (port 443) to public addresses."
        return (
            "The sandbox is limited to the hosts in its `allowed_network` list (and the "
            "package registries). To allow one, add it to `docker: allowed_network:` in "
            "the cli tools bundle of the config."
        )

    # --- running a command -----------------------------------------------

    async def _run_in_container(
        self,
        conversation: str,
        command: str,
        timeout: float,
        stdin: str | None,
        *,
        retry_if_lost: bool,
    ) -> CommandResult:
        try:
            await self._prepare()
            container = await self._ensure_container(conversation)
        except BackendUnavailableError as exc:
            return CommandResult(exit_code=None, stdout="", stderr="", backend_error=str(exc))

        result = await self._cli.exec_command(
            container.name,
            command,
            workdir=self._workspace.root,
            timeout=timeout,
            stdin=stdin,
            forward_env=self._allowed_env,
        )
        if retry_if_lost and _lost_the_container(result):
            # Removed from under us: docker restarted, or someone ran `docker rm`.
            self._containers.pop(conversation, None)
            self._say(
                "The sandbox was lost; starting a new one (files outside the workspace are gone)."
            )
            return await self._run_in_container(
                conversation, command, timeout, stdin, retry_if_lost=False
            )
        if result.exit_code == 0:
            await self._installs.record(container, conversation, command)
        return result

    # --- one container per conversation ----------------------------------

    def _lock_for(self, conversation: str) -> asyncio.Lock:
        return self._conversation_locks.setdefault(conversation, asyncio.Lock())

    async def _ensure_container(self, conversation: str) -> DockerContainer:
        """The conversation's container: the one this object knows, one already
        running that matches the configuration (adopted), or a new one."""
        async with self._lock_for(conversation):
            fingerprint = self._fingerprint()
            known = self._containers.get(conversation)
            if known is not None and known.fingerprint == fingerprint:
                return known

            name = self.container_name(conversation)
            existing = await self._inspect(name)
            if existing is not None:
                kept = await self._keep_or_remove(conversation, name, existing, fingerprint)
                if kept is not None:
                    return kept
            return await self._create(conversation, name, fingerprint)

    async def _inspect(self, name: str) -> _ExistingContainer | None:
        inspected = await self._cli(
            "inspect", "-f",
            f'{{{{.State.Running}}}} {{{{index .Config.Labels "{FINGERPRINT_LABEL}"}}}}',
            name, timeout=INSPECT_SECONDS,
        )  # fmt: skip
        if not inspected.ok:
            return None
        running, _, fingerprint = inspected.stdout.strip().partition(" ")
        return _ExistingContainer(running == "true", fingerprint)

    async def _keep_or_remove(
        self, conversation: str, name: str, existing: _ExistingContainer, fingerprint: str
    ) -> DockerContainer | None:
        """Adopt a container that is running and matches, keep one that no longer does
        while a background command of the conversation runs in it, remove any other."""
        if existing.running and existing.fingerprint == fingerprint:
            return self._remember(conversation, DockerContainer(name, fingerprint))
        if existing.running and await self._has_background_command(name, conversation):
            if self._deferred_for.get(conversation) != fingerprint:
                self._deferred_for[conversation] = fingerprint
                self._say(
                    "The sandbox needs to be recreated (skills or settings changed) "
                    "but a background command is running; it will be after it finishes."
                )
            return self._remember(conversation, DockerContainer(name, existing.fingerprint))
        if existing.running:
            self._say("Recreating the sandbox: skills or settings changed…")
        await self._cli("rm", "-f", name, timeout=ACTION_SECONDS)
        return None

    def _remember(self, conversation: str, container: DockerContainer) -> DockerContainer:
        self._containers[conversation] = container
        return container

    async def _create(self, conversation: str, name: str, fingerprint: str) -> DockerContainer:
        mounts = build_mounts(self._workspace, self._state, conversation)
        container = await self._launcher.start(name, fingerprint, self._image, mounts)
        self._remember(conversation, container)
        self._deferred_for.pop(conversation, None)
        await self._installs.replay(container, conversation)
        self._say(f"Sandbox ready (docker · {self._options.describe_network()}).")
        return container

    def _fingerprint(self) -> str:
        """Everything a container is made with that a later configuration could change."""
        material = {
            "v": 1,
            "image": self._prepared_image,
            "workspace": str(self._workspace.root),
            "state": str(self._state.root),
            "roots": sorted(_identity(path) for path in skill_roots(self._workspace)),
            "network": self._options.allowed_hosts() or "none",
            "memory": self._options.memory,
            "cpus": self._options.cpus,
            "pids": self._options.pids_limit,
            "env": sorted(self._allowed_env),
        }
        return digest(json.dumps(material, sort_keys=True), 20)

    async def _has_background_command(self, container: str, conversation: str) -> bool:
        """A command started with ``background`` is still running in this container."""
        folder = shlex.quote(f"{self.processes_dir}/{safe_id(conversation)}")
        script = (
            f'for d in {folder}/*/; do [ -f "$d/run.pid" ] && [ ! -f "$d/run.exit" ] '
            '&& kill -0 "$(cat "$d/run.pid")" 2>/dev/null && exit 3; done; exit 0'
        )
        checked = await self._cli(
            "exec",
            "-w",
            str(self._workspace.root),
            container,
            "bash",
            "-c",
            script,
            timeout=INSPECT_SECONDS,
        )
        return checked.code == _BACKGROUND_COMMAND_RUNNING

    async def _remove_scope(self) -> None:
        listed = await self._cli(
            "ps", "-aq", *self._scope.filter_arguments, timeout=INSPECT_SECONDS
        )
        if listed.ok and listed.stdout.split():
            await self._cli("rm", "-f", *listed.stdout.split(), timeout=SLOW_ACTION_SECONDS)
        if self._network is not None:
            await self._network.remove()


def _lost_the_container(result: CommandResult) -> bool:
    return result.backend_error is not None and container_is_gone(result.backend_error)


def _identity(path: Path) -> list[object]:
    """A folder as the container sees it: where it is, and which one it is. A folder
    replaced by another at the same path is another mount."""
    try:
        status = os.stat(path)
    except OSError:
        return [str(path), 0, 0]
    return [str(path), status.st_dev, status.st_ino]
