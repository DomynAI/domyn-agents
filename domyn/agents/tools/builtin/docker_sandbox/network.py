"""The only way out of a container: an HTTPS proxy on a network with no route.

Commands run on an ``--internal`` Docker network, which has no route to anywhere.
One proxy container is attached to it and to the default bridge, and tunnels
``CONNECT host:443`` for the hosts it may and refuses everything else
(``egress_proxy.py``). The proxy is the enforcement, not the ``HTTP_PROXY``
variables: a tool that ignores them simply has no network.
"""

import asyncio
from collections.abc import Callable
from pathlib import Path

from domyn.agents.tools.builtin.backends import BackendUnavailableError
from domyn.agents.tools.builtin.docker_sandbox import egress_proxy
from domyn.agents.tools.builtin.docker_sandbox.cli import (
    ACTION_SECONDS,
    INSPECT_SECONDS,
    PULL_SECONDS,
    SLOW_ACTION_SECONDS,
    DockerCli,
    TransientDockerError,
    retry_once,
)
from domyn.agents.tools.builtin.docker_sandbox.images import BASE_IMAGE
from domyn.agents.tools.builtin.docker_sandbox.scope import SandboxScope

PROXY_PORT = 3128

_UNPRIVILEGED_USER = "65534:65534"
_DENIED = "DENY "

# A busy machine can take a good while to start Python in a container, so the proxy
# gets up to 30 s to say it is ready (and the wait ends the moment it does).
_READY_ATTEMPTS = 120
_READY_INTERVAL_SECONDS = 0.25


class EgressNetwork:
    def __init__(
        self,
        cli: DockerCli,
        scope: SandboxScope,
        allowed_hosts: list[str],
        notify: Callable[[str], None],
    ) -> None:
        self._cli = cli
        self._scope = scope
        self._allowed_hosts = allowed_hosts
        self._notify = notify
        self._lock = asyncio.Lock()
        self._ready = False
        self._last_denial = ""

    @property
    def name(self) -> str:
        return self._scope.network_name

    @property
    def proxy_url(self) -> str:
        return f"http://{self._scope.proxy_name}:{PROXY_PORT}"

    async def ensure(self) -> None:
        """The network and the proxy exist and the proxy answers. Safe to call often."""
        async with self._lock:
            if self._ready:
                return
            await self._ensure_network()
            if not await self._proxy_is_running():
                await retry_once(self._start_proxy, describe="starting the network proxy")
            self._ready = True

    async def remove(self) -> None:
        """Remove the network. The proxy is a labelled container, removed with the others."""
        if self._ready:
            await self._cli("network", "rm", self.name, timeout=ACTION_SECONDS)
        self._ready = False

    async def blocked_since(self, since: float) -> list[str]:
        """The hosts the proxy refused a connection to since ``since``, each once."""
        logs = await self._cli(
            "logs",
            "-t",
            "--since",
            str(int(since) - 2),
            self._scope.proxy_name,
            timeout=INSPECT_SECONDS,
        )
        if not logs.ok:
            return []
        hosts: list[str] = []
        newest = self._last_denial
        for line in (logs.stdout + logs.stderr).splitlines():
            stamp, _, event = line.partition(" ")
            if not event.startswith(_DENIED) or stamp <= self._last_denial:
                continue
            newest = max(newest, stamp)
            host = _host_of(event)
            if host not in hosts:
                hosts.append(host)
        self._last_denial = newest
        return hosts

    async def _ensure_network(self) -> None:
        if (await self._cli("network", "inspect", self.name, timeout=INSPECT_SECONDS)).ok:
            return
        created = await self._cli(
            "network",
            "create",
            "--internal",
            *self._scope.label_arguments,
            self.name,
            timeout=ACTION_SECONDS,
        )
        if not created.ok and "already exists" not in created.stderr:
            raise BackendUnavailableError(
                f"could not create the sandbox network: {created.error(200)}"
            )

    async def _proxy_is_running(self) -> bool:
        state = await self._cli(
            "inspect", "-f", "{{.State.Running}}", self._scope.proxy_name, timeout=INSPECT_SECONDS
        )
        return state.ok and state.stdout.strip() == "true"

    async def _start_proxy(self) -> None:
        name = self._scope.proxy_name
        await self._cli("rm", "-f", name, timeout=ACTION_SECONDS)
        await self._pull_base_image_if_missing()
        started = await self._cli(*self._run_arguments(), timeout=SLOW_ACTION_SECONDS)
        if not started.ok:
            raise TransientDockerError(f"could not start the network proxy: {started.error(200)}")
        # Its way out: the default bridge. The command containers never get this.
        connected = await self._cli("network", "connect", "bridge", name, timeout=ACTION_SECONDS)
        if not connected.ok:
            raise TransientDockerError(f"could not connect the proxy: {connected.error(200)}")
        await self._wait_until_ready()

    async def _pull_base_image_if_missing(self) -> None:
        if not (await self._cli("image", "inspect", BASE_IMAGE, timeout=INSPECT_SECONDS)).ok:
            self._notify(f"Pulling {BASE_IMAGE} for the network proxy…")
            await self._cli("pull", BASE_IMAGE, timeout=PULL_SECONDS)

    def _run_arguments(self) -> list[str]:
        return [
            "run", "-d", "--name", self._scope.proxy_name, *self._scope.label_arguments,
            "--network", self.name,
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--read-only", "--pids-limit", "128", "--memory", "128m",
            "--user", _UNPRIVILEGED_USER,
            BASE_IMAGE, "python", "-c", _proxy_source(),
            "--port", str(PROXY_PORT), "--allow", ",".join(self._allowed_hosts),
        ]  # fmt: skip

    async def _wait_until_ready(self) -> None:
        name = self._scope.proxy_name
        for _ in range(_READY_ATTEMPTS):
            logs = await self._cli("logs", name, timeout=INSPECT_SECONDS)
            if "READY" in logs.stdout:
                return
            running = await self._cli(
                "inspect", "-f", "{{.State.Running}}", name, timeout=INSPECT_SECONDS
            )
            if running.stdout.strip() != "true":
                raise TransientDockerError(f"the network proxy stopped: {logs.error(200)}")
            await asyncio.sleep(_READY_INTERVAL_SECONDS)
        raise TransientDockerError("the network proxy did not become ready")


def _host_of(denial: str) -> str:
    """The host in a ``DENY host:port reason`` line of the proxy's log."""
    words = denial.split()
    target = words[1] if len(words) > 1 else "?"
    return target.rpartition(":")[0] or target


def _proxy_source() -> str:
    """The proxy's code, passed to ``python -c`` so the container needs no mount."""
    path = Path(egress_proxy.__file__)
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BackendUnavailableError(f"the sandbox proxy script is missing: {path}") from exc
