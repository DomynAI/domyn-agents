"""What one backend names and labels, and cleaning up after one that died.

Everything the sandbox creates (containers, the network, the proxy) carries the
labels below. The owner is the process that made it, so the next start can tell
what a crashed one left behind; the scope is the owner plus the workspace, so
closing a backend removes its own and nobody else's.
"""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from domyn.agents.logger import get_logger
from domyn.agents.tools.builtin.backends.docker_sandbox.cli import (
    ACTION_SECONDS,
    INSPECT_SECONDS,
    SLOW_ACTION_SECONDS,
    DockerCli,
)

SANDBOX_LABEL = "domyn.sandbox"
OWNER_LABEL = "domyn.owner-pid"
SCOPE_LABEL = "domyn.scope"
FINGERPRINT_LABEL = "domyn.fingerprint"


def digest(text: str, length: int = 10) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:length]


@dataclass(frozen=True)
class SandboxScope:
    scope: str
    owner_pid: int

    @classmethod
    def of(cls, workspace_root: Path) -> Self:
        pid = os.getpid()
        return cls(scope=f"{digest(str(workspace_root), 8)}-{pid}", owner_pid=pid)

    @property
    def label_arguments(self) -> list[str]:
        """``docker run`` / ``network create`` flags that mark a resource as ours."""
        return [
            "--label", f"{SANDBOX_LABEL}=1",
            "--label", f"{OWNER_LABEL}={self.owner_pid}",
            "--label", f"{SCOPE_LABEL}={self.scope}",
        ]  # fmt: skip

    @property
    def filter_arguments(self) -> list[str]:
        """``docker ps`` flags that select this scope's resources."""
        return [
            "--filter",
            f"label={SANDBOX_LABEL}=1",
            "--filter",
            f"label={SCOPE_LABEL}={self.scope}",
        ]

    @property
    def network_name(self) -> str:
        return f"domyn-net-{self.scope}"

    @property
    def proxy_name(self) -> str:
        return f"domyn-egress-{self.scope}"

    def container_name(self, conversation: str) -> str:
        return f"domyn-{self.scope}-{digest(conversation, 12)}"


async def remove_orphans(cli: DockerCli) -> None:
    """Remove the containers and networks of a process that has died."""
    containers = await _owned_by_dead_processes(cli, ("ps", "-a"), name_field="Names")
    if containers:
        get_logger().info(
            f"sandbox: removing {len(containers)} container(s) left by a dead process"
        )
        await cli("rm", "-f", *containers, timeout=SLOW_ACTION_SECONDS)
    networks = await _owned_by_dead_processes(cli, ("network", "ls"), name_field="Name")
    if networks:
        get_logger().info(f"sandbox: removing {len(networks)} network(s) left by a dead process")
        for name in networks:
            await cli("network", "rm", name, timeout=ACTION_SECONDS)


async def _owned_by_dead_processes(
    cli: DockerCli, listing: tuple[str, ...], *, name_field: str
) -> list[str]:
    columns = f'{{{{.{name_field}}}}}\t{{{{.Label "{OWNER_LABEL}"}}}}'
    found = await cli(
        *listing,
        "--filter",
        f"label={SANDBOX_LABEL}=1",
        "--format",
        columns,
        timeout=INSPECT_SECONDS,
    )
    if not found.ok:
        return []
    owned: list[str] = []
    for line in found.stdout.splitlines():
        name, _, owner = line.partition("\t")
        if name and owner.strip().isdigit() and not _is_alive(int(owner)):
            owned.append(name)
    return owned


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
