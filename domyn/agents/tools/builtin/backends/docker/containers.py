"""Starting the container a conversation's commands run in."""

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from domyn.agents.tools.builtin.backends.base import BackendUnavailableError
from domyn.agents.tools.builtin.backends.docker.cli import (
    ACTION_SECONDS,
    CONTAINER_START_SECONDS,
    DockerCli,
    TransientDockerError,
    container_is_gone,
    retry_once,
)
from domyn.agents.tools.builtin.backends.docker.network import EgressNetwork
from domyn.agents.tools.builtin.backends.docker.options import DockerBackendOptions
from domyn.agents.tools.builtin.backends.docker.scope import FINGERPRINT_LABEL, BackendScope
from domyn.agents.tools.builtin.session import PROJECT_DIR_VAR

CONTAINER_HOME = "/tmp/domyn-home"
"""``HOME`` inside the container. Not a mount: ``pip install --user`` and
``npm install -g`` write under it, and a new container starts without them."""

NPM_PREFIX = f"{CONTAINER_HOME}/.npm-global"

_PROXY_VARIABLES = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")
_NO_PROXY = "localhost,127.0.0.1"
_DOCKER_ERROR = 125


@dataclass(frozen=True)
class DockerContainer:
    name: str
    fingerprint: str
    """What it was made with (image, mounts, limits, network). A container whose
    fingerprint differs from the configuration's is made again."""


class ContainerLauncher:
    def __init__(
        self,
        cli: DockerCli,
        options: DockerBackendOptions,
        scope: BackendScope,
        network: EgressNetwork | None,
        workdir: Path,
        notify: Callable[[str], None],
        project_dir: Path | None = None,
    ) -> None:
        self._project_dir = project_dir
        self._cli = cli
        self._options = options
        self._scope = scope
        self._network = network
        self._workdir = workdir
        self._notify = notify

    async def start(
        self, name: str, fingerprint: str, image: str, mounts: list[str]
    ) -> DockerContainer:
        """Start the container, once more if Docker failed the first time. A refusal
        that comes back is real and is raised."""
        self._notify("Starting the container…")
        if self._network is not None:
            await self._network.ensure(image)

        async def attempt() -> DockerContainer:
            return await self._start_once(name, fingerprint, image, mounts)

        return await retry_once(attempt, describe="starting the container")

    async def _start_once(
        self, name: str, fingerprint: str, image: str, mounts: list[str]
    ) -> DockerContainer:
        started = await self._cli(
            *self._run_arguments(name, fingerprint, image, mounts), timeout=CONTAINER_START_SECONDS
        )
        if not started.ok:
            await self._cli("rm", "-f", name, timeout=ACTION_SECONDS)
            raise TransientDockerError(f"could not start the container: {started.error()}")
        await self._probe(name, image)
        return DockerContainer(name, fingerprint)

    async def _probe(self, name: str, image: str) -> None:
        """Make the container's home, as the container's user (the host's uid), and
        check the image has the tools the background-command wrapper needs."""
        probed = await self._cli(
            "exec", name, "sh", "-c",
            f"mkdir -p {CONTAINER_HOME}/.local {NPM_PREFIX} && command -v bash timeout ps >/dev/null",
            timeout=ACTION_SECONDS,
        )  # fmt: skip
        if probed.ok:
            return
        await self._cli("rm", "-f", name, timeout=ACTION_SECONDS)
        docker_failed = (
            probed.code == _DOCKER_ERROR
            or container_is_gone(probed.stderr)
            or "daemon" in probed.stderr.lower()
        )
        if docker_failed:  # the probe did not run, so the image was not looked at
            raise TransientDockerError(f"the container stopped as it started: {probed.error()}")
        raise BackendUnavailableError(
            f"the image {image} lacks bash, timeout or ps: use `docker: image: auto` "
            "or an image that has them"
        )

    def _run_arguments(
        self, name: str, fingerprint: str, image: str, mounts: list[str]
    ) -> list[str]:
        return [
            "run", "-d", "--init", "--name", name,
            *self._scope.label_arguments,
            "--label", f"{FINGERPRINT_LABEL}={fingerprint}",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", str(self._options.pids_limit),
            "--memory", self._options.memory,
            "--cpus", str(self._options.cpus),
            "-w", str(self._workdir),
            *_environment(
                HOME=CONTAINER_HOME,
                PYTHONUSERBASE=f"{CONTAINER_HOME}/.local",
                NPM_CONFIG_PREFIX=NPM_PREFIX,
                PIP_DISABLE_PIP_VERSION_CHECK="1",
                PYTHONDONTWRITEBYTECODE="1",
                # What a skill or the agent reads to know where it is running: not on the
                # machine a config can be edited on (see ``mounts``).
                DOMYN_BACKEND="docker",
                **({PROJECT_DIR_VAR: str(self._project_dir)} if self._project_dir else {}),
            ),
            *_as_host_user(),
            *self._network_arguments(),
            *mounts,
            image, "sleep", "infinity",
        ]  # fmt: skip

    def _network_arguments(self) -> list[str]:
        if self._network is None:
            return ["--network", "none"]
        proxy = self._network.proxy_url
        variables = dict.fromkeys(_PROXY_VARIABLES, proxy)
        variables |= {"no_proxy": _NO_PROXY, "NO_PROXY": _NO_PROXY}
        variables |= {"npm_config_proxy": proxy, "npm_config_https_proxy": proxy}
        return ["--network", self._network.name, *_environment(**variables)]


def _environment(**variables: str) -> list[str]:
    return [flag for name, value in variables.items() for flag in ("-e", f"{name}={value}")]


def _as_host_user() -> list[str]:
    """Run as the user who runs domyn, so files written in the workspace are theirs."""
    if not hasattr(os, "getuid"):
        return []
    return ["--user", f"{os.getuid()}:{os.getgid()}"]
