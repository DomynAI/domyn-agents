"""Run shell commands in a Docker container; everything else stays on this machine.

The agent, its file tools and its approvals run in the terminal process as ever.
Only what ``run_command`` starts is moved into a container that is locked in:
the workspace is all it can write, nothing of the host is mounted, and the only
network it has is an allowlist for package registries. This is the opposite of
running the whole agent inside a container, and the split the Domyn platform uses.

**The transport is ``docker exec``.** There is no relay and no daemon of ours in
the container: one long-lived container per conversation, one ``docker exec`` per
command. The approval gate runs on the host before the command string is sent, so
``/auto``, deny rules and the decision model all work unchanged — the container
is only where the command runs afterwards.

**What the container sees**

* The workspace, read-write, at the *same absolute path* it has on the host, as
  the working directory. Paths in output match the host, and the background
  command directory (``.domyn/processes``) works without a change.
* Skill directories, read-only, at their own paths: builtin skills run their
  scripts through ``run_command``.
* Read-only masks over the places a writable workspace would let a command reach
  the host through: ``.git/hooks`` and ``.git/config`` (a planted hook runs on the
  host at the next ``git commit``), and ``.domyn`` and ``.agents`` — their
  ``config.yaml``, skills and instruction files are read into the *next turn's
  prompt*, so a write there would persist instructions. ``.domyn/processes`` and
  the conversation's own folder stay writable. A mask exists only for what exists
  when the container is created.
* Nothing else: no Docker socket, no home folder, no credentials. Environment
  variables reach a command only when ``allowed_env`` names them.

**The network** is an ``--internal`` Docker network, which has no route out, with
one proxy container attached to it and to the bridge (``scripts/egress_proxy.py``):
it tunnels HTTPS to the allowlisted hosts and refuses everything else. The
proxy is the enforcement, not the ``HTTP_PROXY`` variables — a tool that ignores
them simply has no network.

**Life cycle.** A container is created on a conversation's first command,
destroyed when the conversation ends (``end_conversation``: ``/clear``,
``/resume``, exit) and at ``close``. Containers are named after this process, so a
backend rebuilt by a config reload *adopts* the one already running instead of
leaking it; one left by a process that has died is removed by the next start.
What the agent installed is recorded in the conversation's folder and replayed
when a new container is made for it (``--resume``).

**The claim this supports** is that a command cannot touch the host outside the
workspace, its credentials or its processes. It does not say a command cannot
send data out through an allowlisted registry, and the workspace itself is
writable by design.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import shlex
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField

from domyn.agents.logger import get_logger
from domyn.agents.tools.builtin.backends import (
    BackendUnavailableError,
    CommandBackend,
    CommandResult,
)
from domyn.agents.tools.builtin.session import HOME_DIRNAME, SessionEnvironment, safe_id
from domyn.agents.tools.builtin.session import _ignore_from_version_control as _ignore_in_git

if TYPE_CHECKING:
    from domyn.agents.tools.builtin.workspace import Workspace

BASE_IMAGE = "python:3.12-slim"
"""What the default image is made from, and what the proxy container runs."""

NODE_IMAGE = "node:22-slim"
"""Where the image takes Node and npm from (same Debian family as ``BASE_IMAGE``)."""

DERIVED_IMAGE = "domyn-sandbox:py312-2"
"""``BASE_IMAGE`` plus what an agent working on a project reaches for, built once on
this machine: ``procps`` (the background command wrapper needs ``ps`` and ``pkill``,
which the slim image does not have), ``git`` and ``curl``, and Node with npm."""

_DOCKERFILE = (
    f"FROM {NODE_IMAGE} AS node\n"
    f"FROM {BASE_IMAGE}\n"
    "RUN apt-get update \\\n"
    " && apt-get install -y --no-install-recommends procps git curl ca-certificates \\\n"
    " && rm -rf /var/lib/apt/lists/*\n"
    "COPY --from=node /usr/local/bin/node /usr/local/bin/node\n"
    "COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules\n"
    "RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \\\n"
    " && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx\n"
)

DEFAULT_ALLOW: tuple[str, ...] = (
    "pypi.org",
    "files.pythonhosted.org",
    "registry.npmjs.org",
    "github.com",
)
"""The package registries, and GitHub for ``git clone`` over HTTPS. Anything else is
added with ``docker: allow:``."""

LABEL = "domyn.sandbox"
PROXY_PORT = 3128
SANDBOX_HOME = "/tmp/domyn-home"
REQUIREMENTS_FILE = "requirements.txt"
NPM_GLOBALS_FILE = "npm-global.txt"
NPM_PREFIX = f"{SANDBOX_HOME}/.npm-global"

# How much longer than the command's own limit the host waits. The limit is
# enforced inside the container by `timeout`; this only bounds a `docker exec`
# that has stopped answering.
_EXEC_GRACE_S = 15.0

_INSTALLS = re.compile(r"(?:^|[\s;&|('\"])(?:pip3?|pipx|uv|python3?\s+-m\s+pip)(?:\s|$)")
"""A command that may have changed what is installed. Checked on the text only to
decide whether to look; what is recorded is what ``pip freeze`` then says. A quote
may come before the program: ``run_command`` sends the backend a launcher script with
the model's command quoted inside it, not the command itself."""

_NPM_GLOBAL = re.compile(
    r"(?:^|[\s;&|('\"])npm\s[^;&|\n]*?\s(?:-g|--global|--location[= ]global)(?:\s|$)"
)
"""An ``npm`` command that may have changed the global packages. A local install
lands in the workspace's ``node_modules``, which is the host's and needs no record."""


class DockerOptions(BaseModel):
    """The ``docker:`` block of a ``tool_bundles`` entry. Read when ``sandbox: docker``."""

    model_config = ConfigDict(extra="forbid")

    image: str = "auto"
    """``auto`` builds ``domyn-sandbox`` from ``python:3.12-slim`` on first use. Any
    other value is used as it is and must provide ``bash``, ``timeout`` and ``ps``."""

    network: Literal["allowlist", "none", "open"] = "allowlist"
    """``allowlist``: package registries only. ``none``: no network at all.
    ``open``: the container's ordinary network, with no restriction."""

    allow: list[str] = PydanticField(default_factory=list)
    """Hosts added to the allowlist (port 443). ``*.example.com`` matches subdomains."""

    memory: str = "2g"
    cpus: float = 2.0
    pids_limit: int = 512
    docker_binary: str = "docker"
    """The CLI to run: ``docker``, or ``podman``, or a path."""


@dataclass
class _Container:
    name: str
    fingerprint: str


class _Docker:
    """``docker`` as a subprocess: argv in, ``(code, stdout, stderr)`` out."""

    def __init__(self, binary: str) -> None:
        self.binary = binary

    async def __call__(
        self,
        *args: str,
        timeout: float = 60.0,
        stdin: str | None = None,
        env: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        try:
            process = await asyncio.create_subprocess_exec(
                self.binary,
                *args,
                stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except FileNotFoundError as exc:
            raise BackendUnavailableError(
                f"Docker was not found ('{self.binary}' is not on your PATH).",
                fix="Install Docker from https://docs.docker.com/get-docker/.",
            ) from exc
        try:
            out, err = await asyncio.wait_for(
                process.communicate(input=stdin.encode() if stdin is not None else None),
                timeout=timeout,
            )
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=5)
            return 124, "", f"docker {args[0] if args else ''} timed out after {timeout:.0f}s"
        return (
            process.returncode if process.returncode is not None else 1,
            out.decode("utf-8", "replace"),
            err.decode("utf-8", "replace"),
        )


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _digest(text: str, length: int = 10) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:length]


class DockerBackend(CommandBackend):
    """Commands run in a container per conversation. See the module docstring."""

    name = "docker"
    is_isolated = True
    destroys_on_close = True

    def __init__(
        self,
        workspace: Workspace,
        *,
        options: DockerOptions | None = None,
        allowed_env: Iterable[str] = (),
    ) -> None:
        self._workspace = workspace
        self._options = options or DockerOptions()
        self._allowed_env = [n for n in allowed_env if n]
        self._docker = _Docker(self._options.docker_binary)
        self._owner = os.getpid()
        self._scope = f"{_digest(str(workspace.root), 8)}-{self._owner}"
        self._labels = [
            "--label", f"{LABEL}=1",
            "--label", f"domyn.owner-pid={self._owner}",
            "--label", f"domyn.scope={self._scope}",
        ]  # fmt: skip
        self._network_name = f"domyn-net-{self._scope}"
        self._proxy_name = f"domyn-egress-{self._scope}"

        self._infra = asyncio.Lock()
        self._conversation_locks: dict[str, asyncio.Lock] = {}
        self._containers: dict[str, _Container] = {}
        self._image: str | None = None
        self._prepared = False
        self._network_ready = False
        self._deny_seen = ""
        self._deferred_for: dict[str, str] = {}
        self._loud = False
        self._warned_image = False

    # --- the interface ---------------------------------------------------

    async def start(self) -> None:
        await self.prepare()

    async def prepare(self) -> None:
        """Check Docker and make sure the image exists. The slow, first-time part.

        Called in the foreground at session start, so a pull or a build happens
        under a message and a spinner rather than inside someone's first command.
        """
        self._loud = True
        await self._prepare()

    async def warm_up(self, conversation_id: str) -> None:
        """Have the conversation's container ready, quietly. Never raises: a
        failure here is met again, with the same message, by the first command."""
        self._loud = False
        try:
            await self._prepare()
            await self._ensure_container(conversation_id or "default")
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
        conversation = conversation_id or "default"
        self._loud = True
        for attempt in (1, 2):
            try:
                await self._prepare()
                container = await self._ensure_container(conversation)
            except BackendUnavailableError as exc:
                return CommandResult(exit_code=None, stdout="", stderr="", backend_error=str(exc))

            started = time.time()
            result = await self._exec(container, command, timeout, stdin)
            gone = result.backend_error is not None and _container_is_gone(result.backend_error)
            if gone and attempt == 1:
                # Removed from under us (docker restarted, someone ran `docker rm`).
                self._containers.pop(conversation, None)
                continue
            if result.backend_error is None and not result.timed_out:
                result = await self._after(container, conversation, command, result, started)
            return result
        return result  # pragma: no cover - the loop always returns

    async def end_conversation(self, conversation_id: str) -> None:
        conversation = conversation_id or "default"
        # Under the conversation's lock: a container still being made for it (the
        # warm-up) finishes first, or it would be made after this and never removed.
        async with self._lock_for(conversation):
            self._containers.pop(conversation, None)
            self._deferred_for.pop(conversation, None)
            with contextlib.suppress(Exception):
                await self._docker("rm", "-f", self._container_name(conversation), timeout=30)

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
                code, out, _ = await self._docker(
                    "ps", "-aq",
                    "--filter", f"label={LABEL}=1",
                    "--filter", f"label=domyn.scope={self._scope}",
                    timeout=30,
                )  # fmt: skip
                ids = out.split() if code == 0 else []
                if ids:
                    await self._docker("rm", "-f", *ids, timeout=60)
                if self._network_ready:
                    await self._docker("network", "rm", self._network_name, timeout=30)
        finally:
            self._network_ready = False
            for lock in locks:
                lock.release()

    # --- preparing Docker ------------------------------------------------

    def _say(self, message: str) -> None:
        get_logger().info(f"sandbox: {message}")
        if self._loud and self.on_notice is not None:
            self.on_notice(message)

    async def check(self) -> None:
        """Docker is installed and its daemon answers. The fast part of ``prepare``."""
        self._refuse_unsafe_workspace()
        code, _, err = await self._docker("info", "--format", "{{.ServerVersion}}", timeout=10)
        if code != 0:
            # Short on purpose: the client's own line carries the socket path and reads
            # as noise. It stays in the log.
            get_logger().debug(f"docker info detail: {err.strip()[-400:]}")
            if "permission denied" in err.lower():
                detail = "permission denied connecting to the Docker daemon: is your user allowed to use Docker?"
            else:
                detail = "failed to connect to the docker daemon: is the daemon running?"
            raise BackendUnavailableError(
                f"Docker is installed but not running ({detail}).",
                fix=self._start_hint(),
            )

    def _start_hint(self) -> str:
        binary = Path(self._options.docker_binary).name
        if binary != "docker":
            return f"Start {binary} (for podman on macOS or Windows: `podman machine start`)."
        if sys.platform == "darwin":
            return "Start Docker Desktop (`open -a Docker`) and wait until it says it is running."
        if sys.platform.startswith("linux"):
            return "Start the Docker service (`sudo systemctl start docker`)."
        return "Start Docker Desktop and wait until it says it is running."

    async def _prepare(self) -> None:
        if self._prepared:
            return
        async with self._infra:
            if self._prepared:
                return  # type: ignore[unreachable]  # set by a task that held the lock
            await self.check()
            self._image = await self._ensure_image()
            await self._reap_orphans()
            self._prepared = True

    def _refuse_unsafe_workspace(self) -> None:
        root = self._workspace.root
        home = Path.home().resolve()
        if root == Path(root.anchor) or root == home or root in home.parents:
            raise BackendUnavailableError(
                f"The workspace is {root}: mounting it would hand the container everything in it.",
                fix="Run domyn from a project folder, or set `sandbox: local`.",
            )

    async def _ensure_image(self) -> str:
        wanted = self._options.image
        image = DERIVED_IMAGE if wanted == "auto" else wanted
        code, _, _ = await self._docker("image", "inspect", image, timeout=20)
        if code == 0:
            return image
        if wanted != "auto":
            self._say(f"Pulling sandbox image {image} (first run only)…")
            started = time.time()
            code, _, err = await self._docker("pull", image, timeout=1800)
            if code != 0:
                raise BackendUnavailableError(
                    f"could not pull the sandbox image {image}: {err.strip()[-200:]}"
                )
            self._say(f"Pulled {image} in {time.time() - started:.0f}s.")
            return image

        self._say("Preparing the sandbox image (first run only; Python, Node, git and curl)…")
        started = time.time()
        code, _, err = await self._docker("pull", BASE_IMAGE, timeout=1800)
        if code != 0:
            raise BackendUnavailableError(
                f"could not pull {BASE_IMAGE}: {err.strip()[-200:]}. Check the network, or set "
                "`docker: image:` to an image you already have."
            )
        code, _, err = await self._docker(
            "build", "-t", DERIVED_IMAGE, "-", stdin=_DOCKERFILE, timeout=900
        )
        if code != 0:
            raise BackendUnavailableError(
                f"could not build the sandbox image: {err.strip()[-300:]}"
            )
        self._say(f"Sandbox image ready ({time.time() - started:.0f}s).")
        return DERIVED_IMAGE

    async def _reap_orphans(self) -> None:
        """Remove what a process that has died left behind."""
        for kind, listing in (
            ("container", ("ps", "-a")),
            ("network", ("network", "ls")),
        ):
            code, out, _ = await self._docker(
                *listing,
                "--filter",
                f"label={LABEL}=1",
                "--format",
                '{{.Name}}\t{{.Label "domyn.owner-pid"}}'
                if kind == "network"
                else '{{.Names}}\t{{.Label "domyn.owner-pid"}}',
                timeout=20,
            )
            if code != 0:
                continue
            dead = []
            for line in out.splitlines():
                name, _, pid_text = line.partition("\t")
                if name and pid_text.strip().isdigit() and not _pid_alive(int(pid_text)):
                    dead.append(name)
            if not dead:
                continue
            get_logger().info(f"sandbox: removing {len(dead)} {kind}(s) left by a dead process")
            if kind == "container":
                await self._docker("rm", "-f", *dead, timeout=60)
            else:
                for name in dead:
                    await self._docker("network", "rm", name, timeout=20)

    # --- the network -----------------------------------------------------

    def _allow(self) -> list[str]:
        return list(dict.fromkeys([*DEFAULT_ALLOW, *(h.lower() for h in self._options.allow)]))

    async def _ensure_network(self) -> None:
        if self._options.network != "allowlist" or self._network_ready:
            return
        async with self._infra:
            if self._network_ready:
                return  # type: ignore[unreachable]  # set by a task that held the lock
            labels = self._labels
            code, _, _ = await self._docker("network", "inspect", self._network_name, timeout=15)
            if code != 0:
                code, _, err = await self._docker(
                    "network", "create", "--internal", *labels, self._network_name, timeout=30
                )
                if code != 0 and "already exists" not in err:
                    raise BackendUnavailableError(
                        f"could not create the sandbox network: {err.strip()[-200:]}"
                    )

            code, running, _ = await self._docker(
                "inspect", "-f", "{{.State.Running}}", self._proxy_name, timeout=15
            )
            if code != 0 or running.strip() != "true":
                await self._docker("rm", "-f", self._proxy_name, timeout=30)
                await self._start_proxy(labels)
            self._network_ready = True

    async def _start_proxy(self, labels: list[str]) -> None:
        source = _proxy_source()
        code, _, _ = await self._docker("image", "inspect", BASE_IMAGE, timeout=20)
        if code != 0:
            self._say(f"Pulling {BASE_IMAGE} for the network proxy…")
            await self._docker("pull", BASE_IMAGE, timeout=1800)
        code, _, err = await self._docker(
            "run", "-d", "--name", self._proxy_name, *labels,
            "--network", self._network_name,
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--read-only", "--pids-limit", "128", "--memory", "128m",
            "--user", "65534:65534",
            BASE_IMAGE, "python", "-c", source,
            "--port", str(PROXY_PORT), "--allow", ",".join(self._allow()),
            timeout=60,
        )  # fmt: skip
        if code != 0:
            raise BackendUnavailableError(
                f"could not start the network proxy: {err.strip()[-200:]}"
            )
        # Its way out: the default bridge. The command container never gets this.
        code, _, err = await self._docker(
            "network", "connect", "bridge", self._proxy_name, timeout=30
        )
        if code != 0:
            raise BackendUnavailableError(f"could not connect the proxy: {err.strip()[-200:]}")
        for _ in range(40):
            _, logs, _ = await self._docker("logs", self._proxy_name, timeout=10)
            if "READY" in logs:
                return
            code, running, _ = await self._docker(
                "inspect", "-f", "{{.State.Running}}", self._proxy_name, timeout=10
            )
            if running.strip() != "true":
                raise BackendUnavailableError(f"the network proxy stopped: {logs.strip()[-200:]}")
            await asyncio.sleep(0.25)
        raise BackendUnavailableError("the network proxy did not become ready")

    # --- one container per conversation -----------------------------------

    def _container_name(self, conversation: str) -> str:
        return f"domyn-{self._scope}-{_digest(conversation, 12)}"

    def _conversation_home(self, conversation: str) -> Path:
        return SessionEnvironment(self._workspace.root).paths(conversation).home

    def _skill_roots(self) -> list[Path]:
        return [p for p in self._workspace.extra_roots if p.is_dir() and "," not in str(p)]

    def _fingerprint(self) -> str:
        roots = []
        for path in self._skill_roots():
            try:
                st = path.stat()
                roots.append([str(path), st.st_dev, st.st_ino])
            except OSError:
                continue
        ws = self._workspace.root
        material = {
            "v": 1,
            "image": self._image,
            "workspace": str(ws),
            "roots": sorted(roots),
            "network": self._options.network,
            "allow": self._allow() if self._options.network == "allowlist" else [],
            "memory": self._options.memory,
            "cpus": self._options.cpus,
            "pids": self._options.pids_limit,
            "env": sorted(self._allowed_env),
        }
        return _digest(json.dumps(material, sort_keys=True), 20)

    def _lock_for(self, conversation: str) -> asyncio.Lock:
        return self._conversation_locks.setdefault(conversation, asyncio.Lock())

    async def _ensure_container(self, conversation: str) -> _Container:
        async with self._lock_for(conversation):
            fingerprint = self._fingerprint()
            known = self._containers.get(conversation)
            if known is not None and known.fingerprint == fingerprint:
                return known

            name = self._container_name(conversation)
            code, state, _ = await self._docker(
                "inspect", "-f",
                '{{.State.Running}} {{index .Config.Labels "domyn.fingerprint"}}',
                name, timeout=15,
            )  # fmt: skip
            if code == 0:
                running, _, label = state.strip().partition(" ")
                if running == "true" and label == fingerprint:
                    adopted = _Container(name, fingerprint)
                    self._containers[conversation] = adopted
                    return adopted
                if running == "true" and await self._busy(name, conversation):
                    if self._deferred_for.get(conversation) != fingerprint:
                        self._deferred_for[conversation] = fingerprint
                        self._say(
                            "The sandbox needs to be recreated (skills or settings changed) "
                            "but a background command is running; it will be after it finishes."
                        )
                    kept = _Container(name, label)
                    self._containers[conversation] = kept
                    return kept
                if running == "true":
                    self._say("Recreating the sandbox: skills or settings changed…")
                await self._docker("rm", "-f", name, timeout=30)

            return await self._create(conversation, name, fingerprint)

    async def _create(self, conversation: str, name: str, fingerprint: str) -> _Container:
        self._say("Starting the sandbox…")
        await self._ensure_network()
        mounts = self._mounts(conversation)
        argv = [
            "run", "-d", "--init", "--name", name,
            *self._labels,
            "--label", f"domyn.fingerprint={fingerprint}",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", str(self._options.pids_limit),
            "--memory", self._options.memory,
            "--cpus", str(self._options.cpus),
            "-w", str(self._workspace.root),
            "-e", f"HOME={SANDBOX_HOME}",
            "-e", f"PYTHONUSERBASE={SANDBOX_HOME}/.local",
            "-e", f"NPM_CONFIG_PREFIX={NPM_PREFIX}",
            "-e", "PIP_DISABLE_PIP_VERSION_CHECK=1",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
        ]  # fmt: skip
        if hasattr(os, "getuid"):
            argv += ["--user", f"{os.getuid()}:{os.getgid()}"]
        if self._options.network == "none":
            argv += ["--network", "none"]
        elif self._options.network == "allowlist":
            proxy = f"http://{self._proxy_name}:{PROXY_PORT}"
            argv += ["--network", self._network_name]
            for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
                argv += ["-e", f"{key}={proxy}"]
            argv += ["-e", "no_proxy=localhost,127.0.0.1", "-e", "NO_PROXY=localhost,127.0.0.1"]
            argv += ["-e", f"npm_config_proxy={proxy}", "-e", f"npm_config_https_proxy={proxy}"]
        argv += mounts
        argv += [self._image or DERIVED_IMAGE, "sleep", "infinity"]

        code, _, err = await self._docker(*argv, timeout=120)
        if code != 0:
            await self._docker("rm", "-f", name, timeout=30)
            raise BackendUnavailableError(f"could not start the sandbox: {err.strip()[-300:]}")

        container = _Container(name, fingerprint)
        # `mkdir` as the container's user (the same uid as on the host), so the
        # home exists and is ours, and the tools the wrapper needs are there.
        code, _, err = await self._docker(
            "exec", name, "sh", "-c",
            f"mkdir -p {SANDBOX_HOME}/.local {NPM_PREFIX} && command -v bash timeout ps >/dev/null",
            timeout=30,
        )  # fmt: skip
        if code != 0:
            await self._docker("rm", "-f", name, timeout=30)
            raise BackendUnavailableError(
                f"the image {self._image} lacks bash, timeout or ps: use `docker: image: auto` "
                "or an image that has them"
            )
        self._containers[conversation] = container
        self._deferred_for.pop(conversation, None)
        await self._replay_installs(container, conversation)
        self._say(f"Sandbox ready ({self._summary()}).")
        return container

    def _summary(self) -> str:
        network = {
            "allowlist": "network: registries and GitHub only",
            "none": "no network",
            "open": "network: open",
        }[self._options.network]
        return f"docker · {network}"

    def _mounts(self, conversation: str) -> list[str]:
        ws = self._workspace.root
        mounts: list[str] = []

        def add(path: Path, *, readonly: bool, target: Path | None = None) -> None:
            if "," in str(path) or "," in str(target or path):
                get_logger().warning("sandbox: a path with a comma cannot be mounted; skipped")
                return
            spec = f"type=bind,source={path},target={target or path}"
            mounts.extend(["--mount", spec + (",readonly" if readonly else "")])

        add(ws, readonly=False)

        # Masks, outermost first. A mount nested under another is applied after it.
        dot = ws / HOME_DIRNAME
        self._make_dirs(conversation)
        add(dot, readonly=True)
        add(dot / "processes", readonly=False)
        add(self._conversation_home(conversation), readonly=False)
        git = ws / ".git"
        if git.is_dir() and not git.is_symlink():
            for part, want_dir in (("hooks", True), ("config", False)):
                masked = git / part
                if masked.is_symlink():
                    continue
                if (want_dir and masked.is_dir()) or (not want_dir and masked.is_file()):
                    add(masked, readonly=True)
        agents = ws / ".agents"
        if agents.is_dir() and not agents.is_symlink():
            add(agents, readonly=True)
        for filename in ("AGENTS.md", "AGENT.md", "SOUL.md"):
            masked = ws / filename
            if masked.is_file() and not masked.is_symlink():
                add(masked, readonly=True)

        for root in self._skill_roots():
            add(root, readonly=True)
        return mounts

    def _make_dirs(self, conversation: str) -> None:
        """The writable places under ``.domyn``, made without following a link.

        A cloned repository can commit ``.domyn`` or any folder in it as a link,
        and a bind mount follows one.
        """
        ws = self._workspace.root
        wanted = [
            ws / HOME_DIRNAME / "processes",
            self._conversation_home(conversation),
        ]
        for target in wanted:
            here = ws
            for part in target.relative_to(ws).parts:
                here = here / part
                if here.is_symlink():
                    raise BackendUnavailableError(
                        f"{here} is a link: the sandbox will not mount through it"
                    )
            target.mkdir(parents=True, exist_ok=True)
        _ignore_in_git(self._conversation_home(conversation).parent)

    # --- running a command -------------------------------------------------

    async def _exec(
        self, container: _Container, command: str, timeout: float, stdin: str | None
    ) -> CommandResult:
        limit = max(1, int(timeout))
        script = 'export PATH="$HOME/.local/bin:$HOME/.npm-global/bin:$PATH"\n' + command
        argv = ["exec", "-i", "-w", str(self._workspace.root)]
        env = dict(os.environ)
        for name in self._allowed_env:
            if name in os.environ:
                argv += ["-e", name]
        argv += [container.name, "timeout", "-s", "KILL", str(limit), "bash", "-c", script]

        started = time.monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                self._docker.binary,
                *argv,
                stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                start_new_session=True,
            )
        except OSError as exc:
            return CommandResult(
                exit_code=None,
                stdout="",
                stderr="",
                backend_error=f"could not run docker: {type(exc).__name__}: {exc}",
            )
        try:
            out, err = await asyncio.wait_for(
                process.communicate(input=stdin.encode() if stdin is not None else None),
                timeout=limit + _EXEC_GRACE_S,
            )
        except TimeoutError:
            # `timeout` inside the container should have ended it. If the exec
            # client is still here something is wrong with the container.
            await _kill_client(process)
            return CommandResult(exit_code=None, stdout="", stderr="", timed_out=True)

        elapsed = time.monotonic() - started
        code = process.returncode
        stdout = out.decode("utf-8", "replace")
        stderr = err.decode("utf-8", "replace")
        if code in (124, 137) and elapsed >= limit - 1:
            return CommandResult(exit_code=None, stdout="", stderr="", timed_out=True)
        if code == 125 or (code == 1 and _container_is_gone(stderr) and not stdout):
            return CommandResult(
                exit_code=None,
                stdout="",
                stderr="",
                backend_error=stderr.strip()[-300:] or "docker failed",
            )
        return CommandResult(exit_code=code, stdout=stdout, stderr=stderr)

    async def _busy(self, name: str, conversation: str) -> bool:
        """A background command of this conversation is still running."""
        directory = shlex.quote(f".domyn/processes/{safe_id(conversation)}")
        script = (
            f'for d in {directory}/*/; do [ -f "$d/run.pid" ] && [ ! -f "$d/run.exit" ] '
            '&& kill -0 "$(cat "$d/run.pid")" 2>/dev/null && exit 3; done; exit 0'
        )
        code, _, _ = await self._docker(
            "exec", "-w", str(self._workspace.root), name, "bash", "-c", script, timeout=20
        )
        return code == 3

    # --- after a command: installs and blocked connections ----------------

    async def _after(
        self,
        container: _Container,
        conversation: str,
        command: str,
        result: CommandResult,
        started: float,
    ) -> CommandResult:
        if result.exit_code == 0 and _INSTALLS.search(command):
            await self._record_installs(container, conversation)
        if result.exit_code == 0 and _NPM_GLOBAL.search(command):
            await self._record_npm_globals(container, conversation)
        return result

    async def explain_failure(self, since: float) -> str:
        if self._options.network != "allowlist":
            return ""
        blocked = await self._blocked_since(since)
        if not blocked:
            return ""
        return (
            f"[sandbox] blocked connection(s) to: {', '.join(blocked)}. The sandbox only reaches "
            "an allowlist of hosts (package registries and GitHub). To allow a host, add it under "
            "`docker: allow:` in the cli tools bundle of the config."
        )

    async def _blocked_since(self, started: float) -> list[str]:
        code, out, err = await self._docker(
            "logs", "-t", "--since", str(int(started) - 2), self._proxy_name, timeout=10
        )
        if code != 0:
            return []
        hosts: list[str] = []
        newest = self._deny_seen
        for line in (out + err).splitlines():
            stamp, _, rest = line.partition(" ")
            if not rest.startswith("DENY ") or stamp <= self._deny_seen:
                continue
            newest = max(newest, stamp)
            target = rest.split()[1] if len(rest.split()) > 1 else "?"
            host = target.rpartition(":")[0] or target
            if host not in hosts:
                hosts.append(host)
        self._deny_seen = newest
        return hosts

    def _requirements_path(self, conversation: str) -> Path:
        return self._conversation_home(conversation) / REQUIREMENTS_FILE

    async def _record_installs(self, container: _Container, conversation: str) -> None:
        code, out, _ = await self._docker(
            "exec", container.name, "bash", "-c",
            'export PATH="$HOME/.local/bin:$PATH"; pip freeze --user 2>/dev/null',
            timeout=60,
        )  # fmt: skip
        if code != 0:
            return
        target = self._requirements_path(conversation)
        try:
            if target.is_symlink():
                return
            if target.exists() and target.read_text(encoding="utf-8") == out:
                return
            temporary = target.with_name(target.name + ".tmp")
            temporary.write_text(out, encoding="utf-8")
            os.replace(temporary, target)
        except OSError as exc:
            get_logger().warning(
                f"sandbox: could not record installed packages: {type(exc).__name__}"
            )
            get_logger().debug(f"sandbox requirements detail: {exc}")

    def _npm_globals_path(self, conversation: str) -> Path:
        return self._conversation_home(conversation) / NPM_GLOBALS_FILE

    async def _record_npm_globals(self, container: _Container, conversation: str) -> None:
        """Write the global npm packages as ``name@version`` lines, so a new container
        can install them again. Only what ``npm ls`` reports at the top level."""
        _, out, _ = await self._docker(
            "exec", container.name, "bash", "-c",
            'export PATH="$HOME/.npm-global/bin:$PATH"; npm ls -g --depth=0 --json 2>/dev/null',
            timeout=60,
        )  # fmt: skip
        try:
            found = json.loads(out).get("dependencies") or {}
            lines = sorted(
                f"{name}@{info['version']}"
                for name, info in found.items()
                if isinstance(info, dict) and info.get("version")
            )
        except (ValueError, AttributeError):
            return
        text = "".join(f"{line}\n" for line in lines)
        target = self._npm_globals_path(conversation)
        try:
            if target.is_symlink():
                return
            if target.exists() and target.read_text(encoding="utf-8") == text:
                return
            temporary = target.with_name(target.name + ".tmp")
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, target)
        except OSError as exc:
            get_logger().warning(
                f"sandbox: could not record installed npm packages: {type(exc).__name__}"
            )
            get_logger().debug(f"sandbox npm record detail: {exc}")

    def _recorded(self, target: Path) -> list[str]:
        try:
            if target.is_symlink() or not target.is_file():
                return []
            return [
                ln.strip() for ln in target.read_text(encoding="utf-8").splitlines() if ln.strip()
            ]
        except OSError:
            return []

    async def _replay_installs(self, container: _Container, conversation: str) -> None:
        pip_lines = self._recorded(self._requirements_path(conversation))
        npm_lines = self._recorded(self._npm_globals_path(conversation))
        count = len(pip_lines) + len(npm_lines)
        if not count:
            return
        if self._options.network == "none":
            self._say(
                f"{count} packages were installed in this conversation, but the "
                "sandbox has no network to restore them."
            )
            return
        self._say(f"Restoring {count} package(s) installed earlier in this conversation…")
        if pip_lines:
            code, _, err = await self._docker(
                "exec", "-w", str(self._workspace.root), container.name, "bash", "-c",
                f"pip install --user --quiet -r {shlex.quote(str(self._requirements_path(conversation)))}",
                timeout=900,
            )  # fmt: skip
            if code != 0:
                self._say(f"Could not restore every Python package: {err.strip()[-160:]}")
        if npm_lines:
            packages = " ".join(shlex.quote(line) for line in npm_lines)
            code, _, err = await self._docker(
                "exec", "-w", str(self._workspace.root), container.name, "bash", "-c",
                f"npm install -g --no-audit --no-fund --loglevel=error {packages}",
                timeout=900,
            )  # fmt: skip
            if code != 0:
                self._say(f"Could not restore every npm package: {err.strip()[-160:]}")


def _container_is_gone(text: str) -> bool:
    lowered = text.lower()
    return "no such container" in lowered or "is not running" in lowered


async def _kill_client(process: asyncio.subprocess.Process) -> None:
    import signal

    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    with contextlib.suppress(ProcessLookupError):
        process.kill()
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(process.wait(), timeout=5)


def _proxy_source() -> str:
    path = Path(__file__).resolve().parents[2] / "scripts" / "egress_proxy.py"
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BackendUnavailableError(f"the sandbox proxy script is missing: {path}") from exc


__all__ = [
    "BASE_IMAGE",
    "DEFAULT_ALLOW",
    "DERIVED_IMAGE",
    "DockerBackend",
    "DockerOptions",
]
