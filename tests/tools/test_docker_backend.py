"""The Docker backend.

Two kinds of test. The first needs no Docker: the options, the allowlist proxy's
decisions, what is mounted and masked, the fingerprint, and the wiring into the
bundle and the CLI config. The second runs a real container and is skipped when
Docker is not available — mocking `docker` would test the mock, and what matters
here (a write reaching the host, a read-only mask holding, a timeout killing the
tree, the network refusing a host) only exists in the real thing.
"""

# No `from __future__ import annotations`: this imports the tool modules, which
# rely on real annotations at construction time.
import ast
import asyncio
import json
import os
import runpy
import shutil
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from domyn.agents.tools.builtin import (
    BackendUnavailableError,
    BuiltinBundleConfig,
    CommandBackend,
    CommandResult,
    DockerBackendOptions,
    LocalBackend,
    Workspace,
    build_backend,
    file_tools,
    shell,
)
from domyn.agents.tools.builtin.backends.docker import (
    REGISTRY_HOSTS,
    DockerBackend,
    egress_proxy,
)
from domyn.agents.tools.builtin.backends.docker.cli import DockerCli, DockerOutput
from domyn.agents.tools.builtin.backends.docker.images import (
    IMAGE_REPOSITORY,
    IMAGE_TAG_ENVIRONMENT_VARIABLE,
    LATEST_TAG,
    default_image,
    ensure_image,
)
from domyn.agents.tools.builtin.backends.docker.installs import (
    NPM_GLOBAL_COMMAND,
    PIP_COMMAND,
)
from domyn.agents.tools.builtin.backends.docker.mounts import build_mounts
from domyn.agents.tools.builtin.backends.docker.network import EgressNetwork
from domyn.agents.tools.builtin.backends.docker.scope import BackendScope, remove_orphans
from domyn.agents.tools.builtin.backends.docker.state import StateFolder


def _docker_works() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return (
            subprocess.run(
                ["docker", "info"], capture_output=True, timeout=15, check=False
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


# The image the real-container tests run in: built from docker/docker-backend.Dockerfile
# (`make docker-backend-image`), or one named by the environment, as CI does.
TEST_IMAGE = os.environ.get("DOMYN_TEST_BACKEND_IMAGE", "domyn-docker-backend:dev")


def _image_is_here() -> bool:
    return (
        subprocess.run(
            ["docker", "image", "inspect", TEST_IMAGE], capture_output=True, check=False
        ).returncode
        == 0
    )


needs_docker = pytest.mark.skipif(
    not (_docker_works() and _image_is_here()),
    reason=f"needs Docker and the image {TEST_IMAGE} (`make docker-backend-image`, or DOMYN_TEST_BACKEND_IMAGE)",
)


# --- no Docker needed -------------------------------------------------------


class TestOptions:
    def test_defaults(self) -> None:
        options = DockerBackendOptions()
        assert options.image == "auto"
        assert options.allowed_network == "public"
        assert options.allowed_hosts() == ["*"]

    def test_unknown_keys_are_refused(self) -> None:
        with pytest.raises(ValueError):
            DockerBackendOptions(netwrok="none")  # type: ignore[call-arg]

    def test_network_modes_are_closed(self) -> None:
        with pytest.raises(ValueError):
            DockerBackendOptions(allowed_network="everything")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            DockerBackendOptions(allowed_network="open")  # type: ignore[arg-type]

    def test_a_list_replaces_public_and_keeps_the_registries(self) -> None:
        options = DockerBackendOptions(
            allowed_network=["GitHub.com", "*.example.com", "github.com"]
        )
        assert options.allowed_network == ["github.com", "*.example.com"]
        assert options.allowed_hosts() == [*REGISTRY_HOSTS, "github.com", "*.example.com"]
        assert "*" not in options.allowed_hosts()

    def test_none_has_no_proxy_hosts(self) -> None:
        assert DockerBackendOptions(allowed_network="none").allowed_hosts() == []

    @pytest.mark.parametrize("entry", ["", "https://x.com", "x.com:443", "a/b", "*", "x y"])
    def test_a_list_entry_must_be_a_host(self, entry: str) -> None:
        with pytest.raises(ValueError):
            DockerBackendOptions(allowed_network=[entry])

    def test_a_single_host_is_not_a_mode(self) -> None:
        with pytest.raises(ValueError):
            DockerBackendOptions(allowed_network="github.com")  # type: ignore[arg-type]


class TestWiring:
    def test_build_backend_by_name(self, tmp_path: Path) -> None:
        backend = build_backend(
            "docker", Workspace.at(tmp_path), docker=DockerBackendOptions(memory="1g")
        )
        assert isinstance(backend, DockerBackend)
        assert backend.is_isolated and backend.destroys_on_close

    def test_unknown_names_still_refuse(self, tmp_path: Path) -> None:
        with pytest.raises(BackendUnavailableError, match="local, docker"):
            build_backend("kubernetes", Workspace.at(tmp_path))

    def test_the_bundle_entry_carries_the_block(self, tmp_path: Path) -> None:
        entry = BuiltinBundleConfig(
            name="x",
            bundles=["files", "shell"],
            workspace=str(tmp_path),
            backend="docker",
            docker={"allowed_network": "none"},  # type: ignore[arg-type]
        )
        built = entry.build()
        assert isinstance(built.backend, DockerBackend)

    def test_the_browser_and_the_files_run_where_the_commands_do(self, tmp_path: Path) -> None:
        entry = BuiltinBundleConfig(
            name="x",
            bundles=["files", "shell", "browser"],
            workspace=str(tmp_path),
            backend="docker",
        )
        built = entry.build()
        assert isinstance(built.backend, DockerBackend)
        assert built.browser is not None
        assert built.browser.backend is built.backend

    def test_a_file_tool_is_the_same_to_the_model_wherever_it_runs(self, tmp_path: Path) -> None:
        def tools(backend: str) -> dict[str, Any]:
            built = BuiltinBundleConfig(
                name="x", bundles=["files"], workspace=str(tmp_path), backend=backend
            ).build()
            return {t.name: t for t in built.tools}

        in_container, here = tools("docker"), tools("local")

        assert in_container.keys() == here.keys()
        for name, tool in in_container.items():
            assert tool.invocation_json_schema == here[name].invocation_json_schema
            assert tool.approval_type == here[name].approval_type

    def test_only_docker_and_auto_have_a_lifecycle_to_manage(self, tmp_path: Path) -> None:
        workspace = Workspace.at(tmp_path)
        assert not isinstance(LocalBackend(workspace), DockerBackend)
        for name in ("docker", "auto"):
            assert isinstance(build_backend(name, workspace), DockerBackend)

    def test_cli_config_round_trips_the_block(self) -> None:
        from domyn.agents.interfaces.cli.config import AgentConfig, _docker_options

        agent = AgentConfig(
            backend="docker", docker=DockerBackendOptions(allowed_network=["example.org"])
        )
        dumped = agent.docker.model_dump(exclude_defaults=True)
        assert dumped == {"allowed_network": ["example.org"]}
        assert _docker_options(dumped).allowed_network == ["example.org"]
        assert AgentConfig(backend="docker").docker.model_dump(exclude_defaults=True) == {}
        # An unreadable block must not stop the wizard from starting.
        assert _docker_options({"nope": 1}) == DockerBackendOptions()


class TestWhatIsMounted:
    def _specs(
        self, tmp_path: Path, roots: tuple[Path, ...] = (), conversation: str = "c1"
    ) -> tuple[Path, list[str]]:
        ws = tmp_path / "ws"
        ws.mkdir(exist_ok=True)
        workspace = Workspace.at(ws, also=roots)
        arguments = build_mounts(workspace, StateFolder.resolve(workspace.root, None), conversation)
        return workspace.root, [a for a in arguments if a != "--mount"]

    def test_workspace_and_what_the_session_writes_are_writable(self, tmp_path: Path) -> None:
        ws, specs = self._specs(tmp_path)
        assert f"type=bind,source={ws},target={ws}" in specs
        assert f"type=bind,source={ws}/.domyn/processes,target={ws}/.domyn/processes" in specs
        assert f"type=bind,source={ws}/.domyn/browser,target={ws}/.domyn/browser" in specs
        # The conversation's own folder is writable, so installs can be recorded.
        assert any(s.endswith("/conversations/c1") for s in specs)

    def test_git_hooks_and_config_are_read_only(self, tmp_path: Path) -> None:
        git = tmp_path / "ws" / ".git"
        (git / "hooks").mkdir(parents=True)
        (git / "config").write_text("[core]\n")
        _, specs = self._specs(tmp_path)
        assert any(s.endswith(f"target={git.resolve()}/hooks,readonly") for s in specs)
        assert any(s.endswith(f"target={git.resolve()}/config,readonly") for s in specs)

    def test_instructions_and_skills_are_writable_as_they_are_on_this_machine(
        self, tmp_path: Path
    ) -> None:
        ws = tmp_path / "ws"
        (ws / ".agents").mkdir(parents=True)
        (ws / ".domyn").mkdir()
        (ws / "AGENTS.md").write_text("x")
        skills = tmp_path / "skills"
        skills.mkdir()
        _, specs = self._specs(tmp_path, roots=(skills,))
        assert not any(s.endswith("AGENTS.md,readonly") for s in specs)
        assert not any(s.endswith(".agents,readonly") for s in specs)
        assert not any(".domyn,readonly" in s for s in specs)
        assert f"type=bind,source={skills.resolve()},target={skills.resolve()}" in specs

    def test_nothing_else_is_mounted(self, tmp_path: Path) -> None:
        _, specs = self._specs(tmp_path)
        home = str(Path.home())
        assert not any(home in s and "/ws" not in s for s in specs)
        assert not any("docker.sock" in s for s in specs)

    def test_a_linked_agent_home_is_refused(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (ws / ".domyn").symlink_to(elsewhere)
        with pytest.raises(BackendUnavailableError, match="is a link"):
            self._specs(tmp_path)


class TestFingerprint:
    def test_changes_when_a_skill_root_is_added_or_replaced(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        one, two = tmp_path / "one", tmp_path / "two"
        one.mkdir()
        two.mkdir()
        before = DockerBackend(Workspace.at(ws, also=(one,)))._fingerprint()
        assert DockerBackend(Workspace.at(ws, also=(one,)))._fingerprint() == before
        assert DockerBackend(Workspace.at(ws, also=(one, two)))._fingerprint() != before

        # `domyn setup` builds a new directory beside the old one and swaps it in: the
        # path is the same and the inode is not, which is what a bind mount follows.
        backend = DockerBackend(Workspace.at(ws, also=(one,)))
        fingerprint = backend._fingerprint()
        one.rename(tmp_path / "old")
        one.mkdir()
        assert backend._fingerprint() != fingerprint

    def test_changes_with_the_network_policy(self, tmp_path: Path) -> None:
        ws = Workspace.at(tmp_path)
        a = DockerBackend(ws, options=DockerBackendOptions(allowed_network="none"))._fingerprint()
        b = DockerBackend(
            ws, options=DockerBackendOptions(allowed_network=["github.com"])
        )._fingerprint()
        assert a != b


class TestRefusals:
    @pytest.mark.asyncio
    async def test_a_home_folder_workspace_is_refused(self) -> None:
        backend = DockerBackend(Workspace.at(Path.home()))
        with pytest.raises(BackendUnavailableError, match="Run domyn from a project folder"):
            await backend.prepare()

    @pytest.mark.asyncio
    async def test_a_missing_docker_binary_says_so(self, tmp_path: Path) -> None:
        backend = DockerBackend(
            Workspace.at(tmp_path),
            options=DockerBackendOptions(docker_binary="/nonexistent/docker"),
        )
        result = await backend.run("echo hi", timeout=5)
        assert result.backend_error and "was not found" in result.backend_error
        assert result.exit_code is None

    @pytest.mark.asyncio
    async def test_a_missing_binary_says_how_to_get_docker(self, tmp_path: Path) -> None:
        backend = DockerBackend(
            Workspace.at(tmp_path),
            options=DockerBackendOptions(docker_binary="/nonexistent/docker"),
        )
        with pytest.raises(BackendUnavailableError) as raised:
            await backend.check()
        assert "was not found" in raised.value.problem
        assert raised.value.fix and "docs.docker.com" in raised.value.fix

    @pytest.mark.parametrize(
        ("system", "expected"),
        [
            ("Darwin", "open -a Docker"),
            ("Linux", "systemctl start docker"),
            ("Windows", "Docker Desktop"),
        ],
    )
    def test_the_start_hint_fits_the_platform(
        self, monkeypatch: pytest.MonkeyPatch, system: str, expected: str
    ) -> None:
        monkeypatch.setattr("platform.system", lambda: system)
        assert expected in DockerCli()._how_to_start()

    def test_another_cli_is_named_in_the_hint(self) -> None:
        assert "podman machine start" in DockerCli("/usr/local/bin/podman")._how_to_start()

    @pytest.mark.asyncio
    async def test_a_daemon_that_is_down_is_told_apart_from_a_missing_docker(
        self, tmp_path: Path
    ) -> None:
        # `false` exists and exits 1 for `info`: installed, not answering.
        backend = DockerBackend(
            Workspace.at(tmp_path),
            options=DockerBackendOptions(docker_binary=shutil.which("false") or "false"),
        )
        with pytest.raises(BackendUnavailableError) as raised:
            await backend.check()
        assert "installed but not running" in raised.value.problem
        assert raised.value.fix


class TestInstallDetection:
    @pytest.mark.parametrize(
        "command",
        [
            "pip install x",
            "pip3 install x",
            "uv pip install x",
            "python -m pip install x",
            "cd a && pip install x",
        ],
    )
    def test_installs(self, command: str) -> None:
        assert PIP_COMMAND.search(command)

    @pytest.mark.parametrize("command", ["ls", "echo pipe", "cat pip.txt", "grep pipeline f"])
    def test_not_installs(self, command: str) -> None:
        assert not PIP_COMMAND.search(command)


class TestEgressProxyRunsInABareImage:
    def test_it_imports_the_standard_library_only(self) -> None:
        """It runs as ``python -c <source>`` in ``python:3.12-slim``, where domyn is not
        installed: one import of anything else and the proxy fails to start."""
        tree = ast.parse(Path(egress_proxy.__file__).read_text(encoding="utf-8"))
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            (node.module or "").split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert imported <= set(sys.stdlib_module_names), imported - set(sys.stdlib_module_names)


class TestEgressProxyDecisions:
    def test_exact_and_wildcard_hosts(self) -> None:
        allow = ["pypi.org", "*.example.com"]
        assert egress_proxy.host_allowed("pypi.org", allow)
        assert egress_proxy.host_allowed("PyPI.org.", allow)
        assert egress_proxy.host_allowed("a.example.com", allow)
        assert not egress_proxy.host_allowed("example.com", allow)
        assert not egress_proxy.host_allowed("evilpypi.org", allow)
        assert not egress_proxy.host_allowed("pypi.org.evil.com", allow)

    def test_star_allows_every_name(self) -> None:
        assert egress_proxy.host_allowed("anything.example", ["*"])
        assert not egress_proxy.host_allowed("anything.example", [])

    @pytest.mark.parametrize(
        "address", ["127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254", "::1", "172.16.0.1"]
    )
    def test_non_public_addresses_are_refused(self, address: str) -> None:
        assert not egress_proxy.public_address(address)

    def test_public_addresses_pass(self) -> None:
        assert egress_proxy.public_address("151.101.0.223")


class TestEgressProxyServer:
    """The real server on a loopback port, asked things that never leave the machine."""

    async def _ask(self, request: bytes, allow: list[str]) -> bytes:
        server = await asyncio.start_server(
            lambda r, w: egress_proxy.handle(r, w, allow), host="127.0.0.1", port=0
        )
        port = server.sockets[0].getsockname()[1]
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(request)
            await writer.drain()
            reply = await asyncio.wait_for(reader.read(4096), 10)
            writer.close()
            return reply
        finally:
            server.close()
            await server.wait_closed()

    @pytest.mark.asyncio
    async def test_a_host_off_the_list_is_refused(self) -> None:
        reply = await self._ask(b"CONNECT example.org:443 HTTP/1.1\r\n\r\n", ["pypi.org"])
        assert reply.startswith(b"HTTP/1.1 403")
        assert b"not in the allowlist" in reply

    @pytest.mark.asyncio
    async def test_public_mode_still_refuses_a_private_address(self) -> None:
        reply = await self._ask(b"CONNECT localhost:443 HTTP/1.1\r\n\r\n", ["*"])
        assert reply.startswith(b"HTTP/1.1 403")
        assert b"non-public" in reply

    @pytest.mark.asyncio
    async def test_public_mode_still_refuses_another_port(self) -> None:
        reply = await self._ask(b"CONNECT example.org:22 HTTP/1.1\r\n\r\n", ["*"])
        assert reply.startswith(b"HTTP/1.1 403")

    @pytest.mark.asyncio
    async def test_plain_http_is_refused(self) -> None:
        reply = await self._ask(b"GET http://pypi.org/ HTTP/1.1\r\n\r\n", ["pypi.org"])
        assert reply.startswith(b"HTTP/1.1 403")

    @pytest.mark.asyncio
    async def test_another_port_is_refused(self) -> None:
        reply = await self._ask(b"CONNECT pypi.org:22 HTTP/1.1\r\n\r\n", ["pypi.org"])
        assert reply.startswith(b"HTTP/1.1 403")
        assert b"port 443" in reply

    @pytest.mark.asyncio
    async def test_an_allowed_name_that_resolves_inward_is_refused(self) -> None:
        # `localhost` on the list is how a proxy becomes a way into its own network.
        reply = await self._ask(b"CONNECT localhost:443 HTTP/1.1\r\n\r\n", ["localhost"])
        assert reply.startswith(b"HTTP/1.1 403")
        assert b"non-public" in reply


# --- a real container -------------------------------------------------------


class TestNpmGlobalDetection:
    @pytest.mark.parametrize(
        "command",
        [
            "npm install -g cowsay",
            "npm i --global cowsay",
            "cd a && npm uninstall -g cowsay",
            "npm install cowsay --location=global",
        ],
    )
    def test_global_changes(self, command: str) -> None:
        assert NPM_GLOBAL_COMMAND.search(command)

    @pytest.mark.parametrize("command", ["npm install cowsay", "echo npm -g", "ls -g", "npm test"])
    def test_the_rest(self, command: str) -> None:
        assert not NPM_GLOBAL_COMMAND.search(command)


Match = Callable[[tuple[str, ...]], bool]


class _ScriptedCli(DockerCli):
    """``docker`` with scripted answers: each call gets the first matching rule's next
    answer (the last one repeats), and every call is recorded. No daemon needed."""

    def __init__(self) -> None:
        super().__init__("docker")
        self.calls: list[tuple[str, ...]] = []
        self.exec_results: list[CommandResult] = []
        self._rules: list[tuple[Match, list[DockerOutput]]] = []

    def when(self, match: Match, *answers: DockerOutput) -> None:
        self._rules.append((match, list(answers)))

    async def __call__(self, *args: str, **_: Any) -> DockerOutput:
        self.calls.append(args)
        for match, answers in self._rules:
            if match(args):
                return answers.pop(0) if len(answers) > 1 else answers[0]
        return DockerOutput(0, "", "")

    async def exec_command(self, *_: Any, **__: Any) -> CommandResult:
        return self.exec_results.pop(0)

    def count(self, match: Match) -> int:
        return sum(1 for call in self.calls if match(call))


def _starts(args: tuple[str, ...]) -> bool:
    return args[:1] == ("run",)


def _starts_the_proxy(args: tuple[str, ...]) -> bool:
    return _starts(args) and "--read-only" in args


def _probes(args: tuple[str, ...]) -> bool:
    return args[:1] == ("exec",) and any("command -v" in a for a in args)


def _ok(stdout: str = "") -> DockerOutput:
    return DockerOutput(0, stdout, "")


def _failed(stderr: str, code: int = 125) -> DockerOutput:
    return DockerOutput(code, "", stderr)


@pytest.fixture
def no_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instantly(_: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", instantly)


@pytest_asyncio.fixture
async def scripted(tmp_path: Path, no_waiting: None) -> tuple[DockerBackend, _ScriptedCli]:
    """A prepared backend over a scripted ``docker``, with no network, so a container
    is the only thing it starts."""
    workspace = tmp_path / "project"
    workspace.mkdir()
    cli = _ScriptedCli()
    backend = DockerBackend(
        Workspace.at(workspace),
        options=DockerBackendOptions(allowed_network="none"),
        cli=cli,
    )
    await backend.prepare()
    return backend, cli


class TestABusyDaemon:
    """Docker fails a start now and then when it is loaded. That must not reach the
    person as "could not start the container"."""

    @pytest.mark.asyncio
    async def test_a_start_that_failed_once_is_tried_again(
        self, scripted: tuple[DockerBackend, _ScriptedCli]
    ) -> None:
        backend, cli = scripted
        cli.when(
            _starts,
            _failed("docker: Error response from daemon: No such container: abc"),
            _ok("abc\n"),
        )
        container = await backend._ensure_container("c1")
        assert container.name == backend.container_name("c1")
        assert cli.count(_starts) == 2

    @pytest.mark.asyncio
    async def test_a_start_that_fails_twice_is_reported(
        self, scripted: tuple[DockerBackend, _ScriptedCli]
    ) -> None:
        backend, cli = scripted
        cli.when(_starts, _failed("docker: Error response from daemon: boom"))
        with pytest.raises(BackendUnavailableError, match="could not start the container"):
            await backend._ensure_container("c1")
        assert cli.count(_starts) == 2

    @pytest.mark.asyncio
    async def test_a_probe_the_daemon_failed_does_not_blame_the_image(
        self, scripted: tuple[DockerBackend, _ScriptedCli]
    ) -> None:
        backend, cli = scripted
        cli.when(
            _probes,
            _failed("Error response from daemon: container abc is not running", code=1),
            _ok(),
        )
        container = await backend._ensure_container("c1")
        assert container.name == backend.container_name("c1")
        assert cli.count(_starts) == 2

    @pytest.mark.asyncio
    async def test_an_image_without_the_tools_is_still_named_and_not_retried(
        self, scripted: tuple[DockerBackend, _ScriptedCli]
    ) -> None:
        backend, cli = scripted
        cli.when(_probes, _failed("", code=1))
        with pytest.raises(BackendUnavailableError, match="lacks bash, timeout or ps"):
            await backend._ensure_container("c1")
        assert cli.count(_starts) == 1

    @pytest.mark.asyncio
    async def test_the_proxy_is_started_again_when_the_first_attempt_fails(
        self, tmp_path: Path, no_waiting: None
    ) -> None:
        cli = _ScriptedCli()
        cli.when(
            _starts_the_proxy,
            _failed("Error response from daemon: No such container: p"),
            _ok("p\n"),
        )
        cli.when(lambda a: a[:1] == ("logs",), _ok("READY\n"))
        # Neither the network nor the proxy exists yet.
        cli.when(lambda a: a[:1] == ("inspect",), _failed("No such object", code=1))
        cli.when(lambda a: a[:2] == ("network", "inspect"), _failed("No such network", code=1))
        network = EgressNetwork(cli, BackendScope.of(tmp_path), ["*"], lambda _: None)

        await network.ensure("img")

        assert cli.count(_starts_the_proxy) == 2

    @pytest.mark.asyncio
    async def test_an_image_that_fails_to_pull_once_is_pulled_again(
        self, tmp_path: Path, no_waiting: None
    ) -> None:
        workspace = tmp_path / "project"
        workspace.mkdir()
        cli = _ScriptedCli()
        cli.when(lambda a: a[:2] == ("image", "inspect"), _failed("No such image", code=1))
        cli.when(
            lambda a: a[:1] == ("pull",),
            _failed("the server supports a different API version", code=1),
            _ok(),
        )
        backend = DockerBackend(Workspace.at(workspace), cli=cli)

        await backend.prepare()

        assert cli.count(lambda a: a[:1] == ("pull",)) == 2
        assert cli.count(lambda a: a[:1] == ("build",)) == 0

    @pytest.mark.asyncio
    async def test_an_image_that_cannot_be_pulled_is_reported_after_one_more_try(
        self, tmp_path: Path, no_waiting: None
    ) -> None:
        workspace = tmp_path / "project"
        workspace.mkdir()
        cli = _ScriptedCli()
        cli.when(lambda a: a[:2] == ("image", "inspect"), _failed("No such image", code=1))
        cli.when(lambda a: a[:1] == ("pull",), _failed("no route to host", code=1))
        backend = DockerBackend(Workspace.at(workspace), cli=cli)

        with pytest.raises(BackendUnavailableError, match=r"could not pull the image .*backend"):
            await backend.prepare()

        assert cli.count(lambda a: a[:1] == ("pull",)) == 2

    @pytest.mark.asyncio
    async def test_a_container_lost_between_commands_says_so(
        self, scripted: tuple[DockerBackend, _ScriptedCli]
    ) -> None:
        backend, cli = scripted
        notices: list[str] = []
        backend.on_notice = notices.append
        cli.exec_results = [
            CommandResult(
                exit_code=None, stdout="", stderr="", backend_error="Error: No such container: x"
            ),
            CommandResult(exit_code=0, stdout="", stderr=""),
        ]

        result = await backend.run("true", timeout=5, conversation_id="c1")

        assert result.exit_code == 0
        assert any("container was lost" in n for n in notices)

    @pytest.mark.asyncio
    async def test_a_container_that_cannot_be_started_is_not_reported_as_lost(
        self, scripted: tuple[DockerBackend, _ScriptedCli]
    ) -> None:
        backend, cli = scripted
        notices: list[str] = []
        backend.on_notice = notices.append
        cli.when(_starts, _failed("Error response from daemon: No such container: abc"))

        result = await backend.run("true", timeout=5, conversation_id="c1")

        assert result.backend_error and "could not start the container" in result.backend_error
        assert not any("container was lost" in n for n in notices)
        assert cli.count(_starts) == 2  # tried again once, not once more as a "lost" container


class TestAStateFolderOutsideTheWorkspace:
    """In the CLI the session's working state is under ``DOMYN_PROJECT_DIR`` (``~/.domyn``
    or a project), not in the workspace. The container gets exactly the two places it
    writes in, at their host paths."""

    def _folders(self, tmp_path: Path) -> tuple[Workspace, Path]:
        ws = tmp_path / "ws"
        ws.mkdir()
        state = tmp_path / "state"
        state.mkdir()
        return Workspace.at(ws), state

    def _specs(self, workspace: Workspace, state: Path) -> list[str]:
        arguments = build_mounts(workspace, StateFolder.resolve(workspace.root, state), "c1")
        return [a for a in arguments if a != "--mount"]

    def test_only_the_processes_and_the_conversations_folders_are_mounted(
        self, tmp_path: Path
    ) -> None:
        workspace, state = self._folders(tmp_path)
        specs = self._specs(workspace, state)
        assert f"type=bind,source={state}/processes,target={state}/processes" in specs
        assert f"type=bind,source={state}/conversations/c1,target={state}/conversations/c1" in specs
        assert not any(f"source={state}," in spec for spec in specs), "the whole folder"
        assert not any(f"source={state}/credentials" in spec for spec in specs)

    def test_nothing_is_made_in_the_workspace_and_no_empty_mask_is_asked_for(
        self, tmp_path: Path
    ) -> None:
        workspace, state = self._folders(tmp_path)
        specs = self._specs(workspace, state)
        assert not (workspace.root / ".domyn").exists()
        assert not any(f"source={workspace.root}/.domyn" in spec for spec in specs)

    def test_the_config_is_there_to_read_and_nothing_else_of_the_home_is(
        self, tmp_path: Path
    ) -> None:
        workspace, state = self._folders(tmp_path)
        for name in ("config.yaml", "credentials", "approvals.yaml", "conversations.db"):
            (state / name).write_text("x")

        specs = self._specs(workspace, state)

        assert f"type=bind,source={state}/config.yaml,target={state}/config.yaml,readonly" in specs
        for private in ("credentials", "approvals.yaml", "conversations.db"):
            assert not any(private in spec for spec in specs)
        assert not any(spec == f"type=bind,source={state},target={state}" for spec in specs)

    def test_a_replaced_config_is_another_mount(self, tmp_path: Path) -> None:
        """A file mount pins the inode, and the host replaces the file when it writes."""
        workspace, state = self._folders(tmp_path)
        config = state / "config.yaml"
        config.write_text("one")
        backend = DockerBackend(workspace, state_dir=state)
        before = backend._fingerprint()

        replacement = state / "config.new"
        replacement.write_text("two")
        replacement.replace(config)

        assert backend._fingerprint() != before

    def test_the_container_is_told_where_it_runs_and_which_project_it_is(
        self, tmp_path: Path
    ) -> None:
        workspace, state = self._folders(tmp_path)
        backend = DockerBackend(workspace, state_dir=state)
        arguments = backend._launcher._run_arguments("n", "f", "img", [])
        assert "DOMYN_BACKEND=docker" in arguments
        assert f"DOMYN_PROJECT_DIR={state}" in arguments

    def test_the_shell_tools_are_told_where_it_is(self, tmp_path: Path) -> None:
        workspace, state = self._folders(tmp_path)
        backend = DockerBackend(workspace, state_dir=state)
        assert backend.state_dir == str(state)
        assert backend.processes_dir == f"{state}/processes"

    def test_without_one_it_is_dot_domyn_in_the_workspace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DOMYN_PROJECT_DIR", raising=False)
        workspace, _ = self._folders(tmp_path)
        state = StateFolder.resolve(workspace.root, None)
        assert state.as_seen_by_commands == ".domyn"
        assert state.conversation_home("c1") == workspace.root / ".domyn/conversations/c1"

    def test_the_environment_names_it_when_none_is_given(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace, _ = self._folders(tmp_path)
        monkeypatch.setenv("DOMYN_PROJECT_DIR", str(tmp_path / "proj"))
        assert DockerBackend(workspace).state_dir == str(tmp_path / "proj")

    def test_another_state_folder_is_another_container(self, tmp_path: Path) -> None:
        workspace, _ = self._folders(tmp_path)
        a = DockerBackend(workspace, state_dir=tmp_path / "a")
        b = DockerBackend(workspace, state_dir=tmp_path / "b")
        assert a._fingerprint() != b._fingerprint()

    def test_a_link_below_it_is_refused_but_the_folder_may_be_one(self, tmp_path: Path) -> None:
        workspace, state = self._folders(tmp_path)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (state / "processes").symlink_to(elsewhere)
        with pytest.raises(BackendUnavailableError, match="is a link"):
            self._specs(workspace, state)

        real = tmp_path / "dotfiles"
        real.mkdir()
        linked = tmp_path / "dotfiles-link"
        linked.symlink_to(real, target_is_directory=True)
        assert self._specs(workspace, linked)


class TestTheToolHostEntry:
    """What a container runs for a file tool: quick, because a call is one process."""

    SCRIPT = Path(__file__).parents[2] / "domyn/agents/tools/builtin/backends/docker/tool_host.py"

    def _run(self, workspace: Path, *flags: str) -> subprocess.CompletedProcess[str]:
        request = json.dumps(
            {
                "operation": "read_file",
                "arguments": {"path": "a.txt"},
                "root": str(workspace),
                "extra_roots": [],
            }
        )
        return subprocess.run(
            [sys.executable, "-P", *flags, str(self.SCRIPT)],
            input=request,
            capture_output=True,
            text=True,
            check=True,
        )

    def test_it_answers_a_request_run_by_its_path(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("hello\n")

        answer = self._run(tmp_path).stdout.strip().splitlines()[-1]

        assert json.loads(answer) == {"ok": True, "result": "hello\n"}

    def test_it_does_not_import_the_agent_machinery(self, tmp_path: Path) -> None:
        """What makes a call quick: the MCP client alone is about half a second."""
        (tmp_path / "a.txt").write_text("hello\n")

        imports = self._run(tmp_path, "-X", "importtime").stderr

        for heavy in ("fastmcp", "domyn.agents.tools.mcp", "domyn.agents.tools.base"):
            assert f"| {heavy}\n" not in imports and f"| {heavy} " not in imports, heavy

    def test_it_is_what_the_container_is_told_to_run(self) -> None:
        from domyn.agents.tools.builtin.backends.docker.backend import OPERATIONS_COMMAND

        assert OPERATIONS_COMMAND.endswith("-P /opt/domyn/tool_host.py")
        dockerfile = (
            Path(__file__).parents[2] / "docker" / "docker-backend.Dockerfile"
        ).read_text()
        assert "/opt/domyn/tool_host.py" in dockerfile


class TestTheImage:
    """Pulled, never built: CI publishes the image of each release."""

    def test_a_release_pulls_the_image_of_its_version(self) -> None:
        assert default_image("1.7.0") == f"{IMAGE_REPOSITORY}:1.7.0"
        assert default_image("1.7.1a2") == f"{IMAGE_REPOSITORY}:1.7.1a2"

    @pytest.mark.parametrize("version", ["1.7.1.dev3+g1a2b3c4", "0.1.dev1", "", "1.7.0+local"])
    def test_a_build_without_an_image_of_its_own_pulls_the_latest_release_image(
        self, version: str
    ) -> None:
        assert default_image(version) == f"{IMAGE_REPOSITORY}:{LATEST_TAG}"

    @pytest.mark.asyncio
    async def test_a_versioned_image_already_here_is_not_pulled_again(self) -> None:
        cli = _ScriptedCli()
        cli.when(lambda a: a[:2] == ("image", "inspect"), _ok())

        image = await ensure_image(cli, f"{IMAGE_REPOSITORY}:1.7.0", lambda _: None)

        assert image.endswith(":1.7.0") and cli.count(lambda a: a[:1] == ("pull",)) == 0

    @pytest.mark.asyncio
    async def test_the_rolling_image_is_pulled_each_time_and_a_copy_is_kept_when_that_fails(
        self,
    ) -> None:
        cli = _ScriptedCli()
        cli.when(lambda a: a[:2] == ("image", "inspect"), _ok())
        cli.when(lambda a: a[:1] == ("pull",), _failed("no route to host", code=1))

        image = await ensure_image(cli, f"{IMAGE_REPOSITORY}:{LATEST_TAG}", lambda _: None)

        assert image.endswith(f":{LATEST_TAG}")
        assert cli.count(lambda a: a[:1] == ("pull",)) == 1

    @pytest.mark.asyncio
    async def test_the_environment_names_a_tag_and_it_is_pulled_each_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """How a maintainer tries an image a manual CI run pushed: no configuration to
        edit, and a tag pushed again is not served from a stale copy."""
        monkeypatch.setenv(IMAGE_TAG_ENVIRONMENT_VARIABLE, "my-test")
        cli = _ScriptedCli()
        cli.when(lambda a: a[:2] == ("image", "inspect"), _ok())
        cli.when(lambda a: a[:1] == ("pull",), _ok())

        image = await ensure_image(cli, "auto", lambda _: None)

        assert image == f"{IMAGE_REPOSITORY}:my-test"
        assert cli.count(lambda a: a[:1] == ("pull",)) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("value", ["ghcr.io/me/domyn-docker-backend:test", "repo/name", "a:b"])
    async def test_the_environment_takes_a_tag_and_not_an_image(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv(IMAGE_TAG_ENVIRONMENT_VARIABLE, value)

        with pytest.raises(BackendUnavailableError, match="is a tag"):
            await ensure_image(_ScriptedCli(), "auto", lambda _: None)

    @pytest.mark.asyncio
    async def test_an_image_in_the_configuration_beats_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(IMAGE_TAG_ENVIRONMENT_VARIABLE, "my-test")
        cli = _ScriptedCli()
        cli.when(lambda a: a[:2] == ("image", "inspect"), _ok())

        assert await ensure_image(cli, "my/own:1", lambda _: None) == "my/own:1"

    def test_ci_reads_the_names_from_the_code_and_does_not_repeat_them(self) -> None:
        root = Path(__file__).parents[2]
        script = root / "scripts" / "backend_image.py"
        namespace = runpy.run_path(str(script), run_name="backend_image")

        assert namespace["read"]() == {
            "IMAGE_REPOSITORY": IMAGE_REPOSITORY,
            "LATEST_TAG": LATEST_TAG,
        }
        workflow = (root / ".github" / "workflows" / "docker-backend-image.yml").read_text()
        assert "ghcr.io/domynai" not in workflow and f":{LATEST_TAG}" not in workflow

    @pytest.mark.asyncio
    async def test_a_named_image_is_used_as_it_is(self) -> None:
        cli = _ScriptedCli()
        cli.when(lambda a: a[:2] == ("image", "inspect"), _ok())

        assert await ensure_image(cli, "my/own:1", lambda _: None) == "my/own:1"


class TestDefaults:
    def test_the_registries_are_always_listed(self) -> None:
        assert {"pypi.org", "registry.npmjs.org"} <= set(REGISTRY_HOSTS)

    def test_the_image_is_made_with_what_a_project_and_the_tools_need(self) -> None:
        dockerfile = (
            Path(__file__).parents[2] / "docker" / "docker-backend.Dockerfile"
        ).read_text()
        for wanted in (
            "git",
            "curl",
            "procps",
            "node",
            "agent-browser",
            "download_chrome",
            "/opt/domyn",
        ):
            assert wanted in dockerfile
        assert "sha256sum -c" in dockerfile  # agent-browser is checked against the lock


@pytest_asyncio.fixture
async def make_docker(tmp_path: Path) -> AsyncIterator[Callable[..., DockerBackend]]:
    """Builds backends over a workspace and removes every container, proxy and
    network they made, whatever the test did."""
    made: list[DockerBackend] = []
    workspace = tmp_path / "project"
    workspace.mkdir()

    def make(
        roots: tuple[Path, ...] = (), state_dir: Path | None = None, **options: Any
    ) -> DockerBackend:
        backend = DockerBackend(
            Workspace.at(workspace, also=roots),
            options=DockerBackendOptions(**{"image": TEST_IMAGE, **options}),
            state_dir=state_dir,
        )
        made.append(backend)
        return backend

    make.workspace = workspace  # type: ignore[attr-defined]
    yield make
    for backend in made:
        await backend.close()


async def _sh(backend: CommandBackend, command: str, conversation: str = "c1", timeout: float = 90):
    return await backend.run(command, timeout=timeout, conversation_id=conversation)


async def _tool(backend: CommandBackend, workspace: Workspace, name: str, **arguments: Any) -> Any:
    """A file tool as the runner calls it: its result, with the conversation filled in."""
    tool = {t.name: t for t in file_tools(workspace, host=backend)}[name]
    return await tool.execute(conversation_id="c1", **arguments)


@needs_docker
class TestToolsInTheContainer:
    """The file and browser tools run where the commands do, in the image."""

    @pytest.mark.asyncio
    async def test_a_file_a_command_made_outside_the_workspace_is_read(
        self, make_docker: Any
    ) -> None:
        backend = make_docker()
        workspace = Workspace.at(make_docker.workspace)
        await _sh(backend, "echo from-the-container > /tmp/downloaded.txt")

        result = await _tool(backend, workspace, "read_file", path="/tmp/downloaded.txt")

        assert result.success and "from-the-container" in result.data

    @pytest.mark.asyncio
    async def test_a_write_reaches_the_host_and_is_seen_by_commands(self, make_docker: Any) -> None:
        backend = make_docker()
        workspace = Workspace.at(make_docker.workspace)

        result = await _tool(
            backend, workspace, "write_file", path="made.txt", content="by-the-tool"
        )

        assert result.success, result.error
        assert (make_docker.workspace / "made.txt").read_text() == "by-the-tool"
        assert (await _sh(backend, "cat made.txt")).stdout.strip() == "by-the-tool"

    @pytest.mark.asyncio
    async def test_a_tool_and_a_command_can_both_edit_instructions(self, make_docker: Any) -> None:
        (make_docker.workspace / "AGENTS.md").write_text("rules\n")
        backend = make_docker()
        workspace = Workspace.at(make_docker.workspace)

        written = await _sh(backend, "echo more >> AGENTS.md")
        appended = await _tool(
            backend, workspace, "append_file", path="AGENTS.md", content="and more\n"
        )

        assert written.exit_code == 0 and appended.success, appended.error
        assert (make_docker.workspace / "AGENTS.md").read_text() == "rules\nmore\nand more\n"

    @pytest.mark.asyncio
    async def test_what_is_not_mounted_is_not_there(self, make_docker: Any, tmp_path: Path) -> None:
        """Mounts are the boundary: a file of the host's outside them does not exist."""
        secret = tmp_path / "elsewhere" / "secret.txt"
        secret.parent.mkdir()
        secret.write_text("not for the container")
        backend = make_docker()
        workspace = Workspace.at(make_docker.workspace)

        result = await _tool(backend, workspace, "read_file", path=str(secret))

        assert not result.success and "does not exist" in result.error

    @pytest.mark.asyncio
    async def test_the_whole_container_is_reachable(self, make_docker: Any) -> None:
        backend = make_docker()
        workspace = Workspace.at(make_docker.workspace)

        result = await _tool(backend, workspace, "read_file", path="/etc/os-release")

        assert result.success and "Debian" in result.data

    @pytest.mark.asyncio
    async def test_an_agent_can_edit_its_own_skill(self, make_docker: Any, tmp_path: Path) -> None:
        skills = tmp_path / "skills" / "mine"
        skills.mkdir(parents=True)
        (skills / "SKILL.md").write_text("old\n")
        backend = make_docker(roots=(skills,))
        workspace = Workspace.at(make_docker.workspace, also=(skills,))

        result = await _tool(
            backend, workspace, "write_file", path=str(skills / "SKILL.md"), content="new\n"
        )

        assert result.success, result.error
        assert (skills / "SKILL.md").read_text() == "new\n"

    @pytest.mark.asyncio
    async def test_the_browser_opens_a_page_with_the_images_chrome(self, make_docker: Any) -> None:
        built = BuiltinBundleConfig(
            name="x",
            bundles=["shell", "browser"],
            workspace=str(make_docker.workspace),
            backend="docker",
            docker=DockerBackendOptions(image=TEST_IMAGE),
            browser={"allow_private_urls": True},  # type: ignore[arg-type]
        ).build()
        try:
            await _sh(
                built.backend,
                "echo '<h1>hello</h1>' > index.html"
                " && (nohup python -m http.server 8765 --bind 127.0.0.1 >/dev/null 2>&1 &)"
                " && sleep 1",
            )
            opened = {t.name: t for t in built.tools}["browser_open"]

            result = await opened.execute(url="http://127.0.0.1:8765/", conversation_id="c1")

            assert result.success, result.error
            assert 'heading "hello"' in result.data
        finally:
            await built.close()


@needs_docker
class TestARealContainer:
    @pytest.mark.asyncio
    async def test_a_write_in_the_workspace_reaches_the_host(self, make_docker: Any) -> None:
        backend = make_docker()
        result = await _sh(backend, "echo from-container > made.txt && pwd")
        assert result.exit_code == 0 and result.backend_error is None
        assert result.stdout.strip() == str(make_docker.workspace)
        assert (make_docker.workspace / "made.txt").read_text().strip() == "from-container"
        # Owned by the user who runs domyn, not by root.
        assert (make_docker.workspace / "made.txt").stat().st_uid == os.getuid()

    @pytest.mark.asyncio
    async def test_the_host_outside_the_workspace_is_not_there(
        self, make_docker: Any, tmp_path: Path
    ) -> None:
        secret = tmp_path / "secret.txt"
        secret.write_text("nope")
        backend = make_docker()
        result = await _sh(
            backend, f"cat {secret}; ls {Path.home()} 2>&1; ls /var/run/docker.sock 2>&1"
        )
        assert "nope" not in result.stdout
        assert "No such file" in result.stderr + result.stdout

    @pytest.mark.asyncio
    async def test_no_host_environment_leaks(
        self, make_docker: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DOMYN_TEST_SECRET", "hunter2")
        result = await _sh(make_docker(), "env")
        assert "hunter2" not in result.stdout

    @pytest.mark.asyncio
    async def test_an_allowed_variable_is_passed(
        self, make_docker: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DOMYN_TEST_ALLOWED", "yes-please")
        backend = DockerBackend(
            Workspace.at(make_docker.workspace),
            options=DockerBackendOptions(image=TEST_IMAGE),
            allowed_env=["DOMYN_TEST_ALLOWED"],
        )
        try:
            result = await _sh(backend, "echo $DOMYN_TEST_ALLOWED")
            assert result.stdout.strip() == "yes-please"
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_only_what_runs_code_on_the_host_is_read_only(self, make_docker: Any) -> None:
        git = make_docker.workspace / ".git"
        (git / "hooks").mkdir(parents=True)
        (git / "config").write_text("[core]\n")
        backend = make_docker()
        for command in ("touch .git/hooks/pre-commit", "echo x >> .git/config"):
            result = await _sh(backend, command)
            assert result.exit_code != 0, command
            assert "Read-only" in result.stderr, command
        # What a command may write on this machine it may write here.
        result = await _sh(
            backend, "mkdir -p .domyn/skills .agents && touch .domyn/x .agents/y AGENTS.md SOUL.md"
        )
        assert result.exit_code == 0, result.stderr

    @pytest.mark.asyncio
    async def test_the_config_can_be_read_and_not_written(
        self, make_docker: Any, tmp_path: Path
    ) -> None:
        state = tmp_path / "home"
        state.mkdir()
        (state / "config.yaml").write_text("backend: docker\n")
        (state / "credentials").write_text("KEY=secret\n")
        backend = make_docker(state_dir=state)

        read = await _sh(backend, f"cat {state}/config.yaml && echo $DOMYN_PROJECT_DIR")
        written = await _sh(backend, f"echo 'backend: local' > {state}/config.yaml")
        creds = await _sh(backend, f"cat {state}/credentials")

        assert "backend: docker" in read.stdout and str(state) in read.stdout
        assert written.exit_code != 0 and "Read-only" in written.stderr
        assert creds.exit_code != 0 and "secret" not in creds.stdout
        assert (state / "config.yaml").read_text() == "backend: docker\n"

    @pytest.mark.asyncio
    async def test_state_lives_across_calls(self, make_docker: Any) -> None:
        backend = make_docker()
        await _sh(backend, "echo kept > $HOME/note")
        result = await _sh(backend, "cat $HOME/note")
        assert result.stdout.strip() == "kept"

    @pytest.mark.asyncio
    async def test_a_timeout_kills_the_whole_tree(self, make_docker: Any) -> None:
        backend = make_docker()
        started = asyncio.get_running_loop().time()
        result = await _sh(backend, "sleep 300 & sleep 300 & wait", timeout=3)
        assert result.timed_out
        assert asyncio.get_running_loop().time() - started < 30
        leftover = await _sh(backend, "ps -eo args | grep -c '[s]leep 300'")
        assert leftover.stdout.strip() == "0"

    @pytest.mark.asyncio
    async def test_stdin_and_exit_codes(self, make_docker: Any) -> None:
        backend = make_docker()
        result = await backend.run("cat; exit 3", timeout=20, conversation_id="c1", stdin="typed\n")
        assert result.exit_code == 3 and result.stdout == "typed\n"

    @pytest.mark.asyncio
    async def test_the_network_is_registries_only(self, make_docker: Any) -> None:
        backend = make_docker()
        probe = (
            "python - <<'P'\n"
            "import urllib.request\n"
            "for url in ('https://example.org', 'https://1.1.1.1'):\n"
            "    try:\n"
            "        urllib.request.urlopen(url, timeout=10)\n"
            "        print('REACHED', url)\n"
            "    except Exception as exc:\n"
            "        print('BLOCKED', url, type(exc).__name__)\n"
            "P"
        )
        result = await _sh(backend, probe)
        assert "REACHED" not in result.stdout, result.stdout
        assert result.stdout.count("BLOCKED") == 2
        # And a client that ignores the proxy settings has no route at all.
        direct = await _sh(
            backend,
            "env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY "
            "python -c \"import socket; socket.create_connection(('1.1.1.1', 443), timeout=5)\"",
        )
        assert direct.exit_code != 0

    @pytest.mark.asyncio
    async def test_pip_install_works_through_the_allowlist(self, make_docker: Any) -> None:
        backend = make_docker()
        install = await _sh(backend, "pip install --quiet cowsay")
        assert install.exit_code == 0, install.stderr
        result = await _sh(backend, "python -c 'import cowsay; print(cowsay.__name__)'")
        assert result.stdout.strip() == "cowsay"
        recorded = backend._installs.requirements_path("c1").read_text()
        assert "cowsay" in recorded

    @pytest.mark.asyncio
    async def test_the_image_has_git_curl_node_and_npm(self, make_docker: Any) -> None:
        backend = make_docker()
        result = await _sh(
            backend, "git --version && curl --version | head -1 && node -v && npm -v"
        )
        assert result.exit_code == 0, result.stderr

    @pytest.mark.asyncio
    async def test_global_npm_packages_are_recorded_and_restored(self, make_docker: Any) -> None:
        backend = make_docker()
        install = await _sh(backend, "npm install -g --no-audit --no-fund cowsay", timeout=180)
        assert install.exit_code == 0, install.stderr
        assert "cowsay@" in backend._installs.npm_globals_path("c1").read_text()
        assert (await _sh(backend, "cowsay hi")).exit_code == 0
        # A new container for the same conversation brings them back.
        await backend.end_conversation("c1")
        again = await _sh(backend, "cowsay hi", timeout=180)
        assert again.exit_code == 0, again.stderr

    @pytest.mark.asyncio
    async def test_a_local_npm_install_lands_in_the_workspace(self, make_docker: Any) -> None:
        backend = make_docker()
        install = await _sh(backend, "npm install --no-audit --no-fund cowsay", timeout=180)
        assert install.exit_code == 0, install.stderr
        assert (make_docker.workspace / "node_modules" / "cowsay").is_dir()

    @pytest.mark.asyncio
    async def test_public_reaches_any_public_host_but_not_a_private_address(
        self, make_docker: Any
    ) -> None:
        backend = make_docker()
        ok = await _sh(backend, "curl -sS -o /dev/null -w '%{http_code}' https://example.org")
        assert ok.exit_code == 0 and ok.stdout.strip() == "200", ok.stderr
        began = time.time()
        private = await _sh(backend, "curl -sS https://10.0.0.1/")
        assert private.exit_code != 0
        assert "10.0.0.1" in await backend.explain_failure(began)

    @pytest.mark.asyncio
    async def test_a_list_reaches_only_those_and_the_registries(self, make_docker: Any) -> None:
        backend = make_docker(allowed_network=["github.com"])
        ok = await _sh(backend, "curl -sS -o /dev/null -w '%{http_code}' https://github.com")
        assert ok.exit_code == 0 and ok.stdout.strip() in {"200", "301"}, ok.stderr
        registry = await _sh(
            backend, "curl -sS -o /dev/null -w '%{http_code}' https://pypi.org/simple/"
        )
        assert registry.stdout.strip() == "200", registry.stderr
        began = time.time()
        blocked = await _sh(backend, "curl -sS https://example.org")
        assert blocked.exit_code != 0
        note = await backend.explain_failure(began)
        assert "example.org" in note and "docker: allowed_network:" in note

    @pytest.mark.asyncio
    async def test_no_network_means_none(self, make_docker: Any) -> None:
        backend = make_docker(allowed_network="none")
        result = await _sh(
            backend,
            "python -c \"import socket; socket.create_connection(('1.1.1.1', 443), timeout=5)\"",
        )
        assert result.exit_code != 0


@needs_docker
class TestTheContainersLife:
    @pytest.mark.asyncio
    async def test_a_conversations_end_destroys_its_container_and_a_resume_restores_installs(
        self, make_docker: Any
    ) -> None:
        backend = make_docker()
        await _sh(backend, "pip install --quiet cowsay")
        name = backend.container_name("c1")
        assert await _exists(name)

        await backend.end_conversation("c1")
        assert not await _exists(name)

        notices: list[str] = []
        backend.on_notice = notices.append
        result = await _sh(backend, "python -c 'import cowsay; print(\"back\")'")
        assert result.stdout.strip() == "back"
        assert any("Restoring" in n for n in notices)

    @pytest.mark.asyncio
    async def test_close_removes_containers_proxy_and_network(self, make_docker: Any) -> None:
        backend = make_docker()
        await _sh(backend, "true", conversation="a")
        await _sh(backend, "true", conversation="b")
        names = [
            backend.container_name("a"),
            backend.container_name("b"),
            backend._scope.proxy_name,
        ]
        assert all([await _exists(n) for n in names])
        await backend.close()
        assert not any([await _exists(n) for n in names])
        assert not await _exists(backend._scope.network_name, kind="network")

    @pytest.mark.asyncio
    async def test_a_rebuilt_backend_adopts_the_running_container(self, make_docker: Any) -> None:
        first = make_docker()
        await _sh(first, "echo $$ > $HOME/pid-marker")
        # A config reload builds a new backend over the same workspace, in this process.
        second = make_docker()
        notices: list[str] = []
        second.on_notice = notices.append
        result = await _sh(second, "cat $HOME/pid-marker")
        assert result.exit_code == 0 and result.stdout.strip().isdigit()
        assert not any("Starting" in n for n in notices)

    @pytest.mark.asyncio
    async def test_a_container_removed_underneath_is_recreated(self, make_docker: Any) -> None:
        backend = make_docker()
        await _sh(backend, "true")
        subprocess.run(["docker", "rm", "-f", backend.container_name("c1")], capture_output=True)
        result = await _sh(backend, "echo again")
        assert result.stdout.strip() == "again"

    @pytest.mark.asyncio
    async def test_what_a_dead_process_left_is_removed(self, make_docker: Any) -> None:
        backend = make_docker()
        await backend.prepare()
        gone = subprocess.Popen(["true"])
        gone.wait()
        stale = f"domyn-stale-{gone.pid}"
        started = subprocess.run(
            [
                "docker", "run", "-d", "--name", stale,
                "--label", "domyn.backend=1", "--label", f"domyn.owner-pid={gone.pid}",
                TEST_IMAGE, "sleep", "60",
            ],
            capture_output=True,
        )  # fmt: skip
        # Any other process running the backend reaps a dead owner's containers too, so
        # when the suite runs in parallel it may be gone before this one looks, or even
        # while `docker run` is still starting it. Either way it ends up removed, which
        # is all this test is about.
        assert started.returncode == 0 or "No such container" in started.stderr.decode()
        await remove_orphans(DockerCli())
        assert not await _exists(stale)

    @pytest.mark.asyncio
    async def test_ending_a_conversation_whose_container_is_starting_leaves_nothing(
        self, make_docker: Any
    ) -> None:
        backend = make_docker()
        warming = asyncio.ensure_future(backend.warm_up("c1"))
        while not backend._lock_for("c1").locked():
            await asyncio.sleep(0.01)
        await backend.end_conversation("c1")
        await warming
        assert not await _exists(backend.container_name("c1"))

    @pytest.mark.asyncio
    async def test_closing_while_a_container_is_starting_leaves_nothing(
        self, make_docker: Any
    ) -> None:
        backend = make_docker()
        warming = asyncio.ensure_future(backend.warm_up("c1"))
        while not backend._lock_for("c1").locked():
            await asyncio.sleep(0.01)
        await backend.close()
        await warming
        assert not await _exists(backend.container_name("c1"))
        assert not await _exists(backend._scope.proxy_name)

    @pytest.mark.asyncio
    async def test_a_live_owners_containers_are_left_alone(self, make_docker: Any) -> None:
        backend = make_docker()
        await _sh(backend, "true")
        await remove_orphans(DockerCli())
        assert await _exists(backend.container_name("c1"))


@needs_docker
class TestSkillsReachTheContainer:
    @pytest.mark.asyncio
    async def test_an_edit_on_the_host_is_seen_without_a_restart(
        self, make_docker: Any, tmp_path: Path
    ) -> None:
        skills = tmp_path / "skills"
        (skills / "demo").mkdir(parents=True)
        script = skills / "demo" / "run.py"
        script.write_text("print('v1')")
        backend = make_docker(roots=(skills,))
        assert (await _sh(backend, f"python {script}")).stdout.strip() == "v1"
        script.write_text("print('v2')")
        (skills / "demo" / "new.py").write_text("print('new')")
        assert (await _sh(backend, f"python {script}")).stdout.strip() == "v2"
        assert (await _sh(backend, f"python {skills}/demo/new.py")).stdout.strip() == "new"

    @pytest.mark.asyncio
    async def test_skills_are_writable_as_they_are_on_this_machine(
        self, make_docker: Any, tmp_path: Path
    ) -> None:
        skills = tmp_path / "skills"
        skills.mkdir()
        backend = make_docker(roots=(skills,))
        result = await _sh(backend, f"touch {skills}/mine")
        assert result.exit_code == 0 and (skills / "mine").exists()

    @pytest.mark.asyncio
    async def test_a_new_skill_root_recreates_the_container_and_keeps_installs(
        self, make_docker: Any, tmp_path: Path
    ) -> None:
        one, two = tmp_path / "one", tmp_path / "two"
        for folder in (one, two):
            folder.mkdir()
        (two / "tool.py").write_text("print('two')")
        first = make_docker(roots=(one,))
        await _sh(first, "pip install --quiet cowsay")
        # The config changed and the agent was rebuilt with another skill folder.
        second = make_docker(roots=(one, two))
        result = await _sh(second, f"python {two}/tool.py && python -c 'import cowsay'")
        assert result.exit_code == 0, result.stderr
        assert result.stdout.strip() == "two"

    @pytest.mark.asyncio
    async def test_a_swapped_install_directory_is_picked_up(
        self, make_docker: Any, tmp_path: Path
    ) -> None:
        skills = tmp_path / "skills"
        skills.mkdir()
        (skills / "a.py").write_text("print('old')")
        backend = make_docker(roots=(skills,))
        assert (await _sh(backend, f"python {skills}/a.py")).stdout.strip() == "old"
        # What `domyn setup` does: build beside, swap in.
        skills.rename(tmp_path / "skills-old")
        skills.mkdir()
        (skills / "a.py").write_text("print('new')")
        assert (await _sh(backend, f"python {skills}/a.py")).stdout.strip() == "new"

    @pytest.mark.asyncio
    async def test_a_recreate_waits_for_a_running_command(
        self, make_docker: Any, tmp_path: Path
    ) -> None:
        one, two = tmp_path / "one", tmp_path / "two"
        one.mkdir()
        two.mkdir()
        first = make_docker(roots=(one,))
        run = shell.build_run_command(first)
        state: dict[str, Any] = {}
        await run(command="sleep 120", background=True, conversation_id="c1", state=state)

        second = make_docker(roots=(one, two))
        notices: list[str] = []
        second.on_notice = notices.append
        result = await _sh(second, "ps -eo args | grep -c '[s]leep 120'")
        assert result.stdout.strip() == "1", "the background command was killed"
        assert any("background command is running" in n for n in notices)


@needs_docker
class TestAStateFolderOutsideTheWorkspaceInAContainer:
    @pytest.mark.asyncio
    async def test_the_two_folders_are_writable_and_nothing_around_them_is(
        self, make_docker: Any, tmp_path: Path
    ) -> None:
        state = tmp_path / "state"
        state.mkdir()
        backend = make_docker(state_dir=state)
        ok = await _sh(
            backend, f"echo a > {state}/processes/x && echo b > {state}/conversations/c1/y"
        )
        assert ok.exit_code == 0, ok.stderr
        assert (state / "processes" / "x").read_text().strip() == "a"
        assert (state / "conversations" / "c1" / "y").read_text().strip() == "b"
        beside = await _sh(backend, f"echo no > {state}/elsewhere")
        assert beside.exit_code != 0
        assert not (state / "elsewhere").exists()
        assert not (make_docker.workspace / ".domyn").exists()

    @pytest.mark.asyncio
    async def test_installs_are_recorded_there_and_restored(
        self, make_docker: Any, tmp_path: Path
    ) -> None:
        state = tmp_path / "state"
        state.mkdir()
        backend = make_docker(state_dir=state)
        run = shell.build_run_command(backend)
        await run(command="pip install cowsay", conversation_id="c1", state={})
        assert "cowsay" in (state / "conversations/c1/requirements.txt").read_text()
        await backend.end_conversation("c1")
        again = await run(
            command="python -c 'import cowsay; print(\"py-ok\")'", conversation_id="c1", state={}
        )
        assert "py-ok" in str(again.data), str(again.data)

    @pytest.mark.asyncio
    async def test_background_commands_run_check_and_stop(
        self, make_docker: Any, tmp_path: Path
    ) -> None:
        state = tmp_path / "state"
        state.mkdir()
        backend = make_docker(state_dir=state)
        run = shell.build_run_command(backend)
        check = shell.build_check_command(backend)
        stop = shell.build_stop_command(backend)
        bag: dict[str, Any] = {}
        started = await run(
            command="for i in 1 2 3 4 5 6; do echo tick $i; sleep 1; done",
            background=True,
            conversation_id="c1",
            state=bag,
        )
        assert "still running" in str(started.data), str(started.data)
        handle = next(iter(bag["background_processes"]))
        assert any((state / "processes").rglob("*")), "its folder is in the state folder"
        await asyncio.sleep(2)
        last = None
        async for event in check(handle=handle, conversation_id="c1", state=bag):
            last = event
        assert "tick" in str(last.data)
        assert "Stopped" in str((await stop(handle=handle, conversation_id="c1", state=bag)).data)


@needs_docker
class TestShellToolsInTheContainer:
    """Through ``run_command`` itself: the backend is handed a launcher script with the
    model's command quoted inside it, not the command, and exit codes are the script's."""

    @pytest.mark.asyncio
    async def test_a_failure_names_the_blocked_host(self, make_docker: Any) -> None:
        backend = make_docker(allowed_network=["github.com"])
        run = shell.build_run_command(backend)
        result = await run(command="curl -sS https://example.org", conversation_id="c1", state={})
        text = str(result.data)
        assert "[docker] blocked" in text and "example.org" in text
        assert "docker: allowed_network:" in text

    @pytest.mark.asyncio
    async def test_a_successful_command_gets_no_note(self, make_docker: Any) -> None:
        backend = make_docker()
        run = shell.build_run_command(backend)
        result = await run(command="echo fine", conversation_id="c1", state={})
        assert "[docker]" not in str(result.data)

    @pytest.mark.asyncio
    async def test_installs_are_recorded_and_come_back(self, make_docker: Any) -> None:
        backend = make_docker()
        run = shell.build_run_command(backend)
        state: dict[str, Any] = {}
        pip = await run(command="pip install cowsay", conversation_id="c1", state=state)
        assert "Successfully installed" in str(pip.data)
        assert "cowsay" in backend._installs.requirements_path("c1").read_text()
        npm = await run(
            command="npm install -g --no-audit --no-fund is-odd", conversation_id="c1", state=state
        )
        assert "is-odd@" in backend._installs.npm_globals_path("c1").read_text(), str(npm.data)
        # The same conversation in a new container: both are back.
        await backend.end_conversation("c1")
        again = await run(
            command="python -c 'import cowsay; print(\"py-ok\")' && npm ls -g --depth=0",
            conversation_id="c1",
            state=state,
        )
        assert "py-ok" in str(again.data) and "is-odd@" in str(again.data), str(again.data)

    @pytest.mark.asyncio
    async def test_background_commands_run_check_and_stop(self, make_docker: Any) -> None:
        backend = make_docker()
        run = shell.build_run_command(backend)
        check = shell.build_check_command(backend)
        stop = shell.build_stop_command(backend)
        state: dict[str, Any] = {}

        started = await run(
            command="for i in 1 2 3 4 5 6; do echo tick $i; sleep 1; done",
            background=True,
            conversation_id="c1",
            state=state,
        )
        assert "still running" in str(started.data)
        handle = next(iter(state["background_processes"]))
        await asyncio.sleep(2)
        last = None
        async for event in check(handle=handle, conversation_id="c1", state=state):
            last = event
        assert "tick" in str(last.data)
        stopped = await stop(handle=handle, conversation_id="c1", state=state)
        assert "Stopped" in str(stopped.data)


async def _exists(name: str, kind: str = "container") -> bool:
    argv = (
        ["docker", "network", "inspect", name] if kind == "network" else ["docker", "inspect", name]
    )
    process = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
    )
    return await process.wait() == 0
