"""The Docker sandbox backend.

Two kinds of test. The first needs no Docker: the options, the allowlist proxy's
decisions, what is mounted and masked, the fingerprint, and the wiring into the
bundle and the CLI config. The second runs a real container and is skipped when
Docker is not available — mocking `docker` would test the mock, and what matters
here (a write reaching the host, a read-only mask holding, a timeout killing the
tree, the network refusing a host) only exists in the real thing.
"""

# No `from __future__ import annotations`: this imports the tool modules, which
# rely on real annotations at construction time.
import asyncio
import os
import shutil
import subprocess
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from domyn.agents.scripts import egress_proxy
from domyn.agents.tools.builtin import (
    BackendUnavailableError,
    BuiltinBundleConfig,
    CommandBackend,
    CommandResult,
    DockerOptions,
    LocalBackend,
    Workspace,
    build_backend,
    shell,
)
from domyn.agents.tools.builtin.docker_backend import (
    _DOCKERFILE,
    _INSTALLS,
    _NPM_GLOBAL,
    REGISTRY_HOSTS,
    DockerBackend,
)


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


needs_docker = pytest.mark.skipif(not _docker_works(), reason="Docker is not available")


# --- no Docker needed -------------------------------------------------------


class TestOptions:
    def test_defaults(self) -> None:
        options = DockerOptions()
        assert options.image == "auto"
        assert options.allowed_network == "public"
        assert options.allowed_hosts() == ["*"]

    def test_unknown_keys_are_refused(self) -> None:
        with pytest.raises(ValueError):
            DockerOptions(netwrok="none")  # type: ignore[call-arg]

    def test_network_modes_are_closed(self) -> None:
        with pytest.raises(ValueError):
            DockerOptions(allowed_network="everything")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            DockerOptions(allowed_network="open")  # type: ignore[arg-type]

    def test_a_list_replaces_public_and_keeps_the_registries(self) -> None:
        options = DockerOptions(allowed_network=["GitHub.com", "*.example.com", "github.com"])
        assert options.allowed_network == ["github.com", "*.example.com"]
        assert options.allowed_hosts() == [*REGISTRY_HOSTS, "github.com", "*.example.com"]
        assert "*" not in options.allowed_hosts()

    def test_none_has_no_proxy_hosts(self) -> None:
        assert DockerOptions(allowed_network="none").allowed_hosts() == []

    @pytest.mark.parametrize("entry", ["", "https://x.com", "x.com:443", "a/b", "*", "x y"])
    def test_a_list_entry_must_be_a_host(self, entry: str) -> None:
        with pytest.raises(ValueError):
            DockerOptions(allowed_network=[entry])

    def test_a_single_host_is_not_a_mode(self) -> None:
        with pytest.raises(ValueError):
            DockerOptions(allowed_network="github.com")  # type: ignore[arg-type]


class TestWiring:
    def test_build_backend_by_name(self, tmp_path: Path) -> None:
        backend = build_backend("docker", Workspace.at(tmp_path), docker=DockerOptions(memory="1g"))
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
            sandbox="docker",
            docker={"allowed_network": "none"},  # type: ignore[arg-type]
        )
        built = entry.build()
        assert isinstance(built.backend, DockerBackend)

    def test_the_browser_stays_on_this_machine(self, tmp_path: Path) -> None:
        entry = BuiltinBundleConfig(
            name="x", bundles=["shell", "browser"], workspace=str(tmp_path), sandbox="docker"
        )
        built = entry.build()
        assert isinstance(built.backend, DockerBackend)
        assert built.browser is not None
        assert isinstance(built.browser.backend, LocalBackend)

    def test_other_backends_have_the_new_hooks_as_no_ops(self, tmp_path: Path) -> None:
        local = LocalBackend(Workspace.at(tmp_path))
        assert local.destroys_on_close is False

        async def go() -> None:
            await local.prepare()
            await local.warm_up("c")
            await local.end_conversation("c")

        asyncio.run(go())

    def test_cli_config_round_trips_the_block(self) -> None:
        from domyn.agents.interfaces.cli.config import AgentConfig, _docker_options

        agent = AgentConfig(sandbox="docker", docker=DockerOptions(allowed_network=["example.org"]))
        dumped = agent.docker.model_dump(exclude_defaults=True)
        assert dumped == {"allowed_network": ["example.org"]}
        assert _docker_options(dumped).allowed_network == ["example.org"]
        assert AgentConfig(sandbox="docker").docker.model_dump(exclude_defaults=True) == {}
        # An unreadable block must not stop the wizard from starting.
        assert _docker_options({"nope": 1}) == DockerOptions()


class TestWhatIsMounted:
    def _backend(self, tmp_path: Path, roots: tuple[Path, ...] = ()) -> DockerBackend:
        ws = tmp_path / "ws"
        ws.mkdir(exist_ok=True)
        return DockerBackend(Workspace.at(ws, also=roots))

    def _specs(self, backend: DockerBackend, conversation: str = "c1") -> list[str]:
        mounts = backend._mounts(conversation)
        return [m for m in mounts if m != "--mount"]

    def test_workspace_is_writable_and_the_agents_home_is_not(self, tmp_path: Path) -> None:
        backend = self._backend(tmp_path)
        ws = backend._workspace.root
        specs = self._specs(backend)
        assert f"type=bind,source={ws},target={ws}" in specs
        assert f"type=bind,source={ws}/.domyn,target={ws}/.domyn,readonly" in specs
        assert f"type=bind,source={ws}/.domyn/processes,target={ws}/.domyn/processes" in specs
        # The conversation's own folder is writable, so installs can be recorded.
        assert any(s.endswith("/conversations/c1") for s in specs)

    def test_git_hooks_and_config_are_read_only(self, tmp_path: Path) -> None:
        backend = self._backend(tmp_path)
        git = backend._workspace.root / ".git"
        (git / "hooks").mkdir(parents=True)
        (git / "config").write_text("[core]\n")
        specs = self._specs(backend)
        assert any(s.endswith(f"target={git}/hooks,readonly") for s in specs)
        assert any(s.endswith(f"target={git}/config,readonly") for s in specs)

    def test_instruction_files_and_agents_folder_are_masked_when_present(
        self, tmp_path: Path
    ) -> None:
        backend = self._backend(tmp_path)
        ws = backend._workspace.root
        (ws / ".agents").mkdir()
        (ws / "AGENTS.md").write_text("x")
        specs = self._specs(backend)
        assert any(s.endswith(f"target={ws}/.agents,readonly") for s in specs)
        assert any(s.endswith(f"target={ws}/AGENTS.md,readonly") for s in specs)

    def test_skill_roots_are_read_only(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        skills.mkdir()
        backend = self._backend(tmp_path, roots=(skills,))
        specs = self._specs(backend)
        assert f"type=bind,source={skills.resolve()},target={skills.resolve()},readonly" in specs

    def test_nothing_else_is_mounted(self, tmp_path: Path) -> None:
        specs = self._specs(self._backend(tmp_path))
        home = str(Path.home())
        assert not any(home in s and "/ws" not in s for s in specs)
        assert not any("docker.sock" in s for s in specs)

    def test_a_linked_agent_home_is_refused(self, tmp_path: Path) -> None:
        backend = self._backend(tmp_path)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (backend._workspace.root / ".domyn").symlink_to(elsewhere)
        with pytest.raises(BackendUnavailableError, match="is a link"):
            backend._mounts("c1")


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
        a = DockerBackend(ws, options=DockerOptions(allowed_network="none"))._fingerprint()
        b = DockerBackend(ws, options=DockerOptions(allowed_network=["github.com"]))._fingerprint()
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
            Workspace.at(tmp_path), options=DockerOptions(docker_binary="/nonexistent/docker")
        )
        result = await backend.run("echo hi", timeout=5)
        assert result.backend_error and "was not found" in result.backend_error
        assert result.exit_code is None

    @pytest.mark.asyncio
    async def test_a_missing_binary_says_how_to_get_docker(self, tmp_path: Path) -> None:
        backend = DockerBackend(
            Workspace.at(tmp_path), options=DockerOptions(docker_binary="/nonexistent/docker")
        )
        with pytest.raises(BackendUnavailableError) as raised:
            await backend.check()
        assert "was not found" in raised.value.problem
        assert raised.value.fix and "docs.docker.com" in raised.value.fix

    @pytest.mark.parametrize(
        ("platform", "expected"),
        [
            ("darwin", "open -a Docker"),
            ("linux", "systemctl start docker"),
            ("win32", "Docker Desktop"),
        ],
    )
    def test_the_start_hint_fits_the_platform(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str, expected: str
    ) -> None:
        monkeypatch.setattr("sys.platform", platform)
        assert expected in DockerBackend(Workspace.at(tmp_path))._start_hint()

    @pytest.mark.asyncio
    async def test_a_daemon_that_is_down_is_told_apart_from_a_missing_docker(
        self, tmp_path: Path
    ) -> None:
        # `false` exists and exits 1 for `info`: installed, not answering.
        backend = DockerBackend(
            Workspace.at(tmp_path),
            options=DockerOptions(docker_binary=shutil.which("false") or "false"),
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
        assert _INSTALLS.search(command)

    @pytest.mark.parametrize("command", ["ls", "echo pipe", "cat pip.txt", "grep pipeline f"])
    def test_not_installs(self, command: str) -> None:
        assert not _INSTALLS.search(command)


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
        assert _NPM_GLOBAL.search(command)

    @pytest.mark.parametrize("command", ["npm install cowsay", "echo npm -g", "ls -g", "npm test"])
    def test_the_rest(self, command: str) -> None:
        assert not _NPM_GLOBAL.search(command)


class _ScriptedDocker:
    """A stand-in for ``docker``: each call is answered by the first matching rule,
    and every call is recorded. No daemon needed, so these run everywhere."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.rules: list[tuple[Callable[[tuple[str, ...]], bool], list[tuple[int, str, str]]]] = []

    def when(
        self, match: Callable[[tuple[str, ...]], bool], *answers: tuple[int, str, str]
    ) -> None:
        """Answer matching calls with ``answers`` in turn; the last one repeats."""
        self.rules.append((match, list(answers)))

    async def __call__(self, *args: str, **_: Any) -> tuple[int, str, str]:
        self.calls.append(args)
        for match, answers in self.rules:
            if match(args):
                return answers.pop(0) if len(answers) > 1 else answers[0]
        return 0, "", ""

    def count(self, match: Callable[[tuple[str, ...]], bool]) -> int:
        return sum(1 for call in self.calls if match(call))


def _starts(args: tuple[str, ...]) -> bool:
    return args[:1] == ("run",)


def _probe(args: tuple[str, ...]) -> bool:
    return args[:1] == ("exec",) and any("command -v" in a for a in args)


@pytest.fixture
def scripted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[DockerBackend, _ScriptedDocker]:
    async def no_wait(_: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_wait)
    workspace = tmp_path / "project"
    workspace.mkdir()
    backend = DockerBackend(Workspace.at(workspace), options=DockerOptions(allowed_network="none"))
    fake = _ScriptedDocker()
    backend._docker = fake  # type: ignore[assignment]
    return backend, fake


class TestABusyDaemon:
    """Docker fails a start now and then when it is loaded. That must not reach the
    person as "could not start the sandbox"."""

    @pytest.mark.asyncio
    async def test_a_start_that_failed_once_is_tried_again(
        self, scripted: tuple[DockerBackend, _ScriptedDocker]
    ) -> None:
        backend, fake = scripted
        fake.when(
            _starts,
            (125, "", "docker: Error response from daemon: No such container: abc"),
            (0, "abc\n", ""),
        )
        container = await backend._ensure_container("c1")
        assert container.name == backend._container_name("c1")
        assert fake.count(_starts) == 2

    @pytest.mark.asyncio
    async def test_a_start_that_fails_twice_is_reported(
        self, scripted: tuple[DockerBackend, _ScriptedDocker]
    ) -> None:
        backend, fake = scripted
        fake.when(_starts, (125, "", "docker: Error response from daemon: boom"))
        with pytest.raises(BackendUnavailableError, match="could not start the sandbox"):
            await backend._ensure_container("c1")
        assert fake.count(_starts) == 2

    @pytest.mark.asyncio
    async def test_a_probe_the_daemon_failed_does_not_blame_the_image(
        self, scripted: tuple[DockerBackend, _ScriptedDocker]
    ) -> None:
        backend, fake = scripted
        fake.when(
            _probe,
            (1, "", "Error response from daemon: container abc is not running"),
            (0, "", ""),
        )
        container = await backend._ensure_container("c1")
        assert container.name == backend._container_name("c1")
        assert fake.count(_starts) == 2

    @pytest.mark.asyncio
    async def test_an_image_without_the_tools_is_still_named_and_not_retried(
        self, scripted: tuple[DockerBackend, _ScriptedDocker]
    ) -> None:
        backend, fake = scripted
        fake.when(_probe, (1, "", ""))
        with pytest.raises(BackendUnavailableError, match="lacks bash, timeout or ps"):
            await backend._ensure_container("c1")
        assert fake.count(_starts) == 1

    @pytest.mark.asyncio
    async def test_the_proxy_is_started_again_when_the_first_attempt_fails(
        self, scripted: tuple[DockerBackend, _ScriptedDocker]
    ) -> None:
        backend, fake = scripted
        backend._options = DockerOptions(allowed_network="public")
        fake.when(
            lambda a: a[:1] == ("run",) and "--read-only" in a,
            (125, "", "Error response from daemon: No such container: p"),
            (0, "p\n", ""),
        )
        fake.when(lambda a: a[:1] == ("logs",), (0, "READY\n", ""))
        # `inspect` of the network and the proxy: neither exists yet.
        fake.when(lambda a: a[:1] == ("inspect",), (1, "", "No such object"))
        fake.when(lambda a: a[:2] == ("network", "inspect"), (1, "", "No such network"))
        await backend._ensure_network()
        assert fake.count(lambda a: a[:1] == ("run",) and "--read-only" in a) == 2
        assert backend._network_ready

    @pytest.mark.asyncio
    async def test_a_container_lost_between_commands_says_so(
        self, scripted: tuple[DockerBackend, _ScriptedDocker]
    ) -> None:
        backend, _fake = scripted
        notices: list[str] = []
        backend.on_notice = notices.append
        backend._prepared = True
        backend._image = "img"
        answers = [
            CommandResult(
                exit_code=None, stdout="", stderr="", backend_error="Error: No such container: x"
            ),
            CommandResult(exit_code=0, stdout="", stderr=""),
        ]

        async def exec_(*_: Any, **__: Any) -> CommandResult:
            return answers.pop(0)

        backend._exec = exec_  # type: ignore[method-assign]
        result = await backend.run("true", timeout=5, conversation_id="c1")
        assert result.exit_code == 0
        assert any("sandbox was lost" in n for n in notices)


class TestDefaults:
    def test_the_registries_are_always_listed(self) -> None:
        assert {"pypi.org", "registry.npmjs.org"} <= set(REGISTRY_HOSTS)

    def test_the_image_has_what_a_project_needs(self) -> None:
        for wanted in ("git", "curl", "procps", "node"):
            assert wanted in _DOCKERFILE


@pytest_asyncio.fixture
async def sandbox(tmp_path: Path) -> AsyncIterator[Callable[..., DockerBackend]]:
    """Builds backends over a workspace and removes every container, proxy and
    network they made, whatever the test did."""
    made: list[DockerBackend] = []
    workspace = tmp_path / "project"
    workspace.mkdir()

    def make(roots: tuple[Path, ...] = (), **options: Any) -> DockerBackend:
        backend = DockerBackend(
            Workspace.at(workspace, also=roots), options=DockerOptions(**options)
        )
        made.append(backend)
        return backend

    make.workspace = workspace  # type: ignore[attr-defined]
    yield make
    for backend in made:
        await backend.close()


async def _sh(backend: CommandBackend, command: str, conversation: str = "c1", timeout: float = 90):
    return await backend.run(command, timeout=timeout, conversation_id=conversation)


@needs_docker
class TestARealContainer:
    @pytest.mark.asyncio
    async def test_a_write_in_the_workspace_reaches_the_host(self, sandbox: Any) -> None:
        backend = sandbox()
        result = await _sh(backend, "echo from-container > made.txt && pwd")
        assert result.exit_code == 0 and result.backend_error is None
        assert result.stdout.strip() == str(sandbox.workspace)
        assert (sandbox.workspace / "made.txt").read_text().strip() == "from-container"
        # Owned by the user who runs domyn, not by root.
        assert (sandbox.workspace / "made.txt").stat().st_uid == os.getuid()

    @pytest.mark.asyncio
    async def test_the_host_outside_the_workspace_is_not_there(
        self, sandbox: Any, tmp_path: Path
    ) -> None:
        secret = tmp_path / "secret.txt"
        secret.write_text("nope")
        backend = sandbox()
        result = await _sh(
            backend, f"cat {secret}; ls {Path.home()} 2>&1; ls /var/run/docker.sock 2>&1"
        )
        assert "nope" not in result.stdout
        assert "No such file" in result.stderr + result.stdout

    @pytest.mark.asyncio
    async def test_no_host_environment_leaks(
        self, sandbox: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DOMYN_TEST_SECRET", "hunter2")
        result = await _sh(sandbox(), "env")
        assert "hunter2" not in result.stdout

    @pytest.mark.asyncio
    async def test_an_allowed_variable_is_passed(
        self, sandbox: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DOMYN_TEST_ALLOWED", "yes-please")
        backend = DockerBackend(Workspace.at(sandbox.workspace), allowed_env=["DOMYN_TEST_ALLOWED"])
        try:
            result = await _sh(backend, "echo $DOMYN_TEST_ALLOWED")
            assert result.stdout.strip() == "yes-please"
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_the_masked_paths_are_read_only(self, sandbox: Any) -> None:
        git = sandbox.workspace / ".git"
        (git / "hooks").mkdir(parents=True)
        (git / "config").write_text("[core]\n")
        (sandbox.workspace / "AGENTS.md").write_text("rules")
        backend = sandbox()
        for command in (
            "touch .domyn/config.yaml",
            "touch .domyn/skills",
            "touch .git/hooks/pre-commit",
            "echo x >> .git/config",
            "echo x >> AGENTS.md",
        ):
            result = await _sh(backend, command)
            assert result.exit_code != 0, command
            assert "Read-only" in result.stderr, command
        # The places it does need stay writable.
        result = await _sh(backend, "mkdir -p .domyn/processes/zz && touch .domyn/processes/zz/ok")
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_state_lives_across_calls(self, sandbox: Any) -> None:
        backend = sandbox()
        await _sh(backend, "echo kept > $HOME/note")
        result = await _sh(backend, "cat $HOME/note")
        assert result.stdout.strip() == "kept"

    @pytest.mark.asyncio
    async def test_a_timeout_kills_the_whole_tree(self, sandbox: Any) -> None:
        backend = sandbox()
        started = asyncio.get_running_loop().time()
        result = await _sh(backend, "sleep 300 & sleep 300 & wait", timeout=3)
        assert result.timed_out
        assert asyncio.get_running_loop().time() - started < 30
        leftover = await _sh(backend, "ps -eo args | grep -c '[s]leep 300'")
        assert leftover.stdout.strip() == "0"

    @pytest.mark.asyncio
    async def test_stdin_and_exit_codes(self, sandbox: Any) -> None:
        backend = sandbox()
        result = await backend.run("cat; exit 3", timeout=20, conversation_id="c1", stdin="typed\n")
        assert result.exit_code == 3 and result.stdout == "typed\n"

    @pytest.mark.asyncio
    async def test_the_network_is_registries_only(self, sandbox: Any) -> None:
        backend = sandbox()
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
    async def test_pip_install_works_through_the_allowlist(self, sandbox: Any) -> None:
        backend = sandbox()
        install = await _sh(backend, "pip install --quiet cowsay")
        assert install.exit_code == 0, install.stderr
        result = await _sh(backend, "python -c 'import cowsay; print(cowsay.__name__)'")
        assert result.stdout.strip() == "cowsay"
        recorded = backend._requirements_path("c1").read_text()
        assert "cowsay" in recorded

    @pytest.mark.asyncio
    async def test_the_image_has_git_curl_node_and_npm(self, sandbox: Any) -> None:
        backend = sandbox()
        result = await _sh(
            backend, "git --version && curl --version | head -1 && node -v && npm -v"
        )
        assert result.exit_code == 0, result.stderr

    @pytest.mark.asyncio
    async def test_global_npm_packages_are_recorded_and_restored(self, sandbox: Any) -> None:
        backend = sandbox()
        install = await _sh(backend, "npm install -g --no-audit --no-fund cowsay", timeout=180)
        assert install.exit_code == 0, install.stderr
        assert "cowsay@" in backend._npm_globals_path("c1").read_text()
        assert (await _sh(backend, "cowsay hi")).exit_code == 0
        # A new container for the same conversation brings them back.
        await backend.end_conversation("c1")
        again = await _sh(backend, "cowsay hi", timeout=180)
        assert again.exit_code == 0, again.stderr

    @pytest.mark.asyncio
    async def test_a_local_npm_install_lands_in_the_workspace(self, sandbox: Any) -> None:
        backend = sandbox()
        install = await _sh(backend, "npm install --no-audit --no-fund cowsay", timeout=180)
        assert install.exit_code == 0, install.stderr
        assert (sandbox.workspace / "node_modules" / "cowsay").is_dir()

    @pytest.mark.asyncio
    async def test_public_reaches_any_public_host_but_not_a_private_address(
        self, sandbox: Any
    ) -> None:
        backend = sandbox()
        ok = await _sh(backend, "curl -sS -o /dev/null -w '%{http_code}' https://example.org")
        assert ok.exit_code == 0 and ok.stdout.strip() == "200", ok.stderr
        began = time.time()
        private = await _sh(backend, "curl -sS https://10.0.0.1/")
        assert private.exit_code != 0
        assert "10.0.0.1" in await backend.explain_failure(began)

    @pytest.mark.asyncio
    async def test_a_list_reaches_only_those_and_the_registries(self, sandbox: Any) -> None:
        backend = sandbox(allowed_network=["github.com"])
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
    async def test_no_network_means_none(self, sandbox: Any) -> None:
        backend = sandbox(allowed_network="none")
        result = await _sh(
            backend,
            "python -c \"import socket; socket.create_connection(('1.1.1.1', 443), timeout=5)\"",
        )
        assert result.exit_code != 0


@needs_docker
class TestTheContainersLife:
    @pytest.mark.asyncio
    async def test_a_conversations_end_destroys_its_container_and_a_resume_restores_installs(
        self, sandbox: Any
    ) -> None:
        backend = sandbox()
        await _sh(backend, "pip install --quiet cowsay")
        name = backend._container_name("c1")
        assert await _exists(name)

        await backend.end_conversation("c1")
        assert not await _exists(name)

        notices: list[str] = []
        backend.on_notice = notices.append
        result = await _sh(backend, "python -c 'import cowsay; print(\"back\")'")
        assert result.stdout.strip() == "back"
        assert any("Restoring" in n for n in notices)

    @pytest.mark.asyncio
    async def test_close_removes_containers_proxy_and_network(self, sandbox: Any) -> None:
        backend = sandbox()
        await _sh(backend, "true", conversation="a")
        await _sh(backend, "true", conversation="b")
        names = [backend._container_name("a"), backend._container_name("b"), backend._proxy_name]
        assert all([await _exists(n) for n in names])
        await backend.close()
        assert not any([await _exists(n) for n in names])
        assert not await _exists(backend._network_name, kind="network")

    @pytest.mark.asyncio
    async def test_a_rebuilt_backend_adopts_the_running_container(self, sandbox: Any) -> None:
        first = sandbox()
        await _sh(first, "echo $$ > $HOME/pid-marker")
        # A config reload builds a new backend over the same workspace, in this process.
        second = sandbox()
        notices: list[str] = []
        second.on_notice = notices.append
        result = await _sh(second, "cat $HOME/pid-marker")
        assert result.exit_code == 0 and result.stdout.strip().isdigit()
        assert not any("Starting" in n for n in notices)

    @pytest.mark.asyncio
    async def test_a_container_removed_underneath_is_recreated(self, sandbox: Any) -> None:
        backend = sandbox()
        await _sh(backend, "true")
        subprocess.run(["docker", "rm", "-f", backend._container_name("c1")], capture_output=True)
        result = await _sh(backend, "echo again")
        assert result.stdout.strip() == "again"

    @pytest.mark.asyncio
    async def test_what_a_dead_process_left_is_removed(self, sandbox: Any) -> None:
        backend = sandbox()
        await backend.prepare()
        gone = subprocess.Popen(["true"])
        gone.wait()
        stale = f"domyn-stale-{gone.pid}"
        started = subprocess.run(
            [
                "docker", "run", "-d", "--name", stale,
                "--label", "domyn.sandbox=1", "--label", f"domyn.owner-pid={gone.pid}",
                "domyn-sandbox:py312-2", "sleep", "60",
            ],
            capture_output=True,
        )  # fmt: skip
        # Any other process running the sandbox reaps a dead owner's containers too, so
        # when the suite runs in parallel it may be gone before this one looks, or even
        # while `docker run` is still starting it. Either way it ends up removed, which
        # is all this test is about.
        assert started.returncode == 0 or "No such container" in started.stderr.decode()
        await sandbox()._reap_orphans()
        assert not await _exists(stale)

    @pytest.mark.asyncio
    async def test_ending_a_conversation_whose_container_is_starting_leaves_nothing(
        self, sandbox: Any
    ) -> None:
        backend = sandbox()
        warming = asyncio.ensure_future(backend.warm_up("c1"))
        while not backend._lock_for("c1").locked():
            await asyncio.sleep(0.01)
        await backend.end_conversation("c1")
        await warming
        assert not await _exists(backend._container_name("c1"))

    @pytest.mark.asyncio
    async def test_closing_while_a_container_is_starting_leaves_nothing(self, sandbox: Any) -> None:
        backend = sandbox()
        warming = asyncio.ensure_future(backend.warm_up("c1"))
        while not backend._lock_for("c1").locked():
            await asyncio.sleep(0.01)
        await backend.close()
        await warming
        assert not await _exists(backend._container_name("c1"))
        assert not await _exists(backend._proxy_name)

    @pytest.mark.asyncio
    async def test_a_live_owners_containers_are_left_alone(self, sandbox: Any) -> None:
        backend = sandbox()
        await _sh(backend, "true")
        await sandbox()._reap_orphans()
        assert await _exists(backend._container_name("c1"))


@needs_docker
class TestSkillsReachTheContainer:
    @pytest.mark.asyncio
    async def test_an_edit_on_the_host_is_seen_without_a_restart(
        self, sandbox: Any, tmp_path: Path
    ) -> None:
        skills = tmp_path / "skills"
        (skills / "demo").mkdir(parents=True)
        script = skills / "demo" / "run.py"
        script.write_text("print('v1')")
        backend = sandbox(roots=(skills,))
        assert (await _sh(backend, f"python {script}")).stdout.strip() == "v1"
        script.write_text("print('v2')")
        (skills / "demo" / "new.py").write_text("print('new')")
        assert (await _sh(backend, f"python {script}")).stdout.strip() == "v2"
        assert (await _sh(backend, f"python {skills}/demo/new.py")).stdout.strip() == "new"

    @pytest.mark.asyncio
    async def test_skills_are_read_only(self, sandbox: Any, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        skills.mkdir()
        backend = sandbox(roots=(skills,))
        result = await _sh(backend, f"touch {skills}/evil")
        assert result.exit_code != 0 and "Read-only" in result.stderr

    @pytest.mark.asyncio
    async def test_a_new_skill_root_recreates_the_container_and_keeps_installs(
        self, sandbox: Any, tmp_path: Path
    ) -> None:
        one, two = tmp_path / "one", tmp_path / "two"
        for folder in (one, two):
            folder.mkdir()
        (two / "tool.py").write_text("print('two')")
        first = sandbox(roots=(one,))
        await _sh(first, "pip install --quiet cowsay")
        # The config changed and the agent was rebuilt with another skill folder.
        second = sandbox(roots=(one, two))
        result = await _sh(second, f"python {two}/tool.py && python -c 'import cowsay'")
        assert result.exit_code == 0, result.stderr
        assert result.stdout.strip() == "two"

    @pytest.mark.asyncio
    async def test_a_swapped_install_directory_is_picked_up(
        self, sandbox: Any, tmp_path: Path
    ) -> None:
        skills = tmp_path / "skills"
        skills.mkdir()
        (skills / "a.py").write_text("print('old')")
        backend = sandbox(roots=(skills,))
        assert (await _sh(backend, f"python {skills}/a.py")).stdout.strip() == "old"
        # What `domyn setup` does: build beside, swap in.
        skills.rename(tmp_path / "skills-old")
        skills.mkdir()
        (skills / "a.py").write_text("print('new')")
        assert (await _sh(backend, f"python {skills}/a.py")).stdout.strip() == "new"

    @pytest.mark.asyncio
    async def test_a_recreate_waits_for_a_running_command(
        self, sandbox: Any, tmp_path: Path
    ) -> None:
        one, two = tmp_path / "one", tmp_path / "two"
        one.mkdir()
        two.mkdir()
        first = sandbox(roots=(one,))
        run = shell.build_run_command(first)
        state: dict[str, Any] = {}
        await run(command="sleep 120", background=True, conversation_id="c1", state=state)

        second = sandbox(roots=(one, two))
        notices: list[str] = []
        second.on_notice = notices.append
        result = await _sh(second, "ps -eo args | grep -c '[s]leep 120'")
        assert result.stdout.strip() == "1", "the background command was killed"
        assert any("background command is running" in n for n in notices)


@needs_docker
class TestShellToolsInTheContainer:
    """Through ``run_command`` itself: the backend is handed a launcher script with the
    model's command quoted inside it, not the command, and exit codes are the script's."""

    @pytest.mark.asyncio
    async def test_a_failure_names_the_blocked_host(self, sandbox: Any) -> None:
        backend = sandbox(allowed_network=["github.com"])
        run = shell.build_run_command(backend)
        result = await run(command="curl -sS https://example.org", conversation_id="c1", state={})
        text = str(result.data)
        assert "[sandbox] blocked" in text and "example.org" in text
        assert "docker: allowed_network:" in text

    @pytest.mark.asyncio
    async def test_a_successful_command_gets_no_note(self, sandbox: Any) -> None:
        backend = sandbox()
        run = shell.build_run_command(backend)
        result = await run(command="echo fine", conversation_id="c1", state={})
        assert "[sandbox]" not in str(result.data)

    @pytest.mark.asyncio
    async def test_installs_are_recorded_and_come_back(self, sandbox: Any) -> None:
        backend = sandbox()
        run = shell.build_run_command(backend)
        state: dict[str, Any] = {}
        pip = await run(command="pip install cowsay", conversation_id="c1", state=state)
        assert "Successfully installed" in str(pip.data)
        assert "cowsay" in backend._requirements_path("c1").read_text()
        npm = await run(
            command="npm install -g --no-audit --no-fund is-odd", conversation_id="c1", state=state
        )
        assert "is-odd@" in backend._npm_globals_path("c1").read_text(), str(npm.data)
        # The same conversation in a new container: both are back.
        await backend.end_conversation("c1")
        again = await run(
            command="python -c 'import cowsay; print(\"py-ok\")' && npm ls -g --depth=0",
            conversation_id="c1",
            state=state,
        )
        assert "py-ok" in str(again.data) and "is-odd@" in str(again.data), str(again.data)

    @pytest.mark.asyncio
    async def test_background_commands_run_check_and_stop(self, sandbox: Any) -> None:
        backend = sandbox()
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
