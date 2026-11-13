"""The Docker backend's parts, without Docker: a scripted ``docker``, a stand-in
executable for the subprocess paths, and the proxy over loopback sockets.

CI has no image, so the real-container tests in ``test_docker_backend.py`` are skipped
there. These keep the same code under test everywhere.
"""

import asyncio
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from domyn.agents.tools.builtin import (
    BackendUnavailableError,
    CommandBackend,
    CommandResult,
    DockerBackend,
    DockerBackendOptions,
    Workspace,
    WorkspaceError,
    file_tools,
)
from domyn.agents.tools.builtin.backends import ToolHost
from domyn.agents.tools.builtin.backends.docker import cli as docker_cli
from domyn.agents.tools.builtin.backends.docker import egress_proxy, network
from domyn.agents.tools.builtin.backends.docker.cli import DockerCli, DockerOutput
from domyn.agents.tools.builtin.backends.docker.containers import DockerContainer
from domyn.agents.tools.builtin.backends.docker.installs import (
    InstallLog,
    _npm_packages,
)
from domyn.agents.tools.builtin.backends.docker.mounts import build_mounts
from domyn.agents.tools.builtin.backends.docker.network import EgressNetwork
from domyn.agents.tools.builtin.backends.docker.scope import BackendScope, remove_orphans
from domyn.agents.tools.builtin.backends.docker.state import StateFolder
from domyn.agents.tools.builtin.files import (
    OperationError,
    answer_of,
    main,
    parse_answer,
    request_text,
    serve,
)
from tests.tools.test_docker_backend import _failed, _ok, _ScriptedCli


@pytest.fixture
def no_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instantly(_: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", instantly)


# --- the proxy ---------------------------------------------------------------------------


async def _ask_proxy(request: bytes, allow: list[str], *, hang_up: bool = False) -> bytes:
    server = await asyncio.start_server(
        lambda r, w: egress_proxy.handle(r, w, allow), host="127.0.0.1", port=0
    )
    port = server.sockets[0].getsockname()[1]
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(request)
        await writer.drain()
        if hang_up:
            writer.write_eof()
        answer = await asyncio.wait_for(reader.read(65536), timeout=10)
        writer.close()
        return answer
    finally:
        server.close()
        await server.wait_closed()


class TestTheProxy:
    @pytest.mark.asyncio
    async def test_plain_http_is_refused(self) -> None:
        answer = await _ask_proxy(b"GET http://example.com/ HTTP/1.1\r\n\r\n", ["*"])
        assert answer.startswith(b"HTTP/1.1 403") and b"only HTTPS" in answer

    @pytest.mark.asyncio
    async def test_a_host_off_the_list_is_refused(self) -> None:
        answer = await _ask_proxy(
            b"CONNECT other.org:443 HTTP/1.1\r\n\r\n", ["example.com", "*.example.net"]
        )
        assert answer.startswith(b"HTTP/1.1 403") and b"not in the allowlist" in answer

    @pytest.mark.asyncio
    async def test_a_port_other_than_443_is_refused(self) -> None:
        answer = await _ask_proxy(b"CONNECT example.com:8080 HTTP/1.1\r\n\r\n", ["example.com"])
        assert b"only port 443" in answer

    @pytest.mark.asyncio
    async def test_a_name_that_resolves_to_a_private_address_is_refused(self) -> None:
        answer = await _ask_proxy(b"CONNECT localhost:443 HTTP/1.1\r\n\r\n", ["*"])
        assert b"non-public" in answer

    @pytest.mark.asyncio
    async def test_a_name_that_does_not_resolve_is_refused(self) -> None:
        answer = await _ask_proxy(b"CONNECT nothing.invalid:443 HTTP/1.1\r\n\r\n", ["*"])
        assert b"does not resolve" in answer

    @pytest.mark.asyncio
    async def test_a_client_that_hangs_up_mid_request_is_dropped_quietly(self) -> None:
        assert await _ask_proxy(b"CONNECT exam", ["*"], hang_up=True) == b""

    @pytest.mark.asyncio
    async def test_a_listed_host_is_tunnelled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With the address check satisfied and the upstream a local echo server, bytes
        go through both ways."""
        received: list[bytes] = []

        async def echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            data = await reader.read(100)
            received.append(data)
            writer.write(b"pong")
            await writer.drain()
            writer.close()

        upstream = await asyncio.start_server(echo, host="127.0.0.1", port=0)
        upstream_port = upstream.sockets[0].getsockname()[1]
        real_open = asyncio.open_connection

        async def to_the_echo(address: str, port: int) -> Any:
            return await real_open("127.0.0.1", upstream_port)

        monkeypatch.setattr(egress_proxy, "public_address", lambda _: True)
        monkeypatch.setattr(egress_proxy.asyncio, "open_connection", to_the_echo)
        proxy = await asyncio.start_server(
            lambda r, w: egress_proxy.handle(r, w, ["localhost"]), host="127.0.0.1", port=0
        )
        proxy_port = proxy.sockets[0].getsockname()[1]
        try:
            reader, writer = await real_open("127.0.0.1", proxy_port)
            writer.write(b"CONNECT localhost:443 HTTP/1.1\r\n\r\n")
            await writer.drain()
            established = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=10)
            writer.write(b"ping")
            await writer.drain()
            reply = await asyncio.wait_for(reader.read(100), timeout=10)
            writer.close()
        finally:
            proxy.close()
            upstream.close()
            await proxy.wait_closed()
            await upstream.wait_closed()

        assert established.startswith(b"HTTP/1.1 200")
        assert received == [b"ping"] and reply == b"pong"

    @pytest.mark.asyncio
    async def test_an_unreachable_upstream_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_open = asyncio.open_connection

        async def refused(address: str, port: int) -> Any:
            if port == 443:  # the proxy's way out; the client's own connection is real
                raise OSError("connection refused")
            return await real_open(address, port)

        monkeypatch.setattr(egress_proxy, "public_address", lambda _: True)
        monkeypatch.setattr(egress_proxy.asyncio, "open_connection", refused)

        answer = await _ask_proxy(b"CONNECT localhost:443 HTTP/1.1\r\n\r\n", ["localhost"])

        assert b"unreachable" in answer

    def test_the_list_matches_names_and_their_subdomains_only(self) -> None:
        allow = ["example.com", "*.example.net"]
        assert egress_proxy.host_allowed("Example.com.", allow)
        assert egress_proxy.host_allowed("a.example.net", allow)
        assert not egress_proxy.host_allowed("example.net", allow)
        assert not egress_proxy.host_allowed("evil-example.com", allow)

    def test_only_global_addresses_are_public(self) -> None:
        assert egress_proxy.public_address("8.8.8.8")
        assert not egress_proxy.public_address("10.0.0.1")
        assert not egress_proxy.public_address("127.0.0.1")
        assert not egress_proxy.public_address("not an address")

    def test_main_reads_its_arguments_and_serves(self, monkeypatch: pytest.MonkeyPatch) -> None:
        served: list[tuple[int, list[str]]] = []

        async def serve(port: int, allow: list[str]) -> None:
            served.append((port, allow))

        monkeypatch.setattr(egress_proxy, "serve", serve)
        egress_proxy.main(["--port", "4000", "--allow", "A.com, b.org ,"])

        assert served == [(4000, ["a.com", "b.org"])]


# --- docker as a subprocess ----------------------------------------------------------------


def _fake_docker(tmp_path: Path, script: str) -> DockerCli:
    path = tmp_path / "docker"
    path.write_text("#!/bin/sh\n" + script)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return DockerCli(str(path))


class TestDockerAsASubprocess:
    @pytest.mark.asyncio
    async def test_a_call_returns_what_docker_printed(self, tmp_path: Path) -> None:
        cli = _fake_docker(tmp_path, 'echo "out $1"; echo "err" >&2; exit 3\n')
        result = await cli("version")
        assert (result.code, result.stdout.strip(), result.stderr.strip()) == (
            3,
            "out version",
            "err",
        )

    @pytest.mark.asyncio
    async def test_input_is_piped_in(self, tmp_path: Path) -> None:
        cli = _fake_docker(tmp_path, "cat\n")
        assert (await cli("build", stdin="Dockerfile text")).stdout == "Dockerfile text"

    @pytest.mark.asyncio
    async def test_a_call_that_takes_too_long_is_reported(self, tmp_path: Path) -> None:
        cli = _fake_docker(tmp_path, "sleep 30\n")
        result = await cli("pull", "x", timeout=0.3)
        assert result.code == 124 and "timed out" in result.stderr

    @pytest.mark.asyncio
    async def test_a_missing_docker_says_so(self, tmp_path: Path) -> None:
        with pytest.raises(BackendUnavailableError, match="was not found"):
            await DockerCli(str(tmp_path / "no-docker"))("info")

    @pytest.mark.asyncio
    async def test_the_daemon_check_says_why_it_failed(self, tmp_path: Path) -> None:
        denied = _fake_docker(tmp_path, 'echo "permission denied" >&2; exit 1\n')
        with pytest.raises(BackendUnavailableError, match="permission denied"):
            await denied.check_daemon()
        down = _fake_docker(tmp_path, 'echo "cannot connect" >&2; exit 1\n')
        with pytest.raises(BackendUnavailableError, match="not running"):
            await down.check_daemon()
        up = _fake_docker(tmp_path, "echo 29.0\n")
        await up.check_daemon()

    @pytest.mark.asyncio
    async def test_a_command_in_a_container_has_its_result(self, tmp_path: Path) -> None:
        cli = _fake_docker(tmp_path, 'echo "ran in $6"; exit 4\n')
        result = await cli.exec_command("box", "ls", workdir=tmp_path, timeout=5)
        assert result.exit_code == 4 and "ran in" in result.stdout

    @pytest.mark.asyncio
    async def test_the_limit_is_the_containers_to_enforce(self, tmp_path: Path) -> None:
        cli = _fake_docker(tmp_path, "exit 137\n")
        started = await cli.exec_command("box", "sleep 60", workdir=tmp_path, timeout=1)
        # Killed by `timeout` inside the container, but before its limit: not a timeout.
        assert started.exit_code == 137 or started.timed_out

    @pytest.mark.asyncio
    async def test_docker_failing_is_not_the_commands_failure(self, tmp_path: Path) -> None:
        cli = _fake_docker(tmp_path, 'echo "No such container: box" >&2; exit 125\n')
        result = await cli.exec_command("box", "ls", workdir=tmp_path, timeout=5)
        assert result.backend_error and "No such container" in result.backend_error

    @pytest.mark.asyncio
    async def test_a_client_that_stops_answering_is_ended(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(docker_cli, "EXEC_GRACE_SECONDS", 0.2)
        cli = _fake_docker(tmp_path, "sleep 30\n")
        result = await cli.exec_command("box", "ls", workdir=tmp_path, timeout=1)
        assert result.timed_out

    @pytest.mark.asyncio
    async def test_a_docker_that_cannot_start_is_a_backend_error(self, tmp_path: Path) -> None:
        path = tmp_path / "docker"
        path.write_text("not executable")
        result = await DockerCli(str(path)).exec_command("box", "ls", workdir=tmp_path, timeout=5)
        assert result.backend_error and "could not run docker" in result.backend_error

    def test_the_hint_for_starting_it_follows_the_binary_and_the_system(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert "podman machine start" in DockerCli("podman")._how_to_start()
        for system, expected in (
            ("Darwin", "open -a Docker"),
            ("Linux", "systemctl"),
            ("Plan9", "Docker Desktop"),
        ):
            monkeypatch.setattr(docker_cli.platform, "system", lambda value=system: value)
            assert expected in DockerCli("docker")._how_to_start()


# --- what the agent installed -------------------------------------------------------------


def _install_log(tmp_path: Path, cli: DockerCli, *, can_download: bool = True) -> InstallLog:
    state = StateFolder.resolve(tmp_path, tmp_path / "state")
    state.prepare("c1")
    notices: list[str] = []
    log = InstallLog(cli, state, tmp_path, notices.append, can_download=can_download)
    log.notices = notices  # type: ignore[attr-defined]
    return log


def _in(box: str = "box") -> DockerContainer:
    return DockerContainer(name=box, fingerprint="f")


class TestWhatWasInstalled:
    @pytest.mark.asyncio
    async def test_a_pip_command_records_what_is_installed(self, tmp_path: Path) -> None:
        cli = _ScriptedCli()
        cli.when(lambda a: "pip freeze" in a[-1], _ok("requests==2.0\n"))
        log = _install_log(tmp_path, cli)

        await log.record(_in(), "c1", "pip install requests")
        await log.record(_in(), "c1", "pip install requests")  # unchanged: not rewritten

        assert log.requirements_path("c1").read_text() == "requests==2.0\n"

    @pytest.mark.asyncio
    async def test_a_command_that_is_no_install_records_nothing(self, tmp_path: Path) -> None:
        cli = _ScriptedCli()
        log = _install_log(tmp_path, cli)

        await log.record(_in(), "c1", "ls -l")

        assert cli.calls == [] and not log.requirements_path("c1").exists()

    @pytest.mark.asyncio
    async def test_a_global_npm_install_records_its_packages(self, tmp_path: Path) -> None:
        listing = json.dumps({"dependencies": {"cowsay": {"version": "1.6.0"}, "bad": {}}})
        cli = _ScriptedCli()
        cli.when(lambda a: "npm ls" in a[-1], _ok(listing))
        log = _install_log(tmp_path, cli)

        await log.record(_in(), "c1", "npm install -g cowsay")

        assert log.npm_globals_path("c1").read_text() == "cowsay@1.6.0\n"

    def test_npm_output_that_is_not_json_records_nothing(self) -> None:
        assert _npm_packages("npm ERR! something") is None
        assert _npm_packages('{"dependencies": null}') == ""

    @pytest.mark.asyncio
    async def test_a_new_container_gets_the_packages_back(self, tmp_path: Path) -> None:
        cli = _ScriptedCli()
        log = _install_log(tmp_path, cli)
        log.requirements_path("c1").write_text("requests==2.0\n")
        log.npm_globals_path("c1").write_text("cowsay@1.6.0\n")

        await log.replay(_in(), "c1")

        commands = [call[-1] for call in cli.calls if call[:1] == ("exec",)]
        assert any("pip install --user" in c for c in commands)
        assert any("npm install -g" in c and "cowsay@1.6.0" in c for c in commands)
        assert "Restoring 2 package(s)" in log.notices[0]  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_nothing_recorded_is_nothing_to_restore(self, tmp_path: Path) -> None:
        cli = _ScriptedCli()
        await _install_log(tmp_path, cli).replay(_in(), "c1")
        assert cli.calls == []

    @pytest.mark.asyncio
    async def test_without_a_network_it_says_what_it_cannot_restore(self, tmp_path: Path) -> None:
        cli = _ScriptedCli()
        log = _install_log(tmp_path, cli, can_download=False)
        log.requirements_path("c1").write_text("requests==2.0\n")

        await log.replay(_in(), "c1")

        assert cli.calls == [] and "no network" in log.notices[0]  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_a_package_that_will_not_install_is_said(self, tmp_path: Path) -> None:
        cli = _ScriptedCli()
        cli.when(lambda a: a[:1] == ("exec",), _failed("no matching distribution", code=1))
        log = _install_log(tmp_path, cli)
        log.requirements_path("c1").write_text("nothing==0\n")

        await log.replay(_in(), "c1")

        assert "Could not restore every Python package" in log.notices[-1]  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_a_record_that_is_a_link_is_not_written_through(self, tmp_path: Path) -> None:
        cli = _ScriptedCli()
        cli.when(lambda a: "pip freeze" in a[-1], _ok("requests==2.0\n"))
        log = _install_log(tmp_path, cli)
        target = tmp_path / "elsewhere.txt"
        target.write_text("untouched")
        log.requirements_path("c1").symlink_to(target)

        await log.record(_in(), "c1", "pip install requests")

        assert target.read_text() == "untouched"


# --- what a dead process left, and what the proxy refused ----------------------------------


def _dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


class TestWhatIsLeftBehind:
    @pytest.mark.asyncio
    async def test_only_what_a_dead_process_owned_is_removed(self) -> None:
        dead, alive = _dead_pid(), os.getpid()
        cli = _ScriptedCli()
        cli.when(
            lambda a: a[:2] == ("ps", "-a"), _ok(f"gone\t{dead}\nmine\t{alive}\nodd\tnot-a-pid\n")
        )
        cli.when(
            lambda a: a[:2] == ("network", "ls"), _ok(f"net-gone\t{dead}\nnet-mine\t{alive}\n")
        )

        await remove_orphans(cli)

        removed = [call for call in cli.calls if call[:2] in (("rm", "-f"), ("network", "rm"))]
        assert removed == [("rm", "-f", "gone"), ("network", "rm", "net-gone")]

    @pytest.mark.asyncio
    async def test_a_listing_that_fails_removes_nothing(self) -> None:
        cli = _ScriptedCli()
        cli.when(lambda a: True, _failed("daemon down"))

        await remove_orphans(cli)

        assert all(call[0] != "rm" for call in cli.calls)

    def test_a_scope_names_what_it_makes(self, tmp_path: Path) -> None:
        scope = BackendScope.of(tmp_path)
        assert scope.filter_arguments[0] == "--filter"
        assert scope.network_name.startswith("domyn-net-")
        assert scope.proxy_name.startswith("domyn-egress-")
        assert scope.container_name("c1") != scope.container_name("c2")


class TestWhatTheProxyRefused:
    @pytest.mark.asyncio
    async def test_the_denied_hosts_are_named_once_and_not_again(self) -> None:
        log = (
            "2026-01-01T00:00:01Z DENY example.org:443 not in the allowlist\n"
            "2026-01-01T00:00:02Z DENY example.org:443 not in the allowlist\n"
            "2026-01-01T00:00:03Z DENY 10.0.0.1:443 resolves to a non-public address\n"
            "2026-01-01T00:00:04Z READY 3128\n"
        )
        cli = _ScriptedCli()
        cli.when(lambda a: a[:1] == ("logs",), _ok(log))
        proxy = EgressNetwork(cli, BackendScope.of(Path("/w")), ["*"], lambda _: None)

        first = await proxy.blocked_since(1_700_000_000)
        second = await proxy.blocked_since(1_700_000_000)

        assert first == ["example.org", "10.0.0.1"] and second == []

    @pytest.mark.asyncio
    async def test_no_logs_means_nothing_to_report(self) -> None:
        cli = _ScriptedCli()
        cli.when(lambda a: a[:1] == ("logs",), _failed("no such container"))
        proxy = EgressNetwork(cli, BackendScope.of(Path("/w")), ["*"], lambda _: None)
        assert await proxy.blocked_since(0) == []

    @pytest.mark.asyncio
    async def test_a_ready_network_and_proxy_are_left_alone(self) -> None:
        cli = _ScriptedCli()
        cli.when(lambda a: a[:2] == ("network", "inspect"), _ok())
        cli.when(lambda a: a[:1] == ("inspect",), _ok("true\n"))
        proxy = EgressNetwork(cli, BackendScope.of(Path("/w")), ["*"], lambda _: None)

        await proxy.ensure("img")
        await proxy.ensure("img")  # asked often, done once

        assert [
            call for call in cli.calls if call[0] in ("run", "network") and call[1] != "inspect"
        ] == []
        await proxy.remove()
        assert any(call[:2] == ("network", "rm") for call in cli.calls)

    @pytest.mark.asyncio
    async def test_a_network_that_cannot_be_made_is_reported(self) -> None:
        cli = _ScriptedCli()
        cli.when(lambda a: a[:2] == ("network", "inspect"), _failed("none", code=1))
        cli.when(lambda a: a[:2] == ("network", "create"), _failed("pool overlaps"))
        proxy = EgressNetwork(cli, BackendScope.of(Path("/w")), ["*"], lambda _: None)

        with pytest.raises(BackendUnavailableError, match="could not create"):
            await proxy.ensure("img")

    @pytest.mark.asyncio
    async def test_a_proxy_that_never_says_ready_is_reported(self, no_waiting: None) -> None:
        cli = _ScriptedCli()
        cli.when(lambda a: a[:2] == ("network", "inspect"), _ok())
        # Not running when asked first; running, but never ready, afterwards.
        cli.when(lambda a: a[:1] == ("inspect",), _failed("none", code=1), _ok("true\n"))
        cli.when(lambda a: a[:1] == ("logs",), _ok("starting\n"))
        proxy = EgressNetwork(cli, BackendScope.of(Path("/w")), ["*"], lambda _: None)

        with pytest.raises(BackendUnavailableError, match="did not become ready"):
            await proxy.ensure("img")

    @pytest.mark.asyncio
    async def test_a_connect_that_fails_to_start_the_proxy_is_reported(
        self, no_waiting: None
    ) -> None:
        cli = _ScriptedCli()
        cli.when(lambda a: a[:2] == ("network", "inspect"), _ok())
        cli.when(lambda a: a[:1] == ("inspect",), _failed("none", code=1))
        cli.when(lambda a: a[:1] == ("run",), _ok("id\n"))
        cli.when(lambda a: a[:2] == ("network", "connect"), _failed("no bridge"))
        proxy = EgressNetwork(cli, BackendScope.of(Path("/w")), ["*"], lambda _: None)

        with pytest.raises(BackendUnavailableError, match="could not connect the proxy"):
            await proxy.ensure("img")

    def test_a_missing_proxy_script_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            Path, "read_text", lambda *a, **k: (_ for _ in ()).throw(OSError("gone"))
        )
        with pytest.raises(BackendUnavailableError, match="proxy script is missing"):
            network._proxy_source()


# --- a file tool run through a backend ---------------------------------------------------------


class _Hosting(CommandBackend):
    """A backend whose tool bodies run "elsewhere": the request goes through ``run``
    and ``serve`` answers it, as the container's entry point does."""

    name = "elsewhere"

    def __init__(self, host: ToolHost | None, *, fail: CommandResult | None = None) -> None:
        self._host = host
        self.fail = fail
        self.requests: list[dict[str, Any]] = []
        self.started = 0

    @property
    def tool_host(self) -> ToolHost | None:
        return self._host

    async def start(self) -> None:
        self.started += 1

    async def run(
        self, command: str, *, timeout: float, conversation_id: str = "", stdin: str | None = None
    ) -> CommandResult:
        if self.fail is not None:
            return self.fail
        request = json.loads(stdin or "{}")
        self.requests.append(request)
        return CommandResult(exit_code=0, stdout=json.dumps(serve(request)) + "\n", stderr="")


class TestAFileToolThroughABackend:
    @pytest.mark.asyncio
    async def test_the_body_runs_where_the_backend_says_and_the_answer_comes_back(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "a.txt").write_text("hello\n")
        backend = _Hosting(ToolHost("run-it", roots=(Path("/"),)))
        tool = {t.name: t for t in file_tools(Workspace.at(tmp_path), host=backend)}["read_file"]

        result = await tool.execute(path="a.txt", conversation_id="c1")

        assert result.success and "hello" in result.data
        assert backend.started == 1 and backend.requests[0]["extra_roots"] == ["/"]

    @pytest.mark.asyncio
    async def test_a_refusal_comes_back_as_a_refusal(self, tmp_path: Path) -> None:
        backend = _Hosting(ToolHost("run-it"))
        tool = {t.name: t for t in file_tools(Workspace.at(tmp_path), host=backend)}["read_file"]

        result = await tool.execute(path="../outside.txt", conversation_id="c1")

        assert not result.success and "outside the workspace" in result.error

    @pytest.mark.asyncio
    async def test_a_backend_with_no_host_runs_the_body_here(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("here\n")
        backend = _Hosting(None)
        tool = {t.name: t for t in file_tools(Workspace.at(tmp_path), host=backend)}["read_file"]

        result = await tool.execute(path="a.txt", conversation_id="c1")

        assert result.success and "here" in result.data and backend.requests == []

    @pytest.mark.asyncio
    async def test_a_backend_that_failed_is_an_error_the_model_can_read(
        self, tmp_path: Path
    ) -> None:
        failed = CommandResult(exit_code=None, stdout="", stderr="", backend_error="box is gone")
        backend = _Hosting(ToolHost("run-it"), fail=failed)
        tool = {t.name: t for t in file_tools(Workspace.at(tmp_path), host=backend)}["read_file"]

        result = await tool.execute(path="a.txt", conversation_id="c1")

        assert not result.success and "box is gone" in result.error

    def test_what_a_run_did_is_turned_into_a_result_or_an_error(self) -> None:
        ok = CommandResult(exit_code=0, stdout='{"ok": true, "result": "x"}\n', stderr="")
        assert answer_of(ok, "read_file") == "x"
        with pytest.raises(OperationError, match="took longer"):
            answer_of(CommandResult(exit_code=None, stdout="", stderr="", timed_out=True), "grep")
        with pytest.raises(OperationError, match="failed: boom"):
            answer_of(CommandResult(exit_code=1, stdout="", stderr="boom"), "grep")

    def test_an_answer_is_its_last_line(self) -> None:
        assert parse_answer('a library warning\n{"ok": true, "result": "x"}\n') == "x"
        with pytest.raises(WorkspaceError, match="outside"):
            parse_answer('{"ok": false, "error": "outside", "workspace": true}')
        with pytest.raises(OperationError, match="boom"):
            parse_answer('{"ok": false, "error": "boom", "workspace": false}')
        with pytest.raises(OperationError, match="no answer"):
            parse_answer("")
        with pytest.raises(OperationError, match="no answer"):
            parse_answer("not json")

    def test_a_request_names_the_operation_and_the_roots(self, tmp_path: Path) -> None:
        text = request_text("grep", {"pattern": "x"}, root=tmp_path, extra_roots=(Path("/skills"),))
        assert json.loads(text) == {
            "operation": "grep",
            "arguments": {"pattern": "x"},
            "root": str(tmp_path),
            "extra_roots": ["/skills"],
        }

    def test_serving_never_raises(self, tmp_path: Path) -> None:
        assert (
            serve({"operation": "nothing", "arguments": {}, "root": str(tmp_path)})["ok"] is False
        )
        broken = serve({"operation": "read_file", "arguments": {"nope": 1}, "root": str(tmp_path)})
        assert broken["ok"] is False and not broken["workspace"]

    def test_the_entry_point_reads_a_request_and_prints_an_answer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "a.txt").write_text("hi\n")
        request = request_text("read_file", {"path": "a.txt"}, root=tmp_path, extra_roots=())
        monkeypatch.setattr(sys, "stdin", __import__("io").StringIO(request))

        assert main() == 0

        assert json.loads(capsys.readouterr().out.strip())["ok"] is True


# --- what a command that failed is told -----------------------------------------------------


class TestTheBackendExplainsAFailure:
    @pytest.mark.asyncio
    async def test_a_failed_command_gets_the_backends_note(self, tmp_path: Path) -> None:
        from domyn.agents.tools.builtin import LocalBackend, shell

        workspace = Workspace.at(tmp_path)
        state = tmp_path / "state"
        here = LocalBackend(workspace, state_dir=state)

        class Blocked(DockerBackend):
            """Runs the launcher script here, so the command really fails, and explains."""

            async def run(self, command: str, **kwargs: Any) -> CommandResult:
                return await here.run(command, **kwargs)

            async def explain_failure(self, since: float) -> str:
                return "[docker] blocked connection(s) to: example.org."

        backend = Blocked(
            workspace,
            options=DockerBackendOptions(allowed_network="none"),
            state_dir=state,
            cli=_ScriptedCli(),
        )
        tool = {t.name: t for t in shell.shell_tools(workspace, backend=backend)}["run_command"]

        failed = await tool.execute(command="exit 3", conversation_id="c")
        fine = await tool.execute(command="true", conversation_id="c")

        assert "blocked connection(s) to: example.org" in str(failed.data)
        assert "blocked connection(s)" not in str(fine.data)


# --- small things --------------------------------------------------------------------------


class TestSmallThings:
    def test_a_path_with_a_comma_cannot_be_mounted(self, tmp_path: Path) -> None:
        workspace = Workspace.at(tmp_path / "a,b")
        (tmp_path / "a,b").mkdir(exist_ok=True)
        arguments = build_mounts(workspace, StateFolder.resolve(workspace.root, None), "c1")
        assert not any("a,b" in argument for argument in arguments)

    def test_each_network_setting_describes_itself(self) -> None:
        assert "public" in DockerBackendOptions().describe_network()
        assert (
            "2 allowed"
            in DockerBackendOptions(allowed_network=["a.com", "b.com"]).describe_network()
        )
        assert DockerBackendOptions(allowed_network="none").describe_network() == "no network"
        assert DockerBackendOptions(allowed_network="none").allowed_hosts() == []

    def test_a_package_that_is_not_installed_has_no_version_and_pulls_latest(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from importlib import metadata

        from domyn.agents.tools.builtin.backends.docker import images

        def missing(name: str) -> str:
            raise metadata.PackageNotFoundError(name)

        monkeypatch.setattr(images.metadata, "version", missing)
        assert images.default_image().endswith(f":{images.LATEST_TAG}")

    def test_the_default_backend_runs_tools_in_this_process(self) -> None:
        class Plain(CommandBackend):
            async def run(self, command: str, **kwargs: Any) -> CommandResult:
                return CommandResult(exit_code=0, stdout="", stderr="")

        assert Plain().tool_host is None
