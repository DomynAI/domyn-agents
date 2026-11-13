"""``backend: auto``: Docker when it can be used, this machine when it cannot, and
the fallback is said rather than silent. An explicit ``docker`` still refuses."""

from pathlib import Path
from typing import Any

import pytest

from domyn.agents.tools.builtin import (
    BuiltinBundleConfig,
    DockerBackend,
    DockerBackendOptions,
    LocalBackend,
    Workspace,
    build_backend,
    file_tools,
)
from domyn.agents.tools.builtin.backends import BackendUnavailableError, CommandResult
from domyn.agents.tools.builtin.backends.auto import AutoBackend


@pytest.fixture
def docker_works(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Docker as a machine that has it: what `DockerBackend` does is recorded, and
    nothing is run. `AutoBackend` is one, so it reaches these through `super()`."""
    calls: list[str] = []

    async def check(self: DockerBackend) -> None:
        calls.append("check")

    async def prepare(self: DockerBackend) -> None:
        calls.append("prepare")

    async def run(self: DockerBackend, command: str, **kwargs: Any) -> CommandResult:
        calls.append(f"run {command}")
        return CommandResult(exit_code=0, stdout="in the container", stderr="")

    async def end_conversation(self: DockerBackend, conversation_id: str) -> None:
        calls.append(f"end {conversation_id}")

    async def close(self: DockerBackend) -> None:
        calls.append("close")

    for name, replacement in (
        ("check", check),
        ("prepare", prepare),
        ("run", run),
        ("end_conversation", end_conversation),
        ("close", close),
    ):
        monkeypatch.setattr(DockerBackend, name, replacement)
    return calls


def _auto(tmp_path: Path, **docker: Any) -> AutoBackend:
    backend = build_backend(
        "auto",
        Workspace.at(tmp_path),
        docker=DockerBackendOptions(docker_binary="/no/such/docker", **docker),
    )
    assert isinstance(backend, AutoBackend)
    return backend


class TestFallback:
    @pytest.mark.asyncio
    async def test_without_docker_commands_run_here_and_it_is_said(self, tmp_path: Path) -> None:
        backend = _auto(tmp_path)
        notices: list[str] = []
        backend.on_notice = notices.append

        await backend.check()

        assert backend._on_this_machine and isinstance(backend._local, LocalBackend)
        assert backend.fallback is not None and "Docker was not found" in backend.fallback.problem
        assert backend.is_isolated is False and backend.name == "local"
        assert len(notices) == 1 and "not in a container" in notices[0]

        result = await backend.run("echo hi", timeout=10, conversation_id="c")
        assert result.stdout.strip() == "hi"

    @pytest.mark.asyncio
    async def test_it_decides_once(self, tmp_path: Path) -> None:
        backend = _auto(tmp_path)
        notices: list[str] = []
        backend.on_notice = notices.append
        await backend.check()
        await backend.check()
        await backend.run("true", timeout=10)
        assert len(notices) == 1

    @pytest.mark.asyncio
    async def test_a_command_decides_when_nothing_checked_first(self, tmp_path: Path) -> None:
        backend = _auto(tmp_path)
        result = await backend.run("echo hi", timeout=10)
        assert result.stdout.strip() == "hi"
        assert backend.fallback is not None

    @pytest.mark.asyncio
    async def test_with_docker_it_is_the_chosen_backend(
        self, tmp_path: Path, docker_works: list[str]
    ) -> None:
        backend = AutoBackend(Workspace.at(tmp_path))

        await backend.check()
        assert backend.fallback is None
        assert backend.is_isolated and backend.destroys_on_close and backend.name == "docker"
        assert (await backend.run("ls", timeout=5)).stdout == "in the container"
        await backend.end_conversation("c")
        await backend.close()
        assert docker_works == ["check", "run ls", "end c", "close"]

    @pytest.mark.asyncio
    async def test_with_docker_a_tool_body_is_run_through_the_backend(
        self, tmp_path: Path, docker_works: list[str]
    ) -> None:
        backend = AutoBackend(Workspace.at(tmp_path))
        await backend.check()

        assert backend.tool_host is not None and backend.tool_host.command

    @pytest.mark.asyncio
    async def test_before_deciding_it_is_not_claimed_to_isolate(self, tmp_path: Path) -> None:
        backend = _auto(tmp_path)
        assert backend.is_isolated is False
        assert isinstance(backend, DockerBackend)

    @pytest.mark.asyncio
    async def test_closing_an_undecided_backend_is_fine(self, tmp_path: Path) -> None:
        await _auto(tmp_path).close()


class TestAnImageThatWillNotPull:
    """Docker is there but the image cannot be had: `auto` still starts, on this machine."""

    @pytest.mark.asyncio
    async def test_it_falls_back_when_preparing_fails(
        self, tmp_path: Path, docker_works: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def unpullable(self: DockerBackend) -> None:
            raise BackendUnavailableError("could not pull the image x.", fix="Check the network.")

        monkeypatch.setattr(DockerBackend, "prepare", unpullable)
        backend = AutoBackend(Workspace.at(tmp_path))
        notices: list[str] = []
        backend.on_notice = notices.append

        await backend.check()
        assert backend.name == "docker" and backend.fallback is None
        await backend.prepare()

        assert backend.name == "local" and backend.is_isolated is False
        assert backend.fallback is not None and "could not pull" in backend.fallback.problem
        assert notices and "not in a container" in notices[-1]
        assert (
            await backend.run("echo hi", timeout=10, conversation_id="c")
        ).stdout.strip() == "hi"


class TestToolsFollowTheBackend:
    @pytest.mark.asyncio
    async def test_a_tool_body_runs_on_this_machine_when_commands_do(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("here")
        backend = _auto(tmp_path)
        await backend.check()

        tool = {t.name: t for t in file_tools(Workspace.at(tmp_path), host=backend)}["read_file"]

        result = await tool.execute(path="a.txt", conversation_id="c")

        assert result.success and "here" in result.data
        assert backend.tool_host is None  # in this process, nothing to exec


class TestWhatIsExplicit:
    @pytest.mark.asyncio
    async def test_docker_still_refuses_when_unusable(self, tmp_path: Path) -> None:
        backend = build_backend(
            "docker",
            Workspace.at(tmp_path),
            docker=DockerBackendOptions(docker_binary="/no/such/docker"),
        )
        with pytest.raises(BackendUnavailableError):
            await backend.check()

    def test_local_is_local(self, tmp_path: Path) -> None:
        assert isinstance(build_backend("local", Workspace.at(tmp_path)), LocalBackend)

    def test_a_bundle_entry_defaults_to_auto(self, tmp_path: Path) -> None:
        entry = BuiltinBundleConfig(name="x", bundles=["shell"], workspace=str(tmp_path))
        assert entry.backend == "auto"
        assert isinstance(entry.build().backend, AutoBackend)

    def test_the_browser_goes_where_the_commands_go(self, tmp_path: Path) -> None:
        entry = BuiltinBundleConfig(name="x", bundles=["shell", "browser"], workspace=str(tmp_path))
        built = entry.build()
        assert built.browser is not None
        assert built.browser.backend is built.backend


class TestTheStateFolder:
    """Whichever backend is chosen keeps a conversation's state in the same place."""

    def test_it_is_handed_to_both(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = Workspace.at(tmp_path / "ws")
        (tmp_path / "ws").mkdir(exist_ok=True)
        state = tmp_path / "state"
        auto = build_backend("auto", ws, state_dir=state)
        assert isinstance(auto, AutoBackend)
        assert (
            auto.state_dir == str(state) == build_backend("docker", ws, state_dir=state).state_dir
        )

    def test_the_environment_is_read_once_when_none_is_given(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "ws").mkdir()
        monkeypatch.setenv("DOMYN_PROJECT_DIR", str(tmp_path / "proj"))
        auto = AutoBackend(Workspace.at(tmp_path / "ws"))
        monkeypatch.setenv("DOMYN_PROJECT_DIR", str(tmp_path / "other"))
        assert auto.state_dir == str(tmp_path / "proj")

    @pytest.mark.asyncio
    async def test_the_local_fallback_uses_it_too(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "ws").mkdir()
        state = tmp_path / "state"
        auto = AutoBackend(Workspace.at(tmp_path / "ws"), state_dir=state)

        async def refuse(self: Any) -> None:
            raise BackendUnavailableError("no docker")

        monkeypatch.setattr(
            "domyn.agents.tools.builtin.backends.docker.DockerBackend.check", refuse
        )
        await auto.decide()
        assert auto._on_this_machine and auto._local.state_dir == str(state)

    def test_without_any_it_is_dot_domyn(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "ws").mkdir()
        monkeypatch.delenv("DOMYN_PROJECT_DIR", raising=False)
        assert AutoBackend(Workspace.at(tmp_path / "ws")).state_dir == ".domyn"
