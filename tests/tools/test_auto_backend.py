"""``sandbox: auto``: Docker when it can be used, this machine when it cannot, and
the fallback is said rather than silent. An explicit ``docker`` still refuses."""

from pathlib import Path
from typing import Any

import pytest

from domyn.agents.tools.builtin import (
    BuiltinBundleConfig,
    DockerBackendOptions,
    LocalBackend,
    Workspace,
)
from domyn.agents.tools.builtin.auto_backend import AutoBackend
from domyn.agents.tools.builtin.backend_factory import build_backend
from domyn.agents.tools.builtin.backends import BackendUnavailableError, CommandResult
from domyn.agents.tools.builtin.sandbox import SandboxBackend


class _Fake(SandboxBackend):
    name = "docker"
    is_isolated = True
    destroys_on_close = True

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def check(self) -> None:
        self.calls.append("check")

    async def prepare(self) -> None:
        self.calls.append("prepare")

    async def warm_up(self, conversation_id: str) -> None:
        self.calls.append(f"warm {conversation_id}")

    async def run(self, command: str, **kwargs: Any) -> CommandResult:
        self.calls.append(f"run {command}")
        return CommandResult(exit_code=0, stdout="in the container", stderr="")

    async def end_conversation(self, conversation_id: str) -> None:
        self.calls.append(f"end {conversation_id}")

    async def close(self) -> None:
        self.calls.append("close")


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

        assert isinstance(backend._chosen, LocalBackend)
        assert backend.fallback is not None and "Docker was not found" in backend.fallback.problem
        assert backend.is_isolated is False and backend.name == "local"
        assert len(notices) == 1 and "no sandbox" in notices[0]

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
    async def test_with_docker_it_is_the_sandbox(self, tmp_path: Path) -> None:
        fake = _Fake()
        backend = AutoBackend(Workspace.at(tmp_path), sandbox=fake)

        await backend.check()
        assert backend.fallback is None
        assert backend.is_isolated and backend.destroys_on_close and backend.name == "docker"
        assert (await backend.run("ls", timeout=5)).stdout == "in the container"
        await backend.end_conversation("c")
        await backend.close()
        assert fake.calls == ["check", "run ls", "end c", "close"]

    @pytest.mark.asyncio
    async def test_before_deciding_it_is_not_claimed_to_isolate(self, tmp_path: Path) -> None:
        backend = _auto(tmp_path)
        assert backend.is_isolated is False
        assert isinstance(backend, SandboxBackend)

    @pytest.mark.asyncio
    async def test_closing_an_undecided_backend_is_fine(self, tmp_path: Path) -> None:
        await _auto(tmp_path).close()


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
        assert entry.sandbox == "auto"
        assert isinstance(entry.build().backend, AutoBackend)

    def test_the_browser_keeps_its_own_backend_beside_an_auto_one(self, tmp_path: Path) -> None:
        entry = BuiltinBundleConfig(name="x", bundles=["shell", "browser"], workspace=str(tmp_path))
        built = entry.build()
        assert built.browser is not None
        assert isinstance(built.browser.backend, LocalBackend)
        assert built.browser.backend is not built.backend


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

        monkeypatch.setattr("domyn.agents.tools.builtin.docker_sandbox.DockerBackend.check", refuse)
        chosen = await auto.decide()
        assert isinstance(chosen, LocalBackend) and chosen.state_dir == str(state)

    def test_without_any_it_is_dot_domyn(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "ws").mkdir()
        monkeypatch.delenv("DOMYN_PROJECT_DIR", raising=False)
        assert AutoBackend(Workspace.at(tmp_path / "ws")).state_dir == ".domyn"
