"""What the REPL does for a sandboxed command backend.

The backend itself is tested in ``tests/tools/test_docker_backend.py``; here it is
a fake, so the session's side — preparing under a message, warming the next
conversation's container, destroying the last one's, saying what happens to
running commands — is checked without Docker.
"""

import asyncio
import io
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from rich.console import Console

from domyn.agents.interfaces.cli import Theme
from domyn.agents.tools.builtin import (
    BackendUnavailableError,
    BuiltinTools,
    CommandBackend,
    CommandResult,
)
from tests.cli.test_repl_session import _agent, _repl


class FakeSandbox(CommandBackend):
    name = "fake-sandbox"
    is_isolated = True
    destroys_on_close = True

    def __init__(self, *, fail_prepare: bool = False) -> None:
        self.events: list[tuple[str, str]] = []
        self.fail_prepare = fail_prepare

    async def run(
        self, command: str, *, timeout: float, conversation_id: str = "", stdin: str | None = None
    ) -> CommandResult:
        return CommandResult(exit_code=0, stdout="", stderr="")

    async def prepare(self) -> None:
        self.events.append(("prepare", ""))
        if self.fail_prepare:
            raise RuntimeError("Docker is not running")
        if self.on_notice:
            self.on_notice("Preparing the sandbox image (first run only)…")

    async def warm_up(self, conversation_id: str) -> None:
        self.events.append(("warm_up", conversation_id))

    async def end_conversation(self, conversation_id: str) -> None:
        self.events.append(("end", conversation_id))

    async def close(self) -> None:
        self.events.append(("close", ""))


def _with_sandbox(*script: str, **kwargs: Any):  # type: ignore[no-untyped-def]
    backend = FakeSandbox(**kwargs)
    repl, out = _repl(*script)
    repl._agent.builtin_tools.append(BuiltinTools(backend=backend))
    repl._route_browser_notices()
    return repl, out, backend


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


class TestPreparing:
    @pytest.mark.asyncio
    async def test_the_sandbox_is_prepared_before_the_first_prompt_and_says_so(self) -> None:
        repl, out, backend = _with_sandbox()
        first = repl._state.conversation_id
        await repl.run()
        await _settle()
        assert backend.events[0] == ("prepare", "")
        assert ("warm_up", first) in backend.events
        assert "Preparing the sandbox image" in out.getvalue()

    @pytest.mark.asyncio
    async def test_a_failure_is_reported_and_the_session_carries_on(self) -> None:
        repl, out, backend = _with_sandbox(fail_prepare=True)
        assert await repl.run() == 0
        assert "Sandbox unavailable: Docker is not running" in out.getvalue()
        assert not any(kind == "warm_up" for kind, _ in backend.events)

    @pytest.mark.asyncio
    async def test_a_session_without_a_sandbox_is_untouched(self) -> None:
        repl, out = _repl()
        assert await repl.run() == 0
        assert "Sandbox" not in out.getvalue()


class TestConversationsEnd:
    @pytest.mark.asyncio
    async def test_clear_destroys_the_old_container_and_warms_the_new_one(self) -> None:
        repl, _, backend = _with_sandbox("/clear")
        old = repl._state.conversation_id
        await repl.run()
        await _settle()
        new = repl._state.conversation_id
        assert new != old
        assert ("end", old) in backend.events
        assert ("warm_up", new) in backend.events
        assert backend.events.index(("end", old)) < backend.events.index(("warm_up", new))

    @pytest.mark.asyncio
    async def test_the_session_closing_closes_the_backend(self) -> None:
        repl, _, backend = _with_sandbox()
        await repl.run()
        assert ("close", "") in backend.events


class TestWhatSurvives:
    @pytest.mark.asyncio
    async def test_running_commands_are_said_to_stop_with_the_sandbox(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from domyn.agents.interfaces.cli import processes_view

        monkeypatch.setattr(processes_view, "summary", lambda state: "1 command is running")
        repl, out, _ = _with_sandbox()
        await repl.close()
        text = out.getvalue()
        assert "will be stopped" in text and "will keep going" not in text

    @pytest.mark.asyncio
    async def test_on_this_machine_they_keep_going(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from domyn.agents.interfaces.cli import processes_view

        monkeypatch.setattr(processes_view, "summary", lambda state: "1 command is running")
        repl, out = _repl()
        await repl.close()
        assert "will keep going" in out.getvalue()


class _Down(FakeSandbox):
    """A sandbox whose Docker is not running."""

    def __init__(self, *, fix: str | None = "Start Docker Desktop (`open -a Docker`).") -> None:
        super().__init__()
        self._fix = fix
        self.checked = 0

    async def check(self) -> None:
        self.checked += 1
        raise BackendUnavailableError("Docker is installed but not running.", fix=self._fix)


class TestStartingWithoutDocker:
    """`domyn run` refuses to start when the configured sandbox cannot be used, and says
    both ways out: start Docker, or switch to running commands locally."""

    async def _start(self, backend: CommandBackend, config_file: Path | None) -> tuple[int, str]:
        from domyn.agents.cli.commands.run import command as run_command

        buffer = io.StringIO()
        agent = _agent()
        agent.builtin_tools.append(BuiltinTools(backend=backend))
        with patch.object(run_command, "console", Console(file=buffer, width=100)):
            code = await run_command.start_session(
                agent,
                Theme.build(allow_animation=False),
                initial_prompt=None,
                verbose=False,
                show_status=False,
                config_file=config_file,
            )
        return code, buffer.getvalue()

    @pytest.mark.asyncio
    async def test_it_stops_and_names_both_ways_out(self, tmp_path: Path) -> None:
        backend = _Down()
        config = tmp_path / "config.yaml"

        code, text = await self._start(backend, config)

        assert code == 1
        assert backend.checked == 1
        assert "Docker is installed but not running." in text
        assert "open -a Docker" in text and "run `domyn run` again" in text
        assert "`sandbox: local`" in text and "cli tools bundle" in text
        # Unwrapped: a long path wraps at the width of the console.
        flat = text.replace("\n", "")
        assert "sandbox: local" in text and str(config) in flat

    @pytest.mark.asyncio
    async def test_a_file_less_agent_is_told_to_edit_its_config(self) -> None:
        _, text = await self._start(_Down(), None)

        assert "sandbox: local" in text
        assert " in /" not in text

    @pytest.mark.asyncio
    async def test_no_repl_starts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from domyn.agents.cli.commands.run import command as run_command

        def boom(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("a session was started without a sandbox")

        monkeypatch.setattr(run_command, "Repl", boom)

        code, _ = await self._start(_Down(), None)

        assert code == 1

    @pytest.mark.asyncio
    async def test_a_usable_sandbox_lets_the_session_start(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from domyn.agents.cli.commands.run import command as run_command

        started: list[bool] = []

        class _Stub:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            async def run(self, initial_prompt: str | None) -> int:
                started.append(True)
                return 0

        monkeypatch.setattr(run_command, "Repl", _Stub)

        code, _ = await self._start(FakeSandbox(), None)

        assert code == 0 and started == [True]

    @pytest.mark.asyncio
    async def test_a_session_without_a_sandbox_is_not_asked(self) -> None:
        from domyn.agents.cli.commands.run import command as run_command

        assert await run_command._sandbox_is_usable(
            _agent(), Theme.build(allow_animation=False), None
        )


class TestStartingWithAutoSandbox:
    """`sandbox: auto` starts either way, says where commands run, and warns on a fallback."""

    async def _start(self, backend: CommandBackend, show_status: bool) -> tuple[int, str]:
        from domyn.agents.cli.commands.run import command as run_command

        buffer = io.StringIO()
        agent = _agent()
        agent.builtin_tools.append(BuiltinTools(backend=backend))
        with patch.object(run_command, "console", Console(file=buffer, width=120)):
            code = await run_command.start_session(
                agent,
                Theme.build(allow_animation=False),
                initial_prompt=None,
                verbose=False,
                show_status=show_status,
                config_file=Path("/x/config.yaml"),
            )
        return code, buffer.getvalue()

    @pytest.mark.asyncio
    async def test_a_fallback_warns_and_the_session_starts(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from domyn.agents.cli.commands.run import command as run_command
        from domyn.agents.tools.builtin import DockerBackendOptions, Workspace
        from domyn.agents.tools.builtin.backends import AutoBackend

        started: list[bool] = []

        class _Repl:
            def __init__(self, *a: Any, **k: Any) -> None: ...

            async def run(self, prompt: Any) -> int:
                started.append(True)
                return 0

        monkeypatch.setattr(run_command, "Repl", _Repl)
        backend = AutoBackend(
            Workspace.at(tmp_path), docker=DockerBackendOptions(docker_binary="/no/such/docker")
        )

        code, text = await self._start(backend, show_status=True)

        assert code == 0 and started == [True]
        flat = text.replace("\n", " ")
        assert "Docker was not found" in text and "no sandbox" in text
        assert "sandbox: local" in flat
        assert "commands: this machine (no sandbox)" in flat

    @pytest.mark.asyncio
    async def test_a_usable_docker_shows_the_sandbox_and_no_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from domyn.agents.cli.commands.run import command as run_command

        class _Repl:
            def __init__(self, *a: Any, **k: Any) -> None: ...

            async def run(self, prompt: Any) -> int:
                return 0

        monkeypatch.setattr(run_command, "Repl", _Repl)
        code, text = await self._start(FakeSandbox(), show_status=True)

        assert code == 0
        assert "commands: docker sandbox" in text
        assert "no sandbox" not in text

    @pytest.mark.asyncio
    async def test_the_browser_is_said_to_be_outside_the_sandbox(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from types import SimpleNamespace

        from domyn.agents.cli.commands.run import command as run_command

        class _Repl:
            def __init__(self, *a: Any, **k: Any) -> None: ...

            async def run(self, prompt: Any) -> int:
                return 0

        monkeypatch.setattr(run_command, "Repl", _Repl)
        buffer = io.StringIO()
        agent = _agent()
        agent.builtin_tools.append(BuiltinTools(backend=FakeSandbox(), browser=SimpleNamespace()))
        with patch.object(run_command, "console", Console(file=buffer, width=140)):
            await run_command.start_session(
                agent,
                Theme.build(allow_animation=False),
                initial_prompt=None,
                verbose=False,
                show_status=True,
            )
        assert "commands: docker sandbox  ·  browser: this machine" in buffer.getvalue()
