"""Tests for the command backend seam.

Only the local backend exists, so half of what is pinned here is the *shape*
rather than the behaviour: that a lifecycle exists and is called, that a backend
failure is distinguishable from a command failure, that an unknown backend name
refuses instead of quietly running on the host. Those are the three things a
container backend will rely on, and each of them is easy to get wrong later and
hard to notice.

The rest runs real commands. Mocking the subprocess would test the mock; the
parts worth checking — the working directory, a timeout actually killing
something, the exit code coming back — only exist in the real thing.
"""

# No `from __future__ import annotations`: this imports the tool modules, which
# rely on real annotations at construction time.
import asyncio
from pathlib import Path

import pytest

from domyn.agents.tools.builtin import (
    BackendUnavailableError,
    BuiltinTools,
    CommandBackend,
    CommandResult,
    LocalBackend,
    Workspace,
    build_backend,
    bundle,
)
from domyn.agents.tools.builtin.shell import render


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return Workspace.at(tmp_path)


class TestLocalBackend:
    @pytest.mark.asyncio
    async def test_it_returns_the_exit_code(self, workspace: Workspace) -> None:
        result = await LocalBackend(workspace).run("exit 7", timeout=10)
        assert result.exit_code == 7
        assert not result.timed_out
        assert result.backend_error is None

    @pytest.mark.asyncio
    async def test_it_separates_the_streams(self, workspace: Workspace) -> None:
        result = await LocalBackend(workspace).run("echo out; echo err >&2", timeout=10)
        assert "out" in result.stdout
        assert "err" in result.stderr

    @pytest.mark.asyncio
    async def test_it_runs_in_the_workspace(self, workspace: Workspace) -> None:
        result = await LocalBackend(workspace).run("pwd", timeout=10)
        assert str(workspace.root) in result.stdout

    @pytest.mark.asyncio
    async def test_a_timeout_is_reported_not_raised(self, workspace: Workspace) -> None:
        result = await LocalBackend(workspace).run("sleep 30", timeout=0.5)
        assert result.timed_out
        assert result.exit_code is None

    @pytest.mark.asyncio
    async def test_a_timed_out_process_is_killed(self, workspace: Workspace) -> None:
        """Abandoning it would leave something holding the port or the lock."""
        started = asyncio.get_running_loop().time()
        await LocalBackend(workspace).run("sleep 40", timeout=0.5)
        assert asyncio.get_running_loop().time() - started < 20

    @pytest.mark.asyncio
    async def test_the_whole_process_tree_goes(self, workspace: Workspace) -> None:
        """Killing only the shell left a child alive holding the pipes, and
        `wait()` then blocked on it: a one-second timeout returned after
        forty-seven seconds. Whether the shell forks decides whether it
        happens, which is why this only showed up in CI."""
        started = asyncio.get_running_loop().time()
        result = await LocalBackend(workspace).run("sleep 40 & wait", timeout=1.0)
        elapsed = asyncio.get_running_loop().time() - started
        assert result.timed_out
        assert elapsed < 20, f"waited {elapsed:.0f}s for the tree to die"

    @pytest.mark.asyncio
    async def test_nothing_is_left_running(self, workspace: Workspace) -> None:
        """The promise the old code made in a comment and did not keep.

        Polled rather than checked once: a killed process lingers as a zombie
        until its parent is reaped, and `pgrep` counts zombies. Asserting
        immediately would measure how fast the OS cleans up, which is not the
        property under test — what matters is that nothing keeps *running*.
        """
        import subprocess
        import uuid

        # A unique marker, because this greps the process table: a fixed string
        # can collide with another test's sleep and fail for a reason that has
        # nothing to do with this one.
        marker = f"domyn-orphan-{uuid.uuid4().hex[:8]}"
        await LocalBackend(workspace).run(f"sh -c 'sleep 41 # {marker}' & wait", timeout=1.0)

        survivors = "still here"
        for _ in range(30):
            survivors = subprocess.run(
                ["pgrep", "-f", marker], capture_output=True, text=True
            ).stdout.strip()
            if not survivors:
                break
            await asyncio.sleep(0.1)

        if survivors:
            subprocess.run(["pkill", "-f", marker], check=False)
        assert not survivors, f"orphans still alive after 3s: {survivors}"

    @pytest.mark.asyncio
    async def test_the_lifecycle_is_a_no_op(self, workspace: Workspace) -> None:
        """The local backend has nothing to set up, and must survive being told
        to anyway — including being closed without ever being started."""
        backend = LocalBackend(workspace)
        await backend.close()
        await backend.start()
        await backend.start()
        assert (await backend.run("echo ok", timeout=10)).exit_code == 0
        await backend.close()

    def test_it_does_not_claim_isolation(self, workspace: Workspace) -> None:
        assert LocalBackend(workspace).is_isolated is False


class TestResolvingABackend:
    def test_local_resolves(self, workspace: Workspace) -> None:
        assert isinstance(build_backend("local", workspace), LocalBackend)

    def test_an_unknown_name_refuses(self, workspace: Workspace) -> None:
        """Unlike an unknown tool bundle, which degrades with a warning. Handing
        a local backend to someone who asked to be sandboxed gives them the
        illusion of a boundary, which is worse than having none."""
        with pytest.raises(BackendUnavailableError, match="Refusing to fall back"):
            build_backend("docker", workspace)

    def test_the_bundle_refuses_too(self, tmp_path: Path) -> None:
        with pytest.raises(BackendUnavailableError):
            bundle("shell", root=tmp_path, backend="firecracker")

    def test_a_tool_bundle_refuses_too(self, tmp_path: Path) -> None:
        from domyn.agents.tools.builtin import BuiltinBundleConfig

        config = BuiltinBundleConfig(
            name="cli_tools", bundles=["shell"], workspace=str(tmp_path), sandbox="docker"
        )
        with pytest.raises(BackendUnavailableError):
            config.build()


class TestBackendFailuresAreDistinguishable:
    """ "The container is gone" must never be reported as "your command failed"."""

    @pytest.mark.asyncio
    async def test_a_backend_error_is_reported_as_such(self) -> None:
        result = CommandResult(
            exit_code=None, stdout="", stderr="", backend_error="the sandbox is gone"
        )
        rendered = render(result, timeout=10)
        assert "could not be run" in rendered
        assert "exit code" not in rendered

    @pytest.mark.asyncio
    async def test_a_timeout_says_what_to_do_next(self) -> None:
        rendered = render(
            CommandResult(exit_code=None, stdout="", stderr="", timed_out=True), timeout=5
        )
        assert "larger timeout" in rendered

    def test_an_ordinary_failure_keeps_its_exit_code(self) -> None:
        rendered = render(CommandResult(exit_code=2, stdout="", stderr="boom"), timeout=10)
        assert "exit code: 2" in rendered
        assert "boom" in rendered


class TestTheBundleOwnsTheBackend:
    def test_the_shell_bundle_builds_one(self, tmp_path: Path) -> None:
        built = bundle("shell", root=tmp_path)
        assert isinstance(built, BuiltinTools)
        assert built.backend is not None

    def test_the_files_bundle_needs_none(self, tmp_path: Path) -> None:
        """Nothing to release when nothing runs commands."""
        assert bundle("files", root=tmp_path).backend is None

    def test_one_backend_is_shared(self, tmp_path: Path) -> None:
        """Two backends would be two containers for one session."""
        built = bundle("files", "shell", root=tmp_path)
        assert built.backend is not None

    def test_it_still_iterates_as_a_list_of_tools(self, tmp_path: Path) -> None:
        """Callers that only want to mount them should not have to know about
        any of this."""
        built = bundle("files", root=tmp_path)
        assert len(list(built)) == len(built) > 0

    @pytest.mark.asyncio
    async def test_closing_with_no_backend_is_fine(self, tmp_path: Path) -> None:
        await bundle("files", root=tmp_path).close()


class TestTheSessionReleasesIt:
    """The reason the lifecycle is wired now rather than when it is needed:
    finding the end of a session later, under pressure, is how teardown gets
    forgotten."""

    class _Recording(CommandBackend):
        name = "recording"

        def __init__(self) -> None:
            self.closed = 0

        async def run(
            self,
            command: str,
            *,
            timeout: float,
            conversation_id: str = "",
            stdin: str | None = None,
        ) -> CommandResult:
            return CommandResult(exit_code=0, stdout="", stderr="")

        async def close(self) -> None:
            self.closed += 1

    def _repl_with(self, backend: CommandBackend, tmp_path: Path, *script: str):  # type: ignore[no-untyped-def]
        import io

        from rich.console import Console

        from domyn.agents.interfaces.cli import Theme
        from domyn.agents.interfaces.cli.repl import Repl
        from tests.cli.test_repl_session import ScriptedEditor, _agent

        agent = _agent()
        built = bundle("shell", root=tmp_path, backend=backend)
        agent.tools = list(built)
        agent.builtin_tools = [built]

        repl = Repl(
            agent,
            Console(file=io.StringIO(), width=100),
            Theme.build(allow_animation=False),
            enable_enhanced_input=False,
        )
        repl._input._session = ScriptedEditor(*script)
        return repl

    @pytest.mark.asyncio
    async def test_a_normal_exit_releases_it(self, tmp_path: Path) -> None:
        backend = self._Recording()
        await self._repl_with(backend, tmp_path, "/exit").run()
        assert backend.closed == 1

    @pytest.mark.asyncio
    async def test_end_of_input_releases_it(self, tmp_path: Path) -> None:
        backend = self._Recording()
        await self._repl_with(backend, tmp_path).run()
        assert backend.closed == 1

    @pytest.mark.asyncio
    async def test_a_crash_still_releases_it(self, tmp_path: Path) -> None:
        """A container leaked on every crash is a container still running
        tomorrow."""
        backend = self._Recording()
        repl = self._repl_with(backend, tmp_path)

        async def boom(initial_prompt: str | None = None) -> int:
            raise RuntimeError("the session broke")

        repl._loop = boom  # type: ignore[method-assign]
        with pytest.raises(RuntimeError):
            await repl.run()
        assert backend.closed == 1

    @pytest.mark.asyncio
    async def test_a_failing_teardown_does_not_mask_the_real_error(self, tmp_path: Path) -> None:
        """Raising from a finally would replace the reason the session ended."""

        class Broken(self._Recording):  # type: ignore[name-defined,misc]
            async def close(self) -> None:
                raise RuntimeError("teardown failed")

        repl = self._repl_with(Broken(), tmp_path)

        async def boom(initial_prompt: str | None = None) -> int:
            raise ValueError("the actual problem")

        repl._loop = boom  # type: ignore[method-assign]
        with pytest.raises(ValueError, match="the actual problem"):
            await repl.run()

    @pytest.mark.asyncio
    async def test_every_bundle_on_one_agent_is_released(self, tmp_path: Path) -> None:
        """An agent can declare more than one `tool_bundles` entry — all of
        them need tearing down, not just the first."""
        from tests.cli.test_repl_session import _agent

        first, second = self._Recording(), self._Recording()
        agent = _agent()
        agent.builtin_tools = [
            bundle("shell", root=tmp_path, backend=first),
            bundle("shell", root=tmp_path, backend=second),
        ]

        await agent.close()

        assert first.closed == 1
        assert second.closed == 1

    @pytest.mark.asyncio
    async def test_a_sub_agents_bundle_is_also_released(self, tmp_path: Path) -> None:
        """`Repl.close()` only ever held the root agent's own bundle — a
        sub-agent's sandboxed shell never got torn down."""
        from tests.cli.test_repl_session import _agent

        root_backend, sub_backend = self._Recording(), self._Recording()
        sub = _agent()
        sub.builtin_tools = [bundle("shell", root=tmp_path, backend=sub_backend)]
        root = _agent()
        root.builtin_tools = [bundle("shell", root=tmp_path, backend=root_backend)]
        root.sub_agents = [sub]

        await root.close()

        assert root_backend.closed == 1
        assert sub_backend.closed == 1


class TestStdinIsNotInherited:
    """The child must never read the parent's stdin.

    Left unset, ``create_subprocess_shell`` hands the child whatever this
    process is reading from. In the CLI that is the terminal the user is typing
    into, so a command asking "continue? [y/N]" races prompt_toolkit for their
    keystrokes — it does not hang, it *steals*, which is worse because it looks
    like the user mistyped.

    This is checked end to end rather than by asserting on the call: the whole
    claim is about a file descriptor being inherited, and only a real parent
    with a real stdin can show it. So the backend runs inside a child process
    whose stdin is a pipe with a known secret in it.
    """

    @pytest.mark.asyncio
    async def test_the_child_cannot_read_the_parents_stdin(self, tmp_path: Path) -> None:
        import subprocess
        import sys

        script = (
            "import asyncio, sys\n"
            f"sys.path.insert(0, {str(Path.cwd())!r})\n"
            "from domyn.agents.tools.builtin.backends import LocalBackend\n"
            "from domyn.agents.tools.builtin.workspace import Workspace\n"
            f"b = LocalBackend(Workspace.at({str(tmp_path)!r}), isolate_environment=False)\n"
            "r = asyncio.run(b.run('cat', timeout=10))\n"
            "print('CHILD_SAW:' + r.stdout.strip())\n"
        )

        done = subprocess.run(
            [sys.executable, "-c", script],
            input=b"SECRET-OF-THE-PARENT\n",
            capture_output=True,
            timeout=60,
        )
        assert done.returncode == 0, done.stderr.decode()
        assert b"CHILD_SAW:" in done.stdout
        assert b"SECRET" not in done.stdout

    @pytest.mark.asyncio
    async def test_a_command_reading_input_gets_eof_instead_of_hanging(
        self, workspace: Workspace
    ) -> None:
        # Without DEVNULL this is the case that blocks until the timeout kills
        # it; with it, `cat` sees EOF at once and exits cleanly.
        result = await LocalBackend(workspace).run("cat", timeout=10)
        assert result.exit_code == 0
        assert not result.timed_out
        assert result.stdout == ""


class TestSuppliedStdin:
    """Answers given up front, which is how a prompt gets answered with no
    terminal: `stdin` is the whole input the command will ever get."""

    @pytest.mark.asyncio
    async def test_it_reaches_the_command(self, workspace: Workspace) -> None:
        result = await LocalBackend(workspace).run("cat", timeout=10, stdin="hello\n")
        assert result.stdout.strip() == "hello"

    @pytest.mark.asyncio
    async def test_several_answers_arrive_in_order(self, workspace: Workspace) -> None:
        result = await LocalBackend(workspace).run(
            'read -r a; read -r b; echo "$b then $a"', timeout=10, stdin="first\nsecond\n"
        )
        assert result.stdout.strip() == "second then first"

    @pytest.mark.asyncio
    async def test_it_answers_a_yes_no_prompt(self, workspace: Workspace) -> None:
        # The shape of the case this exists for: a command with no --yes flag.
        script = 'printf "continue? [y/N] "; read -r answer; test "$answer" = y && echo proceeding'
        result = await LocalBackend(workspace).run(script, timeout=10, stdin="y\n")
        assert "proceeding" in result.stdout

    @pytest.mark.asyncio
    async def test_the_tool_hands_it_to_the_command(self, tmp_path: Path) -> None:
        """End to end through the tool, because the path changed underneath it.

        ``run_command`` no longer passes ``stdin`` down to ``CommandBackend.run``:
        the command it builds detaches, so the caller is gone by the time the
        process reads anything and there is no pipe left to write into. The input
        is written to a file the detached command reads instead. The backend
        parameter stays — it is a real capability, and the pod accepts one — but
        it is not what answers a prompt any more.
        """
        from domyn.agents.tools.builtin import bundle

        tools = {t.name: t for t in bundle("shell", root=tmp_path)}
        script = 'printf "continue? [y/N] "; read -r a; test "$a" = y && echo proceeding'
        result = await tools["run_command"].execute(command=script, stdin="y\n")
        assert "proceeding" in str(result.data)

    @pytest.mark.asyncio
    async def test_without_it_the_command_is_not_left_waiting(self, tmp_path: Path) -> None:
        from domyn.agents.tools.builtin import bundle

        tools = {t.name: t for t in bundle("shell", root=tmp_path)}
        result = await tools["run_command"].execute(command="cat; echo done", timeout=10)
        # EOF, not a wait: it must finish rather than yield a handle.
        assert "done" in str(result.data)
        assert "still running" not in str(result.data)


class TestTheTimeoutMessageSaysWhatToDo:
    def test_it_names_both_ways_out(self) -> None:
        text = render(
            CommandResult(exit_code=None, stdout="", stderr="", timed_out=True), timeout=120
        )
        # A model that reads "it cannot be run here" gives up; one that reads
        # this retries in a way that works.
        assert "stdin" in text
        assert "yes |" in text
