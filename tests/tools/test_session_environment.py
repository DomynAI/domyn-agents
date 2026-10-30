"""Tests for per-conversation working directories and their virtualenvs.

The problem these solve, found by asking where a `pip install` actually goes:
commands inherited the environment the CLI was launched from, so under
`uv run domyn` an agent installing a package wrote into *the framework's own*
virtualenv. It could break the runtime executing it, and it was never the
environment anybody thought they were working in.

Directories are named after the conversation id so they can be found again.
Most of what is pinned here is that the isolation is real from the *first*
command — the first version silently fell back to the ambient environment
because `ensurepip` aborts under a uv-managed interpreter, and only the first
command of each session was affected.
"""

import os
import subprocess
from pathlib import Path

import pytest

from domyn.agents.tools.builtin import Workspace
from domyn.agents.tools.builtin.session import (
    SessionEnvironment,
    SessionPaths,
    environment_for,
    restricted_environment,
)


class TestLayout:
    def test_a_conversation_gets_a_directory_named_after_its_id(self, tmp_path: Path) -> None:
        paths = SessionPaths(root=tmp_path, conversation_id="conv_abc123")
        assert paths.home == tmp_path / ".domyn" / "conversations" / "conv_abc123"

    def test_the_name_is_what_makes_it_findable(self, tmp_path: Path) -> None:
        """A folder that cannot be mapped back to a conversation is a folder
        nobody will ever open."""
        paths = SessionPaths(root=tmp_path, conversation_id="conv_abc123")
        assert "conv_abc123" in str(paths.home)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("../../etc/passwd", "______etc_passwd"),
            ("a/b", "a_b"),
            ("", "default"),
            ("ok-id_1", "ok-id_1"),
        ],
    )
    def test_an_id_cannot_place_the_directory_elsewhere(
        self, tmp_path: Path, raw: str, expected: str
    ) -> None:
        """Ids come from a caller — a gateway, a resumed session — and one
        containing `../` would otherwise put a virtualenv anywhere on disk."""
        paths = SessionPaths(root=tmp_path, conversation_id=raw)
        assert paths.home.name == expected
        assert tmp_path in paths.home.parents

    def test_two_conversations_do_not_share(self, tmp_path: Path) -> None:
        a = SessionPaths(root=tmp_path, conversation_id="one")
        b = SessionPaths(root=tmp_path, conversation_id="two")
        assert a.venv != b.venv


class TestBuildingTheEnvironment:
    def test_it_creates_a_usable_virtualenv(self, tmp_path: Path) -> None:
        paths = SessionEnvironment(tmp_path).prepare("c1")
        assert paths is not None
        assert paths.python.exists(), "the interpreter has to actually be there"

    def test_it_ships_a_pip(self, tmp_path: Path) -> None:
        """With the venv's bin first on PATH, a missing pip does not fail — it
        resolves to the next one along and installs somewhere else entirely,
        which is the exact outcome this module exists to prevent."""
        paths = SessionEnvironment(tmp_path).prepare("c1")
        assert paths is not None
        assert (paths.bin / "pip").exists() or (paths.bin / "pip.exe").exists()

    def test_the_first_call_is_already_isolated(self, tmp_path: Path) -> None:
        """The bug this replaced: the stdlib builder needs `ensurepip`, which
        aborts under a uv-managed interpreter, so the first command of every
        session quietly ran in the ambient environment."""
        assert SessionEnvironment(tmp_path).prepare("c1") is not None

    def test_it_is_built_once(self, tmp_path: Path) -> None:
        sessions = SessionEnvironment(tmp_path)
        first = sessions.prepare("c1")
        assert first is not None
        stamp = first.python.stat().st_mtime
        sessions.prepare("c1")
        assert first.python.stat().st_mtime == stamp

    def test_an_output_directory_is_ready(self, tmp_path: Path) -> None:
        paths = SessionEnvironment(tmp_path).prepare("c1")
        assert paths is not None and paths.output.is_dir()

    def test_it_is_kept_out_of_version_control(self, tmp_path: Path) -> None:
        """They live inside the project, which is what makes them findable —
        and puts a virtualenv one `git add -A` away from being committed."""
        SessionEnvironment(tmp_path).prepare("c1")
        marker = tmp_path / ".domyn" / "conversations" / ".gitignore"
        assert marker.is_file()
        assert "*" in marker.read_text()

    def test_an_existing_ignore_file_is_left_alone(self, tmp_path: Path) -> None:
        conversations = tmp_path / ".domyn" / "conversations"
        conversations.mkdir(parents=True)
        (conversations / ".gitignore").write_text("mine\n")
        SessionEnvironment(tmp_path).prepare("c1")
        assert (conversations / ".gitignore").read_text() == "mine\n"

    def test_failure_degrades_instead_of_blocking(self, tmp_path: Path) -> None:
        """Losing the isolated environment means running in the ambient one,
        which is how things worked before. Refusing to run the command at all
        would trade a small problem for a total one."""
        blocked = tmp_path / "file-not-a-dir"
        blocked.write_text("x")
        assert SessionEnvironment(blocked).prepare("c1") is None


class TestTheEnvironmentHandedToCommands:
    def _paths(self, tmp_path: Path) -> SessionPaths:
        return SessionPaths(root=tmp_path, conversation_id="c1")

    def test_the_venv_comes_first_on_path(self, tmp_path: Path) -> None:
        env = environment_for(self._paths(tmp_path), {"PATH": "/usr/bin"})
        assert env["PATH"].split(os.pathsep)[0] == str(self._paths(tmp_path).bin)

    def test_the_launching_virtualenv_is_removed_from_path(self, tmp_path: Path) -> None:
        """Prepending is not enough: the launching venv is also on PATH, and
        under `uv run` it comes first — `python` would keep resolving to the
        runtime the agent is supposed to be isolated from."""
        launcher = tmp_path / "launcher"
        env = environment_for(
            self._paths(tmp_path),
            {
                "VIRTUAL_ENV": str(launcher),
                "PATH": os.pathsep.join([str(launcher / "bin"), "/usr/bin"]),
            },
        )
        assert str(launcher / "bin") not in env["PATH"].split(os.pathsep)

    def test_virtual_env_points_at_the_conversation(self, tmp_path: Path) -> None:
        env = environment_for(self._paths(tmp_path), {})
        assert env["VIRTUAL_ENV"] == str(self._paths(tmp_path).venv)

    def test_variables_that_would_defeat_it_are_dropped(self, tmp_path: Path) -> None:
        env = environment_for(
            self._paths(tmp_path),
            {"PYTHONHOME": "/somewhere", "UV_PROJECT_ENVIRONMENT": "/elsewhere"},
        )
        assert "PYTHONHOME" not in env
        assert "UV_PROJECT_ENVIRONMENT" not in env

    def test_the_rest_of_the_environment_survives(self, tmp_path: Path) -> None:
        """A command still needs HOME, the user's tokens, everything else."""
        env = environment_for(self._paths(tmp_path), {"HOME": "/home/me", "MY_TOKEN": "x"})
        assert env["HOME"] == "/home/me"
        assert env["MY_TOKEN"] == "x"


class TestRestrictedEnvironment:
    """`run_command`'s subprocess must not inherit the whole process
    environment — that would hand a shell command every secret the CLI
    loaded from `.env` or `~/.domyn/credentials` (see `main.py`'s
    `_load_dotenv`/`_load_credentials`)."""

    def test_only_the_safe_set_survives_by_default(self) -> None:
        env = restricted_environment((), base={"PATH": "/usr/bin", "MY_SECRET": "s3cr3t"})
        assert env == {"PATH": "/usr/bin"}

    def test_an_allowed_name_is_let_through(self) -> None:
        env = restricted_environment(
            ["MY_SECRET"], base={"PATH": "/usr/bin", "MY_SECRET": "s3cr3t"}
        )
        assert env["MY_SECRET"] == "s3cr3t"

    def test_the_cli_project_folder_is_let_through(self) -> None:
        """So the agent can tell which configuration the session runs, and pass
        ``--project`` to a ``domyn`` command it runs."""
        env = restricted_environment((), base={"DOMYN_PROJECT_DIR": "/p", "MY_SECRET": "s"})
        assert env == {"DOMYN_PROJECT_DIR": "/p"}

    def test_a_name_that_is_not_present_is_just_absent(self) -> None:
        env = restricted_environment(["NEVER_SET"], base={"PATH": "/usr/bin"})
        assert "NEVER_SET" not in env


class TestItReallyIsolates:
    """The end of the chain: a package installed by one conversation."""

    def test_an_install_lands_in_the_conversation_and_nowhere_else(self, tmp_path: Path) -> None:
        sessions = SessionEnvironment(tmp_path)
        one = sessions.prepare("c1")
        two = sessions.prepare("c2")
        assert one is not None and two is not None

        subprocess.run(
            [str(one.python), "-m", "pip", "install", "-q", "cowsay"],
            check=True,
            capture_output=True,
            timeout=180,
        )

        def imports(paths: SessionPaths) -> bool:
            return (
                subprocess.run(
                    [str(paths.python), "-c", "import cowsay"], capture_output=True
                ).returncode
                == 0
            )

        assert imports(one), "the conversation that installed it should have it"
        assert not imports(two), "another conversation must not"


class TestThroughTheBackend:
    @pytest.mark.asyncio
    async def test_commands_run_inside_the_conversation_environment(self, tmp_path: Path) -> None:
        from domyn.agents.tools.builtin import LocalBackend

        backend = LocalBackend(Workspace.at(tmp_path))
        result = await backend.run('echo "$VIRTUAL_ENV"', timeout=60, conversation_id="c1")
        assert "conversations/c1/venv" in result.stdout

    @pytest.mark.asyncio
    async def test_isolation_can_be_turned_off(self, tmp_path: Path) -> None:
        """Wanted when the agent is meant to work *on* the project it was
        launched from."""
        from domyn.agents.tools.builtin import LocalBackend

        backend = LocalBackend(Workspace.at(tmp_path), isolate_environment=False)
        result = await backend.run('echo "$VIRTUAL_ENV"', timeout=60, conversation_id="c1")
        assert "conversations/c1" not in result.stdout

    @pytest.mark.asyncio
    async def test_no_conversation_means_the_ambient_environment(self, tmp_path: Path) -> None:
        """A caller outside a conversation — a script, a test — is not given a
        directory it never asked for."""
        from domyn.agents.tools.builtin import LocalBackend

        backend = LocalBackend(Workspace.at(tmp_path))
        await backend.run("true", timeout=60, conversation_id="")
        assert not (tmp_path / ".domyn").exists()

    def test_the_id_never_reaches_the_model(self) -> None:
        """It is injected by the runtime, like run_config and state."""
        import tempfile

        from domyn.agents.tools.builtin import bundle

        tools = {t.name: t for t in bundle("shell", root=tempfile.mkdtemp())}
        props = tools["run_command"].tool_definition["function"]["parameters"]["properties"]
        assert "conversation_id" not in props

    @pytest.mark.asyncio
    async def test_a_secret_in_the_process_environment_is_not_visible_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from domyn.agents.tools.builtin import LocalBackend

        monkeypatch.setenv("MY_SECRET", "s3cr3t")
        backend = LocalBackend(Workspace.at(tmp_path))
        result = await backend.run('echo "$MY_SECRET"', timeout=60, conversation_id="")
        assert "s3cr3t" not in result.stdout

    @pytest.mark.asyncio
    async def test_an_allowed_name_reaches_the_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from domyn.agents.tools.builtin import LocalBackend

        monkeypatch.setenv("MY_SECRET", "s3cr3t")
        backend = LocalBackend(Workspace.at(tmp_path), allowed_env=["MY_SECRET"])
        result = await backend.run('echo "$MY_SECRET"', timeout=60, conversation_id="")
        assert "s3cr3t" in result.stdout
