"""The SSH backend.

Two kinds of test. Most run anywhere: the command line given to ``ssh``, how a result is
read, the glob matching, the routing of a path, the skills' copies. The rest put a fake
``ssh`` on the way, which runs the line it is given here, in ``sh``, as a server's login
shell would: the file tools then run through ``RemoteDisk`` on files of this machine,
and must answer exactly what the local tools answer on the same files. Those need the
GNU tools the server needs (``find -printf``, ``realpath -m``), so they are skipped where
they are missing, as on macOS.
"""

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest

from domyn.agents.tools.builtin import Workspace, bundle
from domyn.agents.tools.builtin.backends import (
    BackendUnavailableError,
    SshBackend,
    SshBackendOptions,
    build_backend,
)
from domyn.agents.tools.builtin.backends.ssh.connection import Output, SshConnection
from domyn.agents.tools.builtin.backends.ssh.disk import _match
from domyn.agents.tools.builtin.backends.ssh.skills import SkillCopies
from domyn.agents.tools.builtin.files import build_tools, on_this_machine


def _options(**overrides: Any) -> SshBackendOptions:
    return SshBackendOptions(**{"host": "box", "workspace": "/srv/work", **overrides})


class TestTheOptions:
    def test_the_workspace_is_an_absolute_path_on_the_server(self) -> None:
        with pytest.raises(ValueError, match="absolute"):
            SshBackendOptions(host="box", workspace="work")

    def test_a_tilde_is_refused_since_a_quoted_path_never_expands_it(self) -> None:
        with pytest.raises(ValueError, match="without '~'"):
            SshBackendOptions(host="box", workspace="~/work")
        with pytest.raises(ValueError, match="without '~'"):
            SshBackendOptions(host="box", workspace="/w", state_dir="~/state")

    def test_unknown_keys_are_refused(self) -> None:
        with pytest.raises(ValueError):
            SshBackendOptions.model_validate({"host": "box", "hots": "x"})

    def test_a_backend_without_a_server_or_a_folder_is_refused(self, tmp_path: Path) -> None:
        space = Workspace.at(tmp_path)
        with pytest.raises(BackendUnavailableError, match="host"):
            build_backend("ssh", space, ssh=SshBackendOptions(workspace="/w"))
        with pytest.raises(BackendUnavailableError, match="workspace"):
            build_backend("ssh", space, ssh=SshBackendOptions(host="box"))

    def test_the_factory_builds_one(self, tmp_path: Path) -> None:
        backend = build_backend("ssh", Workspace.at(tmp_path), ssh=_options())
        assert isinstance(backend, SshBackend)
        assert backend.host == "box"


class TestTheCommandLine:
    @pytest.fixture(autouse=True)
    def _ssh_on_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

    def test_nothing_of_this_machine_is_forwarded_and_nothing_is_asked(self) -> None:
        argv = SshConnection(_options()).argv("true")
        flags = " ".join(argv)
        for option in (
            "BatchMode=yes",
            "ForwardAgent=no",
            "ForwardX11=no",
            "ClearAllForwardings=yes",
            "PermitLocalCommand=no",
            "ControlMaster=auto",
        ):
            assert option in flags
        assert argv[-3:] == ["--", "box", "true"]

    def test_the_connection_fields_are_passed(self, tmp_path: Path) -> None:
        argv = SshConnection(
            _options(user="me", port=2222, identity_file=str(tmp_path / "key"), config_file="/c")
        ).argv()
        assert argv[argv.index("-l") + 1] == "me"
        assert argv[argv.index("-p") + 1] == "2222"
        assert argv[argv.index("-i") + 1] == str(tmp_path / "key")
        assert "IdentitiesOnly=yes" in argv
        assert argv[argv.index("-F") + 1] == "/c"

    def test_a_host_that_reads_as_an_option_is_refused(self) -> None:
        with pytest.raises(BackendUnavailableError):
            SshConnection(_options(host="-oProxyCommand=x"))

    def test_the_socket_path_fits_the_budget_of_a_socket(self) -> None:
        # `%C` expands to 40 hex characters.
        socket = SshConnection(_options()).control_dir / ("x" * 40)
        assert len(str(socket)) < 100


class _Connection:
    """A connection that answers from a script."""

    def __init__(self, output: Output | None, *, alive: bool = True) -> None:
        self.output = output
        self.is_alive = alive
        self.sent: list[str] = []

    async def run(self, remote: str, *, stdin: bytes | None, timeout: float) -> Output | None:
        self.sent.append(remote)
        return self.output

    async def alive(self) -> bool:
        return self.is_alive


def _started(
    tmp_path: Path, connection: _Connection, *, skills: tuple[Path, ...] = ()
) -> SshBackend:
    backend = SshBackend(Workspace.at(tmp_path / "here", also=skills), options=_options())
    backend._connection = connection  # type: ignore[assignment]
    backend._root, backend._home = "/srv/work", "/home/me"
    return backend


class TestARunsResult:
    @pytest.mark.asyncio
    async def test_a_command_runs_in_the_workspace_with_its_limit_on_the_server(
        self, tmp_path: Path
    ) -> None:
        connection = _Connection(Output(3, b"out", b"err"))
        result = await _started(tmp_path, connection).run("make test", timeout=30)
        assert (result.exit_code, result.stdout, result.stderr) == (3, "out", "err")
        assert connection.sent == ["cd /srv/work && exec timeout -s KILL 30 bash -c 'make test'"]

    @pytest.mark.asyncio
    async def test_ssh_still_there_after_the_limit_is_a_timeout(self, tmp_path: Path) -> None:
        result = await _started(tmp_path, _Connection(None)).run("sleep 9", timeout=1)
        assert result.timed_out

    @pytest.mark.asyncio
    async def test_255_on_a_lost_connection_is_the_backends_failure(self, tmp_path: Path) -> None:
        connection = _Connection(Output(255, b"", b"Connection reset"), alive=False)
        result = await _started(tmp_path, connection).run("true", timeout=5)
        assert result.backend_error is not None and "Connection reset" in result.backend_error
        assert result.exit_code is None

    @pytest.mark.asyncio
    async def test_255_on_a_live_connection_is_the_commands_own(self, tmp_path: Path) -> None:
        result = await _started(tmp_path, _Connection(Output(255, b"", b""))).run(
            "exit 255", timeout=5
        )
        assert result.exit_code == 255 and result.backend_error is None

    @pytest.mark.asyncio
    async def test_a_backend_that_cannot_start_says_why_as_a_result(self, tmp_path: Path) -> None:
        backend = SshBackend(Workspace.at(tmp_path), options=_options())

        async def refuse() -> None:
            raise BackendUnavailableError("could not connect to box")

        backend.start = refuse  # type: ignore[method-assign]
        result = await backend.run("true", timeout=5)
        assert result.backend_error == "could not connect to box"

    def test_the_state_folder_is_outside_the_workspace_by_default(self, tmp_path: Path) -> None:
        state = _started(tmp_path, _Connection(None)).state_dir
        assert state.startswith("/home/me/.cache/domyn/")
        backend = SshBackend(Workspace.at(tmp_path), options=_options(state_dir="/var/s"))
        assert backend.state_dir == "/var/s"


class TestTheGlob:
    @pytest.mark.parametrize(
        ("names", "pattern", "matches"),
        [
            (("a.py",), ("*.py",), True),
            (("src", "a.py"), ("*.py",), False),
            (("src", "a.py"), ("**", "*.py"), True),
            (("a.py",), ("**", "*.py"), True),
            (("src", "lib", "a.py"), ("src", "**", "*.py"), True),
            (("src", "a.py"), ("src", "**", "*.py"), True),
            (("src", "a.ts"), ("src", "**", "*.py"), False),
            ((".hidden",), ("*",), True),
            (("A.PY",), ("*.py",), False),
        ],
    )
    def test_it_matches_as_pathlib_does(
        self, names: tuple[str, ...], pattern: tuple[str, ...], matches: bool
    ) -> None:
        assert _match(names, pattern) is matches


class TestWhatIsReadHere:
    def test_a_path_in_a_skill_folder_is_this_machines(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        (skills / "demo").mkdir(parents=True)
        space = Workspace.at(tmp_path / "work", also=[skills])
        assert on_this_machine(str(skills / "demo" / "SKILL.md"), space)

    def test_any_other_path_is_the_servers(self, tmp_path: Path) -> None:
        space = Workspace.at(tmp_path / "work", also=[tmp_path / "skills"])
        assert not on_this_machine("src/a.py", space)
        assert not on_this_machine("/etc/hosts", space)
        assert not on_this_machine(None, space)
        # The workspace of this machine is not reachable over ssh: only its skills are.
        assert not on_this_machine(str(tmp_path / "work" / "a.py"), space)


class TestTheSkillsCopies:
    def test_a_command_names_the_copy_of_a_skill_folder(self, tmp_path: Path) -> None:
        copies = SkillCopies([tmp_path / "skills"])
        copy = copies.destination(tmp_path / "skills", "/state/skills")
        command = f"cd '{tmp_path}/skills/demo' && python {tmp_path}/skills/demo/run.py"
        assert copies.rewrite(command, "/state/skills") == (
            f"cd '{copy}/demo' && python {copy}/demo/run.py"
        )

    def test_a_folder_inside_another_is_replaced_first(self, tmp_path: Path) -> None:
        outer, inner = tmp_path / "s", tmp_path / "s" / "inner"
        copies = SkillCopies([outer, inner])
        rewritten = copies.rewrite(f"cat {inner}/x", "/b")
        assert rewritten == f"cat {copies.destination(inner, '/b')}/x"

    def test_a_skill_folder_written_from_home_is_replaced_too(self, tmp_path: Path) -> None:
        root = Path.home() / ".domyn-test-skills"
        copies = SkillCopies([root])
        assert copies.rewrite("ls ~/.domyn-test-skills/x", "/b") == (
            f"ls {copies.destination(root, '/b')}/x"
        )

    def test_the_copy_has_a_name_that_needs_no_quoting(self, tmp_path: Path) -> None:
        copy = SkillCopies([]).destination(tmp_path / "My Skills", "/b")
        assert copy.removeprefix("/b/").isalnum()


# --- through a fake ssh, on the files of this machine -----------------------------------

FAKE_SSH = """#!/bin/sh
# Every option of ours takes a value but -T; -O is a control command.
control=""
while [ $# -gt 0 ]; do
  case "$1" in
    -T) shift ;;
    -O) control=$2; shift 2 ;;
    --) shift; break ;;
    *) shift 2 ;;
  esac
done
shift
[ -n "$control" ] && exit 0
cd "$HOME" && exec sh -c "$1"
"""


def _gnu() -> bool:
    try:
        find = subprocess.run(
            ["find", "/", "-maxdepth", "0", "-printf", ""], capture_output=True, check=False
        )
        real = subprocess.run(["realpath", "-m", "-z", "/"], capture_output=True, check=False)
    except OSError:
        return False
    return find.returncode == 0 and real.returncode == 0 and shutil.which("timeout") is not None


needs_gnu = pytest.mark.skipif(
    sys.platform == "win32" or not _gnu(), reason="needs the GNU tools a server has"
)


@pytest.fixture
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """A "server": a home folder and a workspace on this machine, reached through a fake
    ``ssh`` that runs what it is given there."""
    home = tmp_path / "server-home"
    work = home / "work"
    work.mkdir(parents=True)
    binary = tmp_path / "bin" / "ssh"
    binary.parent.mkdir()
    binary.write_text(FAKE_SSH)
    binary.chmod(0o755)
    monkeypatch.setenv("HOME", str(home))
    (work / "src").mkdir()
    (work / "src" / "a.py").write_text("import os\n# TODO one\nprint('a')\n")
    (work / "src" / "b.txt").write_text("TODO two\n")
    (work / "node_modules" / "x").mkdir(parents=True)
    (work / "node_modules" / "x" / "i.py").write_text("# TODO hidden\n")
    (work / "long.txt").write_text("".join(f"{n}\n" for n in range(1, 2001)))
    (tmp_path / "outside.txt").write_text("secret\n")
    (work / "out.txt").symlink_to(tmp_path / "outside.txt")
    return {"home": home, "work": work.resolve(), "bin": binary, "root": tmp_path}


def _backend(server: dict[str, Path], *, skills: tuple[Path, ...] = ()) -> SshBackend:
    return SshBackend(
        Workspace.at(server["root"] / "here", also=skills),
        options=SshBackendOptions(
            host="fake", workspace=str(server["work"]), binary=str(server["bin"])
        ),
    )


def _answer(func: Any, **arguments: Any) -> str:
    try:
        return str(func(**arguments))
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


@needs_gnu
class TestTheFileToolsOnTheServer:
    CALLS: ClassVar[list[tuple[str, dict[str, Any]]]] = [
        ("read_file", {"path": "src/a.py"}),
        ("read_file", {"path": "long.txt", "start_line": 1990}),
        ("read_file", {"path": "long.txt", "max_lines": 3}),
        ("read_file", {"path": "missing.txt"}),
        ("read_file", {"path": "src"}),
        ("read_file", {"path": "out.txt"}),
        ("read_file", {"path": "../outside.txt"}),
        ("list_dir", {"path": "."}),
        ("list_dir", {"path": "src"}),
        ("glob_files", {"pattern": "**/*.py"}),
        ("glob_files", {"pattern": "src/*"}),
        ("glob_files", {"pattern": "*.txt"}),
        ("glob_files", {"pattern": "../*"}),
        ("grep", {"pattern": r"TODO \w+"}),
        ("grep", {"pattern": "TODO", "glob": "*.py"}),
        ("grep", {"pattern": "TODO", "path": "src/b.txt"}),
        ("grep", {"pattern": "absent"}),
        ("grep", {"pattern": "(?P<w>TODO) one"}),
    ]

    @pytest.mark.asyncio
    async def test_they_answer_what_the_local_tools_answer(self, server: dict[str, Path]) -> None:
        backend = _backend(server)
        await backend.start()
        space, disk = backend.file_disk() or (None, None)
        assert space is not None and disk is not None
        local = build_tools(Workspace.at(server["work"]))
        for name, arguments in self.CALLS:
            remote = build_tools(space, disk)
            here = _answer(local[name], **arguments)
            there = await asyncio.to_thread(_answer, remote[name], **arguments)
            # The refusal lists the roots, which differ by the skill folders only.
            assert there == here, (name, arguments)
        await backend.close()

    @pytest.mark.asyncio
    async def test_writing_reaches_the_servers_files(self, server: dict[str, Path]) -> None:
        backend = _backend(server)
        tools = {
            tool.name: tool
            for tool in bundle(
                "files", workspace=Workspace.at(server["root"] / "here"), backend=backend
            ).tools
        }

        async def call(name: str, **arguments: Any) -> Any:
            result = await tools[name].execute(**arguments)
            return result.error or result.data

        assert "Created" in await call("write_file", path="new/c.md", content="one\n")
        assert "Appended" in await call("append_file", path="new/c.md", content="two\n")
        assert "Edited" in await call("edit_file", path="new/c.md", old="one", new="uno")
        assert (server["work"] / "new" / "c.md").read_text() == "uno\ntwo\n"
        os.link(server["work"] / "new" / "c.md", server["work"] / "hard.md")
        assert "hard link" in await call("write_file", path="hard.md", content="x")
        await backend.close()


@needs_gnu
class TestCommandsOnTheServer:
    @pytest.mark.asyncio
    async def test_a_command_runs_in_the_servers_workspace(self, server: dict[str, Path]) -> None:
        backend = _backend(server)
        result = await backend.run("pwd; read a; echo got $a; exit 4", timeout=10, stdin="x\n")
        assert result.stdout.split() == [str(server["work"]), "got", "x"]
        assert result.exit_code == 4
        await backend.close()

    @pytest.mark.asyncio
    async def test_a_skills_script_runs_from_its_copy(self, server: dict[str, Path]) -> None:
        skills = server["root"] / "skills"
        (skills / "demo").mkdir(parents=True)
        (skills / "demo" / "SKILL.md").write_text("---\nname: demo\n---\nbody\n")
        backend = _backend(server, skills=(skills,))
        result = await backend.run(f"cat {skills}/demo/SKILL.md", timeout=10)
        assert "body" in result.stdout
        copied = result.stdout
        (skills / "demo" / "SKILL.md").write_text("---\nname: demo\n---\nchanged\n")
        result = await backend.run(f"cat {skills}/demo/SKILL.md", timeout=10)
        assert "changed" in result.stdout and result.stdout != copied
        assert str(server["home"] / ".cache" / "domyn") in backend.state_dir
        await backend.close()

    @pytest.mark.asyncio
    async def test_a_missing_workspace_stops_the_start(self, server: dict[str, Path]) -> None:
        backend = SshBackend(
            Workspace.at(server["root"]),
            options=SshBackendOptions(host="fake", workspace="/nowhere", binary=str(server["bin"])),
        )
        with pytest.raises(BackendUnavailableError, match="not a folder"):
            await backend.start()

    @pytest.mark.asyncio
    async def test_a_noisy_login_shell_is_named(self, server: dict[str, Path]) -> None:
        noisy = (
            server["bin"].read_text().replace('exec sh -c "$1"', 'echo welcome; exec sh -c "$1"')
        )
        server["bin"].write_text(noisy)
        with pytest.raises(BackendUnavailableError, match="prints text"):
            await _backend(server).start()
