"""A state folder: where the tools keep what is not the task's.

`LocalBackend` puts the conversations' environments, the background commands' logs and
the browser's files in the folder ``DOMYN_PROJECT_DIR`` names, or in one the caller
passes (``state_dir=``). Without either, nothing changes: ``.domyn`` in the workspace,
as before.
"""

# No `from __future__ import annotations`: this reaches the tool modules, which
# build their tools from real annotations.

import os
from pathlib import Path
from typing import Any

import pytest

from domyn.agents.tools.builtin import (
    BrowserOptions,
    BuiltinBundleConfig,
    LocalBackend,
    Workspace,
    browser_tools,
    processes,
    shell,
)


def _data(result: Any) -> tuple[str, dict[str, Any]]:
    return str(result.data), result.state


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    folder = tmp_path / "work"
    folder.mkdir()
    return folder


class TestWithoutAStateFolder:
    def test_it_is_dot_domyn_in_the_workspace(self, workspace: Path) -> None:
        backend = LocalBackend(Workspace.at(workspace))

        assert backend.state_dir == ".domyn"
        assert backend.sessions is not None
        assert backend.sessions.paths("c1").home == workspace / ".domyn" / "conversations" / "c1"


class TestFromTheEnvironment:
    def test_the_variable_names_it(
        self, workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DOMYN_PROJECT_DIR", str(tmp_path / "home"))

        backend = LocalBackend(Workspace.at(workspace))

        assert backend.state_dir == str(tmp_path / "home")
        assert backend.sessions is not None
        assert backend.sessions.paths("c1").home == tmp_path / "home" / "conversations" / "c1"

    def test_a_folder_passed_wins_over_it(
        self, workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DOMYN_PROJECT_DIR", str(tmp_path / "home"))

        backend = LocalBackend(Workspace.at(workspace), state_dir=tmp_path / "mine")

        assert backend.state_dir == str(tmp_path / "mine")

    def test_it_is_read_when_the_backend_is_built(
        self, workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A session keeps the folder it started with."""
        monkeypatch.setenv("DOMYN_PROJECT_DIR", str(tmp_path / "first"))
        backend = LocalBackend(Workspace.at(workspace))
        monkeypatch.setenv("DOMYN_PROJECT_DIR", str(tmp_path / "second"))

        assert backend.state_dir == str(tmp_path / "first")

    def test_an_empty_variable_is_none(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DOMYN_PROJECT_DIR", "")

        assert LocalBackend(Workspace.at(workspace)).state_dir == ".domyn"


class TestWithAStateFolder:
    @pytest.mark.asyncio
    async def test_background_commands_keep_their_logs_there(
        self, workspace: Path, tmp_path: Path
    ) -> None:
        state_dir = tmp_path / "home"
        backend = LocalBackend(
            Workspace.at(workspace), isolate_environment=False, state_dir=state_dir
        )
        run = shell.build_run_command(backend)
        listing = shell.build_list_commands(backend)
        stop = shell.build_stop_command(backend, grace_s=1)

        text, state = _data(
            await run(command="sleep 30", background=True, conversation_id="c1", state={})
        )
        (handle,) = processes.registry(state)

        assert (state_dir / "processes" / "c1" / handle / "run.pid").is_file()
        assert str(state_dir / "processes" / "c1" / handle / "out.log") in text
        assert not (workspace / ".domyn").exists()
        # The folder is the backend's, not the record's: nothing of it is stored.
        assert "home" not in state["background_processes"][handle]

        listed, state = _data(await listing(conversation_id="c1", state=state))
        assert handle in listed and "running" in listed

        stopped, _ = _data(await stop(handle=handle, conversation_id="c1", state=state))
        assert "Stopped" in stopped

    @pytest.mark.asyncio
    async def test_a_folder_with_shell_characters_in_its_name(
        self, workspace: Path, tmp_path: Path
    ) -> None:
        # The wrapper reads its folder from a quoted variable: in double quotes, `$`,
        # `"` and a backtick in a project's path would be shell.
        state_dir = tmp_path / 'we"ird $(touch pwned) `x`'
        backend = LocalBackend(
            Workspace.at(workspace), isolate_environment=False, state_dir=state_dir
        )
        run = shell.build_run_command(backend)

        text, _ = _data(await run(command="echo hi", stdin="", conversation_id="c1", state={}))

        assert "hi" in text
        assert not (workspace / "pwned").exists() and not (tmp_path / "pwned").exists()
        assert any((state_dir / "processes" / "c1").iterdir())

    def test_conversation_environments_are_there(self, workspace: Path, tmp_path: Path) -> None:
        backend = LocalBackend(Workspace.at(workspace), state_dir=tmp_path / "home")

        assert backend.sessions is not None
        paths = backend.sessions.paths("c1")
        assert paths.home == tmp_path / "home" / "conversations" / "c1"

    def test_a_linked_conversations_folder_is_refused(
        self, workspace: Path, tmp_path: Path
    ) -> None:
        # A project used by its path may be a folder someone sent.
        state_dir = tmp_path / "project"
        state_dir.mkdir()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (state_dir / "conversations").symlink_to(elsewhere)
        backend = LocalBackend(Workspace.at(workspace), state_dir=state_dir)

        assert backend.sessions is not None
        assert backend.sessions.prepare("c1") is None
        assert not any(elsewhere.iterdir())

    def test_the_state_folder_itself_may_be_a_link(self, workspace: Path, tmp_path: Path) -> None:
        real = tmp_path / "dotfiles" / "domyn"
        real.mkdir(parents=True)
        linked = tmp_path / "linked"
        linked.symlink_to(real)
        backend = LocalBackend(Workspace.at(workspace), state_dir=linked)
        assert backend.sessions is not None
        # Only the walk is checked here, not the venv: building one takes seconds.
        paths = backend.sessions.paths("c1")
        here = linked
        for part in paths.home.relative_to(linked).parts:
            here = here / part
            assert not here.is_symlink()

    def test_the_browser_keeps_its_files_there(self, workspace: Path, tmp_path: Path) -> None:
        backend = LocalBackend(Workspace.at(workspace), state_dir=tmp_path / "home")

        _, sessions = browser_tools(backend=backend, options=BrowserOptions())

        assert sessions.browser_dir == f"{tmp_path / 'home'}/browser"


class TestLinksAreNotFollowed:
    """``processes/`` and ``browser/`` are written with ``mkdir -p`` and ``>``, which
    follow a link: a project folder someone sent with ``processes -> elsewhere``
    would have the next background command write its script there."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", ["processes", "browser"])
    async def test_a_linked_state_folder_refuses_commands(
        self, workspace: Path, tmp_path: Path, name: str
    ) -> None:
        state_dir = tmp_path / "project"
        state_dir.mkdir()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (state_dir / name).symlink_to(elsewhere)
        backend = LocalBackend(
            Workspace.at(workspace), isolate_environment=False, state_dir=state_dir
        )

        result = await backend.run("echo hi", timeout=10)
        text, _ = _data(await shell.build_run_command(backend)("sleep 5", background=True))

        assert result.exit_code is None
        assert result.backend_error is not None
        assert f"{state_dir / name} is a link" in result.backend_error
        assert "is a link" in text
        assert not any(elsewhere.iterdir())

    @pytest.mark.asyncio
    async def test_a_linked_dot_domyn_in_the_workspace_is_refused(
        self, workspace: Path, tmp_path: Path
    ) -> None:
        # Without a state folder it is `.domyn` in the workspace, which a cloned
        # repository can commit as a link.
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (workspace / ".domyn").symlink_to(elsewhere)
        backend = LocalBackend(Workspace.at(workspace), isolate_environment=False)

        result = await backend.run("echo hi", timeout=10)

        assert result.backend_error is not None
        assert f"{workspace / '.domyn'} is a link" in result.backend_error

    @pytest.mark.asyncio
    async def test_the_state_folder_itself_may_be_a_link(
        self, workspace: Path, tmp_path: Path
    ) -> None:
        real = tmp_path / "dotfiles" / "domyn"
        (real / "processes").mkdir(parents=True)
        linked = tmp_path / "linked"
        linked.symlink_to(real)
        backend = LocalBackend(Workspace.at(workspace), isolate_environment=False, state_dir=linked)

        result = await backend.run("echo hi", timeout=10)

        assert result.backend_error is None
        assert result.stdout.strip() == "hi"


class TestRefreshKeepsTheFolder:
    def test_the_records_it_returns_are_in_the_backends_folder(self) -> None:
        state: dict[str, Any] = {}
        record = processes.new_record("sleep 1", "c1", home="/state/processes")
        processes.remember(state, record)

        live = processes.refresh(state, {}, home="/state/processes")

        assert [r.dir for r in live] == [f"/state/processes/c1/{record.handle}"]


class TestTheBundle:
    def test_the_file_tools_do_not_reach_it(
        self, workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Outside a project it is ``~/.domyn``, shared by every folder a session runs
        in: the ungated read tools would reach other conversations' logs and output."""
        home = tmp_path / "home"
        (home / "processes" / "c1").mkdir(parents=True)
        (home / "conversations" / "c1").mkdir(parents=True)
        monkeypatch.setenv("DOMYN_PROJECT_DIR", str(home))
        built = BuiltinBundleConfig(name="b", workspace=str(workspace)).build()

        assert built.workspace is not None and built.workspace.extra_roots == ()
        for folder in ("processes", "conversations", "credentials"):
            assert not built.workspace.contains((home / folder).resolve())
        assert built.backend is not None and built.backend.state_dir == str(home)

    def test_without_one_the_workspace_is_unchanged(self, workspace: Path) -> None:
        built = BuiltinBundleConfig(name="b", workspace=str(workspace)).build()

        assert built.workspace is not None and built.workspace.extra_roots == ()
        assert built.backend is not None and built.backend.state_dir == ".domyn"
