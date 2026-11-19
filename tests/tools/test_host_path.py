"""A path the user writes as they see it on their machine, for a tool body in a container."""

from pathlib import Path

import pytest

from domyn.agents.tools.builtin.files import host_path
from domyn.agents.tools.builtin.workspace import Workspace


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    (tmp_path / "work").mkdir()
    return Workspace.at(tmp_path / "work")


class TestPathsAsTheUserSeesThem:
    def test_a_relative_path_is_left_to_the_workspace(self, workspace: Workspace) -> None:
        assert host_path("sub/a.md", workspace) == "sub/a.md"

    def test_an_absolute_path_inside_the_workspace_is_its_real_one(
        self, workspace: Workspace, tmp_path: Path
    ) -> None:
        link = tmp_path / "link"
        link.symlink_to(workspace.root)
        assert host_path(str(link / "a.md"), workspace) == str(workspace.root / "a.md")

    def test_the_home_folder_is_the_hosts_not_the_containers(
        self, workspace: Workspace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(workspace.root.parent))
        assert host_path("~/work/a.md", workspace) == str(workspace.root / "a.md")

    def test_a_home_path_outside_the_mounts_names_the_host_path(
        self, workspace: Workspace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The container has no such file: not found, which is the boundary.
        monkeypatch.setenv("HOME", str(workspace.root.parent))
        assert host_path("~/.ssh/id_rsa", workspace) == str(workspace.root.parent / ".ssh/id_rsa")

    def test_a_container_path_outside_the_mounts_is_left_as_written(
        self, workspace: Workspace
    ) -> None:
        assert host_path("/etc/os-release", workspace) == "/etc/os-release"

    def test_anything_but_text_passes_through(self, workspace: Workspace) -> None:
        assert host_path(None, workspace) is None


class TestTheHomeFolderInTheContainer:
    """`~` is the user's own home there, so a path written from the terminal means the same."""

    def test_home_is_the_hosts_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from domyn.agents.tools.builtin.backends.docker.containers import container_home

        monkeypatch.setenv("HOME", "/Users/someone")
        assert container_home() == "/Users/someone"

    @pytest.mark.parametrize("home", ["/", "/Users/a,b", "/Users/a:b"])
    def test_a_home_that_cannot_be_mounted_at_its_path_falls_back(
        self, home: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from domyn.agents.tools.builtin.backends.docker.containers import (
            CONTAINER_HOME,
            container_home,
        )

        monkeypatch.setenv("HOME", home)
        assert container_home() == CONTAINER_HOME

    def test_it_is_an_empty_writable_tmpfs_with_the_workspace_below_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from domyn.agents.tools.builtin.backends.docker.mounts import build_mounts
        from domyn.agents.tools.builtin.backends.docker.state import StateFolder

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("DOMYN_PROJECT_DIR", raising=False)
        workspace = Workspace.at(tmp_path / "work")
        (tmp_path / "work").mkdir(exist_ok=True)
        arguments = build_mounts(workspace, StateFolder.resolve(workspace.root, None), "c1")

        assert f"{workspace.root.parent}:rw,mode=1777,exec" in arguments
        assert f"type=bind,source={workspace.root},target={workspace.root}" in arguments

    def test_a_workspace_that_is_the_home_is_not_hidden_by_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from domyn.agents.tools.builtin.backends.docker.mounts import build_mounts
        from domyn.agents.tools.builtin.backends.docker.state import StateFolder

        home = tmp_path.resolve()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.delenv("DOMYN_PROJECT_DIR", raising=False)
        workspace = Workspace.at(home)
        arguments = build_mounts(workspace, StateFolder.resolve(workspace.root, None), "c1")

        assert not any(a.startswith(f"{home}:") for a in arguments)


class TestTheRealGlobalFolderIsNeverLeftOpen:
    """The container's home is ``CONTAINER_HOME`` when the real one has a colon, so the
    scratch alias is not at ``~/.domyn``: something else has to cover the real one."""

    @staticmethod
    def _covered(arguments: list[str], hidden: Path) -> bool:
        return any(
            a == f"type=tmpfs,target={hidden},tmpfs-size=1048576,tmpfs-mode=0755"
            or (a.startswith("type=bind,source=") and a.endswith(f",target={hidden}"))
            for a in arguments
        )

    @pytest.mark.parametrize("home_name", ["ab", "a:b"])
    @pytest.mark.parametrize("project", ["elsewhere", "under_global"])
    def test_a_workspace_that_is_the_home_never_exposes_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, home_name: str, project: str
    ) -> None:
        from domyn.agents.tools.builtin.backends.docker.mounts import build_mounts
        from domyn.agents.tools.builtin.backends.docker.state import StateFolder

        home = (tmp_path / home_name).resolve()
        (home / ".domyn" / "projects" / "p").mkdir(parents=True)
        (home / ".domyn" / "credentials").write_text("SECRET")
        proj = tmp_path / "proj" if project == "elsewhere" else home / ".domyn" / "projects" / "p"
        proj.mkdir(exist_ok=True)
        (proj / "config.yaml").write_text("x: 1")
        monkeypatch.setenv("HOME", str(home))
        arguments = build_mounts(Workspace.at(home), StateFolder.resolve(home, proj), "c1")

        assert self._covered(arguments, home / ".domyn")
        assert not any("credentials" in a for a in arguments)


class TestAPathDockerWouldSplit:
    """``--mount`` is read as CSV: a comma or a double quote in a path is quoted, not skipped."""

    @pytest.mark.parametrize(
        ("value", "field"),
        [
            ("/a b", "target=/a b"),
            ("/a:b", "target=/a:b"),
            ("/a,b", '"target=/a,b"'),
            ('/a"b', '"target=/a""b"'),
            ('/a,"b', '"target=/a,""b"'),
            ("/a\nb", '"target=/a\nb"'),
        ],
    )
    def test_the_field_is_quoted_only_when_it_has_to_be(self, value: str, field: str) -> None:
        from domyn.agents.tools.builtin.backends.docker.mounts import _field

        assert _field("target", value) == field

    def test_the_hidden_global_folder_is_covered_for_a_home_with_a_comma(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from domyn.agents.tools.builtin.backends.docker.mounts import build_mounts
        from domyn.agents.tools.builtin.backends.docker.state import StateFolder

        home = (tmp_path / "a,b").resolve()
        (home / ".domyn").mkdir(parents=True)
        (tmp_path / "proj").mkdir()
        ((tmp_path / "proj") / "config.yaml").write_text("x: 1")
        monkeypatch.setenv("HOME", str(home))
        arguments = build_mounts(
            Workspace.at(home), StateFolder.resolve(home, tmp_path / "proj"), "c1"
        )

        assert f'type=tmpfs,"target={home}/.domyn",tmpfs-size=1048576,tmpfs-mode=0755' in arguments
        assert f'type=bind,"source={home}","target={home}"' in arguments
