"""Skills a workspace contains, when the container hides the global `~/.domyn`."""

import os
from pathlib import Path

import pytest

from domyn.agents.tools.builtin.workspace import Workspace

# The Docker backend is refused on Windows until its mounts translate Windows paths
# (`backends/docker/backend.py`, `_ON_WINDOWS`): what these check does not run there.
pytestmark = pytest.mark.skipif(os.name == "nt", reason="no Docker backend on Windows yet")


class TestSkillsUnderAHiddenGlobalHome:
    """A workspace that holds `~/.domyn` hides it in the container, skills excepted."""

    def test_a_skill_folder_the_workspace_contains_is_remembered(self, tmp_path: Path) -> None:
        skills = tmp_path / ".domyn" / "builtin-skills"
        skills.mkdir(parents=True)
        workspace = Workspace.at(tmp_path, also=[skills])

        assert workspace.extra_roots == ()
        assert workspace.covered_roots == (skills.resolve(),)

    def test_they_are_mounted_back_over_the_hidden_folder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from domyn.agents.tools.builtin.backends.docker.mounts import skill_roots

        monkeypatch.setenv("HOME", str(tmp_path.resolve()))
        skills = tmp_path.resolve() / ".domyn" / "builtin-skills"
        skills.mkdir(parents=True)
        mine = tmp_path.resolve() / "project" / "skills"
        mine.mkdir(parents=True)

        roots = skill_roots(Workspace.at(tmp_path, also=[skills, mine]))

        # Only what the container hides: the rest is in the workspace mount already.
        assert roots == [skills]


class TestAHomeReachedThroughALink:
    """The workspace is mounted at its real path, so the folder to hide and the skills to put
    back are the real ones too, not the spelling of ``$HOME``."""

    @staticmethod
    def _home_through_a_link(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
        real = tmp_path / "real"
        (real / ".domyn" / "builtin-skills" / "pres").mkdir(parents=True)
        (real / ".domyn" / "credentials").write_text("SECRET")
        link = tmp_path / "link"
        link.symlink_to(real)
        monkeypatch.setenv("HOME", str(link))
        return real.resolve(), link

    def test_the_real_global_folder_is_hidden(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from domyn.agents.tools.builtin.backends.docker.mounts import build_mounts
        from domyn.agents.tools.builtin.backends.docker.state import StateFolder

        real, link = self._home_through_a_link(tmp_path, monkeypatch)
        (tmp_path / "proj").mkdir()
        (tmp_path / "proj" / "config.yaml").write_text("x: 1")
        arguments = build_mounts(
            Workspace.at(link), StateFolder.resolve(real, tmp_path / "proj"), "c1"
        )
        assert f"type=tmpfs,target={real}/.domyn,tmpfs-size=1048576,tmpfs-mode=0755" in arguments

    def test_the_skills_under_it_are_put_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from domyn.agents.tools.builtin.backends.docker.mounts import skill_roots

        _, link = self._home_through_a_link(tmp_path, monkeypatch)
        skills = link / ".domyn" / "builtin-skills"
        assert skill_roots(Workspace.at(link, also=[skills])) == [skills.resolve()]
