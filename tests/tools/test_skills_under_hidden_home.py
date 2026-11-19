"""Skills a workspace contains, when the container hides the global `~/.domyn`."""

from pathlib import Path

import pytest

from domyn.agents.tools.builtin.workspace import Workspace


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
