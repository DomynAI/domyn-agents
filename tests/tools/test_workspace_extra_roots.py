"""Reaching files that belong to the agent rather than to the task.

The workspace is the one structural protection the built-in tools have, so
widening it is done deliberately and in one place. It exists for skills: they
live outside any project directory, and an agent told where its skill file is and
then refused when it opens it is worse off than one that was never told.

Writing stays behind the approval gate either way — these tests are about what
can be *asked for*, not about what happens unasked.
"""

from pathlib import Path

import pytest

from domyn.agents.tools.builtin import Workspace, WorkspaceError


@pytest.fixture
def places(tmp_path: Path) -> tuple[Path, Path]:
    work = tmp_path / "project"
    skills = tmp_path / "elsewhere" / "skills"
    (work / "src").mkdir(parents=True)
    (skills / "a-skill").mkdir(parents=True)
    (skills / "a-skill" / "SKILL.md").write_text("body")
    (work / "src" / "main.py").write_text("code")
    return work, skills


class TestWhatIsReachable:
    def test_the_workspace_itself(self, places: tuple[Path, Path]) -> None:
        work, skills = places
        assert Workspace.at(work, also=[skills]).resolve("src/main.py").exists()

    def test_an_extra_root(self, places: tuple[Path, Path]) -> None:
        work, skills = places
        space = Workspace.at(work, also=[skills])
        assert space.resolve(skills / "a-skill" / "SKILL.md").exists()

    def test_without_it_the_same_path_is_refused(self, places: tuple[Path, Path]) -> None:
        """The state before this existed, and the user-visible symptom: the agent
        was shown a skill it could not then open."""
        work, skills = places
        with pytest.raises(WorkspaceError):
            Workspace.at(work).resolve(skills / "a-skill" / "SKILL.md")

    def test_everything_else_is_still_refused(self, places: tuple[Path, Path]) -> None:
        work, skills = places
        space = Workspace.at(work, also=[skills])
        with pytest.raises(WorkspaceError):
            space.resolve("/etc/passwd")

    def test_the_parent_of_an_extra_root_is_not_reachable(self, places: tuple[Path, Path]) -> None:
        """Adding a root adds that subtree, not its neighbourhood."""
        work, skills = places
        space = Workspace.at(work, also=[skills])
        with pytest.raises(WorkspaceError):
            space.resolve(skills.parent / "secrets.txt")

    def test_traversal_out_of_an_extra_root_is_refused(self, places: tuple[Path, Path]) -> None:
        work, skills = places
        space = Workspace.at(work, also=[skills])
        with pytest.raises(WorkspaceError):
            space.resolve(skills / ".." / "secrets.txt")

    def test_a_symlink_out_of_an_extra_root_is_refused(
        self, places: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """The case a prefix check cannot see at all, now that there is more than
        one prefix to check against."""
        work, skills = places
        outside = tmp_path / "outside.txt"
        outside.write_text("secret")
        (skills / "link.txt").symlink_to(outside)

        space = Workspace.at(work, also=[skills])
        with pytest.raises(WorkspaceError):
            space.resolve(skills / "link.txt")


class TestKeepingTheRootsTidy:
    def test_no_extra_roots_by_default(self, places: tuple[Path, Path]) -> None:
        """An unchanged call must produce an unchanged boundary."""
        work, _ = places
        assert Workspace.at(work).extra_roots == ()

    def test_a_root_already_inside_the_workspace_is_dropped(
        self, places: tuple[Path, Path]
    ) -> None:
        """Otherwise the error message lists the same place twice and `display`
        picks between two right answers."""
        work, _ = places
        space = Workspace.at(work, also=[work / "src"])
        assert space.extra_roots == ()

    def test_the_workspace_itself_is_not_repeated(self, places: tuple[Path, Path]) -> None:
        work, _ = places
        assert Workspace.at(work, also=[work]).roots == (work.resolve(),)

    def test_duplicates_are_collapsed(self, places: tuple[Path, Path]) -> None:
        work, skills = places
        space = Workspace.at(work, also=[skills, skills])
        assert len(space.extra_roots) == 1

    def test_the_workspace_comes_first(self, places: tuple[Path, Path]) -> None:
        """Relative paths resolve against it, so its position is meaningful."""
        work, skills = places
        assert Workspace.at(work, also=[skills]).roots[0] == work.resolve()

    def test_a_tilde_is_expanded(self) -> None:
        space = Workspace.at(Path.home(), also=["~/.domyn"])
        assert all(str(root).startswith(str(Path.home())) for root in space.roots)


class TestWhatTheErrorAndTheDisplaySay:
    def test_the_error_names_every_allowed_root(self, places: tuple[Path, Path]) -> None:
        """The message is read by the model: naming only one of two places sends
        it to retry inside the wrong one."""
        work, skills = places
        space = Workspace.at(work, also=[skills])
        with pytest.raises(WorkspaceError) as caught:
            space.resolve("/etc/passwd")
        message = str(caught.value)
        assert str(work) in message
        assert str(skills) in message

    def test_workspace_files_stay_relative(self, places: tuple[Path, Path]) -> None:
        work, skills = places
        space = Workspace.at(work, also=[skills])
        assert space.display(work / "src" / "main.py") == "src/main.py"

    def test_an_extra_root_file_is_shown_in_full(self, places: tuple[Path, Path]) -> None:
        """ "SKILL.md" alone does not say which of several places it is in, and in
        an approval prompt that is the whole question."""
        work, skills = places
        space = Workspace.at(work, also=[skills])
        shown = space.display(skills / "a-skill" / "SKILL.md")
        assert shown == str(skills / "a-skill" / "SKILL.md")
