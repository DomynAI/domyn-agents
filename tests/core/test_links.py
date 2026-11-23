"""A Windows junction is a link to every check that refuses to follow one.

Python reports a junction as a plain directory (``is_symlink()`` and ``S_ISLNK``
false, ``is_dir(follow_symlinks=False)`` true), and any user can make one without
privileges, so each check that asked ``is_symlink()`` alone followed it: ``.agents``
made a junction to the home folder would have loaded ``~/skills``. Real junctions,
made with ``mklink /J``, so Windows only.
"""

import os
import subprocess
from pathlib import Path

import pytest

from domyn.agents.core.links import is_link, is_link_stat
from domyn.agents.instructions.discovery import find_instruction_files
from domyn.agents.skills.external_files import expand_files
from domyn.agents.skills.layout import listed_path_is_linked, walk_skill_tree
from domyn.agents.tools.builtin.session import linked_below

pytestmark = pytest.mark.skipif(os.name != "nt", reason="junctions are Windows only")


def _junction(link: Path, target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True
    )
    return link


def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _skill(folder: Path, name: str) -> None:
    _write(folder / "SKILL.md", f"---\nname: {name}\ndescription: d\n---\nbody\n")


class TestWhatAJunctionIs:
    def test_a_junction_is_a_link(self, tmp_path: Path) -> None:
        junction = _junction(tmp_path / "j", tmp_path / "target")
        assert is_link(junction)
        assert not junction.is_symlink(), "what is_symlink alone missed"

    def test_a_folder_is_not(self, tmp_path: Path) -> None:
        (tmp_path / "plain").mkdir()
        assert not is_link(tmp_path / "plain")

    def test_a_directory_entry_too(self, tmp_path: Path) -> None:
        _junction(tmp_path / "j", tmp_path / "target")
        (entry,) = [e for e in os.scandir(tmp_path) if e.name == "j"]
        assert is_link(entry)

    def test_its_lstat_says_so(self, tmp_path: Path) -> None:
        junction = _junction(tmp_path / "j", tmp_path / "target")
        assert is_link_stat(junction.lstat())
        assert not is_link_stat((tmp_path / "target").lstat())


class TestNoCheckFollowsOne:
    def test_a_listed_skills_folder(self, tmp_path: Path) -> None:
        _skill(tmp_path / "elsewhere" / "secret", "secret")
        assert listed_path_is_linked(_junction(tmp_path / "skills", tmp_path / "elsewhere"))

    def test_a_folder_on_the_way_to_a_relative_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`.agents` made a junction to the home folder, in a folder domyn starts in."""
        (tmp_path / "repo").mkdir()
        _junction(tmp_path / "repo" / ".agents", tmp_path / "home")
        monkeypatch.chdir(tmp_path / "repo")
        assert listed_path_is_linked(Path(".agents/skills"))

    def test_the_skills_walk_does_not_enter_one(self, tmp_path: Path) -> None:
        _skill(tmp_path / "outside" / "secret", "secret")
        root = tmp_path / "root"
        _skill(root / "mine", "mine")
        _junction(root / "group", tmp_path / "outside")
        assert [found.path.name for found in walk_skill_tree(root)] == ["mine"]

    def test_the_instructions_walk_does_not_enter_one(self, tmp_path: Path) -> None:
        _write(tmp_path / "outside" / "AGENTS.md")
        root = tmp_path / "root"
        _write(root / "AGENTS.md")
        _junction(root / "linked", tmp_path / "outside")
        assert find_instruction_files(root).paths == [root / "AGENTS.md"]

    def test_the_state_folders_refuse_one(self, tmp_path: Path) -> None:
        state = tmp_path / "state"
        state.mkdir()
        junction = _junction(state / "processes", tmp_path / "LaunchAgents")
        assert linked_below(state, state / "processes" / "c1") == junction

    def test_a_skills_external_folder_skips_what_is_below_one(self, tmp_path: Path) -> None:
        _write(tmp_path / "outside" / "secret.txt")
        scripts = tmp_path / "skill" / "scripts"
        _write(scripts / "run.py")
        _junction(scripts / "linked", tmp_path / "outside")
        assert expand_files(scripts) == [scripts / "run.py"]
