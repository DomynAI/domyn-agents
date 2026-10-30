"""Finding the skills a project keeps in its own folder, for `domyn setup` to offer."""

from pathlib import Path

import pytest

from domyn.agents.skills import LocalDirSkillSource, SkillQuery
from domyn.agents.skills.local import discover_local_skills


def _skill(path: Path, name: str, description: str = "does a thing") -> Path:
    """A skill folder with a SKILL.md, or a loose file when ``path`` ends in .md."""
    if path.suffix == ".md":
        path.parent.mkdir(parents=True, exist_ok=True)
        file = path
    else:
        path.mkdir(parents=True, exist_ok=True)
        file = path / "SKILL.md"
    file.write_text(f"---\nname: {name}\ndescription: {description}\n---\nbody of {name}\n")
    return path


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "skills"


def _names(found: object) -> list[str]:
    return [skill.name for skill in found.skills]  # type: ignore[attr-defined]


class TestWhatCountsAsASkill:
    def test_a_folder_with_a_skill_file_is_one_skill(self, root: Path) -> None:
        _skill(root / "write_essay", "write_essay", "Write essays")

        found = discover_local_skills(root)

        assert _names(found) == ["write_essay"]
        assert found.skills[0].description == "Write essays"
        assert found.skills[0].group is None

    def test_a_loose_markdown_file_in_the_root_is_one_too(self, root: Path) -> None:
        _skill(root / "notes.md", "notes")

        found = discover_local_skills(root)

        assert _names(found) == ["notes"]
        assert found.skills[0].path == root / "notes.md"

    def test_the_lowercase_file_name_is_accepted(self, root: Path) -> None:
        (root / "tidy").mkdir(parents=True)
        (root / "tidy" / "skill.md").write_text("---\nname: tidy\ndescription: d\n---\nb\n")

        assert _names(discover_local_skills(root)) == ["tidy"]

    def test_the_walk_does_not_go_inside_a_skill(self, root: Path) -> None:
        """Its own references and scripts are not skills."""
        _skill(root / "outer", "outer")
        _skill(root / "outer" / "examples" / "inner", "inner")

        assert _names(discover_local_skills(root)) == ["outer"]

    def test_a_loose_file_below_the_root_is_not_a_skill(self, root: Path) -> None:
        """Only the root holds loose files; elsewhere a skill is a folder."""
        _skill(root / "writing" / "stray.md", "stray")

        assert _names(discover_local_skills(root)) == []

    def test_an_empty_or_missing_folder_gives_nothing(self, root: Path, tmp_path: Path) -> None:
        assert discover_local_skills(root).skills == []
        root.mkdir()
        assert discover_local_skills(root).skills == []

    def test_a_file_where_the_folder_should_be_gives_nothing(self, tmp_path: Path) -> None:
        (tmp_path / "skills").write_text("x")

        assert discover_local_skills(tmp_path / "skills").skills == []


class TestGroups:
    def test_the_folders_above_a_skill_are_its_group(self, root: Path) -> None:
        _skill(root / "writing" / "essay-draft", "essay-draft")
        _skill(root / "writing" / "outline", "outline")

        found = discover_local_skills(root)

        assert [(s.name, s.group) for s in found.skills] == [
            ("essay-draft", "writing"),
            ("outline", "writing"),
        ]
        assert found.groups() == {"writing": ["essay-draft", "outline"]}

    def test_a_deeper_skill_is_grouped_by_its_whole_folder_path(self, root: Path) -> None:
        _skill(root / "writing" / "essays" / "draft", "draft")

        assert discover_local_skills(root).skills[0].group == "writing/essays"

    def test_a_skill_in_the_root_has_no_group_and_makes_none(self, root: Path) -> None:
        _skill(root / "alone", "alone")

        found = discover_local_skills(root)

        assert found.skills[0].group is None
        assert found.groups() == {}

    def test_a_folder_holding_no_skill_is_no_group(self, root: Path) -> None:
        (root / "empty").mkdir(parents=True)
        (root / "docs").mkdir()
        (root / "docs" / "readme.txt").write_text("x")

        assert discover_local_skills(root).groups() == {}


class TestPaths:
    def test_the_path_is_the_root_joined_to_the_skill_so_it_stays_relative(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _skill(Path("skills") / "writing" / "outline", "outline")
        _skill(Path("skills") / "notes.md", "notes")

        paths = sorted(s.path.as_posix() for s in discover_local_skills(Path("skills")).skills)

        assert paths == ["skills/notes.md", "skills/writing/outline"]
        assert all(not Path(p).is_absolute() for p in paths)

    def test_the_name_in_the_file_wins_over_the_folder_name(self, root: Path) -> None:
        _skill(root / "folder-name", "frontmatter-name")

        found = discover_local_skills(root)

        assert found.skills[0].name == "frontmatter-name"
        assert found.skills[0].path.name == "folder-name"

    def test_a_listed_path_loads_just_that_skill(self, root: Path) -> None:
        """What the config will hold: one entry per skill, a folder or a loose file."""
        import asyncio

        _skill(root / "a", "a")
        _skill(root / "b", "b")
        _skill(root / "loose.md", "loose")
        found = discover_local_skills(root)
        by_name = {s.name: s.path for s in found.skills}

        for name in ("a", "loose"):
            source = LocalDirSkillSource(paths=[by_name[name]])
            index = asyncio.run(source.index(SkillQuery(task="")))
            assert [d.name for d in index.skills] == [name]


class TestLabels:
    def test_two_skills_with_one_name_get_distinct_labels(self, root: Path) -> None:
        _skill(root / "a" / "summarise", "summarise")
        _skill(root / "b" / "summarise", "summarise")

        labels = [s.label for s in discover_local_skills(root).skills]

        assert len(set(labels)) == 2
        assert labels[0] == "summarise"
        assert "b/summarise" in labels[1]

    def test_a_skill_named_like_a_group_is_told_apart(self, root: Path) -> None:
        """The picker keys rows by label: a skill and a group may not share one."""
        _skill(root / "writing" / "outline", "outline")
        _skill(root / "alone", "writing")

        found = discover_local_skills(root)
        labels = [s.label for s in found.skills]

        assert "writing" not in labels
        assert not set(labels) & set(found.groups())


class TestWhatIsNotWalked:
    def test_hidden_and_dependency_folders_are_skipped(self, root: Path) -> None:
        _skill(root / ".hidden" / "x", "x")
        _skill(root / "node_modules" / "y", "y")
        _skill(root / "__pycache__" / "z", "z")
        _skill(root / "real", "real")

        assert _names(discover_local_skills(root)) == ["real"]

    def test_a_symlinked_folder_is_neither_entered_nor_counted(
        self, root: Path, tmp_path: Path
    ) -> None:
        outside = tmp_path / "outside"
        _skill(outside / "secret", "secret")
        _skill(root / "real", "real")
        (root / "link").symlink_to(outside)

        assert _names(discover_local_skills(root)) == ["real"]

    def test_a_symlink_loop_ends(self, root: Path) -> None:
        _skill(root / "real", "real")
        (root / "loop").symlink_to(root)

        assert _names(discover_local_skills(root)) == ["real"]

    def test_a_symlinked_root_is_not_followed(self, tmp_path: Path) -> None:
        target = tmp_path / "elsewhere"
        _skill(target / "x", "x")
        link = tmp_path / "skills"
        link.symlink_to(target)

        assert discover_local_skills(link).skills == []


class TestMalformedSkills:
    def test_one_that_cannot_be_read_is_reported_and_the_rest_kept(self, root: Path) -> None:
        _skill(root / "good", "good")
        (root / "bad").mkdir()
        (root / "bad" / "SKILL.md").write_bytes(b"\xff\xfe not utf-8 \x00")

        found = discover_local_skills(root)

        assert _names(found) == ["good"]
        assert [path.name for path, _ in found.skipped] == ["bad"]
        assert found.skipped[0][1]  # says why, by type

    def test_the_order_is_by_name_so_it_is_stable(self, root: Path) -> None:
        for name in ("c", "a", "b"):
            _skill(root / name, name)

        assert _names(discover_local_skills(root)) == ["a", "b", "c"]


class TestExternalFilesOfADiscoveredSkill:
    """A discovered skill stays where it is: its `external_files` list is checked, never copied."""

    def _with_external_files(self, root: Path, name: str, entries: list[str]) -> None:
        listed = "".join(f"  - {e}\n" for e in entries)
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: d\nexternal_files:\n{listed}---\nbody\n"
        )

    def test_a_skill_with_every_file_there_reports_nothing(self, root: Path) -> None:
        self._with_external_files(root, "report", ["../shared/helper.py", "scripts/"])
        (root / "shared").mkdir()
        (root / "shared" / "helper.py").write_text("h")
        (root / "report" / "scripts").mkdir()

        assert discover_local_skills(root).skills[0].missing == ()

    def test_a_missing_file_is_recorded_with_why_and_the_skill_is_kept(self, root: Path) -> None:
        self._with_external_files(root, "report", ["../shared/helper.py"])

        found = discover_local_skills(root)

        assert _names(found) == ["report"]
        assert found.skills[0].missing == ("../shared/helper.py (does not exist)",)

    def test_a_skill_with_no_key_has_nothing_to_miss(self, root: Path) -> None:
        _skill(root / "plain", "plain")

        assert discover_local_skills(root).skills[0].missing == ()

    def test_nothing_is_copied_or_created(self, root: Path) -> None:
        self._with_external_files(root, "report", ["../shared/helper.py"])
        before = sorted(p for p in root.parent.rglob("*"))

        discover_local_skills(root)

        assert sorted(p for p in root.parent.rglob("*")) == before

    def test_the_problem_survives_the_unique_labelling(self, root: Path) -> None:
        self._with_external_files(root / "a", "dup", ["gone.py"])
        self._with_external_files(root / "b", "dup", ["gone.py"])

        assert all(s.missing for s in discover_local_skills(root).skills)
