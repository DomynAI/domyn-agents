"""The ``external_files:`` key: files a skill needs, as paths relative to its own SKILL.md."""

import asyncio
from pathlib import Path

import pytest

from domyn.agents.skills import (
    LocalDirSkillSource,
    SkillQuery,
    evaluate_requirements,
    parse_skill_file,
    parse_skill_text,
)
from domyn.agents.skills.external_files import (
    expand_files,
    missing_external_files,
    resolve_external_files,
)


def _parse(front: str) -> list[str]:
    text = f"---\nname: s\ndescription: d\n{front}---\nbody\n"
    descriptor, _ = parse_skill_text(text, fallback_name="s")
    return descriptor.requires.external_files


class TestReadingTheKey:
    def test_a_list_is_read_in_order(self) -> None:
        assert _parse("external_files:\n  - scripts/\n  - ../shared.py\n") == [
            "scripts/",
            "../shared.py",
        ]

    def test_a_single_string_is_one_entry(self) -> None:
        assert _parse("external_files: ../shared.py\n") == ["../shared.py"]

    def test_an_absent_key_is_an_empty_list(self) -> None:
        assert _parse("") == []

    def test_entries_are_stripped(self) -> None:
        assert _parse('external_files:\n  - "  a.py  "\n') == ["a.py"]

    @pytest.mark.parametrize(
        ("front", "expected"),
        [
            ("external_files:\n  a: b\n", []),  # a mapping is not a list of paths
            ("external_files: 3\n", []),
            ("external_files:\n  - ok.py\n  - 7\n  - ''\n  - [x]\n", ["ok.py"]),
        ],
    )
    def test_a_malformed_value_is_ignored_and_the_skill_still_loads(
        self, front: str, expected: list[str]
    ) -> None:
        assert _parse(front) == expected

    def test_it_is_read_nested_under_metadata_requires(self) -> None:
        front = "metadata:\n  requires:\n    external_files: [scripts/, ../shared.py]\n"
        assert _parse(front) == ["scripts/", "../shared.py"]

    def test_the_nested_form_wins_over_the_flat_one(self) -> None:
        front = (
            "external_files: [flat.py]\nmetadata:\n  requires:\n    external_files: [nested.py]\n"
        )
        assert _parse(front) == ["nested.py"]

    def test_a_skill_listing_only_files_has_requirements(self) -> None:
        descriptor, _ = parse_skill_text(
            "---\nname: s\ndescription: d\nexternal_files: [a.py]\n---\nbody\n",
            fallback_name="s",
        )
        assert descriptor.requires.is_empty() is False

    def test_the_old_required_key_is_not_read(self) -> None:
        assert _parse("required:\n  - scripts/\n") == []

    def test_files_that_are_there_are_not_shown_to_the_model(self, tmp_path: Path) -> None:
        (tmp_path / "secret-name.py").write_text("x")
        skill = tmp_path / "SKILL.md"
        skill.write_text(
            "---\nname: s\ndescription: d\nexternal_files:\n  - secret-name.py\n---\nbody\n"
        )
        descriptor, content = parse_skill_file(skill)

        line = evaluate_requirements(descriptor).index_line()

        assert "secret-name.py" not in line
        assert "secret-name.py" not in content.body

    def test_a_missing_file_is_named_in_the_unavailable_reason(self, tmp_path: Path) -> None:
        skill = tmp_path / "SKILL.md"
        skill.write_text("---\nname: s\ndescription: d\nexternal_files:\n  - x.py\n---\nbody\n")
        descriptor, _ = parse_skill_file(skill)

        line = evaluate_requirements(descriptor).index_line()

        assert "[unavailable: file 'x.py' (does not exist)]" in line

    def test_it_is_not_confused_with_the_other_two_keys(self) -> None:
        descriptor, _ = parse_skill_text(
            "---\nname: s\ndescription: d\nenforcement: required\n"
            "metadata:\n  requires:\n    bins: [git]\nexternal_files:\n  - f.py\n---\nb\n",
            fallback_name="s",
        )

        assert descriptor.requires.external_files == ["f.py"]
        assert descriptor.requires.bins == ["git"]
        assert descriptor.enforcement == "required"


@pytest.fixture
def skill_file(tmp_path: Path) -> Path:
    folder = tmp_path / "area" / "skill"
    (folder / "scripts").mkdir(parents=True)
    (folder / "scripts" / "run.py").write_text("x")
    (folder / "SKILL.md").write_text("---\nname: skill\ndescription: d\n---\nb\n")
    (tmp_path / "area" / "shared.py").write_text("s")
    return folder / "SKILL.md"


class TestResolving:
    def test_entries_are_relative_to_the_folder_holding_the_skill_file(
        self, skill_file: Path
    ) -> None:
        found = resolve_external_files(skill_file, ["scripts/run.py", "../shared.py"])

        assert [item.problem for item in found] == [None, None]
        assert found[0].path == (skill_file.parent / "scripts" / "run.py").resolve()
        assert found[1].path == (skill_file.parent.parent / "shared.py").resolve()

    def test_dot_is_the_skills_own_folder(self, skill_file: Path) -> None:
        (item,) = resolve_external_files(skill_file, ["."])

        assert item.problem is None and item.path == skill_file.parent.resolve()

    def test_a_missing_path_is_reported_with_why(self, skill_file: Path) -> None:
        (item,) = resolve_external_files(skill_file, ["gone.py"])

        assert item.problem == "does not exist"

    @pytest.mark.parametrize("entry", ["/etc/hosts", "C:\\Windows\\x.dll", "C:/x.dll"])
    def test_an_absolute_path_is_refused_not_followed(self, skill_file: Path, entry: str) -> None:
        (item,) = resolve_external_files(skill_file, [entry])

        assert item.problem == "absolute path" and item.path is None

    def test_backslashes_are_read_as_separators(self, skill_file: Path) -> None:
        (item,) = resolve_external_files(skill_file, ["scripts\\run.py"])

        assert item.problem is None

    def test_nothing_is_created_or_copied(self, skill_file: Path) -> None:
        before = sorted(skill_file.parent.parent.rglob("*"))

        resolve_external_files(skill_file, ["new/dir/file.py"])

        assert sorted(skill_file.parent.parent.rglob("*")) == before


class TestExpanding:
    def test_a_file_is_itself(self, skill_file: Path) -> None:
        run = skill_file.parent / "scripts" / "run.py"

        assert expand_files(run) == [run]

    def test_a_folder_is_every_file_under_it_without_bytecode(self, skill_file: Path) -> None:
        scripts = skill_file.parent / "scripts"
        (scripts / "__pycache__").mkdir()
        (scripts / "__pycache__" / "run.cpython-312.pyc").write_bytes(b"c")
        (scripts / "deep").mkdir()
        (scripts / "deep" / "more.py").write_text("m")

        names = [p.relative_to(scripts).as_posix() for p in expand_files(scripts)]

        assert names == ["deep/more.py", "run.py"]

    def test_a_symlink_inside_a_folder_is_not_followed(
        self, skill_file: Path, tmp_path: Path
    ) -> None:
        outside = tmp_path / "secret.txt"
        outside.write_text("s")
        scripts = skill_file.parent / "scripts"
        (scripts / "link.txt").symlink_to(outside)

        assert [p.name for p in expand_files(scripts)] == ["run.py"]

    def test_a_folder_that_is_itself_under_a_cache_name_is_still_expanded(
        self, tmp_path: Path
    ) -> None:
        """Only what is *below* the entry is judged: the folders above are the user's."""
        weird = tmp_path / "__pycache__" / "skill"
        weird.mkdir(parents=True)
        (weird / "a.py").write_text("a")

        assert [p.name for p in expand_files(weird)] == ["a.py"]

    def test_a_missing_path_is_nothing(self, tmp_path: Path) -> None:
        assert expand_files(tmp_path / "nope") == []


class TestMissingRequired:
    def test_nothing_missing_is_empty(self, skill_file: Path) -> None:
        assert missing_external_files(skill_file, ["scripts/", "../shared.py"]) == []

    def test_each_problem_names_its_entry_and_why(self, skill_file: Path) -> None:
        assert missing_external_files(skill_file, ["gone.py", "/abs.py", "scripts/"]) == [
            "gone.py (does not exist)",
            "/abs.py (absolute path)",
        ]

    def test_no_entries_is_empty(self, skill_file: Path) -> None:
        assert missing_external_files(skill_file, []) == []


class TestCheckedOnEveryLoad:
    """What a skill lists is checked each time the skills are loaded — every turn in
    a session — so a missing file makes the skill unavailable rather than broken, and
    restoring it is enough."""

    @staticmethod
    def _availability(folder: Path) -> tuple[bool, str | None]:
        source = LocalDirSkillSource(paths=[folder])
        loaded = asyncio.run(source.load(SkillQuery(task="x")))
        (skill,) = loaded.index.skills
        return skill.available, skill.unavailable_reason

    def test_a_missing_listed_file_makes_it_unavailable_until_it_is_back(
        self, tmp_path: Path
    ) -> None:
        folder = tmp_path / "skills" / "report"
        folder.mkdir(parents=True)
        (folder / "SKILL.md").write_text(
            "---\nname: report\ndescription: d\nexternal_files:\n  - build.py\n---\nb\n"
        )

        available, reason = self._availability(tmp_path / "skills")
        assert available is False
        assert reason is not None and "build.py" in reason

        (folder / "build.py").write_text("x")

        assert self._availability(tmp_path / "skills") == (True, None)

    def test_with_the_old_required_key_nothing_is_checked(self, tmp_path: Path) -> None:
        # The file is missing, and the skill is offered anyway: it declared nothing.
        folder = tmp_path / "skills" / "report"
        folder.mkdir(parents=True)
        (folder / "SKILL.md").write_text(
            "---\nname: report\ndescription: d\nrequired:\n  - build.py\n---\nb\n"
        )

        assert self._availability(tmp_path / "skills") == (True, None)
