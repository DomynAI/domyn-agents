"""The builtin skills catalog, and installing a choice of them into a directory."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

from domyn.agents.skills import LocalDirSkillSource, SkillQuery, catalog, parse_skill_file
from domyn.agents.skills.catalog import (
    BUILTIN_SKILLS_ROOT,
    SkillInstallError,
    add_skills,
    install_plan,
    install_skills,
    list_builtin_skills,
    skill_groups,
    unrecognized_entries,
)

DATA_SKILLS = ("data-analysis", "time-series-forecasting", "tabular-classification-regression")


class TestTheManifest:
    def test_every_builtin_skill_is_listed_with_a_description(self) -> None:
        skills = {skill.name: skill for skill in list_builtin_skills()}

        assert {
            "domyn-cli",
            "data-analysis",
            "time-series-forecasting",
            "tabular-classification-regression",
            "presentations",
            "brainstorming",
            "communication",
            "charts",
            "interactive-charts",
            "documents",
            "domyn-docs",
        } <= skills.keys()
        assert all(skill.description for skill in skills.values())


def _plan_sources(skill: object) -> set[Path]:
    return {source for source, _ in install_plan(skill)}  # type: ignore[arg-type]


class TestTheBuiltInsListTheirFiles:
    """The lists are what the installer copies, so these are what stop them rotting."""

    def test_every_entry_of_every_builtin_resolves_inside_the_builtin_skills(self) -> None:
        for skill in list_builtin_skills():
            install_plan(skill)  # raises SkillInstallError for a bad entry

    def test_no_file_in_a_skills_folder_is_left_behind(self) -> None:
        """Listing everything is only safe if a forgotten script is noticed: a skill
        installed without it would fail the first time the model ran it."""
        for skill in list_builtin_skills():
            planned = _plan_sources(skill)
            for file in skill.source.rglob("*"):
                if file.is_symlink() and file.is_dir():
                    # rglob does not go into a linked folder, so its files are never seen
                    # one by one below: the folder has to be listed (`domyn-docs/pages`).
                    assert any(source.is_relative_to(file) for source in planned), (
                        f"{skill.name}: {file.name} is a linked folder not in `external_files`"
                    )
                    continue
                if not file.is_file() or "__pycache__" in file.parts or file.suffix == ".pyc":
                    continue
                assert file.resolve() in planned, (
                    f"{skill.name}: {file.name} is not in `external_files`"
                )

    @pytest.mark.parametrize("module", ["data_utils", "seaborn_style"])
    def test_a_skill_whose_scripts_use_a_shared_module_lists_it(self, module: str) -> None:
        for skill in list_builtin_skills():
            uses = any(
                module in script.read_text(encoding="utf-8")
                for script in skill.source.rglob("*.py")
            )
            if uses:
                names = {relative.name for _, relative in install_plan(skill)}
                assert f"{module}.py" in names, f"{skill.name} uses {module} but does not list it"

    def test_the_three_data_skills_list_the_same_shared_files(self) -> None:
        listed = {
            skill.name: {entry for entry in skill.external_files if entry.startswith("..")}
            for skill in list_builtin_skills()
            if skill.name in DATA_SKILLS
        }

        assert len({frozenset(v) for v in listed.values()}) == 1
        assert "../../resources/seaborn_style.py" in next(iter(listed.values()))

    def test_a_skill_with_nothing_beyond_its_text_has_no_key(self) -> None:
        by_name = {skill.name: skill for skill in list_builtin_skills()}

        for name in ("brainstorming", "communication", "domyn-cli"):
            assert by_name[name].external_files == ()
            assert [r.name for _, r in install_plan(by_name[name])] == ["SKILL.md"]


@pytest.fixture
def fake_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A package of its own for the installer to read: ``cat/one`` and ``cat/two``."""
    root = tmp_path / "pkg" / "builtin"
    root.mkdir(parents=True)
    monkeypatch.setattr(catalog, "BUILTIN_SKILLS_ROOT", root.resolve())
    return root.resolve()


def _fake_skill(root: Path, name: str, external_files: list[str], files: dict[str, str]) -> Path:
    folder = root / "cat" / name
    folder.mkdir(parents=True, exist_ok=True)
    listed = "".join(f"  - {entry}\n" for entry in external_files)
    block = f"external_files:\n{listed}" if external_files else ""
    (folder / "SKILL.md").write_text(f"---\nname: {name}\ndescription: d\n{block}---\nbody\n")
    for relative, text in files.items():
        (folder / relative).parent.mkdir(parents=True, exist_ok=True)
        (folder / relative).write_text(text)
    return folder


def _installed(dest: Path) -> list[str]:
    return sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file())


class TestOnlyWhatIsListedIsCopied:
    def test_an_unlisted_file_in_the_folder_is_left_out(
        self, fake_root: Path, tmp_path: Path
    ) -> None:
        _fake_skill(fake_root, "one", ["run.py"], {"run.py": "x", "notes.md": "y"})

        install_skills(["one"], tmp_path / "out")

        assert _installed(tmp_path / "out") == ["cat/one/SKILL.md", "cat/one/run.py"]

    def test_a_skill_with_no_key_installs_only_its_skill_file(
        self, fake_root: Path, tmp_path: Path
    ) -> None:
        _fake_skill(fake_root, "one", [], {"run.py": "x"})

        install_skills(["one"], tmp_path / "out")

        assert _installed(tmp_path / "out") == ["cat/one/SKILL.md"]

    def test_a_folder_entry_installs_its_files_and_not_bytecode(
        self, fake_root: Path, tmp_path: Path
    ) -> None:
        _fake_skill(
            fake_root,
            "one",
            ["scripts/"],
            {"scripts/a.py": "a", "scripts/sub/b.py": "b", "scripts/__pycache__/a.pyc": "c"},
        )

        install_skills(["one"], tmp_path / "out")

        assert _installed(tmp_path / "out") == [
            "cat/one/SKILL.md",
            "cat/one/scripts/a.py",
            "cat/one/scripts/sub/b.py",
        ]

    def test_dot_means_the_skills_own_folder(self, fake_root: Path, tmp_path: Path) -> None:
        _fake_skill(fake_root, "one", ["."], {"a.py": "a", "ref/b.md": "b"})

        install_skills(["one"], tmp_path / "out")

        assert _installed(tmp_path / "out") == [
            "cat/one/SKILL.md",
            "cat/one/a.py",
            "cat/one/ref/b.md",
        ]

    def test_a_file_above_the_skill_lands_in_the_same_relative_place(
        self, fake_root: Path, tmp_path: Path
    ) -> None:
        _fake_skill(fake_root, "one", ["../shared.py", "../../top.py"], {})
        (fake_root / "cat" / "shared.py").write_text("s")
        (fake_root / "top.py").write_text("t")

        install_skills(["one"], tmp_path / "out")

        assert _installed(tmp_path / "out") == ["cat/one/SKILL.md", "cat/shared.py", "top.py"]

    def test_a_file_two_skills_list_is_copied_once(self, fake_root: Path, tmp_path: Path) -> None:
        _fake_skill(fake_root, "one", ["../shared.py"], {})
        _fake_skill(fake_root, "two", ["../shared.py"], {})
        (fake_root / "cat" / "shared.py").write_text("s")

        paths = install_skills(["one", "two"], tmp_path / "out")

        assert len(paths) == 2
        assert _installed(tmp_path / "out").count("cat/shared.py") == 1

    def test_only_the_chosen_skills_lists_are_followed(
        self, fake_root: Path, tmp_path: Path
    ) -> None:
        _fake_skill(fake_root, "one", ["../a.py"], {})
        _fake_skill(fake_root, "two", ["../b.py"], {})
        (fake_root / "cat" / "a.py").write_text("a")
        (fake_root / "cat" / "b.py").write_text("b")

        install_skills(["two"], tmp_path / "out")

        assert _installed(tmp_path / "out") == ["cat/b.py", "cat/two/SKILL.md"]


class TestABrokenListIsRefused:
    @pytest.mark.parametrize(
        ("entry", "why"),
        [
            ("nope.py", "does not exist"),
            ("/etc/hosts", "absolute"),
            ("../../../outside.py", "outside"),
        ],
    )
    def test_it_raises_before_the_old_directory_is_touched(
        self, fake_root: Path, tmp_path: Path, entry: str, why: str
    ) -> None:
        (fake_root.parent / "outside.py").write_text("o")
        _fake_skill(fake_root, "one", [entry], {})
        out = tmp_path / "out"
        out.mkdir()
        (out / "keep.txt").write_text("keep")

        with pytest.raises(SkillInstallError, match=why):
            install_skills(["one"], out)

        assert (out / "keep.txt").read_text() == "keep"
        assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []

    def test_one_bad_skill_stops_an_install_that_includes_it(
        self, fake_root: Path, tmp_path: Path
    ) -> None:
        _fake_skill(fake_root, "good", [], {})
        _fake_skill(fake_root, "bad", ["missing.py"], {})

        with pytest.raises(SkillInstallError):
            install_skills(["good", "bad"], tmp_path / "out")

        assert not (tmp_path / "out").exists()

    def test_a_bad_list_does_not_break_the_skills_that_do_not_use_it(
        self, fake_root: Path, tmp_path: Path
    ) -> None:
        _fake_skill(fake_root, "good", [], {})
        _fake_skill(fake_root, "bad", ["missing.py"], {})

        install_skills(["good"], tmp_path / "out")

        assert _installed(tmp_path / "out") == ["cat/good/SKILL.md"]


class TestWhatTheWizardNamesMatchesWhatIsWritten:
    def test_a_fresh_install_has_nothing_unrecognized(
        self, fake_root: Path, tmp_path: Path
    ) -> None:
        _fake_skill(fake_root, "one", ["scripts/", "../shared.py"], {"scripts/a.py": "a"})
        (fake_root / "cat" / "shared.py").write_text("s")
        install_skills(["one"], tmp_path / "out")

        assert unrecognized_entries(tmp_path / "out") == []

    def test_a_file_in_no_skills_list_is_named(self, fake_root: Path, tmp_path: Path) -> None:
        _fake_skill(fake_root, "one", [], {})
        install_skills(["one"], tmp_path / "out")
        (tmp_path / "out" / "cat" / "one" / "extra.py").write_text("x")

        assert unrecognized_entries(tmp_path / "out") == [Path("cat/one/extra.py")]

    def test_a_skill_with_a_broken_list_does_not_hide_the_others(
        self, fake_root: Path, tmp_path: Path
    ) -> None:
        _fake_skill(fake_root, "good", [], {})
        _fake_skill(fake_root, "bad", ["missing.py"], {})
        install_skills(["good"], tmp_path / "out")

        assert unrecognized_entries(tmp_path / "out") == []


class TestTheGroups:
    def test_every_skill_is_in_exactly_one_group_named_after_its_category_folder(self) -> None:
        groups = skill_groups()
        names = [name for members in groups.values() for name in members]

        assert sorted(names) == sorted(skill.name for skill in list_builtin_skills())
        assert len(names) == len(set(names))
        for group, members in groups.items():
            assert (BUILTIN_SKILLS_ROOT / group).is_dir()
            for name in members:
                assert (BUILTIN_SKILLS_ROOT / group / name / "SKILL.md").is_file()

    def test_the_productivity_group_has_its_four_skills(self) -> None:
        assert sorted(skill_groups()["productivity"]) == [
            "brainstorming",
            "communication",
            "documents",
            "presentations",
        ]

    def test_a_group_never_has_the_name_of_a_skill(self) -> None:
        """The picker keys rows by label, so the two cannot share one."""
        assert not set(skill_groups()) & {skill.name for skill in list_builtin_skills()}


class TestAdding:
    def test_it_adds_beside_what_is_there_and_replaces_nothing(
        self, fake_root: Path, tmp_path: Path
    ) -> None:
        _fake_skill(fake_root, "one", ["../shared.py"], {})
        _fake_skill(fake_root, "two", ["../shared.py"], {})
        (fake_root / "cat" / "shared.py").write_text("package")
        out = tmp_path / "out"
        install_skills(["one"], out)
        (out / "cat" / "one" / "SKILL.md").write_text("edited")
        (out / "cat" / "shared.py").write_text("exported")
        (out / "mine.md").write_text("not a built-in")

        paths = add_skills(["two"], out)

        assert paths == [out / "cat" / "two"]
        assert _installed(out) == [
            "cat/one/SKILL.md",
            "cat/shared.py",
            "cat/two/SKILL.md",
            "mine.md",
        ]
        assert (out / "cat" / "one" / "SKILL.md").read_text() == "edited"
        assert (out / "cat" / "shared.py").read_text() == "exported"

    def test_a_broken_list_touches_nothing(self, fake_root: Path, tmp_path: Path) -> None:
        _fake_skill(fake_root, "good", [], {})
        _fake_skill(fake_root, "bad", ["missing.py"], {})
        out = tmp_path / "out"

        with pytest.raises(SkillInstallError):
            add_skills(["good", "bad"], out)

        assert not out.exists()

    def test_a_symlinked_directory_is_refused(self, fake_root: Path, tmp_path: Path) -> None:
        _fake_skill(fake_root, "one", [], {})
        (tmp_path / "real").mkdir()
        (tmp_path / "link").symlink_to(tmp_path / "real")

        with pytest.raises(SkillInstallError):
            add_skills(["one"], tmp_path / "link")

        assert not list((tmp_path / "real").iterdir())


class TestInstalling:
    def test_it_mirrors_the_package_layout(self, tmp_path: Path) -> None:
        dest = tmp_path / "skills"

        paths = install_skills(["data-analysis", "charts"], dest)

        assert paths == [
            dest / "data" / "data-analysis",
            dest / "visualization" / "charts",
        ]
        assert (dest / "data" / "data-analysis" / "SKILL.md").is_file()
        assert (dest / "data" / "data-analysis" / "scripts" / "common.py").is_file()
        assert (dest / "resources" / "seaborn_style.py").is_file()
        assert (dest / "resources" / "palette.json").is_file()
        assert (dest / "resources" / "typography.json").is_file()
        assert list((dest / "resources" / "assets" / "fonts").glob("*.ttf"))
        assert (dest / "data" / "data_utils.py").is_file()

    def test_only_what_the_chosen_skills_need_is_copied(self, tmp_path: Path) -> None:
        dest = tmp_path / "skills"

        install_skills(["brainstorming"], dest)

        assert sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()) == [
            "productivity/brainstorming/SKILL.md"
        ]

    def test_the_returned_paths_are_absolute(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)

        paths = install_skills(["charts"], Path("skills"))

        assert all(path.is_absolute() for path in paths)

    def test_bytecode_is_not_copied(self, tmp_path: Path) -> None:
        source = BUILTIN_SKILLS_ROOT / "data" / "data-analysis" / "scripts" / "__pycache__"
        source.mkdir(exist_ok=True)
        (source / "stale.cpython-312.pyc").write_bytes(b"x")
        try:
            install_skills(["data-analysis"], tmp_path / "skills")
        finally:
            (source / "stale.cpython-312.pyc").unlink()

        assert not list((tmp_path / "skills").rglob("*.pyc"))
        assert not list((tmp_path / "skills").rglob("__pycache__"))

    def test_the_binary_assets_come_along(self, tmp_path: Path) -> None:
        install_skills(["presentations"], tmp_path / "skills")

        dest = tmp_path / "skills"
        template = dest / "productivity" / "presentations" / "template" / "layouts.pptx"
        assert template.is_file()
        assert list((dest / "resources" / "assets" / "fonts").glob("*.ttf"))

    def test_an_unknown_name_raises_before_anything_changes(self, tmp_path: Path) -> None:
        dest = tmp_path / "skills"
        dest.mkdir()
        (dest / "keep.txt").write_text("x")

        with pytest.raises(ValueError, match="nope"):
            install_skills(["charts", "nope"], dest)

        assert (dest / "keep.txt").read_text() == "x"


class TestReplacingTheDirectory:
    def test_a_second_install_with_fewer_skills_removes_the_dropped_ones(
        self, tmp_path: Path
    ) -> None:
        dest = tmp_path / "skills"
        install_skills(["data-analysis", "brainstorming"], dest)

        install_skills(["brainstorming"], dest)

        assert not (dest / "data").exists()
        assert not (dest / "resources").exists()
        assert (dest / "productivity" / "brainstorming" / "SKILL.md").is_file()

    def test_an_edited_skill_gets_the_package_version_back(self, tmp_path: Path) -> None:
        dest = tmp_path / "skills"
        install_skills(["brainstorming"], dest)
        skill_file = dest / "productivity" / "brainstorming" / "SKILL.md"
        skill_file.write_text("my edit")

        install_skills(["brainstorming"], dest)

        assert skill_file.read_text() != "my edit"

    def test_a_file_that_is_not_ours_is_deleted_too(self, tmp_path: Path) -> None:
        dest = tmp_path / "skills"
        (dest / "mine").mkdir(parents=True)
        (dest / "mine" / "SKILL.md").write_text("x")

        install_skills(["charts"], dest)

        assert not (dest / "mine").exists()

    def test_an_empty_choice_leaves_an_empty_directory(self, tmp_path: Path) -> None:
        dest = tmp_path / "skills"
        install_skills(["charts"], dest)

        assert install_skills([], dest) == []

        assert dest.is_dir() and not list(dest.iterdir())

    def test_no_staging_folder_is_left_behind(self, tmp_path: Path) -> None:
        install_skills(["charts"], tmp_path / "skills")
        install_skills(["charts"], tmp_path / "skills")

        assert [p.name for p in tmp_path.iterdir()] == ["skills"]

    def test_a_failure_while_copying_leaves_the_old_directory_as_it_was(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dest = tmp_path / "skills"
        install_skills(["brainstorming"], dest)
        before = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*"))

        def explode(*args: object, **kwargs: object) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(catalog.shutil, "copy2", explode)
        with pytest.raises(OSError, match="disk full"):
            install_skills(["charts"], dest)

        assert sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*")) == before
        assert [p.name for p in tmp_path.iterdir()] == ["skills"]

    def test_a_symlinked_directory_is_refused_and_its_target_untouched(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "my-skills-repo"
        target.mkdir()
        (target / "precious.md").write_text("x")
        link = tmp_path / "skills"
        link.symlink_to(target)

        with pytest.raises(SkillInstallError, match="symbolic link"):
            install_skills(["charts"], link)

        assert (target / "precious.md").read_text() == "x"
        assert link.is_symlink()


class TestWhatAReplaceWouldDeleteThatIsNotOurs:
    def test_a_fresh_install_has_nothing_unrecognized(self, tmp_path: Path) -> None:
        install_skills([skill.name for skill in list_builtin_skills()], tmp_path / "skills")

        assert unrecognized_entries(tmp_path / "skills") == []

    def test_a_missing_directory_has_nothing(self, tmp_path: Path) -> None:
        assert unrecognized_entries(tmp_path / "nope") == []

    def test_a_hand_made_skill_is_named_once(self, tmp_path: Path) -> None:
        dest = tmp_path / "skills"
        install_skills(["charts"], dest)
        (dest / "mine").mkdir()
        (dest / "mine" / "SKILL.md").write_text("x")
        (dest / "mine" / "extra.md").write_text("x")

        assert unrecognized_entries(dest) == [Path("mine")]

    def test_a_stray_file_inside_a_known_skill_is_named(self, tmp_path: Path) -> None:
        dest = tmp_path / "skills"
        install_skills(["charts"], dest)
        (dest / "visualization" / "charts" / "notes.txt").write_text("x")

        assert unrecognized_entries(dest) == [Path("visualization/charts/notes.txt")]

    def test_a_skill_from_an_older_version_is_not_an_edit(self, tmp_path: Path) -> None:
        """Content is never compared, or every upgrade would look like a hand edit."""
        dest = tmp_path / "skills"
        install_skills(["charts"], dest)
        (dest / "visualization" / "charts" / "SKILL.md").write_text("an older version")

        assert unrecognized_entries(dest) == []

    def test_bytecode_from_running_a_script_is_not_named(self, tmp_path: Path) -> None:
        dest = tmp_path / "skills"
        install_skills(["data-analysis"], dest)
        cache = dest / "data" / "data-analysis" / "scripts" / "__pycache__"
        cache.mkdir()
        (cache / "common.cpython-312.pyc").write_bytes(b"x")

        assert unrecognized_entries(dest) == []


class TestTheInstalledCopyWorks:
    @pytest.mark.asyncio
    async def test_every_installed_skill_loads_from_its_own_folder(self, tmp_path: Path) -> None:
        dest = tmp_path / "skills"
        paths = install_skills([skill.name for skill in list_builtin_skills()], dest)

        index = await LocalDirSkillSource(paths=paths).index(SkillQuery(task=""))

        # A skill's name is what its file declares, not its folder (`presentations` is
        # `slides-style`).
        assert sorted(d.name for d in index.skills) == sorted(
            parse_skill_file(skill.skill_file, source="builtin")[0].name
            for skill in list_builtin_skills()
        )

    @pytest.mark.parametrize("skill", DATA_SKILLS)
    def test_a_data_skills_shared_modules_are_where_its_scripts_look(
        self, skill: str, tmp_path: Path
    ) -> None:
        """Each `common.py` finds `data_utils` two folders up and `seaborn_style` in
        `resources/` three up, from its own place; the copy has to keep both."""
        dest = tmp_path / "skills"
        install_skills([skill], dest)
        common = dest / "data" / skill / "scripts" / "common.py"

        assert (common.resolve().parents[2] / "data_utils.py").is_file()
        resources = common.resolve().parents[3] / "resources"
        assert (resources / "seaborn_style.py").is_file()
        assert (resources / "palette.json").is_file()
        assert (resources / "typography.json").is_file()

    def test_the_charts_skill_names_a_style_file_that_is_where_it_says(
        self, tmp_path: Path
    ) -> None:
        """The SKILL.md tells the model `<its folder>/../../resources/seaborn_style.py`;
        in an installed tree that has to be a real file."""
        dest = tmp_path / "skills"
        (charts,) = install_skills(["charts"], dest)

        text = (charts / "SKILL.md").read_text(encoding="utf-8")
        assert "../../resources/seaborn_style.py" in text
        assert (charts / ".." / ".." / "resources" / "seaborn_style.py").resolve().is_file()

    def test_the_docs_skill_carries_the_pages_its_index_names(self, tmp_path: Path) -> None:
        """In a source checkout `pages` is a link to `docs/docs`: the copy has to hold
        the pages themselves, or the index points the model at nothing."""
        (docs,) = install_skills(["domyn-docs"], tmp_path / "skills")

        pages = re.findall(r"`(pages/[^`]+\.md)`", (docs / "SKILL.md").read_text(encoding="utf-8"))
        assert pages
        assert not (docs / "pages").is_symlink()
        assert [p for p in pages if not (docs / p).is_file() or (docs / p).is_symlink()] == []

    def test_a_script_runs_from_the_copy(self, tmp_path: Path) -> None:
        """Not only that the files are there: a fresh interpreter, started in the
        installed `scripts/` folder as the agent would, imports through the copy."""
        dest = tmp_path / "skills"
        install_skills(["data-analysis"], dest)
        scripts = dest / "data" / "data-analysis" / "scripts"

        result = subprocess.run(
            [sys.executable, "-I", "-c", _IMPORT_COMMON.format(scripts=str(scripts))],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == str(dest / "data" / "data_utils.py")


_IMPORT_COMMON = """
import sys
sys.path.insert(0, {scripts!r})
import common
print(common.data_utils.__file__)
"""
