"""``domyn export``: the configuration and what it reads, in a folder of its own."""

import asyncio
import copy
import os
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from domyn.agents.cli.cli import app
from domyn.agents.interfaces.cli import config as cli_config
from domyn.agents.interfaces.cli import export as cli_export
from domyn.agents.interfaces.cli.config import collect_skill_workspace_roots
from domyn.agents.interfaces.cli.export import ExportError, export_configuration
from domyn.agents.skills import LocalDirSkillSource, SkillQuery


def _skill(folder: Path, name: str, *, external: list[str] | None = None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    meta = f"name: {name}\ndescription: {name} skill\n"
    if external:
        meta += "external_files:\n" + "".join(f"  - {e}\n" for e in external)
    (folder / "SKILL.md").write_text(f"---\n{meta}---\n\nbody\n", encoding="utf-8")
    return folder


def _config(*paths: str, **agent: Any) -> dict[str, Any]:
    return {
        "version": "1.0",
        "entrypoint": "Assistant",
        "llm_providers": [{"name": "p", "type": "vllm", "model_name": "m"}],
        "agents": [
            {
                "name": "Assistant",
                "llm_provider": "p",
                "skills": {"source": {"type": "local_dir", "paths": list(paths)}},
                **agent,
            }
        ],
    }


def _exported(dest: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load((dest / "config.yaml").read_text(encoding="utf-8"))
    return loaded


def _paths(raw: dict[str, Any]) -> list[str]:
    paths: list[str] = raw["agents"][0]["skills"]["source"]["paths"]
    return paths


P = "${DOMYN_PROJECT_DIR}/"
"""How the exported config names a file copied into the export."""


def _loaded_skills(
    raw: dict[str, Any], cwd: Path, project: Path | None = None
) -> dict[str, tuple[bool, str | None]]:
    """What the loader makes of ``raw`` when started in ``cwd``, with ``project`` as
    ``DOMYN_PROJECT_DIR`` (an export is a project)."""
    before = Path.cwd()
    os.chdir(cwd)
    previous = os.environ.get("DOMYN_PROJECT_DIR")
    if project is not None:
        os.environ["DOMYN_PROJECT_DIR"] = str(project)
    try:
        source = LocalDirSkillSource(paths=collect_skill_workspace_roots(raw))
        loaded = asyncio.run(source.load(SkillQuery(task="x")))
    finally:
        os.chdir(before)
        if previous is None:
            os.environ.pop("DOMYN_PROJECT_DIR", None)
        else:
            os.environ["DOMYN_PROJECT_DIR"] = previous
    return {d.name: (d.available, d.unavailable_reason) for d in loaded.index.skills}


@pytest.fixture(autouse=True)
def autodiscovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real autodiscovery entries, which the CLI conftest points nowhere. Safe
    here: every export is given a temporary working directory to read them against."""
    monkeypatch.setattr(
        cli_config, "LOCAL_SKILLS_DIRS", (Path(".domyn/skills"), Path(".agents/skills"))
    )


@pytest.fixture
def machine(tmp_path: Path) -> dict[str, Path]:
    """A working directory, setup's install directory and a home, all empty."""
    places = {name: tmp_path / name for name in ("cwd", "install", "home")}
    for place in places.values():
        place.mkdir()
    return places


def _export(raw: dict[str, Any], machine: dict[str, Path], dest: Path) -> Any:
    return export_configuration(raw, dest, cwd=machine["cwd"], install_dir=machine["install"])


class TestTheExportLoadsTheSameSkills:
    def test_a_group_folder_is_walked_at_any_depth(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        group = tmp_path / "mine"
        _skill(group / "writing" / "essay", "essay")
        _skill(group / "deep" / "er" / "notes", "notes")
        raw = _config(str(group))

        report = _export(raw, machine, tmp_path / "out")

        assert _paths(_exported(report.dest)) == [P + "skills/mine"]
        assert (report.dest / "skills/mine/deep/er/notes/SKILL.md").is_file()
        assert _loaded_skills(
            _exported(report.dest), machine["cwd"], report.dest
        ) == _loaded_skills(raw, machine["cwd"])

    def test_external_files_above_the_skill_keep_their_place(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        shared = tmp_path / "lib" / "shared.py"
        shared.parent.mkdir()
        shared.write_text("x = 1\n", encoding="utf-8")
        skill = _skill(tmp_path / "lib" / "cat" / "tool", "tool", external=["../../shared.py"])
        raw = _config(str(skill))

        report = _export(raw, machine, tmp_path / "out")

        # The anchor moved up to the folder holding both, and the path into it is kept.
        assert _paths(_exported(report.dest)) == [P + "skills/lib/cat/tool"]
        assert (report.dest / "skills/lib/shared.py").read_text() == "x = 1\n"
        loaded = _loaded_skills(_exported(report.dest), machine["cwd"], report.dest)
        assert loaded == {"tool": (True, None)} == _loaded_skills(raw, machine["cwd"])

    def test_a_relative_path_is_read_against_the_working_directory(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        _skill(machine["cwd"] / "team" / "review", "review")

        report = _export(_config("team"), machine, tmp_path / "out")

        assert _paths(_exported(report.dest)) == [P + "skills/team"]
        assert set(_loaded_skills(_exported(report.dest), machine["cwd"], report.dest)) == {
            "review"
        }

    def test_a_file_shared_by_two_entries_is_copied_once(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        root = tmp_path / "pack"
        _skill(root / "a", "a")
        _skill(root / "b", "b")

        report = _export(_config(str(root / "a"), str(root)), machine, tmp_path / "out")

        # The inner anchor is merged into the outer one, so both entries share a copy.
        assert _paths(_exported(report.dest)) == [P + "skills/pack/a", P + "skills/pack"]
        assert report.files_copied == 2

    def test_two_folders_with_one_name_get_two_places(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        _skill(tmp_path / "x" / "pack" / "one", "one")
        _skill(tmp_path / "y" / "pack" / "two", "two")
        raw = _config(str(tmp_path / "x" / "pack"), str(tmp_path / "y" / "pack"))

        report = _export(raw, machine, tmp_path / "out")

        assert _paths(_exported(report.dest)) == [P + "skills/pack", P + "skills/pack-2"]
        assert set(_loaded_skills(_exported(report.dest), machine["cwd"], report.dest)) == {
            "one",
            "two",
        }

    def test_a_folder_called_skills_is_named_after_its_parent(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        _skill(tmp_path / "project" / "skills" / "one", "one")

        report = _export(_config(str(tmp_path / "project" / "skills")), machine, tmp_path / "out")

        assert _paths(_exported(report.dest)) == [P + "skills/project"]


class TestSetupsSkills:
    def test_only_what_the_skills_declare_is_copied(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        install = machine["install"]
        _skill(install / "data" / "analysis", "analysis", external=["../../plotly_style.py"])
        (install / "plotly_style.py").write_text("style\n", encoding="utf-8")
        # Beside it, nothing a listed skill declares: a skill not chosen, a stray file.
        _skill(install / "data" / "other", "other")
        (install / "stray.py").write_text("x\n", encoding="utf-8")

        report = _export(_config(str(install / "data" / "analysis")), machine, tmp_path / "out")

        assert _paths(_exported(report.dest)) == [P + "builtin-skills/data/analysis"]
        exported = sorted(
            p.relative_to(report.dest).as_posix() for p in report.dest.rglob("*") if p.is_file()
        )
        assert exported == [
            ".info",
            "builtin-skills/data/analysis/SKILL.md",
            "builtin-skills/plotly_style.py",
            "config.yaml",
        ]

    def test_their_folders_do_not_take_names_from_other_skills(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        _skill(machine["install"] / "data" / "analysis", "analysis")
        _skill(tmp_path / "data" / "mine", "mine")
        raw = _config(str(machine["install"] / "data" / "analysis"), str(tmp_path / "data"))

        report = _export(raw, machine, tmp_path / "out")

        assert _paths(_exported(report.dest)) == [
            P + "builtin-skills/data/analysis",
            P + "skills/data",
        ]
        assert _loaded_skills(
            _exported(report.dest), machine["cwd"], report.dest
        ) == _loaded_skills(raw, machine["cwd"])


class TestAutodiscovery:
    def test_both_folders_are_merged_into_skills(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        _skill(machine["cwd"] / ".domyn" / "skills" / "mine", "mine")
        _skill(machine["cwd"] / ".agents" / "skills" / "shared", "shared")
        _skill(machine["install"] / "cat" / "builtin", "builtin")
        raw = _config(
            ".domyn/skills", ".agents/skills", str(machine["install"] / "cat" / "builtin")
        )

        report = _export(raw, machine, tmp_path / "out")

        # Straight into `skills/`, each listed on its own where the first entry was:
        # `skills/` itself also holds the other listed folders.
        assert _paths(_exported(report.dest)) == [
            P + "skills/mine",
            P + "skills/shared",
            P + "builtin-skills/cat/builtin",
        ]
        assert (report.dest / "skills/mine/SKILL.md").is_file()
        assert (report.dest / "skills/shared/SKILL.md").is_file()
        # Run from anywhere, the export loads what a run in the original folder loads.
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        assert _loaded_skills(_exported(report.dest), elsewhere, report.dest) == (
            _loaded_skills(raw, machine["cwd"])
        )

    def test_a_skill_also_listed_by_its_path_is_copied_once(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        # The same folder reached twice: through autodiscovery, and named on its own
        # (an agent following the domyn-cli skill, or a hand edit).
        art = _skill(machine["cwd"] / ".domyn" / "skills" / "ascii-art", "ascii-art")
        _skill(machine["cwd"] / ".domyn" / "skills" / "other", "other")
        raw = _config(".domyn/skills", ".agents/skills", str(art))

        report = _export(raw, machine, tmp_path / "out")

        assert not any("already taken" in warning for warning in report.warnings)
        assert not (report.dest / "skills/ascii-art-2").exists()
        assert _paths(_exported(report.dest)) == [P + "skills/ascii-art", P + "skills/other"]
        assert _loaded_skills(_exported(report.dest), tmp_path, report.dest) == (
            _loaded_skills(raw, machine["cwd"])
        )

    def test_an_identical_copy_from_elsewhere_is_copied_once(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        elsewhere = _skill(tmp_path / "elsewhere" / "ascii-art", "ascii-art", external=["x.py"])
        (elsewhere / "x.py").write_text("print('art')\n")
        copy = machine["cwd"] / ".domyn" / "skills" / "ascii-art"
        shutil.copytree(elsewhere, copy)
        raw = _config(".domyn/skills", str(elsewhere))

        report = _export(raw, machine, tmp_path / "out")

        assert not any("already taken" in warning for warning in report.warnings)
        assert _paths(_exported(report.dest)) == [P + "skills/ascii-art"]
        assert len(list(report.dest.rglob("SKILL.md"))) == 1

    def test_a_copy_of_a_built_in_is_exported_as_the_user_s_own(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        """The built-ins are in `builtin-skills/`, so the copy never meets them, and is
        still listed first, so it still wins the name."""
        builtin = _skill(machine["install"] / "cat" / "art", "art")
        shutil.copytree(builtin, machine["cwd"] / ".domyn" / "skills" / "cat" / "art")
        raw = _config(".domyn/skills", str(builtin))

        report = _export(raw, machine, tmp_path / "out")

        assert _paths(_exported(report.dest)) == [
            P + "skills/cat/art",
            P + "builtin-skills/cat/art",
        ]
        assert _loaded_skills(_exported(report.dest), report.dest, report.dest) == (
            _loaded_skills(raw, machine["cwd"])
        )

    @pytest.mark.parametrize("change", ["a byte", "an extra file", "a missing file"])
    def test_a_copy_that_differs_is_still_its_own(
        self, change: str, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        elsewhere = _skill(tmp_path / "elsewhere" / "ascii-art", "ascii-art", external=["."])
        (elsewhere / "x.py").write_text("print('art')\n")
        copy = machine["cwd"] / ".domyn" / "skills" / "ascii-art"
        shutil.copytree(elsewhere, copy)
        if change == "a byte":
            (copy / "x.py").write_text("print('Art')\n")
        elif change == "an extra file":
            (elsewhere / "y.py").write_text("")
        else:
            (copy / "x.py").unlink()
        raw = _config(".domyn/skills", str(elsewhere))

        report = _export(raw, machine, tmp_path / "out")

        assert any("already taken" in warning for warning in report.warnings)
        assert _paths(_exported(report.dest)) == [P + "skills/ascii-art-2", P + "skills/ascii-art"]

    def test_a_skill_inside_a_listed_folder_is_copied_once(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        # The autodiscovery folder itself listed by its path too: every skill in it is
        # reached twice, and its copy is where each one is listed.
        folder = machine["cwd"] / ".domyn" / "skills"
        _skill(folder / "cat" / "ascii-art", "ascii-art")
        raw = _config(".domyn/skills", str(folder))

        report = _export(raw, machine, tmp_path / "out")

        assert not any("already taken" in warning for warning in report.warnings)
        copies = list(report.dest.rglob("SKILL.md"))
        assert len(copies) == 1, copies
        assert _loaded_skills(_exported(report.dest), tmp_path, report.dest) == (
            _loaded_skills(raw, machine["cwd"])
        )

    def test_categories_are_kept_with_what_they_share(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        category = machine["cwd"] / ".domyn" / "skills" / "writing"
        _skill(category / "essay", "essay", external=["../style.md"])
        (category / "style.md").write_text("style\n", encoding="utf-8")
        raw = _config(".domyn/skills")

        report = _export(raw, machine, tmp_path / "out")

        assert _paths(_exported(report.dest)) == [P + "skills/writing/essay"]
        assert (report.dest / "skills/writing/essay/SKILL.md").is_file()
        assert (report.dest / "skills/writing/style.md").read_text() == "style\n"
        assert _loaded_skills(_exported(report.dest), report.dest, report.dest) == (
            _loaded_skills(raw, machine["cwd"])
        )

    def test_a_name_in_both_is_copied_from_domyn_only(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        ours = _skill(machine["cwd"] / ".domyn" / "skills" / "report", "report")
        (ours / "SKILL.md").write_text(
            "---\nname: report\ndescription: domyn's\n---\n", encoding="utf-8"
        )
        theirs = _skill(machine["cwd"] / ".agents" / "skills" / "report", "report")
        raw = _config(".domyn/skills", ".agents/skills")

        report = _export(raw, machine, tmp_path / "out")

        copied = (report.dest / "skills/report/SKILL.md").read_text(encoding="utf-8")
        assert "domyn's" in copied
        assert any(str(theirs) in d and "takes precedence" in d for d in report.dropped)
        assert _loaded_skills(_exported(report.dest), report.dest, report.dest) == (
            _loaded_skills(raw, machine["cwd"])
        )

    def test_a_clash_by_name_alone_is_settled_too(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        """Different folders, one name: the loader would still read only the first."""
        _skill(machine["cwd"] / ".domyn" / "skills" / "a", "report")
        _skill(machine["cwd"] / ".agents" / "skills" / "b", "report")

        report = _export(_config(".domyn/skills", ".agents/skills"), machine, tmp_path / "out")

        assert _paths(_exported(report.dest)) == [P + "skills/a"]
        assert not (report.dest / "skills/b").exists()

    def test_the_place_a_builtin_has_is_not_taken_and_the_skill_still_wins(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        _skill(machine["install"] / "data" / "analysis", "analysis")
        local = _skill(machine["cwd"] / ".domyn" / "skills" / "data" / "analysis", "analysis")
        # The user's own, not a copy of the built-in.
        with (local / "SKILL.md").open("a", encoding="utf-8") as f:
            f.write("my own version\n")
        raw = _config(".domyn/skills", str(machine["install"] / "data" / "analysis"))

        report = _export(raw, machine, tmp_path / "out")

        assert _paths(_exported(report.dest)) == [
            P + "skills/data/analysis",
            P + "builtin-skills/data/analysis",
        ]
        assert not any("already taken" in w for w in report.warnings)
        assert _loaded_skills(_exported(report.dest), report.dest, report.dest) == (
            _loaded_skills(raw, machine["cwd"])
        )

    def test_a_category_another_listed_folder_has_gets_a_suffix(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        """``skills/mine`` is listed as a folder, so it is walked: a skill put inside
        it would be loaded a second time, through that entry."""
        _skill(tmp_path / "mine" / "one", "one")
        _skill(machine["cwd"] / ".domyn" / "skills" / "mine" / "two", "two")
        raw = _config(".domyn/skills", str(tmp_path / "mine"))

        report = _export(raw, machine, tmp_path / "out")

        assert _paths(_exported(report.dest)) == [P + "skills/mine-2/two", P + "skills/mine"]
        assert sorted(p.name for p in (report.dest / "skills/mine").iterdir()) == ["one"]

    def test_a_file_shared_from_above_the_folder_keeps_its_place(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        """``../../shared.py`` from ``.agents/skills/x`` is ``.agents/shared.py``:
        copied to ``skills/x/../../shared.py``, the top of the export."""
        (machine["cwd"] / ".agents").mkdir()
        (machine["cwd"] / ".agents" / "shared.py").write_text("x = 1\n", encoding="utf-8")
        _skill(machine["cwd"] / ".agents" / "skills" / "x", "x", external=["../../shared.py"])
        raw = _config(".agents/skills")

        report = _export(raw, machine, tmp_path / "out")

        assert (report.dest / "shared.py").read_text(encoding="utf-8") == "x = 1\n"
        assert report.warnings == []
        assert _loaded_skills(_exported(report.dest), report.dest, report.dest) == (
            _loaded_skills(raw, machine["cwd"])
        )

    def test_a_shared_file_never_lands_outside_the_project(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        (tmp_path / "far.py").write_text("x\n", encoding="utf-8")
        _skill(machine["cwd"] / ".domyn" / "skills" / "x", "x", external=["../../../../far.py"])

        # Not copied, so the entry would lead out of the export: refused, not shipped.
        with pytest.raises(ExportError, match=r"external_files: \.\./\.\./\.\./\.\./far\.py"):
            _export(_config(".domyn/skills"), machine, tmp_path / "out")

        assert not (tmp_path / "out").exists()
        assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("far")) == ["far.py"]

    def test_a_shared_file_never_takes_the_config_s_place(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        # `.domyn/config.yaml`, two levels up from the skill: the top of the export.
        (machine["cwd"] / ".domyn").mkdir()
        (machine["cwd"] / ".domyn" / "config.yaml").write_text("a skill's\n", encoding="utf-8")
        _skill(machine["cwd"] / ".domyn" / "skills" / "x", "x", external=["../../config.yaml"])

        report = _export(_config(".domyn/skills"), machine, tmp_path / "out")

        assert (report.dest / "config.yaml").read_text(encoding="utf-8") != "a skill's\n"
        assert any("where the project's own configuration goes" in w for w in report.warnings)

    def test_a_shared_file_never_lands_where_setup_installs(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        """A later setup of the project rewrites `builtin-skills/` from scratch."""
        shared = machine["cwd"] / ".domyn" / "builtin-skills" / "x.py"
        shared.parent.mkdir(parents=True)
        shared.write_text("x\n", encoding="utf-8")
        _skill(
            machine["cwd"] / ".domyn" / "skills" / "x", "x", external=["../../builtin-skills/x.py"]
        )

        report = _export(_config(".domyn/skills"), machine, tmp_path / "out")

        assert not (report.dest / "builtin-skills").exists()
        assert any("which setup rewrites from scratch" in w for w in report.warnings)

    def test_a_shared_file_may_be_called_like_an_instruction_file(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        """Nothing is read without being declared, so the top is a place like any other."""
        (machine["cwd"] / ".domyn").mkdir()
        (machine["cwd"] / ".domyn" / "SOUL.md").write_text("a skill's\n", encoding="utf-8")
        _skill(machine["cwd"] / ".domyn" / "skills" / "x", "x", external=["../../SOUL.md"])

        report = _export(_config(".domyn/skills"), machine, tmp_path / "out")

        assert (report.dest / "SOUL.md").read_text(encoding="utf-8") == "a skill's\n"
        assert report.instruction_files == []

    def test_two_shared_files_for_one_place_keep_the_first(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        for folder, content in ((".domyn", "domyn\n"), (".agents", "agents\n")):
            category = machine["cwd"] / folder / "skills" / "cat"
            category.mkdir(parents=True)
            (category / "shared.py").write_text(content, encoding="utf-8")
        _skill(machine["cwd"] / ".domyn" / "skills" / "cat" / "a", "a", external=["../shared.py"])
        _skill(machine["cwd"] / ".agents" / "skills" / "cat" / "b", "b", external=["../shared.py"])

        report = _export(_config(".domyn/skills", ".agents/skills"), machine, tmp_path / "out")

        assert (report.dest / "skills/cat/shared.py").read_text(encoding="utf-8") == "domyn\n"
        assert any("the skill uses that one instead" in w for w in report.warnings)

    def test_without_a_skills_folder_is_left_out_quietly(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        report = _export(_config(".domyn/skills", ".agents/skills"), machine, tmp_path / "out")

        assert _paths(_exported(report.dest)) == []
        assert report.dropped == []

    def test_the_project_variable_is_not_one_to_set(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        _skill(tmp_path / "s", "s")

        report = _export(_config(str(tmp_path / "s")), machine, tmp_path / "out")

        assert _paths(_exported(report.dest)) == [P + "skills/s"]
        assert report.env_vars == []


class TestWhatIsLeftOut:
    def test_a_missing_path_is_dropped_and_named(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        report = _export(_config(str(tmp_path / "gone")), machine, tmp_path / "out")

        assert _paths(_exported(report.dest)) == []
        assert report.dropped == [f"{tmp_path / 'gone'} (does not exist)"]

    def test_a_linked_path_is_not_followed(self, machine: dict[str, Path], tmp_path: Path) -> None:
        _skill(tmp_path / "real" / "s", "s")
        (tmp_path / "link").symlink_to(tmp_path / "real")

        report = _export(_config(str(tmp_path / "link")), machine, tmp_path / "out")

        assert _paths(_exported(report.dest)) == []
        assert "which export does not follow" in report.dropped[0]

    def test_hidden_files_in_a_skill_folder_stay_behind(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        # Even under a folder the skill declares: the entry opts in to its files, not
        # to a `.env` or a `.git` that happens to sit there.
        skill = _skill(tmp_path / "s", "s", external=["."])
        (skill / ".env").write_text("KEY=secret\n", encoding="utf-8")
        (skill / ".git").mkdir()
        (skill / ".git" / "HEAD").write_text("ref\n", encoding="utf-8")

        report = _export(_config(str(skill)), machine, tmp_path / "out")

        assert sorted(p.name for p in (report.dest / "skills/s").iterdir()) == ["SKILL.md"]

    @pytest.mark.parametrize("entry", ["../../home/.ssh/id_rsa", "../../home/.ssh/", ".env"])
    def test_a_declared_hidden_file_is_never_exported(
        self, machine: dict[str, Path], tmp_path: Path, entry: str
    ) -> None:
        # Named outright, from a skill listed by path: the export's anchor would climb
        # to take it in, and a stranger's skill could carry a private key out.
        (tmp_path / "home" / ".ssh").mkdir(parents=True)
        (tmp_path / "home" / ".ssh" / "id_rsa").write_text("SECRET\n", encoding="utf-8")
        skill = _skill(tmp_path / "repo" / "s", "s", external=[entry])
        (skill / ".env").write_text("KEY=secret\n", encoding="utf-8")

        # From `skills/s` in the export the entry still lands inside it, on nothing.
        report = _export(_config(str(skill)), machine, tmp_path / "out")

        exported = [p.name for p in report.dest.rglob("*") if p.is_file()]
        assert "id_rsa" not in exported and ".env" not in exported
        assert any("never exported" in d for d in report.dropped)

    def test_a_hidden_file_whose_entry_would_lead_out_refuses_the_export(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        # Deeper: from `skills/s`, `../../../../` climbs out of the export, onto the
        # importer's own home.
        (tmp_path / ".ssh").mkdir()
        (tmp_path / ".ssh" / "id_rsa").write_text("SECRET\n", encoding="utf-8")
        skill = _skill(tmp_path / "a" / "b" / "s", "s", external=["../../../.ssh/id_rsa"])
        dest = tmp_path / "out"

        with pytest.raises(ExportError, match="external_files"):
            _export(_config(str(skill)), machine, dest)
        assert not dest.exists()

    def test_a_skill_inside_a_hidden_folder_keeps_its_files(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        # Hidden is judged below the skill: `~/.claude/skills/s` and its `../shared.py`.
        skill = _skill(tmp_path / ".claude" / "skills" / "s", "s", external=["../shared.py"])
        (skill.parent / "shared.py").write_text("x\n", encoding="utf-8")

        report = _export(_config(str(skill)), machine, tmp_path / "out")

        assert "shared.py" in [p.name for p in report.dest.rglob("*") if p.is_file()]

    def test_an_undeclared_file_is_left_behind_and_named(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        skill = _skill(tmp_path / "s", "s", external=["scripts/"])
        (skill / "scripts").mkdir()
        (skill / "scripts" / "run.py").write_text("x\n", encoding="utf-8")
        (skill / "notes.txt").write_text("private\n", encoding="utf-8")

        report = _export(_config(str(skill)), machine, tmp_path / "out")

        assert (report.dest / "skills/s/scripts/run.py").is_file()
        assert not (report.dest / "skills/s/notes.txt").exists()
        assert any("notes.txt" in w and "external_files" in w for w in report.warnings)


class TestOnlyWhatASkillListsTravels:
    """A skill's files besides its SKILL.md travel only when its ``external_files``
    lists them — what the `domyn-cli` skill tells an agent writing one."""

    def test_the_old_required_key_lists_nothing_so_the_script_stays_behind(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        skill = tmp_path / "ascii-art"
        skill.mkdir()
        (skill / "SKILL.md").write_text(
            "---\nname: ascii-art\ndescription: d\nrequired:\n  - ascii_art.py\n---\nbody\n",
            encoding="utf-8",
        )
        (skill / "ascii_art.py").write_text("x\n", encoding="utf-8")

        report = _export(_config(str(skill)), machine, tmp_path / "out")

        assert sorted(p.name for p in (report.dest / "skills/ascii-art").iterdir()) == ["SKILL.md"]
        assert any("ascii_art.py" in w and "external_files" in w for w in report.warnings)

    def test_the_same_skill_with_external_files_brings_it(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        skill = _skill(tmp_path / "ascii-art", "ascii-art", external=["ascii_art.py"])
        (skill / "ascii_art.py").write_text("x\n", encoding="utf-8")

        report = _export(_config(str(skill)), machine, tmp_path / "out")

        assert (report.dest / "skills/ascii-art/ascii_art.py").read_text() == "x\n"
        assert report.warnings == []

    def test_an_unlisted_file_goes_unnoticed_until_the_export_leaves_it_behind(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        # Here the skill loads as available: nothing checks a file it does not list.
        skill = _skill(tmp_path / "s", "s")
        (skill / "helper.py").write_text("x\n", encoding="utf-8")
        raw = _config(str(skill))
        assert _loaded_skills(raw, machine["cwd"]) == {"s": (True, None)}

        report = _export(raw, machine, tmp_path / "out")

        assert not (report.dest / "skills/s/helper.py").exists()
        assert any("helper.py" in w for w in report.warnings)

    def test_a_listed_file_that_is_missing_is_reported_and_unavailable_in_the_copy(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        skill = _skill(tmp_path / "s", "s", external=["gone.py"])
        raw = _config(str(skill))

        report = _export(raw, machine, tmp_path / "out")

        assert any("'gone.py'" in w and "cannot use it" in w for w in report.warnings)
        available, reason = _loaded_skills(_exported(report.dest), machine["cwd"], report.dest)["s"]
        assert available is False
        assert reason is not None and "gone.py" in reason


class TestSecrets:
    def test_a_literal_api_key_becomes_a_reference(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        raw = _config()
        raw["llm_providers"][0]["api_key"] = "sk-literal"
        original = copy.deepcopy(raw)

        report = _export(raw, machine, tmp_path / "out")

        assert _exported(report.dest)["llm_providers"][0]["api_key"] == "${VLLM_API_KEY}"
        assert "sk-literal" not in (report.dest / "config.yaml").read_text()
        assert report.env_vars == ["VLLM_API_KEY"]
        assert raw == original

    def test_references_and_plain_headers_are_kept(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        raw = _config()
        raw["llm_providers"][0]["api_key"] = "${MY_KEY}"
        raw["mcp_servers"] = [
            {"name": "srv", "url": "u", "headers": {"Authorization": "Bearer t", "Accept": "a"}}
        ]

        report = _export(raw, machine, tmp_path / "out")

        headers = _exported(report.dest)["mcp_servers"][0]["headers"]
        assert headers == {"Authorization": "${SRV_AUTHORIZATION}", "Accept": "a"}
        assert report.env_vars == ["MY_KEY", "SRV_AUTHORIZATION"]


class TestInstructions:
    def test_nothing_is_exported_that_is_not_declared(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        """No folder is read without being declared, the home exported from included."""
        (machine["home"] / "SOUL.md").write_text("soul\n", encoding="utf-8")
        (machine["home"] / "AGENT.md").write_text("agent\n", encoding="utf-8")

        report = _export(_config(), machine, tmp_path / "out")

        assert report.instruction_files == []
        assert not (report.dest / "SOUL.md").exists()
        assert not (report.dest / "instructions").exists()

    def test_declared_paths_are_copied_and_rewritten(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        (machine["home"] / "SOUL.md").write_text("soul\n", encoding="utf-8")
        brief = tmp_path / "project"
        brief.mkdir()
        (brief / "AGENTS.md").write_text("brief\n", encoding="utf-8")
        (tmp_path / "task.md").write_text("task\n", encoding="utf-8")
        declared = [str(brief), str(tmp_path / "task.md"), str(machine["home"] / "SOUL.md")]
        raw = _config(standing_instructions={"source": {"type": "local_dir", "paths": declared}})

        report = _export(raw, machine, tmp_path / "out")

        paths = _exported(report.dest)["agents"][0]["standing_instructions"]["source"]["paths"]
        # Nothing collides, so all go straight into `instructions/`, the soul like any file.
        assert paths == [P + "instructions", P + "instructions/task.md", P + "instructions/SOUL.md"]
        assert (report.dest / "instructions/AGENTS.md").read_text() == "brief\n"
        assert (report.dest / "instructions/task.md").read_text() == "task\n"
        assert (report.dest / "instructions/SOUL.md").read_text() == "soul\n"


class TestInstructionsNesting:
    """Every source goes in ``instructions/``, with the least nesting that keeps them
    all: none when no two files share a name, otherwise a folder per source named by
    the fewest trailing parts of its path that tell them apart. ``./.domyn`` and
    ``./.agents`` are walked as the loader walks them, and each file they hold is a
    source of its own, listed by itself."""

    @staticmethod
    def _raw(*declared: str) -> dict[str, Any]:
        return _config(standing_instructions={"source": {"type": "local_dir", "paths": declared}})

    @staticmethod
    def _declared(dest: Path) -> list[str]:
        paths: list[str] = _exported(dest)["agents"][0]["standing_instructions"]["source"]["paths"]
        return paths

    @staticmethod
    def _write(folder: Path, name: str, text: str) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text(text, encoding="utf-8")

    def test_two_of_one_name_get_a_folder_each(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        a, b = tmp_path / "files" / "dir" / "a", tmp_path / "files" / "dir" / "b"
        self._write(a, "AGENTS.md", "a\n")
        self._write(b, "AGENTS.md", "b\n")

        report = _export(self._raw(str(a), str(b)), machine, tmp_path / "out")

        assert self._declared(report.dest) == [P + "instructions/a", P + "instructions/b"]
        assert (report.dest / "instructions/a/AGENTS.md").read_text() == "a\n"
        assert (report.dest / "instructions/b/AGENTS.md").read_text() == "b\n"

    def test_as_deep_as_it_takes_to_tell_them_apart(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        x, y = tmp_path / "x" / "a", tmp_path / "y" / "a"
        self._write(x, "SOUL.md", "x\n")
        self._write(y, "SOUL.md", "y\n")

        report = _export(self._raw(str(x), str(y)), machine, tmp_path / "out")

        assert self._declared(report.dest) == [P + "instructions/x/a", P + "instructions/y/a"]

    def test_agents_and_agent_md_count_as_one_name(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        """A folder contributes only the first of them: together, one would be lost."""
        self._write(tmp_path / "p", "AGENTS.md", "p\n")
        self._write(tmp_path / "q", "AGENT.md", "q\n")

        report = _export(
            self._raw(str(tmp_path / "p"), str(tmp_path / "q")), machine, tmp_path / "out"
        )

        assert self._declared(report.dest) == [P + "instructions/p", P + "instructions/q"]

    def test_one_local_folder_with_files_goes_straight_in(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        self._write(machine["cwd"] / ".domyn", "SOUL.md", "ours\n")

        report = _export(self._raw(".domyn", ".agents"), machine, tmp_path / "out")

        assert self._declared(report.dest) == [P + "instructions/SOUL.md"]
        assert (report.dest / "instructions/SOUL.md").read_text() == "ours\n"

    def test_both_local_folders_without_a_clash_share_it(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        self._write(machine["cwd"] / ".domyn", "SOUL.md", "soul\n")
        self._write(machine["cwd"] / ".agents", "AGENTS.md", "task\n")

        report = _export(self._raw(".domyn", ".agents"), machine, tmp_path / "out")

        assert self._declared(report.dest) == [
            P + "instructions/SOUL.md",
            P + "instructions/AGENTS.md",
        ]

    def test_both_local_folders_with_a_clash_keep_a_folder_each_in_order(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        for folder in (".domyn", ".agents"):
            self._write(machine["cwd"] / folder, "SOUL.md", f"{folder}\n")
        self._write(machine["cwd"] / ".agents", "AGENTS.md", "task\n")

        report = _export(self._raw(".domyn", ".agents"), machine, tmp_path / "out")

        # No leading dot, so nothing lands in a folder the search tools skip.
        assert self._declared(report.dest) == [
            P + "instructions/domyn/SOUL.md",
            P + "instructions/agents/SOUL.md",
            P + "instructions/agents/AGENTS.md",
        ]
        assert (report.dest / "instructions/domyn/SOUL.md").read_text() == ".domyn\n"
        assert (report.dest / "instructions/agents/SOUL.md").read_text() == ".agents\n"
        assert (report.dest / "instructions/agents/AGENTS.md").read_text() == "task\n"

    def test_a_local_folder_is_walked_without_domyn_s_own_folders(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        """As the loader walks it: subfolders included, the state folders skipped."""
        self._write(machine["cwd"] / ".domyn", "AGENTS.md", "top\n")
        self._write(machine["cwd"] / ".domyn" / "team", "AGENTS.md", "team\n")
        self._write(machine["cwd"] / ".domyn" / "skills", "AGENTS.md", "not an instruction\n")

        report = _export(self._raw(".domyn"), machine, tmp_path / "out")

        declared = self._declared(report.dest)
        assert len(declared) == 2
        texts = [(report.dest / d.removeprefix(P)).read_text() for d in declared]
        assert texts == ["top\n", "team\n"]

    def test_a_local_folder_with_neither_file_is_left_out_quietly(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        (machine["cwd"] / ".domyn").mkdir()

        report = _export(self._raw(".domyn", ".agents"), machine, tmp_path / "out")

        assert self._declared(report.dest) == []
        assert report.dropped == []

    def test_a_listed_folder_with_neither_file_is_named(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        (tmp_path / "empty").mkdir()

        report = _export(self._raw(str(tmp_path / "empty")), machine, tmp_path / "out")

        assert self._declared(report.dest) == []
        assert any("no SOUL.md or AGENTS.md in it" in d for d in report.dropped)


class TestTheDestination:
    def test_a_folder_with_something_in_it_is_refused(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        dest = tmp_path / "out"
        dest.mkdir()
        (dest / "keep").write_text("mine\n", encoding="utf-8")

        with pytest.raises(ExportError):
            _export(_config(), machine, dest)
        assert sorted(p.name for p in dest.iterdir()) == ["keep"]

    def test_an_empty_folder_is_used(self, machine: dict[str, Path], tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()

        report = _export(_config(), machine, tmp_path / "out")

        assert (report.dest / "config.yaml").is_file()
        assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []


class TestTheCommand:
    def test_exports_the_named_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = tmp_path / "config.yaml"
        source.write_text(yaml.safe_dump(_config()), encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        result = CliRunner().invoke(app, ["export", "shared", "--config", str(source)])

        assert result.exit_code == 0, result.output
        assert (tmp_path / "shared" / "config.yaml").is_file()

    def test_without_a_config_says_so(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(
            app, ["export", str(tmp_path / "shared"), "--config", str(tmp_path / "none.yaml")]
        )

        assert result.exit_code == 1
        assert not (tmp_path / "shared").exists()


class TestNothingOutsideTheExportIsNamed:
    def test_tool_and_hook_code_is_copied_in(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        for folder in ("a", "b"):
            (tmp_path / folder).mkdir()
            (tmp_path / folder / "tools.py").write_text(f"# {folder}\n", encoding="utf-8")
        raw = {
            **_config(),
            "tools": [
                {"name": "one", "type": "function", "path": str(tmp_path / "a/tools.py")},
                {"name": "two", "type": "function", "path": str(tmp_path / "a/tools.py")},
                {"name": "three", "type": "function", "path": str(tmp_path / "b/tools.py")},
            ],
            "hooks": [{"name": "h", "path": str(tmp_path / "b/tools.py"), "trigger": "x"}],
        }

        report = _export(raw, machine, tmp_path / "out")

        exported = _exported(report.dest)
        assert [t["path"] for t in exported["tools"]] == [
            P + "code/tools.py",
            P + "code/tools.py",
            P + "code/tools-2.py",
        ]
        assert exported["hooks"][0]["path"] == P + "code/tools-2.py"
        assert (report.dest / "code/tools-2.py").read_text(encoding="utf-8") == "# b\n"

    @pytest.mark.parametrize(
        ("extra", "named"),
        [
            ({"tools": [{"name": "t", "type": "function", "path": "/no/such.py"}]}, "/no/such.py"),
            ({"tool_bundles": [{"name": "b", "workspace": "/srv/work"}]}, "/srv/work"),
            (
                {"tool_bundles": [{"name": "b", "browser": {"executable_path": "/opt/x"}}]},
                "/opt/x",
            ),
            (
                {"tool_bundles": [{"name": "b", "browser": {"binary": "~/bin/ab"}}]},
                "~/bin/ab",
            ),
            ({"oauth_providers": [{"name": "o", "storage_dir": "/var/t"}]}, "/var/t"),
        ],
        ids=["missing-code", "workspace", "browser", "binary", "oauth"],
    )
    def test_a_path_it_cannot_copy_refuses_the_export(
        self, machine: dict[str, Path], tmp_path: Path, extra: dict[str, Any], named: str
    ) -> None:
        dest = tmp_path / "out"

        with pytest.raises(ExportError, match="outside the exported folder") as raised:
            _export({**_config(), **extra}, machine, dest)

        assert named in str(raised.value)
        assert not dest.exists()

    def test_a_browser_found_on_path_is_not_a_path(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        raw = {**_config(), "tool_bundles": [{"name": "b", "browser": {"binary": "ab"}}]}

        report = _export(raw, machine, tmp_path / "out")

        assert _exported(report.dest)["tool_bundles"][0]["browser"]["binary"] == "ab"


class TestExternalFilesStayInside:
    def test_a_skill_listed_by_path_exports_and_imports_working(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        # The anchor climbs to take `../../shared/style.md` in: inside, so fine, and the
        # skill finds it in the export and in the project imported from it.
        from domyn.agents.interfaces.cli.project_import import import_project

        (tmp_path / "lib" / "shared").mkdir(parents=True)
        (tmp_path / "lib" / "shared" / "style.md").write_text("s\n", encoding="utf-8")
        skill = _skill(tmp_path / "lib" / "skills" / "s", "s", external=["../../shared/style.md"])

        report = _export(_config(str(skill)), machine, tmp_path / "out")
        assert _loaded_skills(_exported(report.dest), machine["cwd"], report.dest)["s"] == (
            True,
            None,
        )

        project = tmp_path / "project"
        imported = import_project(report.dest, project)
        assert _loaded_skills(imported.config, machine["cwd"], project)["s"] == (True, None)

    def test_an_import_whose_skill_leads_out_is_refused(self, tmp_path: Path) -> None:
        # Hand-edited after export: the next export of that project would copy the
        # importer's file in.
        from domyn.agents.interfaces.cli.project_import import ProjectImportError, import_project

        source = tmp_path / "export"
        _skill(source / "skills" / "s", "s", external=["../../../Documents/secret.pdf"])
        (source / "config.yaml").write_text(
            yaml.safe_dump(_config(P + "skills/s")), encoding="utf-8"
        )

        with pytest.raises(ProjectImportError, match=r"secret\.pdf"):
            import_project(source, tmp_path / "project")
        assert not (tmp_path / "project").exists()


class TestCopiedCodeImports:
    def _export_tool(self, machine: dict[str, Path], tmp_path: Path, code: str) -> Any:
        folder = tmp_path / "mine"
        folder.mkdir(exist_ok=True)
        (folder / "tool.py").write_text(code, encoding="utf-8")
        raw = {
            **_config(),
            "tools": [{"name": "t", "type": "function", "path": str(folder / "tool.py")}],
        }
        return _export(raw, machine, tmp_path / "out")

    def test_the_standard_library_and_domyn_s_own_dependencies_say_nothing(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        report = self._export_tool(
            machine,
            tmp_path,
            "import json, os.path\nimport httpx, yaml\nfrom domyn.agents.core import Part\n",
        )

        assert not [w for w in report.warnings if "code/tool.py" in w]

    def test_a_module_beside_the_file_is_named(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        (tmp_path / "mine").mkdir()
        (tmp_path / "mine" / "helpers.py").write_text("x = 1\n", encoding="utf-8")

        report = self._export_tool(machine, tmp_path, "import helpers\nfrom .sibling import y\n")

        (warning,) = [w for w in report.warnings if "beside" in w]
        assert "helpers" in warning and ".sibling" in warning
        assert not (report.dest / "code" / "helpers.py").exists()

    def test_a_package_domyn_does_not_install_is_named_with_its_distribution(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        # `pytest` is installed here, and no requirement of domyn's.
        report = self._export_tool(machine, tmp_path, "def f():\n    import pytest\n")

        (warning,) = [w for w in report.warnings if "needs packages" in w]
        assert "pytest (pytest)" in warning

    def test_a_module_found_nowhere_is_named(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        report = self._export_tool(machine, tmp_path, "import no_such_module_anywhere\n")

        assert any(
            "no_such_module_anywhere" in w and "cannot be found here" in w for w in report.warnings
        )

    def test_a_file_that_does_not_parse_is_said_so(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        report = self._export_tool(machine, tmp_path, "def (:\n")

        assert any("imports were not checked" in w for w in report.warnings)


class TestTheCopyChecksBothEnds:
    """`_copy_into`, the last check before a file is read: whatever the plan says."""

    def _file(self, tmp_path: Path, name: str = "a.md") -> Path:
        path = tmp_path / "src" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x\n", encoding="utf-8")
        return path

    def test_a_planned_file_is_copied(self, tmp_path: Path) -> None:
        source = self._file(tmp_path)
        staging = tmp_path / "staging"
        staging.mkdir()

        cli_export._copy_into(staging, Path("skills/a.md"), source, {source})

        assert (staging / "skills/a.md").read_text(encoding="utf-8") == "x\n"

    @pytest.mark.parametrize("relative", ["../escape.md", "skills/../../escape.md", "/tmp/x.md"])
    def test_a_target_outside_the_folder_is_refused(self, tmp_path: Path, relative: str) -> None:
        source = self._file(tmp_path)
        staging = tmp_path / "staging"
        staging.mkdir()

        with pytest.raises(ExportError, match="outside the exported folder"):
            cli_export._copy_into(staging, Path(relative), source, {source})
        assert not (tmp_path / "escape.md").exists()

    def test_a_source_the_plan_did_not_choose_is_refused(self, tmp_path: Path) -> None:
        source = self._file(tmp_path)
        staging = tmp_path / "staging"
        staging.mkdir()

        with pytest.raises(ExportError, match="not chosen"):
            cli_export._copy_into(staging, Path("a.md"), source, set())

    @pytest.mark.parametrize("kind", ["link", "credentials", "folder"])
    def test_a_source_that_is_not_a_plain_file_is_refused(self, tmp_path: Path, kind: str) -> None:
        staging = tmp_path / "staging"
        staging.mkdir()
        if kind == "link":
            source = tmp_path / "src" / "link.md"
            self._file(tmp_path).parent.joinpath("link.md").symlink_to(tmp_path / "src" / "a.md")
        elif kind == "credentials":
            source = self._file(tmp_path, "credentials")
        else:
            source = tmp_path / "src" / "dir"
            source.mkdir(parents=True)

        with pytest.raises(ExportError, match="not a file export may copy"):
            cli_export._copy_into(staging, Path("x"), source, {source})
