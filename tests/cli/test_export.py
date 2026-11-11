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
from domyn.agents.tools.builtin.session import StateFolders


def _skill(folder: Path, name: str, *, external: list[str] | None = None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    meta = f"name: {name}\ndescription: {name} skill\n"
    if external:
        meta += "external_files:\n" + "".join(f"  - {e}\n" for e in external)
    (folder / "SKILL.md").write_text(f"---\n{meta}---\n\nbody\n", encoding="utf-8")
    return folder


def _config(*paths: str, **agent: Any) -> dict[str, Any]:
    """A configuration whose skills are read from ``paths``, each whole, through a
    resource; a ``standing_instructions`` given as ``{source: {type: local_dir, paths}}``
    is read through one too."""
    config: dict[str, Any] = {
        "version": "1.0",
        "entrypoint": "Assistant",
        "llm_providers": [{"name": "p", "type": "vllm", "model_name": "m"}],
        "resources": [],
        "agents": [{"name": "Assistant", "llm_provider": "p", **agent}],
    }
    if paths:
        config["resources"].append({"id": "skills", "type": "local_dir", "paths": list(paths)})
        config["agents"][0]["skills"] = {
            "source": {"type": "resource", "resource": "skills", "path": "."}
        }
    declared = config["agents"][0].get("standing_instructions")
    if isinstance(declared, dict) and declared["source"].get("type") == "local_dir":
        config["resources"].append(
            {"id": "instructions", "type": "local_dir", "paths": list(declared["source"]["paths"])}
        )
        config["agents"][0]["standing_instructions"] = {
            "source": {"type": "resource", "resource": "instructions"}
        }
    return config


def _exported(dest: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load((dest / "config.yaml").read_text(encoding="utf-8"))
    return loaded


def _paths(raw: dict[str, Any]) -> list[str]:
    return list(cli_config.source_paths(raw, raw["agents"][0].get("skills"), "skills") or [])


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
    monkeypatch.setattr(cli_config, "AUTODISCOVERY_DIRS", (Path(".domyn"), Path(".agents")))


@pytest.fixture
def machine(tmp_path: Path) -> dict[str, Path]:
    """A working directory, setup's install directory and a home, all empty."""
    places = {name: tmp_path / name for name in ("cwd", "install", "home")}
    for place in places.values():
        place.mkdir()
    return places


def _export(raw: dict[str, Any], machine: dict[str, Path], dest: Path) -> Any:
    return export_configuration(raw, dest, cwd=machine["cwd"])


class TestTheExportLoadsTheSameSkills:
    def test_a_group_folder_is_walked_at_any_depth(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        group = tmp_path / "mine"
        _skill(group / "writing" / "essay", "essay")
        _skill(group / "deep" / "er" / "notes", "notes")
        raw = _config(str(group))

        report = _export(raw, machine, tmp_path / "out")

        assert _paths(_exported(report.dest)) == [P + "resources/mine"]
        assert (report.dest / "resources/mine/deep/er/notes/SKILL.md").is_file()
        assert _loaded_skills(
            _exported(report.dest), machine["cwd"], report.dest
        ) == _loaded_skills(raw, machine["cwd"])

    def test_external_files_inside_the_listed_folder_keep_their_place(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        shared = tmp_path / "lib" / "shared.py"
        shared.parent.mkdir()
        shared.write_text("x = 1\n", encoding="utf-8")
        _skill(tmp_path / "lib" / "cat" / "tool", "tool", external=["../../shared.py"])
        raw = _config(str(tmp_path / "lib"))

        report = _export(raw, machine, tmp_path / "out")

        assert _paths(_exported(report.dest)) == [P + "resources/lib"]
        assert (report.dest / "resources/lib/shared.py").read_text() == "x = 1\n"
        loaded = _loaded_skills(_exported(report.dest), machine["cwd"], report.dest)
        assert loaded == {"tool": (True, None)} == _loaded_skills(raw, machine["cwd"])

    def test_external_files_outside_the_listed_folder_refuse_the_export(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        """A file a skill needs from above the folder it is read through: there is no
        place in the export that keeps the skill working, so nothing is written."""
        shared = tmp_path / "lib" / "shared.py"
        shared.parent.mkdir()
        shared.write_text("x = 1\n", encoding="utf-8")
        skill = _skill(tmp_path / "lib" / "cat" / "tool", "tool", external=["../../shared.py"])

        with pytest.raises(ExportError, match=r"shared\.py.*outside .*tool"):
            _export(_config(str(skill)), machine, tmp_path / "out")
        assert not (tmp_path / "out").exists()

    def test_a_relative_path_is_read_against_the_working_directory(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        _skill(machine["cwd"] / "team" / "review", "review")

        report = _export(_config("team"), machine, tmp_path / "out")

        assert _paths(_exported(report.dest)) == [P + "resources/team"]
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
        assert _paths(_exported(report.dest)) == [P + "resources/pack/a", P + "resources/pack"]
        assert report.files_copied == 2

    def test_two_folders_with_one_name_get_two_places(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        _skill(tmp_path / "x" / "pack" / "one", "one")
        _skill(tmp_path / "y" / "pack" / "two", "two")
        raw = _config(str(tmp_path / "x" / "pack"), str(tmp_path / "y" / "pack"))

        report = _export(raw, machine, tmp_path / "out")

        assert _paths(_exported(report.dest)) == [P + "resources/pack", P + "resources/pack-2"]
        assert set(_loaded_skills(_exported(report.dest), machine["cwd"], report.dest)) == {
            "one",
            "two",
        }

    def test_a_folder_called_skills_is_named_after_its_parent(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        _skill(tmp_path / "project" / "skills" / "one", "one")

        report = _export(_config(str(tmp_path / "project" / "skills")), machine, tmp_path / "out")

        assert _paths(_exported(report.dest)) == [P + "resources/project"]


class TestSetupsSkills:
    """The built-ins are read through their install folder, a resource folder like any."""

    def test_only_what_the_skills_declare_is_copied(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        install = machine["install"]
        _skill(install / "data" / "analysis", "analysis", external=["../../plotly_style.py"])
        (install / "plotly_style.py").write_text("style\n", encoding="utf-8")
        (install / "stray.py").write_text("x\n", encoding="utf-8")

        report = _export(_config(str(install)), machine, tmp_path / "out")

        assert _paths(_exported(report.dest)) == [P + "resources/install"]
        exported = sorted(
            p.relative_to(report.dest).as_posix() for p in report.dest.rglob("*") if p.is_file()
        )
        assert exported == [
            ".info",
            "config.yaml",
            "resources/install/data/analysis/SKILL.md",
            "resources/install/plotly_style.py",
        ]
        assert _loaded_skills(_exported(report.dest), machine["cwd"], report.dest) == {
            "analysis": (True, None)
        }

    def test_a_built_in_listed_by_its_own_folder_cannot_take_what_it_shares(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        """How setup used to list them: the shared file is outside the folder it is
        read through, so the export would break the skill and refuses instead."""
        install = machine["install"]
        skill = _skill(install / "data" / "analysis", "analysis", external=["../../style.py"])
        (install / "style.py").write_text("style\n", encoding="utf-8")

        with pytest.raises(ExportError, match=r"style\.py"):
            _export(_config(str(skill)), machine, tmp_path / "out")
        assert not (tmp_path / "out").exists()


def _setup_config(skills: list[str], instructions: list[str] | None = None) -> dict[str, Any]:
    """What `domyn setup` writes for these paths: `.domyn/skills`, `.domyn`… become the
    autodiscovery resource."""

    def local(paths: list[str] | None) -> dict[str, Any] | None:
        return {"source": {"type": "local_dir", "paths": paths}} if paths else None

    return cli_config.configuration_v1_dict_from(
        cli_config.ProviderConfig(type="vllm", model_name="m", url="http://x"),
        cli_config.AgentConfig(skills=local(skills), standing_instructions=local(instructions)),
        cli_config.CompactionConfig(),
    )


def _entries(raw: dict[str, Any], key: str) -> Any:
    return raw["agents"][0][key]["source"]["resource"]


class TestAutodiscovery:
    """The autodiscovery resource keeps reading the folder the project runs in, and what
    it found here is copied and added to the user's own resources, as `main` did."""

    def test_the_live_folders_stay_and_what_they_held_joins_the_users_own(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        cwd = machine["cwd"]
        _skill(cwd / ".domyn" / "skills" / "writing" / "essay", "essay")
        _skill(cwd / ".agents" / "skills" / "review", "review")
        (cwd / ".domyn" / "SOUL.md").write_text("soul\n", encoding="utf-8")
        (cwd / ".domyn" / "deeper").mkdir()
        (cwd / ".domyn" / "deeper" / "AGENTS.md").write_text("deeper\n", encoding="utf-8")
        raw = _setup_config([".domyn/skills", ".agents/skills"], [".domyn", ".agents"])

        report = _export(raw, machine, tmp_path / "out")

        exported = _exported(report.dest)
        resources = {r["id"]: r["paths"] for r in exported["resources"]}
        assert resources == {
            "autodiscovery": [".domyn", ".agents"],
            # The copied skills folders, read whole like any skills folder of the user's.
            "skills": [P + "resources/domyn/skills", P + "resources/agents/skills"],
            # The instruction files the walk found, listed one by one: a copy is frozen.
            "instructions": [P + "resources/domyn/SOUL.md", P + "resources/domyn/deeper/AGENTS.md"],
        }
        # Skills: the folder a session runs in first, so it wins a name.
        assert _entries(exported, "skills") == [{"id": "autodiscovery", "path": "skills"}, "skills"]
        # Instructions: the copies first, so the live brief is still read last.
        assert _entries(exported, "standing_instructions") == [
            "instructions",
            {"id": "autodiscovery", "walk": True},
        ]
        copied = sorted(
            p.relative_to(report.dest).as_posix()
            for p in (report.dest / "resources").rglob("*")
            if p.is_file()
        )
        assert copied == [
            "resources/agents/skills/review/SKILL.md",
            "resources/domyn/SOUL.md",
            "resources/domyn/deeper/AGENTS.md",
            "resources/domyn/skills/writing/essay/SKILL.md",
        ]

    def test_run_elsewhere_it_reads_the_copies_and_that_folder(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        from domyn.agents.loader import load_from_file

        _skill(machine["cwd"] / ".domyn" / "skills" / "essay", "essay")
        report = _export(_setup_config([".domyn/skills"]), machine, tmp_path / "out")
        elsewhere = tmp_path / "elsewhere"
        _skill(elsewhere / ".domyn" / "skills" / "review", "review")

        before = Path.cwd()
        os.chdir(elsewhere)
        os.environ["DOMYN_PROJECT_DIR"] = str(report.dest)
        try:
            from domyn.agents.loader import ConfigurationV1

            config = ConfigurationV1.from_dict(load_from_file(str(report.dest / "config.yaml")))
            agent = config.agents[0]
            loaded = asyncio.run(agent.skills.index(SkillQuery(task="x")))  # type: ignore[union-attr]
        finally:
            os.chdir(before)
            os.environ.pop("DOMYN_PROJECT_DIR", None)
        assert sorted(s.name for s in loaded.skills) == ["essay", "review"]

    def test_a_name_in_both_folders_still_loads_from_the_earlier(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        ours = _skill(machine["cwd"] / ".domyn" / "skills" / "report", "report")
        with (ours / "SKILL.md").open("a", encoding="utf-8") as f:
            f.write("domyn's\n")
        _skill(machine["cwd"] / ".agents" / "skills" / "report", "report")

        report = _export(
            _setup_config([".domyn/skills", ".agents/skills"]), machine, tmp_path / "out"
        )

        source = LocalDirSkillSource(
            paths=[
                report.dest / "resources/domyn/skills",
                report.dest / "resources/agents/skills",
            ]
        )
        body = asyncio.run(source.load(SkillQuery(task="x"))).body("report")
        assert body is not None and "domyn's" in body.body

    def test_missing_folders_add_no_snapshot(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        report = _export(
            _setup_config([".domyn/skills", ".agents/skills"]), machine, tmp_path / "out"
        )

        exported = _exported(report.dest)
        assert [r["id"] for r in exported["resources"]] == ["autodiscovery"]
        assert _entries(exported, "skills") == {"id": "autodiscovery", "path": "skills"}
        assert report.dropped == []

    def test_a_file_shared_inside_the_folder_keeps_its_place(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        category = machine["cwd"] / ".domyn" / "skills" / "writing"
        _skill(category / "essay", "essay", external=["../style.md"])
        (category / "style.md").write_text("style\n", encoding="utf-8")

        report = _export(_setup_config([".domyn/skills"]), machine, tmp_path / "out")

        assert (report.dest / "resources/domyn/skills/writing/style.md").read_text() == "style\n"
        assert report.warnings == []

    def test_a_file_shared_from_above_the_folder_refuses_the_export(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        (machine["cwd"] / "shared.py").write_text("x = 1\n", encoding="utf-8")
        _skill(machine["cwd"] / ".agents" / "skills" / "x", "x", external=["../../../shared.py"])

        with pytest.raises(ExportError, match=r"shared\.py.*outside"):
            _export(_setup_config([".agents/skills"]), machine, tmp_path / "out")
        assert not (tmp_path / "out").exists()

    def test_an_export_of_an_export_adds_to_what_the_first_copied(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        _skill(machine["cwd"] / ".domyn" / "skills" / "essay", "essay")
        first = _export(_setup_config([".domyn/skills"]), machine, tmp_path / "first")
        again = machine["cwd"].parent / "again"
        _skill(again / ".domyn" / "skills" / "review", "review")
        raw = cli_config.expand_project_dir(_exported(first.dest), first.dest)

        report = export_configuration(raw, tmp_path / "second", cwd=again)

        exported = _exported(report.dest)
        skills = next(r for r in exported["resources"] if r["id"] == "skills")
        # This folder's copy first, then the first export's, now the project's own.
        assert len(skills["paths"]) == 2
        assert _entries(exported, "skills") == [{"id": "autodiscovery", "path": "skills"}, "skills"]


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

        assert sorted(p.name for p in (report.dest / "resources/s").iterdir()) == ["SKILL.md"]

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

        report = _export(_config(str(skill.parent)), machine, tmp_path / "out")

        assert "shared.py" in [p.name for p in report.dest.rglob("*") if p.is_file()]

    def test_an_undeclared_file_is_left_behind_and_named(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        skill = _skill(tmp_path / "s", "s", external=["scripts/"])
        (skill / "scripts").mkdir()
        (skill / "scripts" / "run.py").write_text("x\n", encoding="utf-8")
        (skill / "notes.txt").write_text("private\n", encoding="utf-8")

        report = _export(_config(str(skill)), machine, tmp_path / "out")

        assert (report.dest / "resources/s/scripts/run.py").is_file()
        assert not (report.dest / "resources/s/notes.txt").exists()
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

        assert sorted(p.name for p in (report.dest / "resources/ascii-art").iterdir()) == [
            "SKILL.md"
        ]
        assert any("ascii_art.py" in w and "external_files" in w for w in report.warnings)

    def test_the_same_skill_with_external_files_brings_it(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        skill = _skill(tmp_path / "ascii-art", "ascii-art", external=["ascii_art.py"])
        (skill / "ascii_art.py").write_text("x\n", encoding="utf-8")

        report = _export(_config(str(skill)), machine, tmp_path / "out")

        assert (report.dest / "resources/ascii-art/ascii_art.py").read_text() == "x\n"
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

        assert not (report.dest / "resources/s/helper.py").exists()
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


class TestTheSessionsWorkingStateStaysBehind:
    """``conversations/``, ``processes/`` and ``browser/`` of a domyn home are the
    session's (`home_dir`); a folder of the same name elsewhere is anyone's."""

    @pytest.mark.parametrize("folder", list(StateFolders))
    def test_a_skill_reaching_into_the_home_copies_nothing_from_it(
        self, machine: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, folder: str
    ) -> None:
        home = machine["home"] / ".domyn"
        monkeypatch.setattr(cli_config, "GLOBAL_DIR", home)
        (home / folder / "c1").mkdir(parents=True)
        (home / folder / "c1" / "out.log").write_text("secret\n", encoding="utf-8")
        # A skill in the home whose `external_files` walks up into the state folder.
        _skill(home / "mine" / "s", "s", external=[f"../../{folder}/c1/out.log"])

        report = _export(_config(str(home / "mine" / "s")), machine, tmp_path / "out")

        assert not list(report.dest.rglob("out.log"))
        assert any("out.log" in d and "never exported" in d for d in report.dropped)

    def test_in_a_project_too(
        self, machine: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = machine["home"] / ".domyn"
        monkeypatch.setattr(cli_config, "GLOBAL_DIR", home)
        project = home / "projects" / "alpha"
        (project / "processes" / "c1").mkdir(parents=True)
        (project / "processes" / "c1" / "out.log").write_text("secret\n", encoding="utf-8")
        _skill(project / "skills" / "s", "s", external=["../../processes/c1/out.log"])

        report = _export(_config(str(project / "skills" / "s")), machine, tmp_path / "out")

        assert not list(report.dest.rglob("out.log"))

    @pytest.mark.parametrize("folder", list(StateFolders))
    def test_a_skills_own_folder_of_that_name_travels(
        self, machine: dict[str, Path], tmp_path: Path, folder: str
    ) -> None:
        skill = _skill(tmp_path / "s", "s", external=[f"{folder}/"])
        (skill / folder).mkdir()
        (skill / folder / "notes.md").write_text("mine\n", encoding="utf-8")

        report = _export(_config(str(skill)), machine, tmp_path / "out")

        assert (report.dest / "skills/s" / folder / "notes.md").read_text() == "mine\n"


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
        assert not (report.dest / "resources").exists()

    @staticmethod
    def _raw(*declared: str) -> dict[str, Any]:
        return _config(standing_instructions={"source": {"type": "local_dir", "paths": declared}})

    @staticmethod
    def _declared(dest: Path) -> list[str]:
        raw = _exported(dest)
        declared = raw["agents"][0].get("standing_instructions")
        return list(cli_config.source_paths(raw, declared, "standing_instructions") or [])

    def test_each_declared_path_is_copied_by_position(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        brief = tmp_path / "notes" / "project"
        brief.mkdir(parents=True)
        (brief / "AGENTS.md").write_text("brief\n", encoding="utf-8")
        (brief / "unrelated.txt").write_text("not read\n", encoding="utf-8")
        soul = tmp_path / "me" / "SOUL.md"
        soul.parent.mkdir()
        soul.write_text("soul\n", encoding="utf-8")

        report = _export(self._raw(str(soul), str(brief)), machine, tmp_path / "out")

        assert self._declared(report.dest) == [P + "resources/me/SOUL.md", P + "resources/project"]
        assert (report.dest / "resources/project/AGENTS.md").read_text() == "brief\n"
        assert not (report.dest / "resources/project/unrelated.txt").exists()
        assert report.instruction_files == [
            "resources/me/SOUL.md",
            "resources/project/AGENTS.md",
        ]

    def test_two_files_of_one_folder_stay_in_one_folder(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        notes = tmp_path / "notes"
        notes.mkdir()
        (notes / "SOUL.md").write_text("soul\n", encoding="utf-8")
        (notes / "task.md").write_text("task\n", encoding="utf-8")

        report = _export(
            self._raw(str(notes / "SOUL.md"), str(notes / "task.md")), machine, tmp_path / "out"
        )

        assert self._declared(report.dest) == [
            P + "resources/notes/SOUL.md",
            P + "resources/notes/task.md",
        ]

    def test_two_folders_of_one_name_get_a_folder_each(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        for parent in ("x", "y"):
            (tmp_path / parent / "a").mkdir(parents=True)
            (tmp_path / parent / "a" / "AGENTS.md").write_text(f"{parent}\n", encoding="utf-8")

        report = _export(
            self._raw(str(tmp_path / "x" / "a"), str(tmp_path / "y" / "a")),
            machine,
            tmp_path / "out",
        )

        assert self._declared(report.dest) == [P + "resources/a", P + "resources/a-2"]
        assert (report.dest / "resources/a-2/AGENTS.md").read_text() == "y\n"

    def test_a_walked_folder_is_copied_with_its_subfolders(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        """Subfolders included, the skills folder like any other."""
        domyn = machine["cwd"] / ".domyn"
        (domyn / "deeper").mkdir(parents=True)
        (domyn / "SOUL.md").write_text("soul\n", encoding="utf-8")
        (domyn / "deeper" / "AGENTS.md").write_text("deeper\n", encoding="utf-8")
        (domyn / "skills").mkdir()
        (domyn / "skills" / "AGENTS.md").write_text("skills\n", encoding="utf-8")

        raw = _config()
        raw["resources"] = [{"id": "project", "type": "local_dir", "paths": [".domyn"]}]
        raw["agents"][0]["standing_instructions"] = {
            "source": {"type": "resource", "resource": [{"id": "project", "walk": True}]}
        }

        report = _export(raw, machine, tmp_path / "out")

        assert self._declared(report.dest) == [P + "resources/domyn"]
        # Still walked in the copy: the entry says so, not the folder's name.
        assert _exported(report.dest)["agents"][0]["standing_instructions"]["source"] == {
            "type": "resource",
            "resource": [{"id": "project", "walk": True}],
        }
        assert sorted(
            p.relative_to(report.dest).as_posix() for p in (report.dest / "resources").rglob("*.md")
        ) == [
            "resources/domyn/SOUL.md",
            "resources/domyn/deeper/AGENTS.md",
            "resources/domyn/skills/AGENTS.md",
        ]

    def test_a_local_folder_with_neither_file_is_left_out_quietly(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        report = _export(self._raw(".domyn", ".agents"), machine, tmp_path / "out")

        assert self._declared(report.dest) == []
        assert report.dropped == []

    def test_a_listed_folder_with_neither_file_is_named(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        (tmp_path / "empty").mkdir()

        report = _export(self._raw(str(tmp_path / "empty")), machine, tmp_path / "out")

        assert self._declared(report.dest) == []
        assert any("nothing is read from it" in note for note in report.dropped)


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
            (
                {"resources": [{"id": "workspace", "type": "local_dir", "paths": ["/srv/work"]}]},
                "/srv/work",
            ),
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
        # `lib` holds both the skill and `../../shared/style.md`: the skill finds it in the
        # export and in the project imported from it.
        from domyn.agents.interfaces.cli.project_import import import_project

        (tmp_path / "lib" / "shared").mkdir(parents=True)
        (tmp_path / "lib" / "shared" / "style.md").write_text("s\n", encoding="utf-8")
        _skill(tmp_path / "lib" / "skills" / "s", "s", external=["../../shared/style.md"])

        report = _export(_config(str(tmp_path / "lib")), machine, tmp_path / "out")
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

        cli_export._copy_into(
            staging, Path("skills/a.md"), source, {source}, cli_export.PrivatePlaces.now()
        )

        assert (staging / "skills/a.md").read_text(encoding="utf-8") == "x\n"

    @pytest.mark.parametrize("relative", ["../escape.md", "skills/../../escape.md", "/tmp/x.md"])
    def test_a_target_outside_the_folder_is_refused(self, tmp_path: Path, relative: str) -> None:
        source = self._file(tmp_path)
        staging = tmp_path / "staging"
        staging.mkdir()

        with pytest.raises(ExportError, match="outside the exported folder"):
            cli_export._copy_into(
                staging, Path(relative), source, {source}, cli_export.PrivatePlaces.now()
            )
        assert not (tmp_path / "escape.md").exists()

    def test_a_source_the_plan_did_not_choose_is_refused(self, tmp_path: Path) -> None:
        source = self._file(tmp_path)
        staging = tmp_path / "staging"
        staging.mkdir()

        with pytest.raises(ExportError, match="not chosen"):
            cli_export._copy_into(
                staging, Path("a.md"), source, set(), cli_export.PrivatePlaces.now()
            )

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
            cli_export._copy_into(
                staging, Path("x"), source, {source}, cli_export.PrivatePlaces.now()
            )


class TestWhereTheCopiesGo:
    """The copies join the user's own resource only when nothing else reads it."""

    def test_a_resource_another_agent_reads_is_left_as_it_is(
        self, machine: dict[str, Path], tmp_path: Path
    ) -> None:
        _skill(machine["cwd"] / ".domyn" / "skills" / "essay", "essay")
        _skill(tmp_path / "mine" / "notes", "notes")
        raw = _setup_config([".domyn/skills", str(tmp_path / "mine" / "notes")])
        raw["agents"].append(
            {
                "name": "Helper",
                "llm_provider": raw["agents"][0]["llm_provider"],
                "skills": {"source": {"type": "resource", "resource": "skills", "path": "."}},
            }
        )

        report = _export(raw, machine, tmp_path / "out")

        resources = {r["id"]: r["paths"] for r in _exported(report.dest)["resources"]}
        assert resources["skills"] == [P + "resources/notes"]
        assert resources["skills-2"] == [P + "resources/domyn/skills"]
        assert _entries(_exported(report.dest), "skills") == [
            {"id": "autodiscovery", "path": "skills"},
            "skills-2",
            "skills",
        ]
