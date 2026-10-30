"""Tests for ``domyn skill show|remove``: a skill looked up among those the config loads.

The config is the global one unless ``--project`` selects another. ``remove`` unlists a
skill from that config and never deletes its files; a skill the config does not list
on its own (``.domyn/skills`` autodiscovery, a listed folder of
skills) is left alone with a warning.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import Result
from typer.testing import CliRunner

from domyn.agents.cli.cli import app
from domyn.agents.cli.commands.skills import command as skills_command
from domyn.agents.interfaces.cli import config as cfg
from domyn.agents.skills import SkillsRuntime


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The global config directory, in a folder of its own; the test runs in another."""
    home = tmp_path / "home" / ".domyn"
    home.mkdir(parents=True)
    monkeypatch.setattr(cfg, "GLOBAL_DIR", home)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return home


def _skill(root: Path, name: str) -> Path:
    """A ``<name>/SKILL.md`` skill under ``root``; returns its folder."""
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: what {name} does\n---\n\nThe body of {name}.\n",
        encoding="utf-8",
    )
    return folder


def _config(home: Path, paths: list[str]) -> Path:
    config = cfg.configuration_v1_dict_from(
        cfg.ProviderConfig(type="openai", model_name="m"),
        cfg.AgentConfig(
            name="A",
            instruction="i",
            skills={"source": {"type": "local_dir", "paths": paths}},
        ),
        cfg.CompactionConfig(),
    )
    path = home / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _paths(config: Path) -> list[str]:
    raw: dict[str, Any] = yaml.safe_load(config.read_text(encoding="utf-8"))
    return list(raw["agents"][0]["skills"]["source"]["paths"])


def _invoke(*args: str) -> Result:
    # Wide enough that no message is folded across lines.
    return CliRunner().invoke(app, ["skill", *args], env={"COLUMNS": "500"})


class TestShow:
    def test_it_prints_a_configured_skill(self, home: Path, tmp_path: Path) -> None:
        _config(home, [str(_skill(tmp_path / "mine", "essay"))])

        result = _invoke("show", "essay")

        assert result.exit_code == 0, result.output
        assert "The body of essay." in result.output

    def test_an_unknown_name_fails_and_suggests_a_close_one(
        self, home: Path, tmp_path: Path
    ) -> None:
        _config(home, [str(_skill(tmp_path / "mine", "essay"))])

        result = _invoke("show", "esay")

        assert result.exit_code == 1
        assert "Did you mean essay?" in result.output

    def test_name_hash_section_prints_only_that_section(self, home: Path, tmp_path: Path) -> None:
        folder = _skill(tmp_path / "mine", "essay")
        (folder / "SKILL.md").write_text(
            "---\nname: essay\ndescription: d\n---\n\n"
            "## Structure\n\nIntro, body, end.\n\n## Checklist\n\nSpelling.\n",
            encoding="utf-8",
        )
        _config(home, [str(folder)])

        result = _invoke("show", "essay#structure")

        assert result.exit_code == 0, result.output
        assert "Intro, body, end." in result.output
        assert "Spelling." not in result.output

    def test_an_unknown_section_fails_and_lists_the_sections(
        self, home: Path, tmp_path: Path
    ) -> None:
        folder = _skill(tmp_path / "mine", "essay")
        (folder / "SKILL.md").write_text(
            "---\nname: essay\ndescription: d\n---\n\n## Structure\n\nIntro.\n",
            encoding="utf-8",
        )
        _config(home, [str(folder)])

        result = _invoke("show", "essay#nothing")

        assert result.exit_code == 1
        assert "No section 'nothing' in essay. Sections: Structure" in result.output

    def test_an_unreadable_file_fails(
        self, home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _config(home, [str(_skill(tmp_path / "mine", "essay"))])

        async def unreadable(*args: object, **kwargs: object) -> None:
            raise PermissionError

        monkeypatch.setattr(SkillsRuntime, "fetch", unreadable)

        result = _invoke("show", "essay")

        assert result.exit_code == 1
        assert "Cannot read skill 'essay': PermissionError" in result.output

    def test_with_no_config_it_says_none_are_configured(self, home: Path) -> None:
        result = _invoke("show", "essay")

        assert result.exit_code == 1
        assert "No skills configured" in result.output

    def test_it_reads_the_global_config_even_where_a_project_one_exists(
        self, home: Path, tmp_path: Path
    ) -> None:
        _config(home, [str(_skill(tmp_path / "global", "essay"))])
        project = Path(".domyn")
        project.mkdir()
        _config(project, [str(_skill(tmp_path / "project", "other"))])

        result = _invoke("show", "essay")

        assert result.exit_code == 0, result.output
        assert "The body of essay." in result.output


class TestProject:
    def test_show_reads_the_projects_config(self, home: Path, tmp_path: Path) -> None:
        _config(home, [])
        project = home / "projects" / "p"
        project.mkdir(parents=True)
        # Inside the project: one naming files outside itself is asked about first.
        _config(project, [str(_skill(project / "mine", "essay"))])

        result = _invoke("show", "essay", "--project", "p")

        assert result.exit_code == 0, result.output
        assert "The body of essay." in result.output

    def test_remove_edits_the_projects_config_only(self, home: Path, tmp_path: Path) -> None:
        skill = str(_skill(home / "projects" / "p" / "mine", "essay"))
        global_config = _config(home, [skill])
        before = global_config.read_text(encoding="utf-8")
        project = home / "projects" / "p"
        project_config = _config(project, [skill])

        result = _invoke("remove", "essay", "--project", "p")

        assert result.exit_code == 0, result.output
        assert skill not in project_config.read_text(encoding="utf-8")
        assert global_config.read_text(encoding="utf-8") == before

    def test_an_unknown_project_is_refused(self, home: Path, tmp_path: Path) -> None:
        _config(home, [str(_skill(tmp_path / "mine", "essay"))])

        result = _invoke("show", "essay", "--project", "nope")

        assert result.exit_code != 0


class TestRemove:
    def test_it_unlists_a_listed_skill_folder_and_keeps_its_files(
        self, home: Path, tmp_path: Path
    ) -> None:
        essay = _skill(tmp_path / "mine", "essay")
        other = _skill(tmp_path / "mine", "other")
        config = _config(home, [str(essay), str(other)])

        result = _invoke("remove", "essay")

        assert result.exit_code == 0, result.output
        assert "Removed 'essay'" in result.output
        assert _paths(config) == [str(other)]
        assert (essay / "SKILL.md").is_file()
        # The rest of the config is untouched.
        assert yaml.safe_load(config.read_text())["agents"][0]["name"] == "A"

    def test_it_unlists_a_loose_skill_file(self, home: Path, tmp_path: Path) -> None:
        loose = tmp_path / "mine" / "notes.md"
        loose.parent.mkdir(parents=True)
        loose.write_text("---\nname: notes\ndescription: d\n---\nbody\n", encoding="utf-8")
        config = _config(home, [str(loose)])

        result = _invoke("remove", "notes")

        assert result.exit_code == 0, result.output
        assert _paths(config) == []
        assert loose.is_file()

    def test_a_name_in_another_case_is_found(self, home: Path, tmp_path: Path) -> None:
        config = _config(home, [str(_skill(tmp_path / "mine", "Essay"))])

        result = _invoke("remove", "essay")

        assert result.exit_code == 0, result.output
        assert _paths(config) == []

    def test_an_unknown_name_changes_nothing(self, home: Path, tmp_path: Path) -> None:
        config = _config(home, [str(_skill(tmp_path / "mine", "essay"))])
        before = config.read_text(encoding="utf-8")

        result = _invoke("remove", "nothing")

        assert result.exit_code == 1
        assert "not among the skills" in result.output
        assert config.read_text(encoding="utf-8") == before

    def test_a_skill_from_autodiscovery_is_left_with_a_warning(
        self, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            cfg, "LOCAL_SKILLS_DIRS", (Path(".domyn/skills"), Path(".agents/skills"))
        )
        skill = _skill(Path(".agents/skills"), "local")
        config = _config(home, [".domyn/skills", ".agents/skills"])
        before = config.read_text(encoding="utf-8")

        result = _invoke("remove", "local")

        assert result.exit_code == 1
        assert "./.agents/skills autodiscovery" in " ".join(result.output.split())
        assert "domyn setup" in result.output
        assert config.read_text(encoding="utf-8") == before
        assert (skill / "SKILL.md").is_file()

    def test_a_skill_in_a_listed_folder_is_left_with_a_warning(
        self, home: Path, tmp_path: Path
    ) -> None:
        folder = tmp_path / "mine"
        _skill(folder, "essay")
        _skill(folder, "other")
        config = _config(home, [str(folder)])
        before = config.read_text(encoding="utf-8")

        result = _invoke("remove", "essay")

        assert result.exit_code == 1
        assert f"through the folder {folder}" in result.output
        assert "every skill in it" in result.output
        assert config.read_text(encoding="utf-8") == before

    def test_a_skill_listed_and_in_a_listed_folder_is_unlisted_and_still_warned(
        self, home: Path, tmp_path: Path
    ) -> None:
        folder = tmp_path / "mine"
        essay = _skill(folder, "essay")
        config = _config(home, [str(essay), str(folder)])

        result = _invoke("remove", "essay")

        assert result.exit_code == 0, result.output
        assert _paths(config) == [str(folder)]
        assert "through the folder" in result.output

    def test_a_skill_it_shadowed_is_named(self, home: Path, tmp_path: Path) -> None:
        first = _skill(tmp_path / "first", "essay")
        second = _skill(tmp_path / "second", "essay")
        config = _config(home, [str(first), str(second)])

        result = _invoke("remove", "essay")

        assert result.exit_code == 0, result.output
        assert _paths(config) == [str(second)]
        assert "still loads" in result.output and str(second) in result.output

    def test_an_installed_builtin_says_its_files_stay_until_setup(
        self, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install = home / "skills"
        monkeypatch.setattr(skills_command, "skills_install_dir", lambda: install)
        charts = _skill(install / "visualization", "charts")
        config = _config(home, [str(charts)])

        result = _invoke("remove", "charts")

        assert result.exit_code == 0, result.output
        assert _paths(config) == []
        assert "until the next `domyn setup`" in result.output


def test_an_unknown_action_is_refused(home: Path) -> None:
    result = _invoke("new", "essay")

    assert result.exit_code == 1
    assert "Use show or remove" in result.output
