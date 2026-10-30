"""``domyn setup --project <name> --import <dir>``: a new project started from an export."""

import importlib.metadata
import io
import platform
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml
from rich.console import Console
from typer.testing import CliRunner

from domyn.agents.cli.cli import app
from domyn.agents.interfaces.cli import Theme
from domyn.agents.interfaces.cli import config as cfg
from domyn.agents.interfaces.cli import export as export_mod
from domyn.agents.interfaces.cli import project_import as import_mod
from domyn.agents.interfaces.cli import select as select_mod
from domyn.agents.interfaces.cli import setup as cli_setup
from domyn.agents.interfaces.cli.export import export_configuration
from domyn.agents.interfaces.cli.project_import import ProjectImportError, import_project
from domyn.agents.interfaces.cli.prompt import GoBack
from domyn.agents.skills.catalog import install_skills, list_builtin_skills

P = cfg.PROJECT_DIR_REF
BUILTINS = cfg.BUILTIN_SKILLS_DIRNAME
# Captured at import, before the CLI conftest replaces it for every test.
_REAL_CHOOSE_INSTRUCTIONS = cli_setup._choose_instructions
BUILTIN = list_builtin_skills()[0]


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(cfg, "GLOBAL_DIR", tmp_path / "home" / ".domyn")
    monkeypatch.setattr(cfg, "PROJECT_DIR", tmp_path / "nowhere" / ".domyn")
    # Where setup really installs: the selected project's own `builtin-skills/`, which
    # the CLI conftest points elsewhere for every test.
    monkeypatch.setattr(cli_setup, "skills_install_dir", lambda: cfg.home_dir() / BUILTINS)
    return cfg.GLOBAL_DIR


@pytest.fixture
def answers(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Scripted answers, through the numbered pickers; once used up, every answer is
    empty, which keeps what each question shows."""
    script: list[str] = []
    monkeypatch.setattr(select_mod, "_interactive", lambda theme: False)
    monkeypatch.setattr("builtins.input", lambda *_: script.pop(0) if script else "")
    return script


def _skill(folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name} skill\n---\n\nbody\n", encoding="utf-8"
    )
    return folder


def _config(*skill_paths: str, instructions: list[str] | None = None) -> dict[str, Any]:
    agent: dict[str, Any] = {
        "name": "Writer",
        "description": "d",
        "instruction": "i",
        "llm_provider": "p",
        "skills": {"source": {"type": "local_dir", "paths": list(skill_paths)}},
    }
    if instructions is not None:
        agent["standing_instructions"] = {"source": {"type": "local_dir", "paths": instructions}}
    return {
        "version": "1.0",
        "entrypoint": "Writer",
        "llm_providers": [
            {"name": "p", "type": "vllm", "model_name": "imported-model", "url": "http://x"}
        ],
        "agents": [agent],
    }


def _skill_paths(raw: dict[str, Any] | None) -> list[str]:
    assert raw is not None
    paths: list[str] = raw["agents"][0]["skills"]["source"]["paths"]
    return paths


def _instruction_paths(raw: dict[str, Any] | None) -> list[str]:
    if raw is None:
        return []
    paths: list[str] = raw["source"]["paths"]
    return paths


def _console() -> Console:
    return Console(file=io.StringIO(), width=200)


THEME = Theme.build(allow_animation=False)


@pytest.fixture
def exported(tmp_path: Path) -> Path:
    """An export of a configuration with one built-in skill, a folder of two skills of
    the user's own, an instructions folder and a SOUL.md, both declared."""
    install_dir = tmp_path / "origin" / BUILTINS
    (builtin_path,) = install_skills([BUILTIN.name], install_dir)
    _skill(tmp_path / "mine" / "essay", "essay")
    _skill(tmp_path / "mine" / "poem", "poem")
    instructions = tmp_path / "brief"
    instructions.mkdir()
    (instructions / "AGENTS.md").write_text("write essays\n", encoding="utf-8")
    soul = tmp_path / "me" / "SOUL.md"
    soul.parent.mkdir()
    soul.write_text("be terse\n", encoding="utf-8")

    dest = tmp_path / "shared"
    export_configuration(
        _config(
            str(builtin_path), str(tmp_path / "mine"), instructions=[str(soul), str(instructions)]
        ),
        dest,
        cwd=tmp_path,
        install_dir=install_dir,
    )
    return dest


class TestImportingAnExport:
    def test_every_skill_keeps_its_place(self, exported: Path, tmp_path: Path) -> None:
        project = tmp_path / "new"

        imported = import_project(exported, project)

        builtin_path, mine_path = _skill_paths(imported.config)
        # The built-in in the folder the project's setup installs to, as exported:
        # nothing is recognised or installed again.
        assert builtin_path.startswith(f"{P}/{BUILTINS}/")
        builtin = project / builtin_path.removeprefix(f"{P}/")
        assert builtin.name == BUILTIN.name and (builtin / "SKILL.md").is_file()
        assert mine_path.startswith(f"{P}/skills/")
        assert (project / mine_path.removeprefix(f"{P}/") / "essay" / "SKILL.md").is_file()
        assert imported.copied == [BUILTINS, "instructions", "skills"]

    def test_the_instructions_keep_their_place(self, exported: Path, tmp_path: Path) -> None:
        project = tmp_path / "new"

        imported = import_project(exported, project)

        declared = imported.config["agents"][0]["standing_instructions"]["source"]["paths"]
        assert declared and all(p.startswith(f"{P}/") for p in declared)
        assert all((project / p.removeprefix(f"{P}/")).exists() for p in declared)
        souls = [p for p in declared if p.endswith("/SOUL.md")]
        assert [(project / p.removeprefix(f"{P}/")).read_text() for p in souls] == ["be terse\n"]
        # The config is the wizard's to write.
        assert not (project / cfg.CONFIG_FILENAME).exists()

    def test_private_and_hidden_files_are_left_behind(self, exported: Path, tmp_path: Path) -> None:
        (exported / cfg.CREDENTIALS_FILENAME).write_text("KEY=secret\n")
        (exported / ".env").write_text("KEY=secret\n")
        project = tmp_path / "new"

        import_project(exported, project)

        assert not (project / cfg.CREDENTIALS_FILENAME).exists()
        assert not (project / ".env").exists()


class TestRefused:
    @pytest.mark.parametrize(
        "config",
        [
            _config("/elsewhere/x"),
            _config(".x/s"),
            _config(instructions=["~/.ssh/id_rsa"]),
            _config(instructions=["${DOMYN_PROJECT_DIR}/../../.ssh/id_rsa"]),
            _config(instructions=["${DOMYN_PROJECT_DIR}/${HOME}/.ssh/id_rsa"]),
            {**_config(), "tools": [{"name": "t", "type": "function", "path": "/tmp/t.py"}]},
            {**_config(), "tool_bundles": [{"name": "b", "workspace": "~"}]},
        ],
        ids=["absolute", "relative", "home", "climbing", "variable", "tool", "workspace"],
    )
    def test_an_export_naming_a_path_outside_itself(
        self, tmp_path: Path, config: dict[str, Any]
    ) -> None:
        # Hand-made, or edited after `domyn export`: it must not make this machine
        # read, run or work in a place its author chose.
        source = tmp_path / "hand-made"
        source.mkdir()
        (source / cfg.CONFIG_FILENAME).write_text(yaml.safe_dump(config))
        project = tmp_path / "new"

        with pytest.raises(ProjectImportError, match="outside itself"):
            import_project(source, project)
        assert not project.exists()

    def test_a_project_that_already_holds_files(self, exported: Path, tmp_path: Path) -> None:
        project = tmp_path / "new"
        project.mkdir()
        (project / "notes.txt").write_text("mine")

        with pytest.raises(ProjectImportError, match="not an empty folder"):
            import_project(exported, project)
        assert (project / "notes.txt").read_text() == "mine"

    def test_a_folder_with_no_config(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()

        with pytest.raises(ProjectImportError, match=r"config\.yaml"):
            import_project(tmp_path / "empty", tmp_path / "new")
        assert not (tmp_path / "new").exists()

    def test_the_export_itself(self, exported: Path) -> None:
        with pytest.raises(ProjectImportError, match="already a project"):
            import_project(exported, exported)


class TestTheExportInfo:
    def test_the_export_says_what_made_it_and_names_nothing_of_the_machine(
        self, exported: Path, tmp_path: Path
    ) -> None:
        text = (exported / export_mod.INFO_FILENAME).read_text()
        info = yaml.safe_load(text)

        assert info["export_format_version"] == export_mod.EXPORT_FORMAT_VERSION
        assert info["domyn_agents_version"] == export_mod.installed_version()
        assert datetime.fromisoformat(info["exported_at"]).utcoffset() == timedelta(0)
        assert info["python_version"] == platform.python_version()
        assert info["platform"].startswith(platform.system())
        assert str(Path.home()) not in text and str(tmp_path) not in text
        assert platform.node() not in text

    def test_the_platform_sdk_is_recorded_only_when_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def version(name: str) -> str:
            if name == "domyn-platform-sdk":
                raise importlib.metadata.PackageNotFoundError(name)
            return "1.0"

        monkeypatch.setattr(importlib.metadata, "version", version)
        assert "domyn_platform_sdk_version" not in export_mod.export_info()

        monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.40.0")
        assert export_mod.export_info()["domyn_platform_sdk_version"] == "0.40.0"

    def test_the_same_version_says_nothing_and_the_file_stays_behind(
        self, exported: Path, tmp_path: Path
    ) -> None:
        project = tmp_path / "new"

        imported = import_project(exported, project)

        assert imported.exported_with == export_mod.installed_version()
        assert imported.info["python_version"] == platform.python_version()
        assert imported.version_warning() is None
        assert not (project / export_mod.INFO_FILENAME).exists()

    def test_another_version_is_named_with_this_one_and_the_date(
        self, exported: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (exported / export_mod.INFO_FILENAME).write_text(
            "domyn_agents_version: 0.1.0\nexported_at: '2026-01-02T03:04:05+00:00'\n"
        )
        monkeypatch.setattr(import_mod, "installed_version", lambda: "9.9.9")

        warning = import_project(exported, tmp_path / "new").version_warning()

        assert warning is not None
        assert "0.1.0" in warning and "2026-01-02T03:04:05+00:00" in warning
        assert "this is 9.9.9" in warning

    @pytest.mark.parametrize(
        "info", [None, "", "not: [a, mapping\n", "- 1.0\n", "domyn_agents_version: [1]\n"]
    )
    def test_an_export_that_does_not_say_is_warned_about(
        self, exported: Path, tmp_path: Path, info: str | None
    ) -> None:
        path = exported / export_mod.INFO_FILENAME
        if info is None:
            path.unlink()
        else:
            path.write_text(info)

        imported = import_project(exported, tmp_path / "new")

        assert imported.exported_with is None
        warning = imported.version_warning()
        assert warning is not None and "does not say which version" in warning

    def test_the_command_warns_before_the_wizard_and_goes_on(
        self,
        exported: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        answers: list[str],
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cli_setup, "_keep", _keep_but())
        monkeypatch.setattr(import_mod, "installed_version", lambda: "9.9.9")

        result = CliRunner().invoke(app, ["setup", "--import", str(exported)])

        assert result.exit_code == 0, result.output
        said = " ".join(result.output.split())
        assert export_mod.installed_version() in said
        assert said.index("this is 9.9.9") < said.index("Saved")


class TestTheProjectSkillsPicker:
    @pytest.fixture
    def imported(
        self, exported: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[Path, dict[str, Any]]:
        project = tmp_path / "new"
        config = import_project(exported, project).config
        skills = cfg.expand_project_dir(config, project)["agents"][0]["skills"]
        # The project is selected while it is set up, so setup installs into its own.
        monkeypatch.setattr(cli_setup, "skills_install_dir", lambda: project / BUILTINS)
        return project, skills

    @pytest.fixture
    def picker(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        """What the picker was shown; ``keep`` is what the user leaves ticked."""
        shown: dict[str, Any] = {"keep": set()}

        def multiselect(
            console: Any,
            theme: Any,
            label: str,
            options: list[str],
            *,
            selected: Any = (),
            groups: Any = None,
            **_: Any,
        ) -> Any:
            shown.update(options=list(options), selected=list(selected), groups=dict(groups or {}))
            return [option for option in options if option in shown["keep"]]

        monkeypatch.setattr(cli_setup, "multiselect", multiselect)
        return shown

    def test_it_offers_the_project_s_skills_ticked_and_the_missing_built_ins_unticked(
        self, imported: tuple[Path, dict[str, Any]], picker: dict[str, Any]
    ) -> None:
        project, skills = imported
        picker["keep"] = {BUILTIN.name, "essay", "poem"}

        result = cli_setup._choose_project_skills(_console(), THEME, skills, project)

        others = [skill.name for skill in list_builtin_skills() if skill.name != BUILTIN.name]
        assert others, "the catalog has one skill only: nothing would be missing"
        assert picker["options"] == [BUILTIN.name, "essay", "poem", *others]
        assert picker["selected"] == [BUILTIN.name, "essay", "poem"]
        # All the project's ticked, no built-in added: the paths stay as written.
        assert result == skills
        assert sorted(p.name for p in (project / BUILTINS).rglob("SKILL.md")) == ["SKILL.md"]

    def test_a_missing_built_in_sits_in_its_category_with_the_exported_ones(
        self, imported: tuple[Path, dict[str, Any]], picker: dict[str, Any]
    ) -> None:
        project, skills = imported

        cli_setup._choose_project_skills(_console(), THEME, skills, project)

        group = f"{BUILTINS}/{BUILTIN.category}"
        same_category = [s.name for s in list_builtin_skills() if s.category == BUILTIN.category]
        assert picker["groups"][group] == same_category

    def test_a_missing_built_in_ticked_is_installed_beside_the_exported_ones(
        self, imported: tuple[Path, dict[str, Any]], picker: dict[str, Any]
    ) -> None:
        project, skills = imported
        builtin_path, folder = skills["source"]["paths"]
        exported_file = Path(builtin_path) / "SKILL.md"
        exported_file.write_text(exported_file.read_text() + "\nedited\n", encoding="utf-8")
        extra = next(skill for skill in list_builtin_skills() if skill.name != BUILTIN.name)
        picker["keep"] = {BUILTIN.name, "essay", "poem", extra.name}

        result = cli_setup._choose_project_skills(_console(), THEME, skills, project)

        assert result is not None
        added = project / BUILTINS / extra.relative_path
        assert result["source"]["paths"] == [builtin_path, str(added), folder]
        assert (added / "SKILL.md").is_file()
        # Added, not reinstalled: the exported built-in keeps its edit.
        assert exported_file.read_text().endswith("\nedited\n")

    def test_an_export_with_no_skills_still_offers_the_built_ins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, picker: dict[str, Any]
    ) -> None:
        project = tmp_path / "bare"
        project.mkdir()
        monkeypatch.setattr(cli_setup, "skills_install_dir", lambda: project / BUILTINS)
        picker["keep"] = {BUILTIN.name}

        result = cli_setup._choose_project_skills(_console(), THEME, None, project)

        assert picker["selected"] == []
        assert result == {
            "source": {
                "type": "local_dir",
                "paths": [str(project / BUILTINS / BUILTIN.relative_path)],
            }
        }

    def test_a_folder_with_a_skill_unticked_lists_the_rest_one_by_one(
        self, imported: tuple[Path, dict[str, Any]], picker: dict[str, Any]
    ) -> None:
        project, skills = imported
        builtin, folder = skills["source"]["paths"]
        picker["keep"] = {BUILTIN.name, "poem"}

        result = cli_setup._choose_project_skills(_console(), THEME, skills, project)

        assert result is not None
        assert result["source"]["paths"] == [builtin, f"{folder}/poem"]

    def test_nothing_ticked_lists_nothing(
        self, imported: tuple[Path, dict[str, Any]], picker: dict[str, Any]
    ) -> None:
        project, skills = imported

        result = cli_setup._choose_project_skills(_console(), THEME, skills, project)

        assert result is None


def _choose_and_save(current: Any, project: Path) -> Any:
    """The imported instructions step, then the deletion `run_setup` does once saved."""
    discard: list[Path] = []
    result = cli_setup._choose_imported_instructions(_console(), THEME, current, discard=discard)
    for path in discard:
        cli_setup._discard_imported(path, project)
    return result


class TestTheImportedInstructions:
    """The step runs only when the import's instructions are not kept (`_keep`)."""

    @pytest.fixture
    def project(self, tmp_path: Path) -> Path:
        project = tmp_path / "new"
        (project / "instructions").mkdir(parents=True)
        (project / "instructions" / "SOUL.md").write_text("imported soul\n")
        (project / "instructions" / "AGENTS.md").write_text("imported brief\n")
        return project

    @pytest.fixture
    def more(self, tmp_path: Path) -> Path:
        """A folder with a soul and a brief of its own."""
        more = tmp_path / "more"
        more.mkdir()
        (more / "SOUL.md").write_text("another soul\n")
        (more / "AGENTS.md").write_text("another brief\n")
        return more

    @pytest.fixture(autouse=True)
    def usual_question(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The real question, which the CLI conftest replaces for every test."""
        monkeypatch.setattr(cli_setup, "_choose_instructions", _REAL_CHOOSE_INSTRUCTIONS)

    @staticmethod
    def _current(project: Path) -> dict[str, Any]:
        return {"source": {"type": "local_dir", "paths": [str(project / "instructions")]}}

    def test_the_copies_are_erased_and_a_new_soul_is_welcome(
        self, project: Path, more: Path, answers: list[str]
    ) -> None:
        answers.append(str(more))

        result = _choose_and_save(self._current(project), project)

        assert _instruction_paths(result) == [str(more / "SOUL.md"), str(more / "AGENTS.md")]
        assert not (project / "instructions").exists()

    def test_an_empty_answer_leaves_none(self, project: Path, answers: list[str]) -> None:
        result = _choose_and_save(self._current(project), project)

        assert result is None
        assert not (project / "instructions").exists()

    def test_a_listed_path_outside_the_project_is_only_unlisted(
        self, project: Path, more: Path, answers: list[str]
    ) -> None:
        current = {"source": {"type": "local_dir", "paths": [str(more)]}}

        assert _choose_and_save(current, project) is None
        assert (more / "SOUL.md").is_file()

    def test_the_copies_stay_until_the_setup_is_saved(
        self, project: Path, answers: list[str]
    ) -> None:
        discard: list[Path] = []

        cli_setup._choose_imported_instructions(
            _console(), THEME, self._current(project), discard=discard
        )

        assert discard == [project / "instructions"]
        assert (project / "instructions" / "SOUL.md").is_file()

    def test_an_undeclared_file_is_not_the_project_s_and_stays(
        self, tmp_path: Path, answers: list[str]
    ) -> None:
        project = tmp_path / "bare"
        project.mkdir()
        (project / "SOUL.md").write_text("a file like any other\n")

        assert _choose_and_save(None, project) is None
        assert (project / "SOUL.md").is_file()


# Captured before the CLI conftest replaces it for every wizard test.
_REAL_KEEP = cli_setup._keep


def _keep_but(*changed: str) -> Any:
    """`_keep` answered yes for every section but ``changed``, with the sections asked."""
    asked: list[str] = []

    def keep(console: Any, theme: Any, section: str, current: Any, **_: Any) -> bool:
        asked.append(section)
        return section not in changed

    keep.asked = asked  # type: ignore[attr-defined]
    return keep


class TestTheCommand:
    def test_enter_on_every_section_keeps_the_export(
        self, exported: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answers: list[str]
    ) -> None:
        monkeypatch.chdir(tmp_path)
        keep = _keep_but()
        monkeypatch.setattr(cli_setup, "_keep", keep)

        result = CliRunner().invoke(app, ["setup", "--import", str(exported)])

        assert result.exit_code == 0, result.output
        # Asked as on a re-run, each section showing what the export brought.
        assert keep.asked == ["model", "instructions", "tools", "skills"]
        # Named after the imported folder, under ~/.domyn/projects.
        project = cfg.project_dir(exported.name)
        saved = yaml.safe_load((project / cfg.CONFIG_FILENAME).read_text())
        assert saved["llm_providers"][0]["model_name"] == "imported-model"
        builtin_path, mine_path = _skill_paths(saved)
        assert builtin_path.startswith(f"{P}/{BUILTINS}/")
        assert mine_path.startswith(f"{P}/skills/")
        # The built-in catalog was not offered, so nothing was installed: the one
        # built-in there is the exported one.
        assert len(list((project / BUILTINS).rglob("SKILL.md"))) == 1
        declared = saved["agents"][0]["standing_instructions"]["source"]["paths"]
        assert declared and all(p.startswith(f"{P}/") for p in declared)
        souls = [p for p in declared if p.endswith("/SOUL.md")]
        assert len(souls) == 1 and (project / souls[0].removeprefix(f"{P}/")).is_file()
        assert not any(cfg.is_instruction_autodiscovery(p) for p in declared)
        # The end says what was made, and how to use it.
        said = " ".join(result.output.split())
        assert f"Created project {exported.name}" in said
        assert f"domyn run --project {exported.name}" in said

    def test_the_export_s_code_is_left_behind(
        self, exported: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answers: list[str]
    ) -> None:
        # A tool the export carries: setup keeps no tools, so its code is not copied
        # in, and the import says so.
        (exported / "code").mkdir()
        (exported / "code" / "tool.py").write_text("import os\n", encoding="utf-8")
        config = yaml.safe_load((exported / cfg.CONFIG_FILENAME).read_text())
        config["tools"] = [{"name": "t", "type": "function", "path": f"{P}/code/tool.py"}]
        (exported / cfg.CONFIG_FILENAME).write_text(yaml.safe_dump(config))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cli_setup, "_keep", _keep_but())

        result = CliRunner().invoke(app, ["setup", "--import", str(exported)])

        assert result.exit_code == 0, result.output
        project = cfg.project_dir(exported.name)
        assert not (project / "code").exists()
        assert "Not imported: tool t" in " ".join(result.output.split())

    def test_skills_not_kept_offer_the_project_s_own(
        self, exported: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answers: list[str]
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cli_setup, "_keep", _keep_but("skills"))

        result = CliRunner().invoke(app, ["setup", "--import", str(exported)])

        assert result.exit_code == 0, result.output
        assert "Skills of the imported project" in result.output
        assert len(list((cfg.project_dir(exported.name) / BUILTINS).rglob("SKILL.md"))) == 1

    def test_instructions_not_kept_are_erased_then_asked_as_in_any_setup(
        self, exported: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answers: list[str]
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cli_setup, "_keep", _keep_but("instructions"))
        monkeypatch.setattr(cli_setup, "_choose_instructions", _REAL_CHOOSE_INSTRUCTIONS)
        asked_local: list[bool] = []

        def local_instructions(console: Any, theme: Any, current: Any, *, first_setup: bool) -> Any:
            asked_local.append(first_setup)
            return current

        monkeypatch.setattr(cli_setup, "_choose_local_instructions", local_instructions)

        result = CliRunner().invoke(app, ["setup", "--import", str(exported)])

        assert result.exit_code == 0, result.output
        project = cfg.project_dir(exported.name)
        saved = yaml.safe_load((project / cfg.CONFIG_FILENAME).read_text())
        assert "standing_instructions" not in saved["agents"][0]
        assert not (project / "instructions").exists()
        # Never as a first setup: the export listed what it found.
        assert asked_local == [False]

    def test_instructions_listed_again_after_not_being_kept_stay(
        self, exported: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answers: list[str]
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cli_setup, "_keep", _keep_but("instructions"))
        monkeypatch.setattr(cli_setup, "_choose_instructions", _REAL_CHOOSE_INSTRUCTIONS)
        project = cfg.project_dir(exported.name)
        # The instructions question answered with the project's own folder.
        answers.append(str(project / "instructions"))

        result = CliRunner().invoke(app, ["setup", "--import", str(exported)])

        assert result.exit_code == 0, result.output
        saved = yaml.safe_load((project / cfg.CONFIG_FILENAME).read_text())
        declared = _instruction_paths(saved["agents"][0].get("standing_instructions"))
        assert declared and all(p.startswith(f"{P}/instructions/") for p in declared)
        assert all((project / p.removeprefix(f"{P}/")).is_file() for p in declared)

    def test_going_back_to_keep_them_keeps_their_files(
        self, exported: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answers: list[str]
    ) -> None:
        # Not kept, then back on the path question and kept after all: the copies the
        # step would have erased are still there, as they are only erased once saved.
        monkeypatch.chdir(tmp_path)

        def back(console: Any, theme: Any, current: Any) -> Any:
            raise GoBack

        # The real question, which going back needs to land on: keep the model, not the
        # instructions, then, back there, keep them; Enter on the rest.
        monkeypatch.setattr(cli_setup, "_keep", _REAL_KEEP)
        monkeypatch.setattr(cli_setup, "_choose_instructions", back)
        answers.extend(["", "no", "yes"])

        result = CliRunner().invoke(app, ["setup", "--import", str(exported)])

        assert result.exit_code == 0, result.output
        project = cfg.project_dir(exported.name)
        saved = yaml.safe_load((project / cfg.CONFIG_FILENAME).read_text())
        declared = _instruction_paths(saved["agents"][0]["standing_instructions"])
        assert declared
        assert all((project / p.removeprefix(f"{P}/")).exists() for p in declared)

    def test_a_relative_path_with_a_trailing_slash_is_named_by_its_folder(
        self, exported: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answers: list[str]
    ) -> None:
        monkeypatch.chdir(tmp_path)

        result = CliRunner().invoke(app, ["setup", "--import", f"./{exported.name}/"])

        assert result.exit_code == 0, result.output
        assert (cfg.project_dir(exported.name) / cfg.CONFIG_FILENAME).is_file()

    def test_a_project_of_that_name_is_not_overwritten(
        self, exported: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answers: list[str]
    ) -> None:
        monkeypatch.chdir(tmp_path)
        existing = cfg.project_dir(exported.name)
        existing.mkdir(parents=True)
        (existing / cfg.CONFIG_FILENAME).write_text("mine")

        result = CliRunner().invoke(app, ["setup", "--import", str(exported)])

        assert result.exit_code == 1
        assert "not an empty folder" in " ".join(result.output.split())
        assert (existing / cfg.CONFIG_FILENAME).read_text() == "mine"

    def test_a_folder_name_that_is_no_project_name_is_refused(
        self, exported: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        badly_named = exported.rename(tmp_path / "my export")

        result = CliRunner().invoke(app, ["setup", "--import", str(badly_named)])

        assert result.exit_code == 1
        assert "Rename the folder, or choose a name with --project" in " ".join(
            result.output.split()
        )
        assert not cfg.projects_dir().exists()

    def test_an_aborted_wizard_leaves_no_project(
        self, exported: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answers: list[str]
    ) -> None:
        monkeypatch.chdir(tmp_path)

        def interrupted(*_: Any) -> str:
            raise KeyboardInterrupt

        monkeypatch.setattr("builtins.input", interrupted)

        result = CliRunner().invoke(app, ["setup", "--import", str(exported)])

        assert result.exit_code == 1
        assert not cfg.project_dir(exported.name).exists()

    def test_project_names_it_instead_of_the_folder(
        self, exported: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answers: list[str]
    ) -> None:
        monkeypatch.chdir(tmp_path)

        result = CliRunner().invoke(app, ["setup", "--project", "other", "--import", str(exported)])

        assert result.exit_code == 0, result.output
        assert (cfg.project_dir("other") / cfg.CONFIG_FILENAME).is_file()
        assert not cfg.project_dir(exported.name).exists()
        assert "domyn run --project other" in " ".join(result.output.split())

    def test_project_may_be_a_path_and_the_hint_spells_it_out(
        self, exported: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answers: list[str]
    ) -> None:
        monkeypatch.chdir(tmp_path)

        result = CliRunner().invoke(
            app, ["setup", "--project", "./here", "--import", str(exported)]
        )

        assert result.exit_code == 0, result.output
        folder = (tmp_path / "here").resolve()
        assert (folder / cfg.CONFIG_FILENAME).is_file()
        # Rich wraps long paths mid-word, so compared without any whitespace.
        assert f"domynrun--project{folder}" in "".join(result.output.split())

    def test_project_saves_a_folder_name_that_is_no_project_name(
        self, exported: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answers: list[str]
    ) -> None:
        monkeypatch.chdir(tmp_path)
        badly_named = exported.rename(tmp_path / "my export")

        result = CliRunner().invoke(
            app, ["setup", "--project", "fine", "--import", str(badly_named)]
        )

        assert result.exit_code == 0, result.output
        assert (cfg.project_dir("fine") / cfg.CONFIG_FILENAME).is_file()

    def test_it_cannot_be_shown(
        self, exported: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        result = CliRunner().invoke(app, ["setup", "--show", "--import", str(exported)])

        assert result.exit_code == 1
        assert "--import cannot be combined with --show" in result.output
        assert not cfg.projects_dir().exists()
