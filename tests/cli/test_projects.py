"""``--project``: a folder under ``~/.domyn/projects`` standing in for ``~/.domyn``."""

import asyncio
import copy
import io
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml
from rich.console import Console
from rich.text import Text
from typer.testing import CliRunner

from domyn.agents.cli.cli import app
from domyn.agents.interfaces.cli import Theme
from domyn.agents.interfaces.cli import config as cfg
from domyn.agents.interfaces.cli import export as cli_export
from domyn.agents.interfaces.cli.export import export_configuration
from domyn.agents.loader import interpolate_env_vars, load_application_from_file
from domyn.agents.skills import LocalDirSkillSource, SkillQuery

# Captured at import, before the CLI conftest replaces it for every test.
_REAL_APPROVALS_PATHS = cfg.approvals_paths


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(cfg, "GLOBAL_DIR", tmp_path / "home" / ".domyn")
    monkeypatch.setattr(cfg, "PROJECT_DIR", tmp_path / "nowhere" / ".domyn")
    # The real autodiscovery entries, which the CLI conftest points nowhere: these
    # tests need to tell them apart from a project's own `skills/` folder.
    monkeypatch.setattr(cfg, "LOCAL_SKILLS_DIRS", (Path(".domyn/skills"), Path(".agents/skills")))
    return cfg.GLOBAL_DIR


def _skill(folder: Path, name: str, *, external: list[str] | None = None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    meta = f"name: {name}\ndescription: {name} skill\n"
    if external:
        meta += "external_files:\n" + "".join(f"  - {e}\n" for e in external)
    (folder / "SKILL.md").write_text(f"---\n{meta}---\n\nbody\n", encoding="utf-8")
    return folder


def _config(*skill_paths: str, instructions: list[str] | None = None) -> dict[str, Any]:
    agent: dict[str, Any] = {
        "name": "Assistant",
        "description": "d",
        "instruction": "i",
        "llm_provider": "p",
        "skills": {"source": {"type": "local_dir", "paths": list(skill_paths)}},
    }
    if instructions is not None:
        agent["standing_instructions"] = {"source": {"type": "local_dir", "paths": instructions}}
    return {
        "version": "1.0",
        "entrypoint": "Assistant",
        "llm_providers": [{"name": "p", "type": "vllm", "model_name": "m", "url": "http://x"}],
        "agents": [agent],
    }


def _write_project(name: str, raw: dict[str, Any]) -> Path:
    folder = cfg.project_dir(name)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / cfg.CONFIG_FILENAME).write_text(yaml.safe_dump(raw), encoding="utf-8")
    return folder


def _skill_paths(raw: dict[str, Any]) -> list[str]:
    paths: list[str] = raw["agents"][0]["skills"]["source"]["paths"]
    return paths


class TestNames:
    @pytest.mark.parametrize("name", ["", "../x", "a/b", ".hidden", "-x", "x y"])
    def test_a_name_that_is_not_one_folder_is_refused(self, name: str) -> None:
        with pytest.raises(cfg.ProjectError):
            cfg.project_dir(name)

    def test_a_name_shaped_like_a_folder_of_approvals_is_refused(self) -> None:
        # `projects/<folder>-<12 hex>` already holds a working directory's grants.
        with pytest.raises(cfg.ProjectError):
            cfg.project_dir("work-0123456789ab")

    def test_a_plain_name_is_a_folder_under_projects(self, home: Path) -> None:
        assert cfg.project_dir("my-project") == home / "projects" / "my-project"


class TestSelecting:
    def test_a_project_with_no_config_is_refused_and_the_known_ones_named(self) -> None:
        _write_project("other", _config())

        with pytest.raises(cfg.ProjectError, match="other"):
            cfg.use_project("missing")
        assert cfg.active_project() is None

    def test_setup_may_select_one_that_does_not_exist_yet(self, home: Path) -> None:
        assert cfg.use_project("new", must_exist=False) == home / "projects" / "new"

    def test_the_folder_stands_in_for_the_home_one(self, home: Path) -> None:
        folder = _write_project("p", _config())
        # Even with a `./.domyn` config present, which otherwise wins.
        cfg.PROJECT_DIR.mkdir(parents=True)
        (cfg.PROJECT_DIR / cfg.CONFIG_FILENAME).write_text("{}", encoding="utf-8")

        cfg.use_project("p")

        assert cfg.config_path() == folder / cfg.CONFIG_FILENAME
        assert cfg.home_dir() == folder
        assert cfg.credentials_path() == folder / cfg.CREDENTIALS_FILENAME

    def test_its_instructions_are_only_the_ones_it_declares(self, home: Path) -> None:
        """A ``SOUL.md`` at the project's top is a file like any other: read only when
        the config lists it, as ``~/.domyn``'s is not read either."""
        folder = _write_project("p", _config())
        (folder / "SOUL.md").write_text("soul", encoding="utf-8")

        cfg.use_project("p")

        assert cfg.resolve_instruction_dirs(None) == []
        assert cfg.resolve_instruction_dirs([folder / "SOUL.md"]) == [folder / "SOUL.md"]

    def test_without_a_project_nothing_changes(self, home: Path) -> None:
        assert cfg.config_path() == home / cfg.CONFIG_FILENAME
        assert cfg.home_dir() == home


class TestByPath:
    """``--project <path>``: a project folder anywhere, used the same way."""

    @pytest.mark.parametrize("value", ["./a", "../a", "~/a", "/srv/a", "a/b"])
    def test_a_slash_a_dot_or_a_tilde_makes_it_a_path(self, value: str) -> None:
        assert cfg.is_project_path(value)

    @pytest.mark.parametrize("value", ["alpha", "my-project", "v1.2"])
    def test_anything_else_is_a_name(self, value: str) -> None:
        assert not cfg.is_project_path(value)

    def test_a_relative_path_is_read_against_the_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        folder = tmp_path / "work" / "alpha"
        folder.mkdir(parents=True)
        (folder / cfg.CONFIG_FILENAME).write_text(yaml.safe_dump(_config()), encoding="utf-8")
        monkeypatch.chdir(tmp_path / "work")

        assert cfg.use_project("./alpha") == folder.resolve()
        assert cfg.config_path() == folder.resolve() / cfg.CONFIG_FILENAME
        assert cfg.credentials_path() == folder.resolve() / cfg.CREDENTIALS_FILENAME

    def test_a_name_never_reads_a_folder_of_that_name_here(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "alpha").mkdir()
        (tmp_path / "alpha" / cfg.CONFIG_FILENAME).write_text("{}", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        with pytest.raises(cfg.ProjectError, match="domyn setup --project alpha"):
            cfg.use_project("alpha")

    def test_a_path_with_no_config_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(cfg.ProjectError, match="no project at"):
            cfg.use_project(str(tmp_path / "nothing"))

    def test_a_file_is_not_a_project(self, tmp_path: Path) -> None:
        (tmp_path / "f").write_text("x", encoding="utf-8")

        with pytest.raises(cfg.ProjectError, match="not a project folder"):
            cfg.use_project(str(tmp_path / "f"), must_exist=False)

    def test_selecting_it_sets_the_variable_over_a_stale_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        folder = tmp_path / "proj"
        folder.mkdir()
        (folder / cfg.CONFIG_FILENAME).write_text("{}", encoding="utf-8")
        monkeypatch.setenv(cfg.PROJECT_DIR_VAR, str(tmp_path / "another-project"))

        cfg.use_project(str(folder))

        assert os.environ[cfg.PROJECT_DIR_VAR] == str(folder.resolve())

    def test_selecting_one_by_name_sets_it_too(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / ".domyn"
        monkeypatch.setattr(cfg, "GLOBAL_DIR", home)
        folder = home / "projects" / "p"
        folder.mkdir(parents=True)
        (folder / cfg.CONFIG_FILENAME).write_text("{}", encoding="utf-8")
        monkeypatch.setenv(cfg.PROJECT_DIR_VAR, str(tmp_path / "another-project"))

        cfg.use_project("p")

        assert os.environ[cfg.PROJECT_DIR_VAR] == str(folder)

    def test_no_project_removes_an_inherited_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The agent's shell reads it to tell which project the session runs."""
        monkeypatch.setenv(cfg.PROJECT_DIR_VAR, str(tmp_path / "from-my-shell"))

        cfg.use_project(None)

        assert cfg.PROJECT_DIR_VAR not in os.environ

    def test_its_files_and_the_working_folder_s_skills_both_load(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        folder = tmp_path / "anywhere" / "proj"
        folder.mkdir(parents=True)
        (folder / cfg.CONFIG_FILENAME).write_text(
            yaml.safe_dump(_config("${DOMYN_PROJECT_DIR}/skills/mine", ".domyn/skills")),
            encoding="utf-8",
        )
        _skill(folder / "skills" / "mine" / "essay", "essay")
        elsewhere = tmp_path / "elsewhere"
        _skill(elsewhere / ".domyn" / "skills" / "local", "local")
        monkeypatch.chdir(elsewhere)
        cfg.use_project(str(folder))

        agent = asyncio.run(load_application_from_file(str(cfg.config_path())))

        loaded = asyncio.run(agent.skills.source.load(SkillQuery(task="x")))
        assert sorted(skill.name for skill in loaded.index.skills) == ["essay", "local"]


class TestWhatIsTheProjectsAlone:
    def test_approvals_live_in_the_project(self, home: Path) -> None:
        folder = _write_project("p", _config())

        cfg.use_project("p")
        paths = _REAL_APPROVALS_PATHS()

        assert paths["user"] == folder / cfg.APPROVALS_FILENAME
        assert paths["project"].is_relative_to(folder / cfg.PROJECTS_DIRNAME)

    def test_without_a_project_they_live_at_home(self, home: Path) -> None:
        paths = _REAL_APPROVALS_PATHS()

        assert paths["user"] == home / cfg.APPROVALS_FILENAME
        assert paths["project"].is_relative_to(home / cfg.PROJECTS_DIRNAME)

    def test_the_project_credentials_go_over_the_home_ones(
        self, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in ("SHARED_KEY", "HOME_ONLY", "SHELL_KEY"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(cfg, "_credentials_in_environment", {})
        home.mkdir(parents=True)
        (home / "credentials").write_text(
            "SHARED_KEY=home\nHOME_ONLY=home\nSHELL_KEY=home\n", encoding="utf-8"
        )
        folder = _write_project("p", _config())
        (folder / "credentials").write_text("SHARED_KEY=project\n", encoding="utf-8")
        monkeypatch.setenv("SHELL_KEY", "shell")

        cfg.load_credentials_into_environment()  # startup, before options are parsed
        assert os.environ["SHARED_KEY"] == "home"
        cfg.use_project("p")
        cfg.load_credentials_into_environment()  # `select_project`

        assert os.environ["SHARED_KEY"] == "project"
        assert os.environ["HOME_ONLY"] == "home"
        assert os.environ["SHELL_KEY"] == "shell"
        monkeypatch.delenv("SHARED_KEY")
        monkeypatch.delenv("HOME_ONLY")

    def test_export_never_carries_them_out(self, tmp_path: Path) -> None:
        # Not even when a skill declares them: an `external_files` entry walking up
        # into the project, and a declared folder holding the others.
        folder = _write_project("p", _config())
        skill = _skill(
            folder / "skills" / "mine" / "essay",
            "essay",
            external=["../../../credentials", "data/"],
        )
        (folder / "credentials").write_text("KEY=secret\n", encoding="utf-8")
        (skill / "data").mkdir()
        for private in ("approvals.yaml", "conversations.db", "conversations.db-wal", "ok.csv"):
            (skill / "data" / private).write_text("secret", encoding="utf-8")
        cfg.use_project("p")
        raw = cfg.read_raw_config()
        assert raw is not None
        raw["agents"][0]["skills"]["source"]["paths"] = [str(skill)]

        report = export_configuration(
            raw, tmp_path / "out", cwd=tmp_path, install_dir=folder / cfg.BUILTIN_SKILLS_DIRNAME
        )

        copied = {p.name for p in report.dest.rglob("*") if p.is_file()}
        assert copied == {"config.yaml", ".info", "SKILL.md", "ok.csv"}
        assert len([d for d in report.dropped if "never exported" in d]) == 4


class TestSetupFindsTheKey:
    """`find_api_key`: the environment (a local ``.env`` included), then the project's
    credentials, then ``~/.domyn``'s, as a session reads them."""

    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        # Set, then removed, so the value setup puts in the environment is taken back too.
        monkeypatch.setenv("E2E_KEY", "")
        monkeypatch.delenv("E2E_KEY")
        monkeypatch.chdir(tmp_path)

    def _files(self, home: Path, *, project: str | None, global_: str | None) -> None:
        home.mkdir(parents=True, exist_ok=True)
        if global_:
            (home / "credentials").write_text(f"E2E_KEY={global_}\n", encoding="utf-8")
        folder = _write_project("p", _config())
        if project:
            (folder / "credentials").write_text(f"E2E_KEY={project}\n", encoding="utf-8")
        cfg.use_project("p")

    def test_the_environment_comes_first(self, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._files(home, project="project", global_="global")
        monkeypatch.setenv("E2E_KEY", "shell")

        found = cfg.find_api_key("E2E_KEY")

        assert found is not None
        assert found.value == "shell"
        assert "environment" in found.source
        assert found.path is None

    def test_a_local_dotenv_is_named_as_such(
        self, home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._files(home, project="project", global_="global")
        (tmp_path / ".env").write_text("E2E_KEY=dotenv\n", encoding="utf-8")
        monkeypatch.setenv("E2E_KEY", "dotenv")  # what `_load_dotenv` did at startup

        found = cfg.find_api_key("E2E_KEY")

        assert found is not None
        assert found.value == "dotenv"
        assert str(tmp_path / ".env") in found.source
        assert found.path is None

    def test_then_the_project_credentials(self, home: Path) -> None:
        self._files(home, project="project", global_="global")

        found = cfg.find_api_key("E2E_KEY")

        assert found is not None
        assert found.value == "project"
        assert found.path == cfg.credentials_path()

    def test_then_the_global_credentials(self, home: Path) -> None:
        self._files(home, project=None, global_="global")

        found = cfg.find_api_key("E2E_KEY")

        assert found is not None
        assert found.value == "global"
        assert str(home / "credentials") in found.source
        assert found.path == home / "credentials"

    def test_a_value_loaded_from_a_credentials_file_is_not_the_environment(
        self, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._files(home, project="project", global_="global")
        # As startup leaves it once the project is selected.
        monkeypatch.setenv("E2E_KEY", "project")
        monkeypatch.setattr(cfg, "_credentials_in_environment", {"E2E_KEY": "project"})

        found = cfg.find_api_key("E2E_KEY")

        assert found is not None
        assert "environment" not in found.source

    def test_without_a_project_the_home_file(self, home: Path) -> None:
        home.mkdir(parents=True)
        (home / "credentials").write_text("E2E_KEY=global\n", encoding="utf-8")

        found = cfg.find_api_key("E2E_KEY")

        assert found is not None
        assert found.value == "global"

    def test_nowhere_is_none(self, home: Path) -> None:
        self._files(home, project=None, global_=None)

        assert cfg.find_api_key("E2E_KEY") is None

    def test_a_session_reads_the_global_key_the_project_does_not_set(self, home: Path) -> None:
        self._files(home, project=None, global_="global")
        provider = cfg.ProviderConfig(type="openai", api_key_env="E2E_KEY")

        assert cfg.resolve_api_key(provider) == "global"

    def test_a_session_prefers_the_project_key(self, home: Path) -> None:
        self._files(home, project="project", global_="global")
        provider = cfg.ProviderConfig(type="openai", api_key_env="E2E_KEY")

        assert cfg.resolve_api_key(provider) == "project"

    def test_setup_copies_nothing_into_the_project(self, home: Path) -> None:
        from domyn.agents.interfaces.cli import setup

        self._files(home, project=None, global_="global")
        buffer = io.StringIO()
        provider = cfg.ProviderConfig(type="openai", api_key_env="E2E_KEY")

        with patch("builtins.input", return_value=""):  # Enter keeps the saved key
            setup._ensure_api_key(
                Console(file=buffer, width=200), Theme.build(allow_animation=False), provider
            )

        assert not cfg.credentials_path().exists()
        assert str(home / "credentials") in buffer.getvalue()

    def test_a_key_typed_in_a_project_goes_to_the_project(self, home: Path) -> None:
        from domyn.agents.interfaces.cli import setup

        self._files(home, project=None, global_="global")
        provider = cfg.ProviderConfig(type="openai", api_key_env="E2E_KEY")

        with patch("builtins.input", return_value="mine"):
            setup._ensure_api_key(
                Console(file=io.StringIO()), Theme.build(allow_animation=False), provider
            )

        assert cfg.read_credentials(cfg.credentials_path()) == {"E2E_KEY": "mine"}
        assert cfg.read_credentials(home / "credentials") == {"E2E_KEY": "global"}
        found = cfg.find_api_key("E2E_KEY")
        assert found is not None
        assert (found.value, found.path) == ("mine", cfg.credentials_path())


class TestTheKeyQuestion:
    """`_ensure_api_key`: a key in a credentials file is offered masked, Enter keeps it
    and a key typed replaces it; one from the environment is only reported."""

    KEY = "sk-saved-0123"

    @pytest.fixture(autouse=True)
    def _clean(self, home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("E2E_KEY", "")
        monkeypatch.delenv("E2E_KEY")
        monkeypatch.chdir(tmp_path)
        home.mkdir(parents=True)

    def _ask(self, answer: str) -> str:
        from domyn.agents.interfaces.cli import setup

        buffer = io.StringIO()
        provider = cfg.ProviderConfig(type="openai", api_key_env="E2E_KEY")
        with patch("builtins.input", return_value=answer):
            setup._ensure_api_key(
                Console(file=buffer, width=200), Theme.build(allow_animation=False), provider
            )
        return buffer.getvalue()

    def test_enter_keeps_the_saved_key(self, home: Path) -> None:
        cfg.write_credential("E2E_KEY", self.KEY)

        shown = self._ask("")

        assert cfg.read_credentials() == {"E2E_KEY": self.KEY}
        assert "API key" in shown
        assert "empty = keep" not in shown
        assert "Saved to" not in shown

    def test_a_key_typed_replaces_it(self, home: Path) -> None:
        cfg.write_credential("E2E_KEY", self.KEY)

        shown = self._ask("sk-new")

        assert cfg.read_credentials() == {"E2E_KEY": "sk-new"}
        assert os.environ["E2E_KEY"] == "sk-new"
        assert "Saved to" in shown

    def test_a_key_pasted_is_offered_again_from_its_file(self, home: Path) -> None:
        """Pasted into the environment as a key of the file, so going back to the
        question offers it again rather than reporting it as found."""
        self._ask(self.KEY)

        found = cfg.find_api_key("E2E_KEY")

        assert found is not None
        assert found.path == cfg.credentials_path()

    def test_the_saved_key_is_never_shown(self, home: Path) -> None:
        cfg.write_credential("E2E_KEY", self.KEY)

        assert self.KEY not in self._ask("")

    def test_a_key_in_the_environment_is_only_reported(
        self, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("E2E_KEY", "shell")
        monkeypatch.setattr("builtins.input", lambda: pytest.fail("asked for the key"))

        from domyn.agents.interfaces.cli import setup

        buffer = io.StringIO()
        setup._ensure_api_key(
            Console(file=buffer, width=200),
            Theme.build(allow_animation=False),
            cfg.ProviderConfig(type="openai", api_key_env="E2E_KEY"),
        )

        assert "found (E2E_KEY, in the environment)" in buffer.getvalue()


class TestPathsInAProject:
    """A relative path means the folder domyn runs in, in a project as anywhere; a
    project names its own files with ``${DOMYN_PROJECT_DIR}``."""

    def test_the_config_is_read_as_written(self) -> None:
        written = ["${DOMYN_PROJECT_DIR}/skills/mine", ".domyn/skills", "notes"]
        folder = _write_project("p", _config(*written))

        raw = cfg.read_raw_config(folder / cfg.CONFIG_FILENAME)

        assert raw is not None
        assert _skill_paths(raw) == written

    def test_written_in_project_terms(self, tmp_path: Path) -> None:
        folder = cfg.project_dir("p")
        raw = _config(
            str(folder / "skills/a"),
            str(tmp_path / "elsewhere"),
            ".domyn/skills",
            str(folder / "skills"),
            instructions=[str(folder)],
        )

        written = cfg.in_project_terms(raw, folder)

        assert _skill_paths(written) == [
            "${DOMYN_PROJECT_DIR}/skills/a",
            str(tmp_path / "elsewhere"),
            ".domyn/skills",
            "${DOMYN_PROJECT_DIR}/skills",
        ]
        assert written["agents"][0]["standing_instructions"]["source"]["paths"] == [
            "${DOMYN_PROJECT_DIR}"
        ]
        assert cfg.expand_project_dir(written, folder) == raw

    def test_expanding_touches_only_the_project_variable(self) -> None:
        folder = cfg.project_dir("p")
        raw = _config("${DOMYN_PROJECT_DIR}/skills/a", "${OTHER}/b")
        raw["llm_providers"][0]["api_key"] = "${VLLM_API_KEY}"

        expanded = cfg.expand_project_dir(raw, folder)

        assert _skill_paths(expanded) == [str(folder / "skills/a"), "${OTHER}/b"]
        assert expanded["llm_providers"][0]["api_key"] == "${VLLM_API_KEY}"

    def test_the_agent_loads_the_project_skills_from_anywhere(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        folder = _write_project("p", _config("${DOMYN_PROJECT_DIR}/skills/mine"))
        _skill(folder / "skills" / "mine" / "essay", "essay")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        cfg.use_project("p")

        agent = asyncio.run(load_application_from_file(str(cfg.config_path())))

        loaded = asyncio.run(agent.skills.source.load(SkillQuery(task="x")))
        assert [skill.name for skill in loaded.index.skills] == ["essay"]

    def test_loaded_without_the_variable_it_names_it(self) -> None:
        # Outside the CLI, whoever loads a project's config sets the variable.
        folder = _write_project("p", _config("${DOMYN_PROJECT_DIR}/skills/mine"))

        with pytest.raises(ValueError, match="DOMYN_PROJECT_DIR"):
            asyncio.run(load_application_from_file(str(folder / cfg.CONFIG_FILENAME)))

    def test_setup_saves_a_project_in_project_terms(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from domyn.agents.interfaces.cli import select as select_mod
        from domyn.agents.interfaces.cli import setup as cli_setup

        folder = _write_project("p", _config("${DOMYN_PROJECT_DIR}/skills/old", ".domyn/skills"))
        cfg.use_project("p")
        seen: list[Any] = []

        def _skills(console: Any, theme: Any, current: Any) -> Any:
            seen.append(copy.deepcopy(current))  # what the step was handed
            return {
                "source": {
                    "type": "local_dir",
                    "paths": [str(folder / "skills/new"), ".domyn/skills"],
                }
            }

        monkeypatch.setattr(cli_setup, "_choose_builtin_skills", _skills)
        monkeypatch.setattr(select_mod, "_interactive", lambda theme: False)
        answers = iter(["vllm", "m", "http://x/v1/chat/completions", "", "", "", "", ""])
        monkeypatch.setattr("builtins.input", lambda *_: next(answers, ""))

        saved = cli_setup.run_setup(
            Console(file=io.StringIO()), Theme.build(allow_animation=False), cfg.read_raw_config()
        )

        # The step compared real folders...
        assert seen[0]["source"]["paths"] == [str(folder / "skills/old"), ".domyn/skills"]
        # ...and the file names the project's own files through the variable.
        on_disk = yaml.safe_load((folder / cfg.CONFIG_FILENAME).read_text())
        assert _skill_paths(on_disk) == ["${DOMYN_PROJECT_DIR}/skills/new", ".domyn/skills"]
        assert saved is not None and _skill_paths(saved) == _skill_paths(on_disk)


class TestAnExportIsAProject:
    def test_moved_under_projects_it_loads_what_the_original_did(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = _skill(tmp_path / "mine" / "essay", "essay")
        instructions = tmp_path / "instr"
        instructions.mkdir()
        (instructions / "AGENTS.md").write_text("work on essays\n", encoding="utf-8")
        exported = tmp_path / "shared"
        export_configuration(
            _config(str(source.parent), instructions=[str(instructions)]),
            exported,
            cwd=tmp_path,
            install_dir=tmp_path / "no-install",
        )

        cfg.projects_dir().mkdir(parents=True)
        exported.rename(cfg.project_dir("shared"))
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        cfg.use_project("shared")

        raw = cfg.read_raw_config()
        assert raw is not None
        paths = cfg.collect_skill_workspace_roots(raw)
        loaded = asyncio.run(LocalDirSkillSource(paths=paths).load(SkillQuery(task="x")))
        assert [skill.name for skill in loaded.index.skills] == ["essay"]
        declared = raw["agents"][0]["standing_instructions"]["source"]["paths"]
        assert all(p.startswith("${DOMYN_PROJECT_DIR}/") for p in declared)
        resolved = [Path(interpolate_env_vars(p, "test")) for p in declared]
        assert all(p.is_dir() for p in resolved)
        assert all(p.is_relative_to(cfg.project_dir("shared")) for p in resolved)


class TestTheCommands:
    @pytest.mark.parametrize(
        "args",
        [
            ["run", "--project", "missing", "--no-banner", "hi"],
            ["setup", "--show", "--project", "missing"],
            ["export", "out", "--project", "missing"],
            ["skills", "--project", "missing"],
        ],
    )
    def test_a_missing_project_is_an_error_not_a_new_one(
        self, args: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        result = CliRunner().invoke(app, args)

        assert result.exit_code == 1
        # Rich wraps at the terminal's width, and the message holds a long path.
        assert "domyn setup --project missing" in " ".join(result.output.split())
        assert not cfg.project_dir("missing").exists()

    def test_export_reads_the_project_and_says_how_to_install_the_copy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        folder = _write_project("p", _config("${DOMYN_PROJECT_DIR}/skills/mine"))
        _skill(folder / "skills" / "mine" / "essay", "essay")
        monkeypatch.chdir(tmp_path)

        result = CliRunner().invoke(app, ["export", "copy", "--project", "p"])

        assert result.exit_code == 0, result.output
        # Rich wraps long paths mid-word, so compared without any whitespace.
        output = "".join(result.output.split())
        assert f"domynrun--project{tmp_path / 'copy'}" in output
        assert "domynrun--projectcopy" in output
        exported = yaml.safe_load((tmp_path / "copy" / "config.yaml").read_text())
        (path,) = _skill_paths(exported)
        assert path.startswith("${DOMYN_PROJECT_DIR}/")
        copied = tmp_path / "copy" / path.removeprefix("${DOMYN_PROJECT_DIR}/")
        assert (copied / "essay" / "SKILL.md").is_file()

    def test_setup_show_reads_the_project(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_project("p", _config())
        monkeypatch.chdir(tmp_path)

        result = CliRunner().invoke(app, ["setup", "--show", "--project", "p"])

        assert result.exit_code == 0, result.output
        assert os.fspath(cfg.project_dir("p")) in result.output.replace("\n", "")


class TestTheBanner:
    """`domyn run` names the project under the logo, beside the version."""

    @staticmethod
    def _subtitle(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> Text:
        from domyn.agents.cli.commands.run import command as run_command

        seen: list[Any] = []

        def _banner(console: Any, theme: Any, *, subtitle: Any = None, **_: Any) -> None:
            seen.append(subtitle)
            raise SystemExit(0)  # the banner is all this test is about

        monkeypatch.setattr(run_command, "render_banner", _banner)
        CliRunner().invoke(app, ["run", *args, "hi"])
        assert len(seen) == 1 and isinstance(seen[0], Text)
        return seen[0]

    def test_a_project_is_named(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _write_project("alpha", _config())

        subtitle = self._subtitle(monkeypatch, ["--project", "alpha"])

        assert subtitle.plain.endswith(f"·  Project alpha ({cfg.project_dir('alpha')})")

    def test_the_name_alone_is_in_the_brand_colour(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _write_project("alpha", _config())
        # Each style by its role's name: under the test runner there is no terminal,
        # and the theme would turn every style off.
        monkeypatch.setattr(Theme, "style", lambda self, name: name)

        subtitle = self._subtitle(monkeypatch, ["--project", "alpha"])

        assert [
            subtitle.plain[s.start : s.end] for s in subtitle.spans if s.style == "accent_bold"
        ] == ["alpha"]

    def test_a_project_by_path_is_named_by_its_folder(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        folder = tmp_path / "work" / "beta"
        folder.mkdir(parents=True)
        (folder / cfg.CONFIG_FILENAME).write_text(yaml.safe_dump(_config()), encoding="utf-8")

        subtitle = self._subtitle(monkeypatch, ["--project", str(folder)])

        assert subtitle.plain.endswith(f"·  Project beta ({folder})")

    def test_a_location_under_home_is_written_with_a_tilde(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))

        assert cfg.describe_project(tmp_path / "x" / "p") == "p (~/x/p)"

    def test_without_one_only_the_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        subtitle = self._subtitle(monkeypatch, [])

        assert "Project" not in subtitle.plain
