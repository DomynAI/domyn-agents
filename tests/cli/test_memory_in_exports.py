"""Long-term memory across `domyn export` and `domyn setup --import`: the setting
travels, so memory that was on stays on and the wizard opens on the same choice; what
the agent remembered does not.
"""

import io
from pathlib import Path
from typing import Any

import pytest
import yaml
from rich.console import Console
from typer.testing import CliRunner

from domyn.agents.cli.cli import app
from domyn.agents.interfaces.cli import Theme
from domyn.agents.interfaces.cli import config as cfg
from domyn.agents.interfaces.cli import select as select_mod
from domyn.agents.interfaces.cli import setup as cli_setup
from domyn.agents.interfaces.cli.config import GLOBAL_MEMORY_REF, MemoryChoice, is_global_memory
from domyn.agents.interfaces.cli.export import (
    PrivatePlaces,
    config_paths,
    export_configuration,
    is_private,
    print_export_report,
)
from domyn.agents.interfaces.cli.project_import import ProjectImportError, import_project

P = f"${{{cfg.PROJECT_DIR_VAR}}}"
# Captured at import, before the CLI conftest replaces it for every test.
_REAL_ASK_MEMORY = cli_setup._ask_memory


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(cfg, "GLOBAL_DIR", tmp_path / "home" / ".domyn")
    monkeypatch.setattr(
        cli_setup, "skills_install_dir", lambda: cfg.home_dir() / cfg.BUILTIN_SKILLS_DIRNAME
    )
    return cfg.GLOBAL_DIR


def _config(memory_path: str | None, **options: Any) -> dict[str, Any]:
    agent: dict[str, Any] = {
        "name": "Writer",
        "description": "d",
        "instruction": "i",
        "llm_provider": "p",
    }
    if memory_path is not None:
        agent["memory_manager"] = {
            "type": "filesystem",
            "file_path": memory_path,
            "execution_mode": "agentic",
            **options,
        }
    return {
        "version": "1.0",
        "entrypoint": "Writer",
        "llm_providers": [
            {"name": "p", "type": "vllm", "model_name": "imported-model", "url": "http://x"}
        ],
        "agents": [agent],
    }


def _memory_of(config: dict[str, Any]) -> dict[str, Any] | None:
    manager: dict[str, Any] | None = config["agents"][0].get("memory_manager")
    return manager


def _export(tmp_path: Path, config: dict[str, Any]) -> tuple[Path, Any]:
    dest = tmp_path / "shared"
    report = export_configuration(config, dest, cwd=tmp_path)
    return dest, report


def _written(dest: Path) -> dict[str, Any]:
    written: dict[str, Any] = yaml.safe_load((dest / cfg.CONFIG_FILENAME).read_text())
    return written


def _global_on_this_machine() -> str:
    """What setup writes for ``global`` here: an absolute path."""
    return cfg.memory_file_path("global")


class TestExportKeepsTheSetting:
    def test_a_global_memory_is_written_for_whoever_runs_the_project(self, tmp_path: Path) -> None:
        # Setup writes this machine's absolute path; an export cannot name it.
        dest, _ = _export(tmp_path, _config(_global_on_this_machine()))

        assert _memory_of(_written(dest))["file_path"] == GLOBAL_MEMORY_REF  # type: ignore[index]

    def test_a_project_memory_keeps_its_place_in_the_project(self, tmp_path: Path) -> None:
        dest, _ = _export(tmp_path, _config(f"{P}/memory.md"))

        assert _memory_of(_written(dest))["file_path"] == f"{P}/memory.md"  # type: ignore[index]

    def test_settings_tuned_by_hand_travel_with_it(self, tmp_path: Path) -> None:
        dest, _ = _export(tmp_path, _config(f"{P}/memory.md", recency_limit=7, max_chars=900))

        memory = _memory_of(_written(dest))
        assert memory is not None
        assert (memory["recency_limit"], memory["max_chars"], memory["execution_mode"]) == (
            7,
            900,
            "agentic",
        )

    def test_a_path_written_by_hand_elsewhere_becomes_the_global_memory_with_a_warning(
        self, tmp_path: Path
    ) -> None:
        # The wizard reads it as global, so that is the choice exported.
        dest, report = _export(tmp_path, _config("/data/notes/memory.md"))

        assert _memory_of(_written(dest))["file_path"] == GLOBAL_MEMORY_REF  # type: ignore[index]
        assert any("/data/notes/memory.md" in line for line in report.warnings)

    def test_no_memory_stays_no_memory(self, tmp_path: Path) -> None:
        dest, report = _export(tmp_path, _config(None))

        assert _memory_of(_written(dest)) is None
        assert not any("memor" in line for line in report.dropped + report.warnings)

    def test_the_source_config_is_left_as_it_was(self, tmp_path: Path) -> None:
        raw = _config(_global_on_this_machine())

        _export(tmp_path, raw)

        assert _memory_of(raw)["file_path"] == _global_on_this_machine()  # type: ignore[index]


class TestExportLeavesTheMemoriesBehind:
    @pytest.mark.parametrize("scope", ["global", "project"])
    def test_no_memory_file_is_copied_and_the_report_says_so(
        self, tmp_path: Path, scope: str
    ) -> None:
        source = cfg.memory_file_path(scope)
        if scope == "global":
            Path(source).parent.mkdir(parents=True)
            Path(source).write_text("- the user's secrets\n")

        dest, report = _export(tmp_path, _config(source))

        assert not list(dest.rglob(cfg.MEMORY_FILENAME))
        assert any("what it remembers is not" in line for line in report.dropped)

    def test_the_report_names_it_on_screen(self, tmp_path: Path) -> None:
        _, report = _export(tmp_path, _config(f"{P}/memory.md"))
        console = Console(file=io.StringIO(), width=200)

        print_export_report(console, Theme.build(allow_animation=False), tmp_path, report)

        said = " ".join(console.file.getvalue().split())  # type: ignore[attr-defined]
        assert "the memory setting is exported, what it remembers is not" in said

    def test_a_memory_file_is_private_wherever_it_is(self, tmp_path: Path) -> None:
        # So no skill's external_files can carry it out, and no import copies it in.
        assert is_private(tmp_path / "anywhere" / cfg.MEMORY_FILENAME, PrivatePlaces.now())


class TestTheGlobalMemoryIsTheUsersOwn:
    @pytest.mark.parametrize(
        "path",
        [GLOBAL_MEMORY_REF, "SETUP"],
        ids=["as exported", "as setup writes it"],
    )
    def test_it_is_not_a_path_outside_the_project(self, path: str) -> None:
        path = _global_on_this_machine() if path == "SETUP" else path

        assert is_global_memory(path)
        assert config_paths(_config(path)) == []

    def test_any_other_memory_file_outside_is_one_of_the_paths_checked(self) -> None:
        # What `--project <path>` asks about, and what an import refuses.
        assert not is_global_memory("~/x.md")
        assert ("agent 'Writer' memory_manager", "~/x.md") in config_paths(_config("~/x.md"))


class TestImportKeepsTheSetting:
    def test_a_project_memory_comes_back_and_starts_empty(self, tmp_path: Path) -> None:
        dest, _ = _export(tmp_path, _config(f"{P}/memory.md"))
        project = tmp_path / "new"

        imported = import_project(dest, project)

        assert _memory_of(imported.config)["file_path"] == f"{P}/memory.md"  # type: ignore[index]
        assert not (project / cfg.MEMORY_FILENAME).exists()

    def test_a_global_memory_comes_back_as_this_machines_global_memory(
        self, tmp_path: Path
    ) -> None:
        dest, _ = _export(tmp_path, _config(_global_on_this_machine()))

        imported = import_project(dest, tmp_path / "new")

        assert _memory_of(imported.config)["file_path"] == GLOBAL_MEMORY_REF  # type: ignore[index]
        assert cfg.current_memory_from(imported.config).scope == "global"

    def test_a_memory_file_put_in_the_export_by_hand_is_not_copied(self, tmp_path: Path) -> None:
        dest, _ = _export(tmp_path, _config(f"{P}/memory.md"))
        (dest / cfg.MEMORY_FILENAME).write_text("- someone else's secrets\n")
        project = tmp_path / "new"

        import_project(dest, project)

        assert not (project / cfg.MEMORY_FILENAME).exists()

    def test_an_edited_export_naming_another_memory_file_is_refused(self, tmp_path: Path) -> None:
        # The agent writes to that file: an export's author must not choose it.
        source = tmp_path / "hand-made"
        source.mkdir()
        (source / cfg.CONFIG_FILENAME).write_text(yaml.safe_dump(_config("~/.zshrc")))

        with pytest.raises(ProjectImportError, match="outside itself"):
            import_project(source, tmp_path / "new")


class TestAMemorySettingNeverRefusesAnExport:
    def test_every_memory_path_is_placed(self, tmp_path: Path) -> None:
        # In the project, or as the global memory: never left outside, so never
        # refused (`ExportError`) for naming a path outside the export.
        dest, _ = _export(tmp_path, _config("/anywhere/at/all.md"))

        assert config_paths(_written(dest)) == []


class TestTheImportedProjectsSetup:
    @pytest.mark.parametrize("scope", ["global", "project"])
    def test_the_wizard_opens_on_the_exported_choice_and_enter_keeps_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scope: str
    ) -> None:
        dest, _ = _export(tmp_path, _config(cfg.memory_file_path(scope), recency_limit=7))
        seen = self._wizard(tmp_path, monkeypatch, dest, answer="")

        assert [choice.scope for choice in seen] == [scope]
        memory = _memory_of(self._saved(tmp_path))
        assert memory is not None and cfg.current_memory_from(self._saved(tmp_path)).scope == scope
        assert memory["recency_limit"] == 7  # the same choice keeps what was tuned

    def test_the_choice_can_be_changed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dest, _ = _export(tmp_path, _config(f"{P}/memory.md"))

        self._wizard(tmp_path, monkeypatch, dest, answer="no")

        assert _memory_of(self._saved(tmp_path)) is None

    def _wizard(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dest: Path, answer: str
    ) -> list[MemoryChoice]:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cli_setup, "_keep", lambda *a, **k: True)
        monkeypatch.setattr(select_mod, "_interactive", lambda theme: False)
        script: list[str] = []
        monkeypatch.setattr("builtins.input", lambda *_: script.pop(0) if script else "")
        seen: list[MemoryChoice] = []

        def ask_memory(console: Any, theme: Any, current: MemoryChoice) -> MemoryChoice:
            seen.append(current)
            script.insert(0, answer)
            return _REAL_ASK_MEMORY(console, theme, current)

        monkeypatch.setattr(cli_setup, "_ask_memory", ask_memory)

        result = CliRunner().invoke(app, ["setup", "--import", str(dest)])

        assert result.exit_code == 0, result.output
        return seen

    @staticmethod
    def _saved(tmp_path: Path) -> dict[str, Any]:
        project = cfg.GLOBAL_DIR / "projects" / "shared"
        saved: dict[str, Any] = yaml.safe_load((project / cfg.CONFIG_FILENAME).read_text())
        return saved
