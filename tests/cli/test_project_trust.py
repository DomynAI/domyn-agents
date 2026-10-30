"""Every time a project is used, what it would read outside itself and the code it would
run are shown and asked about (`project_trust`, `select_project`)."""

from pathlib import Path
from typing import Any

import pytest
import typer
import yaml
from rich.console import Console
from typer.testing import CliRunner

from domyn.agents.cli import project as cli_project
from domyn.agents.cli.cli import app
from domyn.agents.interfaces.cli import Theme
from domyn.agents.interfaces.cli import config as cfg

P = cfg.PROJECT_DIR_REF


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(cfg, "GLOBAL_DIR", tmp_path / "home" / ".domyn")
    monkeypatch.setattr(cfg, "PROJECT_DIR", tmp_path / "nowhere" / ".domyn")
    monkeypatch.chdir(tmp_path)
    return cfg.GLOBAL_DIR


def _project(tmp_path: Path, **extra: Any) -> Path:
    """A project folder: a config, and whatever ``extra`` adds to it."""
    folder = tmp_path / "received"
    folder.mkdir(exist_ok=True)
    config = {
        "version": "1.0",
        "entrypoint": "A",
        "llm_providers": [{"name": "p", "type": "vllm", "model_name": "m", "url": "http://x"}],
        "agents": [{"name": "A", "llm_provider": "p"}],
        **extra,
    }
    (folder / cfg.CONFIG_FILENAME).write_text(yaml.safe_dump(config), encoding="utf-8")
    return folder


def _instructions(*paths: str) -> dict[str, Any]:
    source = {"type": "local_dir", "paths": list(paths)}
    return {
        "agents": [{"name": "A", "llm_provider": "p", "standing_instructions": {"source": source}}]
    }


def _with_tool(tmp_path: Path, code: str = "import os\n") -> Path:
    folder = _project(
        tmp_path, tools=[{"name": "t", "type": "function", "path": f"{P}/code/tool.py"}]
    )
    (folder / "code").mkdir(exist_ok=True)
    (folder / "code" / "tool.py").write_text(code, encoding="utf-8")
    return folder


_said: list[str] = []


def _select(folder: Path, **flags: Any) -> Path | None:
    console = Console(record=True, width=300)
    try:
        return cli_project.select_project(
            str(folder), console, Theme.build(allow_animation=False), **flags
        )
    finally:
        _said[:] = [" ".join(console.export_text().split())]


@pytest.fixture
def answer(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """What the yes/no question is answered, in order; asking past them fails."""
    answers: list[str] = []
    monkeypatch.setattr(cli_project, "interactive_terminal", lambda: True)

    def select(*args: Any, **kwargs: Any) -> str:
        assert answers, f"asked unexpectedly: {args[2]}"
        return answers.pop(0)

    monkeypatch.setattr(cli_project, "select", select)
    return answers


class TestFilesOutside:
    @pytest.mark.parametrize(
        ("extra", "named"),
        [
            (_instructions("~/.ssh/id_rsa"), "~/.ssh/id_rsa"),
            (_instructions(f"{P}/../../.ssh/id_rsa"), "/../../.ssh/id_rsa"),
            ({"tool_bundles": [{"name": "b", "workspace": "~"}]}, "workspace"),
        ],
        ids=["home", "climbing", "workspace"],
    )
    def test_without_a_terminal_they_are_refused(
        self, tmp_path: Path, extra: dict[str, Any], named: str
    ) -> None:
        folder = _project(tmp_path, **extra)

        with pytest.raises(typer.Exit):
            _select(folder)

        assert named in _said[0] and "Not used" in _said[0]

    def test_a_skill_leading_out_is_one(self, tmp_path: Path) -> None:
        folder = _project(tmp_path)
        skill = folder / "skills" / "s"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            "---\nname: s\ndescription: d\nexternal_files:\n  - ../../../secret.txt\n---\nb\n",
            encoding="utf-8",
        )

        with pytest.raises(typer.Exit):
            _select(folder)

        assert "secret.txt" in _said[0]

    def test_a_link_out_of_the_folder_is_judged_by_where_it_leads(self, tmp_path: Path) -> None:
        folder = _project(tmp_path)
        (tmp_path / "elsewhere").mkdir()
        (folder / "notes").symlink_to(tmp_path / "elsewhere")
        _project(tmp_path, **_instructions(f"{P}/notes"))

        with pytest.raises(typer.Exit):
            _select(folder)

    def test_files_inside_ask_nothing_however_written(self, tmp_path: Path) -> None:
        folder = _project(tmp_path)
        _project(tmp_path, **_instructions(f"{P}/brief", str(folder / "AGENTS.md")))

        _select(folder)

    def test_on_a_terminal_they_are_asked_about_every_time(
        self, tmp_path: Path, answer: list[str]
    ) -> None:
        folder = _project(tmp_path, tool_bundles=[{"name": "b", "workspace": "/srv/w"}])

        answer.append("yes")
        _select(folder)
        answer.append("yes")
        _select(folder)  # nothing was remembered: asked again
        answer.append("no")
        with pytest.raises(typer.Exit):
            _select(folder)

        assert "/srv/w" in _said[0]

    def test_nothing_is_written_anywhere(self, tmp_path: Path, answer: list[str]) -> None:
        folder = _with_tool(tmp_path)
        before = sorted(p for p in tmp_path.rglob("*"))

        answer.append("yes")
        _select(folder, runs_code=True)

        assert sorted(p for p in tmp_path.rglob("*")) == before


class TestTheCode:
    def test_without_a_terminal_it_is_shown_and_not_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cli_project, "interactive_terminal", lambda: False)
        folder = _with_tool(tmp_path, "import no_such_module_anywhere\n")

        with pytest.raises(typer.Exit):
            _select(folder, runs_code=True)

        said = _said[0]
        assert "code/tool.py" in said and "no_such_module_anywhere" in said
        assert "Not run" in said

    def test_it_is_asked_about_every_run(self, tmp_path: Path, answer: list[str]) -> None:
        folder = _with_tool(tmp_path)

        answer.extend(["yes", "yes"])
        _select(folder, runs_code=True)
        _select(folder, runs_code=True)

        assert not answer

    def test_declined_it_is_not_run(self, tmp_path: Path, answer: list[str]) -> None:
        folder = _with_tool(tmp_path)

        answer.append("no")
        with pytest.raises(typer.Exit):
            _select(folder, runs_code=True)

    def test_commands_that_run_no_code_do_not_ask(self, tmp_path: Path) -> None:
        folder = _with_tool(tmp_path)

        _select(folder)  # `domyn skills`, `export`…: nothing asked, nothing run


class TestTheCommands:
    def test_run_refuses_an_edited_export_without_a_terminal(self, tmp_path: Path) -> None:
        folder = _project(tmp_path, **_instructions("~/.ssh/id_rsa"))

        result = CliRunner().invoke(app, ["run", "--project", str(folder), "--no-banner", "hi"])

        assert result.exit_code == 1
        assert "Not used" in " ".join(result.output.split())

    def test_export_refuses_it_too(self, tmp_path: Path) -> None:
        folder = _project(tmp_path, tool_bundles=[{"name": "b", "workspace": "~"}])

        result = CliRunner().invoke(
            app, ["export", str(tmp_path / "out"), "--project", str(folder)]
        )

        assert result.exit_code == 1
        assert not (tmp_path / "out").exists()


def _named(home: Path) -> Path:
    """``~/.domyn/projects/p``, naming a file outside itself and running code."""
    folder = home / "projects" / "p"
    folder.mkdir(parents=True)
    config = {
        "version": "1.0",
        "entrypoint": "A",
        "llm_providers": [{"name": "p", "type": "vllm", "model_name": "m"}],
        "tools": [{"name": "t", "type": "function", "path": f"{P}/tool.py"}],
        **_instructions("~/notes/AGENTS.md"),
    }
    (folder / cfg.CONFIG_FILENAME).write_text(yaml.safe_dump(config), encoding="utf-8")
    (folder / "tool.py").write_text("import os\n", encoding="utf-8")
    return folder


class TestByName:
    def test_a_project_under_domyn_projects_is_trusted(self, home: Path) -> None:
        # What `domyn setup --project p` or `--import` made: used as it is.
        folder = _named(home)

        console = Console(record=True, width=300)
        selected = cli_project.select_project(
            "p", console, Theme.build(allow_animation=False), runs_code=True
        )

        assert selected == folder
        assert "outside" not in console.export_text()

    def test_the_same_folder_by_path_is_checked(self, home: Path) -> None:
        folder = _named(home)

        with pytest.raises(typer.Exit):
            _select(folder)
