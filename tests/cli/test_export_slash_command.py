"""``/export <folder>``: ``domyn export`` from inside a session, for the session's own
configuration file as it is now."""

import io
from pathlib import Path
from typing import Any

import pytest
import yaml
from rich.console import Console

from domyn.agents import Agent
from domyn.agents.interfaces.cli import Theme
from domyn.agents.interfaces.cli.repl import Repl
from domyn.agents.planner_strategy import ToolUsePlannerStrategy


class _Editor:
    def __init__(self, *lines: str) -> None:
        self._lines = list(lines)

    async def prompt_async(self, *_: Any, **__: Any) -> str:
        if not self._lines:
            raise EOFError
        return self._lines.pop(0)


def _write_config(path: Path, skill_paths: list[str]) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "entrypoint": "Assistant",
                "llm_providers": [{"name": "p", "type": "openai", "model_name": "m"}],
                "agents": [
                    {
                        "name": "Assistant",
                        "description": "test agent",
                        "instruction": "Be brief.",
                        "llm_provider": "p",
                        "skills": {"source": {"type": "local_dir", "paths": skill_paths}},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _session(config: Path | None, *lines: str) -> tuple[Repl, io.StringIO]:
    agent = Agent(
        name="Assistant",
        description="test agent",
        instruction="Be brief.",
        llm_provider={"type": "openai", "model_name": "m"},
        planner=ToolUsePlannerStrategy(),
    )
    buffer = io.StringIO()
    repl = Repl(
        agent,
        Console(file=buffer, width=1000, force_terminal=False),
        Theme.build(allow_animation=False),
        enable_enhanced_input=False,
    )
    repl._config_file = config
    repl._input._session = _Editor(*lines)
    return repl, buffer


@pytest.fixture
def config(tmp_path: Path) -> Path:
    skill = tmp_path / "mine" / "notes"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: notes\ndescription: notes skill\n---\nbody\n", encoding="utf-8"
    )
    path = tmp_path / "config.yaml"
    _write_config(path, [str(skill)])
    return path


class TestExportFromTheSession:
    @pytest.mark.asyncio
    async def test_it_writes_a_project_with_the_config_and_its_skills(
        self, tmp_path: Path, config: Path
    ) -> None:
        dest = tmp_path / "out"
        repl, out = _session(config, f"/export {dest}")

        await repl.run()

        exported = yaml.safe_load((dest / "config.yaml").read_text(encoding="utf-8"))
        assert exported["agents"][0]["skills"]["source"]["paths"] == [
            "${DOMYN_PROJECT_DIR}/skills/notes"
        ]
        assert (dest / "skills" / "notes" / "SKILL.md").is_file()
        # The report `domyn export` prints.
        assert f"Exported {config} to {dest}" in out.getvalue()
        assert "domyn run --project" in out.getvalue()

    @pytest.mark.asyncio
    async def test_it_exports_the_file_as_it_is_now(self, tmp_path: Path, config: Path) -> None:
        _write_config(config, [])
        dest = tmp_path / "out"
        repl, _ = _session(config, f"/export {dest}")

        await repl.run()

        assert not (dest / "skills").exists()

    @pytest.mark.asyncio
    async def test_an_existing_folder_is_refused_and_left_alone(
        self, tmp_path: Path, config: Path
    ) -> None:
        dest = tmp_path / "out"
        dest.mkdir()
        (dest / "keep.txt").write_text("mine", encoding="utf-8")
        repl, out = _session(config, f"/export {dest}")

        await repl.run()

        assert sorted(p.name for p in dest.iterdir()) == ["keep.txt"]
        assert "Export failed" in out.getvalue()

    @pytest.mark.asyncio
    async def test_without_a_folder_it_says_how_to_use_it(self, config: Path) -> None:
        repl, out = _session(config, "/export")

        await repl.run()

        assert "Usage: /export <folder>" in out.getvalue()

    @pytest.mark.asyncio
    async def test_an_agent_from_a_python_module_has_nothing_to_export(
        self, tmp_path: Path
    ) -> None:
        repl, out = _session(None, f"/export {tmp_path / 'out'}")

        await repl.run()

        assert not (tmp_path / "out").exists()
        assert "loaded from a Python file" in out.getvalue()
