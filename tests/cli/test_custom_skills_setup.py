"""`domyn setup` asking for a folder that holds the user's own skills.

Every skill in the folder is listed on its own, by absolute path, and read from where it
is: nothing is copied. A folder the config already lists, under any spelling, is refused
and asked again, and a skill already listed is not listed twice.
"""

import io
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from domyn.agents.interfaces.cli import Theme, setup
from domyn.agents.interfaces.cli import config as cfg
from domyn.agents.interfaces.cli import select as select_mod

FIELDS = ["openai", "m", "", "", "", ""]
"""The answers up to the max tokens and context window; the skill steps follow."""

AUTO_DIRS = (Path(".domyn/skills"), Path(".agents/skills"))
"""The real project skills folders, which the CLI conftest points elsewhere."""


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "GLOBAL_DIR", tmp_path / "home" / ".domyn")
    monkeypatch.setattr(cfg, "PROJECT_DIR", tmp_path / "nowhere" / ".domyn")
    monkeypatch.setattr(select_mod, "_interactive", lambda theme: False)
    for name in ("OPENAI_API_KEY", "LLM_API_KEY"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)


# Captured before the CLI conftest replaces it for every other wizard test.
_REAL_CUSTOM_STEP = setup._choose_custom_skills
_REAL_LOCAL_STEP = setup._choose_local_skills


@pytest.fixture(autouse=True)
def custom_step(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup, "_choose_custom_skills", _REAL_CUSTOM_STEP)


@pytest.fixture
def theme() -> Theme:
    return Theme.build(allow_animation=False)


@pytest.fixture
def console() -> MagicMock:
    return MagicMock()


def _write(path: Path, name: str, extra: str = "") -> None:
    file = path if path.suffix == ".md" else path / "SKILL.md"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(f"---\nname: {name}\ndescription: d\n{extra}---\nbody\n")


@pytest.fixture
def mine(tmp_path: Path) -> Path:
    """A folder of the user's own skills, outside any project.

    A loose ``notes.md``, ``review`` and ``writing/outline`` in a group folder.
    """
    root = tmp_path / "my-skills"
    _write(root / "notes.md", "notes")
    _write(root / "review", "review")
    _write(root / "writing" / "outline", "outline")
    return root


def _skills(root: Path) -> list[str]:
    """What listing ``mine`` adds, in the walk's order: by name within each folder."""
    return [str(root / "notes.md"), str(root / "review"), str(root / "writing" / "outline")]


def _paths(config: dict[str, Any]) -> list[str]:
    skills = config["agents"][0].get("skills")
    return [] if skills is None else skills["source"]["paths"]


def _existing(paths: list[str]) -> dict[str, Any]:
    return cfg.configuration_v1_dict_from(
        cfg.ProviderConfig(type="openai", model_name="m"),
        cfg.AgentConfig(
            name="A",
            instruction="i",
            skills={"source": {"type": "local_dir", "paths": paths}},
        ),
        cfg.CompactionConfig(),
    )


def _run(
    console: Any, theme: Theme, answers: list[str], existing: dict[str, Any] | None = None
) -> dict[str, Any]:
    with patch("builtins.input", side_effect=answers):
        result = setup.run_setup(console, theme, existing, path=Path("config.yaml"))
    assert result is not None
    return result


# `none` for the built-ins keeps these tests to the custom step.
def _answers(*custom: str) -> list[str]:
    return [*FIELDS, "none", *custom]


# A re-run: every field kept, the built-ins as they were.
def _rerun(*custom: str) -> list[str]:
    return [""] * 6 + ["", *custom]


class TestTheQuestion:
    def test_an_empty_answer_adds_nothing(self, console: MagicMock, theme: Theme) -> None:
        result = _run(console, theme, _answers(""))

        assert _paths(result) == []

    def test_each_skill_is_listed_by_its_absolute_path(
        self, mine: Path, console: MagicMock, theme: Theme
    ) -> None:
        result = _run(console, theme, _answers(str(mine)))

        assert _paths(result) == _skills(mine)
        assert str(mine) not in _paths(result)

    def test_a_relative_answer_is_listed_absolute(
        self, mine: Path, console: MagicMock, theme: Theme
    ) -> None:
        result = _run(console, theme, _answers("my-skills"))

        assert _paths(result) == _skills(mine)

    def test_a_home_relative_answer_and_dot_dot_are_expanded(
        self,
        mine: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        console: MagicMock,
        theme: Theme,
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))

        result = _run(console, theme, _answers("~/my-skills/review/.."))

        assert _paths(result) == _skills(mine)

    def test_a_linked_skill_inside_is_not_listed(
        self, mine: Path, tmp_path: Path, console: MagicMock, theme: Theme
    ) -> None:
        _write(tmp_path / "elsewhere" / "secret", "secret")
        (mine / "secret").symlink_to(tmp_path / "elsewhere" / "secret", target_is_directory=True)

        result = _run(console, theme, _answers(str(mine)))

        assert _paths(result) == _skills(mine)

    def test_nothing_is_copied(self, mine: Path, console: MagicMock, theme: Theme) -> None:
        _run(console, theme, _answers(str(mine)))

        install = cfg.skills_install_dir()
        assert not install.exists() or not any(install.iterdir())
        assert (mine / "review" / "SKILL.md").exists()

    def test_it_says_how_many_it_listed(self, mine: Path, theme: Theme) -> None:
        buffer = io.StringIO()

        _run(Console(file=buffer, width=400), theme, _answers(str(mine)))

        assert f"Listed 3 skills from {mine}, read from there, not copied" in buffer.getvalue()

    def test_a_skill_missing_a_file_is_listed_with_a_warning(
        self, mine: Path, theme: Theme
    ) -> None:
        _write(mine / "plot", "plot", "external_files: [scripts/]\n")
        buffer = io.StringIO()

        result = _run(Console(file=buffer, width=400), theme, _answers(str(mine)))

        assert str(mine / "plot") in _paths(result)
        assert "Skill 'plot' lists a file that is missing" in buffer.getvalue()

    def test_ctrl_c_at_the_question_saves_nothing(self, console: MagicMock, theme: Theme) -> None:
        with patch("builtins.input", side_effect=[*_answers(), EOFError]):
            assert setup.run_setup(console, theme, path=Path("config.yaml")) is None
        assert not Path("config.yaml").exists()


class TestARefusedFolder:
    def test_a_missing_folder_is_asked_again(
        self, mine: Path, tmp_path: Path, console: MagicMock, theme: Theme
    ) -> None:
        result = _run(console, theme, _answers(str(tmp_path / "nope"), str(mine)))

        assert _paths(result) == _skills(mine)

    def test_a_folder_without_skills_is_asked_again(
        self, mine: Path, tmp_path: Path, console: MagicMock, theme: Theme
    ) -> None:
        (tmp_path / "empty").mkdir()

        result = _run(console, theme, _answers(str(tmp_path / "empty"), str(mine)))

        assert _paths(result) == _skills(mine)

    def test_a_file_is_refused(self, mine: Path, console: MagicMock, theme: Theme) -> None:
        result = _run(console, theme, _answers(str(mine / "notes.md"), ""))

        assert _paths(result) == []

    def test_a_link_is_refused(
        self, mine: Path, tmp_path: Path, console: MagicMock, theme: Theme
    ) -> None:
        link = tmp_path / "link"
        link.symlink_to(mine, target_is_directory=True)

        result = _run(console, theme, _answers(str(link), ""))

        assert _paths(result) == []

    def test_a_folder_already_listed_is_refused(
        self, mine: Path, tmp_path: Path, console: MagicMock, theme: Theme
    ) -> None:
        _write(tmp_path / "other" / "draft", "draft")

        result = _run(
            console, theme, _rerun(str(mine), str(tmp_path / "other")), _existing([str(mine)])
        )

        assert _paths(result) == [str(tmp_path / "other" / "draft"), str(mine)]

    @pytest.mark.parametrize("spelling", ["my-skills", "./my-skills/", "my-skills/review/.."])
    def test_a_listed_folder_spelled_another_way_is_refused(
        self, spelling: str, mine: Path, console: MagicMock, theme: Theme
    ) -> None:
        result = _run(console, theme, _rerun(str(mine), ""), _existing([spelling]))

        # Written back through `Path` by the built-in step, which keeps it otherwise as is.
        assert [Path(p) for p in _paths(result)] == [Path(spelling)]

    def test_a_listed_link_to_it_is_refused(
        self, mine: Path, tmp_path: Path, console: MagicMock, theme: Theme
    ) -> None:
        link = tmp_path / "link"
        link.symlink_to(mine, target_is_directory=True)

        result = _run(console, theme, _rerun(str(mine), ""), _existing([str(link)]))

        assert _paths(result) == [str(link)]

    def test_the_refusal_names_the_entry_it_matches(self, mine: Path, theme: Theme) -> None:
        buffer = io.StringIO()

        _run(
            Console(file=buffer, width=400),
            theme,
            _rerun(str(mine), ""),
            _existing(["my-skills"]),
        )

        assert f"{mine} is already listed in the config (as my-skills)" in buffer.getvalue()

    def test_a_folder_whose_skills_are_all_listed_is_refused(
        self, mine: Path, theme: Theme
    ) -> None:
        buffer = io.StringIO()

        result = _run(
            Console(file=buffer, width=400),
            theme,
            _rerun(str(mine), ""),
            _existing(_skills(mine)),
        )

        assert _paths(result) == _skills(mine)
        assert f"Every skill in {mine} is already listed" in buffer.getvalue()

    def test_three_refusals_skip_the_question(
        self, tmp_path: Path, console: MagicMock, theme: Theme
    ) -> None:
        missing = str(tmp_path / "nope")

        result = _run(console, theme, _answers(missing, missing, missing))

        assert _paths(result) == []

    def test_the_install_directory_is_refused(self, tmp_path: Path) -> None:
        inside = cfg.skills_install_dir() / "mine"
        inside.mkdir(parents=True)

        problem = setup._custom_dir_problem(inside, [])

        assert problem is not None and "rewrites from scratch" in problem

    @pytest.mark.parametrize("folder", [".domyn/skills", ".agents/skills"])
    def test_a_project_skills_folder_is_left_to_the_next_question(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, folder: str
    ) -> None:
        monkeypatch.setattr(setup, "LOCAL_SKILLS_DIRS", AUTO_DIRS)
        monkeypatch.chdir(tmp_path)
        (tmp_path / folder).mkdir(parents=True)

        problem = setup._custom_dir_problem(tmp_path / folder, [])

        assert problem is not None and "next question" in problem
        assert f"./{folder}," in problem


class TestASkillAlreadyListed:
    def test_is_not_listed_twice(self, mine: Path, theme: Theme) -> None:
        buffer = io.StringIO()
        existing = _existing(["my-skills/review"])

        result = _run(Console(file=buffer, width=400), theme, _rerun(str(mine)), existing)

        assert _paths(result) == [
            str(mine / "notes.md"),
            str(mine / "writing" / "outline"),
            "my-skills/review",
        ]
        assert "Listed 2 skills" in buffer.getvalue()
        assert "(1 already listed)" in buffer.getvalue()

    def test_a_skill_added_later_is_listed_by_answering_the_folder_again(
        self, mine: Path, console: MagicMock, theme: Theme
    ) -> None:
        first = _run(console, theme, _answers(str(mine)))
        _write(mine / "draft", "draft")

        second = _run(console, theme, _rerun(str(mine)), first)

        assert _paths(second) == [str(mine / "draft"), *_skills(mine)]


class TestAReRun:
    def test_an_empty_answer_keeps_the_skills(
        self, mine: Path, console: MagicMock, theme: Theme
    ) -> None:
        first = _run(console, theme, _answers(str(mine)))

        second = _run(console, theme, _rerun(""), first)

        assert _paths(second) == _skills(mine)

    def test_a_source_that_is_not_a_local_directory_is_not_asked(
        self, console: MagicMock, theme: Theme
    ) -> None:
        remote = {"source": {"type": "remote", "url": "https://skills.example"}}
        existing = cfg.configuration_v1_dict_from(
            cfg.ProviderConfig(type="openai", model_name="m"),
            cfg.AgentConfig(name="A", instruction="i", skills=remote),
            cfg.CompactionConfig(),
        )

        # Only the eight fields: neither skill step asks anything.
        result = _run(console, theme, [""] * 6, existing)

        assert result["agents"][0]["skills"] == remote


class TestWhatItLoads:
    def test_every_listed_skill_loads(self, mine: Path, console: MagicMock, theme: Theme) -> None:
        import asyncio

        from domyn.agents.skills import LocalDirSkillSource, SkillQuery

        result = _run(console, theme, _answers(str(mine)))

        source = LocalDirSkillSource(paths=[Path(p) for p in _paths(result)])
        index = asyncio.run(source.index(SkillQuery(task="")))
        assert sorted(d.name for d in index.skills) == ["notes", "outline", "review"]


class TestTheOrder:
    def test_after_project_skills_and_before_the_built_ins(
        self, mine: Path, console: MagicMock, theme: Theme, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from domyn.agents.skills.catalog import list_builtin_skills, skill_groups

        monkeypatch.setattr(setup, "_choose_local_skills", _REAL_LOCAL_STEP)
        for module in (cfg, setup):
            monkeypatch.setattr(module, "LOCAL_SKILLS_DIRS", AUTO_DIRS)
        # Row numbers of the built-in picker count its group rows.
        rows = select_mod._TickList([s.name for s in list_builtin_skills()], skill_groups()).rows
        charts = str(next(i for i, r in enumerate(rows, 1) if r.label == "charts"))

        result = _run(console, theme, [*FIELDS, charts, str(mine), "yes"])

        paths = _paths(result)
        assert paths[:5] == [".domyn/skills", ".agents/skills", *_skills(mine)]
        assert Path(paths[5]).is_absolute() and paths[5].endswith("charts")
