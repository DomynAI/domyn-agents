"""`domyn setup` turning on project skills autodiscovery, and `domyn run` saying when a skill
is gone.

Setup asks one question and, on yes, lists `.domyn/skills` and `.agents/skills` as written:
relative, they are read every turn from the folder `domyn run` starts in. Nothing is walked or
copied by setup.
"""

import io
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml
from rich.console import Console
from typer.testing import CliRunner

from domyn.agents import Agent
from domyn.agents.cli.cli import app
from domyn.agents.cli.commands.run.command import _missing_skills_warning, _startup_index
from domyn.agents.interfaces.cli import Theme, setup
from domyn.agents.interfaces.cli import config as cfg
from domyn.agents.interfaces.cli import select as select_mod
from domyn.agents.skills import InMemorySkillSource, LocalDirSkillSource, SkillsRuntime

AUTO = [".domyn/skills", ".agents/skills"]
"""What a yes writes, in precedence order."""

FIELDS = ["openai", "m", "", "", "", ""]
"""The answers up to the max tokens and context window; the two skill steps follow."""


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "GLOBAL_DIR", tmp_path / "home" / ".domyn")
    monkeypatch.setattr(cfg, "PROJECT_DIR", tmp_path / "nowhere" / ".domyn")
    monkeypatch.setattr(select_mod, "_interactive", lambda theme: False)
    for name in ("OPENAI_API_KEY", "LLM_API_KEY"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


@pytest.fixture
def theme() -> Theme:
    return Theme.build(allow_animation=False)


@pytest.fixture
def console() -> MagicMock:
    return MagicMock()


def _write(path: Path, name: str, description: str = "does a thing") -> None:
    if path.suffix == ".md":
        path.parent.mkdir(parents=True, exist_ok=True)
        file = path
    else:
        path.mkdir(parents=True, exist_ok=True)
        file = path / "SKILL.md"
    file.write_text(f"---\nname: {name}\ndescription: {description}\n---\nbody\n")


# Captured before the CLI conftest replaces it for every other wizard test.
_REAL_LOCAL_STEP = setup._choose_local_skills


@pytest.fixture(autouse=True)
def local_step(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup, "_choose_local_skills", _REAL_LOCAL_STEP)
    for module in (cfg, setup):
        monkeypatch.setattr(module, "LOCAL_SKILLS_DIRS", tuple(Path(p) for p in AUTO))


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project with both skills folders, and the CLI started in it.

    ``.domyn/skills/writing/essay-draft``, ``.domyn/skills/writing/outline``, a loose
    ``.domyn/skills/notes.md``, ``.domyn/skills/write_essay`` and
    ``.agents/skills/review``.
    """
    root = tmp_path / "project"
    _write(root / ".domyn" / "skills" / "writing" / "essay-draft", "essay-draft")
    _write(root / ".domyn" / "skills" / "writing" / "outline", "outline")
    _write(root / ".domyn" / "skills" / "notes.md", "notes")
    _write(root / ".domyn" / "skills" / "write_essay", "write_essay")
    _write(root / ".agents" / "skills" / "review", "review")
    monkeypatch.chdir(root)
    return root


def _paths(config: dict[str, Any]) -> list[str]:
    skills = config["agents"][0].get("skills")
    return [] if skills is None else skills["source"]["paths"]


def _run(
    console: Any,
    theme: Theme,
    answers: list[str],
    existing: dict[str, Any] | None = None,
    target: Path | None = None,
) -> dict[str, Any]:
    with patch("builtins.input", side_effect=answers):
        result = setup.run_setup(console, theme, existing, path=target or Path("config.yaml"))
    assert result is not None
    return result


# `none` for the built-ins keeps these tests to the local step.
def _answers(*local: str) -> list[str]:
    return [*FIELDS, "none", *local]


class TestTheQuestion:
    def test_yes_is_the_default_on_a_first_setup_and_lists_both_folders(
        self, project: Path, console: MagicMock, theme: Theme
    ) -> None:
        result = _run(console, theme, _answers(""))

        assert _paths(result) == AUTO

    def test_answering_no_lists_nothing(
        self, project: Path, console: MagicMock, theme: Theme
    ) -> None:
        result = _run(console, theme, _answers("no"))

        assert _paths(result) == []

    def test_it_is_asked_with_no_skills_folder_since_it_applies_to_every_folder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, console: MagicMock, theme: Theme
    ) -> None:
        monkeypatch.chdir(tmp_path)

        result = _run(console, theme, _answers("yes"))

        assert _paths(result) == AUTO
        assert not (tmp_path / ".domyn").exists() and not (tmp_path / ".agents").exists()

    def test_it_says_where_the_folder_is_read_from(self, project: Path, theme: Theme) -> None:
        buffer = io.StringIO()

        _run(Console(file=buffer, width=160), theme, _answers(""))

        assert (
            "./.domyn/skills and ./.agents/skills are loaded from the folder domyn runs in"
            in buffer.getvalue()
        )

    def test_ctrl_c_at_the_question_saves_nothing(
        self, project: Path, console: MagicMock, theme: Theme
    ) -> None:
        with patch("builtins.input", side_effect=[*_answers(), EOFError]):
            assert setup.run_setup(console, theme, path=Path("config.yaml")) is None
        assert not Path("config.yaml").exists()

    def test_nothing_is_copied(self, project: Path, console: MagicMock, theme: Theme) -> None:
        _run(console, theme, _answers(""))

        assert not cfg.skills_install_dir().exists() or not any(cfg.skills_install_dir().iterdir())


class TestAReRun:
    def test_an_empty_answer_keeps_it_on(
        self, project: Path, console: MagicMock, theme: Theme
    ) -> None:
        first = _run(console, theme, _answers("yes"))

        second = _run(console, theme, [""] * 6 + ["", ""], first)

        assert _paths(second) == AUTO

    def test_an_empty_answer_keeps_it_off(
        self, project: Path, tmp_path: Path, console: MagicMock, theme: Theme
    ) -> None:
        # A config with a `skills:` entry: one with none at all is a first setup to
        # both steps, and gets their defaults.
        team = str(tmp_path / "team")

        second = _run(console, theme, [""] * 6 + ["", ""], _existing([team]))

        assert _paths(second) == [team]

    def test_answering_no_turns_it_off(
        self, project: Path, console: MagicMock, theme: Theme
    ) -> None:
        first = _run(console, theme, _answers("yes"))

        second = _run(console, theme, [""] * 6 + ["", "no"], first)

        assert _paths(second) == []

    def test_it_is_listed_once_however_often_setup_runs(
        self, project: Path, console: MagicMock, theme: Theme
    ) -> None:
        config = _run(console, theme, _answers("yes"))
        for _ in range(2):
            config = _run(console, theme, [""] * 6 + ["", "yes"], config)

        assert _paths(config) == AUTO


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


class TestWhatTheConfigHolds:
    def test_it_comes_before_the_built_ins_so_the_project_wins_a_clash(
        self, project: Path, console: MagicMock, theme: Theme
    ) -> None:
        from domyn.agents.skills.catalog import list_builtin_skills, skill_groups

        # Row numbers of the built-in picker count its group rows.
        rows = select_mod._TickList([s.name for s in list_builtin_skills()], skill_groups()).rows
        charts = str(next(i for i, r in enumerate(rows, 1) if r.label == "charts"))

        result = _run(console, theme, [*FIELDS, charts, ""])

        paths = _paths(result)
        assert paths[:2] == AUTO
        assert Path(paths[2]).is_absolute() and paths[2].endswith("charts")

    def test_a_path_the_user_listed_elsewhere_survives(
        self, project: Path, tmp_path: Path, console: MagicMock, theme: Theme
    ) -> None:
        team = str(tmp_path / "team" / "review")

        result = _run(console, theme, [""] * 6 + ["none", "yes"], _existing([team]))

        assert _paths(result) == [*AUTO, team]

    def test_entries_into_either_folder_belong_to_the_step(
        self, project: Path, console: MagicMock, theme: Theme
    ) -> None:
        existing = _existing(["./.domyn/skills/write_essay", ".agents/skills/review", "skills"])

        on = _run(console, theme, [""] * 6 + ["none", "yes"], existing)
        off = _run(console, theme, [""] * 6 + ["none", "no"], existing)

        # `skills` is an ordinary path now, kept like any other the user listed.
        assert _paths(on) == [*AUTO, "skills"]
        assert _paths(off) == ["skills"]

    def test_declining_the_delete_prompt_still_runs_the_local_step(
        self, project: Path, console: MagicMock, theme: Theme
    ) -> None:
        stray = cfg.skills_install_dir() / "mine"
        stray.mkdir(parents=True)
        (stray / "SKILL.md").write_text("mine")

        result = _run(console, theme, [*FIELDS, "none", "no", ""])

        assert (stray / "SKILL.md").exists()
        assert _paths(result) == AUTO

    def test_a_source_that_is_not_a_local_directory_is_left_alone(
        self, project: Path, console: MagicMock, theme: Theme
    ) -> None:
        remote = {"source": {"type": "remote", "url": "https://skills.example"}}
        existing = cfg.configuration_v1_dict_from(
            cfg.ProviderConfig(type="openai", model_name="m"),
            cfg.AgentConfig(name="A", instruction="i", skills=remote),
            cfg.CompactionConfig(),
        )

        result = _run(console, theme, [""] * 6, existing)

        assert result["agents"][0]["skills"] == remote

    def test_the_config_is_written_as_yaml_a_reader_can_load(
        self, project: Path, console: MagicMock, theme: Theme
    ) -> None:
        _run(console, theme, _answers(""), target=Path("config.yaml"))

        on_disk = yaml.safe_load(Path("config.yaml").read_text())
        assert on_disk["agents"][0]["skills"]["source"]["paths"] == AUTO


class TestWhatItLoads:
    @staticmethod
    def _names(paths: list[str]) -> list[str]:
        import asyncio

        from domyn.agents.skills import SkillQuery

        source = LocalDirSkillSource(paths=[Path(p) for p in paths])
        return sorted(d.name for d in asyncio.run(source.index(SkillQuery(task=""))).skills)

    def test_every_skill_in_both_folders_group_folders_included(
        self, project: Path, console: MagicMock, theme: Theme
    ) -> None:
        paths = _paths(_run(console, theme, _answers("")))

        assert self._names(paths) == ["essay-draft", "notes", "outline", "review", "write_essay"]

    def test_the_domyn_folder_wins_a_name_clash(
        self, project: Path, console: MagicMock, theme: Theme
    ) -> None:
        import asyncio

        from domyn.agents.skills import SkillQuery

        _write(project / ".agents" / "skills" / "notes", "notes", "the shared one")
        paths = _paths(_run(console, theme, _answers("")))

        source = LocalDirSkillSource(paths=[Path(p) for p in paths])
        index = asyncio.run(source.index(SkillQuery(task="")))
        notes = index.by_name("notes")
        assert notes is not None and notes.description == "does a thing"

    def test_each_folder_domyn_starts_in_has_its_own(
        self,
        project: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        console: MagicMock,
        theme: Theme,
    ) -> None:
        paths = _paths(_run(console, theme, _answers("")))
        other = tmp_path / "other"
        _write(other / ".agents" / "skills" / "review" / "checklist", "checklist")
        monkeypatch.chdir(other)

        assert self._names(paths) == ["checklist"]

    def test_a_skill_added_after_setup_is_loaded_without_running_it_again(
        self, project: Path, console: MagicMock, theme: Theme
    ) -> None:
        paths = _paths(_run(console, theme, _answers("")))
        _write(project / ".agents" / "skills" / "fresh", "fresh")

        assert "fresh" in self._names(paths)

    def test_a_folder_without_skills_loads_nothing(
        self,
        project: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        console: MagicMock,
        theme: Theme,
    ) -> None:
        paths = _paths(_run(console, theme, _answers("")))
        (tmp_path / "bare").mkdir()
        monkeypatch.chdir(tmp_path / "bare")

        assert self._names(paths) == []

    def test_a_symlinked_skills_folder_loads_nothing_and_is_not_reachable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outside = tmp_path / "outside"
        _write(outside / "secret", "secret")
        (tmp_path / "repo" / ".domyn").mkdir(parents=True)
        (tmp_path / "repo" / ".domyn" / "skills").symlink_to(outside)
        monkeypatch.chdir(tmp_path / "repo")

        assert self._names([".domyn/skills"]) == []
        assert cfg.collect_skill_workspace_roots(_existing([".domyn/skills"])) == []

    def test_a_symlinked_parent_folder_loads_nothing_and_is_not_reachable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cloned repo committing `.agents -> ~` must not make `.agents/skills` read `~/skills`."""
        outside = tmp_path / "outside"
        _write(outside / "skills" / "secret", "secret")
        (tmp_path / "repo").mkdir()
        (tmp_path / "repo" / ".agents").symlink_to(outside)
        monkeypatch.chdir(tmp_path / "repo")

        assert self._names([".agents/skills"]) == []
        assert cfg.collect_skill_workspace_roots(_existing([".agents/skills"])) == []


def _agent(llm_provider: Any, paths: list[Path] | None = None, **kwargs: Any) -> Agent:
    skills = SkillsRuntime(source=LocalDirSkillSource(paths=paths)) if paths is not None else None
    return Agent(
        name="A",
        description="d",
        instruction="i",
        llm_provider=llm_provider,
        skills=skills,
        **kwargs,
    )


class TestTheStartupWarning:
    def test_a_missing_relative_path_is_named_with_where_it_is_read_from(
        self, llm_provider: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        agent = _agent(llm_provider, [Path("skills/write_essay"), Path("skills/old-draft")])

        warning = _missing_skills_warning(agent)

        assert warning is not None
        assert "Skill not found: skills/write_essay, skills/old-draft" in warning
        assert str(Path.cwd()) in warning
        assert "domyn setup` there" in warning

    def test_a_missing_absolute_path_is_named_without_the_relative_line(
        self, llm_provider: Any, tmp_path: Path
    ) -> None:
        gone = tmp_path / "gone" / "charts"

        warning = _missing_skills_warning(_agent(llm_provider, [gone]))

        assert warning is not None
        assert f"Skill not found: {gone}" in warning
        assert "Relative paths" not in warning
        assert "Run `domyn setup` to update" in warning

    def test_nothing_is_said_when_every_listed_path_is_there(
        self, llm_provider: Any, tmp_path: Path
    ) -> None:
        (tmp_path / "a").mkdir()
        (tmp_path / "b.md").write_text("x")

        assert (
            _missing_skills_warning(_agent(llm_provider, [tmp_path / "a", tmp_path / "b.md"]))
            is None
        )

    def test_only_the_missing_ones_are_named(self, llm_provider: Any, tmp_path: Path) -> None:
        (tmp_path / "here").mkdir()

        warning = _missing_skills_warning(
            _agent(llm_provider, [tmp_path / "here", tmp_path / "gone"])
        )

        assert warning is not None
        assert "gone" in warning and "here," not in warning

    def test_a_long_list_is_cut_with_a_count(self, llm_provider: Any, tmp_path: Path) -> None:
        warning = _missing_skills_warning(
            _agent(llm_provider, [tmp_path / f"s{i}" for i in range(8)])
        )

        assert warning is not None
        assert "and 3 more" in warning
        assert "s4" in warning and "s5" not in warning

    def test_a_home_relative_path_is_expanded_before_it_is_checked(
        self, llm_provider: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "home" / ".domyn" / "skills" / "x").mkdir(parents=True)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))

        assert _missing_skills_warning(_agent(llm_provider, [Path("~/.domyn/skills/x")])) is None

    def test_missing_autodiscovery_folders_are_not_named(
        self, llm_provider: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """They are read from wherever domyn starts: most folders have neither."""
        monkeypatch.chdir(tmp_path)

        agent = _agent(llm_provider, [Path(p) for p in AUTO])

        assert _missing_skills_warning(agent) is None

    def test_a_plain_skills_entry_is_an_ordinary_path(
        self, llm_provider: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        warning = _missing_skills_warning(_agent(llm_provider, [Path("skills")]))

        assert warning is not None
        assert "Skill not found: skills" in warning

    def test_a_symlinked_skills_folder_is_named_as_a_link_not_as_missing(
        self, llm_provider: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "elsewhere").mkdir()
        (tmp_path / ".domyn").mkdir()
        (tmp_path / ".domyn" / "skills").symlink_to(tmp_path / "elsewhere")
        monkeypatch.chdir(tmp_path)

        warning = _missing_skills_warning(_agent(llm_provider, [Path(".domyn/skills")]))

        assert warning is not None
        assert "symbolic link (links are not followed): .domyn/skills" in warning
        assert "Skill not found" not in warning

    def test_a_symlinked_parent_folder_is_named_as_a_link(
        self, llm_provider: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "elsewhere" / "skills").mkdir(parents=True)
        (tmp_path / ".agents").symlink_to(tmp_path / "elsewhere")
        monkeypatch.chdir(tmp_path)

        warning = _missing_skills_warning(_agent(llm_provider, [Path(".agents/skills")]))

        assert warning is not None
        assert "symbolic link (links are not followed): .agents/skills" in warning

    def test_a_broken_link_is_named_once_as_a_link(self, llm_provider: Any, tmp_path: Path) -> None:
        link = tmp_path / "dangling"
        link.symlink_to(tmp_path / "nothing-here")

        warning = _missing_skills_warning(_agent(llm_provider, [link]))

        assert warning is not None
        assert f"symbolic link (links are not followed): {link}" in warning
        assert "Skill not found" not in warning

    def test_a_symlinked_path_listed_by_hand_is_named_too(
        self, llm_provider: Any, tmp_path: Path
    ) -> None:
        _write(tmp_path / "real" / "review", "review")
        (tmp_path / "review").symlink_to(tmp_path / "real" / "review")

        warning = _missing_skills_warning(_agent(llm_provider, [tmp_path / "review"]))

        assert warning is not None
        assert "symbolic link" in warning and str(tmp_path / "review") in warning

    def test_an_agent_with_no_skills_prints_nothing(self, llm_provider: Any) -> None:
        assert _missing_skills_warning(_agent(llm_provider)) is None

    def test_another_kind_of_source_is_not_checked(self, llm_provider: Any) -> None:
        agent = Agent(
            name="A",
            description="d",
            instruction="i",
            llm_provider=llm_provider,
            skills=SkillsRuntime(source=InMemorySkillSource(skills={})),
        )

        assert _missing_skills_warning(agent) is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("show_status", [True, False])
    async def test_the_session_prints_it_with_or_without_the_banner(
        self,
        llm_provider: Any,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        theme: Theme,
        show_status: bool,
    ) -> None:
        import domyn.agents.cli.commands.run.command as run_module

        class _Repl:
            def __init__(self, *args: Any, **kwargs: Any) -> None: ...

            async def run(self, initial_prompt: str | None) -> int:
                return 0

        buffer = io.StringIO()
        monkeypatch.setattr(run_module, "console", Console(file=buffer, width=120))
        monkeypatch.setattr(run_module, "Repl", _Repl)

        await run_module.start_session(
            _agent(llm_provider, [tmp_path / "missing-skill"]),
            theme,
            initial_prompt=None,
            verbose=False,
            show_status=show_status,
        )

        assert "Skill not found" in buffer.getvalue()
        assert buffer.getvalue().count("Skill not found") == 1


def _skill_with_external_files(root: Path, name: str, entries: list[str]) -> None:
    listed = "".join(f"  - {e}\n" for e in entries)
    (root / name).mkdir(parents=True, exist_ok=True)
    (root / name / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: d\nexternal_files:\n{listed}---\nbody\n"
    )


class TestExternalFilesAtStartup:
    @staticmethod
    def _listed(llm_provider: Any, skill_dir: Path) -> Agent:
        return _agent(llm_provider, [skill_dir])

    @pytest.mark.asyncio
    async def test_a_listed_skill_with_a_missing_file_is_named(
        self, llm_provider: Any, tmp_path: Path
    ) -> None:
        _skill_with_external_files(tmp_path, "report", ["../shared/helper.py", "scripts/"])
        agent = self._listed(llm_provider, tmp_path / "report")

        warning = _missing_skills_warning(agent, await _startup_index(agent))

        assert warning is not None
        assert "Skill 'report' needs files that are missing" in warning
        assert "../shared/helper.py (does not exist)" in warning
        assert "scripts/ (does not exist)" in warning

    @pytest.mark.asyncio
    async def test_a_skill_with_every_file_there_is_silent(
        self, llm_provider: Any, tmp_path: Path
    ) -> None:
        _skill_with_external_files(tmp_path, "report", ["scripts/"])
        (tmp_path / "report" / "scripts").mkdir()
        agent = self._listed(llm_provider, tmp_path / "report")

        assert _missing_skills_warning(agent, await _startup_index(agent)) is None

    @pytest.mark.asyncio
    async def test_a_missing_path_and_a_missing_file_are_both_said(
        self, llm_provider: Any, tmp_path: Path
    ) -> None:
        _skill_with_external_files(tmp_path, "report", ["gone.py"])
        agent = _agent(llm_provider, [tmp_path / "report", tmp_path / "removed"])

        warning = _missing_skills_warning(agent, await _startup_index(agent))

        assert warning is not None
        assert "Skill not found" in warning and "removed" in warning
        assert "Skill 'report' needs files that are missing" in warning

    @pytest.mark.asyncio
    async def test_without_an_index_only_the_paths_are_checked(
        self, llm_provider: Any, tmp_path: Path
    ) -> None:
        _skill_with_external_files(tmp_path, "report", ["gone.py"])

        assert _missing_skills_warning(self._listed(llm_provider, tmp_path / "report")) is None

    @pytest.mark.asyncio
    async def test_a_long_list_of_broken_skills_is_cut(
        self, llm_provider: Any, tmp_path: Path
    ) -> None:
        for n in range(8):
            _skill_with_external_files(tmp_path, f"s{n}", ["gone.py"])
        agent = _agent(llm_provider, [tmp_path / f"s{n}" for n in range(8)])

        warning = _missing_skills_warning(agent, await _startup_index(agent))

        assert warning is not None
        assert warning.count("needs files that are missing") == 5
        assert "and 3 more skills with missing files" in warning

    @pytest.mark.asyncio
    async def test_an_agent_with_no_skills_has_no_index(self, llm_provider: Any) -> None:
        assert await _startup_index(_agent(llm_provider)) is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("show_status", [True, False])
    async def test_the_session_says_it_and_reads_each_skill_file_once(
        self,
        llm_provider: Any,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        theme: Theme,
        show_status: bool,
    ) -> None:
        import domyn.agents.cli.commands.run.command as run_module
        from domyn.agents.skills import sources as sources_module

        _skill_with_external_files(tmp_path, "report", ["gone.py"])
        parses: list[Path] = []
        original = sources_module.parse_skill_file

        def counting(path: Path, **kwargs: Any) -> Any:
            parses.append(path)
            return original(path, **kwargs)

        monkeypatch.setattr(sources_module, "parse_skill_file", counting)

        class _Repl:
            def __init__(self, *args: Any, **kwargs: Any) -> None: ...

            async def run(self, initial_prompt: str | None) -> int:
                return 0

        buffer = io.StringIO()
        monkeypatch.setattr(run_module, "console", Console(file=buffer, width=140))
        monkeypatch.setattr(run_module, "Repl", _Repl)

        await run_module.start_session(
            _agent(llm_provider, [tmp_path / "report"]),
            theme,
            initial_prompt=None,
            verbose=False,
            show_status=show_status,
        )

        assert "Skill 'report' needs files that are missing" in buffer.getvalue()
        assert len(parses) == 1
