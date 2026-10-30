"""The agent knowing where its skills are, and being able to open them.

The bug this comes from: with its skills in a folder outside the project, the agent
saw them in its index but went looking for the files in the current directory,
because nothing had ever told it where they were. Two separate gaps —
it was not told, and it could not have reached them anyway.
"""

# No `from __future__ import annotations`: builds real tools.
import asyncio
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from domyn.agents.interfaces.cli import config as cfg_module
from domyn.agents.interfaces.cli.config import (
    LOCAL_SKILLS_DIRS,
    builtin_skill_dirs,
    collect_skill_workspace_roots,
    read_raw_config,
    skill_workspace_roots,
    unique_paths,
)
from domyn.agents.loader import load_from_file
from domyn.agents.skills import (
    AllSkillsSelector,
    IndexSelector,
    LocalDirSkillSource,
    SkillBudget,
    SkillContent,
    SkillQuery,
    SkillsRuntime,
)
from domyn.agents.skills.integration import build_skill_view_tool
from domyn.agents.tools.builtin import BuiltinBundleConfig

# Captured at import, before the CLI conftest points it at a folder of the test's.
_REAL_SKILLS_INSTALL_DIR = cfg_module.skills_install_dir

SKILL = """---
name: user-journeys
description: How to write a user journey document
---
Start with the actor, then the goal.
"""


@pytest.fixture
def elsewhere(tmp_path: Path) -> Path:
    """A skills directory that is nowhere near the workspace."""
    directory = tmp_path / "somewhere-else" / "skills"
    (directory / "user-journeys").mkdir(parents=True)
    (directory / "user-journeys" / "SKILL.md").write_text(SKILL)
    return directory


def _tools(workspace: Path, *, skill_dirs: Iterable[Path] = ()) -> list[Any]:
    """Built-in file tools bound to ``workspace``, with ``skill_dirs`` as extra
    reachable roots — exactly what a `tool_bundles:` entry expands into."""
    bundle = BuiltinBundleConfig(name="test", bundles=["files"], workspace=str(workspace))
    return bundle.build(extra_roots=[str(d) for d in skill_dirs]).tools


class TestTheModelIsToldWhereTheyAre:
    """With the body, not in the index: there it was half of every line, paid on
    every request whether or not the skill was ever opened."""

    def test_skill_view_gives_the_file(self, elsewhere: Path) -> None:
        runtime = SkillsRuntime(source=LocalDirSkillSource(paths=[elsewhere]))
        view = build_skill_view_tool(runtime)._actual_callable

        result = asyncio.run(view(name="user-journeys"))

        assert f"[file: {elsewhere / 'user-journeys' / 'SKILL.md'}]" in result

    def test_an_expanded_skill_gives_the_file_too(self, elsewhere: Path) -> None:
        """Expanded up front, a skill never goes through ``skill_view``."""
        runtime = SkillsRuntime(
            source=LocalDirSkillSource(paths=[elsewhere]), selector=AllSkillsSelector()
        )
        selection, contents = asyncio.run(runtime.resolve(SkillQuery(task="write a journey")))

        payload = runtime.build_payload(selection, contents)

        assert str(elsewhere / "user-journeys" / "SKILL.md") in payload

    def test_the_index_does_not(self, elsewhere: Path) -> None:
        runtime = SkillsRuntime(
            source=LocalDirSkillSource(paths=[elsewhere]), selector=IndexSelector()
        )
        selection, contents = asyncio.run(runtime.resolve(SkillQuery(task="x")))

        payload = runtime.build_payload(selection, contents)

        assert "user-journeys: How to write a user journey document" in payload
        assert "SKILL.md" not in payload

    def test_it_is_told_the_file_can_be_edited(self, elsewhere: Path) -> None:
        """A path with no permission attached gets the agent asking the user to
        make the edit instead of making it."""
        runtime = SkillsRuntime(source=LocalDirSkillSource(paths=[elsewhere]))
        view = build_skill_view_tool(runtime)._actual_callable

        assert "edit" in asyncio.run(view(name="user-journeys")).lower()

    def test_a_skill_with_no_known_file_says_nothing_about_one(self) -> None:
        assert "file:" not in SkillContent(name="n", body="b").header()

    def test_the_path_is_not_guessable_from_the_name(self, tmp_path: Path) -> None:
        """Both layouts are supported, so the model cannot derive the path — which
        is why it has to be given."""
        flat = tmp_path / "skills"
        flat.mkdir()
        (flat / "flat-one.md").write_text(SKILL.replace("user-journeys", "flat-one"))
        runtime = SkillsRuntime(source=LocalDirSkillSource(paths=[flat]))
        view = build_skill_view_tool(runtime)._actual_callable

        assert str(flat / "flat-one.md") in asyncio.run(view(name="flat-one"))

    def test_a_section_keeps_the_file(self, tmp_path: Path) -> None:
        """A section is still read from that file, and its paths are relative to it."""
        directory = tmp_path / "skills"
        directory.mkdir()
        (directory / "sectioned.md").write_text(
            "---\nname: sectioned\ndescription: d\n---\n## Steps\nDo it.\n"
        )
        runtime = SkillsRuntime(source=LocalDirSkillSource(paths=[directory]))
        view = build_skill_view_tool(runtime)._actual_callable

        result = asyncio.run(view(name="sectioned", section="Steps"))

        assert "(section: Steps)" in result
        assert str(directory / "sectioned.md") in result


class TestTheAgentCanActuallyOpenThem:
    def test_a_skill_outside_the_workspace_is_readable(
        self, elsewhere: Path, tmp_path: Path
    ) -> None:
        workspace = tmp_path / "project"
        workspace.mkdir()
        tools = _tools(workspace, skill_dirs=[elsewhere])
        read = next(tool for tool in tools if tool.name == "read_file")

        result = asyncio.run(read.execute(path=str(elsewhere / "user-journeys" / "SKILL.md")))
        assert result.success is True, result.error
        assert "user-journeys" in str(result.data)

    def test_without_the_wiring_it_is_refused(self, elsewhere: Path, tmp_path: Path) -> None:
        """Pins the difference, so the wiring cannot quietly stop being applied."""
        workspace = tmp_path / "project"
        workspace.mkdir()
        tools = _tools(workspace)
        read = next(tool for tool in tools if tool.name == "read_file")

        result = asyncio.run(read.execute(path=str(elsewhere / "user-journeys" / "SKILL.md")))
        assert result.success is False

    def test_editing_one_still_needs_approval(self, elsewhere: Path, tmp_path: Path) -> None:
        """Reach was widened; the gate was not. This is what makes the wider reach
        acceptable."""
        from domyn.agents.core import ApprovalType

        workspace = tmp_path / "project"
        workspace.mkdir()
        tools = _tools(workspace, skill_dirs=[elsewhere])

        write = next(tool for tool in tools if tool.name == "write_file")
        assert write.approval_type is ApprovalType.USER_CONFIRM

    def test_the_rest_of_the_disk_is_still_out_of_reach(
        self, elsewhere: Path, tmp_path: Path
    ) -> None:
        workspace = tmp_path / "project"
        workspace.mkdir()
        secret = tmp_path / "secret.txt"
        secret.write_text("nope")

        tools = _tools(workspace, skill_dirs=[elsewhere])
        read = next(tool for tool in tools if tool.name == "read_file")
        assert asyncio.run(read.execute(path=str(secret))).success is False


class TestOneAnswerForWhereSkillsLive:
    """The two consumers must not be able to disagree: the source that indexes the
    skills, and the workspace that decides whether their files can be opened. When
    they did, the agent was shown skills it could not find."""

    def test_the_local_folders_are_named_in_one_place(self) -> None:
        assert (Path(".domyn/skills"), Path(".agents/skills")) == LOCAL_SKILLS_DIRS
        assert not hasattr(cfg_module, "default_skill_dirs")

    @pytest.mark.parametrize(
        ("entry", "expected"),
        [
            (".domyn/skills", True),
            ("./.agents/skills/", True),
            (Path(".agents/skills"), True),
            ("skills", False),
            (".domyn/skills/review", False),
            ("/abs/.domyn/skills", False),
        ],
    )
    def test_an_autodiscovery_entry_is_recognised_however_it_is_spelled(
        self, entry: str | Path, expected: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The CLI conftest points the folders elsewhere; the real value is what is tested.
        monkeypatch.setattr(cfg_module, "LOCAL_SKILLS_DIRS", LOCAL_SKILLS_DIRS)
        assert cfg_module.is_skill_autodiscovery(entry) is expected

    def test_every_configured_directory_is_used_not_only_the_first(self) -> None:
        """It used to take `skills_dirs[0]` and drop the rest, so a skill in the
        second directory was indexed by nobody. The order is the precedence: an earlier
        directory wins a name clash in `LocalDirSkillSource`."""
        resolved = unique_paths([Path("/one"), Path("/two")])
        assert resolved == [Path("/one"), Path("/two")]

    def test_one_directory_spelled_twice_is_listed_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`./skills` in a config and the absolute default are the same directory:
        walked twice per turn otherwise. The first spelling is kept."""
        (tmp_path / "skills").mkdir()
        monkeypatch.chdir(tmp_path)
        resolved = unique_paths([Path("skills"), tmp_path / "skills"])
        assert resolved == [Path("skills")]

    def test_a_home_relative_spelling_does_not_hide_the_home_skills(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A config listing ``~/.domyn/skills`` names the default home directory.
        Deduplication compares expanded paths, but must not keep the ``~`` spelling
        as the survivor: `LocalDirSkillSource` never expands it, so it looks for a
        directory literally called ``~`` under the cwd and the home skills vanish."""
        home = tmp_path / "home"
        home_skills = home / ".domyn" / "skills"
        (home_skills / "user-journeys").mkdir(parents=True)
        (home_skills / "user-journeys" / "SKILL.md").write_text(SKILL)
        monkeypatch.setenv("HOME", str(home))
        project = tmp_path / "project"
        project.mkdir()
        monkeypatch.chdir(project)

        directories = unique_paths([Path("~/.domyn/skills")])

        runtime = SkillsRuntime(
            source=LocalDirSkillSource(paths=directories), selector=IndexSelector()
        )
        selection, _ = asyncio.run(runtime.resolve(SkillQuery(task="x")))
        assert "user-journeys" in [skill.name for skill in selection.index]

    @pytest.mark.asyncio
    async def test_the_shipped_skills_are_indexed(self) -> None:
        index = await LocalDirSkillSource(paths=builtin_skill_dirs()).index(SkillQuery(task=""))
        assert {
            "data-analysis",
            "tabular-classification-regression",
            "time-series-forecasting",
            "brainstorming",
            "communication",
            "charts",
            "interactive-charts",
            "domyn-cli",
            "domyn-docs",
        } <= {skill.name for skill in index.skills}

    @pytest.mark.asyncio
    async def test_the_shipped_skills_leave_room_in_the_index_budget(self) -> None:
        """Past the budget, every advisory skill is shown in a shorter form.

        The user's own skills share the same budget, so the builtins alone must
        leave most of it for theirs, or a handful of the user's skills would cost
        every skill its description.
        """
        index = await LocalDirSkillSource(paths=builtin_skill_dirs()).index(SkillQuery(task=""))
        used = sum(len(skill.index_line()) for skill in index.skills)
        assert used <= SkillBudget().max_index_chars * 0.4

    def test_the_runtime_and_the_tools_are_given_the_same_answer(
        self, elsewhere: Path, tmp_path: Path
    ) -> None:
        """The regression, end to end: what the index shows must be openable."""
        from domyn.agents.cli.commands.run.command import build_skills_runtime

        directories = unique_paths([elsewhere])
        runtime = build_skills_runtime(directories, expand_all=False)
        assert runtime is not None

        workspace = tmp_path / "project"
        workspace.mkdir()
        tools = _tools(workspace, skill_dirs=directories)
        read = next(tool for tool in tools if tool.name == "read_file")

        selection, _ = asyncio.run(runtime.resolve(SkillQuery(task="x")))
        indexed = selection.index[0].path
        assert indexed is not None
        assert asyncio.run(read.execute(path=str(indexed))).success is True

    def test_an_env_var_skill_path_is_openable(
        self, elsewhere: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The workspace roots are collected from the file as written, the skill
        source from the file with ``${VAR}`` filled in. When only the second was
        filled in, the index pointed at a folder the file tools refused."""
        monkeypatch.setenv("TEAM_SKILLS", str(elsewhere))
        config = tmp_path / "agent.yaml"
        config.write_text(
            "agents:\n"
            "  - name: A\n"
            "    skills:\n"
            "      source:\n"
            "        type: local_dir\n"
            "        paths: ['${TEAM_SKILLS}']\n"
        )
        raw = read_raw_config(config)
        assert raw is not None
        roots = unique_paths(collect_skill_workspace_roots(raw))

        paths = load_from_file(str(config))["agents"][0]["skills"]["source"]["paths"]
        runtime = SkillsRuntime(source=LocalDirSkillSource(paths=paths), selector=IndexSelector())
        selection, _ = asyncio.run(runtime.resolve(SkillQuery(task="x")))
        indexed = selection.index[0].path
        assert indexed is not None

        workspace = tmp_path / "project"
        workspace.mkdir()
        read = next(
            tool for tool in _tools(workspace, skill_dirs=roots) if tool.name == "read_file"
        )
        assert asyncio.run(read.execute(path=str(indexed))).success is True

    def test_started_from_home_autodiscovery_never_meets_the_install_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """From `$HOME`, `./.domyn/skills` is `~/.domyn/skills`: the folder setup rewrites
        from scratch must be another one, or it would be read as the user's own."""
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(cfg_module, "GLOBAL_DIR", home / ".domyn")
        monkeypatch.chdir(home)
        install = _REAL_SKILLS_INSTALL_DIR().resolve()
        # The premise: both live in `~/.domyn`.
        assert install.parent == (home / ".domyn").resolve()

        for folder in LOCAL_SKILLS_DIRS:
            discovered = folder.resolve()
            assert not discovered.is_relative_to(install)
            assert not install.is_relative_to(discovered)


class TestEveryAgentsSkillDirsAreCollected:
    """`collect_skill_workspace_roots`: what becomes reachable by the file tools."""

    def test_every_agents_local_dirs_in_order(self) -> None:
        raw = {
            "agents": [
                {"name": "A", "skills": {"source": {"type": "local_dir", "paths": ["/a"]}}},
                {"name": "B"},
                {"name": "C", "skills": {"source": {"type": "local_dir", "paths": ["/c1", "/c2"]}}},
            ]
        }
        assert collect_skill_workspace_roots(raw) == [Path("/a"), Path("/c1"), Path("/c2")]

    def test_other_sources_and_malformed_entries_contribute_nothing(self) -> None:
        """Only directories known to hold skills may widen the workspace."""
        raw = {
            "agents": [
                {"name": "A", "skills": {"source": {"type": "memory", "paths": ["/nope"]}}},
                {"name": "B", "skills": "not a mapping"},
                "not an agent",
                {"name": "C", "skills": {"source": {"type": "local_dir"}}},
            ]
        }
        assert collect_skill_workspace_roots(raw) == []

    def test_env_var_paths_are_filled_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Filled in the same way the loader fills in the skill source's paths."""
        monkeypatch.setenv("TEAM_SKILLS", "/srv/skills")
        raw = {
            "agents": [
                {
                    "name": "A",
                    "skills": {
                        "source": {"type": "local_dir", "paths": ["${TEAM_SKILLS}", "/plain"]}
                    },
                },
            ]
        }
        assert collect_skill_workspace_roots(raw) == [Path("/srv/skills"), Path("/plain")]

    def test_an_unset_var_is_skipped_not_raised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reporting it is the loader's job, and the loader runs right after."""
        monkeypatch.delenv("TEAM_SKILLS", raising=False)
        raw = {
            "agents": [
                {
                    "name": "A",
                    "skills": {"source": {"type": "local_dir", "paths": ["${TEAM_SKILLS}"]}},
                },
            ]
        }
        assert collect_skill_workspace_roots(raw) == []


class TestWhatTheFileToolsReachForInstalledSkills:
    """`domyn setup` installs skills under one directory, and they share files that sit
    in the folders above them. The model looked for one of those with the file tools
    and was refused: only each skill's own folder was a root."""

    @staticmethod
    def _raw(*paths: Path) -> dict[str, Any]:
        return {
            "agents": [
                {
                    "name": "A",
                    "skills": {"source": {"type": "local_dir", "paths": [str(p) for p in paths]}},
                }
            ]
        }

    @pytest.fixture
    def installed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, list[Path]]:
        from domyn.agents.interfaces.cli import config as cli_config
        from domyn.agents.skills.catalog import install_skills

        install = tmp_path / "home" / "skills"
        monkeypatch.setattr(cli_config, "skills_install_dir", lambda: install)
        return install, install_skills(["charts", "data-analysis"], install)

    def test_the_install_directory_is_added_when_a_listed_skill_is_inside_it(
        self, installed: tuple[Path, list[Path]]
    ) -> None:
        install, paths = installed

        assert skill_workspace_roots(self._raw(*paths)) == [*paths, install]

    def test_it_is_not_added_for_skills_listed_elsewhere(
        self, installed: tuple[Path, list[Path]], tmp_path: Path
    ) -> None:
        elsewhere = tmp_path / "team-skills"

        assert skill_workspace_roots(self._raw(elsewhere)) == [elsewhere]

    def test_it_is_not_added_for_a_config_with_no_skills(
        self, installed: tuple[Path, list[Path]]
    ) -> None:
        assert skill_workspace_roots({"agents": [{"name": "A"}]}) == []

    def test_loading_still_never_sees_the_install_directory(
        self, installed: tuple[Path, list[Path]]
    ) -> None:
        """Only the file tools get the wider root; the skills loaded are the listed ones."""
        _, paths = installed

        assert collect_skill_workspace_roots(self._raw(*paths)) == paths

    def test_the_shared_files_can_be_found_and_read_with_the_file_tools(
        self, installed: tuple[Path, list[Path]], tmp_path: Path
    ) -> None:
        """The case from the session: check that the shared `seaborn_style.py` exists,
        by its absolute folder and a relative pattern, then read it."""
        install, paths = installed
        project = tmp_path / "project"
        project.mkdir()
        tools = _tools(project, skill_dirs=skill_workspace_roots(self._raw(*paths)))
        glob = next(tool for tool in tools if tool.name == "glob_files")
        read = next(tool for tool in tools if tool.name == "read_file")
        listing = next(tool for tool in tools if tool.name == "list_dir")

        style = install / "resources" / "seaborn_style.py"
        found = asyncio.run(glob.execute(pattern="resources/seaborn_style.py", path=str(install)))
        shown = asyncio.run(listing.execute(path=str(style.parent)))
        text = asyncio.run(read.execute(path=str(style)))

        assert found.success is True and "seaborn_style.py" in str(found.data)
        assert shown.success is True and "seaborn_style.py" in str(shown.data)
        assert text.success is True and "seaborn" in str(text.data)

    def test_without_the_install_directory_the_shared_file_is_refused(
        self, installed: tuple[Path, list[Path]], tmp_path: Path
    ) -> None:
        """Pins the difference, so the wiring cannot quietly stop being applied."""
        install, paths = installed
        project = tmp_path / "project"
        project.mkdir()
        tools = _tools(project, skill_dirs=collect_skill_workspace_roots(self._raw(*paths)))
        read = next(tool for tool in tools if tool.name == "read_file")

        style = install / "resources" / "seaborn_style.py"
        assert style.is_file()
        assert asyncio.run(read.execute(path=str(style))).success is False

    def test_writing_there_still_needs_approval(
        self, installed: tuple[Path, list[Path]], tmp_path: Path
    ) -> None:
        from domyn.agents.core import ApprovalType

        _, paths = installed
        project = tmp_path / "project"
        project.mkdir()
        tools = _tools(project, skill_dirs=skill_workspace_roots(self._raw(*paths)))

        write = next(tool for tool in tools if tool.name == "write_file")
        assert write.approval_type is ApprovalType.USER_CONFIRM

    def test_the_rest_of_the_disk_is_still_out_of_reach(
        self, installed: tuple[Path, list[Path]], tmp_path: Path
    ) -> None:
        _, paths = installed
        project = tmp_path / "project"
        project.mkdir()
        secret = tmp_path / "home" / "credentials"
        secret.write_text("KEY=1")
        tools = _tools(project, skill_dirs=skill_workspace_roots(self._raw(*paths)))
        read = next(tool for tool in tools if tool.name == "read_file")

        assert asyncio.run(read.execute(path=str(secret))).success is False
