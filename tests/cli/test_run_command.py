"""Tests for domyn/agents/cli/commands/run/command.py"""

import asyncio
import io
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from domyn.agents.agents import Agent
from domyn.agents.cli.commands.run.command import (
    build_skills_runtime,
    collect_all_agents,
    find_main_agent,
    find_root_agents,
    load_agent_from_yaml,
    load_module_from_path,
)
from domyn.agents.core import (
    ApprovalType,
    BaseEvent,
    ExecutionEventType,
    FeedbackAction,
)
from domyn.agents.interfaces.cli import Theme
from domyn.agents.interfaces.cli.repl import Repl
from domyn.agents.llm import BaseLLMProvider

# ---------------------------------------------------------------------------
# load_module_from_path
# ---------------------------------------------------------------------------


class TestLoadModuleFromPath:
    def test_raises_file_not_found_for_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="File not found"):
            load_module_from_path(str(tmp_path / "nonexistent.py"))

    def test_raises_value_error_for_non_py_extension(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("agents: []")
        with pytest.raises(ValueError, match=r"\.py"):
            load_module_from_path(str(yaml_file))

    def test_loads_module_and_exposes_variables(self, tmp_path: Path) -> None:
        py_file = tmp_path / "my_module.py"
        py_file.write_text("ANSWER = 42\nMESSAGE = 'hello'\n")
        module = load_module_from_path(str(py_file))
        assert module.ANSWER == 42
        assert module.MESSAGE == "hello"

    def test_loaded_module_is_executable_python(self, tmp_path: Path) -> None:
        py_file = tmp_path / "compute.py"
        py_file.write_text("RESULT = 2 + 2 * 5\n")
        module = load_module_from_path(str(py_file))
        assert module.RESULT == 12


# ---------------------------------------------------------------------------
# collect_all_agents
# ---------------------------------------------------------------------------


class TestCollectAllAgents:
    def test_finds_agent_instances_in_module(self, agent: Agent) -> None:
        mock_module = types.ModuleType("test_module")
        mock_module.__file__ = "/project/agent.py"
        mock_module.my_agent = agent
        mock_module.some_string = "not an agent"
        mock_module.some_number = 42

        agents = collect_all_agents(mock_module)

        assert "my_agent" in agents
        assert agents["my_agent"] is agent

    def test_excludes_non_agent_attributes(self, agent: Agent) -> None:
        mock_module = types.ModuleType("test_module")
        mock_module.__file__ = "/project/agent.py"
        mock_module.main_agent = agent
        mock_module.not_an_agent = "string"

        agents = collect_all_agents(mock_module)

        assert len(agents) == 1
        assert "not_an_agent" not in agents

    def test_returns_empty_dict_when_no_agents(self):
        mock_module = types.ModuleType("test_module")
        mock_module.__file__ = "/project/tools.py"
        mock_module.some_function = lambda: None

        agents = collect_all_agents(mock_module)

        assert agents == {}

    def test_uses_lowercase_keys(self, agent: Agent) -> None:
        mock_module = types.ModuleType("test_module")
        mock_module.__file__ = "/project/agent.py"
        mock_module.MyAgent = agent

        agents = collect_all_agents(mock_module)

        assert "myagent" in agents


# ---------------------------------------------------------------------------
# find_root_agents
# ---------------------------------------------------------------------------


class TestFindRootAgents:
    def test_all_agents_are_root_when_no_sub_agents(self, agent: Agent, child_agent: Agent) -> None:
        agents = {"agent": agent, "child": child_agent}
        roots = find_root_agents(agents)
        assert "agent" in roots
        assert "child" in roots

    def test_excludes_child_agents(self, agent: Agent, child_agent: Agent) -> None:
        agent.sub_agents = [child_agent]
        agents = {"parent": agent, "child": child_agent}

        roots = find_root_agents(agents)

        assert "parent" in roots
        assert "child" not in roots

    def test_returns_empty_when_all_are_sub_agents(self, agent: Agent, child_agent: Agent) -> None:
        # Circular: agent contains child, child contains agent
        # This is unusual but the function should handle it gracefully
        agent.sub_agents = [child_agent]
        child_agent.sub_agents = [agent]
        agents = {"agent": agent, "child": child_agent}

        roots = find_root_agents(agents)
        assert len(roots) == 0

    def test_returns_empty_dict_for_empty_input(self):
        assert find_root_agents({}) == {}


# ---------------------------------------------------------------------------
# find_main_agent
# ---------------------------------------------------------------------------


class TestFindMainAgent:
    def test_raises_when_no_agents(self):
        with pytest.raises(ValueError, match="No Agent instances found"):
            find_main_agent({})

    def test_returns_agent_by_exact_name(self, agent: Agent, child_agent: Agent) -> None:
        agents = {"testagent": agent, "childagent": child_agent}
        result = find_main_agent(agents, agent_name="testagent")
        assert result is agent

    def test_name_lookup_is_case_insensitive(self, agent: Agent) -> None:
        agents = {"testagent": agent}
        result = find_main_agent(agents, agent_name="TestAgent")
        assert result is agent

    def test_raises_when_named_agent_not_found(self, agent: Agent) -> None:
        agents = {"testagent": agent}
        with pytest.raises(ValueError, match="not found"):
            find_main_agent(agents, agent_name="unknown_agent")

    def test_auto_detects_single_root(self, agent: Agent, child_agent: Agent) -> None:
        agent.sub_agents = [child_agent]
        agents = {"parent": agent, "child": child_agent}

        result = find_main_agent(agents)

        assert result is agent

    def test_auto_detects_single_agent_as_root(self, agent: Agent):
        agents = {"testagent": agent}
        result = find_main_agent(agents)
        assert result is agent

    def test_fallback_to_name_keyword_when_no_roots(self, agent: Agent, child_agent: Agent) -> None:
        # Both are sub-agents of each other — no root agents
        agent.sub_agents = [child_agent]
        child_agent.sub_agents = [agent]
        # Rename one to match keyword
        main_named = Agent.__new__(Agent)
        object.__setattr__(main_named, "name", "main_service")
        agents_dict = {"main_service": agent, "child_loop": child_agent}
        # Patch find_root_agents to return empty to force fallback
        with patch(
            "domyn.agents.cli.commands.run.command.find_root_agents",
            return_value={},
        ):
            result = find_main_agent(agents_dict)
        # Falls back to keyword match — "main_service" contains "main"
        assert result is agent

    def test_fallback_to_first_agent_when_no_roots_and_no_keywords(
        self, agent: Agent, child_agent: Agent
    ) -> None:
        agents = {"some_agent": agent, "other_agent": child_agent}
        with patch(
            "domyn.agents.cli.commands.run.command.find_root_agents",
            return_value={},
        ):
            result = find_main_agent(agents)
        assert result is agent  # first in dict


# ---------------------------------------------------------------------------
# load_agent_from_yaml
# ---------------------------------------------------------------------------


class TestLoadAgentFromYaml:
    @pytest.mark.asyncio
    async def test_propagates_loader_exceptions(self):
        with (
            patch(
                "domyn.agents.loader.load_application_from_file",
                new_callable=AsyncMock,
                side_effect=FileNotFoundError("config not found"),
            ),
            pytest.raises(FileNotFoundError, match="config not found"),
        ):
            await load_agent_from_yaml("/path/to/missing.yaml")


# ---------------------------------------------------------------------------
# Repl — interactive loop
# ---------------------------------------------------------------------------


def _make_mock_runner(*events: BaseEvent) -> MagicMock:
    """Return a stand-in Runner that yields the given events then stops."""

    async def mock_run(*args, **kwargs):
        for event in events:
            yield event

    runner_instance = MagicMock()
    runner_instance.run = mock_run
    return runner_instance


def _make_repl(
    agent: Agent, runner: MagicMock, inputs: list[str], console: Console | None = None
) -> Repl:
    """Build a Repl whose user input is scripted, with no terminal involved."""
    repl = Repl(
        agent,
        console or MagicMock(),
        Theme.build(allow_animation=False),
        runner=runner,
        enable_enhanced_input=False,
    )
    queue = list(inputs)

    async def ask(label: str = "you") -> str:
        # Async because the real one is: the REPL awaits it from inside its
        # event loop, where prompt_toolkit's synchronous entry point cannot run.
        if not queue:
            raise EOFError
        return queue.pop(0)

    repl._input.ask = ask  # type: ignore[method-assign]
    return repl


class TestRepl:
    """The loop must terminate on every path and survive every event type.

    The turn is driven to completion *before* the next user message is read, so a
    turn that ends without a RESPONSE (an approval pause, an error) cannot leave
    the loop spinning on stale input — which is what the previous structure did.
    """

    @pytest.mark.asyncio
    async def test_exits_on_slash_exit(self, agent: Agent, response_event: BaseEvent) -> None:
        repl = _make_repl(agent, _make_mock_runner(response_event), ["/exit"])
        assert await repl.run() == 0

    @pytest.mark.asyncio
    async def test_exits_on_eof(self, agent: Agent, response_event: BaseEvent) -> None:
        repl = _make_repl(agent, _make_mock_runner(response_event), [])
        assert await repl.run() == 0

    @pytest.mark.asyncio
    async def test_runs_initial_prompt_then_exits(
        self, agent: Agent, response_event: BaseEvent
    ) -> None:
        repl = _make_repl(agent, _make_mock_runner(response_event), [])
        assert await repl.run("Hello") == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "event_name",
        ["response_event", "tool_start_event", "tool_end_event"],
    )
    async def test_handles_event_types(
        self, agent: Agent, event_name: str, request: pytest.FixtureRequest
    ) -> None:
        event = request.getfixturevalue(event_name)
        repl = _make_repl(agent, _make_mock_runner(event), [])
        assert await repl.run("go") == 0

    @pytest.mark.asyncio
    async def test_unknown_slash_command_does_not_end_session(
        self, agent: Agent, response_event: BaseEvent
    ) -> None:
        repl = _make_repl(agent, _make_mock_runner(response_event), ["/nope", "/exit"])
        assert await repl.run() == 0

    @pytest.mark.asyncio
    async def test_clear_starts_new_conversation(
        self, agent: Agent, response_event: BaseEvent
    ) -> None:
        repl = _make_repl(agent, _make_mock_runner(response_event), ["/clear", "/exit"])
        before = repl._state.conversation_id
        await repl.run()
        assert repl._state.conversation_id != before

    @pytest.mark.asyncio
    async def test_clear_wipes_the_screen_and_the_scrollback(
        self, agent: Agent, response_event: BaseEvent
    ) -> None:
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=True)
        repl = _make_repl(agent, _make_mock_runner(response_event), ["/clear", "/exit"], console)
        await repl.run()
        written = buffer.getvalue()
        assert "\x1b[2J" in written
        assert written.index("\x1b[3J") > written.index("\x1b[2J")

    @pytest.mark.asyncio
    async def test_clear_writes_no_escapes_off_a_terminal(
        self, agent: Agent, response_event: BaseEvent
    ) -> None:
        buffer = io.StringIO()
        console = Console(file=buffer)
        repl = _make_repl(agent, _make_mock_runner(response_event), ["/clear", "/exit"], console)
        await repl.run()
        assert "\x1b" not in buffer.getvalue()

    @pytest.mark.asyncio
    async def test_approval_pause_is_resumed_not_looped(self, agent: Agent) -> None:
        """A turn suspended on TOOL_NEEDS_FEEDBACK resumes with TOOL_FEEDBACK.

        The runner is called twice: once for the turn that suspends, once for the
        resume. Without the fix the second call would repeat the *original* user
        input instead.
        """
        needs_feedback = BaseEvent(
            event_type=ExecutionEventType.TOOL_NEEDS_FEEDBACK,
            author="agent",
            action=FeedbackAction(
                name="dangerous_tool",
                approval_type=ApprovalType.USER_CONFIRM,
                parameters=[],
            ),
        )
        calls: list[BaseEvent] = []

        async def mock_run(*args, **kwargs):
            calls.append(kwargs["user_input"])
            if len(calls) == 1:
                yield needs_feedback

        runner_instance = MagicMock()
        runner_instance.run = mock_run

        repl = _make_repl(agent, runner_instance, [])
        # Denial is what an aborted prompt means, and it must not hang the turn.
        with patch("domyn.agents.interfaces.cli.repl.prompt_for_approval", side_effect=EOFError):
            await repl.run("do something risky")

        assert len(calls) == 2
        assert calls[0].event_type is ExecutionEventType.USER_INPUT
        assert calls[1].event_type is ExecutionEventType.TOOL_FEEDBACK
        assert calls[1].action.approve is False


# ---------------------------------------------------------------------------
# build_skills_runtime
# ---------------------------------------------------------------------------


class TestBuildSkillsRuntime:
    def test_returns_none_without_skill_dirs(self, tmp_path: Path, monkeypatch) -> None:
        """No skills configured must mean no behaviour change at all, and a `./skills`
        folder next to it is not a skill directory unless something names it."""
        (tmp_path / "skills").mkdir()
        monkeypatch.chdir(tmp_path)

        assert build_skills_runtime(None, expand_all=False) is None
        assert build_skills_runtime([], expand_all=False) is None

    def test_builds_runtime_for_existing_dir(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        skills.mkdir()
        (skills / "a.md").write_text("---\nname: a\ndescription: d\n---\nbody\n")
        runtime = build_skills_runtime([skills], expand_all=False)
        assert runtime is not None
        assert "a.md" not in runtime.source.describe()

    def test_expand_all_uses_all_selector(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        skills.mkdir()
        runtime = build_skills_runtime([skills], expand_all=True)
        assert runtime is not None
        assert runtime.selector.type == "all"

    def test_sections_are_left_out_of_the_index_unless_asked_for(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        skills.mkdir()
        default = build_skills_runtime([skills], expand_all=False)
        asked = build_skills_runtime([skills], expand_all=False, index_sections=True)
        assert default is not None and asked is not None
        assert default.index_sections is False
        assert asked.index_sections is True


# ---------------------------------------------------------------------------
# run_command: the agent's own skills, extended by the CLI
# ---------------------------------------------------------------------------

SKILL = "---\nname: {name}\ndescription: d\n---\nbody of {name}\n"


class TestRunCommandExtendsTheAgentsSkills:
    """`run_command` extends the agent's runtime instead of replacing it.

    Driven by calling `run_command` directly with every argument explicit; the
    agent is handed in through a fake `resolve_agent_from_file`, and a fake
    `start_session` records what the session would have run with.
    """

    def _run(
        self,
        agent: Agent,
        monkeypatch: pytest.MonkeyPatch,
        *,
        expand_skills: bool = False,
        skill_sections: bool = False,
    ) -> Agent:
        import typer

        from domyn.agents.cli.commands.run import command

        async def _resolve(*args: object, **kwargs: object) -> tuple[Agent, dict]:
            return agent, {}

        started: list[Agent] = []

        async def _start(session_agent: Agent, *args: object, **kwargs: object) -> int:
            started.append(session_agent)
            return 0

        monkeypatch.setattr(command, "resolve_agent_from_file", _resolve)
        monkeypatch.setattr(command, "start_session", _start)

        with pytest.raises(typer.Exit):
            command.run_command(
                file=None,
                prompt=None,
                agent_name=None,
                expand_skills=expand_skills,
                skill_sections=skill_sections,
                verbose=False,
                no_banner=True,
            )
        assert started == [agent]
        return started[0]

    @staticmethod
    def _skills_dir(root: Path, name: str) -> Path:
        directory = root / name
        directory.mkdir()
        (directory / f"{name}.md").write_text(SKILL.format(name=name))
        return directory

    def test_a_local_dir_runtime_keeps_its_paths_and_its_config(
        self, llm_provider: BaseLLMProvider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from domyn.agents.skills import (
            IndexSelector,
            LocalDirSkillSource,
            SkillBudget,
            SkillsRuntime,
        )

        configured = self._skills_dir(tmp_path, "configured")
        runtime = SkillsRuntime(
            source=LocalDirSkillSource(paths=[configured]),
            selector=IndexSelector(),
            budget=SkillBudget(max_index_chars=123),
        )
        agent = Agent(
            name="A", description="d", instruction="i", llm_provider=llm_provider, skills=runtime
        )

        ran = self._run(agent, monkeypatch)

        assert ran.skills is not None
        assert ran.skills.source.paths == [configured]  # type: ignore[attr-defined]
        assert ran.skills.selector is runtime.selector
        assert ran.skills.budget.max_index_chars == 123
        # The agent's own runtime is copied, not mutated.
        assert runtime.source.paths == [configured]  # type: ignore[attr-defined]
        # And `skill_view` reads the runtime attached at startup, not the one the loader bound.
        view = [t for t in ran.tools if t.name == "skill_view"]
        assert len(view) == 1
        assert "body of configured" in asyncio.run(view[0]._actual_callable(name="configured"))  # type: ignore[attr-defined]

    def test_expand_skills_swaps_only_the_selector(
        self, llm_provider: BaseLLMProvider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from domyn.agents.skills import AllSkillsSelector, LocalDirSkillSource, SkillsRuntime

        configured = self._skills_dir(tmp_path, "configured")
        agent = Agent(
            name="A",
            description="d",
            instruction="i",
            llm_provider=llm_provider,
            skills=SkillsRuntime(source=LocalDirSkillSource(paths=[configured])),
        )

        ran = self._run(agent, monkeypatch, expand_skills=True)

        assert ran.skills is not None
        assert isinstance(ran.skills.selector, AllSkillsSelector)
        assert ran.skills.source.paths == [configured]  # type: ignore[attr-defined]

    def test_skill_sections_reaches_a_local_dir_runtime(
        self, llm_provider: BaseLLMProvider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from domyn.agents.skills import LocalDirSkillSource, SkillsRuntime

        configured = self._skills_dir(tmp_path, "configured")
        agent = Agent(
            name="A",
            description="d",
            instruction="i",
            llm_provider=llm_provider,
            skills=SkillsRuntime(source=LocalDirSkillSource(paths=[configured])),
        )

        ran = self._run(agent, monkeypatch, skill_sections=True)

        assert ran.skills is not None
        assert ran.skills.index_sections is True

    def test_a_skills_folder_in_the_project_is_not_scanned_at_run_time(
        self, agent: Agent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Finding skills is `domyn setup`'s job now: a `./skills` folder only matters
        once the config lists what is in it."""
        project = tmp_path / "project"
        project.mkdir()
        self._skills_dir(project, "skills")
        monkeypatch.chdir(project)

        ran = self._run(agent, monkeypatch)

        assert ran.skills is None

    def test_a_config_that_declares_skills_loads_exactly_those(
        self, llm_provider: BaseLLMProvider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from domyn.agents.skills import LocalDirSkillSource, SkillsRuntime

        installed = self._skills_dir(tmp_path, "installed")
        project = tmp_path / "project"
        project.mkdir()
        self._skills_dir(project, "skills")
        monkeypatch.chdir(project)
        agent = Agent(
            name="A",
            description="d",
            instruction="i",
            llm_provider=llm_provider,
            skills=SkillsRuntime(source=LocalDirSkillSource(paths=[installed])),
        )

        ran = self._run(agent, monkeypatch)

        assert ran.skills is not None
        assert ran.skills.source.paths == [installed]  # type: ignore[attr-defined]

    def test_another_source_without_the_flag_is_kept_as_is(
        self, llm_provider: BaseLLMProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: this path once left the runtime variable unassigned."""
        from domyn.agents.skills import InMemorySkillSource, SkillsRuntime

        runtime = SkillsRuntime(source=InMemorySkillSource(skills={"s": SKILL.format(name="s")}))
        agent = Agent(
            name="A", description="d", instruction="i", llm_provider=llm_provider, skills=runtime
        )

        ran = self._run(agent, monkeypatch)

        assert ran.skills is runtime


DELEGATE_CONFIG = """\
version: "1.0"
entrypoint: Remote
agents:
  - name: Remote
    type: delegate
    description: An agent run by a remote runner.
    transport:
      type: websocket
      url: ws://127.0.0.1:9/ws
"""


class TestRunCommandWithARootThatHasNoSkillsField:
    """`domyn run agent.yaml --expand-skills` with a `type: delegate` entrypoint.

    The catalogue reaches the model only through `agent.skills`, loaded by the
    runner and injected by the planner. A `DelegateAgent` has no such field, and
    its `tools` are never advertised to the remote runner — only looked up by
    name when the remote calls one. A `skill_view` attached to it is a tool no
    model is told about, for a catalogue no model is shown.

    Everything is real except `start_session`, which records the agent the
    session would have run with: the YAML goes through `resolve_agent_from_file`
    and the loader, then the same skills wiring as any other root.
    """

    def test_it_is_not_given_a_skill_view(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import typer

        from domyn.agents.agents import DelegateAgent
        from domyn.agents.cli.commands.run import command

        config = tmp_path / "agent.yaml"
        config.write_text(DELEGATE_CONFIG)

        started: list[object] = []

        async def _start(session_agent: object, *args: object, **kwargs: object) -> int:
            started.append(session_agent)
            return 0

        monkeypatch.setattr(command, "start_session", _start)

        with pytest.raises(typer.Exit):
            command.run_command(
                file=config,
                prompt=None,
                agent_name=None,
                expand_skills=True,
                skill_sections=False,
                verbose=False,
                no_banner=True,
            )

        assert len(started) == 1
        ran = started[0]
        assert isinstance(ran, DelegateAgent)
        assert not hasattr(ran, "skills")
        assert "skill_view" not in [t.name for t in ran.tools]
