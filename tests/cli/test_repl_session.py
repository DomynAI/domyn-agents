"""End-to-end tests for the interactive loop.

These drive the real :class:`Repl` — real runner, real renderer, real input
session — with only the LLM and the terminal editor stubbed. That combination is
deliberate: every CLI bug that reached a user so far lived in the seam between
"the REPL is async" and "some terminal library is not", and a test that stubs the
REPL itself cannot see that seam.

The editor stub therefore exposes the *synchronous* prompt_toolkit entry point
too, and fails if anything reaches for it.
"""

import asyncio
import io
from typing import Any

import pytest
from pydantic import Field
from rich.console import Console

from domyn.agents import Agent
from domyn.agents.interfaces.cli import Theme
from domyn.agents.interfaces.cli.repl import Repl
from domyn.agents.llm import BaseLLMProvider, LLMMessage, LLMResponse
from domyn.agents.planner_strategy import ToolUsePlannerStrategy


class ScriptedProvider(BaseLLMProvider):
    """Replies with a fixed answer, so a turn completes without a network."""

    type: str = "scripted"
    model_name: str = "scripted"
    supported_structured_outputs: list[Any] | None = None
    answer: str = "Hello from the agent."

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        return LLMResponse(
            content=self.answer,
            generated_tokens=1,
            execution_time=0.0,
            usage={},
        )

    async def generate_streaming(self, messages: list[LLMMessage], **kwargs: Any) -> Any:
        raise NotImplementedError
        yield  # pragma: no cover


class ScriptedEditor:
    """A terminal editor that plays back a script of user messages.

    Raises ``EOFError`` when the script runs out, which is how a real session
    ends on Ctrl-D.
    """

    def __init__(self, *lines: str) -> None:
        self._lines = list(lines)
        self.sync_calls = 0

    def prompt(self, *args: Any, **kwargs: Any) -> str:
        self.sync_calls += 1
        raise AssertionError(
            "prompt_toolkit's synchronous entry point calls asyncio.run(), "
            "which cannot work inside the REPL's event loop"
        )

    async def prompt_async(self, *args: Any, **kwargs: Any) -> str:
        if not self._lines:
            raise EOFError
        return self._lines.pop(0)


def _agent(answer: str = "Hello from the agent.") -> Agent:
    return Agent(
        name="Assistant",
        description="test agent",
        instruction="Be brief.",
        llm_provider=ScriptedProvider(answer=answer),
        planner=ToolUsePlannerStrategy(),
    )


def _repl(
    *script: str,
    answer: str = "Hello from the agent.",
    config: dict[str, Any] | None = None,
    config_file: Any = None,
) -> tuple[Repl, io.StringIO]:
    agent = _agent(answer)
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=False)
    repl = Repl(
        agent,
        console,
        Theme.build(allow_animation=False),
        config=config,
        config_file=config_file,
        enable_enhanced_input=False,
    )
    repl._input._session = ScriptedEditor(*script)
    return repl, buffer


class TestASessionRuns:
    @pytest.mark.asyncio
    async def test_a_turn_completes_and_the_answer_is_shown(self) -> None:
        """The regression: this used to die on the first keystroke with
        "asyncio.run() cannot be called from a running event loop"."""
        repl, buffer = _repl("hello there", answer="Hi, I am the agent.")
        assert await repl.run() == 0
        assert "Hi, I am the agent." in buffer.getvalue()

    @pytest.mark.asyncio
    async def test_several_turns_run_in_sequence(self) -> None:
        repl, _ = _repl("first", "second", "third")
        assert await repl.run() == 0
        assert repl._state.turns == 3

    @pytest.mark.asyncio
    async def test_end_of_input_closes_the_session_cleanly(self) -> None:
        repl, buffer = _repl()
        assert await repl.run() == 0
        assert "Session closed." in buffer.getvalue()

    @pytest.mark.asyncio
    async def test_an_initial_prompt_runs_before_any_input_is_read(self) -> None:
        """``domyn run "..."`` starts with the prompt already in hand."""
        repl, buffer = _repl(answer="Answered the initial prompt.")
        assert await repl.run(initial_prompt="do the thing") == 0
        assert "Answered the initial prompt." in buffer.getvalue()

    @pytest.mark.asyncio
    async def test_blank_lines_are_ignored_rather_than_sent(self) -> None:
        repl, _ = _repl("", "   ", "real message")
        await repl.run()
        assert repl._state.turns == 1


class TestSlashCommands:
    @pytest.mark.asyncio
    async def test_exit_ends_the_session(self) -> None:
        repl, _ = _repl("/exit", "never reached")
        assert await repl.run() == 0
        assert repl._state.turns == 0

    @pytest.mark.asyncio
    async def test_help_is_shown_and_the_session_continues(self) -> None:
        repl, buffer = _repl("/help")
        await repl.run()
        assert "/help" in buffer.getvalue()

    @pytest.mark.asyncio
    async def test_help_shows_a_command_usage_verbatim(self) -> None:
        """`[revoke <n> | clear]` is a markup tag to a rich console with markup
        on, and rich deletes a tag it cannot resolve — so /help was hiding the
        usage of the one command that has one."""
        repl, buffer = _repl("/help")
        await repl.run()
        assert "/approvals [revoke <n> | clear]" in buffer.getvalue()

    @pytest.mark.asyncio
    async def test_a_slash_command_is_not_sent_to_the_agent(self) -> None:
        repl, _ = _repl("/help")
        await repl.run()
        assert repl._state.turns == 0

    @pytest.mark.asyncio
    async def test_an_unknown_command_does_not_end_the_session(self) -> None:
        repl, _ = _repl("/nonsense", "then a real message")
        await repl.run()
        assert repl._state.turns == 1

    @pytest.mark.asyncio
    async def test_an_unknown_skill_is_reported_and_the_session_continues(
        self, tmp_path: Any
    ) -> None:
        """The view raises when it cannot show a skill; the REPL prints the message."""
        from domyn.agents.skills import LocalDirSkillSource, SkillsRuntime
        from domyn.agents.skills.integration import attach_skills

        (tmp_path / "essay.md").write_text("---\nname: essay\ndescription: d\n---\nbody\n")
        repl, buffer = _repl("/skill zzz", "then a real message")
        attach_skills(repl._agent, SkillsRuntime(source=LocalDirSkillSource(paths=[tmp_path])))

        assert await repl.run() == 0
        assert "Skill not found: zzz." in buffer.getvalue()
        assert repl._state.turns == 1


class TestConfigFile:
    """`/config` shows the file backing the session as a redacted summary —
    its path, then a comprehensive but secret-free rundown built by
    `describe_configuration_v1` — the same way whether that's the saved
    default config or a file named explicitly on the command line.

    It never dumps the file's raw text: `config.yaml` is a full
    `ConfigurationV1` file, and nothing stops a hand-edited one from carrying
    a literal secret instead of an `${ENV_VAR}` reference."""

    @pytest.mark.asyncio
    async def test_it_prints_the_path_and_the_sections(self, tmp_path: Any) -> None:
        config_file = tmp_path / "agent.yaml"
        config_file.write_text(
            "version: '1.0'\n"
            "entrypoint: Assistant\n"
            "llm_providers:\n"
            "  - name: default_provider\n"
            "    type: openai\n"
            "    model_name: gpt-4o\n"
            "agents:\n"
            "  - name: Assistant\n"
            "    description: A test agent\n"
            "    instruction: Be helpful.\n"
            "    llm_provider: default_provider\n"
            "    planner_strategy: tool_use_thought\n",
            encoding="utf-8",
        )

        repl, buffer = _repl("/config", config_file=config_file)
        await repl.run()

        output = buffer.getvalue()
        assert str(config_file.resolve()) in output
        assert "gpt-4o" in output
        assert "Assistant" in output

    @pytest.mark.asyncio
    async def test_a_literal_secret_never_reaches_the_output(self, tmp_path: Any) -> None:
        config_file = tmp_path / "agent.yaml"
        config_file.write_text(
            "version: '1.0'\n"
            "entrypoint: Assistant\n"
            "llm_providers:\n"
            "  - name: default_provider\n"
            "    type: openai\n"
            "    model_name: gpt-4o\n"
            "    api_key: sk-super-secret-value\n"
            "agents:\n"
            "  - name: Assistant\n"
            "    description: A test agent\n"
            "    instruction: Be helpful.\n"
            "    llm_provider: default_provider\n"
            "    planner_strategy: tool_use_thought\n",
            encoding="utf-8",
        )

        repl, buffer = _repl("/config", config_file=config_file)
        await repl.run()

        output = buffer.getvalue()
        assert "sk-super-secret-value" not in output
        assert "hardcoded value" in output

    @pytest.mark.asyncio
    async def test_a_python_module_agent_has_no_file_to_show(self) -> None:
        repl, buffer = _repl("/config", config_file=None)
        await repl.run()

        assert "loaded from a Python module" in buffer.getvalue()


class TestChangeModelIsDefaultConfigOnly:
    """Regression coverage: `/model` used to be gated on `self._config is
    None`, which broke once an explicitly-run `.yaml` file also gets a
    `self._config`. The gate now checks the resolved path against
    `config_path()` instead."""

    @pytest.mark.asyncio
    async def test_it_is_a_noop_for_an_explicitly_named_file(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from domyn.agents.interfaces.cli import repl as repl_mod

        called = False

        def _fail_if_called(*args: Any, **kwargs: Any) -> None:
            nonlocal called
            called = True

        monkeypatch.setattr(repl_mod, "run_setup", _fail_if_called)

        other_file = tmp_path / "other-agent.yaml"
        other_file.write_text("entrypoint: Other\n", encoding="utf-8")
        repl, buffer = _repl("/model", config={"entrypoint": "Other"}, config_file=other_file)
        await repl.run()

        assert not called
        assert "only applies to the default configuration" in buffer.getvalue()

    @pytest.mark.asyncio
    async def test_it_proceeds_for_the_default_config(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from domyn.agents.interfaces.cli import config as cfg
        from domyn.agents.interfaces.cli import repl as repl_mod

        monkeypatch.setattr(cfg, "GLOBAL_DIR", tmp_path / "home" / ".domyn")
        monkeypatch.setattr(cfg, "PROJECT_DIR", tmp_path / "project" / ".domyn")

        called = False

        def _record_call(*args: Any, **kwargs: Any) -> None:
            nonlocal called
            called = True
            return None  # simulates the user aborting the wizard

        monkeypatch.setattr(repl_mod, "run_setup", _record_call)

        default_path = cfg.config_path()
        repl, buffer = _repl("/model", config={"entrypoint": "Assistant"}, config_file=default_path)
        await repl.run()

        assert called
        assert "only applies to the default configuration" not in buffer.getvalue()


class TestChangeModelKeepsTheSessionsSkills:
    """`/model` reloads the agent from `config.yaml`, which only has the config's own
    skill dirs. The runtime the session started with (an agent file may list others)
    must be carried over, and its dirs kept reachable by the file tools."""

    @pytest.mark.asyncio
    async def test_the_runtime_and_its_dirs_survive_the_reload(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from domyn.agents.interfaces.cli import config as cfg
        from domyn.agents.interfaces.cli import repl as repl_mod
        from domyn.agents.skills import LocalDirSkillSource, SkillsRuntime
        from domyn.agents.skills.integration import attach_skills

        monkeypatch.setattr(cfg, "GLOBAL_DIR", tmp_path / "home" / ".domyn")
        monkeypatch.setattr(cfg, "PROJECT_DIR", tmp_path / "project" / ".domyn")

        configured = tmp_path / "configured"
        from_file = tmp_path / "from-file"
        updated = cfg.configuration_v1_dict_from(
            cfg.ProviderConfig(type="openai", model_name="m2"),
            cfg.AgentConfig(skills={"source": {"type": "local_dir", "paths": [str(configured)]}}),
            cfg.CompactionConfig(),
        )
        monkeypatch.setattr(repl_mod, "run_setup", lambda *args, **kwargs: updated)

        reloaded = _agent()
        seen_roots: list[str] = []

        async def _load(path: str, *, extra_workspace_roots: Any = ()) -> Agent:
            seen_roots.extend(extra_workspace_roots)
            return reloaded

        monkeypatch.setattr(repl_mod, "load_application_from_file", _load)

        # What an agent file listing <from-file> left on the session's agent.
        runtime = SkillsRuntime(source=LocalDirSkillSource(paths=[from_file, configured]))
        repl, _ = _repl("/model", config=updated, config_file=cfg.config_path())
        attach_skills(repl._agent, runtime)

        await repl.run()

        assert repl._agent is reloaded
        assert reloaded.skills is runtime
        assert [t.name for t in reloaded.tools].count("skill_view") == 1
        assert sorted(seen_roots) == sorted([str(configured), str(from_file)])

    @pytest.mark.asyncio
    async def test_a_source_that_is_not_a_directory_survives_but_widens_nothing(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-`local_dir` source is carried over like any other runtime, but it
        has no directories, so it gives the file tools nothing new to reach."""
        from domyn.agents.interfaces.cli import config as cfg
        from domyn.agents.interfaces.cli import repl as repl_mod
        from domyn.agents.skills import InMemorySkillSource, SkillsRuntime
        from domyn.agents.skills.integration import attach_skills

        monkeypatch.setattr(cfg, "GLOBAL_DIR", tmp_path / "home" / ".domyn")
        monkeypatch.setattr(cfg, "PROJECT_DIR", tmp_path / "project" / ".domyn")

        skill = "---\nname: in-memory\ndescription: d\n---\nbody of in-memory\n"
        # The saved config declares the same in-memory source.
        updated = cfg.configuration_v1_dict_from(
            cfg.ProviderConfig(type="openai", model_name="m2"),
            cfg.AgentConfig(skills={"source": {"type": "memory", "skills": {"in-memory": skill}}}),
            cfg.CompactionConfig(),
        )
        monkeypatch.setattr(repl_mod, "run_setup", lambda *args, **kwargs: updated)

        reloaded = _agent()
        seen_roots: list[str] = []

        async def _load(path: str, *, extra_workspace_roots: Any = ()) -> Agent:
            seen_roots.extend(extra_workspace_roots)
            return reloaded

        monkeypatch.setattr(repl_mod, "load_application_from_file", _load)

        # What `domyn run` leaves for such an agent: its own runtime, untouched.
        runtime = SkillsRuntime(source=InMemorySkillSource(skills={"in-memory": skill}))
        repl, _ = _repl("/model", config=updated, config_file=cfg.config_path())
        attach_skills(repl._agent, runtime)

        await repl.run()

        # Carried over, and the tool reads it.
        assert repl._agent is reloaded
        assert reloaded.skills is runtime
        view = [t for t in reloaded.tools if t.name == "skill_view"]
        assert len(view) == 1
        assert "body of in-memory" in await view[0]._actual_callable(name="in-memory")  # type: ignore[attr-defined]
        # But the file tools are not given anything new to reach.
        assert seen_roots == []


class TestTheEditorIsDrivenAsynchronously:
    @pytest.mark.asyncio
    async def test_the_synchronous_entry_point_is_never_used(self) -> None:
        repl, _ = _repl("a message")
        await repl.run()
        assert repl._input._session.sync_calls == 0

    @pytest.mark.asyncio
    async def test_the_outer_loop_survives_the_whole_session(self) -> None:
        """A stray ``asyncio.run()`` inside the loop leaves the coroutine
        un-awaited and can close the running loop; check it is still ours."""
        loop = asyncio.get_running_loop()
        repl, _ = _repl("one", "two")
        await repl.run()
        assert asyncio.get_running_loop() is loop
        assert not loop.is_closed()


class _Hangs(BaseLLMProvider):
    """Never answers, so the turn is still running whenever the test acts."""

    type: str = "hangs"
    model_name: str = "hangs"
    supported_structured_outputs: list[Any] | None = None
    started: Any = None

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def generate_streaming(self, messages: list[LLMMessage], **kwargs: Any) -> Any:
        raise NotImplementedError
        yield  # pragma: no cover


class TestCtrlCOnARealEditor:
    """Ctrl-C typed into prompt_toolkit's key processor while the REPL runs a turn.

    Real keys rather than a scripted editor, because the bug this guards against
    lived between the editor and the event loop: the read alongside a turn runs in
    a task of its own, and asyncio re-raises a ``KeyboardInterrupt`` that ends a
    task out of the loop itself. The "Turn interrupted." the REPL had a handler for
    was never reached; the first Ctrl-C mid-turn took the whole CLI down instead.
    """

    @pytest.mark.asyncio
    async def test_one_press_stops_the_turn_and_a_second_ends_the_session(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        pytest.importorskip("prompt_toolkit")
        from prompt_toolkit.application import create_app_session
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        # On a dumb terminal prompt_toolkit swaps in a minimal prompt with its own
        # Ctrl-C; the shell running the tests must not pick which one is tested.
        monkeypatch.setenv("TERM", "xterm-256color")
        # Stopped, so the two presses pair however long the turn takes to unwind.
        monkeypatch.setattr("domyn.agents.interfaces.cli.prompt.monotonic", lambda: 1000.0)
        provider = _Hangs(started=asyncio.Event())
        agent = Agent(
            name="Assistant",
            description="test agent",
            instruction="Be brief.",
            llm_provider=provider,
            planner=ToolUsePlannerStrategy(),
        )
        buffer = io.StringIO()
        with create_pipe_input() as keys, create_app_session(input=keys, output=DummyOutput()):
            repl = Repl(
                agent,
                Console(file=buffer, width=100, force_terminal=True),
                Theme.build(allow_animation=False),
                enable_enhanced_input=False,
            )
            # By hand: the constructor builds the editor only on a real terminal.
            repl._input._setup_enhanced(tmp_path / "history")
            repl._reader.enabled = True

            running = asyncio.ensure_future(repl.run("start something slow"))
            await asyncio.wait_for(provider.started.wait(), 5)
            # Both at once, as a hurried user's would arrive: the second outlives the
            # read the first one ends, and prompt_toolkit hands it to the next read.
            keys.send_text("\x03\x03")
            assert await asyncio.wait_for(running, 5) == 0

        output = buffer.getvalue()
        assert "Turn interrupted." in output
        assert "Session closed." in output


def test_the_renderer_is_wired_to_the_editor_for_the_preview() -> None:
    """The renderer asks the editor to repaint as the answer streams, and reads
    from it how much printed text has reached the terminal: without either, the
    preview lags behind the answer or shows a line twice."""
    import io

    from rich.console import Console

    from domyn.agents import Agent
    from domyn.agents.interfaces.cli import Theme
    from domyn.agents.interfaces.cli.repl import Repl

    agent = Agent(
        name="Assistant", description="d", instruction="i", llm_provider=ScriptedProvider()
    )
    repl = Repl(agent, Console(file=io.StringIO()), Theme.build(), enable_enhanced_input=False)
    assert repl._reader._preview == repl._renderer.preview
    assert repl._renderer.output is repl._input.output
    assert repl._renderer.on_preview == repl._input.refresh
