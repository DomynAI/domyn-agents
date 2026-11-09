"""The memory question of the wizard, the file it writes, and the renderer line.

The halves that matter: what the wizard writes has to load through the real loader
into an agent that can remember, what it reads back has to be what it would write
again, and a memory somebody tuned by hand must survive a re-run.
"""

import io
import re
from unittest.mock import patch

import pytest
import yaml
from rich.console import Console

from domyn.agents.core import BaseEvent, ExecutionEventType, Part, ToolAction
from domyn.agents.interfaces.cli import config as cfg
from domyn.agents.interfaces.cli import select as select_mod
from domyn.agents.interfaces.cli import setup
from domyn.agents.interfaces.cli.config import (
    PROJECT_DIR_VAR,
    DecisionsChoice,
    MemoryChoice,
    MemoryDecider,
    configuration_v1_dict_from,
    current_memory_from,
    current_settings_from,
    describe_configuration_v1,
    memory_file_path,
)
from domyn.agents.interfaces.cli.renderer import EventRenderer
from domyn.agents.interfaces.cli.theme import Theme
from domyn.agents.loader import load_application_from_file
from domyn.agents.memory.managers import FilesystemMemoryManager, MemoryExecutionMode


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "GLOBAL_DIR", tmp_path / "home" / ".domyn")
    return tmp_path


def base() -> dict:
    return {
        "version": "1.0",
        "entrypoint": "Assistant",
        "llm_providers": [
            {
                "name": "default_provider",
                "type": "vllm",
                "model_name": "m",
                "url": "http://x/v1/chat/completions",
                "context_window": 200_000,
            }
        ],
        "agents": [
            {
                "name": "Assistant",
                "description": "d",
                "instruction": "i",
                "llm_provider": "default_provider",
            }
        ],
    }


def written(raw, memory):
    provider, agent, compaction = current_settings_from(raw)
    return configuration_v1_dict_from(provider, agent, compaction, raw, None, memory)


def manager_of(config):
    return config["agents"][0].get("memory_manager")


class TestWhatTheWizardWrites:
    def test_global_is_a_file_in_the_home_folder_and_agentic(self, tmp_path):
        manager = manager_of(written(base(), MemoryChoice(scope="global")))
        assert manager == {
            "type": "filesystem",
            "file_path": str(tmp_path / "home" / ".domyn" / "memory.md"),
            "execution_mode": "agentic",
        }

    def test_project_names_its_own_folder_the_way_a_project_does(self):
        manager = manager_of(written(base(), MemoryChoice(scope="project")))
        assert manager["file_path"] == f"${{{PROJECT_DIR_VAR}}}/memory.md"

    def test_no_writes_nothing(self):
        assert manager_of(written(base(), MemoryChoice(scope="no"))) is None

    def test_a_run_that_never_asked_leaves_the_file_as_it_was(self):
        raw = base()
        raw["agents"][0]["memory_manager"] = {"type": "filesystem", "file_path": "/x/m.md"}
        assert manager_of(written(raw, None)) == raw["agents"][0]["memory_manager"]

    def test_no_removes_what_was_there(self):
        raw = base()
        raw["agents"][0]["memory_manager"] = {"type": "filesystem", "file_path": "/x/m.md"}
        assert manager_of(written(raw, MemoryChoice(scope="no"))) is None

    def test_a_memory_tuned_by_hand_survives_when_the_choice_is_the_same(self):
        raw = written(base(), MemoryChoice(scope="global"))
        raw["agents"][0]["memory_manager"]["recency_limit"] = 7
        again = written(raw, MemoryChoice(scope="global"))
        assert manager_of(again)["recency_limit"] == 7

    def test_changing_the_scope_writes_the_new_path(self):
        raw = written(base(), MemoryChoice(scope="global"))
        again = written(raw, MemoryChoice(scope="project"))
        assert manager_of(again)["file_path"] == f"${{{PROJECT_DIR_VAR}}}/memory.md"

    def test_the_path_helper_agrees(self):
        assert memory_file_path("project") == f"${{{PROJECT_DIR_VAR}}}/memory.md"
        assert memory_file_path("global").endswith("memory.md")


class TestReadingItBack:
    @pytest.mark.parametrize("scope", ["global", "project"])
    def test_what_it_writes_is_what_it_reads(self, scope):
        assert current_memory_from(written(base(), MemoryChoice(scope=scope))).scope == scope

    def test_nothing_declared_reads_as_no(self):
        assert current_memory_from(base()).scope == "no"
        assert current_memory_from(None).scope == "no"

    def test_a_memory_written_by_hand_somewhere_else_reads_as_global(self):
        raw = base()
        raw["agents"][0]["memory_manager"] = {"type": "filesystem", "file_path": "/data/m.md"}
        assert current_memory_from(raw).scope == "global"


class TestTheFileTheWizardWroteActuallyLoads:
    """The half no projection test can reach: YAML on disk, through the real loader."""

    async def built(self, tmp_path, monkeypatch, scope):
        monkeypatch.setenv("VLLM_API_KEY", "not-a-real-key")
        monkeypatch.setenv("DOMYN_PROJECT_DIR", str(tmp_path / "project"))
        path = tmp_path / "config.yaml"
        path.write_text(
            yaml.safe_dump(written(base(), MemoryChoice(scope=scope)), sort_keys=False),
            encoding="utf-8",
        )
        return await load_application_from_file(str(path))

    @pytest.mark.asyncio
    async def test_global_gives_an_agent_that_can_remember_recall_and_forget(
        self, tmp_path, monkeypatch
    ):
        agent = await self.built(tmp_path, monkeypatch, "global")
        assert isinstance(agent.memory_manager, FilesystemMemoryManager)
        assert agent.memory_manager.execution_mode is MemoryExecutionMode.AGENTIC
        assert agent.memory_manager.file_path == str(tmp_path / "home" / ".domyn" / "memory.md")
        assert {"remember", "recall", "forget"} <= {t.name for t in agent.tools}

    @pytest.mark.asyncio
    async def test_project_keeps_it_in_the_projects_folder(self, tmp_path, monkeypatch):
        agent = await self.built(tmp_path, monkeypatch, "project")
        assert agent.memory_manager.file_path == str(tmp_path / "project" / "memory.md")


class TestWhatConfigShows:
    def test_the_memory_section_names_mode_scope_and_file(self):
        raw = written(base(), MemoryChoice(scope="project"))
        section = next(s for s in describe_configuration_v1(raw).sections if s.title == "Memory")
        fields = dict(section.items[0].fields)
        assert fields["mode"] == "agentic" and fields["scope"] == "project"
        assert fields["file"].endswith("memory.md")

    def test_no_memory_no_section(self):
        titles = [s.title for s in describe_configuration_v1(base()).sections]
        assert "Memory" not in titles


class TestTheQuestion:
    @pytest.fixture(autouse=True)
    def numbered(self, monkeypatch):
        monkeypatch.setattr(select_mod, "_interactive", lambda theme: False)

    def ask(self, current, answer, *, project=False, monkeypatch=None):
        console = Console(file=io.StringIO(), width=100, force_terminal=False)
        with (
            patch("builtins.input", side_effect=[answer]),
            patch.object(setup, "active_project", lambda: "p" if project else None),
        ):
            return setup._ask_memory(console, Theme.build(allow_animation=False), current)

    def test_enter_keeps_what_is_there(self):
        assert self.ask(MemoryChoice(scope="global"), "").scope == "global"
        assert self.ask(MemoryChoice(), "").scope == "no"

    def test_it_can_be_turned_on_and_off(self):
        assert self.ask(MemoryChoice(), "global").scope == "global"
        assert self.ask(MemoryChoice(scope="global"), "no").scope == "no"

    def test_the_projects_own_memory_is_offered_only_inside_a_project(self):
        assert self.ask(MemoryChoice(), "project", project=True).scope == "project"
        # Outside one, "project" is not a choice, so it is not accepted.
        with pytest.raises(Exception):  # noqa: B017
            self.ask(MemoryChoice(), "project", project=False)


class TestTheRemeberedLine:
    def renderer(self):
        buffer = io.StringIO()
        console = Console(file=buffer, width=100, force_terminal=False, no_color=True)
        renderer = EventRenderer(console, Theme.build(allow_animation=False))
        renderer.begin_turn()
        return renderer, buffer

    def start(self, call_id, **params):
        return BaseEvent(
            event_type=ExecutionEventType.TOOL_START,
            author="a",
            action=ToolAction(name="remember", parameters=params, call_id=call_id),
        )

    def end(self, call_id, result):
        return BaseEvent(
            event_type=ExecutionEventType.TOOL_END,
            author="a",
            action=ToolAction(name="remember", parameters={}, observation=result, call_id=call_id),
            content=[Part(text=result)],
        )

    def text(self, buffer):
        return re.sub(r"\x1b\[[0-9;]*m", "", buffer.getvalue())

    def test_it_says_what_was_remembered(self):
        renderer, buffer = self.renderer()
        renderer.render(self.start("c1", memory_content="likes tea"))
        renderer.render(self.end("c1", "Memory stored."))
        assert "remembered: likes tea" in self.text(buffer)
        assert "Memory stored." not in self.text(buffer)

    def test_a_repeat_or_a_refusal_is_not_called_remembered(self):
        renderer, buffer = self.renderer()
        renderer.render(self.start("c1", memory_content="likes tea"))
        renderer.render(self.end("c1", "Already remembered."))
        assert "remembered: likes tea" not in self.text(buffer)
        assert "Already remembered." in self.text(buffer)

    def hook_end(self, text, name="update_memory"):
        return BaseEvent(
            event_type=ExecutionEventType.HOOK_END,
            author="a",
            metadata={"hook_name": name},
            content=[Part(text=text)] if text else [],
        )

    def test_the_automatic_memory_is_shown_like_a_call(self):
        renderer, buffer = self.renderer()
        renderer.render(self.hook_end("The user likes tea"))
        assert "remembered The user likes tea" in self.text(buffer)

    def test_it_is_drawn_in_the_tool_calls_yellow(self, monkeypatch):
        """The same two styles as the agent's own `remember` call."""
        renderer, _ = self.renderer()
        printed = []
        renderer._console.print = lambda line, **kw: printed.append(line)
        # The role, whatever the terminal's colours.
        monkeypatch.setattr(Theme, "style", lambda self, name: name)
        renderer.render(self.start("c1", memory_content="likes tea"))
        renderer.render(self.hook_end("The user likes tea"))
        call, automatic = printed

        def styles(line):
            return [str(span.style) for span in line.spans]

        assert styles(automatic) == styles(call)
        assert styles(call) == ["tool_name", "tool_call"]

    def test_nothing_stored_says_nothing(self):
        renderer, buffer = self.renderer()
        renderer.render(self.hook_end(""))
        assert self.text(buffer).strip() == ""

    def test_another_hook_stays_quiet(self):
        renderer, buffer = self.renderer()
        renderer.render(self.hook_end("The user likes tea", name="other"))
        assert "remembered" not in self.text(buffer)

    def test_the_line_cannot_carry_an_escape_sequence(self):
        renderer, buffer = self.renderer()
        renderer.render(self.start("c1", memory_content="red \x1b[31mtext"))
        renderer.render(self.end("c1", "Memory stored."))
        assert "\x1b" not in buffer.getvalue()


class TestInTheWizard:
    """The question is a step of `run_setup`: its answer is saved, and Ctrl-B on it
    goes back to the question before."""

    @pytest.fixture(autouse=True)
    def first_setup(self, monkeypatch):
        monkeypatch.setattr(select_mod, "_interactive", lambda theme: False)
        monkeypatch.setattr(
            setup, "_choose_builtin_skills", lambda console, theme, current: current
        )
        for name in ("OPENAI_API_KEY", "VLLM_API_KEY", "LLM_API_KEY"):
            monkeypatch.delenv(name, raising=False)

    def run(self, answers, tmp_path):
        console = Console(file=io.StringIO(), width=100, force_terminal=False)
        with patch("builtins.input", side_effect=answers):
            return setup.run_setup(
                console, Theme.build(allow_animation=False), path=tmp_path / "config.yaml"
            )

    def test_the_answer_is_saved(self, tmp_path):
        # provider, model, url, api key, max tokens, context window, memory
        result = self.run(["openai", "m", "", "", "", "", "global"], tmp_path)

        assert manager_of(result)["file_path"].endswith("memory.md")

    def test_back_from_it_asks_the_previous_question_again(self, tmp_path):
        from domyn.agents.interfaces.cli.prompt import GoBack

        # Back from memory to the context window, typed this time, then memory again.
        answers = ["openai", "m", "", "", "", "", GoBack(), "64000", "global"]

        result = self.run(answers, tmp_path)

        assert result["llm_providers"][0]["context_window"] == 64000
        assert manager_of(result)["file_path"].endswith("memory.md")


class TestWhoDecidesWhatToRemember:
    """Asked only with a decision model: the answer is the memory's execution_mode."""

    def test_the_decision_model_writes_automatic(self):
        choice = MemoryChoice(scope="global", decided_by=MemoryDecider.MODEL)
        assert manager_of(written(base(), choice))["execution_mode"] == "automatic"

    def test_the_agent_writes_agentic(self):
        choice = MemoryChoice(scope="global", decided_by=MemoryDecider.AGENT)
        assert manager_of(written(base(), choice))["execution_mode"] == "agentic"

    def test_changing_it_keeps_what_was_tuned_by_hand(self):
        raw = written(base(), MemoryChoice(scope="global"))
        raw["agents"][0]["memory_manager"]["recency_limit"] = 7
        again = written(raw, MemoryChoice(scope="global", decided_by=MemoryDecider.MODEL))
        assert manager_of(again)["execution_mode"] == "automatic"
        assert manager_of(again)["recency_limit"] == 7

    def test_not_asked_keeps_the_mode_the_file_has(self):
        raw = written(base(), MemoryChoice(scope="global", decided_by=MemoryDecider.MODEL))
        again = written(raw, MemoryChoice(scope="global"))
        assert manager_of(again)["execution_mode"] == "automatic"

    @pytest.mark.parametrize("decider", [MemoryDecider.AGENT, MemoryDecider.MODEL])
    def test_what_it_writes_is_what_it_reads(self, decider):
        raw = written(base(), MemoryChoice(scope="global", decided_by=decider))
        assert current_memory_from(raw).decided_by == decider

    def test_a_memory_naming_no_mode_reads_as_what_it_runs(self, numbered):
        # No execution_mode runs AUTOMATIC: read as "the agent", Enter wrote agentic.
        raw = written(base(), MemoryChoice(scope="global"))
        del raw["agents"][0]["memory_manager"]["execution_mode"]
        current = current_memory_from(raw).decided_by
        assert current == MemoryDecider.MODEL
        assert self.ask(current, "")[0] == MemoryDecider.MODEL
        assert ("mode", "automatic") in cfg._memory_items(raw)[0].fields

    def test_the_default_mode_is_the_managers(self):
        default = FilesystemMemoryManager.model_fields["execution_mode"].default
        assert default.value == cfg.DEFAULT_MEMORY_MODE

    def ask(self, current, answer, level="active"):
        buffer = io.StringIO()
        console = Console(file=buffer, width=200, force_terminal=False)
        with patch("builtins.input", side_effect=[answer]):
            chosen = setup._ask_memory_decider(
                console,
                Theme.build(allow_animation=False),
                current,
                DecisionsChoice(level=level),
            )
        return chosen, buffer.getvalue()

    @pytest.fixture
    def numbered(self, monkeypatch):
        monkeypatch.setattr(select_mod, "_interactive", lambda theme: False)

    def test_enter_keeps_what_the_file_says(self, numbered):
        assert self.ask(MemoryDecider.AGENT, "")[0] == MemoryDecider.AGENT

    def test_a_new_memory_opens_on_the_decision_model(self, numbered):
        assert self.ask(None, "")[0] == MemoryDecider.MODEL

    def test_at_the_shadow_level_it_says_the_agent_still_decides(self, numbered):
        _, shown = self.ask(None, "", level="shadow")
        assert "the agent's model still decides" in shown

    def test_in_the_wizard_it_follows_the_memory_question(self, tmp_path, monkeypatch):
        monkeypatch.setattr(select_mod, "_interactive", lambda theme: False)
        monkeypatch.setattr(
            setup, "_choose_builtin_skills", lambda console, theme, current: current
        )
        monkeypatch.setattr(
            setup,
            "_ask_decisions",
            lambda console, theme, current, suggested_url=None: DecisionsChoice(
                level="active", base_url="http://d:1", model="m"
            ),
        )
        for name in ("OPENAI_API_KEY", "VLLM_API_KEY", "LLM_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        console = Console(file=io.StringIO(), width=100, force_terminal=False)
        # provider, model, url, api key, max tokens, context window, memory, who decides
        answers = ["openai", "m", "", "", "", "", "global", MemoryDecider.MODEL]
        with patch("builtins.input", side_effect=answers):
            result = setup.run_setup(
                console, Theme.build(allow_animation=False), path=tmp_path / "config.yaml"
            )
        assert manager_of(result)["execution_mode"] == "automatic"

    def test_without_a_memory_it_is_not_asked(self, tmp_path, monkeypatch):
        monkeypatch.setattr(select_mod, "_interactive", lambda theme: False)
        monkeypatch.setattr(
            setup, "_choose_builtin_skills", lambda console, theme, current: current
        )
        monkeypatch.setattr(
            setup,
            "_ask_decisions",
            lambda console, theme, current, suggested_url=None: DecisionsChoice(
                level="active", base_url="http://d:1", model="m"
            ),
        )
        for name in ("OPENAI_API_KEY", "VLLM_API_KEY", "LLM_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        console = Console(file=io.StringIO(), width=100, force_terminal=False)
        # A further answer would be left over: input() raises StopIteration if asked.
        with patch("builtins.input", side_effect=["openai", "m", "", "", "", "", "no"]):
            result = setup.run_setup(
                console, Theme.build(allow_animation=False), path=tmp_path / "config.yaml"
            )
        assert manager_of(result) is None


class TestAddingAndRemovingTheDecisionModel:
    """The decider follows the decision model across runs of the wizard."""

    @pytest.fixture(autouse=True)
    def wizard(self, monkeypatch):
        monkeypatch.setattr(select_mod, "_interactive", lambda theme: False)
        monkeypatch.setattr(
            setup, "_choose_builtin_skills", lambda console, theme, current: current
        )
        for name in ("OPENAI_API_KEY", "VLLM_API_KEY", "LLM_API_KEY"):
            monkeypatch.delenv(name, raising=False)

    def run(self, monkeypatch, path, decisions: DecisionsChoice, answers: list[str]):
        monkeypatch.setattr(
            setup,
            "_ask_decisions",
            lambda console, theme, current, suggested_url=None: decisions,
        )
        buffer = io.StringIO()
        console = Console(file=buffer, width=200, force_terminal=False)
        # provider, model, url, api key, max tokens, context window, then `answers`
        with patch("builtins.input", side_effect=["openai", "m", "", "", "", "", *answers]):
            result = setup.run_setup(
                console,
                Theme.build(allow_animation=False),
                existing=yaml.safe_load(path.read_text()) if path.exists() else None,
                path=path,
            )
        return result, buffer.getvalue()

    ON = DecisionsChoice(level="active", base_url="http://d:1", model="m")
    OFF = DecisionsChoice(level="no")

    def test_removing_it_makes_the_memory_agentic(self, tmp_path, monkeypatch):
        path = tmp_path / "config.yaml"
        first, _ = self.run(monkeypatch, path, self.ON, ["global", MemoryDecider.MODEL])
        assert manager_of(first)["execution_mode"] == "automatic"

        # No decider question: input() would run out of answers.
        second, shown = self.run(monkeypatch, path, self.OFF, ["global"])
        assert "decision_providers" not in second
        assert manager_of(second)["execution_mode"] == "agentic"
        assert "the agent saves memories itself" in shown

    def test_adding_it_to_an_agentic_memory_asks_who_decides(self, tmp_path, monkeypatch):
        path = tmp_path / "config.yaml"
        first, _ = self.run(monkeypatch, path, self.OFF, ["global"])
        assert manager_of(first)["execution_mode"] == "agentic"

        second, shown = self.run(monkeypatch, path, self.ON, ["global", MemoryDecider.MODEL])
        assert "who decides what to remember" in shown
        assert manager_of(second)["execution_mode"] == "automatic"

    def test_a_hand_written_automatic_without_a_decision_model_is_kept(self):
        raw = written(base(), MemoryChoice(scope="global", decided_by=MemoryDecider.MODEL))
        assert "decision_providers" not in raw
        again = configuration_v1_dict_from(
            *current_settings_from(raw),
            raw,
            DecisionsChoice(level="no"),
            MemoryChoice(scope="global"),
        )
        assert manager_of(again)["execution_mode"] == "automatic"
