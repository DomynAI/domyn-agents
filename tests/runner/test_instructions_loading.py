"""Standing instructions are loaded by the runner, once per turn, like skills.

Per turn is what makes an edited ``SOUL.md`` reach the next message, and a
``/clear`` (a new conversation id on the same agent) pick it up with nothing
special. A source that is down must never refuse a turn.
"""

from pathlib import Path
from typing import Any

import pytest

from domyn.agents import Runner
from domyn.agents.agents import Agent
from domyn.agents.core import AgentAction, BaseEvent, ExecutionEventType, Part
from domyn.agents.instructions import (
    AgentInstructions,
    InMemoryInstructionsSource,
    InstructionRuntime,
    LocalDirInstructionsSource,
)
from tests.runner.test_skills_loading import _agent, _run, _scripted, _user_input


class CountingSource(InMemoryInstructionsSource):
    type: str = "counting_instructions"
    calls: int = 0

    async def load(self) -> AgentInstructions:
        self.calls += 1
        return await super().load()


class DeadSource(InMemoryInstructionsSource):
    type: str = "dead_instructions"

    async def load(self) -> AgentInstructions:
        raise OSError("instructions backend unreachable")


def _respond(author: str) -> list[BaseEvent]:
    return [
        BaseEvent(
            event_type=ExecutionEventType.RESPONSE, author=author, content=[Part(text="done")]
        )
    ]


def _with(agent: Agent, source: Any) -> Agent:
    agent.standing_instructions = InstructionRuntime(source=source)
    return agent


class TestLoadedOncePerTurn:
    @pytest.mark.asyncio
    async def test_a_handoff_and_back_loads_once(self, runner: Runner) -> None:
        source = CountingSource(soul="Answer in Italian.")
        sub = _agent("SubAgent")
        main = _with(_agent("MainAgent", sub_agents=[sub]), source)
        calls: list[dict[str, Any]] = []
        main._execute = _scripted(
            [
                [
                    BaseEvent(
                        event_type=ExecutionEventType.AGENT_CALL,
                        author="MainAgent",
                        action=AgentAction(name="SubAgent", parameters={"task": "t"}),
                    )
                ],
                _respond("MainAgent"),
            ],
            calls,
        )
        sub._execute = _scripted([_respond("SubAgent")], [])

        await _run(runner, main, _user_input(), "instructions_once")

        assert len(calls) == 2
        assert source.calls == 1
        assert calls[0]["instructions"] is calls[1]["instructions"]

    @pytest.mark.asyncio
    async def test_an_edit_reaches_the_next_turn(self, runner: Runner, tmp_path: Path) -> None:
        """Including a new conversation, which is all ``/clear`` is."""
        soul = tmp_path / "SOUL.md"
        soul.write_text("Answer in Italian.", encoding="utf-8")
        main = _with(_agent("MainAgent"), LocalDirInstructionsSource(paths=[tmp_path]))
        calls: list[dict[str, Any]] = []
        main._execute = _scripted([_respond("MainAgent"), _respond("MainAgent")], calls)

        await _run(runner, main, _user_input(), "instructions_before")
        soul.write_text("Answer in French.", encoding="utf-8")
        await _run(runner, main, _user_input(), "instructions_after_clear")

        assert "Italian" in calls[0]["instructions"].text
        assert "French" in calls[1]["instructions"].text


class TestEachAgentGetsItsOwn:
    @pytest.mark.asyncio
    async def test_a_sub_agent_is_handed_its_own_not_its_parents(self, runner: Runner) -> None:
        sub = _with(_agent("SubAgent"), InMemoryInstructionsSource(soul="Answer in French."))
        bare = _agent("BareAgent")
        main = _with(
            _agent("MainAgent", sub_agents=[sub, bare]),
            InMemoryInstructionsSource(soul="Answer in Italian."),
        )
        main_calls: list[dict[str, Any]] = []
        sub_calls: list[dict[str, Any]] = []
        main._execute = _scripted(
            [
                [
                    BaseEvent(
                        event_type=ExecutionEventType.AGENT_CALL,
                        author="MainAgent",
                        action=AgentAction(name="SubAgent", parameters={"task": "t"}),
                    )
                ],
                _respond("MainAgent"),
            ],
            main_calls,
        )
        sub._execute = _scripted([_respond("SubAgent")], sub_calls)

        await _run(runner, main, _user_input(), "instructions_per_agent")

        assert main_calls[0]["instructions"].text.count("Italian") == 1
        assert "French" not in main_calls[0]["instructions"].text
        assert "French" in sub_calls[0]["instructions"].text
        assert "Italian" not in sub_calls[0]["instructions"].text


class TestNothingRefusesTheTurn:
    @pytest.mark.asyncio
    async def test_a_dead_source_runs_the_turn_without_them(self, runner: Runner) -> None:
        main = _with(_agent("MainAgent"), DeadSource())
        calls: list[dict[str, Any]] = []
        main._execute = _scripted([_respond("MainAgent")], calls)

        events = await _run(runner, main, _user_input(), "instructions_dead")

        assert calls and not calls[0]["instructions"]
        assert not [e for e in events if e.event_type == ExecutionEventType.ERROR_OCCURRED]


class TestAnAgentWithoutThem:
    @pytest.mark.asyncio
    async def test_is_called_exactly_as_before(self, runner: Runner) -> None:
        main = _agent("MainAgent")
        calls: list[dict[str, Any]] = []
        main._execute = _scripted([_respond("MainAgent")], calls)

        await _run(runner, main, _user_input(), "instructions_none")

        assert "instructions" not in calls[0]


class TestLoadedAlongsideSkills:
    @pytest.mark.asyncio
    async def test_the_two_loads_run_concurrently(self, runner: Runner) -> None:
        """Each source waits for the other to have started: run one after the
        other, this deadlocks and the timeout fails the test."""
        import asyncio

        from domyn.agents.skills import LoadedSkills, SkillQuery, SkillsRuntime
        from tests.runner.test_skills_loading import CountingSource as CountingSkills

        skills_started, instructions_started = asyncio.Event(), asyncio.Event()

        class WaitingSkills(CountingSkills):
            type: str = "waiting_skills"

            async def load(self, query: SkillQuery) -> LoadedSkills:
                skills_started.set()
                await instructions_started.wait()
                return await super().load(query)

        class WaitingInstructions(InMemoryInstructionsSource):
            type: str = "waiting_instructions"

            async def load(self) -> AgentInstructions:
                instructions_started.set()
                await skills_started.wait()
                return await super().load()

        main = _with(
            _agent("MainAgent", skills=SkillsRuntime(source=WaitingSkills())),
            WaitingInstructions(soul="Answer in Italian."),
        )
        calls: list[dict[str, Any]] = []
        main._execute = _scripted([_respond("MainAgent")], calls)

        await asyncio.wait_for(_run(runner, main, _user_input(), "instructions_concurrent"), 5)

        assert "skills" in calls[0] and "Italian" in calls[0]["instructions"].text
