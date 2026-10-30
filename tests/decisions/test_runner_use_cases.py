"""The four use cases that act on the *turn*, driven through a real runner.

`test_runner_integration.py` covers the approval gate, which is the one that
answers inside a tool call. These four are decided once per turn in
`Runner._load_turn_resources` and consumed somewhere else entirely —
`get_tool_definitions`, the compactor, the system prompt, the turn's own
refusal — so the module tests beside this one prove the decision and prove
nothing about whether anyone reads it.

That gap was measured, not guessed: with only the module tests, the bodies of
`_decide_tool_visibility`, `_decide_context_relevance`,
`_apply_conditional_instructions` and `_refuse_over_sensitivity` never ran.
Every assertion here is about the turn, never about the helper.
"""

from typing import Any

import pytest

from domyn.agents import Agent, Runner
from domyn.agents.core import BaseEvent, ExecutionEventType, Part, ToolAction
from domyn.agents.decisions import (
    Answer,
    Decision,
    DecisionMode,
    DecisionProvider,
    DecisionRuntime,
    Provenance,
    UseCasePolicy,
)
from domyn.agents.decisions.context import irrelevant_ids
from domyn.agents.decisions.tools import hidden_for
from domyn.agents.instructions import InMemoryInstructionsSource, InstructionRuntime
from domyn.agents.llm import BaseLLMProvider
from domyn.agents.tools import FunctionTool
from tests.decisions.conftest import async_gen_from_list


class Noul(DecisionProvider):
    """Answers every `noul` with the same probability.

    A probability, not a verdict: `is_true` and `is_false` both read `value`,
    and the gap between them — "it does not know" — is what three of these
    four use cases turn on.
    """

    type_name = "noul"
    provenance = Provenance.NATIVE

    def __init__(self, probability: float):
        super().__init__(timeout_s=5.0)
        self._probability = probability
        self.asked = 0

    async def _decide(self, state: Any, questions: dict[str, Any]) -> Decision:
        self.asked += 1
        return Decision(
            answers={
                name: Answer(
                    known=True,
                    provenance=Provenance.NATIVE,
                    value=self._probability,
                    confidence=max(self._probability, 1.0 - self._probability),
                )
                for name in questions
            },
            latency_ms=151,
        )


class Breaks(DecisionProvider):
    """The endpoint is reachable and answers with a 500, a timeout, or nonsense."""

    type_name = "breaks"
    provenance = Provenance.NATIVE

    async def _decide(self, state: Any, questions: dict[str, Any]) -> Decision:
        raise RuntimeError("the endpoint went away")


def runtime_for(use_case: str, provider: DecisionProvider, **kwargs: Any) -> DecisionRuntime:
    return DecisionRuntime(
        provider=provider,
        policies={use_case: UseCasePolicy(mode=DecisionMode.ACTIVE, threshold=0.8, **kwargs)},
    )


def many_tools(count: int = 14) -> list[FunctionTool]:
    """A catalogue over `MIN_TOOLS`, or the use case declines to act at all."""

    def make(index: int) -> FunctionTool:
        def tool_function(value: str) -> str:
            """Do something with a value."""
            return value

        tool_function.__name__ = f"tool_{index}"
        return FunctionTool(func=tool_function)

    return [make(i) for i in range(count)]


def agent_for(
    llm_provider: BaseLLMProvider,
    runtime: DecisionRuntime | None,
    *,
    tools: list[Any] | None = None,
    instructions: InstructionRuntime | None = None,
) -> Agent:
    agent = Agent(
        name="MainAgent",
        description="main agent description",
        instruction="main agent instruction",
        llm_provider=llm_provider,
        tools=tools if tools is not None else [],
        decisions=runtime,
        standing_instructions=instructions,
    )
    agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="MainAgent",
                content=[Part(text="done")],
            )
        ]
    )
    return agent


async def run_turn(
    runner: Runner,
    agent: Agent,
    name: str,
    text: str = "summarise the failing test",
    turn_id: str = "turn_1",
) -> list[BaseEvent]:
    return [
        event
        async for event in runner.run(
            conversation_id=name,
            application_name="TestApplication",
            user_id="test_user",
            root=agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                author="user",
                content=[Part(text=text)],
                turn_id=turn_id,
            ),
        )
    ]


async def conversation_state(runner: Runner, name: str) -> dict[str, Any]:
    """What the turn left behind, read back the way the next turn would."""
    conversation = await runner.conversation_manager.get_conversation(
        "TestApplication", "test_user", name
    )
    return conversation.state or {}


class AsContext:
    """The two attributes `hidden_for` reads, nothing else."""

    def __init__(self, state: dict[str, Any], turn_id: str = "turn_1"):
        self.state = state
        self.turn_id = turn_id


def decisions_in(events: list[BaseEvent]) -> list[BaseEvent]:
    return [e for e in events if e.event_type == ExecutionEventType.DECISION_MADE]


def use_cases_in(events: list[BaseEvent]) -> list[str]:
    return [(e.metadata or {}).get("use_case") for e in decisions_in(events)]


class TestHidingToolsForTheTurn:
    @pytest.mark.asyncio
    async def test_a_confident_no_hides_the_tool_from_the_turn(self, runner, llm_provider):
        """0.02 is "confidently not needed", and the runner has to leave that
        where `get_tool_definitions` will look for it."""
        agent = agent_for(llm_provider, runtime_for("tools", Noul(0.02)), tools=many_tools())
        await run_turn(runner, agent, "hide-tools")
        state = await conversation_state(runner, "hide-tools")
        assert len(hidden_for(AsContext(state))) == 14

    @pytest.mark.asyncio
    async def test_a_decision_from_another_turn_hides_nothing(self, runner, llm_provider):
        """The stash is keyed by turn: a task that is over must not keep
        hiding tools from the one that follows it."""
        agent = agent_for(llm_provider, runtime_for("tools", Noul(0.02)), tools=many_tools())
        await run_turn(runner, agent, "stale-tools")
        state = await conversation_state(runner, "stale-tools")
        assert hidden_for(AsContext(state, turn_id="turn_2")) == frozenset()

    @pytest.mark.asyncio
    async def test_an_unsure_answer_hides_nothing(self, runner, llm_provider):
        """0.45 is a coin flip. The tool stays, which is today's behaviour."""
        agent = agent_for(llm_provider, runtime_for("tools", Noul(0.45)), tools=many_tools())
        events = await run_turn(runner, agent, "unsure-tools")
        state = await conversation_state(runner, "unsure-tools")
        assert hidden_for(AsContext(state)) == frozenset()
        assert "tools" not in use_cases_in(events)

    @pytest.mark.asyncio
    async def test_a_confident_yes_hides_nothing(self, runner, llm_provider):
        agent = agent_for(llm_provider, runtime_for("tools", Noul(0.97)), tools=many_tools())
        await run_turn(runner, agent, "keep-tools")
        state = await conversation_state(runner, "keep-tools")
        assert hidden_for(AsContext(state)) == frozenset()

    @pytest.mark.asyncio
    async def test_the_stream_says_so_before_the_answer_arrives(self, runner, llm_provider):
        """A consumer has to see why a tool vanished before it notices that it
        has — so the event is drained ahead of the turn, not after it."""
        agent = agent_for(llm_provider, runtime_for("tools", Noul(0.02)), tools=many_tools())
        events = await run_turn(runner, agent, "tools-event")
        assert "tools" in use_cases_in(events)
        decision_at = next(
            i for i, e in enumerate(events) if e.event_type == ExecutionEventType.DECISION_MADE
        )
        answers_at = [
            i for i, e in enumerate(events) if e.event_type == ExecutionEventType.RESPONSE
        ]
        assert answers_at and all(decision_at < position for position in answers_at)

    @pytest.mark.asyncio
    async def test_the_event_names_what_it_hid(self, runner, llm_provider):
        agent = agent_for(llm_provider, runtime_for("tools", Noul(0.02)), tools=many_tools())
        events = await run_turn(runner, agent, "tools-named")
        decision = next(e for e in decisions_in(events) if e.metadata["use_case"] == "tools")
        assert "tool_0" in decision.metadata["hidden"]

    @pytest.mark.asyncio
    async def test_a_small_catalogue_is_never_asked_about(self, runner, llm_provider):
        provider = Noul(0.02)
        agent = agent_for(llm_provider, runtime_for("tools", provider), tools=many_tools(3))
        await run_turn(runner, agent, "small-catalogue")
        assert provider.asked == 0

    @pytest.mark.asyncio
    async def test_a_broken_model_leaves_every_tool_visible(self, runner, llm_provider):
        agent = agent_for(llm_provider, runtime_for("tools", Breaks()), tools=many_tools())
        events = await run_turn(runner, agent, "tools-broken")
        state = await conversation_state(runner, "tools-broken")
        assert hidden_for(AsContext(state)) == frozenset()
        assert ExecutionEventType.RESPONSE in [e.event_type for e in events]


class TestMarkingOldOutputsForCompaction:
    """The ladder judges what is already in the history, so each of these runs
    a turn that leaves a big tool output behind and then asks about the next
    one."""

    def noisy_agent(self, llm_provider: BaseLLMProvider, runtime: DecisionRuntime | None) -> Agent:
        agent = agent_for(llm_provider, runtime)

        async def _execute(*args: Any, **kwargs: Any):
            """A fresh pair of events per turn — the helper beside this one
            replays a list once, and the ladder needs a turn to judge *and* a
            previous turn to have produced something worth judging."""
            yield BaseEvent(
                event_type=ExecutionEventType.TOOL_END,
                author="MainAgent",
                action=ToolAction(
                    name="run_command",
                    parameters={"command": "ls node_modules"},
                    observation="a line of a directory listing\n" * 200,
                ),
            )
            yield BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="MainAgent",
                content=[Part(text="done")],
            )

        agent._execute = _execute
        return agent

    async def two_turns(
        self, runner: Runner, agent: Agent, name: str
    ) -> tuple[list[BaseEvent], dict[str, Any]]:
        await run_turn(runner, agent, name, "list the dependencies")
        events = await run_turn(runner, agent, name, "now fix the failing test", turn_id="turn_2")
        return events, await conversation_state(runner, name)

    @pytest.mark.asyncio
    async def test_a_confident_no_marks_the_old_output(self, runner, llm_provider):
        agent = self.noisy_agent(llm_provider, runtime_for("context_ladder", Noul(0.02)))
        events, state = await self.two_turns(runner, agent, "ladder-drop")
        assert irrelevant_ids(state)
        assert "context_ladder" in use_cases_in(events)

    @pytest.mark.asyncio
    async def test_an_unsure_answer_marks_nothing(self, runner, llm_provider):
        agent = self.noisy_agent(llm_provider, runtime_for("context_ladder", Noul(0.45)))
        events, state = await self.two_turns(runner, agent, "ladder-unsure")
        assert irrelevant_ids(state) == set()
        assert "context_ladder" not in use_cases_in(events)

    @pytest.mark.asyncio
    async def test_an_output_still_needed_is_kept(self, runner, llm_provider):
        agent = self.noisy_agent(llm_provider, runtime_for("context_ladder", Noul(0.97)))
        _, state = await self.two_turns(runner, agent, "ladder-keep")
        assert irrelevant_ids(state) == set()

    @pytest.mark.asyncio
    async def test_a_broken_model_marks_nothing_and_the_turn_finishes(self, runner, llm_provider):
        agent = self.noisy_agent(llm_provider, runtime_for("context_ladder", Breaks()))
        events, state = await self.two_turns(runner, agent, "ladder-broken")
        assert irrelevant_ids(state) == set()
        assert ExecutionEventType.RESPONSE in [e.event_type for e in events]


class TestKeepingTheInstructionSectionsThatApply:
    SOUL = (
        "## Always\n"
        "Be brief.\n\n"
        "## SQL migrations\n"
        "when: the task involves a database migration\n\n"
        "Check the rollback path first.\n"
    )

    def instructions(self) -> InstructionRuntime:
        return InstructionRuntime(source=InMemoryInstructionsSource(soul=self.SOUL))

    def agent(self, llm_provider: BaseLLMProvider, provider: DecisionProvider) -> Agent:
        return agent_for(
            llm_provider,
            runtime_for("instructions", provider),
            instructions=self.instructions(),
        )

    @pytest.mark.asyncio
    async def test_a_confident_no_drops_the_section(self, runner, llm_provider):
        events = await run_turn(runner, self.agent(llm_provider, Noul(0.02)), "instructions-drop")
        assert "instructions" in use_cases_in(events)

    @pytest.mark.asyncio
    async def test_what_was_dropped_is_recorded_not_vanished(self, runner, llm_provider):
        events = await run_turn(
            runner, self.agent(llm_provider, Noul(0.02)), "instructions-recorded"
        )
        decision = next(e for e in decisions_in(events) if e.metadata["use_case"] == "instructions")
        assert any("SQL migrations" in entry for entry in decision.metadata["skipped"])

    @pytest.mark.asyncio
    async def test_an_unsure_answer_keeps_the_users_own_words(self, runner, llm_provider):
        """0.45 must never delete a line someone wrote. This is the assertion a
        test originally got backwards, which is why `is_false` exists."""
        events = await run_turn(runner, self.agent(llm_provider, Noul(0.45)), "instructions-unsure")
        assert "instructions" not in use_cases_in(events)

    @pytest.mark.asyncio
    async def test_a_broken_model_keeps_everything(self, runner, llm_provider):
        events = await run_turn(runner, self.agent(llm_provider, Breaks()), "instructions-broken")
        assert "instructions" not in use_cases_in(events)
        assert ExecutionEventType.RESPONSE in [e.event_type for e in events]


class TestRefusingATurnOnSensitivity:
    def capped(self, llm_provider: BaseLLMProvider, ceiling: str) -> BaseLLMProvider:
        llm_provider.max_sensitivity = ceiling
        return llm_provider

    @pytest.mark.asyncio
    async def test_an_uncapped_provider_is_never_asked(self, runner, llm_provider):
        """A configuration that does not use this pays nothing — no request,
        no latency, no refusal."""
        provider = Noul(0.97)
        agent = agent_for(llm_provider, runtime_for("sensitivity", provider))
        events = await run_turn(runner, agent, "no-ceiling")
        assert provider.asked == 0
        assert ExecutionEventType.RESPONSE in [e.event_type for e in events]

    @pytest.mark.asyncio
    async def test_a_broken_model_blocks_rather_than_sending(self, runner, llm_provider):
        """The one use case that does not degrade to today's behaviour, because
        today's behaviour is "send it anyway"."""
        agent = agent_for(
            self.capped(llm_provider, "internal"), runtime_for("sensitivity", Breaks())
        )
        events = await run_turn(runner, agent, "sensitivity-broken")
        errors = [e for e in events if e.event_type == ExecutionEventType.ERROR_OCCURRED]
        assert [e.error_code for e in errors] == ["SensitivityPolicy"]

    @pytest.mark.asyncio
    async def test_a_refused_turn_produces_no_answer(self, runner, llm_provider):
        agent = agent_for(
            self.capped(llm_provider, "internal"), runtime_for("sensitivity", Breaks())
        )
        events = await run_turn(runner, agent, "sensitivity-no-answer")
        assert ExecutionEventType.RESPONSE not in [e.event_type for e in events]

    @pytest.mark.asyncio
    async def test_on_unknown_allow_lets_it_through(self, runner, llm_provider):
        agent = agent_for(
            self.capped(llm_provider, "internal"),
            runtime_for("sensitivity", Breaks(), on_unknown="allow"),
        )
        events = await run_turn(runner, agent, "sensitivity-allow")
        assert not [e for e in events if e.event_type == ExecutionEventType.ERROR_OCCURRED]
        assert ExecutionEventType.RESPONSE in [e.event_type for e in events]


class TestWithNoDecisionModelTheTurnIsUntouched:
    @pytest.mark.asyncio
    async def test_nothing_is_hidden_marked_or_refused(self, runner, llm_provider):
        agent = agent_for(llm_provider, None, tools=many_tools())
        events = await run_turn(runner, agent, "no-model")
        state = await conversation_state(runner, "no-model")
        assert hidden_for(AsContext(state)) == frozenset()
        assert irrelevant_ids(state) == set()
        assert decisions_in(events) == []
        assert ExecutionEventType.RESPONSE in [e.event_type for e in events]
