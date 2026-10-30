"""A decision model inside a real turn.

The unit tests prove the gate's logic; these prove it is actually wired in —
that a confident ALLOW skips the question the user would have been asked, that
an unsure one does not, and that the event carrying all of it reaches whoever is
watching the stream.
"""

import pytest

from domyn.agents import Agent, Runner
from domyn.agents.core import ApprovalType, BaseEvent, ExecutionEventType, Part, ToolAction
from domyn.agents.decisions import (
    Answer,
    Decision,
    DecisionMode,
    DecisionProvider,
    DecisionRuntime,
    Provenance,
    UseCasePolicy,
)
from domyn.agents.llm import BaseLLMProvider
from domyn.agents.tools import FunctionTool
from tests.decisions.conftest import async_gen_from_list


class Says(DecisionProvider):
    type_name = "says"
    provenance = Provenance.NATIVE

    def __init__(self, value: str, confidence: float):
        super().__init__(timeout_s=5.0)
        self._value = value
        self._confidence = confidence
        self.asked = 0

    async def _decide(self, state, questions):
        self.asked += 1
        return Decision(
            answers={
                name: Answer(
                    known=True,
                    provenance=Provenance.NATIVE,
                    value=self._value,
                    confidence=self._confidence,
                )
                for name in questions
            },
            latency_ms=147,
        )


def _agent(llm_provider: BaseLLMProvider, runtime: DecisionRuntime | None) -> Agent:
    def add_tool(a: int, b: int) -> int:
        return a + b

    agent = Agent(
        name="MainAgent",
        description="main agent description",
        instruction="main agent instruction",
        llm_provider=llm_provider,
        tools=[FunctionTool(func=add_tool, approval_type=ApprovalType.USER_CONFIRM)],
        decisions=runtime,
    )
    agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.TOOL_START,
                author="main_agent",
                action=ToolAction(name="add_tool", parameters={"a": 1, "b": 2}),
            )
        ]
    )
    return agent


async def _run(runner: Runner, agent: Agent, name: str) -> list[BaseEvent]:
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
                content=[Part(text="add 1 and 2")],
                turn_id="turn_1",
            ),
        )
    ]


def _types(events: list[BaseEvent]) -> list[ExecutionEventType]:
    return [event.event_type for event in events]


class TestAConfidentAllowSkipsTheQuestion:
    @pytest.mark.asyncio
    async def test_the_tool_runs_without_asking(self, runner: Runner, llm_provider):
        runtime = DecisionRuntime(
            Says("allow", 0.95), {"approvals": UseCasePolicy(DecisionMode.ACTIVE, 0.8)}
        )
        events = await _run(runner, _agent(llm_provider, runtime), "decide_allow")

        assert ExecutionEventType.TOOL_NEEDS_FEEDBACK not in _types(events)
        assert ExecutionEventType.TOOL_END in _types(events)

    @pytest.mark.asyncio
    async def test_the_decision_reaches_the_stream(self, runner: Runner, llm_provider):
        runtime = DecisionRuntime(
            Says("allow", 0.95), {"approvals": UseCasePolicy(DecisionMode.ACTIVE, 0.8)}
        )
        events = await _run(runner, _agent(llm_provider, runtime), "decide_allow_event")

        made = [e for e in events if e.event_type is ExecutionEventType.DECISION_MADE]
        assert len(made) == 1
        metadata = made[0].metadata
        assert metadata["use_case"] == "approvals"
        assert metadata["acted"] is True
        assert metadata["value"] == "allow"
        assert metadata["latency_ms"] == 147
        assert metadata["tool"] == "add_tool"


class TestAnUnsureAnswerLeavesTheGateAlone:
    @pytest.mark.asyncio
    async def test_the_user_is_still_asked(self, runner: Runner, llm_provider):
        """0.53 is the measured confidence of the one wrong answer in the eval."""
        runtime = DecisionRuntime(
            Says("allow", 0.53), {"approvals": UseCasePolicy(DecisionMode.ACTIVE, 0.8)}
        )
        events = await _run(runner, _agent(llm_provider, runtime), "decide_unsure")

        assert ExecutionEventType.TOOL_NEEDS_FEEDBACK in _types(events)

    @pytest.mark.asyncio
    async def test_it_is_still_recorded(self, runner: Runner, llm_provider):
        runtime = DecisionRuntime(
            Says("allow", 0.53), {"approvals": UseCasePolicy(DecisionMode.ACTIVE, 0.8)}
        )
        events = await _run(runner, _agent(llm_provider, runtime), "decide_unsure_event")

        made = [e for e in events if e.event_type is ExecutionEventType.DECISION_MADE]
        assert made and made[0].metadata["acted"] is False


class TestADenyRefusesTheCall:
    @pytest.mark.asyncio
    async def test_the_tool_does_not_run(self, runner: Runner, llm_provider):
        runtime = DecisionRuntime(
            Says("deny", 0.95), {"approvals": UseCasePolicy(DecisionMode.ACTIVE, 0.8)}
        )
        events = await _run(runner, _agent(llm_provider, runtime), "decide_deny")

        assert ExecutionEventType.TOOL_NEEDS_FEEDBACK not in _types(events)
        refusals = [
            e
            for e in events
            if e.event_type is ExecutionEventType.TOOL_END
            and "not executed" in str(getattr(e.action, "observation", ""))
        ]
        assert refusals


class TestShadowModeChangesNothingAboutTheTurn:
    @pytest.mark.asyncio
    async def test_the_user_is_asked_exactly_as_before(self, runner: Runner, llm_provider):
        provider = Says("allow", 0.99)
        runtime = DecisionRuntime(provider, {"approvals": UseCasePolicy(DecisionMode.SHADOW, 0.8)})
        events = await _run(runner, _agent(llm_provider, runtime), "decide_shadow")

        assert ExecutionEventType.TOOL_NEEDS_FEEDBACK in _types(events)
        assert provider.asked == 1  # measured, not acted on
        made = [e for e in events if e.event_type is ExecutionEventType.DECISION_MADE]
        assert made and made[0].metadata["mode"] == "shadow"


class TestWithNoDecisionModelNothingChanges:
    @pytest.mark.asyncio
    async def test_the_turn_is_exactly_what_it_is_today(self, runner: Runner, llm_provider):
        events = await _run(runner, _agent(llm_provider, None), "decide_absent")

        assert ExecutionEventType.TOOL_NEEDS_FEEDBACK in _types(events)
        assert ExecutionEventType.DECISION_MADE not in _types(events)

    @pytest.mark.asyncio
    async def test_a_broken_model_does_not_break_the_turn(self, runner: Runner, llm_provider):
        class Broken(DecisionProvider):
            type_name = "broken"
            provenance = Provenance.NATIVE

            async def _decide(self, state, questions):
                raise RuntimeError("endpoint down")

        runtime = DecisionRuntime(Broken(), {"approvals": UseCasePolicy(DecisionMode.ACTIVE, 0.8)})
        events = await _run(runner, _agent(llm_provider, runtime), "decide_broken")

        assert ExecutionEventType.TOOL_NEEDS_FEEDBACK in _types(events)
        made = [e for e in events if e.event_type is ExecutionEventType.DECISION_MADE]
        assert made and made[0].metadata["known"] is False
