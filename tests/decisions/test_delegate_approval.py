"""The approval gate on a tool call that arrives from a delegated runtime.

The runner gates these too, and the comment where it does says why: *"leaving
it out here would make approvals bypassable by delegating to an external
agent, which is a hole rather than an omission."* The decision model is
consulted on that path exactly as on the local one.

What had no test is the half after the verdict: the `DECISION_MADE` the runner
yields there. Coverage found it — every other line of that branch is exercised
by `tests/runner/test_delegate_visibility_tool_start.py`, and the one yield was
not. A decision taken and never reported is the shape of bug that looks like a
working feature, so it is worth one test of its own.
"""

from collections.abc import AsyncGenerator
from typing import Any

import pytest

from domyn.agents import Runner
from domyn.agents.agents import DelegateAgent
from domyn.agents.core import (
    ApprovalType,
    BaseEvent,
    ExecutionEventType,
    InteractionContext,
    Part,
    ToolAction,
)
from domyn.agents.decisions import (
    Answer,
    Decision,
    DecisionMode,
    DecisionProvider,
    DecisionRuntime,
    Provenance,
    UseCasePolicy,
)
from domyn.agents.runner import RunnerTransport
from domyn.agents.tools import FunctionTool


class Says(DecisionProvider):
    """Answers the approval verdict with one option, at one confidence."""

    type_name = "says"
    provenance = Provenance.NATIVE

    def __init__(self, value: str, confidence: float):
        super().__init__(timeout_s=5.0)
        self._value = value
        self._confidence = confidence

    async def _decide(self, state: Any, questions: dict[str, Any]) -> Decision:
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
            latency_ms=148,
        )


class Replays(RunnerTransport):
    """An in-process stand-in for the remote runtime."""

    def __init__(self, events: list[BaseEvent]) -> None:
        self._events = events
        self.results: list[BaseEvent] = []

    async def execute(
        self, agent_name: str, context: InteractionContext
    ) -> AsyncGenerator[BaseEvent, None]:
        for event in self._events:
            yield event

    async def send_result(self, event: BaseEvent, conversation_id: str | None = None) -> None:
        # The remote is waiting on a result for the call it sent; a refusal has
        # to come back here or it hangs on a future nobody resolves.
        self.results.append(event)


def runtime_for(value: str, confidence: float, threshold: float = 0.8) -> DecisionRuntime:
    return DecisionRuntime(
        provider=Says(value, confidence),
        policies={"approvals": UseCasePolicy(mode=DecisionMode.ACTIVE, threshold=threshold)},
    )


def delegate_agent(runtime: DecisionRuntime | None) -> tuple[DelegateAgent, list[str]]:
    """A delegate whose *local* `delete_space` tool is gated."""
    deleted: list[str] = []

    def delete_space(space: str) -> str:
        """Delete a space."""
        deleted.append(space)
        return f"deleted {space}"

    agent = DelegateAgent(
        name="RemoteRoot",
        description="root delegate",
        transport=Replays(
            [
                BaseEvent(
                    event_type=ExecutionEventType.TOOL_START,
                    author="ext",
                    action=ToolAction(
                        name="delete_space",
                        parameters={"space": "prod"},
                        call_id="call-1",
                    ),
                ),
                BaseEvent(
                    event_type=ExecutionEventType.RESPONSE,
                    author="ext",
                    content=[Part(text="done")],
                ),
            ]
        ),
        tools=[FunctionTool(func=delete_space, approval_type=ApprovalType.USER_CONFIRM)],
        decisions=runtime,
    )
    return agent, deleted


async def run(agent: DelegateAgent, conversation_id: str) -> list[BaseEvent]:
    return [
        event
        async for event in Runner().run(
            application_name="app",
            user_id="u1",
            root=agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                author="user",
                content=[Part(text="delete the prod space")],
            ),
            conversation_id=conversation_id,
        )
    ]


def decisions_in(events: list[BaseEvent]) -> list[BaseEvent]:
    return [e for e in events if e.event_type is ExecutionEventType.DECISION_MADE]


class TestTheDecisionReachesTheStream:
    @pytest.mark.asyncio
    async def test_a_confident_allow_is_reported(self):
        agent, _ = delegate_agent(runtime_for("allow", 0.96))
        made = decisions_in(await run(agent, "delegate-allow"))
        assert [e.metadata["use_case"] for e in made] == ["approvals"]
        assert made[0].metadata["acted"] is True

    @pytest.mark.asyncio
    async def test_the_event_carries_what_it_decided_and_what_it_cost(self):
        agent, _ = delegate_agent(runtime_for("allow", 0.96))
        decision = decisions_in(await run(agent, "delegate-metadata"))[0]
        assert decision.metadata["value"] == "allow"
        assert decision.metadata["confidence"] == pytest.approx(0.96)
        assert decision.metadata["provenance"] == "native"
        assert decision.metadata["latency_ms"] == 148
        assert decision.metadata["tool"] == "delete_space"

    @pytest.mark.asyncio
    async def test_the_state_never_travels_with_it(self):
        """The state is the tool call and its arguments — customer content. It
        is what the model was shown, and it must not land in an event stream a
        host persists or logs."""
        agent, _ = delegate_agent(runtime_for("allow", 0.96))
        decision = decisions_in(await run(agent, "delegate-no-state"))[0]
        assert "state" not in decision.metadata
        assert "prod" not in str(decision.metadata)

    @pytest.mark.asyncio
    async def test_a_deny_is_reported_too(self):
        agent, _ = delegate_agent(runtime_for("deny", 0.94))
        made = decisions_in(await run(agent, "delegate-deny"))
        assert made and made[0].metadata["value"] == "deny"


class TestTheVerdictStillGovernsTheCall:
    """The event is the reporting half; these pin that reporting did not become
    the *only* thing that happens."""

    @pytest.mark.asyncio
    async def test_a_confident_allow_runs_the_tool_without_asking(self):
        agent, deleted = delegate_agent(runtime_for("allow", 0.96))
        events = await run(agent, "delegate-runs")
        assert deleted == ["prod"]
        assert ExecutionEventType.TOOL_NEEDS_FEEDBACK not in [e.event_type for e in events]

    @pytest.mark.asyncio
    async def test_an_answer_below_the_bar_still_asks(self):
        """0.79 against a 0.8 threshold: the model would allow, and is ignored."""
        agent, deleted = delegate_agent(runtime_for("allow", 0.79))
        events = await run(agent, "delegate-asks")
        assert deleted == []
        assert ExecutionEventType.TOOL_NEEDS_FEEDBACK in [e.event_type for e in events]

    @pytest.mark.asyncio
    async def test_a_deny_refuses_and_tells_the_remote(self):
        """The delegated runtime is waiting on a result for this call."""
        agent, deleted = delegate_agent(runtime_for("deny", 0.94))
        await run(agent, "delegate-refuses")
        assert deleted == []
        assert agent.transport.results, "the remote was left waiting on a call it sent"

    @pytest.mark.asyncio
    async def test_the_decision_is_reported_before_the_refusal(self):
        """Why the call was refused has to be readable before the refusal, or a
        consumer shows a denial with no reason beside it.

        The refusal is a `TOOL_END` carrying an observation, not a `TOOL_ERROR`:
        the remote is waiting on a *result* for the call it sent, and an error
        would leave it waiting."""
        agent, _ = delegate_agent(runtime_for("deny", 0.94))
        events = await run(agent, "delegate-order")
        kinds = [e.event_type for e in events]
        assert kinds.index(ExecutionEventType.DECISION_MADE) < kinds.index(
            ExecutionEventType.TOOL_END
        )
        refusal = next(e for e in events if e.event_type is ExecutionEventType.TOOL_END)
        assert "not executed" in str(getattr(refusal.action, "observation", ""))


class TestWithoutADecisionModel:
    @pytest.mark.asyncio
    async def test_the_delegate_path_is_what_it_is_today(self):
        agent, deleted = delegate_agent(None)
        events = await run(agent, "delegate-none")
        assert decisions_in(events) == []
        assert deleted == []
        assert ExecutionEventType.TOOL_NEEDS_FEEDBACK in [e.event_type for e in events]
