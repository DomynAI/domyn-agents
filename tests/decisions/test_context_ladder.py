"""Which old tool outputs the turn still needs.

Compaction drops by age and size, which cannot be wrong about anything and is
blind to the obvious. This makes it query-aware, in one direction only: it may
move an output to the front of the queue for dropping, never save one.
"""

import pytest

from domyn.agents.core import BaseEvent, ExecutionEventType, ToolAction
from domyn.agents.decisions import (
    Answer,
    Decision,
    DecisionMode,
    DecisionProvider,
    DecisionRuntime,
    Provenance,
    UseCasePolicy,
)
from domyn.agents.decisions.context import MIN_RESULT_CHARS, decide, irrelevant_ids


def tool_end(event_id: str, name: str, size: int) -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.TOOL_END,
        author="agent",
        event_id=event_id,
        action=ToolAction(name=name, parameters={}, observation="x" * size),
    )


def history(*sizes: int) -> list[BaseEvent]:
    return [tool_end(f"e{i}", f"tool_{i}", size) for i, size in enumerate(sizes)]


class Scores(DecisionProvider):
    type_name = "scores"
    provenance = Provenance.NATIVE

    def __init__(self, probabilities: list[float], **kwargs):
        super().__init__(**kwargs)
        self._probabilities = probabilities
        self.asked = 0
        self.state: str | None = None

    async def _decide(self, state, questions):
        self.asked += 1
        self.state = state
        answers = {}
        for position, name in enumerate(questions):
            probability = (
                self._probabilities[position] if position < len(self._probabilities) else 1.0
            )
            answers[name] = Answer(
                known=True,
                provenance=Provenance.NATIVE,
                value=probability,
                confidence=max(probability, 1.0 - probability),
            )
        return Decision(answers=answers)


def runtime(probabilities: list[float], mode=DecisionMode.ACTIVE) -> DecisionRuntime:
    return DecisionRuntime(Scores(probabilities), {"context_ladder": UseCasePolicy(mode, 0.9)})


class TestItMarksWhatIsNoLongerNeeded:
    @pytest.mark.asyncio
    async def test_a_confident_no_is_marked(self):
        state: dict = {}
        dropped = await decide(
            runtime([0.01, 1.0]),
            history=history(5_000, 5_000),
            task="run the tests",
            state=state,
        )
        assert dropped == {"e0"}
        assert irrelevant_ids(state) == {"e0"}

    @pytest.mark.asyncio
    async def test_a_confident_yes_is_left_alone(self):
        assert (
            await decide(runtime([0.99, 0.99]), history=history(5_000, 5_000), task="x", state={})
            == set()
        )

    @pytest.mark.asyncio
    async def test_a_coin_flip_is_left_alone(self):
        """Measured: this model is sure when it says drop and unsure when it
        says keep, so only the first direction is acted on."""
        assert (
            await decide(runtime([0.5, 0.45]), history=history(5_000, 5_000), task="x", state={})
            == set()
        )

    @pytest.mark.asyncio
    async def test_below_the_bar_is_left_alone(self):
        # 0.2 -> 0.8 confidence of "no", under this use case's 0.9.
        assert await decide(runtime([0.2]), history=history(5_000), task="x", state={}) == set()


class TestWhatIsWorthAsking:
    @pytest.mark.asyncio
    async def test_small_outputs_are_not_asked_about(self):
        built = runtime([0.0, 0.0])
        await decide(built, history=history(MIN_RESULT_CHARS - 1, 10), task="x", state={})
        assert built.provider.asked == 0

    @pytest.mark.asyncio
    async def test_only_tool_results_are_considered(self):
        built = runtime([0.0])
        response = BaseEvent(event_type=ExecutionEventType.RESPONSE, author="agent")
        await decide(built, history=[response], task="x", state={})
        assert built.provider.asked == 0

    @pytest.mark.asyncio
    async def test_the_model_sees_a_preview_not_the_whole_output(self):
        """The full outputs are what compaction is trying to get rid of."""
        built = runtime([1.0])
        await decide(built, history=history(50_000), task="x", state={})
        assert built.provider.state is not None
        assert len(built.provider.state) < 5_000


class TestEverythingUncertainChangesNothing:
    @pytest.mark.asyncio
    async def test_with_no_runtime(self):
        assert await decide(None, history=history(5_000), task="x", state={}) == set()

    @pytest.mark.asyncio
    async def test_with_the_use_case_off(self):
        built = runtime([0.0], mode=DecisionMode.OFF)
        assert await decide(built, history=history(5_000), task="x", state={}) == set()
        assert built.provider.asked == 0

    @pytest.mark.asyncio
    async def test_in_shadow_it_measures_and_marks_nothing(self):
        built = runtime([0.0], mode=DecisionMode.SHADOW)
        state: dict = {}
        assert await decide(built, history=history(5_000), task="x", state=state) == set()
        assert built.provider.asked == 1
        assert irrelevant_ids(state) == set()

    @pytest.mark.asyncio
    async def test_with_no_task(self):
        assert await decide(runtime([0.0]), history=history(5_000), task="", state={}) == set()

    @pytest.mark.asyncio
    async def test_when_the_model_will_not_answer(self):
        class Broken(DecisionProvider):
            type_name = "broken"
            provenance = Provenance.NATIVE

            async def _decide(self, state, questions):
                raise RuntimeError("down")

        built = DecisionRuntime(
            Broken(), {"context_ladder": UseCasePolicy(DecisionMode.ACTIVE, 0.9)}
        )
        assert await decide(built, history=history(5_000), task="x", state={}) == set()

    def test_rubbish_in_the_state_marks_nothing(self):
        assert irrelevant_ids({"context_irrelevant": "not a list"}) == set()
        assert irrelevant_ids(None) == set()


class TestCompactionUsesIt:
    @pytest.mark.asyncio
    async def test_a_marked_output_is_pruned_whatever_its_size(self):
        """The whole point: an output under the size threshold, which the age
        and size rule would have kept, goes because the task moved on."""
        from domyn.agents.llm import LLMMessage
        from domyn.agents.planner_strategy import HeuristicEstimator
        from domyn.agents.planner_strategy.context_management.compaction import (
            Budget,
            CompactionPolicy,
            PruneOnlyCompaction,
        )

        small = "y" * 500
        messages = [
            LLMMessage(role="user", content="do the thing"),
            LLMMessage(role="assistant", content="ok"),
            *[LLMMessage(role="user", content=f"turn {i}") for i in range(6)],
        ]
        marked = LLMMessage(role="tool", content=small)
        marked.source_event_id = "e0"
        messages.insert(2, marked)

        policy = CompactionPolicy()
        policy.prune_tool_results_over_chars = 10_000  # size rule would keep it
        budget = Budget(
            context_window=100,
            reserve_tokens=90,
            current_tokens=1_000,
            estimator=HeuristicEstimator(),
        )
        outcome = await PruneOnlyCompaction().compact(
            messages,
            budget=budget,
            policy=policy,
            state={"context_irrelevant": ["e0"]},
        )

        assert outcome.changed
        assert "e0" in outcome.pruned_event_ids
        assert not any(small in str(message.content) for message in outcome.messages)

    @pytest.mark.asyncio
    async def test_with_nothing_marked_it_behaves_exactly_as_before(self):
        from domyn.agents.llm import LLMMessage
        from domyn.agents.planner_strategy import HeuristicEstimator
        from domyn.agents.planner_strategy.context_management.compaction import (
            Budget,
            CompactionPolicy,
            PruneOnlyCompaction,
        )

        messages = [LLMMessage(role="user", content=f"turn {i}") for i in range(8)]
        budget = Budget(
            context_window=100,
            reserve_tokens=90,
            current_tokens=1_000,
            estimator=HeuristicEstimator(),
        )
        outcome = await PruneOnlyCompaction().compact(
            messages, budget=budget, policy=CompactionPolicy(), state={}
        )
        assert outcome.changed is False
