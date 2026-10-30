"""The durable half: a compaction that outlives the turn that performed it.

Without a record, `prune_only` recomputes from the freshly rendered messages and
acts only above the threshold. That is fine for the automatic path — history only
grows — but it makes a **manual** compaction a placebo: prune at 30% of the
window, and the next turn re-renders everything, finds itself below the
threshold, prunes nothing, and the text is back.

So these tests are mostly about survival across a re-render, which is the thing
that is easy to believe works and easy to ship broken.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from domyn.agents.core import BaseEvent, ExecutionEventType, Part, ToolAction
from domyn.agents.llm import LLMMessage, LLMToolCall, ToolCallFunction
from domyn.agents.planner_strategy.context_management.budget import (
    CONTEXT_USAGE_METADATA_KEY,
    ContextUsage,
)
from domyn.agents.planner_strategy.context_management.compaction import (
    CompactionPlan,
    CompactionPolicy,
    Compactor,
    NoOpCompaction,
    PruneOnlyCompaction,
    add_to_record,
    read_record,
)

AGENT = "Assistant"


def _tool_end(index: int, *, size: int = 80, event_id: str | None = None) -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.TOOL_END,
        author=AGENT,
        event_id=event_id or f"evt_{index}",
        action=ToolAction(
            name="read_file",
            call_id=f"c{index}",
            parameters={},
            observation="exit code: 0\n" + f"line of file {index}\n" * size,
        ),
    )


def _rendered(count: int, *, size: int = 80) -> list[LLMMessage]:
    """What the context manager builds from that history, every single turn.

    Shaped like a real conversation — user, the call, its result, the reply —
    rather than a run of bare tool results. That matters here: the splitter
    retreats off any boundary that would separate a call from its result, and a
    list of nothing but results makes it retreat all the way to the start.
    """
    messages = [LLMMessage(role="system", content="sys")]
    for index in range(count):
        messages.append(LLMMessage(role="user", content=f"read file {index}"))
        messages.append(
            LLMMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    LLMToolCall(
                        id=f"c{index}",
                        type="function",
                        function=ToolCallFunction(name="read_file", arguments="{}"),
                    )
                ],
            )
        )
        messages.append(
            LLMMessage(
                role="tool",
                tool_call_id=f"c{index}",
                name="read_file",
                source_event_id=f"evt_{index}",
                content="exit code: 0\n" + f"line of file {index}\n" * size,
            )
        )
        messages.append(LLMMessage(role="assistant", content=f"file {index} read"))
    return messages


class TestTheRecordItself:
    def test_an_absent_record_reads_as_empty(self) -> None:
        assert read_record(None, AGENT).pruned_event_ids == []
        assert read_record({}, AGENT).pruned_event_ids == []

    def test_ids_survive_a_round_trip_through_state(self) -> None:
        """State is plain dicts because the runner persists it — the record has to
        come back from that, not just from memory."""
        state: dict[str, Any] = {}
        add_to_record(state, AGENT, ["evt_1", "evt_2"])
        assert read_record(state, AGENT).pruned_event_ids == ["evt_1", "evt_2"]

    def test_adding_the_same_id_twice_changes_nothing(self) -> None:
        state: dict[str, Any] = {}
        add_to_record(state, AGENT, ["evt_1"])
        add_to_record(state, AGENT, ["evt_1"])
        record = read_record(state, AGENT)
        assert record.pruned_event_ids == ["evt_1"]
        assert record.generation == 1, "a no-op must not look like a change"

    def test_the_generation_moves_only_on_a_real_change(self) -> None:
        state: dict[str, Any] = {}
        assert read_record(state, AGENT).generation == 0
        add_to_record(state, AGENT, ["evt_1"])
        assert read_record(state, AGENT).generation == 1
        add_to_record(state, AGENT, [])
        assert read_record(state, AGENT).generation == 1

    def test_empty_and_falsy_ids_are_ignored(self) -> None:
        state: dict[str, Any] = {}
        add_to_record(state, AGENT, ["", None])  # type: ignore[list-item]
        assert read_record(state, AGENT).pruned_event_ids == []

    def test_two_agents_keep_separate_records(self) -> None:
        """Not tidiness: `context_visibility` means two agents can see different
        slices of one history, so "what has been dropped" is not shared."""
        state: dict[str, Any] = {}
        add_to_record(state, "Assistant", ["evt_1"])
        add_to_record(state, "Researcher", ["evt_9"])
        assert read_record(state, "Assistant").pruned_event_ids == ["evt_1"]
        assert read_record(state, "Researcher").pruned_event_ids == ["evt_9"]

    def test_a_corrupt_record_is_discarded_not_raised(self) -> None:
        """Losing it costs a re-prune. Raising would make the session unusable,
        and this runs on every single render."""
        state: dict[str, Any] = {"compaction_records": {AGENT: {"pruned_event_ids": "not-a-list"}}}
        assert read_record(state, AGENT).pruned_event_ids == []

    def test_a_record_for_another_shape_of_state_does_not_crash(self) -> None:
        assert read_record({"compaction_records": None}, AGENT).pruned_event_ids == []


class TestReplayAcrossARerender:
    def test_a_recorded_drop_is_reapplied_to_a_full_rerender(self) -> None:
        """The whole point. The message list comes back with the text restored;
        replay has to remove it again without being asked."""
        state: dict[str, Any] = {}
        compactor = Compactor()
        compactor.request_from_history([_tool_end(i) for i in range(6)], state, AGENT)

        messages = _rendered(6)
        before = compactor.estimator.estimate_messages(messages)
        after = compactor.estimator.estimate_messages(compactor.replay(messages, state, AGENT))
        assert after < before / 1.5

    def test_replay_is_idempotent(self) -> None:
        state: dict[str, Any] = {}
        compactor = Compactor()
        add_to_record(state, AGENT, ["evt_0", "evt_1"])
        once = compactor.replay(_rendered(4), state, AGENT)
        twice = compactor.replay(once, state, AGENT)
        assert [m.content for m in once] == [m.content for m in twice]

    def test_an_empty_record_returns_the_very_same_list(self) -> None:
        """This runs before every request: it has to cost nothing in the common
        case."""
        compactor = Compactor()
        messages = _rendered(3)
        assert compactor.replay(messages, {}, AGENT) is messages

    def test_replay_ignores_another_agents_record(self) -> None:
        state: dict[str, Any] = {}
        add_to_record(state, "Researcher", ["evt_0", "evt_1"])
        compactor = Compactor()
        messages = _rendered(4)
        assert compactor.replay(messages, state, AGENT) is messages

    def test_size_no_longer_matters_once_recorded(self) -> None:
        """The question changes from "is this big enough to drop" to "was this
        dropped". Asking the first one again would restore anything that shrank."""
        state: dict[str, Any] = {}
        add_to_record(state, AGENT, ["evt_0"])
        small = [
            LLMMessage(
                role="tool",
                tool_call_id="c0",
                name="read_file",
                source_event_id="evt_0",
                content="ok",
            )
        ]
        assert "dropped" in str(Compactor().replay(small, state, AGENT)[0].content)

    def test_the_record_outlives_a_change_of_strategy(self) -> None:
        """Un-pruning would restore text the agent has already been told is gone."""
        state: dict[str, Any] = {}
        add_to_record(state, AGENT, ["evt_0", "evt_1"])
        inert = Compactor(strategy=NoOpCompaction())
        replayed = inert.replay(_rendered(4), state, AGENT)
        assert sum("dropped" in str(m.content) for m in replayed) == 2

    def test_the_originals_are_not_modified(self) -> None:
        state: dict[str, Any] = {}
        add_to_record(state, AGENT, ["evt_0"])
        messages = _rendered(2)
        original = str(messages[1].content)
        Compactor().replay(messages, state, AGENT)
        assert str(messages[1].content) == original


class TestCompactingOnRequest:
    def test_it_records_the_older_results_and_says_how_many(self) -> None:
        state: dict[str, Any] = {}
        event = Compactor().request_from_history([_tool_end(i) for i in range(6)], state, AGENT)
        assert event.event_type is ExecutionEventType.CONTEXT_COMPACTED
        assert event.metadata["changed"] is True
        assert event.metadata["events_dropped"] == 3
        assert read_record(state, AGENT).pruned_event_ids == ["evt_0", "evt_1", "evt_2"]

    def test_the_most_recent_results_are_left_alone(self) -> None:
        """The last couple are usually what the agent is reasoning about; dropping
        them is how a compaction turns into amnesia."""
        state: dict[str, Any] = {}
        Compactor().request_from_history([_tool_end(i) for i in range(6)], state, AGENT)
        assert "evt_5" not in read_record(state, AGENT).pruned_event_ids

    def test_how_many_are_kept_is_configurable(self) -> None:
        state: dict[str, Any] = {}
        compactor = Compactor(policy=CompactionPolicy(keep_recent_tool_results=5))
        compactor.request_from_history([_tool_end(i) for i in range(6)], state, AGENT)
        assert read_record(state, AGENT).pruned_event_ids == ["evt_0"]

    def test_asking_twice_says_there_is_nothing_left(self) -> None:
        """A command that answers nothing looks broken."""
        state: dict[str, Any] = {}
        compactor = Compactor()
        history = [_tool_end(i) for i in range(6)]
        compactor.request_from_history(history, state, AGENT)
        second = compactor.request_from_history(history, state, AGENT)
        assert second.metadata["changed"] is False
        assert "already" in str(second.content[0].text)

    def test_asking_with_nothing_to_drop_says_that_instead(self) -> None:
        state: dict[str, Any] = {}
        event = Compactor().request_from_history([], state, AGENT)
        assert event.metadata["changed"] is False
        assert "nothing to compact" in str(event.content[0].text)

    def test_small_results_are_not_worth_dropping(self) -> None:
        state: dict[str, Any] = {}
        history = [_tool_end(i, size=1) for i in range(6)]
        event = Compactor().request_from_history(history, state, AGENT)
        assert event.metadata["changed"] is False

    def test_the_size_is_read_where_the_payload_actually_lives(self) -> None:
        """A tool result keeps its text in `action.observation`, not in `content`.
        Measuring `content` finds almost nothing and the command concludes there
        was never anything to drop."""
        state: dict[str, Any] = {}
        history = [_tool_end(i) for i in range(6)]
        assert all(not event.content for event in history), "the fixture must have empty content"
        assert Compactor().request_from_history(history, state, AGENT).metadata["changed"] is True

    def test_events_without_an_id_are_skipped(self) -> None:
        state: dict[str, Any] = {}
        history = [_tool_end(0)]
        history[0].event_id = None
        assert Compactor().request_from_history(history, state, AGENT).metadata["changed"] is False

    def test_it_does_not_touch_the_history_it_reads(self) -> None:
        """The invariant the design rests on: events are never rewritten, only a
        note about them is kept."""
        state: dict[str, Any] = {}
        history = [_tool_end(i) for i in range(6)]
        before = [str(event.action.observation) for event in history]
        Compactor().request_from_history(history, state, AGENT)
        assert [str(event.action.observation) for event in history] == before

    def test_it_works_with_an_inert_strategy(self) -> None:
        """Asking explicitly is not the same as configuring an automatic policy:
        a `noop` strategy means "do not compact on your own", not "refuse"."""
        state: dict[str, Any] = {}
        compactor = Compactor(strategy=NoOpCompaction())
        event = compactor.request_from_history([_tool_end(i) for i in range(6)], state, AGENT)
        assert event.metadata["changed"] is True

    def test_the_saving_comes_off_the_number_the_indicator_showed(self) -> None:
        """The CLI takes `tokens_saved` off its last measurement, which includes the
        correction measured on the text: the saving has to include the share of it that
        went with the dropped outputs, or the indicator stays high until the next
        request."""
        compactor = Compactor(strategy=PruneOnlyCompaction())
        rendered = _rendered(6, size=400)
        agent = SimpleNamespace(name=AGENT, llm_provider=SimpleNamespace(context_window=None))
        state: dict[str, Any] = {}

        def shown(messages: list[LLMMessage]) -> int:
            metadata = compactor.usage_metadata(messages, None, agent, state)
            return ContextUsage.model_validate(metadata[CONTEXT_USAGE_METADATA_KEY]).total_tokens

        sent = compactor.usage_metadata(rendered, None, agent, state)
        compactor.observe({"prompt_tokens": int(shown(rendered) * 2.4)}, sent, state, AGENT)
        before = shown(rendered)

        history = [_tool_end(i, size=400) for i in range(6)]
        event = compactor.request_from_history(history, state, AGENT, rendered=rendered)

        assert event.metadata["changed"] is True
        after = shown(compactor.replay(rendered, state, AGENT))
        assert after == before - event.metadata["tokens_saved"]


class TestTheAutomaticPathRecordsToo:
    @pytest.mark.asyncio
    async def test_what_the_threshold_dropped_is_remembered(self) -> None:
        """Otherwise the automatic path reverses itself: the decision is
        recomputed each turn and undone the moment the conversation dips back
        below the threshold."""

        class _Agent:
            name = AGENT

            class llm_provider:  # noqa: N801
                context_window = 400

        state: dict[str, Any] = {}
        compactor = Compactor(
            strategy=PruneOnlyCompaction(),
            policy=CompactionPolicy(reserve_tokens=1, keep_recent_tokens=1),
        )
        messages = _rendered(6)
        plan = compactor.assess(messages, _Agent())
        assert plan.needed is True, plan.reason

        await compactor.run(messages, plan, state, AGENT)
        assert read_record(state, AGENT).pruned_event_ids, "the threshold path recorded nothing"

    @pytest.mark.asyncio
    async def test_a_message_with_no_source_event_is_pruned_but_not_recorded(self) -> None:
        """Count and record are different numbers, and pretending otherwise would
        imply the record covers everything."""
        from domyn.agents.planner_strategy.context_management.compaction import ToolResultPruner

        message = LLMMessage(role="tool", tool_call_id="c", name="t", content="x" * 5_000)
        _, count, ids = ToolResultPruner().prune_reporting([message], over_chars=200)
        assert count == 1
        assert ids == []


class TestSayingWhetherItWasEnough:
    """ "I pruned something" and "the request will now fit" are different claims.

    Only the second is what anyone wanted. Past a point `prune_only` always finds
    one fresh tool output to drop while the total keeps climbing — measured: from
    turn 202 of a heavy session it fires every turn, saves ~6k, and the context
    grows anyway. Reporting that as success is how a session dies on a rejected
    request immediately after being told the context was compacted.
    """

    class _Agent:
        name = AGENT

        class llm_provider:  # noqa: N801
            context_window = 2_000

    @pytest.mark.asyncio
    async def test_getting_under_the_limit_is_reported_as_sufficient(self) -> None:
        compactor = Compactor(policy=CompactionPolicy(reserve_tokens=500, keep_recent_tokens=200))
        messages = _rendered(6, size=40)
        plan = compactor.assess(messages, self._Agent())
        assert plan.needed is True, plan.reason

        _, done = await compactor.run(messages, plan, {}, AGENT)
        assert done.metadata["changed"] is True
        assert done.metadata["sufficient"] is True
        assert done.metadata["tokens_after"] <= done.metadata["threshold_tokens"]

    @pytest.mark.asyncio
    async def test_pruning_that_does_not_get_under_the_limit_says_so(self) -> None:
        """The case that used to report success: plenty pruned, still too big."""

        class _Tiny:
            name = AGENT

            class llm_provider:  # noqa: N801
                context_window = 900

        compactor = Compactor(policy=CompactionPolicy(reserve_tokens=500, keep_recent_tokens=200))
        messages = _rendered(30, size=60)
        plan = compactor.assess(messages, _Tiny())
        assert plan.needed is True

        _, done = await compactor.run(messages, plan, {}, AGENT)
        assert done.metadata["changed"] is True, "the fixture must actually prune something"
        assert done.metadata["sufficient"] is False
        assert done.metadata["tokens_after"] > done.metadata["threshold_tokens"]

    @pytest.mark.asyncio
    async def test_the_threshold_travels_with_the_event(self) -> None:
        """So a front-end can say how far over it still is without recomputing."""
        compactor = Compactor(policy=CompactionPolicy(reserve_tokens=500, keep_recent_tokens=200))
        messages = _rendered(6, size=40)
        plan = compactor.assess(messages, self._Agent())
        _, done = await compactor.run(messages, plan, {}, AGENT)
        assert done.metadata["threshold_tokens"] == 2_000 - 500

    @pytest.mark.asyncio
    async def test_with_no_threshold_it_does_not_claim_a_failure(self) -> None:
        """No window means no limit to be under. Reporting insufficiency there
        would be inventing a failure out of a missing number."""
        compactor = Compactor()
        messages = _rendered(4)
        plan = CompactionPlan(needed=False)
        _, done = await compactor.run(messages, plan, {}, AGENT)
        assert done.metadata["sufficient"] is True
