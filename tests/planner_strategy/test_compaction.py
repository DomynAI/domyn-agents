"""Tests for firing at the window limit, and for what happens when it fires.

Three groups, in order of how much getting them wrong costs:

1. **when it must NOT fire** — an unrequested intervention loses information
   nobody asked to lose;
2. **where it cuts** — a cut between a tool call and its result produces a
   malformed conversation that providers reject;
3. **what it does when it acts** — deterministic, and without ever calling a
   model.
"""

import re
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
import respx

from domyn.agents.core import (
    AgentAction,
    BaseEvent,
    ExecutionEventType,
    InteractionContext,
    Part,
    ToolAction,
)
from domyn.agents.llm import LLMMessage, LLMToolCall, ToolCallFunction, VLLMProvider
from domyn.agents.llm.context_window import resolve_context_window
from domyn.agents.planner_strategy import HeuristicEstimator, ToolUseContextManager
from domyn.agents.planner_strategy.context_management.compaction import (
    Budget,
    CompactionOutcome,
    CompactionPolicy,
    CompactionStrategy,
    Compactor,
    NoOpCompaction,
    PruneOnlyCompaction,
    TokenBudgetSplitter,
    ToolResultPruner,
    add_to_record,
    build_strategy,
    compaction_strategy,
)
from domyn.agents.tools.read_tool_result import read_tool_result


class _Provider:
    def __init__(self, window: int | None) -> None:
        self.context_window = window


class _Agent:
    def __init__(self, window: int | None) -> None:
        self.llm_provider = _Provider(window)
        self.name = "Assistant"


def _msg(role: str, text: str, **kw: Any) -> LLMMessage:
    return LLMMessage(role=role, content=text, **kw)


def _call(name: str = "run_command") -> LLMToolCall:
    return LLMToolCall(
        id="call_1", type="function", function=ToolCallFunction(name=name, arguments="{}")
    )


def _long(chars: int, marker: str = "x") -> str:
    return marker * chars


def _conversation(turns: int, result_chars: int = 4_000) -> list[LLMMessage]:
    """A plausible session: system, then turns with one tool call each."""
    messages = [_msg("system", "instructions")]
    for i in range(turns):
        messages.append(_msg("user", f"question {i}"))
        messages.append(LLMMessage(role="assistant", content=None, tool_calls=[_call()]))
        messages.append(
            LLMMessage(
                role="tool",
                content=f"exit code: 0\n{_long(result_chars)}",
                tool_call_id="call_1",
                name="run_command",
                source_event_id=f"evt_{i}",
            )
        )
        messages.append(_msg("assistant", f"answer {i}"))
    return messages


def _active(**policy: Any) -> Compactor:
    """A compactor that will actually act, for tests about anything else.

    Explicit because the default strategy is inert: with it, `assess` short-
    circuits on "no strategy" and a test meaning to exercise the window or the
    threshold would silently measure something else.
    """
    return Compactor(strategy=PruneOnlyCompaction(), policy=CompactionPolicy(**policy))


class TestWhenItMustNotFire:
    """Every case here is an intervention that would be a defect."""

    def test_unknown_window(self) -> None:
        """The case of every provider that does not declare one — that is, today,
        on vLLM. An invented threshold would compress when it need not."""
        plan = _active().assess(_conversation(50), _Agent(None))
        assert plan.needed is False
        assert plan.reason and "unknown" in plan.reason

    def test_disabled_by_configuration(self) -> None:
        compactor = Compactor(
            strategy=PruneOnlyCompaction(), policy=CompactionPolicy(enabled=False)
        )
        assert compactor.assess(_conversation(50), _Agent(1_000)).needed is False

    def test_below_threshold(self) -> None:
        plan = _active(reserve_tokens=1_000).assess(
            _conversation(1, result_chars=100), _Agent(100_000)
        )
        assert plan.needed is False

    def test_exactly_on_the_threshold_does_not_fire(self) -> None:
        """`>` and not `>=`: on the exact threshold the request still fits."""
        messages = [_msg("user", _long(300))]
        exact = HeuristicEstimator().estimate_messages(messages)
        plan = _active(reserve_tokens=1_000).assess(messages, _Agent(exact + 1_000))
        assert plan.threshold_tokens == exact
        assert plan.needed is False

    def test_one_token_over_the_threshold_fires(self) -> None:
        messages = [_msg("user", _long(300))]
        exact = HeuristicEstimator().estimate_messages(messages)
        plan = _active(reserve_tokens=1_000).assess(messages, _Agent(exact + 1_000 - 1))
        assert plan.needed is True

    def test_what_travels_with_the_conversation_counts_toward_the_threshold(self) -> None:
        """Tool definitions and the turn's injected context go out with every
        request. Leaving them out of the decision is how a CLI said "compaction
        in ~0 tokens" while the next request went out uncompacted."""
        messages = [_msg("user", _long(300))]
        exact = HeuristicEstimator().estimate_messages(messages)
        agent = _Agent(exact + 1_000)  # the conversation alone sits on the threshold
        compactor = _active(reserve_tokens=1_000)
        assert compactor.assess(messages, agent).needed is False

        skills = [_msg("assistant", "the skills index")]
        tools = [{"type": "function", "function": {"name": "t", "description": "d"}}]
        assert compactor.assess(messages, agent, turn_context=skills).needed is True
        assert compactor.assess(messages, agent, tool_defs=tools).needed is True

    def test_a_long_conversation_on_a_large_window_does_not_fire(self) -> None:
        """Proof that in practice nobody notices: fifty turns with 4k-character
        outputs sit comfortably inside 200k tokens."""
        plan = _active().assess(_conversation(50), _Agent(200_000))
        assert plan.needed is False, f"{plan.current_tokens} tokens crossed the threshold"


class TestWhereItCuts:
    def test_never_between_a_call_and_its_result(self) -> None:
        """The constraint that dominates all others: a model handed the result of
        a call it cannot see receives a malformed conversation, and providers
        reject it."""
        splitter = TokenBudgetSplitter()
        estimator = HeuristicEstimator()
        for keep in range(200, 4_000, 137):
            split = splitter.split(
                _conversation(12, result_chars=500),
                policy=CompactionPolicy(keep_recent_tokens=keep),
                estimator=estimator,
            )
            if not split.tail:
                continue
            first = split.tail[0]
            assert not (first.role == "tool" or first.tool_call_id), (
                f"with keep={keep} the tail starts with an orphaned result"
            )
            if split.middle:
                assert not split.middle[-1].tool_calls, (
                    f"with keep={keep} the middle ends with an orphaned call"
                )

    def test_the_head_is_protected(self) -> None:
        split = TokenBudgetSplitter().split(
            _conversation(20),
            policy=CompactionPolicy(keep_recent_tokens=100, protect_first_n=1),
            estimator=HeuristicEstimator(),
        )
        assert split.head[0].role == "system"
        assert len(split.head) == 2  # system + first message

    def test_nothing_to_cut_means_no_middle(self) -> None:
        split = TokenBudgetSplitter().split(
            [_msg("system", "s"), _msg("user", "hello")],
            policy=CompactionPolicy(),
            estimator=HeuristicEstimator(),
        )
        assert split.has_middle is False

    def test_an_empty_list(self) -> None:
        split = TokenBudgetSplitter().split(
            [], policy=CompactionPolicy(), estimator=HeuristicEstimator()
        )
        assert (split.head, split.middle, split.tail) == ([], [], [])


class TestWhatItDoesWhenItActs:
    @pytest.mark.asyncio
    async def test_prune_only_reduces_the_tokens(self) -> None:
        messages = _conversation(20, result_chars=6_000)
        compactor = _active(reserve_tokens=1_000)
        plan = compactor.assess(messages, _Agent(20_000))
        assert plan.needed
        outcome = await compactor.apply(messages, plan)
        assert outcome.changed
        assert compactor.tokens_after(outcome.messages) < plan.current_tokens

    @pytest.mark.asyncio
    async def test_prune_only_calls_no_model(self, respx_mock: Any) -> None:
        """The property that makes this strategy fit for the critical path:
        `respx` records no outgoing request at all."""
        messages = _conversation(20, result_chars=6_000)
        compactor = _active(reserve_tokens=1_000)
        plan = compactor.assess(messages, _Agent(20_000))
        await compactor.apply(messages, plan)
        assert len(respx_mock.calls) == 0

    @pytest.mark.asyncio
    async def test_it_does_not_touch_the_tail(self) -> None:
        messages = _conversation(20, result_chars=6_000)
        compactor = _active(reserve_tokens=1_000, keep_recent_tokens=5_000)
        plan = compactor.assess(messages, _Agent(20_000))
        outcome = await compactor.apply(messages, plan)
        last_result = [m for m in outcome.messages if m.role == "tool"][-1]
        assert "dropped to save context" not in str(last_result.content)

    @pytest.mark.asyncio
    async def test_the_originals_are_not_modified(self) -> None:
        """Additive: a copy is shortened, the source stays whole."""
        messages = _conversation(20, result_chars=6_000)
        before = [str(m.content) for m in messages]
        compactor = _active(reserve_tokens=1_000)
        plan = compactor.assess(messages, _Agent(20_000))
        await compactor.apply(messages, plan)
        assert [str(m.content) for m in messages] == before

    @staticmethod
    def _dropped_message() -> LLMMessage:
        return LLMMessage(
            role="tool",
            content="exit code: 0\n" + _long(5_000),
            tool_call_id="c",
            name="run_command",
            source_event_id="evt_42",
        )

    def test_the_placeholder_says_what_it_was(self) -> None:
        """A bare `[removed]` would make the model re-run the command."""
        pruned, count = ToolResultPruner().prune([self._dropped_message()], over_chars=200)
        assert count == 1
        text = str(pruned[0].content)
        assert "run_command" in text
        assert "exit 0" in text
        assert "2 lines" in text

    @pytest.mark.asyncio
    async def test_a_placeholder_offers_a_read_only_where_one_gives_back_what_it_replaced(
        self,
    ) -> None:
        """Rendered by the context manager, then pruned. A tool's result gives the
        call that reads it again, and a read pruned in turn gives that same read.
        The other tool-role messages, a sub-agent's task and its report, have no
        result to read: a call there could only fail."""
        agent = MagicMock()
        agent.name = "Parent"
        agent.instruction = "Delegate."
        agent.tools = [MagicMock()]
        agent.tools[0].name = "dump"
        agent.planner.use_stop = True  # a sub-agent's report comes as a STOP result
        agent.planner.forward_verbatim_enabled = False
        manager = ToolUseContextManager()
        rows = "".join(f"row {i:04d}\n" for i in range(300))
        delegation = AgentAction(name="Child", call_id="call_child", parameters={"task": rows})
        events = [
            BaseEvent(event_type=ExecutionEventType.USER_INPUT, author="user"),
            BaseEvent(
                event_type=ExecutionEventType.TOOL_START,
                author="Parent",
                action=ToolAction(call_id="call_dump", name="dump", parameters={}),
            ),
            BaseEvent(
                event_type=ExecutionEventType.TOOL_END,
                author="Parent",
                event_id="event_dump",
                action=ToolAction(
                    call_id="call_dump", name="dump", parameters={}, observation=rows
                ),
            ),
        ]

        def context() -> InteractionContext:
            return InteractionContext(
                conversation_id="c",
                user_id="u",
                agent_input=BaseEvent(event_type=ExecutionEventType.USER_INPUT, author="user"),
                conversation_history=list(events),
            )

        read = {"result_id": "event_dump", "offset": 1_000}
        window = read_tool_result(**read, tool_results=manager.tool_results(agent, context()))
        events += [
            BaseEvent(
                event_type=ExecutionEventType.TOOL_START,
                author="Parent",
                action=ToolAction(call_id="call_read", name="read_tool_result", parameters=read),
            ),
            BaseEvent(
                event_type=ExecutionEventType.TOOL_END,
                author="Parent",
                event_id="event_read",
                action=ToolAction(
                    call_id="call_read",
                    name="read_tool_result",
                    parameters=read,
                    observation=window,
                ),
            ),
            BaseEvent(event_type=ExecutionEventType.AGENT_CALL, author="Parent", action=delegation),
            BaseEvent(
                event_type=ExecutionEventType.AGENT_START,
                author="Child",
                action=delegation,
                content=[Part(text=rows)],
            ),
            BaseEvent(
                event_type=ExecutionEventType.AGENT_END, author="Child", content=[Part(text=rows)]
            ),
        ]

        messages = await manager.get_context(context(), agent)
        pruned, _ = ToolResultPruner().prune(messages, over_chars=200)

        results = manager.tool_results(agent, context())
        offered: dict[str, bool] = {}
        for original, placeholder in zip(messages, pruned, strict=True):
            if placeholder is original:
                continue
            call = re.search(
                r'read_tool_result\(result_id="([^"]+)"(?:, offset=(\d+))?\)',
                str(placeholder.content),
            )
            offered[str(original.name)] = call is not None
            if call:
                again = read_tool_result(
                    call.group(1), offset=int(call.group(2) or 0), tool_results=results
                )
                if original.name == "read_tool_result":
                    assert again == original.content, "the same read gives the same text"
                else:
                    assert again.startswith(str(original.content))
        assert offered == {"dump": True, "read_tool_result": True, "Child": False, "STOP": False}

    def test_it_does_not_prune_small_results(self) -> None:
        message = LLMMessage(role="tool", content="ok", tool_call_id="c", name="t")
        _, count = ToolResultPruner().prune([message], over_chars=200)
        assert count == 0

    def test_pruning_is_idempotent(self) -> None:
        """The placeholder is short, so a second pass leaves it alone."""
        messages = _conversation(5, result_chars=5_000)
        once, first = ToolResultPruner().prune(messages, over_chars=200)
        twice, second = ToolResultPruner().prune(once, over_chars=200)
        assert first > 0 and second == 0
        assert [str(m.content) for m in once] == [str(m.content) for m in twice]

    @pytest.mark.asyncio
    async def test_nothing_to_prune_means_no_change(self) -> None:
        """`changed=False` rather than a "changed" list identical to the input:
        otherwise the CLI would announce a recovery that never happened."""
        messages = [_msg("system", "s")] + [_msg("user", _long(400)) for _ in range(30)]
        compactor = _active(reserve_tokens=100)
        plan = compactor.assess(messages, _Agent(1_000))
        outcome = await compactor.apply(messages, plan)
        assert outcome.changed is False


class TestFailingWithoutBreaking:
    @pytest.mark.asyncio
    async def test_a_raising_strategy_does_not_kill_the_turn(self) -> None:
        """An oversized request is a smaller problem than a dead turn: the
        provider will reject it with a readable error."""

        class Broken(CompactionStrategy):
            async def compact(self, messages, **kwargs):  # type: ignore[no-untyped-def]
                raise RuntimeError("the strategy is broken")

        messages = _conversation(20, result_chars=6_000)
        compactor = Compactor(strategy=Broken(), policy=CompactionPolicy(reserve_tokens=1_000))
        plan = compactor.assess(messages, _Agent(20_000))
        outcome = await compactor.apply(messages, plan)
        assert outcome.changed is False
        assert outcome.messages == messages

    @pytest.mark.asyncio
    async def test_noop_changes_nothing(self) -> None:
        messages = _conversation(20)
        compactor = Compactor(strategy=NoOpCompaction(), policy=CompactionPolicy(reserve_tokens=1))
        plan = compactor.assess(messages, _Agent(100))
        outcome = await compactor.apply(messages, plan)
        assert outcome.messages == messages and outcome.changed is False


class TestReplacingTheStrategy:
    """The test that matters more than the others: a foreign strategy, injected
    by name, is invoked and its result arrives intact."""

    @pytest.mark.asyncio
    async def test_a_foreign_strategy_is_invoked(self) -> None:
        called: list[bool] = []

        @compaction_strategy("test_only")
        class TestOnly(CompactionStrategy):
            async def compact(self, messages, **kwargs):  # type: ignore[no-untyped-def]
                called.append(True)
                return CompactionOutcome(
                    messages=[LLMMessage(role="user", content="all replaced")],
                    changed=True,
                    note="done my way",
                )

        compactor = Compactor(
            strategy=build_strategy("test_only"), policy=CompactionPolicy(reserve_tokens=1)
        )
        plan = compactor.assess(_conversation(3), _Agent(10))
        outcome = await compactor.apply(_conversation(3), plan)

        assert called, "the injected strategy was not invoked"
        assert len(outcome.messages) == 1
        assert outcome.note == "done my way"

    def test_an_unknown_name_raises(self) -> None:
        """Falling back to `noop` would give someone who mistyped a line of
        configuration the illusion of an active compaction."""
        with pytest.raises(ValueError, match="unknown compaction strategy"):
            build_strategy("does_not_exist")

    def test_the_shipped_strategies_are_registered(self) -> None:
        assert isinstance(build_strategy("noop"), NoOpCompaction)
        assert isinstance(build_strategy("prune_only"), PruneOnlyCompaction)


class TestTheThreshold:
    def test_a_window_smaller_than_the_reserve_gives_a_zero_threshold(self) -> None:
        """That is, "always". It is the right reading: if there is no room for a
        reply, any conversation is too long."""
        budget = Budget(
            context_window=100,
            reserve_tokens=16_384,
            current_tokens=1,
            estimator=HeuristicEstimator(),
        )
        assert budget.threshold_tokens == 0
        assert budget.over_threshold is True


class TestDoNotTrustTheWindowValue:
    """The value comes from an attribute on a foreign object, and several things
    convert to an integer silently.

    This group exists because two tests fell over: their agents were `MagicMock`,
    `int(MagicMock())` is 1, and a one-token window makes every conversation too
    long. Not a production defect, but misplaced trust.
    """

    @pytest.mark.parametrize(
        "window",
        [None, 0, -1, True, False, "128000", 1.5, object()],
        ids=["none", "zero", "negative", "true", "false", "string", "float", "object"],
    )
    def test_a_non_integer_window_leaves_everything_inert(self, window: Any) -> None:
        agent = _Agent(None)
        agent.llm_provider.context_window = window
        plan = _active().assess(_conversation(50), agent)
        assert plan.needed is False
        assert plan.reason and "unknown" in plan.reason

    def test_a_test_double_does_not_switch_compaction_on(self) -> None:
        """The concrete case that broke two other tests: a fake agent must not
        make anything fire just because of how it is built."""
        from unittest.mock import MagicMock

        assert _active().assess(_conversation(10), MagicMock()).needed is False


class TestTheDefaultStrategy:
    """What a bare `Compactor()` does, which is what every planner mounts.

    It prunes. The opposite was pinned here while the default was inert, on the
    reasoning that a behaviour change should be opted into; that reasoning
    ignored what the alternative at the threshold actually is — not "leave the
    conversation alone" but a rejected request and a lost turn.
    """

    def test_the_default_acts(self) -> None:
        compactor = Compactor()
        assert compactor.active is True
        assert compactor.strategy_name == "two_phase"

    def test_the_default_prunes_before_it_spends_anything(self) -> None:
        """`two_phase` is `prune_only` plus a fallback. It is only defensible as a
        default because the model call needs the free phase to have failed first —
        so that ordering is pinned here, not left to the strategy's docstring."""
        from domyn.agents.planner_strategy.context_management.compaction import (
            TwoPhaseCompaction,
        )

        assert isinstance(Compactor().strategy, TwoPhaseCompaction)


class TestTheInertStrategy:
    """Opting out stays possible, and stays honest: a CLI announcing a compaction
    that will not happen makes its reader plan around an imaginary threshold."""

    def test_it_is_not_active_and_has_no_name(self) -> None:
        compactor = Compactor(strategy=NoOpCompaction())
        assert compactor.active is False
        assert compactor.strategy_name is None

    def test_it_says_why_it_would_do_nothing(self) -> None:
        plan = Compactor(strategy=NoOpCompaction()).assess(_conversation(50), _Agent(200_000))
        assert plan.needed is False
        assert plan.reason and "no compaction strategy" in plan.reason

    def test_the_strategy_reason_precedes_the_window_reason(self) -> None:
        """Without a strategy, configuring the window would change nothing: the
        message has to name the first thing to fix."""
        plan = Compactor(strategy=NoOpCompaction()).assess(_conversation(10), _Agent(None))
        assert plan.reason and "no compaction strategy" in plan.reason

    def test_a_strategy_that_acts_is_active_and_names_itself(self) -> None:
        compactor = Compactor(strategy=PruneOnlyCompaction())
        assert compactor.active is True
        assert compactor.strategy_name == "prune_only"


class TestOneLongTurnStillCompacts:
    """One request, then tool outputs past the tail budget: the shape of an
    agentic task. The tail used to stretch back to the request, so compaction
    found nothing to do exactly when one turn had filled the window."""

    @staticmethod
    def _long_turn(outputs: int) -> list[LLMMessage]:
        """An earlier exchange, then the last request — outside the protected
        head, where only the tail could have kept it — and its tool outputs."""
        messages = [
            _msg("system", "sys"),
            _msg("user", "an earlier question"),
            _msg("assistant", "an earlier answer"),
            _msg("user", "THE REQUEST"),
        ]
        for index in range(outputs):
            messages.append(LLMMessage(role="assistant", content=None, tool_calls=[_call()]))
            messages.append(
                LLMMessage(
                    role="tool",
                    content=f"exit code: 0\n{_long(6_000)}",
                    tool_call_id="call_1",
                    name="run_command",
                    source_event_id=f"evt_{index}",
                )
            )
        return messages

    @pytest.mark.asyncio
    async def test_its_older_outputs_are_pruned(self) -> None:
        messages = self._long_turn(8)
        compactor = _active(reserve_tokens=1_000)
        plan = compactor.assess(messages, _Agent(12_000))
        assert plan.needed, plan.reason

        outcome = await compactor.apply(messages, plan)

        assert outcome.changed, "the turn's own outputs were out of reach"
        assert "dropped to save context" not in str(outcome.messages[-1].content)
        assert any(m.content == "THE REQUEST" for m in outcome.messages)


class TestPruningDoesNotEatItsOwnPlaceholder:
    def test_a_second_pass_leaves_the_placeholder_intact(self) -> None:
        """Otherwise it writes a placeholder *about* the placeholder — "1 lines",
        no file name, no exit code — destroying the summary that made the drop
        tolerable in the first place."""
        message = LLMMessage(
            role="tool",
            content="exit code: 0\n" + _long(5_000),
            tool_call_id="c",
            name="run_command",
        )
        pruner = ToolResultPruner()
        once, _ = pruner.prune([message], over_chars=200)
        twice, count = pruner.prune(once, over_chars=1)
        assert count == 0
        assert str(twice[0].content) == str(once[0].content)
        assert "run_command" in str(twice[0].content)


def _total(metadata: dict[str, Any]) -> int:
    usage = metadata["context_usage"]
    return int(
        usage["message_tokens"] + usage["tool_definition_tokens"] + usage["correction_tokens"]
    )


def _measured(compactor: Compactor, messages: list[LLMMessage], reported: int) -> dict[str, Any]:
    """State after the provider counted ``reported`` tokens for ``messages``."""
    state: dict[str, Any] = {}
    sent = compactor.usage_metadata(messages, None, _Agent(None), state)
    compactor.observe({"prompt_tokens": reported}, sent, state, "Assistant")
    return state


class TestMeasuringFromTheProvidersCount:
    """The provider counts the request it answered; the next one is that count plus
    an estimate of what was added since. Measured live: chars/3 overcounts prose
    ~1.8x and undercounts UUIDs and hex ~2.4x, so either way the estimate alone
    puts the threshold in the wrong place."""

    def _setup(self, ratio: float) -> tuple[Compactor, dict[str, Any], list[LLMMessage], int]:
        compactor = _active(reserve_tokens=1_000)
        sent = _conversation(2)
        estimated = _total(compactor.usage_metadata(sent, None, _Agent(None)))
        state = _measured(compactor, sent, int(estimated * ratio))
        following = [*sent, _msg("user", "one more question")]
        return compactor, state, following, estimated

    def test_a_count_above_the_estimate_fires_where_the_estimate_would_not(self) -> None:
        compactor, state, following, estimated = self._setup(ratio=2.4)
        alone = _total(compactor.usage_metadata(following, None, _Agent(None)))
        agent = _Agent(1_000 + alone + estimated // 2)

        assert compactor.assess(following, agent).needed is False
        assert compactor.assess(following, agent, state=state).needed is True

    def test_a_count_below_the_estimate_holds_back_where_the_estimate_would_fire(self) -> None:
        compactor, state, following, estimated = self._setup(ratio=0.56)
        alone = _total(compactor.usage_metadata(following, None, _Agent(None)))
        agent = _Agent(1_000 + alone - estimated // 4)

        assert compactor.assess(following, agent).needed is True
        assert compactor.assess(following, agent, state=state).needed is False

    def test_the_indicator_and_the_threshold_still_read_one_number(self) -> None:
        compactor, state, following, _ = self._setup(ratio=2.4)
        agent = _Agent(1_000_000)
        shown = _total(compactor.usage_metadata(following, None, agent, state))
        assert shown == compactor.assess(following, agent, state=state).current_tokens

    def test_a_record_changed_without_carrying_the_count_retires_it(self) -> None:
        """The count described a rendering that changed, and nothing said how much of
        it went: a compaction carries it over, any other change leaves no base."""
        compactor, state, following, _ = self._setup(ratio=2.4)
        agent = _Agent(1_000_000)
        add_to_record(state, "Assistant", ["evt_0"])

        retired = compactor.assess(following, agent, state=state).current_tokens
        assert retired == compactor.assess(following, agent).current_tokens

    def test_a_count_on_another_basis_is_ignored(self) -> None:
        """Anthropic's `input_tokens` with a cache in play: 4 tokens of 8,561.
        Trusted, it would read as an empty context that never compacts."""
        compactor = _active(reserve_tokens=1_000)
        sent = _conversation(2)
        state = _measured(compactor, sent, reported=4)
        agent = _Agent(1_000_000)
        assert compactor.assess(sent, agent, state=state).current_tokens == (
            compactor.assess(sent, agent).current_tokens
        )

    def test_a_count_from_another_deployment_is_ignored(self) -> None:
        """A `/model`, a conversation resumed on another model, a model the gateway
        replaced: another tokenizer counts the same text differently."""
        compactor = _active(reserve_tokens=1_000)
        sent = _conversation(2)
        url = "http://llm.test/v1/chat/completions"
        state: dict[str, Any] = {}
        usage = compactor.usage_metadata(sent, None, _Agent(None), state)
        counted_by = VLLMProvider(model_name="qwen", url=url)
        compactor.observe(
            {"prompt_tokens": 2 * _total(usage)}, usage, state, "Assistant", counted_by
        )

        def on(model: str) -> SimpleNamespace:
            provider = VLLMProvider(model_name=model, url=url, context_window=1_000_000)
            return SimpleNamespace(name="Assistant", llm_provider=provider)

        alone = compactor.assess(sent, on("qwen")).current_tokens
        assert compactor.assess(sent, on("qwen"), state=state).current_tokens > alone
        assert compactor.assess(sent, on("deepseek"), state=state).current_tokens == alone


class _Logged:
    """Returns the estimates a live run logged, by how many messages a list has, so a
    test can replay that run's numbers exactly."""

    def __init__(self, by_length: dict[int, int]) -> None:
        self.by_length = by_length

    def estimate_messages(self, messages: list[LLMMessage]) -> int:
        return self.by_length[len(messages)]

    def estimate_tool_definitions(self, tool_defs: list[dict[str, Any]]) -> int:
        return 0


class TestTheCorrectionAfterACompaction:
    """The correction belongs to the text it was measured on, so a compaction that drops
    part of that text drops the same share of it. Measured with Qwen3-VL-32B through
    the gateway (window 32768, threshold 16384) on an agent reading hex tool output:
    kept whole, a summary the provider then counted at 12,036 was reported still over
    the limit; dropped whole, a pruning that kept the dense text was reported under it,
    and went out at 22,730."""

    def _budget(self, estimated: int, correction: int, logged: dict[int, int]) -> Budget:
        return Budget(
            context_window=32_768,
            reserve_tokens=16_384,
            current_tokens=estimated + correction,
            message_tokens=estimated,
            correction_tokens=correction,
            correction_base_tokens=estimated,
            estimator=_Logged({10: estimated, **logged}),
        )

    def test_a_compaction_that_kept_the_measured_text_keeps_its_correction(self) -> None:
        """Nothing was pruned: the provider counted the request that went out at 22,730."""
        budget = self._budget(10_147, 12_583, {})
        unchanged = [_msg("user", "x")] * 10
        assert budget.tokens_for(unchanged) == 22_730
        assert budget.fits(unchanged) is False

    def test_a_summary_that_dropped_most_of_it_takes_that_share_along(self) -> None:
        """Counted 12,036 by the provider; with the whole correction it read 18,624."""
        budget = self._budget(9_993, 12_656, {4: 5_968})
        assert budget.fits([_msg("user", "x")] * 4) is True

    def test_a_correction_larger_than_the_threshold_still_leaves_a_tail(self) -> None:
        """Estimated 12,000, counted 30,000. Charged as a fixed cost, the correction left
        no room under the threshold, and a summary took everything up to the last
        request."""
        budget = Budget(
            context_window=32_768,
            reserve_tokens=16_384,
            current_tokens=30_000,
            message_tokens=12_000,
            correction_tokens=18_000,
            correction_base_tokens=12_000,
            estimator=HeuristicEstimator(),
        )
        tail = budget.tail_tokens(20_000)
        assert tail > 0
        assert tail * 30_000 / 12_000 <= budget.threshold_tokens / 2

    @pytest.mark.asyncio
    async def test_the_result_reported_is_what_the_next_request_measures(self) -> None:
        """Retired by the compaction, the count left the request it produced measured on
        the estimate alone, while the compaction had judged it with the whole
        correction: two numbers for one request."""
        compactor = _active(reserve_tokens=1_000, keep_recent_tokens=1_000)
        conversation = _conversation(8)
        agent = _Agent(None)
        state: dict[str, Any] = {}
        sent = compactor.usage_metadata(conversation, None, agent, state)
        compactor.observe({"prompt_tokens": int(_total(sent) * 2.4)}, sent, state, "Assistant")
        agent.llm_provider.context_window = 1_000 + _total(sent)
        plan = compactor.assess(conversation, agent, state=state)
        assert plan.needed

        compacted, done = await compactor.run(conversation, plan, state, "Assistant")

        following = compactor.assess(compacted, agent, state=state)
        assert done.metadata["changed"] is True
        assert done.metadata["tokens_after"] == following.current_tokens
        assert done.metadata["sufficient"] is (
            following.current_tokens <= following.budget.threshold_tokens
        )


def _rejection(body: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://llm.test/v1/chat/completions")
    response = httpx.Response(400, request=request, text=body)
    return httpx.HTTPStatusError("400", request=request, response=response)


class TestLearningTheWindowFromARejection:
    """Through an OpenAI-compatible gateway nothing declares the window, and the same
    weights are served with different limits; the rejection states the real one."""

    @pytest.mark.parametrize(
        ("declared", "kept"),
        [(None, 40_960), (150_000, 40_960), (32_000, 32_000)],
        ids=["undeclared", "declared-too-large", "declared-smaller-is-a-choice"],
    )
    def test_the_stated_window_is_adopted(self, declared: int | None, kept: int) -> None:
        agent = _Agent(declared)
        error = _rejection("This model's maximum context length is 40960 tokens.")
        _active().overflow_plan(_conversation(2), agent, error=error)
        assert agent.llm_provider.context_window == kept

    def test_once_learned_the_next_request_is_compacted_before_it_is_rejected(self) -> None:
        agent = _Agent(None)
        compactor = _active(reserve_tokens=1_000)
        conversation = _conversation(6)
        assert compactor.assess(conversation, agent).needed is False, "no window, no threshold"

        error = _rejection("This model's maximum context length is 2000 tokens.")
        compactor.overflow_plan(conversation, agent, error=error)

        assert compactor.assess(conversation, agent).needed is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_a_provider_built_again_for_the_next_turn_starts_from_it(self) -> None:
        """A host that builds its providers for every request: kept on the instance
        alone, the window would go with it, and the next long request would be refused
        again."""
        respx.get("http://llm.test/v1/models").mock(
            return_value=httpx.Response(200, json={"object": "list", "data": []})
        )

        def served() -> VLLMProvider:
            return VLLMProvider(model_name="m", url="http://llm.test/v1/chat/completions")

        agent = SimpleNamespace(name="Assistant", llm_provider=served())
        assert await resolve_context_window(agent.llm_provider) is None, "reports nothing"

        error = _rejection("This model's maximum context length is 40960 tokens.")
        _active().overflow_plan(_conversation(2), agent, error=error)

        assert await resolve_context_window(served()) == 40_960
