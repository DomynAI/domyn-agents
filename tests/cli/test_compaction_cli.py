"""What the user sees when the conversation gets compacted.

This is the half the mechanism tests do not cover, and the easiest to leave
broken: compaction can work perfectly and stay **invisible**, and then the agent
just looks forgetful for no apparent reason. Half of these tests check something
is said, the other half that nothing is said when it is not true.
"""

# No `from __future__ import annotations`: builds real agents and tools.
import io
from typing import Any

import pytest
from rich.console import Console

from domyn.agents import BaseEvent, ExecutionEventType, Part
from domyn.agents.interfaces.cli import Theme
from domyn.agents.interfaces.cli.renderer import EventRenderer
from domyn.agents.planner_strategy import (
    CompactionPolicy,
    Compactor,
    PruneOnlyCompaction,
)


def _renderer(verbose: bool = False) -> tuple[EventRenderer, io.StringIO]:
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=False)
    return EventRenderer(console, Theme.build(allow_animation=False), verbose=verbose), buffer


def _event(kind: ExecutionEventType, text: str = "", **metadata: Any) -> BaseEvent:
    return BaseEvent(
        event_type=kind, author="Assistant", content=[Part(text=text)], metadata=metadata
    )


class TestWhatIsVisible:
    def test_the_start_is_announced_even_without_verbose(self) -> None:
        """Not a lifecycle event to hide: this is the moment the agent loses
        information, and with a summarizing strategy it is also a pause to
        account for."""
        renderer, buffer = _renderer(verbose=False)
        renderer.begin_turn()
        renderer.render(_event(ExecutionEventType.CONTEXT_COMPACTION_START, tokens_before=124_000))
        assert "compacting context" in buffer.getvalue()

    def test_the_start_says_how_much_there_was(self) -> None:
        renderer, buffer = _renderer()
        renderer.begin_turn()
        renderer.render(_event(ExecutionEventType.CONTEXT_COMPACTION_START, tokens_before=124_000))
        assert "124k" in buffer.getvalue()

    def test_the_end_says_before_after_and_saving(self) -> None:
        renderer, buffer = _renderer()
        renderer.begin_turn()
        renderer.render(
            _event(
                ExecutionEventType.CONTEXT_COMPACTED,
                tokens_before=124_000,
                tokens_after=38_000,
                changed=True,
            )
        )
        output = buffer.getvalue()
        assert "124k" in output and "38k" in output and "86k" in output

    def test_when_there_was_nothing_to_do_it_says_so(self) -> None:
        """Implying a recovery that did not happen is how people stop
        investigating a session that keeps being too large."""
        renderer, buffer = _renderer()
        renderer.begin_turn()
        renderer.render(
            _event(
                ExecutionEventType.CONTEXT_COMPACTED,
                tokens_before=124_000,
                tokens_after=124_000,
                changed=False,
            )
        )
        output = buffer.getvalue()
        assert "nothing to compact" in output
        assert "->" not in output

    def test_without_numbers_it_at_least_says_it_happened(self) -> None:
        """Metadata can be missing (a third-party planner): a line without
        numbers beats no line."""
        renderer, buffer = _renderer()
        renderer.begin_turn()
        renderer.render(_event(ExecutionEventType.CONTEXT_COMPACTED))
        assert "compacted" in buffer.getvalue()


class TestInARealTurn:
    """Through the runner, because it sits between the planner and the screen."""

    @pytest.mark.asyncio
    async def test_a_tiny_window_makes_it_fire_and_say_so(self) -> None:
        from tests.cli.test_repl_session import _repl

        repl, buffer = _repl("hello")
        # A window so small any conversation exceeds it: the most direct way to
        # force the trigger without inventing a long session.
        repl._agent.llm_provider.context_window = 100
        repl._agent.planner.compactor = Compactor(
            strategy=PruneOnlyCompaction(),
            policy=CompactionPolicy(reserve_tokens=1, keep_recent_tokens=1),
        )
        await repl.run()

        assert "compacting context" in buffer.getvalue(), "the trigger was not announced"

    @pytest.mark.asyncio
    async def test_without_a_known_window_it_neither_fires_nor_speaks(self) -> None:
        """Today's case for every provider that does not declare a window: the
        guarantee that this work does not change existing behaviour."""
        from tests.cli.test_repl_session import _repl

        repl, buffer = _repl("hello")
        repl._agent.planner.compactor = Compactor(strategy=PruneOnlyCompaction())
        assert repl._agent.llm_provider.context_window is None
        await repl.run()

        assert "compacting context" not in buffer.getvalue()

    @pytest.mark.asyncio
    async def test_the_threshold_only_arrives_when_a_compaction_will_happen(self) -> None:
        """The link between the two halves: the planner measures, and the
        measurement carries the threshold — but only when a compaction will
        actually happen.

        This test asserted the opposite and passed, because the code announced a
        threshold even with the inert strategy mounted by default. That was a
        false promise in exactly the field a user comes to read.
        """
        from tests.cli.test_repl_session import _repl

        repl, _ = _repl("hello")
        repl._agent.llm_provider.context_window = 100_000
        repl._agent.planner.compactor = Compactor(
            strategy=PruneOnlyCompaction(), policy=CompactionPolicy(reserve_tokens=8_000)
        )
        await repl.run()

        usage = repl._state.last_usage
        assert usage is not None
        assert usage.threshold_tokens == 100_000 - 8_000
        assert usage.strategy == "prune_only"
        assert usage.tokens_until_threshold is not None


class TestBothPlanners:
    """The gap that shipped a measurement which did not work.

    ``ToolUseWithThoughtPlannerStrategy`` is a **full copy** of its parent's loop,
    and the CLI uses that one (``CLI_PLANNER = "tool_use_thought"``). My tests used
    the parent, so they all passed while ``/context`` said "no request" after a
    real conversation.

    Parametrized on purpose: as long as the two loops stay duplicated, anything
    touching the context has to be verified on both.
    """

    @staticmethod
    def _planners() -> list[Any]:
        from domyn.agents.planner_strategy import (
            ToolUsePlannerStrategy,
            ToolUseWithThoughtPlannerStrategy,
        )

        return [ToolUsePlannerStrategy, ToolUseWithThoughtPlannerStrategy]

    @pytest.mark.parametrize("planner_name", ["tool_use", "tool_use_thought"])
    @pytest.mark.asyncio
    async def test_the_measurement_arrives_with_every_planner(self, planner_name: str) -> None:
        from domyn.agents.interfaces.cli.config import CLI_PLANNER
        from tests.cli.test_repl_session import _repl

        by_name = {cls().type: cls for cls in self._planners()}
        assert CLI_PLANNER in by_name, "the CLI's planner is not among those covered"

        repl, _ = _repl("hello")
        repl._agent.planner = by_name[planner_name]()
        await repl.run()

        usage = repl._state.last_usage
        assert usage is not None, (
            f"no measurement collected with {planner_name}: "
            "the planner -> LLM_START -> CLI path is broken"
        )
        assert usage.message_tokens > 0

    @pytest.mark.parametrize("planner_name", ["tool_use", "tool_use_thought"])
    @pytest.mark.asyncio
    async def test_compaction_fires_with_every_planner(self, planner_name: str) -> None:
        from tests.cli.test_repl_session import _repl

        by_name = {cls().type: cls for cls in self._planners()}
        repl, buffer = _repl("hello")
        repl._agent.planner = by_name[planner_name]()
        repl._agent.llm_provider.context_window = 100
        repl._agent.planner.compactor = Compactor(
            strategy=PruneOnlyCompaction(),
            policy=CompactionPolicy(reserve_tokens=1, keep_recent_tokens=1),
        )
        await repl.run()

        assert "compacting context" in buffer.getvalue(), (
            f"the trigger was not announced with {planner_name}"
        )

    @pytest.mark.parametrize("planner_name", ["tool_use", "tool_use_thought"])
    @pytest.mark.asyncio
    async def test_when_the_indicator_says_it_is_due_it_happens(self, planner_name: str) -> None:
        """The reported bug: "compaction in ~0 tokens" on screen and the next
        request went out uncompacted — the indicator counted the tool
        definitions, the trigger did not. Here the conversation alone is far
        under the threshold and the definitions alone are over it."""
        from domyn.agents.tools import FunctionTool
        from tests.cli.test_repl_session import _repl

        def lookup(query: str) -> str:
            """Look something up.

            Args:
                query: What to look up.
            """
            return query

        tool = FunctionTool(func=lookup)
        tool.description = "x" * 9_000
        by_name = {cls().type: cls for cls in self._planners()}
        repl, buffer = _repl("hello")
        repl._agent.planner = by_name[planner_name]()
        repl._agent.tools = [tool]
        repl._agent.llm_provider.context_window = 3_000
        repl._agent.planner.compactor = Compactor(
            strategy=PruneOnlyCompaction(), policy=CompactionPolicy(reserve_tokens=1)
        )
        await repl.run()

        usage = repl._state.last_usage
        assert usage is not None and usage.tokens_until_threshold == 0
        assert "compacting context" in buffer.getvalue(), (
            f"the indicator said the threshold was reached and {planner_name} did not compact"
        )


class TestWhenTheProviderRejectsTheRequestAsTooLong:
    """The estimate under-counted and the provider said so. Compacting harder and
    asking again saves the turn; asking again forever would not end it."""

    @staticmethod
    def _repl(
        planner_name: str, rejections: int, context_window: int | None = 100_000
    ) -> tuple[Any, io.StringIO, Any]:
        import httpx

        from domyn.agents.planner_strategy import (
            ToolUsePlannerStrategy,
            ToolUseWithThoughtPlannerStrategy,
        )
        from tests.cli.test_repl_session import ScriptedProvider, _repl

        class Rejecting(ScriptedProvider):
            rejections: int = 0
            calls: int = 0

            async def generate(self, messages: Any, **kwargs: Any) -> Any:
                self.calls += 1
                if self.calls <= self.rejections:
                    request = httpx.Request("POST", "http://llm.test/v1/chat/completions")
                    body = "This model's maximum context length is 40960 tokens."
                    response = httpx.Response(400, request=request, text=body)
                    raise httpx.HTTPStatusError("400", request=request, response=response)
                return await super().generate(messages, **kwargs)

        planners = {
            "tool_use": ToolUsePlannerStrategy,
            "tool_use_thought": ToolUseWithThoughtPlannerStrategy,
        }
        repl, buffer = _repl("hello")
        repl._agent.planner = planners[planner_name]()
        provider = Rejecting(rejections=rejections, context_window=context_window)
        repl._agent.llm_provider = provider
        return repl, buffer, provider

    @pytest.mark.parametrize(
        "context_window",
        # Undeclared is the platform's case: a rejection is then the only signal.
        [100_000, None],
        ids=["declared window", "no window"],
    )
    @pytest.mark.parametrize("planner_name", ["tool_use", "tool_use_thought"])
    @pytest.mark.asyncio
    async def test_it_compacts_and_the_retry_answers(
        self, planner_name: str, context_window: int | None
    ) -> None:
        repl, buffer, provider = self._repl(
            planner_name, rejections=1, context_window=context_window
        )
        await repl.run()

        assert provider.calls == 2
        assert "compacting context" in buffer.getvalue()
        assert "rejected the request as too long" in buffer.getvalue(), "the reason went unsaid"
        assert "Hello from the agent." in buffer.getvalue()

    @pytest.mark.parametrize("planner_name", ["tool_use", "tool_use_thought"])
    @pytest.mark.asyncio
    async def test_a_second_rejection_ends_the_turn(self, planner_name: str) -> None:
        repl, buffer, provider = self._repl(planner_name, rejections=99)
        await repl.run()

        assert provider.calls == 2, "retried more than once"
        assert "Hello from the agent." not in buffer.getvalue()


class TestSayingWhenPruningIsNoLongerEnough:
    """A compaction that pruned plenty and is still over the limit has to say so.

    Otherwise the next thing the person sees is a rejected request, with nothing
    connecting it to the line that just told them the context was compacted.
    """

    def test_the_warning_appears(self) -> None:
        renderer, buffer = _renderer()
        renderer.begin_turn()
        renderer.render(
            _event(
                ExecutionEventType.CONTEXT_COMPACTED,
                tokens_before=195_967,
                tokens_after=189_318,
                changed=True,
                sufficient=False,
                threshold_tokens=133_616,
            )
        )
        out = buffer.getvalue()
        assert "still over the limit" in out
        assert "189.3k" in out, "the numbers must still be there"

    def test_a_sufficient_compaction_carries_no_warning(self) -> None:
        renderer, buffer = _renderer()
        renderer.begin_turn()
        renderer.render(
            _event(
                ExecutionEventType.CONTEXT_COMPACTED,
                tokens_before=140_000,
                tokens_after=30_000,
                changed=True,
                sufficient=True,
                threshold_tokens=133_616,
            )
        )
        assert "still over the limit" not in buffer.getvalue()

    def test_an_event_without_the_flag_is_not_treated_as_a_failure(self) -> None:
        """Older events, and the manual path, carry no `sufficient` key."""
        renderer, buffer = _renderer()
        renderer.begin_turn()
        renderer.render(
            _event(
                ExecutionEventType.CONTEXT_COMPACTED,
                tokens_before=140_000,
                tokens_after=30_000,
                changed=True,
            )
        )
        assert "still over the limit" not in buffer.getvalue()

    def test_nothing_to_compact_comes_first_and_the_warning_still_follows(self) -> None:
        """ "Nothing was done" is read first, but it is not the whole story: when
        the request is still over the limit, the next line the person sees is a
        rejection, and only the warning connects the two. Seen live after a
        provider rejected a request nothing could be dropped from."""
        renderer, buffer = _renderer()
        renderer.begin_turn()
        renderer.render(
            _event(ExecutionEventType.CONTEXT_COMPACTED, changed=False, sufficient=False)
        )
        out = buffer.getvalue()
        assert "nothing to compact" in out
        assert "still over the limit" in out
        assert out.index("nothing to compact") < out.index("still over the limit")


class TestMeasuringFromTheProvidersCountInARealSession:
    """Through the runner, two turns: the count of the first request has to survive
    the turn boundary in the conversation state, and both planner loops have to
    record it — they are copies of each other."""

    @staticmethod
    def _repl(planner_name: str, *, counted: bool) -> tuple[Any, io.StringIO]:
        from domyn.agents.planner_strategy import (
            HeuristicEstimator,
            ToolUsePlannerStrategy,
            ToolUseWithThoughtPlannerStrategy,
        )
        from tests.cli.test_repl_session import ScriptedProvider, _repl

        class DenseCounting(ScriptedProvider):
            """Counts three times what chars/3 estimates, as UUIDs and hex do, on a
            window that fits the estimate and not the count. Sized on the first
            request, since its absolute size depends on the agent's preamble."""

            counted: bool = True

            async def generate(self, messages: Any, **kwargs: Any) -> Any:
                estimator = HeuristicEstimator()
                estimate = estimator.estimate_messages(messages)
                estimate += estimator.estimate_tool_definitions(kwargs.get("tools") or [])
                if self.context_window is None:
                    self.context_window = CompactionPolicy().reserve_tokens + 2 * estimate
                response = await super().generate(messages, **kwargs)
                if self.counted:
                    response.usage = {"prompt_tokens": 3 * estimate, "completion_tokens": 5}
                return response

        planners = {
            "tool_use": ToolUsePlannerStrategy,
            "tool_use_thought": ToolUseWithThoughtPlannerStrategy,
        }
        repl, buffer = _repl("first", "second")
        repl._agent.planner = planners[planner_name]()
        repl._agent.llm_provider = DenseCounting(counted=counted)
        return repl, buffer

    @pytest.mark.parametrize("planner_name", ["tool_use", "tool_use_thought"])
    @pytest.mark.asyncio
    async def test_the_second_request_is_measured_on_the_first_ones_count(
        self, planner_name: str
    ) -> None:
        repl, buffer = self._repl(planner_name, counted=True)
        await repl.run()
        assert "compacting context" in buffer.getvalue()

    @pytest.mark.parametrize("planner_name", ["tool_use", "tool_use_thought"])
    @pytest.mark.asyncio
    async def test_without_a_count_the_estimate_alone_decides(self, planner_name: str) -> None:
        """The control: same window, same estimate, no count — no compaction."""
        repl, buffer = self._repl(planner_name, counted=False)
        await repl.run()
        assert "compacting context" not in buffer.getvalue()
