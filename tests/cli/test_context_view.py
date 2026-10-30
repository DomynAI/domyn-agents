"""Tests for /context and for the indicator.

The command is easy; the indicator is the part with a requirement that is hard to
test and easy to get wrong: **it has to stay quiet**. In a fifty-turn session
anything printed always becomes noise and stops being read, so half of these
tests verify that nothing is printed.
"""

import io
from typing import Any

import pytest
from rich.console import Console

from domyn.agents.interfaces.cli import Theme, context_view
from domyn.agents.planner_strategy.context_management.budget import ContextUsage


@pytest.fixture
def theme() -> Theme:
    return Theme.build(allow_animation=False)


def _render(theme: Theme, usage: ContextUsage | None) -> str:
    buffer = io.StringIO()
    context_view.render(Console(file=buffer, width=100, force_terminal=False), theme, usage)
    return buffer.getvalue()


def _usage(
    messages: int = 1_000, tools: int = 0, window: int | None = None, **kw: Any
) -> ContextUsage:
    return ContextUsage(
        message_tokens=messages, tool_definition_tokens=tools, context_window=window, **kw
    )


class TestTheIndicatorStaysQuiet:
    """Every line here is a case where printing something would be a defect."""

    def test_before_any_request(self, theme: Theme) -> None:
        assert context_view.indicator(None) is None

    def test_when_the_window_is_unknown(self, theme: Theme) -> None:
        """A percentage of an invented window would look like data."""
        assert context_view.indicator(_usage(window=None)) is None

    def test_when_there_is_still_room(self, theme: Theme) -> None:
        assert context_view.indicator(_usage(messages=1_000, window=100_000)) is None

    def test_just_under_the_attention_threshold(self, theme: Theme) -> None:
        window = 100_000
        just_under = int(window * context_view.ATTENTION_FRACTION) - 1
        assert context_view.indicator(_usage(messages=just_under, window=window)) is None


class TestTheIndicatorSpeaks:
    def test_past_the_attention_threshold(self, theme: Theme) -> None:
        shown = context_view.indicator(_usage(messages=75_000, window=100_000))
        assert shown is not None
        text, _ = shown
        assert "75%" in text

    def test_the_tone_changes_when_it_is_serious(self, theme: Theme) -> None:
        """No longer information, a warning: the style has to distinguish it."""
        attention = context_view.indicator(_usage(messages=75_000, window=100_000))
        serious = context_view.indicator(_usage(messages=95_000, window=100_000))
        assert attention and serious
        assert attention[1] == "meta"
        assert serious[1] == "warning"

    def test_with_a_policy_it_shows_the_distance_not_the_share(self, theme: Theme) -> None:
        """When a threshold exists, the useful thing is how far it is."""
        shown = context_view.indicator(
            _usage(messages=75_000, window=100_000, threshold_tokens=84_000, strategy="prune_only")
        )
        assert shown is not None
        text, _ = shown
        assert "compaction" in text
        assert "9k" in text  # 84_000 - 75_000

    def test_tool_definitions_count_towards_the_share(self, theme: Theme) -> None:
        """They are not an extra: they occupy the window like everything else."""
        without = context_view.indicator(_usage(messages=60_000, window=100_000))
        with_tools = context_view.indicator(_usage(messages=60_000, tools=25_000, window=100_000))
        assert without is None and with_tools is not None


class TestTheCommand:
    def test_with_no_requests_it_says_so(self, theme: Theme) -> None:
        assert "nothing to measure" in _render(theme, None)

    def test_it_separates_conversation_from_tools(self, theme: Theme) -> None:
        out = _render(theme, _usage(messages=8_000, tools=2_000, window=100_000))
        assert "conversation" in out
        assert "tool definitions" in out
        assert "total" in out

    def test_it_shows_the_share_of_the_window(self, theme: Theme) -> None:
        assert "50%" in _render(theme, _usage(messages=50_000, window=100_000))

    def test_without_a_window_it_explains_how_to_get_one(self, theme: Theme) -> None:
        """Without that number no policy can act: saying so is useful, silence
        leaves the user with a screen that does not explain why it is mute."""
        assert "context_window" in _render(theme, _usage(messages=1_000, window=None))

    def test_it_warns_when_tools_outweigh_the_conversation(self, theme: Theme) -> None:
        """It is counter-intuitive and changes where it is worth intervening: no
        compaction reduces tool definitions."""
        out = _render(theme, _usage(messages=3_000, tools=20_000, window=100_000))
        assert "outweigh the conversation" in out

    def test_it_does_not_warn_in_the_normal_case(self, theme: Theme) -> None:
        out = _render(theme, _usage(messages=20_000, tools=3_000, window=100_000))
        assert "outweigh the conversation" not in out

    def test_it_does_not_divide_by_zero(self, theme: Theme) -> None:
        """A zero measurement is possible and must not bring the command down."""
        _render(theme, _usage(messages=0, tools=0, window=100_000))


class TestTheMeasurementComesFromThePlanner:
    """The part unit tests do not cover: that the number makes the journey.

    The planner measures and puts the data in ``LLM_START`` metadata; the REPL
    collects it. In between sits the runner. If that path breaks, every test above
    keeps passing and ``/context`` shows nothing — which is how an observability
    feature dies in silence.
    """

    @pytest.mark.asyncio
    async def test_after_a_turn_the_measurement_exists(self) -> None:
        from tests.cli.test_repl_session import _repl

        repl, _ = _repl("hello")
        await repl.run()

        usage = repl._state.last_usage
        assert usage is not None, "no measurement collected: the path is broken"
        assert usage.message_tokens > 0

    @pytest.mark.asyncio
    async def test_clear_forgets_the_measurement(self) -> None:
        """Keeping it would say something false about the new conversation."""
        from tests.cli.test_repl_session import _repl

        repl, _ = _repl("hello", "/clear")
        await repl.run()
        assert repl._state.last_usage is None

    @pytest.mark.asyncio
    async def test_the_command_shows_what_was_collected(self) -> None:
        from tests.cli.test_repl_session import _repl

        repl, buffer = _repl("hello", "/context")
        await repl.run()
        output = buffer.getvalue()
        assert "conversation" in output
        assert "tool definitions" in output

    @pytest.mark.asyncio
    async def test_without_a_known_window_the_indicator_stays_away(self) -> None:
        """Today's case for every provider that does not declare a window: the
        measurement is collected, but nothing is announced."""
        from tests.cli.test_repl_session import _repl

        repl, buffer = _repl("hello")
        await repl.run()
        assert repl._state.last_usage is not None
        assert repl._state.last_usage.context_window is None
        assert "context " not in buffer.getvalue()

    @pytest.mark.asyncio
    async def test_with_a_known_window_and_a_full_context_it_appears(self) -> None:
        """The visible behaviour no other test covers: the line that shows up on
        its own. Without this the indicator could never print and every test about
        its thresholds would keep passing."""
        from tests.cli.test_repl_session import _repl

        repl, buffer = _repl("hello")
        # A tiny window: any conversation fills it.
        repl._agent.llm_provider.context_window = 10
        await repl.run()

        assert "context " in buffer.getvalue()


class TestDoNotPromiseACompactionThatWillNotHappen:
    """Born from a real defect: with the inert strategy mounted by default,
    ``/context`` announced "until compaction 89.4k" and the indicator "compaction
    in ~11.6k tokens" — while no compaction would have happened. The reader plans
    their work around that threshold.
    """

    def test_without_a_strategy_the_command_says_so(self, theme: Theme) -> None:
        out = _render(theme, _usage(messages=20_000, window=128_000))
        assert "no compaction configured" in out
        assert "until compaction" in out  # the field is there, the promise is not

    def test_with_a_strategy_it_shows_distance_and_name(self, theme: Theme) -> None:
        out = _render(
            theme,
            _usage(
                messages=100_000, window=128_000, threshold_tokens=111_616, strategy="prune_only"
            ),
        )
        assert "11.6k" in out
        assert "prune_only" in out

    def test_the_indicator_promises_nothing_without_a_strategy(self, theme: Theme) -> None:
        """The share remains — that is true, and it matters that the window will
        run out — but no distance to an event that will not arrive."""
        shown = context_view.indicator(_usage(messages=100_000, window=128_000))
        assert shown is not None
        assert "compaction" not in shown[0]
        assert "78%" in shown[0]

    def test_the_indicator_only_promises_with_a_strategy(self, theme: Theme) -> None:
        shown = context_view.indicator(
            _usage(
                messages=100_000, window=128_000, threshold_tokens=111_616, strategy="prune_only"
            )
        )
        assert shown is not None and "compaction in" in shown[0]

    @pytest.mark.asyncio
    async def test_from_the_planner_the_default_is_active(self) -> None:
        """Through a real turn: the shipped default must actually do something.

        This assertion used to be the opposite, when the default was inert. It is
        pinned in this direction now because a compaction nobody can reach is a
        compaction nobody exercises — and because at the threshold the
        alternative to pruning is a rejected request.
        """
        from tests.cli.test_repl_session import _repl

        repl, _ = _repl("hello")
        repl._agent.llm_provider.context_window = 128_000
        await repl.run()

        usage = repl._state.last_usage
        assert usage is not None
        assert usage.strategy == "two_phase"
        assert usage.threshold_tokens == 128_000 - 16_384

    @pytest.mark.asyncio
    async def test_from_the_planner_an_explicit_noop_still_promises_nothing(self) -> None:
        """Opting out has to keep working, and keep being honest about it."""
        from domyn.agents.planner_strategy.context_management.compaction import (
            Compactor,
            NoOpCompaction,
        )
        from tests.cli.test_repl_session import _repl

        repl, _ = _repl("hello")
        repl._agent.llm_provider.context_window = 128_000
        repl._agent.planner.compactor = Compactor(strategy=NoOpCompaction())
        await repl.run()

        usage = repl._state.last_usage
        assert usage is not None
        assert usage.threshold_tokens is None, "threshold announced with the inert strategy"
        assert usage.strategy is None

    @pytest.mark.asyncio
    async def test_from_the_planner_an_active_strategy_brings_the_threshold(self) -> None:
        from domyn.agents.planner_strategy import (
            Compactor,
            PruneOnlyCompaction,
        )
        from tests.cli.test_repl_session import _repl

        repl, _ = _repl("hello")
        repl._agent.llm_provider.context_window = 128_000
        repl._agent.planner.compactor = Compactor(strategy=PruneOnlyCompaction())
        await repl.run()

        usage = repl._state.last_usage
        assert usage is not None
        assert usage.threshold_tokens == 128_000 - 16_384
        assert usage.strategy == "prune_only"
