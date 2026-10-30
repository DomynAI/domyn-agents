"""Tests for token estimation.

An estimate cannot be tested for exactness — it is not exact by construction. It
is tested for the properties its callers depend on: that it never underestimates
(an underestimate costs a rejected request), that it grows with content, and that
it does not treat expensive things as free.
"""

import pytest

from domyn.agents.llm import LLMMessage, LLMToolCall, ToolCallFunction
from domyn.agents.planner_strategy.context_management.budget import (
    ContextUsage,
    HeuristicEstimator,
    TokenEstimator,
    measure,
)


@pytest.fixture
def estimator() -> HeuristicEstimator:
    return HeuristicEstimator()


def _call(name: str = "run_command", arguments: str = '{"command": "ls"}') -> LLMToolCall:
    return LLMToolCall(
        id="call_1", type="function", function=ToolCallFunction(name=name, arguments=arguments)
    )


class TestTheMinimalContract:
    def test_it_satisfies_the_protocol(self, estimator: HeuristicEstimator) -> None:
        assert isinstance(estimator, TokenEstimator)

    def test_nothing_costs_zero(self, estimator: HeuristicEstimator) -> None:
        assert estimator.estimate_text("") == 0
        assert estimator.estimate_messages([]) == 0
        assert estimator.estimate_tool_definitions([]) == 0

    def test_one_character_costs_at_least_one_token(self, estimator: HeuristicEstimator) -> None:
        """Rounding up: never zero for non-empty content."""
        assert estimator.estimate_text("a") >= 1


class TestNeverUnderestimate:
    """The property that matters: an error in this direction costs a turn."""

    def test_it_rounds_up(self, estimator: HeuristicEstimator) -> None:
        # 4 characters / 3.0 = 1.33 -> 2, not 1
        assert estimator.estimate_text("abcd") == 2

    def test_a_message_costs_more_than_its_text(self, estimator: HeuristicEstimator) -> None:
        """The role envelope exists in every chat template: ignoring it would
        badly underestimate a conversation of many short messages."""
        text = "a" * 30
        text_only = estimator.estimate_text(text)
        one_message = estimator.estimate_messages([LLMMessage(role="user", content=text)])
        assert one_message > text_only

    def test_tool_calls_are_not_free(self, estimator: HeuristicEstimator) -> None:
        without = estimator.estimate_messages([LLMMessage(role="assistant", content="ok")])
        with_call = estimator.estimate_messages(
            [LLMMessage(role="assistant", content="ok", tool_calls=[_call()])]
        )
        assert with_call > without

    def test_images_do_not_cost_their_length(self, estimator: HeuristicEstimator) -> None:
        """An image is a short reference and a huge cost: estimating it by
        characters would make it look free."""
        message = LLMMessage(
            role="user",
            content=[{"type": "image_url", "image_url": {"url": "http://x/y.png"}}],
        )
        assert estimator.estimate_messages([message]) >= estimator.image_tokens

    def test_the_default_is_conservative(self) -> None:
        """3.0 rather than the canonical 4.0: see the reasoning on the field."""
        assert HeuristicEstimator().chars_per_token == 3.0
        assert HeuristicEstimator().estimate_text("a" * 100) > 25  # 4.0 would give 25


class TestMonotonicity:
    def test_more_text_never_costs_less(self, estimator: HeuristicEstimator) -> None:
        previous = 0
        for length in range(0, 500, 37):
            current = estimator.estimate_text("x" * length)
            assert current >= previous
            previous = current

    def test_more_messages_never_cost_less(self, estimator: HeuristicEstimator) -> None:
        messages: list[LLMMessage] = []
        previous = 0
        for i in range(10):
            messages.append(LLMMessage(role="user", content=f"message number {i}"))
            current = estimator.estimate_messages(messages)
            assert current > previous
            previous = current


class TestToolDefinitions:
    """They travel outside the messages and no compaction touches them:
    measuring them separately is what makes that cost visible."""

    def test_one_tool_costs_something(self, estimator: HeuristicEstimator) -> None:
        tool_def = {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run a command in the shell",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
        assert estimator.estimate_tool_definitions([tool_def]) > 0

    def test_many_tools_cost_proportionally(self, estimator: HeuristicEstimator) -> None:
        tool_def = {"type": "function", "function": {"name": "t", "parameters": {}}}
        one = estimator.estimate_tool_definitions([tool_def])
        forty = estimator.estimate_tool_definitions([tool_def] * 40)
        assert forty > one * 20

    def test_the_estimate_is_deterministic(self, estimator: HeuristicEstimator) -> None:
        """Stable serialization: two equal dicts with keys in a different order
        must estimate identically, or the number oscillates between requests."""
        a = {"function": {"name": "t", "description": "d"}, "type": "function"}
        b = {"type": "function", "function": {"description": "d", "name": "t"}}
        assert estimator.estimate_tool_definitions([a]) == estimator.estimate_tool_definitions([b])


class TestConfigurability:
    def test_tuning_chars_per_token_changes_the_estimate(self) -> None:
        conservative = HeuristicEstimator(chars_per_token=3.0)
        optimistic = HeuristicEstimator(chars_per_token=4.0)
        text = "a" * 120
        assert conservative.estimate_text(text) > optimistic.estimate_text(text)

    def test_nonsense_values_are_refused(self) -> None:
        with pytest.raises(ValueError):
            HeuristicEstimator(chars_per_token=0)


class TestContextUsage:
    """The shape the measurement travels in, from planner to front-end."""

    def test_the_total_sums_both_lines(self) -> None:
        usage = ContextUsage(message_tokens=100, tool_definition_tokens=40)
        assert usage.total_tokens == 140

    def test_an_unknown_window_does_not_become_zero(self) -> None:
        """`None` and `0%` are different statements: "I do not know how much I am
        using" versus "I am using nothing". Confusing them is a UI bug."""
        assert ContextUsage(message_tokens=100).fraction_used is None

    def test_the_share_includes_tool_definitions(self) -> None:
        usage = ContextUsage(message_tokens=500, tool_definition_tokens=500, context_window=1_000)
        assert usage.fraction_used == 1.0

    def test_without_a_threshold_there_is_no_distance_to_announce(self) -> None:
        assert ContextUsage(message_tokens=10).tokens_until_threshold is None

    def test_the_distance_never_goes_below_zero(self) -> None:
        """Past the threshold the answer is "zero", not a negative number a
        front-end would print as such."""
        usage = ContextUsage(message_tokens=900, threshold_tokens=800)
        assert usage.tokens_until_threshold == 0

    def test_measure_needs_no_optional_arguments(self) -> None:
        usage = measure([LLMMessage(role="user", content="hello")])
        assert usage.total_tokens > 0
        assert usage.tool_definition_tokens == 0
        assert usage.context_window is None


class TestNothingIsInferredFromAModelName:
    """There used to be a table of windows keyed by model name. It is gone.

    A name does not determine a window: the same weights served with
    `--max-model-len 32768` or `131072` have different limits. It got a real
    deployment wrong — `deepseek-ai/DeepSeek-V4-Flash` picked up 64,000 tokens
    from a `deepseek` key — which is worse than an admitted guess, because it reads
    as knowledge. Hosted APIs that cannot be asked (OpenAI) have a catalogue keyed
    by their exact ids; a self-hosted endpoint never consults one.
    """

    def test_the_table_and_its_lookup_are_gone(self) -> None:
        from domyn.agents.interfaces.cli import config

        assert not hasattr(config, "KNOWN_CONTEXT_WINDOWS")
        assert not hasattr(config, "guess_context_window")

    @pytest.mark.asyncio
    async def test_a_self_hosted_endpoint_never_takes_a_window_from_a_name(self) -> None:
        """Not even a name the hosted catalogue knows: on vLLM it says nothing."""
        from unittest.mock import patch

        from domyn.agents.llm import VLLMProvider
        from domyn.agents.llm.context_window import resolve_context_window

        for model in ("gpt-4o", "deepseek-ai/DeepSeek-V4-Flash"):
            provider = VLLMProvider(
                model_name=model, url=f"http://{model}.invalid/v1/chat/completions"
            )
            with patch.object(VLLMProvider, "fetch_context_window", return_value=None):
                assert await resolve_context_window(provider) is None, model
            assert provider.context_window is None, model


def test_the_usage_metadata_key_is_not_private() -> None:
    """No `_` prefix: it is telemetry a front-end must read, not a secret to be
    stripped before leaving the process."""
    from domyn.agents.planner_strategy.context_management.budget import (
        CONTEXT_USAGE_METADATA_KEY,
    )

    assert not CONTEXT_USAGE_METADATA_KEY.startswith("_")


def test_source_event_id_never_reaches_the_wire() -> None:
    """It is our bookkeeping, not part of the protocol: a provider receiving an
    unknown field rejects the request."""
    message = LLMMessage(role="user", content="x", source_event_id="evt_1")
    assert message.source_event_id == "evt_1"
    assert "source_event_id" not in message.to_dict()
    assert "source_event_id" not in message.model_dump()
