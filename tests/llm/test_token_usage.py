"""`TokenUsage.from_provider`: every provider's usage, one shape.

The shapes below are what the providers return on `LLMResponse.usage`, as each
provider module copies it from the wire.
"""

from domyn.agents.llm.base import LLMResponse, TokenUsage, add_token_usage, token_usage_metadata


class TestChatCompletions:
    """vLLM, OpenAI chat, Cerebras, OpenRouter, Domyn."""

    def test_plain_counts(self) -> None:
        usage = TokenUsage.from_provider(
            {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        )
        assert usage is not None
        assert usage.as_metadata() == {
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "requests": 1,
        }

    def test_details_give_cache_and_reasoning(self) -> None:
        usage = TokenUsage.from_provider(
            {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_tokens_details": {"cached_tokens": 64, "cache_write_tokens": 10},
                "completion_tokens_details": {"reasoning_tokens": 30},
            }
        )
        assert usage is not None
        assert usage.cache_read_tokens == 64
        assert usage.cache_write_tokens == 10
        assert usage.reasoning_tokens == 30

    def test_details_set_to_null_are_not_counts(self) -> None:
        """vLLM sends `prompt_tokens_details: null` when prefix caching is off."""
        usage = TokenUsage.from_provider(
            {"prompt_tokens": 5, "completion_tokens": 1, "prompt_tokens_details": None}
        )
        assert usage is not None
        assert usage.cache_read_tokens is None
        assert "cache_read_tokens" not in usage.as_metadata()

    def test_total_is_derived_when_not_reported(self) -> None:
        usage = TokenUsage.from_provider({"prompt_tokens": 7, "completion_tokens": 3})
        assert usage is not None and usage.total_tokens == 10


class TestOpenAIResponses:
    def test_input_and_output_details(self) -> None:
        usage = TokenUsage.from_provider(
            {
                "input_tokens": 200,
                "output_tokens": 40,
                "total_tokens": 240,
                "input_tokens_details": {"cached_tokens": 128},
                "output_tokens_details": {"reasoning_tokens": 12},
            }
        )
        assert usage is not None
        assert usage.input_tokens == 200, "already includes the cached tokens"
        assert usage.cache_read_tokens == 128
        assert usage.reasoning_tokens == 12


class TestAnthropic:
    def test_input_is_the_whole_prompt(self) -> None:
        """Anthropic's `input_tokens` is only the uncached tail."""
        usage = TokenUsage.from_provider(
            {
                "input_tokens": 4,
                "output_tokens": 20,
                "cache_read_input_tokens": 8400,
                "cache_creation_input_tokens": 157,
            }
        )
        assert usage is not None
        assert usage.input_tokens == 8561
        assert usage.cache_read_tokens == 8400
        assert usage.cache_write_tokens == 157
        assert usage.total_tokens == 8581


class TestGatewayFrontingAnthropic:
    def test_cache_is_not_counted_twice(self) -> None:
        """LiteLLM sends `prompt_tokens`, cache included, beside Anthropic's cache keys."""
        usage = TokenUsage.from_provider(
            {
                "prompt_tokens": 8561,
                "completion_tokens": 20,
                "cache_read_input_tokens": 8400,
                "cache_creation_input_tokens": 157,
            }
        )
        assert usage is not None
        assert usage.input_tokens == 8561
        assert usage.cache_read_tokens == 8400
        assert usage.cache_write_tokens == 157


class TestNothingReported:
    def test_empty_or_missing(self) -> None:
        assert TokenUsage.from_provider({}) is None
        assert TokenUsage.from_provider(None) is None
        assert TokenUsage.from_provider({"model": "x"}) is None

    def test_non_counts_are_ignored(self) -> None:
        assert TokenUsage.from_provider({"prompt_tokens": True, "completion_tokens": -1}) is None
        assert TokenUsage.from_provider({"prompt_tokens": "12"}) is None

    def test_zero_is_a_count(self) -> None:
        usage = TokenUsage.from_provider({"prompt_tokens": 0, "completion_tokens": 0})
        assert usage is not None
        assert usage.input_tokens == 0 and usage.output_tokens == 0

    def test_metadata_is_empty(self) -> None:
        assert token_usage_metadata(None) == {}


class TestSumming:
    def test_a_field_one_side_lacks_is_kept_from_the_other(self) -> None:
        a = TokenUsage(input_tokens=10, output_tokens=2, total_tokens=12)
        b = TokenUsage(input_tokens=11, output_tokens=3, total_tokens=14, reasoning_tokens=1)
        total = a + b
        assert total.as_metadata() == {
            "input_tokens": 21,
            "output_tokens": 5,
            "total_tokens": 26,
            "reasoning_tokens": 1,
            "requests": 2,
        }

    def test_a_response_without_usage_leaves_the_total(self) -> None:
        response = LLMResponse(content="x", generated_tokens=0, execution_time=0.0)
        total = TokenUsage(input_tokens=1)
        assert add_token_usage(total, response) is total
        assert add_token_usage(None, response) is None
        assert add_token_usage(None, None) is None

    def test_metadata_key(self) -> None:
        assert token_usage_metadata(TokenUsage(input_tokens=1)) == {
            "token_usage": {"input_tokens": 1, "requests": 1}
        }
