"""Telling "the prompt did not fit" apart from every other provider error.

A false negative costs a turn that compaction could have saved; a false positive
costs a compaction nobody needed. Each provider words it differently, and the
LiteLLM gateway wraps vLLM's wording in its own.
"""

import httpx
import pytest

from domyn.agents.llm.base import (
    context_limit_from_error,
    is_context_overflow,
    reported_prompt_tokens,
)


def _rejection(status: int, body: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://llm.test/v1/chat/completions")
    response = httpx.Response(status, request=request, text=body)
    return httpx.HTTPStatusError(f"{status}", request=request, response=response)


@pytest.mark.parametrize(
    "body",
    [
        # A LiteLLM gateway's error for a vLLM backend, verbatim.
        '{"error":{"message":"litellm.ContextWindowExceededError: litellm.BadRequestError: '
        "ContextWindowExceededError: Hosted_vllmException - This model's maximum context "
        'length is 40960 tokens. However, you requested 40000 output tokens"}}',
        "This model's maximum context length is 131072 tokens.",
        '{"error":{"code":"context_length_exceeded"}}',
        "Your input exceeds the context window of this model",
        "prompt is too long: 213462 tokens > 200000 maximum",
        # vLLM's engine, for a prompt exactly the window's length, verbatim.
        "The decoder prompt (length 65536) plus the number of requested output tokens "
        "(at least 1) is longer than the maximum model length of 65536. Make sure that "
        "`max_model_len` is no smaller than the number of text tokens (prompt + requested "
        "output tokens).",
    ],
    ids=["gateway", "vllm", "openai-code", "openai-text", "anthropic", "vllm-engine"],
)
def test_a_prompt_too_long_is_recognised(body: str) -> None:
    assert is_context_overflow(_rejection(400, body)) is True


def test_asking_for_more_output_than_the_model_allows_is_not_overflow() -> None:
    """The prompt fits; `max_tokens` does not. Compacting would change nothing."""
    body = (
        "max_tokens=100000000 cannot be greater than max_model_len=max_total_tokens=40960. "
        "Please request fewer output tokens."
    )
    assert is_context_overflow(_rejection(400, body)) is False


def test_vllms_encoder_budget_is_neither_overflow_nor_a_window() -> None:
    """Worded like the decoder prompt's rejection, but the number is the encoder's own
    budget (`mm_encoder_cache_size`), and compacting the conversation does not shorten
    an encoder input."""
    error = _rejection(
        400,
        "The encoder prompt (length 9000) is longer than the maximum model length of 8192. "
        "Make sure that `max_model_len` is no smaller than the number of text tokens.",
    )
    assert is_context_overflow(error) is False
    assert context_limit_from_error(error) is None


def test_only_a_rejected_request_counts() -> None:
    """A rate limit or a server error is not about this prompt, whatever it says."""
    assert is_context_overflow(_rejection(429, "maximum context length")) is False
    assert is_context_overflow(_rejection(500, "maximum context length")) is False


def test_an_error_without_a_response_is_not_overflow() -> None:
    assert is_context_overflow(ValueError("maximum context length")) is False


@pytest.mark.parametrize(
    ("body", "window"),
    [
        (
            # A LiteLLM gateway's error for a vLLM backend, verbatim.
            '{"error":{"message":"litellm.ContextWindowExceededError: litellm.BadRequestError: '
            "ContextWindowExceededError: Hosted_vllmException - This model's maximum context "
            'length is 40960 tokens. However, you requested 4000 output tokens"}}',
            40960,
        ),
        (
            # OpenRouter, verbatim: "endpoint", where vLLM says "model".
            "This endpoint's maximum context length is 16384 tokens. However, you requested "
            "about 17991 tokens (17975 of text input, 16 in the output).",
            16384,
        ),
        (
            # vLLM's later check through the gateway, verbatim: the one gpt-oss meets,
            # with the window in parentheses.
            '{"error":{"message":"litellm.BadRequestError: Hosted_vllmException - {\\"error\\":'
            '{\\"message\\":\\"Input length (170860) exceeds model\'s maximum context length '
            '(128000).\\",\\"type\\":\\"BadRequestError\\",\\"param\\":null,\\"code\\":400}}. '
            "Received Model Group=openai/gpt-oss-120b Available Model Group Fallbacks=None"
            '","type":null,"param":null,"code":"400"}}',
            128000,
        ),
        ("prompt is too long: 213462 tokens > 200000 maximum", 200000),
        (
            # vLLM's engine through the gateway, which does not map it to a context error.
            '{"error":{"message":"litellm.BadRequestError: Hosted_vllmException - The decoder '
            "prompt (length 70412) is longer than the maximum model length of 65536. Make sure "
            'that `max_model_len` is no smaller than the number of text tokens."}}',
            65536,
        ),
        (
            # The gateway's own pre-call check: one entry per deployment it ruled out, run
            # together. A request that fits the widest is routed there.
            '{"error":{"message":"litellm.ContextWindowExceededError: litellm._pre_call_checks: '
            "Context Window exceeded for given call. No models have context window large "
            "enough for this call.\\nModel=hosted_vllm/Qwen/Qwen3-32B, Max Input Tokens=32768, "
            'Got=40123Model=hosted_vllm/Qwen/Qwen3-32B, Max Input Tokens=40960, Got=40123"}}',
            40960,
        ),
    ],
    ids=[
        "gateway",
        "openrouter",
        "vllm-input-length-via-gateway",
        "anthropic",
        "vllm-engine-via-gateway",
        "litellm-pre-call",
    ],
)
def test_a_rejection_that_states_the_window_gives_it(body: str, window: int) -> None:
    assert context_limit_from_error(_rejection(400, body)) == window


@pytest.mark.parametrize(
    "error",
    [
        _rejection(400, '{"error":{"code":"context_length_exceeded"}}'),
        _rejection(400, "max_tokens=50000 cannot be greater than max_model_len=40960"),
        _rejection(429, "This model's maximum context length is 40960 tokens."),
    ],
    ids=["overflow-without-a-number", "output-cap", "not-a-rejection"],
)
def test_no_window_is_read_where_none_is_stated(error: httpx.HTTPStatusError) -> None:
    """A number from another kind of error would become a wrong window."""
    assert context_limit_from_error(error) is None


@pytest.mark.parametrize(
    ("usage", "prompt"),
    [
        ({"prompt_tokens": 9704, "completion_tokens": 2}, 9704),
        ({"input_tokens": 6630, "input_tokens_details": {"cached_tokens": 0}}, 6630),
        ({"input_tokens": 8561, "output_tokens": 4}, 8561),
        # Anthropic with a cache in play, as measured: `input_tokens` alone is 4.
        ({"input_tokens": 4, "cache_creation_input_tokens": 8557, "output_tokens": 4}, 8561),
        ({"input_tokens": 4, "cache_read_input_tokens": 8557}, 8561),
    ],
    ids=[
        "chat-completions",
        "openai-responses",
        "anthropic",
        "anthropic-cache-write",
        "anthropic-cache-read",
    ],
)
def test_the_reported_prompt_is_the_whole_prompt(usage: dict, prompt: int) -> None:
    assert reported_prompt_tokens(usage) == prompt


@pytest.mark.parametrize(
    "usage", [None, {}, {"output_tokens": 4}, {"prompt_tokens": True}, {"prompt_tokens": "9704"}]
)
def test_a_usage_without_a_count_reports_nothing(usage: dict | None) -> None:
    """`None`, not 0: "no count" and "an empty prompt" lead to opposite decisions."""
    assert reported_prompt_tokens(usage) is None
