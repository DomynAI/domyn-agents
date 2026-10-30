"""Tests for VLLM Provider - generation_params merging."""

import json
import os

import httpx
import pytest
import respx

from domyn.agents.llm import (
    LLMMessage,
    LLMResponse,
    LLMToolCall,
    StructuredOutput,
    StructuredOutputType,
    VLLMProvider,
)
from tests.llm.test_inference_logging import recording_logger

vllm_api_key = os.getenv("VLLM_API_KEY") or os.getenv("VLLM_API_KEY_DEFAULT")


class TestVLLMProviderGenerationParams:
    """Test cases for VLLMProvider generation_params merging."""

    def test_default_generation_params(self):
        """Test that default generation_params are set correctly."""
        provider = VLLMProvider(
            url="https://llm.example.com/v1/chat/completions",
            model_name="test-model",
        )

        assert provider.generation_params["temperature"] == 0.7
        assert provider.generation_params["max_tokens"] == 2000

    def test_partial_generation_params_merge(self):
        """Test that partial generation_params are merged with defaults."""
        provider = VLLMProvider(
            url="https://llm.example.com/v1/chat/completions",
            model_name="test-model",
            generation_params={"temperature": 0.9},
        )

        # Should have the custom temperature
        assert provider.generation_params["temperature"] == 0.9
        # Should still have the default max_tokens
        assert provider.generation_params["max_tokens"] == 2000

    def test_full_generation_params_override(self):
        """Test that all generation_params can be overridden."""
        provider = VLLMProvider(
            url="https://llm.example.com/v1/chat/completions",
            model_name="test-model",
            generation_params={"temperature": 0.9, "max_tokens": 1000},
        )

        assert provider.generation_params["temperature"] == 0.9
        assert provider.generation_params["max_tokens"] == 1000

    def test_empty_generation_params(self):
        """Test that empty generation_params dict still gets defaults."""
        provider = VLLMProvider(
            url="https://llm.example.com/v1/chat/completions",
            model_name="test-model",
            generation_params={},
        )

        # Should have the default values
        assert provider.generation_params["temperature"] == 0.7
        assert provider.generation_params["max_tokens"] == 2000

    def test_additional_generation_params(self):
        """Test that additional generation_params are preserved."""
        provider = VLLMProvider(
            url="https://llm.example.com/v1/chat/completions",
            api_key=vllm_api_key,
            model_name="test-model",
            generation_params={
                "temperature": 0.8,
                "top_p": 0.95,
                "frequency_penalty": 0.5,
            },
        )

        # Should have custom temperature
        assert provider.generation_params["temperature"] == 0.8
        # Should have default max_tokens
        assert provider.generation_params["max_tokens"] == 2000
        # Should have additional params
        assert provider.generation_params["top_p"] == 0.95
        assert provider.generation_params["frequency_penalty"] == 0.5


VLLM_URL = "https://test.gateway.com/v1/chat/completions"

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


# ---------------------------------------------------------------------------
# preprocess_structured_outputs
# ---------------------------------------------------------------------------


class TestVLLMPreprocessStructuredOutputs:
    """Tests for VLLMProvider.preprocess_structured_outputs."""

    def setup_method(self):
        self.provider = VLLMProvider(url=VLLM_URL, model_name="test-model")

    def test_none_returns_empty_dict(self):
        """None input must return an empty dict (no structured output params)."""
        result = self.provider.preprocess_structured_outputs(None)
        assert result == {}

    def test_llguidance_returns_correct_params(self):
        """LLGUIDANCE type must produce the guidance backend params."""
        so = StructuredOutput(type=StructuredOutputType.LLGUIDANCE, rules='{"type":"object"}')
        result = self.provider.preprocess_structured_outputs(so)
        assert "structured_outputs" in result
        assert result["structured_outputs"]["_backend"] == "guidance"
        assert result["structured_outputs"]["grammar"] == '{"type":"object"}'

    def test_gbnf_returns_correct_params(self):
        """GBNF type must produce the xgrammar backend params."""
        so = StructuredOutput(type=StructuredOutputType.GBNF, rules="root ::= value")
        result = self.provider.preprocess_structured_outputs(so)
        assert "structured_outputs" in result
        assert result["structured_outputs"]["_backend"] == "xgrammar"
        assert result["structured_outputs"]["grammar"] == "root ::= value"

    def test_json_schema_raises_not_implemented(self):
        """JSON_SCHEMA type must raise NotImplementedError."""
        so = StructuredOutput(type=StructuredOutputType.JSON_SCHEMA, rules={"type": "object"})
        with pytest.raises(NotImplementedError):
            self.provider.preprocess_structured_outputs(so)

    def test_llguidance_grammar_forwarded_verbatim(self):
        """Grammar value must be forwarded without modification."""
        grammar = '{"type":"string","pattern":"^[A-Z]+"}'
        so = StructuredOutput(type=StructuredOutputType.LLGUIDANCE, rules=grammar)
        result = self.provider.preprocess_structured_outputs(so)
        assert result["structured_outputs"]["grammar"] == grammar


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


class TestVLLMGenerate:
    """Tests for VLLMProvider.generate."""

    def setup_method(self):
        self.provider = VLLMProvider(
            url=VLLM_URL,
            model_name="test-model",
            api_key="test-key",
        )

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_success_plain_text(self):
        """Successful call returns content and correct token count."""
        respx.post(VLLM_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "Hello!", "tool_calls": None}}],
                    "usage": {"completion_tokens": 5},
                },
            )
        )
        messages = [LLMMessage(role="user", content="Hi")]
        response = await self.provider.generate(messages)

        assert isinstance(response, LLMResponse)
        assert response.content == "Hello!"
        assert response.generated_tokens == 5
        assert response.tool_calls == []
        assert response.execution_time > 0

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_http_error_raises(self):
        """HTTP error must bubble up as HTTPStatusError."""
        respx.post(VLLM_URL).mock(return_value=httpx.Response(500, json={"error": "oops"}))
        with pytest.raises(httpx.HTTPStatusError):
            await self.provider.generate([LLMMessage(role="user", content="Hi")])

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_adds_api_key_header(self):
        """Bearer token must be present in Authorization header when api_key is set."""
        captured = {}

        def capture(request):
            captured["auth"] = request.headers.get("authorization")
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok", "tool_calls": None}}],
                    "usage": {"completion_tokens": 1},
                },
            )

        respx.post(VLLM_URL).mock(side_effect=capture)
        await self.provider.generate([LLMMessage(role="user", content="Hi")])
        assert captured["auth"] == "Bearer test-key"

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_no_api_key_no_auth_header(self):
        """When api_key is None the Authorization header must not be sent."""
        provider = VLLMProvider(url=VLLM_URL, model_name="test-model")
        captured = {}

        def capture(request):
            captured["auth"] = request.headers.get("authorization")
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok", "tool_calls": None}}],
                    "usage": {"completion_tokens": 1},
                },
            )

        respx.post(VLLM_URL).mock(side_effect=capture)
        await provider.generate([LLMMessage(role="user", content="Hi")])
        assert captured["auth"] is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_sends_tools_in_payload(self):
        """Tools list must appear in the request payload."""
        captured = {}

        def capture(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok", "tool_calls": None}}],
                    "usage": {"completion_tokens": 1},
                },
            )

        respx.post(VLLM_URL).mock(side_effect=capture)
        await self.provider.generate(
            [LLMMessage(role="user", content="weather?")], tools=SAMPLE_TOOLS
        )
        assert "tools" in captured["body"]
        assert captured["body"]["tools"][0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_sends_tool_choice(self):
        """tool_choice must appear in the payload when provided."""
        captured = {}

        def capture(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "", "tool_calls": None}}],
                    "usage": {"completion_tokens": 0},
                },
            )

        respx.post(VLLM_URL).mock(side_effect=capture)
        await self.provider.generate(
            [LLMMessage(role="user", content="weather?")],
            tools=SAMPLE_TOOLS,
            tool_choice="auto",
        )
        assert captured["body"]["tool_choice"] == "auto"

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_no_tools_not_in_payload(self):
        """When tools=None neither 'tools' nor 'tool_choice' must appear in payload."""
        captured = {}

        def capture(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "hi", "tool_calls": None}}],
                    "usage": {"completion_tokens": 2},
                },
            )

        respx.post(VLLM_URL).mock(side_effect=capture)
        await self.provider.generate([LLMMessage(role="user", content="Hi")])
        assert "tools" not in captured["body"]
        assert "tool_choice" not in captured["body"]

    @pytest.mark.parametrize(
        ("configured", "sent"),
        [({}, None), ({"truncate_prompt_tokens": -1}, -1)],
        ids=["not asked for by default", "kept when configured"],
    )
    @pytest.mark.asyncio
    @respx.mock
    async def test_prompt_truncation_is_opt_in(self, configured: dict, sent: int | None):
        """vLLM's own default rejects a prompt that does not fit, and the tool-use
        planners answer that rejection by compacting and retrying. Asking for
        truncation by default turned it into a silent cut of the prompt's
        beginning — system prompt, tool definitions — with nothing to recover."""
        captured = {}

        def capture(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "hi"}}],
                    "usage": {"completion_tokens": 1},
                },
            )

        respx.post(VLLM_URL).mock(side_effect=capture)
        provider = VLLMProvider(url=VLLM_URL, model_name="test-model", generation_params=configured)
        await provider.generate([LLMMessage(role="user", content="Hi")])
        assert captured["body"].get("truncate_prompt_tokens") == sent

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_parses_tool_calls(self):
        """Tool calls in the response must be parsed into LLMToolCall objects."""
        respx.post(VLLM_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_123",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": '{"city":"Rome"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"completion_tokens": 15},
                },
            )
        )
        response = await self.provider.generate(
            [LLMMessage(role="user", content="weather in Rome?")], tools=SAMPLE_TOOLS
        )
        assert len(response.tool_calls) == 1
        tc = response.tool_calls[0]
        assert isinstance(tc, LLMToolCall)
        assert tc.id == "call_123"
        assert tc.function.name == "get_weather"
        assert json.loads(tc.function.arguments) == {"city": "Rome"}

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_parses_multiple_tool_calls(self):
        """Multiple tool calls must all be parsed."""
        respx.post(VLLM_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_001",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": '{"city":"Rome"}',
                                        },
                                    },
                                    {
                                        "id": "call_002",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": '{"city":"Paris"}',
                                        },
                                    },
                                ],
                            }
                        }
                    ],
                    "usage": {"completion_tokens": 30},
                },
            )
        )
        response = await self.provider.generate(
            [LLMMessage(role="user", content="compare weather")], tools=SAMPLE_TOOLS
        )
        assert len(response.tool_calls) == 2
        assert response.tool_calls[0].id == "call_001"
        assert response.tool_calls[1].id == "call_002"

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_no_tool_calls_returns_empty_list(self):
        """Plain-text response must yield tool_calls == []."""
        respx.post(VLLM_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "Just text.", "tool_calls": None}}],
                    "usage": {"completion_tokens": 3},
                },
            )
        )
        response = await self.provider.generate([LLMMessage(role="user", content="Hi")])
        assert response.tool_calls == []
        assert response.content == "Just text."

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_appends_system_prompt(self):
        """append_text_system_prompt must be appended to the first message."""
        provider = VLLMProvider(
            url=VLLM_URL,
            model_name="test-model",
            append_text_system_prompt=" /sys_suffix",
        )
        captured = {}

        def capture(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok", "tool_calls": None}}],
                    "usage": {"completion_tokens": 1},
                },
            )

        respx.post(VLLM_URL).mock(side_effect=capture)
        messages = [
            LLMMessage(role="system", content="You are helpful."),
            LLMMessage(role="user", content="Hi"),
        ]
        await provider.generate(messages)
        assert captured["body"]["messages"][0]["content"] == "You are helpful.\n /sys_suffix"

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_appends_last_message(self):
        """append_text_last_message must be appended to the last message."""
        provider = VLLMProvider(
            url=VLLM_URL,
            model_name="test-model",
            append_text_last_message=" /suffix",
        )
        captured = {}

        def capture(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok", "tool_calls": None}}],
                    "usage": {"completion_tokens": 1},
                },
            )

        respx.post(VLLM_URL).mock(side_effect=capture)
        messages = [LLMMessage(role="user", content="Hello")]
        await provider.generate(messages)
        assert captured["body"]["messages"][-1]["content"] == "Hello\n /suffix"

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_reasoning_content_forwarded(self):
        """reasoning_content from the model response must be set on LLMResponse."""
        respx.post(VLLM_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "The answer is 42.",
                                "tool_calls": None,
                                "reasoning_content": "Let me think... 42 is the answer.",
                            }
                        }
                    ],
                    "usage": {"completion_tokens": 10},
                },
            )
        )
        response = await self.provider.generate([LLMMessage(role="user", content="What is 42?")])
        assert response.reasoning_content == "Let me think... 42 is the answer."

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_with_llguidance_structured_output(self):
        """LLGUIDANCE structured output params must appear in the payload."""
        captured = {}

        def capture(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": '{"name":"Alice"}', "tool_calls": None}}],
                    "usage": {"completion_tokens": 8},
                },
            )

        respx.post(VLLM_URL).mock(side_effect=capture)
        so = StructuredOutput(type=StructuredOutputType.LLGUIDANCE, rules='{"type":"object"}')
        await self.provider.generate(
            [LLMMessage(role="user", content="Give me JSON")], structured_outputs=so
        )

        assert "structured_outputs" in captured["body"]
        assert captured["body"]["structured_outputs"]["_backend"] == "guidance"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VLLM_URL = "https://test.endpoint.com/v1/chat/completions"


def _sse(*payloads: dict) -> bytes:
    """Encode a sequence of dicts as SSE lines followed by [DONE]."""
    lines = "".join(f"data: {json.dumps(p)}\n\n" for p in payloads)
    lines += "data: [DONE]\n\n"
    return lines.encode()


def _delta_chunk(
    content: str | None = None,
    tool_calls: list | None = None,
    logprobs: dict | None = None,
) -> dict:
    """Build a minimal streaming chunk dict."""
    delta: dict = {}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    choice: dict = {"delta": delta}
    if logprobs is not None:
        choice["logprobs"] = logprobs
    return {"choices": [choice]}


class TestVLLMGenerateStreaming:
    """Tests for VLLMProvider.generate_streaming — including tool-call accumulation."""

    @pytest.fixture(autouse=True)
    def provider(self):
        self.provider = VLLMProvider(url=VLLM_URL, model_name="test-model")

    # ------------------------------------------------------------------
    # Plain text streaming
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_yields_text_tokens(self):
        """Each text delta must be yielded as a separate LLMResponse."""
        body = _sse(
            _delta_chunk(content="Hello"),
            _delta_chunk(content=" world"),
        )
        respx.post(VLLM_URL).mock(return_value=httpx.Response(200, content=body))

        chunks = [
            r
            async for r in self.provider.generate_streaming([LLMMessage(role="user", content="Hi")])
        ]

        texts = [c.content for c in chunks if c.content]
        assert texts == ["Hello", " world"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_yields_logprobs_per_content_chunk(self):
        """Each content chunk's logprobs travel with that chunk, unset otherwise."""
        hello_logprobs = {"content": [{"token": "Hello", "logprob": -0.1}]}
        world_logprobs = {"content": [{"token": " world", "logprob": -0.2}]}
        body = _sse(
            _delta_chunk(content="Hello", logprobs=hello_logprobs),
            _delta_chunk(content=" world", logprobs=world_logprobs),
        )
        respx.post(VLLM_URL).mock(return_value=httpx.Response(200, content=body))

        chunks = [
            r
            async for r in self.provider.generate_streaming([LLMMessage(role="user", content="Hi")])
        ]

        content_chunks = [c for c in chunks if c.content]
        assert [c.logprobs for c in content_chunks] == [hello_logprobs, world_logprobs]

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_logprobs_absent_when_not_requested(self):
        """No logprobs key on the wire means None on the response, not an error."""
        body = _sse(_delta_chunk(content="Hello"))
        respx.post(VLLM_URL).mock(return_value=httpx.Response(200, content=body))

        chunks = [
            r
            async for r in self.provider.generate_streaming([LLMMessage(role="user", content="Hi")])
        ]

        assert all(c.logprobs is None for c in chunks)

    @pytest.mark.parametrize(
        "closing",
        [
            {"choices": [], "usage": {"prompt_tokens": 85, "completion_tokens": 18}},
            {"choices": [{"index": 0, "delta": {}}], "usage": {"prompt_tokens": 85}},
        ],
        ids=["vllm-no-choices", "gateway-empty-delta"],
    )
    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_usage_reaches_the_last_chunk(self, closing: dict) -> None:
        """`include_usage` closes the stream with the counts, in either shape;
        a consumer reads them off the last chunk."""
        body = _sse(_delta_chunk(content="Hello"), closing)
        respx.post(VLLM_URL).mock(return_value=httpx.Response(200, content=body))

        chunks = [
            r
            async for r in self.provider.generate_streaming([LLMMessage(role="user", content="Hi")])
        ]

        assert "".join(c.content for c in chunks) == "Hello"
        assert chunks[-1].usage == closing["usage"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_usage_travels_with_the_tool_calls(self) -> None:
        """With tool calls the last chunk is the one carrying them: the usage
        goes there, not after it, where a consumer would take it for the answer."""
        call = {"index": 0, "id": "c1", "type": "function"}
        call["function"] = {"name": "lookup", "arguments": "{}"}
        body = _sse(_delta_chunk(tool_calls=[call]), {"choices": [], "usage": {"prompt_tokens": 9}})
        respx.post(VLLM_URL).mock(return_value=httpx.Response(200, content=body))

        chunks = [
            r
            async for r in self.provider.generate_streaming([LLMMessage(role="user", content="Hi")])
        ]

        assert chunks[-1].tool_calls and chunks[-1].usage == {"prompt_tokens": 9}

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_logs_error_body(self, recording_logger):  # noqa: F811 — fixture
        """A 4xx on the streamed request must surface the response body in logs.

        Guards the ``client.stream()`` + ``await response.aread()`` path: the
        error body must be read before the stream context closes, otherwise it
        is lost.

        The body is logged at DEBUG level in the structured payload (not in the
        message, not at ERROR) to avoid leaking sensitive data in production
        logs — so it is only visible through a payload-aware injected logger.
        """
        respx.post(VLLM_URL).mock(
            return_value=httpx.Response(400, json={"error": {"message": "boom-xyz"}})
        )

        with pytest.raises(httpx.HTTPStatusError):
            async for _ in self.provider.generate_streaming(
                [LLMMessage(role="user", content="Hi")]
            ):
                pass

        assert "vLLM API 400" in recording_logger.dump("error")
        debug_payloads = [p for level, _, p in recording_logger.records if level == "debug" and p]
        assert any("boom-xyz" in str(p.get("body", "")) for p in debug_payloads)

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_no_tool_call_chunk_when_no_tools(self):
        """If there are no tool_calls, no chunk with tool_calls must be yielded."""
        body = _sse(_delta_chunk(content="ok"))
        respx.post(VLLM_URL).mock(return_value=httpx.Response(200, content=body))

        chunks = [
            r
            async for r in self.provider.generate_streaming([LLMMessage(role="user", content="Hi")])
        ]

        assert all(not r.tool_calls for r in chunks)

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_sends_tools_in_payload(self):
        """tools and tool_choice must be forwarded in the request payload."""
        captured = {}

        def capture(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, content=_sse(_delta_chunk(content="ok")))

        respx.post(VLLM_URL).mock(side_effect=capture)

        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
        await self.provider.generate_streaming(
            [LLMMessage(role="user", content="weather?")],
            tools=tools,
            tool_choice="auto",
        ).__anext__()

        assert captured["body"]["tools"] == tools
        assert captured["body"]["tool_choice"] == "auto"

    # ------------------------------------------------------------------
    # Single tool call — arguments spread across multiple chunks
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_single_tool_call_accumulated(self):
        """A tool call whose arguments arrive in 3 chunks must be fully reassembled."""
        body = _sse(
            # chunk 1: id + name + first fragment of arguments
            _delta_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_001",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"cit'},
                    }
                ]
            ),
            # chunk 2: more arguments
            _delta_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "function": {"arguments": 'y": "Rome'},
                    }
                ]
            ),
            # chunk 3: closing brace
            _delta_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "function": {"arguments": '"}'},
                    }
                ]
            ),
        )
        respx.post(VLLM_URL).mock(return_value=httpx.Response(200, content=body))

        chunks = [
            r
            async for r in self.provider.generate_streaming(
                [LLMMessage(role="user", content="weather?")]
            )
        ]

        tool_chunks = [c for c in chunks if c.tool_calls]
        assert len(tool_chunks) == 1, "exactly one chunk must carry the completed tool calls"

        tc = tool_chunks[0].tool_calls[0]
        assert tc.id == "call_001"
        assert tc.function.name == "get_weather"
        assert tc.function.arguments == '{"city": "Rome"}'

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_tool_call_no_text_chunks_yielded(self):
        """While accumulating a tool call, no intermediate text chunk must be yielded."""
        body = _sse(
            _delta_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_x",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"a":1,'},
                    }
                ]
            ),
            _delta_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "function": {"arguments": '"b":2}'},
                    }
                ]
            ),
        )
        respx.post(VLLM_URL).mock(return_value=httpx.Response(200, content=body))

        chunks = [
            r
            async for r in self.provider.generate_streaming(
                [LLMMessage(role="user", content="add")]
            )
        ]

        text_chunks = [c for c in chunks if c.content]
        assert text_chunks == [], "no text chunks must be emitted for a pure tool-call response"

    # ------------------------------------------------------------------
    # Multiple tool calls in the same response
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_multiple_tool_calls(self):
        """Two parallel tool calls with interleaved argument chunks must both be correct."""
        body = _sse(
            # first chunks for tool 0 and tool 1 (id + name)
            _delta_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_A",
                        "type": "function",
                        "function": {"name": "tool_a", "arguments": '{"x":'},
                    },
                    {
                        "index": 1,
                        "id": "call_B",
                        "type": "function",
                        "function": {"name": "tool_b", "arguments": '{"y":'},
                    },
                ]
            ),
            # argument continuations
            _delta_chunk(
                tool_calls=[
                    {"index": 0, "function": {"arguments": "1}"}},
                    {"index": 1, "function": {"arguments": "2}"}},
                ]
            ),
        )
        respx.post(VLLM_URL).mock(return_value=httpx.Response(200, content=body))

        chunks = [
            r
            async for r in self.provider.generate_streaming(
                [LLMMessage(role="user", content="call both")]
            )
        ]

        tool_chunks = [c for c in chunks if c.tool_calls]
        assert len(tool_chunks) == 1
        tcs = tool_chunks[0].tool_calls
        assert len(tcs) == 2

        by_id = {tc.id: tc for tc in tcs}
        assert by_id["call_A"].function.name == "tool_a"
        assert by_id["call_A"].function.arguments == '{"x":1}'
        assert by_id["call_B"].function.name == "tool_b"
        assert by_id["call_B"].function.arguments == '{"y":2}'

    # ------------------------------------------------------------------
    # Mixed: text tokens + tool call in the same response
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_mixed_text_and_tool_call(self):
        """Text tokens yielded before the tool-call chunk must be preserved."""
        body = _sse(
            _delta_chunk(content="Sure, "),
            _delta_chunk(content="calling now."),
            _delta_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_Z",
                        "type": "function",
                        "function": {"name": "do_it", "arguments": "{}"},
                    }
                ]
            ),
        )
        respx.post(VLLM_URL).mock(return_value=httpx.Response(200, content=body))

        chunks = [
            r
            async for r in self.provider.generate_streaming([LLMMessage(role="user", content="go")])
        ]

        texts = [c.content for c in chunks if c.content]
        assert texts == ["Sure, ", "calling now."]

        tool_chunks = [c for c in chunks if c.tool_calls]
        assert len(tool_chunks) == 1
        assert tool_chunks[0].tool_calls[0].id == "call_Z"
        assert tool_chunks[0].tool_calls[0].function.name == "do_it"

    # ------------------------------------------------------------------
    # HTTP error
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_http_error_raises(self):
        """A non-2xx response must raise an httpx.HTTPStatusError."""
        respx.post(VLLM_URL).mock(return_value=httpx.Response(500))

        with pytest.raises(httpx.HTTPStatusError):
            async for _ in self.provider.generate_streaming(
                [LLMMessage(role="user", content="Hi")]
            ):
                pass

    # ------------------------------------------------------------------
    # id/type/name update in subsequent chunks (else branch)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_tool_call_id_type_name_updated_in_subsequent_chunk(self):
        """
        If a backend sends id/type/name in a chunk that is NOT the first for that
        index (i.e. the else branch of the accumulator), they must overwrite the
        previously stored empty/placeholder values.

        Sequence:
          chunk 1 - index 0 arrives with NO id/type/name (only arguments fragment)
          chunk 2 - index 0 arrives again with id, type and name  (exercises the else branch)
          chunk 3 - index 0 final arguments fragment
        """
        body = _sse(
            # chunk 1: first occurrence of index 0, but id/type/name are absent
            _delta_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "function": {"arguments": '{"ci'},
                    }
                ]
            ),
            # chunk 2: same index — id, type and name arrive late (else branch)
            _delta_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_late",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "ty"},
                    }
                ]
            ),
            # chunk 3: only the closing fragment of arguments
            _delta_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "function": {"arguments": '": "Rome"}'},
                    }
                ]
            ),
        )
        respx.post(VLLM_URL).mock(return_value=httpx.Response(200, content=body))

        chunks = [
            r
            async for r in self.provider.generate_streaming(
                [LLMMessage(role="user", content="weather?")]
            )
        ]

        tool_chunks = [c for c in chunks if c.tool_calls]
        assert len(tool_chunks) == 1

        tc = tool_chunks[0].tool_calls[0]
        # id and name must reflect the values from chunk 2, not the empty defaults
        assert tc.id == "call_late"
        assert tc.type == "function"
        assert tc.function.name == "get_weather"
        # arguments must be the full concatenation of all three chunks
        assert tc.function.arguments == '{"city": "Rome"}'
