"""Tests for Anthropic LLM Provider."""

import json
from unittest.mock import patch

import httpx
import pytest
import respx

from domyn.agents.llm import AnthropicProvider
from domyn.agents.llm.base import LLMMessage, LLMResponse, reported_prompt_tokens


class TestAnthropicProvider:
    """Test cases for AnthropicProvider."""

    def test_init(self):
        """Test provider initialization."""
        provider = AnthropicProvider(model_name="claude-3-5-sonnet-20241022", api_key="test-key")

        assert provider.type == "anthropic"
        assert provider.name == "claude-3-5-sonnet-20241022"
        assert provider.url == "https://api.anthropic.com/v1/messages"

    def test_generation_params_merging(self):
        """Test that partial generation_params are merged with defaults."""
        # Test partial params - only override temperature
        provider = AnthropicProvider(
            model_name="claude-3-5-sonnet-20241022",
            api_key="test-key",
            generation_params={"temperature": 0.5},
        )

        # Should have custom temperature
        assert provider.generation_params["temperature"] == 0.5
        # Should preserve default values
        assert provider.generation_params["max_tokens"] == 4096

    def test_init_without_api_key(self):
        """Test provider initialization without API key."""
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="ANTHROPIC_API_KEY"),
        ):
            AnthropicProvider(model_name="claude-3-5-sonnet-20241022")

    def test_convert_messages_simple(self):
        """Test message conversion with simple messages."""
        provider = AnthropicProvider(model_name="claude-3-5-sonnet-20241022", api_key="test-key")

        messages = [
            LLMMessage(role="user", content="Hello"),
            LLMMessage(role="assistant", content="Hi there!"),
            LLMMessage(role="user", content="How are you?"),
        ]

        system_prompt, converted = provider._convert_messages(messages)

        assert system_prompt is None
        assert len(converted) == 3
        assert converted[0] == {"role": "user", "content": "Hello"}
        assert converted[1] == {"role": "assistant", "content": "Hi there!"}
        assert converted[2] == {"role": "user", "content": "How are you?"}

    def test_convert_messages_with_system(self):
        """Test message conversion with system messages."""
        provider = AnthropicProvider(model_name="claude-3-5-sonnet-20241022", api_key="test-key")

        messages = [
            LLMMessage(role="system", content="You are a helpful assistant."),
            LLMMessage(role="user", content="Hello"),
        ]

        system_prompt, converted = provider._convert_messages(messages)

        assert system_prompt == "You are a helpful assistant."
        assert len(converted) == 1
        assert converted[0] == {"role": "user", "content": "Hello"}

    def test_convert_messages_with_multiple_system(self):
        """Test message conversion with multiple system messages."""
        provider = AnthropicProvider(model_name="claude-3-5-sonnet-20241022", api_key="test-key")

        messages = [
            LLMMessage(role="system", content="You are a helpful assistant."),
            LLMMessage(role="system", content="Be concise in your responses."),
            LLMMessage(role="user", content="Hello"),
        ]

        system_prompt, converted = provider._convert_messages(messages)

        assert system_prompt == "You are a helpful assistant.\n\nBe concise in your responses."
        assert len(converted) == 1
        assert converted[0] == {"role": "user", "content": "Hello"}

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_success(self):
        """Test successful generation."""
        provider = AnthropicProvider(model_name="claude-3-5-sonnet-20241022", api_key="test-key")

        messages = [LLMMessage(role="user", content="Hello")]

        mock_response_data = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hi there! How can I help you?"}],
            "model": "claude-3-5-sonnet-20241022",
            "usage": {"input_tokens": 5, "output_tokens": 8},
        }

        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=mock_response_data)
        )

        response = await provider.generate(messages, structured_outputs=None)

        assert isinstance(response, LLMResponse)
        assert response.content == "Hi there! How can I help you?"
        assert response.generated_tokens == 8
        assert response.usage["input_tokens"] == 5
        assert response.usage["output_tokens"] == 8
        assert response.execution_time > 0

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_with_system_message(self):
        """Test generation with system message."""
        provider = AnthropicProvider(model_name="claude-3-5-sonnet-20241022", api_key="test-key")

        messages = [
            LLMMessage(role="system", content="You are a helpful assistant."),
            LLMMessage(role="user", content="Hello"),
        ]

        mock_response_data = {
            "content": [{"type": "text", "text": "Hi!"}],
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }

        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=mock_response_data)
        )

        response = await provider.generate(messages, structured_outputs=None)

        assert response.content == "Hi!"

        # Verify the request included system parameter
        request = route.calls.last.request
        payload = request.read()
        import json

        payload_data = json.loads(payload)
        assert payload_data["system"] == "You are a helpful assistant."
        assert payload_data["messages"] == [{"role": "user", "content": "Hello"}]

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_with_multiple_text_blocks(self):
        """Test generation with multiple text blocks in response."""
        provider = AnthropicProvider(model_name="claude-3-5-sonnet-20241022", api_key="test-key")

        messages = [LLMMessage(role="user", content="Hello")]

        mock_response_data = {
            "content": [
                {"type": "text", "text": "Hello! "},
                {"type": "text", "text": "How can I assist you today?"},
            ],
            "usage": {"input_tokens": 5, "output_tokens": 10},
        }

        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=mock_response_data)
        )

        response = await provider.generate(messages, structured_outputs=None)

        # Should concatenate all text blocks
        assert response.content == "Hello! How can I assist you today?"

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_http_error(self):
        """Test generation with HTTP error."""
        provider = AnthropicProvider(model_name="claude-3-5-sonnet-20241022", api_key="test-key")

        messages = [LLMMessage(role="user", content="Hello")]

        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(500, json={"error": "Internal Server Error"})
        )

        with pytest.raises(httpx.HTTPStatusError):
            await provider.generate(messages, structured_outputs=None)

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_streaming_success(self):
        """Test successful streaming generation."""
        provider = AnthropicProvider(model_name="claude-3-5-sonnet-20241022", api_key="test-key")

        messages = [LLMMessage(role="user", content="Hello")]

        # Create SSE format streaming response
        streaming_content = (
            "event: message_start\n"
            'data: {"type":"message_start"}\n'
            "\n"
            "event: content_block_start\n"
            'data: {"type":"content_block_start","index":0}\n'
            "\n"
            "event: content_block_delta\n"
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}\n'
            "\n"
            "event: content_block_delta\n"
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" there"}}\n'
            "\n"
            "event: content_block_delta\n"
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"!"}}\n'
            "\n"
            "event: content_block_stop\n"
            'data: {"type":"content_block_stop","index":0}\n'
            "\n"
            "event: message_delta\n"
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":8}}\n'
            "\n"
            "event: message_stop\n"
            'data: {"type":"message_stop"}\n'
            "\n"
        )

        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, content=streaming_content)
        )

        responses = []
        async for response in provider.generate_streaming(messages):
            responses.append(response)

        # 3 text deltas, then the closing chunk carrying the usage
        assert [r.content for r in responses] == ["Hi", " there", "!", ""]
        assert responses[-1].usage == {"output_tokens": 8}

        # Check token counting
        for i, response in enumerate(responses[:3]):
            assert response.generated_tokens == i + 1
            assert response.execution_time > 0

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_streaming_with_system_message(self):
        """Test streaming with system message."""
        provider = AnthropicProvider(model_name="claude-3-5-sonnet-20241022", api_key="test-key")

        messages = [
            LLMMessage(role="system", content="Be concise."),
            LLMMessage(role="user", content="Hello"),
        ]

        streaming_content = (
            "event: content_block_delta\n"
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}\n'
            "\n"
            "event: message_stop\n"
            'data: {"type":"message_stop"}\n'
            "\n"
        )

        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, content=streaming_content)
        )

        responses = []
        async for response in provider.generate_streaming(messages):
            responses.append(response)

        assert len(responses) == 1
        assert responses[0].content == "Hi"

        # Verify system message was passed
        request = route.calls.last.request
        payload = request.read()
        import json

        payload_data = json.loads(payload)
        assert payload_data["system"] == "Be concise."
        assert payload_data["stream"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_streaming_empty_events(self):
        """Test streaming with no text events."""
        provider = AnthropicProvider(model_name="claude-3-5-sonnet-20241022", api_key="test-key")

        messages = [LLMMessage(role="user", content="Hello")]

        # Only message events, no content deltas
        streaming_content = (
            "event: message_start\n"
            'data: {"type":"message_start"}\n'
            "\n"
            "event: message_stop\n"
            'data: {"type":"message_stop"}\n'
            "\n"
        )

        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, content=streaming_content)
        )

        responses = []
        async for response in provider.generate_streaming(messages):
            responses.append(response)

        # Should get no responses since there are no content deltas
        assert len(responses) == 0

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_streaming_error(self):
        """Test streaming with API error."""
        provider = AnthropicProvider(model_name="claude-3-5-sonnet-20241022", api_key="test-key")

        messages = [LLMMessage(role="user", content="Hello")]

        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(500, json={"error": "Streaming Error"})
        )

        with pytest.raises(httpx.HTTPStatusError):
            async for _ in provider.generate_streaming(messages):
                pass

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_streaming_usage_info(self):
        """Test that streaming collects usage information."""
        provider = AnthropicProvider(model_name="claude-3-5-sonnet-20241022", api_key="test-key")

        messages = [LLMMessage(role="user", content="Hello")]

        streaming_content = (
            "event: content_block_delta\n"
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}\n'
            "\n"
            "event: message_delta\n"
            'data: {"type":"message_delta","usage":{"output_tokens":5}}\n'
            "\n"
            "event: message_stop\n"
            'data: {"type":"message_stop"}\n'
            "\n"
        )

        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, content=streaming_content)
        )

        responses = []
        async for response in provider.generate_streaming(messages):
            responses.append(response)

        assert [r.content for r in responses] == ["Hello", ""]
        assert responses[-1].usage == {"output_tokens": 5}

    @pytest.mark.parametrize("stream", [False, True], ids=["plain", "streamed"])
    @pytest.mark.asyncio
    @respx.mock
    async def test_the_cache_counts_are_kept(self, stream: bool):
        """With a cache in play `input_tokens` is only the uncached tail of the
        prompt — measured, 4 of 8,561 — so dropping the cache counts made the
        prompt look nearly empty to anyone reading its size."""
        provider = AnthropicProvider(model_name="claude-haiku-5-5", api_key="test-key")
        usage = {
            "input_tokens": 4,
            "cache_read_input_tokens": 8557,
            "cache_creation_input_tokens": 0,
        }
        if stream:
            start = {"type": "message_start", "message": {"usage": {**usage, "output_tokens": 1}}}
            body = (
                f"event: message_start\ndata: {json.dumps(start)}\n\n"
                'event: content_block_delta\ndata: {"type":"content_block_delta",'
                '"delta":{"type":"text_delta","text":"ok"}}\n\n'
                'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":2}}\n\n'
            )
            respx.post("https://api.anthropic.com/v1/messages").mock(
                return_value=httpx.Response(200, content=body)
            )
            chunks = [
                c
                async for c in provider.generate_streaming([LLMMessage(role="user", content="hi")])
            ]
            reported = chunks[-1].usage
        else:
            data = {
                "content": [{"type": "text", "text": "ok"}],
                "usage": {**usage, "output_tokens": 2},
            }
            respx.post("https://api.anthropic.com/v1/messages").mock(
                return_value=httpx.Response(200, json=data)
            )
            reported = (await provider.generate([LLMMessage(role="user", content="hi")])).usage

        assert reported_prompt_tokens(reported) == 8561

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_streaming_malformed_json(self):
        """Test streaming with malformed JSON (should be skipped)."""
        provider = AnthropicProvider(model_name="claude-3-5-sonnet-20241022", api_key="test-key")

        messages = [LLMMessage(role="user", content="Hello")]

        streaming_content = (
            "event: content_block_delta\n"
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Good"}}\n'
            "\n"
            "event: content_block_delta\n"
            "data: {invalid json}\n"  # This should be skipped
            "\n"
            "event: content_block_delta\n"
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":" response"}}\n'
            "\n"
        )

        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, content=streaming_content)
        )

        responses = []
        async for response in provider.generate_streaming(messages):
            responses.append(response)

        # Should only get 2 responses (malformed JSON skipped)
        assert len(responses) == 2
        assert responses[0].content == "Good"
        assert responses[1].content == " response"

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_filters_none_params(self):
        """Test that None values in generation_params are filtered out."""
        provider = AnthropicProvider(model_name="claude-3-5-sonnet-20241022", api_key="test-key")

        messages = [LLMMessage(role="user", content="Hello")]

        mock_response_data = {
            "content": [{"type": "text", "text": "Hi!"}],
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }

        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=mock_response_data)
        )

        await provider.generate(messages, structured_outputs=None)

        # Verify that top_k (which is None) was filtered out
        request = route.calls.last.request
        payload = request.read()
        import json

        payload_data = json.loads(payload)
        assert "max_tokens" in payload_data


# ---------------------------------------------------------------------------
# Helpers — Anthropic SSE format
# ---------------------------------------------------------------------------

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


def _sse_lines(*events: tuple[str, dict]) -> bytes:
    """
    Build a raw SSE byte string from (event_type, data_dict) pairs.
    Each event becomes:
        event: <type>
        data: <json>
        (blank line)
    """
    out = ""
    for event_type, data in events:
        out += f"event: {event_type}\n"
        out += f"data: {json.dumps(data)}\n"
        out += "\n"
    return out.encode()


def _text_delta(idx: int, text: str) -> tuple[str, dict]:
    return (
        "content_block_delta",
        {
            "index": idx,
            "delta": {"type": "text_delta", "text": text},
        },
    )


def _tool_start(idx: int, tool_id: str, name: str) -> tuple[str, dict]:
    return (
        "content_block_start",
        {
            "index": idx,
            "content_block": {"type": "tool_use", "id": tool_id, "name": name},
        },
    )


def _tool_delta(idx: int, partial_json: str) -> tuple[str, dict]:
    return (
        "content_block_delta",
        {
            "index": idx,
            "delta": {"type": "input_json_delta", "partial_json": partial_json},
        },
    )


class TestAnthropicGenerateStreamingToolCalls:
    """Tests for generate_streaming tool-call accumulation in AnthropicProvider."""

    @pytest.fixture(autouse=True)
    def provider(self):
        self.provider = AnthropicProvider(
            model_name="claude-3-5-sonnet-20241022", api_key="test-key"
        )

    # ------------------------------------------------------------------
    # Payload forwarding
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_sends_tools_in_payload(self):
        """tools must be converted to Anthropic format and sent in the payload."""
        captured = {}

        def capture(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, content=b"")

        respx.post(ANTHROPIC_URL).mock(side_effect=capture)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                },
            }
        ]
        # consume the generator (it will yield nothing — empty response body)
        async for _ in self.provider.generate_streaming(
            [LLMMessage(role="user", content="Hi")], tools=tools, tool_choice="auto"
        ):
            pass

        assert "tools" in captured["body"]
        assert captured["body"]["tools"][0]["name"] == "get_weather"
        assert captured["body"]["tool_choice"] == {"type": "auto"}

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_no_tools_not_in_payload(self):
        """When tools=None the key must be absent from the payload."""
        captured = {}

        def capture(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, content=b"")

        respx.post(ANTHROPIC_URL).mock(side_effect=capture)
        async for _ in self.provider.generate_streaming([LLMMessage(role="user", content="Hi")]):
            pass

        assert "tools" not in captured["body"]
        assert "tool_choice" not in captured["body"]

    # ------------------------------------------------------------------
    # Single tool call — arguments spread across multiple deltas
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_single_tool_call_accumulated(self):
        """input_json_delta fragments must be concatenated into complete arguments."""
        body = _sse_lines(
            _tool_start(0, "toolu_001", "get_weather"),
            _tool_delta(0, '{"ci'),
            _tool_delta(0, 'ty": "Rome'),
            _tool_delta(0, '"}'),
        )
        respx.post(ANTHROPIC_URL).mock(return_value=httpx.Response(200, content=body))

        chunks = [
            r
            async for r in self.provider.generate_streaming(
                [LLMMessage(role="user", content="weather?")]
            )
        ]

        tool_chunks = [c for c in chunks if c.tool_calls]
        assert len(tool_chunks) == 1
        tc = tool_chunks[0].tool_calls[0]
        assert tc.id == "toolu_001"
        assert tc.function.name == "get_weather"
        assert tc.function.arguments == '{"city": "Rome"}'

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_tool_call_no_intermediate_text(self):
        """While accumulating a tool call, no text chunk must be yielded."""
        body = _sse_lines(
            _tool_start(0, "toolu_x", "add"),
            _tool_delta(0, '{"a":1,'),
            _tool_delta(0, '"b":2}'),
        )
        respx.post(ANTHROPIC_URL).mock(return_value=httpx.Response(200, content=body))

        chunks = [
            r
            async for r in self.provider.generate_streaming(
                [LLMMessage(role="user", content="add")]
            )
        ]

        assert [c for c in chunks if c.content] == []

    # ------------------------------------------------------------------
    # Multiple tool calls
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_multiple_tool_calls(self):
        """Two tool calls at different indices must be reconstructed independently."""
        body = _sse_lines(
            _tool_start(0, "toolu_A", "tool_a"),
            _tool_start(1, "toolu_B", "tool_b"),
            _tool_delta(0, '{"x":1}'),
            _tool_delta(1, '{"y":2}'),
        )
        respx.post(ANTHROPIC_URL).mock(return_value=httpx.Response(200, content=body))

        chunks = [
            r
            async for r in self.provider.generate_streaming(
                [LLMMessage(role="user", content="both")]
            )
        ]

        tool_chunks = [c for c in chunks if c.tool_calls]
        assert len(tool_chunks) == 1
        by_id = {tc.id: tc for tc in tool_chunks[0].tool_calls}
        assert by_id["toolu_A"].function.name == "tool_a"
        assert by_id["toolu_A"].function.arguments == '{"x":1}'
        assert by_id["toolu_B"].function.name == "tool_b"
        assert by_id["toolu_B"].function.arguments == '{"y":2}'

    # ------------------------------------------------------------------
    # Mixed: text tokens + tool call
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_mixed_text_and_tool_call(self):
        """Text tokens yielded before the tool-call chunk must be preserved."""
        body = _sse_lines(
            _text_delta(0, "Sure, "),
            _text_delta(0, "calling now."),
            _tool_start(1, "toolu_Z", "do_it"),
            _tool_delta(1, "{}"),
        )
        respx.post(ANTHROPIC_URL).mock(return_value=httpx.Response(200, content=body))

        chunks = [
            r
            async for r in self.provider.generate_streaming([LLMMessage(role="user", content="go")])
        ]

        assert [c.content for c in chunks if c.content] == ["Sure, ", "calling now."]
        tool_chunks = [c for c in chunks if c.tool_calls]
        assert len(tool_chunks) == 1
        assert tool_chunks[0].tool_calls[0].id == "toolu_Z"
        assert tool_chunks[0].tool_calls[0].function.name == "do_it"

    # ------------------------------------------------------------------
    # No tool calls → no extra chunk
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_no_tool_calls_no_extra_chunk(self):
        """A plain-text response must not yield any chunk with tool_calls."""
        body = _sse_lines(
            _text_delta(0, "Hello"),
            _text_delta(0, " world"),
        )
        respx.post(ANTHROPIC_URL).mock(return_value=httpx.Response(200, content=body))

        chunks = [
            r
            async for r in self.provider.generate_streaming([LLMMessage(role="user", content="Hi")])
        ]

        assert all(not c.tool_calls for c in chunks)

    # ------------------------------------------------------------------
    # HTTP error
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_http_error_raises(self):
        """A non-2xx response must raise httpx.HTTPStatusError."""
        respx.post(ANTHROPIC_URL).mock(return_value=httpx.Response(500))

        with pytest.raises(httpx.HTTPStatusError):
            async for _ in self.provider.generate_streaming(
                [LLMMessage(role="user", content="Hi")]
            ):
                pass
