import json
from collections.abc import AsyncGenerator
from unittest.mock import patch

import httpx
import pytest
import respx

from domyn.agents.llm import CerebrasProvider, LLMMessage, LLMResponse


class TestCerebrasProvider:
    """Test cases for CerebrasProvider."""

    def setup_method(self):
        """Setup for each test."""
        self.provider = CerebrasProvider(model_name="llama3.1-8b", api_key="test-api-key")

    def test_init_with_defaults(self):
        """Test provider initialization with default parameters."""
        provider = CerebrasProvider(model_name="llama3.1-8b", api_key="test-api-key")

        assert provider.name == "llama3.1-8b"
        assert provider.url == "https://api.cerebras.ai/v1/chat/completions"
        assert provider.generation_params["temperature"] == 0.7
        assert provider.generation_params["max_tokens"] == 8000
        assert provider.append_text_system_prompt is None
        assert provider.append_text_last_message is None
        assert provider.type == "cerebras"

    def test_generation_params_merging(self):
        """Test that partial generation_params are merged with defaults."""
        # Test partial params - only override temperature
        provider = CerebrasProvider(
            model_name="llama3.1-8b",
            api_key="test-key",
            generation_params={"temperature": 0.9},
        )

        # Should have custom temperature
        assert provider.generation_params["temperature"] == 0.9
        # Should preserve default max_tokens
        assert provider.generation_params["max_tokens"] == 8000

    def test_generation_params_empty_dict(self):
        """Test that empty generation_params dict gets defaults."""
        provider = CerebrasProvider(
            model_name="llama3.1-8b",
            api_key="test-key",
            generation_params={},
        )

        # Should have default values
        assert provider.generation_params["temperature"] == 0.7
        assert provider.generation_params["max_tokens"] == 8000

    def test_init_without_api_key(self):
        """Test provider initialization without API key."""
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(
                ValueError,
                match="API key must be provided for CerebrasProvider",
            ),
        ):
            CerebrasProvider(model_name="llama3.1-8b")

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_success(self):
        """Test successful generation."""
        messages = [LLMMessage(role="user", content="Hello")]

        mock_response_data = {
            "choices": [{"message": {"content": "Hi there! How can I help you?"}}],
            "usage": {"completion_tokens": 15},
        }

        respx.post("https://api.cerebras.ai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_response_data)
        )

        response = await self.provider.generate(messages)

        assert isinstance(response, LLMResponse)
        assert response.content == "Hi there! How can I help you?"
        assert response.generated_tokens == 15
        assert response.execution_time > 0

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_with_append_text(self):
        """Test generation with appended text to prompt."""
        messages = [LLMMessage(role="user", content="Hello")]

        mock_response_data = {
            "choices": [{"message": {"content": "Response"}}],
            "usage": {"completion_tokens": 5},
        }

        route = respx.post("https://api.cerebras.ai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_response_data)
        )

        await self.provider.generate(messages)

        # Verify that the append_text was added to the last message
        request_body = json.loads(route.calls.last.request.content)
        assert request_body["messages"][-1]["content"] == "Hello"

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_retry_on_empty_content(self):
        """Test generation retries when content is empty or None."""
        messages = [LLMMessage(role="user", content="Hello")]

        # First response with None content
        empty_response_data = {
            "choices": [{"message": {"content": None}}],
            "usage": {"completion_tokens": 0},
        }

        # Second response with valid content
        valid_response_data = {
            "choices": [{"message": {"content": "Valid response"}}],
            "usage": {"completion_tokens": 10},
        }

        route = respx.post("https://api.cerebras.ai/v1/chat/completions")
        route.side_effect = [
            httpx.Response(200, json=empty_response_data),
            httpx.Response(200, json=valid_response_data),
        ]

        response = await self.provider.generate(messages)

        assert response.content == "Valid response"
        assert response.generated_tokens == 10
        # Verify it was called twice
        assert route.call_count == 2

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_retry_on_soft_error_without_choices(self):
        """Cerebras returns overload/queue soft errors as HTTP 200 with an error
        body that has no `choices`. It must be retried (not KeyError on choices)."""
        messages = [LLMMessage(role="user", content="Hello")]

        soft_error = {
            "message": "We're experiencing high traffic right now! Please try again soon.",
            "type": "too_many_requests_error",
            "param": "queue",
            "code": "queue_exceeded",
        }
        valid_response_data = {
            "choices": [{"message": {"content": "Valid response"}}],
            "usage": {"completion_tokens": 10},
        }

        route = respx.post("https://api.cerebras.ai/v1/chat/completions")
        route.side_effect = [
            httpx.Response(200, json=soft_error),
            httpx.Response(200, json=valid_response_data),
        ]

        response = await self.provider.generate(messages)

        assert response.content == "Valid response"
        assert route.call_count == 2

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_retry_on_empty_data(self):
        """Test generation retries when response data is empty dict."""
        messages = [LLMMessage(role="user", content="Hello")]

        # First response with empty dict
        # Second response with valid content
        valid_response_data = {
            "choices": [{"message": {"content": "Valid response"}}],
            "usage": {"completion_tokens": 10},
        }

        route = respx.post("https://api.cerebras.ai/v1/chat/completions")
        route.side_effect = [
            httpx.Response(200, json={}),
            httpx.Response(200, json=valid_response_data),
        ]

        response = await self.provider.generate(messages)

        assert response.content == "Valid response"
        assert response.generated_tokens == 10
        # Verify it was called twice
        assert route.call_count == 2

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_http_error(self):
        """Test generation with HTTP error."""
        messages = [LLMMessage(role="user", content="Hello")]

        respx.post("https://api.cerebras.ai/v1/chat/completions").mock(
            return_value=httpx.Response(500, json={"error": "Internal Server Error"})
        )

        with pytest.raises(httpx.HTTPStatusError):
            await self.provider.generate(messages)

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_with_custom_generation_params(self):
        """Test generation with custom parameters."""
        messages = [LLMMessage(role="user", content="Hello")]

        mock_response_data = {
            "choices": [{"message": {"content": "Response"}}],
            "usage": {"completion_tokens": 5},
        }

        route = respx.post("https://api.cerebras.ai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_response_data)
        )

        provider = CerebrasProvider(
            model_name="llama3.1-8b",
            api_key="test-key",
            generation_params={"temperature": 0.9, "max_tokens": 1000, "top_p": 0.95},
        )

        await provider.generate(messages)

        request_body = json.loads(route.calls.last.request.content)
        assert request_body["temperature"] == 0.9
        assert request_body["max_tokens"] == 1000
        assert request_body["top_p"] == 0.95

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_streaming_success(self):
        """Test successful streaming generation."""
        messages = [LLMMessage(role="user", content="Hello")]

        async def streaming_content_generator() -> AsyncGenerator[str, None]:
            streaming_data = [
                'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n',
                'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
                'data: {"choices":[{"delta":{"content":"!"}}]}\n\n',
                'data: {"choices":[{"delta":{"content":" How"}}]}\n\n',
                'data: {"choices":[{"delta":{"content":" can"}}]}\n\n',
                'data: {"choices":[{"delta":{"content":" I"}}]}\n\n',
                'data: {"choices":[{"delta":{"content":" help"}}]}\n\n',
                'data: {"choices":[{"delta":{"content":"?"}}]}\n\n',
                "data: [DONE]\n\n",
            ]
            for chunk in streaming_data:
                yield chunk.encode()

        respx.post("https://api.cerebras.ai/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                stream=streaming_content_generator(),
                headers={"Content-Type": "text/event-stream"},
            )
        )

        responses = []
        async for response in self.provider.generate_streaming(messages):
            responses.append(response)

        assert len(responses) == 7  # Empty content not yielded
        assert responses[0].content == "Hello"
        assert responses[1].content == "!"
        assert responses[2].content == " How"
        assert responses[-1].content == "?"

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_streaming_malformed_json(self):
        """Test streaming with malformed JSON (should be skipped)."""
        messages = [LLMMessage(role="user", content="Hello")]

        async def streaming_content():
            yield b'data: {"choices":[{"delta":{"content":"Good"}}]}\n\n'
            yield b"data: {invalid json}\n\n"  # This should be skipped
            yield b'data: {"choices":[{"delta":{"content":" response"}}]}\n\n'
            yield b"data: [DONE]\n\n"

        respx.post("https://api.cerebras.ai/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, stream=streaming_content(), headers={"Content-Type": "text/event-stream"}
            )
        )

        responses = []
        async for response in self.provider.generate_streaming(messages):
            responses.append(response)

        # Should only get 2 responses (malformed JSON skipped)
        assert len(responses) == 2
        assert responses[0].content == "Good"
        assert responses[1].content == " response"

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_streaming_empty_delta(self):
        """Test streaming with empty delta content."""
        messages = [LLMMessage(role="user", content="Hello")]

        streaming_content = (
            'data: {"choices":[{"delta":{}}]}\n\n'  # No content field
            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
            'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'  # No content field
            "data: [DONE]\n\n"
        )

        respx.post("https://api.cerebras.ai/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, content=streaming_content, headers={"Content-Type": "text/event-stream"}
            )
        )

        responses = []
        async for response in self.provider.generate_streaming(messages):
            responses.append(response)

        # Should only get 1 response (only one with content)
        assert len(responses) == 1
        assert responses[0].content == "Hello"

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_streaming_with_custom_kwargs(self):
        """Test streaming generation with custom kwargs."""
        messages = [LLMMessage(role="user", content="Hello")]

        streaming_content = (
            'data: {"choices":[{"delta":{"content":"Response"}}]}\n\ndata: [DONE]\n\n'
        )

        route = respx.post("https://api.cerebras.ai/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, content=streaming_content, headers={"Content-Type": "text/event-stream"}
            )
        )

        responses = []
        async for response in self.provider.generate_streaming(
            messages, custom_param="value", temperature=0.3
        ):
            responses.append(response)

        request_body = json.loads(route.calls.last.request.content)
        assert request_body["custom_param"] == "value"
        assert request_body["temperature"] == 0.3

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_streaming_http_error(self):
        """Test streaming generation with HTTP error."""
        messages = [LLMMessage(role="user", content="Hello")]

        respx.post("https://api.cerebras.ai/v1/chat/completions").mock(
            return_value=httpx.Response(500, json={"error": "Internal Server Error"})
        )

        with pytest.raises(httpx.HTTPStatusError):
            async for _ in self.provider.generate_streaming(messages):
                pass

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_streaming_multiple_chunks_per_message(self):
        """Test streaming with multiple data chunks in a single message."""
        messages = [LLMMessage(role="user", content="Hello")]

        # Multiple data chunks without separators
        streaming_content = b'data: {"choices":[{"delta":{"content":"First"}}]}\ndata: {"choices":[{"delta":{"content":" Second"}}]}\ndata: [DONE]'

        respx.post("https://api.cerebras.ai/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, content=streaming_content, headers={"Content-Type": "text/event-stream"}
            )
        )

        responses = []
        async for response in self.provider.generate_streaming(messages):
            responses.append(response)

        # Should parse both chunks
        assert len(responses) == 2
        assert responses[0].content == "First"
        assert responses[1].content == " Second"

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_with_wrong_api_key(self):
        """Test generation with wrong API key."""
        provider = CerebrasProvider(model_name="llama3.1-8b", api_key="wrong-key")
        messages = [LLMMessage(role="user", content="Hello")]

        route = respx.post("https://api.cerebras.ai/v1/chat/completions").mock(
            return_value=httpx.Response(401, text="Unauthorized")
        )

        with pytest.raises(httpx.HTTPStatusError):
            await provider.generate(messages)

        assert route.called

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_with_authorization_header(self):
        """Test that authorization header is properly set."""
        messages = [LLMMessage(role="user", content="Hello")]

        mock_response_data = {
            "choices": [{"message": {"content": "Response"}}],
            "usage": {"completion_tokens": 5},
        }

        route = respx.post("https://api.cerebras.ai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_response_data)
        )

        await self.provider.generate(messages)

        # Verify Authorization header
        assert route.calls.last.request.headers["Authorization"] == "Bearer test-api-key"
        assert route.calls.last.request.headers["Content-Type"] == "application/json"

    def test_from_config(self):
        """Test provider creation from config."""
        config = {
            "url": "https://api.cerebras.ai/v1/chat/completions",
            "model_name": "llama3.1-70b",
            "api_key": "test-key",
            "name": "custom-cerebras",
            "generation_params": {"temperature": 0.5, "max_tokens": 2000},
        }

        provider = CerebrasProvider.from_config(config)

        assert provider.url == "https://api.cerebras.ai/v1/chat/completions"
        assert provider.model_name == "llama3.1-70b"
        assert provider.api_key == "test-key"
        assert provider.name == "custom-cerebras"
        assert provider.generation_params["temperature"] == 0.5
        assert provider.generation_params["max_tokens"] == 2000

    def test_from_config_with_defaults(self):
        """Test provider creation from config with default values."""
        config = {
            "url": "https://api.cerebras.ai/v1/chat/completions",
            "model_name": "llama3.1-8b",
            "api_key": "test-api-key",
        }

        provider = CerebrasProvider.from_config(config)

        assert provider.url == "https://api.cerebras.ai/v1/chat/completions"
        assert provider.model_name == "llama3.1-8b"
        assert provider.name == "llama3.1-8b"
        assert provider.generation_params["temperature"] == 0.7
        assert provider.generation_params["max_tokens"] == 2000

    def test_from_config_missing_required(self):
        """Test provider creation from config with missing required fields."""
        config = {
            "model_name": "llama3.1-8b"
            # Missing url
        }

        with pytest.raises(ValueError, match="Missing required Cerebras config keys"):
            CerebrasProvider.from_config(config)

    def test_from_config_missing_model_name(self):
        """Test provider creation from config with missing model_name."""
        config = {
            "url": "https://api.cerebras.ai/v1/chat/completions"
            # Missing model_name
        }

        with pytest.raises(ValueError, match="Missing required Cerebras config keys"):
            CerebrasProvider.from_config(config)

    def test_name_defaults_to_model_name(self):
        """Test that name defaults to model_name if not provided."""
        provider = CerebrasProvider(model_name="llama3.1-8b", api_key="test-key")
        assert provider.name == "llama3.1-8b"

    def test_from_config_with_custom_append_text(self):
        """Test provider creation from config with custom append_text."""
        config = {
            "url": "https://api.cerebras.ai/v1/chat/completions",
            "model_name": "llama3.1-8b",
            "api_key": "test-key",
            "append_text_last_message": " /custom_suffix",
        }

        provider = CerebrasProvider.from_config(config)

        assert provider.append_text_last_message == " /custom_suffix"

    def test_from_config_with_default_append_text(self):
        """Test provider creation from config uses default append_text when not specified."""
        config = {
            "url": "https://api.cerebras.ai/v1/chat/completions",
            "model_name": "llama3.1-8b",
            "api_key": "test-key",
        }

        provider = CerebrasProvider.from_config(config)

        # Should use the default append_text value
        assert provider.append_text_last_message is None
        assert provider.append_text_system_prompt is None


# ---------------------------------------------------------------------------
# Tool-calling tests
# ---------------------------------------------------------------------------

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    }
]


class TestCerebrasProviderToolCalling:
    """Tests for tool-calling support in CerebrasProvider."""

    def setup_method(self):
        self.provider = CerebrasProvider(model_name="llama3.1-8b", api_key="test-api-key")

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_sends_tools_in_payload(self):
        """Tools list must be forwarded as-is in the request payload."""
        messages = [LLMMessage(role="user", content="What is the weather in Rome?")]

        captured_request = None

        def capture(request):
            nonlocal captured_request
            captured_request = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "It is sunny.", "tool_calls": None}}],
                    "usage": {"completion_tokens": 5},
                },
            )

        respx.post("https://api.cerebras.ai/v1/chat/completions").mock(side_effect=capture)

        await self.provider.generate(messages, tools=SAMPLE_TOOLS)

        assert captured_request is not None
        assert "tools" in captured_request
        assert captured_request["tools"][0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_sends_tool_choice(self):
        """tool_choice must be forwarded in the payload when provided."""
        messages = [LLMMessage(role="user", content="Get weather")]

        captured_request = None

        def capture(request):
            nonlocal captured_request
            captured_request = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "", "tool_calls": None}}],
                    "usage": {"completion_tokens": 0},
                },
            )

        respx.post("https://api.cerebras.ai/v1/chat/completions").mock(side_effect=capture)

        await self.provider.generate(messages, tools=SAMPLE_TOOLS, tool_choice="auto")

        assert captured_request["tool_choice"] == "auto"

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_no_tools_not_in_payload(self):
        """When tools=None the payload must not contain a 'tools' key."""
        messages = [LLMMessage(role="user", content="Hello")]

        captured_request = None

        def capture(request):
            nonlocal captured_request
            captured_request = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "Hi!", "tool_calls": None}}],
                    "usage": {"completion_tokens": 3},
                },
            )

        respx.post("https://api.cerebras.ai/v1/chat/completions").mock(side_effect=capture)

        await self.provider.generate(messages)

        assert "tools" not in captured_request
        assert "tool_choice" not in captured_request

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_parses_tool_calls_from_response(self):
        """Tool calls in the LLM response must be parsed into LLMToolCall objects."""
        from domyn.agents.llm import LLMToolCall

        messages = [LLMMessage(role="user", content="What is the weather in Rome?")]

        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Rome", "unit": "celsius"}',
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {"completion_tokens": 20},
        }

        respx.post("https://api.cerebras.ai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        response = await self.provider.generate(messages, tools=SAMPLE_TOOLS)

        assert len(response.tool_calls) == 1
        tc = response.tool_calls[0]
        assert isinstance(tc, LLMToolCall)
        assert tc.id == "call_abc123"
        assert tc.type == "function"
        assert tc.function.name == "get_weather"
        assert json.loads(tc.function.arguments) == {"city": "Rome", "unit": "celsius"}

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_parses_multiple_tool_calls(self):
        """Multiple tool calls in one response must all be parsed."""
        messages = [LLMMessage(role="user", content="Compare weather in Rome and Paris")]

        mock_response = {
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
                                    "arguments": '{"city": "Rome"}',
                                },
                            },
                            {
                                "id": "call_002",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Paris"}',
                                },
                            },
                        ],
                    }
                }
            ],
            "usage": {"completion_tokens": 35},
        }

        respx.post("https://api.cerebras.ai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        response = await self.provider.generate(messages, tools=SAMPLE_TOOLS)

        assert len(response.tool_calls) == 2
        assert response.tool_calls[0].id == "call_001"
        assert response.tool_calls[1].id == "call_002"
        assert json.loads(response.tool_calls[0].function.arguments) == {"city": "Rome"}
        assert json.loads(response.tool_calls[1].function.arguments) == {"city": "Paris"}

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_tool_call_response_has_empty_content(self):
        """When model only returns tool calls, content must be empty string (not None)."""
        messages = [LLMMessage(role="user", content="What is the weather?")]

        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_xyz",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Milan"}',
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {"completion_tokens": 10},
        }

        respx.post("https://api.cerebras.ai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        response = await self.provider.generate(messages, tools=SAMPLE_TOOLS)

        assert response.content == ""
        assert len(response.tool_calls) == 1

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_no_tool_calls_returns_empty_list(self):
        """When model returns plain text with no tool calls, tool_calls must be []."""
        messages = [LLMMessage(role="user", content="Hello")]

        mock_response = {
            "choices": [{"message": {"content": "Hello there!", "tool_calls": None}}],
            "usage": {"completion_tokens": 5},
        }

        respx.post("https://api.cerebras.ai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        response = await self.provider.generate(messages)

        assert response.tool_calls == []
        assert response.content == "Hello there!"

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_retries_on_empty_response_then_succeeds(self):
        """If the first response body is empty, the provider must retry."""
        messages = [LLMMessage(role="user", content="Hello")]

        call_count = 0

        def flaky_response(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(200, content=b"{}")
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "Got it on retry!",
                                "tool_calls": None,
                            }
                        }
                    ],
                    "usage": {"completion_tokens": 8},
                },
            )

        respx.post("https://api.cerebras.ai/v1/chat/completions").mock(side_effect=flaky_response)

        response = await self.provider.generate(messages)

        assert call_count == 2
        assert response.content == "Got it on retry!"
