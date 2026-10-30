import json
from collections.abc import AsyncGenerator
from unittest.mock import patch

import httpx
import pytest
import respx

from domyn.agents.llm import LLMMessage, LLMResponse, OpenRouterProvider


class TestOpenRouterProvider:
    """Test cases for OpenRouterProvider."""

    def setup_method(self):
        """Setup for each test."""
        self.provider = OpenRouterProvider(model_name="fake/model", api_key="test-api-key")

    def test_generation_params_merging(self):
        """Test that partial generation_params are merged with defaults."""
        # Test partial params - only override temperature
        provider = OpenRouterProvider(
            model_name="fake/model",
            api_key="test-key",
            generation_params={"temperature": 0.9},
        )

        # Should have custom temperature
        assert provider.generation_params["temperature"] == 0.9
        # Should preserve default max_tokens
        assert provider.generation_params["max_tokens"] == 8000

    def test_generation_params_empty_dict(self):
        """Test that empty generation_params dict gets defaults."""
        provider = OpenRouterProvider(
            model_name="fake/model",
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
            ),
        ):
            OpenRouterProvider(model_name="fake/model")

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_success(self):
        """Test successful generation."""
        messages = [LLMMessage(role="user", content="Hello")]

        mock_response_data = {
            "choices": [{"message": {"content": "Hi there!"}}],
            "usage": {"completion_tokens": 10},
        }

        respx.get(
            url__regex=r"https://openrouter\.ai/api/v1/models\?supported_parameters=\w+"
        ).mock(
            return_value=httpx.Response(
                200,
                content=b'{"data":[]}',
            )
        )

        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_response_data)
        )

        response = await self.provider.generate(messages)

        assert isinstance(response, LLMResponse)
        assert response.content == "Hi there!"
        assert response.generated_tokens == 10
        assert response.execution_time > 0

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_http_error(self):
        """Test generation with HTTP error."""
        messages = [LLMMessage(role="user", content="Hello")]

        respx.get(
            url__regex=r"https://openrouter\.ai/api/v1/models\?supported_parameters=\w+"
        ).mock(
            return_value=httpx.Response(
                200,
                content=b'{"data":[]}',
            )
        )

        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(500, json={"error": "Internal Server Error"})
        )

        with pytest.raises(httpx.HTTPStatusError):
            await self.provider.generate(messages)

    @pytest.mark.asyncio
    @respx.mock
    async def test_reasoning_stripped_when_unsupported(self):
        """`reasoning_supported=False` is authoritative: a `reasoning` key injected
        via generation_params must NOT reach the model — the merge happens after the
        gate, so the key has to be stripped (else a hard 400 on models without
        thinking support)."""
        respx.get(
            url__regex=r"https://openrouter\.ai/api/v1/models\?supported_parameters=\w+"
        ).mock(return_value=httpx.Response(200, content=b'{"data":[]}'))
        post_route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "hi"}}],
                    "usage": {"completion_tokens": 1},
                },
            )
        )
        provider = OpenRouterProvider(
            model_name="some/non-reasoning-model",
            api_key="test-key",
            reasoning_supported=False,
            generation_params={"temperature": 0.7, "reasoning": {"effort": "high"}},
        )

        await provider.generate([LLMMessage(role="user", content="hi")])

        sent = json.loads(post_route.calls.last.request.content)
        assert "reasoning" not in sent

    @pytest.mark.asyncio
    @respx.mock
    async def test_reasoning_sent_when_supported(self):
        """`reasoning_supported=True` → the `reasoning` param is sent (default effort
        medium). Capability comes from config (set upstream by the platform), not a
        runtime probe."""
        respx.get(
            url__regex=r"https://openrouter\.ai/api/v1/models\?supported_parameters=\w+"
        ).mock(return_value=httpx.Response(200, content=b'{"data":[]}'))
        post_route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "hi"}}],
                    "usage": {"completion_tokens": 1},
                },
            )
        )
        provider = OpenRouterProvider(
            model_name="some/reasoning-model",
            api_key="test-key",
            reasoning_supported=True,
        )

        await provider.generate([LLMMessage(role="user", content="hi")])

        sent = json.loads(post_route.calls.last.request.content)
        assert sent.get("reasoning") == {"effort": "medium"}

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_streaming_success(self):
        """Test successful streaming generation."""
        messages = [LLMMessage(role="user", content="Hello")]

        async def streaming_content_generator() -> AsyncGenerator[str, None]:
            for r in json.loads(
                '["data: {\\"id\\":\\"chatcmpl-CVEnFEVohfzgHWGNvKB2mov3mwi9R\\",\\"object\\":\\"chat.completion.chunk\\",\\"created\\":1761561069,\\"model\\":\\"gpt-4o-mini-2024-07-18\\",\\"service_tier\\":\\"default\\",\\"system_fingerprint\\":\\"fp_560af6e559\\",\\"choices\\":[{\\"index\\":0,\\"delta\\":{\\"role\\":\\"assistant\\",\\"content\\":\\"\\",\\"refusal\\":null},\\"logprobs\\":null,\\"finish_reason\\":null}],\\"obfuscation\\":\\"mHBsXH\\"}", "", "data: {\\"id\\":\\"chatcmpl-CVEnFEVohfzgHWGNvKB2mov3mwi9R\\",\\"object\\":\\"chat.completion.chunk\\",\\"created\\":1761561069,\\"model\\":\\"gpt-4o-mini-2024-07-18\\",\\"service_tier\\":\\"default\\",\\"system_fingerprint\\":\\"fp_560af6e559\\",\\"choices\\":[{\\"index\\":0,\\"delta\\":{\\"content\\":\\"Hello\\"},\\"logprobs\\":null,\\"finish_reason\\":null}],\\"obfuscation\\":\\"7Tb\\"}", "", "data: {\\"id\\":\\"chatcmpl-CVEnFEVohfzgHWGNvKB2mov3mwi9R\\",\\"object\\":\\"chat.completion.chunk\\",\\"created\\":1761561069,\\"model\\":\\"gpt-4o-mini-2024-07-18\\",\\"service_tier\\":\\"default\\",\\"system_fingerprint\\":\\"fp_560af6e559\\",\\"choices\\":[{\\"index\\":0,\\"delta\\":{\\"content\\":\\"!\\"},\\"logprobs\\":null,\\"finish_reason\\":null}],\\"obfuscation\\":\\"O0vXHbI\\"}", "", "data: {\\"id\\":\\"chatcmpl-CVEnFEVohfzgHWGNvKB2mov3mwi9R\\",\\"object\\":\\"chat.completion.chunk\\",\\"created\\":1761561069,\\"model\\":\\"gpt-4o-mini-2024-07-18\\",\\"service_tier\\":\\"default\\",\\"system_fingerprint\\":\\"fp_560af6e559\\",\\"choices\\":[{\\"index\\":0,\\"delta\\":{\\"content\\":\\" How\\"},\\"logprobs\\":null,\\"finish_reason\\":null}],\\"obfuscation\\":\\"T81k\\"}", "", "data: {\\"id\\":\\"chatcmpl-CVEnFEVohfzgHWGNvKB2mov3mwi9R\\",\\"object\\":\\"chat.completion.chunk\\",\\"created\\":1761561069,\\"model\\":\\"gpt-4o-mini-2024-07-18\\",\\"service_tier\\":\\"default\\",\\"system_fingerprint\\":\\"fp_560af6e559\\",\\"choices\\":[{\\"index\\":0,\\"delta\\":{\\"content\\":\\" can\\"},\\"logprobs\\":null,\\"finish_reason\\":null}],\\"obfuscation\\":\\"Ym0j\\"}", "", "data: {\\"id\\":\\"chatcmpl-CVEnFEVohfzgHWGNvKB2mov3mwi9R\\",\\"object\\":\\"chat.completion.chunk\\",\\"created\\":1761561069,\\"model\\":\\"gpt-4o-mini-2024-07-18\\",\\"service_tier\\":\\"default\\",\\"system_fingerprint\\":\\"fp_560af6e559\\",\\"choices\\":[{\\"index\\":0,\\"delta\\":{\\"content\\":\\" I\\"},\\"logprobs\\":null,\\"finish_reason\\":null}],\\"obfuscation\\":\\"4pUBE2\\"}", "", "data: {\\"id\\":\\"chatcmpl-CVEnFEVohfzgHWGNvKB2mov3mwi9R\\",\\"object\\":\\"chat.completion.chunk\\",\\"created\\":1761561069,\\"model\\":\\"gpt-4o-mini-2024-07-18\\",\\"service_tier\\":\\"default\\",\\"system_fingerprint\\":\\"fp_560af6e559\\",\\"choices\\":[{\\"index\\":0,\\"delta\\":{\\"content\\":\\" assist\\"},\\"logprobs\\":null,\\"finish_reason\\":null}],\\"obfuscation\\":\\"t\\"}", "", "data: {\\"id\\":\\"chatcmpl-CVEnFEVohfzgHWGNvKB2mov3mwi9R\\",\\"object\\":\\"chat.completion.chunk\\",\\"created\\":1761561069,\\"model\\":\\"gpt-4o-mini-2024-07-18\\",\\"service_tier\\":\\"default\\",\\"system_fingerprint\\":\\"fp_560af6e559\\",\\"choices\\":[{\\"index\\":0,\\"delta\\":{\\"content\\":\\" you\\"},\\"logprobs\\":null,\\"finish_reason\\":null}],\\"obfuscation\\":\\"FSYx\\"}", "", "data: {\\"id\\":\\"chatcmpl-CVEnFEVohfzgHWGNvKB2mov3mwi9R\\",\\"object\\":\\"chat.completion.chunk\\",\\"created\\":1761561069,\\"model\\":\\"gpt-4o-mini-2024-07-18\\",\\"service_tier\\":\\"default\\",\\"system_fingerprint\\":\\"fp_560af6e559\\",\\"choices\\":[{\\"index\\":0,\\"delta\\":{\\"content\\":\\" today\\"},\\"logprobs\\":null,\\"finish_reason\\":null}],\\"obfuscation\\":\\"cy\\"}", "", "data: {\\"id\\":\\"chatcmpl-CVEnFEVohfzgHWGNvKB2mov3mwi9R\\",\\"object\\":\\"chat.completion.chunk\\",\\"created\\":1761561069,\\"model\\":\\"gpt-4o-mini-2024-07-18\\",\\"service_tier\\":\\"default\\",\\"system_fingerprint\\":\\"fp_560af6e559\\",\\"choices\\":[{\\"index\\":0,\\"delta\\":{\\"content\\":\\"?\\"},\\"logprobs\\":null,\\"finish_reason\\":null}],\\"obfuscation\\":\\"vC4i6HS\\"}", "", "data: {\\"id\\":\\"chatcmpl-CVEnFEVohfzgHWGNvKB2mov3mwi9R\\",\\"object\\":\\"chat.completion.chunk\\",\\"created\\":1761561069,\\"model\\":\\"gpt-4o-mini-2024-07-18\\",\\"service_tier\\":\\"default\\",\\"system_fingerprint\\":\\"fp_560af6e559\\",\\"choices\\":[{\\"index\\":0,\\"delta\\":{},\\"logprobs\\":null,\\"finish_reason\\":\\"stop\\"}],\\"obfuscation\\":\\"KU\\"}", "", "data: [DONE]"]'
            ):
                # Emit real SSE framing (terminated by a blank line). The old
                # parser split on the "data:" substring, so it tolerated a stream
                # with no line terminators; aiter_lines requires proper framing,
                # which is what the wire actually sends.
                yield (r + "\n\n").encode() if r else b"\n\n"

        respx.get(
            url__regex=r"https://openrouter\.ai/api/v1/models\?supported_parameters=\w+"
        ).mock(
            return_value=httpx.Response(
                200,
                content=b'{"data":[]}',
            )
        )

        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                stream=streaming_content_generator(),
                headers={"Content-Type": "text/event-stream"},
            )
        )
        responses = []
        async for response in self.provider.generate_streaming(messages):
            responses.append(response)

        assert len(responses) == 9
        assert responses[0].content == "Hello"
        assert responses[1].content == "!"

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_streaming_with_custom_model(self):
        """Test successful streaming generation with a custom model."""
        messages = [LLMMessage(role="user", content="generate 3 words")]

        provider = OpenRouterProvider(model_name="fake/model", api_key="test-api-key")

        content = 'data: {"id":"chatcmpl-CVFRmyERIuYQw9r6Oi3d1FxDzOkLn","object":"chat.completion.chunk","created":1761563582,"model":"anthropic/claude-3-sonnet","service_tier":"default","system_fingerprint":null,"choices":[{"index":0,"delta":{"role":"assistant","content":"","refusal":null},"finish_reason":null}],"obfuscation":"5R"}\n\ndata: {"id":"chatcmpl-CVFRmyERIuYQw9r6Oi3d1FxDzOkLn","object":"chat.completion.chunk","created":1761563582,"model":"anthropic/claude-3-sonnet","service_tier":"default","system_fingerprint":null,"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}],"obfuscation":"6M"}\n\ndata: {"id":"chatcmpl-CVFRmyERIuYQw9r6Oi3d1FxDzOkLn","object":"chat.completion.chunk","created":1761563582,"model":"anthropic/claude-3-sonnet","service_tier":"default","system_fingerprint":null,"choices":[{"index":0,"delta":{"content":" there"},"finish_reason":null}],"obfuscation":"OEMj4VXzW6eg6C"}\n\ndata: [DONE]\n\n'
        respx.get(
            url__regex=r"https://openrouter\.ai/api/v1/models\?supported_parameters=\w+"
        ).mock(
            return_value=httpx.Response(
                200,
                content=b'{"data":[]}',
            )
        )
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                content=content.encode(),
                headers={"Content-Type": "text/event-stream"},
            )
        )

        responses = []
        async for response in provider.generate_streaming(messages):
            responses.append(response)

        assert len(responses) == 2
        assert responses[0].content == "Hi"
        assert responses[1].content == " there"

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_streaming_with_usage(self):
        """Test streaming generation with usage information."""
        messages = [LLMMessage(role="user", content="Hello")]

        streaming_content = (
            'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":" there"}}], "usage":{"completion_tokens":5}}\n\n'
            "data: [DONE]\n\n"
        )

        respx.get(
            url__regex=r"https://openrouter\.ai/api/v1/models\?supported_parameters=\w+"
        ).mock(
            return_value=httpx.Response(
                200,
                content=b'{"data":[]}',
            )
        )

        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, content=streaming_content, headers={"Content-Type": "text/event-stream"}
            )
        )

        responses = []
        async for response in self.provider.generate_streaming(messages):
            responses.append(response)

        assert [r.content for r in responses] == ["Hi", " there", ""]
        assert responses[0].generated_tokens == 1
        assert responses[1].generated_tokens == 2
        assert responses[-1].usage == {"completion_tokens": 5}

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_streaming_malformed_json(self):
        """Test streaming with malformed JSON (should be skipped)."""
        messages = [LLMMessage(role="user", content="Hello")]

        streaming_content = (
            'data: {"choices":[{"delta":{"content":"Good"}}]}\r\n'
            "data: {invalid json}\r\n"
            'data: {"choices":[{"delta":{"content":" response"}}]}\r\n'
            "data: [DONE]\r\n"
        )

        respx.get(
            url__regex=r"https://openrouter\.ai/api/v1/models\?supported_parameters=\w+"
        ).mock(
            return_value=httpx.Response(
                200,
                content=b'{"data":[]}',
            )
        )

        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                content=streaming_content.encode(),
                headers={"Content-Type": "text/event-stream"},
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

        respx.get(
            url__regex=r"https://openrouter\.ai/api/v1/models\?supported_parameters=\w+"
        ).mock(
            return_value=httpx.Response(
                200,
                content=b'{"data":[]}',
            )
        )

        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
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
    async def test_generate_with_wrong_api_key(self):
        """Test generation with wrong API key."""
        provider = OpenRouterProvider(model_name="anthropic/claude-3-haiku", api_key="wrong-key")
        messages = [LLMMessage(role="user", content="Hello")]

        respx.get(
            url__regex=r"https://openrouter\.ai/api/v1/models\?supported_parameters=\w+"
        ).mock(
            return_value=httpx.Response(
                200,
                content=b'{"data":[]}',
            )
        )

        route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(401, text="Unauthorized")
        )

        with pytest.raises(httpx.HTTPStatusError):
            await provider.generate(messages)

        assert route.called

    def test_from_config(self):
        """Test provider creation from config."""
        config = {
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "model_name": "anthropic/claude-3-haiku",
            "api_key": "test-key",
            "name": "custom-claude",
            "generation_params": {"temperature": 0.5, "max_tokens": 2000},
        }

        provider = OpenRouterProvider.from_config(config)

        assert provider.url == "https://openrouter.ai/api/v1/chat/completions"
        assert provider.model_name == "anthropic/claude-3-haiku"
        assert provider.api_key == "test-key"
        assert provider.name == "custom-claude"
        assert provider.generation_params["temperature"] == 0.5
        assert provider.generation_params["max_tokens"] == 2000

    def test_from_config_missing_required(self):
        """Test provider creation from config with missing required fields."""
        config = {
            "model_name": "anthropic/claude-3-haiku"
            # Missing url
        }

        with pytest.raises(ValueError, match="Missing required OpenRouter config keys"):
            OpenRouterProvider.from_config(config)


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

MOCK_MODELS_RESPONSE = b'{"data":[]}'


class TestOpenRouterProviderToolCalling:
    """Tests for tool-calling support in OpenRouterProvider."""

    def setup_method(self):
        self.provider = OpenRouterProvider(model_name="fake/model", api_key="test-api-key")

    def _mock_models(self):
        respx.get(
            url__regex=r"https://openrouter\.ai/api/v1/models\?supported_parameters=\w+"
        ).mock(return_value=httpx.Response(200, content=MOCK_MODELS_RESPONSE))

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_sends_tools_in_payload(self):
        """Tools list must be forwarded as-is in the request payload."""
        self._mock_models()
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

        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(side_effect=capture)

        await self.provider.generate(messages, tools=SAMPLE_TOOLS)

        assert captured_request is not None
        assert "tools" in captured_request
        assert captured_request["tools"][0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_sends_tool_choice(self):
        """tool_choice must be forwarded in the payload when provided."""
        self._mock_models()
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

        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(side_effect=capture)

        await self.provider.generate(messages, tools=SAMPLE_TOOLS, tool_choice="auto")

        assert captured_request["tool_choice"] == "auto"

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_no_tools_not_in_payload(self):
        """When tools=None the payload must not contain a 'tools' key."""
        self._mock_models()
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

        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(side_effect=capture)

        await self.provider.generate(messages)

        assert "tools" not in captured_request
        assert "tool_choice" not in captured_request

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_parses_tool_calls_from_response(self):
        """Tool calls in the LLM response must be parsed into LLMToolCall objects."""
        from domyn.agents.llm import LLMToolCall

        self._mock_models()
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

        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
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
        self._mock_models()
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

        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
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
        self._mock_models()
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

        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        response = await self.provider.generate(messages, tools=SAMPLE_TOOLS)

        assert response.content == ""
        assert len(response.tool_calls) == 1

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_no_tool_calls_returns_empty_list(self):
        """When model returns plain text with no tool calls, tool_calls must be []."""
        self._mock_models()
        messages = [LLMMessage(role="user", content="Hello")]

        mock_response = {
            "choices": [{"message": {"content": "Hello there!", "tool_calls": None}}],
            "usage": {"completion_tokens": 5},
        }

        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        response = await self.provider.generate(messages)

        assert response.tool_calls == []
        assert response.content == "Hello there!"

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_retries_on_empty_response_then_succeeds(self):
        """If the first response is empty, the provider must retry and return the second."""
        self._mock_models()
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

        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(side_effect=flaky_response)

        response = await self.provider.generate(messages)

        assert call_count == 2
        assert response.content == "Got it on retry!"
