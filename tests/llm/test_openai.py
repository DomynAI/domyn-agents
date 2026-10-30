"""Tests for OpenAI LLM Provider."""

import json

import httpx
import pydantic
import pytest
import respx

from domyn.agents.llm import LLMMessage, LLMResponse, LLMToolCall, OpenAIProvider

OPENAI_URL = "https://api.openai.com/v1/responses"


def _sse(*events: dict) -> bytes:
    """Encode a sequence of event dicts as SSE data lines followed by [DONE]."""
    lines = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
    lines += "data: [DONE]\n\n"
    return lines.encode()


def _text_delta(content: str, output_index: int = 0) -> dict:
    return {
        "type": "response.output_text.delta",
        "output_index": output_index,
        "content_index": 0,
        "delta": content,
    }


def _fc_added(output_index: int, item_id: str, call_id: str, name: str) -> dict:
    return {
        "type": "response.output_item.added",
        "output_index": output_index,
        "item": {
            "type": "function_call",
            "id": item_id,
            "call_id": call_id,
            "name": name,
            "arguments": "",
        },
    }


def _fc_args_delta(output_index: int, delta: str) -> dict:
    return {
        "type": "response.function_call_arguments.delta",
        "output_index": output_index,
        "delta": delta,
    }


def _fc_done(output_index: int, item_id: str, call_id: str, name: str, arguments: str) -> dict:
    return {
        "type": "response.output_item.done",
        "output_index": output_index,
        "item": {
            "type": "function_call",
            "id": item_id,
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
        },
    }


class TestOpenAIProvider:
    """Test cases for OpenAIProvider."""

    def setup_method(self):
        """Setup for each test."""
        self.provider = OpenAIProvider(model_name="gpt-4o-mini", api_key="test-api-key")

    def test_init(self):
        """Test provider initialization."""
        assert self.provider.type == "openai"
        assert self.provider.url == "https://api.openai.com/v1/responses"
        assert self.provider.name == "gpt-4o-mini"

    def test_generation_params_merging(self):
        """Test that partial generation_params are merged with defaults."""
        provider = OpenAIProvider(
            model_name="gpt-4o-mini",
            api_key="test-key",
            generation_params={"temperature": 0.5},
        )

        assert provider.generation_params["temperature"] == 0.5
        assert provider.generation_params["max_output_tokens"] == 4000
        assert provider.generation_params["service_tier"] == "auto"
        assert provider.generation_params["reasoning_effort"] == "medium"

    def test_generation_params_empty_dict(self):
        """Test that empty generation_params dict gets defaults."""
        provider = OpenAIProvider(
            model_name="gpt-4o-mini",
            api_key="test-key",
            generation_params={},
        )

        assert provider.generation_params["temperature"] == 1
        assert provider.generation_params["max_output_tokens"] == 4000
        assert provider.generation_params["service_tier"] == "auto"
        assert provider.generation_params["reasoning_effort"] == "medium"

    def test_reasoning_effort_transformed_for_reasoning_models(self):
        """reasoning_effort must be transformed to a reasoning object for o-series/gpt-5 models."""
        provider = OpenAIProvider(model_name="gpt-5-nano", api_key="test-key")
        params = provider.compute_generation_params

        assert "reasoning_effort" not in params
        assert params["reasoning"] == {"effort": "medium"}

    def test_reasoning_effort_dropped_for_non_reasoning_models(self):
        """reasoning_effort must be dropped entirely for models that don't support it."""
        provider = OpenAIProvider(model_name="gpt-4o-mini", api_key="test-key")
        params = provider.compute_generation_params

        assert "reasoning_effort" not in params
        assert "reasoning" not in params

    def test_init_without_api_key(self):
        """Test provider initialization rejects explicit None API key."""
        from unittest.mock import patch

        with patch.dict("os.environ", {}, clear=True), pytest.raises(pydantic.ValidationError):
            OpenAIProvider(model_name="gpt-4o-mini", api_key=None)

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_success(self):
        """Test successful generation."""
        messages = [LLMMessage(role="user", content="Hello")]

        mock_response_data = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Hi there!"}],
                }
            ],
            "usage": {"output_tokens": 10},
        }

        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=mock_response_data))

        response = await self.provider.generate(messages, structured_outputs=None)

        assert isinstance(response, LLMResponse)
        assert response.content == "Hi there!"
        assert response.generated_tokens == 10
        assert response.execution_time > 0

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_with_tool_calls(self):
        """Test generation that returns tool calls."""
        messages = [LLMMessage(role="user", content="What's the weather?")]

        mock_response_data = {
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_001",
                    "call_id": "call_001",
                    "name": "get_weather",
                    "arguments": '{"city": "Rome"}',
                }
            ],
            "usage": {"output_tokens": 5},
        }

        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=mock_response_data))

        response = await self.provider.generate(messages, structured_outputs=None)

        assert response.content == ""
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id == "call_001"
        assert response.tool_calls[0].function.name == "get_weather"
        assert response.tool_calls[0].function.arguments == '{"city": "Rome"}'

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_http_error(self):
        """Test generation with HTTP error."""
        messages = [LLMMessage(role="user", content="Hello")]

        respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(500, json={"error": "Internal Server Error"})
        )

        with pytest.raises(httpx.HTTPStatusError):
            await self.provider.generate(messages, structured_outputs=None)

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_streaming_success(self):
        """Test successful streaming generation."""
        messages = [LLMMessage(role="user", content="Hello")]

        body = _sse(
            _text_delta("Hello"),
            _text_delta("!"),
            _text_delta(" How"),
            _text_delta(" can"),
            _text_delta(" I"),
            _text_delta(" assist"),
            _text_delta(" you"),
            _text_delta(" today"),
            _text_delta("?"),
        )

        respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(
                200, content=body, headers={"Content-Type": "text/event-stream"}
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
    async def test_generate_streaming_success_with_reasoning(self):
        """reasoning_effort must be sent as a reasoning object for reasoning models."""
        messages = [LLMMessage(role="user", content="Hello")]

        provider = OpenAIProvider(model_name="gpt-5-nano", api_key="test-api-key")
        captured: dict = {}

        def capture(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                content=_sse(_text_delta("Hi"), _text_delta(" there"), _text_delta("!")),
                headers={"Content-Type": "text/event-stream"},
            )

        respx.post(OPENAI_URL).mock(side_effect=capture)

        responses = []
        async for response in provider.generate_streaming(messages):
            responses.append(response)

        assert len(responses) == 3
        assert responses[0].content == "Hi"
        assert responses[1].content == " there"
        assert captured["body"]["reasoning"] == {"effort": "medium"}
        assert "reasoning_effort" not in captured["body"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_streaming_with_usage(self):
        """Test streaming generation token counting."""
        messages = [LLMMessage(role="user", content="Hello")]

        body = _sse(_text_delta("Hi"), _text_delta(" there"))

        respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(
                200, content=body, headers={"Content-Type": "text/event-stream"}
            )
        )

        responses = []
        async for response in self.provider.generate_streaming(messages):
            responses.append(response)

        assert len(responses) == 2
        assert responses[0].generated_tokens == 1
        assert responses[1].generated_tokens == 2

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_streaming_malformed_json(self):
        """Test streaming with malformed JSON (should be skipped)."""
        messages = [LLMMessage(role="user", content="Hello")]

        streaming_content = (
            'data: {"type":"response.output_text.delta","delta":"Good"}\r\n'
            "data: {invalid json}\r\n"
            'data: {"type":"response.output_text.delta","delta":" response"}\r\n'
            "data: [DONE]\r\n"
        )

        respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(
                200,
                content=streaming_content.encode(),
                headers={"Content-Type": "text/event-stream"},
            )
        )

        responses = []
        async for response in self.provider.generate_streaming(messages):
            responses.append(response)

        assert len(responses) == 2
        assert responses[0].content == "Good"
        assert responses[1].content == " response"

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_streaming_empty_delta(self):
        """Test streaming with empty delta content."""
        messages = [LLMMessage(role="user", content="Hello")]

        body = _sse(
            {"type": "response.output_text.delta", "delta": ""},  # empty — should not yield
            _text_delta("Hello"),
            {"type": "response.created", "response": {}},  # non-text event — ignored
        )

        respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(
                200, content=body, headers={"Content-Type": "text/event-stream"}
            )
        )

        responses = []
        async for response in self.provider.generate_streaming(messages):
            responses.append(response)

        assert len(responses) == 1
        assert responses[0].content == "Hello"

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_without_api_key(self):
        """Test generation without API key."""
        with pytest.raises(pydantic.ValidationError):
            _ = OpenAIProvider(model_name="gpt-4o-mini", api_key=None)

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_with_wrong_api_key(self):
        """Test generation with wrong API key."""
        provider = OpenAIProvider(model_name="gpt-4o-mini", api_key="wrong-key")
        messages = [LLMMessage(role="user", content="Hello")]

        route = respx.post(OPENAI_URL).mock(return_value=httpx.Response(401, text="Unauthorized"))

        with pytest.raises(httpx.HTTPStatusError):
            await provider.generate(messages, structured_outputs=None)

        assert route.called

    @pytest.mark.asyncio
    @respx.mock
    async def test_serialize_messages_tool_result(self):
        """Tool role messages must be serialized as function_call_output."""
        captured = {}

        def capture(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"output": [], "usage": {"output_tokens": 0}})

        respx.post(OPENAI_URL).mock(side_effect=capture)

        messages = [
            LLMMessage(role="user", content="Hi"),
            LLMMessage(
                role="assistant",
                tool_calls=[
                    LLMToolCall(
                        id="call_001",
                        type="function",
                        function={"name": "get_weather", "arguments": "{}"},
                    )
                ],
            ),
            LLMMessage(role="tool", content="Sunny", tool_call_id="call_001"),
        ]

        await self.provider.generate(messages, structured_outputs=None)

        input_items = captured["body"]["input"]
        assert input_items[0] == {"role": "user", "content": "Hi"}
        assert input_items[1] == {
            "type": "function_call",
            "id": "fc_call_001",
            "call_id": "call_001",
            "name": "get_weather",
            "arguments": "{}",
        }
        assert input_items[2] == {
            "type": "function_call_output",
            "call_id": "call_001",
            "output": "Sunny",
        }

    @pytest.mark.asyncio
    @respx.mock
    async def test_tools_adapted_to_flat_format(self):
        """Tools in nested Chat Completions format must be flattened for the Responses API."""
        captured = {}

        def capture(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"output": [], "usage": {"output_tokens": 0}})

        respx.post(OPENAI_URL).mock(side_effect=capture)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Add two numbers",
                    "parameters": {"type": "object", "properties": {"a": {}, "b": {}}},
                },
            }
        ]

        await self.provider.generate([LLMMessage(role="user", content="Hi")], tools=tools)

        expected = [
            {
                "type": "function",
                "name": "add",
                "description": "Add two numbers",
                "parameters": {"type": "object", "properties": {"a": {}, "b": {}}},
                "strict": False,
            }
        ]
        assert captured["body"]["tools"] == expected
        assert "input" in captured["body"]
        assert "messages" not in captured["body"]

    @pytest.mark.asyncio
    @respx.mock
    @pytest.mark.parametrize(("declared", "sent"), [({}, False), ({"strict": True}, True)])
    async def test_strict_is_always_sent(self, declared, sent):
        """An omitted `strict` means true on the Responses API, so it is always sent."""
        captured = {}

        def capture(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"output": [], "usage": {"output_tokens": 0}})

        respx.post(OPENAI_URL).mock(side_effect=capture)

        tools = [{"type": "function", "function": {"name": "add", "parameters": {}, **declared}}]
        await self.provider.generate([LLMMessage(role="user", content="Hi")], tools=tools)

        assert captured["body"]["tools"][0]["strict"] is sent


class TestOpenAIGenerateStreamingToolCalls:
    """Tests for generate_streaming tool-call accumulation in OpenAIProvider."""

    @pytest.fixture(autouse=True)
    def provider(self):
        self.provider = OpenAIProvider(model_name="gpt-4o-mini", api_key="test-key")

    # ------------------------------------------------------------------
    # Payload forwarding
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_sends_tools_in_payload(self):
        """tools and tool_choice must appear in the request payload (flattened format)."""
        captured = {}

        def capture(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, content=_sse(_text_delta("ok")))

        respx.post(OPENAI_URL).mock(side_effect=capture)

        tools = [
            {"type": "function", "function": {"name": "add", "description": "", "parameters": {}}}
        ]
        await self.provider.generate_streaming(
            [LLMMessage(role="user", content="Hi")],
            tools=tools,
            tool_choice="auto",
        ).__anext__()

        expected_tools = [
            {
                "type": "function",
                "name": "add",
                "description": "",
                "parameters": {},
                "strict": False,
            }
        ]
        assert captured["body"]["tools"] == expected_tools
        assert captured["body"]["tool_choice"] == "auto"
        assert "input" in captured["body"]
        assert "messages" not in captured["body"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_no_tools_not_in_payload(self):
        """When tools=None, the key must not appear in the payload."""
        captured = {}

        def capture(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, content=_sse(_text_delta("ok")))

        respx.post(OPENAI_URL).mock(side_effect=capture)
        await self.provider.generate_streaming([LLMMessage(role="user", content="Hi")]).__anext__()

        assert "tools" not in captured["body"]
        assert "tool_choice" not in captured["body"]
        assert "input" in captured["body"]

    # ------------------------------------------------------------------
    # Single tool call — arguments spread across multiple chunks
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_single_tool_call_accumulated(self):
        """Arguments arriving in 3 separate chunks must be fully reassembled."""
        body = _sse(
            _fc_added(0, "fc_001", "call_001", "get_weather"),
            _fc_args_delta(0, '{"ci'),
            _fc_args_delta(0, 'ty": "Rome'),
            _fc_args_delta(0, '"}'),
            _fc_done(0, "fc_001", "call_001", "get_weather", '{"city": "Rome"}'),
        )
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, content=body))

        chunks = [
            r
            async for r in self.provider.generate_streaming(
                [LLMMessage(role="user", content="weather?")]
            )
        ]

        tool_chunks = [c for c in chunks if c.tool_calls]
        assert len(tool_chunks) == 1
        tc = tool_chunks[0].tool_calls[0]
        assert tc.id == "call_001"
        assert tc.function.name == "get_weather"
        assert tc.function.arguments == '{"city": "Rome"}'

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_tool_call_no_intermediate_text(self):
        """No text chunk must be emitted while accumulating a tool call."""
        body = _sse(
            _fc_added(0, "fc_x", "call_x", "add"),
            _fc_args_delta(0, '{"a":1,'),
            _fc_args_delta(0, '"b":2}'),
            _fc_done(0, "fc_x", "call_x", "add", '{"a":1,"b":2}'),
        )
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, content=body))

        chunks = [
            r
            async for r in self.provider.generate_streaming(
                [LLMMessage(role="user", content="add")]
            )
        ]

        assert [c for c in chunks if c.content] == []

    # ------------------------------------------------------------------
    # Multiple tool calls in the same response
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_multiple_tool_calls(self):
        """Two parallel tool calls with interleaved argument chunks."""
        body = _sse(
            _fc_added(0, "fc_A", "call_A", "tool_a"),
            _fc_added(1, "fc_B", "call_B", "tool_b"),
            _fc_args_delta(0, '{"x":'),
            _fc_args_delta(1, '{"y":'),
            _fc_args_delta(0, "1}"),
            _fc_args_delta(1, "2}"),
            _fc_done(0, "fc_A", "call_A", "tool_a", '{"x":1}'),
            _fc_done(1, "fc_B", "call_B", "tool_b", '{"y":2}'),
        )
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, content=body))

        chunks = [
            r
            async for r in self.provider.generate_streaming(
                [LLMMessage(role="user", content="both")]
            )
        ]

        tool_chunks = [c for c in chunks if c.tool_calls]
        assert len(tool_chunks) == 1
        by_id = {tc.id: tc for tc in tool_chunks[0].tool_calls}
        assert by_id["call_A"].function.arguments == '{"x":1}'
        assert by_id["call_B"].function.arguments == '{"y":2}'

    # ------------------------------------------------------------------
    # output_item.done finalizes accumulated call data
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_output_item_done_finalizes_accumulated_call(self):
        """output_item.done must provide the authoritative final arguments and call_id."""
        body = _sse(
            _fc_added(0, "fc_001", "call_001", "get_weather"),
            _fc_args_delta(0, '{"ci'),
            _fc_args_delta(0, 'ty": "Rome'),
            _fc_done(0, "fc_001", "call_001", "get_weather", '{"city": "Rome"}'),
        )
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, content=body))

        chunks = [
            r
            async for r in self.provider.generate_streaming(
                [LLMMessage(role="user", content="weather?")]
            )
        ]

        tool_chunks = [c for c in chunks if c.tool_calls]
        assert len(tool_chunks) == 1
        tc = tool_chunks[0].tool_calls[0]
        assert tc.id == "call_001"
        assert tc.type == "function"
        assert tc.function.name == "get_weather"
        assert tc.function.arguments == '{"city": "Rome"}'

    # ------------------------------------------------------------------
    # Mixed: text + tool call
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_mixed_text_and_tool_call(self):
        """Text tokens before the tool-call events must be preserved."""
        body = _sse(
            _text_delta("Sure, "),
            _text_delta("calling now."),
            _fc_added(0, "fc_Z", "call_Z", "do_it"),
            _fc_args_delta(0, "{}"),
            _fc_done(0, "fc_Z", "call_Z", "do_it", "{}"),
        )
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, content=body))

        chunks = [
            r
            async for r in self.provider.generate_streaming([LLMMessage(role="user", content="go")])
        ]

        assert [c.content for c in chunks if c.content] == ["Sure, ", "calling now."]
        tool_chunks = [c for c in chunks if c.tool_calls]
        assert len(tool_chunks) == 1
        assert tool_chunks[0].tool_calls[0].id == "call_Z"

    # ------------------------------------------------------------------
    # No tool calls → no extra chunk
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_no_tool_calls_no_extra_chunk(self):
        """A plain-text response must not yield any chunk with tool_calls."""
        body = _sse(_text_delta("Hello"), _text_delta(" world"))
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, content=body))

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
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(500))

        with pytest.raises(httpx.HTTPStatusError):
            async for _ in self.provider.generate_streaming(
                [LLMMessage(role="user", content="Hi")]
            ):
                pass
