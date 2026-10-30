"""Tests for CustomLLMProvider."""

import json

import httpx
import pytest
import respx

from domyn.agents.llm import CustomLLMProvider, LLMMessage, LLMResponse, LLMToolCall

ENDPOINT = "https://custom.example.com/v1/chat/completions"


class TestCustomLLMProviderInit:
    def test_init_minimal(self):
        provider = CustomLLMProvider(
            model_name="my-model",
            url=ENDPOINT,
        )
        assert provider.type == "custom"
        assert provider.headers == {}
        assert provider.body == {}
        assert provider.name == "my-model"

    def test_init_with_headers_and_body(self):
        provider = CustomLLMProvider(
            model_name="my-model",
            url=ENDPOINT,
            headers={"Authorization": "Bearer tok", "X-Custom": "val"},
            body={"temperature": 0.5, "max_tokens": 256},
        )
        assert provider.headers["Authorization"] == "Bearer tok"
        assert provider.headers["X-Custom"] == "val"
        assert provider.body["temperature"] == 0.5


class TestCustomLLMProviderGenerate:
    def setup_method(self):
        self.provider = CustomLLMProvider(
            model_name="my-model",
            url=ENDPOINT,
            headers={"Authorization": "Bearer secret"},
            body={"temperature": 0.7},
        )

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_success(self):
        mock_response = {
            "choices": [{"message": {"content": "Hello there!"}}],
            "usage": {"completion_tokens": 5},
        }
        respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=mock_response))

        result = await self.provider.generate([LLMMessage(role="user", content="Hi")])

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello there!"
        assert result.generated_tokens == 5

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_sends_custom_headers(self):
        captured = {}

        def capture(request):
            captured["headers"] = dict(request.headers)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"completion_tokens": 1},
                },
            )

        respx.post(ENDPOINT).mock(side_effect=capture)

        await self.provider.generate([LLMMessage(role="user", content="Hi")])

        assert captured["headers"].get("authorization") == "Bearer secret"

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_merges_body(self):
        captured = {}

        def capture(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"completion_tokens": 1},
                },
            )

        respx.post(ENDPOINT).mock(side_effect=capture)

        await self.provider.generate([LLMMessage(role="user", content="Hi")])

        assert captured["body"]["temperature"] == 0.7
        assert captured["body"]["model"] == "my-model"

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_http_error(self):
        respx.post(ENDPOINT).mock(return_value=httpx.Response(500, json={"error": "oops"}))

        with pytest.raises(httpx.HTTPStatusError):
            await self.provider.generate([LLMMessage(role="user", content="Hi")])

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_with_tool_calls(self):
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "add", "arguments": '{"a":1,"b":2}'},
                            }
                        ],
                    }
                }
            ],
            "usage": {"completion_tokens": 10},
        }
        respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=mock_response))

        result = await self.provider.generate([LLMMessage(role="user", content="add 1 and 2")])

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_1"
        assert result.tool_calls[0].function.name == "add"


def _sse(*payloads: dict) -> bytes:
    lines = "".join(f"data: {json.dumps(p)}\n\n" for p in payloads)
    lines += "data: [DONE]\n\n"
    return lines.encode()


def _delta(content: str | None = None, tool_calls: list | None = None) -> dict:
    delta: dict = {}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    return {"choices": [{"delta": delta}]}


class TestCustomLLMProviderStreaming:
    def setup_method(self):
        self.provider = CustomLLMProvider(
            model_name="my-model",
            url=ENDPOINT,
            headers={"Authorization": "Bearer secret"},
        )

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_streaming_success(self):
        body = _sse(_delta("Hello"), _delta(" world"))
        respx.post(ENDPOINT).mock(return_value=httpx.Response(200, content=body))

        chunks = [
            r
            async for r in self.provider.generate_streaming([LLMMessage(role="user", content="Hi")])
        ]

        assert len(chunks) == 2
        assert chunks[0].content == "Hello"
        assert chunks[1].content == " world"

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_streaming_sends_stream_true(self):
        captured = {}

        def capture(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, content=_sse(_delta("ok")))

        respx.post(ENDPOINT).mock(side_effect=capture)

        async for _ in self.provider.generate_streaming([LLMMessage(role="user", content="Hi")]):
            break

        assert captured["body"]["stream"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_streaming_sends_custom_headers(self):
        captured = {}

        def capture(request):
            captured["headers"] = dict(request.headers)
            return httpx.Response(200, content=_sse(_delta("ok")))

        respx.post(ENDPOINT).mock(side_effect=capture)

        async for _ in self.provider.generate_streaming([LLMMessage(role="user", content="Hi")]):
            break

        assert captured["headers"].get("authorization") == "Bearer secret"

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_streaming_tool_call_accumulated(self):
        body = _sse(
            _delta(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_x",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":'},
                    }
                ]
            ),
            _delta(tool_calls=[{"index": 0, "function": {"arguments": '"Rome"}'}}]),
        )
        respx.post(ENDPOINT).mock(return_value=httpx.Response(200, content=body))

        chunks = [
            r
            async for r in self.provider.generate_streaming(
                [LLMMessage(role="user", content="weather?")]
            )
        ]

        tool_chunks = [c for c in chunks if c.tool_calls]
        assert len(tool_chunks) == 1
        tc = tool_chunks[0].tool_calls[0]
        assert tc.id == "call_x"
        assert tc.function.name == "get_weather"
        assert tc.function.arguments == '{"city":"Rome"}'

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_streaming_http_error(self):
        respx.post(ENDPOINT).mock(return_value=httpx.Response(401))

        with pytest.raises(httpx.HTTPStatusError):
            async for _ in self.provider.generate_streaming(
                [LLMMessage(role="user", content="Hi")]
            ):
                pass
