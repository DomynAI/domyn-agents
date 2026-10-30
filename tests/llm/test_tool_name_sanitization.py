"""Tool/agent name sanitization is confined to the provider boundary.

The canonical name (registry, execution events, frontend) stays original;
only the request payload sent to a strict provider is sanitized, and the
original is restored from the response so the runner keeps dispatching on the
canonical name. These tests pin that round-trip for every provider — the four
chat-completions providers (openai, open_router, cerebras, vllm) share the
single ``sanitize_tool_names`` entry point; anthropic sanitizes its own shape.
"""

import json
from unittest.mock import MagicMock

import httpx
import pytest
import respx

from domyn.agents.llm import (
    AnthropicProvider,
    CerebrasProvider,
    CustomLLMProvider,
    LLMMessage,
    LLMToolCall,
    OpenAIProvider,
    OpenRouterProvider,
    ToolCallFunction,
    VLLMProvider,
)
from domyn.agents.tools.base import (
    build_sanitized_name_map,
    sanitize_tool_choice,
    sanitize_tool_name,
    sanitize_tool_names,
)

RAW_NAME = "Weather Agent"
SAN_NAME = "Weather_Agent"


def _tool_def(name: str = RAW_NAME) -> dict:
    """An OpenAI-format tool definition with an invalid (space) name."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "forecast",
            "parameters": {"type": "object", "properties": {}},
        },
    }


class TestSanitizeHelpers:
    def test_replaces_invalid_chars(self):
        assert sanitize_tool_name("Weekly Sales Brief") == "Weekly_Sales_Brief"

    def test_is_idempotent(self):
        once = sanitize_tool_name(RAW_NAME)
        assert sanitize_tool_name(once) == once

    def test_keeps_valid_name_unchanged(self):
        assert sanitize_tool_name("CodeAgent") == "CodeAgent"

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            sanitize_tool_name("")

    def test_rejects_non_str(self):
        with pytest.raises(ValueError):
            sanitize_tool_name(None)  # type: ignore[arg-type]

    def test_name_map_round_trips(self):
        mapping = build_sanitized_name_map([_tool_def()])
        assert mapping == {SAN_NAME: RAW_NAME}

    def test_name_map_warns_on_collision(self):
        logger = MagicMock()
        # "Web agent" and "Web.agent" both sanitize to "Web_agent" ('-' is valid,
        # space and '.' are not).
        build_sanitized_name_map([_tool_def("Web agent"), _tool_def("Web.agent")], logger=logger)
        logger.warning.assert_called_once()

    def test_tool_choice_openai_shape_sanitized(self):
        out = sanitize_tool_choice({"type": "function", "function": {"name": RAW_NAME}})
        assert out == {"type": "function", "function": {"name": SAN_NAME}}

    def test_tool_choice_anthropic_shape_sanitized(self):
        out = sanitize_tool_choice({"type": "tool", "name": RAW_NAME})
        assert out == {"type": "tool", "name": SAN_NAME}

    def test_tool_choice_string_and_nameless_passthrough(self):
        assert sanitize_tool_choice("auto") == "auto"
        assert sanitize_tool_choice({"type": "any"}) == {"type": "any"}

    def test_sanitize_tool_names_combined_entry_point(self):
        # The single boundary call: mutates messages in place, returns sanitized
        # tools + tool_choice and the reverse map. Shared by all CC providers.
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": RAW_NAME, "arguments": "{}"}}],
            }
        ]
        tools, tool_choice, name_map = sanitize_tool_names(
            messages,
            [_tool_def()],
            {"type": "function", "function": {"name": RAW_NAME}},
        )
        assert messages[0]["tool_calls"][0]["function"]["name"] == SAN_NAME  # mutated in place
        assert tools[0]["function"]["name"] == SAN_NAME
        assert tool_choice == {"type": "function", "function": {"name": SAN_NAME}}
        assert name_map == {SAN_NAME: RAW_NAME}

    def test_sanitize_tool_names_handles_none(self):
        # No tools / no tool_choice → passthrough, empty map, no crash.
        tools, tool_choice, name_map = sanitize_tool_names([], None, None)
        assert tools is None
        assert tool_choice is None
        assert name_map == {}


class TestAnthropicBoundary:
    @pytest.mark.asyncio
    @respx.mock
    async def test_tool_def_sanitized_out_and_name_restored_in(self):
        provider = AnthropicProvider(model_name="claude-3-5-sonnet", api_key="k")
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "content": [{"type": "tool_use", "id": "t1", "name": SAN_NAME, "input": {}}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )
        )

        resp = await provider.generate(
            [LLMMessage(role="user", content="weather in Turin")], tools=[_tool_def()]
        )

        sent = json.loads(route.calls.last.request.read())
        assert sent["tools"][0]["name"] == SAN_NAME  # sanitized on the way out
        assert resp.tool_calls[0].function.name == RAW_NAME  # restored on the way in

    @pytest.mark.asyncio
    @respx.mock
    async def test_replayed_tool_use_name_sanitized(self):
        provider = AnthropicProvider(model_name="claude-3-5-sonnet", api_key="k")
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(
                200, json={"content": [], "usage": {"input_tokens": 1, "output_tokens": 1}}
            )
        )

        messages = [
            LLMMessage(role="user", content="weather in Turin"),
            LLMMessage(
                role="assistant",
                content="",
                tool_calls=[
                    LLMToolCall(
                        id="t1",
                        type="function",
                        function=ToolCallFunction(name=RAW_NAME, arguments="{}"),
                    )
                ],
            ),
        ]
        await provider.generate(messages, tools=[_tool_def()])

        sent = json.loads(route.calls.last.request.read())
        tool_use = sent["messages"][1]["content"][0]
        assert tool_use["type"] == "tool_use"
        assert tool_use["name"] == SAN_NAME

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_restores_name(self):
        provider = AnthropicProvider(model_name="claude-3-5-sonnet", api_key="k")
        stream = (
            "event: content_block_start\n"
            'data: {"type":"content_block_start","index":0,'
            '"content_block":{"type":"tool_use","id":"t1","name":"' + SAN_NAME + '"}}\n'
            "\n"
            "event: content_block_delta\n"
            'data: {"type":"content_block_delta","index":0,'
            '"delta":{"type":"input_json_delta","partial_json":"{}"}}\n'
            "\n"
            "event: message_stop\n"
            'data: {"type":"message_stop"}\n'
            "\n"
        )
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, content=stream)
        )

        tool_calls = []
        async for r in provider.generate_streaming(
            [LLMMessage(role="user", content="hi")], tools=[_tool_def()]
        ):
            if r.tool_calls:
                tool_calls = r.tool_calls
        assert tool_calls[0].function.name == RAW_NAME

    @pytest.mark.asyncio
    @respx.mock
    async def test_forced_tool_choice_sanitized(self):
        provider = AnthropicProvider(model_name="claude-3-5-sonnet", api_key="k")
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(
                200, json={"content": [], "usage": {"input_tokens": 1, "output_tokens": 1}}
            )
        )
        await provider.generate(
            [LLMMessage(role="user", content="hi")],
            tools=[_tool_def()],
            tool_choice={"type": "tool", "name": RAW_NAME},
        )
        sent = json.loads(route.calls.last.request.read())
        assert sent["tool_choice"] == {"type": "tool", "name": SAN_NAME}


class TestOpenAIBoundary:
    URL = "https://api.openai.com/v1/responses"

    @pytest.mark.asyncio
    @respx.mock
    async def test_tool_def_sanitized_out_and_name_restored_in(self):
        provider = OpenAIProvider(model_name="gpt-4o", api_key="k")
        route = respx.post(self.URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "output": [
                        {
                            "type": "function_call",
                            "id": "fc_t1",
                            "call_id": "t1",
                            "name": SAN_NAME,
                            "arguments": "{}",
                        }
                    ],
                    "usage": {"output_tokens": 1},
                },
            )
        )

        resp = await provider.generate(
            [LLMMessage(role="user", content="weather in Turin")], tools=[_tool_def()]
        )

        sent = json.loads(route.calls.last.request.read())
        # Tools are flattened to Responses API format; name must still be sanitized
        assert sent["tools"][0]["name"] == SAN_NAME
        assert resp.tool_calls[0].function.name == RAW_NAME

    @pytest.mark.asyncio
    @respx.mock
    async def test_replayed_tool_use_name_sanitized(self):
        provider = OpenAIProvider(model_name="gpt-4o", api_key="k")
        route = respx.post(self.URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "output": [
                        {"type": "message", "content": [{"type": "output_text", "text": "ok"}]}
                    ],
                    "usage": {"output_tokens": 1},
                },
            )
        )

        messages = [
            LLMMessage(role="user", content="weather in Turin"),
            LLMMessage(
                role="assistant",
                content="",
                tool_calls=[
                    LLMToolCall(
                        id="t1",
                        type="function",
                        function=ToolCallFunction(name=RAW_NAME, arguments="{}"),
                    )
                ],
            ),
        ]
        await provider.generate(messages, tools=[_tool_def()])

        sent = json.loads(route.calls.last.request.read())
        # Replayed tool call is a function_call item in input; name must be sanitized
        fc_item = next(i for i in sent["input"] if i.get("type") == "function_call")
        assert fc_item["name"] == SAN_NAME
        assert fc_item["id"].startswith("fc_")
        assert fc_item["call_id"] == "t1"

    @pytest.mark.asyncio
    @respx.mock
    async def test_forced_tool_choice_sanitized(self):
        provider = OpenAIProvider(model_name="gpt-4o", api_key="k")
        route = respx.post(self.URL).mock(
            return_value=httpx.Response(
                200,
                json={"output": [], "usage": {"output_tokens": 1}},
            )
        )
        await provider.generate(
            [LLMMessage(role="user", content="hi")],
            tools=[_tool_def()],
            tool_choice={"type": "function", "function": {"name": RAW_NAME}},
        )
        sent = json.loads(route.calls.last.request.read())
        assert sent["tool_choice"] == {"type": "function", "function": {"name": SAN_NAME}}


class TestOpenRouterBoundary:
    """OpenRouter speaks the OpenAI wire format, so it sanitizes ``function.name``;
    the upstream provider (Anthropic/Bedrock/Vertex) is what enforces the charset
    and rejected the raw name as ``tools.0.custom.name`` before this fix. These
    pin OpenRouter's own request/response wiring (retry loop + SSE accumulator),
    not the shared helpers.
    """

    URL = "https://openrouter.ai/api/v1/chat/completions"

    @pytest.mark.asyncio
    @respx.mock
    async def test_tool_def_and_replay_sanitized_out_name_restored_in(self):
        provider = OpenRouterProvider(model_name="anthropic/claude-opus-4.1", api_key="k")
        provider._verified_structured_output_support = True  # skip /models probe (orthogonal)
        route = respx.post(self.URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "t1",
                                        "type": "function",
                                        "function": {"name": SAN_NAME, "arguments": "{}"},
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"completion_tokens": 1},
                },
            )
        )

        messages = [
            LLMMessage(role="user", content="weather in Turin"),
            LLMMessage(
                role="assistant",
                content="",
                tool_calls=[
                    LLMToolCall(
                        id="t1",
                        type="function",
                        function=ToolCallFunction(name=RAW_NAME, arguments="{}"),
                    )
                ],
            ),
        ]
        resp = await provider.generate(messages, tools=[_tool_def()])

        sent = json.loads(route.calls.last.request.read())
        assert sent["tools"][0]["function"]["name"] == SAN_NAME  # tool def sanitized out
        assert sent["messages"][1]["tool_calls"][0]["function"]["name"] == SAN_NAME  # replay too
        assert resp.tool_calls[0].function.name == RAW_NAME  # restored on the way in

    @pytest.mark.asyncio
    @respx.mock
    async def test_forced_tool_choice_sanitized(self):
        provider = OpenRouterProvider(model_name="anthropic/claude-opus-4.1", api_key="k")
        provider._verified_structured_output_support = True  # skip /models probe (orthogonal)
        route = respx.post(self.URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"completion_tokens": 1},
                },
            )
        )
        await provider.generate(
            [LLMMessage(role="user", content="hi")],
            tools=[_tool_def()],
            tool_choice={"type": "function", "function": {"name": RAW_NAME}},
        )
        sent = json.loads(route.calls.last.request.read())
        assert sent["tool_choice"] == {"type": "function", "function": {"name": SAN_NAME}}

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_restores_name(self):
        provider = OpenRouterProvider(model_name="anthropic/claude-opus-4.1", api_key="k")
        provider._verified_structured_output_support = True  # skip /models probe (orthogonal)
        stream = (
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"t1",'
            '"type":"function","function":{"name":"' + SAN_NAME + '","arguments":"{}"}}]}}]}\n\n'
            "data: [DONE]\n\n"
        )
        respx.post(self.URL).mock(return_value=httpx.Response(200, content=stream))

        tool_calls = []
        async for r in provider.generate_streaming(
            [LLMMessage(role="user", content="hi")], tools=[_tool_def()]
        ):
            if r.tool_calls:
                tool_calls = r.tool_calls
        assert tool_calls[0].function.name == RAW_NAME


class TestCerebrasBoundary:
    """Cerebras is OpenAI wire format too; a space in a tool name breaks its
    constrained-decoding grammar compilation (verified via curl: raw name → 400,
    sanitized → 200). Same boundary fix as openai/open_router; these pin
    Cerebras' own retry loop + SSE accumulator wiring.
    """

    URL = "https://api.cerebras.ai/v1/chat/completions"

    @pytest.mark.asyncio
    @respx.mock
    async def test_tool_def_and_replay_sanitized_out_name_restored_in(self):
        provider = CerebrasProvider(model_name="gpt-oss-120b", api_key="k")
        route = respx.post(self.URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "t1",
                                        "type": "function",
                                        "function": {"name": SAN_NAME, "arguments": "{}"},
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"completion_tokens": 1},
                },
            )
        )

        messages = [
            LLMMessage(role="user", content="weather in Turin"),
            LLMMessage(
                role="assistant",
                content="",
                tool_calls=[
                    LLMToolCall(
                        id="t1",
                        type="function",
                        function=ToolCallFunction(name=RAW_NAME, arguments="{}"),
                    )
                ],
            ),
        ]
        resp = await provider.generate(messages, tools=[_tool_def()])

        sent = json.loads(route.calls.last.request.read())
        assert sent["tools"][0]["function"]["name"] == SAN_NAME  # tool def sanitized out
        assert sent["messages"][1]["tool_calls"][0]["function"]["name"] == SAN_NAME  # replay too
        assert resp.tool_calls[0].function.name == RAW_NAME  # restored on the way in

    @pytest.mark.asyncio
    @respx.mock
    async def test_forced_tool_choice_sanitized(self):
        provider = CerebrasProvider(model_name="gpt-oss-120b", api_key="k")
        route = respx.post(self.URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"completion_tokens": 1},
                },
            )
        )
        await provider.generate(
            [LLMMessage(role="user", content="hi")],
            tools=[_tool_def()],
            tool_choice={"type": "function", "function": {"name": RAW_NAME}},
        )
        sent = json.loads(route.calls.last.request.read())
        assert sent["tool_choice"] == {"type": "function", "function": {"name": SAN_NAME}}

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_restores_name(self):
        provider = CerebrasProvider(model_name="gpt-oss-120b", api_key="k")
        stream = (
            'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"t1",'
            '"type":"function","function":{"name":"' + SAN_NAME + '","arguments":"{}"}}]}}]}\n\n'
            "data: [DONE]\n\n"
        )
        respx.post(self.URL).mock(return_value=httpx.Response(200, content=stream))

        tool_calls = []
        async for r in provider.generate_streaming(
            [LLMMessage(role="user", content="hi")], tools=[_tool_def()]
        ):
            if r.tool_calls:
                tool_calls = r.tool_calls
        assert tool_calls[0].function.name == RAW_NAME


class TestVLLMBoundary:
    """vLLM is the OpenAI wire format too. It does not enforce the charset today,
    but routing it through the shared boundary is idempotent (no cost on valid
    names) and future-proofs it against a backend that does — exactly the
    surprise Cerebras sprung. These pin the round-trip + SSE accumulator.
    """

    URL = "https://vllm.local/v1/chat/completions"

    @pytest.mark.asyncio
    @respx.mock
    async def test_tool_def_and_replay_sanitized_out_name_restored_in(self):
        provider = VLLMProvider(model_name="llama-3.3-70b", url=self.URL)
        route = respx.post(self.URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "t1",
                                        "type": "function",
                                        "function": {"name": SAN_NAME, "arguments": "{}"},
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"completion_tokens": 1},
                },
            )
        )

        messages = [
            LLMMessage(role="user", content="weather in Turin"),
            LLMMessage(
                role="assistant",
                content="",
                tool_calls=[
                    LLMToolCall(
                        id="t1",
                        type="function",
                        function=ToolCallFunction(name=RAW_NAME, arguments="{}"),
                    )
                ],
            ),
        ]
        resp = await provider.generate(messages, tools=[_tool_def()])

        sent = json.loads(route.calls.last.request.read())
        assert sent["tools"][0]["function"]["name"] == SAN_NAME  # tool def sanitized out
        assert sent["messages"][1]["tool_calls"][0]["function"]["name"] == SAN_NAME  # replay too
        assert resp.tool_calls[0].function.name == RAW_NAME  # restored on the way in

    @pytest.mark.asyncio
    @respx.mock
    async def test_forced_tool_choice_sanitized(self):
        provider = VLLMProvider(model_name="llama-3.3-70b", url=self.URL)
        route = respx.post(self.URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"completion_tokens": 1},
                },
            )
        )
        await provider.generate(
            [LLMMessage(role="user", content="hi")],
            tools=[_tool_def()],
            tool_choice={"type": "function", "function": {"name": RAW_NAME}},
        )
        sent = json.loads(route.calls.last.request.read())
        assert sent["tool_choice"] == {"type": "function", "function": {"name": SAN_NAME}}

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_restores_name(self):
        provider = VLLMProvider(model_name="llama-3.3-70b", url=self.URL)
        stream = (
            'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"t1",'
            '"type":"function","function":{"name":"' + SAN_NAME + '","arguments":"{}"}}]}}]}\n\n'
            "data: [DONE]\n\n"
        )
        respx.post(self.URL).mock(return_value=httpx.Response(200, content=stream))

        tool_calls = []
        async for r in provider.generate_streaming(
            [LLMMessage(role="user", content="hi")], tools=[_tool_def()]
        ):
            if r.tool_calls:
                tool_calls = r.tool_calls
        assert tool_calls[0].function.name == RAW_NAME


class TestCustomBoundary:
    """CustomLLMProvider fronts any OpenAI-compatible endpoint — this is how
    cohere (no dedicated provider) is reached when not via OpenRouter, and
    cohere enforces the same tool-name charset. Same shared boundary.
    """

    URL = "https://custom.local/v1/chat/completions"

    @pytest.mark.asyncio
    @respx.mock
    async def test_tool_def_and_replay_sanitized_out_name_restored_in(self):
        provider = CustomLLMProvider(model_name="command-r-plus", url=self.URL)
        route = respx.post(self.URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "t1",
                                        "type": "function",
                                        "function": {"name": SAN_NAME, "arguments": "{}"},
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"completion_tokens": 1},
                },
            )
        )

        messages = [
            LLMMessage(role="user", content="weather in Turin"),
            LLMMessage(
                role="assistant",
                content="",
                tool_calls=[
                    LLMToolCall(
                        id="t1",
                        type="function",
                        function=ToolCallFunction(name=RAW_NAME, arguments="{}"),
                    )
                ],
            ),
        ]
        resp = await provider.generate(messages, tools=[_tool_def()])

        sent = json.loads(route.calls.last.request.read())
        assert sent["tools"][0]["function"]["name"] == SAN_NAME  # tool def sanitized out
        assert sent["messages"][1]["tool_calls"][0]["function"]["name"] == SAN_NAME  # replay too
        assert resp.tool_calls[0].function.name == RAW_NAME  # restored on the way in

    @pytest.mark.asyncio
    @respx.mock
    async def test_forced_tool_choice_sanitized(self):
        provider = CustomLLMProvider(model_name="command-r-plus", url=self.URL)
        route = respx.post(self.URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"completion_tokens": 1},
                },
            )
        )
        await provider.generate(
            [LLMMessage(role="user", content="hi")],
            tools=[_tool_def()],
            tool_choice={"type": "function", "function": {"name": RAW_NAME}},
        )
        sent = json.loads(route.calls.last.request.read())
        assert sent["tool_choice"] == {"type": "function", "function": {"name": SAN_NAME}}

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_restores_name(self):
        provider = CustomLLMProvider(model_name="command-r-plus", url=self.URL)
        stream = (
            'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"t1",'
            '"type":"function","function":{"name":"' + SAN_NAME + '","arguments":"{}"}}]}}]}\n\n'
            "data: [DONE]\n\n"
        )
        respx.post(self.URL).mock(return_value=httpx.Response(200, content=stream))

        tool_calls = []
        async for r in provider.generate_streaming(
            [LLMMessage(role="user", content="hi")], tools=[_tool_def()]
        ):
            if r.tool_calls:
                tool_calls = r.tool_calls
        assert tool_calls[0].function.name == RAW_NAME
