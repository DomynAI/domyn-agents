"""Tests for Anthropic provider tool-calling support."""

import json

import httpx
import pytest
import respx

from domyn.agents.llm import AnthropicProvider, LLMMessage, LLMToolCall, ToolCallFunction


class TestAnthropicToolConversion:
    def setup_method(self):
        self.provider = AnthropicProvider(
            model_name="claude-3-5-sonnet-20241022", api_key="test-key"
        )

    def test_convert_tools(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]
        result = self.provider._convert_tools(tools)
        assert len(result) == 1
        assert result[0]["name"] == "search"
        assert result[0]["description"] == "Search the web"
        assert result[0]["input_schema"]["properties"]["query"]["type"] == "string"

    def test_convert_messages_tool_call(self):
        """Assistant message with tool_calls → Anthropic tool_use content block."""
        messages = [
            LLMMessage(
                role="assistant",
                content="Let me search.",
                tool_calls=[
                    LLMToolCall(
                        id="tc_1",
                        function=ToolCallFunction(name="search", arguments='{"query": "AI"}'),
                    )
                ],
            )
        ]
        _, converted = self.provider._convert_messages(messages)

        assert len(converted) == 1
        msg = converted[0]
        assert msg["role"] == "assistant"
        assert isinstance(msg["content"], list)
        text_blocks = [b for b in msg["content"] if b["type"] == "text"]
        tool_blocks = [b for b in msg["content"] if b["type"] == "tool_use"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "Let me search."
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["id"] == "tc_1"
        assert tool_blocks[0]["name"] == "search"
        assert tool_blocks[0]["input"] == {"query": "AI"}

    def test_convert_messages_tool_result(self):
        """Tool result message → Anthropic user message with tool_result block."""
        messages = [
            LLMMessage(
                role="tool",
                tool_call_id="tc_1",
                name="search",
                content="Search results here",
            )
        ]
        _, converted = self.provider._convert_messages(messages)

        assert len(converted) == 1
        msg = converted[0]
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        assert msg["content"][0]["type"] == "tool_result"
        assert msg["content"][0]["tool_use_id"] == "tc_1"
        assert msg["content"][0]["content"] == "Search results here"

    def test_convert_messages_merges_consecutive_tool_results(self):
        """Multiple consecutive tool results should be merged into one user message."""
        messages = [
            LLMMessage(role="tool", tool_call_id="tc_1", name="a", content="result_a"),
            LLMMessage(role="tool", tool_call_id="tc_2", name="b", content="result_b"),
        ]
        _, converted = self.provider._convert_messages(messages)

        assert len(converted) == 1
        assert len(converted[0]["content"]) == 2
        assert converted[0]["content"][0]["tool_use_id"] == "tc_1"
        assert converted[0]["content"][1]["tool_use_id"] == "tc_2"

    def test_convert_messages_full_tool_use_roundtrip(self):
        """Full conversation: user → assistant(tool) → tool result → assistant."""
        messages = [
            LLMMessage(role="user", content="What is AI?"),
            LLMMessage(
                role="assistant",
                content="Let me search.",
                tool_calls=[
                    LLMToolCall(
                        id="tc_1",
                        function=ToolCallFunction(name="search", arguments='{"query": "AI"}'),
                    )
                ],
            ),
            LLMMessage(role="tool", tool_call_id="tc_1", name="search", content="AI is..."),
            LLMMessage(role="assistant", content="Based on search: AI is amazing."),
        ]
        _, converted = self.provider._convert_messages(messages)

        assert len(converted) == 4
        assert converted[0] == {"role": "user", "content": "What is AI?"}
        assert converted[1]["role"] == "assistant"
        assert any(b["type"] == "tool_use" for b in converted[1]["content"])
        assert converted[2]["role"] == "user"
        assert converted[2]["content"][0]["type"] == "tool_result"
        assert converted[3] == {"role": "assistant", "content": "Based on search: AI is amazing."}


class TestAnthropicGenerateWithTools:
    def setup_method(self):
        self.provider = AnthropicProvider(
            model_name="claude-3-5-sonnet-20241022", api_key="test-key"
        )

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_with_tools_sends_tool_definitions(self):
        """Tools should be converted and sent in the request."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ]

        mock_data = {
            "content": [{"type": "text", "text": "I'll search for you."}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=mock_data)
        )

        messages = [LLMMessage(role="user", content="Search for me")]
        await self.provider.generate(messages, tools=tools)

        payload = json.loads(route.calls.last.request.content)
        assert "tools" in payload
        assert payload["tools"][0]["name"] == "search"
        assert "input_schema" in payload["tools"][0]

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_parses_tool_use_blocks(self):
        """Response with tool_use block → tool_calls on LLMResponse."""
        mock_data = {
            "content": [
                {"type": "text", "text": "Let me search."},
                {
                    "type": "tool_use",
                    "id": "tc_42",
                    "name": "search",
                    "input": {"query": "AI trends"},
                },
            ],
            "usage": {"input_tokens": 10, "output_tokens": 15},
        }
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=mock_data)
        )

        messages = [LLMMessage(role="user", content="Search for AI trends")]
        response = await self.provider.generate(messages)

        assert response.content == "Let me search."
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id == "tc_42"
        assert response.tool_calls[0].function.name == "search"
        assert json.loads(response.tool_calls[0].function.arguments) == {"query": "AI trends"}

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_parses_thinking_blocks(self):
        """Response with thinking block → thinking_blocks on LLMResponse."""
        mock_data = {
            "content": [
                {"type": "thinking", "thinking": "Let me reason about this..."},
                {"type": "text", "text": "The answer is 42."},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=mock_data)
        )

        messages = [LLMMessage(role="user", content="What is the answer?")]
        response = await self.provider.generate(messages)

        assert response.content == "The answer is 42."
        assert response.thinking_blocks is not None
        assert len(response.thinking_blocks) == 1
        assert response.thinking_blocks[0]["thinking"] == "Let me reason about this..."

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_no_tools_no_tool_calls(self):
        """Normal text response → empty tool_calls."""
        mock_data = {
            "content": [{"type": "text", "text": "Hello!"}],
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=mock_data)
        )

        messages = [LLMMessage(role="user", content="Hi")]
        response = await self.provider.generate(messages)

        assert response.content == "Hello!"
        assert response.tool_calls == []
        assert response.thinking_blocks is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_with_tool_choice(self):
        """tool_choice string should be sent as {type: ...} to Anthropic."""
        mock_data = {
            "content": [{"type": "text", "text": "OK"}],
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=mock_data)
        )

        messages = [LLMMessage(role="user", content="Hi")]
        tools = [
            {
                "type": "function",
                "function": {"name": "test", "description": "test", "parameters": {}},
            }
        ]
        await self.provider.generate(messages, tools=tools, tool_choice="auto")

        payload = json.loads(route.calls.last.request.content)
        assert payload["tool_choice"] == {"type": "auto"}
