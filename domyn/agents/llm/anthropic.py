"""Anthropic LLM Provider implementation using httpx."""

import json
import os
import time
from collections.abc import AsyncGenerator
from typing import Any, ClassVar

import httpx
from pydantic import Field

from domyn.agents.llm.base import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
    LLMToolCall,
    StructuredOutput,
    StructuredOutputType,
    ToolCallFunction,
    log_inference_request,
    log_inference_response,
)
from domyn.agents.llm.context_window import get_json, positive_int
from domyn.agents.logger import get_logger
from domyn.agents.tools.base import (
    build_sanitized_name_map,
    sanitize_tool_choice,
    sanitize_tool_name,
)

_CACHE_USAGE_KEYS = ("cache_read_input_tokens", "cache_creation_input_tokens")
"""Counted apart from `input_tokens`: see `reported_prompt_tokens` in `llm/base.py`."""


class AnthropicProvider(BaseLLMProvider):
    """Provider for Anthropic Claude models"""

    type: str = "anthropic"
    url: str = Field(
        default="https://api.anthropic.com/v1/messages",
        description="API endpoint URL",
    )
    api_key: str | None = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"),
        description="API key for Anthropic authentication",
    )

    default_generation_params: ClassVar[dict[str, Any]] = {
        "max_tokens": 4096,
    }

    generation_params: dict[str, Any] = Field(
        default_factory=lambda: {
            "max_tokens": 4096,
        },
        description="Generation parameters for the model",
    )

    supported_structured_outputs: list[StructuredOutputType] | None = Field(
        default=None, description="Supported structured output types"
    )

    context_window_catalog: ClassVar[str | None] = "anthropic"

    async def fetch_context_window(self) -> int | None:
        """Anthropic answers per model: ``GET /v1/models/{id}`` → ``max_input_tokens``."""
        if not self.url.endswith("/messages"):
            return None
        model = await get_json(
            f"{self.url.removesuffix('/messages')}/models/{self.model_name}",
            {"x-api-key": self.api_key or "", "anthropic-version": "2023-06-01"},
        )
        return positive_int(model.get("max_input_tokens")) if isinstance(model, dict) else None

    def model_post_init(self, context: Any, /) -> None:
        """Initialize the provider after model creation."""
        if not self.name:
            self.name = self.model_name
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable or api_key parameter is required"
            )

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-format tool definitions to Anthropic format.

        Names are sanitized here (provider boundary) to satisfy Anthropic's
        ``^[a-zA-Z0-9_-]{1,128}$`` contract; the original is restored from the
        response via ``build_sanitized_name_map`` so the runner never sees the
        sanitized form.
        """
        anthropic_tools = []
        for tool in tools:
            func = tool.get("function", {})
            raw_name = func.get("name", "")
            anthropic_tools.append(
                {
                    "name": sanitize_tool_name(raw_name) if raw_name else raw_name,
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                }
            )
        return anthropic_tools

    def _convert_messages(
        self, messages: list[LLMMessage]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """
        Convert LLMMessage format to Anthropic format.

        Anthropic requires system messages to be separate from the messages list.
        Handles tool-use messages: assistant messages with tool_calls become content
        blocks with type "tool_use", and tool result messages become user messages
        with type "tool_result" content blocks.

        Returns (system_prompt, messages_list)
        """
        system_prompt = None
        anthropic_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                content_str = (
                    msg.content if isinstance(msg.content, str) else str(msg.content or "")
                )
                if system_prompt is None:
                    system_prompt = content_str
                else:
                    system_prompt += "\n\n" + content_str

            elif msg.role == "assistant" and msg.tool_calls:
                # Assistant message with tool calls → Anthropic content blocks
                content_blocks: list[dict[str, Any]] = []
                if msg.content:
                    text = msg.content if isinstance(msg.content, str) else str(msg.content)
                    if text.strip():
                        content_blocks.append({"type": "text", "text": text})
                for tc in msg.tool_calls:
                    args = tc.function.arguments
                    # Replayed tool_use names must match the (sanitized) tool
                    # definitions sent in the request. ``sanitize_tool_name`` is
                    # idempotent, so already-valid names pass through unchanged.
                    tc_name = (
                        sanitize_tool_name(tc.function.name)
                        if tc.function.name
                        else tc.function.name
                    )
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc_name,
                            "input": json.loads(args) if isinstance(args, str) else args,
                        }
                    )
                anthropic_messages.append({"role": "assistant", "content": content_blocks})

            elif msg.role == "tool":
                # Tool result → Anthropic user message with tool_result content block
                content_str = (
                    msg.content if isinstance(msg.content, str) else str(msg.content or "")
                )
                tool_result_block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": content_str,
                }
                # Merge consecutive tool results into a single user message
                if (
                    anthropic_messages
                    and anthropic_messages[-1]["role"] == "user"
                    and isinstance(anthropic_messages[-1]["content"], list)
                    and anthropic_messages[-1]["content"]
                    and anthropic_messages[-1]["content"][0].get("type") == "tool_result"
                ):
                    anthropic_messages[-1]["content"].append(tool_result_block)
                else:
                    anthropic_messages.append(
                        {
                            "role": "user",
                            "content": [tool_result_block],
                        }
                    )

            else:
                # Regular user/assistant message
                if isinstance(msg.content, list):
                    anthropic_messages.append({"role": msg.role, "content": msg.content})
                else:
                    content_str = (msg.content or "").strip()
                    if content_str:
                        anthropic_messages.append({"role": msg.role, "content": content_str})

        return system_prompt, anthropic_messages

    async def generate(
        self,
        messages: list[LLMMessage],
        structured_outputs: StructuredOutput | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate content using Anthropic API with httpx"""
        logger = get_logger()
        start = time.time()

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        system_prompt, anthropic_messages = self._convert_messages(messages)

        # Map sanitized → original tool names to restore them in the response.
        name_map = build_sanitized_name_map(tools, logger=logger)

        params = {**self.generation_params}
        params = {k: v for k, v in params.items() if v is not None}

        # Build request parameters
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": anthropic_messages,
            **params,
        }

        # Add system prompt if present
        if system_prompt:
            payload["system"] = system_prompt

        # Add tools if present
        if tools:
            payload["tools"] = self._convert_tools(tools)
        if tool_choice is not None:
            if isinstance(tool_choice, str):
                payload["tool_choice"] = {"type": tool_choice}
            else:
                payload["tool_choice"] = sanitize_tool_choice(tool_choice)

        log_inference_request(logger, self.url, payload)

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(self.url, headers=headers, json=payload)
            if response.is_error:
                body_bytes = await response.aread()
                logger.error(f"Anthropic API {response.status_code} on {self.url}")
                logger.debug(
                    "Anthropic API error body",
                    payload={"body": body_bytes.decode("utf-8", "ignore")[:2000]},
                )
            response.raise_for_status()
            data = response.json()

            # Extract content, tool calls, and thinking blocks from response
            content = ""
            tool_calls: list[LLMToolCall] = []
            thinking_blocks: list[dict[str, Any]] = []

            for block in data.get("content", []):
                block_type = block.get("type")
                if block_type == "text":
                    content += block.get("text", "")
                elif block_type == "tool_use":
                    tool_calls.append(
                        LLMToolCall(
                            id=block["id"],
                            type="function",
                            function=ToolCallFunction(
                                name=name_map.get(block["name"], block["name"]),
                                arguments=json.dumps(block.get("input", {})),
                            ),
                        )
                    )
                elif block_type == "thinking":
                    thinking_blocks.append(
                        {
                            "type": "thinking",
                            "thinking": block.get("thinking", ""),
                        }
                    )

            # Extract usage information. The cache counts are kept: with a cache in
            # play `input_tokens` is only the uncached tail of the prompt, and the
            # prompt's real size is the sum of the three.
            usage_data = data.get("usage", {})
            usage = {
                "input_tokens": usage_data.get("input_tokens", 0),
                "output_tokens": usage_data.get("output_tokens", 0),
                **{key: usage_data[key] for key in _CACHE_USAGE_KEYS if key in usage_data},
            }

            log_inference_response(
                logger,
                self.url,
                {
                    "response": content,
                    "generated_tokens": usage_data.get("output_tokens", 0),
                    "execution_time": time.time() - start,
                    "tool_calls": [tc.model_dump() for tc in tool_calls],
                    "usage": usage,
                },
            )

            return LLMResponse(
                content=content,
                usage=usage,
                generated_tokens=usage_data.get("output_tokens", 0),
                execution_time=time.time() - start,
                tool_calls=tool_calls,
                thinking_blocks=thinking_blocks if thinking_blocks else None,
            )

    async def generate_streaming(
        self,
        messages: list[LLMMessage],
        structured_outputs: StructuredOutput | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[LLMResponse, None]:
        """Generate content with streaming using Anthropic API with httpx"""
        logger = get_logger()
        start = time.time()

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        system_prompt, anthropic_messages = self._convert_messages(messages)

        # Map sanitized → original tool names to restore them in the response.
        name_map = build_sanitized_name_map(tools, logger=logger)

        # TODO: structured_outputs is ignored here and in generate(); support JSON_SCHEMA.
        params = {**self.generation_params}
        params = {k: v for k, v in params.items() if v is not None}

        # Build request parameters
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": anthropic_messages,
            "stream": True,
            **params,
        }

        # Add system prompt if present
        if system_prompt:
            payload["system"] = system_prompt

        # Add tools if present (converting from OpenAI format)
        if tools:
            payload["tools"] = self._convert_tools(tools)
        if tool_choice is not None:
            if isinstance(tool_choice, str):
                payload["tool_choice"] = {"type": tool_choice}
            else:
                payload["tool_choice"] = sanitize_tool_choice(tool_choice)

        log_inference_request(logger, self.url, payload, stream=True)

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(self.url, headers=headers, json=payload)
            if response.is_error:
                body_bytes = await response.aread()
                logger.error(f"Anthropic API {response.status_code} on {self.url}")
                logger.debug(
                    "Anthropic API error body",
                    payload={"body": body_bytes.decode("utf-8", "ignore")[:2000]},
                )
            response.raise_for_status()

            token_count = 0
            usage_info: dict[str, Any] = {}
            event_type = None
            content_started = False

            # Accumulator for streaming tool-call deltas, keyed by content-block index.
            # Anthropic sends: content_block_start (id+name), then content_block_delta
            # with input_json_delta (partial arguments JSON string).
            # Each entry: {"id": str, "name": str, "arguments": str}
            tool_call_acc: dict[int, dict] = {}

            async for line in response.aiter_lines():
                line = line.strip()

                # Parse SSE format
                if line.startswith("event: "):
                    event_type = line[7:]
                elif line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])

                        # content_block_start → may open a tool_use block
                        if event_type == "content_block_start":
                            block = data.get("content_block", {})
                            idx = data.get("index", 0)
                            if block.get("type") == "tool_use":
                                block_name = block.get("name", "")
                                tool_call_acc[idx] = {
                                    "id": block.get("id", ""),
                                    "name": name_map.get(block_name, block_name),
                                    "arguments": "",
                                }

                        # content_block_delta → text token or arguments fragment
                        elif event_type == "content_block_delta":
                            delta = data.get("delta", {})
                            idx = data.get("index", 0)

                            if delta.get("type") == "text_delta":
                                text = delta.get("text", "").replace("\x00", "")
                                if not content_started:
                                    text = text.lstrip()
                                if text:
                                    content_started = True
                                    token_count += 1
                                    yield LLMResponse(
                                        content=text,
                                        usage=usage_info,
                                        generated_tokens=token_count,
                                        execution_time=time.time() - start,
                                    )

                            elif delta.get("type") == "input_json_delta":
                                # Accumulate partial JSON for the tool call at this index
                                if idx in tool_call_acc:
                                    tool_call_acc[idx]["arguments"] += delta.get("partial_json", "")

                        # message_start → the prompt's size, message_delta → the reply's
                        elif event_type == "message_start":
                            usage = (data.get("message") or {}).get("usage") or {}
                            for key in ("input_tokens", *_CACHE_USAGE_KEYS):
                                if key in usage:
                                    usage_info[key] = usage[key]
                        elif event_type == "message_delta":
                            usage = data.get("usage", {})
                            if usage:
                                usage_info["output_tokens"] = usage.get("output_tokens", 0)

                    except json.JSONDecodeError:
                        # Skip malformed JSON
                        pass

        # Build final LLMToolCall objects from the accumulated deltas
        tool_calls: list[LLMToolCall] = [
            LLMToolCall(
                id=acc["id"],
                type="function",
                function=ToolCallFunction(
                    name=acc["name"],
                    arguments=acc["arguments"],
                ),
            )
            for acc in tool_call_acc.values()
        ]

        # If there were tool calls, yield one final response carrying them all —
        # with the usage, which a consumer reads off the last chunk. Without tool
        # calls the usage gets a closing chunk of its own.
        if tool_calls:
            yield LLMResponse(
                content="",
                usage=usage_info,
                generated_tokens=token_count,
                execution_time=time.time() - start,
                tool_calls=tool_calls,
            )
        elif usage_info:
            yield LLMResponse(
                content="",
                usage=usage_info,
                generated_tokens=token_count,
                execution_time=time.time() - start,
            )

        log_inference_response(
            logger,
            self.url,
            {
                "generated_tokens": token_count,
                "execution_time": time.time() - start,
                "tool_calls": [tc.model_dump() for tc in tool_calls],
                "usage": usage_info,
            },
        )
