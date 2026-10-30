import json
import os
import time
from collections.abc import AsyncGenerator
from typing import Any, ClassVar

import httpx
from pydantic import Field, computed_field

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
from domyn.agents.logger import get_logger
from domyn.agents.tools.base import sanitize_tool_name, sanitize_tool_names

MODELS_SUPPORT_REASONING_EFFORT = ["o1", "o1-mini", "o3-mini", "gpt-5", "gpt-5-mini", "gpt-5-nano"]
MODELS_SUPPORT_TEMPERATURE = ["gpt-5-mini", "gpt-5-nano", "gpt-5", "gpt-4o", "gpt-4o-mini"]


class OpenAIProvider(BaseLLMProvider):
    """Provider for OpenAI models"""

    type: str = "openai"
    url: str = Field(default="https://api.openai.com/v1/responses", description="API endpoint URL")

    # TODO: raise in model_post_init when api_key is empty, as AnthropicProvider does.
    api_key: str = Field(
        default=os.getenv("OPENAI_API_KEY"), description="API key for authentication"
    )
    supported_structured_outputs: list[StructuredOutputType] = Field(
        default_factory=lambda: [StructuredOutputType.JSON_SCHEMA],
        description="List of supported structured output types",
    )
    # OpenAI's /v1/models lists no window: the catalogue is the only source.
    context_window_catalog: ClassVar[str | None] = "openai"

    default_generation_params: ClassVar[dict[str, Any]] = {
        "reasoning_effort": "medium",
        "service_tier": "auto",
        "store": False,
        "temperature": 1,
        "max_output_tokens": 4000,
    }

    generation_params: dict[str, Any] = Field(
        default_factory=lambda: {
            "reasoning_effort": "medium",
            "service_tier": "auto",
            "store": False,
            "temperature": 1,
            "max_output_tokens": 4000,
        },
        description="Generation parameters for the model",
    )

    @computed_field
    @property
    def compute_generation_params(self) -> dict[str, Any]:
        gen_params = self.generation_params.copy()
        if self.model_name not in MODELS_SUPPORT_TEMPERATURE:
            gen_params.pop("temperature", None)
        # Transform reasoning_effort → reasoning object required by the Responses API.
        # Drop it entirely for models that don't support reasoning.
        reasoning_effort = gen_params.pop("reasoning_effort", None)
        if reasoning_effort and self.model_name in MODELS_SUPPORT_REASONING_EFFORT:
            gen_params["reasoning"] = {"effort": reasoning_effort}
        return gen_params

    def model_post_init(self, context: Any, /) -> None:
        if not self.name:
            self.name = self.model_name

    @staticmethod
    def _serialize_messages(messages: list[LLMMessage]) -> list[dict[str, Any]]:
        """Serialize LLMMessages to Responses API input format.

        - tool role → function_call_output item
        - assistant with tool_calls → one function_call item per call
        - all others → role/content dicts
        """
        result: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "tool":
                result.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.tool_call_id,
                        "output": msg.content or "",
                    }
                )
            elif msg.tool_calls:
                for tc in msg.tool_calls:
                    result.append(
                        {
                            "type": "function_call",
                            # Responses API requires id to start with "fc"; call_id is the
                            # matching key stored in LLMToolCall.id (starts with "call_").
                            "id": f"fc_{tc.id}",
                            "call_id": tc.id,
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    )
            else:
                d: dict[str, Any] = {"role": msg.role}
                if msg.content is not None:
                    d["content"] = msg.content
                if msg.name is not None:
                    d["name"] = msg.name
                result.append(d)
        return result

    @staticmethod
    def _adapt_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Chat Completions nested tool format to Responses API flat format."""
        result = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                fn = tool["function"]
                adapted: dict[str, Any] = {
                    "type": "function",
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                    # The Responses API defaults to strict, which makes every optional
                    # field required; the rest of the code assumes Chat Completions' false.
                    "strict": bool(fn.get("strict", False)),
                }
                result.append(adapted)
            else:
                result.append(tool)
        return result

    @staticmethod
    async def preprocess_structured_outputs(structured_outputs: StructuredOutput | None) -> dict:
        if structured_outputs is None:
            return {}
        match structured_outputs.type:
            case StructuredOutputType.JSON_SCHEMA:
                return {
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "agent_step",
                            "strict": True,
                            "schema": structured_outputs.rules,
                        }
                    }
                }
            case _:
                raise ValueError(
                    f"Structured output type {structured_outputs.type} not supported by OpenAIProvider"
                )

    async def generate(
        self,
        messages: list[LLMMessage],
        structured_outputs: StructuredOutput | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Call OpenAI Responses API"""
        logger = get_logger()

        structured_outputs_params = await self.preprocess_structured_outputs(structured_outputs)

        start = time.time()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        serialized_messages = self._serialize_messages(messages)
        # Sanitize tool names at the boundary; name_map restores originals in the response.
        # sanitize_tool_names handles Chat Completions message format; also sanitize
        # function_call items (Responses API replay format) which it doesn't reach.
        tools, tool_choice, name_map = sanitize_tool_names(
            serialized_messages, tools, tool_choice, logger=logger
        )
        for item in serialized_messages:
            if item.get("type") == "function_call" and item.get("name"):
                item["name"] = sanitize_tool_name(item["name"])

        payload: dict[str, Any] = {
            "model": self.model_name,
            "input": serialized_messages,
        } | structured_outputs_params
        payload = {**payload, **self.compute_generation_params}

        if tools:
            payload["tools"] = self._adapt_tools(tools)
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        log_inference_request(logger, self.url, payload)

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(self.url, headers=headers, json=payload)
            if response.is_error:
                body_bytes = await response.aread()
                logger.error(f"OpenAI API {response.status_code} on {self.url}")
                logger.debug(
                    "OpenAI API error body",
                    payload={"body": body_bytes.decode("utf-8", "ignore")[:2000]},
                )
            response.raise_for_status()
            data = response.json()

            output_text = ""
            tool_calls: list[LLMToolCall] = []
            reasoning_content = None

            for item in data.get("output", []):
                item_type = item.get("type")
                if item_type == "message":
                    for part in item.get("content", []):
                        if part.get("type") == "output_text":
                            output_text += part.get("text", "")
                elif item_type == "function_call":
                    tool_calls.append(
                        LLMToolCall(
                            id=item["call_id"],
                            type="function",
                            function=ToolCallFunction(
                                name=name_map.get(item["name"], item["name"]),
                                arguments=item["arguments"],
                            ),
                        )
                    )
                elif item_type == "reasoning":
                    summaries = item.get("summary", [])
                    if summaries:
                        reasoning_content = " ".join(
                            s.get("text", "") for s in summaries if s.get("type") == "summary_text"
                        )

            usage = data.get("usage", {})
            generated_tokens = usage.get("output_tokens", 0)

            log_inference_response(
                logger,
                self.url,
                {
                    "response": output_text,
                    "generated_tokens": generated_tokens,
                    "execution_time": time.time() - start,
                    "tool_calls": [tc.model_dump() for tc in tool_calls],
                    "usage": usage,
                },
            )

            return LLMResponse(
                content=output_text,
                usage=usage,
                generated_tokens=generated_tokens,
                reasoning_content=reasoning_content,
                execution_time=time.time() - start,
                tool_calls=tool_calls,
            )

    async def generate_streaming(
        self,
        messages: list[LLMMessage],
        structured_outputs: StructuredOutput | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[LLMResponse, None]:
        """Call OpenAI Responses API with streaming"""
        logger = get_logger()
        structured_outputs_params = await self.preprocess_structured_outputs(structured_outputs)
        start = time.time()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        serialized_messages = self._serialize_messages(messages)
        tools, tool_choice, name_map = sanitize_tool_names(
            serialized_messages, tools, tool_choice, logger=logger
        )
        for item in serialized_messages:
            if item.get("type") == "function_call" and item.get("name"):
                item["name"] = sanitize_tool_name(item["name"])

        payload: dict[str, Any] = {
            "model": self.model_name,
            "input": serialized_messages,
            "stream": True,
        }
        payload = {
            **payload,
            **self.compute_generation_params,
            **kwargs,
        } | structured_outputs_params

        if tools:
            payload["tools"] = self._adapt_tools(tools)
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        log_inference_request(logger, self.url, payload, stream=True)

        token_count = 0
        full_output = ""
        content_started = False
        # Diagnostic: SSE data lines we failed to json-parse. Should stay ~0 with
        # line-buffered reading; a non-zero count paired with the invalid-arguments
        # error below means the upstream genuinely sent a malformed data line.
        dropped_sse_fragments = 0

        # Accumulator for streaming function-call deltas, keyed by output_index.
        # Each entry: {"id": str, "call_id": str, "name": str, "arguments": str}
        tool_call_acc: dict[int, dict[str, str]] = {}
        reasoning = ""
        usage: dict[str, Any] = {}

        # Stream line-by-line: httpx buffers across network chunks and yields
        # complete SSE lines. Re-chunking the body on arbitrary byte boundaries
        # (the old aiter_bytes + split("data:")) split events mid-JSON, dropping
        # tool-call argument deltas and corrupting the accumulated arguments.
        async with (
            httpx.AsyncClient(timeout=60.0) as client,
            client.stream("POST", self.url, headers=headers, json=payload) as response,
        ):
            if response.is_error:
                body_bytes = await response.aread()
                logger.error(f"OpenAI API {response.status_code} on {self.url}")
                logger.debug(
                    "OpenAI API error body",
                    payload={"body": body_bytes.decode("utf-8", "ignore")[:2000]},
                )
            response.raise_for_status()
            # Line-buffered read: httpx reassembles SSE lines across network
            # chunks. The old aiter_bytes()+splitlines() was line-aware only
            # within a single chunk, so an event split at a chunk boundary was
            # dropped, corrupting accumulated function-call arguments.
            async for line in response.aiter_lines():
                line = line.strip()
                # Skip blank lines and SSE comments (e.g. ": OPENROUTER PROCESSING").
                if not line.startswith("data:"):
                    continue
                raw = line[len("data:") :].strip()
                if not raw or raw == "[DONE]":
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    dropped_sse_fragments += 1
                    logger.warning("OpenAI SSE: unparsed data line")
                    logger.debug("OpenAI SSE: unparsed data line", payload={"data_line": raw[:200]})
                    continue

                event_type = data.get("type", "")

                if event_type == "response.output_text.delta":
                    content = data.get("delta", "").replace("\x00", "")
                    if not content_started:
                        content = content.lstrip()
                    if content:
                        content_started = True
                        token_count += 1
                        full_output += content
                        yield LLMResponse(
                            content=content,
                            usage={},
                            generated_tokens=token_count,
                            execution_time=time.time() - start,
                        )

                elif event_type == "response.output_item.added":
                    item = data.get("item", {})
                    if item.get("type") == "function_call":
                        idx = data.get("output_index", 0)
                        tool_call_acc[idx] = {
                            "id": item.get("id", ""),
                            "call_id": item.get("call_id", ""),
                            "name": item.get("name", ""),
                            "arguments": "",
                        }

                elif event_type == "response.function_call_arguments.delta":
                    idx = data.get("output_index", 0)
                    if idx in tool_call_acc:
                        tool_call_acc[idx]["arguments"] += data.get("delta", "")

                elif event_type == "response.output_item.done":
                    item = data.get("item", {})
                    if item.get("type") == "function_call":
                        idx = data.get("output_index", 0)
                        if idx in tool_call_acc:
                            tool_call_acc[idx]["call_id"] = item.get(
                                "call_id", tool_call_acc[idx]["call_id"]
                            )
                            tool_call_acc[idx]["name"] = item.get(
                                "name", tool_call_acc[idx]["name"]
                            )
                            tool_call_acc[idx]["arguments"] = item.get(
                                "arguments", tool_call_acc[idx]["arguments"]
                            )

                elif event_type == "response.completed":
                    usage = (data.get("response") or {}).get("usage") or usage

        # Build final LLMToolCall objects from the accumulated deltas
        tool_calls: list[LLMToolCall] = [
            LLMToolCall(
                id=acc["call_id"],
                type="function",
                function=ToolCallFunction(
                    name=name_map.get(acc["name"], acc["name"]),
                    arguments=acc["arguments"],
                ),
            )
            for acc in tool_call_acc.values()
        ]

        # Safety net: accumulated arguments must be valid JSON. If not, a streaming
        # delta was lost (historically an SSE fragment dropped at a chunk boundary).
        for tc in tool_calls:
            try:
                json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                logger.error(
                    "OpenAI SSE: accumulated tool-call arguments are invalid "
                    f"JSON | tool={tc.function.name} error={type(e).__name__} "
                    f"dropped_sse_fragments={dropped_sse_fragments}"
                )
                logger.debug(
                    "OpenAI SSE: invalid tool-call arguments payload | "
                    f"error={e} arguments={tc.function.arguments!r}"
                )

        # If there were tool calls, yield one final response carrying them all —
        # with the usage, which a consumer reads off the last chunk. Without tool
        # calls the usage gets a closing chunk of its own.
        if tool_calls:
            yield LLMResponse(
                content="",
                usage=usage,
                generated_tokens=token_count,
                execution_time=time.time() - start,
                tool_calls=tool_calls,
                reasoning_content=reasoning or None,
            )
        elif usage:
            yield LLMResponse(
                content="",
                usage=usage,
                generated_tokens=token_count,
                execution_time=time.time() - start,
            )

        log_inference_response(
            logger,
            self.url,
            {
                "response": full_output,
                "generated_tokens": token_count,
                "execution_time": time.time() - start,
                "tool_calls": [tc.model_dump() for tc in tool_calls],
                "usage": usage,
            },
        )
