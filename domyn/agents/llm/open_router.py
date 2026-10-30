import asyncio
import json
import os
import time
from collections.abc import AsyncGenerator
from typing import Any, ClassVar

import httpx
from httpx import AsyncClient
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
from domyn.agents.logger import get_logger
from domyn.agents.tools.base import sanitize_tool_names


class OpenRouterProvider(BaseLLMProvider):
    """Provider for LLM hosted on OpenRouter"""

    type: str = "open_router"
    url: str = Field(
        default="https://openrouter.ai/api/v1/chat/completions",
        description="URL of the OpenRouter server",
    )
    api_key: str = Field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""),
        description="API key for authentication",
    )

    default_generation_params: ClassVar[dict[str, Any]] = {
        "temperature": 0.7,
        "max_tokens": 8000,
    }

    generation_params: dict[str, Any] = Field(
        default_factory=lambda: {"temperature": 0.7, "max_tokens": 8000},
        description="Generation parameters for the model",
    )
    supported_structured_outputs: list[StructuredOutputType] | None = Field(
        default=None, description="Supported structured output types"
    )
    max_retries: int = Field(default=3, description="Maximum number of retries for API calls")
    reasoning_supported: bool = Field(
        default=False,
        description=(
            "Whether the model accepts the `reasoning` parameter. Set by the caller "
            "from OpenRouter's model metadata; when False the "
            "`reasoning` param is never sent — avoids 400s on models without "
            "thinking support (e.g. llama-4 on Vertex)."
        ),
    )

    _verified_structured_output_support: bool = False

    def model_post_init(self, context: Any, /) -> None:
        if not self.name:
            self.name = self.model_name
        if not self.api_key:
            raise ValueError(f"API key must be provided for OpenRouterProvider: {self.model_name}")

    async def verify_structured_outputs_support(self, client: AsyncClient) -> None:
        # TODO: probe once before the first request: the planner reads supported_structured_outputs
        # before generate() calls this, so the first request goes without structured outputs.
        logger = get_logger()
        if not self._verified_structured_output_support:
            response = await client.get(
                url="https://openrouter.ai/api/v1/models?supported_parameters=structured_outputs"
            )
            if response.is_error:
                body_bytes = await response.aread()
                logger.error(f"OpenRouter API {response.status_code} on {response.request.url}")
                logger.debug(
                    "OpenRouter API error body",
                    payload={"body": body_bytes.decode("utf-8", "ignore")[:2000]},
                )
            response.raise_for_status()
            data = response.json()
            supported_models = [model["id"] for model in data["data"]]
            has_structured_output_support = self.model_name in supported_models
            if has_structured_output_support:
                if self.supported_structured_outputs is None:
                    self.supported_structured_outputs = [
                        StructuredOutputType.JSON_SCHEMA,
                    ]
            else:
                self.supported_structured_outputs = None
            self._verified_structured_output_support = True

    async def preprocess_structured_outputs(
        self,
        client: AsyncClient,
        structured_outputs: StructuredOutput | None,
    ) -> dict:
        await self.verify_structured_outputs_support(client=client)
        if structured_outputs is None:
            structured_outputs_params = {}
        else:
            match structured_outputs.type:
                case StructuredOutputType.JSON_SCHEMA:
                    structured_outputs_params = {
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "agent_step",
                                "strict": True,
                                "schema": structured_outputs.rules,
                            },
                        }
                    }
                case _:
                    raise ValueError(
                        f"Structured output type {structured_outputs.type} not supported by OpenRouterProvider"
                    )
        return structured_outputs_params

    async def generate(
        self,
        messages: list[LLMMessage],
        structured_outputs: StructuredOutput | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Call OpenRouter API"""
        logger = get_logger()

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        start = time.time()

        async with httpx.AsyncClient(timeout=60.0) as client:
            structured_outputs_params = await self.preprocess_structured_outputs(
                client=client, structured_outputs=structured_outputs
            )
            serialized_messages = [message.to_dict() for message in messages]
            # Sanitize tool names at the boundary; name_map restores originals in the response.
            tools, tool_choice, name_map = sanitize_tool_names(
                serialized_messages, tools, tool_choice, logger=logger
            )

            payload: dict[str, Any] = {
                "model": self.model_name,
                "messages": serialized_messages,
            } | structured_outputs_params
            # Only models that advertise reasoning support accept the param;
            # sending it to others (e.g. llama-4 on Vertex) is a hard 400.
            if self.reasoning_supported:
                payload["reasoning"] = {"effort": "medium"}
            payload = {**payload, **self.generation_params, **kwargs}
            # Gate is authoritative: strip `reasoning` if generation_params/kwargs
            # re-introduced it for a model that doesn't support it (else a hard 400).
            if not self.reasoning_supported:
                payload.pop("reasoning", None)

            if tools:
                payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

            log_inference_request(logger, self.url, payload)

            count = 0
            while count < self.max_retries:  # retry on empty response
                response = await client.post(self.url, headers=headers, json=payload)
                if response.is_error:
                    body_bytes = await response.aread()
                    logger.error(f"OpenRouter API {response.status_code} on {self.url}")
                    logger.debug(
                        "OpenRouter API error body",
                        payload={"body": body_bytes.decode("utf-8", "ignore")[:2000]},
                    )
                response.raise_for_status()
                data = json.loads(response.text.strip())

                if not data:
                    await asyncio.sleep(0.5)  # wait a bit before retrying
                    count += 1
                    continue

                message_data = data["choices"][0]["message"]
                content = message_data.get("content") or ""

                tool_calls: list[LLMToolCall] = []
                for tc in message_data.get("tool_calls") or []:
                    tool_calls.append(
                        LLMToolCall(
                            id=tc["id"],
                            type=tc.get("type", "function"),
                            function=ToolCallFunction(
                                name=name_map.get(tc["function"]["name"], tc["function"]["name"]),
                                arguments=tc["function"]["arguments"],
                            ),
                        )
                    )

                log_inference_response(
                    logger,
                    self.url,
                    {
                        "response": content,
                        "generated_tokens": data.get("usage", {}).get("completion_tokens", 0),
                        "execution_time": time.time() - start,
                        "tool_calls": [tc.model_dump() for tc in tool_calls],
                        "usage": data.get("usage", {}),
                    },
                )

                if content or tool_calls:
                    return LLMResponse(
                        content=content,
                        usage=data.get("usage", {}),
                        generated_tokens=data.get("usage", {}).get("completion_tokens", 0),
                        reasoning_content=data["choices"][0]["message"].get("reasoning_content")
                        or data["choices"][0]["message"].get("reasoning", None),
                        execution_time=time.time() - start,
                        tool_calls=tool_calls,
                    )

                count += 1

    async def generate_streaming(
        self,
        messages: list[LLMMessage],
        structured_outputs: StructuredOutput | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[LLMResponse, None]:
        """Call OpenRouter API with streaming support"""
        logger = get_logger()

        start = time.time()
        headers = {
            "Content-Type": "application/json",
        }
        headers["Authorization"] = f"Bearer {self.api_key}"
        async with httpx.AsyncClient(timeout=60.0) as client:
            structured_outputs_params = await self.preprocess_structured_outputs(
                client=client, structured_outputs=structured_outputs
            )
            if self.append_text_system_prompt:
                messages[0].content += "\n" + self.append_text_system_prompt

            if self.append_text_last_message:
                messages[-1].content += "\n" + self.append_text_last_message

            serialized_messages = [message.to_dict() for message in messages]
            # Sanitize tool names at the boundary; name_map restores originals in the response.
            tools, tool_choice, name_map = sanitize_tool_names(
                serialized_messages, tools, tool_choice, logger=logger
            )

            payload = {
                "model": self.model_name,
                "messages": serialized_messages,
                "stream": True,
            }
            # Only models that advertise reasoning support accept the param;
            # sending it to others (e.g. llama-4 on Vertex) is a hard 400.
            if self.reasoning_supported:
                payload["reasoning"] = {"effort": "medium"}
            payload = {**payload, **self.generation_params, **kwargs} | structured_outputs_params
            # Gate is authoritative: strip `reasoning` if generation_params/kwargs
            # re-introduced it for a model that doesn't support it (else a hard 400).
            if not self.reasoning_supported:
                payload.pop("reasoning", None)

            if tools:
                payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

            log_inference_request(logger, self.url, payload, stream=True)

            token_count = 0
            full_output = ""
            content_started = False
            reasoning_sent = False
            # Diagnostic: SSE data lines we failed to json-parse. Should stay ~0
            # now that aiter_lines() yields complete lines (the old
            # aiter_bytes()+split("data:") could split an event mid-JSON). A
            # non-zero count paired with the invalid-arguments ERROR below means
            # the upstream genuinely sent a malformed data line.
            dropped_sse_fragments = 0

            # Accumulator for streaming tool-call deltas, keyed by their index.
            # Each entry: {"id": str, "type": str, "name": str, "arguments": str}
            tool_call_acc: dict[int, dict] = {}
            reasoning = ""
            usage: dict[str, Any] = {}

            # Stream line-by-line: httpx buffers across network chunks and
            # yields complete SSE lines. Re-chunking the body on arbitrary byte
            # boundaries (the old aiter_bytes + split("data:")) split events
            # mid-JSON, dropping tool-call argument deltas and corrupting the
            # accumulated arguments string.
            async with client.stream("POST", self.url, headers=headers, json=payload) as response:
                if response.is_error:
                    body_bytes = await response.aread()
                    logger.error(f"OpenRouter API {response.status_code} on {self.url}")
                    logger.debug(
                        "OpenRouter API error body",
                        payload={"body": body_bytes.decode("utf-8", "ignore")[:2000]},
                    )
                response.raise_for_status()
                async for line in response.aiter_lines():
                    line = line.strip()
                    # Skip blank lines and SSE comments (e.g. ": OPENROUTER PROCESSING").
                    if not line.startswith("data:"):
                        continue
                    payload_str = line[len("data:") :].strip()
                    if not payload_str or payload_str == "[DONE]":
                        continue
                    try:
                        data = json.loads(payload_str)
                    except json.JSONDecodeError:
                        # Should be ~0 now that lines are complete; if it fires,
                        # the upstream genuinely sent a malformed data line.
                        dropped_sse_fragments += 1
                        logger.warning("OpenRouter SSE: unparsed data line")
                        logger.debug(
                            "OpenRouter SSE: unparsed data line",
                            payload={"data_line": payload_str[:200]},
                        )
                        continue

                    # When OpenRouter reports usage it closes the stream with it.
                    usage = data.get("usage") or usage
                    if not data.get("choices"):
                        continue
                    delta = data["choices"][0]["delta"]

                    # --- tool-call delta accumulation ---
                    for tc_delta in delta.get("tool_calls") or []:
                        idx = tc_delta["index"]
                        if idx not in tool_call_acc:
                            tool_call_acc[idx] = {
                                "id": tc_delta.get("id", ""),
                                "type": tc_delta.get("type", "function"),
                                "name": tc_delta.get("function", {}).get("name", ""),
                                "arguments": "",
                            }
                        else:
                            if tc_delta.get("id"):
                                tool_call_acc[idx]["id"] = tc_delta["id"]
                            if tc_delta.get("type"):
                                tool_call_acc[idx]["type"] = tc_delta["type"]
                            if tc_delta.get("function", {}).get("name"):
                                tool_call_acc[idx]["name"] = tc_delta["function"]["name"]
                        tool_call_acc[idx]["arguments"] += tc_delta.get("function", {}).get(
                            "arguments", ""
                        )

                    # --- reasoning delta accumulation ---
                    if delta.get("reasoning"):
                        reasoning += delta.get("reasoning")

                    # --- text content ---
                    content = (delta.get("content") or "").replace("\x00", "")
                    if not content_started:
                        content = content.lstrip()
                    if content:
                        # Reasoning arrives before the answer and is complete once
                        # the first content token appears. Surfacing it here, in a
                        # chunk of its own, lets a consumer show the thinking
                        # *before* the answer it produced. Held to the end it would
                        # arrive after, which reads backwards.
                        if reasoning and not reasoning_sent:
                            reasoning_sent = True
                            yield LLMResponse(
                                content="",
                                usage={},
                                generated_tokens=token_count,
                                execution_time=time.time() - start,
                                reasoning_content=reasoning,
                            )
                        content_started = True
                        token_count += 1
                        full_output += content
                        yield LLMResponse(
                            content=content,
                            usage={},
                            generated_tokens=token_count,
                            execution_time=time.time() - start,
                        )

        # Build final LLMToolCall objects from the accumulated deltas
        tool_calls: list[LLMToolCall] = [
            LLMToolCall(
                id=acc["id"],
                type=acc["type"],
                function=ToolCallFunction(
                    name=name_map.get(acc["name"], acc["name"]),
                    arguments=acc["arguments"],
                ),
            )
            for acc in tool_call_acc.values()
        ]

        # Diagnostic: the accumulated arguments string must be valid JSON. If it
        # is not, a streaming delta was lost — almost always an SSE fragment
        # dropped above at a chunk boundary. Surface it loudly together with the
        # drop count so the cause is unambiguous downstream of this provider.
        for tc in tool_calls:
            try:
                json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                logger.error(
                    "OpenRouter SSE: accumulated tool-call arguments are invalid "
                    f"JSON | tool={tc.function.name} error={type(e).__name__} "
                    f"dropped_sse_fragments={dropped_sse_fragments}"
                )
                logger.debug(
                    "OpenRouter SSE: invalid tool-call arguments payload | "
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

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OpenRouterProvider":
        required = ["url", "model_name"]
        missing = [k for k in required if k not in config]
        if missing:
            raise ValueError(f"Missing required OpenRouter config keys: {missing}")
        return cls(
            url=config["url"],
            model_name=config["model_name"],
            generation_params=config.get("generation_params")
            or {"temperature": 0.7, "max_tokens": 2000},
            api_key=config.get("api_key"),
            name=config.get("name") or config["model_name"],
            append_text_last_message=config.get("append_text_last_message"),
            append_text_system_prompt=config.get("append_text_system_prompt"),
            reasoning_supported=config.get("reasoning_supported", False),
        )
