import asyncio
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
from domyn.agents.logger import get_logger
from domyn.agents.tools.base import sanitize_tool_names


class CerebrasProvider(BaseLLMProvider):
    """Provider for LLM hosted on Cerebras"""

    type: str = "cerebras"
    url: str = Field(
        default="https://api.cerebras.ai/v1/chat/completions",
        description="URL of the Cerebras server",
    )
    api_key: str = Field(
        default=os.getenv("CEREBRAS_API_KEY"), description="API key for authentication"
    )
    context_window_catalog: ClassVar[str | None] = "cerebras"

    default_generation_params: ClassVar[dict[str, Any]] = {
        "temperature": 0.7,
        "max_tokens": 8000,
    }

    generation_params: dict[str, Any] = Field(
        default_factory=lambda: {"temperature": 0.7, "max_tokens": 8000},
        description="Generation parameters for the model",
    )

    supported_structured_outputs: list[StructuredOutputType] | None = Field(
        # JSON_SCHEMA off by default: it stopped working on Cerebras (2026-01). Opt in via
        # supported_structured_outputs.
        default=None,
        description="Supported structured output types",
    )
    max_retries: int = Field(default=3, description="Maximum number of retries for API calls")

    def model_post_init(self, context: Any, /) -> None:
        if not self.name:
            self.name = self.model_name
        if self.api_key is None:
            raise ValueError(f"API key must be provided for CerebrasProvider: {self.model_name}")

    @staticmethod
    def preprocess_structured_outputs(
        structured_outputs: StructuredOutput | None,
    ) -> dict[str, Any]:
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
                        f"Structured output type {structured_outputs.type} not supported by CerebrasProvider"
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
        """Call Cerebras API"""
        logger = get_logger()

        structured_outputs_params = self.preprocess_structured_outputs(
            structured_outputs=structured_outputs
        )

        start = time.time()
        headers = {
            "Content-Type": "application/json",
        }
        headers["Authorization"] = f"Bearer {self.api_key}"

        if self.append_text_system_prompt:
            messages[0].content += "\n" + self.append_text_system_prompt

        if self.append_text_last_message:
            messages[-1].content += "\n" + self.append_text_last_message

        serialized_messages = [message.to_dict() for message in messages]
        # Sanitize tool names at the boundary; name_map restores originals in the response.
        tools, tool_choice, name_map = sanitize_tool_names(
            serialized_messages, tools, tool_choice, logger=logger
        )

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": serialized_messages,
        } | structured_outputs_params
        payload = {**payload, **self.generation_params}

        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        log_inference_request(logger, self.url, payload)

        # TODO: reuse a caller-provided httpx.AsyncClient instead of opening one per request.
        async with httpx.AsyncClient(timeout=60.0) as client:
            count = 0
            while count < self.max_retries:  # fix for cerebras that return invalid json response
                response = await client.post(self.url, headers=headers, json=payload)
                if response.is_error:
                    body_bytes = await response.aread()
                    logger.error(f"Cerebras API {response.status_code} on {self.url}")
                    logger.debug(
                        "Cerebras API error body",
                        payload={"body": body_bytes.decode("utf-8", "ignore")[:2000]},
                    )
                response.raise_for_status()
                data = json.loads(response.text.strip())

                if not data:
                    await asyncio.sleep(0.5)  # wait a bit before retrying
                    count += 1
                    continue

                # Defensive: retry if a 2xx body lacks `choices` instead of
                # crashing on `data["choices"]` with an opaque KeyError. A
                # choices-less response was observed once as a *streaming* chunk
                # the raw body was never captured and it has not reproduced, so this
                # non-streaming guard is precautionary.
                if "choices" not in data:
                    logger.warning("Cerebras 200 without 'choices' (soft error?), retrying")
                    logger.debug(
                        "Cerebras 200 without 'choices' body",
                        payload={"body": json.dumps(data)[:2000]},
                    )
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

                if content or tool_calls:
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
        """Call Cerebras API with streaming support"""
        logger = get_logger()

        structured_outputs_params = self.preprocess_structured_outputs(
            structured_outputs=structured_outputs
        )

        start = time.time()
        headers = {
            "Content-Type": "application/json",
        }
        headers["Authorization"] = f"Bearer {self.api_key}"

        serialized_messages = [message.to_dict() for message in messages]
        # Sanitize tool names at the boundary; name_map restores originals in the response.
        tools, tool_choice, name_map = sanitize_tool_names(
            serialized_messages, tools, tool_choice, logger=logger
        )

        payload = {
            "model": self.model_name,
            "messages": serialized_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        } | structured_outputs_params
        payload = {**payload, **self.generation_params, **kwargs}
        # vLLM-only parameter passed by the react planner; not part of the Cerebras API.
        payload.pop("add_generation_prompt", None)

        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        log_inference_request(logger, self.url, payload, stream=True)

        token_count = 0
        full_output = ""
        content_started = False
        reasoning_sent = False

        # Accumulator for streaming tool-call deltas, keyed by their index.
        # Each entry: {"id": str, "type": str, "name": str, "arguments": str}
        tool_call_acc: dict[int, dict] = {}
        reasoning = ""
        usage: dict[str, Any] = {}

        async with (
            httpx.AsyncClient(timeout=60.0) as client,
            client.stream("POST", self.url, headers=headers, json=payload) as response,
        ):
            if response.is_error:
                body_bytes = await response.aread()
                logger.error(f"Cerebras API {response.status_code} on {self.url}")
                logger.debug(
                    "Cerebras API error body",
                    payload={"body": body_bytes.decode("utf-8", "ignore")[:2000]},
                )
            response.raise_for_status()
            async for line in response.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                c = line[5:].strip()
                if not c or c == "[DONE]":
                    continue
                try:
                    data = json.loads(c)
                    # Requested with `include_usage`: a closing chunk of its own.
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
                        content = content.lstrip("\n")
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
                except json.JSONDecodeError:
                    pass

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
    def from_config(cls, config: dict[str, Any]) -> "CerebrasProvider":
        required = ["url", "model_name"]
        missing = [k for k in required if k not in config]
        if missing:
            raise ValueError(f"Missing required Cerebras config keys: {missing}")
        return cls(
            url=config["url"],
            model_name=config["model_name"],
            generation_params=config.get("generation_params")
            or {"temperature": 0.7, "max_tokens": 2000},
            api_key=config.get("api_key"),
            name=config.get("name") or config["model_name"],
            append_text_last_message=config.get("append_text_last_message"),
            append_text_system_prompt=config.get("append_text_system_prompt"),
        )
