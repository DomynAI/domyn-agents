"""Custom LLM provider for arbitrary OpenAI-compatible endpoints."""

import json
import time
from collections.abc import AsyncGenerator
from typing import Any

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


class CustomLLMProvider(BaseLLMProvider):
    """Provider for any OpenAI-compatible endpoint with configurable headers and body extras."""

    type: str = "custom"
    url: str = Field(..., description="Full URL of the chat completions endpoint")
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Custom HTTP headers sent with every request (e.g. Authorization)",
    )
    body: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra key-value pairs merged into the request body",
    )
    supported_structured_outputs: list[StructuredOutputType] | None = None

    def model_post_init(self, context: Any, /) -> None:
        if not self.name:
            self.name = self.model_name

    def _context_window_headers(self) -> dict[str, str]:
        """The endpoint's own headers: its credentials live there, not in an api key."""
        return dict(self.headers)

    async def generate(
        self,
        messages: list[LLMMessage],
        structured_outputs: StructuredOutput
        | None = None,  # TODO: Add support or add a warning if structured_outputs is provided but not supported
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        logger = get_logger()

        start = time.time()
        request_headers = {"Content-Type": "application/json", **self.headers}

        serialized_messages = [m.to_dict() for m in messages]
        # Sanitize tool names at the boundary; name_map restores originals in the response.
        tools, tool_choice, name_map = sanitize_tool_names(
            serialized_messages, tools, tool_choice, logger=logger
        )

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": serialized_messages,
            **self.body,
            **kwargs,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        # TODO: Add tracing headers
        # TODO: Add support for self.append_text_system_prompt and self.append_text_last_message

        log_inference_request(logger, self.url, payload)

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(self.url, headers=request_headers, json=payload)
            if response.is_error:
                body_bytes = await response.aread()
                logger.error(f"Custom LLM API {response.status_code} on {self.url}")
                logger.debug(
                    "Custom LLM API error body",
                    payload={"body": body_bytes.decode("utf-8", "ignore")[:2000]},
                )
            response.raise_for_status()
            data = response.json()

            message_data = data["choices"][0]["message"]

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
                    "response": message_data.get("content") or "",
                    "generated_tokens": data.get("usage", {}).get("completion_tokens", 0),
                    "execution_time": time.time() - start,
                    "tool_calls": [tc.model_dump() for tc in tool_calls],
                    "usage": data.get("usage", {}),
                },
            )

            return LLMResponse(
                content=message_data.get("content") or "",
                usage=data.get("usage", {}),
                generated_tokens=data.get("usage", {}).get("completion_tokens", 0),
                reasoning_content=message_data.get("reasoning_content")
                or message_data.get("reasoning"),
                finish_reason=data["choices"][0].get("finish_reason"),
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
        logger = get_logger()

        start = time.time()
        request_headers = {"Content-Type": "application/json", **self.headers}

        serialized_messages = [m.to_dict() for m in messages]
        # Sanitize tool names at the boundary; name_map restores originals in the response.
        tools, tool_choice, name_map = sanitize_tool_names(
            serialized_messages, tools, tool_choice, logger=logger
        )

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": serialized_messages,
            "stream": True,
            **self.body,
            **kwargs,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        log_inference_request(logger, self.url, payload, stream=True)

        # TODO: Add tracing headers
        # TODO: Add support for self.append_text_system_prompt and self.append_text_last_message

        token_count = 0
        full_output = ""
        content_started = False
        reasoning_sent = False
        finish_reason: str | None = None
        tool_call_acc: dict[int, dict] = {}
        reasoning = ""
        usage: dict[str, Any] = {}

        async with (
            httpx.AsyncClient(timeout=300.0) as client,
            client.stream("POST", self.url, headers=request_headers, json=payload) as response,
        ):
            if response.is_error:
                body_bytes = await response.aread()
                logger.error(f"Custom LLM API {response.status_code} on {self.url}")
                logger.debug(
                    "Custom LLM API error body",
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
                    # Only if the endpoint is asked for it (`stream_options` in
                    # `body`): a closing chunk, with no choices or an empty delta.
                    usage = data.get("usage") or usage
                    if not data.get("choices"):
                        continue
                    # Recorded, not acted on here: a truncated stream still
                    # yields what arrived, and only the consumer knows whether
                    # a cut-off answer is usable.
                    if data["choices"][0].get("finish_reason"):
                        finish_reason = data["choices"][0]["finish_reason"]
                    delta = data["choices"][0]["delta"]

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

                    if delta.get("reasoning"):
                        reasoning += delta["reasoning"]
                    elif delta.get("reasoning_content"):
                        reasoning += delta["reasoning_content"]

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

        tool_calls: list[LLMToolCall] = [
            LLMToolCall(
                id=acc["id"],
                type=acc["type"],
                function=ToolCallFunction(
                    name=name_map.get(acc["name"], acc["name"]), arguments=acc["arguments"]
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
                finish_reason=finish_reason,
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
