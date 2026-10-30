import json
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
from domyn.agents.tracing import inject_trace_headers


class VLLMProvider(BaseLLMProvider):
    """Provider for LLM hosted on vLLM (using openai API)"""

    type: str = "vllm"
    url: str = Field(..., description="URL of the VLLM server")
    api_key: str | None = Field(default=None, description="API key for authentication")
    timeout: float = Field(
        default=300.0,
        description="HTTP client timeout in seconds for requests to the vLLM server. "
        "Increase for slow/large models that exceed the default read timeout.",
    )
    supported_structured_outputs: list[StructuredOutputType] | None = Field(
        default=[
            StructuredOutputType.LLGUIDANCE,
            StructuredOutputType.GBNF,
        ],
        description="Supported structured output types",
    )

    # No `truncate_prompt_tokens`: vLLM's own default rejects a prompt that does
    # not fit, and that rejection is what the tool-use planners answer by
    # compacting and retrying. `-1` cuts the prompt to fit by dropping its
    # beginning — the system prompt, the tool definitions — and answers as if
    # nothing had happened. Set it explicitly to opt into that.
    default_generation_params: ClassVar[dict[str, Any]] = {
        "temperature": 0.7,
        "max_tokens": 2000,
    }

    generation_params: dict[str, Any] = Field(
        default_factory=lambda: {
            "temperature": 0.7,
            "max_tokens": 2000,
        },
        description="Generation parameters for the model.",
    )

    def model_post_init(self, context: Any, /) -> None:
        if not self.name:
            self.name = self.model_name

    def preprocess_structured_outputs(
        self, structured_outputs: StructuredOutput | None = None
    ) -> dict:
        if structured_outputs is None:
            structured_outputs_params = {}
        else:
            match structured_outputs.type:
                case StructuredOutputType.LLGUIDANCE:
                    structured_outputs_params = {
                        "structured_outputs": {
                            "grammar": structured_outputs.rules,
                            "_backend": "guidance",
                        }
                    }
                case StructuredOutputType.GBNF:
                    structured_outputs_params = {
                        "structured_outputs": {
                            "grammar": structured_outputs.rules,
                            "_backend": "xgrammar",
                        }
                    }
                case StructuredOutputType.JSON_SCHEMA:
                    raise NotImplementedError  # TODO: implement JSON_SCHEMA structured outputs
                case _:
                    raise ValueError(
                        f"Unsupported structured output type: {structured_outputs.type}"
                    )

        return structured_outputs_params

    async def generate(
        self,
        messages: list[LLMMessage],
        structured_outputs: StructuredOutput | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Call vLLM API"""
        logger = get_logger()

        structured_outputs_params = self.preprocess_structured_outputs(structured_outputs)

        start = time.time()
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
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

        payload = {
            "model": self.model_name,
            "messages": serialized_messages,
        }
        payload = {**payload, **self.generation_params, **kwargs} | structured_outputs_params

        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if logprobs is not None:
            payload["logprobs"] = logprobs
        if top_logprobs is not None:
            payload["top_logprobs"] = top_logprobs

        # inject trace headers if tracing is enabled
        headers = inject_trace_headers(headers)

        log_inference_request(logger, self.url, payload)

        # TODO: reuse a caller-provided httpx.AsyncClient instead of opening one per request.
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.url, headers=headers, json=payload)
            if response.is_error:
                body_bytes = await response.aread()
                logger.error(f"vLLM API {response.status_code} on {self.url}")
                logger.debug(
                    "vLLM API error body",
                    payload={"body": body_bytes.decode("utf-8", "ignore")[:2000]},
                )
            response.raise_for_status()
            data = response.json()

            full_output = data["choices"][0]["message"]["content"]
            token_count = data["usage"]["completion_tokens"]

            tool_calls: list[LLMToolCall] = []
            for tc in data["choices"][0]["message"].get("tool_calls") or []:
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
                    "response": full_output,
                    "generated_tokens": token_count,
                    "execution_time": time.time() - start,
                    "tool_calls": [tc.model_dump() for tc in tool_calls],
                    "usage": data.get("usage", {}),
                },
            )
            return LLMResponse(
                content=full_output or "",
                usage=data.get("usage", {}),
                generated_tokens=token_count,
                reasoning_content=data["choices"][0]["message"].get("reasoning_content")
                or data["choices"][0]["message"].get("reasoning", None),
                finish_reason=data["choices"][0].get("finish_reason"),
                logprobs=data["choices"][0].get("logprobs"),
                execution_time=time.time() - start,
                tool_calls=tool_calls,
            )

    async def generate_streaming(
        self,
        messages: list[LLMMessage],
        structured_outputs: StructuredOutput | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[LLMResponse, None]:
        """Call vLLM API with true token-by-token streaming."""
        logger = get_logger()

        structured_outputs_params = self.preprocess_structured_outputs(structured_outputs)

        start = time.time()
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
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
        }
        payload = {**payload, **self.generation_params, **kwargs} | structured_outputs_params

        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if logprobs is not None:
            payload["logprobs"] = logprobs
        if top_logprobs is not None:
            payload["top_logprobs"] = top_logprobs

        # inject trace headers if tracing is enabled
        headers = inject_trace_headers(headers)

        log_inference_request(logger, self.url, payload, stream=True)

        token_count = 0
        full_output = ""
        content_started = False
        reasoning_sent = False
        finish_reason: str | None = None

        # Accumulator for streaming tool-call deltas, keyed by their index.
        # Each entry: {"id": str, "type": str, "name": str, "arguments": str}
        tool_call_acc: dict[int, dict] = {}
        reasoning = ""
        usage: dict[str, Any] = {}

        async with (
            httpx.AsyncClient(timeout=self.timeout) as client,
            client.stream("POST", self.url, headers=headers, json=payload) as response,
        ):
            if response.is_error:
                body_bytes = await response.aread()
                logger.error(f"vLLM API {response.status_code} on {self.url}")
                logger.debug(
                    "vLLM API error body",
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
                    # Requested with `include_usage`, it closes the stream: in a
                    # chunk with no choices (vLLM) or with an empty delta (LiteLLM).
                    usage = data.get("usage") or usage
                    if not data["choices"]:
                        continue
                    # Recorded, not acted on here: a truncated stream still
                    # yields what arrived, and only the consumer knows whether
                    # a cut-off answer is usable.
                    if data["choices"][0].get("finish_reason"):
                        finish_reason = data["choices"][0]["finish_reason"]
                    chunk_logprobs = data["choices"][0].get("logprobs")
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
                            # id/type/name only arrive in the first chunk
                            if tc_delta.get("id"):
                                tool_call_acc[idx]["id"] = tc_delta["id"]
                            if tc_delta.get("type"):
                                tool_call_acc[idx]["type"] = tc_delta["type"]
                            if tc_delta.get("function", {}).get("name"):
                                tool_call_acc[idx]["name"] = tc_delta["function"]["name"]
                        # arguments arrive in pieces — always concatenate
                        tool_call_acc[idx]["arguments"] += tc_delta.get("function", {}).get(
                            "arguments", ""
                        )

                    # --- reasoning delta accumulation ---
                    if delta.get("reasoning"):
                        reasoning += delta.get("reasoning")
                    elif delta.get("reasoning_content"):
                        reasoning += delta.get("reasoning_content")

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
                            logprobs=chunk_logprobs,
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

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "VLLMProvider":
        required = ["url", "model_name"]
        missing = [k for k in required if k not in config]
        if missing:
            raise ValueError(f"Missing required VLLMProvider config keys: {missing}")
        return cls(
            url=config["url"],
            model_name=config["model_name"],
            generation_params=config.get("generation_params")
            or {
                "temperature": 0.7,
                "max_tokens": 2000,
            },
            api_key=config.get("api_key"),
            name=config.get("name") or config["model_name"],
            append_text_last_message=config.get("append_text_last_message"),
            append_text_system_prompt=config.get("append_text_system_prompt"),
        )
