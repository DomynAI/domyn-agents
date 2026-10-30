"""LLM provider for the models of a Domyn platform space, called through ``domyn-platform-sdk``."""

import os
import time
from collections.abc import AsyncGenerator
from typing import Any

from pydantic import AliasChoices, Field

from domyn.agents.integrations import domyn_platform_sdk
from domyn.agents.integrations.domyn_platform_sdk import PlatformSettings
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


def _from_env(name: str) -> Any:
    return lambda: os.environ.get(name, "")


class DomynProvider(BaseLLMProvider):
    """The models a Domyn platform space offers, reached with the user's own API key.

    ``model_name`` is the name the space's configuration gives the model. The platform holds the
    provider credentials; the endpoint, headers and host rules are the SDK's. It needs
    ``domyn-platform-sdk`` installed next to domyn-agents (see ``integrations/domyn_platform_sdk.py``).

    Every connection field falls back to the variable the rest of the platform integration reads, so a
    provider with only a ``model_name`` works where ``DOMYN_API_KEY``, ``DOMYN_SPACE_ID`` and
    ``DOMYN_BASE_URL`` are set.
    """

    type: str = "domyn"
    api_key: str = Field(default_factory=_from_env("DOMYN_API_KEY"), repr=False)
    space_id: str = Field(default_factory=_from_env("DOMYN_SPACE_ID"))
    base_url: str = Field(
        default_factory=_from_env("DOMYN_BASE_URL"),
        validation_alias=AliasChoices("base_url", "url"),
        description="The platform's host, with or without a scheme. Also read as ``url``.",
    )
    configuration_id: str | None = Field(
        default_factory=lambda: os.environ.get("DOMYN_CONFIGURATION_ID") or None,
        description="Configuration to resolve the model from; the space's last released one when empty.",
    )
    generation_params: dict[str, Any] = Field(default_factory=dict)
    body: dict[str, Any] = Field(
        default_factory=dict, description="Extra key-value pairs merged into the request body"
    )
    supported_structured_outputs: list[StructuredOutputType] | None = None

    def model_post_init(self, context: Any, /) -> None:
        if not self.name:
            self.name = self.model_name

    @property
    def settings(self) -> PlatformSettings:
        return PlatformSettings(
            api_key=self.api_key,
            space_id=self.space_id,
            base_url=self.base_url,
            configuration_id=self.configuration_id,
        )

    def _payload(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        serialized = [m.to_dict() for m in messages]
        # Sanitize tool names at the boundary; name_map restores originals in the response.
        tools, tool_choice, name_map = sanitize_tool_names(
            serialized, tools, tool_choice, logger=get_logger()
        )
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": serialized,
            **self.generation_params,
            **self.body,
            **kwargs,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        return payload, name_map

    async def generate(
        self,
        messages: list[LLMMessage],
        structured_outputs: StructuredOutput | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        logger = get_logger()
        start = time.time()
        payload, name_map = self._payload(messages, tools, tool_choice, kwargs)
        log_inference_request(logger, self.base_url, payload)

        data = await domyn_platform_sdk.chat_completion(self.settings, payload)

        choice = data["choices"][0]
        message = choice["message"]
        usage = data.get("usage") or {}
        tool_calls = [
            LLMToolCall(
                id=tc["id"],
                type=tc.get("type", "function"),
                function=ToolCallFunction(
                    name=name_map.get(tc["function"]["name"], tc["function"]["name"]),
                    arguments=tc["function"]["arguments"],
                ),
            )
            for tc in message.get("tool_calls") or []
        ]
        log_inference_response(
            logger,
            self.base_url,
            {
                "response": message.get("content") or "",
                "generated_tokens": usage.get("completion_tokens", 0),
                "execution_time": time.time() - start,
                "tool_calls": [tc.model_dump() for tc in tool_calls],
                "usage": usage,
            },
        )
        return LLMResponse(
            content=message.get("content") or "",
            usage=usage,
            generated_tokens=usage.get("completion_tokens", 0),
            reasoning_content=message.get("reasoning_content") or message.get("reasoning"),
            finish_reason=choice.get("finish_reason"),
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
        payload, name_map = self._payload(messages, tools, tool_choice, kwargs)
        log_inference_request(logger, self.base_url, payload, stream=True)

        token_count = 0
        full_output = ""
        content_started = False
        reasoning_sent = False
        finish_reason: str | None = None
        tool_call_acc: dict[int, dict[str, Any]] = {}
        reasoning = ""
        usage: dict[str, Any] = {}

        async for data in domyn_platform_sdk.stream_chat_completion(self.settings, payload):
            # Only if the platform is asked for it (`stream_options` in `body`): a closing chunk,
            # with no choices or an empty delta.
            usage = data.get("usage") or usage
            if not data.get("choices"):
                continue
            # Recorded, not acted on here: a truncated stream still yields what arrived, and only
            # the consumer knows whether a cut-off answer is usable.
            if data["choices"][0].get("finish_reason"):
                finish_reason = data["choices"][0]["finish_reason"]
            delta = data["choices"][0].get("delta") or {}

            for tc_delta in delta.get("tool_calls") or []:
                function = tc_delta.get("function") or {}
                acc = tool_call_acc.setdefault(
                    tc_delta["index"],
                    {"id": "", "type": "function", "name": "", "arguments": ""},
                )
                if tc_delta.get("id"):
                    acc["id"] = tc_delta["id"]
                if tc_delta.get("type"):
                    acc["type"] = tc_delta["type"]
                if function.get("name"):
                    acc["name"] = function["name"]
                acc["arguments"] += function.get("arguments") or ""

            reasoning += delta.get("reasoning") or delta.get("reasoning_content") or ""

            content = (delta.get("content") or "").replace("\x00", "")
            if not content_started:
                content = content.lstrip("\n")
            if not content:
                continue
            # Reasoning arrives before the answer and is complete once the first content token
            # appears. Surfacing it here, in a chunk of its own, lets a consumer show the thinking
            # *before* the answer it produced.
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

        tool_calls = [
            LLMToolCall(
                id=acc["id"],
                type=acc["type"],
                function=ToolCallFunction(
                    name=name_map.get(acc["name"], acc["name"]), arguments=acc["arguments"]
                ),
            )
            for acc in tool_call_acc.values()
        ]

        # If there were tool calls, yield one final response carrying them all, with the usage,
        # which a consumer reads off the last chunk. Without tool calls the usage gets a closing
        # chunk of its own.
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
            self.base_url,
            {
                "response": full_output,
                "generated_tokens": token_count,
                "execution_time": time.time() - start,
                "tool_calls": [tc.model_dump() for tc in tool_calls],
                "usage": usage,
            },
        )
