---
title: LLM providers
sidebar_label: LLM providers
description: "Connect an agent to the models of a Domyn platform space, OpenAI, Anthropic, vLLM, Cerebras, OpenRouter or any OpenAI-compatible endpoint, or to a model of your own"
---

# LLM providers

An LLM provider connects an agent to a model. Every `Agent` has one (`llm_provider`), and agents in a team can each use a different one. A provider sends the conversation and the agent's tools to the model's API and returns its reply: text, tool calls, or both. Every built-in provider supports native tool calling and streaming, and talks to its API directly over HTTP, without vendor SDKs.

## Built-in providers

| `type` | Class | API | Default URL | Key variable |
|---|---|---|---|---|
| `domyn` | `DomynProvider` | The models of a Domyn platform space, through the platform | the platform's address, `url` is required | `DOMYN_API_KEY` |
| `openai` | `OpenAIProvider` | OpenAI Responses API | `https://api.openai.com/v1/responses` | `OPENAI_API_KEY` |
| `anthropic` | `AnthropicProvider` | Anthropic Messages API | `https://api.anthropic.com/v1/messages` | `ANTHROPIC_API_KEY` |
| `vllm` | `VLLMProvider` | OpenAI Chat Completions, as served by vLLM | none, `url` is required | none |
| `cerebras` | `CerebrasProvider` | Cerebras Chat Completions | `https://api.cerebras.ai/v1/chat/completions` | `CEREBRAS_API_KEY` |
| `open_router` | `OpenRouterProvider` | OpenRouter Chat Completions | `https://openrouter.ai/api/v1/chat/completions` | `OPENROUTER_API_KEY` |
| `custom` | `CustomLLMProvider` | Any OpenAI-compatible Chat Completions endpoint | none, `url` is required | none |

The key variable is read when `api_key` is not given. For `openai` and `cerebras` it is read once, when `domyn.agents.llm` is first imported, so set it before that import, or pass `api_key` explicitly. `cerebras` refuses to start without a key; `openai` starts and sends no key.

## Declaring a provider

In Python, create the provider and pass it to the agent:

```python
import os

from domyn.agents import Agent
from domyn.agents import AnthropicProvider

llm = AnthropicProvider(
    model_name="claude-sonnet-4-5",
    api_key=os.environ["ANTHROPIC_API_KEY"],
)

agent = Agent(name="Assistant", description="...", instruction="...", llm_provider=llm)
```

The classes are importable from `domyn.agents.llm`.

In a [YAML configuration](configuration.md), providers are declared under `llm_providers` and referenced by `name` from each agent. Use `${VAR}` for keys, so they stay out of the file:

```yaml
llm_providers:
  - name: claude
    type: anthropic
    model_name: claude-sonnet-4-5
    api_key: ${ANTHROPIC_API_KEY}

  - name: local
    type: vllm
    model_name: Qwen/Qwen3-8B
    url: http://localhost:8000/v1/chat/completions
    context_window: 32768
    generation_params:
      temperature: 0.2

agents:
  - name: Assistant
    llm_provider: claude
    ...
```

A provider without a `name` is named after its `model_name`.

The [interactive CLI](../cli/configuration.md) writes its provider to `~/.domyn/config.yaml` in this same form, from the answers to `domyn setup` or `/model`.

## Options

### All providers

| Option | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | the `model_name` | Name the provider is referenced by in YAML. |
| `model_name` | `str` | required | The model, as the API names it. |
| `context_window` | `int` | none | The model's context window, in tokens. When it is not set, it is asked of the provider before each turn (the `/models` listing of vLLM, of OpenRouter or of a gateway that publishes it, or Anthropic's model API), or read from a catalogue shipped with the package for `openai` and `cerebras`; answers are kept for five minutes. A request the provider rejects as too long states it too, and lowers a window set too high. [Context compaction](context-compaction.md) compacts at a threshold only when a window is known; a rejected request is compacted and retried either way. |

### `openai`, `anthropic`, `vllm`, `cerebras`, `open_router`

| Option | Type | Default | Description |
|---|---|---|---|
| `url` | `str` | see [Built-in providers](#built-in-providers) | API endpoint. |
| `api_key` | `str` | the key variable | Sent as `Authorization: Bearer <key>`, or as `x-api-key` for `anthropic`. For `vllm` it is sent only when set. |
| `generation_params` | `dict` | per provider, below | Request fields added to every call, such as `temperature` or `max_tokens`. The values you give are merged over the defaults, so you only list what you change. |

Default `generation_params`:

| `type` | Default |
|---|---|
| `openai` | `reasoning_effort: medium`, `service_tier: auto`, `store: false`, `temperature: 1`, `max_output_tokens: 4000` |
| `anthropic` | `max_tokens: 4096` |
| `vllm` | `temperature: 0.7`, `max_tokens: 2000` |
| `cerebras` | `temperature: 0.7`, `max_tokens: 8000` |
| `open_router` | `temperature: 0.7`, `max_tokens: 8000` |

`openai` sends `temperature` only for `gpt-4o`, `gpt-4o-mini`, `gpt-5`, `gpt-5-mini` and `gpt-5-nano`, and sends `reasoning_effort`, as the Responses API's `reasoning.effort`, only for `o1`, `o1-mini`, `o3-mini`, `gpt-5`, `gpt-5-mini` and `gpt-5-nano`. For any other model, both are left out of the request.

### Provider-specific options

| Provider | Option | Type | Default | Description |
|---|---|---|---|---|
| `vllm` | `timeout` | `float` | `300.0` | Seconds to wait for the server, for slow or large models. |
| `cerebras`, `open_router` | `max_retries` | `int` | `3` | Attempts when the API answers with an empty body. |
| `open_router` | `reasoning_supported` | `bool` | `False` | Sends `reasoning: {effort: medium}`. Set it only for models that accept the parameter: others reject the request. |

### `domyn`

The models a space of the Domyn platform offers. You authenticate with your own platform API key and name the space; the platform holds the credentials of the model's provider and forwards the request. It needs the platform SDK next to Domyn Agents, version 0.37.0 or later (see [Installation](../getting-started/installation.md#the-domyn-platform-sdk)): without it, or with an older one, the provider stops and gives the requirement.

| Option | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | required | The model, named as the space's configuration names it, not as its provider does. |
| `url` | `str` | `DOMYN_BASE_URL` | The platform's address, with or without `https://`. Also read as `base_url`. |
| `api_key` | `str` | `DOMYN_API_KEY` | Your platform API key. |
| `space_id` | `str` | `DOMYN_SPACE_ID` | The space whose configuration holds the model. |
| `configuration_id` | `str` | `DOMYN_CONFIGURATION_ID`, else the space's last released configuration | The configuration to take the model from. |
| `generation_params`, `body` | `dict` | `{}` | Request fields added to every call, such as `temperature` or `max_tokens`. |

`domyn setup` lists the space's models when it can reach the platform, so you can choose the name instead of typing it.

With the three variables set, a provider needs only the model:

```yaml
llm_providers:
  - name: platform
    type: domyn
    model_name: qwen3-32b
    context_window: 32768
```

### `custom`

For any endpoint that speaks OpenAI Chat Completions, with full control over the request. It has no `api_key` and no `generation_params`: authentication goes in `headers`, and request fields in `body`.

| Option | Type | Default | Description |
|---|---|---|---|
| `url` | `str` | required | The full chat completions URL. |
| `headers` | `dict[str, str]` | `{}` | HTTP headers sent with every request, for example `Authorization`. |
| `body` | `dict` | `{}` | Fields added to every request body, for example `temperature` or `max_tokens`. |

```yaml
llm_providers:
  - name: gateway
    type: custom
    model_name: Qwen/Qwen3-32B
    url: https://llm.example.com/v1/chat/completions
    headers:
      Authorization: Bearer ${GATEWAY_API_KEY}
    body:
      temperature: 0.7
      max_tokens: 512
```

## Your own provider

A provider is a subclass of `BaseLLMProvider` that implements `generate` and `generate_streaming`:

```python
import time
from collections.abc import AsyncGenerator
from typing import Any

from domyn.agents.core.decorators import llm_provider
from domyn.agents.llm import BaseLLMProvider, LLMMessage, LLMResponse


@llm_provider(name="echo", model_name="echo-1", generation_params={"prefix": "echo:"})
class EchoProvider(BaseLLMProvider):
    type: str = "echo"
    supported_structured_outputs: None = None
    generation_params: dict[str, Any] = {}

    async def generate(
        self, messages: list[LLMMessage], structured_outputs=None, tools=None, tool_choice=None, **kwargs: Any
    ) -> LLMResponse:
        start = time.monotonic()
        text = f"{self.generation_params['prefix']} {messages[-1].content}"
        return LLMResponse(
            content=text,
            generated_tokens=len(text.split()),
            execution_time=time.monotonic() - start,
        )

    async def generate_streaming(
        self, messages: list[LLMMessage], **kwargs: Any
    ) -> AsyncGenerator[LLMResponse, None]:
        yield await self.generate(messages, **kwargs)
```

What the methods receive and return:

- `messages` is the conversation, as `LLMMessage` objects with a `role` (`system`, `user`, `assistant`, `tool`), `content`, and, for tool use, `tool_calls` and `tool_call_id`.
- `tools`, when the agent has any, is a list of tool definitions in the OpenAI Chat Completions format: `{"type": "function", "function": {"name", "description", "parameters"}}`.
- The reply is an `LLMResponse`: `content`, `generated_tokens` and `execution_time` (seconds) are required. A tool call is returned in `tool_calls`, as `LLMToolCall(id=..., function=ToolCallFunction(name=..., arguments=<JSON string>))`.
- `usage` carries the provider's token counts. [Context compaction](context-compaction.md) measures the next request from the prompt count (`prompt_tokens`, or `input_tokens` with any cache counts), so return it when the API gives one. `reasoning_content` becomes the thought of a `tool_use_thought` agent when a call carries none (see [Planner strategies](planner-strategies.md)).
- `fetch_context_window()` returns the model's window, or `None`. By default it reads the `/models` listing beside an OpenAI-style `url`; override it for an API that says the window elsewhere.
- `generate_streaming` yields the reply in pieces, each an `LLMResponse` whose `content` is the next piece. It is used when the agent has `stream` enabled.

Without the decorator, create an instance and pass it to the agent, as with the built-in providers. `@llm_provider` registers one instance instead, created on first use with the decorator's arguments, so that YAML can use it by name, either directly or through a `reference` that changes some of its options:

```yaml
llm_providers:
  - name: loud_echo
    reference: echo
    config_params:
      generation_params:
        prefix: "ECHO:"

agents:
  - name: Assistant
    llm_provider: loud_echo     # or: echo
    ...
```

The module that defines the provider must be imported before the configuration is loaded. The decorator replaces the class with a function that returns that one instance: once decorated, `EchoProvider(...)` returns the registered provider rather than a new one, and `EchoProvider` cannot be used with `isinstance`. `config_params` sets any option except `type`, `model_name`, `name`, `url` and `reference`, and is merged into options that are dictionaries, such as `generation_params`.

## Next steps

- [Planner strategies](planner-strategies.md): how an agent uses its provider on each step.
- [Context compaction](context-compaction.md): what `context_window` drives.
- [YAML configuration](configuration.md): the rest of a YAML configuration.
