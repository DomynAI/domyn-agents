<div align="center">

<picture>
  <source srcset="docs/static/img/logo-white.svg" media="(prefers-color-scheme: dark)">
  <source srcset="docs/static/img/logo.svg" media="(prefers-color-scheme: light)">
  <img src="docs/static/img/logo.svg" alt="Domyn" height="60">
</picture>

Build AI agents in Python or YAML, run them on any model, and work with them from the terminal.

[![CI](https://github.com/domynai/domyn-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/domynai/domyn-agents/actions/workflows/ci.yml)
[![Coverage](https://github.com/domynai/domyn-agents/raw/coverage-badge/coverage-badge.svg)](https://github.com/domynai/domyn-agents/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14%20%7C%203.15-blue)](https://github.com/domynai/domyn-agents/actions/workflows/ci.yml?query=branch%3Amain)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-domynai.github.io-blue)](https://domynai.github.io/domyn-agents/)
[![Built by Domyn](https://img.shields.io/badge/built%20by-Domyn-0000FF)](https://domyn.com)

[Documentation](https://domynai.github.io/domyn-agents/) · [Examples](examples/) · [Contributing](CONTRIBUTING.md)

</div>

---

Domyn Agents is two things:

- **A framework for building agents.** Define an agent in Python or in a YAML file, give it tools, and run it on any model, from OpenAI and Anthropic to a model you serve yourself with vLLM. Agents can hand work to each other, and a tool that should not run unsupervised waits for a person's approval.
- **An agent for your terminal.** `domyn run` opens a session with an agent that reads your files, edits them and runs commands, asking before anything that changes your machine. It runs the agents you build too.

## Installation

Domyn Agents is published on PyPI as part of `domyn` (the GitHub repository is `domyn-agents`).

For the terminal agent, install the `domyn` command. On macOS (Apple Silicon) and Linux, the installer brings its own Python:

```sh
curl -fsSL https://github.com/domynai/domyn-agents/releases/latest/download/install.sh | sh
```

or install it with [uv](https://docs.astral.sh/uv/), on Python 3.11 or later:

```sh
uv tool install "domyn[cli]"
```

To install the development version from the repository instead, use `uv tool install "domyn[cli] @ git+https://github.com/domynai/domyn-agents"`.

To build agents in a Python project (Python 3.11 or later), add it as a dependency:

```sh
uv add domyn
```

With pip, run `pip install "domyn[cli]"`. Agents import from `domyn.agents` (`from domyn.agents import Agent, Runner`). Optional extras add integrations: `tracing` (OpenTelemetry), `langchain`, `langgraph`, `slack`, `telegram` and `audio` (local speech to text).

## Use the terminal agent

Start it in the directory you want to work on:

```sh
domyn run
```

The first time, a setup wizard asks for a provider, a model and an API key, and saves them under `~/.domyn/`. Then you chat with the agent. It can read the files in this directory freely, and it asks for your approval before it writes a file or runs a command: once, for the rest of the conversation, in this project, or always.

Conversations are saved, and `domyn run --resume` picks one up where you left it. Type `/help` in a session to see its commands. See [Interactive sessions](docs/docs/cli/sessions.md).

## Build your own agent

### In YAML

An agent that keeps notes in the current directory, with no Python. Save it as `notes.yaml`:

```yaml
llm_providers:
  - name: openrouter
    type: open_router
    model_name: nvidia/nemotron-3-ultra-550b-a55b:free
    api_key: ${OPENROUTER_API_KEY}

tool_bundles:
  - name: files
    bundles: [files]

agents:
  - name: NoteKeeper
    description: Keeps notes as markdown files.
    instruction: >
      You are NoteKeeper, a useful assistant. You keep the user's notes as markdown files in the current directory,
      one file per topic. Read a file before changing it.
    llm_provider: openrouter
    tool_bundles: [files]
```

and start a session with it:

```sh
export OPENROUTER_API_KEY=...
domyn run notes.yaml
```

The `files` bundle gives the agent the built-in file tools. It reads without asking, and every write waits for your approval. `${OPENROUTER_API_KEY}` is read from the environment, or from a `.env` file in the current directory. See [YAML configuration](docs/docs/guides/configuration.md).

### In Python

A support agent with two tools. Any function is a tool: its docstring is the description the model reads, and its type annotations are the schema of its arguments. The refund changes something real, so it waits for a person's approval. Save it as `support.py`:

```python
from domyn.agents import Agent
from domyn.agents import ApprovalType
from domyn.agents import OpenRouterProvider
from domyn.agents import FunctionTool


def get_order_status(order_id: str) -> str:
    """Look up the status of an order."""
    return f"Order {order_id} was delivered on 2 October."


def refund_order(order_id: str, amount: float) -> str:
    """Refund an order, in full or in part."""
    return f"Refunded {amount:.2f} EUR on order {order_id}."


agent = Agent(
    name="SupportAgent",
    description="Answers questions about orders",
    instruction="You are SupportAgent, a helpful assistant. You help customers with their orders. Check an order's status before refunding it.",
    llm_provider=OpenRouterProvider(model_name="nvidia/nemotron-3-ultra-550b-a55b:free"),
    tools=[
        FunctionTool(func=get_order_status),
        FunctionTool(func=refund_order, approval_type=ApprovalType.USER_CONFIRM),
    ],
)
```

With `OPENROUTER_API_KEY` set, `domyn run support.py` opens a session with it. Ask for a refund, and the session shows you the call and its arguments before the tool runs. See [Tools](docs/docs/guides/tools.md) and [Tool approvals](docs/docs/guides/approvals.md).

### In your application

The `Runner` executes an agent and yields an event for every step: model calls, tool calls, handoffs and the answer. The same agent can then drive a web service, a chat channel or a batch job:

```python
import asyncio

from domyn.agents import BaseEvent, ExecutionEventType, Part
from domyn.agents import Runner

from support import agent


async def main() -> None:
    question = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="Where is order 1234?")],
    )
    async for event in Runner().run(
        application_name="support",
        user_id="user-1",
        conversation_id="conv-1",
        root=agent,
        user_input=question,
    ):
        if event.event_type == ExecutionEventType.RESPONSE:
            print("".join(part.text or "" for part in event.content or []))
        elif event.event_type == ExecutionEventType.ERROR_OCCURRED:
            print("Error:", event.error_message)


asyncio.run(main())
```

When a tool needs approval, the turn ends with a `TOOL_NEEDS_FEEDBACK` event, and your application answers it in the next turn. See [Events and streaming](docs/docs/guides/events-and-streaming.md).

## Features

- **[Any model](docs/docs/guides/llm-providers.md)**: vLLM, OpenAI, Anthropic, Cerebras, OpenRouter or any OpenAI-compatible endpoint, with native tool calling or a prompt-based protocol for endpoints without it.
- **[Tools](docs/docs/guides/tools.md)**: Python functions, streaming tools, remote MCP servers, and built-in file and shell tools confined to a workspace.
- **[Multi-agent](docs/docs/guides/orchestration.md)**: agents hand work to each other, with control over what each one sees and whom it can call. [Remote agents](docs/docs/guides/remote-agents.md) take part like local ones.
- **[Human in the loop](docs/docs/guides/approvals.md)**: a tool can wait for approval, remembered for one call, a conversation or longer, and rules or hooks can decide some calls on their own.
- **[Skills](docs/docs/guides/skills.md)**: teach agents reusable know-how with `SKILL.md` files, loaded only when a task needs them.
- **[Standing instructions](docs/docs/guides/standing-instructions.md)**: `SOUL.md` and `AGENTS.md` files that shape every conversation.
- **[Long conversations](docs/docs/guides/context-compaction.md)**: the conversation is compacted before it outgrows the model's context window.
- **[Hooks](docs/docs/guides/hooks.md)**: run your code before and after agents and tools, to log, validate or change what they do.

## Documentation

The [documentation](https://domynai.github.io/domyn-agents/) has guides for the terminal agent and for building agents, and a reference for the [CLI](docs/docs/reference/cli.md), the [YAML schema](docs/docs/reference/yaml-schema.md) and the [event types](docs/docs/reference/events.md).

## Contributing

Contributions are welcome. Search the issues first; anything beyond a small fix starts from an issue a maintainer has accepted. [CONTRIBUTING.md](CONTRIBUTING.md) explains how to propose a change, set up the environment, run the tests and the checks, and build the documentation site. If you work with a coding agent, have it follow [AGENTS.md](AGENTS.md). Everyone taking part is expected to follow the [code of conduct](CODE_OF_CONDUCT.md).

To report a security vulnerability, see [SECURITY.md](SECURITY.md) rather than opening an issue.

## License

Domyn Agents is released under the [Apache License 2.0](LICENSE).
