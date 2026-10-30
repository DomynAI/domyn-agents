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

Domyn Agents is a powerful and extensible agent harness. It works with any model, from Domyn's, to other OSS models running in vLLM, or closed model providers. Domyn Agents is provided with an extensive cli for setup and configuration.

Domyn Agents also allows to build your own agents. Define them with the Python library or a YAML config, give them tools, let them hand work to each other, and require a person's approval for any tool that should not run unsupervised. `domyn run` runs the agents you build too.

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

## Getting started

Start it in the directory you want to work on:

```sh
domyn run
```

The first time, a setup wizard asks for a provider, a model and an API key, and saves them under `~/.domyn/`. Then you chat with the agent. It can read the files in this directory freely, and it asks for your approval before it writes a file or runs a command: once, for the rest of the conversation, in this project, or always.

Conversations are saved, and `domyn run --resume` picks one up where you left it. Type `/help` in a session to see its commands. See [Interactive sessions](docs/docs/cli/sessions.md).

To build your own agents, in Python or in YAML, see the [documentation](https://domynai.github.io/domyn-agents/) or the [examples](examples/README.md).

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
