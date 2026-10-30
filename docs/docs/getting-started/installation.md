---
title: Installation
sidebar_label: Installation
description: "Install Domyn Agents and set up your API keys"
---

# Installation

Domyn Agents is published on PyPI as `domyn`; the GitHub repository is `domyn-agents`. The library requires **Python 3.11 or later**. The CLI can also be installed without Python, as a standalone build.

## Install the CLI

### Standalone installer

The installer downloads a build of the `domyn` command that carries its own Python, so it needs neither Python nor uv on your machine:

```bash
curl -fsSL https://github.com/domynai/domyn-agents/releases/latest/download/install.sh | sh
```

Builds exist for macOS on Apple Silicon and for Linux on x86-64 and ARM64 with glibc 2.33 or later (Ubuntu 22.04, Debian 12, RHEL 9 or newer). Intel Macs and musl systems such as Alpine are not supported.

The installer checks the download against its published checksum, puts the runtime in `~/.local/share/domyn` and the `domyn` command in `~/.local/bin`, and adds that directory to your `PATH` in your shell profile if it is not there already. Open a new terminal afterwards, or reload the profile it names. Running it again replaces the installed version.

| Variable | Default | Description |
|---|---|---|
| `DOMYN_VERSION` | the latest release | The release to install, for example `1.5.0`. |
| `DOMYN_HOME` | `~/.local/share/domyn` | Where the runtime goes. |
| `DOMYN_BIN_DIR` | `~/.local/bin` | Where the `domyn` command goes. |

Set them for the `sh` that runs the script: `curl -fsSL …/install.sh | DOMYN_VERSION=1.5.0 sh`.

To uninstall, delete the two paths: `rm -rf ~/.local/share/domyn ~/.local/bin/domyn`. Your configuration and conversations in `~/.domyn` stay where they are.

The standalone build is the CLI only, with the `audio` extra included, so [`/voice`](../cli/sessions.md#talking-instead-of-typing) works without installing anything else. To use Domyn Agents as a library, [add it to a project](#add-it-to-a-project).

### With uv

To install the `domyn` command into a Python environment of its own, install Domyn Agents as a [uv](https://docs.astral.sh/uv/) tool with the `cli` extra:

```bash
uv tool install "domyn[cli]"
```

To install the development version from GitHub instead:

```bash
uv tool install "domyn[cli] @ git+https://github.com/domynai/domyn-agents"
```

Whichever way you installed it, check that it works:

```bash
domyn --version
```

## Add it to a project

To use Domyn Agents as a library in your own code, add it to your project's dependencies:

```bash
uv add domyn
```

With pip:

```bash
pip install "domyn[cli]"
```

To use the development version from GitHub, add `"domyn @ git+https://github.com/domynai/domyn-agents"` instead, or clone the repository and run `pip install "./domyn-agents[cli]"`.

## Optional extras

The base package covers agents, tools, every built-in LLM provider and the CLI. Extras add the dependencies of optional integrations:

| Extra | What it adds |
|---|---|
| `cli` | The line editor of the interactive CLI: completion, history and multi-line messages. `domyn run` installs it into its environment when it is missing. |
| `tracing` | OpenTelemetry tracing |
| `langgraph` | Serving a LangGraph graph through Domyn Agents, for example with `domyn connect` |
| `langchain` | LangChain and `langchain-openai`, to build such a graph with LangChain models |
| `slack` | Slack channel |
| `telegram` | Telegram channel |
| `audio` | [Transcription](../guides/transcription.md) on your machine, for voice notes and voice input (Python 3.11 to 3.14) |

Name several extras in one install, for example `domyn[cli,tracing]`.

### The Domyn platform SDK

`domyn connect`, the platform's tools and the `domyn` LLM provider need `domyn-platform-sdk`, version 0.37.0 or later. It is not an extra: it is distributed with the Domyn platform, not on a public index, so install it next to Domyn Agents from where your platform provides it:

```bash
pip install "domyn-platform-sdk>=0.37.0"
```

Without it, or with an older one, those commands stop and give this requirement.

## API keys

Each provider reads its API key from an environment variable:

| Provider | Variable |
|---|---|
| Domyn platform | `DOMYN_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Cerebras | `CEREBRAS_API_KEY` |
| vLLM | `VLLM_API_KEY` |
| Any other OpenAI-compatible endpoint | `LLM_API_KEY` |

You can provide the key in three ways. When the same variable is set in more than one place, the first one in this list wins:

1. **Your shell environment**, for example `export OPENAI_API_KEY=...`.
2. **A `.env` file** in the directory where you run `domyn`, or in one of its parent directories. Every `domyn` command loads it automatically, so the key does not need to live in your shell profile. Keep this file out of version control.
3. **`~/.domyn/credentials`**, where the setup wizard saves a key you paste in. The file is readable only by you, and the key never goes into `~/.domyn/config.yaml`.

:::tip
Inside a YAML configuration file, refer to a key as `${OPENAI_API_KEY}` instead of writing it in the file. It is replaced with the variable's value when the file loads. See [YAML configuration](../guides/configuration.md).
:::

## Next steps

- [Quickstart: CLI](quickstart-cli.md): chat with an agent from the terminal.
- [Quickstart: library](quickstart-library.md): build your own agent in Python.
