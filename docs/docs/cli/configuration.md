---
title: Configuring the CLI
sidebar_label: Configuring the CLI
description: "The CLI's configuration file, the settings you are most likely to change, and what lives in ~/.domyn"
---

# Configuring the CLI

The setup wizard saves the session's configuration to `~/.domyn/config.yaml`. `domyn run` uses it whatever folder it starts in: a `config.yaml` in a `./.domyn` folder there is not read. To give a folder a configuration of its own, make it a [project](#projects), or pass the file to `domyn run`. Run `domyn setup` to go through the wizard again, use `/model` to change only the model, and `/config` to see what a session runs on.

## The setup wizard

`domyn setup` asks, in this order:

| Part | What it asks |
|---|---|
| Model | The provider, the model, the URL, the API key, the maximum tokens of a reply and the context window. For `domyn`, also the space id and the configuration id, and the model is picked from the models of the space. Then, without asking, it sends the model a small image to find out whether it reads images (see below). See the [CLI quickstart](../getting-started/quickstart-cli.md). |
| Instructions | The agent's name (empty: `Assistant`, at most 30 characters), then a folder or file holding your `SOUL.md` / `AGENTS.md`, then whether to also load them from `./.domyn` and `./.agents` in the folder the session starts in, then extra instructions for the agent, the start of its system prompt (empty: none). See [Standing instructions](../guides/standing-instructions.md). |
| Tools | A checklist of the [built-in tools](tools-and-approvals.md): `files`, `shell` and `browser`, all ticked the first time. Untick them all for an agent that only talks. |
| Skills | A checklist of the built-in skills, by group, all ticked the first time; then a folder of your own skills; then whether to load skills from `./.domyn/skills` and `./.agents/skills` in the folder the session starts in. See [Skills](../guides/skills.md). |

Run again over an existing configuration, the wizard first shows each part as it is and asks whether to keep it; Enter keeps it and skips its questions. In a part you do not keep, each field opens with its saved value: Enter keeps it, and erasing it gives back the default the question names, such as the provider's own URL or the last released configuration. Keeping the skills does not reinstall the built-in ones, so a built-in skill changed by an upgrade arrives only when you do not keep them.

On a terminal, Ctrl-B goes back to the previous question, which opens on the answer you gave it. Ctrl-C stops the wizard and saves nothing.

The image check runs on every `domyn setup` and every `/model`, even when you keep the model, and its line is printed under the model's settings: `Vision: enabled` when the model took the image, `Vision: not supported for this model (if not expected doublecheck the parameters)` otherwise. Enabled, the agent gets [`read_image`](tools-and-approvals.md) and you can [attach images](sessions.md#attaching-images) to a message. The answer is saved as `vision` on the provider, and it **replaces whatever the file said**: a `vision: true` written by hand is removed if the check fails. Any failure counts as no, a wrong API key or an endpoint that is down included, so if a model that reads images comes out as not supported, check the URL and the key and run `/model` again.

Renaming the agent changes which past messages it recognizes as its own: in a conversation started under the old name, resumed with `--resume` or still open, the model no longer sees its earlier replies and tool results, only your messages.

## The configuration file

The file is an ordinary [YAML configuration](../guides/configuration.md), so anything you can declare there you can declare here, such as more agents, MCP servers or hooks. The wizard writes something like this:

```yaml
version: '1.0'
entrypoint: Assistant
llm_providers:
- name: default_provider
  type: openai
  model_name: gpt-4o
  generation_params:
    max_tokens: 16000
planner_strategies:
- name: tool_use_thought
  type: tool_use_thought
  compactor:
    strategy: two_phase
agents:
- name: Assistant
  description: A general-purpose assistant
  instruction: You are a helpful assistant. Be concise and accurate.
  llm_provider: default_provider
  planner_strategy: tool_use_thought
  stream: true
  skills:
    source:
      type: local_dir
      paths:
      - .domyn/skills
      - .agents/skills
      - /home/me/.domyn/builtin-skills/productivity/brainstorming
  standing_instructions:
    source:
      type: local_dir
      paths:
      - /home/me/notes/SOUL.md
      - .domyn
      - .agents
  tool_bundles:
  - cli_tools
tool_bundles:
- name: cli_tools
  bundles:
  - files
  - shell
  - browser
  workspace: null
  backend: auto
  isolate_environment: true
  allowed_env: []
```

The relative paths are the folders the wizard loads from the directory the session starts in, when you answered yes to them; the absolute ones are the files and skills it found or installed.

The settings you are most likely to change:

| Setting | What it does |
|---|---|
| `context_window` | The model's context window, in tokens. Left out (`auto` in the wizard), the session asks the provider at start and assumes 150,000 when it cannot answer; `/context` shows the window it got. Set it when the provider reports none, or a wrong one: the session compacts the conversation based on it. |
| `send_reasoning` | On the provider, `true` unless written. `false` stops the model's reasoning being sent back to it, for a model that ignores it (a vLLM chat template that does not read `reasoning_content`): it then gets the short thoughts instead. Kept when `domyn setup` or `/model` rewrites the file. |
| `vision` | On the provider, `true` when the model reads images, which gives the agent `read_image` with the `files` tools. Written by `domyn setup` and `/model` from the image they send the model, and replaced by that check every time they run: set it by hand only for a model you changed by hand, and expect the next `domyn setup` or `/model` to decide it again. |
| `stream` | `true` prints the answer as it is generated, `false` when it is complete. |
| `bundles` | The [built-in tools](tools-and-approvals.md): `files`, `shell` and `browser`. Remove one to take those tools away, or leave the list empty for an agent that only talks. |
| `skills` | The [skills](../guides/skills.md) the agent loads. Only what is listed here is loaded. |
| `standing_instructions` | The [standing instructions](../guides/standing-instructions.md) added to the system prompt on every turn. |
| `workspace` | The directory the file tools are confined to. `null` means the directory the session starts from. |
| `backend` | Where the tools run. `auto` (default) uses a [Docker container](tools-and-approvals.md) when Docker is running, and your machine, with a warning, when it is not. `docker` always uses the container and refuses to start without Docker. `local` runs everything on your machine. **A container is not your machine**: the config is read-only there, and only mounted files exist; see [the differences](tools-and-approvals.md#running-the-tools-in-a-docker-container). The container's options go in a `docker` block: `image`, `allowed_network` (`public`, `none` or a list of hosts; see [what commands can reach](tools-and-approvals.md#what-commands-can-reach-on-the-network)), `memory`, `cpus` and `docker_binary`. |
| `isolate_environment` | `false` lets commands use the environment you run `domyn` from instead of a virtual environment per conversation. |
| `allowed_env` | Names of environment variables that commands may see. |
| `compactor.strategy` | How the conversation is shortened: `two_phase` (default), `prune_only` (never calls the model) or `noop` (never shortens). |

### The browser

With `browser` in `bundles`, its options go in a `browser` block on the same entry. Every key is optional:

```yaml
tool_bundles:
- name: cli_tools
  bundles:
  - files
  - shell
  - browser
  browser:
    allowed_domains:
    - github.com
    - docs.python.org
    allow_private_urls: true
```

| Option | What it does |
|---|---|
| `allowed_domains` | The only sites the browser may contact, the requests a page makes included. Empty (the default) means any public site. |
| `allow_private_urls` | `true` lets the browser open addresses on your machine or your network, such as a dev server on `localhost`. Default `false`. |
| `engine` | `chrome` (default) or `lightpanda`, a much lighter browser that renders nothing, installed separately. |
| `executable_path` | The browser to launch, for Lightpanda when it is not on your `PATH`. |
| `cdp_url` | Drive a browser that is already running, over the Chrome DevTools Protocol, instead of launching one. It cannot be combined with `allowed_domains`. |
| `headed` | `true` shows the browser window. |
| `idle_timeout_s` | Seconds without a browser command before the browser closes itself. Default 300; `0` keeps it open. |
| `command_timeout_s` | The longest one browser command may take, page load included. Default 60. |

An unknown key in this block is refused rather than ignored, so a misspelt `allowed_domains` cannot leave the browser unfenced.

API keys are never written to this file. See [Installation](../getting-started/installation.md) for where they are read from.

## Projects

A project is a separate configuration, kept side by side with the default one: its own model and API key, tools, skills, standing instructions, approvals and conversation history. Use one to switch between two setups, say one for work and one for personal use, or to share a setup with someone else.

| Command | What it does |
|---|---|
| `domyn setup --project work` | Creates the project `work` in `~/.domyn/projects/work` with the setup wizard, or changes it if it exists. |
| `domyn run --project work` | Starts a session with it. The banner shows the project's name. |
| `domyn export <folder>` | Saves the configuration, with the skills and instruction files it uses, into a new folder: a project that works on any machine. With `--project work`, it exports that project. `/export <folder>` does the same from inside a session. |
| `domyn setup --import <folder>` | Creates a new project from an exported folder, named after the folder unless `--project` gives another name. The wizard shows each exported setting and asks whether to keep it. |

`--project` also takes the path of a project folder anywhere on disk: anything containing `/` or starting with `.` or `~`, such as `--project ./shared`.

An export also holds a hidden `.info` file saying what made it: the version of domyn-agents, the date, the Python version and the platform, and the version of `domyn-platform-sdk` when it is installed. Nothing in it names your machine or you. `domyn setup --import` compares that version with its own and, when they differ or the export does not say, prints a warning before the wizard. The import goes on either way, but the exported configuration and built-in skills are as that version wrote them.

When you choose not to keep the exported skills during an import, the wizard lists them, ticked, and below them, unticked, the built-in skills the export does not include, so you can add any of them. The built-ins you tick are installed into the project's `builtin-skills` folder; the exported ones stay as they came, edits included. To replace them with this version's, run `domyn setup --project <name>` afterwards.

A project reads nothing from `~/.domyn` except API keys and, when its [memory](#memory) includes the global one, `~/.domyn/memory.md`. A key is read from the project's own `credentials` file first, then from `~/.domyn/credentials`, so a key saved once serves every project, and a project can use a different one by setting its own. Its conversations and approvals are its own. An export names nothing outside its own folder: the files the configuration reads (skills, instruction files, the code of tools and hooks) are copied in, and a path that cannot be, such as a tool bundle's workspace, refuses the export, saying which. `domyn setup --import` refuses an export that names anything outside itself, so an edited export cannot make your machine read a file of yours. A project used by its path (`--project ./folder`) is checked every time: if it names files outside its folder, domyn lists them and asks before using it, and `domyn run` also shows the code of its tools and hooks, with what each imports, and asks before running it. Without a terminal to ask on, such a project is refused. A project used by name (`--project <name>`, a folder under `~/.domyn/projects` that setup made or imported) is trusted and used as it is. An export never includes API keys, approvals, conversation history or what the agent remembers. The [memory](#memory) setting is kept: a project whose memory was on still has it on after `domyn setup --import`, which shows the exported choice and lets you change it. A project memory starts empty there, and the global memory, written to or only read, is the global memory of whoever imports it (`~/.domyn/memory.md`). When the configuration needs a key, the export says which environment variable to set.

## Voice

[`/voice`](sessions.md#talking-instead-of-typing) transcribes on your machine by default. A `voice` block, at the top level of the file beside the agent, changes how: its `transcriber` takes the same fields as any [transcriber](../guides/transcription.md).

```yaml
voice:
  transcriber:
    type: local
    language: it       # detected when left out; Parakeet always detects it
```

To transcribe on a server instead, such as a speech model served by vLLM:

```yaml
voice:
  transcriber:
    type: openai
    url: http://localhost:8000/v1/audio/transcriptions
    model: openai/whisper-large-v3-turbo
    api_key_env: ASR_API_KEY
```

The setting is read when voice is turned on, so a change takes effect at the next `/voice` in a new session.

## Memory

The agent can remember facts about you between conversations: your name, how you like things done, what you are working on. It is off until you turn it on in `domyn setup`.

There are two memories, each a `memory.md` file: the **global** one, `~/.domyn/memory.md`, shared by every configuration that uses it, and a **project**'s own, in the project's folder. A [project](#projects) always reads the global memory too, so setup only asks where the agent learns (writes what it remembers):

| Choice | Learns into | What follows |
|---|---|---|
| **No** | nothing | The agent remembers nothing between conversations, even when your default configuration has a memory. |
| **Yes, and inform other projects too (writes to global memory)** | the global memory | What it learns here is known in other projects too. The project's own memory is only read. |
| **Yes, scoped only to this project** | the project's memory | It knows what was learned in your other configurations, but what it learns here stays in the project. The global memory is only read. |

Outside a project the choices are **No** and **Yes (global memory - shared with all projects)**, which learns into the global memory. Once you choose, setup prints the file it learns into and, in a project, the one it only reads.

A config written by hand can still have a memory that reads only the global or only the project file; setup keeps it as written when you press Enter on that question.

A memory that is only read is never changed by the agent: a fact found there is not stored again, `/memory` marks its entries `(read only)`, and `/memory forget` and `/memory clear` leave them alone, saying so. To change one, edit its file, or run `/memory` in a configuration that learns into it.

The memory is a markdown file you can open and edit; one bullet is one fact. The agent decides what to keep, and says so as it does (`remembered: …`).

With a [decision model](../guides/decision-models.md) configured, setup also asks **who decides what to remember**:

- **the agent**: the agent's model stores a fact when it judges one worth keeping, with its `remember` tool (`execution_mode: agentic`).
- **the decision model**: after every turn the decision model says whether the turn told it something new about you; only on a yes is the agent's model asked to write the fact (`execution_mode: automatic`). It runs in the background once the answer is out, so the turn does not wait for it, and nothing is printed for it: `/memory` lists what it stored.

Without a decision model the question is not asked, and the agent decides. Removing the decision model in a later setup gives the decision back to the agent, and setup says so. It reads the memory at the start of each conversation, and can look back through it page by page. See [Conversations and memory](../guides/conversations-and-memory.md) for how it works, and `/memory` under [Commands](sessions.md#commands) to see or change it.

In `config.yaml`, memory is a `memory_manager` on the agent:

```yaml
agents:
  - name: Assistant
    memory_manager:
      type: filesystem
      file_path: /Users/you/.domyn/memory.md
      execution_mode: agentic
```

`file_path` is the memory it learns into; `read_only_sources` lists extra memory files it only reads, which are never part of the memory it writes: nothing is stored, forgotten or cleared in them. A project set to `project + global, writes to project`:

```yaml
    memory_manager:
      type: filesystem
      file_path: ${DOMYN_PROJECT_DIR}/memory.md
      read_only_sources:
        - /Users/you/.domyn/memory.md
      execution_mode: agentic
```

## What lives in `~/.domyn`

On Windows, `~/.domyn` is `%USERPROFILE%\.domyn`.

| Path | What it holds |
|---|---|
| `config.yaml` | The session's configuration |
| `credentials` | API keys pasted into the wizard, readable only by you |
| `conversations.db` | Every saved conversation, whatever folder you run `domyn` from |
| `conversations/` | One folder per conversation that ran a command, with its Python environment and `output/` files |
| `processes/` | The output of commands left running in the background, one folder per command |
| `browser/` | Text of long pages the [browser tools](#the-browser) saved to read in parts |
| `memory.md` | The global memory: what the agent remembers about you, for every configuration whose [memory](#memory) reads it |
| `approvals.yaml` | Approvals that last "always" |
| `state.yaml` | Settings the CLI remembers between sessions: [`/verbose`](sessions.md#commands) and [`/auto`](tools-and-approvals.md). A project has its own, so a setting in one does not reach another. Not exported |
| `projects/` | Your [projects](#projects), and approvals that last for one project |
| `builtin-skills/` | The built-in [skills](../guides/skills.md) you chose in `domyn setup`, copied from the package. Every `domyn setup` deletes and rewrites it, so keep your own skills elsewhere, such as `./.domyn/skills` |
| `models/` | The speech model [`/voice`](sessions.md#talking-instead-of-typing) downloads the first time |

A [project](#projects) has the same layout in its own folder, and what a session leaves behind goes there instead of `~/.domyn`. Nothing is written to the folder you start `domyn` in. To remove old conversations and their environments, use [`domyn clean`](../reference/cli.md#domyn-clean).

`~/.domyn` is not a folder for [standing instructions](../guides/standing-instructions.md): a `SOUL.md` or `AGENTS.md` there is not read, unless you give its path to `domyn setup`.

## Next steps

- [YAML configuration](../guides/configuration.md): everything a configuration file can declare.
- [LLM providers](../guides/llm-providers.md): every provider and its options.
- [Context compaction](../guides/context-compaction.md): how `compactor` works.
