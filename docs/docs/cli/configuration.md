---
title: Configuring the CLI
sidebar_label: Configuring the CLI
description: "The CLI's configuration file, the settings you are most likely to change, and what lives in ~/.domyn"
---

# Configuring the CLI

The setup wizard saves the session's configuration to `~/.domyn/config.yaml`. Run `domyn setup` to go through the wizard again, use `/model` to change only the model, and `/config` to see what a session runs on.

## The setup wizard

`domyn setup` asks, in this order:

| Part | What it asks |
|---|---|
| Model | The provider, the model, the URL, the API key, the maximum tokens of a reply and the context window. For `domyn`, also the space id and the configuration id, and the model is picked from the models of the space. See the [CLI quickstart](../getting-started/quickstart-cli.md). |
| Instructions | The agent's name (empty: `Assistant`, at most 30 characters), then a folder or file holding your `SOUL.md` / `AGENTS.md`, then whether to also load them from `./.domyn` and `./.agents` in the folder the session starts in, then extra instructions for the agent, the start of its system prompt (empty: none). See [Standing instructions](../guides/standing-instructions.md). |
| Tools | A checklist of the [built-in tools](tools-and-approvals.md): `files`, `shell` and `browser`, all ticked the first time. Untick them all for an agent that only talks. |
| Skills | A checklist of the built-in skills, by group, all ticked the first time; then a folder of your own skills; then whether to load skills from `./.domyn/skills` and `./.agents/skills` in the folder the session starts in. See [Skills](../guides/skills.md). |

Run again over an existing configuration, the wizard first shows each part as it is and asks whether to keep it; Enter keeps it and skips its questions. In a part you do not keep, each field opens with its saved value: Enter keeps it, and erasing it gives back the default the question names, such as the provider's own URL or the last released configuration. Keeping the skills does not reinstall the built-in ones, so a built-in skill changed by an upgrade arrives only when you do not keep them.

On a terminal, Ctrl-B goes back to the previous question, which opens on the answer you gave it. Ctrl-C stops the wizard and saves nothing.

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
  sandbox: local
  isolate_environment: true
  allowed_env: []
```

The relative paths are the folders the wizard loads from the directory the session starts in, when you answered yes to them; the absolute ones are the files and skills it found or installed.

The settings you are most likely to change:

| Setting | What it does |
|---|---|
| `context_window` | The model's context window, in tokens. Left out (`auto` in the wizard), the session asks the provider at start and assumes 150,000 when it cannot answer; `/context` shows the window it got. Set it when the provider reports none, or a wrong one: the session compacts the conversation based on it. |
| `stream` | `true` prints the answer as it is generated, `false` when it is complete. |
| `bundles` | The [built-in tools](tools-and-approvals.md): `files`, `shell` and `browser`. Remove one to take those tools away, or leave the list empty for an agent that only talks. |
| `skills` | The [skills](../guides/skills.md) the agent loads. Only what is listed here is loaded. |
| `standing_instructions` | The [standing instructions](../guides/standing-instructions.md) added to the system prompt on every turn. |
| `workspace` | The directory the file tools are confined to. `null` means the directory the session starts from. |
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

A project reads nothing from `~/.domyn` except API keys: a key is read from the project's own `credentials` file first, then from `~/.domyn/credentials`, so a key saved once serves every project, and a project can use a different one by setting its own. Its conversations and approvals are its own. An export names nothing outside its own folder: the files the configuration reads (skills, instruction files, the code of tools and hooks) are copied in, and a path that cannot be, such as a tool bundle's workspace, refuses the export, saying which. `domyn setup --import` refuses an export that names anything outside itself, so an edited export cannot make your machine read a file of yours. A project used by its path (`--project ./folder`) is checked every time: if it names files outside its folder, domyn lists them and asks before using it, and `domyn run` also shows the code of its tools and hooks, with what each imports, and asks before running it. Without a terminal to ask on, such a project is refused. A project used by name (`--project <name>`, a folder under `~/.domyn/projects` that setup made or imported) is trusted and used as it is. An export never includes API keys, approvals or conversation history: when the configuration needs a key, the export says which environment variable to set.

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

## What lives in `~/.domyn`

| Path | What it holds |
|---|---|
| `config.yaml` | The session's configuration |
| `credentials` | API keys pasted into the wizard, readable only by you |
| `conversations.db` | Every saved conversation |
| `approvals.yaml` | Approvals that last "always" |
| `projects/` | Your [projects](#projects), and approvals that last for one project |
| `builtin-skills/` | The built-in [skills](../guides/skills.md) you chose in `domyn setup`, copied from the package. Every `domyn setup` deletes and rewrites it, so keep your own skills elsewhere, such as `./.domyn/skills` |
| `models/` | The speech model [`/voice`](sessions.md#talking-instead-of-typing) downloads the first time |

`~/.domyn` is not a folder for [standing instructions](../guides/standing-instructions.md): a `SOUL.md` or `AGENTS.md` there is not read, unless you give its path to `domyn setup`.

## Next steps

- [YAML configuration](../guides/configuration.md): everything a configuration file can declare.
- [LLM providers](../guides/llm-providers.md): every provider and its options.
- [Context compaction](../guides/context-compaction.md): how `compactor` works.
