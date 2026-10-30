---
title: CLI commands
sidebar_label: CLI commands
description: "Every domyn command and its options"
---

# CLI commands

The `domyn` command is installed with the package (see [Installation](../getting-started/installation.md)). Every command accepts `-h` / `--help`.

| Command | What it does |
|---|---|
| [`domyn run`](#domyn-run) | Chat with an agent in the terminal |
| [`domyn setup`](#domyn-setup) | Choose the provider, model and agent of `domyn run` |
| [`domyn init`](#domyn-init) | Create a new project with an example agent |
| [`domyn skills`](#domyn-skills) | List the skills the configuration loads |
| [`domyn skill`](#domyn-skill) | Show a skill, or stop loading it |
| [`domyn export`](#domyn-export) | Save the configuration, with its skills and instructions, into a new folder to share |
| [`domyn connect`](#domyn-connect) | Connect an agent to the Domyn platform as a relay worker |

The commands available inside a `domyn run` session, such as `/model` or `/resume`, are listed in [Sessions and commands](../cli/sessions.md).

## Global options

Given before the command: `domyn --log-level DEBUG run`.

| Option | Type | Default | Description |
|---|---|---|---|
| `--log-level` | `str` | `WARNING` for `run`, `setup`, `init`, `skills` and `skill`; `INFO` for the servers | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Any other value is taken as `INFO`. |
| `--version` | | | Prints the version and exits. |

## Environment

Before running any command, `domyn` reads a `.env` file from the current directory or the nearest parent directory that has one, then `~/.domyn/credentials`, where the setup wizard saves API keys. A variable already set in the environment is never replaced, and `.env` comes before the credentials file. These variables are what `${VAR}` in a [YAML configuration](../guides/configuration.md) refers to. With `--project`, the project's own `credentials` file is read first, and `~/.domyn/credentials` supplies only the keys the project does not set.

`run`, `setup`, `skills`, `skill` and `export` take `--project <name>` to work on a [project](../cli/configuration.md#projects), a separate configuration, instead of the default one in `~/.domyn`. A name means `~/.domyn/projects/<name>`; a value containing `/` or starting with `.` or `~` is the path of a project folder.

## `domyn run`

```bash
domyn run [FILE] [MESSAGE] [OPTIONS]
```

Opens a chat session with an agent. See [Sessions and commands](../cli/sessions.md) for the session itself.

| Argument | Description |
|---|---|
| `FILE` | A `.py` or `.yaml` file defining the agent. Without it, the agent configured in `~/.domyn/config.yaml`; the first time, the setup wizard runs before the session. |
| `MESSAGE` | Sent as the first message of the session. |

From a Python file, the agent is the one that is not a sub-agent of another; from a YAML file, its `entrypoint`.

| Option | Type | Default | Description |
|---|---|---|---|
| `--agent-name`, `-a` | `str` | | The variable holding the agent, when a Python file defines more than one. |
| `--resume` | flag | off | Continues a past conversation: `--resume <conversation_id>`, or `--resume` alone to choose one from a list. The id takes the place of `MESSAGE`, so the two cannot be combined. For a conversation started from a file, give the same `FILE`. |
| `--expand-skills` | flag | off | Gives the model every [skill](../guides/skills.md) in full up front, instead of an index it reads from on demand. Uses more of the context window. |
| `--skill-sections` | flag | off | Lists each skill's sections in the index. Uses more of the context window. |
| `--verbose`, `-v` | flag | off | Shows internal events: model calls, hooks, hand-offs. |
| `--no-banner` | flag | off | Skips the startup banner. |
| `--project` | `str` | | Runs a [project](../cli/configuration.md#projects) instead of the default configuration. The banner shows its name. |
| `--voice` | flag | off | Starts with push-to-talk on, as `/voice` does: hold the space bar to speak. Needs the `audio` extra. See [Talking instead of typing](../cli/sessions.md#talking-instead-of-typing). |

The skills and the [standing instructions](../guides/standing-instructions.md) are those the configuration declares; no option adds or removes any.

When the session ends, it prints the conversation id and the command that resumes it, with `FILE` when the session was started from one: `domyn run agent.yaml --resume <conversation_id>`.

## `domyn setup`

```bash
domyn setup [--show] [--project NAME] [--import FOLDER]
```

Runs the setup wizard: provider, model and API key, standing instructions, built-in tools and skills, saved to `~/.domyn/config.yaml` and, for the key, `~/.domyn/credentials`. Run again over an existing configuration, it shows each part and asks whether to keep it. See the [CLI quickstart](../getting-started/quickstart-cli.md) and [Configuring the CLI](../cli/configuration.md).

| Option | Description |
|---|---|
| `--show` | Prints the active configuration and exits. |
| `--project` | Sets up a [project](../cli/configuration.md#projects) instead of the default configuration, creating it if it does not exist. |
| `--import` | Creates a new project from a folder made by `domyn export`, named after the folder unless `--project` gives a name. The wizard shows each exported setting and asks whether to keep it; when you do not keep the skills, it also offers the built-in skills the export does not include, unticked. It warns when the export was made by another version of domyn-agents, or does not say which. Cannot be combined with `--show`. |

## `domyn init`

```bash
domyn init [PROJECT_NAME] [--directory DIR] [--force]
```

Creates a new project with `uv`: a `pyproject.toml` that depends on `domyn[cli]`, an example `agent.py` with two tools, a `README.md` and a `Dockerfile` that runs `domyn serve agent.py`, then installs the dependencies with `uv sync`. `uv` must be installed. Without `PROJECT_NAME`, it asks for one.

| Option | Type | Default | Description |
|---|---|---|---|
| `--directory`, `-d` | `str` | `./PROJECT_NAME` | Where to create the project. |
| `--force`, `-f` | flag | off | Deletes the directory first if it exists and is not empty. Without it, `init` stops. |

Then set the provider in `agent.py` and run `uv run domyn run agent.py`.

## `domyn skills`

```bash
domyn skills
```

Lists the [skills](../guides/skills.md) the configuration loads, `./.domyn/config.yaml` when the current directory has one and `~/.domyn/config.yaml` otherwise: their names, descriptions, sections and files, and why any is unavailable. A relative path in the configuration is read from the current directory, as `domyn run` reads it.

| Option | Type | Description |
|---|---|---|
| `--project` | `str` | Uses a [project](../cli/configuration.md#projects)'s configuration. |

## `domyn skill`

```bash
domyn skill show|remove NAME
```

Works on the skills the global `~/.domyn/config.yaml` loads, whatever the current directory, or a project's with `--project`.

| Action | Description |
|---|---|
| `show` | Prints the skill: its file, its sections and its text. `NAME#SECTION` prints one `##` section. |
| `remove` | Removes the entry that names the skill, its folder or its `.md` file, from the configuration. No file is deleted. A skill loaded through a listed folder, such as `./.domyn/skills`, has no entry of its own: nothing is removed, and the command says how to stop loading it. |

| Option | Description |
|---|---|
| `--project` | Uses a [project](../cli/configuration.md#projects)'s configuration. |

## `domyn export`

```bash
domyn export FOLDER [--config FILE] [--project NAME]
```

Saves the configuration into the new folder `FOLDER`, with the skills and standing instructions it uses, so it can be shared or run elsewhere. The folder is a [project](../cli/configuration.md#projects): run it where it is with `domyn run --project FOLDER`, move it to `~/.domyn/projects/<name>` and run `domyn run --project <name>`, or create a new project from it with `domyn setup --import FOLDER`.

| Option | Type | Default | Description |
|---|---|---|---|
| `--config` | path | the one `domyn run` uses | The configuration file to export. |
| `--project` | `str` | | Exports a project instead of the default configuration. |

A skill is copied as its `SKILL.md` plus the files its `external_files` lists; anything else in its folder is left out and named in the report. API keys, approvals and conversation history are never copied: a key written in the configuration is replaced by a `${VAR}` reference, and the report names the variables to set. `FOLDER` must not exist, or be empty. The folder also gets a hidden `.info` file with the version of domyn-agents, the date, the Python version and the platform, which `domyn setup --import` checks. `/export FOLDER` does the same from inside a session.

## `domyn connect`

```bash
domyn connect MODULE:SYMBOL --channel-id ID --base-url HOST
```

Serves an agent or a LangGraph graph on a relay channel, for a `DelegateAgent` in another application to call. `domyn expose` is an older name for the same command, still accepted. It needs the [platform SDK](../getting-started/installation.md#the-domyn-platform-sdk), which provides the relay's address and headers. See [Remote agents](../guides/remote-agents.md) for the argument, the relay, and LangGraph graphs.

| Option | Environment variable | Type | Default | Description |
|---|---|---|---|---|
| `--channel-id` | `DOMYN_CHANNEL_ID` | `str` | | Channel to serve. |
| `--space-id` | `DOMYN_SPACE_ID` | `str` | | Space the channel belongs to. |
| `--configuration-id` | `DOMYN_CONFIGURATION_ID` | `str` | | Configuration of the space to use. Optional. |
| `--base-url` | `DOMYN_BASE_URL` | `str` | | Relay host, with an optional port. The relay is reached at `wss://<host>/relay/v1/ws`, or `ws://` for `localhost`, `127.0.0.1` and `::1`. |
| `--target` | | `str` | | Serves several channels from one process. Repeatable; each is `channel_id=…,space_id=…,base_url=…,configuration_id=…`, where only `channel_id` is required. `space_id` and `base_url` default to `--space-id` and `--base-url`; `configuration_id` defaults to `--configuration-id` only when the target sets no `space_id`. |
| `--max-concurrency` | | `int` | `8` | Turns run at the same time. Beyond that, as many more wait, and further requests are refused. |
| `--platform-tools` / `--no-platform-tools` | | flag | on when the SDK is installed, else off | Adds to the entrypoint agent every tool the platform offers that it does not already declare. `--platform-tools` without the SDK stops and names the version it needs. |
| `--tool-refresh-interval` | | `float` | `0` | With `--platform-tools`, looks for new platform tools every this many seconds. `0` never does. |

`DOMYN_API_KEY` must be set; it is sent to the relay in the `api-key` header.
