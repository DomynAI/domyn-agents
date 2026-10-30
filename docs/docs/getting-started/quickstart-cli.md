---
title: "Quickstart: CLI"
sidebar_label: "Quickstart: CLI"
description: "Install the domyn command and chat with an agent that works on the files of a directory"
---

# Quickstart: CLI

This guide takes you from a fresh install to a chat with an agent in your terminal. The agent can read and edit the files of a directory and run commands in it, asking you before it changes anything.

Install the CLI before you start. On macOS (Apple Silicon) and Linux, the installer needs no Python:

```bash
curl -fsSL https://github.com/domynai/domyn-agents/releases/latest/download/install.sh | sh
```

or, with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install "domyn[cli]"
```

For the details of both, and for how API keys are read, see [Installation](installation.md).

## 1. Start a session

The built-in agent can read files and run commands in the directory you start it from. For a first try, use a folder you don't mind it working in:

```bash
mkdir domyn-quickstart && cd domyn-quickstart
domyn run
```

## 2. Configure a model

The first time you run `domyn`, a setup wizard asks how to reach a model. Each field opens with its current value: press Enter to keep it, or erase it (Ctrl-U) and press Enter to go back to the default the question names (`empty = …`). Without a terminal, in a pipe, the value is shown in brackets and Enter keeps it. Ctrl-B goes back to the previous question.

| Question | What to enter |
|---|---|
| provider | Where the model is served: `domyn` (the models of your Domyn platform space, the first choice), `openai`, `anthropic`, `open_router`, `cerebras`, `vllm` or `custom` (any OpenAI-compatible endpoint). |
| model | The model name, as your provider spells it. The wizard prints where to find the valid names. For `domyn`, when the platform URL, space and API key are known (for example from `DOMYN_BASE_URL`, `DOMYN_SPACE_ID` and `DOMYN_API_KEY`), it lists the models of the space for you to choose from, and an `other` row lets you type a name that is not listed. |
| url | Leave it empty to use the provider's default endpoint. It is required for `domyn`, where it is the platform's address, and for `vllm` and `custom`. |
| space id, configuration id | Only for `domyn`: the space whose configuration holds the model, and optionally which configuration to use (empty: the last released one). The platform URL and the space id are taken from `DOMYN_BASE_URL` and `DOMYN_SPACE_ID` when they are set: the wizard says it found them, without printing them, and Enter uses them. Type another value to replace one, and erase a saved one to go back to the variable. |
| api key | If the provider's variable (for example `OPENAI_API_KEY`) is already set, the wizard finds it. Otherwise paste the key, and it is saved to `~/.domyn/credentials`. A key saved there appears hidden (`********`) when the wizard runs again or you go back to it: press Enter to keep it, or type a new one. |
| max tokens, context window | Limits for the model's replies and for its context. The defaults work for most models. |
| agent name | The name of your agent. Leave it empty for `Assistant`. |
| SOUL.md / AGENTS.md | A folder or file with [standing instructions](../guides/standing-instructions.md) for your agent, and whether to load them from `./.domyn` and `./.agents` too. Leave the first empty if you have none. |
| extra instructions | The start of the agent's system prompt. Leave it empty if you have none. |
| built-in tools | The tools the agent gets: `files`, `shell` and `browser`. Keep them all ticked for this guide. |
| skills | The built-in [skills](../guides/skills.md) to install, a folder of your own, and whether to load skills from `./.domyn/skills` and `./.agents/skills`. The defaults are fine. |

The answers are saved to `~/.domyn/config.yaml`, so the wizard does not run again. Run `domyn setup` to change them later.

## 3. Ask something

When the wizard finishes, the session opens. Create a file for the agent to work with, then ask about it:

```text
you › Create a file called notes.md with three tips for writing clear documentation
```

The agent can read, search and list files on its own, because reading cannot damage anything. Writing a file or running a command needs your approval first, so the session shows the call and asks `run it?`. Choose an answer with the arrow keys, or type its number, and press Enter:

- **yes, just this once** lets it run. The other `yes` answers also remember the approval: for the rest of the conversation, in this project, or always.
- **no** stops it. The agent is told the call was refused and continues without it.

The file tools work only inside the directory you started the session from: they cannot read or write a file outside it, even with your approval. Commands start in that directory too, but a command can reach any path you could reach yourself, which is why every command needs your approval.

## 4. Find your way around

Type `/help` to list the session commands. The ones you need first:

| Command | What it does |
|---|---|
| `/help` | Lists every command |
| `/model` | Changes the provider or model |
| `/clear` | Starts a new conversation and clears the screen |
| `/exit` | Ends the session |

When the session ends, it prints the conversation id. To continue the conversation later, run `domyn run --resume <conversation_id>`, or `domyn run --resume` to pick it from a list.

For everything else the session can do, see [Sessions and commands](../cli/sessions.md).

## Next steps

- [Sessions and commands](../cli/sessions.md): everything a session can do.
- [Files, commands and approvals](../cli/tools-and-approvals.md): what the agent can reach, and how approvals work.
- [Configuring the CLI](../cli/configuration.md): change the model, the tools and the agent.
- [Quickstart: library](quickstart-library.md): build your own agent in Python and chat with it the same way.
