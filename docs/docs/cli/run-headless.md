---
title: Unattended tasks
sidebar_label: Unattended tasks
description: "Hand the agent one task with domyn run-headless: it carries it out with nobody to answer and no follow-up, needs no setup, and exits when it is done"
---

# Unattended tasks

`domyn run-headless` hands the agent one task and lets it work on its own until it is done. Nobody approves its tool calls, nobody answers its questions, and no message follows: what counts is what the agent leaves behind when it exits. It is for running domyn where nobody is at the keyboard, such as a container, a CI job or a benchmark.

It needs no `domyn setup` and reads nothing of one. The endpoint, the model and the key are given on the command line, and everything the session writes goes into a folder of its own. To run with a configuration of your own instead, give it with [`--config`](#running-a-configuration-of-your-own).

```bash
export MY_ENDPOINT_KEY=...
domyn run-headless \
  --model_url http://localhost:8000/v1 \
  --model_name Qwen/Qwen3-32B \
  --api-key-env MY_ENDPOINT_KEY \
  "The tests in tests/test_parser.py fail since the last commit: find out why and fix it"
```

For a long task, or one you do not want on the command line, put it in a file:

```bash
domyn run-headless --model_url http://localhost:8000/v1 --model_name Qwen/Qwen3-32B --prompt-file task.md
```

## How it differs from `domyn run`

`domyn run "<message>"` also starts with a message, but it is a [session](sessions.md): it waits for you after the first answer, asks before a tool changes anything, and reads the configuration `domyn setup` wrote. `domyn run-headless` is built for nobody being there:

| | `domyn run "<message>"` | `domyn run-headless "<task>"` |
|---|---|---|
| Tool calls | asked about, unless you turned on `/auto` | all approved, as with `/auto`: the [short list of commands refused even then](tools-and-approvals.md#approvals) still is |
| After the first answer | waits for your next message | exits |
| A message starting with `/` | a [slash command](sessions.md#commands) | the task, sent as written (`/app/data.csv has a bad row…`) |
| What the agent is told | your standing instructions | that it works alone: do the work, ask nothing, check the result (below) |
| Configuration | `~/.domyn/config.yaml` or a project, from `domyn setup` | built from the command line, or a file given with `--config` |
| Keys | `.env`, `~/.domyn/credentials` | only the variable `--api-key-env` names, set in the environment or a `.env`. With `--config`, the variables the file names, from `~/.domyn/credentials` too |

Auto-approve is turned on for this session only: it is not remembered, so an interactive session of the same configuration starts as you left it.

## The endpoint and the model

`domyn run-headless` talks to any OpenAI-compatible endpoint (vLLM, a gateway, a hosted API).

- `--model_url` is the endpoint: `http://host:8000`, `http://host:8000/v1` or the full `…/v1/chat/completions` all work.
- `--model_name` is the model's name as the endpoint serves it.
- `--api-key-env` names the environment variable holding the key. Without it no key is sent; naming a variable that is not set is an error. Nothing else is read for the endpoint: `OPENAI_BASE_URL`, `OPENAI_API_KEY` and the keys `domyn setup` saved are ignored, so a variable set for another tool cannot send the task elsewhere.

The model's limits are left to the endpoint unless you give them:

| Option | Without it |
|---|---|
| `--context-window` | asked of the endpoint when the session starts, as for any [provider](../guides/llm-providers.md); [compaction](../guides/context-compaction.md) uses it |
| `--max-tokens` | the endpoint's own limit on a reply. When given, compaction keeps that much room under the window for the reply |
| `--temperature` | the endpoint's default |
| `--reasoning-effort` | `default`: no `reasoning_effort` is sent and the endpoint's default applies. Give `low`, `medium` or `high` (or whatever the endpoint accepts) to send one |

## What the agent works with

Without `--config`, the session runs with the following.

- **Tools**: the [file tools and `run_command`](tools-and-approvals.md), on the machine `domyn run-headless` runs on (never in a Docker container). Commands see every variable of the environment it was started with, the key's included, and use the Python environment they find rather than one of their own. The file tools reach everything under `--workspace`, which is `/` unless you give a folder.
- **Skills**: none. The built-in [skills](../guides/skills.md) are not installed; a `--config` file can declare skills of its own.
- **Standing instructions**: a short text telling the model it works alone, that the message is a task to carry out rather than a question, that nobody will answer a question or read a plan, to install what is missing and fix errors on the way, to follow the request's names and paths exactly, to check the result before finishing, to leave services running, and to end with a short summary. Without it a model tends to describe the steps and offer to carry them out, and with nobody to accept, the session ends on the offer. `--instructions <file>` replaces the text with the file's, `--no-instructions` runs without any. See [Standing instructions](../guides/standing-instructions.md).
- **Steps**: up to 1000 model calls in the turn (`--max-iterations`).

## Running a configuration of your own

`--config <file>` runs a configuration file instead of the one built from the options above: the format of `~/.domyn/config.yaml`, or of any [YAML application](../guides/configuration.md). The file decides the model, the tools and where they run, the skills, the standing instructions and the limits, and is used as written.

```bash
domyn run-headless --config my-agent.yaml --prompt-file task.md
```

- The session is still unattended: every tool call approved, one turn, no follow-up. The standing instructions are the file's own, so the text above is not added: declare it in the file if you want it (a session without it tends to stop at a plan).
- The options that build the configuration (`--model_url`, `--model_name`, `--api-key-env`, `--context-window`, `--max-tokens`, `--temperature`, `--reasoning-effort`, `--max-iterations`, `--instructions`, `--no-instructions`, `--workspace`) cannot be given with it: set them in the file. `--prompt-file`, `--state-dir` and `--verbose` work as without it.
- The file's `${VAR}` references are read from the environment, a `.env` and `~/.domyn/credentials`, so a configuration `domyn setup` wrote finds the key it saved.
- The file is copied unchanged into the state folder as its `config.yaml`, and the session runs that copy. `${DOMYN_PROJECT_DIR}` in it means the state folder; relative paths mean the folder you run `domyn run-headless` from, as everywhere.

A good starting point is the configuration `domyn run-headless` builds: run it once without `--config`, and copy `config.yaml` from the state folder.

## Where things are kept

`--state-dir` is the folder the session keeps everything in: its `config.yaml` (the one built from the options, or a copy of the `--config` file), `conversations.db` with the whole conversation, and the tools' working files. Without it, a new temporary folder is made each time. It plays the part of `~/.domyn` for the session, so nothing in `~/.domyn` is read or written.

The answer, the tool calls and the agent's summary are printed as the session goes; redirected to a file or a pipe, the output is plain text. At the end the session prints the command that reopens the conversation, `domyn run --project <state-dir> --resume <conversation_id>`, if you want to look at it or continue it yourself.

The exit code is `0` once the turn has ended, whatever its outcome: an error from the model ends the turn too, and is printed. It is `130` when the turn is interrupted with Ctrl+C, `2` for a mistake on the command line (no task, no `--model_url`, an unset key variable), and `1` when the agent cannot be built.

## In a container or a benchmark

- Give the task with `--prompt-file`. On the command line, a `pkill -f <word>` the agent runs to stop its own processes matches domyn's command line too whenever the task contains the word, and stops domyn.
- Give a `--state-dir` outside `/tmp` when the agent may clean up `/tmp`: the conversation is there.
- Keep `--workspace /` when the task's files can be anywhere in the container (`/app`, `/etc`, `/tmp`), as with other agents run this way; give a folder to keep the file tools inside it.
