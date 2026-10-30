---
title: Sessions and commands
sidebar_label: Sessions and commands
description: "Chat with an agent from the terminal with domyn run: starting a session, working in it, its commands, conversations and the context window"
---

# Sessions and commands

`domyn run` opens a chat session with an agent in your terminal. The agent can read files, edit them and run commands in the directory you start it from, and it asks before doing anything that changes something. To try it for the first time, start with the [CLI quickstart](../getting-started/quickstart-cli.md).

## Starting a session

| Command | What it opens |
|---|---|
| `domyn run` | A session with the agent configured in `~/.domyn/config.yaml`. The first time, the setup wizard runs before it. |
| `domyn run "<message>"` | The same session, with `<message>` sent as the first message. |
| `domyn run agent.py` | A session with the agent defined in a Python file. |
| `domyn run agent.yaml` | A session with the agent defined in a [YAML file](../guides/configuration.md). |
| `domyn run --project <name>` | A session with a [project](configuration.md#projects): a separate configuration with its own model, keys, skills and history. |
| `domyn run --resume <conversation_id>` | A past conversation, to continue it. Without an id, you pick it from a list. For a conversation started from a file, give the file too: `domyn run agent.yaml --resume <conversation_id>`. |

When a Python file defines more than one agent, `domyn run` picks the one that is not a sub-agent of another; if there are several such agents, it takes the first and says which. To choose a different one, pass the name of its variable with `--agent-name`.

Other options:

| Option | What it does |
|---|---|
| `--expand-skills` | Gives the model every [skill](../guides/skills.md) in full up front, instead of an index it reads from on demand. Uses more of the context window. |
| `--skill-sections` | Lists each skill's sections in the index. Uses more of the context window. |
| `--verbose`, `-v` | Shows the internal events: model calls, hooks, agent handoffs. |
| `--no-banner` | Skips the startup banner. |

Which skills and [standing instructions](../guides/standing-instructions.md) a session reads is not an option: it is in the configuration, and `domyn setup` asks about it (see [Configuring the CLI](configuration.md)).

## During a session

Type a message and press Enter. To start a new line without sending, press Ctrl+J, or Alt+Enter (on a Mac, Option+Enter, if the terminal is set to use Option as Meta). The answer is printed as it is generated, together with each tool call the agent makes, one line per call. While the model is working or a tool is running, a clock shows how long that step has taken (`thinking` or `executing`), and when the turn ends the session prints its total time.

The answer and the model's reasoning are rendered as Markdown: headings, lists, quotes, code blocks with syntax colours, and tables with borders. A line appears once it is complete; the line still being written, or a table still being received, is shown above the input line until it is. When the output is not a terminal that shows colour, such as a pipe, a file or a terminal with `NO_COLOR` set, the text is printed as the model wrote it, so output redirected to a file keeps the Markdown source.

**While the agent is working** you can keep typing:

- **A message** is delivered to the agent at its next step, so you can correct or add to a request without waiting for the answer.
- **A command that only shows something**, such as `/context` or `/help`, runs at once, and so does `/auto`.
- **Any other command**, such as `/clear` or `/model`, is held and runs when the agent finishes. A held `/clear` clears the screen right after the answer is printed; `/resume` shows that conversation again.

**Ctrl-C** depends on what is happening:

| When | Ctrl-C |
|---|---|
| The agent is working | Stops the current turn. The session stays open. |
| You have typed something | Clears the line. |
| The line is empty | Shows `press Ctrl-C again to quit`. A second Ctrl-C within a moment ends the session. |

**Changing the configuration** needs no restart. The configuration file is read again before every message, so a change to it, made by hand, by the agent or by `domyn setup` in another terminal, is used from the next message, and the session says `Reloaded <file>: it changed since the last message.` A file that no longer loads is reported once, and the session keeps the configuration it had. Only the options given to `domyn run` and API keys saved in the credentials file need a restart.

Skills and standing instructions are read again every turn too. When a turn adds or removes one, for example a skill the agent wrote, the session says so in a line such as `Now: 4 skills (+weekly-report)`.

Ctrl-D and `/exit` also end the session. When it ends, if anything was said, it prints the conversation id and the command that resumes it, which is how you get back to it.

### Talking instead of typing

`/voice`, or starting with `domyn run --voice`, turns on push-to-talk. It needs the `audio` extra, which the standalone installer already includes (with uv: `uv tool install 'domyn[cli,audio]'`), and a microphone; the first time, the speech model is downloaded (about 670 MB, into `~/.domyn/models`) and macOS asks whether the terminal may use the microphone.

- **Hold the space bar** on an empty line and speak; **let go** to send. What you said is transcribed on your machine and sent as if you had typed it, so it appears as your message.
- **Tap** it instead, and recording goes on until the next tap: for a long message, or a keyboard without key repeat.
- **During a turn**, the space bar stops the agent, as Ctrl-C does, and what you say next becomes the next message.
- A line above the prompt says when the microphone is open (`● recording`), and when the transcription is running.
- A space in the middle of a sentence is still a space, and approvals are still answered on the keyboard.

A terminal reports when a key is pressed, not when it is let go, so the release is read from the key repeat stopping: the message is sent a fraction of a second after you let go. `/voice` again turns it off. The language is detected from what you say; to transcribe somewhere else, or to fix the language, see [Voice](configuration.md#voice).

## Commands

Type `/help` in a session for this list. Commands complete with Tab.

| Command | What it does |
|---|---|
| **Session** | |
| `/help` | Lists the commands. Also `/?`. |
| `/exit` | Ends the session. Also `/quit` and `/q`. |
| `/clear` | Starts a new conversation and clears the screen, scrollback included. The previous one stays saved, and `/resume` brings it back. |
| `/resume [conversation_id]` | Continues a past conversation, redrawing it on a cleared screen. Without an id, shows the recent ones to pick from. |
| `/verbose` | Turns on or off the display of internal events and library logs |
| `/voice` | Turns push-to-talk on or off: hold the space bar to speak, let go to send. See [Talking instead of typing](#talking-instead-of-typing). |
| **Agent and model** | |
| `/agent` | Shows the active agent, its tools, its sub-agents, the conversation id and the conversation's working directory |
| `/model` | Changes the provider and the model, and saves them to `~/.domyn/config.yaml`. Only for a session on the saved configuration, not one started from a file. |
| `/config` | Shows the path and the contents of the configuration file this session runs on, without secrets. A session started from a Python file has none. |
| `/skills` | Lists the skills available to the agent |
| `/skill <name>` | Shows one skill in full |
| `/skills-autodiscover [on\|off]` | Shows, or turns on or off, loading skills from `./.domyn/skills` and `./.agents/skills` of the folder `domyn run` started in. Saved to the configuration and used from the next message. |
| `/export <folder>` | Saves this session's configuration, with the skills and instruction files it uses, into a new folder: a [project](configuration.md#projects) you can share or run elsewhere. Never includes API keys, approvals or history. |
| **Approvals** | |
| `/approvals` | Lists the approvals in force: for this conversation, this project and always. `/approvals revoke <n>` removes the ones you name by number, `/approvals clear` removes them all. |
| `/auto` | Turns automatic approval on or off, as Shift+Tab does. While on, every tool call runs without asking, and the prompt says so, until you turn it off or the session ends. |
| **Context** | |
| `/context` | Shows how much of the model's context window the last request used, and what used it. Empty until the first request. |
| `/compact` | Shortens the conversation now. The shorter version is used from the next request. |
| **Background commands** | |
| `/ps` | Lists the commands this conversation started that are still running in the background |
| `/stop <handle>` | Stops one of them, by the handle `/ps` shows |

## Conversations

Every conversation is saved to `~/.domyn/conversations.db`. `/clear` starts a new one without deleting the old one, and `/resume` or `domyn run --resume` brings any of them back.

A resumed conversation is drawn again as it looked: your messages, the answers and the tool calls, up to the last 20 turns, with a line counting the earlier ones that are not shown. The agent still receives the whole conversation. A `── resumed here ──` line marks where the new part begins. The conversation id printed when a session ends is the only way to name one directly, so keep it if you want to come back to it. A conversation started from a file is resumed with the same file, `domyn run agent.yaml --resume <conversation_id>`; the command printed at the end already includes it.

## Context window

Every model can read only a limited amount of text at once, its **context window**. `/context` shows how much of it the conversation uses.

When a conversation gets close to the limit, the session shortens it automatically: first it replaces old tool outputs with a short note saying what they were, and if that is not enough, it has the model summarize the older part of the conversation. You see when this happens. `/compact` replaces old tool outputs right away, without waiting for the limit. For how it works and how to change it, see [Context compaction](../guides/context-compaction.md).

## Next steps

- [Files, commands and approvals](tools-and-approvals.md): what the agent can do in your directory, and how you approve it.
- [Configuring the CLI](configuration.md): the model, the tools and the files in `~/.domyn`.
- [CLI commands](../reference/cli.md): every `domyn` command and option.
