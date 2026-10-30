---
title: Standing instructions
sidebar_label: Standing instructions
description: "Shape every turn with SOUL.md and AGENTS.md: who the agent is for you, and what it is working on"
---

# Standing instructions

Standing instructions are markdown files whose text is added to an agent's system prompt on every turn. They let you change how an agent behaves without editing its definition: your tone and preferences in one file, the project you are working on in another.

| File | What it says | Follows |
|---|---|---|
| `SOUL.md` | Who the agent is for you: tone, language, standing preferences, things it must not do | You, in every project |
| `AGENTS.md` | What the agent is working on: the project, its conventions, where things are | The task, so it changes when the task does |
| Any other file | A one-off brief, such as today's task | One session |

`AGENT.md` is accepted in place of `AGENTS.md`; when a directory has both, `AGENTS.md` is read.

## In the interactive CLI

The [interactive CLI](../cli/sessions.md) reads the standing instructions its configuration declares, and nothing else: there is no command-line option for them, and `~/.domyn` is not read. `domyn setup` asks two questions, after the agent's name and before the agent's own `instruction`:

1. **A folder or file with your `SOUL.md` / `AGENTS.md`.** A file is saved as you give it. A folder is searched once, subfolders included: its first `SOUL.md` and every `AGENTS.md` are saved, each by its absolute path. An edit to one of those files reaches the next message, but a file added to the folder later is read only after you run `domyn setup` again. Unlike the other questions, an empty answer removes the current paths, so the wizard prints them first.
2. **Whether to load `SOUL.md` / `AGENTS.md` from `./.domyn` and `./.agents`** in the folder `domyn run` starts in. Yes writes both paths as they are, relative, so they are read again on every turn from whichever folder the session starts in, subfolders included, and a file added there is read from the next message. domyn's own folders under `.domyn`, such as `conversations` and `skills`, are not searched. This puts the text of a repository you cloned into the system prompt on every turn, which is why it is a question.

The paths are written in that order, so your own `SOUL.md` wins over a project's, and a project's `AGENTS.md` comes after yours. To change them, run `domyn setup` again or edit `standing_instructions` in `~/.domyn/config.yaml`.

No symbolic link is followed: not a listed path, not a folder on the way to a relative one, and not a file in a listed folder. A cloned repository could otherwise link its `AGENTS.md` to any file on your machine and have it sent to the model.

The line printed when the session starts names the files in use, and says so again when one is added or removed. `/config` shows the paths your configuration declares. They apply to the main agent only.

## What the agent receives

On every turn, the files are read again and their text is placed in the system prompt, right after the agent's own `instruction`, which is never replaced. Each file is a block headed by its name and absolute path:

```text
You are a coding assistant.

[SOUL.md — /home/me/notes/SOUL.md]
Answer in British English. Be terse.

[AGENTS.md — /home/me/work/billing/AGENTS.md]
You are working on the billing service.

[brief.md — /home/me/work/billing/brief.md]
Today: fix the invoice rounding bug.
```

- **One soul.** An agent has one identity, so only the first `SOUL.md` in the order of the paths is used; any later one is not read. An empty `SOUL.md` counts as none, and the next one takes its place.
- **Order.** The `SOUL.md` comes first, then every `AGENTS.md`, then the other files, whatever the order the paths were given in. Within one kind, the order of the paths is kept. When two instructions conflict, models tend to follow the later one, which is why the most specific should come last.
- **Every turn.** An edit to a file reaches the next message, without restarting the session. A file that is missing or empty adds nothing, and a file that cannot be read is skipped with a warning.
- **The path in the header** tells the agent which file a rule comes from, so you can ask it why it does something and learn which file to edit.
- **Each file once**, even when it is reached through two paths.

### Size limit

The text of all standing instructions together, not counting the headers, is limited to 32 KiB (`max_bytes`), because they are sent with every request and no [compaction](context-compaction.md) can shorten them. Over the limit, the file that crosses it is cut at the limit, and its header says `(truncated: instruction budget exceeded)`. The files after it, in the order above, are left out. Both are logged as a warning.

## Declaring them on an agent

Any agent can declare its own standing instructions, in YAML or in Python. An agent that declares none has none: a team member does not inherit them from the main agent.

```yaml
agents:
  - name: Assistant
    ...
    standing_instructions:
      source:
        type: local_dir
        paths: [./instructions, ~/notes/brief.md]
```

```python
from domyn.agents.instructions import InstructionRuntime, LocalDirInstructionsSource

agent = Agent(
    ...,
    standing_instructions=InstructionRuntime(
        source=LocalDirInstructionsSource(paths=["./instructions"]),
    ),
)
```

Relative paths are resolved against the current directory, not the file that declares them.

The text can also be given directly, without files:

```yaml
    standing_instructions:
      source:
        type: memory
        soul: Answer in British English.
        task: Review pull request 12.
```

Its blocks are headed `[soul]`, `[agent]` and `[task]`.

When such an agent is run with `domyn run`, its source is used as it is declared: nothing is added to it.

`attach_instructions(agent, runtime)` sets the runtime on an agent that already exists.

### Options

`standing_instructions`:

| Option | Type | Default | Description |
|---|---|---|---|
| `source` | source | required | Where the instructions come from: `local_dir` or `memory`. |
| `max_bytes` | `int` | `32768` | Limit on the text of all instructions together, in bytes. See [Size limit](#size-limit). |

`local_dir` source:

| Option | Type | Default | Description |
|---|---|---|---|
| `paths` | list of paths | `[]` | Directories, from which `SOUL.md` and `AGENTS.md` (or `AGENT.md`) are read, and files, read whatever their name. Only the files directly in a directory are read, except `.domyn` and `.agents` written as relative paths, which are searched with their subfolders. `~` is expanded. |

`memory` source:

| Option | Type | Default | Description |
|---|---|---|---|
| `soul` | `str` | `""` | Who the agent is. |
| `agent` | `str` | `""` | What the agent is working on. |
| `task` | `str` | `""` | A one-off brief. |

## Next steps

- [Skills](skills.md): instructions the agent loads only when a task needs them.
- [YAML configuration](configuration.md): the rest of an agent's YAML definition.
- [Sessions and commands](../cli/sessions.md): the session the default files apply to.
