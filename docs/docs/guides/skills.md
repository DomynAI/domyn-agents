---
title: Skills
sidebar_label: Skills
description: "Give an agent reusable instructions it reads only when a task needs them"
---

# Skills

A skill is a set of instructions for one kind of task — how to write a release note, how to review a contract, the steps of a deployment — kept in a Markdown file. The agent always sees a one-line index of its skills, and reads a skill in full only when the task calls for it. Dozens of skills cost the context a few hundred tokens until one is needed.

Use a skill for know-how that applies to some tasks and not others. Instructions that apply to every message belong in the agent's `instruction`, or in [standing instructions](standing-instructions.md).

Skill files use the [agentskills.io](https://agentskills.io) format, so skills written for other tools that use it work here, and the other way round.

## Writing a skill

A skill is a `SKILL.md` file: YAML frontmatter, then the instructions in Markdown.

```markdown
---
name: user-journeys
description: How to write a user journey document. Use when asked to draft or review one.
---

Write user journeys from the user's point of view.

## Structure

1. Persona
2. Goal
3. Steps

## Review checklist

- Every step has an observable outcome.
```

| Key | Description |
|---|---|
| `name` | The skill's name. Defaults to the name of the directory holding `SKILL.md`, or to the file name without `.md`. |
| `description` | What the skill is for **and when to use it**. The model chooses a skill from this line alone, so write it for that. Without one, the first line of the body is used, which is rarely written for the purpose. |
| `version` | Free-form, for your own tracking. |
| `enforcement` | `advisory` (default) or `required`. See [Required skills](#required-skills). |
| `metadata.requires` | What the skill needs to be usable. See [Requirements](#requirements). |

Each `##` heading starts a **section**. The model can read one section of a skill instead of the whole file, which keeps a long skill cheap.

A directory of skills can hold both layouts side by side, and group them in folders:

```text
skills/
├── user-journeys/
│   └── SKILL.md          # a skill with its own directory, for supporting files
├── release-notes.md      # a single-file skill
└── writing/              # a group: every skill under it is read
    └── essays/
        └── SKILL.md
```

A directory holding a `SKILL.md` is one skill, and nothing inside it is read as another. Any other directory is a group, entered at any depth. A single-file skill is only read directly in the listed directory. Hidden folders and dependency folders such as `node_modules` or `venv` are skipped, and **no symbolic link is followed**, including the listed directory itself.

### Requirements

A skill can declare what it needs, under `metadata.requires` or as top-level keys:

```yaml
---
name: release-notes
description: How to write release notes. Use when preparing a release.
metadata:
  requires:
    bins: [git]
    env: [GITHUB_TOKEN]
    os: [darwin, linux]
    tools: [read_file]
    external_files:
      - scripts/
      - ../shared_helper.py
---
```

| Key | Met when |
|---|---|
| `bins` | Each program is on the `PATH`. |
| `env` | Each environment variable is set and not empty. |
| `os` | The platform is one of those listed, as Python's `sys.platform` names it: `darwin`, `linux`, `win32`. |
| `tools` | The agent has each of these tools. |
| `external_files` | Each file the skill needs besides its `SKILL.md` exists. |

`external_files` lists paths **relative to the `SKILL.md`**, with forward slashes; `..` is allowed, an absolute path is not. An entry can be a file, or a folder, meaning every file under it. Nothing is copied: the files must already be where the skill says.

A skill whose requirements are not met stays in the index, marked with the reason, for example `[unavailable: env 'GITHUB_TOKEN']` or `[unavailable: file 'scripts/' (does not exist)]`, so the model knows it exists and why it cannot use it. Requirements are checked on every turn, so a missing file put back counts from the next message.

## Giving skills to an agent

An agent reads its skills through a `SkillsRuntime`:

```python
from domyn.agents import Agent
from domyn.agents.skills import LocalDirSkillSource, SkillsRuntime

agent = Agent(
    name="Writer",
    description="Writes product documents",
    instruction="Help the user write product documents.",
    llm_provider=llm,
    skills=SkillsRuntime(source=LocalDirSkillSource(paths=["./skills"])),
)
```

`LocalDirSkillSource` reads every skill in `paths`. When two directories hold a skill of the same name, the one in the earlier directory is used. A relative path is resolved against the current directory, and `~` is expanded.

Skills can also be given as text, with no files:

```python
from domyn.agents.skills import InMemorySkillSource

SkillsRuntime(source=InMemorySkillSource(skills={
    "tone": "---\ndescription: House tone of voice. Use for any customer-facing text.\n---\nBe warm and brief.",
}))
```

In a [YAML configuration](configuration.md), `skills` is declared on the agent:

```yaml
agents:
  - name: Writer
    ...
    skills:
      source:
        type: local_dir
        paths: [./skills]
```

Skill files are read again on every turn, so an edited skill applies from the next message. In a [team](orchestration.md), each agent has its own skills.

## How a skill reaches the model

On every model call of a turn, the agent's skills are added to the request as a message right after the system prompt. By default it is the index:

```text
Available skills (call skill_view to read one in full):
- release-notes: How to write release notes. Use when preparing a release.
- user-journeys: How to write a user journey document. Use when asked to draft or review one.
```

The agent also gets a `skill_view` tool, added automatically. When a skill looks relevant, the model calls `skill_view(name="user-journeys")` for the whole skill, or `skill_view(name="user-journeys", section="Review checklist")` for one section, and reads it as the tool's result. The result starts with the skill's file:

```text
Skill: user-journeys
[file: skills/user-journeys/SKILL.md] — a real file you can read and edit with your file tools when asked to change this skill.
...
```

The path is there so that an agent asked to change a skill knows where it is, and so that a skill can name its own files relative to it. A skill added in full before the turn (see [Choosing skills before the turn](#choosing-skills-before-the-turn)) starts the same way. Whether the agent can open the file depends on its tools.

### When the index does not fit

The index has a budget, `max_index_chars`. When the skills do not fit, none is left out: all of them are shown in a shorter form, the first that fits of

1. every description whole;
2. every description cut to `max_description_chars`;
3. names only, on one line: `Also available, by name only: release-notes, user-journeys, ...`.

The same form is used for every skill, so which skill keeps its description never depends on its name. A `required` skill is always shown whole. If even the names do not fit, the index lists the names that do and ends with how many it left out, so the model knows there are more and can find them with `skill_search`.

### Searching for a skill

The agent also gets a `skill_search` tool. It searches every skill, including those the index shows by name only or not at all, and answers with the best matches, at most 10, each with its description cut to 120 characters. It never lists the whole catalogue: `query` is required, and `limit` (default 5) is capped at 10.

The search is done by a model, given the query and the catalogue: skills and queries may be in different languages, and a model matches them by meaning. That model is the agent's own unless `search.provider` names another. If it fails, does not answer within `search.timeout_s`, or there is none, the search falls back to matching words, ignoring accents and comparing parts of words, so that close languages still meet; the result then says so.

In `domyn run`, the status line at the start of a session says when the index is over its budget, and `/skills` says in which form the model sees the skills.

### Choosing skills before the turn

The `selector` decides whether some skills are read in full before the model starts, instead of on request:

| `type` | Class | What the model gets |
|---|---|---|
| `index` | `IndexSelector` | **Default.** The index only. The model reads skills with `skill_view`. |
| `all` | `AllSkillsSelector` | The index, followed by the full text of every usable skill. Costs context for every skill, used or not. |
| `llm` | `LLMSelector` | Each time the agent starts working on a task, before its first model call, a separate model call is given the index and the task and picks the skills it needs. Those are added in full, or the one section that applies. If that call fails, times out or picks nothing, the turn goes on with the index only. |

```python
from domyn.agents.skills import LLMSelector

SkillsRuntime(source=LocalDirSkillSource(paths=["./skills"]), selector=LLMSelector(max_expansions=2))
```

```yaml
    skills:
      source: {type: local_dir, paths: [./skills]}
      selector: {type: llm, max_expansions: 2}
```

With every selector, the whole index stays in the request, so the model can still open a skill the selector did not pick.

### Options

`SkillsRuntime`:

| Option | Type | Default | Description |
|---|---|---|---|
| `source` | `LocalDirSkillSource` \| `InMemorySkillSource` | required | Where the skills come from. In YAML: `type: local_dir` with `paths`, or `type: memory` with `skills`. |
| `selector` | selector | `IndexSelector` | See [Choosing skills before the turn](#choosing-skills-before-the-turn). |
| `index_sections` | `bool` | `False` | Lists each skill's sections in the index. Useful when the model should ask for one section without opening the skill first; costs context on every request. |
| `budget` | `SkillBudget` | see below | Limits on how much skill text enters the request. |
| `search` | `SkillSearch` | see below | How `skill_search` searches. |

`SkillBudget`:

| Option | Type | Default | Description |
|---|---|---|---|
| `max_index_chars` | `int` | `10000` | Size of the index. Over it, skills are shown in a shorter form: see [When the index does not fit](#when-the-index-does-not-fit). |
| `max_description_chars` | `int` | `120` | Length a description is cut to when the index does not fit with whole ones. |
| `max_body_chars` | `int` | `20000` | Total size of the skills added in full before the turn. |
| `max_expansions_per_turn` | `int` | `5` | How many skills can be added in full before the turn. |

`SkillSearch`, under `search`:

| Option | Type | Default | Description |
|---|---|---|---|
| `provider` | `str` | the agent's provider | Name of a registered provider to search with, as for `LLMSelector`. |
| `timeout_s` | `float` | `10.0` | Seconds to wait for the model before matching words instead. |
| `generation_params` | `dict` | `{}` | Passed to the provider with the request. With a reasoning model, turn its thinking off here, as the provider spells it: for Qwen3 served by vLLM, `{chat_template_kwargs: {enable_thinking: false}}`. |
| `max_candidates` | `int` | `1000` | Past this many skills, the model is shown only the best word matches. |
| `description_chars` | `int` | `120` | Length descriptions are cut to, in what the model searches and in the results. |
| `instruction` | `str` | built in | Replaces the prompt of the search call. `{limit}` is replaced with the number of results asked for. |

```yaml
    skills:
      source: {type: local_dir, paths: [./skills]}
      search: {provider: small_model, timeout_s: 5}
```

`LLMSelector`:

| Option | Type | Default | Description |
|---|---|---|---|
| `provider` | `str` | the agent's provider | Name of a provider registered with [`@llm_provider`](llm-providers.md), to pick skills with a smaller, faster model. A provider declared under `llm_providers` in a YAML file is not registered, so it cannot be named here. An unknown name falls back to the agent's provider, with a warning in the log. |
| `max_expansions` | `int` | `3` | How many skills, or sections, it may pick. |
| `timeout_s` | `float` | `10.0` | Seconds to wait for the pick before going on with the index only. |
| `instruction` | `str` | built in | Replaces the prompt of the picking call. `{limit}` is replaced with `max_expansions`. |

## Required skills

A skill with `enforcement: required` is a rule the agent must not run without, such as a compliance policy. If its requirements are not met, the turn is refused before the agent does anything: the stream is `agent_start` followed by `error_occurred`, with `error_code` `RequiredSkillUnavailable` and a message naming the skill and what is missing:

```text
Cannot run this turn: required skill(s) unavailable for agent 'Advisor': compliance (unavailable: env 'COMPLIANCE_RULES_URL')
```

The check covers every agent of the team, including those the turn would not have reached. Fixing the requirement is enough: the next message runs normally.

An `advisory` skill that is not usable never blocks a turn: it is marked in the index, and no selector adds it in full.

## In the interactive CLI

The [interactive CLI](../cli/sessions.md) loads a skill only because its configuration names it: the `paths` of the agent's `skills` source, in `~/.domyn/config.yaml` or in the file passed to `domyn run`. Nothing else is searched, and there is no option that adds a directory. When two skills share a name, the one listed earlier wins.

`domyn setup` fills that list. It asks about three kinds of skill, built-in ones first, and writes them in this order:

1. **The project's own skills.** One yes/no question: load skills from `./.domyn/skills` and `./.agents/skills` in the folder `domyn run` starts in. Yes writes both paths as they are, relative, so they are read again on every turn from whichever folder the session starts in, and a skill added to either is loaded from the next message. When both hold a skill of the same name, `./.domyn/skills` wins. It puts the text of a repository you cloned into the prompt, which is why it is a question.
2. **A folder of your own skills.** The wizard reads the folder once and lists each skill in it by its absolute path. Nothing is copied: each skill is read from where it is. A skill added to that folder later is loaded only after you run `domyn setup` again and give the folder again.
3. **The built-in skills.** A checklist, by group. The ones you tick are copied into `~/.domyn/builtin-skills` and listed by absolute path. **Every `domyn setup` deletes and rewrites `~/.domyn/builtin-skills`**, so an edit made there is lost and nothing of yours belongs there; to change a built-in skill, untick it and keep your own copy in `./.domyn/skills`. The skills inside the package are not a fallback: a configuration that lists none has none.

So a project skill wins over a skill of yours, and both win over a built-in of the same name.

A relative path is read from the folder `domyn run` starts in. When a listed path goes through a symbolic link, or is not there, `domyn run` says so when it starts, except that a missing `./.domyn/skills` or `./.agents/skills` is not reported, because most folders have neither. It also names any listed skill whose `external_files` are missing.

`--expand-skills` reads every skill in full up front (the `all` selector), and `--skill-sections` lists sections in the index. The skill directories are also reachable by the session's file tools, so you can ask the agent to edit a skill; writing still asks for your approval.

Inside a session, `/skills` lists the skills with their files and sections, and `/skill <name>` shows one in full.

From the shell:

| Command | What it does |
|---|---|
| `domyn skills` | Lists the skills the configuration loads, with their files, and why any is unavailable. |
| `domyn skill show <name>` | Prints one skill the global `~/.domyn/config.yaml` loads, with its file and sections. `<name>#<section>` prints one `##` section. |
| `domyn skill remove <name>` | Removes the entry that names the skill, its folder or its `.md` file, from `~/.domyn/config.yaml`. The files stay where they are. A skill loaded through a listed folder, such as `./.domyn/skills`, has no entry of its own, so nothing is removed and the command says how to stop loading it. |

`--project`, for a project's own `./.domyn/config.yaml`, is not supported.

## Next steps

- [Standing instructions](standing-instructions.md): instructions that apply to every message.
- [YAML configuration](configuration.md): declaring skills in a YAML file.
- [Sessions and commands](../cli/sessions.md): skills in a terminal session.
