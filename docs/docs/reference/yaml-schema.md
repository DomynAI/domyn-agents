---
title: YAML configuration
sidebar_label: YAML configuration
description: "Every key of a YAML application file, with its type and default"
---

# YAML configuration

Every key of a YAML application file. How to write and run one is in [YAML configuration](../guides/configuration.md).

Any string value can contain `${NAME}`, replaced with the environment variable `NAME` when the file is loaded; an unset variable stops the load with an error. Relative paths (`path`, a resource's `paths`) are resolved against the current directory.

## Top level

| Key | Type | Default | Description |
|---|---|---|---|
| `version` | `"1.0"` | `"1.0"` | Format version. Must be the string `"1.0"`: unquoted, `1.0` is a number, and the load fails with `Unsupported configuration version`. |
| `entrypoint` | `str` | the last agent, in dependency order | Name of the agent the user talks to. |
| `llm_providers` | list | `[]` | See [`llm_providers`](#llm_providers). |
| `planner_strategies` | list | `[]` | See [`planner_strategies`](#planner_strategies). |
| `context_managers` | list | `[]` | Context managers referenced by name from `planner_strategies`. Each needs `name` and `type`, and the `react` ones more keys. See [Planner strategies](../guides/planner-strategies.md). |
| `tools` | list | `[]` | See [`tools`](#tools). |
| `resources` | list | `[]` | See [`resources`](#resources). |
| `tool_bundles` | list | `[]` | See [`tool_bundles`](#tool_bundles). |
| `mcp_servers` | list | `[]` | See [`mcp_servers`](#mcp_servers). |
| `oauth_providers` | list | `[]` | See [`oauth_providers`](#oauth_providers). |
| `hooks` | list | `[]` | See [`hooks`](#hooks). |
| `agents` | list | required | See [`agents`](#agents). |
| `metadata` | mapping | none | Free-form keys for the application that reads the file. The loader keeps them and does nothing with them; the interactive CLI's `/config` shows their names only. |
| `final` | `bool` | `false` | Kept with the configuration as a marker for the application. Nothing in the framework enforces it. |

## `llm_providers`

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | the `model_name` | Name agents refer to. |
| `type` | `domyn` \| `openai` \| `anthropic` \| `vllm` \| `cerebras` \| `open_router` \| `custom` | required, unless `reference` | The provider. |
| `model_name` | `str` | required | The model. |
| `reference` | `str` | | Instead of `type`: the name of a provider registered in Python with `@llm_provider`, copied under this entry's `name`. |
| `config_params` | `dict` | | With `reference`: options to change on the copy; dictionaries are merged. With `type`: options added to the entry's own. Any option except `type`, `model_name`, `name`, `url` and `reference`. |

The other keys depend on `type` (`url`, `api_key`, `generation_params`, `context_window`, …). See [LLM providers](../guides/llm-providers.md).

## `planner_strategies`

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | the `type` | Name agents refer to. |
| `type` | `tool_use_thought` \| `tool_use` \| `react` | required | The strategy. |
| `context_manager` | `str` | per type | Name of an entry of `context_managers`. |

The other options (`max_iterations`, `compactor`, …) are listed in [Planner strategies](../guides/planner-strategies.md).

## `tools`

### `type: function` and `type: StreamableFunctionTool`

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Name of the function in `path`, which is also the name the model sees. |
| `type` | `function` \| `StreamableFunctionTool` | required | `StreamableFunctionTool` for an async generator. See [Tools](../guides/tools.md). |
| `path` | `str` | required | The Python file that defines the function. |
| `approval_type` | `user_confirm` \| `params_update` | none | Each call waits for approval. See [Tool approvals](../guides/approvals.md). |
| `hooks` | list of `str` | `[]` | Names of `pre_tool_approval`, `pre_tool_call` and `post_tool_call` hooks. |

The description the model reads is the function's docstring.

### `type: mcp`

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | The tool's name on the server. |
| `type` | `mcp` | required | |
| `server_url` | `str` | required | The server's streamable HTTP endpoint. |
| `headers` | `dict[str, str]` | none | HTTP headers sent to the server, for example `Authorization`. |
| `use_oauth` | `bool` | `false` | Sign in with OAuth. |
| `oauth_provider` | `str` | the in-memory default | Name of an entry of `oauth_providers`. |
| `approval_type` | `user_confirm` \| `params_update` | none | Each call waits for approval. |
| `hooks` | list of `str` | `[]` | Names of `pre_tool_approval`, `pre_tool_call` and `post_tool_call` hooks. |

### `reference`

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Name agents refer to. |
| `reference` | `str` | required | A tool registered in Python with `@tool()`. |
| `hooks` | list of `str` | `[]` | Hooks to attach. |

Two more types exist: `rest`, an HTTP endpoint called as a tool, and `delegate`, a tool declared here and run by the application, used with [remote agents](../guides/remote-agents.md).

An entry whose `type` is unknown, or that cannot be built from its keys, is skipped with a warning in the log naming the tool and its type, and the agents that list it are loaded without it.

## `resources`

Places files live, each with an id the rest of the file refers to. A resource does nothing on its own: a tool bundle, a skills source or an instructions source that names it uses it. The id never comes from an agent's name, so renaming an agent does not move its files.

| Key | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | required | Unique in the file, and different from every agent's name. |
| `type` | `local_dir` | required | The kind of place. Applications register types of their own. |
| `paths` | `str` or list of `str` | required | `local_dir`: folders on this machine, in order. The first is the primary one: a tool bundle's workspace, where writes go. The others are read too, and the earlier one wins when two hold the same file. An absolute path to a folder that does not exist is a warning when the file loads. |
| `read_only` | `bool` | `false` | Enforced by the resource: the `files` tools refuse to write to it, and a resource type that is mounted is mounted read-only. A `local_dir` cannot keep commands from writing: a `shell` bound to it still runs them, each asking for approval. |

The load stops with an error naming what to change for: two resources with one id, a resource id equal to an agent's name, an id no resource declares, a bundle bound to no resource or to one that is not a folder on this machine, and a skills or instructions source of `type: local_dir` (what an agent reads is always a resource). A folder that does not exist is a warning when it is an absolute path; a relative one names a folder of wherever the process runs, which may well lack it.

## `tool_bundles`

A bundle names no folder: each agent binds it to a resource, `{use: <bundle>, workspace: <resource id>}`, so one bundle serves agents that work in different places.

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Name agents refer to with `use`. |
| `bundles` | list of `files` \| `shell` \| `browser` | `[files, shell]` | Which tools: `files` (`read_file`, `list_dir`, `glob_files`, `grep`, `write_file`, `append_file`, `edit_file`), `shell` (`run_command`, `check_command`, `list_commands`, `stop_command`), `browser`. |
| `sandbox` | `local` | `local` | Where commands run. |
| `isolate_environment` | `bool` | `true` | Commands see only a fixed set of environment variables (`PATH`, `HOME`, …), plus `allowed_env`. |
| `allowed_env` | list of `str` | `[]` | Further environment variables commands may see. |

## `mcp_servers`

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Name agents refer to in `mcp_servers`. |
| `url` | `str` | required | The server's streamable HTTP endpoint. |
| `tool_autodiscovery` | `bool` | `true` | Fetch the server's tools when the agent is initialized. |
| `headers` | `dict[str, str]` | none | HTTP headers sent to the server. |
| `use_oauth` | `bool` | `false` | Sign in with OAuth. |
| `oauth_provider` | `str` | the in-memory default | Name of an entry of `oauth_providers`. |
| `timeout` | `float` | `30.0` | Seconds to wait for the server. |

## `oauth_providers`

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Name servers and tools refer to. `__default__` is reserved. |
| `type` | `local` \| `in_memory` | `local` | `local` keeps the tokens on disk, `in_memory` only for the life of the process. |
| `storage_dir` | `str` | `~/.domyn/mcp_oauth` | Where `local` keeps the tokens. |
| `scopes` | list of `str` | `[read]` | Scopes to request. |
| `client_name` | `str` | `Domyn Agent` | Name the client registers with. |
| `client_id` | `str` | none | A client registered in advance with the server. Without it, the client registers itself. |
| `client_secret` | `str` | none | Secret of `client_id`, when the server requires one. |
| `callback_port` | `int` | `45678` | Local port of the sign-in callback, `http://localhost:<port>/callback`. |

A server or tool with `use_oauth` and no `oauth_provider` uses an in-memory provider with the defaults above.

## `hooks`

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Name of the function in `path`, and the name agents and tools refer to. |
| `type` | `listener` \| `updater` | required, unless `reference` | `listener` observes and runs concurrently; `updater` can change what it receives. |
| `trigger` | `pre_agent_call` \| `post_agent_call` \| `on_agent_event` \| `pre_tool_approval` \| `pre_tool_call` \| `post_tool_call` | required | When it runs. The first three attach to agents, the last three to tools. |
| `path` | `str` | required, unless `reference` or `url` | The Python file that defines the function. |
| `url` | `str` | | Instead of `path`: an HTTP endpoint called as the hook. |
| `method` | `GET` \| `POST` \| `PUT` \| `DELETE` \| `PATCH` | `POST` | With `url`: the HTTP method. |
| `headers` | `dict[str, str]` | none | With `url`: HTTP headers sent with each call. |
| `timeout` | `float` | `300.0` | With `url`: seconds to wait for the endpoint. |
| `reference` | `str` | | Instead of `path` and `type`: a hook registered in Python with `@hook`. |
| `break_on_error` | `bool` | `true` | Whether the hook raising stops the turn. |
| `silent` | `bool` | `false` | Keep the events the hook produces out of the conversation history. They are still emitted. |
| `on_event_triggers` | list of event types | see [Hooks](../guides/hooks.md) | For `on_agent_event`: the event types it runs on, such as `tool_end`. |

See [Hooks](../guides/hooks.md).

## `agents`

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | 1 to 30 characters. |
| `description` | `str` | required | What the agent does, at most 1,000 characters. In a team, the main agent chooses a member from it. |
| `instruction` | `str` | required | The system instruction. |
| `llm_provider` | `str` | required | Name of an entry of `llm_providers`, or of a provider registered in Python. |
| `planner_strategy` | `str` | `tool_use_thought` with its defaults | Name of an entry of `planner_strategies`, or of a strategy registered in Python. |
| `max_iterations` | `int` | the strategy's | Kept on the agent, but not read by any strategy: the limit on a turn is the `max_iterations` of its `planner_strategy`. |
| `memory_manager` | mapping | none | Long-term memory: `type` (`filesystem` or `remote`), `execution_mode` (`automatic` or `agentic`) and the manager's options. See [Conversations and memory](../guides/conversations-and-memory.md). |
| `stream` | `bool` | `false` | Stream the reply. See [Events and streaming](../guides/events-and-streaming.md). |
| `tools` | list of `str` | `[]` | Names of entries of `tools`, or of tools registered in Python. |
| `tool_bundles` | list of `{use, workspace}` | `[]` | Each entry mounts the bundle `use` names, working in the resource `workspace` names. A bare bundle name is refused: it names no folder. |
| `resources` | list of `str` | `[]` | Ids of resources the agent may use beyond those its bundles and sources name. Optional: naming a resource anywhere on the agent already grants it. An application reads the whole set from `Agent.resources`. |
| `mcp_servers` | list of `str` | `[]` | Names of entries of `mcp_servers`. |
| `sub_agents` | list of `str` | `[]` | The agents of its team. See [Orchestration](../guides/orchestration.md). |
| `context_visibility` | `standard` \| `peers` \| `global` | `standard` | What history the agent sees. See [Orchestration](../guides/orchestration.md). |
| `handoff_scope` | `explicit` \| `direct_peers` \| `global` | `explicit` | Which agents it may hand off to. See [Orchestration](../guides/orchestration.md). |
| `hooks` | list of `str` | `[]` | Names of `pre_agent_call`, `post_agent_call` and `on_agent_event` hooks. |
| `skills` | mapping | none | `source: {type: resource, resource: <id or list>, path: skills}` reads the `path` folder (default `skills`) under every folder of the resources named, earlier first. An entry of the list may be `{id, path}` to read another folder in that resource. `type: memory` holds the skills in the file. See [Skills](../guides/skills.md). |
| `standing_instructions` | mapping | none | `source: {type: resource, resource: <id or list>}` reads `SOUL.md` and `AGENTS.md` under every folder of the resources named (`path` for a subfolder); an entry `{id, walk: true}` reads its subfolders too, hidden and dependency folders skipped and no link followed. See [Standing instructions](../guides/standing-instructions.md). |
| `type` | `agent` \| `delegate` | `agent` | `delegate` for an agent that runs in another process. See [Remote agents](../guides/remote-agents.md). |

Keys that are not recognized are ignored without an error.
