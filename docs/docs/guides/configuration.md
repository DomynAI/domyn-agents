---
title: YAML configuration
sidebar_label: YAML configuration
description: "Describe an application in a YAML file: providers, strategies, tools, hooks and agents, and how to load and run it"
---

# YAML configuration

An application can be described in a YAML file instead of Python code: the model providers, the planner strategies, the tools, the hooks and the agents, each declared once and referred to by name. The file is loaded into the same objects you would build in Python, so everything on the other pages applies to agents loaded from YAML too.

This page shows how a file is organized and how to run it. [YAML configuration](../reference/yaml-schema.md) lists every key.

## A complete file

```yaml
version: "1.0"
entrypoint: Assistant

llm_providers:
  - name: main_llm
    type: openai
    model_name: gpt-4o
    api_key: ${OPENAI_API_KEY}

planner_strategies:
  - name: default
    type: tool_use_thought
    max_iterations: 15

tools:
  - name: get_order_status
    type: function
    path: tools/orders.py

agents:
  - name: OrderAgent
    description: Looks up the status of an order.
    instruction: Answer questions about orders using your tools.
    llm_provider: main_llm
    planner_strategy: default
    tools:
      - get_order_status

  - name: Assistant
    description: Front desk of the shop.
    instruction: Send order questions to OrderAgent and answer from what it found.
    llm_provider: main_llm
    planner_strategy: default
    sub_agents:
      - OrderAgent
```

Sections are lists of named entries, and agents refer to them by name: `llm_provider: main_llm`, `planner_strategy: default`, `tools: [get_order_status]`, `sub_agents: [OrderAgent]`. Order within the file does not matter: an agent can list a sub-agent declared after it.

| Key | Description |
|---|---|
| `version` | `"1.0"`, in quotes: an unquoted `1.0` is read as a number and rejected. Can be omitted. |
| `entrypoint` | The agent the user talks to. If omitted, the loader takes the last agent once agents are ordered so that each comes after its sub-agents; name it explicitly when the file has more than one top-level agent. |
| `llm_providers` | Model connections. See [LLM providers](llm-providers.md). |
| `planner_strategies` | How agents reason and call tools. See [Planner strategies](planner-strategies.md). |
| `context_managers` | Custom context managers for the strategies. See [Planner strategies](planner-strategies.md). |
| `tools` | Python functions and single MCP tools. See [Tools](#tools). |
| `tool_bundles` | The built-in file and shell tools. See [Built-in tools](#built-in-tools). |
| `mcp_servers`, `oauth_providers` | Remote MCP servers whose tools agents use. See [MCP servers](#mcp-servers). |
| `hooks` | Code that runs around agents and tools. See [Hooks](#hooks). |
| `agents` | The agents. See [Agents](#agents). |

## Running a file

From the command line, `domyn run` opens an interactive session with the entrypoint:

```bash
domyn run assistant.yaml
```

See [CLI commands](../reference/cli.md).

In Python, `load_application_from_file` returns the entrypoint agent, ready for a [`Runner`](events-and-streaming.md):

```python
from domyn.agents import load_application_from_file

agent = await load_application_from_file("assistant.yaml")
```

It also connects to the MCP servers the agents use and fetches their tools. To build the agent without contacting them, load the file yourself and pass `mcp_init=False`:

```python
from domyn.agents.loader import load_application_from_config_new, load_from_file

agent = await load_application_from_config_new(load_from_file("assistant.yaml"), mcp_init=False)
```

Then call `await agent.initialize_mcp_tools()` on each agent before it needs them.

## Secrets and environment variables

Any string in the file can contain `${NAME}`, replaced with the environment variable `NAME` when the file is loaded:

```yaml
llm_providers:
  - name: main_llm
    type: vllm
    model_name: Qwen/Qwen3-8B
    url: http://localhost:8000/v1/chat/completions
    api_key: ${VLLM_API_KEY}
```

If the variable is not set, loading stops with an error naming it, so a missing key is never sent as the literal text `${VLLM_API_KEY}`.

The `domyn` command first reads a `.env` file from the current directory or the nearest parent that has one, so keys can live in a file kept out of version control:

```bash
# .env
VLLM_API_KEY=sk-...
```

A variable already set in the environment takes precedence over the same name in `.env`. In Python, `load_application_from_file` reads only the environment: load the `.env` yourself first, for example with `python-dotenv`.

## Tools

A function tool points to a Python file:

```yaml
tools:
  - name: get_order_status
    type: function
    path: tools/orders.py
```

- `name` is the name of the function in that file, and it is also the name the model sees. The model reads the function's docstring as the description.
- `path` is resolved against the current directory, not against the YAML file: run from the directory the paths are written for.
- `approval_type: user_confirm` makes every call wait for a person's approval. See [Tool approvals](approvals.md).

A [streaming tool](tools.md), an async generator, uses `type: StreamableFunctionTool` with the same keys.

A single tool of an MCP server:

```yaml
tools:
  - name: search
    type: mcp
    server_url: https://api.example.com/mcp
```

`name` is the tool's name on the server.

A tool registered in Python with `@tool()` can be used by name: list it in an agent's `tools` directly, or declare it with `reference` to add hooks:

```yaml
tools:
  - name: lookup_order
    reference: lookup_order
    hooks: [clean_args]
```

The Python module that registers it must be imported before the file is loaded.

The `type` of an entry under `tools` is one of `function`, `StreamableFunctionTool` and `mcp`, spelled exactly so. Two more exist for other uses: `rest`, a call to an HTTP endpoint, and `delegate`, a tool the model sees but that runs in another process. An entry with any other `type` is skipped, with a warning in the log, and the agents that list it load without it.

### Built-in tools

The built-in [file and shell tools](../cli/tools-and-approvals.md) are added in bundles. Declare the bundle once and list it in each agent that should have it:

```yaml
tool_bundles:
  - name: workspace_tools
    bundles: [files, shell]
    workspace: ./project

agents:
  - name: Coder
    ...
    tool_bundles: [workspace_tools]
```

`files` adds `read_file`, `list_dir`, `glob_files`, `grep`, `write_file`, `append_file` and `edit_file`; `shell` adds `run_command`, `check_command`, `list_commands` and `stop_command`. Without `bundles`, both are added. The tools can reach only the `workspace` directory (the current directory by default), and writing files or running commands needs approval.

## MCP servers

To give an agent every tool of an MCP server, declare the server and list it in the agent's `mcp_servers`:

```yaml
mcp_servers:
  - name: analytics
    url: https://analytics.example.com/mcp
    headers:
      Authorization: Bearer ${ANALYTICS_TOKEN}

agents:
  - name: Analyst
    ...
    mcp_servers: [analytics]
```

Clients connect with the classic `initialize` handshake (`protocol_mode: legacy`). Set `protocol_mode: auto` to let the client try the newer protocol revision first; a server that enforces `Mcp-Param-*` headers, such as GitHub's, then rejects calls with "header mismatch: missing Mcp-Param-... header".

For a server that requires OAuth, set `use_oauth: true`. To choose the scopes or the client name, or to keep the tokens between runs, declare an OAuth provider and name it:

```yaml
oauth_providers:
  - name: company_sso
    scopes: [read, write]
    client_name: Support Assistant

mcp_servers:
  - name: tickets
    url: https://tickets.example.com/mcp
    use_oauth: true
    oauth_provider: company_sso
```

A named provider keeps the tokens in `~/.domyn/mcp_oauth/` by default, so the user signs in again only when they can no longer be refreshed. Without `oauth_provider`, the tokens are kept in memory only, and the user signs in again each time the application starts. Servers that name the same provider and share a URL share one sign-in. See [Tools](tools.md) for what happens at sign-in.

## Hooks

Hooks are declared once under `hooks`, each with the phase it runs in (`trigger`), and attached by name to agents or tools:

```yaml
hooks:
  - name: log_start
    type: listener
    trigger: pre_agent_call
    path: hooks.py
  - name: clean_args
    type: updater
    trigger: pre_tool_call
    path: hooks.py

tools:
  - name: get_order_status
    type: function
    path: tools/orders.py
    hooks: [clean_args]

agents:
  - name: OrderAgent
    ...
    hooks: [log_start]
```

| Key | Description |
|---|---|
| `name` | The function's name in `path`, and the name agents and tools use. |
| `path` | The Python file, resolved against the current directory. |
| `type` | `listener` (observes, runs concurrently) or `updater` (can change what it receives). |
| `trigger` | `pre_agent_call`, `post_agent_call` or `on_agent_event` for an agent; `pre_tool_approval`, `pre_tool_call` or `post_tool_call` for a tool. |

A hook registered in Python with `@hook` is declared with `reference` instead of `path` and `type`. The function's parameters depend on the phase. See [Hooks](hooks.md).

## Agents

```yaml
agents:
  - name: OrderAgent
    description: Looks up the status of an order.
    instruction: Answer questions about orders using your tools.
    llm_provider: main_llm
    planner_strategy: default
    stream: true
    tools: [get_order_status]
    sub_agents: [RefundAgent]
    context_visibility: standard
    handoff_scope: explicit
```

`name`, `description`, `instruction` and `llm_provider` are required. `llm_provider` and `planner_strategy` name entries of the file, or a provider or strategy registered in Python. An agent without `planner_strategy` uses `tool_use_thought` with its defaults.

Keys that are not recognized are ignored without an error.

An agent can also declare:

- `skills`, the skill directories it reads (see [Skills](skills.md));
- `standing_instructions`, its `SOUL.md` and `AGENTS.md` (see [Standing instructions](standing-instructions.md)).

```yaml
    skills:
      source:
        type: local_dir
        paths: [./skills]
    standing_instructions:
      source:
        type: local_dir
        paths: [./instructions]
```

An agent that runs in another process is declared with `type: delegate` and a `transport`. See [Remote agents](remote-agents.md).

## Next steps

- [YAML configuration](../reference/yaml-schema.md): every key, with its type and default.
- [Orchestration](orchestration.md): how the agents of a team work together.
- [Events and streaming](events-and-streaming.md): run the loaded agent from your own code.
