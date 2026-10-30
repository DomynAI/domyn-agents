---
title: Tools
sidebar_label: Tools
description: "Give agents tools: Python functions, streaming tools and remote MCP servers"
---

# Tools

Tools are how an agent acts: it calls them to search, read data, call an API or change something. A tool can be a Python function, an async generator that streams its output, or a tool on a remote MCP server. The model decides when to call a tool from its name, its description and the description of its arguments, so those three matter as much as the code.

## Function tools

Wrap any Python function in a `FunctionTool`:

```python
from domyn.agents import FunctionTool


async def search_documents(query: str, max_results: int = 5) -> str:
    """Search the document store for relevant content.

    Args:
        query: The search query in natural language.
        max_results: Maximum number of results to return.
    """
    results = await perform_search(query, limit=max_results)
    return format_results(results)


tool = FunctionTool(func=search_documents)
```

Then pass it to an agent with `Agent(..., tools=[tool])`.

What the model sees is built from the function:

- **Name**: the function's name, `search_documents`.
- **Description**: the first part of the docstring, before the `Args:` section.
- **Arguments**: one per parameter, with the type taken from its annotation. A parameter without a default value is required.
- **Argument descriptions**: taken from the `Args:` section of the docstring, written in the Google style shown above. Once a docstring has an `Args:` section, it must describe every parameter: a parameter missing from it makes `FunctionTool(...)` raise a `ValidationError`. Without an `Args:` section, each parameter gets a generic description.

Instead of the `Args:` section, you can describe a parameter next to its type with `Annotated`:

```python
from typing import Annotated


def get_forecast(city: Annotated[str, "The city name"], days: int = 1) -> str:
    """Get the weather forecast for a city."""
    ...
```

The function can be sync or async. What it returns is what the model reads as the tool's result.

### Registering a tool with `@tool`

The `@tool()` decorator turns a function into a `FunctionTool` and registers it by name, so a [YAML configuration](configuration.md) can refer to it with `reference:`:

```python
from domyn.agents.core.decorators import tool


@tool()
async def search_documents(query: str) -> str:
    """Search documents."""
    ...
```

`name` and `description` override the defaults: `@tool(name="search", description="Search documents")`.

:::warning
Always write the parentheses. A bare `@tool` without them does not raise an error, but leaves a function that is not a tool.
:::

### Config parameters

Some arguments should come from your configuration, not from the model: an API key, a tenant id, an internal URL. Declare them as `ConfigParam`s:

```python
from domyn.agents.tools import ConfigParam


async def call_api(query: str, api_key: str) -> str:
    """Call an external API."""
    return await make_request(query, api_key=api_key)


tool = FunctionTool(
    func=call_api,
    config_params=[ConfigParam(name="api_key", value="sk-...", is_secret=True)],
)
```

A config parameter is left out of the arguments the model sees, so the model never knows it exists. Its value is passed to the function on every call.

### Requiring approval

A tool can require a person to approve each call before it runs:

```python
from domyn.agents import ApprovalType

refund_tool = FunctionTool(func=process_refund, approval_type=ApprovalType.USER_CONFIRM)
```

| `approval_type` | The person can |
|---|---|
| `ApprovalType.USER_CONFIRM` | Approve or refuse the call |
| `ApprovalType.PARAMS_UPDATE` | Also change the arguments before approving |

Instead of running the tool, the runner emits a `TOOL_NEEDS_FEEDBACK` event and waits for the answer. How to answer it, and how approvals can be remembered, is in [Tool approvals](approvals.md).

## Streaming tools

A tool that takes a while can show progress while it works. Write it as an async generator and wrap it in a `StreamableFunctionTool`:

```python
from domyn.agents import Part
from domyn.agents.tools.base import UpdateToolResult
from domyn.agents.tools import (
    StreamableFunctionTool,
    StreamableToolResult,
)


async def analyze_dataset(dataset_id: str):
    """Analyze a dataset."""
    yield StreamableToolResult(data=Part(text="Loading the dataset...\n"))
    data = await load_data(dataset_id)

    yield UpdateToolResult(data={"loaded": len(data)})
    result = await run_analysis(data)

    yield f"Analysis complete: {result.summary}"


tool = StreamableFunctionTool(func=analyze_dataset)
```

What the generator yields decides where it goes:

| Yielded value | Becomes |
|---|---|
| `StreamableToolResult(data=Part(text=...))` | Text shown to the user right away, as a partial `RESPONSE` event. The model reads it too, appended to the tool's result. |
| `UpdateToolResult(data={...})` | A `TOOL_UPDATE` event carrying the data, for a progress bar or a status line in your interface. The model does not see it. |
| Any other value | The tool's result, the one the model reads. When there are several, only the last one counts. |

When the agent runs as a sub-agent, or its planner runs with `use_stop=True`, its text chunks are not shown to the user: they are added to the tool's result instead.

## MCP tools

An agent can use the tools of a remote [MCP](https://modelcontextprotocol.io) server over streamable HTTP.

### Using every tool of a server

Declare the server on the agent, then call `initialize_mcp_tools()` to fetch its tools:

```python
from domyn.agents import Agent
from domyn.agents.tools.mcp import MCPServerStreamableHTTPConfig

agent = Agent(
    name="MyAgent",
    description="Agent with remote tools",
    instruction="...",
    llm_provider=llm,
    mcp_servers=[
        MCPServerStreamableHTTPConfig(
            name="analytics",
            url="https://analytics.example.com/mcp",
        )
    ],
)

await agent.initialize_mcp_tools()
```

After the call, `agent.tools` holds one tool for each tool of the server, with the name and description the server gives it. Until you call it, the agent has none of them, and nothing tells you so.

:::note
An agent loaded from a YAML file with `load_application_from_file` is already initialized. Call `initialize_mcp_tools()` only for an agent you build in Python.
:::

Set `tool_autodiscovery=False` to declare a server without fetching its tools.

### Using a single tool

To use one tool of a server, create an `MCPTool` and initialize it:

```python
from domyn.agents import MCPTool

search_tool = MCPTool(
    server_url="https://api.example.com/mcp",
    tool_name="search",
)
await search_tool.initialize()
agent.tools.append(search_tool)
```

You can also put it in `Agent(..., tools=[search_tool])` and let `agent.initialize_mcp_tools()` initialize it with the others.

### Authentication

For a server that takes a token or an API key, pass it as a header:

```python
MCPServerStreamableHTTPConfig(
    name="analytics",
    url="https://analytics.example.com/mcp",
    headers={"Authorization": "Bearer <token>"},
)
```

`MCPTool` takes the same `headers` argument.

For a server that requires OAuth, set `use_oauth=True`. The first time the agent connects, a browser window opens for you to sign in. With `use_oauth=True` alone, the tokens are kept in memory, so you sign in again each time the process starts. To keep them on disk, under `~/.domyn/mcp_oauth/`, use an OAuth provider of type `local`, as below.

```python
MCPServerStreamableHTTPConfig(
    name="secure_service",
    url="https://api.example.com/mcp",
    use_oauth=True,
)
```

By default the agent asks for the `read` scope and registers itself as "Domyn Agent". To change that, build the OAuth settings yourself and pass them as `oauth_provider`:

```python
from domyn.agents.tools.mcp.oauth_providers import OAuthProviderConfig, build_oauth_provider

url = "https://api.example.com/mcp"
oauth = OAuthProviderConfig(name="my_oauth", scopes=["read", "write"], client_name="My Agent")

MCPServerStreamableHTTPConfig(
    name="secure_service",
    url=url,
    use_oauth=True,
    oauth_provider=build_oauth_provider(oauth, url),
)
```

An `OAuthProviderConfig` is of type `local` unless you set `type="in_memory"`, so the tokens of this example are kept on disk, under `~/.domyn/mcp_oauth/`. In YAML, the same settings go in a named `oauth_providers:` entry. See [YAML configuration](configuration.md).

## Several calls at once

A model can ask for several tools in one answer: read three files, query two servers. The calls that need no approval run at the same time, so the turn waits for the slowest of them rather than for their sum, and their results reach the model together in its next request.

A call to a tool that requires approval runs alone and in order: everything the model asked for before it has finished, and nothing after it has started. Those are the calls another may depend on, such as writing a file and then running the tests, and an approval is asked about a call that has not begun. While the approval is pending, the calls after it wait; they run once it is answered, before the model is asked again. See [Tool approvals](approvals.md#several-calls-in-one-answer).

It is on by default, and an option of the planner strategy, so each agent decides. Turn it off and every call runs after the one before it, in order:

```python
from domyn.agents.planner_strategy import ToolUseWithThoughtPlannerStrategy

planner = ToolUseWithThoughtPlannerStrategy(parallel_tool_calls=False)
```

or, in YAML:

```yaml
planner_strategies:
  - name: one_at_a_time
    type: tool_use_thought
    parallel_tool_calls: false
```

How many run at the same time at most is the `Runner`'s: `Runner(max_parallel_tool_calls=4)` (default 8).

Tools that run together share the conversation's `state`, so a tool that changes it should not assume another one is not changing it at the same time.

## Tool hooks

A hook runs your code around every call of a tool: to check or clean the arguments, to change the result, or to log. Write it with `@hook` and attach it to the tool with `hooks=`:

```python
from domyn.agents.core.decorators import hook
from domyn.agents.hooks import HookPhase


@hook(trigger=HookPhase.PRE_TOOL_CALL)
def strip_query(tool, tool_args):
    tool_args["query"] = tool_args["query"].strip()
    return {"new_tool_args": tool_args}


@hook(trigger=HookPhase.POST_TOOL_CALL)
def shorten_result(tool, result):
    result.data = result.data[:2000]
    return result


@hook(trigger=HookPhase.PRE_TOOL_CALL, is_listener=True)
def log_call(tool, tool_args):
    print(f"Calling {tool.name} with query={tool_args['query']!r}")


tool = FunctionTool(func=search_documents, hooks=[strip_query, shorten_result, log_call])
```

| Trigger | Runs | The function receives | To change something, return |
|---|---|---|---|
| `PRE_TOOL_CALL` | Before the tool runs | `tool`, `tool_args` | `{"new_tool_args": tool_args}` |
| `POST_TOOL_CALL` | After the tool returns | `tool`, `result` (a `ToolResult`, with the return value in `result.data`) | The `ToolResult`, or `{"new_result": result}` |

The function must have parameters with these names: a hook missing one is refused when it is defined. `tool_args` holds the arguments of the call, together with a few keys the runner adds, such as `state` and `conversation_id`.

A hook is an **updater** by default: hooks of this kind run one after the other, each receiving what the previous one returned. With `is_listener=True` it is a **listener**: listeners run concurrently, after the updaters, and what they return is ignored.

`PRE_TOOL_CALL` runs after the decision to run the tool has been taken. A hook that must allow or refuse a call uses `PRE_TOOL_APPROVAL` instead. Every trigger, and hooks on agents, are covered in [Hooks](hooks.md).
