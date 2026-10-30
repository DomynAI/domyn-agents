---
title: Planner strategies
sidebar_label: Planner strategies
description: "Choose how an agent reasons and calls tools, and configure its iteration limit, context and compaction"
---

# Planner strategies

A planner strategy runs one agent's turn. On each iteration it builds the request from the conversation history (through its **context manager**), calls the model, and executes what the model returns: a tool call, a handoff to another agent, or a final reply. Tool results and sub-agent results are appended to the history, and the loop repeats until the model replies or `max_iterations` is reached. This is the ReAct loop: reason, act, observe.

The three built-in strategies run the same loop and differ in how the model is asked to act.

## Choosing a strategy

| `type` | Class | How the model calls tools | Use it when |
|---|---|---|---|
| `tool_use_thought` | `ToolUseWithThoughtPlannerStrategy` | Native tool calling, with the model's reasoning recorded before each call | **Default.** Any model with native tool calling. |
| `tool_use` | `ToolUsePlannerStrategy` | Native tool calling, without the reasoning | You do not need the reasoning. Tools have no extra `reasoning` argument, and the system prompt is shorter by one instruction. |
| `react` | `ReactPlannerStrategy` | A text format described in the system prompt, parsed from the model's reply | The model, or the endpoint serving it, has no native tool calling. |

**Reasoning.** `tool_use_thought` adds a required `reasoning` argument to every tool and agent the model can call, and an instruction to the system prompt asking the model to state its reasoning before each call. The `reasoning` argument is recorded as the `thought` of the action, on the `TOOL_START` or `AGENT_CALL` event, and the [interactive CLI](../cli/sessions.md) prints it above the call. When the first call of a reply has no `reasoning`, its thought comes from the provider's reasoning field (`reasoning_content`, as with DeepSeek R1, then `thinking_blocks`, as with Anthropic extended thinking), and otherwise from the text the model wrote before the call. With `tool_use`, `thought` is empty.

**Text protocol.** With `react`, every step must be written in the exact format the prompt describes (`<thought>…</thought><action>{"name": …, "arguments": …}</action>`). A reply outside the format is a step the planner cannot parse, so prefer a `tool_use` strategy whenever native tool calling is available.

## Setting the strategy

An `Agent` without a `planner` uses `ToolUseWithThoughtPlannerStrategy()` with its defaults. To choose another strategy, or to change an option, pass one explicitly:

```python
from domyn.agents.planner_strategy import ToolUsePlannerStrategy

agent = Agent(..., planner=ToolUsePlannerStrategy(max_iterations=15))
```

`ReactPlannerStrategy()` needs nothing more: its default context manager sends the whole history and asks the model for the `<thought>`/`<action>` text format.

```python
from domyn.agents.planner_strategy import ReactPlannerStrategy

agent = Agent(..., planner=ReactPlannerStrategy())
```

In a [YAML configuration](configuration.md), strategies are declared under `planner_strategies` and referenced by name from each agent. An agent that names none gets `tool_use_thought` with its defaults, so the section is needed only to choose another strategy or change an option:

```yaml
planner_strategies:
  - name: default
    type: tool_use_thought
    max_iterations: 15
  - name: text_protocol
    type: react

agents:
  - name: Assistant
    ...
    planner_strategy: default
```

Several agents can reference the same strategy. In a [team](orchestration.md), each agent can use a different one.

## Options

### All strategies

| Option | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | the `type` | Name the strategy is referenced by in YAML. |
| `max_iterations` | `int` | `25` | Maximum model calls in one turn of this agent. When reached, the agent stops and replies `Reached maximum iterations (<n>)`. |
| `forward_verbatim_enabled` | `bool` | `True` | Offers the model the `FORWARD_VERBATIM` tool, which returns a previous tool or sub-agent result unchanged as the agent's reply. See [Orchestration](orchestration.md). |
| `use_stop` | `bool` | `False` | For a sub-agent. Offers two extra tools: `STOP(info)`, which ends the agent's work and returns `info` to the calling agent, and `ASK_USER_MISSING_INFO(text)`, which ends it with a request for information. The calling agent receives the question with an instruction to ask the user and call the sub-agent again. |

### `tool_use` and `tool_use_thought`

| Option | Type | Default | Description |
|---|---|---|---|
| `context_manager` | `ToolUseContextManager` / `ToolUseWithThoughtContextManager` | a new instance, with the defaults below | Builds the messages and tool definitions sent to the model. |
| `compactor` | `Compactor` | `two_phase`, reserving 16,384 tokens | Shortens the conversation before a request that would not fit the model's context window. See [Context compaction](context-compaction.md). |
| `parallel_tool_calls` | `bool` | `True` | Runs the calls of one model response that need no approval at the same time; `False` runs every call after the one before it. See [Several calls at once](tools.md#several-calls-at-once). |

Context manager options:

| Option | Type | Default | Description |
|---|---|---|---|
| `max_tool_result_chars` | `int` | `20000` | Maximum characters of a single tool result sent to the model. A longer result is cut keeping its start and its end, with a line saying which characters are missing, that what they contain cannot be told from the rest, and how to read them: an agent with tools of its own, or agents to hand off to, also gets a `read_tool_result` tool that reads a result from its history, a part that fits this limit at a time, or lists where a text occurs in it. |

```python
from domyn.agents.planner_strategy import (
    ToolUseWithThoughtContextManager,
)
from domyn.agents.planner_strategy import ToolUseWithThoughtPlannerStrategy

planner = ToolUseWithThoughtPlannerStrategy(
    context_manager=ToolUseWithThoughtContextManager(max_tool_result_chars=8000),
)
```

An agent with tools of its own, or agents to hand off to, gets the `read_tool_result` definition with every request, about 970 characters. An agent with neither does not, so it sends no tools because of it (a vLLM server started without `--enable-auto-tool-choice` rejects a request with any), and its cut results do not say how to read the rest. Neither do those of an agent with a tool of its own named `read_tool_result`, which keeps its own.

In YAML, context managers are declared under `context_managers` and referenced by name. The `type` is `thought_tool_use` for `tool_use_thought` and `tool_use` for `tool_use`:

```yaml
context_managers:
  - name: short_results
    type: thought_tool_use
    max_tool_result_chars: 8000

planner_strategies:
  - name: default
    type: tool_use_thought
    context_manager: short_results
```

### `react`

| Option | Type | Default | Description |
|---|---|---|---|
| `context_manager` | `ContextManager` | `full_messages` with `structural_tags` prompting | Builds the prompt and the history, and defines the text format the model must follow. |
| `verbose` | `bool` | `False` | When the model returns no action, also emits a `RESPONSE` event carrying a `thought` that says the agent is finishing. |

`react` does not support `compactor`: a `react` strategy declared with one is rejected when it is loaded.

In YAML, a context manager for `react` is declared under `context_managers` like the ones above, and needs more keys than its Python default suggests:

- `prompting`, the text format: `{type: structural_tags}` or `{type: json_schema}`. Required.
- For `type: full_messages`, `narrow_non_visible_agents`: `true` leaves out the tool results of agents this one cannot see, `false` keeps them. Required.
- For the summarizing types (`old_summary`, `agents_summary`, `interactions_summary`), `llm_provider`: the name of an entry of `llm_providers`, the model that writes the summaries. Required.

```yaml
context_managers:
  - name: full
    type: full_messages
    prompting: {type: structural_tags}
    narrow_non_visible_agents: false

planner_strategies:
  - name: react
    type: react
    context_manager: full
```
