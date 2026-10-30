---
title: Hooks
sidebar_label: Hooks
description: "Run your own code around agents, tools and events: log, check, rewrite or stop a turn"
---

# Hooks

A hook is a function the runner calls at a fixed point of a turn: when an agent starts or finishes, before or after a tool runs, or on each event of an agent. Hooks are how you add logging, guardrails and rewrites without changing the agent: a hook can observe what happens, change the arguments of a tool call or its result, rewrite an event, or stop the turn with a message of its own.

## Writing a hook

A hook is a function decorated with `@hook`, with the phase it runs in as `trigger`:

```python
from domyn.agents.core.decorators import hook
from domyn.agents.hooks import HookPhase


@hook(trigger=HookPhase.PRE_TOOL_CALL)
def clean_args(tool, tool_args):
    tool_args["city"] = tool_args["city"].strip().title()
    return {"new_tool_args": tool_args}
```

and is attached to the agents or tools it applies to:

```python
from domyn.agents import FunctionTool

weather = FunctionTool(func=get_weather, hooks=[clean_args])
agent = Agent(..., tools=[weather], hooks=[log_start])
```

The function can be plain or `async`. Its parameter names are fixed by the phase, and checked when the hook is defined: a function missing one of them is refused. Extra parameters are allowed, for [`config_params`](#options).

## Phases

| `trigger` | Runs | Parameters | Attached to |
|---|---|---|---|
| `PRE_AGENT_CALL` | Each time the agent starts in a turn, including when a main agent resumes after a team member answered | `context`, `agent` | an agent |
| `POST_AGENT_CALL` | Each time the agent stops in a turn: when it has replied, and also when it hands off to a team member | `context`, `agent` | an agent |
| `PRE_TOOL_APPROVAL` | Before the approval gate of a tool that needs approval | `tool`, `tool_args` | a tool |
| `PRE_TOOL_CALL` | Before the tool runs | `tool`, `tool_args` | a tool |
| `POST_TOOL_CALL` | After the tool returns | `tool`, `result` | a tool |
| `ON_AGENT_EVENT` | On each event of the agent | `event`, `context` | an agent |

`context` is the agent's `InteractionContext` for the turn: `context.agent_input` is the event that started it, `context.conversation_history` what came before, and `context.state` the [conversation state](conversations-and-memory.md). `tool_args` are the arguments the tool will be called with, `result` is the tool's `ToolResult`, and `event` is a `BaseEvent` (see [Event types](../reference/events.md)).

`PRE_TOOL_APPROVAL` hooks decide whether a call may run without asking a person, and are covered in [Tool approvals](approvals.md).

## Updaters and listeners

| Kind | Declared with | Can change what it receives | Runs |
|---|---|---|---|
| Updater | `@hook(...)`, the default | yes | one after another, each receiving what the previous one returned |
| Listener | `@hook(..., is_listener=True)` | no | concurrently with the other listeners |

A listener returns nothing, or content to show (see [Showing content](#showing-content)). An updater returns the new version of what it received, as below. An agent or tool updater that returns nothing fails, like a hook that raises.

### Agent hooks

An updater `PRE_AGENT_CALL` or `POST_AGENT_CALL` hook returns the `context`, changed or not. To **stop the turn** before the agent runs, return it with `break_execution` and the message to show:

```python
@hook(trigger=HookPhase.PRE_AGENT_CALL)
def no_passwords(context, agent):
    if "password" in context.agent_input.action.parameters.get("task", ""):
        return {
            "new_context": context,
            "break_execution": True,
            "emitted_content": "I can't help with passwords.",
        }
    return context
```

The message reaches the user as a `response` event, also when the agent streams, and the turn ends there.

### Tool hooks

An updater `PRE_TOOL_CALL` hook returns `{"new_tool_args": tool_args}`. Besides the model's arguments, `tool_args` holds the values the runner passes to the tool itself, such as `state`, `run_config` and `conversation_id`, so change the dictionary you received and return all of it, as in `clean_args` above.

An updater `POST_TOOL_CALL` hook returns the new `ToolResult`, or `{"new_result": ...}` with a `ToolResult` or a dictionary of its fields. The model reads the new result:

```python
@hook(trigger=HookPhase.POST_TOOL_CALL)
def shout(tool, result):
    result.data = result.data.upper()
    return result
```

A tool hook cannot stop the call.

### Event hooks

An `ON_AGENT_EVENT` hook runs on the events of its agent: the input it starts from (the user's message, or the task from the agent that called it), its hand-offs, its tool calls and results, and its reply. On the main agent it also runs on its `agent_end`. It sees events of its own agent that are not passed to the stream: on a team member, its reply to the main agent, which reaches the stream only as the member's `agent_end`. It does not see the events of other agents. `on_event_triggers` limits it to some event types:

```python
from domyn.agents import ExecutionEventType, Part


@hook(trigger=HookPhase.ON_AGENT_EVENT, on_event_triggers=[ExecutionEventType.RESPONSE])
def mask_city(event, context):
    parts = [Part(text=(p.text or "").replace("Rome", "[city]")) for p in event.content]
    return event.model_copy(update={"content": parts})
```

An updater returns the event to use in place of the one it received, or nothing to leave it unchanged. For an agent that streams, the reply arrives as `RESPONSE_TOKEN` events, so a hook that rewrites the reply must list that type, and sees one piece at a time.

## Showing content

A hook can add content to the stream: a listener that returns a string, or any hook that returns `{"emitted_content": ...}` with its other fields, puts it in the `content` of its `hook_end` event. A hook written as a generator can report progress first: each `HookResult(is_update=True, emitted_content=...)` it yields becomes a `hook_update` event.

```python
from domyn.agents.hooks import HookResult


@hook(trigger=HookPhase.PRE_AGENT_CALL, is_listener=True)
async def check_quota(context, agent):
    yield HookResult(is_update=True, emitted_content="Checking your quota...")
    yield HookResult(emitted_content="Quota OK")
```

## Errors

A hook that raises, or that returns something its phase does not accept, produces a `hook_error` event with the exception's type as `error_code`. What follows depends on where it runs:

- **Agent and event hooks**: with `break_on_error=True`, the default, the turn stops with a generic message as the `response`. With `break_on_error=False`, the turn goes on as if the hook had not run.
- **Tool hooks**: the call goes on, with the arguments or the result the hook received. The `hook_error` event still carries the generic message in its `content` when `break_on_error=True`, so an interface that shows it should not present it as a refusal.

## Options

| `@hook` argument | Type | Default | Description |
|---|---|---|---|
| `trigger` | `HookPhase` | required | The phase. See [Phases](#phases). |
| `name` | `str` | the function's name | The name in events (`metadata["hook_name"]`) and in YAML. |
| `is_listener` | `bool` | `False` | A listener instead of an updater. |
| `config_params` | `dict` | `{}` | Extra keyword arguments passed to the function on every call, for settings that are not part of the phase's parameters. |
| `on_event_triggers` | list of `ExecutionEventType` | `USER_INPUT`, `TOOL_FEEDBACK`, `AGENT_CALL`, `AGENT_START`, `AGENT_END`, `TOOL_START`, `TOOL_UPDATE`, `TOOL_NEEDS_FEEDBACK`, `TOOL_END`, `RESPONSE_TOKEN`, `RESPONSE` | `ON_AGENT_EVENT` only: the event types the hook runs on. |
| `input_fields` | list of `InputFields` | none | The fields the hook receives instead of the phase's parameters, from the [REST hooks](#rest-hooks) table. The function's parameters must then have those names. |
| `metadata` | `dict` | `{}` | Free-form data kept on the hook. |
| `break_on_error` | `bool` | `True` | Whether an error in an agent or event hook stops the turn. |
| `silent` | `bool` | `False` | Keeps the hook's own events out of the stored conversation. They still appear in the stream. |

`@hook` also registers the hook by its name, so a [YAML configuration](configuration.md) can use it with `reference`.

```python
@hook(trigger=HookPhase.PRE_TOOL_CALL, config_params={"default_country": "IT"})
def add_country(tool, tool_args, default_country):
    tool_args["city"] = f"{tool_args['city']}, {default_country}"
    return {"new_tool_args": tool_args}
```

## In YAML

Hooks are declared under `hooks`, from a Python file or a URL, and attached by name to agents and tools:

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
  - name: mask_city
    type: updater
    trigger: on_agent_event
    path: hooks.py
    on_event_triggers: [response]
    silent: true

tools:
  - name: get_weather
    type: function
    path: tools.py
    hooks: [clean_args]

agents:
  - name: Assistant
    ...
    tools: [get_weather]
    hooks: [log_start, mask_city]
```

In a file, the function is plain: the `@hook` decorator is not needed. `break_on_error`, `on_event_triggers`, `silent` and `config_params` are set as keys. Every key is in the [YAML reference](../reference/yaml-schema.md).

### REST hooks

A hook declared with `url` instead of `path` calls an HTTP endpoint:

```yaml
hooks:
  - name: audit
    type: listener
    trigger: pre_tool_call
    url: https://audit.example.com/hooks/tool-call
    headers:
      Authorization: Bearer ${AUDIT_TOKEN}
    input_fields: [tool_name, tool_args]
```

The runner sends a `POST` with a JSON body. Without `input_fields`, the body holds the phase's parameters, serialized (`tool` and `tool_args`, for instance), and `config_params`. With `input_fields`, it holds only the fields named:

| Phase | `input_fields` available |
|---|---|
| `pre_agent_call`, `post_agent_call` | `agent_name`, `agent_input` (the text of the request), `last_event`, `interaction_history`, `interaction_context` |
| `pre_tool_approval`, `pre_tool_call` | `tool_name`, `tool_args` |
| `post_tool_call` | `tool_name`, `tool_result` |
| `on_agent_event` | `current_event`, `last_event`, `interaction_history`, `interaction_context` |

The endpoint answers with status 200 and a JSON object, read as a function's return value: `{}` for a listener, `{"new_tool_args": {...}}` or `{"new_result": {...}}` for a tool updater. Any other status is a hook error, with the status code as `error_code`.

| Key | Default | Description |
|---|---|---|
| `url` | required | The endpoint. |
| `method` | `POST` | The HTTP method. |
| `headers` | none | HTTP headers sent with every call. |
| `timeout` | `300.0` | Seconds to wait for the answer. |
| `input_fields` | none | The fields to send, from the table above. |

`tool_args` includes the `state` and `run_config` the tool receives, so an endpoint that gets them sees the data they carry.

## Events

Every hook run is visible in the stream: `hook_start` before it, then `hook_update` for each update, and `hook_end` with its content, or `hook_error`. Each carries the hook's name in `metadata["hook_name"]`. See [Event types](../reference/events.md).

## Next steps

- [Tool approvals](approvals.md): `PRE_TOOL_APPROVAL` hooks, and the rest of the approval gate.
- [Events and streaming](events-and-streaming.md): the events an `ON_AGENT_EVENT` hook sees.
- [YAML configuration](configuration.md): declaring hooks with the rest of an application.
