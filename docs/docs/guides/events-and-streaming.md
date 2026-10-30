---
title: Events and streaming
sidebar_label: Events and streaming
description: "Run an agent, follow its turn event by event, and stream its reply to a user"
---

# Events and streaming

A turn of an agent is a stream of events. The `Runner` executes the agent and yields an event for each step: the agent starting, each model call, each tool call and its result, each hand-off in a team, and the reply. An interface that shows an agent to a user — a web app, a chat bot, the [interactive CLI](../cli/sessions.md) — is a loop over these events.

This page shows how to read that stream. [Event types](../reference/events.md) lists every type with its fields.

## Running a turn

```python
import asyncio

from domyn.agents import BaseEvent, ExecutionEventType, Part
from domyn.agents import Runner

runner = Runner()


async def ask(agent, text: str, conversation_id: str | None = None) -> str | None:
    user_input = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text=text)],
    )
    async for event in runner.run(
        application_name="weather-app",
        user_id="user-42",
        root=agent,
        user_input=user_input,
        conversation_id=conversation_id,
    ):
        conversation_id = event.conversation_id
        match event.event_type:
            case ExecutionEventType.TOOL_START:
                print(f"[{event.author} calls {event.action.name}: {event.action.thought}]")
            case ExecutionEventType.RESPONSE_TOKEN | ExecutionEventType.RESPONSE:
                print("".join(p.text or "" for p in event.content), end="", flush=True)
    print()
    return conversation_id
```

| Argument | Description |
|---|---|
| `application_name` | Name of your application. Conversations are stored per application. |
| `user_id` | Who is talking. Approval grants and stored conversations are kept per user. |
| `root` | The agent the user talks to. In a team, the main agent. |
| `user_input` | The message, as a `USER_INPUT` event. |
| `conversation_id` | The conversation to continue. Without one, the runner starts a new conversation and every event of the turn carries its id: pass it back on the next turn to continue. |

## What a turn looks like

An agent with one tool, asked about the weather, produces these events:

| `event_type` | `author` | What it carries |
|---|---|---|
| `agent_start` | `Assistant` | The user's message, as `content` and as the `task` in `action.parameters` |
| `llm_start` | `Assistant` | |
| `llm_end` | `Assistant` | |
| `tool_start` | `Assistant` | `action`: `get_weather`, `parameters={'city': 'Rome', 'reasoning': 'Need the weather.'}`, `thought='Need the weather.'` |
| `tool_end` | `Assistant` | The same `action`, with `observation='Sunny in Rome'` |
| `llm_start` | `Assistant` | |
| `llm_end` | `Assistant` | |
| `response` | `Assistant` | `content`: `It is sunny in Rome.` |
| `agent_end` | `Assistant` | |

With the default `tool_use_thought` planner, `action.parameters` also holds the `reasoning` argument the planner adds to each call, the same text as `action.thought`.

The user's own message is not yielded: the stream starts with `agent_start`. The turn ends with the root agent's `agent_end`, or earlier when it [pauses for approval](#pausing-for-approval), [fails](#errors), or a hook stops it, in which case the last event is the hook's `response`.

## The event

Every event is a `BaseEvent`. The fields an interface uses:

| Field | Description |
|---|---|
| `event_type` | What happened, an `ExecutionEventType`. |
| `author` | The agent the event belongs to. In a team, this is how events of different agents are told apart. |
| `content` | A list of `Part`s. A part's `text` is meant for the user; its `thought` is the agent's reasoning. |
| `action` | For tool and agent events, the call: `name`, `parameters`, `thought` (why the model made it), `call_id`, and on `tool_end` the `observation`, the result the model reads. |
| `is_partial` | `True` for a piece of a longer output that more pieces will follow. |
| `error_code`, `error_message` | Set on errors. |
| `metadata` | Extra data, depending on the type: token counts on `llm_start`, the approval subject on `tool_needs_feedback`. |
| `conversation_id`, `turn_id`, `interaction_id`, `event_id`, `timestamp` | Identifiers, for storing or correlating events. |

## Showing a turn to a user

Most interfaces handle a handful of types and ignore the rest:

| Show | Events |
|---|---|
| The reply | `response_token` while it streams; `response` when it does not stream, and for text a tool streams (`is_partial`) |
| That the agent is working | `llm_start` and `llm_end` |
| Tool calls | `tool_start` (with `action.thought`), then `tool_end` with the result, or `tool_error` |
| Progress of a long tool | `tool_update` (see [streaming tools](tools.md)) |
| Hand-offs in a team | `agent_call`, then the other agent's events, ending with its `agent_end` |
| A request for approval | `tool_needs_feedback` |
| A failure | `error_occurred` |

Rendering both `response_token` and `response` never shows a reply twice: a streamed reply is not repeated as a `response`.

## Streaming

`Agent.stream` decides how the reply arrives. It is `False` by default; the interactive CLI turns it on.

- **`stream=False`**: the reply is one `response` event.
- **`stream=True`**: the reply arrives as `response_token` events, each carrying the next piece in `content`, as the model produces it. No `response` follows.

So an interface that only handles `response` shows nothing for a streaming agent. Handle both.

With streaming on, `response` is still emitted for replies that were not streamed:

- a reply the agent forwards unchanged from a tool or a team member (`FORWARD_VERBATIM`, see [Orchestration](orchestration.md)), marked with `forward_verbatim` in `metadata`;
- the message of a hook that stops the turn (see [Hooks](hooks.md));
- a reply that asks the user a question and waits for the answer (`need_feedback` set on the event).

A provider streams only if it implements `generate_streaming`, which all [built-in providers](llm-providers.md) do.

## Teams

When the main agent hands off, the stream shows `agent_call` from the main agent (with the task in `action.parameters` and the reason in `action.thought`), then `agent_start` and the events of the team member, with its name as `author`. Its answer arrives as its `agent_end`, with the answer in `content`, and the main agent continues.

A team member's answer is never streamed and never a `response`, even when its `stream` is on: it is addressed to the main agent, not to the user. What the user should read is the main agent's reply.

## Pausing for approval

A tool that needs a person's approval stops the turn: the stream yields `tool_start`, then `tool_needs_feedback`, and ends. The `action` of that event is the pending call, and `metadata["approval_subject"]` describes what is being approved.

To answer, run the next turn with a `TOOL_FEEDBACK` event instead of a user message, carrying the action of the `tool_needs_feedback` event (`pending` below) with `approve` set:

```python
feedback = BaseEvent(
    event_type=ExecutionEventType.TOOL_FEEDBACK,
    author="user",
    action=pending.action.model_copy(update={"approve": True}),
)
async for event in runner.run(..., user_input=feedback, conversation_id=conversation_id):
    ...
```

The stream continues from `tool_end`: with the tool's result when approved, or with a refusal as the result when not, which the model reads and answers. [Tool approvals](approvals.md) covers who is asked, and how a grant can last beyond one call.

## Errors

| Event | Meaning | The turn |
|---|---|---|
| `tool_error` | A tool raised. `error_code` is the exception type, and the message is the result the model reads. | Continues: the model sees the error and decides what to do. |
| `hook_error` | A hook raised, or returned something its phase does not accept. `error_code` is the exception type. | For an agent or event hook: stops, with a generic message as a `response`, unless the hook was declared with `break_on_error=False`. For a tool hook: continues, and the call runs as if the hook had not. See [Hooks](hooks.md). |
| `error_occurred` | The agent could not continue, for example because the model's API failed. `error_message` says why. | Ends: it is the last event. |

## Compaction

When a conversation approaches the model's context window, or the provider rejects a request as too long, the agent shortens it before calling the model, and the stream shows `context_compaction_start` and `context_compacted` around it, with token counts in `metadata`. An interface can show them to explain a pause, and why the agent may no longer remember details from early in the conversation. See [Context compaction](context-compaction.md).

## Next steps

- [Event types](../reference/events.md): every type, when it is emitted and what it carries.
- [Tool approvals](approvals.md): the approval round trip.
- [Conversations and memory](conversations-and-memory.md): what is kept between turns.
