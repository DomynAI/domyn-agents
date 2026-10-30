---
title: Event types
sidebar_label: Event types
description: "Every event type the runner emits or accepts, when it occurs and what it carries"
---

# Event types

Every event is a `BaseEvent` (`domyn.agents.core`) whose `event_type` is an `ExecutionEventType`. This page uses the values, such as `tool_start`; in Python, compare with the enum member, such as `ExecutionEventType.TOOL_START`. How to consume them is in [Events and streaming](../guides/events-and-streaming.md).

## Fields

| Field | Type | Description |
|---|---|---|
| `event_type` | `ExecutionEventType` | What happened. |
| `author` | `str` | The agent the event belongs to; `user` for inputs. |
| `content` | `list[Part]` | Text and other parts. See [Parts](#parts). |
| `action` | action | The call the event is about, for tool and agent events. See [Actions](#actions). |
| `is_partial` | `bool` | More pieces of the same output follow. |
| `error_code` | `str \| None` | On errors: a short code, such as the exception type. |
| `error_message` | `str \| None` | On errors: the description. |
| `metadata` | `dict` | Type-specific data, listed per type below. |
| `need_feedback` | `bool` | The event is a question to the user, and the turn ends with it. |
| `event_id` | `str` | Unique id of the event. |
| `conversation_id` | `str` | The conversation. |
| `turn_id`, `interaction_id` | `str` | Identify the turn the event belongs to. A turn resumed after an approval keeps both. |
| `timestamp` | `datetime` | When the event was created. |

### Parts

| Field | Description |
|---|---|
| `text` | Text for the user. |
| `thought` | The agent's reasoning, or a status line, not meant as the answer. |
| `code`, `html`, `chart`, `rich_text`, `link`, `file`, `metadata`, `guardrail_result` | Structured parts, for interfaces that render them. |

### Actions

| Class | On | Fields |
|---|---|---|
| `AgentAction` | `agent_start`, `agent_call`, a [remote agent](../guides/remote-agents.md)'s `agent_end` | `name` (the agent), `parameters` (the `task`, or the agent's own invocation parameters), `thought` |
| `ToolAction` | `tool_start`, `tool_update`, `tool_end`, `tool_error`, a local team member's `agent_end` | `name`, `parameters`, `thought`, `call_id`, `observation` (the result, on `tool_end` and `tool_error`) |
| `FeedbackAction` | `tool_needs_feedback`, `tool_feedback` | as `ToolAction`, plus `approval_type`, `approve`, `remember_scope`, `remember_pattern`; `parameters` is a list of name–value pairs that can be edited |
| `AuthenticationAction` | `tool_needs_feedback` | as `ToolAction`, for an MCP tool whose OAuth credentials must be renewed before it runs |

Every action also has an optional `message`.

## Emitted by the runner

### Agents

| Type | When | Carries |
|---|---|---|
| `agent_start` | An agent starts its part of the turn: the root agent first, then each team member it hands off to. | `action`: `AgentAction` with the task; `content`: the task as text. |
| `agent_call` | An agent hands off to a team member. Followed by that member's `agent_start`. | `action`: `AgentAction` naming the member, with its arguments and the reason in `thought`. |
| `agent_end` | An agent finishes. For the root agent, the last event of the turn. | For a team member, its answer in `content` and its final call in `action`. Empty for the root agent. |

### Model calls

| Type | When | Carries |
|---|---|---|
| `llm_start` | Before each model call. | `metadata["context_usage"]`: `message_tokens` and `tool_definition_tokens` (estimated), `correction_tokens` (the provider's count of the last request minus the estimate of it, 0 without a count), `context_window`, `threshold_tokens` and `strategy`, each `None` when it does not apply. `threshold_tokens` is set only when a compaction can happen; `content`: a status line. |
| `llm_end` | After each model call. | `content`: a status line and, for a `tool_use_thought` agent, the model's reasoning as a `thought` part. |
| `response` | The reply, when it is not streamed; text a streaming tool shows right away (`is_partial`); a reply forwarded unchanged with `FORWARD_VERBATIM` (`metadata["forward_verbatim"]`); the message of a hook that stops the turn; a question to the user (`need_feedback`). | `content`: the text. |
| `response_token` | A piece of a streamed reply, for an agent with `stream=True`. | `content`: the next piece; `is_partial` is `True`. |
| `external_part` | Content pushed into a running turn with `Runner.push_parts`, from outside the agent. | `content`; the model sees it only when its `metadata` marks it visible. |
| `llm_provider_error` | The model's API failed, with the `react` planner only. | `error_message`. |

### Tools

| Type | When | Carries |
|---|---|---|
| `tool_start` | The model called a tool. | `action`: `ToolAction` with the arguments and the `thought`. |
| `tool_update` | A streaming tool reported progress with `UpdateToolResult`. Not shown to the model. | `content[0].metadata`: the data the tool yielded. |
| `tool_end` | The tool returned. Also when a call was refused at the approval gate: the refusal is the result. | `action.observation`: the result. |
| `tool_error` | The tool raised. | `error_code`: the exception type; `error_message` and `action.observation`: its message. |
| `tool_needs_feedback` | The tool needs approval before it runs. The turn ends with it. | `action`: `FeedbackAction` for the pending call; `metadata`: `approval_subject`, `approval_scopes`, `approval_pattern`, `approval_parts`. |

### Hooks

| Type | When | Carries |
|---|---|---|
| `hook_start` | Before a hook runs. | `metadata["hook_name"]`. |
| `hook_update` | A hook emitted content without finishing. | `content`, `metadata["hook_name"]`. |
| `hook_end` | A hook finished. | `content` it emitted, `metadata["hook_name"]`. |
| `hook_error` | A hook raised. | `error_code`: the exception type; `error_message`; `metadata["hook_name"]`. |

### Compaction and errors

| Type | When | Carries |
|---|---|---|
| `context_compaction_start` | The conversation is about to be shortened: the request reached the threshold, the provider rejected it as too long, or a compaction was requested. | `metadata`: `tokens_before`, `threshold_tokens` and `rejected` (the provider rejected the request), or `requested: True`. |
| `context_compacted` | The compaction finished. | `metadata`: `changed` (something was removed or summarized). At the threshold, also `tokens_before`, `tokens_after`, `threshold_tokens` and `sufficient` (the result fits under the threshold); on request, `requested: True`, `events_dropped` and `tokens_saved`. |
| `error_occurred` | The agent could not continue. The last event of the turn. | `error_message`; `content`: a description, from a planner; `error_code`, from the runner, such as `RequiredSkillUnavailable`. |

## Accepted as input

These are passed to `Runner.run()` as `user_input`.

| Type | Purpose |
|---|---|
| `user_input` | A message from the user: `content` with its text. |
| `tool_feedback` | The answer to a `tool_needs_feedback`: its `action`, with `approve` set and, optionally, edited `parameters` and a `remember_scope`. See [Tool approvals](../guides/approvals.md). |
| `compaction_requested` | Asks for a compaction now, instead of waiting for the threshold. The stream is only `context_compaction_start` and `context_compacted` (only the latter, with `changed: False`, for a planner that cannot compact), and the request is not added to the conversation. See [Context compaction](../guides/context-compaction.md). |
