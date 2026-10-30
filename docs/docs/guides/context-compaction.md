---
title: Context compaction
sidebar_label: Context compaction
description: "Keep a long conversation within the model's context window by dropping old tool outputs and summarizing older turns"
---

# Context compaction

A model reads a limited amount of text per request, its **context window**. An agent sends the whole conversation on every model call, so a long conversation, or a few large tool outputs, eventually outgrows it, and the provider rejects the request. Compaction shortens the conversation before that happens: it replaces old tool outputs with a one-line note, and, if that is not enough, has the model summarize the older part of the conversation. When a request is rejected anyway, it compacts and sends it again.

Compaction runs in the `tool_use` and `tool_use_thought` [planner strategies](planner-strategies.md), not in `react`.

## When it happens

Compaction needs to know the window. Set `context_window` on the [LLM provider](llm-providers.md) to declare it. Without it, the agent asks the provider before each turn: vLLM, OpenRouter, Anthropic and a gateway that publishes it report it, and for OpenAI and Cerebras, which do not, a catalogue shipped with the package has it. When nothing answers, nothing is compacted automatically until the provider rejects a request as too long (see below). The [interactive CLI](../cli/configuration.md) assumes 150,000 tokens when it cannot find out.

Before each model call, the agent measures the size of the whole request: the conversation, the tool definitions, and the [skills](skills.md) and memories added for this turn. When it exceeds the **threshold**, the window minus `reserve_tokens` (16,384 by default), the conversation is compacted first, and the shorter version is sent. The reserve leaves room for the reply.

Only the conversation is compacted. The tool definitions and the turn's skills and memories count towards the threshold but are never shortened, so an agent whose tools alone take most of the window cannot get under it: the `tool_definition_tokens` in the `llm_start` event's `metadata["context_usage"]` says how much they take.

## How the size is estimated

The estimate counts three characters per token, rounding up, plus a small allowance per message, per tool call. On its own it is off both ways: it overcounts prose, and undercounts token-dense text such as hashes, UUIDs or encoded data by more than two times. So the size is anchored to the provider's own count: each response says how many tokens its request had (`usage`), and the next request is that count plus an estimate of what was added since. After a compaction, the share of the difference that belongs to the text kept is carried over. What was added since the last count is only estimated, so a request that is mostly fresh, dense tool output can still be undercounted, and rejected.

## What gets compacted

The conversation is split in three parts:

- **The head**: the system prompt and the conversation's first message. Never summarized, since it frames the whole conversation.
- **The tail**: the most recent messages, about `keep_recent_tokens` (20,000 by default) worth, and never more than half the room the conversation has under the threshold, so that a small window can still be compacted. Never compacted, since it is what the agent is working on.
- **The middle**: everything in between, which is what gets shortened.

The cut never separates a tool call from its result, which providers would reject.

Compaction happens in two phases:

1. **Pruning.** Each tool output before the tail longer than 200 characters is replaced with a note saying what it was, such as `[fetch_report output dropped to save context — 3 lines. Read it again with read_tool_result(result_id="event_…")]`, with the exit code when the output had one. The agent can read the output again with that call, since the conversation keeps it whole. A sub-agent's task or report, pruned the same way, has no tool output to read and its note gives no call. It needs no model call, and in an agent conversation tool outputs are usually most of the text.
2. **Summarizing.** If the conversation is still over the threshold, the agent's model is asked to summarize the middle: what the user asked for, what was established, what was tried and failed, and what is still open. The summary replaces the middle, as one message starting with `[conversation summary]`. The user's latest message is never summarized, wherever the tail begins. If the summary fails, the pruning is kept.

The conversation itself is never rewritten: every event stays in the history. What was dropped and the summary are recorded in the conversation's state, for each agent, and applied again on every later request, so a compaction lasts and a summary is paid for once. When a later compaction summarizes again, the previous summary is part of what it summarizes.

## When the provider rejects a request

The estimate can still fall short, or the window can be unknown. When the provider rejects a request because the prompt is too long, the agent compacts the conversation to about half its estimated size and sends the request again. It does this once per turn: a second rejection ends the turn with an `error_occurred` event. When the rejection states the window, as vLLM, OpenAI, OpenRouter, Anthropic and LiteLLM's do (`maximum context length is 32768 tokens`), that window is used for the requests that follow, and remembered for the same endpoint and model until the process exits.

This needs no `context_window`. It works with every provider: a rejection is recognized by its HTTP status, 400 or 413, and by the provider's wording, such as `maximum context length` or `prompt is too long`. Any other error is not retried. With the `noop` strategy, or with compaction disabled, nothing is retried.

## Strategies

| `strategy` | What it does |
|---|---|
| `two_phase` | **Default.** Prunes, and summarizes only if pruning was not enough. |
| `prune_only` | Prunes only, and never calls a model. A conversation whose text alone exceeds the threshold stays over it. |
| `noop` | Never compacts automatically. The threshold is not reported either, since it would never be acted on. |

## Configuring

### In Python

The strategy and its limits are the `compactor` of the planner strategy:

```python
from domyn.agents.planner_strategy import (
    CompactionPolicy,
    Compactor,
    PruneOnlyCompaction,
)
from domyn.agents.planner_strategy import ToolUseWithThoughtPlannerStrategy

planner = ToolUseWithThoughtPlannerStrategy(
    compactor=Compactor(
        strategy=PruneOnlyCompaction(),
        policy=CompactionPolicy(reserve_tokens=8000, keep_recent_tokens=4000),
    ),
)
```

The strategies are `TwoPhaseCompaction`, `PruneOnlyCompaction` and `NoOpCompaction`. `TwoPhaseCompaction(llm_provider=...)` sends the summaries to another model, for example a cheaper one, instead of the agent's.

### In YAML

In a [YAML configuration](configuration.md), the `compactor` is set on the strategy entry:

```yaml
planner_strategies:
  - name: default
    type: tool_use_thought
    compactor:
      strategy: prune_only
      reserve_tokens: 8000
      keep_recent_tokens: 4000
```

An unknown `strategy` name falls back to `two_phase`, with a warning in the log. A `react` strategy declared with a `compactor` is rejected when the file is loaded.

### In the interactive CLI

`~/.domyn/config.yaml` is a YAML configuration too: the setup wizard writes the `compactor` block under `planner_strategies`, and `context_window` on the provider. Edit them there.

### Options

| Option | Type | Default | Description |
|---|---|---|---|
| `strategy` | `str` | `two_phase` | See [Strategies](#strategies). In Python, the `strategy` of the `Compactor`. |
| `reserve_tokens` | `int` | `16384` | Room kept free below the window, for the reply. |
| `keep_recent_tokens` | `int` | `20000` | Size of the recent part of the conversation that is never compacted. |

`CompactionPolicy` has a few more, in Python only:

| Option | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `True` | `False` turns automatic compaction off. |
| `protect_first_n` | `int` | `1` | Messages after the system prompt that are never compacted. |
| `keep_last_user_messages` | `int` | `1` | The latest user messages a summary never replaces. |
| `prune_tool_results_over_chars` | `int` | `200` | Tool outputs at or below this length are never pruned. |
| `keep_recent_tool_results` | `int` | `3` | For a [compaction on request](#compacting-on-request): how many of the latest results it leaves alone. |

## Compacting on request

To compact now rather than at the threshold, run a turn with a `COMPACTION_REQUESTED` event instead of a user message:

```python
from domyn.agents import BaseEvent, ExecutionEventType, Part

request = BaseEvent(
    event_type=ExecutionEventType.COMPACTION_REQUESTED,
    author="user",
    content=[Part(text="compact")],
)
async for event in runner.run(..., user_input=request, conversation_id=conversation_id):
    ...
```

It drops the older tool outputs of the agent the user is talking to, and calls no model. It leaves alone the last three results (`keep_recent_tool_results`), counting both tool outputs and the `agent_end` event that closes each agent's run, so after a few turns of one tool call each, only the latest tool output is kept. It does not need a `context_window`, and works with every strategy, `noop` included. The stream is only `context_compaction_start` and `context_compacted`, whose text says how many outputs were dropped and whose `metadata["tokens_saved"]` says how much the next request shrinks. The request is not added to the conversation. The shorter conversation is used from the next request.

In the interactive CLI, `/compact` sends this request, and `/context` shows what the last request contained: the conversation and the tool definitions, the correction the provider's count gave them, their total, the share of the window used, and how far the request is from the threshold.

## Events

An automatic compaction shows in the event stream as `context_compaction_start`, before the work starts, and `context_compacted`, when it ends. A summary takes a model call, so an interface can use them to explain the pause, and why the agent may no longer have the details of earlier turns. After a rejected request, `context_compaction_start` carries `rejected: true` in its `metadata`.

`context_compacted` reports two different things in its `metadata`: `changed`, whether anything was dropped or summarized, and `sufficient`, whether the result is under the threshold. A compaction can change a lot and still not be sufficient: with `prune_only`, a conversation whose text alone is over the threshold reports `changed: false, sufficient: false` on every request.

Every `llm_start` also carries `metadata["context_usage"]`, the measurement of the request about to be sent. [Event types](../reference/events.md) lists the fields of all three.

## Next steps

- [LLM providers](llm-providers.md): setting `context_window`.
- [Planner strategies](planner-strategies.md): the strategies that compact, and their other options.
- [Conversations and memory](conversations-and-memory.md): where the conversation and its state are kept.
