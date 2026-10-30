---
title: Conversations and memory
sidebar_label: Conversations and memory
description: "Keep conversations across turns and restarts, store data per conversation, and remember facts about a user across conversations"
---

# Conversations and memory

An agent keeps three kinds of information between one message and the next:

| | What it holds | Lasts |
|---|---|---|
| [**The conversation**](#conversations) | Every event of every turn: messages, tool calls, replies | As long as the conversation is stored |
| [**Conversation state**](#conversation-state) | Data your tools keep for this conversation, such as a cart or a working directory | As long as the conversation |
| [**Long-term memory**](#long-term-memory) | Facts about the user, such as their name or preferences | Across all of the user's conversations |

## Conversations

The `Runner` stores each turn in a **conversation manager**, keyed by application name, user id and conversation id. A turn run with the id of an existing conversation continues it: the agent's model receives the earlier turns as history.

```python
from domyn.agents.conversation_manager import SqliteConversationManager
from domyn.agents import Runner

runner = Runner(conversation_manager=SqliteConversationManager("conversations.db"))
```

| Manager | Where conversations live |
|---|---|
| `InMemoryConversationManager` | In the process. **The default** when `Runner()` is created without one. Gone when the process stops. |
| `SqliteConversationManager(path="./conversation.db")` | In a SQLite file. A conversation can be continued by any runner, in any process, that opens the same file. |

A turn without a `conversation_id` starts a new conversation, and every event of the turn carries the new id (see [Events and streaming](events-and-streaming.md)). Keep it, and pass it to `runner.run(..., conversation_id=...)` to continue.

The manager can also list and delete conversations:

```python
store = SqliteConversationManager("conversations.db")

for info in await store.get_user_conversations("shop", "user-42"):
    print(info.conversation_id, info.updated_at, len(info.events_history))

await store.delete_conversation("shop", "user-42", conversation_id)
await store.cleanup_old_conversations(older_than_days=30, application_name="shop")
```

A `ConversationInfo` carries the `conversation_id`, `application_name`, `user_id`, the `state`, the `events_history` (every event of every turn, including the user's messages) and `created_at` / `updated_at`.

### What the agent sees

On each model call, the agent receives the conversation's earlier turns as history, then the current turn. In a [team](orchestration.md), what each agent sees of the history is set by its `context_visibility`. A long conversation is shortened before it outgrows the model's context window by [context compaction](context-compaction.md).

### In the interactive CLI

The [interactive CLI](../cli/sessions.md) stores every session in `~/.domyn/conversations.db` with a `SqliteConversationManager`. `domyn run --resume` and `/resume` continue one.

## Conversation state

Each conversation has a `state`: a dictionary that is saved with the conversation and given back on every later turn. Use it for data a tool needs again later in the same conversation.

A tool receives the state by declaring a parameter named `state`. The parameter is not shown to the model. The tool gets a copy: to change the state, return a `ToolResult` with the new state, which **replaces** the old one, so return all of it:

```python
from domyn.agents.tools import ToolResult
from domyn.agents import FunctionTool


def add_to_cart(item: str, state: dict) -> ToolResult:
    """Add an item to the user's cart.

    Args:
        item: The product to add.
    """
    cart = [*state.get("cart", []), item]
    return ToolResult(
        success=True,
        data=f"Cart: {', '.join(cart)}",
        state={**state, "cart": cart},
    )


tool = FunctionTool(func=add_to_cart)
```

`data` is the result the model reads. With a `SqliteConversationManager`, a turn that adds `apples` and a later turn, in another process, that adds `pears` return `Cart: apples, pears`.

The runner keeps its own entries in the state too, for context compaction, which is why the example copies the existing state rather than starting from an empty one.

A tool can also declare `conversation_id`, to keep its own data per conversation outside the state.

## Long-term memory

Long-term memory keeps facts about a user and gives them to the agent in every later conversation with that user. It is enabled per agent with a `memory_manager`:

```python
from domyn.agents.memory.managers import FilesystemMemoryManager, MemoryExecutionMode

agent = Agent(
    ...,
    memory_manager=FilesystemMemoryManager(
        file_path="memories.jsonl",
        execution_mode=MemoryExecutionMode.AGENTIC,
    ),
)
```

In YAML:

```yaml
agents:
  - name: Assistant
    ...
    memory_manager:
      type: filesystem
      file_path: memories.jsonl
      execution_mode: agentic
```

`FilesystemMemoryManager` appends each memory as a line of JSON to `file_path`, with the user id, the agent name and the time.

### Recall

Before each model call, the agent's most recent memories about the user (up to `recency_limit`) are added to the request, as one message right after the system prompt. They are the memories the agent stored itself, or that its direct sub-agents stored.

### Storing

With `execution_mode` set to `agentic`, the agent gets a `remember` tool, which its model calls with a fact to keep, such as "The user prefers metric units". The memory is stored under `run_config["user_id"]`, so pass it to the runner:

```python
runner.run(..., user_id="user-42", run_config={"user_id": "user-42"})
```

Without it, `remember` answers that the memory was not stored.

### Options

| Option | Type | Default | Description |
|---|---|---|---|
| `file_path` | `str` | `local_memory.jsonl` | The file memories are written to. |
| `execution_mode` | `MemoryExecutionMode` | | Set it to `agentic`: the agent decides what to store. See [Storing](#storing). |
| `recency_limit` | `int` | `5` | How many memories, the most recent first, are given to the agent. |

### Managing memories

```python
memories = await manager.recall_memories(user_id="user-42", recency_limit=20)
await manager.delete_memory(memories[0].id)
await manager.reset_memory(user_id="user-42")
```

Each `Memory` has an `id`, the `memory` text, `user_id`, `agent_name` and `timestamp`. `reset_memory` can be limited to one agent with `agent_name`.

## Next steps

- [Context compaction](context-compaction.md): how a long conversation is kept within the model's context window.
- [Events and streaming](events-and-streaming.md): the events a turn adds to the conversation.
- [Tools](tools.md): writing the tools that use the state.
