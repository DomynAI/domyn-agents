# LangGraph integration

Connect a LangGraph compiled graph to the Domyn platform with a single command:

```bash
domyn connect <module>:<symbol> \
    --channel-id <channel_id> \
    --space-id   <space_id> \
    --base-url   <base_url>
```

(`domyn connect` (formerly `domyn expose`, still accepted) is an alias for `domyn serve --channel relay`.) It opens an
outbound WebSocket to the relay, waits for `AGENT_START` events from the
platform, runs your graph, and streams trace events (`LLM_START` /
`TOOL_START` / `TOOL_END` / …) back as Domyn `BaseEvent`s. The adapter
(`LangGraphAdapter`) is auto-detected from the entrypoint symbol — anything
with an `ainvoke(input, config)` method is treated as a LangGraph runnable.

## Install

```bash
pip install "domyn[langgraph]"
```

The `langgraph` extra pulls in `langchain-core` and `langgraph` alongside the
core `domyn` package.

## Quickstart — a minimal graph

`ws_react_agent.py`:

```python
from typing import Any, TypedDict

from langchain_core.language_models.fake import FakeListLLM
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from domyn.agents.integrations.langgraph.utils import input_mapper


class GraphState(TypedDict):
    question: str
    answer: str


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression like '2+2'."""
    return str(eval(expression, {"__builtins__": {}}, {}))


_llm = FakeListLLM(responses=["It's probably 42."])


def answer(state: GraphState) -> GraphState:
    q = state["question"]
    if any(op in q for op in "+-*/"):
        return {"question": q, "answer": calculator.invoke({"expression": q})}
    return {"question": q, "answer": str(_llm.invoke(q))}


# The platform sends AGENT_START with action.parameters = {"task": "...",
# "user_request": "..."}; @input_mapper translates that dict into whatever
# input shape your compiled graph expects (here: GraphState).
@input_mapper(lambda d: GraphState(question=d.get("task", ""), answer=""))
def build_graph() -> Any:
    builder = StateGraph(GraphState)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    return builder.compile()


graph = build_graph()
```

Expose it:

```bash
DOMYN_API_KEY=domyn_*** domyn connect ws_react_agent:graph \
    --channel-id <channel_id> \
    --space-id   <space_id> \
    --base-url   <base_url>
```

## Anatomy of the entrypoint

The `MODULE:SYMBOL` argument resolves the runnable two ways:

- `path/to/file.py:symbol` — a file path, loaded ad-hoc.
- `module.path:symbol` — a regular Python import (run from the directory
  containing the module, or with the module on `PYTHONPATH`).

`SYMBOL` may be either a compiled graph **instance** (e.g. `graph` above) or a
zero-arg **factory** (e.g. `build_graph`). For factories, the loader calls
them once at startup. The `@input_mapper` decorator works on either form —
the mapper is recorded both on the factory and on the instance it produces,
so both `:build_graph` and `:graph` resolve it.

In the example above, `ws_react_agent:graph` means: from the file
`ws_react_agent.py`, take the `graph` variable.

## CLI flags

| Flag | What it is |
| --- | --- |
| `--channel-id` | Issued by the platform when you provision a worker. It opens a direct, persistent communication tunnel from your agent process to the platform. |
| `--space-id` | The platform space the agent serves. Specific to the platform deployment you're connecting to. You can find it in the url when opening the platform, after `spaces/` e.g.: `https://platform.example.com/console/{organization-id}/spaces/{space-id}/agents` |
| `--base-url` | The platform's API host (e.g. `api.platform.example.com`). Specific to the platform deployment you're connecting to. The scheme defaults to `https://` if omitted. |
| `--max-concurrency` | Optional. Caps in-flight invocations on this worker (default `8`). |

`DOMYN_API_KEY` must be set in the environment.

## Lifecycle in one paragraph

1. The CLI imports your entrypoint, picks up any `@input_mapper`, and
   `DomynRelayChannel` opens a WebSocket to `wss://<base-url>/relay/v1/ws`.
2. On each `AGENT_START` from the platform, `LangGraphAdapter` applies the
   input mapper (if any) to `action.parameters` and calls `graph.ainvoke(...)`
   with `config["configurable"]["thread_id"]` set to the conversation id, so a
   graph built with a checkpointer keeps multi-turn history across calls.
3. A langchain callback handler (`PlatformCallbackHandler`) attached to the
   run emits one `BaseEvent` per LangGraph callback (`LLM_START`/`LLM_END`,
   `TOOL_START`/`TOOL_END`, etc.); they're streamed back over the same
   WebSocket.
4. When `ainvoke` returns, the adapter sends a terminal `AGENT_END` carrying
   the result; on exception it sends `AGENT_END` with `error_code`/`error_message`.
5. The connection auto-reconnects on drops (full-jitter exponential backoff).

## Returning a structured result from a tool

By default a tool's return value is stringified into the `TOOL_END` event's
`action.observation` and nothing else reaches the platform. To return something
richer — an observation distinct from what the model saw, display content, or
**metadata for downstream consumers** — put a `ToolResult`-shaped dict in the
ToolMessage's `artifact`:

```python
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command


@tool("kg_search")
async def kg_search(query: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    answer, usage = await run_the_pipeline(query)
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=answer,  # what the model sees
                    tool_call_id=tool_call_id,
                    artifact={  # what the platform sees
                        "success": True,
                        "data": answer,
                        "metadata": {"token_usage": usage},
                    },
                )
            ]
        }
    )
```

The artifact is read whether the tool returns a bare `ToolMessage` (the
`response_format="content_and_artifact"` route) or a `Command` wrapping one —
LangGraph's own way of updating graph state from a tool. Fields:

| field | where it goes |
| --- | --- |
| `success` | required — an artifact without it is not a `ToolResult` and is ignored |
| `data` | `action.observation`. Optional in an artifact — omit it for a metadata-only result and the observation falls back to the message text |
| `metadata` | `BaseEvent.metadata` on the `TOOL_END` / `TOOL_ERROR` |
| `emitted_content` | the event's `content` parts |
| `error`, `error_code` | the event's `error_message` / `error_code` |

**The event type is keyed on `error`, not on `success`**: a non-empty `error`
makes it a `TOOL_ERROR`, and `success: false` on its own does not. That is the
same rule the runner applies to a local tool's result. Set both.

Extra keys are rejected: an artifact that is not a `ToolResult` is treated as
ordinary domain data and ignored, and the tool's content is read instead.
Without an artifact, the observation is the message text.

**Do not put bulk payloads in `metadata`.** It rides on the same WebSocket frame
as the rest of the event, and the relay connection carries the default 1 MiB
frame limit — an oversized frame is not a dropped event, it closes the
connection (close code `1009`) and the turn dies with it. Raw per-token
logprobs are the known way to blow past this: a single tool call has been
measured at 1.7–2.6 MB of them. Send ids, counts and small summaries; keep the
bulk where it was produced and reference it.

## Platform tools (optional)

Tools hosted on the Domyn platform can be referenced from a node and
`LangGraphAdapter` will round-trip their execution over the WebSocket: a
`TOOL_START` goes out, the platform executes the tool, a `TOOL_END` comes
back, and the node resumes.
