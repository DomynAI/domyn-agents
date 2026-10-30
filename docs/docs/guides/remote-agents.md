---
title: Remote agents
sidebar_label: Remote agents
description: "Add an agent that runs in another process to a team, and serve a Domyn agent or a LangGraph graph to other runners"
---

# Remote agents

A team member does not have to run in the same process as the rest of the team. It can run on another machine, belong to a service another team operates, or be a graph built with LangGraph. In the calling application it is a `DelegateAgent`: the main agent hands off to it like any other [team member](orchestration.md), but its turn is executed by a remote runner, and the events of that turn come back over a **transport**.

## How it fits together

```text
 caller                                        worker
┌────────────────────────────┐                ┌───────────────────────────────┐
│ Runner                     │                │ domyn connect worker.py:agent │
│  └─ DelegateAgent          │                │  └─ Agent or LangGraph graph  │
│      └─ transport ─────────┼──►  relay  ◄───┼── relay connection            │
└────────────────────────────┘    channel     └───────────────────────────────┘
```

The two sides never connect to each other. Both connect to a **relay** as clients, on the same **channel**, and the relay forwards events between them. The caller sends an `AGENT_START` carrying the task; the worker runs the agent and sends back its events, ending with the reply.

Domyn Agents does not include a relay: you run one, or use one your infrastructure already provides. With the `websocket` transport and `domyn connect`, a relay must:

- accept WebSocket connections at `/relay/v1/ws`, with the channel in the `channel-id` header;
- forward every message a client sends to the other clients connected to the same channel;
- keep the recent messages of each channel, and on each new connection first send the ones after the event id in the `last-event-id` header, or all of them when the header is missing.

The last point matters because both sides close and reopen their connection during a turn: a message sent while one of them is reconnecting must still reach it. Both sides drop the messages they have already seen, so replaying more than needed is harmless.

## Adding a remote agent to a team

In a [YAML configuration](configuration.md), a remote agent is an entry under `agents` with `type: delegate` and a `transport` (the provider is declared as in [Orchestration](orchestration.md)):

```yaml
agents:
  - name: Specialist
    type: delegate
    description: Counts the words in a text.
    transport:
      type: websocket
      url: wss://relay.example.com/relay/v1/ws
      headers:
        channel-id: <channel id>

  - name: Assistant
    description: Front desk
    instruction: Send counting questions to Specialist.
    llm_provider: main_llm
    sub_agents:
      - Specialist
```

In Python:

```python
from domyn.agents import DelegateAgent
from domyn.agents.runner.websocket_transport import WebSocketTransport

specialist = DelegateAgent(
    name="Specialist",
    description="Counts the words in a text.",
    transport=WebSocketTransport(
        ws_url="wss://relay.example.com/relay/v1/ws",
        headers={"channel-id": "<channel id>"},
    ),
)

assistant = Agent(..., sub_agents=[specialist])
```

The main agent's model sees `Specialist` among the agents it can call, with its `description`, and calls it with a `task`. A `DelegateAgent` can also be the `entrypoint` itself: the application then forwards each user message to the remote agent and returns its reply.

## What the caller sees

Every event of the remote turn is passed through, with the delegate's name as `author`: `LLM_START` and `LLM_END`, `TOOL_START` and `TOOL_END` for the tools the remote agent runs, and its reply. Events of the remote agent's own team members keep their own `author`. The caller only observes these tool calls: the remote agent runs them, except the tools declared in the delegate's own `tools` (see [Options](#options)), which the caller runs and whose results it sends back.

How the remote turn ends depends on where the delegate sits:

- **As a team member**, its final reply becomes an `AGENT_END` carrying the reply, and the main agent continues. The tool calls the remote agent made are in the main agent's history too, so its model sees what was done as well as the answer.
- **As the entrypoint**, its final reply is the turn's `RESPONSE`.

If no message arrives within the transport's `timeout`, the remote turn ends with `error_code` `delegate_timeout` and a text saying that the delegated agent stopped responding. For a team member, this is its `AGENT_END`: the text goes to the main agent's model as the result of the call, and the main agent continues. As the entrypoint, it is the turn's `RESPONSE`.

If the caller's turn is cancelled, both transports signal the remote runner to stop: they send it an `AGENT_END` with `error_code` `user_cancelled`. A turn that has already ended is not signalled.

## Transports

### `websocket`

Connects to the relay over WebSocket. The connection is opened for each step of the remote turn and closed after it, so no socket stays open between turns. On each connection, the transport also sends the conversation id and the last event it received, and drops any event it has already seen or that belongs to another turn.

| Option | Type | Default | Description |
|---|---|---|---|
| `url` | `str` | required | Relay URL, `ws://` or `wss://`. Python: `ws_url`. |
| `headers` | `dict[str, str]` | `{}` | Headers sent on every connection. `channel-id` selects the channel. |
| `timeout` | `float` | `60.0` | Seconds to wait to connect, and for each next message. |
| `ping_interval` | `float \| None` | `20.0` | Seconds between keep-alive pings. `None` disables them. |

### `sse`

Talks to the relay over HTTP: a `POST` opens a Server-Sent Events stream that carries the remote events, and a result for the remote agent is sent as the body of the request that reopens it. Each `POST` carries the channel, space and conversation as the `channelId`, `spaceId` and `conversationId` query parameters, and the last event received in a `Last-Event-ID` header. Each event in the stream must be a JSON envelope with the event under `payload`; an event sent bare is dropped. `domyn connect` speaks WebSocket only, so a relay used with `sse` has to serve both.

| Option | Type | Default | Description |
|---|---|---|---|
| `url` | `str` | required | Relay messages endpoint, `https://…`. |
| `headers` | `dict[str, str]` | `{}` | `channel-id` selects the channel and `space-id` the space, sent as query parameters. Also sent as HTTP headers, unless `meta` is set. |
| `meta` | `dict[str, str]` | the `headers` | HTTP headers sent with every request. |
| `timeout` | `float` | `60.0` | Seconds to wait to connect, to send, and for the next event. |

### Your own transport

A transport is a subclass of `RunnerTransport` (`domyn.agents.runner.transport`) that implements `execute()`, an async generator that yields the remote events for one turn, and `send_result()`, which sends an event back to the remote runner. `cancel()` is optional: by default a cancelled turn sends nothing. Declaring a `type` class variable makes it available as `transport.type` in YAML as soon as its module is imported:

```python
import asyncio

from domyn.agents import ExecutionEventType
from domyn.agents.runner import RunnerTransport

# One pair of queues per url: events to the remote runner, events from it.
QUEUES: dict[str, tuple[asyncio.Queue, asyncio.Queue]] = {}


class QueueTransport(RunnerTransport):
    type = "queue"

    def __init__(self, url: str) -> None:
        self.outbox, self.inbox = QUEUES.setdefault(url, (asyncio.Queue(), asyncio.Queue()))

    @classmethod
    def from_config(cls, config: dict) -> "QueueTransport":
        return cls(url=config["url"])

    async def execute(self, agent_name, context, run_config=None):
        await self.outbox.put(context.agent_input)  # the AGENT_START carrying the task
        while True:
            event = await self.inbox.get()
            yield event.model_copy(update={"author": agent_name})
            if event.event_type in (ExecutionEventType.RESPONSE, ExecutionEventType.AGENT_END) and not event.is_partial:
                return

    async def send_result(self, event, conversation_id=None):
        await self.outbox.put(event)
```

```yaml
agents:
  - name: Specialist
    type: delegate
    description: Counts the words in a text.
    transport:
      type: queue
      url: specialist
```

Whatever reads `outbox` plays the remote runner: it answers each `AGENT_START` by putting the remote events on `inbox`, ending with the reply.

## Options

| Option | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Name the agent is called by. |
| `description` | `str` | required | What the agent does. The main agent's model chooses it from this text. |
| `transport` | transport | required | See [Transports](#transports). In YAML, a mapping with a `type`. Without one, the agent fails when it is called. |
| `tools` | list of tools | `[]` | Tools the remote agent may ask the caller to run. A function tool listed here is run by the caller, and its result is sent back to the remote agent. |
| `invocation_parameters` | JSON Schema | `{"task": string}` | The arguments the main agent's model passes when it calls this agent, in place of the default `task`. They reach the remote agent as the parameters of its `AGENT_START`. |
| `visibility` | `isolated` \| `scoped` \| `trusted` \| `global` | `isolated` | Whether the caller's `run_config` (user id, tokens and other request data) is sent to the remote agent. Only `trusted` and `global` send it, beside the event rather than inside it. |
| `context_visibility` | `ContextVisibility` | `STANDARD` | What history the agent sees, as for [any agent](orchestration.md). |
| `handoff_scope` | `HandoffScope` | `EXPLICIT` | Which agents it may hand off to, as for [any agent](orchestration.md). |
| `hooks` | list of hooks | `[]` | [Hooks](hooks.md) with the `pre_agent_call`, `post_agent_call` and `on_agent_event` triggers. |

## Serving an agent with `domyn connect`

`domyn connect` makes an agent available on a relay channel. It connects to the relay, waits for `AGENT_START` events, runs the agent for each one and sends its events back. It runs until it is stopped.

:::note
The relay's address and headers, and the platform's tools, come from the Domyn platform SDK, so `domyn connect` needs the [platform SDK](../getting-started/installation.md#the-domyn-platform-sdk) installed next to Domyn Agents. Without it, or with a version older than it needs, the command stops and names the requirement. To reach a relay of your own with no SDK, build a `DomynRelayChannel(ws_url=..., headers=...)` in Python instead.
:::

```bash
export DOMYN_API_KEY=<relay key>
domyn connect worker.py:specialist \
    --channel-id <channel id> \
    --base-url relay.example.com
```

The argument names what to serve:

- `path/to/file.py:symbol`: a variable in a file. The file's directory is not added to the import path, so a file that imports a module next to it needs the `module:symbol` form, or that directory on `PYTHONPATH`.
- `module:symbol`: a variable in an importable module. The current directory is added to the import path.
- `path/to/file.py`: the file's root agent, the one that is not a sub-agent of another.
- `path/to/config.yaml`: the `entrypoint` agent of a [YAML configuration](configuration.md).

What the variable holds decides how it is run:

- an `Agent`: it runs under a `Runner`, with its tools and its team, exactly as it would locally;
- a LangGraph compiled graph, or anything else with an `ainvoke` method: see [LangGraph graphs](#langgraph-graphs);
- a function with no arguments that returns a LangGraph graph: it is called once, at startup. This works for graphs only: for an `Agent`, name the variable that holds the instance.

| Option | Environment variable | Default | Description |
|---|---|---|---|
| `--channel-id` | `DOMYN_CHANNEL_ID` | | Channel to serve. |
| `--space-id` | `DOMYN_SPACE_ID` | | Space of the channel, sent to the relay in the `space-id` header. |
| `--configuration-id` | `DOMYN_CONFIGURATION_ID` | | Sent to the relay in the `configuration-id` header, and used when platform tools are looked up. |
| `--base-url` | `DOMYN_BASE_URL` | | Relay host, with an optional port. The relay is reached at `wss://<host>/relay/v1/ws`, or `ws://` for `localhost`, `127.0.0.1` and `::1`. |
| `--target` | | | Serves several channels from one process, possibly in different spaces or on different relays. Repeatable; each is `channel_id=…,space_id=…,base_url=…,configuration_id=…`, where only `channel_id` is required. `space_id` and `base_url` default to `--space-id` and `--base-url`; `configuration_id` defaults to `--configuration-id` only when the target sets no `space_id`. Without `--target`, the one channel is `--channel-id`. |
| `--max-concurrency` | | `8` | Turns run at the same time. Beyond that, as many more wait, and further requests are refused. |
| `--platform-tools` / `--no-platform-tools` | | on when the SDK is installed, else off | Attaches to the served agent every tool the platform offers that the agent does not already declare. `--no-platform-tools` turns it off; `--platform-tools` without the SDK stops and names the version it needs. |
| `--tool-refresh-interval` | | `0` | With `--platform-tools`, looks the platform tools up again every this many seconds. `0` does not. |

`DOMYN_API_KEY` must be set; the command stops if it is not. It is sent to the relay in the `api-key` header, for a relay that checks who connects.

### LangGraph graphs

Install the `langgraph` extra:

```bash
uv add "domyn[langgraph]"
```

A compiled graph is served the same way. The caller sends a `task` (and possibly a `user_request`), which rarely matches the graph's input; `@input_mapper` converts it:

```python
from typing import Any, TypedDict

from langchain_core.language_models.fake import FakeListLLM
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from domyn.agents.integrations.langgraph import input_mapper


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


@input_mapper(lambda d: GraphState(question=d.get("task", ""), answer=""))
def build_graph() -> Any:
    builder = StateGraph(GraphState)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    return builder.compile()


graph = build_graph()
```

```bash
domyn connect graph.py:graph --channel-id <channel id> --base-url relay.example.com
```

`@input_mapper` can decorate the factory or the compiled graph, so both `graph.py:graph` and `graph.py:build_graph` find it. Without it, the parameters are passed to the graph as they are.

The graph's model and tool calls reach the caller as `LLM_START`, `LLM_END`, `TOOL_START` and `TOOL_END` events. The reply is taken from the graph's final state: the text of the last message in `messages`; otherwise `result`, if it is text; otherwise the whole state as JSON. The graph above has neither, so `6*7` is answered with `{"question": "6*7", "answer": "42"}`.

## Next steps

- [Orchestration](orchestration.md): how a main agent hands off to its team.
- [Events and streaming](events-and-streaming.md): the events a remote turn produces.
