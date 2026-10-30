---
title: Orchestration
sidebar_label: Orchestration
description: "Compose agents that hand off work to each other, and control what each one sees and can call"
---

# Orchestration

One agent with many tools gets harder to steer as the tools pile up. An alternative is a team: a main agent that talks to the user and hands each request to a specialist with its own instructions and tools. This page shows how to build one, and how to control what each agent sees and which agents it can call.

## Handing off to a sub-agent

Give the main agent its specialists as `sub_agents`:

```python
from domyn.agents import Agent

order_agent = Agent(
    name="OrderAgent",
    description="Retrieves information about customer orders",
    instruction="You help users check order status and details.",
    llm_provider=llm,
    tools=[get_order_tool, list_orders_tool],
)

payment_agent = Agent(
    name="PaymentAgent",
    description="Handles payment questions and refunds",
    instruction="You help users with payment questions and process refunds.",
    llm_provider=llm,
    tools=[get_payment_tool, refund_tool],
)

main_agent = Agent(
    name="CustomerService",
    description="Main customer service agent",
    instruction="You are a customer service assistant. Route each question to the right specialist.",
    llm_provider=llm,
    tools=[lookup_customer_tool],
    sub_agents=[order_agent, payment_agent],
)
```

Run `main_agent` as usual. It is the agent the user talks to, and the only one you pass to the runner.

When a user asks "Where is order 42?":

1. The main agent's model sees each sub-agent as one more tool it can call, named after the agent and described by its `description`. It calls `OrderAgent`, and writes a **task** for it: the request, restated with every detail the sub-agent needs.
2. `OrderAgent` starts with that task, calls its own tools, and finishes with an answer.
3. Control returns to the main agent, which reads what `OrderAgent` did and found, and answers the user, or hands off again.

A sub-agent's answer goes back to the agent that called it, not to the user. A sub-agent can have sub-agents of its own, to any depth.

In the [event stream](events-and-streaming.md), a handoff shows up as an `AGENT_CALL` from the calling agent, an `AGENT_START` when the sub-agent begins, and an `AGENT_END` when it finishes. Every event carries the name of the agent that produced it in `author`.

### Writing descriptions that route well

The main agent decides whom to call only from the sub-agents' names and descriptions, so write the description for that reader:

- **Say what the agent handles, and what it does not.** "Handles payment questions and refunds. Does not change orders." routes better than "Payment agent".
- **Keep specialists distinct.** Two descriptions that could both match a request make the choice a coin toss.
- **Tell the main agent how to delegate** in its `instruction`: for example, to send each question to the right specialist and to answer from what the specialist found.

## What an agent sees, and whom it can call

Two settings on each agent, independent of each other, control a team:

| Setting | Controls | Default |
|---|---|---|
| `context_visibility` | Which other agents' work this agent reads in its history | `STANDARD` |
| `handoff_scope` | Which agents this agent can hand off to | `EXPLICIT` |

```python
from domyn.agents import ContextVisibility, HandoffScope

payment_agent = Agent(
    name="PaymentAgent",
    ...,
    context_visibility=ContextVisibility.PEERS,
    handoff_scope=HandoffScope.EXPLICIT,
)
```

### `context_visibility`: what it reads

| Value | The agent reads the work of |
|---|---|
| `STANDARD` | Itself and its own sub-agents, at any depth |
| `PEERS` | Also its siblings (the other sub-agents of its parent) and their sub-agents |
| `GLOBAL` | Every agent in the tree |

With `STANDARD`, a specialist starts from the task it was given and sees nothing of what its siblings did. Use `PEERS` when it needs their results: a `PaymentAgent` that should read the order `OrderAgent` just looked up. Use `GLOBAL` for an agent that needs the whole picture, such as one that summarizes the conversation. Every agent you open up adds to its context, so widen only where it is needed.

### `handoff_scope`: whom it can call

| Value | The agent can hand off to |
|---|---|
| `EXPLICIT` | Exactly the agents in its `sub_agents` |
| `DIRECT_PEERS` | Its own sub-agents and its siblings, but not into a sibling's sub-agents |
| `GLOBAL` | Any agent in the tree, the main agent included |

With `EXPLICIT`, the shape of the tree is the shape of the conversation: work flows down and comes back up. `DIRECT_PEERS` lets specialists pass work sideways, for example `OrderAgent` handing a refund to `PaymentAgent` without going through the main agent. A handoff to a sibling goes to that sibling itself, never past it into its own sub-agents, so the sibling still decides how its part is done. `GLOBAL` makes every agent reachable, which suits a utility agent that any other may need.

The two settings combine freely: an agent can read the whole tree (`context_visibility=GLOBAL`) and still call only its own sub-agents (`handoff_scope=EXPLICIT`).

:::note
Older configurations use a single `handoff_visibility` key. It is still accepted, and translated into the two settings above with a warning: replace it with `context_visibility` and `handoff_scope`.
:::

## Passing an answer through unchanged

Often a sub-agent's answer is already what the user should read, and having the main agent rewrite it only costs a model call and risks changing it. For that, every agent can also call `FORWARD_VERBATIM`: it names the result of a sub-agent or of a tool in its history, and that text becomes the agent's answer exactly as it was written.

The model decides when to use it. It is told to do so only for a complete answer that reads well on its own, and not for a partial answer, a raw record or a list of ids, an answer that points to sources outside its own text, a request for a different format, length or language, or one that combines several results. To turn it off for an agent, give it a planner with `forward_verbatim_enabled=False`:

```python
from domyn.agents.planner_strategy import ToolUseWithThoughtPlannerStrategy

main_agent = Agent(
    ...,
    planner=ToolUseWithThoughtPlannerStrategy(forward_verbatim_enabled=False),
)
```

## The same team in YAML

In a [YAML configuration](configuration.md), declare each agent once and list sub-agents by name. The two settings are written in lowercase:

```yaml
version: "1.0"
entrypoint: CustomerService

llm_providers:
  - name: main_llm
    type: openai
    model_name: gpt-4o
    api_key: ${OPENAI_API_KEY}

planner_strategies:
  - name: planner
    type: tool_use_thought

agents:
  - name: OrderAgent
    description: Retrieves information about customer orders
    instruction: You help users check order status and details.
    llm_provider: main_llm
    planner_strategy: planner

  - name: PaymentAgent
    description: Handles payment questions and refunds
    instruction: You help users with payment questions and refunds.
    llm_provider: main_llm
    planner_strategy: planner
    context_visibility: peers

  - name: CustomerService
    description: Main customer service agent
    instruction: Route each question to the right specialist.
    llm_provider: main_llm
    planner_strategy: planner
    sub_agents:
      - OrderAgent
      - PaymentAgent
```

`entrypoint` names the agent the user talks to.

## Next steps

- [Remote agents](remote-agents.md): team members that run in another process, including LangGraph graphs.
- [Events and streaming](events-and-streaming.md): follow a team's work event by event.
