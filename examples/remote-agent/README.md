# Remote agent

This example shows how to add an agent that runs in another process, or on another
machine, to a multi-agent system. The remote agent is reached through a relay. To the agent
that calls it, the remote agent is a sub-agent like any other: its turn runs remotely, and
its events come back through the relay.

The configuration defines the coffee shop's `FrontDesk`, which you talk to. `FrontDesk` has
one sub-agent, `Roastery`, which knows the beans in stock and the roasting schedule.
`Roastery` runs in a separate process.

## Features

- **Delegate agents.** `config.yaml` declares `Roastery` with `type: delegate` and a
  websocket `transport`, instead of a model and tools. `FrontDesk` lists it among its
  `sub_agents`.
- **Serving an agent on a relay.** `remote_agent.py` builds `Roastery` as an ordinary
  `Agent`, with its own model and tools, and serves it with an `AgentServer` on the relay's
  `roastery` channel. The `channel-id` header in `config.yaml` names that same channel.

## Files

- `config.yaml`: the configuration of the system you talk to. It defines `FrontDesk`, and
  `Roastery` declared as a delegate that is reached through the relay at
  `ws://localhost:8765`.
- `remote_agent.py`: the other side. It starts the relay and serves `Roastery` on it. In a
  real setup, another team could run this on another machine.
- `mock_relay.py`: a minimal in-memory relay. It is enough for this example, but it is not
  meant for production.

## Run it

Copy `.env.example` to `.env` in this folder and fill it in (see [Setup](../README.md#setup)).

Start the remote side first, and leave it running:

```bash
cd examples/remote-agent
uv run --env-file .env python remote_agent.py
```

Then, in another terminal:

```bash
cd examples/remote-agent
uv run domyn run config.yaml
```

## Try it

- "Is the Ethiopian Yirgacheffe in stock?"
- "When is the next decaf roast?"

`FrontDesk` asks `Roastery`, which answers from the other process.

## Learn more

- [Remote agents](../../docs/docs/guides/remote-agents.md)
- [Orchestration](../../docs/docs/guides/orchestration.md)
