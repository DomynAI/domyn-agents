# Handoffs and hooks

This example shows three things:

- how to build a multi-agent system, in which one agent hands requests to other agents;
- how to control what each agent sees and which agents it can hand work to;
- how hooks run your own code around tool calls.

The configuration defines three agents of an online coffee shop. You only talk to
`FrontDesk`, which finds the customer and then hands the request to one of two specialists:
`OrderAgent`, for parcels and deliveries, or `BillingAgent`, for payments and refunds.
`FrontDesk` then answers you based on what the specialist found.

## Features

- **Sub-agents and handoffs.** `FrontDesk` lists `OrderAgent` and `BillingAgent` as its
  `sub_agents`. To `FrontDesk`'s model, each sub-agent is one more tool, described by the
  sub-agent's `description`.
- **Context visibility.** By default, an agent sees only its own work and the work of its
  sub-agents. `BillingAgent` has `context_visibility=PEERS`, so it also sees what
  `OrderAgent` found.
- **Handoff scope.** `OrderAgent` has `handoff_scope=DIRECT_PEERS`, so it can hand a refund
  straight to `BillingAgent` without going back through `FrontDesk`.
- **A hook that decides approvals.** `refund_policy` runs before you are asked to approve a
  refund (`HookPhase.PRE_TOOL_APPROVAL`). It allows refunds up to 50 EUR, refuses refunds
  over 500 EUR, and leaves the ones in between for you to decide.
- **A hook that only observes.** `audit` runs after every tool call
  (`HookPhase.POST_TOOL_CALL`) and prints it. It is a listener, so it cannot change the
  result.

## Files

- `config.py`: the configuration of the system, made of the three agents, their tools and
  the two hooks. The shop's data is a few made-up dictionaries at the top of the file.

## Run it

Copy `.env.example` to `.env` in this folder and fill it in (see [Setup](../README.md#setup)).

```bash
cd examples/handoffs-and-hooks
uv run domyn run config.py
```

## Try it

Start with "I'm Anna Rossi, where are my orders?", then ask for a refund of each order:

- "The moka pot arrived broken": the refund is 35 EUR, so the hook allows it without asking
  you.
- "The milk frother too": the refund is 89 EUR, so you are asked to approve it.
- "The espresso machine never arrived": the refund is 649 EUR, so the hook refuses it.

Every tool call also prints an `[audit]` line.

## Learn more

- [Orchestration](../../docs/docs/guides/orchestration.md)
- [Hooks](../../docs/docs/guides/hooks.md)
- [Approvals](../../docs/docs/guides/approvals.md)
