# Tools and approvals

This example shows three things:

- how to turn Python functions into tools;
- how to make a tool wait for your approval before it runs;
- how a tool can keep data in the conversation state.

The configuration defines a single agent, the support agent of an online coffee shop. It has
three tools, which work on a small table of made-up orders.

## Features

- **Function tools.** Any Python function can be a tool. The model reads the function's
  docstring as the tool's description, and its type annotations as the schema of the
  arguments. `get_order` runs without asking you.
- **Approval with a yes or no (`ApprovalType.USER_CONFIRM`).** `refund_order` asks you to
  confirm before it runs.
- **Approval with editable arguments (`ApprovalType.PARAMS_UPDATE`).**
  `change_shipping_address` shows you its arguments before it runs, and you can change them.
- **Conversation state.** `refund_order` records the orders it has refunded in the
  conversation state. Because of this, it refuses to refund the same order twice, even
  several messages later. The model never sees the state.

## Files

- `config.py`: the configuration of the system, made of the three tools and the agent that
  uses them.

## Run it

Copy `.env.example` to `.env` in this folder and fill it in (see [Setup](../README.md#setup)).

```bash
cd examples/tools-and-approvals
uv run domyn run config.py
```

## Try it

- "Where is order 1001?": the agent calls `get_order` without asking you.
- "Send order 1002 to Via Roma 1, Milan instead": the agent shows you the new address, and
  you can edit it before the change is made.
- "Refund order 1001, it arrived broken": the agent asks you to confirm. If you then ask for
  the same refund again, the tool refuses it, because the state shows the order is already
  refunded.

## Learn more

- [Tools](../../docs/docs/guides/tools.md)
- [Approvals](../../docs/docs/guides/approvals.md)
