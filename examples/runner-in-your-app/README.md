# The Runner in your app

This example shows four things:

- how to run an agentic system from your own application, using the `Runner` instead of
  `domyn run`;
- how to answer tool approvals in code;
- how to keep conversations across runs;
- how to give an agent long-term memory.

`domyn run` is itself an application built on the `Runner`. `app.py` is a much smaller one,
a terminal chat, which shows what your own application would do.

## Features

- **The Runner and its events.** Each message is one `runner.run(...)` call, which yields an
  event for every step. These events include the answer as it is written
  (`RESPONSE_TOKEN`), each tool call (`TOOL_START`) and each tool result (`TOOL_END`).
  `app.py` prints them.
- **Approvals in code.** When a tool needs approval, the turn ends with a
  `TOOL_NEEDS_FEEDBACK` event. The application then asks its user, and runs the next turn
  with a `TOOL_FEEDBACK` event that carries the answer.
- **Saved conversations.** The `Runner` keeps conversations in a SQLite file
  (`SqliteConversationManager`), so you can continue a conversation in a later run by
  giving its id.
- **Long-term memory.** The agent has `AGENTIC` memory, which means it gets a `remember`
  tool and decides for itself what to keep. Whatever it keeps outlives the conversation and
  is recalled in every new one.

## Files

- `config.py`: the configuration of the system. It defines a single agent, the assistant of
  an online coffee shop, with two tools and long-term memory. One of the tools,
  `refund_order`, waits for approval.
- `app.py`: the application, which runs the agent of `config.py` with a `Runner`.

## Run it

Copy `.env.example` to `.env` in this folder and fill it in (see [Setup](../README.md#setup)).

```bash
cd examples/runner-in-your-app
uv run --env-file .env python app.py
```

## Try it

1. Tell the agent something about yourself, for example "I'm Marco, I only drink decaf".
   The agent keeps it with `remember`.
2. Ask for a refund of order 1001. The application asks you to confirm in the terminal.
3. Quit with an empty line.
4. Run the application again. It starts a new conversation, and the agent still knows who
   you are.
5. To continue the same conversation instead, run the application with the id it printed
   when you quit.

The memories and the conversations are kept in your system's temporary folder, in
`domyn_example_memory.jsonl` and `domyn_example_conversations.db`. Delete both files to
start over.

## Learn more

- [Events and streaming](../../docs/docs/guides/events-and-streaming.md)
- [Conversations and memory](../../docs/docs/guides/conversations-and-memory.md)
- [Approvals](../../docs/docs/guides/approvals.md)
