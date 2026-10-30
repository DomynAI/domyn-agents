"""The agent of ``config.py`` inside your own application: the ``Runner`` and its events.

``domyn run`` is one application built on the ``Runner``; this is a minimal other one, a
terminal chat:

- every message is one ``runner.run(...)`` call, which yields an event for every step: the
  answer as it is written (``RESPONSE_TOKEN``), each tool call and its result;
- a tool that needs approval ends the turn with ``TOOL_NEEDS_FEEDBACK``; the application asks
  its user and runs the next turn with a ``TOOL_FEEDBACK`` event carrying the answer;
- conversations are kept in a SQLite file, so a conversation can be continued in another run
  by passing its id.

See README.md to run it.
"""

import asyncio
import os
import sys
import tempfile
import uuid

from config import agent

from domyn.agents import BaseEvent, ExecutionEventType, Part, Runner
from domyn.agents.conversation_manager import SqliteConversationManager

USER_ID = "user-1"

runner = Runner(
    conversation_manager=SqliteConversationManager(
        path=os.path.join(tempfile.gettempdir(), "domyn_example_conversations.db")
    )
)


async def run_turn(conversation_id: str, turn_input: BaseEvent) -> BaseEvent | None:
    """Run one turn and print it. Returns the event that needs an answer, if any."""
    waiting = None
    async for event in runner.run(
        application_name="coffee-shop",
        user_id=USER_ID,
        conversation_id=conversation_id,
        root=agent,
        user_input=turn_input,
        run_config={"user_id": USER_ID},  # the user the memories belong to
    ):
        match event.event_type:
            case ExecutionEventType.RESPONSE_TOKEN:  # a piece of the answer
                print(event.content[0].text or "", end="", flush=True)
            case ExecutionEventType.RESPONSE if not event.is_partial:
                # Sent instead of the tokens when the answer was not streamed.
                print("".join(part.text or "" for part in event.content or []), end="")
            case ExecutionEventType.TOOL_START:
                # `reasoning` is the model's note on why it calls the tool, not an argument.
                arguments = {k: v for k, v in event.action.parameters.items() if k != "reasoning"}
                print(f"  -> {event.action.name} {arguments}")
            case ExecutionEventType.TOOL_END:
                print(f"  <- {event.action.observation}")
            case ExecutionEventType.TOOL_NEEDS_FEEDBACK:
                waiting = event
            case ExecutionEventType.ERROR_OCCURRED:
                print(f"Error: {event.error_message}")
    print()
    return waiting


def _ask(prompt: str) -> str:
    """Read a line; Ctrl-D reads as an empty one."""
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


async def main() -> None:
    conversation_id = sys.argv[1] if len(sys.argv) > 1 else f"conv-{uuid.uuid4().hex[:8]}"
    print(f"Conversation {conversation_id}. An empty line or Ctrl-D quits.")
    while message := _ask("you> "):
        turn_input = BaseEvent(
            event_type=ExecutionEventType.USER_INPUT,
            author="user",
            content=[Part(text=message)],
        )
        # A turn that stops for an approval is answered by the next one.
        while pending := await run_turn(conversation_id, turn_input):
            arguments = {p.name: p.value for p in pending.action.parameters}
            answer = _ask(f"Run {pending.action.name} {arguments}? [y/N] ")
            turn_input = BaseEvent(
                event_type=ExecutionEventType.TOOL_FEEDBACK,
                author="user",
                action=pending.action.model_copy(update={"approve": answer.lower() == "y"}),
            )
    print(f"Continue it with: uv run --env-file .env python app.py {conversation_id}")


if __name__ == "__main__":
    asyncio.run(main())
