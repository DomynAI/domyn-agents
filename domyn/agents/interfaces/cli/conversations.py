"""Past conversations: where they are kept, and how to get back into one.

The session used to hold its history in memory, so closing the terminal ended
the conversation in the strongest sense — nothing was written anywhere and the
id printed on the way out named something that no longer existed. Here it goes
to SQLite under the same directory as the rest of the CLI's state, and the id
becomes an address.

Resuming needs nothing from this module beyond the id: ``Runner.run`` loads an
existing conversation, its state and its agent stack when handed one it already
knows. What is here is the part the runner cannot do — finding the id again.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from domyn.agents.interfaces.cli.config import config_dir
from domyn.agents.logger import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from rich.console import Console

    from domyn.agents.conversation_manager import BaseConversationManager, ConversationInfo
    from domyn.agents.core import BaseEvent
    from domyn.agents.interfaces.cli.theme import Theme

STORE_FILENAME = "conversations.db"

PREVIEW_CHARS = 60

# Turns drawn back on resume. A terminal keeps about a thousand lines of
# scrollback by default (iTerm2, VS Code), some thirty turns with tool calls:
# anything older would be printed only to scroll out of reach.
HISTORY_TURNS = 20


def store_path() -> Path:
    """Beside the config, so a project that pins its own agent keeps its own
    history and does not spill into the user's global one."""
    return config_dir() / STORE_FILENAME


def build_manager() -> BaseConversationManager:
    from domyn.agents.conversation_manager import SqliteConversationManager

    path = store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteConversationManager(path=str(path))


async def recent(
    manager: BaseConversationManager, application_name: str, user_id: str
) -> list[ConversationInfo]:
    """Conversations newest first, or an empty list if they cannot be read.

    Unreadable is not a crash: the store is written by other versions of this
    code, and a history that cannot be listed must not stop a session starting.
    """
    try:
        found = await manager.get_user_conversations(application_name, user_id)
    except Exception as exc:
        get_logger().warning(f"could not list conversations: {type(exc).__name__}")
        get_logger().debug(f"conversation listing detail: {exc}")
        return []
    return sorted(found, key=lambda info: info.updated_at, reverse=True)


def preview(info: ConversationInfo) -> str:
    """The first thing the user said, which is what they will recognise it by."""
    from domyn.agents.core import ExecutionEventType

    for event in info.events_history:
        if event.event_type != ExecutionEventType.USER_INPUT:
            continue
        text = " ".join((part.text or "").strip() for part in event.content or [])
        text = " ".join(text.split())
        if text:
            return (text[: PREVIEW_CHARS - 1] + "…") if len(text) > PREVIEW_CHARS else text
    return "(nothing was said)"


def exchanges(info: ConversationInfo) -> int:
    from domyn.agents.core import ExecutionEventType

    return sum(1 for e in info.events_history if e.event_type == ExecutionEventType.USER_INPUT)


def label(info: ConversationInfo, now: datetime | None = None) -> str:
    """One row of the picker: when, how long, and what it was about."""
    count = exchanges(info)
    turns = f"{count} message{'s' if count != 1 else ''}"
    return f"{_ago(info.updated_at, now)}  ·  {turns}  ·  {preview(info)}"


def _ago(moment: datetime, now: datetime | None = None) -> str:
    minutes = ((now or datetime.now()) - moment).total_seconds() / 60
    if minutes < 1.5:
        return "just now"
    for size, unit in ((60, "min"), (24, "hour"), (7, "day")):
        if minutes < size:
            value = int(minutes)
            return f"{value} {unit}{'s' if value != 1 and unit != 'min' else ''} ago"
        minutes /= size
    return moment.strftime("%d %b")


def pick(console: Console, theme: Theme, found: list[ConversationInfo]) -> ConversationInfo | None:
    """Choose one, or ``None`` if there is nothing to choose or the user backed out."""
    from domyn.agents.interfaces.cli.select import select

    if not found:
        console.print("  No past conversations yet.", style=theme.style("meta"), highlight=False)
        return None

    rows = [label(info) for info in found]
    # Rows can repeat — two conversations opened and closed in the same minute
    # with the same first message — and `select` answers with the string, so an
    # index has to survive the round trip.
    numbered = [f"{n + 1}. {row}" for n, row in enumerate(rows)]
    chosen = select(console, theme, "Resume which conversation?", numbered)
    if chosen is None:
        return None
    return found[numbered.index(chosen)]


def render_history(
    console: Console,
    theme: Theme,
    info: ConversationInfo,
    *,
    verbose: bool = False,
    turns: int = HISTORY_TURNS,
) -> None:
    """Put the last turns of the conversation back on screen, as they looked.

    The model is handed the whole history regardless — this is for the person,
    who otherwise gets a prompt identical to a fresh one and has to remember
    what they were doing. It used to be the last three exchanges, one dimmed
    line each and cut at the edge, which read as a resume that had lost them.

    Drawn by the same renderer a live turn uses, so an answer, a tool call or an
    error looks the same coming back as it did going out. Whole turns, the last
    ``turns`` of them, and a line saying how many came before: older ones would
    only scroll past the terminal's scrollback, and the count says they were
    kept rather than lost.
    """
    from domyn.agents.core import ExecutionEventType
    from domyn.agents.interfaces.cli.renderer import EventRenderer

    console.print(
        f"  Resumed {info.conversation_id}  ·  {exchanges(info)} messages  ·  "
        f"last {_ago(info.updated_at)}",
        style=theme.style("meta"),
        highlight=False,
    )
    console.print()
    grouped = _turns(replayable(info.events_history))
    hidden = max(0, len(grouped) - turns)
    if hidden:
        console.print(
            f"  … {hidden} earlier turn{'s' if hidden != 1 else ''} not shown",
            style=theme.style("meta"),
            highlight=False,
        )
        console.print()
    # No spinner: nothing is running, and a spinner resumed after every event
    # would draw an activity line through the middle of the replay.
    renderer = EventRenderer(console, theme, verbose=verbose)
    for n, group in enumerate(grouped[hidden:]):
        if n:
            renderer.finish()
            console.print()
        for event in group:
            if event.event_type is not ExecutionEventType.USER_INPUT:
                renderer.render(event)
                continue
            renderer.echo_user(_text(event))
            if event is group[0]:
                # After the anchor, not before: the editor leaves this line live,
                # and an answer straight under it has no blank line to separate.
                # A message steered in later is still the person's own line, and
                # the answer below it gets the separation it had live.
                renderer.begin_turn()
    renderer.finish()
    if grouped:
        console.print()
    console.print(
        "  ── resumed here ──" if info.events_history else "  (nothing was said)",
        style=theme.style("meta"),
        highlight=False,
    )
    console.print()


def _turns(events: list[BaseEvent]) -> list[list[BaseEvent]]:
    """The events split into turns, each opening on the person's message.

    A message steered into a running turn shares its turn id and stays inside
    it; one with none (stored by an older version) is taken as a turn of its
    own. Anything before the first message is a group of its own.
    """
    from domyn.agents.core import ExecutionEventType

    grouped: list[list[BaseEvent]] = []
    turn: str | None = None
    for event in events:
        opens = event.event_type is ExecutionEventType.USER_INPUT and (
            not grouped or event.turn_id is None or event.turn_id != turn
        )
        if opens:
            turn = event.turn_id
            grouped.append([])
        elif not grouped:
            grouped.append([])
        grouped[-1].append(event)
    return grouped


def _text(event: BaseEvent) -> str:
    return " ".join(part.text for part in event.content or [] if part.text)


def replayable(events: list[BaseEvent]) -> list[BaseEvent]:
    """The stored events a live session would have shown, in order.

    The store keeps more than the screen did: the approval round trip, which the
    session drew as its own prompt rather than as an event and must not be
    drawn as a panel waiting for an answer that was given long ago; the input
    asking for a compaction; and a sub-agent's answer, which reached the screen
    as the return to its caller rather than as an answer of its own.
    """
    from domyn.agents.core import METADATA_CONVERTED_TO_AGENT_END, ExecutionEventType

    not_replayed = {
        ExecutionEventType.RESPONSE_TOKEN,
        ExecutionEventType.TOOL_NEEDS_FEEDBACK,
        ExecutionEventType.TOOL_FEEDBACK,
        ExecutionEventType.COMPACTION_REQUESTED,
    }
    return [
        event
        for event in events
        if event.event_type not in not_replayed
        and not (
            event.event_type is ExecutionEventType.RESPONSE
            and event.metadata.get(METADATA_CONVERTED_TO_AGENT_END)
        )
    ]


async def find(
    manager: BaseConversationManager,
    application_name: str,
    user_id: str,
    conversation_id: str,
) -> ConversationInfo | None:
    """One conversation by id, or ``None`` when there is no such conversation.

    An id that was never used would otherwise be created on first use and look
    like a resume that lost everything.
    """
    try:
        if not await manager.check_conversation_exists(application_name, user_id, conversation_id):
            return None
        return await manager.get_conversation(application_name, user_id, conversation_id)
    except Exception as exc:
        get_logger().warning(f"could not read conversation: {type(exc).__name__}")
        get_logger().debug(f"conversation read detail: {exc}")
        return None
