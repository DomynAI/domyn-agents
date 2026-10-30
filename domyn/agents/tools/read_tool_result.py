"""Read back a tool result the model was not shown whole.

The tool-use context manager cuts a long result down to its start and its end,
and compaction drops old ones; both say so where they do it, with the call that
reads the result back (``read_call``). Nothing is stored for that: the whole
result is already in the conversation, as the event its tool produced, so the
reader looks it up among the events the calling agent can see. What visibility
hides from an agent stays hidden here too.

Offsets count characters of a result *as the model is shown it*, which is not
always its observation alone: what a streamable tool showed the user while it ran
comes after it. ``ShownTexts`` builds that text, and the context manager and this
tool both use it on the same events, in the same order, so the offset a note
gives is the offset this tool reads at.

No agent lists this tool. The tool-use context manager offers it to every agent
with tools of its own or agents to hand off to, none of its tools by that name;
``Agent.get_tool_by_name`` resolves its name, and the runner hands it a
``ToolResults``, made by the agent's context manager, through the reserved
``tool_results`` argument.
"""

# No `from __future__ import annotations`: FunctionTool reads the annotations.
import json
from collections.abc import Callable, Iterable
from typing import Any, Self

from domyn.agents.core import (
    METADATA_CONVERTED_TO_AGENT_END,
    BaseEvent,
    ExecutionEventType,
    ToolAction,
)
from domyn.agents.core.utils import parts_to_text
from domyn.agents.tools.function_tool import FunctionTool

# What a search shows of each match: this much of its line on either side.
_BEFORE = 80
_AFTER = 120
# Between the text a read returns and its note.
_GAP = "\n\n"

_RESULT_EVENTS = (ExecutionEventType.TOOL_END, ExecutionEventType.TOOL_ERROR)


class ShownTexts:
    """The text of each tool result as a model is shown it: its observation, then
    what the tool showed the user while it ran.

    Filled walking the events in the order a request renders them: a chunk a tool
    streamed is kept (``take_chunk``) until the result of its call arrives, which
    takes it (``text_of``).
    """

    def __init__(self) -> None:
        self._streamed: dict[str, list[str]] = {}

    def take_chunk(self, event: BaseEvent) -> bool:
        """Keep *event* if a streamable tool showed it to the user while it ran,
        and say whether it did."""
        # `Any`: mypy reads `BaseEvent.action` as `None`, its declared union being
        # built at runtime.
        action: Any = event.action
        if (
            event.event_type != ExecutionEventType.RESPONSE
            or not event.is_partial
            or not isinstance(action, ToolAction)
            or event.metadata.get(METADATA_CONVERTED_TO_AGENT_END)
        ):
            return False
        text = parts_to_text(event.content)
        if text:
            self._streamed.setdefault(action.call_id, []).append(text)
        return True

    def text_of(self, event: BaseEvent) -> str:
        """The text of the tool result *event*, with what its call streamed."""
        observation = str(getattr(event.action, "observation", None) or "")
        streamed = self._streamed.pop(getattr(event.action, "call_id", ""), None)
        if not streamed:
            return observation
        return (
            f"{observation}{_GAP}Shown to the user while the tool ran:\n" + "".join(streamed)
        ).lstrip()


class ToolResults:
    """The tool results an agent sees, as its model is shown them: what
    ``read_tool_result`` reads.

    Made by the agent's context manager, which knows both halves: the events the
    agent's requests are rendered from, and how much of one result reaches its
    model whole (``budget``, what a read is sized to).

    Never copied. A function tool deep-copies its arguments on every call: a copy
    of this would be a copy of the conversation. The tool's pre-hooks never see it:
    the runner hands it to the tool after them.
    """

    def __init__(self, events: Callable[[], Iterable[BaseEvent]], budget: int) -> None:
        self._events = events
        # Characters a read may return, its note included, and still reach the
        # model whole.
        self.budget = budget

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        return self

    def text(self, result_id: str) -> str:
        """The result *result_id*, as the model is shown it.

        Raises ``ValueError`` when the agent sees no tool result by that id.
        """
        texts = ShownTexts()
        for event in self._events():
            if texts.take_chunk(event):
                continue
            action: Any = event.action  # see `ShownTexts.take_chunk`
            if event.event_type in _RESULT_EVENTS and action:
                # Every result takes its call's chunks, as when it is rendered,
                # not only the one asked for.
                text = texts.text_of(event)
                if event.event_id == result_id:
                    return text
        raise ValueError(
            f"No tool result with id {result_id!r} in this conversation. Use the "
            "result_id given where the result was cut or dropped."
        )


def read_tool_result(
    result_id: str,
    offset: int = 0,
    find: str = "",
    tool_results: ToolResults | None = None,
) -> str:
    """Read a tool result that was cut or dropped to save context.

    Where a result was cut or dropped, a note says so and gives the call that
    reads it. A read returns the result from offset on, as much as fits, then a
    note with the call that reads on. With find it searches instead, from offset
    on, and lists the offset of each place the text occurs, with its line around
    it.

    Args:
        result_id: The result's id, as the note gives it.
        offset: Where to start, in characters from the start of the result.
        find: Text to look for instead of reading, matched exactly as written.
    """
    if tool_results is None:
        raise ValueError("This agent has no tool results to read back.")
    text = tool_results.text(result_id)
    start = _offset(offset, len(text))
    if find:
        return _search(text, find, start, result_id, tool_results.budget)
    return _window(text, start, result_id, tool_results.budget)


READ_TOOL_RESULT = FunctionTool(func=read_tool_result)  # type: ignore[no-untyped-call]


def read_call(result_id: str, offset: int = 0, find: str = "") -> str:
    """The call that reads a result from ``offset``, written as notes give it."""
    arguments = [f"result_id={json.dumps(result_id)}"]
    if offset:
        arguments.append(f"offset={offset}")
    if find:
        arguments.append(f"find={json.dumps(find, ensure_ascii=False)}")
    return f"{READ_TOOL_RESULT.name}({', '.join(arguments)})"


def read_back(event: BaseEvent) -> str | None:
    """The call that reads the tool result *event* back; ``None`` without an id.

    For a read of another result it is that same read: it gives the same text
    again, where reading the read back would number a text of its own.
    """
    arguments = getattr(event.action, "parameters", None)
    if (
        event.event_type == ExecutionEventType.TOOL_END
        and getattr(event.action, "name", None) == READ_TOOL_RESULT.name
        and isinstance(arguments, dict)
    ):
        try:
            return read_call(
                str(arguments["result_id"]),
                _number(arguments.get("offset", 0)),
                str(arguments.get("find") or ""),
            )
        except (KeyError, ValueError):
            pass
    return read_call(event.event_id) if event.event_id else None


def characters(start: int, end: int, total: int) -> str:
    """A span of a result, as the notes about it write it."""
    return f"characters {start:,} to {end:,} of {total:,}"


def _number(value: Any) -> int:
    # The schema says integer and a model may still send "7971".
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"offset must be a number of characters, not {value!r}.") from None


def _offset(value: Any, total: int) -> int:
    offset = _number(value)
    if not 0 <= offset <= total:
        raise ValueError(f"offset {offset} is outside the result, which has {total} characters.")
    return offset


def _window(text: str, start: int, result_id: str, budget: int) -> str:
    total = len(text)

    def read_on(end: int) -> str:
        return f"[{characters(start, end, total)}; read on with {read_call(result_id, end)}]"

    # No end is wider than the total, so this is the longest the note gets. At
    # least one character, so a read always gets somewhere.
    size = max(budget - len(_GAP) - len(read_on(total)), 1)
    end = min(start + size, total)
    if end < total:
        note = read_on(end)
    else:
        note = f"[{characters(start, end, total)}, the end of the result]"
    return text[start:end] + _GAP + note


def _search(text: str, find: str, start: int, result_id: str, budget: int) -> str:
    def more(listed: int, at: int) -> str:
        return (
            f"[{listed:,} matches listed; more follow — list them with "
            f"{read_call(result_id, at, find)}]"
        )

    # Neither number is wider than the length of the text: the longest the note gets.
    room = budget - len(more(len(text), len(text))) - 1
    lines: list[str] = []
    used = 0
    at = text.find(find, start)
    while at != -1:
        line = f"offset={at}: {_around(text, at)}"
        # The first match is always listed, so a search always gets somewhere.
        if lines and used + len(line) + 1 > room:
            lines.append(more(len(lines), at))
            break
        lines.append(line)
        used += len(line) + 1
        at = text.find(find, at + len(find))
    if not lines:
        return f"No match for {json.dumps(find, ensure_ascii=False)} from offset {start} on."
    return "\n".join(lines)


def _around(text: str, at: int) -> str:
    """A short line of text around ``at``, never crossing a line break."""
    start = max(at - _BEFORE, 0)
    newline = text.rfind("\n", start, at)
    if newline != -1:
        start = newline + 1
    end = min(at + _AFTER, len(text))
    newline = text.find("\n", at, end)
    if newline != -1:
        end = newline
    return text[start:end]
