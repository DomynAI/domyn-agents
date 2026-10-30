"""Renders runner events to the terminal.

Extracted from ``domyn run`` so any terminal front-end can reuse it, and so the
rendering rules live in one place instead of being tangled with the
conversation loop.

The transcript has a **two-level shape**, because a turn has two kinds of output
and conflating them is what makes an agent CLI hard to read:

* The *work* — reasoning, tool calls, tool results, handoffs — is indented and
  dimmed. It is a trace: worth being able to follow, not worth reading closely.
* The *answer* is flush left behind a brand-coloured marker, with a blank line
  above it. It is what the user asked for, and it should be findable by scrolling
  without reading anything.

Together with the brand-coloured user prompt, that gives two anchors of the
same colour per exchange: everything between them is one turn.

Other behaviours worth calling out:

* **Reasoning is shown, not hidden.** ``tool_use_thought`` extracts the model's
  reasoning and emits it as ``Part(thought=...)`` on ``LLM_END``. That event is
  otherwise verbose-only, so the reasoning used to be invisible unless the user
  turned on internal events — which also turned on everything else. The thought
  is now pulled out of it and shown on its own.
* **Thoughts are de-duplicated.** The same reasoning is attached both to
  ``LLM_END`` and to every action it produced, so a turn with three tool calls
  would print it four times.
* **Streaming.** ``RESPONSE_TOKEN`` events are printed as they arrive; the final
  ``RESPONSE`` that follows carries the same text in full, so it must not be
  printed again. Note the ordering trap: closing the stream is what clears the
  "a stream was open" flag, so the flag has to be read *before* closing, or the
  answer is printed twice.
* **Markdown, where it can be seen as such.** On a terminal that takes colour,
  the answer and the reasoning are rendered as markdown by ``markdown.py``, a
  line at a time as they stream; that module says why a line and not a token.
  Anywhere else — a pipe, a file, ``NO_COLOR`` — the source is printed as it
  came: without styles it is the more readable of the two, and
  ``domyn run … > out.md`` keeps what the model wrote. While the steering
  editor holds the screen, the part of the answer not final yet is drawn by the
  editor instead (:meth:`EventRenderer.preview`).
* **Streaming and reasoning overlap, and cannot fully be separated.** A model
  without native thinking writes its reasoning as a preamble in ``content``, and
  nothing distinguishes that from an answer until the tool calls arrive — by
  which point it has already been streamed to the screen. It is therefore shown
  at answer level rather than indented, and the renderer only makes sure it is
  not then printed a *second* time as a thought. Models that populate
  ``reasoning_content`` or ``thinking_blocks`` do not have this problem: their
  reasoning never enters ``content``, so it is classified correctly from the
  start and rendered as part of the trace.
* **Unknown events are silent by default.** The previous implementation printed
  ``Event: <type>`` for anything it did not handle. That turned control-flow
  events into noise and, worse, made ``TOOL_NEEDS_FEEDBACK`` look handled when
  it was not. Unrecognised events now show only in verbose mode.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from rich.text import Text

from domyn.agents.core import BaseEvent, ExecutionEventType
from domyn.agents.interfaces.cli.calls import as_written, summary
from domyn.agents.interfaces.cli.markdown import MarkdownStream, render_markdown

if TYPE_CHECKING:
    from rich.console import Console

    from domyn.agents.interfaces.cli.prompt import OutputProgress
    from domyn.agents.interfaces.cli.spinner import Spinner
    from domyn.agents.interfaces.cli.theme import Theme


def _thousands(tokens: int) -> str:
    """Tokens in readable form: nobody reads 184213 as a quantity."""
    if tokens < 1_000:
        return str(tokens)
    return f"{tokens / 1_000:.1f}k".replace(".0k", "k")


# Internal lifecycle events: meaningful when debugging, noise otherwise.
_VERBOSE_ONLY = frozenset(
    {
        ExecutionEventType.LLM_START,
        ExecutionEventType.LLM_END,
        ExecutionEventType.AGENT_START,
        ExecutionEventType.HOOK_START,
        ExecutionEventType.HOOK_END,
        ExecutionEventType.HOOK_UPDATE,
        ExecutionEventType.TOOL_UPDATE,
    }
)

_ERRORS = frozenset(
    {
        ExecutionEventType.ERROR_OCCURRED,
        ExecutionEventType.TOOL_ERROR,
        ExecutionEventType.LLM_PROVIDER_ERROR,
        ExecutionEventType.HOOK_ERROR,
    }
)


# Indent for the trace: everything the agent does on the way to an answer. Deep
# enough to read as subordinate to the answer, shallow enough not to waste width.
_WORK = "  "
_WORK_DETAIL = "    "

# Rows of the answer an editor shows while they are not final yet, at least: see
# `preview`, which allows half the window when it is taller.
PREVIEW_ROWS = 8
# How often a streaming answer asks for that region to be repainted, at most. The
# editor's own timer repaints ten times a second; tokens arrive faster than that,
# and the text would advance in visible steps.
_PREVIEW_REPAINT_S = 1 / 30
# An OSC 8 hyperlink, opening or closing.
_HYPERLINK = re.compile(r"\x1b\]8;[^\x1b\x07]*(?:\x1b\\|\x07)")


def _track_step(spinner: Spinner, kind: ExecutionEventType) -> None:
    """Tell the spinner when a model call or a tool starts and stops, for its clock.

    Once the answer streams it goes quiet altogether until the call ends.

    Errors end a step too: a provider failure yields no LLM_END, and a clock left
    running through the error would be a lie. So does an approval: the human is
    the slow part then, and the session restarts the tool's clock once the
    answer is in (no second TOOL_START follows it).
    """
    if kind is ExecutionEventType.RESPONSE_TOKEN:
        spinner.streaming()
    elif kind is ExecutionEventType.LLM_START:
        spinner.thinking()
    elif kind is ExecutionEventType.TOOL_START:
        spinner.executing()
    elif kind in _STEP_ENDS or kind in _ERRORS:
        spinner.idle()


_STEP_ENDS = frozenset(
    {
        ExecutionEventType.LLM_END,
        ExecutionEventType.TOOL_END,
        ExecutionEventType.TOOL_NEEDS_FEEDBACK,
    }
)


class EventRenderer:
    """Stateful renderer: tracks the shape of the turn across events."""

    def __init__(
        self,
        console: Console,
        theme: Theme,
        *,
        verbose: bool = False,
        spinner: Spinner | None = None,
    ) -> None:
        self._console = console
        self._theme = theme
        self._verbose = verbose
        self._spinner = spinner
        self._streaming = False
        self._shown_thoughts: set[str] = set()
        self._streamed = ""
        self._wrote_work = False
        # Messages this front-end has already put on screen itself, waiting to be
        # matched against the USER_INPUT the runner echoes back when it absorbs
        # them. A list, not a set: two identical steers are two lines, and the
        # second must not be swallowed by the first.
        self._echoed: list[str] = []
        self._live_prompt = False
        self._pending = ""
        self._need_marker = False
        # Decided once, from what the output is: see the module docstring.
        self._markdown = theme.capabilities.color and bool(getattr(console, "is_terminal", False))
        self._answer: MarkdownStream | None = None
        # Called when `preview` has something new, for whoever draws it to repaint
        # sooner than its own timer would. At most every `_PREVIEW_REPAINT_S`.
        self.on_preview: Callable[[], None] | None = None
        self._previewed_at = 0.0
        # Where printed text stands on its way to the terminal, when it goes
        # through a proxy that writes later. Rows printed but not written yet stay
        # in `preview` until they are, each with the `sent` count that covers it.
        self.output: OutputProgress | None = None
        self._in_flight: list[tuple[int, Text]] = []

    def begin_turn(self) -> None:
        """Start a new exchange. Called once per user message.

        Resets the per-turn de-duplication of thoughts: the same reasoning in a
        later turn is new information and should be shown again.
        """
        self._shown_thoughts.clear()
        self._streamed = ""
        self._wrote_work = False
        # A message is absorbed by the turn it was typed into, so anything still
        # unmatched here belongs to a turn that is over and can only produce a
        # false suppression later.
        self._echoed.clear()
        self._pending = ""
        self._need_marker = False
        self._answer = None
        self._in_flight.clear()

    def set_live_prompt(self, live: bool) -> None:
        """Say whether an editor is holding the bottom of the screen.

        It changes how the answer is written, and the reason is a hard property of
        printing underneath a prompt_toolkit application: a write that does not end
        in a newline is **erased**. The editor repaints itself starting wherever the
        cursor was left, so the tail of that line is inside the region it clears on
        its next repaint. Partial writes close together survive by accident — the
        stdout proxy batches them into one write — which is why a tool call, whose
        two halves are printed back to back, looks fine while a token stream a
        quarter of a second apart disappears completely.

        So with a prompt live the answer is emitted a whole line at a time. It
        arrives in lines rather than in words, which is a real loss; it arriving at
        all is worth more.

        Rendered as markdown the answer never leaves a partial line in the first
        place (see ``markdown.py``), so this only changes the plain path: a
        terminal without colour, where the answer is printed as it came.
        """
        self._live_prompt = live

    def render(self, event: BaseEvent) -> None:
        """Render one event. Safe to call for every event the runner yields."""
        # The activity line is erased before anything is written and redrawn
        # after, so it never ends up interleaved with the transcript. Doing it
        # here, once, is why no individual print has to remember.
        if self._spinner is not None:
            self._spinner.pause()
            _track_step(self._spinner, event.event_type)
        try:
            self._render(event)
        finally:
            # Not while a token stream is open: the answer is arriving on that
            # same line, and a spinner would fight it for the cursor.
            if self._spinner is not None and not self._streaming:
                self._spinner.resume()

    def _render(self, event: BaseEvent) -> None:
        kind = event.event_type

        if kind is ExecutionEventType.RESPONSE_TOKEN:
            self._render_token(event)
            return

        # Any non-token event ends an open stream. Whether one *was* open has to
        # be captured first: closing it clears the flag, so a `RESPONSE` handler
        # that consulted it afterwards would always see "no stream" and print the
        # answer a second time under its own marker.
        was_streaming = self._streaming
        self._close_stream()

        if kind is ExecutionEventType.RESPONSE:
            self._render_response(event, already_streamed=was_streaming)
        elif kind is ExecutionEventType.LLM_END:
            # Reasoning rides on an otherwise verbose-only event: pull it out so
            # showing the thinking does not mean showing every lifecycle event.
            self._render_thoughts(event)
            self._render_lifecycle(event)
        elif kind is ExecutionEventType.TOOL_START:
            self._render_tool_start(event)
        elif kind is ExecutionEventType.TOOL_END:
            self._render_tool_end(event)
        elif kind is ExecutionEventType.AGENT_CALL:
            self._render_handoff(event, entering=True)
        elif kind is ExecutionEventType.AGENT_END:
            self._render_handoff(event, entering=False)
        elif kind in _ERRORS:
            self._render_error(event)
        elif kind is ExecutionEventType.CONTEXT_COMPACTION_START:
            self._render_compaction_start(event)
        elif kind is ExecutionEventType.CONTEXT_COMPACTED:
            self._render_compacted(event)
        elif kind is ExecutionEventType.DECISION_MADE:
            self._render_decision(event)
        elif kind is ExecutionEventType.USER_INPUT:
            self._render_steering(event)
        elif kind is ExecutionEventType.TOOL_NEEDS_FEEDBACK:
            # Owned by the conversation loop, which prompts the human. Rendering
            # it here would duplicate the approval panel.
            pass
        elif kind is ExecutionEventType.TOOL_UPDATE and self._progress_of(event):
            self._render_progress(event)
        elif kind in _VERBOSE_ONLY:
            self._render_lifecycle(event)
        elif self._verbose:
            self._work(f"[{kind.value}]", "meta")

    @staticmethod
    def _progress_of(event: BaseEvent) -> str:
        """The line a running command last printed, if this event carries one.

        Read defensively: TOOL_UPDATE is general-purpose and other producers put
        other things in it — the langgraph and hermes integrations send
        heartbeats through it. Anything without a progress line falls through to
        the lifecycle handler and stays behind `/verbose`.
        """
        for part in event.content or []:
            line = (getattr(part, "metadata", None) or {}).get("progress")
            if isinstance(line, str) and line.strip():
                return line.strip()
        return ""

    def _render_progress(self, event: BaseEvent) -> None:
        """What a command is printing, while it prints it.

        Shown outside `/verbose`, unlike the rest of TOOL_UPDATE: it is the only
        thing between the person and several minutes of nothing. Dim and
        indented — it is the command talking, not the agent.
        """
        self._work(f"{self._theme.symbols.tool_result} {self._progress_of(event)}", "meta")

    def _render_compaction_start(self, event: BaseEvent) -> None:
        """What is happening, while it happens.

        Always shown, not only under verbose: this is the one moment the agent
        loses information, and the person watching has to be able to connect it
        to behaviour that would otherwise look like a defect. With a summarizing
        strategy there is also a pause to account for."""
        meta = event.metadata or {}
        before = meta.get("tokens_before")
        detail = f" ({_thousands(before)} tokens)" if isinstance(before, int) else ""
        if meta.get("rejected"):
            detail += ": the model rejected the request as too long"
        self._work(f"{self._theme.symbols.tool_call} compacting context{detail}", "meta")

    def _render_decision(self, event: BaseEvent) -> None:
        """One line when a decision model changed something about this turn.

        Shown unprompted only when it **acted** — when a question was not asked,
        a tool was hidden, an instruction left out. Otherwise the session would
        narrate a background service on every turn, which is how a trace becomes
        wallpaper. The rest is behind `/verbose`, where the shadow runs that
        exist precisely to be inspected can be read.

        The confidence is always printed next to the verdict, because the number
        *is* the mechanism: a reader who cannot see that 0.79 fell below a 0.8
        threshold cannot tell a model that abstained from one that was not asked.
        """
        meta = event.metadata or {}
        use_case = str(meta.get("use_case", "decision"))
        acted = bool(meta.get("acted"))
        if not acted and not self._verbose:
            return

        summary = meta.get("summary")
        if isinstance(summary, str) and summary:
            detail = summary
        else:
            value = meta.get("value")
            confidence = meta.get("confidence")
            if meta.get("known") is False:
                detail = f"no answer ({meta.get('reason') or 'unknown'})"
            elif isinstance(confidence, (int, float)):
                detail = f"{value} ({float(confidence):.2f})"
            else:
                detail = str(value)

        latency = meta.get("latency_ms")
        timing = f" · {int(latency)} ms" if isinstance(latency, (int, float)) else ""
        mode = str(meta.get("mode", ""))
        # A shadow decision changed nothing by definition, and saying so is the
        # difference between reading the trace and misreading it.
        shadow = " · shadow, not applied" if mode == "shadow" else ""
        self._work(
            f"decision[{use_case}]: {detail}{timing}{shadow}",
            self._decision_style(meta, acted=acted),
        )

    def _decision_style(self, meta: dict, *, acted: bool) -> str:
        """The style for one decision line.

        A decision that acted is coloured by what it decided, because the two
        verdicts mean opposite things to the person reading: `allow` is a
        confirmation that did not happen, `deny` a call that was refused. Both
        are green/red for the same reason a diff is — the hue carries the
        meaning without a word being read.

        Everything else stays dim. A shadow run, a hidden tool, an instruction
        left out: those changed nothing about what the user was asked, and
        colouring them would make the trace compete with the answer for a
        decision that had no consequence.
        """
        if not acted:
            return "meta"
        value = meta.get("value")
        if meta.get("use_case") == "approvals":
            if value == "allow":
                return "decision_allow"
            if value == "deny":
                return "decision_deny"
        return "decision"

    def _render_compacted(self, event: BaseEvent) -> None:
        """How it went, with the numbers.

        If the strategy found nothing to do it says so: implying a recovery that
        did not happen is how people stop investigating a session that keeps
        being too large."""
        meta = event.metadata or {}
        if meta.get("requested"):
            # `/compact` says in its own words what it dropped, or why it dropped
            # nothing — "already dropped" and "nothing worth dropping" ask for
            # different next steps.
            note = " ".join(part.text for part in event.content if part.text).strip()
            saved = meta.get("tokens_saved")
            if isinstance(saved, int) and saved > 0:
                note = f"{note} (-{_thousands(saved)} tokens)"
            self._work(note or "context compacted", "meta" if meta.get("changed") else "warning")
            return
        before, after = meta.get("tokens_before"), meta.get("tokens_after")
        # No early return on `changed=False`: the case that recovers nothing is
        # also the one that most often stays over the limit, and leaving before
        # the warning below would report a dead end as a quiet nothing.
        if not meta.get("changed", True):
            self._work("nothing to compact: context left as it is", "warning")
        elif isinstance(before, int) and isinstance(after, int):
            saved = max(0, before - after)
            self._work(
                f"context compacted: {_thousands(before)} -> {_thousands(after)} tokens"
                f" (-{_thousands(saved)})",
                "meta",
            )
        else:
            self._work("context compacted", "meta")

        if meta.get("sufficient") is False:
            # It pruned, and it was not enough. Saying only the first part is how
            # the next line the person sees is a rejected request they have no
            # reason to connect to this.
            self._work(
                "still over the limit: dropping tool output is no longer enough"
                " — summarise or start a new conversation",
                "warning",
            )

    def finish(self) -> None:
        """Close any dangling stream. Call when a turn ends."""
        if self._spinner is not None:
            self._spinner.pause()
        self._close_stream()

    def preview(self) -> str:
        """The answer's unfinished end, for an editor to draw above its prompt.

        Rendered as markdown the answer is printed a line at a time, once a line
        is final, and whatever is not final yet — the line being written, a table
        still being collected — would otherwise be invisible until it is. An
        editor repaints its own region anyway, so it can show that part there
        and replace it on every repaint, the way a full-screen interface streams:
        the transcript gets final lines only, the region gets the rest.

        ANSI text, a line per row, or empty when there is nothing to show. The
        last rows only — half the window, at least :data:`PREVIEW_ROWS` — each cut
        to the width: the region is a window onto the end of the answer, and one
        that grew with a long table would push the prompt off the screen.
        Hyperlinks are left out: prompt_toolkit would print their escape as text,
        and they come back with the line once it is printed for good.

        Rows already printed but not on the terminal yet come first (see
        `_unwritten`), so a line is never on screen in both places, or in neither.
        """
        rows = self._unwritten()
        answer = self._answer
        if answer is not None and self._streaming:
            rows += answer.preview()
        rows = rows[-max(PREVIEW_ROWS, self._console.height // 2) :]
        if not rows:
            return ""
        width = self._console.width
        with self._console.capture() as capture:
            for row in rows:
                row = row.copy()
                row.truncate(width, overflow="ellipsis")
                self._console.print(row, soft_wrap=True)
        return _HYPERLINK.sub("", capture.get())

    # --- internals ---------------------------------------------------------

    def _work(self, text: str, style: str, *, detail: bool = False) -> None:
        """Print one line of the trace, indented under the answer."""
        self._wrote_work = True
        indent = _WORK_DETAIL if detail else _WORK
        self._console.print(
            f"{indent}{text}", style=self._theme.style(style), highlight=False, markup=False
        )

    def echo_user(self, text: str, *, label: str = "you") -> None:
        """Put a user message into the transcript, as the anchor it always was.

        The between-turns editor leaves its own line behind, which is where the
        anchor normally comes from. A read that runs *alongside* a turn cannot: its
        line is a live widget at the bottom of the screen, repainted as the
        transcript scrolls past it, and leaving that behind on submit drops a
        ``you >`` — with whatever the right-hand timer happened to say — into the
        middle of the agent's work, above the answer it precedes. So that editor
        erases itself and the message is printed here instead, in the same shape
        the editor would have left, at the point it took effect: when the runner
        absorbed it, when the command it held finally ran, or when it was handed to
        the next turn. Never on submit — that is what the queue rows above the
        prompt are for.

        Recorded as well as printed: the runner echoes an absorbed message back as
        ``USER_INPUT``, and without knowing this copy exists the renderer would
        show it a second time — see :meth:`_render_steering`.
        """
        flat = " ".join(text.split())
        if not flat:
            return
        if self._spinner is not None:
            self._spinner.pause()
        # Marker and message in one print, for the reason in `set_live_prompt`:
        # printed separately, the marker is a partial line and the editor's next
        # repaint takes it away, leaving the message with nothing in front of it.
        line = Text()
        line.append(f"{label} {self._theme.symbols.prompt} ", style=self._style("user_prompt"))
        line.append(flat)
        self._console.print(line, highlight=False, markup=False)
        self._echoed.append(flat)
        # It is not the trace, but the answer still needs separating from it: a
        # steer landing immediately above an unspaced answer reads as one block.
        self._wrote_work = True

    def _render_steering(self, event: BaseEvent) -> None:
        """Show a message that entered the turn from somewhere this front-end is not.

        The only ``USER_INPUT`` that reaches a consumer is one the runner absorbed
        partway through a turn: the event that opened the turn is stored, not
        yielded. A message typed here is printed by :meth:`echo_user` as it is
        absorbed — before this runs, by the front-end that recognised its id — and
        is therefore dropped here, or the same sentence would land twice in a row.
        Until then it was a row above the prompt rather than a line of transcript,
        which is why there is nothing to show for the seconds it spent queued.

        What is left is a message steered from *another* front-end on the same
        conversation, which has never been on this screen. That one is drawn as
        trace: it is something that happened to the turn rather than something the
        person at this terminal did, and the two anchors per exchange described at
        the top of this module must stay two.
        """
        text = " ".join(part.text for part in event.content if part.text).strip()
        if not text:
            return
        one_line = " ".join(text.split())
        if one_line in self._echoed:
            self._echoed.remove(one_line)
            return
        self._work(f"{self._theme.symbols.prompt} {self._truncate(one_line, limit=100)}", "meta")

    def _render_lifecycle(self, event: BaseEvent) -> None:
        if self._verbose:
            self._work(f"[{event.event_type.value}] {event.author}", "meta")

    def _close_stream(self) -> None:
        if not self._streaming:
            return
        if self._answer is not None:
            answer, self._answer = self._answer, None
            self._print_rows(answer.finish())
        elif self._live_prompt:
            # Whatever is left is the last line of the answer, and it is the one
            # that would otherwise be lost: it is by definition the part no
            # newline has arrived for.
            self._drain(final=True)
        else:
            self._console.print()
        self._streaming = False

    def _render_token(self, event: BaseEvent) -> None:
        for part in event.content:
            # Reasoning can arrive mid-stream, ahead of the answer it explains,
            # for a model whose thinking is a separate field. It closes any open
            # stream first: a thought belongs to the trace, not inside a
            # half-printed answer.
            if part.thought:
                self._close_stream()
                self._show_thought(part.thought)
            if part.text:
                if not self._streaming:
                    self._begin_answer()
                self._streaming = True
                self._streamed += part.text
                if self._answer is not None:
                    self._print_rows(self._answer.feed(part.text))
                    self._preview_changed()
                elif self._live_prompt:
                    self._pending += part.text
                    self._drain(final=False)
                else:
                    self._console.print(part.text, end="", highlight=False, markup=False)

    def _preview_changed(self) -> None:
        if self.on_preview is None:
            return
        now = time.monotonic()
        if now - self._previewed_at >= _PREVIEW_REPAINT_S:
            self._previewed_at = now
            self.on_preview()

    def _begin_answer(self) -> None:
        """Open the answer block: blank line, then the brand-coloured marker.

        The marker is what makes an answer findable when scrolling back past a
        long trace — the whole reason the trace is indented and this is not. The
        blank line only appears when there was a trace to separate from; on a
        plain question-and-answer turn it would just be a gap.
        """
        if self._wrote_work:
            self._console.print()
        if self._markdown:
            # Handed to the stream rather than printed: the marker goes in front of
            # the answer's first line, which is two columns narrower for it, and
            # only the stream knows when that line is final.
            self._answer = MarkdownStream(self._console, self._theme, lead=self._marker())
            return
        if self._live_prompt:
            # Held back rather than printed: on its own it is a partial line, and
            # `set_live_prompt` explains where those go. It is written as part of
            # the answer's first line instead.
            self._need_marker = True
            return
        self._console.print(
            f"{self._theme.symbols.agent} ",
            style=self._theme.style("agent_marker"),
            highlight=False,
            end="",
        )

    def _marker(self) -> Text:
        return Text(f"{self._theme.symbols.agent} ", style=self._theme.style("agent_marker"))

    def _unwritten(self) -> list[Text]:
        """Rows printed whose text the proxy has not written to the terminal yet.

        Shown by `preview` until it has: the proxy writes up to its flush interval
        later, and in between the editor repaints. Without them a line just made
        final would vanish from the region before it appears in the transcript.
        """
        output = self.output
        if output is None:
            return []
        self._in_flight = [entry for entry in self._in_flight if entry[0] > output.written]
        return [row for _, row in self._in_flight]

    def _print_rows(self, rows: list[Text]) -> None:
        for row in rows:
            # soft_wrap: the stream has already fitted every row to the width, and
            # a code line wider than the window is left for the terminal to fold —
            # one line to a copy-paste, where a break of ours would make it two.
            self._console.print(row, soft_wrap=True)
            # A mark per row, not per call: the proxy writes a line as soon as its
            # newline is in, so of two rows printed together the first can be on
            # the terminal while the second is not, and shown in both places.
            output = self.output
            if output is not None and output.sent > output.written:
                self._in_flight.append((output.sent, row))

    def _drain(self, *, final: bool) -> None:
        """Print every complete line the answer has produced so far.

        "Complete" means up to a newline the model wrote, or up to the width of
        the terminal — the second is what keeps a single-paragraph answer from
        arriving in one lump at the very end, which is what buffering strictly on
        newlines would do to the common case. Breaking at the last space before the
        edge puts the break where the terminal would have wrapped it anyway, so the
        result reads as a normally wrapped paragraph rather than as chunks.
        """
        while self._pending:
            head, newline, tail = self._pending.partition("\n")
            if newline:
                self._pending = tail
                self._emit_answer_line(head)
                continue
            limit = self._answer_width()
            if len(head) <= limit:
                break
            cut = head.rfind(" ", 0, limit + 1)
            if cut <= 0:
                cut = limit
            self._pending = head[cut:].lstrip(" ")
            self._emit_answer_line(head[:cut].rstrip(" "))

        if final:
            rest, self._pending = self._pending, ""
            if rest or self._need_marker:
                self._emit_answer_line(rest)

    def _answer_width(self) -> int:
        width = int(getattr(self._console, "width", 80) or 80)
        # The marker sits on the first line only, and takes two columns from it.
        return max(20, width - (2 if self._need_marker else 0))

    def _emit_answer_line(self, text: str) -> None:
        line = Text()
        if self._need_marker:
            line.append(f"{self._theme.symbols.agent} ", style=self._style("agent_marker"))
            self._need_marker = False
        line.append(text)
        self._console.print(line, highlight=False, markup=False)

    def _style(self, name: str) -> str | None:
        """A rich style, or ``None`` when the theme has colour switched off.

        ``Text.append`` wants ``None`` where ``Console.print`` is happy with an
        empty string.
        """
        return self._theme.style(name) or None

    def _render_thoughts(self, event: BaseEvent) -> None:
        for part in event.content:
            if part.thought:
                self._show_thought(part.thought)

    def _show_thought(self, thought: str) -> None:
        """Print reasoning once per turn, however many events carry it."""
        text = thought.strip()
        if not text or text in self._shown_thoughts:
            return
        if text and text in self._streamed:
            # Already on screen, streamed before it was known to be reasoning.
            # See the module docstring: with a model that writes its reasoning as
            # a preamble, nothing can tell the two apart until the tool calls
            # arrive, and by then the text has been printed. Printing it a second
            # time as a thought is strictly worse than leaving it where it is.
            self._shown_thoughts.add(text)
            return
        self._shown_thoughts.add(text)
        if self._markdown:
            # Rendered like the answer, but drawn as trace: indented, and in the
            # thought style underneath whatever the markdown adds.
            self._wrote_work = True
            self._print_rows(
                render_markdown(
                    text,
                    self._console,
                    self._theme,
                    margin=_WORK,
                    style=self._theme.style("thought"),
                )
            )
            return
        for line in text.splitlines():
            self._work(line, "thought")

    def _render_response(self, event: BaseEvent, *, already_streamed: bool = False) -> None:
        for part in event.content:
            if part.thought:
                self._show_thought(part.thought)
        # The final RESPONSE repeats, in full, text that already arrived
        # token-by-token. _close_stream has emitted the closing newline, so there
        # is nothing left to print.
        if already_streamed:
            return
        for part in event.content:
            if part.text:
                self._begin_answer()
                if self._answer is not None:
                    answer, self._answer = self._answer, None
                    self._print_rows(answer.feed(part.text) + answer.finish())
                    continue
                if self._live_prompt:
                    # The marker is waiting to be written with the first line of
                    # the answer rather than on its own; the same drain that serves
                    # the streaming path does both.
                    self._pending += part.text
                    self._drain(final=True)
                    continue
                # highlight=False as well as markup=False: rich colours numbers,
                # paths and quoted strings by default, which puts stray cyan
                # inside an answer that is meant to be neutral.
                self._console.print(
                    part.text,
                    style=self._theme.style("response"),
                    markup=False,
                    highlight=False,
                )

    def _render_tool_start(self, event: BaseEvent) -> None:
        # See repl._deny for why this is Any: the core's action union is not
        # resolvable by mypy, which then treats the None guard as dead.
        action: Any = event.action
        if action is None:
            return
        if action.thought:
            self._show_thought(action.thought)
        params = getattr(action, "parameters", None) or {}
        sym = self._theme.symbols
        self._wrote_work = True
        # Two spans, not a base style: the arguments would inherit the name's bold.
        line = Text()
        line.append(f"{_WORK}{sym.tool_call} {action.name}", style=self._theme.style("tool_name"))
        shown = (
            summary(str(action.name), params) if isinstance(params, Mapping) else as_written(params)
        )
        if shown:
            line.append(f" {shown}", style=self._theme.style("tool_call"))
        # One line whatever the call: cropped at the edge, never wrapped. A Text,
        # so `[/]` is not markup and `:a:` no emoji.
        self._console.print(line, no_wrap=True, overflow="ellipsis")

    def _render_tool_end(self, event: BaseEvent) -> None:
        sym = self._theme.symbols
        action: Any = event.action
        observation = getattr(action, "observation", None) if action else None
        if observation:
            self._work(
                f"{sym.tool_result} {self._truncate(str(observation))}", "tool_result", detail=True
            )
        for part in event.content:
            if part.text:
                self._work(
                    f"{sym.tool_result} {self._truncate(part.text)}", "tool_result", detail=True
                )

    def _render_handoff(self, event: BaseEvent, *, entering: bool) -> None:
        action: Any = event.action
        if action is None:
            return
        sym = self._theme.symbols
        arrow = sym.handoff_in if entering else sym.handoff_out
        label = "handoff" if entering else "return"
        self._work(f"{arrow} {label} {action.name}", "handoff")

    def _render_error(self, event: BaseEvent) -> None:
        """Show why it failed.

        The detail lives in different places depending on who raised: provider and
        planner errors set ``error_message``/``error_code`` on the event, while tool
        errors put it in the action's observation. Reading only one of them is how
        you end up printing "(no detail)" at a user who cannot then act on it.
        """
        action: Any = event.action
        candidates = [
            str(getattr(event, "error_message", "") or ""),
            str(getattr(action, "observation", "") or "") if action is not None else "",
            " ".join(p.text for p in event.content if p.text),
        ]
        detail = next((c for c in candidates if c.strip()), "")

        code = str(getattr(event, "error_code", "") or "")
        label = f"{event.event_type.value}"
        if code:
            label += f" [{code}]"
        self._work(f"{label}: {detail or '(no detail available)'}", "error")

    @staticmethod
    def _truncate(text: str, limit: int = 400) -> str:
        flat = text.strip().replace("\n", " ")
        return flat if len(flat) <= limit else flat[: limit - 1] + "…"
