"""A one-line activity indicator for the gaps between output.

A turn is mostly waiting, and a slow one looks exactly like a hung one — whether
to wait or hit Ctrl-C depends on telling them apart. The moving ball says the
process is alive; the clock says how long the current step has taken, and
what kind of step it is.

It is a **single line, redrawn in place, and erased before anything else
prints**. That rule is the whole design: interleaved with real output it leaves
debris in the transcript, which is worse than no spinner at all. The renderer
pauses it before writing and resumes after, and pausing erases.

Logging is the one writer that does not go through the renderer, so a filter on
the root handlers erases the line before a record is emitted — without it a
warning lands halfway through the ball.

Written straight to the console's stream with carriage returns rather than
through ``rich.Live``, which owns a region and repaints it, fighting a renderer
that prints freely above.

The clock is shown **only while a step is in flight**, labelled with which:
``thinking`` from ``LLM_START`` to ``LLM_END``, ``executing`` from
``TOOL_START`` to ``TOOL_END``. Each step restarts it, and between steps there
is no clock at all — only the ball. The renderer reports the edges
(:meth:`Spinner.thinking`, :meth:`Spinner.executing`, :meth:`Spinner.idle`). A
clock spanning both kinds would answer neither "is the model slow" nor "is the
command stuck". The turn's total is printed once, when it ends, by the session.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console

    from domyn.agents.interfaces.cli.theme import Theme

# Wide enough to read as movement, narrow enough to sit beside the timer on any
# terminal. Shared with `status()`, so the shape does not change with the drawer.
_TRACK = 14

# ~12 fps: continuous enough, without spending an SSH session's bandwidth on it.
_FRAME_S = 0.08

_THINKING = "thinking"
_EXECUTING = "executing"
_LABEL_WIDTH = max(len(_THINKING), len(_EXECUTING))
# Steps shorter than this never show a label: a quick tool call would flash one.
_SHOW_AFTER_S = 0.1

_ERASE_LINE = "\r\033[2K"
_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"


class Spinner:
    """Draws a moving ball, and the model call's clock, until something else needs the line."""

    def __init__(self, console: Console, theme: Theme) -> None:
        self._console = console
        self._theme = theme
        self._task: asyncio.Task[None] | None = None
        self._drawing = False
        # The step in flight and when it began; None between steps.
        self._phase: tuple[str, float] | None = None
        # The answer is arriving: nothing to wait for, so nothing is drawn.
        self._quiet = False
        self._frame = 0
        self._dirty = False

        caps = theme.capabilities
        # Without colour or animation this is punctuation twitching in a log file.
        self._enabled = caps.animation and caps.color
        self._ball = "●" if caps.unicode else "o"
        self._rail = "·" if caps.unicode else "."

    @property
    def enabled(self) -> bool:
        return self._enabled

    def thinking(self) -> None:
        """A model call began: show the clock, labelled and timed from now.

        None of these are gated on :attr:`enabled` or on the spinner running: the
        editor draws :meth:`status` while the spinner itself stands down, and
        needs the clock.
        """
        self._phase = (_THINKING, time.monotonic())
        self._quiet = False

    def executing(self) -> None:
        """A tool began running: show the clock, labelled and timed from now."""
        self._phase = (_EXECUTING, time.monotonic())
        self._quiet = False

    def idle(self) -> None:
        """The step is over: no clock until the next one starts."""
        self._phase = None
        self._quiet = False

    def streaming(self) -> None:
        """The model's answer has started arriving: neither ball nor clock.

        The text itself is the sign of life, and a "thinking" clock under an
        answer being written says the opposite of what is happening. Lasts
        until the next step boundary (LLM_END, TOOL_START, ...).
        """
        self._phase = None
        self._quiet = True

    def _clock(self) -> str:
        if self._phase is None:
            return ""
        label, started_at = self._phase
        elapsed = time.monotonic() - started_at
        if elapsed < _SHOW_AFTER_S:
            # A step this short would only flash its label past the eye.
            return ""
        # Padded to one width, so the line does not twitch between the two.
        return f"{label:<{_LABEL_WIDTH}} {elapsed:5.1f}s"

    def status(self) -> tuple[str, str]:
        """The same movement and clock, as text for something else to draw.

        Two writers cannot both own the bottom line, and the one with an editor on
        it wins — so while a read shares the screen with the turn, the editor draws
        this on the row above its prompt and the spinner stands down.

        **Returned in two pieces** because the two halves are not styled alike:
        :meth:`_draw` gives the track the brand colour and dims the clock. One
        string would either lose that or make the caller guess where to cut, and
        the line would end up looking different depending on who drew it. They
        are named, not ordered — the caller lays them out as ``_draw`` does.

        Plain text, no escapes: whoever draws it owns the styling. Same track as
        the spinner's own line — it is the same indicator, drawn elsewhere. The
        clock is empty between steps.
        """
        if self._quiet:
            return "", ""
        clock = self._clock()
        if not self._enabled:
            # No animation: the clock alone still answers the question.
            return clock, ""
        position = int(time.monotonic() / _FRAME_S) % _TRACK
        track = self._rail * position + self._ball + self._rail * (_TRACK - position - 1)
        return clock, track

    def attach_to_logging(self) -> None:
        """Erase the line before any log record is written.

        A filter rather than a wrapped handler: the only hook that runs before
        every emit without replacing what the host installed. Always returns
        ``True`` — it filters nothing, it just needs telling that a write is due.
        """
        if not self._enabled:
            return
        pause_filter = _PauseBeforeLogging(self)
        for handler in logging.getLogger().handlers:
            if not any(isinstance(f, _PauseBeforeLogging) for f in handler.filters):
                handler.addFilter(pause_filter)

    async def start(self) -> None:
        """Begin drawing. Idempotent."""
        if not self._enabled or self._task is not None:
            return
        self._drawing = True
        self._task = asyncio.create_task(self._loop())

    def pause(self) -> None:
        """Erase the line and stop drawing. Called before anything else prints."""
        if not self._enabled:
            return
        self._drawing = False
        self._erase()

    def resume(self) -> None:
        """Draw again. The clock is left alone: it times the model call, not the gap."""
        if not self._enabled or self._task is None:
            return
        self._frame = 0
        self._drawing = True

    async def stop(self) -> None:
        """Stop for good and leave the line clean."""
        # A turn interrupted mid-step never sees the step's end.
        self._phase = None
        self._quiet = False
        task, self._task = self._task, None
        self._drawing = False
        if task is not None:
            task.cancel()
            # Awaited, so it cannot repaint after the caller has printed something.
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._erase()

    # --- internals ---------------------------------------------------------

    def _write(self, text: str) -> None:
        file = getattr(self._console, "file", None)
        if file is None:  # pragma: no cover - defensive
            return
        file.write(text)
        file.flush()

    def _erase(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        self._write(_ERASE_LINE + _SHOW_CURSOR)

    async def _loop(self) -> None:
        try:
            while True:
                if self._drawing:
                    self._draw()
                await asyncio.sleep(_FRAME_S)
        except asyncio.CancelledError:
            raise
        except Exception:
            # A decoration must never take a session down, and cannot report a
            # problem without becoming the noise it exists to avoid.
            self._drawing = False

    def _draw(self) -> None:
        if self._quiet:
            return
        clock = self._clock()
        position = self._frame % _TRACK
        self._frame += 1

        track = self._rail * position + self._ball + self._rail * (_TRACK - position - 1)
        brand = self._theme.styles.brand
        dim = "\033[2m"
        reset = "\033[0m"
        # Escapes by hand: rewritten many times a second, and must not go through
        # the renderer that is being asked to stay out of the way.
        colour = f"\033[38;2;{_rgb(brand)}m" if brand.startswith("#") else ""
        self._dirty = True
        # Track first, at the left margin: same column as the transcript, same
        # order as `status()`.
        line = f"{_ERASE_LINE}{_HIDE_CURSOR}{colour}{track}{reset}"
        if clock:
            line += f"  {dim}{clock}{reset}"
        self._write(line)


class _PauseBeforeLogging(logging.Filter):
    """Lets the spinner clear the line before a record is printed."""

    def __init__(self, spinner: Spinner) -> None:
        super().__init__()
        self._spinner = spinner

    def filter(self, record: logging.LogRecord) -> bool:
        self._spinner.pause()
        return True


def _rgb(hex_colour: str) -> str:
    value = hex_colour.lstrip("#")
    return f"{int(value[0:2], 16)};{int(value[2:4], 16)};{int(value[4:6], 16)}"
