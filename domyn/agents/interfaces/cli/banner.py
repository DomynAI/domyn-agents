"""Animated startup banner.

The logo ships as pre-rendered ANSI art in ``assets/logo_<width>.ans`` — truecolor
half-blocks (``▀▄``) of the Domyn lockup — alongside ``logo_<width>.frames.gz``,
the keyframes of the reveal. Rendering these at startup would mean carrying an
image library and decoding a 1.8 MB GIF just to print a banner, so they are baked
offline instead.

The keyframes are gzipped because they are not small: the reveal is 2.3 MB of
escape codes, and that same repetition compresses to 124 KB. Their per-frame
durations travel in the archive alongside them, so the reveal plays at the
source's own cadence rather than at whatever rate happens to look plausible.
The last keyframe is byte-identical to the static frame, so the transient
overlay closes onto exactly what stays in the scrollback — no jump at the end.

Three things gate it, each learned the hard way:

* **Width.** One size is baked, and it is used only when it fits. Art wider than
  the terminal soft-wraps into unreadable noise, and re-rendering it smaller is
  not an option without shipping an image library. Intermediate sizes were tried
  and dropped: each costs as much as the full one and buys a marginally better
  fit, so anything narrower falls back to the wordmark instead.
* **Height.** Equally important, and the reason an earlier version "scrolled":
  the lockup is ~9:1 over 17 rows, and that pushes a short terminal's
  scrollback. Art is only used when it fits with room left for the prompt.
* **Colour and Unicode.** The art is *entirely* colour, so without it the blocks
  carry no shape at all; ``NO_COLOR``, a pipe or ``TERM=dumb`` fall back to the
  wordmark rather than printing meaningless glyphs. Half-blocks also need UTF-8.

The art animates in place, never row by row: 17 rows printed progressively read as
the screen scrolling, the opposite of the intended effect. Each frame is one write
over the previous one — no ``Live``, which blanks a row before redrawing it and left
the logo empty between frames — inside a synchronized-output block, on the frames'
own schedule. The wordmark fallback keeps its short row-by-row reveal: with four
rows it reads as a reveal, not as scrolling.
"""

from __future__ import annotations

import gzip
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from rich.text import Text

if TYPE_CHECKING:
    from rich.console import Console

    from domyn.agents.interfaces.cli.theme import Theme

_ASSETS = Path(__file__).parent / "assets"

# Baked widths, widest first: the picker takes the first that fits, so the size
# follows the terminal. 150 is the full lockup as the animation draws it, for a
# large window. 98 (12 rows) and 80 (9 rows) are for the windows below it: cuts
# with their geometry chosen so every stroke lands on whole pixels and the
# letters redrawn on the pixel grid, which a plain shrink does not give. See
# COMPACT_CUTS and TIDY in scripts/bake_banner_assets.py.
_LOGO_WIDTHS = (150, 98, 80)

# Rows kept free below the banner for the subtitle, the hint line and the first
# exchange. Without this the logo technically "fits" and still pushes everything
# the user came to read off the top of the window.
_RESERVED_ROWS = 8

# Total wall time the reveal may take. Split across rows, so a taller logo does
# not make startup slower.
_REVEAL_BUDGET_S = 0.36

# Ceiling on the reveal, not its intended length: the frames carry their own
# durations and normally sum to ~4.2 s. This only exists so a corrupt or
# hand-edited archive cannot hold the session hostage before it starts.
_ANIMATION_CEILING_S = 8.0

# How often the skip-key reader wakes during the reveal. Frames last 20 or 40 ms,
# so a coarser wait holds a short frame for a whole poll and the cadence turns
# uneven; a zero-timeout ``select`` is cheap enough at 200 Hz.
_KEY_POLL_S = 1 / 200

# Synchronized output (DEC 2026): the terminal holds what it is sent until the
# frame is complete, then paints it at once. Terminals that do not know it ignore it.
_SYNC_BEGIN = "\x1b[?2026h"
_SYNC_END = "\x1b[?2026l"
_HIDE_CURSOR = "\x1b[?25l"
_SHOW_CURSOR = "\x1b[?25h"
_ERASE_LINE_TAIL = "\x1b[K"

_WORDMARK = (
    r"  ___   ___  __  __ __   __ _  _ ",
    r" |   \ / _ \|  \/  |\ \ / /| \| |",
    r" | |) | (_) | |\/| | \ V / | .` |",
    r" |___/ \___/|_|  |_|  |_|  |_|\_|",
)

_WORDMARK_ASCII = (
    r" ____   ___  __  __ _   _ _   _ ",
    r"|  _ \ / _ \|  \/  | \ / | \ | |",
    r"| |_| | |_| | |\/| |  V  |  \| |",
    r"|____/ \___/|_|  |_|_| |_|_|\_|",
)

_COMPACT = "DOMYN AGENTS"


@lru_cache(maxsize=len(_LOGO_WIDTHS))
def _load_logo(width: int) -> tuple[str, ...] | None:
    """Read one pre-rendered logo, or ``None`` when it is missing.

    Cached because a session may redraw the banner, and missing art must never be
    fatal — a banner is decoration, and an install with stripped data files should
    still start.
    """
    path = _ASSETS / f"logo_{width}.ans"
    try:
        return tuple(path.read_text(encoding="utf-8").rstrip("\n").split("\n"))
    except OSError:
        return None


@lru_cache(maxsize=len(_LOGO_WIDTHS))
def _load_frames(width: int) -> tuple[tuple[str, float], ...]:
    """Read the reveal as ``(frame, seconds)`` pairs, or ``()`` when unavailable.

    The archive leads with a comma-separated list of per-frame durations in
    milliseconds; the frames follow. Never raises: a missing, truncated or
    malformed archive costs the animation, not the banner, and certainly not the
    session.
    """
    path = _ASSETS / f"logo_{width}.frames.gz"
    try:
        blob = gzip.decompress(path.read_bytes()).decode("utf-8")
    except (OSError, EOFError, UnicodeDecodeError):
        return ()

    header, _, rest = blob.partition("\f")
    frames = rest.split("\f") if rest else []
    try:
        durations = [int(value) for value in header.split(",")]
    except ValueError:
        return ()
    if not frames or len(durations) != len(frames):
        return ()
    return tuple((frame, ms / 1000) for frame, ms in zip(frames, durations, strict=True))


def _skip_key_pressed() -> bool:
    """Whether the user pressed a key that should skip the reveal.

    Only meaningful on a real, interactive terminal: a pipe or a file has nobody
    at it, so this always returns ``False`` there and the banner plays out on its
    own. Pure ``select`` on a pipe would *also* report readable (EOF), which is
    exactly why the TTY check comes first.

    On Windows ``select`` takes sockets only; the console has its own pair, which
    asks whether a key is waiting and reads it, so it is not left for the REPL.
    """
    if not bool(getattr(sys.stdin, "isatty", lambda: False)()):
        return False
    if sys.platform == "win32":
        import msvcrt

        if not msvcrt.kbhit():
            return False
        msvcrt.getwch()
        return True
    else:
        import select

        readable, _, _ = select.select([sys.stdin], [], [], 0)
        if not readable:
            return False

        # One byte is enough to know a key was pressed; the rest stays buffered for
        # the REPL that starts right after the banner. ``.buffer`` when present (a
        # text stream), the stream itself otherwise (a binary stream).
        try:
            ch = getattr(sys.stdin, "buffer", sys.stdin).read(1)
        except (OSError, ValueError, AttributeError):
            return False
        return bool(ch)


def _wait_for_skip(stop: Callable[[], bool]) -> bool:
    """Wait until the user presses a skip key or ``stop()`` turns true.

    Polls in short slices so the reveal can keep playing underneath: a blocking
    read would freeze the animation on the last frame until the key arrives,
    which reads as a hang. The keypress itself is consumed here; nothing is left
    for the REPL to swallow.

    Returns ``True`` when the user skipped, ``False`` when ``stop()`` won first.
    """
    while not stop():
        if _skip_key_pressed():
            return True
        time.sleep(_KEY_POLL_S)
    return False


def _mute_keys() -> Callable[[], None] | None:
    """Turn off echo and line buffering on stdin; return the undo, or ``None``.

    ``None`` when there is nothing to do or nothing could be done: Windows, a pipe.
    """
    if sys.platform == "win32":
        return None
    try:
        import termios

        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
        quiet = termios.tcgetattr(fd)
        quiet[3] &= ~(termios.ECHO | termios.ICANON)
        termios.tcsetattr(fd, termios.TCSADRAIN, quiet)
    except (ImportError, OSError, ValueError, AttributeError):
        # No ``termios`` module, or no tty behind stdin: nothing to mute. Its own
        # ``error`` is ``OSError``, and naming it here would break the case where
        # the import itself failed.
        return None

    def restore() -> None:
        with suppress(OSError, termios.error):
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)

    return restore


@contextmanager
def _keys_not_echoed() -> Iterator[None]:
    """Keep a keypress from being echoed or line-buffered while the reveal plays.

    In the terminal's default mode a key is held until Enter, so skipping would
    need a newline, and the typed character is drawn by the terminal itself, in
    the middle of the animation. The Windows console reads keys unechoed already.
    Always restored, and ``ISIG`` is left alone so Ctrl-C still works.
    """
    restore = _mute_keys()
    try:
        yield
    finally:
        if restore is not None:
            restore()


def _paint_rows(console: Console, frame: str) -> list[str]:
    """The frame as the terminal will receive it, one string per row.

    Rendered through the console so rich downgrades the 24-bit colours where the
    terminal lacks them. Each row ends by clearing what is left of the previous
    frame to its right, so a row is overwritten in place and never blanked first.
    """
    with console.capture() as captured:
        console.print(Text.from_ansi(frame), highlight=False, no_wrap=True, crop=False, end="")
    return [row + _ERASE_LINE_TAIL for row in captured.get().split("\n")]


def _animate_art(console: Console, frames: tuple[tuple[str, float], ...]) -> None:
    """Play the reveal in place, at the frames' own cadence, then clear it.

    Nothing is left behind: the caller prints the final frame afterwards,
    permanently, and that is what survives in the scrollback.

    Frames are written straight to the terminal, not through ``Live``: ``Live``
    erases every row (``ESC[2K``) and rewrites it on a timer of its own, so a
    terminal painting between the two shows the logo blank, and its 30 Hz is out of
    step with frames of 20 and 40 ms. Here a frame is one write over the previous
    one, inside a synchronized-output block, made when it is due.

    The schedule is absolute — a frame is due at the sum of the durations before it
    — so render time comes out of the wait and a slow frame is caught up, not
    shifted.

    A keypress skips the rest: the overlay is cleared at once and the caller's static
    frame follows, so the user lands on the finished logo without waiting out the
    cadence.
    """
    out = console.file
    rows = 0
    started = time.perf_counter()
    deadline = started + _ANIMATION_CEILING_S
    due = started

    try:
        with _keys_not_echoed():
            out.write(_HIDE_CURSOR)
            for frame, seconds in frames:
                now = time.perf_counter()
                if now > deadline:
                    break
                painted = _paint_rows(console, frame)
                if rows:
                    # Same height every frame; guard anyway so a hand-edited
                    # archive cannot move the cursor off the art.
                    painted = (painted + [_ERASE_LINE_TAIL] * rows)[:rows]
                    move_up = f"\x1b[{rows}A\r"
                else:
                    rows = len(painted)
                    move_up = ""
                out.write(f"{_SYNC_BEGIN}{move_up}{chr(10).join(painted)}\n{_SYNC_END}")
                out.flush()

                due += seconds

                # Default argument binds the value now, not the loop variable later.
                def _frame_done(finish: float = due) -> bool:
                    return time.perf_counter() >= finish

                if _wait_for_skip(_frame_done):
                    break
    finally:
        clear = f"\x1b[{rows}A\r\x1b[J" if rows else ""
        out.write(f"{_SYNC_BEGIN}{clear}{_SYNC_END}{_SHOW_CURSOR}")
        out.flush()


def _pick_art_width(console: Console, theme: Theme) -> int | None:
    """The widest baked logo this terminal can show, or ``None`` for no art."""
    caps = theme.capabilities
    if not (caps.color and caps.unicode):
        return None
    max_rows = max(0, console.size.height - _RESERVED_ROWS)
    for width in _LOGO_WIDTHS:
        if width > console.width:
            continue
        rows = _load_logo(width)
        if rows is not None and len(rows) <= max_rows:
            return width
    return None


def _pick_rows(console: Console, theme: Theme) -> tuple[str, ...]:
    """Choose what to draw: the colour logo, the wordmark, or a compact line."""
    width = _pick_art_width(console, theme)
    if width is not None:
        art = _load_logo(width)
        if art is not None:
            return art

    if console.width < 40:
        return (_COMPACT,)
    return _WORDMARK if theme.capabilities.unicode else _WORDMARK_ASCII


def render_banner(
    console: Console,
    theme: Theme,
    *,
    subtitle: str | Text | None = None,
    animate: bool = True,
) -> None:
    """Print the logo, animated when the terminal allows it.

    Args:
        console: rich console to draw on.
        theme: resolved theme, providing capabilities and styles.
        subtitle: optional line under the logo (version, agent, workspace). A string
            is drawn dim; a ``Text`` keeps its own styles, so a part of it can stand out.
        animate: caller-side opt-out (e.g. ``--no-banner``). Terminal capabilities
            can still veto the animation.
    """
    art_width = _pick_art_width(console, theme)
    rows = _pick_rows(console, theme)
    is_art = art_width is not None
    should_animate = animate and theme.capabilities.animation

    accent = theme.style("accent_bold")
    dim = theme.style("meta")

    try:
        if is_art and should_animate and art_width is not None:
            # Plays and erases itself; the static frame below is what remains.
            frames = _load_frames(art_width)
            if frames:
                _animate_art(console, frames)

        # The wordmark is revealed row by row; the art is not — see module docs.
        delay = (_REVEAL_BUDGET_S / len(rows)) if should_animate and not is_art else 0.0

        for row in rows:
            if is_art:
                # Parsed into styled spans rather than printed as a plain string.
                # rich measures a raw string by counting characters, and each cell
                # of the art carries ~20 characters of escape codes — so it reads a
                # 160-column row as thousands of columns, wraps it, and the logo
                # collapses into noise. from_ansi also lets rich downgrade the
                # 24-bit colours sensibly on terminals that lack truecolor.
                console.print(Text.from_ansi(row), highlight=False, no_wrap=True, crop=False)
            else:
                console.print(row, style=accent, highlight=False)
            if delay:
                if _skip_key_pressed():
                    delay = 0.0
                else:
                    time.sleep(delay)
        if isinstance(subtitle, Text):
            console.print(subtitle, highlight=False)
        elif subtitle:
            console.print(subtitle, style=dim, highlight=False)
        console.print()
    except KeyboardInterrupt:
        # Leave the shell prompt on a clean line rather than glued to a
        # half-drawn banner, then let the caller decide whether to exit.
        console.print()
        raise
