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

The art animates in place, over a ``Live`` overlay, never row by row: printing 17
rows progressively reads as the screen scrolling, which is the opposite of the
intended effect. The wordmark fallback keeps its short row-by-row reveal — with
four rows it reads as a reveal, not as scrolling.
"""

from __future__ import annotations

import gzip
import os
import sys
import time
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from rich.text import Text

if TYPE_CHECKING:
    from rich.console import Console

    from domyn.agents.interfaces.cli.theme import Theme

_ASSETS = Path(__file__).parent / "assets"

# Baked widths, widest first: the picker takes the first that fits. 150 is the
# full lockup but needs a window wider than most people run, so narrower cuts
# exist to keep the art from being something only a maximised terminal ever
# sees. Nothing below 100 — see WIDTHS in scripts/bake_banner_assets.py.
_LOGO_WIDTHS = (150, 120, 100)

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

# How often the skip-key reader wakes to check for a keypress while the reveal
# plays. Each wake costs a tiny syscall, so this is deliberately coarse: 30 Hz
# keeps the perceived delay under ~33 ms without hammering the terminal.
_KEY_POLL_S = 1 / 30

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
    """
    if not (bool(getattr(sys.stdin, "isatty", lambda: False)()) and os.name == "posix"):
        return False

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


def _animate_art(console: Console, frames: tuple[tuple[str, float], ...]) -> None:
    """Play the reveal over a transient overlay, at the frames' own cadence.

    Transient so the frames leave nothing behind: the caller prints the final
    frame afterwards, permanently, and that is what survives in the scrollback.

    Each frame's sleep subtracts the time already spent rendering it. Without
    that the render cost adds to the delay instead of being part of it, and the
    animation runs slower than authored and unevenly from frame to frame.

    A keypress during the reveal skips the rest: the transient overlay
    closes immediately and the caller's static frame follows, so the user lands
    on the finished logo without waiting out the cadence.
    """
    from rich.live import Live

    deadline = time.perf_counter() + _ANIMATION_CEILING_S

    with Live(console=console, refresh_per_second=30, transient=True) as live:
        for frame, seconds in frames:
            started = time.perf_counter()
            if started > deadline:
                break
            live.update(Text.from_ansi(frame))
            frame_end = started + seconds

            # Default argument binds the value now, not the loop variable later.
            def _frame_done(finish: float = frame_end) -> bool:
                return time.perf_counter() >= finish

            if _wait_for_skip(_frame_done):
                break


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
