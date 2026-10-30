#!/usr/bin/env python3
"""Bake the CLI startup banner from the source logo animation.

The CLI ships the banner pre-rendered — see ``domyn/agents/interfaces/cli/banner.py``
for why — which means the assets are build output, not source. This is the build
step. Run it whenever the logo changes:

    uv run --with pillow python scripts/bake_banner_assets.py path/to/logo.gif

It writes, per supported width:

* ``logo_<w>.ans``       the final flush-left frame, printed permanently
* ``logo_<w>.frames.gz`` the reveal, gzipped: a ``\\f``-separated list whose
  first record is the per-frame durations in milliseconds and whose remaining
  records are the frames themselves. The durations travel with the frames so
  the CLI replays the source's own cadence without having to know anything
  about how the animation was built.

Both come out of the same run, so the last keyframe is byte-identical to the
static frame and the transient overlay closes onto exactly what gets printed.

Pillow is the only dependency, and it is deliberately *not* a dependency of the
package: decoding the animation happens here, once, not on every startup.
"""

import argparse
import gzip
import io
from pathlib import Path

from PIL import Image

ASSETS = Path(__file__).resolve().parent.parent / "domyn/agents/interfaces/cli/assets"

# Must match ``_LOGO_WIDTHS`` in banner.py — anything else ships art the CLI
# never looks for.
#
# Widest first, and 100 is the floor. 150 needs a window most people do not run:
# at 120 and 100 the lettering still reads, but by 80 the letterforms collapse
# into texture and the hand-drawn ASCII wordmark is genuinely the better
# fallback. Each extra size costs ~60-80 KB gzipped, which is nothing next to a
# banner nobody sees.
WIDTHS = (150, 120, 100)

# Every frame of the source, no subsampling. Dropping frames means either
# playing the reveal faster than it was designed to run or holding each frame
# longer, and both were tried and both read as a stutter: the mark visibly
# steps as it draws itself instead of flowing. At one width the full sequence
# is ~137 KB gzipped, which is worth it.
N_FRAMES = None

# Below this alpha a pixel counts as transparent — the logo animation fades in,
# so a plain "non-zero" test would drag near-invisible pixels into the crop.
ALPHA_THRESHOLD = 32

# A terminal cell is about twice as tall as it is wide. Half-blocks give two
# vertical samples per cell, so the vertical resolution works out square.
CHAR_ASPECT = 2.0

RESET = "\033[0m"

# Pixels at least this bright in every channel are the white wordmark, and they
# are *not* emitted as white.
#
# The logo is white lettering next to a blue mark, drawn for a dark background.
# Hardcoding white makes the lettering invisible on a light-themed terminal —
# white on white — and no amount of picking a different white fixes both themes
# at once. Detecting the background would: terminals answer an OSC 11 query with
# their background colour. But that needs a reply-parsing round trip on stdin
# before the first frame, it is unanswered on plenty of terminals and over ssh,
# and it fails by hanging or by leaking the reply into the prompt.
#
# So these cells are emitted with SGR 39 — "default foreground" — and the
# terminal supplies the colour it already uses for text. Light on dark themes,
# dark on light ones, correct on both without asking anyone anything. The blue
# mark keeps its exact colours: it reads on either background.
WHITE_THRESHOLD = 232


def frame_duration_ms(path: Path) -> int:
    """The source's own frame delay. Replaying at any other rate is a redesign
    of the animation, not a rendering choice."""
    with Image.open(path) as gif:
        return int(gif.info.get("duration", 40) or 40)


def load_frames(path: Path) -> list[tuple[Image.Image, int]]:
    """Every GIF frame as composed RGBA, with its count of visible pixels."""
    frames: list[tuple[Image.Image, int]] = []
    gif = Image.open(path)
    try:
        while True:
            rgba = gif.convert("RGBA")
            # getchannel().histogram() runs in C. Counting alpha bytes in Python
            # instead costs ~15 ms per 1920x1080 frame — seconds over a whole GIF.
            visible = sum(rgba.getchannel("A").histogram()[ALPHA_THRESHOLD:])
            frames.append((rgba, visible))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    return frames


def trim_to_peak(frames: list[tuple[Image.Image, int]]) -> list[Image.Image]:
    """Cut at the fullest frame, dropping any fade-out used to loop the GIF."""
    peak = max(range(len(frames)), key=lambda i: frames[i][1])
    return [img for img, _ in frames[: peak + 1]]


def content_bbox(img: Image.Image) -> tuple[int, int, int, int] | None:
    alpha = img.getchannel("A")
    box = alpha.point(lambda a: 255 if a >= ALPHA_THRESHOLD else 0).getbbox()
    return box  # type: ignore[no-any-return]


def crop_box(frames: list[Image.Image]) -> tuple[int, int, int, int] | None:
    """One crop covering every frame's content, with breathing room.

    Union rather than per-frame, or the framing would shift between frames. The
    source is mostly transparent padding, and cropping it is what makes the
    lockup sharp: otherwise most of the available character cells are spent on
    empty space.
    """
    boxes = [b for b in (content_bbox(f) for f in frames) if b]
    if not boxes:
        return None
    left = min(b[0] for b in boxes)
    top = min(b[1] for b in boxes)
    right = max(b[2] for b in boxes)
    bottom = max(b[3] for b in boxes)

    img_w, img_h = frames[0].size
    w, h = right - left, bottom - top
    pad_x, pad_y = int(w * 0.04), int(h * 0.04)
    # Extra room on the left so the closing slide has somewhere to travel.
    return (
        max(0, left - pad_x - int(w * 0.15)),
        max(0, top - pad_y),
        min(img_w, right + pad_x),
        min(img_h, bottom + pad_y),
    )


# Frames of the closing slide. Unlike the reveal these are synthesised here, so
# both the count and the pacing are ours to pick: brisker than the reveal, so
# the lockup snaps into place rather than drifting.
SLIDE_STEPS = 10
SLIDE_FRAME_MS = 20


def slide_left(img: Image.Image, steps: int) -> list[Image.Image]:
    """Translate the lockup to the left margin, easing out."""
    box = content_bbox(img)
    if not box or box[0] <= 0:
        return [img]
    out = []
    for i in range(steps + 1):
        t = i / steps
        shift = round(box[0] * (1 - (1 - t) ** 2))
        canvas = Image.new("RGBA", img.size, (0, 0, 0, 0))
        canvas.paste(img, (-shift, 0), img)
        out.append(canvas)
    return out


Cell = tuple[int, int, int] | None
"""A cell half's colour, or ``None`` for the terminal's own foreground."""


def _ink(r: int, g: int, b: int) -> Cell:
    return None if min(r, g, b) >= WHITE_THRESHOLD else (r, g, b)


def _fg(colour: Cell) -> str:
    if colour is None:
        return "\033[39m"
    return f"\033[38;2;{colour[0]};{colour[1]};{colour[2]}m"


def _bg(colour: Cell) -> str:
    if colour is None:
        return "\033[49m"
    return f"\033[48;2;{colour[0]};{colour[1]};{colour[2]}m"


def to_half_blocks(img: Image.Image, width: int) -> str:
    """Render to truecolor half-blocks: one character per two vertical pixels.

    Colour codes are only emitted when the colour actually changes, which is
    what keeps a 160-column frame at ~21 KB instead of several times that.

    Cells whose colour is the terminal's default are the reason the glyph is
    chosen per cell rather than fixed: the default can only be applied to the
    *foreground*, so a cell whose lower half is default becomes ``▄`` with the
    upper half as its background, instead of ``▀`` with the halves swapped.
    """
    w, h = img.size
    rows = max(1, round(width * (h / w) / CHAR_ASPECT))
    px = img.resize((width, rows * 2), Image.LANCZOS).load()
    assert px is not None

    lines = []
    for row in range(rows):
        parts: list[str] = []
        # Every line is terminated with RESET, so a row starts with both channels
        # at the terminal default — which is exactly what ``None`` means here.
        last_fg: Cell = None
        last_bg: Cell = None
        for x in range(width):
            tr, tg, tb, ta = px[x, row * 2]
            br, bg, bb, ba = px[x, row * 2 + 1]
            top, bottom = ta >= ALPHA_THRESHOLD, ba >= ALPHA_THRESHOLD

            if not top and not bottom:
                parts.append(RESET + " ")
                last_fg = last_bg = None
            elif top and bottom:
                upper, lower = _ink(tr, tg, tb), _ink(br, bg, bb)
                if upper is None and lower is None:
                    # Both halves are the terminal's own colour, so one solid
                    # glyph says it; whatever background is still set is covered.
                    seq = "" if last_fg is None else _fg(None)
                    last_fg = None
                    parts.append(seq + "█")
                    continue
                if upper is None or lower is None:
                    # The default belongs in the foreground, so the glyph follows
                    # which half is default rather than being fixed as ``▀``.
                    glyph = "▀" if upper is None else "▄"
                    ink, paint = None, (lower if upper is None else upper)
                else:
                    glyph, ink, paint = "▀", upper, lower
                seq = ""
                if ink != last_fg:
                    seq += _fg(ink)
                    last_fg = ink
                if paint != last_bg:
                    seq += _bg(paint)
                    last_bg = paint
                parts.append(seq + glyph)
            else:
                # Only one half is visible, so it is drawn as a foreground glyph
                # over a reset background rather than as a two-colour cell.
                ink = _ink(*(px[x, row * 2][:3] if top else px[x, row * 2 + 1][:3]))
                glyph = "▀" if top else "▄"
                # RESET already restores the default foreground, so a default
                # cell needs nothing after it.
                seq = RESET + ("" if ink is None else _fg(ink))
                parts.append(seq + glyph)
                last_fg, last_bg = ink, None
        lines.append("".join(parts) + RESET)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gif", type=Path, help="source logo animation")
    parser.add_argument("--out", type=Path, default=ASSETS)
    args = parser.parse_args()

    source_ms = frame_duration_ms(args.gif)
    frames = trim_to_peak(load_frames(args.gif))
    box = crop_box(frames)
    if box is not None:
        frames = [f.crop(box) for f in frames]

    # The two beats of the reveal: the mark draws itself, then the finished
    # lockup slides to the margin.
    full = frames + slide_left(frames[-1], steps=SLIDE_STEPS)[1:]
    if N_FRAMES is None:
        keyframes = full
    else:
        keep = sorted({round(i * (len(full) - 1) / (N_FRAMES - 1)) for i in range(N_FRAMES)})
        keyframes = [full[i] for i in keep]
    print(f"{len(frames)} reveal + {len(full) - len(frames)} slide -> {len(keyframes)} keyframes")

    args.out.mkdir(parents=True, exist_ok=True)
    for width in WIDTHS:
        rendered = [to_half_blocks(img, width) for img in keyframes]
        (args.out / f"logo_{width}.ans").write_text(rendered[-1] + "\n", encoding="utf-8")
        durations = [source_ms] * len(frames) + [SLIDE_FRAME_MS] * (len(keyframes) - len(frames))
        blob = "\f".join([",".join(str(d) for d in durations), *rendered]).encode("utf-8")
        # mtime=0: without it gzip stamps the current time into the header and
        # every re-bake shows up as a diff even when nothing changed.
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=9, mtime=0) as gz:
            gz.write(blob)
        packed = buffer.getvalue()
        (args.out / f"logo_{width}.frames.gz").write_bytes(packed)
        print(
            f"  {width:>3} cols x {rendered[-1].count(chr(10)) + 1} rows  "
            f"{sum(durations) / 1000:.2f}s  "
            f"frames {len(blob) // 1024} KB -> {len(packed) // 1024} KB"
        )


if __name__ == "__main__":
    main()
