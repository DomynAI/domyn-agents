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
import math
from pathlib import Path

from PIL import Image, ImageChops

ASSETS = Path(__file__).resolve().parent.parent / "domyn/agents/interfaces/cli/assets"

# Must match ``_LOGO_WIDTHS`` in banner.py — anything else ships art the CLI
# never looks for.
#
# Widest first. 150 is the lockup as the animation draws it, for a large window;
# the two cuts below it are baked separately, from a chosen geometry rather than a
# width (COMPACT_CUTS). Each extra size costs ~50-120 KB gzipped, which is nothing
# next to a banner nobody sees.
WIDTHS = (150,)

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
    if px is None:
        raise RuntimeError("Failed to load image pixels")

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


# The compact cut. Shrinking the frames to a width gives letters whose x-height
# and baseline fall between pixel rows: an "o" a pixel below the other letters.
# So its geometry is chosen, not derived from a width:
#
# * the scale makes the cap height of the cut (its COMPACT_CUTS value) pixels (CAP_SRC is the source's,
#   measured on the GIF this was made for; x-height is 0.768 of it, which at 10
#   is within a quarter of a pixel of a whole number), the lettering keeps the
#   font's own proportions,
# * the baseline (BASELINE_SRC) is moved onto a cell boundary,
# * the area filter (BOX) averages coverage honestly, and the alpha cut is low
#   enough to keep the weight of the strokes but above the o's overshoot of the
#   baseline, which would otherwise be drawn as a row of its own,
# * the canvas starts at the left edge of the finished lockup, so the glyph is
#   whole when the slide ends, and is as many columns wide as the cut is named,
#   which must hold the lockup where the slide starts (checked).
# Canvas width in columns -> cap height in pixels. 98 is the finished one (the
# stems come out three whole pixels, 12 rows); 80 is for windows narrower than
# that and is rougher: the "o" and "m" touch at the top.
COMPACT_CUTS = {98: 13.0, 80: 10.0}
COMPACT_ALPHA = 70
CAP_SRC = 198.0
BASELINE_SRC = 635.0


# Letters of the compact cuts, redrawn on the pixel grid once the word is whole.
# A filter cannot give a stroke that is 2.4 source pixels wide the same width in
# every letter: the "o" came out lopsided with a hole missing in its bottom, the
# legs of the "m" and "n" differed, the counters had square inner corners. These
# keep the font's own structure (the notch above a counter, the arch that ends one
# column short of its stem, the D's counter measured on the source) with the
# strokes made even. The "y" is left as rendered.
#
# Per cut: ``cap`` is the cap height in pixels; ``letters`` are
# (column, row below the cap line, rows) with the column counted from the first
# column of the D's stem; ``stems`` are the runs of ink across the x-height, from
# that column, once the word is whole (the letters spread apart while it is
# revealed, so nothing is redrawn before that); ``clear`` are the column ranges the
# redrawn letters replace, which leave the y alone; ``stem_row`` is how many rows
# above the baseline that run is read.
TIDY: dict[int, dict] = {
    98: {
        "cap": 13,
        "letters": (
            # D: closed by a two-row bar, 3 wide on its first counter row, then 4
            # until it narrows the same way at the bottom; the stem side is straight.
            # Stem and right wall are 3 wide, like the legs of the m and n. (The
            # font's D is a column wider, with a 5-wide counter.)
            (
                0,
                0,
                (
                    "#######...",
                    "########..",
                    "###...###.",
                    "###....###",
                    "###....###",
                    "###....###",
                    "###....###",
                    "###....###",
                    "###....###",
                    "###....###",
                    "###...###.",
                    "########..",
                    "#######...",
                ),
            ),
            # o: symmetric in both directions; 10 wide, the font's (counter 4). Its
            # bars are two rows at the centre: the font's are 2.4 pixels, like the
            # stems, and a solid third row made it heavy. The outline runs 4, 8, 10 wide, as
            # a circle of that diameter does, so it reads round.
            (
                11,
                3,
                (
                    "...####...",
                    ".########.",
                    "####..####",
                    "###....###",
                    "###....###",
                    "###....###",
                    "###....###",
                    "####..####",
                    ".########.",
                    "...####...",
                ),
            ),
            # m: three equal stems, two equal counters, rounded like the n's. 15
            # wide, so that it keeps the column the font leaves between it and the y.
            (
                22,
                3,
                (
                    "###.#####.####.",
                    "###############",
                    "###############",
                    "####.#####.####",
                    "###...###...###",
                    "###...###...###",
                    "###...###...###",
                    "###...###...###",
                    "###...###...###",
                    "###...###...###",
                ),
            ),
            # n
            (
                50,
                3,
                (
                    "###.#####.",
                    "##########",
                    "##########",
                    "####..####",
                    "###....###",
                    "###....###",
                    "###....###",
                    "###....###",
                    "###....###",
                    "###....###",
                ),
            ),
        ),
        "stems": (
            (0, 2),
            (8, 10),
            (12, 14),
            (20, 22),
            (24, 26),
            (30, 32),
            (37, 39),
            (42, 44),
            (46, 48),
            (52, 54),
            (59, 61),
        ),
        "clear": ((-1, 39), (52, 61)),
        "stem_row": 6,
        # The y's tail, from the baseline down: one stroke of constant width that
        # steps left once per cell row (two pixel rows), so every cell row is flat
        # and the line stays straight. (first column, last column per row; the
        # columns cleared before drawing.)
        "tail": ((42, 44), (42, 44), (41, 43), (41, 43)),
        "tail_clear": (36, 48),
        # The y is the filter's, not drawn here: its pixels move left with the o, m
        # and n, which sit two columns closer than the filter put them.
        "shift": (40, 51, -2),
    },
    80: {
        "cap": 10,
        "letters": (
            # D: the outer curve is the font's, scaled: of its width the edge has
            # 74%, 91%, 96% at the first rows, then all of it.
            (
                0,
                0,
                (
                    "#####..",
                    "######.",
                    "##..###",
                    "##...##",
                    "##...##",
                    "##...##",
                    "##...##",
                    "##..###",
                    "######.",
                    "#####..",
                ),
            ),
            (
                8,
                2,
                (
                    "..####..",
                    ".######.",
                    "###..###",
                    "##....##",
                    "##....##",
                    "###..###",
                    ".######.",
                    "..####..",
                ),
            ),
            (
                17,
                2,
                (
                    "##.####.###.",
                    "############",
                    "###.####.###",
                    "##...##...##",
                    "##...##...##",
                    "##...##...##",
                    "##...##...##",
                    "##...##...##",
                ),
            ),
            (
                38,
                2,
                (
                    "##.###.",
                    "#######",
                    "###.###",
                    "##...##",
                    "##...##",
                    "##...##",
                    "##...##",
                    "##...##",
                ),
            ),
        ),
        "stems": (
            (0, 2),
            (7, 8),
            (10, 12),
            (14, 16),
            (18, 21),
            (23, 26),
            (28, 30),
            (32, 34),
            (36, 38),
            (40, 42),
            (45, 47),
        ),
        "clear": ((-1, 30), (40, 47)),
        "stem_row": 6,
        "tail": ((32, 34), (32, 34), (31, 33), (31, 33)),
        "tail_clear": (28, 36),
        # The y is the filter's, not drawn here: its pixels move two columns left
        # with the letters after the D, which is one column narrower.
        "shift": (31, 39, -2),
    },
}


def tidy_word(px: Image.Image, baseline_row: int, cut: int, tolerance: int = 1) -> bool:
    """Redraw the letters of ``TIDY[cut]`` on a rendered compact frame.

    Only when the whole word is there; while it is being revealed the frame is
    left as the filter made it. ``baseline_row`` is the pixel row just below the
    baseline. Returns whether the letters were redrawn.

    ``tolerance`` is how far, in columns, a stem may be from where the cut expects
    it. The caller widens it once a frame has matched: the letters of a reveal
    overshoot their place by a column or two just before they settle, and a frame
    that fell back to the filter's own letters there showed them jumping left.
    """
    cfg = TIDY[cut]
    pix = px.load()
    assert pix is not None

    def inked(x: int, y: int) -> bool:
        r, g, b, a = pix[x, y]
        return a >= COMPACT_ALPHA and min(r, g, b) >= 200

    cap_top = baseline_row - cfg["cap"]
    first = None
    for x in range(px.width):
        if any(inked(x, y) for y in range(max(0, cap_top), baseline_row)):
            first = x
            break
    if first is None:
        return False
    row = baseline_row - cfg["stem_row"]
    found: list[tuple[int, int]] = []
    start = None
    for x in range(px.width + 1):
        if x < px.width and inked(x, row):
            if start is None:
                start = x
        elif start is not None:
            found.append((start - first, x - 1 - first))
            start = None
    stems = cfg["stems"]
    if len(found) != len(stems) or any(
        abs(a - ra) > tolerance or abs(b - rb) > tolerance
        for (a, b), (ra, rb) in zip(found, stems, strict=True)
    ):
        return False
    moved: list[tuple[int, int, tuple[int, int, int, int]]] = []
    if "shift" in cfg:
        lo, hi, dx = cfg["shift"]
        for y in range(max(0, cap_top - 1), px.height):
            for x in range(max(0, first + lo), min(px.width, first + hi + 1)):
                if pix[x, y][3] > 0 and min(pix[x, y][:3]) >= 200:
                    moved.append((x + dx, y, pix[x, y]))
                    pix[x, y] = (0, 0, 0, 0)
    for lo, hi in cfg["clear"]:
        for y in range(max(0, cap_top - 1), baseline_row + 1):
            for x in range(max(0, first + lo), min(px.width, first + hi + 1)):
                r, g, b, _a = pix[x, y]
                if min(r, g, b) >= 200:
                    pix[x, y] = (0, 0, 0, 0)
    for x, y, value in moved:
        if 0 <= x < px.width:
            pix[x, y] = value
    lo_clear, hi_clear = cfg["tail_clear"]
    for dy, (lo, hi) in enumerate(cfg["tail"]):
        y = baseline_row + dy
        if y >= px.height:
            break
        for x in range(max(0, first + lo_clear), min(px.width, first + hi_clear + 1)):
            if min(pix[x, y][:3]) >= 200:
                pix[x, y] = (0, 0, 0, 0)
        for x in range(first + lo, first + hi + 1):
            if 0 <= x < px.width:
                pix[x, y] = (255, 255, 255, 255)
    for col, row_offset, rows in cfg["letters"]:
        top = cap_top + row_offset
        for dy, line in enumerate(rows):
            for dx, c in enumerate(line):
                if c == "#":
                    x, y = first + col + dx, top + dy
                    if 0 <= x < px.width and 0 <= y < px.height:
                        pix[x, y] = (255, 255, 255, 255)
    return True


# Cuts whose glyph is snapped to a circle on the outside (see round_glyph).
ROUND_GLYPH = (80,)
# A glyph counts as fully drawn from this size on (source pixels): while it draws
# itself the circle would complete it.
GLYPH_DRAWN = 245


def glyph_box(img: Image.Image) -> tuple[int, int, int, int] | None:
    """Bounding box of the mark: opaque pixels that are not the white word."""
    r, g, b, a = img.split()
    dark = ImageChops.darker(ImageChops.darker(r, g), b).point(lambda v: 255 if v < 200 else 0)
    solid = a.point(lambda v: 255 if v >= 128 else 0)
    return ImageChops.multiply(dark, solid).getbbox()


def round_glyph(px: Image.Image, cx: float, cy: float, radius: float) -> None:
    """Make the outside of the mark a circle, on a rendered frame.

    At a few pixels across, the mark's edge is a stairway of whatever the filter
    left on each pixel: gaps in the rim, a stray pixel outside it. The disc the
    mark fills is measured on the source; a pixel with under a third of its area
    inside is cleared, and an empty one on the rim with over half of it inside is
    given the colour of the nearest ink, so the outline is the circle's. ``cx``,
    ``cy`` and ``radius`` are in pixels of ``px``.
    """
    pix = px.load()
    assert pix is not None

    def inside(x: int, y: int) -> float:
        hits = 0
        for i in range(4):
            for j in range(4):
                dx = x + (i + 0.5) / 4 - cx
                dy = y + (j + 0.5) / 4 - cy
                hits += dx * dx + dy * dy <= radius * radius
        return hits / 16

    def is_ink(x: int, y: int) -> bool:
        r, g, b, a = pix[x, y]
        return a >= COMPACT_ALPHA and min(r, g, b) < 200

    reach = int(radius) + 3
    x0, y0 = int(cx) - reach, int(cy) - reach
    for y in range(max(0, y0), min(px.height, y0 + 2 * reach + 1)):
        for x in range(max(0, x0), min(px.width, x0 + 2 * reach + 1)):
            area = inside(x, y)
            r, g, b, a = pix[x, y]
            white = min(r, g, b) >= 200 and a > 0
            if white:
                continue
            if area < 1 / 3 and a > 0:
                pix[x, y] = (0, 0, 0, 0)
            elif area > 0.5 and not is_ink(x, y):
                dist = math.hypot(x + 0.5 - cx, y + 0.5 - cy)
                if dist < radius - 1.5:
                    continue
                best = None
                for ny in range(y - 2, y + 3):
                    for nx in range(x - 2, x + 3):
                        if 0 <= nx < px.width and 0 <= ny < px.height and is_ink(nx, ny):
                            d = (nx - x) ** 2 + (ny - y) ** 2
                            if best is None or d < best[0]:
                                best = (d, pix[nx, ny])
                if best is not None:
                    pix[x, y] = (*best[1][:3], 255)


def place_glyph(
    px: Image.Image,
    padded: Image.Image,
    render_box: tuple[float, float, float, float],
    mark: tuple[int, int, int, int],
    scale: float,
    top: float,
    margin: int,
    ref: tuple[float, float, float] | None = None,
) -> None:
    """Redraw the mark with its centre on the middle of a pixel, then round it.

    ``ref`` is the mark's centre and radius in source pixels, if they are not to be
    read off this frame: the mark spins on the spot, and its box, which follows the
    swirl, moves a pixel or two as it turns; the circle should not.

    A mark whose centre falls a quarter of a pixel off the grid gives its top row
    a different coverage from its bottom one: one pixel up, six down. It is
    rendered again, shifted by that quarter, so the two ends match, and its
    pixels replace the frame's own.
    """
    left, up, right, down = render_box
    mid_x, mid_y = (mark[0] + mark[2]) / 2, (mark[1] + mark[3]) / 2
    radius_src = min(mark[2] - mark[0], mark[3] - mark[1]) / 2
    if ref is not None:
        mid_x, mid_y, radius_src = ref
    # Where the centre falls in the rendered pixels: the picture is read from
    # ``render_box``, which is moved by the sub-column shift ``compact_frames`` makes.
    cx = (mid_x + margin - left) * scale
    cy = (mid_y + margin - up) * scale
    radius = radius_src * scale
    tx, ty = math.floor(cx) + 0.5, math.floor(cy) + 0.5
    dx, dy = tx - cx, ty - cy
    moved = padded.resize(
        px.size,
        Image.BOX,
        box=(left - dx / scale, up - dy / scale, right - dx / scale, down - dy / scale),
    )
    src, dst = moved.load(), px.load()
    assert src is not None and dst is not None
    reach = int(radius) + 3
    for y in range(max(0, int(ty) - reach), min(px.height, int(ty) + reach + 1)):
        for x in range(max(0, int(tx) - reach), min(px.width, int(tx) + reach + 1)):
            if min(dst[x, y][:3]) >= 200 and dst[x, y][3] > 0:
                continue  # the word is not the mark's
            r, g, b, a = src[x, y]
            dst[x, y] = (0, 0, 0, 0) if (min(r, g, b) >= 200 and a > 0) else src[x, y]
    round_glyph(px, tx, ty, radius)


def render_compact_px(
    img: Image.Image,
    crop_top: int,
    width: int,
    letters: "tuple[list[Template], list[tuple[int, int]]] | None" = None,
    x_shift: float = 0.0,
    glyph_ref: "tuple[float, float, float] | None" = None,
) -> tuple[Image.Image, bool]:
    """One cropped frame at the compact geometry, as pixels, and whether its word
    was redrawn (see ``COMPACT_CAP``).

    With ``letters`` (the templates and the column of each letter in view, from
    ``letter_places``) those letters are redrawn (``tidy_letters``). Without, the word
    is redrawn only if it is exactly where ``TIDY`` expects it (``tidy_word``), which
    is how the templates themselves are made. ``glyph_ref`` is the centre and the
    radius of the mark, in source pixels, smoothed over the frames around this one.

    ``crop_top`` is the top of the crop in source pixels, to find the baseline;
    ``width`` names the cut in ``COMPACT_CUTS``. ``x_shift`` moves the picture right
    by that many source pixels, which ``compact_frames`` uses to put the lockup's left
    edge on a whole column.
    """
    global ALPHA_THRESHOLD
    scale = COMPACT_CUTS[width] / CAP_SRC
    box = content_bbox(img)
    if box is not None and (box[2] + x_shift) * scale > width:
        raise ValueError(
            f"compact banner: content reaches column {(box[2] + x_shift) * scale:.1f}, past {width}"
        )
    base = BASELINE_SRC - crop_top
    rows_above = math.ceil(base * scale)
    rows_above += rows_above % 2
    top = base - rows_above / scale
    height = 2 * math.ceil((img.height - top) * scale / 2)
    margin = 160
    padded = Image.new("RGBA", (img.width + 2 * margin, img.height + 2 * margin), (0, 0, 0, 0))
    padded.paste(img, (margin, margin))
    render_box = (
        margin - x_shift,
        top + margin,
        margin - x_shift + width / scale,
        top + margin + height / scale,
    )
    px = padded.resize((width, height), Image.BOX, box=render_box)
    saved, ALPHA_THRESHOLD = ALPHA_THRESHOLD, COMPACT_ALPHA
    try:
        if width in ROUND_GLYPH:
            mark = glyph_box(img)
            if mark and mark[2] - mark[0] >= GLYPH_DRAWN and mark[3] - mark[1] >= GLYPH_DRAWN:
                place_glyph(
                    px,
                    padded,
                    render_box,
                    (mark[0], mark[1], mark[2], mark[3]),
                    scale,
                    top,
                    margin,
                    glyph_ref,
                )
        if letters is not None:
            return px, tidy_letters(px, *letters)
        redrawn = width in TIDY and tidy_word(px, round((base - top) * scale), width)
        return px, redrawn
    finally:
        ALPHA_THRESHOLD = saved


# Columns cleared beyond a redrawn letter, each side: the filter's stroke can be a
# column wider than the template, and what is left of it would stand beside the
# new letter. Everything is cleared before anything is drawn.
CLEAR_MARGIN = 2

# A letter counts as fully in view when its ink is this much of its finished width.
# While one is still behind the mark it is narrower than that.
LETTER_VISIBLE = 0.92

Template = tuple[int, list[tuple[int, int]], int]
"""A letter: its width, its ink as ``(column, row)`` offsets from its left edge, and
where that edge is in the finished word, from the word's own left edge."""


def _white_ink(px: Image.Image) -> list[list[bool]]:
    pix = px.load()
    assert pix is not None
    return [
        [pix[x, y][3] >= COMPACT_ALPHA and min(pix[x, y][:3]) >= 200 for x in range(px.width)]
        for y in range(px.height)
    ]


def _first_ink(ink: list[list[bool]], cap_top: int, baseline_row: int) -> int | None:
    """The leftmost column with white ink between the cap line and the baseline."""
    for x in range(len(ink[0])):
        if any(ink[y][x] for y in range(max(0, cap_top), min(len(ink), baseline_row))):
            return x
    return None


def letter_templates(px: Image.Image, cap_top: int, baseline_row: int, cut: int) -> list[Template]:
    """The five letters of a finished, redrawn frame, left to right.

    Cut where ``TIDY[cut]`` says each letter is: D, o, m and n are listed there and
    the y is what lies between the m and the n. Splitting at blank columns instead
    would work for one cut and not the other, where the m, y and n touch.
    """
    ink = _white_ink(px)
    first = _first_ink(ink, cap_top, baseline_row)
    assert first is not None
    placed = [(col, col + len(rows[0])) for col, _, rows in TIDY[cut]["letters"]]
    d, o, m, n = placed
    spans = [d, o, m, (m[1], n[0]), n]
    out: list[Template] = []
    for lo, hi in spans:
        cells = [
            (x - first - lo, y)
            for y, row in enumerate(ink)
            for x in range(first + lo, min(px.width, first + hi))
            if row[x]
        ]
        out.append((hi - lo, cells, lo))
    return out


def word_clusters(img: Image.Image, crop_top: int) -> list[tuple[float, float]]:
    """Left and right edge, in source pixels, of each run of white ink in the word.

    Read on the source, where a letter's edge is known to a fraction of a pixel:
    that is what lets a letter be put on a whole column that moves smoothly from
    frame to frame, which matching the rendered pixels does not (the filter's edge
    flips by a column or two between frames).
    """
    base = int(BASELINE_SRC - crop_top)
    r, g, b, a = img.crop((0, base - int(CAP_SRC), img.width, base)).split()
    white = ImageChops.multiply(
        ImageChops.darker(ImageChops.darker(r, g), b).point(lambda v: 255 if v >= 232 else 0),
        a.point(lambda v: 255 if v >= 128 else 0),
    )
    columns = [
        white.crop((x, 0, x + 1, white.height)).getbbox() is not None for x in range(img.width)
    ]
    runs: list[tuple[float, float]] = []
    start = None
    for x, filled in enumerate([*columns, False]):
        if filled and start is None:
            start = x
        elif not filled and start is not None:
            if x - start >= MIN_LETTER_SRC:
                runs.append((float(start), float(x)))
            start = None
    return runs


# Narrowest run of white ink, in source pixels, that is a letter and not a speck.
MIN_LETTER_SRC = 40


def letter_places(
    clusters: list[tuple[float, float]],
    finished: list[tuple[float, float]],
    templates: list[Template],
    offsets: list[float],
    scale: float,
    x_shift: float,
) -> list[tuple[int, int]]:
    """Which letters are fully in view in a frame, and on which column to draw each.

    The reveal brings the letters in from the right, so the clusters in view are the
    last ones of the word. A letter is drawn only if its cluster is nearly as wide
    as it will be, and its column is its source position, scaled, offset by where
    the finished frame has it, and rounded: a smooth path becomes a smooth path.
    """
    n = len(templates)
    if not clusters or len(clusters) > n:
        return []
    first_letter = n - len(clusters)
    out: list[tuple[int, int]] = []
    for j, (lo, hi) in enumerate(clusters):
        k = first_letter + j
        done = finished[k][1] - finished[k][0]
        if hi - lo < LETTER_VISIBLE * done:
            continue
        out.append((k, round((lo + x_shift) * scale + offsets[k])))
    return out


def tidy_letters(px: Image.Image, templates: list[Template], places: list[tuple[int, int]]) -> bool:
    """Redraw the letters in ``places`` from their templates, each on its column.

    The ink round each is cleared first, a little wider than the letter. When every
    letter is there, everything from the first to the right edge of the picture is
    cleared: nothing else is right of the word. Returns whether any was redrawn.
    """
    if not places:
        return False
    ink = _white_ink(px)
    width = px.width
    pix = px.load()
    assert pix is not None
    n = len(templates)

    def clear(lo: int, hi: int) -> None:
        for x in range(max(0, lo), min(width, hi)):
            for y in range(px.height):
                if ink[y][x]:
                    pix[x, y] = (0, 0, 0, 0)

    if len(places) == n:
        clear(places[0][1] - CLEAR_MARGIN, width)
    else:
        for k, p in places:
            clear(p - CLEAR_MARGIN, p + templates[k][0] + CLEAR_MARGIN)
        if places[-1][0] == n - 1:
            clear(places[-1][1] + templates[-1][0] + CLEAR_MARGIN, width)
    for k, p in places:
        for dx, y in templates[k][1]:
            if 0 <= p + dx < width:
                pix[p + dx, y] = (255, 255, 255, 255)
    return True


def compact_text(px: Image.Image, width: int) -> str:
    """Half-block text of compact pixels, read at the compact alpha cut."""
    global ALPHA_THRESHOLD
    saved, ALPHA_THRESHOLD = ALPHA_THRESHOLD, COMPACT_ALPHA
    try:
        return to_half_blocks(px, width)
    finally:
        ALPHA_THRESHOLD = saved


def to_half_blocks_compact(img: Image.Image, crop_top: int, width: int) -> str:
    """One cropped frame at the compact geometry, as half-block text."""
    return compact_text(render_compact_px(img, crop_top, width)[0], width)


# Frames either side of one that its mark's centre and size are averaged over.
GLYPH_SMOOTH = 12


def glyph_refs(frames: list[Image.Image]) -> list[tuple[float, float, float] | None]:
    """The mark's centre and radius in each frame, source pixels, smoothed.

    Its box follows the swirl as it turns, so read frame by frame its centre wanders
    a few source pixels and its size more: a pixel or two on the rendered mark, and
    the mark appears to bob. The median over the frames around each one keeps what
    really moves (the mark shrinks and travels as the word appears) and drops that.
    ``None`` for a frame in which the mark is still being drawn.
    """
    marks = [glyph_box(f) for f in frames]
    drawn = [
        m is not None and m[2] - m[0] >= GLYPH_DRAWN and m[3] - m[1] >= GLYPH_DRAWN for m in marks
    ]
    out: list[tuple[float, float, float] | None] = []
    for i, m in enumerate(marks):
        if not drawn[i] or m is None:
            out.append(None)
            continue
        near = [
            marks[j]
            for j in range(max(0, i - GLYPH_SMOOTH), min(len(marks), i + GLYPH_SMOOTH + 1))
            if drawn[j]
        ]
        xs = sorted((q[0] + q[2]) / 2 for q in near if q)
        ys = sorted((q[1] + q[3]) / 2 for q in near if q)
        rs = sorted(min(q[2] - q[0], q[3] - q[1]) / 2 for q in near if q)
        mid = len(xs) // 2
        out.append((xs[mid], ys[mid], rs[mid]))
    return out


def hold_glyph(
    refs: list[tuple[float, float, float] | None], scale: float
) -> list[tuple[float, float, float] | None]:
    """``refs`` with the centre and radius held until they move a whole pixel.

    Even smoothed, a value that sits near a pixel boundary rounds one way and the
    other from frame to frame, and the mark bobs by a pixel. Held, it changes only
    when the drift is a full rendered pixel, and never goes back.
    """
    out: list[tuple[float, float, float] | None] = []
    held: tuple[float, float, float] | None = None
    # Walked from the last frame back, so the finished mark keeps the position the
    # static banner has.
    for ref in reversed(refs):
        if ref is None:
            out.append(None)
            continue
        if held is None:
            held = ref
        else:
            held = tuple(  # type: ignore[assignment]
                new if abs(new - old) * scale >= 1.0 else old
                for new, old in zip(ref, held, strict=True)
            )
        out.append(held)
    return out[::-1]


def compact_frames(frames: list[Image.Image], crop_top: int, width: int) -> list[str]:
    """The reveal and the closing slide of one compact cut, as half-block text.

    The slide moves the finished frame by whole columns. Re-rendering the source
    at each offset, as the full-size cut does, samples the letters at a different
    fraction of a pixel every frame, and their strokes shimmer as they travel; a
    column is one pixel wide, so a whole-column move is exact.
    """
    scale = COMPACT_CUTS[width] / CAP_SRC
    final_box = content_bbox(frames[-1])
    # The lockup's left edge, in compact pixels, rounded up to a whole column and
    # then put there: the picture is moved right by under one column, so that the
    # slide's end (the edge on column 0) is the same pixels, whole columns away,
    # as the frame it starts from. Letters sampled at one fraction of a pixel
    # while the word settles and another once it has slid change shape on arrival.
    edge = final_box[0] * scale if final_box else 0.0
    x_shift = (math.ceil(edge) - edge) / scale
    finished, redrawn = render_compact_px(frames[-1], crop_top, width, None, x_shift)
    assert redrawn or width not in TIDY, "the finished word is not where TIDY expects it"
    # The baseline's pixel row, as render_compact_px puts it: the rows above it,
    # rounded up to an even number so it falls on a cell boundary.
    baseline_row = math.ceil((BASELINE_SRC - crop_top) * scale)
    baseline_row += baseline_row % 2
    cap_top = baseline_row - TIDY[width]["cap"]
    templates = letter_templates(finished, cap_top, baseline_row, width)
    # Where each finished letter is, in source and in rendered pixels: a letter
    # in an earlier frame is drawn at its source position plus the difference.
    finished_clusters = word_clusters(frames[-1], crop_top)
    assert len(finished_clusters) == len(templates), "the finished word is not five letters"
    first = _first_ink(_white_ink(finished), cap_top, baseline_row)
    assert first is not None
    offsets = [
        first + t[2] - (c[0] + x_shift) * scale
        for t, c in zip(templates, finished_clusters, strict=True)
    ]
    refs = hold_glyph(glyph_refs(frames), scale)
    out: list[str] = []
    px = finished
    for img, ref in zip(frames, refs, strict=True):
        places = letter_places(
            word_clusters(img, crop_top), finished_clusters, templates, offsets, scale, x_shift
        )
        px, _ = render_compact_px(img, crop_top, width, (templates, places), x_shift, ref)
        out.append(compact_text(px, width))
    ink = px.getchannel("A").point(lambda a: 255 if a >= COMPACT_ALPHA else 0).getbbox()
    left = ink[0] if ink else 0
    for i in range(1, SLIDE_STEPS + 1):
        t = i / SLIDE_STEPS
        shift = round(left * (1 - (1 - t) ** 2))
        moved = Image.new("RGBA", px.size, (0, 0, 0, 0))
        # No mask: a frame used as its own mask squares the alpha of its soft edges
        # and dims the white of the word below the threshold that reads as white.
        moved.paste(px, (-shift, 0))
        out.append(compact_text(moved, width))
    return out


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
    full = frames + slide_left(frames[-1], steps=SLIDE_STEPS)[1:]  # full-size cut only
    if N_FRAMES is None:
        keyframes = full
    else:
        keep = sorted({round(i * (len(full) - 1) / (N_FRAMES - 1)) for i in range(N_FRAMES)})
        keyframes = [full[i] for i in keep]
    print(f"{len(frames)} reveal + {len(full) - len(frames)} slide -> {len(keyframes)} keyframes")

    args.out.mkdir(parents=True, exist_ok=True)
    for width in (*WIDTHS, *COMPACT_CUTS):
        if width in COMPACT_CUTS:
            top = box[1] if box is not None else 0
            rendered = compact_frames(frames, top, width)
        else:
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
