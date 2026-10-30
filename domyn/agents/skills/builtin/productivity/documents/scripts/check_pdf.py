"""Check a PDF for colliding text: what a reader sees as words printed over each other, over
a chart or another picture, cut by a line, or running off the page.

    python check_pdf.py document.pdf            # a report; exit status 1 if anything collides
    python check_pdf.py document.pdf --json     # the same, as JSON

It reads where every glyph actually is (pymupdf), not where a layout meant it to be, so it
catches collisions whatever made them. What it reports:

- text on text: two runs of text whose glyphs overlap. Each run is measured by its glyph
  band (from the cap height to the descender, around its baseline) rather than its line
  box, so lines set closely but legibly are not reported;
- text on an image: text overlapping a picture that is not a page background (a chart, a
  screenshot). An image covering most of the page is a background, and text on it is the design;
- text crossed by a line: a drawn line through the glyphs (a table's grid line through a
  cell's text that overflowed, a rule through a title). An underline, below the baseline,
  is not a crossing;
- text outside the margins: text past the left or right margin of `pdf_layout.json`
  (beside this file), or off the page. Pass `--margin` to check against another value, or
  `--no-margins` to check only the page edges.

Needs `pymupdf` (`pip install pymupdf`).
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pymupdf

# Where glyphs are, relative to the baseline, as a share of the font size: from the cap
# height down to the descender. A line box is taller (it holds the accents and the line
# gap), and two line boxes of a closely set paragraph touch without any ink touching.
CAP_HEIGHT = 0.70
DESCENDER = 0.18
# Overlaps smaller than this, in points, are rounding, not a collision.
TOLERANCE = 0.75
# An image covering this share of the page is its background.
BACKGROUND_SHARE = 0.85
# How far past each edge text is looked for, in points.
SHIFT = 2000.0

LAYOUT_FILE = Path(__file__).resolve().with_name("pdf_layout.json")


@dataclass
class Run:
    """A run of text in one font and size, measured by its ink, without surrounding spaces."""

    page: int
    text: str
    x0: float
    x1: float
    baseline: float
    size: float
    line: tuple[int, int]

    @property
    def top(self) -> float:
        return self.baseline - CAP_HEIGHT * self.size

    @property
    def bottom(self) -> float:
        return self.baseline + DESCENDER * self.size


@dataclass
class Collision:
    page: int
    kind: str
    detail: str
    where: tuple[float, float, float, float]


def _runs(page: pymupdf.Page, number: int, shift: float) -> list[Run]:
    """The page's runs of text.

    pymupdf joins what sits on one baseline in one font into one span, so "10" printed into
    "Step ten" comes back as the single span "10Step ten": a span is cut into runs wherever
    a character starts before the one ahead of it ends, and the overlap of those runs is
    then found like any other. Coordinates are moved back by `shift` (see `check`)."""
    runs: list[Run] = []
    for b, block in enumerate(page.get_text("rawdict")["blocks"]):
        for li, line in enumerate(block.get("lines", [])):
            for span in line["spans"]:
                pieces: list[list[dict[str, Any]]] = [[]]
                for ch in span["chars"]:
                    if ch["c"].isspace():
                        pieces[-1].append(ch)
                        continue
                    ink = [c for c in pieces[-1] if not c["c"].isspace()]
                    if ink and ch["bbox"][0] < ink[-1]["bbox"][2] - TOLERANCE:
                        pieces.append([])
                    pieces[-1].append(ch)
                for chars in pieces:
                    ink = [c for c in chars if not c["c"].isspace()]
                    if not ink:
                        continue
                    runs.append(
                        Run(
                            page=number,
                            text="".join(c["c"] for c in chars).strip(),
                            x0=min(c["bbox"][0] for c in ink) - shift,
                            x1=max(c["bbox"][2] for c in ink) - shift,
                            baseline=span["origin"][1] - shift,
                            size=span["size"],
                            line=(b, li),
                        )
                    )
    return runs


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return min(a1, b1) - max(a0, b0)


def _box(run: Run) -> tuple[float, float, float, float]:
    return (round(run.x0, 1), round(run.top, 1), round(run.x1, 1), round(run.bottom, 1))


def _short(text: str, n: int = 40) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def _text_on_text(runs: list[Run]) -> list[Collision]:
    found = []
    ordered = sorted(runs, key=lambda r: r.top)
    for i, a in enumerate(ordered):
        for b in ordered[i + 1 :]:
            if b.top >= a.bottom - TOLERANCE:
                break  # sorted by top: nothing further down can reach a
            if _overlap(a.x0, a.x1, b.x0, b.x1) <= TOLERANCE:
                continue
            if _overlap(a.top, a.bottom, b.top, b.bottom) <= TOLERANCE:
                continue
            found.append(
                Collision(
                    a.page,
                    "text on text",
                    f"{_short(a.text)!r} and {_short(b.text)!r}",
                    _box(a),
                )
            )
    return found


def _images(page: pymupdf.Page, area: float, shift: float) -> list[pymupdf.Rect]:
    """Where the page's pictures are drawn, backgrounds (most of the page's `area`)
    excepted. Only embedded images: a gradient fill (a table's header band) is listed as an
    image by get_image_info, and text on it is the design."""
    rects = [r for img in page.get_images() for r in page.get_image_rects(img[0])]
    moved = [pymupdf.Rect(r.x0 - shift, r.y0 - shift, r.x1 - shift, r.y1 - shift) for r in rects]
    return [r for r in moved if r.width * r.height < BACKGROUND_SHARE * area]


def _text_on_images(runs: list[Run], images: list[pymupdf.Rect]) -> list[Collision]:
    found = []
    for run in runs:
        for img in images:
            if (
                _overlap(run.x0, run.x1, img.x0, img.x1) > TOLERANCE
                and _overlap(run.top, run.bottom, img.y0, img.y1) > TOLERANCE
            ):
                found.append(
                    Collision(run.page, "text on an image", repr(_short(run.text)), _box(run))
                )
    return found


def _lines(page: pymupdf.Page, shift: float) -> list[tuple[pymupdf.Point, pymupdf.Point]]:
    """Every stroked straight segment: lines, and the edges of stroked rectangles."""
    segments: list[tuple[pymupdf.Point, pymupdf.Point]] = []
    for path in page.get_drawings():
        if path.get("type") not in ("s", "fs") or not path.get("width"):
            continue
        for item in path["items"]:
            if item[0] == "l":
                segments.append((item[1], item[2]))
            elif item[0] == "re":
                r = item[1]
                corners = [r.tl, r.tr, r.br, r.bl]
                segments += [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    back = pymupdf.Point(shift, shift)
    return [(p - back, q - back) for p, q in segments]


def _lines_through_text(
    runs: list[Run], segments: list[tuple[pymupdf.Point, pymupdf.Point]]
) -> list[Collision]:
    found = []
    for run in runs:
        for p, q in segments:
            vertical = abs(p.x - q.x) < 0.5
            horizontal = abs(p.y - q.y) < 0.5
            if vertical:
                inside = run.x0 + TOLERANCE < p.x < run.x1 - TOLERANCE
                if inside and _overlap(run.top, run.bottom, min(p.y, q.y), max(p.y, q.y)) > 1:
                    found.append(
                        Collision(
                            run.page, "text crossed by a line", repr(_short(run.text)), _box(run)
                        )
                    )
                    break
            elif horizontal:
                # Through the letters (between the cap height and just above the baseline),
                # so an underline, below the baseline, is not one.
                through = run.top + TOLERANCE < p.y < run.baseline - 0.1 * run.size
                if through and _overlap(run.x0, run.x1, min(p.x, q.x), max(p.x, q.x)) > 1:
                    found.append(
                        Collision(
                            run.page, "text crossed by a line", repr(_short(run.text)), _box(run)
                        )
                    )
                    break
    return found


def _outside(
    runs: list[Run], size: tuple[float, float], margins: tuple[float, float] | None
) -> list[Collision]:
    (width, height), (left, right) = size, margins if margins else (0.0, 0.0)
    lo, hi = left - TOLERANCE, width - right + TOLERANCE
    found = []
    for run in runs:
        off_page = run.x0 < 0 or run.x1 > width or run.top < 0 or run.bottom > height
        if off_page or run.x0 < lo or run.x1 > hi:
            kind = "text off the page" if off_page else "text outside the margins"
            found.append(Collision(run.page, kind, repr(_short(run.text)), _box(run)))
    return found


def check(path: Path | str, margins: tuple[float, float] | None = None) -> list[Collision]:
    """Every collision in the PDF; `margins` = (left, right) in points, None for the page
    edges only."""
    found: list[Collision] = []
    with pymupdf.open(path) as doc:
        for number, page in enumerate(doc, start=1):
            size = (page.rect.width, page.rect.height)
            # pymupdf drops every character that lies wholly off the page, i.e. exactly the
            # text a reader loses. So the page box is widened, in memory only, by SHIFT on
            # every side, and every position is moved back by it.
            box = page.mediabox
            page.set_mediabox(
                pymupdf.Rect(box.x0 - SHIFT, box.y0 - SHIFT, box.x1 + SHIFT, box.y1 + SHIFT)
            )
            runs = _runs(page, number, SHIFT)
            found += _text_on_text(runs)
            found += _text_on_images(runs, _images(page, size[0] * size[1], SHIFT))
            found += _lines_through_text(runs, _lines(page, SHIFT))
            found += _outside(runs, size, margins)
    return found


def layout_margins() -> tuple[float, float] | None:
    if not LAYOUT_FILE.is_file():
        return None
    margins = json.loads(LAYOUT_FILE.read_text())["page"]["margins"]
    return margins["left"], margins["right"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--json", action="store_true", help="print the collisions as JSON")
    parser.add_argument(
        "--margin", type=float, help="left and right margin in points (default: pdf_layout.json)"
    )
    parser.add_argument(
        "--no-margins", action="store_true", help="check only the page edges, not the margins"
    )
    args = parser.parse_args(argv)
    margins: tuple[float, float] | None
    if args.no_margins:
        margins = None
    elif args.margin is not None:
        margins = (args.margin, args.margin)
    else:
        margins = layout_margins()
    found = check(args.pdf, margins)
    if args.json:
        print(json.dumps([asdict(c) for c in found], indent=2, ensure_ascii=False))
    elif not found:
        print(f"{args.pdf}: no collisions")
    else:
        print(f"{args.pdf}: {len(found)} collision(s)")
        for c in found:
            x0, y0, x1, y1 = c.where
            print(f"  page {c.page}: {c.kind}: {c.detail}  at ({x0}, {y0})–({x1}, {y1})")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
