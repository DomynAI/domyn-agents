"""PDF documents with reportlab, in the house style the slides use: the shared palette (its
backgrounds included) and DM Sans, taken from `skills/builtin/resources/`.

    import sys
    sys.path.insert(0, "<dir of this SKILL.md>/scripts")
    import pdf_layout as d

    story = [
        d.h1("Executive summary"),
        d.intro("Agents cut onboarding time by 40% across the three pilot teams."),
        d.p("The pilot ran for <b>six weeks</b> with 42 analysts."),
        d.bullets(["Faster first answer", "Fewer escalations"]),
        d.table([["Team", "Before", "After"], ["Risk", "9 days", "5 days"]]),
        d.chart(Path("files/onboarding.png"), caption="Days to first answer, by team"),
    ]
    d.build(Path("Pilot report.pdf"), story, title="Pilot report", subtitle="Q3 2026")

`build` adds the cover, the content pages' background and page number, and, when asked, the
closing page; the story is only the content.

Nothing about the look is written here. Every page setting (size, margins, spacing,
positions on the fixed pages, which backgrounds a page may use) is read from
`pdf_layout.json` beside this file (`LAYOUT`); every colour from the palette (`PALETTE`),
which the layout names by key ("text.light_background.title") and never as hex; and the
fonts and the size of every text role the slides share (title, body, caption…) from the
typography (`TYPOGRAPHY`), which the layout names by role. So a document matches the slides
and the charts (`seaborn_style.py`, the same palette) by construction, and a setting changes in
the JSON, not in the code.

The fonts are embedded, so the PDF looks the same wherever it is opened, unlike a deck,
which only names its fonts.

The resources are found from this file's place in the tree (`parents[3]`, i.e.
`skills/builtin/`), so the module is imported from the skill, never copied out of it.

Needs `reportlab` (which brings `pillow`).
"""

import json
import math
import re
from collections.abc import Iterable, Sequence
from functools import cache
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from PIL import Image
from reportlab.lib import pagesizes
from reportlab.lib.colors import Color, HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Flowable,
    Frame,
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus import Image as PictureFlowable

RESOURCES = Path(__file__).resolve().parents[3] / "resources"
PALETTE = json.loads((RESOURCES / "palette.json").read_text())
TYPOGRAPHY = json.loads((RESOURCES / "typography.json").read_text())
LAYOUT = json.loads(Path(__file__).resolve().with_name("pdf_layout.json").read_text())

__all__ = [
    "LAYOUT",
    "PALETTE",
    "build",
    "bullets",
    "callout",
    "chart",
    "escape",
    "h1",
    "h2",
    "h3",
    "intro",
    "numbered",
    "p",
    "page_break",
    "spacer",
    "table",
]

_ALIGN = {"left": TA_LEFT, "right": TA_RIGHT, "center": TA_CENTER}

# -- reading the settings ------------------------------------------------------------------


def _palette(key: str) -> Any:
    """A palette entry by its dotted key, as the layout names it: "text.light_background.body"."""
    node: Any = PALETTE
    for part in key.split("."):
        node = node[part]
    return node


def _colour(key: str) -> Color:
    return HexColor(_palette(key))


def _over_white(stop: dict[str, Any]) -> Color:
    """A palette gradient stop ({"color", "alpha"}) as the opaque colour it gives on white."""
    c = HexColor(stop["color"])
    a = stop.get("alpha", 1.0)
    return Color(*(a * v + (1 - a) for v in (c.red, c.green, c.blue)))


def _font(key: str) -> str:
    """The registered name of a style of the typography's `styles`."""
    return f"layout-{key}"


@cache
def _register_fonts() -> None:
    for key, style in TYPOGRAPHY["styles"].items():
        if style["file"]:  # a style not shipped (thin) is never drawn in a PDF
            path = RESOURCES / "assets" / "fonts" / style["file"]
            pdfmetrics.registerFont(TTFont(_font(key), str(path)))
    pdfmetrics.registerFontFamily(
        _font("regular"),
        normal=_font("regular"),
        bold=_font("bold"),
        italic=_font("italic"),
        boldItalic=_font("bold_italic"),
    )


_register_fonts()


def _page_size() -> tuple[float, float]:
    size = getattr(pagesizes, LAYOUT["page"]["size"].upper(), None)
    if not isinstance(size, tuple):
        raise ValueError(f"page size {LAYOUT['page']['size']!r}: a reportlab size, e.g. A4, LETTER")
    return size


def _text_width() -> float:
    margins = LAYOUT["page"]["margins"]
    return _page_size()[0] - margins["left"] - margins["right"]


def _text_height() -> float:
    margins = LAYOUT["page"]["margins"]
    return _page_size()[1] - margins["top"] - margins["bottom"]


def _text(name: str) -> dict[str, Any]:
    """`LAYOUT["text"][name]`, its `role` resolved into the role's `font` and `size`."""
    spec = dict(LAYOUT["text"][name])
    if "role" in spec:
        role = TYPOGRAPHY["roles"][spec.pop("role")]
        spec.setdefault("font", role["style"])
        spec.setdefault("size", role["size"])
    if spec.get("leading", spec["size"]) < spec["size"]:
        raise ValueError(
            f"text {name!r}: leading {spec['leading']} is smaller than its size {spec['size']}; "
            "lines would print over each other. Raise it in pdf_layout.json"
        )
    return spec


@cache
def _style(name: str) -> ParagraphStyle:
    """The paragraph style of `LAYOUT["text"][name]`."""
    spec = _text(name)
    kw: dict[str, Any] = {
        "fontName": _font(spec["font"]),
        "fontSize": spec["size"],
        "leading": spec.get("leading", spec["size"] * 1.35),
        "spaceBefore": spec.get("space_before", 0),
        "spaceAfter": spec.get("space_after", 0),
        "alignment": _ALIGN[spec.get("align", "left")],
        # A line holding larger text (<font size=...>) grows to fit it, instead of being set
        # at the style's leading and printing over the line above.
        "autoLeading": "max",
    }
    if "color" in spec:
        kw["textColor"] = _colour(spec["color"])
    return ParagraphStyle(name, **kw)


def _set_font(canvas: Any, name: str) -> None:
    spec = _text(name)
    canvas.setFont(_font(spec["font"]), spec["size"])
    canvas.setFillColor(_colour(spec["color"]))


# -- backgrounds ---------------------------------------------------------------------------


def _draw_background(canvas: Any, name: str) -> None:
    """Fill the page with a background of the palette's `backgrounds`, by name: a colour, or
    a gradient as the slides draw it, linear at `angle` degrees clockwise from left-to-right
    (spanning the page, corner to corner along that direction) or radial from the top-left
    corner."""
    w, h = canvas._pagesize
    spec = _palette(f"backgrounds.{name}")
    if isinstance(spec, str):
        canvas.setFillColor(HexColor(spec))
        canvas.rect(0, 0, w, h, stroke=0, fill=1)
        return
    colours = [HexColor(s["color"]) for s in spec["stops"]]
    positions = [s["pos"] for s in spec["stops"]]
    canvas.saveState()
    path = canvas.beginPath()
    path.rect(0, 0, w, h)
    canvas.clipPath(path, stroke=0, fill=0)
    if spec["gradient"] == "radial":
        canvas.radialGradient(0, h, math.hypot(w, h), colours, positions, extend=True)
    else:
        angle = math.radians(spec["angle"])
        dx, dy = math.cos(angle), -math.sin(angle)  # clockwise on the page: y points up here
        half = (w * abs(dx) + h * abs(dy)) / 2
        cx, cy = w / 2, h / 2
        x0, y0, x1, y1 = cx - dx * half, cy - dy * half, cx + dx * half, cy + dy * half
        canvas.linearGradient(x0, y0, x1, y1, colours, positions, extend=True)
    canvas.restoreState()


def _draw_paragraph(
    canvas: Any, text: str, style: ParagraphStyle, x: float, y: float, w: float, anchor: str
) -> float:
    """Draw a paragraph with its top (anchor="top") or bottom (anchor="bottom") at y; returns
    its height."""
    para = Paragraph(text, style)
    _, h = para.wrapOn(canvas, w, 10_000)
    para.drawOn(canvas, x, y - h if anchor == "top" else y)
    return h


def _fit(canvas: Any, text: str, name: str, width: float, height: float) -> ParagraphStyle:
    """The style `name`, made smaller (down to the cover's `min_scale` of its size) until
    `text` fits in width × height. Raises when even the smallest does not: the text is too
    long for its place, and has to be shortened rather than printed over something."""
    base = _style(name)
    least = LAYOUT["cover"]["min_scale"]
    scale = 1.0
    while True:
        style = ParagraphStyle(
            f"{name}-{scale:.2f}",
            base,
            fontSize=base.fontSize * scale,
            leading=base.leading * scale,
        )
        _, needed = Paragraph(text, style).wrapOn(canvas, width, 10_000)
        if needed <= height:
            return style
        if scale <= least + 1e-9:
            raise ValueError(
                f"{name.replace('_', ' ')} too long: it needs {needed:.0f}pt of the "
                f"{height:.0f}pt it has even at {least:.0%} of its size. Shorten it."
            )
        scale = max(least, scale - 0.05)


def _draw_rule(canvas: Any, spec: dict[str, Any], y: float) -> None:
    margins, w = LAYOUT["page"]["margins"], canvas._pagesize[0]
    canvas.setStrokeColor(_colour(spec["rule_color"]))
    canvas.setLineWidth(spec["rule_width"])
    canvas.line(margins["left"], y, w - margins["right"], y)


# -- the content (the story) ---------------------------------------------------------------


class _Heading(KeepTogether):
    """A heading, and the room it needs below it: `build` starts a new page before it when
    less than `room` points are left, so a heading never ends a page alone. (keepWithNext
    would tie it to the whole next flowable instead, and push a long table, with its heading,
    to the next page, leaving this one empty.)"""

    def __init__(self, flowables: list[Flowable], room: float) -> None:
        super().__init__(flowables)
        self.room = room


def h1(text: str) -> Flowable:
    """A section title, with the slides' title rule under it."""
    rule = LAYOUT["h1_rule"]
    line = HRFlowable(
        width="100%",
        thickness=rule["width"],
        color=_colour(rule["color"]),
        spaceBefore=rule["space_before"],
        spaceAfter=rule["space_after"],
    )
    return _Heading([Paragraph(text, _style("h1")), line], LAYOUT["text"]["h1"]["room"])


def h2(text: str) -> Flowable:
    """A subsection heading."""
    return _Heading([Paragraph(text, _style("h2"))], LAYOUT["text"]["h2"]["room"])


def h3(text: str) -> Flowable:
    """A minor heading."""
    return _Heading([Paragraph(text, _style("h3"))], LAYOUT["text"]["h3"]["room"])


def intro(text: str) -> Flowable:
    """The lead line under a section title, as the slides' intro line: at most two lines."""
    return Paragraph(text, _style("intro"))


def p(text: str) -> Flowable:
    """A body paragraph. `text` is reportlab markup: <b>, <i>, <u>, <a href="...">; a literal
    &, < or > in text you did not write as markup goes through `escape()`."""
    return Paragraph(text, _style("body"))


# A bullet glyph typed at the start of the text, or a dash/asterisk followed by a space
# ("- item"); "-5% churn" keeps its minus.
TYPED_BULLET = re.compile(r"^\s*(?:[•·▪‣◦]\s*|[-–—*]\s+)+")
TYPED_NUMBER = re.compile(r"^\s*\d+[.)]\s+")


def _list(items: Iterable[str], kind: str) -> Flowable:
    spec = LAYOUT["lists"]
    strip = TYPED_BULLET if kind == "bullet" else TYPED_NUMBER
    rows = [ListItem(Paragraph(strip.sub("", s), _style("body"))) for s in items]
    size = spec["bullet_size"] if kind == "bullet" else spec["number_size"]
    dedent, indent = spec["bullet_dedent"], spec["indent"]
    if kind == "number":
        # Room for the widest number, so "10" does not run into its item's text.
        widest = pdfmetrics.stringWidth(str(len(rows)), _font("regular"), size)
        dedent = max(dedent, widest + spec["number_gap"])
        indent = max(indent, dedent + indent - spec["bullet_dedent"])
    return ListFlowable(
        rows,
        bulletType="bullet" if kind == "bullet" else "1",
        start=spec["bullet"] if kind == "bullet" else None,
        bulletFontName=_font("regular"),
        bulletFontSize=size,
        bulletColor=_colour(spec["color"]),
        leftIndent=indent,
        bulletDedent=dedent,
        spaceAfter=spec["space_after"],
    )


def bullets(items: Iterable[str]) -> Flowable:
    """A bulleted list, one string per item, without a typed mark (one is removed anyway)."""
    return _list(items, "bullet")


def numbered(items: Iterable[str]) -> Flowable:
    """A numbered list, one string per item."""
    return _list(items, "number")


class _BandedTable(Table):
    """The template's table, with the header band drawn under its first row: a rounded
    rectangle with the palette's radial fill (spreading from the top-left corner) and its
    gradient outline. A table split across pages repeats its header, and with it the band."""

    def draw(self) -> None:
        spec = LAYOUT["table"]
        c = self.canv
        h0, w = self._rowHeights[0], self._width
        x, y = 0.0, self._height - h0
        r = spec["header_band_radius"] * h0
        outline = _palette(spec["header_band_outline"])
        fill = _palette(spec["header_band_fill"])
        reach = math.hypot(w, h0)
        c.saveState()
        path = c.beginPath()
        path.roundRect(x, y, w, h0, r)
        c.clipPath(path, stroke=0, fill=0)
        angle = math.radians(spec["header_band_outline_angle"])
        c.linearGradient(
            x,
            y + h0,
            x + reach * math.cos(angle),
            y + h0 - reach * math.sin(angle),
            [_over_white(s) for s in outline],
            [s["pos"] for s in outline],
            extend=True,
        )
        c.restoreState()
        c.saveState()
        path = c.beginPath()
        path.roundRect(x + 1, y + 1, w - 2, h0 - 2, max(r - 1, 0))
        c.clipPath(path, stroke=0, fill=0)
        c.radialGradient(
            x,
            y + h0,
            reach,
            [_over_white(s) for s in fill],
            [s["pos"] for s in fill],
            extend=True,
        )
        c.restoreState()
        super().draw()


def table(
    rows: Sequence[Sequence[Any]],
    col_widths: Sequence[float] | None = None,
    align: Sequence[str] | None = None,
) -> Flowable:
    """The template's table: header row on the header band; grid lines between body rows
    and columns only (none around the table, none in the header); every other body row
    filled, from the first. Rows are lists of cells (strings as markup, or flowables); the
    first row is the header and repeats on every page. `col_widths` in points (default:
    equal shares of the text width), `align` per column ("left", "right", "center"; numbers
    read best right-aligned)."""
    spec = LAYOUT["table"]
    ncols = len(rows[0])
    aligns = [_ALIGN[a] for a in align or []]
    aligns += [TA_LEFT] * (ncols - len(aligns))

    def cell(value: Any, header: bool, col: int) -> Any:
        if isinstance(value, Flowable):
            return value
        base = _style("table_header" if header else "table_cell")
        return Paragraph(
            str(value), ParagraphStyle(f"{base.name}{col}", base, alignment=aligns[col])
        )

    data = [[cell(v, ri == 0, ci) for ci, v in enumerate(row)] for ri, row in enumerate(rows)]
    widths = list(col_widths) if col_widths else [_text_width() / ncols] * ncols
    if sum(widths) > _text_width():  # never wider than the text: scaled down, in proportion
        widths = [w * _text_width() / sum(widths) for w in widths]
    commands: list[tuple[Any, ...]] = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), spec["padding_x"]),
        ("RIGHTPADDING", (0, 0), (-1, -1), spec["padding_x"]),
        ("TOPPADDING", (0, 0), (-1, -1), spec["padding_y"]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), spec["padding_y"]),
        ("TOPPADDING", (0, 0), (-1, 0), spec["header_padding_y"]),
        ("BOTTOMPADDING", (0, 0), (-1, 0), spec["header_padding_y"]),
        ("INNERGRID", (0, 1), (-1, -1), spec["grid_width"], _colour(spec["grid_color"])),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_colour(spec["band_row_color"]), None]),
    ]
    t = _BandedTable(data, colWidths=widths, repeatRows=1, style=TableStyle(commands))
    t.spaceBefore, t.spaceAfter = spec["space_before"], spec["space_after"]
    return t


def chart(path: Path | str, caption: str | None = None, width: float | None = None) -> Flowable:
    """A chart PNG (from seaborn_style.save, which draws in the same palette) at the text width,
    or `width` points, keeping its proportions; with an optional caption under it."""
    img = Image.open(path)
    # Never wider than the text, nor taller than a page leaves for it with its caption: a
    # bigger picture would run into the margin, or stop the build.
    w = min(width or _text_width(), _text_width())
    tallest = _text_height() - (3 * _style("caption").leading if caption else 0)
    h = w * img.height / img.width
    if h > tallest:
        w, h = w * tallest / h, tallest
    picture = PictureFlowable(str(path), width=w, height=h)
    parts: list[Flowable] = [picture]
    if caption:
        parts.append(Paragraph(caption, _style("caption")))
    parts.append(Spacer(1, LAYOUT["chart"]["space_after"]))
    return KeepTogether(parts)


def callout(text: str, kind: str = "highlight") -> Flowable:
    """A card with one short message (a key finding, a recommendation): "highlight" is the
    slides' card 01 (card_highlight, dark-background text), "light" is card 02 (card_light,
    outlined in the light divider). Text is markup, as in p()."""
    spec = LAYOUT["callout"]
    if kind not in ("highlight", "light"):
        raise ValueError(f"no callout {kind!r}: 'highlight' or 'light'")
    look = spec[kind]
    style = ParagraphStyle("callout", _style("callout"), textColor=_colour(look["text"]))
    commands: list[tuple[Any, ...]] = [
        ("BACKGROUND", (0, 0), (-1, -1), _colour(look["fill"])),
        ("ROUNDEDCORNERS", [spec["radius"]] * 4),
        ("LEFTPADDING", (0, 0), (-1, -1), spec["padding_x"]),
        ("RIGHTPADDING", (0, 0), (-1, -1), spec["padding_x"]),
        ("TOPPADDING", (0, 0), (-1, -1), spec["padding_y"]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), spec["padding_y"]),
    ]
    if "outline" in look:
        commands.append(("BOX", (0, 0), (-1, -1), look["outline_width"], _colour(look["outline"])))
    t = Table([[Paragraph(text, style)]], colWidths=[_text_width()], style=commands)
    t.spaceBefore, t.spaceAfter = spec["space_before"], spec["space_after"]
    return t


def spacer(points: float = 12) -> Flowable:
    """Vertical space, in points; never negative, which would print over what is above."""
    return Spacer(1, max(0.0, points))


def page_break() -> Flowable:
    return PageBreak()


# -- the fixed pages -----------------------------------------------------------------------


class _Document(BaseDocTemplate):
    def __init__(self, path: Path, options: dict[str, Any], **kw: Any) -> None:
        super().__init__(str(path), **kw)
        self.options = options
        w, h = self.pagesize
        m = LAYOUT["page"]["margins"]
        content = Frame(
            m["left"],
            m["bottom"],
            w - m["left"] - m["right"],
            h - m["top"] - m["bottom"],
            id="text",
        )
        empty = Frame(0, 0, w, h, id="empty")
        self.addPageTemplates(
            [
                PageTemplate("cover", [empty], onPage=self._cover),
                PageTemplate("content", [content], onPage=self._content),
                PageTemplate("closing", [empty], onPage=self._closing),
            ]
        )

    def _footer_y(self) -> float:
        return LAYOUT["page"]["footer_y"]

    def _cover(self, canvas: Any, _doc: Any) -> None:
        o, spec = self.options, LAYOUT["cover"]
        (w, h), m = self.pagesize, LAYOUT["page"]["margins"]
        _draw_background(canvas, o["cover"])
        bottom = h * spec["title_bottom"]
        width = w - m["left"] - m["right"]
        title_width = width - spec["title_right_inset"]
        # The title grows up from `bottom`, the subtitle down from under the rule: each is
        # made smaller, down to `min_scale`, until it fits before the top margin / the footer.
        title = _fit(canvas, o["title"], "cover_title", title_width, h - m["top"] - bottom)
        _draw_paragraph(canvas, o["title"], title, m["left"], bottom, title_width, "bottom")
        _draw_rule(canvas, spec, bottom - spec["rule_gap"])
        if o["subtitle"]:
            top = bottom - spec["subtitle_gap"]
            floor = self._footer_y() + spec["subtitle_clearance"]
            sub = _fit(canvas, o["subtitle"], "cover_subtitle", width, top - floor)
            _draw_paragraph(canvas, o["subtitle"], sub, m["left"], top, width, "top")

    def _content(self, canvas: Any, _doc: Any) -> None:
        o, m = self.options, LAYOUT["page"]["margins"]
        _draw_background(canvas, o["page_background"])
        _set_font(canvas, "page_number")
        rise = LAYOUT["page"]["footer_text_rise"]
        canvas.drawString(m["left"], self._footer_y() + rise, str(canvas.getPageNumber()))

    def _closing(self, canvas: Any, _doc: Any) -> None:
        o, spec = self.options, LAYOUT["closing"]
        (w, h), m = self.pagesize, LAYOUT["page"]["margins"]
        background = o["cover"]
        width = w - m["left"] - m["right"]
        _draw_background(canvas, background)
        top = h * spec["top"]
        _draw_paragraph(
            canvas, "Thank you", _style("closing_title"), m["left"], top, width, "bottom"
        )
        _draw_rule(canvas, spec, top - spec["rule_gap"])
        if o["contacts"]:
            more = top - spec["explore_gap"]
            _draw_paragraph(
                canvas, "Explore more", _style("explore_more"), m["left"], more, width, "top"
            )
            y = top - spec["contacts_gap"]
            for markup in o["contacts"]:
                y -= _draw_paragraph(canvas, markup, _style("contact"), m["left"], y, width, "top")


def build(
    path: Path | str,
    story: Sequence[Flowable],
    *,
    title: str,
    subtitle: str = "",
    cover: str | None = None,
    page_background: str | None = None,
    closing: bool = False,
    contacts: Sequence[str] = (),
    author: str = "",
) -> Path:
    """Write the document: the cover, the story on content pages (page number bottom left),
    then the closing page if asked for (external documents get one).
    `cover` and `page_background` name backgrounds of the palette the layout's `backgrounds`
    allows (default: its `default_cover` / `default_content`). `contacts` are lines for the
    closing page, as markup (a link: `<u><a href="...">...</a></u>`); without them it shows
    only "Thank you". `title` and `subtitle` are plain text. Returns the path."""
    allowed = LAYOUT["backgrounds"]
    cover = cover or allowed["default_cover"]
    page_background = page_background or allowed["default_content"]
    if cover not in allowed["cover"]:
        raise ValueError(f"cover background {cover!r}: one of {', '.join(allowed['cover'])}")
    if page_background not in allowed["content"]:
        raise ValueError(
            f"page background {page_background!r}: one of {', '.join(allowed['content'])}"
        )
    path = Path(path)
    options = {
        "title": escape(title),
        "subtitle": escape(subtitle),
        "cover": cover,
        "page_background": page_background,
        "contacts": list(contacts),
    }
    doc = _Document(
        path,
        options,
        pagesize=_page_size(),
        title=title,
        subject=subtitle,
        author=author,
    )
    flow: list[Flowable] = [NextPageTemplate("content"), PageBreak()]
    for item in story:
        if isinstance(item, _Heading):
            flow.append(CondPageBreak(item.room))
        flow.append(item)
    if closing:
        flow += [NextPageTemplate("closing"), PageBreak(), Spacer(1, 1)]
    doc.build(flow)
    return path
