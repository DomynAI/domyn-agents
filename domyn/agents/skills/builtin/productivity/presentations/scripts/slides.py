"""Build a deck in the house style with python-pptx, on the skill's PowerPoint template, with
every colour, font and background taken from the shared resources.

    import sys
    sys.path.insert(0, "<dir of SKILL.md>/scripts")
    import slides as sl

    deck = sl.Deck()
    deck.title_slide("Agents in production", "Board · October 2026")
    deck.text_slide("Agents cut onboarding time by 40%", ["Three pilot teams", "Six weeks"])
    deck.two_cards(
        "Claims before and after",
        [("Before: manual", "Adjusters spend 60% of their time on paperwork.", "High cost"),
         ("After: assisted", "The agent extracts, checks and recommends.", "70% faster")],
    )
    deck.closing_slide()
    deck.save("Q3 Board Update.pptx")   # Q3 Board Update.pptx only if nothing collides

The layouts are not in this file: they are the slide layouts of `../template/layouts.pptx`, a
PowerPoint template edited in PowerPoint. Its shapes name what they take from the shared
resources, `label [key=value; ...]` (see `inject`), and opening it writes those in, so the
template holds geometry and never a colour, font or image of its own that counts. A `Deck`
method picks a layout by name and fills its placeholders (`slot=`) with the text it is given;
the slides stay real slides of their layouts: placeholders, outline, "Reset slide".

Imported from the skill, never copied: it finds the template, the shared resources and the
collision check (`check_pptx.py`, beside it) from its own place in the tree. What no layout
covers goes on `Deck.content_slide()` with the low-level helpers (`text`, `bullets`, `card`,
`rule`, `picture`), which measure their text as they write it, as the layouts do.

Needs python-pptx (which brings Pillow and lxml).
"""

import copy
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.oxml import parse_xml
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import check_pptx  # noqa: E402  (beside this file; the check's measure is fits()'s too)

SKILL_DIR = HERE.parent
TEMPLATE = SKILL_DIR / "template" / "layouts.pptx"
RESOURCES = Path(__file__).resolve().parents[3] / "resources"
P = json.loads((RESOURCES / "palette.json").read_text())
T = json.loads((RESOURCES / "typography.json").read_text())
ASSETS = RESOURCES / "assets"

EMU = 914400  # per inch

__all__ = [
    "BOLD",
    "LIGHT",
    "REGULAR",
    "THIN",
    "TITLE",
    "Deck",
    "P",
    "PillarCard",
    "T",
    "TwoCard",
    "bullets",
    "card",
    "inject",
    "link",
    "picture",
    "rule",
    "run",
    "text",
]

# -- colours and type ------------------------------------------------------------------------


def rgb(h: str) -> RGBColor:
    return RGBColor.from_string(h.lstrip("#"))


def _style(key: str) -> tuple[str, bool]:
    st = T["styles"][key]
    return (st["family"], st["bold"])


# Type styles: (face, bold), from typography.json. SKILL.md's Typography says which takes which.
REGULAR, BOLD, TITLE, LIGHT, THIN = (
    _style(k) for k in ("regular", "bold", "title", "light", "thin")
)

LIGHT_TEXT = P["text"]["light_background"]
DARK_TEXT = P["text"]["dark_background"]


def _what_fits(box: Any) -> str | None:
    """The longest start of a one-paragraph text, in whole words, that fits its box: measured
    exactly as the check measures it, so it is the length to rewrite to, not an estimate."""
    paragraphs = box.text_frame.paragraphs
    if len(paragraphs) != 1 or not paragraphs[0].runs:
        return None
    runs = paragraphs[0].runs
    kept = [r.text for r in runs]
    words = "".join(kept).split()
    best, lo, hi = "", 1, len(words)
    try:
        while lo <= hi:
            mid = (lo + hi) // 2
            runs[0].text = " ".join(words[:mid])
            for r in runs[1:]:
                r.text = ""
            if check_pptx.fit(box).ok:
                best, lo = runs[0].text, mid + 1
            else:
                hi = mid - 1
    finally:
        for r, t in zip(runs, kept, strict=True):
            r.text = t
    return best


def fits(box: Any) -> Any:
    """Raise if a text box's text does not fit it, saying exactly what does: for one line or
    paragraph, the words that fit; for several, the lines the box holds. Every helper that
    writes text calls it."""
    f = check_pptx.fit(box)
    if f.ok:
        return box
    start = box.text_frame.text.split("\n")[0][:40]
    paragraphs = box.text_frame.paragraphs
    if len(paragraphs) == 1:
        room = _what_fits(box)
        if room:
            hint = (
                f"what fits is {room!r} ({len(room)} characters): rewrite it to that length or less"
            )
        else:
            hint = "not even its first word fits: use a shorter word"
    else:
        lines = max(1, int((f.holds + check_pptx.OVERFLOW_SHARE * f.pitch) / f.pitch))
        takes = max(lines + 1, round(f.needed / f.pitch))
        hint = (
            f"the box holds about {lines} lines at this size and the text takes about {takes} "
            f"({len(paragraphs)} paragraphs): cut points or shorten them by about "
            f"{takes - lines} lines"
        )
    raise ValueError(
        f"{start!r} does not fit: it needs {f.needed:.2f} in, the box holds {f.holds:.2f} in; "
        f"{hint}. Change the content, never the size (SKILL.md, Step 3)."
    )


# -- the low-level helpers: for "Other content" on a content_slide() -------------------------


def run(
    p: Any, s: str, size: str | float, color: str, style: tuple[str, bool] | None = None
) -> Any:
    """A run in paragraph p. size is a role of typography.json ("title", "body", …), which gives
    the style too, or, for an element only slides have, a size in pt (style then defaults to
    REGULAR)."""
    if isinstance(size, str):
        role = T["roles"][size]
        size, style = role["size"], style or _style(role["style"])
    style = style or REGULAR
    r = p.add_run()
    r.text = s
    r.font.name, r.font.bold = style
    r.font.size = Pt(size)
    r.font.color.rgb = rgb(color)
    return r


def text(
    slide: Any,
    x: float,
    y: float,
    w: float,
    h: float,
    s: str,
    size: str | float,
    color: str,
    style: tuple[str, bool] | None = None,
    align: Any = PP_ALIGN.LEFT,
    anchor: Any = MSO_ANCHOR.TOP,
    spacing: float | None = None,
) -> Any:
    r"""One text box at (x, y, w, h) inches, with zero margins; '\n' in s starts a new paragraph
    with the same style. size is a role or a size in pt, as for run(). Raises if s does not fit."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap, tf.auto_size, tf.vertical_anchor = True, MSO_AUTO_SIZE.NONE, anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if spacing:
            p.line_spacing = spacing
        run(p, line, size, color, style)
    return fits(box)


def link(p: Any, s: str, url: str, size: str | float, color: str, style: Any = None) -> Any:
    """An underlined hyperlink run that keeps the text colour instead of the theme's link blue."""
    r = run(p, s, size, color, style)
    r.font.underline = True
    r.hyperlink.address = url
    h = r._r.find(qn("a:rPr")).find(qn("a:hlinkClick"))
    ext = etree.SubElement(
        etree.SubElement(h, qn("a:extLst")),
        qn("a:ext"),
        uri="{A12FA001-AC4F-418D-AE19-62706E023703}",
    )
    etree.SubElement(
        ext, "{http://schemas.microsoft.com/office/drawing/2018/hyperlinkcolor}hlinkClr", val="tx"
    )
    return r


def _plain(shape: Any) -> Any:
    """Drop the theme style python-pptx attaches to new shapes: it adds a shadow and a default
    outline, which turns a hairline into a thick grey bar."""
    style = shape._element.find(qn("p:style"))
    if style is not None:
        shape._element.remove(style)
    return shape


def rule(slide: Any, x1: float, y: float, x2: float, color: str, pt: float = 0.75) -> Any:
    """A horizontal rule from x1 to x2 at y."""
    ln = _plain(
        slide.shapes.add_connector(
            MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y), Inches(x2), Inches(y)
        )
    )
    ln.line.color.rgb, ln.line.width = rgb(color), Pt(pt)
    return ln


def card(
    slide: Any,
    x: float,
    y: float,
    w: float,
    h: float,
    radius: float,
    fill: str | None = None,
    outline: str | None = None,
) -> Any:
    """A rounded rectangle. radius is python-pptx's adjustment (a share of the short side).
    Add the card before the text that sits on it."""
    s = _plain(
        slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
        )
    )
    s.adjustments[0] = radius
    if fill:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
    if outline:
        s.line.color.rgb, s.line.width = rgb(outline), Pt(1)
    else:
        s.line.fill.background()
    return s


def _grad(stops: list[dict[str, Any]], angle_deg: float | None = None, radial: bool = False) -> Any:
    g = etree.Element(qn("a:gradFill"), rotWithShape="1")
    lst = etree.SubElement(g, qn("a:gsLst"))
    for st in stops:
        gs = etree.SubElement(lst, qn("a:gs"), pos=str(int(st["pos"] * 100000)))
        c = etree.SubElement(gs, qn("a:srgbClr"), val=st["color"].lstrip("#").upper())
        if st.get("alpha", 1) < 1:  # a stop without one is opaque
            etree.SubElement(c, qn("a:alpha"), val=str(int(st["alpha"] * 100000)))
    if radial:  # a circle spreading from the top-left corner
        path = etree.SubElement(g, qn("a:path"), path="circle")
        etree.SubElement(path, qn("a:fillToRect"), r="100000", b="100000")
        etree.SubElement(g, qn("a:tileRect"), l="-100000", t="-100000")
    else:
        etree.SubElement(g, qn("a:lin"), ang=str(int((angle_deg or 0) * 60000)), scaled="0")
    return g


def gradient(
    shape: Any, stops: list[dict[str, Any]], angle_deg: float | None = None, radial: bool = False
) -> None:
    """Linear (angle_deg) or radial fill with per-stop transparency:
    stops = [{"pos", "color", "alpha"}, ...]."""
    spPr = shape._element.spPr
    for tag in ("a:solidFill", "a:noFill", "a:gradFill"):
        for e in spPr.findall(qn(tag)):
            spPr.remove(e)
    spPr.find(qn("a:prstGeom")).addnext(_grad(stops, angle_deg, radial))


def gradient_outline(
    shape: Any, stops: list[dict[str, Any]], angle_deg: float, pt: float = 1
) -> None:
    """Outline with a linear gradient, same stops format. Call after gradient()."""
    spPr = shape._element.spPr
    for e in spPr.findall(qn("a:ln")):
        spPr.remove(e)
    ln = etree.Element(qn("a:ln"), w=str(int(pt * 12700)))
    ln.append(_grad(stops, angle_deg))
    spPr.find(qn("a:gradFill")).addnext(ln)


def shadow(shape: Any, color: str, alpha: float, blur_pt: float, dist_pt: float) -> None:
    """Soft drop shadow straight down. Call after fill and line are set."""
    eff = etree.SubElement(shape._element.spPr, qn("a:effectLst"))
    sh = etree.SubElement(
        eff,
        qn("a:outerShdw"),
        blurRad=str(int(blur_pt * 12700)),
        dist=str(int(dist_pt * 12700)),
        dir="5400000",
        algn="t",
        rotWithShape="0",
    )
    c = etree.SubElement(sh, qn("a:srgbClr"), val=color.lstrip("#").upper())
    etree.SubElement(c, qn("a:alpha"), val=str(int(alpha * 100000)))


NO_GRID = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"  # PowerPoint's "No Style, No Grid"


def _cell_line(tag: str, color: str | None = None, pt: float = 0.5) -> Any:
    ln = etree.Element(qn(tag), w=str(int(pt * 12700)), cmpd="sng")
    if color:
        etree.SubElement(
            etree.SubElement(ln, qn("a:solidFill")), qn("a:srgbClr"), val=color.lstrip("#").upper()
        )
    else:
        etree.SubElement(ln, qn("a:noFill"))
    return ln


def table(
    slide: Any, x: float, y: float, rows: int, cols: int, col_w: float = 1.33, row_h: float = 0.41
) -> Any:
    """The template's table, empty: no style; header row unfilled and unlined; body rows with
    0.5pt inner grid lines (no outer border) and every other body row filled, from the first.
    Deck.table_slide() draws the header band under it and fills it."""
    gf = slide.shapes.add_table(
        rows, cols, Inches(x), Inches(y), Inches(col_w * cols), Inches(row_h * rows)
    )
    gf._element.graphic.graphicData.tbl.tblPr.find(qn("a:tableStyleId")).text = NO_GRID
    t = gf.table
    t.first_row = t.horz_banding = False
    grid, band = P["lines"]["table_grid"], P["surfaces"]["table_band_row"]
    for c in t.columns:
        c.width = Inches(col_w)
    for ri, r in enumerate(t.rows):
        r.height = Inches(row_h)
        for ci, cell in enumerate(r.cells):
            cell.margin_left = cell.margin_right = Inches(0.10)
            cell.margin_top = cell.margin_bottom = 0
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tcPr = cell._tc.get_or_add_tcPr()
            for e in list(tcPr):
                tcPr.remove(e)
            body = ri > 0
            tcPr.append(_cell_line("a:lnL", grid if body and ci > 0 else None))
            tcPr.append(_cell_line("a:lnR", grid if body and ci < cols - 1 else None))
            tcPr.append(_cell_line("a:lnT", grid if ri > 1 else None))
            tcPr.append(_cell_line("a:lnB", grid if body and ri < rows - 1 else None))
            if body and ri % 2 == 1:
                etree.SubElement(
                    etree.SubElement(tcPr, qn("a:solidFill")),
                    qn("a:srgbClr"),
                    val=band.lstrip("#").upper(),
                )
            else:
                etree.SubElement(tcPr, qn("a:noFill"))
    return t


# A bullet glyph typed at the start of the text, or a dash/asterisk followed by a space
# ("- item"); "-5% churn" keeps its minus.
TYPED_BULLET = re.compile(r"^\s*(?:[•·▪‣◦]\s*|[-–—*]\s+)+")


def bullets(box: Any, color: str, check: bool = True) -> Any:
    """Turn every paragraph of a text() box into a bullet: • in the regular face at 8pt in
    `color`, 0.2 in hanging, 8pt before. A mark typed at the start of the text is removed, and
    calling this twice on one box still gives one bullet. Raises if they no longer fit."""
    for p in box.text_frame.paragraphs:
        if p.runs:
            p.runs[0].text = TYPED_BULLET.sub("", p.runs[0].text)
        p.space_before = Pt(8)
        pPr = p._p.get_or_add_pPr()
        for tag in (
            "a:buClr",
            "a:buClrTx",
            "a:buSzPts",
            "a:buSzPct",
            "a:buSzTx",
            "a:buFont",
            "a:buFontTx",
            "a:buChar",
            "a:buAutoNum",
            "a:buNone",
        ):
            for e in pPr.findall(qn(tag)):
                pPr.remove(e)
        pPr.set("marL", str(Inches(0.2)))
        pPr.set("indent", str(-Inches(0.1)))
        etree.SubElement(
            etree.SubElement(pPr, qn("a:buClr")), qn("a:srgbClr"), val=color.lstrip("#").upper()
        )
        etree.SubElement(pPr, qn("a:buSzPts"), val="800")
        etree.SubElement(pPr, qn("a:buFont"), typeface=REGULAR[0])
        etree.SubElement(pPr, qn("a:buChar"), char="•")
    return fits(box) if check else box  # the bullets' indent and spacing take room


def picture(slide: Any, path: Path | str, x: float, y: float, w: float, h: float) -> Any:
    """A picture (a chart PNG from seaborn_style, a screenshot) inside the box (x, y, w, h),
    at its own proportions, centred in it: never stretched."""
    with Image.open(path) as im:
        aspect = im.width / im.height
    pw, ph = (w, w / aspect) if w / h < aspect else (h * aspect, h)
    left, top = x + (w - pw) / 2, y + (h - ph) / 2
    return slide.shapes.add_picture(str(path), Inches(left), Inches(top), Inches(pw), Inches(ph))


# -- the template: what its shapes name, written in from the shared resources ---------------

# Theme colour slots, from the palette: what PowerPoint offers in its colour pickers and
# applies to anything a person adds to the deck by hand.
THEME_COLOURS = {
    "dk1": "text.light_background.title",
    "lt1": "surfaces.slide_plain",
    "dk2": "text.light_background.heading",
    "lt2": "surfaces.table_band_row",
    "accent1": "primary",
    "accent2": "secondary",
    "accent3": "tertiary",
    "accent4": "quaternary",
    "accent5": "quinary",
    "accent6": "senary",
    "hlink": "secondary",
    "folHlink": "quaternary",
}
# The palette's accents: never text on a blue or violet background (blue on blue).
ACCENTS = ("primary", "secondary", "tertiary", "quaternary", "quinary", "senary")
# The backgrounds of palette.json that take text.light_background; every other one is dark.
LIGHT_BACKGROUNDS = ("white", "tint")
BLUE_BACKGROUNDS = ("blue", "violet")
LAYOUT_KEYS = {"background"}  # what a layout's own name may tag
TAG = re.compile(r"^(?P<label>.*?)\s*\[(?P<body>[^\]]*)\]\s*$")
COLOUR_KEYS = ("color", "fill", "outline", "line", "gradient", "shadow")
KEYS = {*COLOUR_KEYS, "role", "style", "size", "spacing", "align", "anchor", "bullets", "slot",
        "image", "angle", "blur", "dist", "width"}  # fmt: skip
STYLE_KEYS = ("regular", "bold", "title", "light", "thin")
ALIGN = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}


def palette(key: str) -> Any:
    """A palette entry by its dotted key, as the template names it: "text.light_background.body"."""
    node: Any = P
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f"the template names {key!r}, which palette.json does not have")
        node = node[part]
    return node


def parse_tag(name: str, where: str, keys: set[str] = KEYS) -> tuple[str, dict[str, str]]:
    """`label [key=value; ...]` -> (label, {key: value}). Every shape of a layout has one, and
    so does every layout's own name (`keys=LAYOUT_KEYS`)."""
    m = TAG.match(name or "")
    if not m:
        raise ValueError(f"{where}: {name!r} names nothing it takes from the resources")
    tags = {}
    for item in filter(None, (i.strip() for i in m["body"].split(";"))):
        key, _, value = item.partition("=")
        if key.strip() not in keys:
            raise ValueError(f"{where}: {name!r} has an unknown key {key.strip()!r}")
        tags[key.strip()] = value.strip()
    return m["label"], tags


def type_of(tags: dict[str, str], where: str) -> tuple[float, tuple[str, bool]]:
    """(size in pt, style) of a text shape: from its role, or its style and size."""
    if "role" in tags:
        if tags["role"] not in T["roles"]:
            raise ValueError(f"{where}: no role {tags['role']!r} in typography.json")
        role = T["roles"][tags["role"]]
        return role["size"], _style(role["style"])
    if tags.get("style") not in STYLE_KEYS or "size" not in tags:
        raise ValueError(f"{where}: a text shape names a role, or a style and a size")
    return float(tags["size"]), _style(tags["style"])


def _set_runs(shape: Any, tags: dict[str, str], where: str) -> None:
    """Font, size and colour of every run (and field) of a fixed text from its tags."""
    size, (family, bold) = type_of(tags, where)
    colour = palette(tags["color"]).lstrip("#").upper()
    for p in shape.text_frame.paragraphs:
        for child in p._p:
            if child.tag not in (qn("a:r"), qn("a:fld")):
                continue
            rpr = child.find(qn("a:rPr"))
            if rpr is None:
                rpr = etree.Element(qn("a:rPr"))
                child.insert(0, rpr)
            rpr.set("sz", str(round(size * 100)))
            rpr.set("b", "1" if bold else "0")
            for tag in ("a:solidFill", "a:latin"):
                for e in rpr.findall(qn(tag)):
                    rpr.remove(e)
            fill = etree.SubElement(rpr, qn("a:solidFill"))
            etree.SubElement(fill, qn("a:srgbClr"), val=colour)
            etree.SubElement(rpr, qn("a:latin"), typeface=family)


def _set_image(layout: Any, shape: Any, file: str, where: str) -> None:
    """Point a picture at the resources' file, in place of the template's stand-in."""
    path = ASSETS / file
    if not path.is_file():
        raise ValueError(f"{where}: {shape.name!r} names {file!r}, not in {ASSETS}")
    blip = shape._element.find(qn("p:blipFill")).find(qn("a:blip"))
    old = blip.get(qn("r:embed"))
    _, rid = layout.part.get_or_add_image_part(str(path))
    blip.set(qn("r:embed"), rid)
    if old != rid:
        layout.part.drop_rel(old)


def _set_theme(prs: Any) -> None:
    master = prs.slide_master
    theme = master.part.part_related_by(RT.THEME)
    # python-pptx's parser, as for every other part: entities are never resolved, so a
    # tampered template cannot pull in a local file or a URL (XXE)
    xml = parse_xml(theme.blob)
    scheme = xml.find(".//" + qn("a:clrScheme"))
    for slot, key in THEME_COLOURS.items():
        el = scheme.find(qn(f"a:{slot}"))
        for child in list(el):
            el.remove(child)
        etree.SubElement(el, qn("a:srgbClr"), val=palette(key).lstrip("#").upper())
    family = T["styles"]["regular"]["family"]
    for font in ("a:majorFont", "a:minorFont"):
        xml.find(".//" + qn(font)).find(qn("a:latin")).set("typeface", family)
    theme._blob = etree.tostring(xml, xml_declaration=True, encoding="UTF-8", standalone=True)
    # the master's own background: the plain slide
    c_sld = master._element.find(qn("p:cSld"))
    for bg in c_sld.findall(qn("p:bg")):
        c_sld.remove(bg)
    bg = etree.Element(qn("p:bg"))
    bg_pr = etree.SubElement(bg, qn("p:bgPr"))
    fill = etree.SubElement(bg_pr, qn("a:solidFill"))
    etree.SubElement(fill, qn("a:srgbClr"), val=palette("surfaces.slide_plain").lstrip("#").upper())
    etree.SubElement(bg_pr, qn("a:effectLst"))
    c_sld.insert(0, bg)


def _set_background(layout: Any, name: str, where: str) -> None:
    """A layout's own background (what PowerPoint's Format Background sets): the palette's
    `backgrounds[name]`, a colour or a gradient, linear at `angle` degrees clockwise from
    left-to-right or radial from the top-left corner."""
    if name not in P["backgrounds"]:
        raise ValueError(
            f"{where}: no background {name!r} in palette.json; one of {', '.join(P['backgrounds'])}"
        )
    spec = P["backgrounds"][name]
    c_sld = layout._element.find(qn("p:cSld"))
    for old in c_sld.findall(qn("p:bg")):
        c_sld.remove(old)
    bg = etree.Element(qn("p:bg"))
    bg_pr = etree.SubElement(bg, qn("p:bgPr"))
    if isinstance(spec, str):
        fill = etree.SubElement(bg_pr, qn("a:solidFill"))
        etree.SubElement(fill, qn("a:srgbClr"), val=spec.lstrip("#").upper())
    else:
        radial = spec["gradient"] == "radial"
        bg_pr.append(_grad(spec["stops"], spec.get("angle", 0), radial=radial))
    etree.SubElement(bg_pr, qn("a:effectLst"))
    c_sld.insert(0, bg)


def _white_links(element: Any, clr_map: Any) -> None:
    """Map the theme's link colours to white on a layout or slide (its clrMapOvr): link() asks
    for the text colour through an Office 2019 extension, and every viewer that ignores it
    (some presentation apps, older PowerPoint) draws links in the theme's hlink, a blue."""
    for old in element.findall(qn("p:clrMapOvr")):
        element.remove(old)
    ovr = etree.Element(qn("p:clrMapOvr"))
    mapping = etree.SubElement(ovr, qn("a:overrideClrMapping"), dict(clr_map.attrib))
    mapping.set("hlink", "lt1")
    mapping.set("folHlink", "lt1")
    element.find(qn("p:cSld")).addnext(ovr)


def inject(prs: Any) -> None:
    """Write the shared resources into a presentation opened from the template: the theme's
    colours and fonts, each layout's background, which its name tags
    (`Content · navy [background=navy]`), and, on every shape of every layout, what its tag
    names. The tag is then taken off the layout's name, so a deck lists its layouts by their
    labels. Raises on a layout or shape with no tag, a key, background or file the resources
    do not have, or an accent colour as text on a blue background: the template cannot
    drift from the resources unnoticed."""
    _set_theme(prs)
    for layout in prs.slide_layouts:
        label, tags = parse_tag(layout.name, "the template", LAYOUT_KEYS)
        if "background" not in tags:
            raise ValueError(f"layout {label!r} names no background: [background=<name>]")
        background = tags["background"]
        _set_background(layout, background, f"layout {label!r}")
        layout.name = label
        accent_text = []
        for shape in layout.shapes:
            where = f"layout {layout.name!r}"
            label, tags = parse_tag(shape.name, where)
            where = f"{where}, {label!r}"
            if "image" in tags:
                _set_image(layout, shape, tags["image"], where)
            if "fill" in tags:
                shape.fill.solid()
                shape.fill.fore_color.rgb = rgb(palette(tags["fill"]))
            if "gradient" in tags:
                gradient(shape, palette(tags["gradient"]), float(tags.get("angle", 0)))
            if "outline" in tags:
                shape.line.color.rgb = rgb(palette(tags["outline"]))
            if "line" in tags:
                shape.line.color.rgb = rgb(palette(tags["line"]))
                shape.line.width = Pt(float(tags.get("width", 0.75)))
            if "shadow" in tags:
                for eff in shape._element.spPr.findall(qn("a:effectLst")):
                    shape._element.spPr.remove(eff)
                shadow(
                    shape,
                    **palette(tags["shadow"]),
                    blur_pt=float(tags["blur"]),
                    dist_pt=float(tags["dist"]),
                )
            if "color" in tags:
                _set_runs(shape, tags, where)
                if tags["color"] in ACCENTS:
                    accent_text.append(label)
        if background in BLUE_BACKGROUNDS and accent_text:
            raise ValueError(
                f"layout {layout.name!r}: {', '.join(map(repr, accent_text))} set an accent "
                f"colour as text on the {background} background: blue on blue, use "
                "text.dark_background"
            )
        if _dark(background):
            _white_links(layout._element, prs.slide_master._element.find(qn("p:clrMap")))


# -- the layouts -----------------------------------------------------------------------------


class TwoCard(NamedTuple):
    """One card of Deck.two_cards(): a one-line heading, a body of up to 3 lines, a takeaway
    of up to 2 lines. A plain (heading, body, takeaway) tuple works the same."""

    heading: str
    body: str
    takeaway: str


class PillarCard(NamedTuple):
    """One card of Deck.three_cards() / four_cards(): a short title (one line on three cards,
    two on four), a one-line subtitle, and short bullets (≤ 4 on three cards, ≤ 3 on four). A plain (title, subtitle,
    bullets) tuple works the same."""

    title: str
    subtitle: str
    bullets: Sequence[str]


def _too_many(what: str, n: int, most: int, fix: str) -> ValueError:
    return ValueError(f"{n} {what}, the layout holds {most}: {fix} (SKILL.md, Step 3)")


def _own_paragraph(p: Any, align: Any, spacing: float) -> None:
    """Give a placeholder's paragraph every property it would otherwise inherit from the
    master's text styles (bullets, an indent, 90% line spacing, space before): written on the
    slide, it reads the same in PowerPoint and to the collision check, which reads only what
    the slide itself says. bullets() then adds its own."""
    p.alignment = align
    p.line_spacing = spacing
    p.space_before = Pt(0)
    p.space_after = Pt(0)
    ppr = p._p.get_or_add_pPr()
    ppr.set("marL", "0")
    ppr.set("indent", "0")
    for tag in ("a:buNone", "a:buChar", "a:buAutoNum"):
        for e in ppr.findall(qn(tag)):
            ppr.remove(e)
    etree.SubElement(ppr, qn("a:buNone"))


def _dark(background: str | None) -> bool:
    return background is not None and background not in LIGHT_BACKGROUNDS


def _variant(kind: str, background: str | None, default: str) -> str:
    """The layout of `kind` on a background: `kind` itself on its default one."""
    return kind if background in (None, default) else f"{kind} · {background}"


def _records(method: Any) -> Any:
    """A layout method that, when the content does not suit its layout (too many cards or
    bullets, text that does not fit), records why with the slide's number and carries on,
    so one run reports the whole deck. save() refuses while anything is recorded."""
    import functools

    @functools.wraps(method)
    def build(self: Any, *args: Any, **kwargs: Any) -> Any:
        self._where = f"slide {len(self.prs.slides) + 1} ({method.__name__})"
        try:
            return method(self, *args, **kwargs)
        except ValueError as e:
            self.problems.append(f"{self._where}: {e}")
            return None

    return build


class Deck:
    """One presentation on the template: start every slide with one of its methods, save it
    with save()."""

    def __init__(self) -> None:
        self.prs = Presentation(str(TEMPLATE))
        inject(self.prs)
        self.layouts = {layout.name: layout for layout in self.prs.slide_layouts}
        self._filled: dict[int, set[int]] = {}  # slide id -> placeholder idx written
        # Everything that does not fit, from every slide: a method records it and carries on,
        # so one run lists the whole deck's problems and save() refuses to save until none is
        # left. One fix per run would take a run per problem.
        self.problems: list[str] = []
        self._where = ""  # "slide 3 (agenda)", while a method builds it

    # --- slides and their placeholders

    def _start(self, layout: str, kind: str = "") -> Any:
        if layout not in self.layouts:
            options = sorted(n for n in self.layouts if n.split(" · ")[0] == layout.split(" · ")[0])
            raise ValueError(
                f"the template has no layout {layout!r}; {kind or 'it'} has: {', '.join(options)}"
            )
        s = self.prs.slides.add_slide(self.layouts[layout])
        # a slide's own mapping is the master's unless it states one: repeat the layout's
        ovr = self.layouts[layout]._element.find(qn("p:clrMapOvr"))
        if ovr is not None and ovr.find(qn("a:overrideClrMapping")) is not None:
            for old in s._element.findall(qn("p:clrMapOvr")):
                s._element.remove(old)
            s._element.find(qn("p:cSld")).addnext(copy.deepcopy(ovr))
        self._filled[s.slide_id] = set()
        return s

    def _slots(self, s: Any) -> dict[str, tuple[Any, dict[str, str]]]:
        """slot name -> (the layout's placeholder, its tags)."""
        out = {}
        for ph in s.slide_layout.placeholders:
            _, tags = parse_tag(ph.name, f"layout {s.slide_layout.name!r}")
            out[tags["slot"]] = (ph, tags)
        return out

    def _slot(self, s: Any, slot: str) -> tuple[Any, dict[str, str], Any]:
        layout_ph, tags = self._slots(s)[slot]
        idx = layout_ph.placeholder_format.idx
        ph = next(p for p in s.placeholders if p.placeholder_format.idx == idx)
        return ph, tags, layout_ph

    def _geometry(self, s: Any, slot: str) -> tuple[float, float, float, float]:
        """Where a slot is, in inches, as the template places it."""
        layout_ph, _ = self._slots(s)[slot]
        return tuple(
            v / EMU for v in (layout_ph.left, layout_ph.top, layout_ph.width, layout_ph.height)
        )  # type: ignore[return-value]

    def _fill(
        self, s: Any, slot: str, words: str, colour: str | None = None, check: bool = True
    ) -> Any:
        """Write words into a slot, one paragraph per line, in the type and colour its tags
        name. The geometry and frame of the layout's placeholder are written on the slide too,
        so the slide reads the same in PowerPoint and to the collision check."""
        ph, tags, layout_ph = self._slot(s, slot)
        sp = ph._element
        sppr = sp.find(qn("p:spPr"))
        for child in list(sppr):
            sppr.remove(child)
        sppr.append(copy.deepcopy(layout_ph._element.spPr.find(qn("a:xfrm"))))
        old = sp.find(qn("p:txBody"))
        body = copy.deepcopy(layout_ph._element.find(qn("p:txBody")))
        for p in body.findall(qn("a:p")):
            body.remove(p)
        etree.SubElement(body, qn("a:p"))
        if old is not None:
            sp.replace(old, body)
        else:
            sp.append(body)
        ph.name = slot
        where = f"layout {s.slide_layout.name!r}, {slot!r}"
        size, style = type_of(tags, where)
        colour = colour or palette(tags["color"])
        tf = ph.text_frame
        for i, line in enumerate(words.split("\n")):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            _own_paragraph(p, ALIGN[tags.get("align", "left")], float(tags.get("spacing", 1.0)))
            run(p, line, size, colour, style)
        self._filled[s.slide_id].add(ph.placeholder_format.idx)
        if check:
            self._measure(ph, slot, tags.get("bullets") == "yes", colour)
        elif tags.get("bullets") == "yes":
            bullets(ph, colour, check=False)
        return ph

    def _measure(self, ph: Any, slot: str, bulleted: bool = False, colour: str = "") -> None:
        """Bullet and measure a filled slot; what does not fit is recorded, not raised."""
        try:
            if bulleted:
                bullets(ph, colour)
            else:
                fits(ph)
        except ValueError as e:
            self.problems.append(f"{self._where}, {slot}: {e}")

    def _finish(self, s: Any) -> Any:
        """Remove the placeholders a slide did not use: an empty one shows its prompt."""
        for ph in list(s.placeholders):
            if ph.placeholder_format.idx not in self._filled[s.slide_id]:
                ph._element.getparent().remove(ph._element)
        return s

    # --- layout 1: the title slide

    @_records
    def title_slide(self, title: str, subtitle: str, background: str = "blue") -> Any:
        """The first slide: the title (≤ 2 lines, about 30 characters in all), the subtitle (one
        line, e.g. audience · date). "navy" and "violet" are the alternatives,
        only if the person asks for a different title look."""
        s = self._start(_variant("Title", background, "blue"), "the title slide")
        self._fill(s, "title", title)
        self._fill(s, "subtitle", subtitle)
        return self._finish(s)

    # --- every content slide

    def content_slide(
        self, title: str, background: str | None = None, intro: str | None = None
    ) -> Any:
        """A content slide, plain white unless background is given (one of the "Content"
        layouts'): the title (one line, ≤ 40 characters), its rule, the intro line if any (≤ 2
        lines) and the slide number. Returns the slide; its content goes from y 1.25 (1.57
        with an intro) to 4.90. The layouts below start from their own; use this only for
        content no layout covers."""
        self._where = f"slide {len(self.prs.slides) + 1} (content_slide)"
        s = self._start(_variant("Content", background, "white"), "a content slide")
        self._fill(s, "title", title)
        if intro:
            self._fill(s, "intro", intro)
        return self._finish(s)

    # --- layout 2: the agenda

    @_records
    def agenda(
        self, takeaway: str, summary: str, topics: Sequence[tuple[str, str]], dark: bool = False
    ) -> Any:
        """The agenda: a key takeaway (≤ 2 lines at 30pt, about 20 characters), a summary (1–2
        lines) and up to 6 topics, each (topic ≤ 30 characters, one-line description of
        about 55). dark=True is the occasional navy accent."""
        if not 1 <= len(topics) <= 6:
            raise _too_many("topics", len(topics), 6, "group them into fewer sections")
        s = self._start("Agenda · dark" if dark else "Agenda")
        self._fill(s, "title", "Agenda")
        self._fill(s, "takeaway", takeaway)
        self._fill(s, "summary", summary)
        for i, (topic, description) in enumerate(topics, start=1):
            self._fill(s, f"number {i}", f"{i:02d}")
            self._fill(s, f"topic {i}", topic)
            self._fill(s, f"description {i}", description)
            if i < len(topics):  # a separator under every row but the last, from the row's place
                x, y, _, _ = self._geometry(s, f"number {i}")
                tx, _, tw, _ = self._geometry(s, f"topic {i}")
                rule(s, x, y + 0.56, tx + tw, P["lines"]["divider_light"])
        return self._finish(s)

    # --- layout 3: the section divider

    @_records
    def divider_slide(
        self, heading: str, background: str = "tint", highlight: str | None = None
    ) -> Any:
        """A section divider: the heading centred (one line, ≤ 45 characters), with highlight,
        one of its words, in primary on the light background; the slide number. No title, no
        rule."""
        s = self._start(_variant("Section divider", background, "tint"), "the section divider")
        lit = bool(highlight) and not _dark(background) and highlight in heading
        if not lit:
            self._fill(s, "heading", heading)
            return self._finish(s)
        before, _, after = heading.partition(highlight)  # type: ignore[arg-type]
        ph = self._fill(s, "heading", before, check=False)
        _, tags, _ = self._slot(s, "heading")
        size, style = type_of(tags, "heading")
        paragraph = ph.text_frame.paragraphs[0]
        run(paragraph, highlight, size, P["primary"], style)  # type: ignore[arg-type]
        run(paragraph, after, size, palette(tags["color"]), style)
        self._measure(ph, "heading")
        return self._finish(s)

    # --- layout 4: two cards

    @_records
    def two_cards(
        self, title: str, cards: Sequence[TwoCard | tuple[str, str, str]], intro: str | None = None
    ) -> Any:
        """Two cards, for a two-part argument: before/after, problem/solution. Each card is
        (heading: one line, about 30 characters; body: ≤ 3 lines, about 160; takeaway: ≤ 2
        lines, about 120). Card 01 is the highlighted one."""
        if len(cards) != 2:
            raise ValueError(
                f"two_cards takes 2 cards, not {len(cards)}: three_cards, four_cards or text_slide"
            )
        s = self._start("Two cards")
        self._fill(s, "title", title)
        if intro:
            self._fill(s, "intro", intro)
        for i, item in enumerate(cards, start=1):
            heading, body, takeaway = TwoCard(*item)
            self._fill(s, f"heading {i}", heading)
            self._fill(s, f"body {i}", body)
            self._fill(s, f"takeaway {i}", takeaway)
        return self._finish(s)

    # --- layouts 5 and 6: three and four cards

    def _pillar_cards(
        self, layout: str, title: str, cards: Sequence[Any], most_bullets: int
    ) -> Any:
        s = self._start(layout)
        self._fill(s, "title", title)
        for i, item in enumerate(cards, start=1):
            card_title, subtitle, points = PillarCard(*item)
            if isinstance(points, str):
                points = [points]
            if len(points) > most_bullets:
                raise _too_many(
                    f"bullets in {card_title!r}", len(points), most_bullets, "keep the strongest"
                )
            self._fill(s, f"title {i}", card_title)
            self._fill(s, f"subtitle {i}", subtitle)
            if points:
                self._fill(s, f"bullets {i}", "\n".join(points))
        return self._finish(s)

    @_records
    def three_cards(
        self,
        title: str,
        cards: Sequence[PillarCard | tuple[str, str, Sequence[str]]],
        dark: bool = False,
    ) -> Any:
        """Three cards, for three pillars, offers or options. Each card is (title: one line,
        about 12 characters; subtitle: one line; bullets: ≤ 4 short ones). dark=True is the
        occasional navy accent."""
        if len(cards) != 3:
            raise ValueError(
                f"three_cards takes 3 cards, not {len(cards)}: two_cards or four_cards"
            )
        return self._pillar_cards("Three cards · dark" if dark else "Three cards", title, cards, 4)

    @_records
    def four_cards(
        self,
        title: str,
        cards: Sequence[PillarCard | tuple[str, str, Sequence[str]]],
        dark: bool = False,
    ) -> Any:
        """Four cards, for four pillars. Each card is (title: ≤ 2 lines, about 20 characters;
        subtitle: one line; bullets: ≤ 3 short ones). dark=True is the occasional navy accent."""
        if len(cards) != 4:
            raise ValueError(
                f"four_cards takes 4 cards, not {len(cards)}: three_cards, or a second slide"
            )
        return self._pillar_cards("Four cards · dark" if dark else "Four cards", title, cards, 3)

    # --- layout 7: the table

    @_records
    def table_slide(self, title: str, rows: Sequence[Sequence[Any]], col_w: float = 1.33) -> Any:
        """A table, for comparison data: rows of cells, the first being the header; ≤ 6
        columns and ≤ 7 rows, short cell text (one line per cell). col_w is each column's
        width in inches; the table may be as wide as the template's table area."""
        cols = len(rows[0]) if rows else 0
        if not rows or any(len(r) != cols for r in rows):
            raise ValueError("table_slide needs rows of the same length, the header first")
        if cols > 6:
            raise _too_many("columns", cols, 6, "drop or merge columns, or split the table")
        if len(rows) > 7:
            raise _too_many("rows", len(rows), 7, "split the table across two slides")
        s = self._start("Table")
        self._fill(s, "title", title)
        x, y, w, _ = self._geometry(s, "table")  # the table area: taken, then left empty
        if col_w * cols > w + 1e-9:
            raise ValueError(
                f"{cols} columns of {col_w} in make {col_w * cols:.2f} in; the most is {w:.2f}"
            )
        band = card(s, x - 0.01, y, col_w * cols, 0.40, 0.17)
        gradient(band, P["surfaces"]["table_header_band"], radial=True)
        gradient_outline(band, P["surfaces"]["table_header_band_outline"], 36, pt=1)
        t = table(s, x, y, len(rows), cols, col_w=col_w)
        for ri, row in enumerate(rows):
            for ci, value in enumerate(row):
                paragraph = t.cell(ri, ci).text_frame.paragraphs[0]
                if ri == 0:
                    run(paragraph, str(value), "table_header", P["primary"])
                else:
                    run(paragraph, str(value), "table_cell", P["text"]["light_background"]["title"])
        return self._finish(s)

    # --- layout 8: the closing slide

    @_records
    def closing_slide(self, contacts: Sequence[str] = ()) -> Any:
        """The closing slide: "Thank you", and, when contacts are given, "Explore more" and the
        contacts, one line each, as plain text (no links on a dark slide). Pass contacts only
        if the person gives them."""
        s = self._start("Closing")
        if contacts:
            self._fill(s, "explore more", "Explore more")
            self._fill(s, "contacts", "\n".join(contacts))
        return self._finish(s)

    # --- layout 9: text

    @_records
    def text_slide(
        self, title: str, points: Sequence[str], intro: str | None = None, dark: bool = False
    ) -> Any:
        """A list of points, the slide most decks need most: ≤ 6 bullets, each ≤ 2 lines (about
        180 characters), one point per bullet, without a typed mark. dark=True is the
        occasional navy accent."""
        if isinstance(points, str):
            points = [points]
        if not 1 <= len(points) <= 6:
            raise _too_many("bullets", len(points), 6, "split into two slides, or use cards")
        s = self._start(("Text with intro" if intro else "Text") + (" · dark" if dark else ""))
        self._fill(s, "title", title)
        if intro:
            self._fill(s, "intro", intro)
        self._fill(s, "points", "\n".join(points))
        return self._finish(s)

    # --- saving

    def save(self, path: Path | str, render: bool = True) -> Path:
        """The only way to save the deck. Writes `<name>.unchecked.pptx`, runs the collision
        check on it, and only if nothing collides renames it to the file asked for, printing
        `<file>: no collisions`. Otherwise it raises and the deck stays `<name>.unchecked.pptx`:
        not deliverable. A file already at `path` is removed first, so a failed build never
        leaves an earlier one looking finished. render=True also draws every slide into a PNG,
        in `<name> check/` beside it, for the visual review."""
        path = Path(path)
        path.unlink(missing_ok=True)
        if self.problems:
            listed = "\n".join(f"  {p}" for p in self.problems)
            raise RuntimeError(
                f"{len(self.problems)} slide(s) do not fit their layout, so nothing was saved. "
                "Fix every one of them in the build script, then run it once more:\n" + listed
            )
        unchecked = path.with_name(f"{path.stem}.unchecked{path.suffix}")
        self.prs.save(unchecked)
        report = check_pptx.check(unchecked)
        if render:
            try:
                check_pptx.render(unchecked, path.with_name(f"{path.stem} check"))
            except Exception as e:  # LibreOffice missing or failing: the check still decides
                print(f"render failed ({type(e).__name__}): {e}", file=sys.stderr)
        if report.findings:
            found = "\n".join(f"  slide {f.slide}: {f.kind}: {f.detail}" for f in report.findings)
            raise RuntimeError(
                f"collision check failed: {len(report.findings)} collision(s). The deck is NOT "
                f"deliverable and was left as {unchecked.name}. Fix the build script (change the "
                f"content, SKILL.md Step 3) and run it again:\n{found}"
            )
        unchecked.replace(path)
        for note in report.notes:
            print(f"  note: {note}")
        print(f"{path}: no collisions")
        return path
