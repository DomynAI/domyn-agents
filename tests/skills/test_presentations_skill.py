"""The presentations skill (`skills/builtin/productivity/presentations/`): a SKILL.md, the
PowerPoint template its layouts are (`template/layouts.pptx`), the module the model builds decks
with (`scripts/slides.py`: it writes the shared resources into the template, fills its layouts'
placeholders, and saves through the collision check), and the check itself
(`scripts/check_pptx.py`).

What can break: the template naming a palette key, role or file the resources do not have, or
carrying a picture of its own (a logo, a background image); a layout filled off its template; a capacity the
module does not enforce; and a deck that collides reaching the name it was asked to be saved
under.

The module tests need python-pptx, which the test environment does not install; they are
skipped without it: `uv run --with python-pptx pytest tests/skills/test_presentations_skill.py`.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

import domyn.agents
from domyn.agents.skills import LocalDirSkillSource, SkillQuery

PRODUCTIVITY = Path(domyn.agents.__file__).parent / "skills" / "builtin" / "productivity"
SKILL = PRODUCTIVITY / "presentations"
SKILL_MD = (SKILL / "SKILL.md").read_text()
SCRIPTS = SKILL / "scripts"
TEMPLATE = SKILL / "template" / "layouts.pptx"
RESOURCES = PRODUCTIVITY.parent / "resources"
PALETTE = json.loads((RESOURCES / "palette.json").read_text())
TYPOGRAPHY = json.loads((RESOURCES / "typography.json").read_text())
BACKGROUNDS = PALETTE["backgrounds"]
EMU = 914400
BODY = "New analysts needed three weeks to answer their first question on their own."


def python_blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", SKILL_MD, re.S)


def palette_value(path: str) -> Any:
    node: Any = PALETTE
    for key in path.split("."):
        node = node[key]
    return node


class TestTheSkill:
    @pytest.mark.asyncio
    async def test_it_is_indexed_with_the_builtins(self) -> None:
        index = await LocalDirSkillSource(paths=[PRODUCTIVITY]).index(SkillQuery(task=""))
        assert "slides-style" in {skill.name for skill in index.skills}

    def test_it_ships_its_module_and_its_template(self) -> None:
        assert {p.name for p in SCRIPTS.glob("*.py")} == {"slides.py", "check_pptx.py"}
        assert [p.relative_to(SKILL) for p in SKILL.rglob("*.pptx")] == [
            Path("template/layouts.pptx")
        ]

    def test_the_template_holds_no_picture(self) -> None:
        """No logo and no background image: its backgrounds are the palette's colours."""
        import zipfile

        with zipfile.ZipFile(TEMPLATE) as z:
            assert not [n for n in z.namelist() if n.startswith("ppt/media/")]
        assert TEMPLATE.stat().st_size < 200_000

    def test_it_gives_no_helper_code_to_copy(self) -> None:
        """The helpers are imported from scripts/slides.py; SKILL.md only calls them."""
        for block in python_blocks():
            assert not re.search(r"^\s*def |^\s*class ", block, re.M), block[:80]
        assert "import slides as sl" in SKILL_MD

    def test_every_layout_method_is_documented(self) -> None:
        source = (SCRIPTS / "slides.py").read_text()
        deck = source[source.index("class Deck") :]
        methods = set(re.findall(r"\n    def ([a-z]\w*)\(", deck))
        assert {"two_cards", "three_cards", "four_cards", "table_slide", "save"} <= methods
        for method in methods:
            assert f"deck.{method}(" in SKILL_MD, method


class TestWhatTheSpecNames:
    def test_it_names_no_picture_of_its_own(self) -> None:
        assert set(re.findall(r"[\w-]+\.(?:jpg|png)", SKILL_MD)) <= {"onboarding.png"}
        assert "logo" not in SKILL_MD.lower()

    def test_every_background_is_described(self) -> None:
        """A background the table does not list is one the model never picks."""
        for name in BACKGROUNDS:
            assert f"| `{name}` | `backgrounds.{name}`" in SKILL_MD, name

    def test_every_palette_key_it_names_exists(self) -> None:
        dotted = re.findall(r"`((?:text|surfaces|lines|backgrounds)\.[\w.]+)`", SKILL_MD)
        indexed = [
            ".".join(re.findall(r'\["(\w+)"\]', chain))
            for chain in re.findall(r'P((?:\["\w+"\])+)', SKILL_MD)
        ]
        names = set(dotted + indexed)
        assert names
        for name in names:
            palette_value(name)  # KeyError names the missing key

    @pytest.mark.parametrize("key", ["primary", "secondary", "tertiary", "quaternary"])
    def test_the_accents_exist(self, key: str) -> None:
        assert re.fullmatch(r"#[0-9A-F]{6}", PALETTE[key])

    def test_every_colour_is_a_hex_code(self) -> None:
        def colours(node: Any) -> list[str]:
            if isinstance(node, dict):  # a background's "gradient" is its kind, not a colour
                return [
                    c
                    for k, v in node.items()
                    if k != "gradient"
                    for c in (colours(v) if k != "color" else [v])
                ]
            if isinstance(node, list):
                return [c for item in node for c in colours(item)]
            return [node] if isinstance(node, str) else []

        found = colours(PALETTE)
        assert found and all(re.fullmatch(r"#[0-9A-F]{6}", c) for c in found)

    def test_every_background_is_a_colour_or_an_opaque_gradient_of_the_palette(self) -> None:
        """Its colours are the palette's own, and a gradient has no transparency: there is
        nothing under a background to show through, in a deck or a PDF."""
        others = json.dumps({k: v for k, v in PALETTE.items() if k != "backgrounds"})
        for name, spec in BACKGROUNDS.items():
            if isinstance(spec, str):
                assert spec in others, name
                continue
            assert spec["gradient"] in ("linear", "radial"), name
            assert ("angle" in spec) == (spec["gradient"] == "linear"), name
            positions = [stop["pos"] for stop in spec["stops"]]
            assert positions[0] == 0 and positions[-1] == 1 and positions == sorted(positions)
            for stop in spec["stops"]:
                assert set(stop) == {"pos", "color"} and stop["color"] in others, (name, stop)

    @pytest.mark.parametrize(
        ("backgrounds", "colours", "least"),
        [
            (["white", "tint"], ["title", "heading", "body"], 4.5),
            (["blue", "navy", "violet"], ["heading", "body"], 4.5),
            (["blue", "navy"], ["muted"], 4.5),
            (["violet"], ["muted"], 3.0),  # 18pt and larger only, as the Contrast table says
            (["white"], ["muted"], 4.5),
            (["tint"], ["muted"], 4.2),  # the one exception: short descriptions at 11pt+
        ],
    )
    def test_the_text_colours_stay_readable_across_every_gradient(
        self, backgrounds: list[str], colours: list[str], least: float
    ) -> None:
        """Text can sit anywhere on a slide, so every colour of a gradient has to keep the
        contrast the Contrast table promises, not only its average."""

        def luminance(hex_colour: str) -> float:
            def channel(v: float) -> float:
                return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

            r, g, b = (channel(int(hex_colour[i : i + 2], 16) / 255) for i in (1, 3, 5))
            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        def contrast(a: str, b: str) -> float:
            la, lb = sorted((luminance(a), luminance(b)), reverse=True)
            return (la + 0.05) / (lb + 0.05)

        for name in backgrounds:
            spec = BACKGROUNDS[name]
            stops = [spec] if isinstance(spec, str) else [st["color"] for st in spec["stops"]]
            family = "light_background" if name in ("white", "tint") else "dark_background"
            for key in colours:
                text = PALETTE["text"][family][key]
                worst = min(contrast(text, stop) for stop in stops)
                assert worst >= least, (name, key, round(worst, 2))
        navy = BACKGROUNDS["navy"]["stops"]  # the dark agenda's numbers
        assert min(contrast(PALETTE["tertiary"], st["color"]) for st in navy) >= 4.5

    def test_every_role_it_names_is_in_the_typography(self) -> None:
        named = set(re.findall(r"role `(\w+)`", SKILL_MD))
        assert named and named <= set(TYPOGRAPHY["roles"])

    def test_it_lists_every_role(self) -> None:
        """A role the Typography section does not list is one the model never uses."""
        for role in TYPOGRAPHY["roles"]:
            assert f"`{role}`" in SKILL_MD, role

    def test_the_module_types_no_family_name_and_no_colour(self) -> None:
        """Families come from typography.json and colours from palette.json, so the slides,
        the PDFs and the charts cannot drift apart."""
        source = (SCRIPTS / "slides.py").read_text()
        assert "DM Sans" not in source
        assert not re.findall(r'"#[0-9A-Fa-f]{6}"', source)
        assert "`DM Sans Thin`" in SKILL_MD and "not among these files" in SKILL_MD


# --- the module -----------------------------------------------------------------------------


@pytest.fixture
def sl() -> Any:
    pytest.importorskip("pptx")
    sys.path.insert(0, str(SCRIPTS))
    try:
        import slides
    finally:
        sys.path.remove(str(SCRIPTS))
    return slides


@pytest.fixture
def deck(sl: Any) -> Any:
    return sl.Deck()


def blank(sl: Any, deck: Any, background: str | None = None) -> Any:
    """A slide for the low-level helpers: a content slide, its title left empty."""
    s = deck.content_slide("Pilot results by team", background=background)
    return s


def children(element: Any) -> list[str]:
    return [c.tag.split("}")[1] for c in element]


def boxes(slide: Any) -> dict[str, tuple[float, ...]]:
    """Text -> (x, y, w, h) in inches, for every text box of the slide."""
    return {
        sh.text_frame.text: tuple(round(v / EMU, 2) for v in (sh.left, sh.top, sh.width, sh.height))
        for sh in slide.shapes
        if sh.has_text_frame and sh.text_frame.text
    }


def cards(shapes: Any) -> list[tuple[float, ...]]:
    """(x, y, w, h) of the rounded rectangles among shapes, left to right."""
    found = [
        tuple(round(v / EMU, 2) for v in (sh.left, sh.top, sh.width, sh.height))
        for sh in shapes
        if sh.shape_type == 1 and sh.auto_shape_type == 5  # MSO_SHAPE.ROUNDED_RECTANGLE
    ]
    return sorted(found)


class TestText:
    def test_the_box_has_no_margins_and_no_autofit(self, sl: Any, deck: Any) -> None:
        box = sl.text(blank(sl, deck), 1, 1, 3, 1, "a\nb", 11, "#000000")
        tf = box.text_frame
        assert (tf.margin_left, tf.margin_right, tf.margin_top, tf.margin_bottom) == (0, 0, 0, 0)
        assert len(tf.paragraphs) == 2

    @pytest.mark.parametrize("style", ["REGULAR", "BOLD", "TITLE", "LIGHT", "THIN"])
    def test_every_run_carries_its_style(self, sl: Any, deck: Any, style: str) -> None:
        box = sl.text(blank(sl, deck), 1, 1, 3, 1, "a\nb", 11, "#000000", style=getattr(sl, style))
        for p in box.text_frame.paragraphs:
            for r in p.runs:
                assert (r.font.name, bool(r.font.bold)) == getattr(sl, style)

    def test_a_role_sets_size_and_style(self, sl: Any, deck: Any) -> None:
        box = sl.text(blank(sl, deck), 1, 1, 7, 1, "A title", "title", "#000000")
        r = box.text_frame.paragraphs[0].runs[0]
        assert r.font.size.pt == TYPOGRAPHY["roles"]["title"]["size"]
        assert (r.font.name, r.font.bold) == sl.TITLE

    def test_a_size_in_pt_defaults_to_regular(self, sl: Any, deck: Any) -> None:
        box = sl.text(blank(sl, deck), 1, 1, 7, 1, "53", 18, "#000000")
        r = box.text_frame.paragraphs[0].runs[0]
        assert r.font.size.pt == 18 and (r.font.name, r.font.bold) == sl.REGULAR


class TestBullets:
    def test_a_typed_mark_is_removed_and_a_minus_kept(self, sl: Any, deck: Any) -> None:
        box = sl.text(
            blank(sl, deck), 1, 1, 3, 2, "• one\n- two\n-5% churn\n•three\nfour", 11, "#FFFFFF"
        )
        sl.bullets(box, "#FFFFFF")
        assert [p.text for p in box.text_frame.paragraphs] == [
            "one",
            "two",
            "-5% churn",
            "three",
            "four",
        ]

    def test_calling_it_twice_still_gives_one_bullet(self, sl: Any, deck: Any) -> None:
        from pptx.oxml.ns import qn

        box = sl.text(blank(sl, deck), 1, 1, 3, 2, "one\ntwo", 11, "#FFFFFF", spacing=1.12)
        sl.bullets(box, "#FFFFFF")
        sl.bullets(box, "#FFFFFF")
        for p in box.text_frame.paragraphs:
            ppr = p._p.find(qn("a:pPr"))
            assert children(ppr) == ["lnSpc", "spcBef", "buClr", "buSzPts", "buFont", "buChar"]

    def test_the_bullet_is_the_template_s(self, sl: Any, deck: Any) -> None:
        from pptx.oxml.ns import qn

        box = sl.text(blank(sl, deck), 1, 1, 3, 1, "one", 11, "#FFFFFF")
        sl.bullets(box, "#FFFFFF")
        ppr = box.text_frame.paragraphs[0]._p.find(qn("a:pPr"))
        assert ppr.find(qn("a:buFont")).get("typeface") == TYPOGRAPHY["styles"]["regular"]["family"]
        assert ppr.find(qn("a:buSzPts")).get("val") == "800"
        assert ppr.find(qn("a:buChar")).get("char") == "•"


class TestTable:
    def lines(self, cell: Any) -> dict[str, bool]:
        from pptx.oxml.ns import qn

        tcpr = cell._tc.find(qn("a:tcPr"))
        return {
            side: tcpr.find(qn(f"a:{side}")).find(qn("a:solidFill")) is not None
            for side in ("lnL", "lnR", "lnT", "lnB")
        }

    def filled(self, cell: Any) -> bool:
        from pptx.oxml.ns import qn

        return cell._tc.find(qn("a:tcPr")).find(qn("a:solidFill")) is not None

    def test_the_header_row_has_no_lines_and_no_fill(self, sl: Any, deck: Any) -> None:
        t = sl.table(blank(sl, deck), 0.30, 1.24, 4, 3)
        for cell in t.rows[0].cells:
            assert not any(self.lines(cell).values()) and not self.filled(cell)

    def test_the_body_has_an_inner_grid_and_no_outer_border(self, sl: Any, deck: Any) -> None:
        t = sl.table(blank(sl, deck), 0.30, 1.24, 4, 3)
        assert self.lines(t.cell(1, 0)) == {"lnL": False, "lnR": True, "lnT": False, "lnB": True}
        assert self.lines(t.cell(2, 1)) == {"lnL": True, "lnR": True, "lnT": True, "lnB": True}
        assert self.lines(t.cell(3, 2)) == {"lnL": True, "lnR": False, "lnT": True, "lnB": False}

    def test_every_other_body_row_is_banded_from_the_first(self, sl: Any, deck: Any) -> None:
        t = sl.table(blank(sl, deck), 0.30, 1.24, 5, 2)
        assert [self.filled(t.cell(r, 0)) for r in range(5)] == [False, True, False, True, False]

    def test_the_cell_children_are_in_schema_order(self, sl: Any, deck: Any) -> None:
        from pptx.oxml.ns import qn

        t = sl.table(blank(sl, deck), 0.30, 1.24, 3, 2)
        tcpr = t.cell(1, 0)._tc.find(qn("a:tcPr"))
        assert children(tcpr) == ["lnL", "lnR", "lnT", "lnB", "solidFill"]


class TestShapes:
    def test_the_header_band_is_radial_with_a_gradient_outline(self, sl: Any, deck: Any) -> None:
        from pptx.oxml.ns import qn

        s = deck.table_slide("Results", [["Team", "After"], ["Risk", "5 days"]])
        band = next(sh for sh in s.shapes if sh.shape_type == 1)
        sppr = band._element.spPr
        assert children(sppr) == ["xfrm", "prstGeom", "gradFill", "ln"]
        assert sppr.find(qn("a:gradFill")).find(qn("a:path")).get("path") == "circle"
        assert sppr.find(qn("a:ln")).find(qn("a:gradFill")) is not None

    def test_a_link_keeps_the_text_colour(self, sl: Any, deck: Any) -> None:
        from pptx.oxml.ns import qn

        box = sl.text(blank(sl, deck, "navy"), 1, 1, 3, 1, "Write to ", 12, "#FFFFFF")
        r = sl.link(
            box.text_frame.paragraphs[0],
            "team@example.com",
            "mailto:team@example.com",
            12,
            "#FFFFFF",
        )
        assert r.hyperlink.address == "mailto:team@example.com"
        assert r.font.underline
        ext = r._r.find(qn("a:rPr")).find(qn("a:hlinkClick")).find(".//{*}hlinkClr")
        assert ext.get("val") == "tx"

    def test_a_picture_keeps_its_proportions_inside_its_box(
        self, sl: Any, deck: Any, tmp_path: Path
    ) -> None:
        from PIL import Image

        Image.new("RGB", (900, 300), "#cccccc").save(tmp_path / "chart.png")
        pic = sl.picture(blank(sl, deck), tmp_path / "chart.png", 0.29, 1.25, 9.39, 3.65)
        assert pic.width / pic.height == pytest.approx(3, rel=0.01)
        assert pic.width / EMU == pytest.approx(9.39, abs=0.01)
        assert pic.top / EMU > 1.25  # centred vertically in the taller box


def raw_layout(prs: Any, label: str) -> Any:
    """A layout of the template as opened, before inject(): its name still carries its tag."""
    return next(layout for layout in prs.slide_layouts if layout.name.startswith(f"{label} ["))


def layout_shapes(deck: Any, name: str) -> list[Any]:
    return list(deck.layouts[name].shapes)


def tagged(shapes: list[Any], label: str) -> Any:
    return next(sh for sh in shapes if sh.name.startswith(f"{label} ["))


class TestTheTemplate:
    """The layouts are the template's; every colour, font and background on them is the shared
    resources', written in when a Deck opens it."""

    def test_every_shape_of_every_layout_names_what_it_takes(self, sl: Any) -> None:
        from pptx import Presentation

        prs = Presentation(str(TEMPLATE))
        for layout in prs.slide_layouts:
            for shape in layout.shapes:
                _, tags = sl.parse_tag(shape.name, layout.name)
                assert tags, (layout.name, shape.name)
                assert not shape.is_placeholder or "slot" in tags, (layout.name, shape.name)

    def test_every_layout_the_module_starts_exists(self, deck: Any) -> None:
        source = (SCRIPTS / "slides.py").read_text()
        named = set(re.findall(r'self\._start\(\s*"([^"]+)"', source))
        assert named and named <= set(deck.layouts)

    def test_a_colour_typed_into_the_template_is_overwritten(self, sl: Any, tmp_path: Path) -> None:
        from pptx import Presentation
        from pptx.dml.color import RGBColor

        prs = Presentation(str(TEMPLATE))
        card = tagged(list(raw_layout(prs, "Two cards").shapes), "card 2")
        card.fill.solid()
        card.fill.fore_color.rgb = RGBColor(0xFF, 0, 0)
        sl.inject(prs)
        assert str(card.fill.fore_color.rgb) == PALETTE["surfaces"]["card_light"][1:]

    def test_fixed_text_takes_its_role(self, deck: Any) -> None:
        thanks = tagged(layout_shapes(deck, "Closing"), "thank you")
        r = thanks.text_frame.paragraphs[0].runs[0]
        role = TYPOGRAPHY["roles"]["closing_title"]
        assert r.font.size.pt == role["size"]
        assert r.font.name == TYPOGRAPHY["styles"][role["style"]]["family"]

    def test_the_theme_is_the_resources(self, deck: Any) -> None:
        from pptx.opc.constants import RELATIONSHIP_TYPE as RT
        from pptx.oxml import parse_xml

        theme = deck.prs.slide_master.part.part_related_by(RT.THEME)
        xml = parse_xml(theme.blob)
        ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
        accent1 = xml.find(".//a:clrScheme/a:accent1/a:srgbClr", ns).get("val")
        assert accent1 == PALETTE["primary"][1:]
        latin = xml.find(".//a:majorFont/a:latin", ns).get("typeface")
        assert latin == TYPOGRAPHY["styles"]["regular"]["family"]

    def test_every_layout_tags_its_background_and_a_deck_shows_the_label(
        self, sl: Any, deck: Any
    ) -> None:
        from pptx import Presentation

        raw = [layout.name for layout in Presentation(str(TEMPLATE)).slide_layouts]
        tags = [sl.parse_tag(name, "the template", sl.LAYOUT_KEYS) for name in raw]
        assert [label for label, _ in tags] == list(deck.layouts)  # the tag is gone in a deck
        for label, tag in tags:
            assert tag["background"] in BACKGROUNDS, label
            _, _, variant = label.partition(" · ")
            if variant and variant != "dark":  # the name says the background it has
                assert tag["background"] == variant, label
        assert {tag["background"] for _, tag in tags} == set(BACKGROUNDS)

    def test_a_background_is_the_layout_s_own_fill(self, deck: Any) -> None:
        """What Format Background sets: a flat colour or the palette's gradient, never a shape."""
        from pptx.oxml.ns import qn

        def fill(layout: Any) -> Any:
            return layout._element.find(qn("p:cSld")).find(qn("p:bg")).find(qn("p:bgPr"))[0]

        white = fill(deck.layouts["Content"])
        assert white.tag == qn("a:solidFill")
        assert white[0].get("val") == BACKGROUNDS["white"][1:]
        for name, background in (("Title", "blue"), ("Text · dark", "navy")):
            gradient = fill(deck.layouts[name])
            assert gradient.tag == qn("a:gradFill"), name
            stops = [gs[0].get("val") for gs in gradient.find(qn("a:gsLst"))]
            assert stops == [st["color"][1:] for st in BACKGROUNDS[background]["stops"]], name
            assert not gradient.findall(".//" + qn("a:alpha")), name
            assert (gradient.find(qn("a:path")) is not None) == (background == "navy"), name
        for name, layout in deck.layouts.items():
            assert not [sh for sh in layout.shapes if sh.name.startswith("background")], name

    @pytest.mark.parametrize(
        ("name", "message"),
        [
            ("Content · navy", "'Content · navy' names nothing it takes from the resources"),
            ("Content · navy [background=midnight]", "no background 'midnight' in palette.json"),
            ("Content · navy [fill=navy]", "unknown key 'fill'"),
            ("Content · navy []", "names no background"),
        ],
    )
    def test_a_layout_tag_the_palette_cannot_meet_is_refused(
        self, sl: Any, name: str, message: str
    ) -> None:
        from pptx import Presentation

        prs = Presentation(str(TEMPLATE))
        prs.slide_layouts[0].name = name
        with pytest.raises(ValueError, match=re.escape(message)):
            sl.inject(prs)

    def test_no_layout_has_a_picture(self, deck: Any) -> None:
        for name, layout in deck.layouts.items():
            assert not [sh for sh in layout.shapes if sh.shape_type == 13], name

    def test_a_template_s_xml_entities_are_never_resolved(self, sl: Any, tmp_path: Path) -> None:
        """The theme is parsed to be rewritten: an external entity planted in it (XXE) must
        not pull a local file into the deck."""
        from pptx import Presentation
        from pptx.opc.constants import RELATIONSHIP_TYPE as RT

        secret = tmp_path / "secret.txt"
        secret.write_text("the secret")
        prs = Presentation(str(TEMPLATE))
        theme = prs.slide_master.part.part_related_by(RT.THEME)
        xml = theme.blob.decode()
        start = xml.index("<a:theme")
        doctype = f'<!DOCTYPE a:theme [<!ENTITY e SYSTEM "{secret.as_uri()}">]>'
        planted = (
            xml[:start] + doctype + xml[start:].replace("</a:theme>", "<a:x>&e;</a:x></a:theme>")
        )
        theme._blob = planted.encode()
        sl.inject(prs)
        assert b"the secret" not in theme.blob

    def test_a_key_the_palette_does_not_have_is_refused(self, sl: Any) -> None:
        from pptx import Presentation

        prs = Presentation(str(TEMPLATE))
        rule = tagged(list(raw_layout(prs, "Content").shapes), "title rule")
        rule.name = "title rule [line=lines.no_such_line]"
        with pytest.raises(
            ValueError, match=r"'lines\.no_such_line', which palette\.json does not have"
        ):
            sl.inject(prs)

    def test_an_accent_as_text_on_a_blue_background_is_refused(self, sl: Any) -> None:
        from pptx import Presentation

        prs = Presentation(str(TEMPLATE))
        shape = tagged(list(raw_layout(prs, "Closing").shapes), "explore more")
        shape.name = "explore more [slot=explore more; role=explore_more; color=tertiary]"
        with pytest.raises(ValueError, match="blue on blue"):
            sl.inject(prs)

    def test_the_closing_slide_has_no_links(self, deck: Any) -> None:
        from lxml import etree

        closing = deck.closing_slide(contacts=["Write to team@example.com"])
        assert b"hlinkClick" not in etree.tostring(closing._element)
        text = " ".join(sh.text_frame.text for sh in closing.shapes if sh.has_text_frame)
        assert "Explore more" in text and "team@example.com" in text

    def test_without_contacts_the_closing_slide_only_thanks(self, deck: Any) -> None:
        """No contacts are invented: without them there is no "Explore more" either."""
        closing = deck.closing_slide()
        assert not list(closing.placeholders)

    def test_dark_slides_map_the_link_colour_to_white(self, deck: Any) -> None:
        from pptx.oxml.ns import qn

        def mapping(element: Any) -> Any:
            return element.find(qn("p:clrMapOvr")).find(qn("a:overrideClrMapping"))

        closing = deck.closing_slide()
        for element in (closing.slide_layout._element, closing._element):
            assert mapping(element).get("hlink") == mapping(element).get("folHlink") == "lt1"
        content = deck.content_slide("A title")
        assert mapping(content._element) is None

    def test_an_untagged_shape_is_refused(self, sl: Any) -> None:
        from pptx import Presentation

        prs = Presentation(str(TEMPLATE))
        tagged(list(raw_layout(prs, "Content").shapes), "title rule").name = "Line 3"
        with pytest.raises(ValueError, match="'Line 3' names nothing it takes from the resources"):
            sl.inject(prs)

    def test_the_slides_are_slides_of_their_layouts(self, deck: Any) -> None:
        """Placeholders, so the outline lists the titles and "Reset slide" works."""
        s = deck.two_cards(
            "Claims before and after", [("Before", BODY, "Cost"), ("After", BODY, "Gain")]
        )
        assert s.slide_layout.name == "Two cards"
        assert s.shapes.title.text == "Claims before and after"
        assert all(sh.is_placeholder for sh in s.shapes)

    def test_no_placeholder_inherits_the_master_s_bullets(self, deck: Any) -> None:
        from pptx.oxml.ns import qn

        s = deck.agenda("Ready for scale", "Three pilots.", [("Where we are", "The pilots")])
        for ph in s.placeholders:
            for p in ph.text_frame.paragraphs:
                ppr = p._p.find(qn("a:pPr"))
                assert ppr.get("marL") == "0" and ppr.find(qn("a:buNone")) is not None, ph.name

    def test_unused_placeholders_are_removed(self, deck: Any) -> None:
        s = deck.agenda("Ready for scale", "Three pilots.", [("Where we are", "The pilots")])
        assert sorted(ph.name for ph in s.placeholders) == sorted(
            ["title", "takeaway", "summary", "number 1", "topic 1", "description 1"]
        )


# --- the layouts: every position as the template gives it -------------------------------------


class TestLayouts:
    def test_a_content_slide_has_its_fixed_elements_in_place(self, deck: Any) -> None:
        s = deck.content_slide("Pilot results", intro="What changed in the quarter")
        b = boxes(s)
        assert b["Pilot results"][:3] == (0.29, 0.22, 7.65)
        assert b["What changed in the quarter"][:3] == (0.29, 0.94, 9.18)
        layout = layout_shapes(deck, "Content")
        number = tagged(layout, "slide number")
        assert number.text_frame.paragraphs[0]._p.find(".//{*}fld").get("type") == "slidenum"
        assert len(s.shapes) == 2  # title and intro; the rule and number are the layout's

    def test_two_cards(self, deck: Any) -> None:
        s = deck.two_cards(
            "Claims before and after",
            [("Before: manual", BODY, "High cost"), ("After: assisted", BODY, "70% faster")],
        )
        assert cards(s.slide_layout.shapes) == [(0.29, 1.57, 4.44, 3.29), (4.96, 1.57, 4.44, 3.29)]
        b = boxes(s)
        assert b["Before: manual"] == (0.48, 2.2, 3.84, 0.34)
        assert b["After: assisted"][:2] == (5.15, 2.2)
        assert b["High cost"][:2] == (0.48, 3.8)

    def test_the_heading_that_collided_on_slide_11_is_refused(self, deck: Any) -> None:
        """A two-card heading that wraps would print over the body: refused when written, with
        the words that fit."""
        deck.two_cards(
            "Claims before and after",
            [
                ("Before: Manual, slow, inconsistent", BODY, "High cost"),
                ("After: AI-augmented, consistent, auditable", BODY, "70% faster"),
            ],
        )
        assert len(deck.problems) == 1
        assert deck.problems[0].startswith("slide 1 (two_cards), heading 2:")
        assert "what fits is 'After: AI-augmented, consistent,' (32 characters)" in deck.problems[0]

    @pytest.mark.parametrize("dark", [False, True])
    def test_three_cards(self, deck: Any, dark: bool) -> None:
        items = [(n, "From your sources", ["Sourced", "Fast"]) for n in ("Answer", "Draft", "Act")]
        s = deck.three_cards("Three ways to start", items, dark=dark)
        assert s.slide_layout.name == ("Three cards · dark" if dark else "Three cards")
        found = cards(s.slide_layout.shapes)
        assert [c[0] for c in found] == [0.29, 3.52, 6.76]
        assert {c[1:] for c in found} == {(1.29, 2.92, 3.5)}
        assert boxes(s)["Answer"][:2] == (0.52, 2.0)

    def test_four_cards(self, deck: Any) -> None:
        items = [("Act", "With approval", ["Workflows"])] * 4
        s = deck.four_cards("Four capabilities", items)
        found = cards(s.slide_layout.shapes)
        assert [c[0] for c in found] == [0.29, 2.69, 5.09, 7.48]
        assert {c[1:] for c in found} == {(1.29, 2.2, 3.43)}

    def test_table_slide(self, deck: Any) -> None:
        s = deck.table_slide("Results by team", [["Team", "After"], ["Risk", "5 days"]])
        assert cards(s.shapes)[0] == (0.29, 1.24, 2.66, 0.4)  # the header band, at the table area
        t = next(sh for sh in s.shapes if sh.has_table).table
        header = t.cell(0, 0).text_frame.paragraphs[0].runs[0]
        assert str(header.font.color.rgb) == PALETTE["primary"][1:]
        assert header.font.size.pt == TYPOGRAPHY["roles"]["table_header"]["size"]

    def test_agenda(self, deck: Any) -> None:
        topics = [("Where we are", "The pilots"), ("What changed", "Onboarding"), ("Next", "Q4")]
        s = deck.agenda("Ready for scale", "Three pilots.", topics)
        b = boxes(s)
        assert [b[n][:2] for n in ("01", "02", "03")] == [(4.28, 1.13), (4.28, 1.8), (4.28, 2.48)]
        separators = [sh for sh in s.shapes if sh.shape_type == 9]
        assert len(separators) == len(topics) - 1

    def test_text_slide(self, deck: Any) -> None:
        s = deck.text_slide("Agents cut onboarding by 40%", ["One", "Two"], intro="Measured")
        assert s.slide_layout.name == "Text with intro"
        assert boxes(s)["One\nTwo"] == (0.29, 1.57, 9.39, 3.33)

    def test_a_background_the_template_has_no_layout_for_is_refused(self, deck: Any) -> None:
        with pytest.raises(ValueError, match="no layout 'Content · midnight'"):
            deck.content_slide("Pilot results", background="midnight")

    def test_the_spec_makes_plain_white_the_default(self) -> None:
        for layout in ("Two cards", "Three cards", "Four cards", "Table", "Text"):
            assert f"{layout} (plain white)" in SKILL_MD, layout


class TestCapacities:
    """What a layout cannot take is recorded with its slide, and the deck is not saved."""

    @pytest.mark.parametrize(
        ("build", "message"),
        [
            (lambda d: d.two_cards("T", [("h", "b", "t")] * 3), "two_cards takes 2 cards, not 3"),
            (
                lambda d: d.four_cards("T", [("t", "s", ["b"])] * 5),
                "four_cards takes 4 cards, not 5",
            ),
            (
                lambda d: d.three_cards(
                    "T", [("Answer", "s", ["b"] * 5)] + [("t", "s", ["b"])] * 2
                ),
                "5 bullets in 'Answer', the layout holds 4",
            ),
            (lambda d: d.text_slide("T", ["point"] * 7), "7 bullets, the layout holds 6"),
            (lambda d: d.table_slide("T", [["c"] * 7] * 2), "7 columns, the layout holds 6"),
            (lambda d: d.table_slide("T", [["c", "d"]] * 8), "8 rows, the layout holds 7"),
            (lambda d: d.agenda("T", "S", [("a", "b")] * 7), "7 topics, the layout holds 6"),
        ],
    )
    def test_it_is_recorded_and_the_deck_not_saved(
        self, deck: Any, tmp_path: Path, build: Any, message: str
    ) -> None:
        build(deck)
        assert len(deck.problems) == 1 and message in deck.problems[0]
        with pytest.raises(RuntimeError, match="nothing was saved"):
            deck.save(tmp_path / "deck.pptx", render=False)
        assert not list(tmp_path.iterdir())


class TestOneRunReportsEverything:
    """A build that stopped at the first problem took one run per problem: the agent shortened
    one text, ran again, met the next. Now one run lists them all, each with what fits."""

    def test_every_slide_s_problems_in_one_run(self, deck: Any, tmp_path: Path) -> None:
        deck.agenda(
            "Agents are ready for regulated work", "Three pilots.", [("Where", "The pilots")]
        )
        deck.text_slide("Fine", ["A point"])
        deck.two_cards(
            "Claims",
            [
                ("Before: Manual, slow, inconsistent", BODY, "Cost"),
                ("After: AI-augmented, consistent, auditable", BODY, "Gain"),
            ],
        )
        deck.three_cards("Pillars", [("A", "s", ["b"] * 5), ("B", "s", ["b"]), ("C", "s", ["b"])])
        with pytest.raises(RuntimeError) as raised:
            deck.save(tmp_path / "deck.pptx", render=False)
        report = str(raised.value)
        assert "3 slide(s) do not fit" in report
        for where in (
            "slide 1 (agenda), takeaway",
            "slide 3 (two_cards), heading 2",
            "slide 4 (three_cards)",
        ):
            assert where in report

    def test_what_fits_is_exact(self, sl: Any, deck: Any) -> None:
        """The words it names fit the box, and one more word does not: a length to rewrite to,
        not an estimate."""
        text = "Agents are ready for regulated work in every team"
        deck.agenda(text, "Three pilots.", [("Where", "The pilots")])
        fits = re.search(r"what fits is '([^']+)'", deck.problems[0]).group(1)
        s = deck.prs.slides[-1]
        box = next(ph for ph in s.placeholders if ph.name == "takeaway")
        run = box.text_frame.paragraphs[0].runs[0]
        run.text = fits
        assert sl.check_pptx.fit(box).ok
        run.text = " ".join(text.split()[: len(fits.split()) + 1])
        assert not sl.check_pptx.fit(box).ok

    def test_many_lines_say_how_many_fit(self, deck: Any) -> None:
        deck.three_cards("Pillars", [("A", "s", [BODY] * 4), ("B", "s", ["b"]), ("C", "s", ["b"])])
        assert re.search(
            r"bullets 1: .* holds about \d+ lines at this size and the text takes about \d+",
            deck.problems[0],
        )

    def test_content_slide_still_returns_its_slide(self, deck: Any) -> None:
        s = deck.content_slide("A title far too long to fit on the one line a slide title has here")
        assert (
            s is not None and deck.problems and "slide 1 (content_slide), title" in deck.problems[0]
        )


# --- saving: only a deck that passes the check gets its name ----------------------------------


class TestSave:
    def test_a_clean_deck_gets_its_name(
        self, deck: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        deck.content_slide("Pilot results by team")
        path = deck.save(tmp_path / "deck.pptx", render=False)
        assert path.exists() and not (tmp_path / "deck.unchecked.pptx").exists()
        assert f"{tmp_path / 'deck.pptx'}: no collisions" in capsys.readouterr().out

    def test_a_colliding_deck_stays_unchecked(
        self, sl: Any, deck: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "deck.pptx").write_bytes(b"an earlier build")
        monkeypatch.setattr(sl, "fits", lambda box: box)  # let it be written; saving refuses it
        deck.content_slide("Agents cut onboarding time by forty percent across every team")
        with pytest.raises(RuntimeError, match="NOT deliverable") as raised:
            deck.save(tmp_path / "deck.pptx", render=False)
        assert "text crossed by a line" in str(raised.value)  # the layout's title rule
        assert not (tmp_path / "deck.pptx").exists()  # the earlier build is gone too
        assert (tmp_path / "deck.unchecked.pptx").exists()

    def test_a_deck_with_every_layout_at_capacity_passes(self, deck: Any, tmp_path: Path) -> None:
        deck.title_slide("Agents in production", "Board · October 2026")
        deck.agenda(
            "Ready for scale",
            "Three pilots, one platform, a plan for Q4.",
            [(f"Topic {i}", "The pilots and their results") for i in range(1, 7)],
        )
        deck.divider_slide("Where agents pay off", highlight="agents")
        deck.text_slide("Agents cut onboarding by 40%", [(BODY + " " + BODY)[:150]] * 6)
        deck.two_cards(
            "Claims before and after",
            [("Before: manual and slow", BODY + " " + BODY[:60], "High cost, high risk")] * 2,
        )
        deck.three_cards(
            "Three ways to start", [("Answer", "From your sources", ["A", "B", "C", "D"])] * 3
        )
        deck.four_cards(
            "Four capabilities", [("Act", "With approval", ["A", "B", "C"])] * 4, dark=True
        )
        deck.table_slide(
            "Results by team", [["Team", "Before", "After"]] + [["Risk", "9", "5"]] * 6
        )
        deck.closing_slide(contacts=["Write to team@example.com", "example.com/agents"])
        assert deck.save(tmp_path / "deck.pptx", render=False).exists()

    @pytest.mark.parametrize("background", ["navy", "violet"])
    def test_every_alternative_title_passes(
        self, deck: Any, tmp_path: Path, background: str
    ) -> None:
        deck.title_slide("Agents in production", "Board · October 2026", background=background)
        assert deck.save(tmp_path / "deck.pptx", render=False).exists()

    def test_every_divider_and_content_background_passes(self, deck: Any, tmp_path: Path) -> None:
        for name in deck.layouts:
            kind, _, background = name.partition(" · ")
            if background not in BACKGROUNDS:  # the default, or "· dark": a variant
                continue
            if kind == "Section divider":
                deck.divider_slide("Where agents pay off", background=background)
            elif kind == "Content":
                deck.content_slide("Pilot results by team", background=background, intro="By team")
        assert deck.save(tmp_path / "deck.pptx", render=False).exists()


# --- the collision checker, on decks built to collide -----------------------------------------


@pytest.fixture
def checker(sl: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    # These decks collide on purpose, which the helpers refuse to write: the measure they
    # apply is the checker's own (TestCapacities, TestLayouts).
    monkeypatch.setattr(sl, "fits", lambda box: box)
    return sl.check_pptx


def kinds(deck: Any, checker: Any, tmp_path: Path) -> list[str]:
    deck.prs.save(tmp_path / "deck.pptx")
    return sorted({f.kind for f in checker.check(tmp_path / "deck.pptx").findings})


class TestTheCollisionCheckerFinds:
    def test_a_title_wrapped_onto_its_rule(self, deck: Any, checker: Any, tmp_path: Path) -> None:
        deck.content_slide("Agents cut onboarding time by forty percent across every team")
        assert kinds(deck, checker, tmp_path) == [
            "text crossed by a line",
            "text overflowing its box",
        ]

    def test_two_boxes_on_each_other(
        self, sl: Any, deck: Any, checker: Any, tmp_path: Path
    ) -> None:
        s = deck.content_slide("Pilot results by team")
        sl.text(s, 0.29, 1.6, 4, 1, BODY, 11, "#000000")
        sl.text(s, 0.29, 1.65, 4, 1, "A second paragraph", 11, "#000000")
        assert kinds(deck, checker, tmp_path) == ["text on text"]

    def test_text_on_a_picture(self, sl: Any, deck: Any, checker: Any, tmp_path: Path) -> None:
        from PIL import Image

        Image.new("RGB", (900, 300), "#cccccc").save(tmp_path / "chart.png")
        s = deck.content_slide("Pilot results by team")
        sl.picture(s, tmp_path / "chart.png", 0.29, 1.6, 6.0, 2.0)
        sl.text(s, 0.5, 2.2, 3.0, 0.3, "A label on the chart", 11, "#000000")
        assert kinds(deck, checker, tmp_path) == ["text on a picture"]

    def test_text_out_of_the_layout_s_card(
        self, sl: Any, deck: Any, checker: Any, tmp_path: Path
    ) -> None:
        """The cards are the layout's, not the slide's: text across one's edge is still found."""
        s = deck.two_cards("Claims", [("Before", BODY, "Cost"), ("After", BODY, "Gain")])
        sl.text(s, 4.0, 4.5, 1.6, 0.2, "Across the card's edge", 9, "#000000")
        assert "text out of its card" in kinds(deck, checker, tmp_path)

    def test_text_out_of_its_card(self, sl: Any, deck: Any, checker: Any, tmp_path: Path) -> None:
        s = deck.content_slide("Pilot results by team")
        sl.card(s, 0.29, 1.57, 2.0, 0.5, 0.03, fill="#FFFFFF")
        sl.text(s, 0.48, 1.65, 1.6, 0.3, "Short", 11, "#000000")
        sl.text(s, 0.48, 1.95, 3.0, 0.3, "Longer than the card it sits on", 11, "#000000")
        assert kinds(deck, checker, tmp_path) == ["text out of its card"]

    def test_text_under_a_card(self, sl: Any, deck: Any, checker: Any, tmp_path: Path) -> None:
        s = deck.content_slide("Pilot results by team")
        sl.text(s, 0.48, 1.65, 1.6, 0.3, "Hidden", 11, "#000000")
        sl.card(s, 0.29, 1.57, 2.0, 0.5, 0.03, fill="#FFFFFF")
        assert kinds(deck, checker, tmp_path) == ["text under a shape"]

    def test_a_table_row_that_grows(self, sl: Any, deck: Any, checker: Any, tmp_path: Path) -> None:
        s = deck.content_slide("Pilot results by team")
        t = sl.table(s, 0.30, 1.24, 3, 3)
        sl.run(t.cell(1, 1).text_frame.paragraphs[0], BODY, 10, "#000000")
        assert kinds(deck, checker, tmp_path) == ["table row grown"]

    def test_text_past_the_safe_area(
        self, sl: Any, deck: Any, checker: Any, tmp_path: Path
    ) -> None:
        s = deck.content_slide("Pilot results by team")
        sl.text(s, 0.05, 2.0, 2.0, 0.3, "Too far left", 11, "#000000")
        assert kinds(deck, checker, tmp_path) == ["text outside the safe area"]


class TestTheCollisionCheckerLeavesAlone:
    def test_the_closing_slide(self, deck: Any, checker: Any, tmp_path: Path) -> None:
        deck.closing_slide()
        deck.closing_slide(contacts=["Write to team@example.com"])
        assert kinds(deck, checker, tmp_path) == []

    def test_text_on_a_coloured_background(self, deck: Any, checker: Any, tmp_path: Path) -> None:
        """A background is a full-slide shape: all text is on it, none out of it."""
        for background in BACKGROUNDS:
            deck.content_slide("Pilot results by team", background=background, intro="By team")
        assert kinds(deck, checker, tmp_path) == []
