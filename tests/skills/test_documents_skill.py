"""The documents skill (`skills/builtin/productivity/documents/`): a SKILL.md and the
module it builds PDF documents with, `scripts/pdf_layout.py`, which takes every colour, background
and font from the shared resources (`skills/builtin/resources/`).

What can break: the module reaching for a resource that is not shipped, a colour of its own
instead of the palette's, the spec naming a block or a background the module does not have,
and the build itself (a heading stranded at the bottom of a page, a table that does not
split, fonts not embedded, a picture of its own: a logo or a background image).

The build tests need reportlab, which the test environment does not install; they are
skipped without it: `uv run --with reportlab --with pymupdf pytest tests/skills/test_documents_skill.py`.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

import domyn.agents
from domyn.agents.interfaces.cli.config import builtin_skill_dirs
from domyn.agents.skills import LocalDirSkillSource, SkillQuery

BUILTIN = Path(domyn.agents.__file__).parent / "skills" / "builtin"
SKILL = BUILTIN / "productivity" / "documents"
SKILL_MD = (SKILL / "SKILL.md").read_text()
MODULE = SKILL / "scripts" / "pdf_layout.py"
LAYOUT_FILE = SKILL / "scripts" / "pdf_layout.json"
LAYOUT = json.loads(LAYOUT_FILE.read_text())
RESOURCES = BUILTIN / "resources"
PALETTE = json.loads((RESOURCES / "palette.json").read_text())
TYPOGRAPHY = json.loads((RESOURCES / "typography.json").read_text())
BACKGROUNDS = PALETTE["backgrounds"]


class TestTheSkill:
    @pytest.mark.asyncio
    async def test_it_is_indexed_with_the_builtins(self) -> None:
        index = await LocalDirSkillSource(paths=builtin_skill_dirs()).index(SkillQuery(task=""))
        skill = next(s for s in index.skills if s.name == "documents-style")
        assert skill.path is not None and Path(skill.path).resolve() == SKILL / "SKILL.md"

    def test_the_module_finds_the_shared_resources_from_its_place(self) -> None:
        text = MODULE.read_text()
        assert 'Path(__file__).resolve().parents[3] / "resources"' in text
        assert MODULE.resolve().parents[3] / "resources" == BUILTIN / "resources"

    def test_the_module_has_no_colour_of_its_own(self) -> None:
        """Every colour is read from the palette by key, as seaborn_style does."""
        assert not re.findall(r"#[0-9A-Fa-f]{6}\b", MODULE.read_text())

    def test_it_names_no_picture_of_its_own(self) -> None:
        names = set(re.findall(r"[\w-]+\.(?:jpg|png)", MODULE.read_text() + SKILL_MD))
        assert names <= {"onboarding.png"}
        assert "logo" not in MODULE.read_text().lower()

    def test_the_layout_names_only_palette_keys_never_hex(self) -> None:
        """Colours in the layout are dotted palette keys: every one resolves, and none is hex."""
        assert not re.findall(r"#[0-9A-Fa-f]{6}\b", LAYOUT_FILE.read_text())

        def keys(node: object, field: str = "") -> list[str]:
            if isinstance(node, dict):
                return [k for f, v in node.items() for k in keys(v, f)]
            if isinstance(node, list):
                return []
            colour_field = field in {"color", "fill", "text", "outline", "plain"} or field.endswith(
                ("_color", "_fill", "_outline")
            )
            return [node] if colour_field and isinstance(node, str) else []

        found = keys(LAYOUT)
        assert len(found) > 20
        for key in found:
            node: object = PALETTE
            for part in key.split("."):
                assert isinstance(node, dict) and part in node, f"{key!r} is not in the palette"
                node = node[part]

    def test_every_background_the_layout_names_is_the_palette_s(self) -> None:
        named = LAYOUT["backgrounds"]
        for name in [*named["cover"], *named["content"]]:
            assert name in BACKGROUNDS, name
            assert f"`{name}`" in SKILL_MD, name
        assert named["default_cover"] in named["cover"]
        assert named["default_content"] in named["content"]

    def test_every_text_style_has_its_type_from_the_typography(self) -> None:
        """A shared element names a role, so its size is the slides'; only what a PDF alone has
        (the cover, h3) sets its own, in a style the typography ships."""
        for name, spec in LAYOUT["text"].items():
            if "role" in spec:
                assert spec["role"] in TYPOGRAPHY["roles"], name
                assert "size" not in spec and "font" not in spec, name
            else:
                assert TYPOGRAPHY["styles"][spec["font"]]["file"], name
        assert "fonts" not in LAYOUT

    def test_every_leading_holds_its_size(self) -> None:
        for name, spec in LAYOUT["text"].items():
            size = TYPOGRAPHY["roles"][spec["role"]]["size"] if "role" in spec else spec["size"]
            assert spec.get("leading", size) >= size, name

    def test_every_block_the_spec_names_is_exported(self) -> None:
        exported = re.search(r"__all__ = \[(.*?)\]", MODULE.read_text(), re.S)
        assert exported
        names = set(re.findall(r'"(\w+)"', exported.group(1)))
        named = set(re.findall(r"\| `(\w+)\(", SKILL_MD))
        assert named and named <= names


# -- the module, run ---------------------------------------------------------------------


@pytest.fixture
def d() -> Any:
    pytest.importorskip("reportlab")
    sys.path.insert(0, str(MODULE.parent))
    try:
        import pdf_layout
    finally:
        sys.path.remove(str(MODULE.parent))
    return pdf_layout


@pytest.fixture
def pdf() -> Any:
    return pytest.importorskip("pymupdf")


def build(d: Any, tmp_path: Path, story: list[Any], **kw: Any) -> Path:
    kw.setdefault("title", "A report")
    return d.build(tmp_path / "doc.pdf", story, **kw)


class TestTheBlocks:
    def test_a_typed_mark_is_removed_and_a_minus_kept(self, d: Any) -> None:
        items = d.bullets(["• one", "- two", "-5% churn"])
        assert [i._flowables[0].text for i in items._flowables] == ["one", "two", "-5% churn"]

    def test_a_typed_number_is_removed(self, d: Any) -> None:
        items = d.numbered(["1. one", "2) two", "2026 was good"])
        assert [i._flowables[0].text for i in items._flowables] == ["one", "two", "2026 was good"]

    def test_an_unknown_callout_or_background_is_refused(self, d: Any, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="highlight"):
            d.callout("x", kind="loud")
        with pytest.raises(ValueError, match="cover background"):
            build(d, tmp_path, [d.p("x")], cover="white")
        with pytest.raises(ValueError, match="page background"):
            build(d, tmp_path, [d.p("x")], page_background="blue")


class TestTheDocument:
    def test_every_block_builds_and_the_fonts_are_embedded(
        self, d: Any, pdf: Any, tmp_path: Path
    ) -> None:
        story = [
            d.h1("Summary"),
            d.intro("The main finding."),
            d.p("Body with <b>bold</b>, <i>italic</i> and " + d.escape("R&D <costs>") + "."),
            d.bullets(["one", "two"]),
            d.numbered(["first", "second"]),
            d.callout("Key finding"),
            d.callout("A note", kind="light"),
            d.h2("Detail"),
            d.h3("More"),
            d.table([["a", "b"], ["1", "2"]], align=["left", "right"]),
        ]
        path = build(d, tmp_path, story, subtitle="Q3", closing=True)
        doc = pdf.open(path)
        assert doc.page_count == 3  # cover, content, closing
        fonts = {f[3].split("+")[-1] for page in doc for f in page.get_fonts()}
        embedded = {  # a font's file is named after it (DMSans-Light.ttf is DMSans-Light)
            key: Path(TYPOGRAPHY["styles"][key]["file"]).stem
            for key in ("regular", "bold", "title", "light")
        }
        assert {embedded["regular"], embedded["bold"], embedded["title"]} <= fonts
        assert embedded["light"] in fonts  # "Thank you"
        assert "R&D <costs>" in doc[1].get_text()
        assert doc.metadata["title"] == "A report"

    def test_a_long_table_repeats_its_header_on_every_page(
        self, d: Any, pdf: Any, tmp_path: Path
    ) -> None:
        rows = [["Team", "Score"]] + [[f"team {i}", str(i)] for i in range(80)]
        doc = pdf.open(build(d, tmp_path, [d.table(rows)]))
        pages = [doc[i].get_text() for i in range(1, doc.page_count)]
        assert len(pages) >= 2
        assert all("Team" in text for text in pages)
        assert "team 79" in pages[-1]

    @pytest.mark.parametrize(("left", "moves"), [(100, True), (300, False)])
    def test_a_heading_does_not_end_a_page(
        self, d: Any, pdf: Any, tmp_path: Path, left: float, moves: bool
    ) -> None:
        """Fill the first content page to `left` points above its bottom, then a section: with
        too little room the heading opens the next page, with enough it stays."""
        from reportlab.lib.pagesizes import A4

        margins = d.LAYOUT["page"]["margins"]
        frame = A4[1] - margins["top"] - margins["bottom"]
        story = [d.spacer(frame - left), d.h1("Next section"), d.p("Its text.")]
        doc = pdf.open(build(d, tmp_path, story))
        page = next(i for i in range(doc.page_count) if "Next section" in doc[i].get_text())
        assert page == (2 if moves else 1)
        assert "Its text." in doc[page].get_text()

    @pytest.mark.parametrize(("cover", "page"), [("blue", "white"), ("violet", "tint")])
    def test_no_picture_of_its_own_and_the_backgrounds_are_the_palette_s(
        self, d: Any, pdf: Any, tmp_path: Path, cover: str, page: str
    ) -> None:
        story = [d.p("page"), d.page_break(), d.p("page")]
        path = build(d, tmp_path, story, cover=cover, page_background=page, closing=True)
        doc = pdf.open(path)
        assert not [x for p in doc for x in p.get_images()]  # no logo, no background image
        colours = [cover, page, page, cover]
        for i, name in enumerate(colours):
            pix = doc[i].get_pixmap(dpi=10)
            spec = BACKGROUNDS[name]  # a gradient starts at the top-left corner
            want = bytes.fromhex((spec if isinstance(spec, str) else spec["stops"][0]["color"])[1:])
            # within one step a channel: the colour is drawn from a float and rounded back
            assert all(abs(a - b) <= 1 for a, b in zip(pix.pixel(1, 1), want, strict=True)), i
        assert path.stat().st_size < 200_000

    def test_a_linear_gradient_runs_corner_to_corner_as_on_a_slide(
        self, d: Any, pdf: Any, tmp_path: Path
    ) -> None:
        """Along its angle the gradient spans the page: the far corner is its last colour."""
        doc = pdf.open(build(d, tmp_path, [d.p("x")], cover="blue"))
        pix = doc[0].get_pixmap(dpi=10)
        last = bytes.fromhex(BACKGROUNDS["blue"]["stops"][-1]["color"][1:])
        corner = pix.pixel(pix.width - 2, pix.height - 2)
        # sampled a few points in from the corner, where it has not quite reached the end
        assert all(abs(a - b) <= 8 for a, b in zip(corner, last, strict=True)), corner

    def test_the_closing_page_shows_contacts_only_when_given(
        self, d: Any, pdf: Any, tmp_path: Path
    ) -> None:
        bare = pdf.open(build(d, tmp_path, [d.p("x")], closing=True))[-1].get_text()
        assert "Thank you" in bare and "Explore more" not in bare
        contacts = ['Write to <u><a href="mailto:team@example.com">team@example.com</a></u>']
        path = tmp_path / "with contacts.pdf"
        d.build(path, [d.p("x")], title="A report", closing=True, contacts=contacts)
        page = pdf.open(path)[-1]
        assert "Explore more" in page.get_text() and "team@example.com" in page.get_text()
        assert [link["uri"] for link in page.get_links()] == ["mailto:team@example.com"]


# -- the collision checker ---------------------------------------------------------------

CHECKER = SKILL / "scripts" / "check_pdf.py"


@pytest.fixture
def checker(pdf: Any) -> Any:
    sys.path.insert(0, str(CHECKER.parent))
    try:
        import check_pdf
    finally:
        sys.path.remove(str(CHECKER.parent))
    return check_pdf


@pytest.fixture
def page(tmp_path: Path) -> Any:
    """A blank A4 canvas; draw on it, then `.save()` and check `tmp_path / "page.pdf"`."""
    canvas = pytest.importorskip("reportlab.pdfgen.canvas")
    from reportlab.lib.pagesizes import A4

    c = canvas.Canvas(str(tmp_path / "page.pdf"), pagesize=A4)
    c.setFont("Helvetica", 11)
    return c


def kinds(
    checker: Any, path: Path, margins: tuple[float, float] | None = (56.7, 56.7)
) -> list[str]:
    return [c.kind for c in checker.check(path, margins)]


class TestTheCheckerFinds:
    def test_text_printed_over_text(self, checker: Any, page: Any, tmp_path: Path) -> None:
        page.drawString(100, 700, "The first line of a paragraph")
        page.drawString(100, 694, "a second line printed over it")
        page.save()
        assert kinds(checker, tmp_path / "page.pdf") == ["text on text"]

    def test_a_number_run_into_its_text(self, checker: Any, page: Any, tmp_path: Path) -> None:
        page.drawString(100, 700, "10")
        page.drawString(108, 700, "Step ten")
        page.save()
        assert kinds(checker, tmp_path / "page.pdf") == ["text on text"]

    def test_text_on_a_chart(self, checker: Any, page: Any, tmp_path: Path) -> None:
        from PIL import Image

        Image.new("RGB", (200, 100), "#cccccc").save(tmp_path / "chart.png")
        page.drawImage(str(tmp_path / "chart.png"), 100, 600, 200, 100)
        page.drawString(120, 650, "a label on the chart")
        page.save()
        assert kinds(checker, tmp_path / "page.pdf") == ["text on an image"]

    def test_a_line_through_text(self, checker: Any, page: Any, tmp_path: Path) -> None:
        page.drawString(100, 700, "a cell that overflowed")
        page.line(150, 690, 150, 715)  # a column's grid line
        page.save()
        assert kinds(checker, tmp_path / "page.pdf") == ["text crossed by a line"]

    def test_text_off_the_page(self, checker: Any, page: Any, tmp_path: Path) -> None:
        page.drawString(-30, 700, "lost at the edge")
        page.save()
        assert kinds(checker, tmp_path / "page.pdf") == ["text off the page"]

    def test_text_past_the_margin(self, checker: Any, page: Any, tmp_path: Path) -> None:
        page.drawString(500, 700, "into the margin")  # ends past 538, before 595
        page.save()
        assert kinds(checker, tmp_path / "page.pdf") == ["text outside the margins"]
        assert kinds(checker, tmp_path / "page.pdf", margins=None) == []


class TestTheCheckerLeavesAlone:
    def test_an_underlined_link(self, checker: Any, page: Any, tmp_path: Path) -> None:
        page.drawString(100, 700, "example.com")
        page.line(100, 698.5, 152, 698.5)
        page.save()
        assert kinds(checker, tmp_path / "page.pdf") == []

    def test_lines_set_closely(self, checker: Any, page: Any, tmp_path: Path) -> None:
        for i in range(5):
            page.drawString(100, 700 - i * 12, "A closely set line of body text, gyp")
        page.save()
        assert kinds(checker, tmp_path / "page.pdf") == []

    def test_text_on_a_page_background(self, checker: Any, page: Any, tmp_path: Path) -> None:
        from PIL import Image
        from reportlab.lib.pagesizes import A4

        Image.new("RGB", (60, 85), "#eeeeee").save(tmp_path / "bg.png")
        page.drawImage(str(tmp_path / "bg.png"), 0, 0, *A4)
        page.drawString(100, 700, "text on the background")
        page.save()
        assert kinds(checker, tmp_path / "page.pdf") == []

    def test_the_command_line_says_so(self, checker: Any, page: Any, tmp_path: Path) -> None:
        page.drawString(100, 700, "fine")
        page.save()
        assert checker.main([str(tmp_path / "page.pdf")]) == 0


# -- what the module does so that the checker has nothing to find ------------------------


class TestCollisionsTheModulePrevents:
    def test_content_that_used_to_collide_builds_clean(
        self, d: Any, checker: Any, tmp_path: Path
    ) -> None:
        """Everything here collided, ran off the page or stopped the build before."""
        from PIL import Image

        Image.new("RGB", (900, 1600), "#dddddd").save(tmp_path / "tall.png")
        Image.new("RGB", (900, 520), "#dddddd").save(tmp_path / "wide.png")
        # A word with no break opportunity, wider than its table column.
        unbroken = "-".join(["segment"] * 10)
        story = [
            d.p('Inline <font size="22">large text</font> in body leading, ' * 6),
            d.numbered([f"Step {i}" for i in range(1, 14)]),
            d.table([["A", "B"], ["one", "two"]], col_widths=[400, 300]),
            d.table([["Key", "Value"], [unbroken, "https://example.com/" + unbroken]]),
            d.chart(tmp_path / "wide.png", width=700),
            d.chart(tmp_path / "tall.png", caption="taller than a page"),
            d.p("Before a negative spacer."),
            d.spacer(-40),
            d.p("After a negative spacer."),
        ]
        path = build(
            d,
            tmp_path,
            story,
            title="A cover title written as a whole sentence that keeps going for five lines",
            subtitle="A subtitle far too long for one line, with an audience, a date and a place",
            closing=True,
        )
        assert checker.check(path, checker.layout_margins()) == []

    def test_a_cover_title_that_cannot_fit_is_refused(self, d: Any, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="cover title too long"):
            build(d, tmp_path, [d.p("x")], title="An endless title " * 30)
