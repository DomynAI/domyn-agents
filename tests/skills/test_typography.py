"""The shared typography (`skills/builtin/resources/typography.json`): the fonts and text
sizes slides and documents both read, as they both read the palette."""

import json
from pathlib import Path

import pytest

import domyn.agents

RESOURCES = Path(domyn.agents.__file__).parent / "skills" / "builtin" / "resources"
TYPOGRAPHY = json.loads((RESOURCES / "typography.json").read_text())
FONTS = RESOURCES / "assets" / "fonts"


@pytest.mark.parametrize("key", sorted(TYPOGRAPHY["styles"]))
def test_every_style_s_file_is_the_family_it_names(key: str) -> None:
    """PowerPoint finds a run's font by family name and bold flag, reportlab by file: the two
    have to be the same font, or a document and a deck set the same role in different faces."""
    image_font = pytest.importorskip("PIL.ImageFont")
    style = TYPOGRAPHY["styles"][key]
    if style["file"] is None:
        assert key == "thin"  # documented as not shipped; the master does not ship it either
        return
    family, face = image_font.truetype(str(FONTS / style["file"]), 10).getname()
    # FreeType reports the typographic family ("DM Sans", "Light"); PowerPoint stores the
    # legacy one, which carries the weights that are neither regular nor bold ("DM Sans Light").
    words = face.split()
    weight = [w for w in words if w not in ("Regular", "Bold", "Italic")]
    assert " ".join([family, *weight]) == style["family"]
    assert ("Bold" in words) == style["bold"]
    assert ("Italic" in words) == style.get("italic", False)


def test_the_five_slide_styles_exist() -> None:
    assert {"title", "bold", "regular", "light", "thin"} <= set(TYPOGRAPHY["styles"])


@pytest.mark.parametrize("name", sorted(TYPOGRAPHY["roles"]))
def test_every_role_is_a_shipped_style_and_a_size(name: str) -> None:
    role = TYPOGRAPHY["roles"][name]
    assert TYPOGRAPHY["styles"][role["style"]]["file"]
    assert isinstance(role["size"], (int, float)) and 6 <= role["size"] <= 60
    assert role["use"]
