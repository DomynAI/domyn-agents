"""How the builtin skills name their files: relative to the SKILL.md that names them.

The model is given one location for a skill, the `[file: …]` of its SKILL.md, in the header
above the body wherever the body reaches it (`skill_view`, or expansion), so a path written
relative to it is one the model can resolve, and a placeholder like
"<this skill's directory>" is one it has to guess at (it guessed the working directory). Every
skill that names a file outside itself says so in the same sentence, and every path it names
exists.
"""

import re
from pathlib import Path

import pytest

import domyn.agents

BUILTIN = Path(domyn.agents.__file__).parent / "skills" / "builtin"
SKILLS = sorted(BUILTIN.rglob("SKILL.md"))
NOTICE = (
    "Paths in this file are relative to this SKILL.md, whose location is the `[file: …]` line"
    " above this text"
)

# `../../resources/palette.json`, `scripts/check_pptx.py`, `../prepare_table.py` …
RELATIVE = re.compile(r"`((?:\.\./)+[\w./-]*[\w/]|scripts/[\w./-]*[\w/])")


def named_paths(text: str) -> set[str]:
    return set(RELATIVE.findall(text))


def ids(path: Path) -> str:
    return str(path.parent.relative_to(BUILTIN))


@pytest.mark.parametrize("skill_md", SKILLS, ids=ids)
def test_every_relative_path_exists(skill_md: Path) -> None:
    for rel in named_paths(skill_md.read_text()):
        assert (skill_md.parent / rel).exists(), rel


@pytest.mark.parametrize("skill_md", SKILLS, ids=ids)
def test_a_skill_naming_files_says_what_they_are_relative_to(skill_md: Path) -> None:
    text = skill_md.read_text()
    if named_paths(text):
        assert NOTICE in " ".join(text.split()), "the paths notice is missing"


@pytest.mark.parametrize("skill_md", SKILLS, ids=ids)
def test_no_directory_is_left_to_guess(skill_md: Path) -> None:
    text = skill_md.read_text()
    for placeholder in ("<this skill's directory>", "<skill-dir>", "builtin/resources/"):
        assert placeholder not in text


def test_the_skills_that_use_resources_are_covered() -> None:
    covered = {ids(p) for p in SKILLS if named_paths(p.read_text())}
    assert {
        "productivity/presentations",
        "productivity/documents",
        "visualization/charts",
        "visualization/interactive-charts",
        "data/data-analysis",
        "data/tabular-classification-regression",
        "data/time-series-forecasting",
    } <= covered
