"""Tests for finding the text an edit meant.

The corpus here was run against the real matchers of two other harnesses, not
against a description of them: opencode's `edit` (`packages/core/src/tool/plugin/
edit.ts`, commit `fd3c9c4`) and DeepSeek Harness's `apply_patch`
(`packages/fs/fs-patch/src/apply.ts`, commit `4b399c39`). On all 42 cases of the
comparison the level and the number of matches agree with opencode's `edit`
exactly. Where this module deliberately differs from DSH, the case says so:

* DSH's `edit` is exact-only — it fails every punctuation and trailing-whitespace
  case below. Its tolerance lives in `apply_patch`, which is why the divergences
  are with the patch matcher, not with the edit tool.
* DSH's patch matcher forgives *leading* whitespace too (its `trim` pass). This
  does not: a re-indented block has no anchor without context lines, and
  guessing which block was meant is the failure this module exists to avoid.
* DSH's patch matcher is line-anchored, so it cannot match a fragment from the
  middle of a line. This can, because `edit_file` matches strings, not lines.
"""

import pytest

from domyn.agents.tools.builtin import matching

# (content, old, level, count) — the level that found it, and how many places.
# `none` means the edit must be refused.
CASES = [
    # exact: the case that should stay the case
    ("def add(a, b):\n    return a + b\n", "    return a + b", "exact", 1),
    ("def f():\n    a = 1\n    b = 2\n    return a\n", "    a = 1\n    b = 2", "exact", 1),
    ("    x = 1   \n", "    x = 1", "exact", 1),
    ("    return a + b\n", "a + b", "exact", 1),
    ("a\nb\n", "a\nb\n", "exact", 1),
    ("a\nb\n", "b\n", "exact", 1),
    ("a\nb", "b", "exact", 1),
    # punctuation: one character for one character
    ("it's fine\n", "it\u2019s fine", "punctuation", 1),
    ('say "hi"\n', "say \u201chi\u201d", "punctuation", 1),
    ("a - b\n", "a \u2013 b", "punctuation", 1),
    ("a - b\n", "a \u2014 b", "punctuation", 1),
    ("a\u00a0b\n", "a b", "punctuation", 1),
    ("| a\u00a0b | c |\n", "| a b | c |", "punctuation", 1),
    ("def f():\n    # don't\n    pass\n", "def f():\n    # don\u2019t\n    pass", "punctuation", 1),
    ("# 1 - 10 items\n", "# 1 \u2013 10 items", "punctuation", 1),
    # trailing: whole lines, whitespace at their ends
    ("    x = 1\n", "    x = 1   ", "trailing", 1),
    ("    a\n    b  \n    c\n", "    a\n    b\n    c", "trailing", 1),
    (
        "def f():\n    a = 1\n    b = 2  \n    return a\n",
        "def f():\n    a = 1\n    b = 2\n    return a",
        "trailing",
        1,
    ),
    (
        "class A:\n    x = 1 \n\n    def f(self):\n        return 1 \n",
        "class A:\n    x = 1\n\n    def f(self):\n        return 1",
        "trailing",
        1,
    ),
    # ambiguity is an error at every level
    ("print(x)\nprint(x)\n", "print(x)", "exact", 2),
    ("x  \nx\n", "x", "exact", 2),
    # and no level forgives a difference that changes the text
    ("    return a\n", "        return a", "none", 0),
    ("    return a\n", "\treturn a", "none", 0),
    ("  x = 1\n", "    x = 1", "none", 0),
    ("a  b\n", "a b", "none", 0),
    ("- one\n- two\n", "\u2022 one\n\u2022 two", "none", 0),
    ("    return total\n", "    return sum", "none", 0),
    ("    x = 1\n", "    x = 2", "none", 0),
    ("    a = 1\n    b = 2\n", "    a = 1\n    c = 3\n    b = 2", "none", 0),
    # an edit that ends with a line break the file does not have there
    ("a\nb", "b\n", "none", 0),
    ("a\n", "", "none", 0),
]


@pytest.mark.parametrize(("content", "old", "level", "count"), CASES)
def test_the_level_that_finds_it(content: str, old: str, level: str, count: int) -> None:
    found = matching.find(content, old)
    assert len(found) == count
    if found:
        assert found[0].level == level


@pytest.mark.parametrize(("content", "old", "level", "count"), CASES)
def test_strict_keeps_the_levels_that_cannot_move_an_edit(
    content: str, old: str, level: str, count: int
) -> None:
    """``loose=False`` drops the trailing level and nothing else."""
    assert len(matching.find(content, old, loose=False)) == (count if level != "trailing" else 0)


def test_the_offsets_are_where_the_text_is() -> None:
    """A match that is found but misplaced is worse than one that is not found."""
    content = "def f():\n    a = 1\n    return a\n"
    (found,) = matching.find(content, "    a = 1")
    assert content[found.start : found.end] == "    a = 1"


def test_the_offsets_survive_a_typographic_character() -> None:
    """The plain text must be the same length as the text, or every offset after it moves."""
    content = "x = \u201cit\u2019s\u201d\ny = 2\n"
    (found,) = matching.find(content, 'x = "it\'s"')
    assert content[found.start : found.end] == "x = \u201cit\u2019s\u201d"


def test_a_trailing_match_covers_whole_lines() -> None:
    content = "one\n  two  \nthree\n"
    (found,) = matching.find(content, "  two\nthree")
    assert content[found.start : found.end] == "  two  \nthree"


def test_a_trailing_match_can_end_with_the_line_break() -> None:
    content = "one\n  two  \nthree\n"
    (found,) = matching.find(content, "  two\nthree\n")
    assert content[found.start : found.end] == "  two  \nthree\n"


def test_an_exact_match_wins_over_a_looser_one_elsewhere() -> None:
    """A looser level is never consulted once a stricter one matched."""
    content = "it\u2019s here\nit's there\n"
    found = matching.find(content, "it's there")
    assert [(m.level, m.start) for m in found] == [("exact", content.index("it's there"))]


def test_a_fragment_from_the_middle_of_a_line_is_not_a_trailing_match() -> None:
    """There is no line end to forgive, so the level must not pretend there is."""
    assert matching.find("    return a + b\n", "return a + b  ") == []
    assert matching.find("    return a + b\n    return c\n", "a + b\n    return c  ") == []


def test_a_leading_whitespace_difference_is_not_a_trailing_match() -> None:
    """DSH's patch matcher forgives it; without context lines it is a guess."""
    assert matching.find("        deep = 1\n", "    deep = 2\n") == []
    assert matching.find("    a\n        b\n", "    a\n    b\n") == []


def test_an_empty_old_finds_nothing() -> None:
    """``str.find("")`` answers 0 forever: a match here would be a hang."""
    assert matching.find("a\n", "") == []
