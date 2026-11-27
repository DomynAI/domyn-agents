"""Tests for the diff an edit result carries.

The point of the diff is that the model can notice a match that landed on a
fragment of a line, so the tests are about what it shows, not about the format
being pretty.
"""

from domyn.agents.tools.builtin import diff


def test_an_unchanged_text_has_no_diff() -> None:
    assert diff.changes("a\nb\n", "a\nb\n") == ""


def test_a_changed_line_shows_both_sides() -> None:
    assert diff.changes("a\nb\nc\n", "a\nB\nc\n") == "@@ line 1 @@\n a\n-b\n+B\n c"


def test_the_hunk_names_the_line_it_starts_at() -> None:
    before = "".join(f"line {n}\n" for n in range(1, 11))
    after = before.replace("line 7", "changed")
    assert diff.changes(before, after).startswith("@@ line 5 @@")


def test_two_distant_changes_are_two_hunks() -> None:
    before = "".join(f"line {n}\n" for n in range(1, 21))
    after = before.replace("line 2", "two").replace("line 19", "nineteen")
    shown = diff.changes(before, after)
    assert shown.count("@@ line") == 2


def test_a_long_diff_is_counted_not_cropped() -> None:
    before = "".join(f"line {n}\n" for n in range(1, 101))
    after = "".join(f"changed {n}\n" for n in range(1, 101))
    shown = diff.changes(before, after, limit=10)
    assert shown.endswith("more lines")
    assert shown.count("\n") == 10


def test_a_lost_final_newline_is_shown() -> None:
    """``splitlines`` drops it, so without the marker this reads as no change."""
    assert diff.changes("a\nb\n", "a\nb") == "\\ No newline at end of file"


def test_a_gained_final_newline_is_shown() -> None:
    assert diff.changes("a\nb", "a\nb\n") == "\\ Newline at end of file"


def test_a_fragment_that_dismantles_a_line_is_visible() -> None:
    """The failure this exists for: three 'successes' that ate the line."""
    before = 'UNIT = "kg\u00a0per\u00a0m3"\n'
    after = before.replace('UNIT = "kg', "")
    shown = diff.changes(before, after)
    assert "-" + before.rstrip("\n") in shown
    assert '+\u00a0per\u00a0m3"' in shown
