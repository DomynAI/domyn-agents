"""Finding the text an edit meant, when it is not quite the text the file holds.

``edit_file`` matches a string the model wrote against a file it read earlier.
The two are usually identical and occasionally are not: a typographic quote
where the file has a straight one, a space at the end of a line the model
dropped while retyping a block. Neither is a mistake about *what* to change,
and both used to cost a turn — the edit failed, the model re-read the file, and
the second attempt was the first one with one character taken out.

The match is tried at three levels, and the first that finds anything wins.
That order matters more than the levels themselves: a looser level is never
consulted when a stricter one matched, so a fuzzy match cannot displace an
exact one that exists somewhere else in the file. Ambiguity is an error at
every level.

What is deliberately absent is any notion of similarity. Comparing two strings
by how alike they look picks the nearest block, and the nearest block is
sometimes the wrong one — silently. Each level here names a difference that is
known to be meaningless, and forgives nothing else.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

EXACT = "exact"
"""The text is there, character for character."""

PUNCTUATION = "punctuation"
"""It is there once typographic quotes, dashes and spaces are read as ASCII."""

TRAILING = "trailing"
"""It is there as whole lines, once whitespace at the ends of those lines is ignored."""


@dataclass(frozen=True)
class Match:
    """One place ``old`` was found, and the level that found it."""

    start: int
    """Where the match begins in the text searched."""

    end: int
    """Where it ends, exclusive."""

    level: str
    """``EXACT``, ``PUNCTUATION`` or ``TRAILING`` — which level found it."""


def find(content: str, old: str, *, loose: bool = True) -> list[Match]:
    """Every occurrence of ``old`` in ``content``, at the first level that finds any.

    An empty ``old`` matches nothing: it is a call that would replace every
    position in the file, which is never what was meant.
    """
    if not old:
        return []
    for level, found in _levels(content, old, loose=loose):
        if found:
            return [Match(start, end, level) for start, end in found]
    return []


def _levels(content: str, old: str, *, loose: bool) -> Iterator[tuple[str, list[tuple[int, int]]]]:
    """The levels, in the order they are tried, each with what it finds."""
    yield EXACT, _occurrences(content, old)
    # ``_plain`` maps one character to one character, so an offset found in the
    # plain text is an offset into the text itself. That is the whole reason
    # this level needs no map of its own.
    yield PUNCTUATION, _occurrences(_plain(content), _plain(old))
    if loose:
        yield TRAILING, _line_occurrences(content, old)


def _occurrences(content: str, old: str) -> list[tuple[int, int]]:
    """Every place ``old`` appears in ``content``, left to right, non-overlapping."""
    if not old:
        # ``str.find("")`` answers 0 forever, so this is a hang, not a match.
        return []
    found: list[tuple[int, int]] = []
    at = content.find(old)
    while at != -1:
        found.append((at, at + len(old)))
        at = content.find(old, at + len(old))
    return found


_PLAIN = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u00a0": " ",
        "\u2002": " ",
        "\u2003": " ",
        "\u2004": " ",
        "\u2005": " ",
        "\u2006": " ",
        "\u2007": " ",
        "\u2008": " ",
        "\u2009": " ",
        "\u200a": " ",
        "\u202f": " ",
        "\u205f": " ",
        "\u3000": " ",
    }
)
"""Typographic variants of ASCII punctuation, each mapped to a single character.

One character in, one character out. A mapping that changed the length would
move every offset after it, and the match would be applied to the wrong place
in the file — quietly, and only in files that contain one of these characters.
"""


def _plain(text: str) -> str:
    """``text`` with typographic punctuation read as the ASCII it stands for."""
    return text.translate(_PLAIN)


def _line_occurrences(content: str, old: str) -> list[tuple[int, int]]:
    """Places where ``old`` appears as whole lines, ignoring trailing whitespace.

    Only a block that starts at the beginning of a line and covers whole lines
    is eligible. A fragment from the middle of a line has no line ends to
    forgive, and treating it as if it did would mean guessing where its line
    began — which is the kind of guess that lands an edit on the wrong line.
    """
    wanted = old.split("\n")
    ends_with_newline = old.endswith("\n")
    if ends_with_newline:
        wanted.pop()
    spans = _line_spans(content)
    found: list[tuple[int, int]] = []
    for first in range(len(spans) - len(wanted) + 1):
        window = spans[first : first + len(wanted)]
        if not all(
            content[start:end].rstrip() == line.rstrip()
            for (start, end, _), line in zip(window, wanted, strict=False)
        ):
            continue
        _, last_end, last_after = window[-1]
        if ends_with_newline and last_after == last_end:
            # ``old`` ends with a line break the file does not have there.
            continue
        found.append((window[0][0], last_after if ends_with_newline else last_end))
    return found


def _line_spans(content: str) -> list[tuple[int, int, int]]:
    """``(start, end, end with the line break)`` for every line of ``content``.

    A file that ends with a line break has one more, empty, line after it. That
    is not an artefact: an edit may legitimately end with the file's last
    newline, and it needs a line to end on.
    """
    spans: list[tuple[int, int, int]] = []
    start = 0
    while True:
        newline = content.find("\n", start)
        if newline == -1:
            spans.append((start, len(content), len(content)))
            return spans
        spans.append((start, newline, newline + 1))
        start = newline + 1
