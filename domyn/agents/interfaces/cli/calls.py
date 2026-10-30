r"""How a tool call reads on screen: one line in the trace, once in an approval.

Three places show a call, and they used to show it the same way — every
argument in full, one after another — so a 30-line command filled the trace
line, the approval heading and the approval's parameters with the same text,
and an edit read as a wall of ``\n``. Each place now has one job:

* **The trace line** (:func:`summary`) says what the call does, on one line: the
  argument that names it first (the command, the path, the pattern), the others
  in short. A value on several lines is counted, not printed.
* **The approval heading** (:func:`approval_view`) is the subject — what a grant
  is matched against — when it fits on one line, otherwise the tool's name.
* **The approval body** has every argument the heading does not already show,
  in full and once: a command as the block it is, an edit as a diff.

Which argument names a built-in call, and which one is its content, is how this
front-end presents those tools rather than part of what a tool is, so it lives
here (:data:`_SHAPES`) and not on the tools. Any other tool reads ``key=value``.
"""

import difflib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rich.cells import cell_len

from domyn.agents.approvals import visible_parameters


def as_written(value: object, *, multiline: bool = False) -> str:
    r"""A tool argument the way the model wrote it, for a person to read.

    Not ``repr``: it doubles every backslash and escapes quotes, so a call to
    ``grep "a\|b"`` used to read ``'grep "a\\|b"'`` — one command shown three
    ways on one approval screen. A string comes out as written; only what a
    terminal would act on rather than print (ESC, a carriage return, NUL, a bidi
    override, a non-breaking space) is escaped, so an argument cannot redraw the
    question asking about it, nor pass for a different command. Newlines are
    kept with ``multiline``, otherwise written ``\n`` to stay on one line.
    Anything that is not a string keeps its ``repr``.
    """
    if not isinstance(value, str):
        return repr(value)
    if not value:
        return "''"  # an empty argument still has to show
    return _escaped(value, multiline=multiline)


def _escaped(text: str, *, multiline: bool = False) -> str:
    """``as_written`` for a piece of a value: an empty line stays empty."""
    if text.isprintable():
        return text
    return "".join(
        ch if ch.isprintable() or ch == "\t" or (multiline and ch == "\n") else repr(ch)[1:-1]
        for ch in text
    )


@dataclass(frozen=True)
class _Shape:
    """How one built-in tool's call reads."""

    names: str
    """The argument that says what the call is about: the trace line leads with it."""
    content: str | None = None
    """The argument that is the call's content, shown under the heading without a label."""
    diff: tuple[str, str] | None = None
    """``(old, new)``: shown together, as a diff."""


_SHAPES: dict[str, _Shape] = {
    "run_command": _Shape("command", content="command"),
    "check_command": _Shape("handle"),
    "stop_command": _Shape("handle"),
    "read_file": _Shape("path"),
    "list_dir": _Shape("path"),
    "glob_files": _Shape("pattern"),
    "grep": _Shape("pattern"),
    "write_file": _Shape("path", content="content"),
    "append_file": _Shape("path", content="content"),
    "edit_file": _Shape("path", diff=("old", "new")),
}


def _lines(text: str) -> list[str]:
    """The lines of a value; a final newline does not start one more."""
    return text.removesuffix("\n").split("\n")


def _is_block(value: object) -> bool:
    """Whether a value is on several lines."""
    return isinstance(value, str) and "\n" in value.removesuffix("\n")


def summary(name: str, params: Mapping[str, Any]) -> str:
    """What follows the tool's name on the trace line: one line, however long the call.

    It may run past the terminal's edge — the caller crops it there — but never
    onto a second line. A value on several lines is counted; the argument that
    names the call shows its first line before the count.
    """
    shown = visible_parameters(dict(params))
    shape = _SHAPES.get(name)
    lead = shape.names if shape is not None and shape.names in shown else None
    parts: list[str] = []
    if lead is not None:
        value = shown[lead]
        if _is_block(value):
            lines = _lines(value)
            parts.append(f"{_escaped(lines[0])}  (+{len(lines) - 1} lines)")
        else:
            parts.append(as_written(value))
    for key, value in shown.items():
        if key == lead:
            continue
        shown_value = f"({len(_lines(value))} lines)" if _is_block(value) else as_written(value)
        parts.append(f"{key}={shown_value}")
    return "  ".join(parts)


@dataclass(frozen=True)
class Field:
    """One argument, in full, under the approval heading."""

    name: str | None
    """``None`` for the call's content, shown without a label."""
    text: str
    """As written; a value on several lines keeps them."""


@dataclass(frozen=True)
class DiffLine:
    """One line of an edit."""

    mark: str
    """``"-"`` removed, ``"+"`` added, ``" "`` kept."""
    text: str


@dataclass(frozen=True)
class Diff:
    """An edit, as the lines it removes and the lines it adds."""

    lines: tuple[DiffLine, ...]


def diff_lines(old: object, new: object) -> tuple[DiffLine, ...]:
    """``old`` turned into ``new``, line by line: what goes, what comes, what stays."""
    before, after = _diff_side(old), _diff_side(new)
    lines: list[DiffLine] = []
    matcher = difflib.SequenceMatcher(None, before, after, autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            lines.extend(DiffLine(" ", line) for line in before[i1:i2])
            continue
        lines.extend(DiffLine("-", line) for line in before[i1:i2])
        lines.extend(DiffLine("+", line) for line in after[j1:j2])
    return tuple(lines)


def _diff_side(value: object) -> list[str]:
    if not isinstance(value, str):
        return [as_written(value)]
    return [_escaped(line) for line in _lines(value)] if value else []


def approval_view(
    name: str, params: Mapping[str, Any], subject: str, *, room: int
) -> tuple[str, list[Field | Diff]]:
    """The approval's heading and body: the call, once.

    The heading is the subject when it fits in ``room`` columns — it is what a
    grant is matched against — otherwise the tool's name. It is the name too
    when the subject holds an argument on several lines flattened into one:
    ``$ a b`` reads as one command where two run. The body has every argument
    the heading does not show, in full: the call's content unlabelled, an edit
    as a diff, anything else under its name.
    """
    shown = visible_parameters(dict(params))
    flattened = (" ".join(value.split()) for value in shown.values() if _is_block(value))
    heading = as_written(subject)
    is_subject = cell_len(heading) <= room and not any(f and f in subject for f in flattened)
    if not is_subject:
        heading = name
    shape = _SHAPES.get(name)
    pair = shape.diff if shape is not None else None
    if pair is not None and not all(key in shown for key in pair):
        pair = None
    body: list[Field | Diff] = []
    diffed = False
    for key, value in shown.items():
        if pair is not None and key in pair:
            if not diffed:  # once, where the first of the two is
                body.append(Diff(diff_lines(shown[pair[0]], shown[pair[1]])))
                diffed = True
            continue
        if is_subject and _in_heading(heading, key, value, shape):
            continue
        label = None if shape is not None and key == shape.content else key
        text = (
            _escaped(value.removesuffix("\n"), multiline=True)
            if _is_block(value)
            else as_written(value)
        )
        body.append(Field(label, text))
    return heading, body


def _in_heading(heading: str, key: str, value: object, shape: _Shape | None) -> bool:
    """Whether the heading already shows this argument, so the body can leave it out.

    Checked on the text, not assumed from how a tool builds its subject: if the
    two ever disagree, an argument is shown twice rather than not at all.
    """
    if _is_block(value):
        return False
    text = as_written(value)
    if shape is not None and key == shape.names and heading.endswith(f" {text}"):
        return True  # `$ git status`: the subject is the command itself
    piece = f"{key}={text}"
    start = 0
    while (at := heading.find(piece, start)) >= 0:
        end = at + len(piece)
        if (at == 0 or heading[at - 1] == " ") and (end == len(heading) or heading[end] == " "):
            return True
        start = at + 1
    return False
