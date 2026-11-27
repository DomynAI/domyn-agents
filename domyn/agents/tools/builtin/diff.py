"""What an edit changed, in the form the model can check.

A tool that answers "Edited src/app.py." asks the model to trust it. The model
cannot see the file it just changed, so it cannot notice that the match landed
on a fragment of a line, or that a block it meant to rewrite came out with one
line fewer. Both have been observed: a model that cannot find its anchor
narrows it until something matches, and three calls reported as successes
dismantled the line between them.

So the result carries the change. Not the whole file — the hunks, with a couple
of lines of context, and a cap, because a result is read on every call and the
context it costs is paid on every turn after it.
"""

from __future__ import annotations

import difflib

CONTEXT = 2
"""Lines kept either side of a change, so the model can see where it landed."""

LIMIT = 40
"""Lines of diff before the rest is counted instead of shown.

A tool result is kept whole up to ``max_tool_result_chars`` (20 000 by default)
and cropped past it, keeping the start and the end. A diff that long would be
cropped in the middle, which is the one place it must not be: the cap here is
chosen so a diff is either complete or honestly shortened.
"""


def changes(before: str, after: str, *, context: int = CONTEXT, limit: int = LIMIT) -> str:
    """The lines ``after`` removes and adds to ``before``, or ``""`` if none.

    Both sides are the text as the model reads it — LF line endings — so a CRLF
    file does not show every line as changed.
    """
    old = before.splitlines()
    new = after.splitlines()
    lines: list[str] = []
    for group in difflib.SequenceMatcher(None, old, new, autojunk=False).get_grouped_opcodes(
        context
    ):
        lines.append(f"@@ line {group[0][1] + 1} @@")
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                lines.extend(f" {line}" for line in old[i1:i2])
                continue
            lines.extend(f"-{line}" for line in old[i1:i2])
            lines.extend(f"+{line}" for line in new[j1:j2])
    ending = _final_newline(before, after)
    if ending:
        lines.append(ending)
    if not lines:
        return ""
    if len(lines) > limit:
        lines = [*lines[:limit], f"… {len(lines) - limit} more lines"]
    return "\n".join(lines)


def _final_newline(before: str, after: str) -> str:
    r"""The unified-diff marker for a file that gained or lost its last line break.

    ``splitlines`` drops it, so without this a file rewritten from
    ``"a\nb\n"`` to ``"a\nb"`` would read as unchanged.
    """
    if not before or not after or before.endswith("\n") == after.endswith("\n"):
        return ""
    return "\\ Newline at end of file" if after.endswith("\n") else "\\ No newline at end of file"
