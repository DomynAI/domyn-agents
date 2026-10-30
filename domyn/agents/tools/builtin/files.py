"""Reading, searching and changing files.

Split by risk rather than by subject matter, because that split is what the
approval gate keys on. Reading and searching are ungated: they cannot damage
anything, and gating them would train the user to approve without reading.
Writing and editing are gated, with the *path* as the approved subject — so
"yes, you may keep editing this file" is expressible, while "yes, you may write
anywhere" is not.

``append_file`` exists because of a failure mode that has nothing to do with
files: a model writing a whole HTML document into one JSON string argument has
to escape every quote and newline in it, and over a few thousand characters it
misses one. Splitting the write into several smaller calls is the only route
out that does not depend on the model getting a long escape right.

``edit_file`` replaces an exact string rather than taking a diff or a line
range. Line numbers go stale between the read and the write, and a diff is a
format the model gets subtly wrong. An exact match that must occur exactly once
fails loudly when the file is not what the model thought it was, which is the
failure you want.

Documents — PDF, Word, Excel, PowerPoint, OpenDocument, RTF, EPUB — are read by
converting them to markdown first, because a model handed the bytes of a PDF
learns nothing from them. Which files are documents, and what a failed
conversion means, is ``domyn.agents.core.documents``: the same answers every
other reader of files gets. A file's name decides only what is tried first; the
content settles it, so text saved under a document's name is read as text and
a document saved under another name is converted. Converted text is windowed
like any other file, so its line numbers are lines of the markdown, not of the
file. A PDF is read page by page: what a page holds that is not text is marked
on that page, and the first window says what was left out, since the mark may
be windows away.
"""

# No `from __future__ import annotations` here: FunctionTool reads the signature
# of each tool at construction time and cannot resolve string annotations. The
# tools are defined in this module, so the import has to stay off.

import io
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

import anydoc

from domyn.agents.core import ApprovalType, documents
from domyn.agents.tools.builtin.workspace import Workspace, WorkspaceError, is_ignored

# The most one read returns. It has to fit inside what the model is actually
# shown: the tool-use context manager keeps the start and the end of a longer
# tool result, within ``max_tool_result_chars`` (20 000 by default, event-id
# header included). A longer window loses its middle, and the note's
# ``start_line`` then reads on past lines the model never saw.
MAX_READ_CHARS = 12_000
MAX_MATCHES = 200
MAX_LISTED = 500


# How many lines a read returns when the caller does not say. Enough to see a
# whole small file at once, small enough that a large one arrives in pieces the
# model can decide about.
DEFAULT_READ_LINES = 500

# Largest document converted. The converter parses a document whole — a PDF
# keeps its index at the end, and the Office and OpenDocument formats are zip
# or compound-file packages — so this is a refusal rather than a window, and a
# conversion needs many times the file's size in memory while it runs.
MAX_DOCUMENT_BYTES = 25 * 1024 * 1024

# Runs of pages a line names before it counts the rest: a PDF with a scan on
# every other page must not open with a line of hundreds of numbers.
_LISTED_RUNS = 20


def _positive_int(value: object, fallback: int) -> int:
    """Read a number the model supplied, whatever shape it arrived in.

    The schema says integer and a model may still send ``"200"``; see
    ``shell._seconds`` for the same lesson learned from a crash. Anything
    unusable falls back rather than failing: the caller asked to read a file.
    """
    try:
        number = int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback
    return number if number > 0 else fallback


def _read_text(path: str, workspace: Workspace) -> str:
    """The whole file, for callers that need it — ``edit_file`` matching.

    Not what the model gets: see ``read_file``, which reads a window.
    """
    resolved = workspace.resolve(path, must_exist=True)
    if resolved.is_dir():
        raise WorkspaceError(f"'{path}' is a directory. Use list_dir for that.")
    try:
        return resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise WorkspaceError(
            f"'{path}' is not UTF-8 text. Binary files cannot be read here."
        ) from None


def _contained(workspace: Workspace, found: Path) -> bool:
    """Whether a file a glob found is inside the workspace once resolved.

    Checked as every path a tool is given is checked: a glob can climb out with
    ``..``, and a symlink in the workspace can point anywhere.
    """
    try:
        workspace.resolve(found)
    except WorkspaceError:
        return False
    return True


def _files_to_search(workspace: Workspace, scope: Path, glob: str) -> list[Path]:
    """The files under the directory ``scope`` that ``glob`` names.

    Read as ripgrep reads a glob, which is what models write: without a ``/`` it
    matches a file's name at any depth, with one a path from the root that holds
    ``scope``: the workspace, as ``path`` is, or the extra root ``scope`` is in.
    Read from ``scope`` instead, ``src/*.py`` under ``src`` named
    ``src/src/*.py``, and the search found nothing without saying why.
    """
    if "/" not in glob:
        found = scope.rglob(glob)
    elif glob.startswith("**/") and "/" not in glob[3:]:
        # A name at any depth: what it names under ``scope`` is the same from
        # either place, and reading it from ``scope`` walks ``scope`` only.
        found = scope.glob(glob)
    else:
        # From the root that holds ``scope``, but only what is under ``scope``:
        # ``path`` still bounds the search. A directory the glob names may be
        # ``scope`` itself or one above it (``**/src/*.py`` under ``src``), which
        # a glob read from ``scope`` cannot see; and an extra root, such as a
        # skill directory, is outside the workspace root.
        base = next(
            (root for root in workspace.roots if root == scope or root in scope.parents),
            workspace.root,
        )
        found = (p for p in base.glob(glob) if p == scope or scope in p.parents)
    return sorted(
        p
        for p in found
        if p.is_file() and not is_ignored(p, workspace.root) and _contained(workspace, p)
    )


def _matches(workspace: Workspace, base: Path, pattern: str, argument: str) -> list[Path]:
    """The files under ``base`` that a glob pattern names, never one outside the workspace.

    ``Path.glob`` does not know about the workspace. A pattern with ``..`` climbs out
    of ``base`` (``../*`` lists the parent), and a match can be a symlink to anywhere,
    which the caller then reads. So the pattern has to be relative and stay inside
    ``base``, and every match is resolved and checked against the roots, the same way
    a path the model typed is.

    What goes wrong is reported in words the model can act on: pathlib's own errors
    (``NotImplementedError: Non-relative patterns are unsupported``) say neither
    what was wrong nor what to do instead.
    """
    parts = PurePosixPath(pattern.replace("\\", "/")).parts
    if not pattern.strip():
        raise WorkspaceError(f"{argument} is empty. Give a pattern such as '**/*.py'.")
    if pattern.startswith("~") or PurePosixPath(pattern).is_absolute() or ".." in parts:
        raise WorkspaceError(
            f"{argument} {pattern!r} must be relative to the folder searched and stay inside "
            "it: no absolute path, '~' or '..'. To search another folder, give it as 'path' "
            "and keep the pattern relative to it, e.g. path='/abs/folder', pattern='*.py'."
        )
    try:
        candidates = sorted(base.glob(pattern))
    except (ValueError, NotImplementedError) as exc:
        raise WorkspaceError(f"{argument} {pattern!r} is not a usable pattern: {exc}") from None

    inside: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if workspace.contains(resolved) and not is_ignored(
            candidate, workspace.anchor(base) or workspace.root
        ):
            inside.append(candidate)
    return inside


@dataclass(frozen=True)
class _Window:
    lines: list[str]
    more: bool
    """Something follows the window."""
    last_seen: int
    continues_at: int | None = None
    """Where the last line goes on, in characters from its start, when it alone
    exceeded the budget and was cut."""
    line_length: int | None = None
    """The first line's length, when the read was asked to start past its end."""
    document: bool = False
    """The lines are a document's converted text, not the file's own."""
    omitted: str = ""
    """What a document's conversion left out, said on its first window."""


def _read_window(resolved: Path, start: int, count: int, start_char: int = 0) -> _Window:
    """Lines ``start..start+count-1`` of a text file, and what is around them.

    Iterated rather than read whole and sliced. Slicing would still pull the
    entire file into memory, which makes "read it in pieces" true only for the
    model and not for the process doing the reading — and a large enough file
    would take the session down before the window was ever computed.
    """
    with resolved.open(encoding="utf-8") as handle:
        return _window(handle, start, count, start_char)


def _window(source: Iterable[str], start: int, count: int, start_char: int = 0) -> _Window:
    """Lines ``start..start+count-1`` of ``source``, within the read budget.

    ``start_char`` skips that many characters of the first line. It is the way
    on through a line too long for one read, which is cut where the budget
    ends: continuing from the next line would skip the rest of it.
    """
    lines: list[str] = []
    budget = MAX_READ_CHARS
    last = 0

    for number, line in enumerate(source, start=1):
        last = number
        if number < start:
            continue
        offset = start_char if number == start else 0
        if offset:
            length = len(line.rstrip("\r\n"))
            if offset >= length:
                return _Window([], more=True, last_seen=last, line_length=length)
            line = line[offset:]
        if len(lines) >= count:
            # One line past the window is enough to know something follows;
            # counting the rest would mean reading the whole file to say so.
            return _Window(lines, more=True, last_seen=last)
        if len(line) > budget:
            if lines:
                # The window is full. Stop before this line so the next read
                # starts on it whole: cutting it here would report a
                # truncation that did not happen.
                return _Window(lines, more=True, last_seen=last)
            # A minified bundle or a base64 blob is one line and can be
            # megabytes. Without cutting it the line budget is no budget at
            # all, and the caller gets the whole file it asked not to get. A
            # line over only by its line break fits: cut there, the next read
            # would be offered a start_char with nothing left after it.
            if len(line.rstrip("\r\n")) > budget:
                lines.append(line[:budget])
                return _Window(lines, more=True, last_seen=last, continues_at=offset + budget)
        budget -= len(line)
        lines.append(line)
    return _Window(lines, more=False, last_seen=last)


def _read_file(path: str, resolved: Path, start: int, count: int, start_char: int = 0) -> _Window:
    """A window of a file, read the way ``domyn.agents.core.documents`` reads files.

    A text file is read a window at a time, never whole. What has to be read
    whole goes through ``documents.extract_text`` instead: a document, which
    the converter parses whole; a CSV, whose reading depends on whether all of
    it is UTF-8; and a file that turns out not to be UTF-8, which may be a
    document under another name.

    Over ``MAX_DOCUMENT_BYTES`` nothing is converted. A file whose name says it
    is a document is refused unless it reads as text; any other file is read
    as text or refused as binary.
    """
    size = resolved.stat().st_size
    named = anydoc.format_from_path(resolved)
    if size > MAX_DOCUMENT_BYTES or named is None:
        try:
            return _read_window(resolved, start, count, start_char)
        except UnicodeDecodeError:
            if size > MAX_DOCUMENT_BYTES:
                if named not in (None, "csv"):
                    raise _too_large(path, size) from None
                raise _binary(path) from None
    try:
        extracted = documents.extract_text(resolved)
    except documents.NotTextError:
        raise _binary(path) from None
    except documents.DocumentError as error:
        raise _conversion_failure(path, error) from None
    if extracted.converted:
        return _document_window(extracted, start, count, start_char)
    # Read whole, but windowed as a text file is: newlines translated the same.
    return _window(io.StringIO(extracted.text, newline=None), start, count, start_char)


def _binary(path: str) -> WorkspaceError:
    return WorkspaceError(f"'{path}' is not UTF-8 text. Binary files cannot be read here.")


def _too_large(path: str, size: int) -> WorkspaceError:
    return WorkspaceError(
        f"'{path}' is {_megabytes(size)}, over the {_megabytes(MAX_DOCUMENT_BYTES)} "
        "limit for converting a document. A document is converted whole, so reading "
        "part of it is not possible either. Retrying will not help; try a smaller or "
        "split-up version."
    )


def _refuse_text_over_document(path: str, resolved: Path) -> None:
    """Refuse to write text over a document, which ``read_file`` shows converted.

    What the model read of a document is a markdown conversion, not the file,
    and every write tool here writes plain text: writing that back, appending
    to it or editing it would replace the document instead of changing it, and
    a later read would show the text as if nothing had happened. Decided as
    reading decides (``documents.document_format``): a document under another
    name is protected, text under a document's name (a TSV saved as ``.xls``)
    is not, and a CSV is protected only when it is not UTF-8, the one kind that
    reads converted. A file over ``MAX_DOCUMENT_BYTES`` is never converted, so
    it is left to the approval gate.
    """
    if not resolved.is_file() or resolved.stat().st_size > MAX_DOCUMENT_BYTES:
        return
    kind = documents.document_format(resolved.read_bytes(), resolved)
    if kind is not None:
        raise WorkspaceError(
            f"'{path}' is a document ({kind}): what read_file shows of it is a markdown "
            "conversion, not its contents, and write_file, append_file and edit_file write "
            "plain text, so they would replace the document instead of changing it. Write "
            f"the changed text to a new file instead, such as "
            f"'{PurePosixPath(path).with_suffix('.md')}'."
        )


def _conversion_failure(path: str, error: documents.DocumentError) -> WorkspaceError:
    """What to tell the model about a document that could not be converted.

    Every one of these is a property of the file, so each says retrying will
    not help: a model told only "could not read it" tries again in a slightly
    different way.
    """
    if error.kind == "encrypted":
        reason = "is password-protected, so its content cannot be read"
    elif error.kind == "needs_ocr":
        reason = _needs_ocr(error)
    elif error.kind == "no_text":
        reason = "has no text the converter can extract"
    elif error.kind == "resource_limit":
        reason = (
            "could not be converted: it crosses a safety limit on how deeply "
            "nested or how heavily compressed a document may be, or on how long "
            "or how much memory reading one may take"
        )
    else:
        reason = (
            "could not be read as a document: it appears corrupt, truncated, or "
            "not the kind of file its extension claims"
        )
    return WorkspaceError(f"'{path}' {reason}. Retrying will not help.")


def _document_window(
    extracted: documents.Text, start: int, count: int, start_char: int = 0
) -> _Window:
    """Lines ``start..start+count-1`` of a document's converted text. The first
    window also says what the conversion left out."""
    window = _window(_document_lines(extracted.text), start, count, start_char)
    omitted = _omitted(extracted) if start == 1 and not start_char else ""
    return replace(window, document=True, omitted=omitted)


def _omitted(extracted: documents.Text) -> str:
    """A line saying what a PDF's text leaves out, or nothing.

    Each page not read and each image left out is marked on its page, but a
    later window may hold the mark, so it is said up front too, naming the
    pages: told only how many, a model reads on hunting for them.
    """
    pages, images = extracted.pages_not_read, extracted.images_not_read or 0
    if not pages and not images:
        return ""
    listed = []
    if pages:
        which = "is" if len(pages) == 1 else "are"
        listed.append(
            f"{_page_numbers(pages)} of its {extracted.page_count} pages, which {which} not text"
        )
    if images:
        listed.append(f"{images} image{'' if images == 1 else 's'} on its pages of text")
    each = "It is" if len(pages) + images == 1 else "Each is"
    return f"[Not included: {', and '.join(listed)}. {each} marked on its page.]"


def _needs_ocr(error: documents.DocumentError) -> str:
    """Why a PDF whose pages need OCR could not be read. Only when they are all
    its pages is it said to have no text: one the page reader cannot open is
    converted whole by anydoc, which refuses it for any page without text."""
    pages, count = error.pages, error.page_count
    if not pages or not count:
        return "could not be converted: it is scanned or made of images, and only text is read"
    if len(pages) == count:
        which = "its only page is" if count == 1 else f"all its {count} pages are"
        return f"has no text on any page: {which} scanned or made of images, and only text is read"
    verb = "is" if len(pages) == 1 else "are"
    return (
        f"could not be converted: {_page_numbers(pages)} of its {count} pages {verb} "
        "scanned or made of images, and only text is read"
    )


def _page_numbers(pages: tuple[int, ...]) -> str:
    """``page 32``, ``pages 2, 5-7``: consecutive pages joined into a run, and
    the runs past ``_LISTED_RUNS`` counted rather than listed."""
    runs: list[list[int]] = []
    for page in pages:
        if runs and page == runs[-1][-1] + 1:
            runs[-1].append(page)
        else:
            runs.append([page])
    shown, rest = runs[:_LISTED_RUNS], runs[_LISTED_RUNS:]
    listed = ", ".join(str(run[0]) if len(run) == 1 else f"{run[0]}-{run[-1]}" for run in shown)
    if rest:
        listed += f" and {sum(len(run) for run in rest)} more"
    return f"page {listed}" if len(pages) == 1 else f"pages {listed}"


def _document_lines(text: str) -> Iterator[str]:
    """The converted text as lines no window has to cut.

    A converter's line is a paragraph, and a dense page can make one longer
    than a whole window. Cut the way a text file's line is cut, the rest of it
    could never be read, so it is split here instead: a document's line numbers
    count lines of the converted text anyway, never lines of the file.
    """
    for line in io.StringIO(text):
        while len(line) > MAX_READ_CHARS:
            yield line[:MAX_READ_CHARS]
            line = line[MAX_READ_CHARS:]
        yield line


def _megabytes(size: int) -> str:
    return f"{size / (1024 * 1024):.1f} MB"


def build_tools(workspace: Workspace) -> dict[str, Callable[..., str]]:
    """Build every file callable bound to one workspace."""

    def read_file(path: str, start_line: int = 1, max_lines: int = 0, start_char: int = 0) -> str:
        """Read a text file or a document, or a range of lines from it.

        Large files come back in pieces: read a window, then call again with
        start_line set past the end of what you got. A line too long for one
        read is cut, and the note after it gives the start_char to go on with.

        PDF, Word, Excel, PowerPoint, OpenDocument, RTF and EPUB files come back
        converted to markdown: headings as # headings, tables as | tables |, a
        spreadsheet as one table per sheet. Their line numbers count lines of
        that text, not of the file. A PDF's pages each start with a
        "<!-- Page N -->" line, N counting the file's pages from 1, not the
        numbers printed on them. What a PDF holds that is not text, a scanned
        page or a large image, is not read: it is marked on its page with a
        line in [brackets] saying why, and the first lines say what was left
        out. A document cannot be changed with write_file, append_file or
        edit_file, which write plain text.

        Args:
            path: Path to the file, relative to the workspace root.
            start_line: First line to return, counting from 1.
            max_lines: How many lines to return. 0 uses the default.
            start_char: Where to start inside start_line, in characters from its
                start. Only for a line cut because it was too long for one read:
                the note after it gives the value.
        """
        resolved = workspace.resolve(path, must_exist=True)
        if resolved.is_dir():
            raise WorkspaceError(f"'{path}' is a directory. Use list_dir for that.")

        start = _positive_int(start_line, 1)
        count = _positive_int(max_lines, DEFAULT_READ_LINES)
        offset = _positive_int(start_char, 0)

        window = _read_file(path, resolved, start, count, offset)

        if window.line_length is not None:
            raise WorkspaceError(
                f"'{path}' line {start} has {window.line_length} characters, so there is "
                f"nothing at character {offset}."
            )
        if not window.lines:
            if window.last_seen == 0:
                # A document with no text in it is not an empty file.
                return "(no text in this document)" if window.document else "(empty file)"
            # An empty result and a genuinely empty file look identical to a
            # model; saying how long the file is turns a dead end into a retry.
            raise WorkspaceError(
                f"'{path}' has {window.last_seen} lines, so there is nothing at line {start}."
            )

        body = "".join(window.lines)
        if window.omitted:
            body = f"{window.omitted}\n\n{body}"
        first, final = start, start + len(window.lines) - 1
        if not window.more and start == 1 and not offset:
            return body

        span = f"lines {first}-{final}"
        if offset:
            span += f", line {first} from character {offset}"
        if window.continues_at is not None:
            # Continuing from the next line would silently skip the rest of
            # the one that was cut: the way on is inside it.
            note = (
                f"\n\n[{span}; line {final} continues — read on with "
                f"start_line={final}, start_char={window.continues_at}]"
            )
        elif window.more:
            note = f"\n\n[{span}; more follow — read on with start_line={final + 1}]"
        else:
            note = f"\n\n[{span}, end of file]"
        return body.rstrip("\n") + note

    def list_dir(path: str = ".") -> str:
        """List the contents of a directory.

        Args:
            path: Directory to list, relative to the workspace root.
        """
        resolved = workspace.resolve(path, must_exist=True)
        if not resolved.is_dir():
            raise WorkspaceError(f"'{path}' is not a directory.")
        entries = sorted(resolved.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        lines = [
            f"{'dir ' if entry.is_dir() else 'file'}  {workspace.display(entry)}"
            for entry in entries[:MAX_LISTED]
            if not is_ignored(entry, workspace.root)
        ]
        if len(entries) > MAX_LISTED:
            lines.append(f"... and {len(entries) - MAX_LISTED} more")
        return "\n".join(lines) or "(empty)"

    def glob_files(pattern: str, path: str = ".") -> str:
        """Find files by name pattern, e.g. '**/*.py' or 'src/*.ts'.

        Args:
            pattern: Glob pattern, relative to `path`. Not an absolute path and no '..'.
            path: Folder to search in, relative to the workspace root or an absolute
                path inside one of the allowed folders.
        """
        base = workspace.resolve(path, must_exist=True)
        if not base.is_dir():
            raise WorkspaceError(f"'{path}' is not a directory.")
        found = [
            workspace.display(p)
            for p in _matches(workspace, base, pattern, "pattern")
            if p.is_file()
        ]
        if not found:
            return f"No files match {pattern!r}."
        if len(found) > MAX_LISTED:
            return "\n".join(found[:MAX_LISTED]) + f"\n... and {len(found) - MAX_LISTED} more"
        return "\n".join(found)

    def grep(pattern: str, path: str = ".", glob: str = "**/*") -> str:
        """Search file contents for a regular expression.

        Only text files are searched: PDF, Word, Excel and other documents are
        not, so read those with read_file instead.

        Args:
            pattern: Regular expression to search for.
            path: The directory to search in, or one file, relative to the
                workspace root.
            glob: Which files under path to search. Without a '/' it matches the
                file name at any depth, e.g. '*.py'; with one, the path from the
                workspace root, e.g. 'src/**/*.py'. Not used when path is a file.
        """
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise WorkspaceError(f"{pattern!r} is not a valid regular expression: {exc}") from None

        scope = workspace.resolve(path, must_exist=True)
        candidates = [scope] if scope.is_file() else _files_to_search(workspace, scope, glob)
        hits: list[str] = []
        searched = 0
        for candidate in candidates:
            try:
                text = candidate.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                # Binary files and unreadable ones are skipped rather than
                # reported: a search that lists every image it could not read is
                # a search whose output nobody reads.
                continue
            searched += 1
            for number, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    hits.append(f"{workspace.display(candidate)}:{number}: {line.strip()[:200]}")
                    if len(hits) >= MAX_MATCHES:
                        hits.append(f"... stopped at {MAX_MATCHES} matches; narrow the search")
                        return "\n".join(hits)
        if hits:
            return "\n".join(hits)
        # Two different answers. "No matches" read after a search that read no
        # file is taken for "the text is not there", when the arguments were wrong.
        where = workspace.display(scope)
        if not searched:
            if scope.is_file():
                return f"Nothing was searched: {where!r} is not a text file."
            return f"Nothing was searched: no text file under {where!r} matches the glob {glob!r}."
        files = "1 file" if searched == 1 else f"{searched} files"
        return f"No matches for {pattern!r} in {files} under {where!r}."

    def write_file(path: str, content: str) -> str:
        """Create a file or replace its entire contents.

        Args:
            path: Path to write, relative to the workspace root.
            content: The full new contents of the file.
        """
        resolved = workspace.resolve(path)
        _refuse_text_over_document(path, resolved)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        existed = resolved.exists()
        resolved.write_text(content, encoding="utf-8")
        verb = "Overwrote" if existed else "Created"
        return f"{verb} {workspace.display(resolved)} ({len(content)} characters)."

    def append_file(path: str, content: str) -> str:
        """Add text to the end of a file, creating it if needed.

        Use this to write a long file in several smaller pieces instead of one
        large call.

        Args:
            path: Path to append to, relative to the workspace root.
            content: Text to add at the end.
        """
        resolved = workspace.resolve(path)
        _refuse_text_over_document(path, resolved)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with resolved.open("a", encoding="utf-8") as handle:
            handle.write(content)
        return f"Appended {len(content)} characters to {workspace.display(resolved)}."

    def edit_file(path: str, old: str, new: str) -> str:
        """Replace an exact piece of text in a file, once.

        The old text must appear exactly once, so include enough surrounding
        context to make it unique.

        Args:
            path: File to edit, relative to the workspace root.
            old: Exact text to replace.
            new: Text to put in its place.
        """
        resolved = workspace.resolve(path, must_exist=True)
        _refuse_text_over_document(path, resolved)
        content = _read_text(path, workspace)
        occurrences = content.count(old)
        if occurrences == 0:
            raise WorkspaceError(
                f"That text does not appear in {workspace.display(resolved)}. "
                "Read the file again — it may not be what you expect."
            )
        if occurrences > 1:
            raise WorkspaceError(
                f"That text appears {occurrences} times in {workspace.display(resolved)}. "
                "Include more surrounding context so the match is unique."
            )
        resolved.write_text(content.replace(old, new, 1), encoding="utf-8")
        return f"Edited {workspace.display(resolved)}."

    return {
        "read_file": read_file,
        "list_dir": list_dir,
        "glob_files": glob_files,
        "grep": grep,
        "write_file": write_file,
        "append_file": append_file,
        "edit_file": edit_file,
    }


def file_tools(workspace: Workspace) -> list[Any]:
    """Every file tool, gated according to what it can damage."""
    from domyn.agents.tools.function_tool import FunctionTool

    built = build_tools(workspace)
    gated = {"write_file", "append_file", "edit_file"}

    tools: list[Any] = []
    for name, func in built.items():
        tools.append(
            FunctionTool(
                func=func,
                approval_type=ApprovalType.USER_CONFIRM if name in gated else None,
                # The path, so "keep editing this file" is expressible and
                # "write anywhere" is not.
                approval_subject=["path"] if name in gated else None,
            )
        )
    return tools
