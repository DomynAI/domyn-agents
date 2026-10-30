"""A file's text, whatever kind of file it is.

Everything that shows a model the contents of a file has the same questions to
answer: is it text, or a document to convert to markdown first; what does a
failed conversion mean; what did a conversion leave out; would writing text
over it destroy it. This module answers them once, so a file reads the same
wherever it is read — a local workspace, an object store, a sandbox's mounts.
Only the answers live here: size limits, windows, caching and the words shown to
the model belong to the caller, because they depend on where the file is and who
reads it. The exception is the line a PDF's text carries for each thing it
leaves out, which has to stand where that thing stood.

Conversion happens on this machine only: reading a file must never upload it
anywhere, so nothing here goes to anydoc's hosted OCR. ``firecrawl-anydoc``
converts every format but PDF. A PDF goes page by page to anydoc's own PDF
engine, ``pdf-inspector``, because anydoc returns a PDF only whole and judges
whether its pages have text from a sample of eight of them: a page without text
outside the sample vanished from the text without a word, and one inside it
made the whole document unreadable. pdf-inspector runs in a child process, one
per PDF (``domyn/agents/scripts/pdf_reader.py``): it holds the GIL through each
of its calls, a second and more on a long document, which in the caller's
process would stop every other thread for as long. A PDF still being read
after ``_READ_SECONDS``, or needing more than ``_MEMORY_BYTES`` of memory
(enforced on Linux), is stopped and refused as ``resource_limit``. At most two
are read at once (``_READERS``), so a host has to leave room for twice
``_MEMORY_BYTES`` beyond its own memory: under Kubernetes a container that runs
out of memory loses every process in it, the caller included. A process that
cannot start the reader — a frozen or compiled build (PyInstaller, Nuitka),
whose ``sys.executable`` is the application itself or no file at all — reads the
PDF in its own process instead: the same pages, without that isolation. One
pdf-inspector cannot open at all is asked of anydoc, which says why by the type
of its error, or reads it whole.

The rules, in the order they apply:

- A name anydoc knows as a document format — PDF, Word, Excel, PowerPoint,
  OpenDocument, RTF, EPUB, and variants such as ``.docm`` — is converted, and
  the content decides the format. What is not the document its name claims is
  read as text when it is plainly text: "export to Excel" writes tab-separated
  text and calls it ``.xls``. Plainly means UTF-8 with no NUL byte and no
  document's signature, so a damaged document is still reported as one.
- A CSV is read as written, like any text file. One that is not UTF-8 — an
  Excel export in Windows-1252 — is converted to a table, because anydoc
  detects its encoding and nothing here could.
- Any other file is its own text when it is UTF-8.
- A file that is none of those is converted when its content is a document
  under another name (no extension, ``.bin``, a template such as ``.dotx``),
  and is not text otherwise.

A PDF is read page by page, each page under a ``<!-- Page N -->`` line, N
counting its pages from 1, so a reader can cite a page. What a page holds that
is not text is said, never dropped in silence:

- A scan whose scanner left the text it recognized on the page, invisible over
  the image, is read from that text, with a line saying it may contain
  recognition errors.
- Any other page that is not text — scanned, made of images, or drawn without
  characters — gets a line saying why it is not read, followed by the text
  that is written on it, if any: a stamp or a protocol number on a scan.
  Nothing here reads the text in an image.
- A page pdf-inspector does not take as text although text is written on it —
  a few characters it could not map, a table of contents' dot leaders, lines
  it took for page numbers — is read as text all the same.
- A character that could not be decoded shows as U+FFFD, and a line after its
  page's text counts them.
- An image on a page of text gets a line after that page's text when it is
  about a tenth of an A4 page or larger: a figure, a screenshot, a photographed
  table.
- A page with neither text nor images says so, and that a drawing on it, if
  any, is not read.
- A PDF fails, as ``needs_ocr``, only when no page has text to read.

Of the text a PDF draws but nobody sees, text drawn invisible (rendering mode
3) is read only as a scan's recognition layer, which is where a scanner puts
what it read. On a page of text it is left out, as it is where an instruction
meant only for a model hides; a scan's layer is read for what it is, so it can
still carry one. Text hidden any other way — white on white, half a point high,
clipped, covered by a shape, outside the page, in a layer switched off — is read
as if it were visible, so a PDF can still hand a model words its reader never
sees. Nor is all a reader sees read: an annotation (a comment box, an
"APPROVED" stamp) and a form drawn only by XFA are not. And the lines this
module adds cannot be told from the PDF's own text: a PDF can write a
``<!-- Page N -->`` or a ``[Page not included: …]`` line itself.
"""

import json
import os
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import anydoc

from domyn.agents.logger import get_logger

__all__ = [
    "DocumentError",
    "FailureKind",
    "NotTextError",
    "Text",
    "document_format",
    "extract_text",
    "extract_text_bytes",
]

FailureKind = Literal["encrypted", "needs_ocr", "no_text", "resource_limit", "corrupt"]
"""Why a document could not be converted. Each is a property of the file, so
retrying cannot help.

- ``encrypted``: it needs a password.
- ``needs_ocr``: a PDF none of whose pages has text to read — each is scanned,
  made of images or drawn without characters; ``pages`` names them. A PDF with
  any text converts, its other pages marked. The exception is a PDF the page
  reader cannot open, which anydoc converts whole or not at all: it refuses
  one with any page without text, so ``pages`` may name only some of the
  ``page_count``, and the others have text that could not be read.
- ``no_text``: a format the converter knows but cannot take any text from.
- ``resource_limit``: its structure crosses one of the converter's safety
  limits (nesting depth, compression ratio), however small the file, or a PDF
  is still being read after ``_READ_SECONDS`` or needs more memory than
  ``_MEMORY_BYTES``.
- ``corrupt``: damaged, truncated, or not the kind of file its name claims.
"""

_PDF_HEAD = 1024
"""A PDF's signature is within its first kilobyte, so that much tells one apart."""

_MIN_IMAGE_AREA = 48_000.0
"""The size, in square points, an image on a page of text has to reach to be
marked: about a tenth of an A4 or US Letter page, 7.7 cm square. A figure or a
screenshot reaches it, a logo or an icon does not."""

_READER = Path(__file__).parents[1] / "scripts" / "pdf_reader.py"
"""The script that reads a PDF in a child process, run with this interpreter."""

_READ_SECONDS = 30.0
"""How long reading a PDF may take before it is stopped. The slowest PDF under
25 MB measured, a hundred pages of vector drawings, took 8 seconds."""

_MEMORY_BYTES = 1 << 30
"""The memory reading one PDF may take, enforced on Linux. The most a real PDF
measured took was 212 MB, a datasheet of 75 pages drawn with 160 000 path
operators. One of 4 MB built to inflate to 2 GiB of drawing operators stops here
within a second, where it took 4 GB."""

_READERS = threading.BoundedSemaphore(2)
"""PDFs read at once by this process, each by a child of its own: whatever the
calls asking, reading PDFs takes at most twice ``_MEMORY_BYTES``."""

_READ_WHOLE = (
    "[Read whole, not page by page: the page reader cannot open this PDF. Its "
    "pages are not marked, and a page with no text may be missing without a mark.]"
)
_LAYER = (
    "[A scanned page, read from the text its scanner recognized and left on it, "
    "invisible: it may contain recognition errors.]"
)
_SHAPES = "[Some of this page may be drawn as shapes, not written as text, and is not included.]"
_EMPTY = "[No text or images on this page; a drawing on it, if any, is not read.]"
_SCANNED = "it is scanned or made of images, and only text is read"
_UNREAD = {
    "vector_text": "its text is drawn as shapes, not written as characters",
    "suspected_garbled_text": "its text cannot be decoded",
}
"""Why a page with no text written on it is not read, by pdf-inspector's reason
for it; with no reason, or another, it is scanned or made of images."""


@dataclass(frozen=True)
class Text:
    """A file's text, as a model is shown it."""

    text: str
    converted: bool
    """A document's text converted to markdown, not the file's own: its lines
    are not the file's lines, and writing it back would replace the document."""
    pages_not_read: tuple[int, ...] = ()
    """1-indexed pages of a PDF that are not text — scanned, made of images, or
    drawn without characters — and are not read: each is marked in ``text``,
    saying why. A scan read from its recognition layer is not one of them. A
    caller showing ``text`` a window at a time says so up front: a later window
    may be the first to mark them."""
    images_not_read: int | None = None
    """Images on a PDF's pages of text whose content is not in ``text``, each
    marked after its page's text. ``None`` for any other file, and for a PDF
    read whole: their images are not counted."""
    page_count: int | None = None
    """Pages in the document, for a PDF read page by page; ``None`` otherwise."""


class NotTextError(Exception):
    """Neither text nor a document the converter recognizes: a binary file."""


class DocumentError(Exception):
    """A document that cannot be converted, and why (``kind``).

    The message is the converter's own, meant for logs; what a model is told
    is the caller's to word.
    """

    def __init__(
        self,
        kind: FailureKind,
        message: str,
        *,
        pages: tuple[int, ...] = (),
        page_count: int | None = None,
    ) -> None:
        super().__init__(message)
        self.kind: FailureKind = kind
        self.pages = pages
        """1-indexed pages not read, for ``needs_ocr``; empty otherwise."""
        self.page_count = page_count
        """Pages in the document, for ``needs_ocr``; ``None`` otherwise."""


def extract_text(path: str | PathLike[str]) -> Text:
    """The text a model is shown for the file at ``path``: its own, or a
    document's as markdown.

    A file whose name says it is a document is handed to its converter by its
    path, so its bytes are never held beside its text. Empty text is returned
    as it is: a document that converts to nothing is not an error, and only the
    caller knows how to say so.

    Raises:
        DocumentError: a document that cannot be converted.
        NotTextError: a file that is neither text nor a document.
    """
    path = Path(path)
    return _extract(_File(name=path, path=path))


def extract_text_bytes(data: bytes, name: str | PathLike[str]) -> Text:
    """``extract_text`` for a file held in memory, known by its ``name``.

    For a file with no local path, such as one fetched from an object store.
    The name matters as much as for a path: it decides what is tried first,
    and it is the only way to tell a CSV, which has no signature.
    """
    return _extract(_File(name=name, data=data))


def document_format(data: bytes, name: str | PathLike[str]) -> anydoc.Format | None:
    """The document a file holds, when reading shows it converted.

    What a write tool asks before writing text over a file: the model was
    shown a document's converted text, and writing that back would replace the
    document instead of changing it. Decided in the order reading decides, so a
    document under another name is one, text under a document's name is not,
    and a CSV is one only when it is not UTF-8. A document that fails to
    convert still is one: writing text over it would destroy it all the same.
    """
    named = _named_format(name)
    if named is None:
        if _utf8(data) is not None:
            return None
        return anydoc.format_from_bytes(data) or anydoc.format_from_path(name)
    found = anydoc.format_from_bytes(data)
    if found is not None:
        return found
    return None if _plain_text(data) is not None else named


def _named_format(name: str | PathLike[str]) -> anydoc.Format | None:
    """The format a name says a file is converted as. A CSV is text, so it has none."""
    found = anydoc.format_from_path(name)
    return None if found == "csv" else found


def _utf8(data: bytes) -> str | None:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _plain_text(data: bytes) -> str | None:
    """Its text, when a file that did not convert is text rather than a damaged
    document: UTF-8 with no NUL byte, the mark of a container's fragment, and
    no document's signature at its start."""
    if b"\x00" in data or anydoc.format_from_bytes(data) is not None:
        return None
    return _utf8(data)


@dataclass(frozen=True)
class _File:
    """A file known by its path, or by its bytes when there is no path."""

    name: str | PathLike[str]
    path: Path | None = None
    data: bytes = b""
    """The file's bytes, used only when ``path`` is ``None``."""

    def read(self) -> bytes:
        return self.path.read_bytes() if self.path is not None else self.data

    def head(self) -> bytes:
        if self.path is None:
            return self.data[:_PDF_HEAD]
        with self.path.open("rb") as handle:
            return handle.read(_PDF_HEAD)

    def convert(self, named: anydoc.Format) -> Text:
        """A file whose name says it is a document, converted.

        The content decides the format and the name is the fallback, for a
        container the content does not settle. A PDF is told apart by its
        first kilobyte and read page by page. Any other document goes to
        anydoc, by its path when it has one, so the converter reads the file
        itself.
        """
        if anydoc.format_from_bytes(self.head()) == "pdf":
            return _pdf(self.path if self.path is not None else self.data)
        try:
            if self.path is not None:
                markdown = anydoc.to_markdown(self.path, ocr="reject")
            else:
                found = anydoc.format_from_bytes(self.data) or named
                markdown = anydoc.to_markdown_bytes(self.data, found, ocr="reject")
        except anydoc.ConvertError as error:
            raise _failure(error) from error
        return Text(markdown, converted=True)


def _extract(file: _File) -> Text:
    named = _named_format(file.name)
    if named is not None:
        try:
            return file.convert(named)
        except DocumentError as error:
            # Not the document its name claims, or a damaged one: text when it
            # is plainly text, the damaged document it looks like otherwise.
            text = _plain_text(file.read()) if error.kind == "corrupt" else None
            if text is None:
                raise
            return Text(text, converted=False)

    data = file.read()
    text = _utf8(data)
    if text is not None:
        return Text(text, converted=False)
    # Told from the bytes already read, so a binary file is read once. The name
    # can only add a CSV, which has no signature.
    found = anydoc.format_from_bytes(data) or anydoc.format_from_path(file.name)
    if found is None:
        raise NotTextError(f"{file.name!s} is neither UTF-8 text nor a document")
    if found == "pdf":
        return _pdf(data)
    try:
        return Text(anydoc.to_markdown_bytes(data, found, ocr="reject"), converted=True)
    except anydoc.ConvertError as error:
        raise _failure(error) from error


def _failure(error: anydoc.ConvertError) -> DocumentError:
    """The converter's exception, as the kind of failure it is.

    Dispatched on the exception's type, never its message, so a reworded
    message cannot change what a failure means. Anything unrecognized is
    ``corrupt``: the one outcome whose advice holds for every failure.
    """
    message = str(error)
    if isinstance(error, anydoc.EncryptedError):
        return DocumentError("encrypted", message)
    if isinstance(error, anydoc.NeedsOcrError):
        return DocumentError(
            "needs_ocr", message, pages=tuple(error.pages), page_count=error.page_count
        )
    if isinstance(error, anydoc.UnsupportedError):
        return DocumentError("no_text", message)
    if isinstance(error, anydoc.ResourceLimitError):
        return DocumentError("resource_limit", message)
    return DocumentError("corrupt", message)


def _pdf(source: Path | bytes) -> Text:
    """A PDF's text, page by page, each page under a ``<!-- Page N -->`` line.

    The line is what a reader cites a page by, and how it finds the page a mark
    further down belongs to.

    Raises:
        DocumentError: ``resource_limit`` for a PDF still being read after
            ``_READ_SECONDS`` or needing more than ``_MEMORY_BYTES``,
            ``corrupt`` for one that stops the reader, ``needs_ocr`` for one
            with no text on any page, and anydoc's reason for one pdf-inspector
            cannot open (``encrypted``, ``corrupt``...).
    """
    reply = _read_pdf(source)
    if "error" in reply:
        return _whole(source, str(reply["error"]))
    return _by_page(reply["pages"])


def _read_pdf(source: Path | bytes) -> dict[str, Any]:
    """The reader's reply for the PDF: from a child process, or from this
    one when no child can be started (``_reader_unavailable``)."""
    reason = _reader_unavailable()
    if reason is None:
        try:
            return _read_in_child(source)
        except _NotStartedError as error:
            reason = str(error)
    return _read_here(source, reason)


def _reader_unavailable() -> str | None:
    """Why this process cannot start the reader, or None when it can.

    A frozen or compiled build ships no Python to run a script with:
    PyInstaller's ``sys.executable`` is the application itself, which would
    start again, and Nuitka's names no file at all. Neither ships
    ``domyn/agents/scripts/pdf_reader.py`` as a file.
    """
    if getattr(sys, "frozen", False):
        return "a frozen build has no Python to run it"
    if "__compiled__" in globals():
        return "a compiled build has no Python to run it"
    if not os.path.isfile(sys.executable) or not os.access(sys.executable, os.X_OK):
        return "sys.executable is not a program to run it"
    if not _READER.is_file():
        return "its script is not a file"
    return None


class _NotStartedError(Exception):
    """The reader's process could not be started at all."""


def _read_in_child(source: Path | bytes) -> dict[str, Any]:
    """The reader's reply for the PDF, from a child process.

    ``-P`` keeps the reader's own folder off its import path, so nothing put
    beside it can shadow the standard library's modules or pdf-inspector.

    Raises:
        _NotStartedError: the process could not be started, so nothing was read.
    """
    command = [sys.executable, "-P", str(_READER), str(_MEMORY_BYTES)]
    if isinstance(source, Path):
        command.append(str(source))
    data = source if isinstance(source, bytes) else None
    try:
        with _READERS:
            done = subprocess.run(
                command,
                input=data,
                stdin=subprocess.DEVNULL if data is None else None,
                capture_output=True,
                timeout=_READ_SECONDS,
                check=False,
            )
    except subprocess.TimeoutExpired as error:
        raise DocumentError(
            "resource_limit", f"still being read after {_READ_SECONDS:.0f} seconds"
        ) from error
    except OSError as error:
        # Before anything ran: no reader, rather than a file it could not read.
        raise _NotStartedError(f"starting it failed ({type(error).__name__})") from error
    if done.returncode < 0 and -done.returncode in (signal.SIGABRT, signal.SIGKILL):
        # Aborted by an allocation past _MEMORY_BYTES, or killed by the system
        # for memory: a PDF too large to read here, not a damaged one.
        get_logger().warning(f"documents: the PDF reader ran out of memory ({done.returncode})")
        raise DocumentError(
            "resource_limit", f"needs more than {_MEMORY_BYTES >> 20} MiB of memory to read"
        )
    reply = _reply(done.stdout) if done.returncode == 0 else None
    if reply is None:
        # A crash: what stops pdf-inspector is a property of the file, like
        # anything else it cannot read.
        get_logger().warning(f"documents: the PDF reader failed with exit status {done.returncode}")
        get_logger().debug(f"PDF reader detail: {done.stderr.decode(errors='replace')[-4000:]}")
        raise DocumentError("corrupt", f"the PDF reader stopped (exit status {done.returncode})")
    return reply


_READ_HERE: set[str] = set()
"""Why PDFs are read in this process rather than a child, each logged once."""


def _read_here(source: Path | bytes, reason: str) -> dict[str, Any]:
    """The reader's reply for the PDF, read in this process, because
    ``reason``.

    The same pages as the child reads, without its isolation: pdf-inspector
    holds the GIL while it reads, every other thread waiting, and neither
    ``_READ_SECONDS`` nor ``_MEMORY_BYTES`` applies. Imported here, so a
    compiled build bundles the module and pdf-inspector with it.
    """
    if reason not in _READ_HERE:
        _READ_HERE.add(reason)
        get_logger().warning(
            f"documents: reading PDFs in this process, as the PDF reader cannot be started: {reason}"
        )
    from domyn.agents.scripts import pdf_reader

    data = source.read_bytes() if isinstance(source, Path) else source
    with _READERS:
        try:
            return pdf_reader.read(data)
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except MemoryError as error:
            raise DocumentError("resource_limit", "ran out of memory reading it") from error
        except BaseException as error:
            # What would have stopped the child: a panic in pdf-inspector's Rust
            # code reaches Python as a BaseException.
            get_logger().warning(f"documents: the PDF reader failed: {type(error).__name__}")
            get_logger().debug(f"PDF reader detail: {error}")
            raise DocumentError(
                "corrupt", f"the PDF reader stopped ({type(error).__name__})"
            ) from error


def _reply(out: bytes) -> dict[str, Any] | None:
    try:
        reply = json.loads(out)
    except ValueError:
        return None
    if isinstance(reply, dict) and ("error" in reply or "pages" in reply):
        return reply
    return None


def _whole(source: Path | bytes, refusal: str) -> Text:
    """A PDF pdf-inspector cannot open, asked of anydoc instead.

    pdf-inspector says why in words only. anydoc says it by the type of its
    exception, so an encrypted PDF is told from a damaged one without reading a
    message. It also reads some files pdf-inspector refuses, but only whole:
    the text then says so first, since it judges from a sample of the pages
    whether they have text, and a page without may be missing without a mark.
    """
    get_logger().info("documents: the PDF page reader cannot open a file; converting it whole")
    get_logger().debug(f"PDF page reader refusal: {refusal}")
    try:
        if isinstance(source, Path):
            markdown = anydoc.to_markdown(source, ocr="reject")
        else:
            markdown = anydoc.to_markdown_bytes(source, "pdf", ocr="reject")
    except anydoc.ConvertError as error:
        raise _failure(error) from error
    except OSError as error:
        # What anydoc raises, unwrapped, for a part it cannot decode: a
        # cross-reference stream whose PNG predictor is broken, say.
        raise DocumentError("corrupt", str(error)) from error
    if not markdown.strip():
        return Text("", converted=True)
    return Text(f"{_READ_WHOLE}\n\n{markdown}", converted=True)


@dataclass(frozen=True)
class _Shown:
    """What one page of a PDF shows the model."""

    text: str
    read: bool
    """It holds text from the page."""
    unread: bool = False
    """What the page holds is not text, and is not read."""
    images: int = 0
    """Images on it marked as not read."""


def _by_page(pages: list[dict[str, Any]]) -> Text:
    """A PDF's pages joined, each under its ``<!-- Page N -->`` line, from
    the reader's reply.

    Raises:
        DocumentError: ``needs_ocr``, when no page has text to read. A document
            of nothing but lines saying so would be no reading at all.
    """
    count = len(pages)
    parts: list[str] = []
    unread: list[int] = []
    images = 0
    read = False
    for number, page in enumerate(pages, start=1):
        shown = _page(page)
        parts += [f"<!-- Page {number} -->", shown.text]
        read = read or shown.read
        images += shown.images
        if shown.unread:
            unread.append(number)
    if not read:
        if unread:
            raise DocumentError(
                "needs_ocr",
                f"no page has text to read: {len(unread)} of its {count} pages are "
                "scanned or made of images",
                pages=tuple(unread),
                page_count=count,
            )
        # Nothing but empty pages: no text, which is not an error.
        return Text("", converted=True, images_not_read=0, page_count=count)
    return Text(
        "\n\n".join(parts) + "\n",
        converted=True,
        pages_not_read=tuple(unread),
        images_not_read=images,
        page_count=count,
    )


def _page(page: dict[str, Any]) -> _Shown:
    """What a page shows, from what the reader found on it."""
    reason, written = page["reason"], page["written"]
    if page["markdown"]:
        return _text_page(page["markdown"], page["images"])
    if reason == "invisible_text_layer" and page["layer"]:
        # Its image is the page itself, so no image is marked.
        return _Shown(_counted(f"{_LAYER}\n\n{page['layer']}"), read=True)
    scan = reason in ("scanned", "invisible_text_layer")
    if not scan and written:
        # Text is written on it, whatever pdf-inspector judged: a few
        # characters it could not map, dot leaders, lines it took for page
        # numbers. Read from the positioned text, a line per baseline.
        return _text_page(written, page["images"], shapes=reason == "vector_text")
    if not scan and not page["images"] and reason not in _UNREAD:
        return _Shown(_EMPTY, read=False)
    why = _UNREAD.get(reason or "", _SCANNED)
    if written:
        line = f"[Page not included: {why}. Only the text written on it follows.]"
        return _Shown(_counted(f"{line}\n{written}"), read=True, unread=True)
    return _Shown(f"[Page not included: {why}.]", read=False, unread=True)


def _text_page(text: str, sizes: list[list[float]], *, shapes: bool = False) -> _Shown:
    """A page read as text, with a line for what it leaves out: what is drawn
    as shapes, and the images large enough to hold content."""
    large = sum(1 for width, height in sizes if width * height >= _MIN_IMAGE_AREA)
    parts = [_counted(text)]
    if shapes:
        parts.append(_SHAPES)
    if large == 1:
        parts.append("[Not included: an image on this page.]")
    elif large:
        parts.append(f"[Not included: {large} images on this page.]")
    return _Shown("\n\n".join(parts), read=True, images=large)


def _counted(text: str) -> str:
    """``text``, followed by a line counting the characters in it that could
    not be decoded, which pdf-inspector writes as U+FFFD: told nothing, a model
    reads "�40" as 40 rather than as a number whose sign is lost."""
    lost = text.count("\ufffd")
    if not lost:
        return text
    which = "character" if lost == 1 else "characters"
    return f"{text}\n\n[{lost} {which} on this page could not be decoded: each shows as \ufffd.]"
