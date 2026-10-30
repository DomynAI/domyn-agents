"""A PDF's pages as pdf-inspector reads them, in a process of their own.

``domyn.agents.core.documents`` runs this file, with the interpreter it runs in,
for every PDF it reads; a process that cannot start one, a frozen or compiled
build, imports it and calls ``read`` instead. pdf-inspector holds the GIL
through each of its calls, a
second and more on a long document, so in the reader's own process it would
stop every other thread for as long: a server's other conversations, the CLI's
spinner and editor. Here it stops nothing, ``documents`` can stop it after a
time, and a crash ends this process rather than the caller's. So does running
out of memory, but only because this process limits its own on Linux
(``_limit_memory``): under Kubernetes a container that runs out of memory loses
every process in it, so a PDF built to inflate to gigabytes would otherwise end
the caller along with this one.

The first argument is the memory this process may take, in bytes; the PDF is
the file named by the second, else the bytes on standard input. The reply, on
standard output, is ``{"error": <message>}`` when
pdf-inspector cannot read the file at all, and ``{"pages": [...]}`` otherwise,
one object per page, in order:

- ``markdown``: the page's text, when pdf-inspector reads the page as text;
  empty otherwise.
- ``reason``: why it does not, as pdf-inspector names it (``scanned``,
  ``invisible_text_layer``, ``suspected_garbled_text``, ``vector_text``...);
  ``null`` when it reads the page, or gives no reason.
- ``written``: on a page it does not read, the visible text, a line per
  baseline: a stamp on a scan, or the text of a page it judged wrongly.
- ``layer``: on a scan with an invisible text layer, the layer's text.
- ``images``: each image drawn on the page, as ``[width, height]`` in points.

An exit status other than 0, or anything else on standard output, is the reader
failing. Out of memory it dies by SIGABRT, as pdf-inspector's own code does when
an allocation fails. Only the standard library and pdf_inspector are imported,
so starting this costs about what starting Python does.
"""

import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any

import pdf_inspector

_INVISIBLE = 3
"""The PDF text rendering mode that neither fills nor strokes: invisible text."""

_WHOLE_PAGE = [[0.0, 0.0, 1e6, 1e6]]
"""A region, in points from the page's top-left corner, larger than any page."""

_IMAGE_PLACEHOLDER = re.compile(r"\[Image: [^\]]*\]")
"""What pdf-inspector writes in a region's text where an image is drawn."""


def read(data: bytes) -> dict[str, Any]:
    """The reply for the PDF in ``data``, as the module docstring describes it."""
    try:
        extracted = pdf_inspector.extract_pages_markdown_bytes(data).pages
    except ValueError as error:
        # pdf-inspector's one exception for a file it cannot read.
        return {"error": str(error)}
    pages: list[dict[str, Any]] = []
    for page in extracted:
        markdown = "" if page.needs_ocr else page.markdown.strip()
        reason = page.ocr_reason if page.needs_ocr else None
        pages.append(
            {
                "markdown": markdown,
                "reason": reason,
                "written": "",
                "layer": "",
                "images": [],
            }
        )
    _add_positions(data, pages)
    _add_layers(data, pages)
    return {"pages": pages}


def _add_positions(data: bytes, pages: list[dict[str, Any]]) -> None:
    """Each page's images, and the visible text on the pages not read as text.

    A second parse the pages' text does not depend on, so it is best effort:
    whatever it raises — pdf-inspector's ``ValueError``, or the
    ``BaseException`` pyo3 raises for a panic in its Rust code — costs the
    images and the stamps, never a document read already. Only memory running
    out inside that code ends this process, and the document with it.
    """
    try:
        items = pdf_inspector.extract_text_with_positions_bytes(data)
    except BaseException:
        return
    lines: dict[int, list[list[str]]] = {}
    baseline: dict[int, float] = {}
    for item in items:
        if not 1 <= item.page <= len(pages):
            continue
        page = pages[item.page - 1]
        if item.item_type == "image":
            page["images"].append([item.width, item.height])
            continue
        if page["markdown"] or item.item_type != "text" or item.render_mode == _INVISIBLE:
            continue
        text = item.text.strip()
        if not text:
            continue
        rows = lines.setdefault(item.page, [])
        if rows and abs(item.y - baseline[item.page]) < max(item.font_size, 1.0) / 2:
            rows[-1].append(text)
        else:
            rows.append([text])
        baseline[item.page] = item.y
    for number, rows in lines.items():
        pages[number - 1]["written"] = "\n".join(" ".join(row) for row in rows)


def _add_layers(data: bytes, pages: list[dict[str, Any]]) -> None:
    """The invisible text on the pages pdf-inspector calls scans with a text
    layer: what a scanner's recognition wrote over its image, read whole from a
    region larger than the page. pdf-inspector leaves out of it the short
    invisible runs that are no recognition layer."""
    wanted = [i for i, page in enumerate(pages) if page["reason"] == "invisible_text_layer"]
    if not wanted:
        return
    try:
        found = pdf_inspector.extract_text_in_regions_bytes(
            data, [(index, _WHOLE_PAGE) for index in wanted]
        )
    except BaseException:
        # Best effort, as in _add_positions: the page is then marked as a scan.
        return
    for page in found:
        if 0 <= page.page < len(pages):
            text = "\n".join(region.text for region in page.regions)
            pages[page.page]["layer"] = _tidy(text)


def _tidy(text: str) -> str:
    """A region's text without its layout: the spaces that place each line, the
    image placeholders, and runs of empty lines."""
    lines = [_IMAGE_PLACEHOLDER.sub("", line).strip() for line in text.splitlines()]
    tidy: list[str] = []
    for line in lines:
        if line or (tidy and tidy[-1]):
            tidy.append(line)
    return "\n".join(tidy).strip()


def main() -> None:
    _limit_memory(int(sys.argv[1]))
    data = Path(sys.argv[2]).read_bytes() if len(sys.argv) > 2 else sys.stdin.buffer.read()
    try:
        reply = json.dumps(read(data))
    except MemoryError:
        # Python's own allocations ran out: said the way pdf-inspector says it.
        os.abort()
    sys.stdout.write(reply)


def _limit_memory(limit: int) -> None:
    """Keep this process within ``limit`` bytes where the system enforces it,
    on Linux: an allocation past it fails here, before the container the
    caller runs in runs out.

    ``RLIMIT_DATA`` rather than ``RLIMIT_AS``: it counts the memory a process
    can write to, while a limit on address space also counts what malloc
    reserves for each thread and may never use. macOS enforces neither on the
    memory a process maps, so there nothing is set.
    """
    if platform.system() != "Linux":
        return
    import resource

    _, hard = resource.getrlimit(resource.RLIMIT_DATA)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_DATA, (limit, hard))


if __name__ == "__main__":
    main()
