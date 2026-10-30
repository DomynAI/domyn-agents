"""Markdown for the terminal, rendered a line at a time while the answer streams.

Models write markdown whether they are asked to or not, and printed as it comes
it is noise: ``**`` around every emphasis, ``##`` in front of every heading,
pipes where a table should be. This turns it into what it describes.

**Why a line at a time.** A full-screen interface can re-render the unfinished
end of an answer on every token, because it owns the screen and can repaint it.
This CLI prints into the scrollback, where a line once written stays as written;
and while the steering editor holds the bottom of the screen, a write that does
not end in a newline is erased by its next repaint (see
``EventRenderer.set_live_prompt``). So nothing is printed until it is final, and
the unit of final is the line: each source line is rendered as soon as its
newline arrives. Two things wait longer, both because what comes later changes
how what came first must look:

* **A table** is held until it ends, since its columns cannot be sized before
  every row is known — and so is a line that might be a table's header, for the
  one line it takes to find out.
* **A long line** — for prose that is a whole paragraph, since models do not wrap
  their own — goes out a visual line at a time as it overflows the width, but
  only up to its first delimiter that is still open. ``**bold`` is literal text
  until the closing ``**`` arrives, and a line printed with the asterisks in it
  cannot be taken back.

A delimiter still open when its line ends is printed as written. CommonMark lets
``**`` run across a soft line break, but a line held back for a later one to
close it would hold the whole paragraph, and models do not break a span there.

**The look**: headings without their ``#`` (from level three the marks
stay, as a depth cue), dimmed bullets and quote bars, code between visible dimmed
fences with the body indented two columns, tables with drawn borders, and a
link's address in brackets after its text when the two differ. Colour follows
``theme.py``: attributes for structure, one tint for code and links, syntax
colours inside code blocks.

Whether to render at all is the caller's decision; see ``EventRenderer``.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from markdown_it import MarkdownIt
from rich import box
from rich.cells import cell_len, get_character_cell_size
from rich.style import Style
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from rich.console import Console, JustifyMethod
    from rich.syntax import Syntax

    from domyn.agents.interfaces.cli.theme import Theme

# Block structure is decided here, from the start of each line: CommonMark cut
# down to what a model writes. Inline structure is markdown-it's.
_FENCE = re.compile(r"^(?P<indent> *)(?P<marks>`{3,}|~{3,})[ \t]*(?P<info>.*?)[ \t]*$")
_HEADING = re.compile(r"^ {0,3}(?P<marks>#{1,6})(?:[ \t]+(?P<text>.*?))?(?:[ \t]+#+)?[ \t]*$")
_RULE = re.compile(r"^ {0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})$")
_ITEM = re.compile(r"^(?P<indent> *)(?P<marker>[-*+]|\d{1,9}[.)])(?:[ \t]+(?P<body>.*))?$")
_QUOTE = re.compile(r"^ {0,3}> ?")
_DELIMITER_CELL = re.compile(r"^:?-+:?$")

_INLINE = MarkdownIt("commonmark").enable("strikethrough")
_EMPHASIS = {
    "strong_open": Style(bold=True),
    "em_open": Style(italic=True),
    "s_open": Style(strike=True),
}
_EMPHASIS_CLOSE = frozenset({"strong_close", "em_close", "s_close"})

# Lines of a code block handed to the lexer ahead of the ones being coloured, so
# a string or comment that opened earlier is still known to be one. Bounded, so a
# long block costs the same per line at its end as at its start.
_LEXER_CONTEXT = 40

# After an attempt to print part of an unfinished line finds nothing it may print,
# the characters to wait for before parsing it again.
_RETRY_AFTER = 16


def _is_row(line: str) -> bool:
    """Whether a line could belong to a table. Models always open rows with a pipe."""
    return line.lstrip().startswith("|")


def _cells(row: str) -> list[str]:
    """A table row's cells: split on pipes that are neither escaped nor inside code."""
    text = row.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|") and not text.endswith("\\|"):
        text = text[:-1]
    cells: list[str] = []
    current: list[str] = []
    ticks = 0  # length of the backtick run that opened a code span; 0 outside one
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text):
            current.append(text[index : index + 2])
            index += 2
            continue
        if char == "`":
            end = index
            while end < len(text) and text[end] == "`":
                end += 1
            run = end - index
            if not ticks:
                ticks = run
            elif run == ticks:
                ticks = 0
            current.append(text[index:end])
            index = end
            continue
        if char == "|" and not ticks:
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        index += 1
    if ticks:
        # A backtick that never closed would have swallowed every pipe after it.
        return [cell.strip() for cell in re.split(r"(?<!\\)\|", text)]
    cells.append("".join(current).strip())
    return cells


def _alignments(line: str, columns: int) -> list[JustifyMethod] | None:
    """The column alignments, when this is the delimiter row under a table's header."""
    if "|" not in line or "-" not in line:
        return None
    cells = _cells(line)
    if len(cells) != columns or not all(_DELIMITER_CELL.match(cell) for cell in cells):
        return None
    aligns: list[JustifyMethod] = []
    for cell in cells:
        if cell.startswith(":") and cell.endswith(":"):
            aligns.append("center")
        elif cell.endswith(":"):
            aligns.append("right")
        else:
            aligns.append("left")
    return aligns


def _first_open(content: str, before: str) -> int | None:
    """Where, in plain text markdown-it left alone, something could still open.

    A delimiter markdown-it did not pair is left in a text token; one that a
    later character can still pair is what makes the text so far provisional. A
    lone ``~`` is a path (``~/.domyn``), not half of ``~~``, and an ``_`` inside a
    word cannot open emphasis — both are too common to hold a line back for.
    """
    previous = before
    for index, char in enumerate(content):
        if char in "*`[":
            return index
        if char == "~" and content[index + 1 : index + 2] == "~":
            return index
        if char == "_" and not previous.isalnum():
            return index
        previous = char
    return None


def _join(*parts: Text) -> Text:
    """Concatenate, each part keeping its own style.

    ``Text + Text`` would not: the left operand's style is the base of the whole
    result, so a dimmed bullet would dim the item after it.
    """
    joined = Text()
    for part in parts:
        joined.append_text(part)
    return joined


def _styled(text: str, style: Style) -> Text:
    return Text.assemble((text, style))


def _word_start(plain: str, index: int) -> int:
    while index > 0 and not plain[index - 1].isspace():
        index -= 1
    return index


def _break(plain: str, start: int, limit: int, room: int) -> tuple[int, int]:
    """End of the visual line that starts at ``start``, and where the next begins.

    Greedy, at the last space that fits; a word wider than the whole line is cut.
    Greedy is what makes printing part of a line safe: the breaks already taken
    depend only on the text before them, never on what arrives after.
    """
    cells = 0
    last_space = -1
    index = start
    while index < limit:
        char = plain[index]
        width = get_character_cell_size(char)
        if cells + width > room:
            break
        if char == " ":
            last_space = index
        cells += width
        index += 1
    if index < len(plain) and plain[index] == " ":
        cut = index
    elif last_space > start:
        cut = last_space
    else:
        cut = max(index, start + 1)
    end = cut
    while end > start and plain[end - 1] == " ":
        end -= 1
    return end, cut


@dataclass
class _Line:
    """A line of prose on its way out: what goes in front of it, and its body."""

    first: Text  # in front of its first visual line: bullet, quote bar, heading marks
    hang: Text  # in front of every visual line after that
    offset: int  # where the body starts in the source line
    end: int | None = None  # where it stops, when not at the end (a heading's closing #s)
    base: Style = field(default_factory=Style.null)
    # The list levels open once this line is printed — (source indent, column its
    # text starts at) — or None when it leaves them as they are.
    lists: list[tuple[int, int]] | None = None
    done: int = 0  # characters of the rendered body already printed
    rows: int = 0  # visual lines already printed
    retry: int = 0  # body length below which trying to print more cannot help


@dataclass
class _Fence:
    marks: str
    indent: int
    syntax: Syntax | None  # None when the language is unknown: printed uncoloured
    context: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)


@dataclass
class _Table:
    header: list[str]
    aligns: list[JustifyMethod]
    rows: list[list[str]] = field(default_factory=list)


class MarkdownStream:
    """Markdown in, finished terminal lines out, each as soon as it is final.

    :meth:`feed` takes text in whatever pieces it arrives and returns the lines
    that became final; :meth:`finish` returns the rest. Lines come back already
    fitted to the console's width, to be printed without wrapping again — code
    excepted, which is left for the terminal to fold so that a long command is
    still one line to a copy-paste.
    """

    def __init__(
        self,
        console: Console,
        theme: Theme,
        *,
        lead: Text | None = None,
        margin: str = "",
        style: str = "",
    ) -> None:
        """``lead`` goes in front of the first line printed — the answer's marker —
        and counts against that line's width; ``margin`` goes in front of every
        line and ``style`` underneath everything, which is how the trace draws
        reasoning."""
        self._console = console
        self._lead = lead
        self._margin = margin
        self._base = Style.parse(style) if style else None

        def resolve(name: str) -> Style:
            return Style.parse(theme.style(name))

        self._code = resolve("md_code")
        self._link = resolve("md_link")
        self._link_url = resolve("md_link_url")
        self._title = resolve("md_title")
        self._heading = resolve("md_heading")
        self._quote = resolve("md_quote")
        self._marker = resolve("md_marker")
        self._palette = theme.styles.syntax
        self._symbols = theme.symbols
        self._unicode = theme.capabilities.unicode

        self._partial = ""  # everything after the last newline
        self._open: _Line | None = None  # the unfinished line, once it has been laid out
        self._fence: _Fence | None = None
        self._table: _Table | None = None
        self._header: str | None = None  # a line that may turn out to head a table
        self._lists: list[tuple[int, int]] = []
        self._blank = False  # a blank line owed before whatever is printed next
        self._started = False
        self._out: list[Text] = []

    # --- public API --------------------------------------------------------

    def feed(self, text: str) -> list[Text]:
        """Take the next piece of the source; return the lines that are now final."""
        self._partial += text
        while True:
            line, newline, rest = self._partial.partition("\n")
            if not newline:
                break
            self._partial = rest
            self._complete(line)
        self._flush_code()
        self._advance()
        return self._drain()

    def finish(self) -> list[Text]:
        """The source is over: return everything still held back."""
        return self._end(close_fence=True)

    def preview(self) -> list[Text]:
        """What :meth:`finish` would return if the source ended now, leaving it open.

        For a front-end that owns a region it can repaint: drawn under the lines
        already printed and replaced, not appended to, on the next call, it shows
        what is still held back — the line being written, a table still being
        collected — before it is final. Worked out on a copy, so the stream itself
        is untouched. A code block still open is shown without the closing fence
        :meth:`finish` would add: the model has not closed it yet.
        """
        return self._copy()._end(close_fence=False)

    def _copy(self) -> MarkdownStream:
        """A twin that can be finished without finishing this one.

        Shallow, but for what the stream changes in place: an unfinished line's
        counters, the code lines waiting to be coloured, a table's rows, and the
        output. Everything else is replaced rather than mutated, so sharing it is
        safe — the lexer included, which keeps no state between calls.
        """
        twin = copy.copy(self)
        twin._open = copy.copy(self._open)
        if self._fence is not None:
            twin._fence = replace(self._fence, pending=list(self._fence.pending))
        if self._table is not None:
            twin._table = replace(self._table, rows=list(self._table.rows))
        twin._out = []
        return twin

    def _end(self, *, close_fence: bool) -> list[Text]:
        if self._partial:
            line, self._partial = self._partial, ""
            self._complete(line)
        self._open = None
        if self._header is not None:
            header, self._header = self._header, None
            self._prose(header)
        self._flush_table()
        if self._fence is not None:
            self._flush_code()
            if close_fence:
                # Closed even though the model never closed it: the block has to
                # end somewhere, and it did end, with the answer.
                self._close_fence(self._fence.marks)
        self._blank = False
        return self._drain()

    # --- lines -------------------------------------------------------------

    def _complete(self, raw: str) -> None:
        raw = raw.rstrip("\r")
        if self._fence is not None:
            self._fence_line(raw)
            return
        line = raw.expandtabs(4)
        if self._open is not None:
            unfinished, self._open = self._open, None
            if unfinished.rows:
                # Part of it is on screen: it has to be finished in the shape it
                # was started in.
                self._settle(unfinished, line, final=True)
                return
        if self._table is not None:
            if _is_row(line):
                self._table.rows.append(_cells(line))
                return
            self._flush_table()
        if self._header is not None:
            header, self._header = self._header, None
            columns = _cells(header)
            aligns = _alignments(line, len(columns))
            if aligns is not None:
                self._table = _Table(columns, aligns)
                return
            self._prose(header)
        if not line.strip():
            self._blank = self._started
            return
        fence = _FENCE.match(line)
        if fence and not (fence["marks"][0] == "`" and "`" in fence["info"]):
            self._open_fence(fence)
            return
        if _is_row(line):
            self._header = line
            return
        self._prose(line)

    def _prose(self, raw: str) -> None:
        line = self._shape(raw)
        if line is None:
            self._lists = []
            width = self._width() - (cell_len(self._lead.plain) if self._lead else 0)
            self._emit(_styled(self._symbols.rule * max(1, width), self._marker))
            return
        self._settle(line, raw, final=True)

    def _shape(self, raw: str, *, quoted: bool = False) -> _Line | None:
        """How a line of prose is laid out. ``None`` for a thematic break."""
        quote = _QUOTE.match(raw)
        if quote:
            bar = _styled(f"{self._symbols.quote} ", self._marker)
            inner = self._shape(raw[quote.end() :], quoted=True)
            if inner is None:
                return _Line(bar, bar, offset=len(raw), lists=[])
            return _Line(
                _join(bar, inner.first),
                _join(bar, inner.hang),
                offset=quote.end() + inner.offset,
                end=None if inner.end is None else quote.end() + inner.end,
                base=self._quote + inner.base,
                lists=[],
            )
        if _RULE.match(raw):
            return None
        heading = _HEADING.match(raw)
        if heading:
            level = len(heading["marks"])
            marks = _styled(f"{'#' * level} ", self._heading) if level >= 3 else Text()
            if heading["text"] is None:
                start = end = len(raw)
            else:
                start, end = heading.start("text"), heading.end("text")
            base = self._title if level == 1 else self._heading
            return _Line(marks, Text(), offset=start, end=end, base=base, lists=[])
        item = _ITEM.match(raw)
        if item:
            indent = len(item["indent"])
            levels = [] if quoted else [level for level in self._lists if level[0] < indent]
            marker = item["marker"]
            glyph = self._symbols.bullet if marker in "-*+" else marker
            first = _join(Text("  " * len(levels)), _styled(f"{glyph} ", self._marker))
            hang = Text(" " * cell_len(first.plain))
            offset = item.start("body") if item["body"] is not None else len(raw)
            opened = None if quoted else [*levels, (indent, cell_len(first.plain))]
            return _Line(first, hang, offset=offset, lists=opened)
        indent = len(raw) - len(raw.lstrip(" "))
        if indent and self._lists and not quoted:
            # A continuation of a list item: under that item's text, not the bullet.
            owners = [level for level in self._lists if level[0] < indent]
            pad = Text(" " * (owners[-1][1] if owners else indent))
            return _Line(pad, pad, offset=indent)
        pad = Text(" " * indent)
        return _Line(pad, pad, offset=indent, lists=None if quoted else [])

    def _advance(self) -> None:
        """Print whatever of the unfinished line can no longer change."""
        if self._fence is not None or self._table is not None or self._header is not None:
            return
        raw = self._partial.expandtabs(4)
        if not raw.strip():
            return
        line = self._open
        if line is None:
            if cell_len(raw) <= self._width() or _is_row(raw) or _FENCE.match(raw):
                return
            if _HEADING.match(raw):
                return
            line = self._shape(raw)
            if line is None:
                return
            # Laid out once and kept: by the time a line overflows, whatever
            # marks its start (a bullet, a quote) has long since arrived.
            self._open = line
        body = raw[line.offset :]
        # The rendered body is never longer than its source, so while the source
        # still fits there is nothing that could be complete.
        if len(body) < line.retry or cell_len(body) - line.done <= self._room(line):
            return
        if not self._settle(line, raw, final=False):
            line.retry = len(body) + _RETRY_AFTER

    def _settle(self, line: _Line, raw: str, *, final: bool) -> bool:
        """Print the line's body, or as much of it as is settled. True if anything was."""
        end = len(raw) if line.end is None else line.end
        body = raw[line.offset : end]
        rendered, first_open = self._inline(body.rstrip() if final else body, line.base)
        plain = rendered.plain
        stop = len(plain)
        if not final and first_open is not None:
            # Up to the word that holds the open delimiter: if it closes, that word
            # gets shorter and may fit on the line before, so neither is final.
            stop = _word_start(plain, first_open)
        printed = False
        position = line.done
        while True:
            if line.rows:
                while position < stop and plain[position] == " ":
                    position += 1
            room = self._room(line)
            if cell_len(plain[position:stop]) <= room:
                if final and (position < stop or not line.rows):
                    self._row(line, rendered[position:stop])
                    position = stop
                    printed = True
                break
            cut, position_next = _break(plain, position, stop, room)
            self._row(line, rendered[position:cut])
            position = position_next
            printed = True
        line.done = position
        return printed

    def _row(self, line: _Line, piece: Text) -> None:
        row = _join(line.hang if line.rows else line.first, piece)
        if not line.rows and line.lists is not None:
            self._lists = line.lists
        line.rows += 1
        self._emit(row)

    def _room(self, line: _Line) -> int:
        prefix = line.hang if line.rows else line.first
        lead = cell_len(self._lead.plain) if self._lead is not None else 0
        return max(1, self._width() - cell_len(prefix.plain) - lead)

    def _width(self) -> int:
        # Read on every line rather than once: the window can be resized mid-answer.
        return max(1, self._console.width - cell_len(self._margin))

    # --- inline ------------------------------------------------------------

    def _inline(self, source: str, base: Style) -> tuple[Text, int | None]:
        """One line of inline markdown, and where its first still-open delimiter is.

        The offset is into the *rendered* text; ``None`` means every delimiter found
        its partner, so nothing later in the line can change how this much looks.
        """
        tokens = _INLINE.parseInline(source)
        children = (tokens[0].children or []) if tokens else []
        text = Text()
        stack = [base]
        links: list[tuple[str, int]] = []
        first_open: int | None = None
        for token in children:
            kind = token.type
            current = Style.combine(stack)
            if kind == "text":
                if first_open is None:
                    at = _first_open(token.content, text.plain[-1:])
                    if at is not None:
                        first_open = len(text.plain) + at
                text.append(token.content, current)
            elif kind == "code_inline":
                text.append(token.content, current + self._code)
            elif kind in _EMPHASIS:
                stack.append(_EMPHASIS[kind])
            elif kind in _EMPHASIS_CLOSE:
                if len(stack) > 1:
                    stack.pop()
            elif kind == "link_open":
                href = str(token.attrs.get("href", ""))
                links.append((href, len(text.plain)))
                stack.append(self._link + Style(link=href) if href else self._link)
            elif kind == "link_close":
                if len(stack) > 1:
                    stack.pop()
                href, start = links.pop() if links else ("", len(text.plain))
                label = text.plain[start:]
                if href and label not in (href, href.removeprefix("mailto:")):
                    # The address, too: in a terminal without hyperlinks the text
                    # alone is a link nobody can follow.
                    outer = Style.combine(stack)
                    text.append(" ", outer)
                    text.append(f"({href})", outer + self._link_url + Style(link=href))
            elif kind == "image":
                src = str(token.attrs.get("src", ""))
                style = current + self._link + (Style(link=src) if src else Style.null())
                text.append(token.content or src, style)
            elif kind in ("softbreak", "hardbreak"):
                text.append(" ", current)
            else:
                # Inline HTML, and whatever markdown-it adds later: as written.
                text.append(token.content, current)
        return text, first_open

    # --- code blocks -------------------------------------------------------

    def _open_fence(self, match: re.Match[str]) -> None:
        indent = len(match["indent"])
        marks = match["marks"]
        language = match["info"].split()[0] if match["info"].split() else ""
        if not indent:
            self._lists = []
        self._emit(_join(Text(" " * indent), _styled(marks + language, self._marker)))
        self._fence = _Fence(marks=marks, indent=indent, syntax=self._syntax(language))

    def _syntax(self, language: str) -> Syntax | None:
        # Imported here: it brings in Pygments, which an answer without code never needs.
        from rich.syntax import ANSISyntaxTheme, Syntax

        if not language:
            return None
        lexer = Syntax("", language).lexer
        if lexer is None:
            return None
        palette = ANSISyntaxTheme(
            {tuple(name.split(".")): Style.parse(style) for name, style in self._palette.items()}
        )
        return Syntax("", lexer, theme=palette)

    def _fence_line(self, raw: str) -> None:
        fence = self._fence
        if fence is None:
            return
        closing = raw.strip()
        mark = fence.marks[0]
        if closing and set(closing) == {mark} and len(closing) >= len(fence.marks):
            self._flush_code()
            self._close_fence(closing)
            return
        # Up to the fence's own indentation is taken off each line, as CommonMark
        # does: a block inside a list item is indented for the list, not the code.
        strip = min(fence.indent, len(raw) - len(raw.lstrip(" ")))
        fence.pending.append(raw[strip:])

    def _flush_code(self) -> None:
        """Colour and print the code lines received so far.

        Batched per :meth:`feed` rather than per line: a whole answer handed over at
        once goes through the lexer once, and a streamed one a line at a time — with
        the lines before it as context either way.
        """
        fence = self._fence
        if fence is None or not fence.pending:
            return
        lines, fence.pending = fence.pending, []
        if fence.syntax is None:
            rows = [Text(line) for line in lines]
        else:
            context = fence.context
            source = "\n".join([*context, *lines])
            coloured = fence.syntax.highlight(source).split("\n", allow_blank=True)
            rows = list(coloured)[len(context) : len(context) + len(lines)]
        fence.context = [*fence.context, *lines][-_LEXER_CONTEXT:]
        pad = " " * (fence.indent + 2)
        for row in rows:
            self._emit(_join(Text(pad), row))

    def _close_fence(self, marks: str) -> None:
        fence = self._fence
        if fence is None:
            return
        self._fence = None
        self._emit(_join(Text(" " * fence.indent), _styled(marks, self._marker)))

    # --- tables ------------------------------------------------------------

    def _flush_table(self) -> None:
        table, self._table = self._table, None
        if table is None:
            return
        self._lists = []
        grid = Table(
            box=box.ROUNDED if self._unicode else box.ASCII,
            border_style=self._marker,
            header_style="bold",
            expand=False,
        )
        columns = len(table.header)
        for header, justify in zip(table.header, table.aligns, strict=True):
            grid.add_column(self._inline(header, Style.null())[0], justify=justify)
        for cells in table.rows:
            padded = [*cells, *[""] * columns][:columns]
            grid.add_row(*(self._inline(cell, Style.null())[0] for cell in padded))
        if self._lead is not None:
            # The marker cannot sit in front of a border without pushing that one
            # row out of line with the rest: it gets a line of its own.
            self._emit(Text())
        options = self._console.options.update(width=self._width())
        for segments in self._console.render_lines(grid, options, pad=False):
            line = Text()
            for segment in segments:
                if not segment.control:
                    line.append(segment.text, segment.style)
            self._emit(line)

    # --- output ------------------------------------------------------------

    def _emit(self, row: Text) -> None:
        if self._blank and self._started:
            self._out.append(Text())
        self._blank = False
        parts = [Text(self._margin)]
        if self._lead is not None:
            parts.append(self._lead)
            self._lead = None
        row = _join(*parts, row)
        if self._base is not None:
            row.stylize_before(self._base)
        self._out.append(row)
        self._started = True

    def _drain(self) -> list[Text]:
        out, self._out = self._out, []
        return out


def render_markdown(
    source: str,
    console: Console,
    theme: Theme,
    *,
    lead: Text | None = None,
    margin: str = "",
    style: str = "",
) -> list[Text]:
    """A whole document at once: what the stream would have printed, in one go."""
    stream = MarkdownStream(console, theme, lead=lead, margin=margin, style=style)
    return stream.feed(source) + stream.finish()
