"""Tests for markdown rendered into the scrollback while the answer streams.

The contract, and the reason for each group below: nothing is printed until it
is final, so the order the text arrives in — token by token, or all at once —
must never show on screen; and "final" comes as early as it safely can, a line at
a time, because waiting for a whole block is waiting for a whole list.
"""

import io
import random
import re

import pytest
from rich.console import Console
from rich.segment import Segment
from rich.text import Text

from domyn.agents import BaseEvent, ExecutionEventType, Part
from domyn.agents.interfaces.cli import Capabilities, Theme
from domyn.agents.interfaces.cli.markdown import MarkdownStream, render_markdown
from domyn.agents.interfaces.cli.prompt import OutputProgress
from domyn.agents.interfaces.cli.renderer import EventRenderer

DOCUMENT = """# Setup

Installed **4 skills**, not 3, from `AikidoSec/aikido-claude-plugin` \
(see [the README](https://github.com/AikidoSec/aikido-claude-plugin)); they live in \
`~/.domyn/skills/` next to snake_case_names, and this sentence is long on purpose.

## What each one does

| Skill | Purpose | Tools |
|---|:---:|--:|
| `setup` | Sign-in through MCP (`aikido_login`) | 1 |
| `scan` | SAST and secrets scan of the code | 2 |

- the **skills** are installed and visible
- the MCP tools behind them cannot be wired as they are, unless through an HTTP \
proxy in front of the stdio server
  - nested one
  - nested two
    1. numbered and nested

> A quote with **emphasis**
> > and a nested one

### Third level

```json
{ "command": "npx", "args": ["-y", "@aikidosec/mcp@1.0.25"], "a_key_long_enough": "to overflow" }
```

```python
def f(x):
    \"\"\"A docstring
    that spans lines\"\"\"
    return x + 1  # comment
```

---

1. leave them installed
2. add stdio support
"""


def _theme(*, unicode: bool = True) -> Theme:
    base = Theme.build(allow_animation=False)
    return type(base)(
        capabilities=Capabilities(color=True, unicode=unicode, animation=False),
        symbols=type(base.symbols)(unicode=unicode),
        styles=base.styles,
    )


def _console(width: int = 60) -> Console:
    # height pinned: without it rich forces 80 columns wherever TERM says dumb.
    return Console(
        file=io.StringIO(),
        width=width,
        height=40,
        force_terminal=True,
        color_system="truecolor",
    )


def _stream(width: int = 60, **kwargs: object) -> MarkdownStream:
    return MarkdownStream(_console(width), _theme(), **kwargs)  # type: ignore[arg-type]


def _render(pieces: list[str], width: int = 60) -> list[Text]:
    stream = _stream(width)
    rows: list[Text] = []
    for piece in pieces:
        rows += stream.feed(piece)
    return rows + stream.finish()


def _plain(rows: list[Text]) -> list[str]:
    return [row.plain.rstrip() for row in rows]


def _visible(rows: list[Text]) -> list[list[tuple[str, str]]]:
    """What each row looks like: runs of text and the style they are drawn in.

    Not the spans: the lexer splits a run differently when handed a block whole
    than a line at a time, and two spans that draw the same are the same here.
    """
    console = _console()
    return [
        [(segment.text, str(segment.style)) for segment in Segment.simplify(row.render(console))]
        for row in rows
    ]


def _style_at(row: Text, needle: str) -> str:
    """The style the first character of ``needle`` is drawn with in ``row``."""
    return str(row.get_style_at_offset(_console(), row.plain.index(needle)))


class TestTheOrderOfArrivalNeverShows:
    """Printed once, never repainted: every piece must add to the screen exactly
    what the whole document would have put there."""

    @pytest.mark.parametrize("width", [40, 60, 100])
    def test_one_character_at_a_time(self, width: int) -> None:
        whole = _render([DOCUMENT], width)
        streamed = _render(list(DOCUMENT), width)
        assert _visible(streamed) == _visible(whole)

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_in_pieces_of_any_size(self, seed: int) -> None:
        rng = random.Random(seed)
        pieces, index = [], 0
        while index < len(DOCUMENT):
            size = rng.randint(1, 15)
            pieces.append(DOCUMENT[index : index + size])
            index += size
        whole = _render([DOCUMENT])
        assert _visible(_render(pieces)) == _visible(whole)

    def test_nothing_but_code_is_wider_than_the_window(self) -> None:
        in_code = False
        for line in _plain(_render([DOCUMENT], 40)):
            if line.strip().startswith("```"):
                in_code = not in_code
            elif not in_code:
                assert len(line) <= 40, line


class TestALineGoesOutWhenItEnds:
    def test_before_the_list_it_belongs_to_is_over(self) -> None:
        stream = _stream()
        assert _plain(stream.feed("- one\n- tw")) == ["• one"]
        assert _plain(stream.feed("o\n")) == ["• two"]

    def test_a_long_paragraph_goes_out_before_its_newline(self) -> None:
        """Models do not wrap their own paragraphs: waiting for the newline would
        be waiting for the whole paragraph."""
        stream = _stream(width=40)
        words = " ".join(f"word{n}" for n in range(40))
        early = stream.feed(words)
        rest = stream.feed("\n") + stream.finish()
        assert early, "nothing was printed until the paragraph ended"
        assert " ".join(_plain(early + rest)) == words
        assert all(len(line) <= 40 for line in _plain(early + rest))

    def test_an_open_delimiter_holds_its_line_back(self) -> None:
        """``**bold`` is literal text until the closing ``**``: printed early, the
        asterisks would stay on screen after the text turned bold."""
        stream = _stream(width=40)
        opened = stream.feed("**" + " ".join(f"bold{n}" for n in range(20)))
        assert opened == []
        rows = stream.feed("**\n") + stream.finish()
        assert "*" not in "".join(_plain(rows))
        assert "bold" in _style_at(rows[0], "bold0")

    def test_a_delimiter_still_open_at_the_newline_is_printed_as_written(self) -> None:
        """The newline makes a line final. Holding it for a later line to close
        the span would hold the whole paragraph back; the markers show instead,
        and the next line starts clean."""
        stream = _stream()
        assert _plain(stream.feed("**never closed\n")) == ["**never closed"]
        rows = stream.feed("next line\n")
        assert _plain(rows) == ["next line"]
        assert "bold" not in _style_at(rows[0], "next")


class TestBlocks:
    def test_headings_lose_their_marks_until_level_three(self) -> None:
        rows = _plain(render_markdown("# One\n## Two\n### Three\n", _console(), _theme()))
        assert rows == ["One", "Two", "### Three"]

    def test_a_wrapped_item_hangs_under_its_text(self) -> None:
        source = "- " + " ".join(["item"] * 20) + "\n  - nested\n"
        rows = _plain(render_markdown(source, _console(40), _theme()))
        assert rows[0].startswith("• item")
        assert all(line.startswith("  item") for line in rows[1:-1])
        assert rows[-1] == "  • nested"

    def test_a_link_shows_its_address_when_the_text_is_different(self) -> None:
        rows = _plain(
            render_markdown(
                "[docs](https://x.dev/a) and <https://x.dev/b>\n", _console(80), _theme()
            )
        )
        assert rows == ["docs (https://x.dev/a) and https://x.dev/b"]

    def test_blank_lines_collapse_and_do_not_pad_the_ends(self) -> None:
        rows = _plain(render_markdown("\n\nfirst\n\n\n\nsecond\n\n", _console(), _theme()))
        assert rows == ["first", "", "second"]


class TestTables:
    SOURCE = "| Skill | Tools |\n|---|--:|\n| `setup` | 1 |\n| `scan` | 22 |\n"

    def test_it_waits_for_its_last_row(self) -> None:
        """Columns cannot be sized before every row is known."""
        stream = _stream()
        assert stream.feed(self.SOURCE) == []
        rows = _plain(stream.feed("after\n"))
        assert rows[-1] == "after"
        table = rows[:-1]
        assert len({len(line) for line in table}) == 1, "the columns are not aligned"
        assert any("setup" in line and "1" in line for line in table)

    def test_a_pipe_that_starts_no_table_is_still_printed(self) -> None:
        rows = _plain(render_markdown("| not a table\nnext line\n", _console(), _theme()))
        assert rows == ["| not a table", "next line"]

    def test_the_marker_gets_its_own_line_in_front_of_one(self) -> None:
        """In front of the top border it would push that one row out of line."""
        rows = _plain(render_markdown(self.SOURCE, _console(), _theme(), lead=Text("⏺ ")))
        assert rows[0] == "⏺"
        assert len({len(line) for line in rows[1:]}) == 1


class TestCodeBlocks:
    def test_the_fences_stay_and_the_body_is_indented(self) -> None:
        rows = _plain(render_markdown("```sh\nls -la\n```\n", _console(), _theme()))
        assert rows == ["```sh", "  ls -la", "```"]

    def test_a_long_line_is_left_whole_for_the_terminal_to_fold(self) -> None:
        """A break of ours would be a newline in the middle of a pasted command."""
        command = "curl " + "-H 'x: y' " * 20
        rows = _plain(render_markdown(f"```\n{command}\n```\n", _console(40), _theme()))
        assert rows[1] == "  " + command.rstrip()

    def test_colouring_knows_about_the_lines_before(self) -> None:
        """Streamed a line at a time, the second line of a docstring is still
        inside the string."""
        stream = _stream()
        rows: list[Text] = []
        for line in ["```python\n", 'x = """doc\n', 'still doc"""\n', "```\n"]:
            rows += stream.feed(line)
        continued = next(row for row in rows if "still doc" in row.plain)
        opened = next(row for row in rows if "doc" in row.plain and "x =" in row.plain)
        assert _style_at(continued, "still doc") == _style_at(opened, '"""doc')

    def test_a_block_the_model_never_closed_is_closed_at_the_end(self) -> None:
        rows = _plain(render_markdown("```\nunfinished", _console(), _theme()))
        assert rows == ["```", "  unfinished", "```"]


def _renderer(*, terminal: bool, live: bool = False) -> tuple[EventRenderer, io.StringIO]:
    buffer = io.StringIO()
    console = Console(
        file=buffer,
        width=60,
        height=40,
        force_terminal=terminal,
        color_system="truecolor" if terminal else None,
    )
    renderer = EventRenderer(console, _theme())
    renderer.begin_turn()
    renderer.set_live_prompt(live)
    return renderer, buffer


def _screen(buffer: io.StringIO) -> str:
    return re.sub(r"\x1b\]8;[^\x1b]*\x1b\\|\x1b\[[0-9;]*m", "", buffer.getvalue())


def _token(text: str) -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.RESPONSE_TOKEN,
        author="agent",
        content=[Part(text=text)],
        is_partial=True,
    )


class TestTheRendererUsesIt:
    def test_on_a_terminal_the_answer_is_rendered(self) -> None:
        renderer, buffer = _renderer(terminal=True)
        for piece in ("Use **bold** and `co", "de`.\n"):
            renderer.render(_token(piece))
        renderer.render(
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="agent",
                content=[Part(text="Use **bold** and `code`.\n")],
            )
        )
        assert _screen(buffer).splitlines() == ["⏺ Use bold and code."]

    def test_anywhere_else_the_source_is_printed_as_it_came(self) -> None:
        """``domyn run … > out.md`` should get what the model wrote."""
        renderer, buffer = _renderer(terminal=False)
        renderer.render(_token("Use **bold**\n"))
        renderer.finish()
        assert "Use **bold**" in buffer.getvalue()

    def test_under_a_live_prompt_every_write_ends_a_line(self) -> None:
        """The steering editor erases whatever is left without a newline."""
        renderer, buffer = _renderer(terminal=True, live=True)
        words = [
            "A",
            "paragraph",
            "that",
            "streams",
            "in,",
            "word",
            "by",
            "word,",
            "**with**",
            "markup.",
        ]
        for word in words:
            renderer.render(_token(word + " "))
            assert buffer.getvalue() == "" or buffer.getvalue().endswith("\n")
        renderer.finish()
        assert "A paragraph that streams in" in _screen(buffer)

    def test_reasoning_is_rendered_as_trace(self) -> None:
        renderer, buffer = _renderer(terminal=True)
        renderer.render(
            BaseEvent(
                event_type=ExecutionEventType.LLM_END,
                author="agent",
                content=[Part(thought="**Checking** the `config`\n- first\n- second")],
            )
        )
        assert _screen(buffer).splitlines() == [
            "  Checking the config",
            "  • first",
            "  • second",
        ]


class TestThePreview:
    """What a front-end that can repaint shows of the part not printed yet.

    Worked out on a copy, so asking for it must never change what the stream
    prints; and at any moment it is what `finish` would print if the answer
    ended there — the same rows, which is why nothing jumps when they land.
    """

    def test_it_shows_the_line_being_written(self) -> None:
        stream = _stream()
        assert stream.feed("Use **bo") == []
        assert _plain(stream.preview()) == ["Use **bo"]

    def test_asking_for_it_changes_nothing(self) -> None:
        stream, rows = _stream(), []
        for char in DOCUMENT:
            rows += stream.feed(char)
            stream.preview()
        assert _visible(rows + stream.finish()) == _visible(_render([DOCUMENT]))

    @pytest.mark.parametrize("cut", [50, 400, 470])
    def test_it_is_what_finish_would_print(self, cut: int) -> None:
        """Outside a code block: there `finish` closes the fence, the preview not."""
        stream = _stream()
        stream.feed(DOCUMENT[:cut])
        preview = stream.preview()
        assert _visible(preview) == _visible(stream.finish())

    def test_a_table_shows_before_its_last_row(self) -> None:
        stream = _stream()
        assert stream.feed("| a | b |\n|---|---|\n| 1 | 2 |\n") == []
        shown = _plain(stream.preview())
        assert any("1" in row and "2" in row for row in shown)
        assert shown == _plain(stream.finish())

    def test_an_open_code_block_has_no_closing_fence_yet(self) -> None:
        stream = _stream()
        stream.feed("```python\nx = 1\ny = ")
        assert _plain(stream.preview()) == ["  y ="]

    def test_nothing_held_is_nothing_to_show(self) -> None:
        stream = _stream()
        stream.feed("A whole line.\n")
        assert stream.preview() == []


class _Counted(io.StringIO):
    """A terminal behind a proxy that writes later: counts what it is handed."""

    def __init__(self, progress: OutputProgress) -> None:
        super().__init__()
        self._progress = progress

    def write(self, text: str) -> int:
        self._progress.sent += len(text)
        return super().write(text)


def _previewing(height: int = 40) -> tuple[EventRenderer, OutputProgress]:
    progress = OutputProgress()
    console = Console(
        file=_Counted(progress),
        width=60,
        height=height,
        force_terminal=True,
        color_system="truecolor",
    )
    renderer = EventRenderer(console, _theme())
    renderer.output = progress
    renderer.begin_turn()
    renderer.set_live_prompt(True)
    return renderer, progress


def _drawn(ansi: str) -> list[str]:
    return re.sub(r"\x1b\[[0-9;]*m", "", ansi).splitlines()


class TestTheRendererPreview:
    """`EventRenderer.preview` is what the editor draws above its prompt while the
    answer streams: the unfinished end, plus whatever was printed but has not
    reached the terminal yet — the proxy writes later, and the editor repaints
    in between."""

    def test_nothing_while_no_answer_streams(self) -> None:
        renderer, _ = _previewing()
        assert renderer.preview() == ""

    def test_the_unfinished_end_with_its_marker(self) -> None:
        renderer, _ = _previewing()
        renderer.render(_token("Use **bo"))
        assert _drawn(renderer.preview()) == ["⏺ Use **bo"]

    def test_not_off_a_terminal(self) -> None:
        renderer, _ = _renderer(terminal=False, live=True)
        renderer.render(_token("Use **bo"))
        assert renderer.preview() == ""

    def test_a_printed_row_stays_until_the_terminal_has_it(self) -> None:
        """Gone from the stream the moment it is printed, but the proxy writes it
        up to its flush interval later: in between it would be on screen nowhere."""
        renderer, progress = _previewing()
        renderer.render(_token("line one\nline tw"))
        assert _drawn(renderer.preview()) == ["⏺ line one", "line tw"]
        progress.written = progress.sent
        assert _drawn(renderer.preview()) == ["line tw"]

    def test_each_row_leaves_when_its_own_text_is_written(self) -> None:
        """Two rows printed together can reach the terminal in two writes; the
        first must not be drawn a second time while the other is still on its way."""
        renderer, progress = _previewing()
        renderer.render(_token("first\nsecond\nthi"))
        first_mark = renderer._in_flight[0][0]
        progress.written = first_mark
        assert _drawn(renderer.preview()) == ["second", "thi"]

    def test_hyperlinks_are_left_out(self) -> None:
        """prompt_toolkit would draw their escape sequence as text."""
        renderer, _ = _previewing()
        renderer.render(_token("see [docs](https://example.com) and"))
        drawn = renderer.preview()
        assert "\x1b]8;" not in drawn
        assert "docs" in drawn

    def test_a_tall_table_is_a_window_onto_its_end(self) -> None:
        renderer, _ = _previewing(height=40)
        rows = "".join(f"| r{n:02d} | {n} |\n" for n in range(40))
        renderer.render(_token("| a | b |\n|---|---|\n" + rows))
        drawn = _drawn(renderer.preview())
        assert len(drawn) == 20, "half the window"
        assert any("r39" in line for line in drawn)
        assert not any("r00" in line for line in drawn)

    def test_the_editor_is_asked_to_repaint_but_not_on_every_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import domyn.agents.interfaces.cli.renderer as renderer_module

        clock = [100.0]
        monkeypatch.setattr(renderer_module.time, "monotonic", lambda: clock[0])
        renderer, _ = _previewing()
        asked: list[None] = []
        renderer.on_preview = lambda: asked.append(None)
        for piece in ("a", "b", "c"):
            renderer.render(_token(piece))
        assert len(asked) == 1
        clock[0] += 0.1
        renderer.render(_token("d"))
        assert len(asked) == 2
