"""Arrow-key selection for a short list of options.

Rendered inline, the way comparable CLIs do it: the list stays in the transcript,
the highlighted row moves with ↑/↓, Enter confirms. No full-screen dialog, which
would clear the terminal and lose the context the user was reading.

Falls back to a numbered prompt whenever the fancy version cannot work — no
TTY, ``TERM=dumb``. That fallback is not a nicety: the
setup wizard has to stay drivable from a pipe, or scripted installs and tests
break.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from domyn.agents.interfaces.cli.prompt import (
    GoBack,
    back_key_on,
    interactive_terminal,
    read_line,
)

if TYPE_CHECKING:
    from rich.console import Console

    from domyn.agents.interfaces.cli.theme import Theme


# One look for every pick-one question, whoever draws it: the picker below (setup,
# /model, an approval outside a session) and the question the session's editor
# carries above its line (an approval while a turn runs). Two copies had drifted
# apart in colour and wording.
CHOICE_STYLES: dict[str, str] = {
    "choice-label": "bold",
    # A cyan bar with the text in the terminal's background colour. Spelled out:
    # the picker once got the reverse only from prompt_toolkit's built-in rule
    # for a class named `selected`, and lost it with the name.
    "choice-selected": "reverse bold cyan",
    "choice-option": "",
    "choice-hint": "italic",
}


def choice_rows(
    theme: Theme,
    label: str,
    options: list[str],
    index: int,
    keys: list[str],
    *,
    descriptions: dict[str, str] | None = None,
) -> list[tuple[str, str]]:
    """A pick-one question as prompt_toolkit fragments, ending in a newline.

    ``keys`` are the hint's entries after ``↑↓ move`` (``enter select``…), since
    what the other keys do depends on who asks.

    Rows are numbered, the same numbers the fallback list prints, so an option can
    be named by typing its number wherever it is drawn.
    """
    # Same rule as everywhere else in the CLI: Unicode only when the terminal
    # declares it can render it.
    unicode = theme.capabilities.unicode
    pointer_glyph = "\u276f" if unicode else ">"
    descriptions = descriptions or {}
    rows: list[tuple[str, str]] = [("class:choice-label", f"  {label}\n")]
    for position, option in enumerate(options):
        selected = position == index
        pointer = f"{pointer_glyph} " if selected else "  "
        style = "class:choice-selected" if selected else "class:choice-option"
        rows.append((style, f"  {pointer}{position + 1}. {option}"))
        if option in descriptions:
            rows.append(("class:choice-hint", f"  {descriptions[option]}"))
        rows.append(("", "\n"))
    separator, move = (" \u00b7 ", "\u2191\u2193 move") if unicode else (" - ", "up/down move")
    hint = separator.join([move, *keys])
    rows.append(("class:choice-hint", f"  {hint}\n"))
    return rows


def typed_number(text: str, count: int) -> int | None:
    """The 0-based option ``text`` names, if it is nothing but a valid number.

    Whole-line only: ``2 more things`` is a sentence, not a pick, and ``7`` of four
    options is a number someone meant to say, not one they meant to choose.
    """
    text = text.strip()
    if text.isdigit() and 1 <= int(text) <= count:
        return int(text) - 1
    return None


def _interactive(theme: Theme) -> bool:
    return interactive_terminal() and theme.capabilities.color


def select(
    console: Console,
    theme: Theme,
    label: str,
    options: list[str],
    *,
    current: str | None = None,
    descriptions: dict[str, str] | None = None,
    aliases: dict[str, str] | None = None,
    max_tries: int | None = None,
) -> str | None:
    """Ask the user to pick one option.

    Returns the chosen value, or ``None`` if the user aborted. Inside
    :func:`~domyn.agents.interfaces.cli.prompt.back_key`, Ctrl-B raises ``GoBack``.

    Only the numbered fallback reads typed text, so only it uses the last two:
    ``aliases`` maps extra answers to an option (``y`` for a yes), and
    ``max_tries`` gives up with ``None`` after that many unusable answers — a
    stdin that keeps repeating one would otherwise hold the caller forever.
    """
    if not options:
        return None

    start = options.index(current) if current in options else 0

    if _interactive(theme):
        chosen = _select_with_arrows(theme, label, options, start, descriptions or {})
        if chosen is not None and back_key_on():
            _echo(console, theme, label, chosen)
        return chosen
    return _select_with_numbers(
        console, theme, label, options, start, descriptions or {}, aliases or {}, max_tries
    )


def _echo(console: Console, theme: Theme, label: str, answer: str) -> None:
    """An answered picker inside ``back_key``: it erased itself, and stays as this line."""
    from rich.text import Text

    separator = " " if label.endswith("?") else ": "
    console.print(
        Text.assemble((f"  {label}{separator}", theme.style("accent")), answer), highlight=False
    )


def _select_with_numbers(
    console: Console,
    theme: Theme,
    label: str,
    options: list[str],
    start: int,
    descriptions: dict[str, str],
    aliases: dict[str, str],
    max_tries: int | None,
) -> str | None:
    console.print(f"  {label}:", style=theme.style("accent"), highlight=False)
    for index, option in enumerate(options):
        marker = "*" if index == start else " "
        hint = f"  — {descriptions[option]}" if option in descriptions else ""
        console.print(
            f"   {marker} {index + 1}) {option}{hint}", style=theme.style("meta"), highlight=False
        )

    default = options[start]
    tries = 0
    while max_tries is None or tries < max_tries:
        tries += 1
        try:
            raw = read_line(console, f"  choice [{default}]: ", style=theme.style("accent")).strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return None
        if not raw:
            return default
        if raw in options:
            return raw
        if raw.lower() in aliases:
            return aliases[raw.lower()]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        console.print(
            f"  pick 1-{len(options)} or a name", style=theme.style("warning"), highlight=False
        )
    return None


def _select_with_arrows(
    theme: Theme,
    label: str,
    options: list[str],
    start: int,
    descriptions: dict[str, str],
) -> str | None:
    """Inline arrow-key picker.

    Run with ``in_thread=True``: a dedicated thread with its own event loop.
    ``Application.run()`` refuses to start when an event loop is already running,
    and this is reached from ``/model`` *inside* the REPL's loop — so calling it
    plainly works from a shell and breaks in a session, which is the worst kind
    of bug.
    """
    from prompt_toolkit.application import Application
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style

    index = start
    aborted = False
    back = back_key_on()
    went_back = False
    # Digits typed since the last move, so ``1`` then ``2`` reaches row 12 of a
    # long /model list. A digit only moves the highlight; Enter still confirms, as
    # in the editor, where a digit cannot pick because it may start a message.
    digits = ""

    def render() -> FormattedText:
        return FormattedText(
            choice_rows(
                theme,
                label,
                options,
                index,
                [
                    "enter select",
                    f"1-{len(options)} jump",
                    *(["ctrl-b back"] if back else []),
                    "ctrl-c cancel",
                ],
                descriptions=descriptions,
            )
        )

    bindings = KeyBindings()

    @bindings.add("up")
    @bindings.add("k")
    def _up(event: Any) -> None:
        nonlocal index, digits
        index, digits = (index - 1) % len(options), ""

    @bindings.add("down")
    @bindings.add("j")
    def _down(event: Any) -> None:
        nonlocal index, digits
        index, digits = (index + 1) % len(options), ""

    def _jump(event: Any) -> None:
        nonlocal index, digits
        for candidate in (digits + event.data, event.data):
            named = typed_number(candidate, len(options))
            if named is not None:
                index, digits = named, candidate
                return
        digits = ""

    for digit in "0123456789":
        bindings.add(digit)(_jump)

    @bindings.add("enter")
    def _accept(event: Any) -> None:
        event.app.exit()

    @bindings.add("c-c")
    @bindings.add("escape")
    def _cancel(event: Any) -> None:
        nonlocal aborted
        aborted = True
        event.app.exit()

    if back:

        @bindings.add("c-b")
        def _back(event: Any) -> None:
            nonlocal went_back
            went_back = True
            event.app.exit()

    application: Application[None] = Application(
        layout=Layout(
            HSplit(
                [
                    # Wrapped, not cut at the edge: an approval's options carry the
                    # pattern they grant, and a cut row hides what is being given.
                    Window(FormattedTextControl(render), dont_extend_height=True, wrap_lines=True)
                ]
            )
        ),
        key_bindings=bindings,
        style=Style.from_dict(CHOICE_STYLES),
        full_screen=False,
        # Inside `back_key` the answer is printed back instead (`_echo`).
        erase_when_done=back,
    )

    application.run(in_thread=True)
    # Raised here, in the caller's thread, not from the binding in the picker's.
    if went_back:
        raise GoBack
    return None if aborted else options[index]


@dataclass(frozen=True)
class _Row:
    """One line of a tick-any picker: an option, or a group standing for several."""

    label: str
    leaves: tuple[str, ...]
    """The options this row ticks: itself for an option, its members for a group."""
    is_group: bool = False
    indent: int = 0


class _TickList:
    """What a tick-any picker holds, with no terminal in it.

    Options may be gathered under group labels. A group is a row like any other that
    the cursor can stop on: ticking it ticks all its options, and its own box is
    worked out from them (all, some or none) rather than stored, so it can never
    disagree with them. Only options are ever returned. Both pickers, arrows and
    numbers, are drawn from this, and it is what the tests drive.
    """

    def __init__(
        self,
        options: Sequence[str],
        groups: Mapping[str, Sequence[str]] | None = None,
        selected: Iterable[str] = (),
    ) -> None:
        self.options = list(options)
        self.rows: list[_Row] = []
        seen: set[str] = set()
        for group, members in (groups or {}).items():
            if group in self.options:
                raise ValueError(f"group {group!r} has the name of an option")
            unknown = [member for member in members if member not in self.options]
            if unknown:
                raise ValueError(f"group {group!r} names unknown options: {', '.join(unknown)}")
            twice = [member for member in members if member in seen]
            if twice:
                raise ValueError(f"options in more than one group: {', '.join(twice)}")
            if not members:
                continue
            seen.update(members)
            self.rows.append(_Row(group, tuple(members), is_group=True))
            self.rows.extend(_Row(member, (member,), indent=1) for member in members)
        # Options in no group still have to be reachable.
        self.rows.extend(_Row(option, (option,)) for option in self.options if option not in seen)
        self.ticked = {option for option in selected if option in self.options}

    def state(self, index: int) -> str:
        """``"all"``, ``"some"`` or ``"none"`` of a row's options are ticked."""
        leaves = self.rows[index].leaves
        count = sum(1 for leaf in leaves if leaf in self.ticked)
        return "none" if count == 0 else "all" if count == len(leaves) else "some"

    def toggle(self, index: int) -> None:
        """Flip an option; for a group, tick the rest of it, or clear it if it was all ticked."""
        row = self.rows[index]
        if not row.is_group:
            self.ticked.symmetric_difference_update(row.leaves)
        elif self.state(index) == "all":
            self.ticked.difference_update(row.leaves)
        else:
            self.ticked.update(row.leaves)

    def toggle_all(self) -> None:
        if len(self.ticked) == len(self.options):
            self.ticked.clear()
        else:
            self.ticked.update(self.options)

    def selected(self) -> list[str]:
        """The ticked options, in the order they were given."""
        return [option for option in self.options if option in self.ticked]


def multiselect(
    console: Console,
    theme: Theme,
    label: str,
    options: list[str],
    *,
    selected: Iterable[str] = (),
    descriptions: dict[str, str] | None = None,
    groups: Mapping[str, Sequence[str]] | None = None,
    max_tries: int | None = None,
) -> list[str] | None:
    """Ask the user to tick any number of options.

    Returns the ticked ones in the order of ``options`` (possibly none), or ``None``
    if the user aborted. ``selected`` is what starts ticked. Inside
    :func:`~domyn.agents.interfaces.cli.prompt.back_key`, Ctrl-B raises ``GoBack``,
    and the list erases itself when it ends, with nothing
    printed back: unlike a pick, what was ticked is the caller's to say, and every
    caller does (``Tools: …``, ``Installed 11 skills``).

    ``groups`` gathers options under labels, drawn as a row with its options indented
    below it: ticking the group ticks them all, and each can still be ticked on its
    own. The groups are only in the picker: what comes back is options. A group's
    label may have a ``descriptions`` entry like an option.

    On a terminal: ↑/↓ move, Space ticks, ``a`` ticks or clears all, Enter confirms.
    Otherwise the numbered fallback, which is what keeps the wizard drivable from a
    pipe: every row is numbered, groups included, and the answer is the new selection
    as numbers (``1,3``, ``2-4``; a group's number takes all its options), ``all`` or
    ``none``, and an empty answer keeps what is ticked. ``max_tries`` gives up with
    ``None`` after that many unusable answers.
    """
    if not options:
        return []
    ticks = _TickList(options, groups, selected)
    if _interactive(theme):
        return _multiselect_with_arrows(theme, label, ticks, descriptions or {})
    return _multiselect_with_numbers(console, theme, label, ticks, descriptions or {}, max_tries)


def _box(theme: Theme, state: str) -> str:
    """The mark before a row: ticked, partly ticked (a group), or not."""
    glyphs = (
        {"all": "\u25c9", "some": "\u25d0", "none": "\u25ef"}
        if theme.capabilities.unicode
        else {"all": "[x]", "some": "[-]", "none": "[ ]"}
    )
    return glyphs[state]


_INDENT = "    "


def multi_choice_rows(
    theme: Theme,
    label: str,
    ticks: _TickList,
    index: int,
    *,
    descriptions: dict[str, str] | None = None,
    back: bool = False,
) -> list[tuple[str, str]]:
    """The `choice_rows` look for a tick-any question: a box before each row, a group's
    options indented below it. ``back`` adds Ctrl-B to the hint."""
    pointer_glyph = "\u276f" if theme.capabilities.unicode else ">"
    descriptions = descriptions or {}
    rows: list[tuple[str, str]] = [("class:choice-label", f"  {label}\n")]
    for position, row in enumerate(ticks.rows):
        current = position == index
        pointer = f"{pointer_glyph} " if current else "  "
        style = "class:choice-selected" if current else "class:choice-option"
        box = _box(theme, ticks.state(position))
        rows.append((style, f"  {pointer}{_INDENT * row.indent}{box} {row.label}"))
        if row.label in descriptions:
            rows.append(("class:choice-hint", f"  {descriptions[row.label]}"))
        rows.append(("", "\n"))
    separator, move = (
        (" \u00b7 ", "\u2191\u2193 move") if theme.capabilities.unicode else (" - ", "up/down move")
    )
    keys = ["space tick", "a all/none", "enter confirm", *(["ctrl-b back"] if back else [])]
    hint = separator.join([move, *keys, "ctrl-c cancel"])
    rows.append(("class:choice-hint", f"  {hint}\n"))
    return rows


def parse_numbers(text: str, ticks: _TickList) -> list[str] | None:
    """``1,3`` / ``2-4`` / ``all`` / ``none`` as the options they name; ``None`` if unusable.

    Numbers are rows, groups included: a group's number names all its options, and a
    range names every row it covers.
    """
    text = text.strip().lower()
    if text == "all":
        return list(ticks.options)
    if text == "none":
        return []
    count = len(ticks.rows)
    named: set[str] = set()
    for piece in text.replace(" ", "").split(","):
        low, dash, high = piece.partition("-")
        if not low.isdigit() or (dash and not high.isdigit()):
            return None
        first, last = int(low), int(high) if dash else int(low)
        if not 1 <= first <= last <= count:
            return None
        for number in range(first, last + 1):
            named.update(ticks.rows[number - 1].leaves)
    return [option for option in ticks.options if option in named]


def _multiselect_with_numbers(
    console: Console,
    theme: Theme,
    label: str,
    ticks: _TickList,
    descriptions: dict[str, str],
    max_tries: int | None,
) -> list[str] | None:
    console.print(f"  {label}:", style=theme.style("accent"), highlight=False)
    marks = {"all": "x", "some": "-", "none": " "}
    for index, row in enumerate(ticks.rows):
        hint = f"  \u2014 {descriptions[row.label]}" if row.label in descriptions else ""
        # markup=False: `[x]` is a rich tag, and a description may hold brackets too.
        console.print(
            f"   {_INDENT * row.indent}[{marks[ticks.state(index)]}] {index + 1}) {row.label}{hint}",
            style=theme.style("meta"),
            highlight=False,
            markup=False,
        )

    current = ticks.selected()
    tries = 0
    while max_tries is None or tries < max_tries:
        tries += 1
        try:
            raw = read_line(
                console,
                "  choose (e.g. 1,3 or 2-4, all, none; empty keeps the ticked ones): ",
                style=theme.style("accent"),
            ).strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return None
        if not raw:
            return current
        chosen = parse_numbers(raw, ticks)
        if chosen is not None:
            return chosen
        console.print(
            f"  numbers from 1 to {len(ticks.rows)}, a range like 2-4, all or none",
            style=theme.style("warning"),
            highlight=False,
        )
    return None


def _multiselect_with_arrows(
    theme: Theme,
    label: str,
    ticks: _TickList,
    descriptions: dict[str, str],
) -> list[str] | None:
    """The inline picker of `_select_with_arrows`, with a tick per row.

    Run with ``in_thread=True`` for the same reason: this is reached from inside
    the REPL's event loop as well as from a shell.
    """
    from prompt_toolkit.application import Application
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style

    index = 0
    aborted = False
    back = back_key_on()
    went_back = False

    def render() -> FormattedText:
        return FormattedText(
            multi_choice_rows(theme, label, ticks, index, descriptions=descriptions, back=back)
        )

    bindings = KeyBindings()

    @bindings.add("up")
    @bindings.add("k")
    def _up(event: Any) -> None:
        nonlocal index
        index = (index - 1) % len(ticks.rows)

    @bindings.add("down")
    @bindings.add("j")
    def _down(event: Any) -> None:
        nonlocal index
        index = (index + 1) % len(ticks.rows)

    @bindings.add("space")
    def _toggle(event: Any) -> None:
        ticks.toggle(index)

    @bindings.add("a")
    def _toggle_all(event: Any) -> None:
        ticks.toggle_all()

    @bindings.add("enter")
    def _accept(event: Any) -> None:
        event.app.exit()

    @bindings.add("c-c")
    @bindings.add("escape")
    def _cancel(event: Any) -> None:
        nonlocal aborted
        aborted = True
        event.app.exit()

    if back:

        @bindings.add("c-b")
        def _back(event: Any) -> None:
            nonlocal went_back
            went_back = True
            event.app.exit()

    application: Application[None] = Application(
        layout=Layout(
            HSplit([Window(FormattedTextControl(render), dont_extend_height=True, wrap_lines=True)])
        ),
        key_bindings=bindings,
        style=Style.from_dict(CHOICE_STYLES),
        full_screen=False,
        # Inside `back_key` the answer is printed back instead (`_echo`).
        erase_when_done=back,
    )

    application.run(in_thread=True)
    # Raised here, in the caller's thread, not from the binding in the picker's.
    if went_back:
        raise GoBack
    return None if aborted else ticks.selected()
