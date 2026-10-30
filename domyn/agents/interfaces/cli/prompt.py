"""User input for the interactive session.

Every read goes through ``prompt_toolkit`` (completion, history, multiline
editing). It is a dependency of the CLI (the ``cli`` extra, installed by ``domyn
init``, and by ``domyn run`` when it is missing — ``cli/cli_dependencies.py``),
imported lazily so embedding the library drags in no terminal deps. The
one exception is a stdin that is not a terminal — a pipe has no terminal for the
editor to drive, so it is read with :func:`input`.

Enter submits; Alt/Option+Enter and Ctrl+J insert a newline. Ctrl-C clears what
is typed, and a second one right after quits, like Ctrl-D; while a turn runs it
stops the turn instead.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any

from domyn.agents.interfaces.cli.commands import completion_names

if TYPE_CHECKING:
    from rich.console import Console

    from domyn.agents.interfaces.cli.theme import Theme
    from domyn.agents.interfaces.cli.voice import VoiceInput

_HISTORY_PATH = Path.home() / ".domyn" / "cli_history"


# Ten a second: the clock shows tenths, and slower reads as a stutter.
_STATUS_REFRESH_S = 0.1

# How close two Ctrl-C have to be to count as one "quit".
_DOUBLE_PRESS_S = 0.8


class GoBack(Exception):  # noqa: N818 — a request, like KeyboardInterrupt, not an error
    """Ctrl-B on a question asked inside :func:`back_key`: go back to the previous one."""


# Off everywhere but the setup wizard.
_back_key: ContextVar[bool] = ContextVar("back_key", default=False)


@contextlib.contextmanager
def back_key() -> Iterator[None]:
    """Make Ctrl-B raise :class:`GoBack` from the questions asked inside the block.

    Only on a terminal: a pipe has no key to press. Not Esc: a lone Esc can only be
    told from the start of Option+← (``ESC b``) by waiting for the key after it, so it
    went back half a second late. In a text field Ctrl-B no longer moves the cursor a
    character to the left; ← still does.

    Inside the block a question also erases itself when it ends, and an answered one
    stays as a line printed through the console: everything left on screen is then
    the console's, whose rows the wizard counts to erase back to an earlier question.
    """
    token = _back_key.set(True)
    try:
        yield
    finally:
        _back_key.reset(token)


def back_key_on() -> bool:
    """Whether Ctrl-B goes back here (:func:`back_key`)."""
    return _back_key.get()


def interactive_terminal() -> bool:
    """Whether stdin is a terminal the editor can drive.

    A piped stdin is not: prompt_toolkit fails its event loop and takes the
    session with it, and ``printf … | domyn run`` must work.
    """
    return bool(getattr(sys.stdin, "isatty", lambda: False)())


def read_line(
    console: Console,
    message: str,
    *,
    style: Any = None,
    password: bool = False,
    prefill: str = "",
) -> str:
    """Read one line outside the session's own editor: approvals, wizard fields.

    A real line editor: bare :func:`input` only edits if ``readline`` was
    imported, which nothing does, so ←/→ printed ``^[[D`` into the answer instead
    of moving the cursor. ``in_thread`` because these are reached synchronously
    from inside the REPL's running loop, where ``prompt()`` refuses to start.
    Raises ``EOFError`` / ``KeyboardInterrupt`` as :func:`input` does.

    ``password`` hides what is typed; a pipe cannot, and echoes.

    ``prefill`` starts the line with that text, to be kept with Enter, edited, or
    erased: the answer is the line as the user left it. A pipe cannot pre-fill,
    so it shows the text in brackets and returns it for a blank answer, which
    makes Enter mean the same on both.

    Inside :func:`back_key`, Ctrl-B raises :class:`GoBack`, and the line erases itself
    when it ends: the answer is printed back through ``console``.

    ``message`` is text, never markup: every caller shows a value in brackets
    (``[auto]``, ``[gpt-4o]``, a pattern), and rich would read any that starts
    lowercase as a tag and delete it. ``[http://localhost:28080]`` parses as a
    style tag and renders as **nothing**, so the wizard's own
    ``url (required) [http://…]:`` lost its default and a required field looked
    empty — measured on `domyn setup` re-runs, where it invites retyping a URL
    that was already right. The house rule for the same hazard is in
    ``interfaces/cli/calls.py``: print what a user typed as text, never as
    markup.
    """
    if not interactive_terminal():
        # A pre-filled secret is masked here as the editor masks it.
        bracketed = "*" * len(prefill) if password else prefill
        shown = f"{message}[{bracketed}] " if prefill else message
        console.print(shown, end="", style=style, highlight=False, markup=False)
        answer = input()
        return answer if answer.strip() else prefill or answer

    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import ANSI, FormattedText
    from prompt_toolkit.key_binding import KeyBindings

    # The editor redraws its own line, so the label has to be part of what it
    # draws — printed by rich beforehand it would be overwritten. Rendered by
    # rich all the same, so it keeps the theme's colour.
    with console.capture() as captured:
        console.print(message, end="", style=style, highlight=False, markup=False)
    back = back_key_on()
    bindings = KeyBindings()
    hint = None
    if back:

        @bindings.add("c-b")
        def _back(event: Any) -> None:
            event.app.exit(exception=GoBack())

        hint = FormattedText([("italic", "ctrl-b back")])
    session: PromptSession[str] = PromptSession(erase_when_done=True) if back else PromptSession()
    # ``in_thread`` re-raises the worker's KeyboardInterrupt / EOFError / GoBack here.
    answer = session.prompt(
        ANSI(captured.get()),
        is_password=password,
        in_thread=True,
        default=prefill,
        key_bindings=bindings,
        rprompt=hint,
    )
    if back:
        from rich.text import Text

        shown = "*" * len(answer) if password else answer
        console.print(Text.assemble((message, style or ""), shown), highlight=False)
    return answer


@dataclass
class OutputProgress:
    """How much of what was printed through :func:`shared_stdout` has reached the
    terminal, in characters: ``sent`` when handed to the proxy, ``written`` once the
    proxy has written it.

    The proxy writes from a thread of its own, up to ``sleep_between_writes`` after
    it was handed the text, and the editor goes on repainting in between. Anything
    the editor draws in place of printed text until it is printed — the answer's
    unfinished end — needs to know when that moment is, or the text is on screen
    in neither place for the length of the gap.
    """

    sent: int = 0
    written: int = 0


@contextlib.contextmanager
def shared_stdout(progress: OutputProgress | None = None) -> Iterator[None]:
    """Route everything printed anywhere to *above* the live prompt.

    Two writers share one terminal while a read runs alongside a turn, and only
    the editor repaints a region instead of appending — a plain ``print`` moves
    the cursor out from under it. The proxy defers each write until the editor can
    erase its line, let it through, and redraw.

    ``prompt_toolkit.patch_stdout`` but for two settings:

    * ``raw=True``: the default strips VT100 escapes, i.e. every colour the
      renderer emits, leaving a whole turn monochrome for no visible reason.
    * A short flush interval: the default 0.2s turns a token stream into five
      chunks a second instead of a flow.

    And ``progress``, when given, counts what goes in and what comes out.
    """
    proxy = _tracked_proxy(progress)
    with proxy:
        saved_out, saved_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = proxy, proxy
        try:
            yield
        finally:
            sys.stdout, sys.stderr = saved_out, saved_err
    if progress is not None:
        # Closed: whatever it was going to write is written. Text it still held
        # without a newline is dropped by prompt_toolkit on close, and must not
        # leave the count waiting for it forever.
        progress.written = progress.sent


def _tracked_proxy(progress: OutputProgress | None) -> Any:
    from prompt_toolkit.application import run_in_terminal
    from prompt_toolkit.patch_stdout import StdoutProxy

    if progress is None:
        return StdoutProxy(raw=True, sleep_between_writes=0.02)
    counted = progress

    class _Tracked(StdoutProxy):
        """Counts on the way in, and on the way out at the moment of writing.

        ``_write_and_flush`` is prompt_toolkit's own, with one line added: the
        count goes up *inside* the write, before the editor redraws after it, so
        that redraw already knows the text is in the transcript. Counted anywhere
        later and that redraw shows it twice; anywhere earlier, not at all.
        """

        def write(self, data: str) -> int:
            with self._lock:
                counted.sent += len(data)
                self._write(data)
            return len(data)

        def _write_and_flush(self, loop: asyncio.AbstractEventLoop | None, text: str) -> None:
            def write_and_flush() -> None:
                self._output.enable_autowrap()
                self._output.write_raw(text)
                self._output.flush()
                # Never past `sent`: a write still queued in the loop when the
                # proxy closed lands after `shared_stdout` has settled the count.
                counted.written = min(counted.sent, counted.written + len(text))

            def write_and_flush_in_loop() -> None:
                run_in_terminal(write_and_flush, in_executor=False)

            if loop is None:
                write_and_flush()
            else:
                loop.call_soon_threadsafe(write_and_flush_in_loop)

    return _Tracked(raw=True, sleep_between_writes=0.02)


@dataclass
class _Choice:
    """A question drawn inside the editor, above the line being typed."""

    question: str
    options: list[str]
    index: int
    answer: asyncio.Future[int | None]

    def settle(self, picked: int | None) -> None:
        if not self.answer.done():
            self.answer.set_result(picked)


class InputSession:
    """Reads one user message at a time.

    Raises ``EOFError`` when the user wants out — Ctrl-D, or two Ctrl-C in quick
    succession — and ``KeyboardInterrupt`` for a Ctrl-C during a read alongside a
    turn, which the caller reads as "stop the agent". A lone Ctrl-C at the prompt
    raises nothing: it clears what is typed. The fallback :func:`input` raises
    ``KeyboardInterrupt`` for every Ctrl-C; the caller decides what each one means.
    """

    def __init__(
        self,
        console: Console,
        theme: Theme,
        *,
        history_path: Path | None = None,
        enable_enhanced: bool = True,
        auto_mode: Callable[[], bool] | None = None,
        on_toggle_auto: Callable[[], None] | None = None,
        voice: VoiceInput | None = None,
    ) -> None:
        """``auto_mode`` says whether auto-approve is on, and is read on every
        repaint; ``on_toggle_auto`` flips it, and is what Shift+Tab calls. The
        session owns the flag, not the editor: ``/auto`` changes it too.
        """
        self._console = console
        self._theme = theme
        self._auto_mode = auto_mode
        self._on_toggle_auto = on_toggle_auto
        # Push-to-talk, when voice is on: see `voice.py`. Held here because the
        # editor owns the keys and the rows above the prompt.
        self.voice = voice
        self._session: Any = None
        self._bindings: Any = None
        # Borrowed by the first mid-turn read, so it has something to give back.
        # None until then: the idle prompt keeps what the session was built with.
        self._menu_space: int | None = None
        self._choice: _Choice | None = None
        # Which of the two reads is live. One set of bindings serves both, and
        # Ctrl-C means something different in each: see `_setup_enhanced`.
        self._alongside_turn = False
        # When the Ctrl-C a second one would pair with was pressed, or None.
        self._armed_at: float | None = None
        self._hint_repaint: asyncio.TimerHandle | None = None
        # What has been printed through the proxy of a read alongside a turn, and
        # how much of it is on the terminal yet: see `OutputProgress`.
        self.output = OutputProgress()

        if enable_enhanced and interactive_terminal():
            self._setup_enhanced(history_path or _HISTORY_PATH)

    @property
    def enhanced(self) -> bool:
        return self._session is not None

    @property
    def supports_concurrent_input(self) -> bool:
        """Whether a read may run while something else writes to the terminal.

        Only the enhanced editor can: the fallback is a blocking :func:`input` that
        cannot be cancelled, and one left pending would swallow the next turn's
        line. Scripted use stays sequential — a pipe has nobody at it to change
        their mind.
        """
        return self._session is not None

    def offer_choice(
        self, question: str, options: list[str], *, start: int = 0
    ) -> asyncio.Future[int | None]:
        """Put a pick-one question inside the editor; the future is the answer.

        Drawn above the line being typed, so the editor stays usable while it is
        open — a message or a slash command can still be written and sent. On an
        empty line ↑/↓ move and Enter picks; once something is typed those keys
        are the editor's again, and Enter sends the text, not the choice. The one
        exception is a line holding nothing but an option's number: Enter picks
        that option, and the highlight and hint say so before it is pressed; ↑/↓
        erase the number and move on from the option it named.
        Digits are never taken from the line — any of them may start a message.
        Ctrl-C answers ``None``: the question is what it interrupts, not the
        session.

        Only the enhanced editor can carry one; ask :attr:`supports_concurrent_input`.
        """
        answer: asyncio.Future[int | None] = asyncio.get_running_loop().create_future()
        self._choice = _Choice(question, list(options), start, answer)
        self._redraw()
        return answer

    def withdraw_choice(self) -> None:
        """Take the question off the editor, answering ``None`` if nobody did."""
        choice, self._choice = self._choice, None
        if choice is not None:
            choice.settle(None)
            self._redraw()

    def toggle_auto(self) -> None:
        """What Shift+Tab does: flip auto-approve and show it at once."""
        if self._on_toggle_auto is None:
            return
        self._on_toggle_auto()
        self._redraw()

    def _auto_on(self) -> bool:
        return self._auto_mode is not None and bool(self._auto_mode())

    def _auto_rows(self) -> list[tuple[str, str]]:
        """The auto-approve line, drawn directly above the ``you`` prompt.

        Nothing at all when it is off: the line is there to say the session is
        not asking, and a line that is always there stops being read — the key
        is announced at startup, in /help and on every approval instead.

        Above the prompt, not in prompt_toolkit's bottom toolbar: that one sits
        under the rows reserved for the completion menu, eight lines away from
        the text it is about, and is not drawn at all by a terminal that does
        not answer a cursor position request. Left out of the final render, so
        a submitted message does not leave a copy of it in the scrollback.
        """
        from prompt_toolkit.application.current import get_app

        if not self._auto_on() or get_app().is_done:
            return []
        rows = [("class:auto-mode", f"{self._theme.symbols.auto_mode} auto-approve on")]
        if self._on_toggle_auto is not None:
            rows.append(("class:auto-mode-hint", " · shift+tab to turn off"))
        rows.append(("", "\n"))
        return rows

    def _voice_rows(self) -> list[tuple[str, str]]:
        """The voice line, above the auto-approve one; left out of the final
        render like it, so a sent message leaves no copy in the scrollback."""
        from prompt_toolkit.application.current import get_app

        if self.voice is None or get_app().is_done:
            return []
        return self.voice.rows()

    def refresh(self) -> None:
        """Repaint now rather than at the next tick, if a read is on screen.

        For what changes faster than the status timer: the answer's unfinished
        end, drawn above the prompt while it streams.
        """
        self._redraw()

    def _redraw(self) -> None:
        app = getattr(self._session, "app", None)
        if app is not None and getattr(app, "is_running", False):
            app.invalidate()

    def draft(self) -> str:
        """Text typed but not submitted, or ``""``.

        Read *before* cancelling a read: ``CancelledError`` discards the buffer,
        and losing a half-typed sentence to an event the user did not cause is what
        stops them typing while the agent works.
        """
        session = self._session
        if session is None:
            return ""
        return str(getattr(getattr(session, "default_buffer", None), "text", "") or "")

    @contextlib.contextmanager
    def _transient(self) -> Iterator[None]:
        """Make the editor erase its own line when the read ends.

        A read alongside a turn is a widget, not a transcript entry: left behind it
        drops a ``you >`` frozen at an arbitrary instant into the middle of the
        agent's work, once per exit. Nothing is lost — the REPL prints the message
        where it took effect.

        Flipped on the application because ``erase_when_done`` is an
        ``Application`` constructor argument read at exit, with no per-call
        keyword. Restored after: the same session serves the between-turns read,
        whose line *is* the transcript's anchor.
        """
        app = getattr(self._session, "app", None)
        if app is None:
            yield
            return
        previous = getattr(app, "erase_when_done", False)
        app.erase_when_done = True
        try:
            yield
        finally:
            app.erase_when_done = previous

    def _setup_enhanced(self, history_path: Path) -> None:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.filters import Condition, has_completions
        from prompt_toolkit.history import FileHistory, InMemoryHistory
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.keys import Keys

        history: Any
        try:
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history = FileHistory(str(history_path))
        except OSError:
            # A read-only home (containers, CI) must not break the session.
            history = InMemoryHistory()

        bindings = KeyBindings()
        alongside_turn = Condition(lambda: self._alongside_turn)

        @bindings.add(Keys.Enter)
        def _submit(event: Any) -> None:
            # A line sent closes whatever the last Ctrl-C opened: otherwise a press
            # made just before sending pairs with one made in the next read, for an
            # unrelated reason.
            self._disarm()
            event.current_buffer.validate_and_handle()

        @bindings.add(Keys.Escape, Keys.Enter)
        def _newline_alt(event: Any) -> None:
            event.current_buffer.insert_text("\n")

        @bindings.add("c-j")
        def _newline_ctrl_j(event: Any) -> None:
            event.current_buffer.insert_text("\n")

        # Only the key, not ``<sigint>``: in raw mode Ctrl-C arrives as a key, so a
        # real SIGINT is somebody else asking the process to stop, and it keeps
        # prompt_toolkit's own meaning. So does a dumb terminal (TERM=dumb): there
        # prompt_toolkit swaps in a minimal prompt that ranks its own Ctrl-C above
        # these, and a Ctrl-C quits at once, as it always did.
        @bindings.add("c-c", filter=~alongside_turn)
        def _clear_or_quit(event: Any) -> None:
            # Text is looked at first, and a press with text never quits, however
            # soon after another it comes: the one before may be the Ctrl-C that
            # interrupted a turn, and the line typed since is somebody's next
            # message, not a confirmation.
            #
            # Quitting is Ctrl-D's EOFError rather than an exception of its own:
            # the REPL already ends the session on it and releases what the tools
            # hold on the way out, and a second way to say it would be a second
            # way out to keep correct.
            buffer = event.current_buffer
            if buffer.text:
                buffer.reset()
                self._arm(event.app)
            elif self._armed():
                self._disarm()
                event.app.exit(exception=EOFError(), style="class:exiting")
            else:
                self._arm(event.app)

        @bindings.add("c-c", filter=alongside_turn)
        def _interrupt_turn(event: Any) -> None:
            # prompt_toolkit's own Ctrl-C, which the REPL turns into "Turn
            # interrupted." The arming is the only addition: a quick second press
            # lands on the prompt the interrupted turn gives back, and quits there.
            self._arm(event.app)
            event.app.exit(exception=KeyboardInterrupt(), style="class:aborting")

        # A choice open in the editor takes the keys an empty line has no use
        # for. Registered after the ones above so they win while it is open.
        choosing = Condition(lambda: self._choice is not None)
        on_empty_line = Condition(lambda: not self.draft())
        names_an_option = Condition(lambda: self._typed_pick() is not None)

        # A line holding only an option's number keeps the arrows too: the
        # highlight already sits on that option, so ↑/↓ move on from there and the
        # number is erased — it named a row, it was never the start of a message,
        # and recalling history over it is not what the screen promised.
        @bindings.add("up", filter=choosing & (on_empty_line | names_an_option))
        def _choice_up(event: Any) -> None:
            self._move_choice(-1, event.current_buffer)

        @bindings.add("down", filter=choosing & (on_empty_line | names_an_option))
        def _choice_down(event: Any) -> None:
            self._move_choice(1, event.current_buffer)

        @bindings.add(Keys.Enter, filter=choosing & (on_empty_line | names_an_option))
        def _choice_pick(event: Any) -> None:
            typed = self._typed_pick()
            choice = self._choice
            # Reset, not submitted: the number answered the question, it is not a
            # message, and it has no place in the history either.
            event.current_buffer.reset()
            if choice is not None:
                self._settle_choice(typed if typed is not None else choice.index)

        @bindings.add("c-c", filter=choosing)
        def _choice_abort(event: Any) -> None:
            self._settle_choice(None)

        # Auto-approve, alongside a turn as well as between turns: this editor is
        # the one reading in both. Not while the completion menu is open, where
        # Shift+Tab already means "previous entry".
        @bindings.add("s-tab", filter=~has_completions)
        def _toggle_auto(event: Any) -> None:
            self.toggle_auto()

        # Push-to-talk. Only on an empty line, so a space in a sentence is a
        # space, and never while a question is open: approvals stay on the
        # keyboard. Every space goes to voice while it is recording or
        # transcribing: they are the repeats of the key being held.
        voice_on = Condition(lambda: self.voice is not None and self.voice.enabled)
        voice_busy = Condition(lambda: self.voice is not None and self.voice.busy)

        @bindings.add(" ", filter=voice_on & ~choosing & (on_empty_line | voice_busy))
        def _push_to_talk(event: Any) -> None:
            voice = self.voice
            if voice is None:
                return
            starting = not voice.busy
            voice.on_space(event.app)
            if starting and voice.recording and self._alongside_turn:
                # Talking over the agent stops it, as Ctrl-C would, and what is
                # said becomes the next message. The recording carries on into
                # the read that follows the interrupted turn.
                event.app.exit(exception=KeyboardInterrupt(), style="class:aborting")

        self._bindings = bindings
        self._session = PromptSession(
            completer=WordCompleter(completion_names(), sentence=True),
            complete_while_typing=True,
            history=history,
            style=self._prompt_style(),
            color_depth=self._color_depth(),
        )

    def _move_choice(self, step: int, buffer: Any) -> None:
        choice = self._choice
        if choice is None:
            return
        typed = self._typed_pick()
        if typed is not None:
            choice.index = typed
            buffer.reset()
        choice.index = (choice.index + step) % len(choice.options)

    def _settle_choice(self, picked: int | None) -> None:
        choice, self._choice = self._choice, None
        if choice is not None:
            choice.settle(picked)

    def _typed_pick(self) -> int | None:
        """The option the line names by number, if a choice is open and it does."""
        from domyn.agents.interfaces.cli.select import typed_number

        choice = self._choice
        if choice is None:
            return None
        return typed_number(self.draft(), len(choice.options))

    def _armed(self) -> bool:
        """Whether a Ctrl-C now would be the second of a pair."""
        return self._armed_at is not None and monotonic() - self._armed_at < _DOUBLE_PRESS_S

    def _arm(self, app: Any) -> None:
        self._armed_at = monotonic()
        # The hint is worked out from `_armed_at` on every repaint, so all this has
        # to do is ask for one once the window has closed. The timer changes
        # nothing itself, which is why one outliving the read that set it is
        # harmless: the application belongs to the session, shared by every read,
        # and a repaint of whichever read is live draws it correctly. A little
        # late on purpose: the event loop runs callbacks up to its clock
        # resolution early, and a repaint inside the window would keep the hint.
        if self._hint_repaint is not None:
            self._hint_repaint.cancel()
        self._hint_repaint = asyncio.get_running_loop().call_later(
            _DOUBLE_PRESS_S + 0.05, app.invalidate
        )
        app.invalidate()

    def _disarm(self) -> None:
        self._armed_at = None
        if self._hint_repaint is not None:
            self._hint_repaint.cancel()
            self._hint_repaint = None

    def _exit_hint(self) -> list[tuple[str, str]]:
        """``press Ctrl-C again to quit`` on a row of its own while that is true.

        A lone Ctrl-C on an empty line otherwise does nothing visible, and a key
        that seems dead is a key people keep pressing. The idle editor's first
        row, so it lands right under the last line printed (``Completed in``,
        after a turn) and in its column. On the right-hand side of the prompt it
        sat a screen width from where the eye is, and went unseen.

        Never alongside a turn: a Ctrl-C there stops the turn, it does not quit.
        """
        if self._alongside_turn or not self._armed():
            return []
        # A finished read's last frame stays behind as the anchor of the message
        # sent from it, and a hint drawn there would outlive what it says.
        if self._session.app.is_done:
            return []
        return [("class:exit-hint", "  press Ctrl-C again to quit\n")]

    def _prompt_style(self) -> Any:
        """Style for the prompt prompt_toolkit draws itself.

        The editor renders outside rich, so ``Styles.user_prompt`` never reaches it
        and the marker stayed uncoloured. Both come from the same ``Styles.brand``
        hex, so they cannot drift apart.
        """
        from prompt_toolkit.styles import Style

        # Here, not at the top: select.py reads lines through this module.
        from domyn.agents.interfaces.cli.select import CHOICE_STYLES

        if not self._theme.capabilities.color:
            return None
        return Style.from_dict(
            {
                "prompt": f"bold fg:{self._theme.styles.brand}",
                # Yellow like approvals: this CLI's colour for "waiting on
                # something". Not brand — that marks a transcript exchange.
                "queue-tag": "bold fg:ansiyellow",
                "queue-text": "fg:ansibrightblack",
                # The spinner's own colour: same signal, wherever it is drawn.
                # Split where the spinner splits it when it owns a line of its
                # own: a dim clock, then the moving track in the brand colour.
                # One style for both makes the line look different depending on
                # who happened to draw it.
                "status-clock": "fg:ansibrightblack",
                "status-track": f"fg:{self._theme.styles.brand}",
                "auto-mode": f"bold fg:{self._theme.styles.auto}",
                "auto-mode-hint": "fg:ansibrightblack",
                # Red while the microphone is open: it should be impossible to
                # miss that it is.
                "voice-recording": "bold fg:ansired",
                "voice-hint": "fg:ansibrightblack",
                "voice-notice": "fg:ansiyellow",
                # The picker's own look, so a question reads the same whether
                # it is drawn here or on its own (setup, /model).
                **CHOICE_STYLES,
                # Dim like the queue: an instruction about the prompt, not part
                # of the conversation. `nobold` because the editor draws its
                # whole message as `class:prompt`, which is bold, and the hint is
                # part of that message: it keeps the weight of the line above it.
                "exit-hint": "fg:ansibrightblack nobold",
            }
        )

    def _color_depth(self) -> Any:
        """Match the editor's colour depth to the rest of the CLI.

        prompt_toolkit defaults to 8-bit whatever the terminal can do, so it draws
        the brand blue as the nearest palette entry while rich, right below, prints
        the exact one. rich has already detected it; borrow the answer.
        """
        from prompt_toolkit.output import ColorDepth

        if getattr(self._console, "color_system", None) == "truecolor":
            return ColorDepth.TRUE_COLOR
        return None

    async def ask(
        self,
        label: str = "you",
        *,
        default: str = "",
        status: Callable[[], tuple[str, str]] | None = None,
        concurrent: bool = False,
        pending: Callable[[], list[tuple[str, str]]] | None = None,
        preview: Callable[[], str] | None = None,
    ) -> str:
        """Return the next user message, stripped.

        Async because the caller is: ``PromptSession.prompt()`` calls
        ``asyncio.run()`` internally and raises inside a running loop.
        ``prompt_async`` is the same editor on the loop already running.

        ``default`` seeds the buffer, so a draft interrupted by something the user
        did not do comes back intact. ``status`` is drawn on its own line above the
        prompt and refreshed on a timer: while a turn runs it carries what the
        spinner would say, the spinner being unable to share the bottom line with
        an editor. Above rather than to the right of it, where it would sit a
        screen width from the text it reports on. ``concurrent`` says something
        else is writing to this terminal too; see :func:`shared_stdout`.

        ``pending`` returns ``(tag, text)`` rows for what the user has typed that
        has not taken effect yet. In the prompt rather than the transcript because
        they are not history: printed at submit time a queued message leaves a copy
        stranded mid-answer, while inside the editor's repaint region a row can
        vanish the moment the runner takes it in. Callable for the same reason
        ``status`` is — nothing else asks it to re-read.

        ``preview`` returns the unfinished end of the answer being written, as
        ANSI text, drawn at the top of the editor's region: the part that is not
        final yet, so not in the transcript yet, still has somewhere to be seen.
        Re-read on every repaint, like ``pending``.

        All five are ignored on the fallback path, which must keep calling
        :func:`input` exactly as before.
        """
        symbol = self._theme.symbols.prompt

        if self._session is None:
            # No editor and no Shift+Tab here, so the label carries it.
            if self._auto_on():
                label = f"{label} [auto]"
            # markup=False: rich would read `[auto]` as a tag and delete it.
            self._console.print(
                f"{label} {symbol} ",
                end="",
                style=self._theme.style("user_prompt"),
                markup=False,
            )
            # Blocking on purpose: nothing else to run, and a thread would break
            # the tests and scripted callers that patch ``input``.
            return input().strip()

        # Set on the session, not passed to the call: these are *sticky* — a
        # keyword is assigned only when not None, so nothing means "back to the
        # default", and the idle prompt would inherit a stopped turn's clock.
        self._alongside_turn = concurrent
        voice_on = self.voice is not None and self.voice.enabled
        self._session.refresh_interval = (
            _STATUS_REFRESH_S
            if status is not None or pending is not None or preview is not None or voice_on
            else 0.0
        )
        # No reserved menu space while a turn runs: the default eight free rows
        # are invisible when nothing else draws, and an eight-line hole scrolled
        # through on every line of output when the renderer is.
        if concurrent:
            if self._menu_space is None:
                self._menu_space = getattr(self._session, "reserve_space_for_menu", 8)
            self._session.reserve_space_for_menu = 0
        elif self._menu_space is not None:
            self._session.reserve_space_for_menu = self._menu_space

        # prompt_toolkit draws its own prompt, so it must not go through rich.
        kwargs: dict[str, Any] = {
            "key_bindings": self._bindings,
            "multiline": True,
            "default": default,
        }
        if self.voice is not None:
            voice = self.voice
            kwargs["pre_run"] = lambda: voice.attach(self._session.app)

        # Always the live prompt, re-read on every repaint, so the auto-approve
        # line comes and goes with Shift+Tab without restarting the read.
        prompt = self._live_prompt(label, symbol, pending, status, preview)
        if not concurrent:
            text: str = await self._session.prompt_async(prompt, **kwargs)
            return text.strip()

        with shared_stdout(self.output), self._transient():
            text = await self._session.prompt_async(prompt, **kwargs)
        return text.strip()

    def _live_prompt(
        self,
        label: str,
        symbol: str,
        pending: Callable[[], list[tuple[str, str]]] | None,
        status: Callable[[], tuple[str, str]] | None,
        preview: Callable[[], str] | None = None,
    ) -> Callable[[], Any]:
        """A prompt that carries the queue and the activity line above the cursor.

        Re-evaluated on every repaint, so rows come and go without the read being
        restarted. Truncated rather than wrapped: wrapping moves the cursor line
        down mid-sentence, and these are a status display, not the message.

        The activity line goes last, directly over the ``you``: it is the row that
        changes every repaint, so the queue above it stays where the eye left it.
        """
        from prompt_toolkit.formatted_text import ANSI, FormattedText, to_formatted_text

        mark = self._theme.symbols.bullet
        tag_width = 6

        def message() -> Any:
            # The answer's unfinished end first, right under the part of it that
            # is already printed: it is the same text, continued.
            fragments: list[Any] = []
            if preview is not None:
                text = preview()
                if text:
                    # prompt_toolkit draws the whole message as `class:prompt`, the
                    # prompt's bold brand blue: the answer keeps its own style.
                    fragments.extend(to_formatted_text(ANSI(text), style="noinherit"))
            fragments.extend(self._choice_rows())
            limit = max(24, self._console.width - tag_width - 8)
            for tag, text in pending() if pending is not None else []:
                one_line = " ".join(text.split())
                if len(one_line) > limit:
                    one_line = one_line[: limit - 1].rstrip() + "…"
                fragments.append(("class:queue-tag", f"{mark} {tag:<{tag_width}}"))
                fragments.append(("class:queue-text", f" {one_line}\n"))
            if status is not None:
                clock, track = status()
                # Either half alone is still worth a row: the clock is only there
                # during a model call, the track whenever there is animation.
                if clock or track:
                    # Track then clock, which is the order `Spinner._draw` uses
                    # on a line of its own. Same indicator, same layout.
                    if track:
                        fragments.append(("class:status-track", f"  {track}"))
                        if clock:
                            fragments.append(("class:status-clock", f"  {clock}"))
                    else:
                        fragments.append(("class:status-clock", f"  {clock}"))
                    fragments.append(("", "\n"))
            # Before the auto-approve line, which stays directly over `you`. Only
            # ever drawn between turns, where nothing above it is, so it is the
            # first row: right under the last line printed.
            fragments.extend(self._exit_hint())
            fragments.extend(self._voice_rows())
            fragments.extend(self._auto_rows())
            fragments.append(("class:prompt", f"{label} {symbol} "))
            return FormattedText(fragments)

        return message

    def _choice_rows(self) -> list[tuple[str, str]]:
        """The open choice, if any, as prompt fragments ending in a newline.

        First, above the queue and the status: it is the thing waiting on the
        user, and the rows under it change on every repaint. Wrapped by the
        editor rather than cut: an approval's option carries the pattern it
        grants.
        """
        choice = self._choice
        if choice is None:
            return []
        from domyn.agents.interfaces.cli.select import choice_rows

        typed = self._typed_pick()
        if typed is None:
            index = choice.index
            keys = [
                "enter select",
                f"1-{len(choice.options)} + enter pick",
                "ctrl-c deny",
                *(["shift+tab auto-approve"] if self._on_toggle_auto is not None else []),
                "or type a message or /command",
            ]
        else:
            # Spelled out because a bare number is also something a person might
            # mean to send: Enter must not do something the screen did not say.
            index = typed
            arrow = "\u2192" if self._theme.capabilities.unicode else "->"
            keys = [f"enter {arrow} {choice.options[typed]}", "add text to send as a message"]
        return choice_rows(self._theme, choice.question, choice.options, index, keys)
