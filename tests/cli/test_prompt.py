"""Tests for reading user input.

The bug these exist to prevent: the enhanced editor was driven through
``PromptSession.prompt()``, the synchronous entry point, from a REPL that runs
inside an event loop. ``prompt()`` calls ``asyncio.run()`` internally, so the
first keystroke of a real session died with "asyncio.run() cannot be called from
a running event loop" — while every test passed, because a test has no TTY and
therefore never builds the enhanced session at all.

So the tests below drive the enhanced path explicitly, from inside a running
loop, rather than letting capability detection quietly skip it.
"""

import asyncio
import contextlib
import io
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from domyn.agents.interfaces.cli import Theme
from domyn.agents.interfaces.cli import prompt as prompt_mod
from domyn.agents.interfaces.cli.prompt import InputSession


def _session(**kwargs: Any) -> tuple[InputSession, io.StringIO]:
    buffer = io.StringIO()
    console = Console(file=buffer, width=80)
    return InputSession(console, Theme.build(), **kwargs), buffer


class StubEditor:
    """Stands in for ``PromptSession``, exposing both entry points.

    Having ``prompt`` present and recording its use is the point: the test can
    then assert the *synchronous* one was never touched.
    """

    def __init__(self, reply: str = "hello") -> None:
        self.reply = reply
        self.sync_calls = 0
        self.async_calls = 0

    def prompt(self, *args: Any, **kwargs: Any) -> str:
        self.sync_calls += 1
        # What the real one does from inside a loop.
        asyncio.get_running_loop()
        raise AssertionError("the synchronous entry point must not be used")

    async def prompt_async(self, *args: Any, **kwargs: Any) -> str:
        self.async_calls += 1
        return self.reply


class TestEnhancedEditor:
    @pytest.mark.asyncio
    async def test_it_is_driven_by_the_running_loop(self) -> None:
        """The regression test: this is called from inside an event loop, which
        is exactly where the synchronous entry point blows up."""
        session, _ = _session(enable_enhanced=False)
        editor = StubEditor("write the release notes")
        session._session = editor

        assert await session.ask() == "write the release notes"
        assert editor.async_calls == 1
        assert editor.sync_calls == 0

    @pytest.mark.asyncio
    async def test_input_is_stripped(self) -> None:
        session, _ = _session(enable_enhanced=False)
        session._session = StubEditor("  padded  \n")
        assert await session.ask() == "padded"

    @pytest.mark.asyncio
    async def test_the_label_reaches_the_editor(self) -> None:
        session, _ = _session(enable_enhanced=False)
        captured: list[Any] = []

        class Recording(StubEditor):
            async def prompt_async(self, *args: Any, **kwargs: Any) -> str:
                captured.append(args[0])
                return ""

        session._session = Recording()
        await session.ask("you")
        # A callable, re-read on every repaint so the label can follow Shift+Tab.
        message = captured[0]() if callable(captured[0]) else captured[0]
        assert "you" in str(message)


class TestFallback:
    @pytest.mark.asyncio
    async def test_plain_input_is_used_without_the_editor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session, buffer = _session(enable_enhanced=False)
        monkeypatch.setattr("builtins.input", lambda: "  typed  ")
        assert await session.ask() == "typed"
        assert "you" in buffer.getvalue(), "rich draws the prompt when the editor does not"

    @pytest.mark.asyncio
    async def test_eof_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The REPL reads Ctrl-D as "end the session", so it must reach it."""

        def _eof() -> str:
            raise EOFError

        session, _ = _session(enable_enhanced=False)
        monkeypatch.setattr("builtins.input", _eof)
        with pytest.raises(EOFError):
            await session.ask()


class TestCapabilityGating:
    def test_a_piped_stdin_never_builds_the_editor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``printf ... | domyn run`` has no terminal for prompt_toolkit to
        drive: it warns, fails to start its loop and takes the session with it."""
        monkeypatch.setattr("sys.stdin", io.StringIO("piped"))
        session, _ = _session(enable_enhanced=True)
        assert not session.enhanced

    def test_a_terminal_always_gets_the_editor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """prompt_toolkit is a dependency of the CLI: no plain-input mode on a TTY."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
        session, buffer = _session(enable_enhanced=True)
        assert session.enhanced
        assert buffer.getvalue() == ""

    def test_a_read_only_home_falls_back_to_memory_history(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Containers and CI mount read-only homes; losing history is fine,
        failing to start is not."""
        pytest.importorskip("prompt_toolkit")
        monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
        blocked = tmp_path / "blocked"
        blocked.write_text("not a directory")
        session, _ = _session(enable_enhanced=True, history_path=blocked / "sub" / "history")
        assert session.enhanced

    def test_the_editor_exposes_the_async_entry_point(self) -> None:
        """Guards the assumption the fix rests on, against a prompt_toolkit
        upgrade that renames or drops it."""
        pytest.importorskip("prompt_toolkit")
        from prompt_toolkit import PromptSession

        assert callable(PromptSession.prompt_async)


class TestPromptStyling:
    """The editor draws its own prompt, outside rich, so the theme has to be
    handed to it explicitly — otherwise the user prompt stays uncoloured while
    everything rich prints below it is themed."""

    def _themed(self, *, color: bool, truecolor: bool) -> InputSession:
        from domyn.agents.interfaces.cli import Capabilities

        base = Theme.build()
        theme = type(base)(
            capabilities=Capabilities(color=color, unicode=True, animation=False),
            symbols=base.symbols,
            styles=base.styles,
        )
        console = Console(
            file=io.StringIO(),
            force_terminal=True,
            color_system="truecolor" if truecolor else "standard",
        )
        return InputSession(console, theme, enable_enhanced=False)

    def test_the_prompt_carries_the_brand_colour(self) -> None:
        pytest.importorskip("prompt_toolkit")
        session = self._themed(color=True, truecolor=True)
        rendered = str(session._prompt_style().style_rules)
        assert "#1f6dff" in rendered

    def test_it_is_the_same_hex_the_renderer_uses(self) -> None:
        """One constant, so the prompt and the agent's marker cannot drift."""
        from domyn.agents.interfaces.cli.theme import Styles

        assert Styles.brand in Styles.user_prompt
        assert Styles.brand == Styles.agent_marker

    def test_no_style_without_colour_support(self) -> None:
        pytest.importorskip("prompt_toolkit")
        assert self._themed(color=False, truecolor=False)._prompt_style() is None

    def test_truecolor_is_requested_when_the_terminal_has_it(self) -> None:
        """prompt_toolkit defaults to 8-bit whatever the terminal can do, and
        renders the brand blue as its nearest palette entry — a visibly
        different blue from the one rich prints on the next line."""
        pytest.importorskip("prompt_toolkit")
        from prompt_toolkit.output import ColorDepth

        assert self._themed(color=True, truecolor=True)._color_depth() is ColorDepth.TRUE_COLOR

    def test_the_default_is_left_alone_on_a_limited_terminal(self) -> None:
        pytest.importorskip("prompt_toolkit")
        assert self._themed(color=True, truecolor=False)._color_depth() is None


class TestTheClockAboveTheInput:
    """What the editor draws while a turn runs, in place of the spinner.

    Where it sits and how it is styled are pinned in ``test_live_prompt.py``,
    next to the rest of the live-prompt behaviour. What is left here is what the
    line has to keep being true of, repaint after repaint.
    """

    def _session(self) -> InputSession:
        import io

        from rich.console import Console

        from domyn.agents.interfaces.cli import Theme

        return InputSession(
            Console(file=io.StringIO()), Theme.build(allow_animation=False), enable_enhanced=False
        )

    def test_it_is_re_read_on_every_repaint(self) -> None:
        """A clock drawn once is a clock that stopped."""
        ticks = iter([("1.0s", "o."), ("1.1s", ".o")])
        message = self._session()._live_prompt("you", ">", pending=None, status=lambda: next(ticks))
        assert "1.0s" in "".join(t for _, t in message())
        assert "1.1s" in "".join(t for _, t in message())

    def test_its_width_never_changes(self) -> None:
        """It shares a screen with an input somebody is typing into, and a width
        that moved would drag the line under it sideways ten times a second."""
        session = self._session()

        def width(clock: str, track: str) -> int:
            message = session._live_prompt("you", ">", pending=None, status=lambda: (clock, track))
            return len("".join(text for _, text in message()))

        assert width(" 1.0s", "o.......") == width("12.3s", "....o...")

    def test_a_terminal_without_animation_still_gets_the_clock(self) -> None:
        """The clock alone answers the only question the line is asked — slow, or
        hung? — and a track of punctuation that cannot move answers nothing."""
        message = self._session()._live_prompt(
            "you", ">", pending=None, status=lambda: ("1.2s", "")
        )
        assert "1.2s" in "".join(t for _, t in message())


class TestReadLine:
    """The one-line reads outside the session editor: approvals, wizard fields."""

    @pytest.mark.asyncio
    async def test_a_terminal_reads_through_the_editor_in_its_own_thread(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Approvals are asked from inside the REPL's running loop, where the
        synchronous ``prompt()`` refuses to start without ``in_thread``."""
        import prompt_toolkit

        calls: list[dict[str, Any]] = []

        class Stub:
            def prompt(self, message: Any, **kwargs: Any) -> str:
                calls.append({"message": message, **kwargs})
                return "y"

        monkeypatch.setattr(prompt_toolkit, "PromptSession", Stub)
        monkeypatch.setattr(prompt_mod, "interactive_terminal", lambda: True)
        monkeypatch.setattr("builtins.input", lambda: pytest.fail("input() on a terminal"))

        assert prompt_mod.read_line(Console(), "  run it? [Y/n] ") == "y"
        assert calls[0]["in_thread"] is True
        assert "run it?" in "".join(t for _, t in calls[0]["message"].__pt_formatted_text__())

    def test_a_pipe_reads_with_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(prompt_mod, "interactive_terminal", lambda: False)
        monkeypatch.setattr("builtins.input", lambda: "piped")
        buffer = io.StringIO()
        assert prompt_mod.read_line(Console(file=buffer), "label: ") == "piped"
        assert "label: " in buffer.getvalue()

    @pytest.mark.parametrize("terminal", [True, False], ids=["terminal", "pipe"])
    def test_a_value_in_brackets_is_shown_as_typed(
        self, monkeypatch: pytest.MonkeyPatch, terminal: bool
    ) -> None:
        """`[auto]`, a model name, a lowercase pattern: rich reads each as a markup
        tag and deletes it, and the current value a prompt offers goes missing."""
        import prompt_toolkit

        shown: list[str] = []

        class Stub:
            def prompt(self, message: Any, **kwargs: Any) -> str:
                shown.append("".join(t for _, t in message.__pt_formatted_text__()))
                return ""

        monkeypatch.setattr(prompt_toolkit, "PromptSession", Stub)
        monkeypatch.setattr(prompt_mod, "interactive_terminal", lambda: terminal)
        monkeypatch.setattr("builtins.input", lambda: "")
        buffer = io.StringIO()

        prompt_mod.read_line(Console(file=buffer), "  context window [auto]: ")

        assert "[auto]" in (shown[0] if terminal else buffer.getvalue())

    def test_a_prefilled_password_is_masked_on_a_pipe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(prompt_mod, "interactive_terminal", lambda: False)
        monkeypatch.setattr("builtins.input", lambda: "")
        buffer = io.StringIO()

        answer = prompt_mod.read_line(
            Console(file=buffer), "  API key: ", password=True, prefill="sk-secret"
        )

        assert answer == "sk-secret"
        assert "sk-secret" not in buffer.getvalue()
        assert "[*********]" in buffer.getvalue()

    def test_a_url_in_the_prompt_is_not_eaten_as_markup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The wizard shows the current value in brackets so Enter keeps it.
        Rich reads `[http://…]` as a style tag and renders nothing, so the
        prompt for a *required* url looked empty on every `domyn setup` re-run
        and invited retyping a url that was already right."""
        monkeypatch.setattr(prompt_mod, "interactive_terminal", lambda: False)
        monkeypatch.setattr("builtins.input", lambda: "")
        buffer = io.StringIO()
        prompt_mod.read_line(Console(file=buffer), "  url [http://localhost:28080]: ")
        assert "http://localhost:28080" in buffer.getvalue()

    def test_a_bracketed_value_that_is_a_real_style_name_survives_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`[red]` is both a plausible value and a style rich would apply."""
        monkeypatch.setattr(prompt_mod, "interactive_terminal", lambda: False)
        monkeypatch.setattr("builtins.input", lambda: "")
        buffer = io.StringIO()
        prompt_mod.read_line(Console(file=buffer), "  colour [red]: ")
        assert "[red]" in buffer.getvalue()


class TestAChoiceInsideTheEditor:
    """An approval asked inside the live editor, which stays usable meanwhile.

    Driven through a real ``PromptSession`` on a pipe, because what is under test
    is which key binding wins: a stub editor has none.
    """

    async def _run(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, keys: str) -> Any:
        from prompt_toolkit.application import create_app_session
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        monkeypatch.setattr(prompt_mod, "interactive_terminal", lambda: True)
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
            session, _ = _session(history_path=tmp_path / "history")
            assert session.enhanced
            read = asyncio.ensure_future(session.ask(pending=list))
            answer = session.offer_choice("run it?", ["yes", "always", "no"])
            await asyncio.sleep(0.05)
            pipe.send_text(keys)
            await asyncio.wait({read, answer}, timeout=2, return_when=asyncio.FIRST_COMPLETED)
            await asyncio.sleep(0.05)
            outcome = (
                answer.result() if answer.done() else "open",
                read.result() if read.done() else "reading",
                session._choice_rows() != [],
            )
            read.cancel()
            session.withdraw_choice()
            return outcome

    @pytest.mark.asyncio
    async def test_arrows_and_enter_on_an_empty_line_pick(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        assert await self._run(monkeypatch, tmp_path, "\x1b[B\r") == (1, "reading", False)

    @pytest.mark.asyncio
    async def test_typed_text_is_sent_and_the_choice_stays_open(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        assert await self._run(monkeypatch, tmp_path, "/auto\r") == ("open", "/auto", True)

    @pytest.mark.asyncio
    async def test_a_lone_option_number_picks_it(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        assert await self._run(monkeypatch, tmp_path, " 2 \r") == (1, "reading", False)

    @pytest.mark.asyncio
    async def test_an_arrow_after_a_number_erases_it_and_moves_from_its_option(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # 3 names "no"; up moves to "always" and empties the line, so Enter picks.
        assert await self._run(monkeypatch, tmp_path, "3\x1b[A\r") == (1, "reading", False)

    @pytest.mark.asyncio
    async def test_an_arrow_after_a_number_does_not_recall_history(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / "history").write_text("\n# 2026-01-01\n+earlier message\n")
        assert await self._run(monkeypatch, tmp_path, "2\x1b[A\r") == (0, "reading", False)

    @pytest.mark.asyncio
    async def test_a_message_starting_with_a_number_is_sent_not_picked(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        outcome = await self._run(monkeypatch, tmp_path, "2 more things\r")
        assert outcome == ("open", "2 more things", True)

    @pytest.mark.asyncio
    async def test_a_number_naming_no_option_is_sent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        assert await self._run(monkeypatch, tmp_path, "7\r") == ("open", "7", True)

    @pytest.mark.asyncio
    async def test_ctrl_c_answers_none_instead_of_interrupting(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        assert await self._run(monkeypatch, tmp_path, "\x03") == (None, "reading", False)

    @pytest.mark.asyncio
    async def test_the_rows_draw_the_question_and_mark_the_selected_option(self) -> None:
        session, _ = _session(enable_enhanced=False)
        session.offer_choice("run it?", ["yes", "no"], start=1)
        rows = session._choice_rows()
        assert "run it?" in rows[0][1]
        selected = [text for style, text in rows if style == "class:choice-selected"]
        assert len(selected) == 1 and selected[0].rstrip().endswith("no")
        assert "type a message" in rows[-1][1]
        session.withdraw_choice()

    @pytest.mark.asyncio
    async def test_a_typed_number_moves_the_highlight_and_says_what_enter_does(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session, _ = _session(enable_enhanced=False)
        session.offer_choice("run it?", ["yes", "always", "no"], start=0)
        monkeypatch.setattr(session, "draft", lambda: "3")
        rows = session._choice_rows()
        session.withdraw_choice()
        selected = [text for style, text in rows if style == "class:choice-selected"]
        assert selected[0].rstrip().endswith("3. no")
        assert "no" in rows[-1][1] and "send as a message" in rows[-1][1]

    @pytest.mark.asyncio
    async def test_the_rows_look_like_the_setup_picker(self) -> None:
        """Same fragments, same styles, as the picker setup and /model draw."""
        from domyn.agents.interfaces.cli.select import CHOICE_STYLES

        session, _ = _session(enable_enhanced=False)
        session.offer_choice("run it?", ["yes", "no"], start=0)
        rows = session._choice_rows()
        session.withdraw_choice()
        used = {style.removeprefix("class:") for style, _ in rows if style}
        assert used <= set(CHOICE_STYLES)
        assert rows[-1][1].strip().startswith(("\u2191\u2193 move", "up/down move"))


class TestShiftTabAutoApprove:
    """Auto-approve from the keyboard, in the editor that reads between turns and
    alongside them alike. Driven through a real ``PromptSession`` on a pipe for
    the key itself, because what matters is that the binding wins."""

    @staticmethod
    def _wired(**kwargs: Any) -> tuple[InputSession, dict[str, bool]]:
        flag = {"on": False}

        def flip() -> None:
            flag["on"] = not flag["on"]

        session, _ = _session(auto_mode=lambda: flag["on"], on_toggle_auto=flip, **kwargs)
        return session, flag

    async def _press(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, keys: str
    ) -> tuple[bool, str]:
        """Press ``keys`` in a live read; the flag, and the prompt as drawn then."""
        from prompt_toolkit.application import create_app_session
        from prompt_toolkit.formatted_text import fragment_list_to_text, to_formatted_text
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        monkeypatch.setattr(prompt_mod, "interactive_terminal", lambda: True)
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
            session, flag = self._wired(history_path=tmp_path / "history")
            read = asyncio.ensure_future(session.ask())
            await asyncio.sleep(0.05)
            pipe.send_text(keys)
            await asyncio.sleep(0.1)
            drawn = fragment_list_to_text(to_formatted_text(session._session.message))
            read.cancel()
            return flag["on"], drawn

    @pytest.mark.asyncio
    async def test_shift_tab_puts_the_line_right_above_the_prompt(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Directly over `you`, not in the bottom toolbar: that one sits under the
        rows reserved for the completion menu, a screen's worth of blank lines
        away from the prompt it is about."""
        on, drawn = await self._press(monkeypatch, tmp_path, "\x1b[Z")
        assert on is True
        assert drawn.splitlines()[-2:] == [
            f"{Theme.build().symbols.auto_mode} auto-approve on · shift+tab to turn off",
            f"you {Theme.build().symbols.prompt} ",
        ]

    @pytest.mark.asyncio
    async def test_pressed_again_the_line_goes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        on, drawn = await self._press(monkeypatch, tmp_path, "\x1b[Z\x1b[Z")
        assert on is False
        assert "auto-approve" not in drawn

    @pytest.mark.asyncio
    async def test_it_is_not_left_in_the_scrollback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The final render of a submitted line stays in the transcript; the line
        is about the editor, not about the message."""
        session, flag = self._wired(enable_enhanced=False)
        flag["on"] = True
        app = MagicMock(is_done=True)
        monkeypatch.setattr("prompt_toolkit.application.current.get_app", lambda: app)
        assert session._auto_rows() == []
        app.is_done = False
        assert session._auto_rows()[0][0] == "class:auto-mode"

    def test_the_colour_is_orange_not_the_approval_yellow(self) -> None:
        session = TestPromptStyling()._themed(color=True, truecolor=True)
        styles = dict(session._prompt_style().style_rules)
        assert styles["auto-mode"] == f"bold fg:{Theme.build().styles.auto}"
        assert "yellow" not in styles["auto-mode"]

    @pytest.mark.asyncio
    async def test_without_the_editor_the_label_says_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pipe has no editor to draw the line and no Shift+Tab: the label is
        all there is. Printed without markup, or rich deletes `[auto]` as a tag."""
        session, flag = self._wired(enable_enhanced=False)
        flag["on"] = True
        monkeypatch.setattr("builtins.input", lambda: "hi")
        buffer = session._console.file
        await session.ask()
        assert "you [auto]" in buffer.getvalue()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_an_open_choice_offers_the_key(self) -> None:
        session, _ = self._wired(enable_enhanced=False)
        session.offer_choice("run it?", ["yes", "no"])
        assert "shift+tab auto-approve" in session._choice_rows()[-1][1]
        session.withdraw_choice()

        plain, _ = _session(enable_enhanced=False)
        plain.offer_choice("run it?", ["yes", "no"])
        assert "shift+tab" not in plain._choice_rows()[-1][1]
        plain.withdraw_choice()


class TestTheKeyIsAnnouncedAtStartup:
    """Nothing shows auto-approve while it is off, so the startup line is where
    the key is learnt."""

    async def _status_line(self, monkeypatch: pytest.MonkeyPatch, terminal: bool) -> str:
        from domyn.agents import Agent
        from domyn.agents.cli.commands.run import command

        buffer = io.StringIO()
        monkeypatch.setattr(command, "console", Console(file=buffer, width=200))
        monkeypatch.setattr(command, "interactive_terminal", lambda: terminal)
        agent = Agent(name="Assistant", description="d", instruction="i", llm_provider=MagicMock())

        async def done() -> int:
            return 0

        with patch.object(command, "Repl", new=MagicMock(return_value=MagicMock())) as repl:
            repl.return_value.run = MagicMock(side_effect=lambda *a, **k: done())
            await command.start_session(
                agent,
                Theme.build(allow_animation=False),
                initial_prompt=None,
                verbose=False,
                show_status=True,
            )
        return buffer.getvalue()

    @pytest.mark.asyncio
    async def test_a_terminal_is_told_about_shift_tab(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert "shift+tab auto-approve" in await self._status_line(monkeypatch, terminal=True)

    @pytest.mark.asyncio
    async def test_a_pipe_is_not(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert "shift+tab" not in await self._status_line(monkeypatch, terminal=False)


class _Clock:
    """What ``monotonic`` returns to the code under test.

    Stopped by default, so whether two presses pair is decided by the test rather
    than by how fast the machine happened to process them. ``step`` moves every
    reading past the last one, which is presses made far apart.
    """

    def __init__(self) -> None:
        self.now = 1000.0
        self.step = 0.0

    def __call__(self) -> float:
        self.now += self.step
        return self.now


@contextlib.contextmanager
def _keyboard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, output: Any = None
) -> Iterator[tuple[InputSession, Any, _Clock]]:
    """A real editor, typed into through a pipe, on a clock the test controls.

    Real rather than a stub because what is under test is which of two bindings
    for one key wins: prompt_toolkit's own Ctrl-C aborts the read, and a stub
    editor has no opinion about that.
    """
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    clock = _Clock()
    monkeypatch.setattr("domyn.agents.interfaces.cli.prompt.monotonic", clock)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    # On a dumb terminal prompt_toolkit swaps in a minimal prompt that ranks its own
    # Ctrl-C above ours, and the shell running the tests must not pick which one is
    # under test.
    monkeypatch.setenv("TERM", "xterm-256color")
    with (
        create_pipe_input() as keys,
        create_app_session(input=keys, output=output or DummyOutput()),
    ):
        session, _ = _session(enable_enhanced=True, history_path=tmp_path / "history")
        assert session.enhanced
        yield session, keys, clock


async def _until(condition: Callable[[], bool], timeout: float = 2.0) -> None:
    """Wait for the editor to have drawn something: it repaints on its own schedule."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        if loop.time() > deadline:
            raise AssertionError("the editor never drew it")
        await asyncio.sleep(0.01)


class TestCtrlC:
    """At the prompt Ctrl-C clears what is typed and a second one straight after
    quits; while a turn runs it stops the turn. Every keystroke goes through
    prompt_toolkit's key processor, as a terminal's would."""

    @pytest.mark.asyncio
    async def test_it_clears_what_is_typed_and_the_read_goes_on(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """What the feature is for: a Ctrl-C used to end the session and take the
        half-typed message with it."""
        with _keyboard(monkeypatch, tmp_path) as (session, keys, _):
            keys.send_text("half a thought\x03the real message\r")
            assert await session.ask() == "the real message"

    @pytest.mark.asyncio
    async def test_once_on_an_empty_line_does_not_quit(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with _keyboard(monkeypatch, tmp_path) as (session, keys, _):
            keys.send_text("\x03still here\r")
            assert await session.ask() == "still here"

    @pytest.mark.asyncio
    async def test_twice_in_a_row_quits_like_ctrl_d(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``EOFError``, because that is what the REPL already ends the session on,
        releasing what the tools hold on the way out."""
        with _keyboard(monkeypatch, tmp_path) as (session, keys, _):
            keys.send_text("\x03\x03")
            with pytest.raises(EOFError):
                await session.ask()

    @pytest.mark.asyncio
    async def test_twice_with_text_clears_then_quits(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with _keyboard(monkeypatch, tmp_path) as (session, keys, _):
            keys.send_text("a draft\x03\x03")
            with pytest.raises(EOFError):
                await session.ask()

    @pytest.mark.asyncio
    async def test_presses_far_apart_never_pair(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with _keyboard(monkeypatch, tmp_path) as (session, keys, clock):
            clock.step = 1.0
            keys.send_text("\x03\x03\x03still here\r")
            assert await session.ask() == "still here"

    @pytest.mark.asyncio
    async def test_a_press_with_text_never_quits(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """However soon after another it comes: the press before may be the one
        that stopped a turn, and what was typed since is a message."""
        with _keyboard(monkeypatch, tmp_path) as (session, keys, _):
            keys.send_text("\x03a new line\x03sent\r")
            assert await session.ask() == "sent"

    @pytest.mark.asyncio
    async def test_sending_a_line_ends_the_pairing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Otherwise a press made just before sending pairs with the next one,
        made in another read for another reason."""
        with _keyboard(monkeypatch, tmp_path) as (session, keys, _):
            keys.send_text("\x03hi\r")
            assert await session.ask() == "hi"
            keys.send_text("\x03ok\r")
            assert await session.ask() == "ok"

    @pytest.mark.asyncio
    async def test_during_a_turn_it_still_interrupts(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The REPL's "Turn interrupted." hangs on this ``KeyboardInterrupt``."""
        with _keyboard(monkeypatch, tmp_path) as (session, keys, _):
            keys.send_text("\x03")
            with pytest.raises(KeyboardInterrupt):
                await session.ask(concurrent=True)

    @pytest.mark.asyncio
    async def test_a_quick_press_after_an_interrupted_turn_quits(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with _keyboard(monkeypatch, tmp_path) as (session, keys, _):
            keys.send_text("\x03")
            with pytest.raises(KeyboardInterrupt):
                await session.ask(concurrent=True)
            keys.send_text("\x03")
            with pytest.raises(EOFError):
                await session.ask()

    @pytest.mark.asyncio
    async def test_no_hint_alongside_a_turn(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A turn can start straight after an interrupted one, on a queued message,
        with the press still counted: a Ctrl-C there stops the turn, so a hint
        saying it quits would be false."""
        pytest.importorskip("prompt_toolkit")
        from prompt_toolkit.formatted_text import fragment_list_to_text, to_formatted_text

        with _keyboard(monkeypatch, tmp_path) as (session, keys, _):
            keys.send_text("\x03")
            with pytest.raises(KeyboardInterrupt):
                await session.ask(concurrent=True)
            read = asyncio.ensure_future(session.ask(concurrent=True))
            await asyncio.sleep(0.05)
            drawn = fragment_list_to_text(to_formatted_text(session._session.message))
            read.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await read

        assert "press Ctrl-C again" not in drawn

    @pytest.mark.asyncio
    async def test_the_hint_appears_once_a_second_press_would_quit(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A lone Ctrl-C on an empty line otherwise changes nothing on screen. On a
        row of its own at the left margin, the first of the editor: on the right of
        the prompt it sat a screen width from the cursor and went unseen."""
        pytest.importorskip("prompt_toolkit")
        from prompt_toolkit.formatted_text import fragment_list_to_text, to_formatted_text
        from prompt_toolkit.output.plain_text import PlainTextOutput

        screen = io.StringIO()
        with _keyboard(monkeypatch, tmp_path, output=PlainTextOutput(screen)) as (
            session,
            keys,
            _,
        ):
            read = asyncio.ensure_future(session.ask())
            await _until(lambda: "you" in screen.getvalue())
            assert "press Ctrl-C again" not in screen.getvalue()

            keys.send_text("\x03")
            await _until(lambda: "press Ctrl-C again to quit" in screen.getvalue())
            drawn = fragment_list_to_text(to_formatted_text(session._session.message))
            keys.send_text("still here\r")
            assert await read == "still here"

        assert drawn.splitlines() == [
            "  press Ctrl-C again to quit",
            f"you {Theme.build().symbols.prompt} ",
        ]

    @pytest.mark.asyncio
    async def test_the_hint_goes_once_a_second_press_would_no_longer_quit(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Nothing in the read changes when the window closes: only the repaint
        `_arm` schedules takes the hint away, and it has to land after the window."""
        monkeypatch.setattr(prompt_mod, "_DOUBLE_PRESS_S", 0.2)
        with _keyboard(monkeypatch, tmp_path) as (session, keys, _):
            # A real clock this time: what is under test is the timer against it.
            monkeypatch.setattr(prompt_mod, "monotonic", time.monotonic)
            app = session._session.app
            hints: list[object] = []
            repaint = app.invalidate

            def recording() -> None:
                hints.append(session._exit_hint())
                repaint()

            monkeypatch.setattr(app, "invalidate", recording)

            read = asyncio.ensure_future(session.ask())
            await asyncio.sleep(0.05)
            keys.send_text("\x03")
            await _until(lambda: any(hints))
            await asyncio.sleep(0.4)

            assert hints[-1] == []
            keys.send_text("still here\r")
            assert await read == "still here"

    @pytest.mark.asyncio
    async def test_the_hint_is_not_left_in_the_scrollback(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A finished read's last frame stays in the scrollback; a hint drawn there
        would outlive what it says. Ctrl-D ends the read while still armed."""
        pytest.importorskip("prompt_toolkit")
        from prompt_toolkit.output.plain_text import PlainTextOutput

        screen = io.StringIO()
        with _keyboard(monkeypatch, tmp_path, output=PlainTextOutput(screen)) as (
            session,
            keys,
            _,
        ):
            read = asyncio.ensure_future(session.ask())
            await _until(lambda: "you" in screen.getvalue())
            keys.send_text("\x03")
            await _until(lambda: "press Ctrl-C again" in screen.getvalue())
            before = len(screen.getvalue())

            keys.send_text("\x04")
            with pytest.raises(EOFError):
                await read

            assert "press Ctrl-C again" not in screen.getvalue()[before:]


class TestGoingBackInReadLine:
    """Inside ``back_key`` (the setup wizard) Ctrl-B goes back a question.

    Driven through a real ``PromptSession`` on a pipe: what is under test is the key
    binding, which a stub editor has none of.
    """

    def _read(self, monkeypatch: pytest.MonkeyPatch, keys: str) -> str:
        from prompt_toolkit.application import create_app_session
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        monkeypatch.setattr(prompt_mod, "interactive_terminal", lambda: True)
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
            pipe.send_text(keys)
            return prompt_mod.read_line(Console(file=io.StringIO()), "  Model: ", prefill="ab cd")

    def test_ctrl_b_goes_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with prompt_mod.back_key(), pytest.raises(prompt_mod.GoBack):
            self._read(monkeypatch, "\x02")

    def test_ctrl_b_moves_the_cursor_outside_the_wizard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Its usual meaning: a character to the left, then typed there.
        assert self._read(monkeypatch, "\x02X\r") == "ab cXd"

    def test_option_left_still_moves_a_word(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Option+← arrives as ESC b: one word left, then typed there.
        with prompt_mod.back_key():
            assert self._read(monkeypatch, "\x1bbX\r") == "ab Xcd"

    def test_an_answer_is_printed_back_inside_the_wizard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The line erases itself when it ends, so what stays is printed through the console.
        from prompt_toolkit.application import create_app_session
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        monkeypatch.setattr(prompt_mod, "interactive_terminal", lambda: True)
        out = io.StringIO()
        with (
            prompt_mod.back_key(),
            create_pipe_input() as pipe,
            create_app_session(input=pipe, output=DummyOutput()),
        ):
            pipe.send_text("m2\r")
            prompt_mod.read_line(Console(file=out), "  Model: ")
            pipe.send_text("sk-123\r")
            prompt_mod.read_line(Console(file=out), "  Key: ", password=True)

        assert out.getvalue() == "  Model: m2\n  Key: ******\n"

    def test_back_key_is_off_once_the_block_ends(self) -> None:
        with prompt_mod.back_key():
            assert prompt_mod.back_key_on()
        assert not prompt_mod.back_key_on()
