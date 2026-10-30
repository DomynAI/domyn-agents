"""Tests for a transcript written underneath a live editor.

Mid-turn steering puts a prompt_toolkit application on the bottom line for a
whole turn, which changes what "printing" means. Everything here is one of two
rules:

* **Nothing the editor drew may survive in the scrollback.** Its line is a
  widget, repainted as output scrolls past, so left behind it drops a ``you >``
  frozen at an arbitrary instant into the agent's work, once per exit. The REPL
  prints the messages themselves, where they took effect.
* **Nothing may be written without a trailing newline.** The editor repaints
  from wherever the cursor was left, so the tail of a partial line is inside the
  region it erases next. Writes close together survive by accident — the stdout
  proxy batches them — which is why a tool call printed in two halves looked
  fine while a token stream a quarter of a second apart vanished.
"""

import io
import re
from typing import Any

import pytest
from rich.console import Console

from domyn.agents import BaseEvent, ExecutionEventType, Part
from domyn.agents.interfaces.cli import Capabilities, Theme
from domyn.agents.interfaces.cli.prompt import InputSession
from domyn.agents.interfaces.cli.renderer import EventRenderer


def _renderer(*, live: bool) -> tuple[EventRenderer, io.StringIO]:
    buffer = io.StringIO()
    console = Console(file=buffer, width=60, no_color=True)
    renderer = EventRenderer(console, Theme.build(allow_animation=False))
    renderer.begin_turn()
    renderer.set_live_prompt(live)
    return renderer, buffer


def _lines(buffer: io.StringIO) -> list[str]:
    plain = re.sub(r"\x1b\[[0-9;]*m", "", buffer.getvalue())
    return [line.rstrip() for line in plain.splitlines()]


def _token(text: str) -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.RESPONSE_TOKEN,
        author="agent",
        content=[Part(text=text)],
        is_partial=True,
    )


def _response(text: str) -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.RESPONSE, author="agent", content=[Part(text=text)]
    )


def _user_input(text: str) -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text=text)]
    )


class TestTheAnswerSurvivesALivePrompt:
    def test_every_write_ends_a_line(self) -> None:
        """The regression that motivated all of this: streamed word by word under
        a live editor, the answer reached the terminal and was then erased by the
        next repaint — the turn ended with no answer in the transcript at all."""
        renderer, buffer = _renderer(live=True)

        for word in ("Here ", "is ", "the ", "answer."):
            renderer.render(_token(word))
        renderer.finish()

        assert buffer.getvalue().endswith("\n")
        assert "Here is the answer." in " ".join(_lines(buffer))

    def test_it_is_still_written_a_piece_at_a_time(self) -> None:
        """Buffering strictly to the newlines the model writes would hold a
        one-paragraph answer back until the very end, which is the common case.
        It breaks at the width instead, where the terminal would have wrapped."""
        renderer, buffer = _renderer(live=True)

        for _ in range(12):
            renderer.render(_token("a long stretch of answer "))
        written = _lines(buffer)
        renderer.finish()

        assert written, "nothing was emitted until the stream closed"
        assert all(len(line) <= 60 for line in _lines(buffer))

    def test_the_marker_stays_on_the_first_line(self) -> None:
        """It is what makes an answer findable when scrolling back, and it cannot
        be printed on its own: alone it is a partial line."""
        renderer, buffer = _renderer(live=True)

        renderer.render(_token("The answer, at some length.\n"))
        renderer.finish()

        lines = [line for line in _lines(buffer) if line]
        assert lines[0].startswith(Theme.build().symbols.agent)

    def test_an_unstreamed_answer_keeps_its_marker_too(self) -> None:
        """The non-streaming path opens the answer the same way, and used to lose
        the marker entirely once it stopped being printed separately."""
        renderer, buffer = _renderer(live=True)

        renderer.render(_response("Short."))

        assert [line for line in _lines(buffer) if line] == [
            f"{Theme.build().symbols.agent} Short."
        ]

    def test_without_a_live_prompt_nothing_changes(self) -> None:
        """Tokens go straight out, unwrapped and unbuffered: there is no editor to
        erase them, and word-by-word is what streaming is for."""
        renderer, buffer = _renderer(live=False)

        renderer.render(_token("Here "))

        assert buffer.getvalue().endswith("Here ")


class TestAMessageAppearsExactlyOnce:
    def test_the_echo_is_the_transcript_anchor(self) -> None:
        renderer, buffer = _renderer(live=True)

        renderer.echo_user("only the python files")

        assert _lines(buffer) == [f"you {Theme.build().symbols.prompt} only the python files"]

    def test_the_runner_echoing_it_back_is_not_shown_again(self) -> None:
        """The runner yields an absorbed message as USER_INPUT, seconds later at a
        tool-cycle boundary. Printed there as well, the same sentence would be in
        two places with the agent's work in between."""
        renderer, buffer = _renderer(live=True)

        renderer.echo_user("only the python files")
        renderer.render(_user_input("only the python files"))

        assert len(_lines(buffer)) == 1

    def test_a_message_from_another_front_end_is_shown(self) -> None:
        """It has never been on this screen, and this is the only place it can
        appear."""
        renderer, buffer = _renderer(live=True)

        renderer.render(_user_input("stop and summarise"))

        assert any("stop and summarise" in line for line in _lines(buffer))

    def test_two_identical_messages_are_two_lines(self) -> None:
        """Suppression consumes one echo, not every match: somebody who types the
        same thing twice meant it twice."""
        renderer, buffer = _renderer(live=True)

        renderer.echo_user("hurry up")
        renderer.echo_user("hurry up")
        renderer.render(_user_input("hurry up"))
        renderer.render(_user_input("hurry up"))

        assert len(_lines(buffer)) == 2


class _StubApp:
    """``erase_when_done`` lives on the application, not the session."""

    def __init__(self) -> None:
        self.erase_when_done = False


class _StubEditor:
    """A ``PromptSession`` with only the attributes ``ask`` reaches for."""

    def __init__(self) -> None:
        self.refresh_interval = 99.0
        self.reserve_space_for_menu = 8
        self.app = _StubApp()
        self.seen: list[dict[str, Any]] = []

    async def prompt_async(self, *args: Any, **kwargs: Any) -> str:
        self.seen.append(
            {
                "prompt": args[0] if args else None,
                "refresh_interval": self.refresh_interval,
                "reserve_space_for_menu": self.reserve_space_for_menu,
                "erase_when_done": self.app.erase_when_done,
            }
        )
        return "typed"


def _fragments(prompt: Any) -> list[tuple[str, str]]:
    """The rows the editor was handed, whether it got a callable or a literal."""
    return list(prompt() if callable(prompt) else prompt)


def _input_session() -> tuple[InputSession, _StubEditor]:
    session = InputSession(
        Console(file=io.StringIO(), width=80), Theme.build(), enable_enhanced=False
    )
    editor = _StubEditor()
    session._session = editor
    return session, editor


class TestTheEditorIsPutBackAsItWasFound:
    @pytest.mark.asyncio
    async def test_a_mid_turn_read_erases_itself(self) -> None:
        session, editor = _input_session()

        await session.ask(concurrent=True)

        assert editor.seen[0]["erase_when_done"] is True
        assert editor.app.erase_when_done is False, "the between-turns read must leave its line"

    @pytest.mark.asyncio
    async def test_the_status_does_not_outlive_the_turn(self) -> None:
        """These settings are sticky: a keyword is assigned only when not None, so
        nothing means "back to the default" and the idle prompt would inherit a
        stopped turn's clock."""
        session, editor = _input_session()

        await session.ask(status=lambda: "1.0s", concurrent=True)
        await session.ask()

        assert editor.seen[1]["refresh_interval"] == 0.0

    @pytest.mark.asyncio
    async def test_the_status_is_drawn_above_the_prompt(self) -> None:
        """Not on the right, a screen width from the text it reports on."""
        session, editor = _input_session()

        await session.ask(status=lambda: ("1.0s", "···●··"), concurrent=True)

        drawn = _fragments(editor.seen[0]["prompt"])
        # Two fragments, not one: the track carries the brand colour and the
        # clock is dim, exactly as `Spinner._draw` styles them on a line of its
        # own. A single fragment made the seconds blue here and not there.
        assert drawn[0] == ("class:status-track", "  ···●··")
        assert drawn[1] == ("class:status-clock", "  1.0s")
        assert drawn[2][1] == "\n"
        assert drawn[3][0] == "class:prompt"
        assert drawn[3][1].startswith("you ")

    @pytest.mark.asyncio
    async def test_the_status_sits_below_the_queue(self) -> None:
        """The row that changes every repaint goes last, so the queue above it
        stays where the eye left it."""
        session, editor = _input_session()

        await session.ask(
            status=lambda: ("1.0s", "···●··"),
            pending=lambda: [("steer", "hurry up")],
            concurrent=True,
        )

        classes = [style for style, _ in _fragments(editor.seen[0]["prompt"])]
        assert classes == [
            "class:queue-tag",
            "class:queue-text",
            "class:status-track",
            "class:status-clock",
            "",
            "class:prompt",
        ]

    @pytest.mark.asyncio
    async def test_the_completion_menu_space_comes_back(self) -> None:
        """Eight rows held free below the prompt are invisible when nothing else
        draws, and a hole at the foot of the screen when the renderer is."""
        session, editor = _input_session()

        await session.ask(concurrent=True)
        await session.ask()

        assert editor.seen[0]["reserve_space_for_menu"] == 0
        assert editor.seen[1]["reserve_space_for_menu"] == 8


class TestTheAnswerAboveThePrompt:
    """While the answer streams, its unfinished end is drawn by the editor, at the
    top of the region it repaints: right under the part already printed, which
    it continues."""

    @staticmethod
    def _message(**kwargs: Any) -> str:
        session = InputSession(
            Console(file=io.StringIO()), Theme.build(allow_animation=False), enable_enhanced=False
        )
        message = session._live_prompt("you", ">", **kwargs)
        return "".join(fragment[1] for fragment in message())

    def test_it_comes_first_above_the_queue_and_the_clock(self) -> None:
        drawn = self._message(
            pending=lambda: [("queued", "do this next")],
            status=lambda: ("1.2s", ""),
            preview=lambda: "\x1b[1mhalf a line\x1b[0m\n",
        )
        assert drawn.startswith("half a line\n")
        assert drawn.index("half a line") < drawn.index("do this next") < drawn.index("1.2s")

    def test_nothing_to_show_draws_nothing(self) -> None:
        with_it = self._message(pending=None, status=None, preview=lambda: "")
        assert with_it == self._message(pending=None, status=None)

    def test_it_keeps_the_answers_style_not_the_prompts(self) -> None:
        """prompt_toolkit draws the whole message as ``class:prompt``, bold brand
        blue: a preview that took it read blue and bold until the line was printed
        as the plain text it is."""
        from prompt_toolkit.formatted_text import to_formatted_text

        base = Theme.build(allow_animation=False)
        theme = type(base)(
            capabilities=Capabilities(color=True, unicode=True, animation=False),
            symbols=base.symbols,
            styles=base.styles,
        )
        session = InputSession(Console(file=io.StringIO()), theme, enable_enhanced=False)
        message = session._live_prompt(
            "you", ">", pending=None, status=None, preview=lambda: "plain \x1b[35mcode\x1b[0m\n"
        )
        style = session._prompt_style()
        # As prompt_toolkit draws a prompt's message.
        drawn = to_formatted_text(message(), style="class:prompt")
        cells = [
            (char, style.get_attrs_for_style_str(fragment[0]))
            for fragment in drawn
            for char in fragment[1]
        ]
        text = "".join(char for char, _ in cells)
        plain, code, prompt = (text.index(word) for word in ("plain", "code", "you"))
        assert not any(attrs.bold or attrs.color for _, attrs in cells[plain : plain + 5])
        assert all(attrs.color == "ansimagenta" for _, attrs in cells[code : code + 4])
        assert all(attrs.bold and attrs.color for _, attrs in cells[prompt : prompt + 3])


class TestWhatTheProxyHasWritten:
    """The editor draws a printed line until the proxy has written it: the proxy
    writes from a thread, up to its flush interval later, and repaints happen in
    between. So it counts what it is handed and what it has written."""

    @staticmethod
    def _quiet_terminal() -> Any:
        from prompt_toolkit.application import create_app_session
        from prompt_toolkit.output import DummyOutput

        return create_app_session(output=DummyOutput())

    def test_what_goes_in_comes_out(self) -> None:
        import sys
        import time

        from domyn.agents.interfaces.cli.prompt import OutputProgress, shared_stdout

        progress = OutputProgress()
        with self._quiet_terminal(), shared_stdout(progress):
            sys.stdout.write("a line\n")
            assert progress.sent == len("a line\n")
            deadline = time.monotonic() + 2
            while progress.written < progress.sent and time.monotonic() < deadline:
                time.sleep(0.01)
            assert progress.written == progress.sent

    def test_closing_it_settles_the_count(self) -> None:
        """Text left without a newline is dropped on close; a count still waiting
        for it would keep a line in the editor for the rest of the session."""
        import sys

        from domyn.agents.interfaces.cli.prompt import OutputProgress, shared_stdout

        progress = OutputProgress()
        with self._quiet_terminal(), shared_stdout(progress):
            sys.stdout.write("no newline")
        assert progress.written == progress.sent == len("no newline")
