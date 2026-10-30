"""Tests for the activity line.

Almost all of these are about it getting *out of the way*. A spinner that draws
correctly and leaves debris in the transcript is a net loss, so what is pinned
here is: it erases itself before anything else prints, it never appears where
escape codes are unwelcome, and it cannot outlive the turn it belongs to.
"""

import asyncio
import io
import re

import pytest
from rich.console import Console

from domyn.agents.interfaces.cli import Capabilities, Theme
from domyn.agents.interfaces.cli import spinner as spinner_module
from domyn.agents.interfaces.cli.spinner import Spinner

ERASE = "\r\x1b[2K"


@pytest.fixture(autouse=True)
def _label_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Most tests read the label right after starting a step; the delay that
    hides it from quick steps has tests of its own."""
    monkeypatch.setattr(spinner_module, "_SHOW_AFTER_S", 0.0)


def _spinner(
    *, color: bool = True, unicode: bool = True, animation: bool = True
) -> tuple[Spinner, io.StringIO]:
    base = Theme.build()
    theme = type(base)(
        capabilities=Capabilities(color=color, unicode=unicode, animation=animation),
        symbols=base.symbols,
        styles=base.styles,
    )
    buffer = io.StringIO()
    return Spinner(Console(file=buffer), theme), buffer


def _screen(buffer: io.StringIO) -> list[str]:
    r"""What a terminal would actually show, honouring `\r` and `ESC[2K`.

    Flattening the escapes instead would report every erased frame as if it had
    stayed on screen, which is the opposite of what these tests are checking.
    """
    raw, lines, cur, col, i = buffer.getvalue(), [], [], 0, 0
    while i < len(raw):
        if raw.startswith("\x1b[2K", i):
            cur, col, i = [], 0, i + 4
            continue
        match = re.match(r"\x1b\[[0-9;?]*[a-zA-Z]", raw[i:])
        if match:
            i += match.end()
            continue
        ch, i = raw[i], i + 1
        if ch == "\r":
            col = 0
        elif ch == "\n":
            lines.append("".join(cur))
            cur, col = [], 0
        else:
            if col < len(cur):
                cur[col] = ch
            else:
                cur.append(ch)
            col += 1
    lines.append("".join(cur))
    return [line for line in lines if line.strip()]


def _frames(buffer: io.StringIO) -> list[str]:
    """Only the drawn frames. The trailing chunk after the last erase is the
    cursor-restore escape, which is not a frame."""
    return [chunk for chunk in buffer.getvalue().split(ERASE) if re.search("[●o]", chunk)]


def _clocks(buffer: io.StringIO) -> list[float]:
    return [float(m.group(1)) for f in _frames(buffer) if (m := re.search(r"(\d+\.\d+)s", f))]


class TestItDraws:
    @pytest.mark.asyncio
    async def test_it_shows_elapsed_time_and_a_ball(self) -> None:
        spinner, buffer = _spinner()
        await spinner.start()
        spinner.thinking()
        await asyncio.sleep(0.25)
        await spinner.stop()

        frames = _frames(buffer)
        assert frames, "nothing was drawn"
        assert re.search(r"\d+\.\d+s", frames[0]), "the wait time is the useful half"
        assert "●" in frames[0]

    @pytest.mark.asyncio
    async def test_the_ball_moves(self) -> None:
        spinner, buffer = _spinner()
        await spinner.start()
        await asyncio.sleep(0.3)
        await spinner.stop()

        positions = {frame.index("●") for frame in _frames(buffer)}
        assert len(positions) > 1, "a still ball says nothing a static line would not"

    @pytest.mark.asyncio
    async def test_it_carries_the_brand_colour(self) -> None:
        spinner, buffer = _spinner()
        await spinner.start()
        await asyncio.sleep(0.15)
        await spinner.stop()
        assert "38;2;31;109;255" in buffer.getvalue()


class TestItGetsOutOfTheWay:
    @pytest.mark.asyncio
    async def test_pausing_erases_the_line(self) -> None:
        """Otherwise the next thing printed lands on top of a half-drawn
        spinner and the transcript has to be squinted at."""
        spinner, buffer = _spinner()
        await spinner.start()
        await asyncio.sleep(0.15)
        spinner.pause()
        assert buffer.getvalue().endswith(ERASE + "\x1b[?25h")
        await spinner.stop()

    @pytest.mark.asyncio
    async def test_it_stops_drawing_while_paused(self) -> None:
        spinner, buffer = _spinner()
        await spinner.start()
        await asyncio.sleep(0.15)
        spinner.pause()
        before = len(buffer.getvalue())
        await asyncio.sleep(0.25)
        assert len(buffer.getvalue()) == before
        await spinner.stop()

    @pytest.mark.asyncio
    async def test_it_draws_again_after_resume(self) -> None:
        spinner, buffer = _spinner()
        await spinner.start()
        spinner.pause()
        spinner.resume()
        await asyncio.sleep(0.2)
        await spinner.stop()
        assert _frames(buffer)

    @pytest.mark.asyncio
    async def test_the_cursor_is_restored(self) -> None:
        """A hidden cursor left behind outlives the process and the user has to
        reset their terminal."""
        spinner, buffer = _spinner()
        await spinner.start()
        await asyncio.sleep(0.15)
        await spinner.stop()
        assert buffer.getvalue().endswith("\x1b[?25h")

    @pytest.mark.asyncio
    async def test_nothing_is_written_after_stop(self) -> None:
        """`stop` awaits the task, so a late frame cannot land on top of
        whatever the caller printed next."""
        spinner, buffer = _spinner()
        await spinner.start()
        await asyncio.sleep(0.15)
        await spinner.stop()
        settled = buffer.getvalue()
        await asyncio.sleep(0.25)
        assert buffer.getvalue() == settled


class TestTheClockTimesOneStep:
    @pytest.mark.asyncio
    async def test_no_clock_outside_a_model_call(self) -> None:
        """Between calls the ball alone says the process is alive; a clock there
        times a tool whose own output already says what it is doing."""
        spinner, buffer = _spinner()
        await spinner.start()
        await asyncio.sleep(0.2)
        await spinner.stop()
        assert _frames(buffer), "the ball still moves"
        assert not _clocks(buffer)

    @pytest.mark.asyncio
    async def test_it_stops_when_the_call_ends(self) -> None:
        spinner, buffer = _spinner()
        await spinner.start()
        spinner.thinking()
        await asyncio.sleep(0.2)
        spinner.idle()
        mark = len(buffer.getvalue())
        await asyncio.sleep(0.2)
        await spinner.stop()
        assert not re.search(r"\d+\.\d+s", buffer.getvalue()[mark:])

    @pytest.mark.asyncio
    async def test_each_call_restarts_it(self) -> None:
        """After four model calls "3.1s" is useful and "47s" is not."""
        spinner, buffer = _spinner()
        await spinner.start()
        spinner.thinking()
        await asyncio.sleep(0.3)
        spinner.idle()
        spinner.thinking()
        await asyncio.sleep(0.1)
        await spinner.stop()
        assert _clocks(buffer)[-1] < 0.3, "the clock should have restarted"

    @pytest.mark.asyncio
    async def test_a_pause_does_not_restart_it(self) -> None:
        """A tool line printed mid-call pauses the spinner; the model is still
        thinking, so the clock carries on."""
        spinner, buffer = _spinner()
        await spinner.start()
        spinner.thinking()
        await asyncio.sleep(0.3)
        spinner.pause()
        spinner.resume()
        await asyncio.sleep(0.1)
        await spinner.stop()
        assert _clocks(buffer)[-1] >= 0.3

    def test_it_says_which_kind_of_step(self) -> None:
        """A long wait means something different for each: a slow model, or a
        command that may be stuck."""
        spinner = _spinner()[0]
        spinner.thinking()
        thinking = spinner.status()[0]
        spinner.executing()
        executing = spinner.status()[0]
        assert thinking.startswith("thinking")
        assert executing.startswith("executing")
        assert len(thinking) == len(executing), "switching must not make the line twitch"

    def test_a_quick_step_shows_no_label(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A tool call that returns at once would only flash "executing"."""
        monkeypatch.setattr(spinner_module, "_SHOW_AFTER_S", 0.1)
        spinner = _spinner()[0]
        spinner.executing()
        assert spinner.status()[0] == ""

    def test_a_longer_step_shows_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import time

        monkeypatch.setattr(spinner_module, "_SHOW_AFTER_S", 0.1)
        spinner = _spinner()[0]
        spinner.executing()
        time.sleep(0.15)
        assert spinner.status()[0].startswith("executing")

    @pytest.mark.asyncio
    async def test_stopping_forgets_an_unfinished_call(self) -> None:
        """An interrupted turn never sees its LLM_END."""
        spinner, _ = _spinner()
        spinner.thinking()
        await spinner.stop()
        assert spinner.status()[0] == ""


class TestCapabilityGates:
    @pytest.mark.parametrize(
        ("color", "animation"),
        [(False, True), (True, False), (False, False)],
    )
    @pytest.mark.asyncio
    async def test_it_stays_silent_where_escapes_are_unwelcome(
        self, color: bool, animation: bool
    ) -> None:
        """A redirected stream or a dumb terminal would get a line of
        punctuation twitching in a log file."""
        spinner, buffer = _spinner(color=color, animation=animation)
        assert not spinner.enabled
        await spinner.start()
        await asyncio.sleep(0.2)
        await spinner.stop()
        assert buffer.getvalue() == ""

    @pytest.mark.asyncio
    async def test_it_has_an_ascii_ball(self) -> None:
        spinner, buffer = _spinner(unicode=False)
        await spinner.start()
        await asyncio.sleep(0.15)
        await spinner.stop()
        assert "o" in _frames(buffer)[0]
        assert "●" not in buffer.getvalue()


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_starting_twice_runs_one_task(self) -> None:
        spinner, _ = _spinner()
        await spinner.start()
        await spinner.start()
        await spinner.stop()

    @pytest.mark.asyncio
    async def test_stopping_without_starting_is_fine(self) -> None:
        spinner, buffer = _spinner()
        await spinner.stop()
        assert buffer.getvalue() == ""

    @pytest.mark.asyncio
    async def test_resume_before_start_does_nothing(self) -> None:
        spinner, buffer = _spinner()
        spinner.resume()
        await asyncio.sleep(0.15)
        assert buffer.getvalue() == ""


class TestTheRendererDrivesIt:
    def _renderer(self):  # type: ignore[no-untyped-def]
        from domyn.agents.interfaces.cli.renderer import EventRenderer

        spinner, buffer = _spinner()
        base = Theme.build(allow_animation=False)
        renderer = EventRenderer(Console(file=buffer, width=90), base, spinner=spinner)
        renderer.begin_turn()
        return renderer, spinner, buffer

    @pytest.mark.asyncio
    async def test_it_is_paused_before_output_and_resumed_after(self) -> None:
        from domyn.agents import BaseEvent, ExecutionEventType, Part

        renderer, spinner, _ = self._renderer()
        await spinner.start()
        renderer.render(
            BaseEvent(event_type=ExecutionEventType.RESPONSE, author="a", content=[Part(text="hi")])
        )
        assert spinner._drawing, "it should be running again once the line is free"
        await spinner.stop()

    @pytest.mark.asyncio
    async def test_it_stays_paused_while_an_answer_streams(self) -> None:
        """The answer is arriving on that same line; a spinner would fight it
        for the cursor."""
        from domyn.agents import BaseEvent, ExecutionEventType, Part

        renderer, spinner, _ = self._renderer()
        await spinner.start()
        renderer.render(
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE_TOKEN,
                author="a",
                content=[Part(text="par")],
            )
        )
        assert not spinner._drawing
        await spinner.stop()

    @pytest.mark.parametrize(
        ("opening", "closing", "label_after"),
        [
            ("LLM_START", "LLM_END", ""),
            ("LLM_START", "LLM_PROVIDER_ERROR", ""),
            ("LLM_START", "TOOL_START", "executing"),
            ("TOOL_START", "TOOL_END", ""),
            ("TOOL_START", "TOOL_ERROR", ""),
            ("TOOL_START", "TOOL_NEEDS_FEEDBACK", ""),
            ("TOOL_START", "LLM_START", "thinking"),
        ],
    )
    def test_events_drive_the_clock(self, opening: str, closing: str, label_after: str) -> None:
        from domyn.agents import BaseEvent, ExecutionEventType

        renderer, spinner, _ = self._renderer()
        assert spinner.status()[0] == ""
        renderer.render(BaseEvent(event_type=ExecutionEventType[opening], author="a"))
        assert spinner.status()[0].endswith("s")
        renderer.render(BaseEvent(event_type=ExecutionEventType[closing], author="a"))
        assert spinner.status()[0].split(" ")[0] == label_after

    def test_streaming_hides_it_until_the_call_ends(self) -> None:
        """While the answer is being written the text is the sign of life; the
        editor draws ``status()`` regardless, so it has to come back empty."""
        from domyn.agents import BaseEvent, ExecutionEventType, Part

        renderer, spinner, _ = self._renderer()
        renderer.render(BaseEvent(event_type=ExecutionEventType.LLM_START, author="a"))
        renderer.render(
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE_TOKEN,
                author="a",
                content=[Part(text="par")],
            )
        )
        assert spinner.status() == ("", "")
        renderer.render(BaseEvent(event_type=ExecutionEventType.LLM_END, author="a"))
        clock, track = spinner.status()
        assert clock == "" and "●" in track, "the ball is back between steps"

    @pytest.mark.asyncio
    async def test_finishing_a_turn_pauses_it(self) -> None:
        renderer, spinner, _ = self._renderer()
        await spinner.start()
        renderer.finish()
        assert not spinner._drawing
        await spinner.stop()

    @pytest.mark.asyncio
    async def test_rendered_as_markdown_it_goes_quiet_like_plain_text(self) -> None:
        """The markdown path is the same `render()`: once the answer streams there
        is no clock, in the spinner or in what the editor draws, even while a
        table is still being collected and nothing of it is on screen yet."""
        from domyn.agents import BaseEvent, ExecutionEventType, Part
        from domyn.agents.interfaces.cli.renderer import EventRenderer

        buffer = io.StringIO()
        base = Theme.build()
        theme = type(base)(
            capabilities=Capabilities(color=True, unicode=True, animation=True),
            symbols=base.symbols,
            styles=base.styles,
        )
        console = Console(file=buffer, width=80, height=40, force_terminal=True)
        spinner = Spinner(console, theme)
        renderer = EventRenderer(console, theme, spinner=spinner)
        renderer.begin_turn()

        def token(text: str) -> BaseEvent:
            return BaseEvent(
                event_type=ExecutionEventType.RESPONSE_TOKEN,
                author="a",
                content=[Part(text=text)],
            )

        assert renderer._markdown
        await spinner.start()
        renderer.render(BaseEvent(event_type=ExecutionEventType.LLM_START, author="a"))
        await asyncio.sleep(0.2)
        assert spinner.status()[0].startswith("thinking")

        renderer.render(token("| a | b |\n|---|---|\n| 1 | 2 |\n"))
        assert spinner.status() == ("", ""), "a clock under an answer being written"
        await asyncio.sleep(0.2)
        renderer.render(token("done\n"))
        await asyncio.sleep(0.2)
        renderer.finish()
        await spinner.stop()

        lines = _screen(buffer)
        assert lines[-1] == "done"
        assert not any("●" in line or "thinking" in line for line in lines), (
            "the spinner left debris behind"
        )


class TestLoggingDoesNotCollideWithIt:
    """Logging is the one writer that does not pass through the renderer: a
    handler writes to the stream on its own schedule, from anywhere in the
    library. Seen in a real session as
    `···●··········2026-08-02 20:56:17 - WARNING - ...`."""

    @pytest.mark.asyncio
    async def test_a_log_record_gets_a_clean_line(self) -> None:
        import logging

        spinner, buffer = _spinner()
        logging.basicConfig(level=logging.WARNING, stream=buffer, format="%(message)s", force=True)
        spinner.attach_to_logging()

        await spinner.start()
        await asyncio.sleep(0.2)
        logging.getLogger("t").warning("Malformed tool-call arguments, retrying (1/3)")
        await asyncio.sleep(0.15)
        await spinner.stop()

        assert _screen(buffer) == ["Malformed tool-call arguments, retrying (1/3)"]

    def test_attaching_twice_installs_one_filter(self) -> None:
        import logging

        spinner, buffer = _spinner()
        logging.basicConfig(level=logging.WARNING, stream=buffer, force=True)
        spinner.attach_to_logging()
        spinner.attach_to_logging()
        handler = logging.getLogger().handlers[0]
        assert sum(type(f).__name__ == "_PauseBeforeLogging" for f in handler.filters) == 1

    def test_it_filters_nothing(self) -> None:
        """It is a hook, not a filter: dropping a record would be a bug hiding
        behind a decoration."""
        import logging

        from domyn.agents.interfaces.cli.spinner import _PauseBeforeLogging

        spinner, _ = _spinner()
        record = logging.LogRecord("t", logging.WARNING, "", 0, "msg", None, None)
        assert _PauseBeforeLogging(spinner).filter(record) is True

    def test_a_disabled_spinner_attaches_nothing(self) -> None:
        import logging

        spinner, buffer = _spinner(color=False)
        logging.basicConfig(level=logging.WARNING, stream=buffer, force=True)
        before = len(logging.getLogger().handlers[0].filters)
        spinner.attach_to_logging()
        assert len(logging.getLogger().handlers[0].filters) == before


class TestTheSameLineWhenSomeoneElseDrawsIt:
    """While a turn runs the editor owns the bottom line, so the spinner hands
    over the text and lets it draw.

    Which means there are now two ways to draw one thing, and the risk is that
    they drift: the line would change appearance depending on who happened to be
    drawing it. That is not hypothetical — it happened, and what arrived was a
    clock in the brand colour and a ball on the far right of the screen.
    """

    @pytest.mark.asyncio
    async def test_the_two_halves_are_named_not_ordered(self) -> None:
        """Which comes first is the caller's business — and the caller lays them
        out as ``_draw`` does, so the line does not change shape depending on who
        drew it. What the spinner owes is that the pieces are distinguishable."""
        spinner, buffer = _spinner()
        spinner.thinking()
        await spinner.start()
        await asyncio.sleep(0.15)
        drawn = _frames(buffer)[0]
        clock, track = spinner.status()
        await spinner.stop()

        assert "s" in clock and "●" not in clock
        assert "●" in track and "s" not in track
        assert drawn.index("●") < drawn.index("s"), "the spinner leads with the track"

    def test_it_comes_apart_where_the_styling_changes(self) -> None:
        """Two pieces rather than one string, because they are not styled alike:
        the clock is dim and the track carries the brand colour. Handed over as
        one string, whoever draws it either loses that or has to guess where to
        cut — and guessing produced blue seconds."""
        spinner = _spinner()[0]
        spinner.thinking()
        clock, track = spinner.status()
        assert re.fullmatch(r"thinking +\d+\.\ds", clock.strip())
        assert "●" in track
        assert "●" not in clock

    def test_without_animation_the_clock_is_still_there(self) -> None:
        """A twitching line of punctuation in a log file helps nobody. The
        clock alone still answers the only question the line is asked: slow, or
        hung?"""
        spinner = _spinner(animation=False)[0]
        spinner.thinking()
        clock, track = spinner.status()
        assert re.fullmatch(r"thinking +\d+\.\ds", clock.strip())
        assert track == ""

    def test_the_mark_moves_with_time(self) -> None:
        import time

        spinner = _spinner()[0]
        positions = set()
        for _ in range(4):
            positions.add(spinner.status()[1].index("●"))
            time.sleep(0.1)
        assert len(positions) > 1, "a still mark says nothing a static line would not"

    def test_the_track_is_there_between_calls(self) -> None:
        """The editor keeps the row, so the prompt does not jump up and down
        each time a model call starts and ends."""
        clock, track = _spinner()[0].status()
        assert clock == ""
        assert "●" in track

    def test_its_width_never_changes_during_a_call(self) -> None:
        """A width that moved would make the line twitch ten times a second."""
        spinner = _spinner()[0]
        spinner.thinking()
        widths = set()
        for started in (0.0, 9.9, 12.3):
            spinner._phase = ("thinking", __import__("time").monotonic() - started)
            clock, track = spinner.status()
            widths.add(len(f"{clock}{track}"))
        assert len(widths) == 1

    def test_it_carries_no_escapes_of_its_own(self) -> None:
        """Whoever draws it owns the styling — here that is an editor, which
        would print the escapes rather than obey them."""
        spinner = _spinner()[0]
        spinner.thinking()
        clock, track = spinner.status()
        assert "\x1b" not in clock + track
