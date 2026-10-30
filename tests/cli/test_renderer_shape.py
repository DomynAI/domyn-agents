"""Tests for the shape of a rendered turn.

Not "does it print the text" — that was never the hard part. What matters is
whether a user can find the answer in a transcript full of trace, which comes
down to three things: the answer is flush left behind a marker, the work is
indented, and the reasoning appears exactly once however many events carry it.
"""

import io
import re

import pytest
from rich.console import Console

from domyn.agents.core import BaseEvent, ExecutionEventType, Part, ToolAction
from domyn.agents.interfaces.cli import Capabilities, Theme
from domyn.agents.interfaces.cli.renderer import EventRenderer


def _theme(*, color: bool = False) -> Theme:
    base = Theme.build(allow_animation=False)
    return type(base)(
        capabilities=Capabilities(color=color, unicode=True, animation=False),
        symbols=base.symbols,
        styles=base.styles,
    )


def _renderer(*, verbose: bool = False, color: bool = False) -> tuple[EventRenderer, io.StringIO]:
    buffer = io.StringIO()
    # color_system pinned: rich detects it from the environment even when the
    # terminal is forced, so on a CI runner the brand's 24-bit colour comes out
    # as the nearest 16-colour approximation and an assertion about the exact
    # escape fails for a reason that has nothing to do with the code.
    console = Console(
        file=buffer,
        width=100,
        force_terminal=color,
        no_color=not color,
        color_system="truecolor" if color else None,
    )
    renderer = EventRenderer(console, _theme(color=color), verbose=verbose)
    renderer.begin_turn()
    return renderer, buffer


def _lines(buffer: io.StringIO) -> list[str]:
    plain = re.sub(r"\x1b\[[0-9;]*m", "", buffer.getvalue())
    return [line.rstrip() for line in plain.splitlines()]


def _response(text: str = "The answer.") -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.RESPONSE, author="agent", content=[Part(text=text)]
    )


def _llm_end(thought: str) -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.LLM_END, author="agent", content=[Part(thought=thought)]
    )


def _tool_start(name: str = "search", thought: str | None = None, **params: object) -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.TOOL_START,
        author="agent",
        action=ToolAction(name=name, parameters=params, thought=thought),
    )


class TestTheAnswerIsFindable:
    def test_it_carries_a_marker_and_starts_at_the_left_margin(self) -> None:
        renderer, buffer = _renderer()
        renderer.render(_response("The answer."))
        answer = next(line for line in _lines(buffer) if "The answer." in line)
        assert answer.startswith("⏺ "), answer
        assert not answer.startswith(" ")

    def test_a_blank_line_separates_it_from_the_trace(self) -> None:
        renderer, buffer = _renderer()
        renderer.render(_tool_start())
        renderer.render(_response())
        lines = _lines(buffer)
        answer_at = next(i for i, line in enumerate(lines) if "The answer." in line)
        assert lines[answer_at - 1] == "", "the answer must be set apart from the work above it"

    def test_no_gap_when_there_was_no_trace(self) -> None:
        """On a plain question-and-answer turn the blank line is just a gap."""
        renderer, buffer = _renderer()
        renderer.render(_response())
        assert _lines(buffer)[0].startswith("⏺ ")


class TestTheWorkIsSubordinate:
    def test_tool_calls_are_indented(self) -> None:
        renderer, buffer = _renderer()
        renderer.render(_tool_start("search", query="x"))
        line = next(line for line in _lines(buffer) if "search" in line)
        assert line.startswith("  →"), line

    def test_tool_arguments_read_as_written(self) -> None:
        """Not a dict's repr, which doubled every backslash in the pattern."""
        renderer, buffer = _renderer()
        renderer.render(_tool_start("grep", pattern='"a\\|b"', path="src"))
        line = next(line for line in _lines(buffer) if "grep" in line)
        assert '→ grep "a\\|b"  path=src' in line, line

    def test_a_call_takes_one_line_however_long(self) -> None:
        """Cropped at the edge: wrapped, it ran back to the left margin."""
        renderer, buffer = _renderer()
        renderer.render(_tool_start("run_command", command="echo " + "x" * 300))
        lines = [line for line in _lines(buffer) if line]
        assert len(lines) == 1, lines
        assert lines[0].endswith("…"), lines

    def test_a_value_on_several_lines_is_counted_not_printed(self) -> None:
        renderer, buffer = _renderer()
        renderer.render(
            _tool_start("write_file", path="notes.txt", content="one\ntwo\nthree\n", reasoning="r")
        )
        assert _lines(buffer) == ["  → write_file notes.txt  content=(3 lines)"]

    def test_tool_results_are_indented_further_than_the_call(self) -> None:
        renderer, buffer = _renderer()
        renderer.render(_tool_start("search"))
        renderer.render(
            BaseEvent(
                event_type=ExecutionEventType.TOOL_END,
                author="agent",
                content=[Part(text="found 3 things")],
            )
        )
        lines = _lines(buffer)
        call = next(line for line in lines if "search" in line)
        result = next(line for line in lines if "found 3 things" in line)
        indent = lambda s: len(s) - len(s.lstrip())  # noqa: E731
        assert indent(result) > indent(call)

    def test_reasoning_is_indented_with_the_work(self) -> None:
        renderer, buffer = _renderer()
        renderer.render(_llm_end("I should search first."))
        assert _lines(buffer)[0].startswith("  I should search first.")

    def test_errors_are_indented_too(self) -> None:
        renderer, buffer = _renderer()
        renderer.render(
            BaseEvent(
                event_type=ExecutionEventType.ERROR_OCCURRED,
                author="agent",
                error_message="it broke",
            )
        )
        assert _lines(buffer)[0].startswith("  ")


class TestReasoningIsVisible:
    def test_it_shows_without_verbose(self) -> None:
        """It rides on LLM_END, which is otherwise verbose-only. Requiring
        --verbose to see the thinking would also turn on every lifecycle event."""
        renderer, buffer = _renderer(verbose=False)
        renderer.render(_llm_end("Let me think about this."))
        assert "Let me think about this." in buffer.getvalue()

    def test_the_rest_of_llm_end_stays_hidden(self) -> None:
        renderer, buffer = _renderer(verbose=False)
        renderer.render(_llm_end("visible reasoning"))
        assert "llm_end" not in buffer.getvalue().lower()

    def test_multiline_reasoning_keeps_its_indent(self) -> None:
        renderer, buffer = _renderer()
        renderer.render(_llm_end("first line\nsecond line"))
        assert all(line.startswith("  ") for line in _lines(buffer) if line)


class TestReasoningIsNotRepeated:
    def test_the_same_thought_on_llm_end_and_on_the_action_prints_once(self) -> None:
        """The planner attaches the reasoning to LLM_END *and* to every action it
        produced, so a turn with three tool calls would print it four times."""
        renderer, buffer = _renderer()
        thought = "I need to look this up."
        renderer.render(_llm_end(thought))
        renderer.render(_tool_start("search", thought=thought))
        renderer.render(_tool_start("fetch", thought=thought))
        assert buffer.getvalue().count(thought) == 1

    def test_different_reasoning_is_shown_each_time(self) -> None:
        renderer, buffer = _renderer()
        renderer.render(_llm_end("first thought"))
        renderer.render(_llm_end("second thought"))
        assert "first thought" in buffer.getvalue()
        assert "second thought" in buffer.getvalue()

    def test_a_new_turn_shows_it_again(self) -> None:
        """Repeating in a later turn is new information, not a duplicate."""
        renderer, buffer = _renderer()
        renderer.render(_llm_end("same thought"))
        renderer.begin_turn()
        renderer.render(_llm_end("same thought"))
        assert buffer.getvalue().count("same thought") == 2


class TestBrandColour:
    def test_the_marker_is_brand_coloured_and_the_answer_is_not(self) -> None:
        """The answer stays uncoloured so it does not compete with its own
        marker — the marker is the anchor, the text is the content."""
        renderer, buffer = _renderer(color=True)
        renderer.render(_response("plain answer"))
        output = buffer.getvalue()
        marker_at = output.index("⏺")
        assert "38;2;31;109;255" in output[:marker_at], "marker should carry the brand colour"
        assert "38;2;31;109;255" not in output[output.index("plain answer") :]

    def test_the_user_prompt_uses_the_same_hue(self) -> None:
        from domyn.agents.interfaces.cli.theme import Styles

        assert "#1f6dff" in Styles.user_prompt
        assert "#1f6dff" in Styles.agent_marker

    def test_the_answer_text_is_never_syntax_highlighted(self) -> None:
        """rich colours numbers, paths and quoted strings by default. In an
        answer that is meant to read as plain prose it just looks like a bug."""
        renderer, buffer = _renderer(color=True)
        renderer.render(_response("it costs 18 euros at /tmp/x"))
        answer = buffer.getvalue()[buffer.getvalue().index("it costs") :]
        assert "\x1b" not in answer, answer

    def test_nothing_is_emitted_without_colour_support(self) -> None:
        renderer, buffer = _renderer(color=False)
        renderer.render(_response())
        assert "\x1b" not in buffer.getvalue()


class TestStreaming:
    def test_a_streamed_answer_gets_the_marker_once(self) -> None:
        renderer, buffer = _renderer()
        for chunk in ("Hel", "lo ", "there"):
            renderer.render(
                BaseEvent(
                    event_type=ExecutionEventType.RESPONSE_TOKEN,
                    author="agent",
                    content=[Part(text=chunk)],
                )
            )
        renderer.render(_response("Hello there"))
        output = buffer.getvalue()
        assert output.count("⏺") == 1
        assert output.count("Hello there") == 1, "the final RESPONSE repeats the streamed text"


@pytest.mark.parametrize("unicode_ok", [True, False])
def test_the_marker_has_an_ascii_fallback(unicode_ok: bool) -> None:
    base = Theme.build(allow_animation=False)
    theme = type(base)(
        capabilities=Capabilities(color=False, unicode=unicode_ok, animation=False),
        symbols=type(base.symbols)(unicode=unicode_ok),
        styles=base.styles,
    )
    buffer = io.StringIO()
    renderer = EventRenderer(Console(file=buffer, width=100), theme)
    renderer.begin_turn()
    renderer.render(_response("answer"))
    assert buffer.getvalue().startswith("⏺" if unicode_ok else "*")
