"""Tests for recovering from a tool call whose arguments are not valid JSON.

Seen in a real session: the model wrote a file's contents into a JSON string
argument, missed an escape, and the planner retried the *identical* request
three times over forty-five seconds before giving up. Retrying without saying
what was wrong asks a model to spot a mistake it has no way of knowing it made.

So what is pinned here is that the retry differs from the attempt before it.
"""

# No `from __future__ import annotations`: builds real tools.
import json
from typing import Any

import pytest
from pydantic import PrivateAttr

from domyn.agents import Agent, BaseEvent, ExecutionEventType, Part, Runner
from domyn.agents.llm import BaseLLMProvider, LLMResponse, LLMToolCall, ToolCallFunction
from domyn.agents.planner_strategy import PlannerStrategyFactory
from domyn.agents.planner_strategy.tool_use import (
    malformed_args_feedback,
    parse_tool_call_args_with_errors,
)
from domyn.agents.tools import FunctionTool

BROKEN = '{"path": "a.txt", "content": "he said "hi"", "reasoning": "x"}'
FIXED = json.dumps({"path": "a.txt", "content": 'he said "hi"', "reasoning": "x"})


def _call(arguments: str, name: str = "write_file") -> LLMToolCall:
    return LLMToolCall(id="c1", function=ToolCallFunction(name=name, arguments=arguments))


class TestParsing:
    def test_valid_arguments_parse(self) -> None:
        parsed, ok, errors = parse_tool_call_args_with_errors([_call(FIXED)])
        assert ok and not errors
        assert parsed["c1"]["path"] == "a.txt"

    def test_a_broken_call_is_reported_with_its_reason(self) -> None:
        """The reason is the whole point: it is what the retry is given."""
        parsed, ok, errors = parse_tool_call_args_with_errors([_call(BROKEN)])
        assert not ok
        assert parsed["c1"] == {}
        assert len(errors) == 1
        assert "write_file" in errors[0], "the model needs to know which call failed"
        assert "line 1 column" in errors[0], "and where"

    def test_one_broken_call_does_not_hide_the_others(self) -> None:
        _, ok, errors = parse_tool_call_args_with_errors(
            [_call(FIXED), _call(BROKEN, name="run_command")]
        )
        assert not ok
        assert len(errors) == 1
        assert "run_command" in errors[0]

    def test_no_tool_calls_is_not_a_failure(self) -> None:
        parsed, ok, errors = parse_tool_call_args_with_errors(None)
        assert ok and not errors and parsed == {}


class TestTheFeedbackMessage:
    def test_it_names_the_characters_that_actually_break(self) -> None:
        """ "Produce valid JSON" is advice the model believed it was following.
        The failures are all escaping, in arguments carrying file contents or
        shell commands."""
        content = str(malformed_args_feedback(["write_file: Expecting ','"]).content)
        assert "newline" in content
        assert "backslash" in content
        assert "double quote" in content

    def test_it_carries_the_parser_error(self) -> None:
        content = str(malformed_args_feedback(["write_file: Expecting ',' delimiter"]).content)
        assert "Expecting ','" in content

    def test_it_asks_for_the_same_call_again(self) -> None:
        """Not for a different approach: the intent was fine, the encoding was
        not, and a model told to "try something else" abandons the task."""
        content = str(malformed_args_feedback([]).content).lower()
        assert "same call again" in content

    def test_it_survives_having_no_details(self) -> None:
        assert malformed_args_feedback([]).content


class _RecoveringProvider(BaseLLMProvider):
    """Breaks the JSON once, then gets it right — like a real model given a hint."""

    type: str = "mock"
    name: str = "mock"
    model_name: str = "mock"
    supported_structured_outputs: None = None
    _calls: int = PrivateAttr(default=0)
    _seen: list = PrivateAttr(default_factory=list)

    async def generate(self, messages: Any, **kwargs: Any) -> LLMResponse:
        self._calls += 1
        self._seen.append([str(getattr(m, "content", "") or "") for m in messages])
        if self._calls == 1:
            return LLMResponse(
                content="",
                generated_tokens=1,
                execution_time=0.0,
                tool_calls=[_call(BROKEN)],
            )
        return LLMResponse(content="Done.", generated_tokens=1, execution_time=0.0)

    async def generate_streaming(self, messages: Any, **kwargs: Any) -> Any:
        yield


def _agent(provider: BaseLLMProvider, planner: str) -> Agent:
    def write_file(path: str, content: str) -> str:
        """Write a file.

        Args:
            path: Where.
            content: What.
        """
        return "ok"

    return Agent(
        name="A",
        description="d",
        instruction="i",
        llm_provider=provider,
        planner=PlannerStrategyFactory.from_config({"type": planner}, save_in_registry=False),
        tools=[FunctionTool(func=write_file)],
    )


class TestTheRetryDiffers:
    @pytest.mark.parametrize("planner", ["tool_use", "tool_use_thought"])
    @pytest.mark.asyncio
    async def test_the_second_attempt_is_told_what_was_wrong(self, planner: str) -> None:
        provider = _RecoveringProvider()
        async for _ in Runner().run(
            application_name="app",
            user_id="u",
            root=_agent(provider, planner),
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text="go")]
            ),
            conversation_id="c1",
        ):
            pass

        assert len(provider._seen) >= 2, "it should have retried"
        second = " ".join(provider._seen[1])
        assert "not valid JSON" in second

    @pytest.mark.parametrize("planner", ["tool_use", "tool_use_thought"])
    @pytest.mark.asyncio
    async def test_the_first_attempt_is_not_polluted(self, planner: str) -> None:
        """The hint must only appear after a failure — a model told about
        escaping before it has made a mistake is being primed for one."""
        provider = _RecoveringProvider()
        async for _ in Runner().run(
            application_name="app",
            user_id="u",
            root=_agent(provider, planner),
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text="go")]
            ),
            conversation_id="c1",
        ):
            pass
        assert "not valid JSON" not in " ".join(provider._seen[0])

    @pytest.mark.parametrize("planner", ["tool_use", "tool_use_thought"])
    @pytest.mark.asyncio
    async def test_it_recovers_instead_of_burning_every_retry(self, planner: str) -> None:
        provider = _RecoveringProvider()
        events = []
        async for event in Runner().run(
            application_name="app",
            user_id="u",
            root=_agent(provider, planner),
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text="go")]
            ),
            conversation_id="c1",
        ):
            events.append(event)

        assert not any(
            "still malformed" in str(getattr(e, "error_message", "") or "") for e in events
        )


class TestRepairingWhatCanBeRepaired:
    """One repair is worth making automatically and the other is not.

    A literal newline inside a JSON string is not ambiguous, it is *illegal* —
    escaping it is the only reading there is. An unescaped quote is a different
    matter: where the string ends becomes a guess, and guessing at the
    boundaries of content about to be written to disk is worse than failing.
    """

    def test_raw_newlines_are_escaped_and_the_call_goes_through(self) -> None:
        raw = '{"path":"d.html","content":"<div class=\\"c\\">\n  <h1>x</h1>\n</div>"}'
        parsed, ok, errors = parse_tool_call_args_with_errors([_call(raw)])
        assert ok and not errors
        assert parsed["c1"]["content"] == '<div class="c">\n  <h1>x</h1>\n</div>'

    def test_the_repaired_content_keeps_its_newlines(self) -> None:
        """Escaping them must preserve the text, not flatten it — the file the
        model meant to write has line breaks in it."""
        raw = '{"content":"a\nb"}'
        parsed, _, _ = parse_tool_call_args_with_errors([_call(raw)])
        assert parsed["c1"]["content"] == "a\nb"

    def test_tabs_and_carriage_returns_too(self) -> None:
        parsed, ok, _ = parse_tool_call_args_with_errors([_call('{"c":"a\tb\rc"}')])
        assert ok
        assert parsed["c1"]["c"] == "a\tb\rc"

    def test_an_unescaped_quote_is_not_guessed_at(self) -> None:
        _, ok, errors = parse_tool_call_args_with_errors(
            [_call('{"path":"d.html","content":"<div class="c">x</div>"}')]
        )
        assert not ok and errors

    def test_a_control_character_outside_a_string_is_left_alone(self) -> None:
        """Whitespace between tokens is legal JSON and must not be rewritten."""
        from domyn.agents.planner_strategy.tool_use import repair_control_characters

        assert repair_control_characters('{\n  "a": 1\n}') is None

    def test_an_already_escaped_sequence_is_untouched(self) -> None:
        from domyn.agents.planner_strategy.tool_use import repair_control_characters

        assert repair_control_characters('{"a":"line\\nbreak"}') is None

    def test_a_backslash_before_a_quote_does_not_end_the_string(self) -> None:
        r"""`"...\"` is an escaped quote, still inside the string; treating it as
        the end would put the repair out of step for the rest of the payload."""
        raw = '{"a":"he said \\"hi\\"\nthen left"}'
        parsed, ok, _ = parse_tool_call_args_with_errors([_call(raw)])
        assert ok
        assert parsed["c1"]["a"] == 'he said "hi"\nthen left'


class TestTheFeedbackOffersAWayOut:
    def test_it_suggests_splitting_a_long_value(self) -> None:
        """Told only to escape better, a model tries the same long call again.
        `append_file` is the route that does not depend on getting a thousand
        escapes right."""
        content = str(malformed_args_feedback([]).content).lower()
        assert "smaller calls" in content
