"""The ``Completed in`` line says what the turn cost, beside how long it took.

Summed over every model call of the turn: each call sends the conversation again,
so that sum, not the conversation's size, is what the turn cost.
"""

import io
import json
import re
from typing import Any

import pytest
from pydantic import PrivateAttr
from rich.console import Console

from domyn.agents import Agent
from domyn.agents.interfaces.cli import Theme
from domyn.agents.interfaces.cli.repl import Repl, _turn_tokens_text
from domyn.agents.llm import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
    LLMToolCall,
    TokenUsage,
    ToolCallFunction,
)
from domyn.agents.planner_strategy import ToolUsePlannerStrategy
from domyn.agents.tools import FunctionTool


class TestTheText:
    def test_nothing_reported_adds_nothing(self) -> None:
        assert _turn_tokens_text(None) == ""
        assert _turn_tokens_text(TokenUsage(cache_read_tokens=5)) == ""

    def test_total_with_input_and_output(self) -> None:
        usage = TokenUsage(input_tokens=11_800, output_tokens=512, total_tokens=12_312)
        assert _turn_tokens_text(usage) == " · 12.3k tokens (11.8k in, 512 out)"

    def test_total_alone_when_the_split_is_unknown(self) -> None:
        assert _turn_tokens_text(TokenUsage(total_tokens=900)) == " · 900 tokens"


def _usage(prompt: int, completion: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


class _Scripted(BaseLLMProvider):
    """Answers with ``responses`` in order, the last one repeated."""

    type: str = "scripted"
    model_name: str = "scripted"
    supported_structured_outputs: list[Any] | None = None
    responses: list[LLMResponse]
    _calls: int = PrivateAttr(default=0)

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        response = self.responses[min(self._calls, len(self.responses) - 1)]
        self._calls += 1
        return response

    async def generate_streaming(self, messages: list[LLMMessage], **kwargs: Any) -> Any:
        raise NotImplementedError
        yield  # pragma: no cover


class _Editor:
    def __init__(self, *lines: str) -> None:
        self._lines = list(lines)

    async def prompt_async(self, *args: Any, **kwargs: Any) -> str:
        if not self._lines:
            raise EOFError
        return self._lines.pop(0)


def _answer(usage: dict[str, int]) -> LLMResponse:
    return LLMResponse(content="done", usage=usage, generated_tokens=1, execution_time=0.0)


def _look_up_call(usage: dict[str, int]) -> LLMResponse:
    arguments = json.dumps({"term": "x"})
    return LLMResponse(
        content="",
        usage=usage,
        generated_tokens=1,
        execution_time=0.0,
        tool_calls=[
            LLMToolCall(id="c1", function=ToolCallFunction(name="look_up", arguments=arguments))
        ],
    )


async def _completed_lines(responses: list[LLMResponse], *script: str) -> list[str]:
    def look_up(term: str) -> str:
        """Look something up.

        Args:
            term: What.
        """
        return "found"

    agent = Agent(
        name="Assistant",
        description="test agent",
        instruction="Be brief.",
        llm_provider=_Scripted(responses=responses),
        planner=ToolUsePlannerStrategy(),
        tools=[FunctionTool(func=look_up)],
        stream=False,
    )
    buffer = io.StringIO()
    repl = Repl(
        agent,
        Console(file=buffer, width=200, force_terminal=False),
        Theme.build(allow_animation=False),
        enable_enhanced_input=False,
    )
    repl._input._session = _Editor(*script)
    await repl.run()
    return re.findall(r"Completed in .*", buffer.getvalue())


@pytest.mark.asyncio
async def test_one_call_shows_its_counts() -> None:
    (line,) = await _completed_lines([_answer(_usage(1_500, 40))], "hi")
    assert line.endswith(" · 1.5k tokens (1.5k in, 40 out)")


@pytest.mark.asyncio
async def test_every_call_of_the_turn_is_added() -> None:
    """A tool call, then the answer: two model calls, one turn."""
    (line,) = await _completed_lines(
        [_look_up_call(_usage(1_000, 20)), _answer(_usage(1_200, 30))], "hi"
    )
    assert line.endswith(" · 2.2k tokens (2.2k in, 50 out)")


@pytest.mark.asyncio
async def test_each_turn_counts_only_its_own() -> None:
    first, second = await _completed_lines([_answer(_usage(100, 10))], "hi", "again")
    assert first.endswith(" · 110 tokens (100 in, 10 out)")
    assert second.endswith(" · 110 tokens (100 in, 10 out)")


@pytest.mark.asyncio
async def test_no_usage_reported_leaves_the_time_alone() -> None:
    (line,) = await _completed_lines([_answer({})], "hi")
    assert re.fullmatch(r"Completed in [\d.]+s\s*", line)
