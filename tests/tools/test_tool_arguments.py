"""Tests for what a model is told when it gets a tool's arguments wrong.

Measured in real sessions: of 96 failed ``edit_file`` calls, 95 failed on the
argument names — ``old_text``/``new_text`` and ``old_string``/``new_string``,
the vocabularies of other harnesses — and not one on the text not being found.
The keys were dropped silently, so the call failed with a TypeError naming the
Python function, and the model retried the same way: ninety-three times in one
session.
"""

import pytest

from domyn.agents.tools import FunctionTool

pytestmark = pytest.mark.asyncio


def add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)


def edits(path: str, old: str, new: str) -> str:
    """Replace old with new in path."""
    return f"edited {path}"


def writes(path: str, content: str) -> str:
    """Write content to a path."""
    return f"wrote {path}"


async def test_a_correct_call_is_untouched() -> None:
    result = await FunctionTool(func=add).execute(a=1, b=2)
    assert result.success and result.data == "3"


async def test_runtime_arguments_are_still_ignored() -> None:
    """``state`` and ``context`` are the runner's, not the model's."""
    result = await FunctionTool(func=add).execute(a=1, b=2, state={"x": 1}, context=None)
    assert result.success and result.data == "3"


@pytest.mark.parametrize(
    ("wrong", "right"),
    [("old_text", "old"), ("old_string", "old"), ("new_text", "new"), ("new_string", "new")],
)
async def test_a_borrowed_name_is_named_and_matched(wrong: str, right: str) -> None:
    arguments: dict[str, str] = {"path": "a.txt", "old": "x", "new": "y"}
    arguments[wrong] = arguments.pop(right)
    result = await FunctionTool(func=edits).execute(**arguments)
    assert not result.success
    assert result.error_code == "ToolArgumentsError"
    assert f"`{wrong}` looks like `{right}`" in result.error


async def test_a_borrowed_path_name_is_matched_too() -> None:
    result = await FunctionTool(func=writes).execute(file_path="a.txt", content="x")
    assert "`file_path` looks like `path`" in result.error


async def test_the_message_lists_what_the_tool_takes() -> None:
    result = await FunctionTool(func=writes).execute(path="a.txt", old_text="x", new_text="y")
    assert "writes is missing `content`" in result.error
    assert "It takes: path, content." in result.error


async def test_a_missing_argument_without_a_wrong_one_is_still_named() -> None:
    result = await FunctionTool(func=add).execute(a=1)
    assert result.error == "add is missing `b`. It takes: a, b."


async def test_an_extra_name_that_is_not_an_argument_is_not_reported() -> None:
    """``reasoning`` is the planner's, and a call that works is not a mistake."""
    result = await FunctionTool(func=add).execute(a=1, b=2, reasoning="because")
    assert result.success and result.error is None


async def test_the_planners_reasoning_is_not_called_a_wrong_argument() -> None:
    """It is in every real call, and reporting it would blame the right thing."""
    result = await FunctionTool(func=edits).execute(path="a", old="x", reasoning="why")
    assert "reasoning" not in result.error
    assert "is missing `new`" in result.error
