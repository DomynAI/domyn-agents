"""End-to-end approval in a CLI session.

Everything below the fake LLM is real: a real ``Agent`` with a real gated tool, a
real ``Runner``, the real REPL and the real approval prompt. Only the model is
stubbed, because the point is to prove the whole chain holds together —

    planner asks for a tool
      → runner sees ``approval_type`` and suspends with TOOL_NEEDS_FEEDBACK
        → REPL notices the turn ended on a pause and prompts the human
          → answer becomes TOOL_FEEDBACK and re-enters the runner
            → the tool runs, or the model is told it was refused

— which the unit tests each cover a link of, and none covers as a whole.
"""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from domyn.agents import Runner
from domyn.agents.agents import Agent
from domyn.agents.core import ApprovalType
from domyn.agents.interfaces.cli import Theme
from domyn.agents.interfaces.cli.repl import Repl
from domyn.agents.llm import BaseLLMProvider, LLMResponse, LLMToolCall, ToolCallFunction
from domyn.agents.planner_strategy import PlannerStrategyFactory
from domyn.agents.tools import FunctionTool

DANGEROUS_RESULT = "deleted 42 rows"


class _ScriptedProvider(BaseLLMProvider):
    """Asks for the gated tool once, then answers.

    Native tool calling, so the tool_use planner drives it — the same path a real
    provider takes.
    """

    type: str = "mock"
    name: str = "mock-llm"
    model_name: str = "mock-model"
    supported_structured_outputs: None = None

    async def generate(self, messages: Any, **kwargs: Any) -> LLMResponse:
        already_called = any(
            getattr(message, "role", "") == "tool"
            or "delete_rows" in str(getattr(message, "content", "") or "")
            for message in messages
        )

        if not already_called:
            return LLMResponse(
                content="",
                generated_tokens=1,
                execution_time=0.0,
                tool_calls=[
                    LLMToolCall(
                        id="call-1",
                        function=ToolCallFunction(
                            name="delete_rows",
                            arguments=json.dumps({"table": "orders", "reasoning": "cleanup"}),
                        ),
                    )
                ],
            )

        return LLMResponse(
            content="All done.",
            generated_tokens=1,
            execution_time=0.0,
        )

    async def generate_streaming(self, messages: Any, **kwargs: Any) -> Any:
        yield


@pytest.fixture
def executed() -> list[str]:
    return []


@pytest.fixture
def agent(executed: list[str]) -> Agent:
    def delete_rows(table: str, reasoning: str = "") -> str:
        """Delete rows from a table.

        Args:
            table: Table to delete from.
            reasoning: Why the call is being made.
        """
        executed.append(table)
        return DANGEROUS_RESULT

    tool = FunctionTool(func=delete_rows, approval_type=ApprovalType.USER_CONFIRM)

    return Agent(
        name="Dba",
        description="database assistant",
        instruction="Use the tools available to you.",
        llm_provider=_ScriptedProvider(),
        planner=PlannerStrategyFactory.from_config({"type": "tool_use"}, save_in_registry=False),
        tools=[tool],
    )


def _repl(agent: Agent) -> tuple[Repl, MagicMock]:
    console = MagicMock()
    repl = Repl(
        agent,
        console,
        Theme.build(allow_animation=False),
        runner=Runner(),
        enable_enhanced_input=False,
    )
    # The session ends after the scripted turn.
    repl._input.ask = MagicMock(side_effect=EOFError)  # type: ignore[method-assign]
    return repl, console


def _printed(console: MagicMock) -> str:
    return " ".join(str(call.args[0]) for call in console.print.call_args_list if call.args)


class TestApprovalRoundTrip:
    @pytest.mark.asyncio
    async def test_approving_runs_the_tool(self, agent: Agent, executed: list[str]) -> None:
        repl, console = _repl(agent)

        # "y" answers the approval prompt, which reads from stdin directly.
        with patch("builtins.input", return_value="y"):
            await repl.run("clean up the orders table")

        assert executed == ["orders"], "the tool must run once approved"
        output = _printed(console)
        assert "approval required" in output
        assert "delete_rows" in output

    @pytest.mark.asyncio
    async def test_denying_does_not_run_the_tool(self, agent: Agent, executed: list[str]) -> None:
        repl, console = _repl(agent)

        with patch("builtins.input", return_value="n"):
            await repl.run("clean up the orders table")

        assert executed == [], "a denied tool must not execute"
        assert "denied" in _printed(console)

    @pytest.mark.asyncio
    async def test_denial_is_reported_to_the_model_so_the_turn_continues(
        self, agent: Agent, executed: list[str]
    ) -> None:
        """Denial is not a dead end: the runner turns it into an observation the
        model reads, so the conversation stays coherent instead of hanging."""
        seen: list[str] = []
        original = _ScriptedProvider.generate

        async def spy(self: Any, messages: Any, **kwargs: Any) -> LLMResponse:
            seen.extend(str(getattr(m, "content", "") or "") for m in messages)
            return await original(self, messages, **kwargs)

        with (
            patch("builtins.input", return_value="n"),
            patch.object(_ScriptedProvider, "generate", spy),
        ):
            await repl_run(agent, "clean up the orders table")

        assert executed == []
        assert any("refused" in text.lower() for text in seen), (
            "the model must be told the call was refused"
        )

    @pytest.mark.asyncio
    async def test_aborted_prompt_counts_as_denial(self, agent: Agent, executed: list[str]) -> None:
        """Ctrl-D at the prompt must not leave the turn suspended forever."""
        repl, _ = _repl(agent)

        with patch("builtins.input", side_effect=EOFError):
            await repl.run("clean up the orders table")

        assert executed == []

    @pytest.mark.asyncio
    async def test_ungated_tool_runs_without_asking(self, executed: list[str]) -> None:
        """The gate must only fire for tools that declare it."""

        def delete_rows(table: str, reasoning: str = "") -> str:
            """Delete rows from a table.

            Args:
                table: Table to delete from.
                reasoning: Why the call is being made.
            """
            executed.append(table)
            return DANGEROUS_RESULT

        agent = Agent(
            name="Dba",
            description="database assistant",
            instruction="Use the tools available to you.",
            llm_provider=_ScriptedProvider(),
            planner=PlannerStrategyFactory.from_config(
                {"type": "tool_use"}, save_in_registry=False
            ),
            tools=[FunctionTool(func=delete_rows)],  # no approval_type
        )
        repl, console = _repl(agent)

        asked = MagicMock(side_effect=AssertionError("must not prompt for an ungated tool"))
        with patch("builtins.input", asked):
            await repl.run("clean up the orders table")

        assert executed == ["orders"]
        assert "approval required" not in _printed(console)


async def repl_run(agent: Agent, prompt: str) -> None:
    repl, _ = _repl(agent)
    await repl.run(prompt)
