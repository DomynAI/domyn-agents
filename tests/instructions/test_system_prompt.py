"""Standing instructions reaching the model.

The agent merges what the runner loaded into its instruction and hands that to
the planner as ``instruction=``; both prompt builders render it in place of
``agent.instruction``. ``agent.instruction`` itself is never rewritten.
"""

# No `from __future__ import annotations`: builds real agents.
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import MagicMock

import pytest

from domyn.agents import Agent
from domyn.agents.instructions import (
    AgentInstructions,
    InMemoryInstructionsSource,
    Instruction,
    InstructionKind,
    InstructionRuntime,
    attach_instructions,
)
from domyn.agents.llm import BaseLLMProvider, LLMMessage, LLMResponse
from domyn.agents.planner_strategy import (
    JSONPrompting,
    ToolUseContextManager,
    ToolUsePlannerStrategy,
    ToolUseWithThoughtContextManager,
    ToolUseWithThoughtPlannerStrategy,
)
from tests.planner_strategy.test_tool_use import make_context, make_mock_agent


class MockLLMProvider(BaseLLMProvider):
    type: str = "mock"
    name: str = "mock-llm"
    model_name: str = "mock-model"
    supported_structured_outputs: None = None

    async def generate(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:  # type: ignore
        return LLMResponse(content="", generated_tokens=0, execution_time=0.0)

    async def generate_streaming(  # type: ignore
        self, messages: list[LLMMessage], **kwargs
    ) -> AsyncGenerator[LLMResponse, None]:
        yield LLMResponse(content="", generated_tokens=0, execution_time=0.0)


def _agent(**kwargs: Any) -> Agent:
    fields = {
        "name": "Assistant",
        "description": "A test agent",
        "instruction": "Be brief.",
        "llm_provider": MockLLMProvider(),
    }
    return Agent(**{**fields, **kwargs})


def _runtime(soul: str = "Answer in Italian.") -> InstructionRuntime:
    return InstructionRuntime(source=InMemoryInstructionsSource(soul=soul))


def _loaded(text: str = "Answer in Italian.") -> AgentInstructions:
    return AgentInstructions(soul=(Instruction(text=text, instruction_kind=InstructionKind.SOUL),))


class TestMerging:
    def test_appended_after_the_agents_own_instruction(self) -> None:
        """Appended, never substituted: that value is the author's. After it,
        because of two conflicting lines the later one is the one followed."""
        merged = _runtime().merge("You are a customer service agent.", _loaded())

        assert merged.index("customer service") < merged.index("Answer in Italian.")

    def test_an_empty_instruction_gains_no_blank_lines(self) -> None:
        assert _runtime().merge("", _loaded()).startswith("[soul]")

    def test_nothing_loaded_leaves_it_unchanged(self) -> None:
        assert _runtime().merge("Be brief.", AgentInstructions()) == "Be brief."


class TestTheAgentHandsItToThePlanner:
    @staticmethod
    async def _planner_kwargs(agent: Agent, **execute_kwargs: Any) -> dict[str, Any]:
        seen: dict[str, Any] = {}

        async def _execute(agent_arg, context, **kwargs):  # type: ignore[no-untyped-def]
            seen.update(kwargs)
            return
            yield

        agent.planner = MagicMock()
        agent.planner.execute = _execute
        async for _ in agent._execute(MagicMock(), **execute_kwargs):
            pass
        return seen

    @pytest.mark.asyncio
    async def test_what_the_runner_loaded_is_merged(self) -> None:
        agent = _agent(standing_instructions=_runtime("ignored, the preload wins"))

        kwargs = await self._planner_kwargs(agent, instructions=_loaded())

        assert "Be brief." in kwargs["instruction"]
        assert "Answer in Italian." in kwargs["instruction"]

    @pytest.mark.asyncio
    async def test_outside_a_runner_it_loads_on_demand(self) -> None:
        agent = _agent(standing_instructions=_runtime())

        kwargs = await self._planner_kwargs(agent)

        assert "Answer in Italian." in kwargs["instruction"]

    @pytest.mark.asyncio
    async def test_the_agents_instruction_is_never_rewritten(self) -> None:
        """One agent serves concurrent conversations; writing into it would leak
        one user's soul into another's turn — and append again on every turn."""
        agent = _agent(standing_instructions=_runtime())

        await self._planner_kwargs(agent)
        await self._planner_kwargs(agent)

        assert agent.instruction == "Be brief."

    @pytest.mark.asyncio
    async def test_an_agent_without_them_passes_nothing(self) -> None:
        """So a planner written before the argument keeps working."""
        assert "instruction" not in await self._planner_kwargs(_agent())

    @pytest.mark.asyncio
    async def test_nothing_loaded_passes_nothing(self) -> None:
        agent = _agent(standing_instructions=_runtime())

        kwargs = await self._planner_kwargs(agent, instructions=AgentInstructions())

        assert "instruction" not in kwargs

    @pytest.mark.asyncio
    async def test_a_failing_source_degrades_to_none(self) -> None:
        class Broken(InMemoryInstructionsSource):
            type: str = "broken_for_test"

            async def load(self) -> AgentInstructions:
                raise OSError("down")

        agent = _agent(standing_instructions=InstructionRuntime(source=Broken()))

        assert "instruction" not in await self._planner_kwargs(agent)

    @pytest.mark.asyncio
    async def test_an_instruction_built_asynchronously_is_left_alone(self) -> None:
        """A coroutine cannot be concatenated."""

        async def later() -> str:
            return "Be brief."

        coroutine = later()
        try:
            agent = _agent(instruction=coroutine, standing_instructions=_runtime())

            assert "instruction" not in await self._planner_kwargs(agent)
        finally:
            coroutine.close()


class TestThePromptBuilders:
    @pytest.mark.parametrize(
        "manager",
        [ToolUseContextManager(), ToolUseWithThoughtContextManager()],
        ids=["tool_use", "thought_tool_use"],
    )
    def test_the_tool_use_branch_renders_the_override(self, manager: ToolUseContextManager) -> None:
        prompt = manager._build_system_prompt(_agent(), None, "Be brief.\n\nAnswer in Italian.")

        assert "Answer in Italian." in prompt

    def test_the_protocol_note_still_comes_last(self) -> None:
        """FORWARD_VERBATIM is a wire-format requirement, not a preference."""
        prompt = ToolUseContextManager()._build_system_prompt(_agent(), None, "Answer in Italian.")

        assert prompt.index("Answer in Italian.") < prompt.index("FORWARD_VERBATIM")

    def test_without_an_override_it_is_the_agents_instruction(self) -> None:
        prompt = ToolUseContextManager()._build_system_prompt(_agent(), None)

        assert prompt.startswith("Be brief.")

    def test_the_react_branch_renders_the_override(self) -> None:
        text = JSONPrompting.get_agent_instructions_text(
            _agent(instruction=""), "Answer in Italian."
        )

        assert "Answer in Italian." in text


@pytest.mark.parametrize("planner_cls", [ToolUsePlannerStrategy, ToolUseWithThoughtPlannerStrategy])
@pytest.mark.asyncio
async def test_both_tool_use_loops_put_it_in_the_system_message(planner_cls: Any) -> None:
    """The two loops are copies of each other, so each is driven for real."""
    from unittest.mock import AsyncMock

    agent = make_mock_agent()
    agent.planner = MagicMock()
    agent.planner.use_stop = False
    agent.llm_provider = AsyncMock()
    agent.llm_provider.generate = AsyncMock(
        return_value=LLMResponse(
            content="done", generated_tokens=1, execution_time=0.1, tool_calls=[]
        )
    )

    async for _ in planner_cls().execute(
        agent, make_context(), instruction="You are helpful.\n\nAnswer in Italian."
    ):
        pass

    messages = agent.llm_provider.generate.call_args.kwargs["messages"]
    assert messages[0].role == "system"
    assert "Answer in Italian." in messages[0].content


class TestAttaching:
    def test_it_is_idempotent(self) -> None:
        agent = _agent()
        runtime = _runtime()

        attach_instructions(agent, runtime)
        attach_instructions(agent, runtime)

        assert agent.standing_instructions is runtime
        assert agent.instruction == "Be brief."

    def test_it_leaves_sub_agents_alone(self) -> None:
        """A standing preamble repeated down a tree is paid at every handoff."""
        child = _agent(name="Child")
        parent = _agent(sub_agents=[child])

        attach_instructions(parent, _runtime())

        assert child.standing_instructions is None

    def test_it_skips_an_agent_that_declares_no_field(self) -> None:
        class Foreign:
            __slots__ = ()

        foreign = Foreign()

        assert attach_instructions(foreign, _runtime()) is foreign
