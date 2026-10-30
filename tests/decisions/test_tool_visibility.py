"""Hiding tools the turn plainly does not need.

The rule is the mirror of the approval gate's: there the base is "ask" and
*allowing* needs a real number; here the base is "the tool is there" and
*hiding* does. A tool hidden by mistake produces "I can't read files" instead
of a question, so everything uncertain keeps it.
"""

import pytest

from domyn.agents import BaseEvent, ExecutionEventType, Part
from domyn.agents.core import InteractionContext
from domyn.agents.decisions import (
    Answer,
    Decision,
    DecisionMode,
    DecisionProvider,
    DecisionRuntime,
    Provenance,
    UseCasePolicy,
)
from domyn.agents.decisions.tools import ALWAYS_VISIBLE, decide, hidden_for


class Tool:
    def __init__(self, name: str, description: str = "does something"):
        self.name = name
        self.description = description


def tools(count: int = 14) -> list[Tool]:
    return [Tool(f"tool_{i}") for i in range(count)]


def context(turn_id: str = "turn_1") -> InteractionContext:
    return InteractionContext(
        conversation_id="c1",
        user_id="u1",
        interaction_id="i1",
        turn_id=turn_id,
        agent_input=BaseEvent(
            event_type=ExecutionEventType.USER_INPUT,
            author="user",
            content=[Part(text="a task")],
        ),
    )


class Scores(DecisionProvider):
    type_name = "scores"
    provenance = Provenance.NATIVE

    def __init__(self, probabilities: list[float], **kwargs):
        super().__init__(**kwargs)
        self._probabilities = probabilities
        self.asked = 0

    async def _decide(self, state, questions):
        self.asked += 1
        answers = {}
        for position, name in enumerate(questions):
            probability = (
                self._probabilities[position] if position < len(self._probabilities) else 1.0
            )
            answers[name] = Answer(
                known=True,
                provenance=Provenance.NATIVE,
                value=probability,
                confidence=max(probability, 1.0 - probability),
            )
        return Decision(answers=answers)


def runtime(probabilities: list[float], mode=DecisionMode.ACTIVE) -> DecisionRuntime:
    return DecisionRuntime(Scores(probabilities), {"tools": UseCasePolicy(mode, 0.9)})


class TestItOnlyEverHides:
    @pytest.mark.asyncio
    async def test_a_confident_no_hides_the_tool(self):
        built = context()
        result = await decide(
            runtime([0.01] + [1.0] * 13), tools=tools(), task="write a poem", context=built
        )
        assert result is not None
        assert result.hidden == ("tool_0",)
        assert hidden_for(built) == {"tool_0"}

    @pytest.mark.asyncio
    async def test_a_confident_yes_keeps_it(self):
        assert (
            await decide(runtime([0.99] * 14), tools=tools(), task="anything", context=context())
            is None
        )

    @pytest.mark.asyncio
    async def test_a_coin_flip_keeps_it(self):
        """Measured on this shape of question, the model is sure when it says
        hide (0.96-0.98) and unsure when it says keep (0.57-0.72). Only the
        direction it is good at is acted on."""
        assert (
            await decide(runtime([0.45] * 14), tools=tools(), task="anything", context=context())
            is None
        )

    @pytest.mark.asyncio
    async def test_below_the_threshold_it_keeps_it(self):
        # 0.2 -> 1-0.2 = 0.8, under the 0.9 bar this use case asks for.
        assert (
            await decide(runtime([0.2] * 14), tools=tools(), task="anything", context=context())
            is None
        )


class TestWhatIsNeverHidden:
    @pytest.mark.asyncio
    async def test_the_protocol_tools_are_not_candidates(self):
        every = [Tool(name) for name in ALWAYS_VISIBLE] + tools(14)
        built = context()
        result = await decide(runtime([0.0] * 30), tools=every, task="anything", context=built)
        assert result is not None
        assert not ALWAYS_VISIBLE & set(result.hidden)

    @pytest.mark.asyncio
    async def test_a_small_catalogue_is_left_alone(self):
        """Hiding three tools saves tokens nobody was short of."""
        built = runtime([0.0] * 5)
        assert await decide(built, tools=tools(5), task="anything", context=context()) is None
        assert built.provider.asked == 0


class TestTheDecisionIsForOneTurnOnly:
    @pytest.mark.asyncio
    async def test_another_turn_sees_the_whole_list(self):
        built = context(turn_id="turn_1")
        await decide(runtime([0.0] * 14), tools=tools(), task="anything", context=built)
        assert hidden_for(built) == {"tool_0"} | {f"tool_{i}" for i in range(1, 14)}

        built.turn_id = "turn_2"
        assert hidden_for(built) == frozenset()

    def test_a_context_with_nothing_stored_hides_nothing(self):
        assert hidden_for(context()) == frozenset()

    def test_rubbish_in_the_state_hides_nothing(self):
        built = context()
        built.state["tool_visibility"] = "not a Visibility"
        assert hidden_for(built) == frozenset()


class TestEverythingUncertainKeepsEveryTool:
    @pytest.mark.asyncio
    async def test_with_no_runtime(self):
        assert await decide(None, tools=tools(), task="x", context=context()) is None

    @pytest.mark.asyncio
    async def test_with_the_use_case_off(self):
        built = runtime([0.0] * 14, mode=DecisionMode.OFF)
        assert await decide(built, tools=tools(), task="x", context=context()) is None
        assert built.provider.asked == 0

    @pytest.mark.asyncio
    async def test_in_shadow_it_measures_and_hides_nothing(self):
        built = runtime([0.0] * 14, mode=DecisionMode.SHADOW)
        ctx = context()
        assert await decide(built, tools=tools(), task="x", context=ctx) is None
        assert built.provider.asked == 1
        assert hidden_for(ctx) == frozenset()

    @pytest.mark.asyncio
    async def test_with_no_task(self):
        built = runtime([0.0] * 14)
        assert await decide(built, tools=tools(), task="", context=context()) is None
        assert built.provider.asked == 0

    @pytest.mark.asyncio
    async def test_when_the_model_will_not_answer(self):
        class Broken(DecisionProvider):
            type_name = "broken"
            provenance = Provenance.NATIVE

            async def _decide(self, state, questions):
                raise RuntimeError("down")

        built = DecisionRuntime(Broken(), {"tools": UseCasePolicy(DecisionMode.ACTIVE, 0.9)})
        assert await decide(built, tools=tools(), task="x", context=context()) is None


class TestTheFilterReachesTheRequest:
    """The decision is worth nothing if the tool definitions still go out."""

    @pytest.mark.asyncio
    async def test_a_hidden_tool_is_not_in_the_definitions(self):
        from domyn.agents import Agent
        from domyn.agents.llm import BaseLLMProvider
        from domyn.agents.planner_strategy import (
            ToolUseContextManager,
        )
        from domyn.agents.tools import FunctionTool

        class Mock(BaseLLMProvider):
            type: str = "mock"
            name: str = "m"
            model_name: str = "m"
            supported_structured_outputs: None = None

            async def generate(self, messages, **kwargs):
                pass

            async def generate_streaming(self, messages, **kwargs):
                yield

        def keep_me(a: int) -> int:
            """Keep me."""
            return a

        def hide_me(a: int) -> int:
            """Hide me."""
            return a

        agent = Agent(
            name="A",
            description="d",
            instruction="i",
            llm_provider=Mock(),
            tools=[FunctionTool(func=keep_me), FunctionTool(func=hide_me)],
        )
        built = context()
        manager = ToolUseContextManager()

        before = manager.get_tool_definitions(agent, built)
        assert {"keep_me", "hide_me"} <= {d["function"]["name"] for d in before}

        await decide(
            runtime([1.0, 0.0]),
            tools=[Tool("keep_me"), Tool("hide_me"), *tools(12)],
            task="anything",
            context=built,
        )
        after = {d["function"]["name"] for d in manager.get_tool_definitions(agent, built)}
        assert "keep_me" in after
        assert "hide_me" not in after
