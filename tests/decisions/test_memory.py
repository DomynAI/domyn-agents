"""Whether a turn holds something worth remembering, asked of a decision model.

With the ``memory`` use case active the model answers the yes/no and the LLM is
called only after a yes, to write the fact; anything short of a confident answer
takes the path without a decision model, the LLM deciding and writing in one call.
"""

import os
import tempfile
from unittest.mock import AsyncMock

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
from domyn.agents.decisions.memory import Verdict, decide, last_exchange
from domyn.agents.llm import LLMResponse
from domyn.agents.memory.managers import FilesystemMemoryManager, MemoryExecutionMode


class Says(DecisionProvider):
    """Answers every ``noul`` with one probability, and records what it was shown."""

    type_name = "says"
    provenance = Provenance.NATIVE

    def __init__(self, probability: float | None, **kwargs):
        super().__init__(**kwargs)
        self._probability = probability
        self.asked = 0
        self.state: str | None = None

    async def _decide(self, state, questions):
        self.asked += 1
        self.state = state
        if self._probability is None:
            return Decision()
        p = self._probability
        return Decision(
            answers={
                name: Answer(
                    known=True, provenance=Provenance.NATIVE, value=p, confidence=max(p, 1 - p)
                )
                for name in questions
            }
        )


def runtime(probability, mode=DecisionMode.ACTIVE, threshold=0.5) -> DecisionRuntime:
    return DecisionRuntime(Says(probability), {"memory": UseCasePolicy(mode, threshold)})


def event(kind: ExecutionEventType, text: str, author: str = "user") -> BaseEvent:
    return BaseEvent(event_type=kind, author=author, content=[Part(text=text)])


def context() -> InteractionContext:
    earlier = event(ExecutionEventType.USER_INPUT, "hello")
    earlier_reply = event(ExecutionEventType.RESPONSE, "hi!", "Assistant")
    user = event(ExecutionEventType.USER_INPUT, "I moved to Rome last month")
    reply = event(ExecutionEventType.RESPONSE, "Nice, how do you like it?", "Assistant")
    return InteractionContext(
        conversation_id="c",
        user_id="u1",
        agent_input=user,
        conversation_history=[earlier, earlier_reply, user, reply],
        run_config={"user_id": "u1"},
    )


def agent_whose_llm_says(text: str, decisions: DecisionRuntime | None = None):
    agent = AsyncMock()
    agent.name = "Assistant"
    agent.decisions = decisions
    agent.llm_provider.generate.return_value = LLMResponse(
        content=text, usage={}, generated_tokens=1, execution_time=0.01
    )
    return agent


@pytest.fixture
def manager():
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        path = f.name
    os.unlink(path)
    yield FilesystemMemoryManager(file_path=path, execution_mode=MemoryExecutionMode.AUTOMATIC)
    if os.path.exists(path):
        os.unlink(path)


class TestTheVerdict:
    @pytest.mark.asyncio
    async def test_a_confident_yes_saves(self):
        assert await decide(runtime(0.9), user="I am vegan", assistant="ok") is Verdict.SAVE

    @pytest.mark.asyncio
    async def test_a_confident_no_skips(self):
        assert await decide(runtime(0.1), user="what time is it", assistant="") is Verdict.SKIP

    @pytest.mark.asyncio
    async def test_between_the_bars_is_undecided(self):
        decided = await decide(runtime(0.6, threshold=0.8), user="I am vegan", assistant="")
        assert decided is Verdict.UNDECIDED

    @pytest.mark.asyncio
    async def test_no_answer_is_undecided(self):
        assert await decide(runtime(None), user="I am vegan", assistant="") is Verdict.UNDECIDED

    @pytest.mark.asyncio
    async def test_shadow_asks_and_does_not_act(self):
        rt = runtime(0.99, mode=DecisionMode.SHADOW)
        assert await decide(rt, user="I am vegan", assistant="") is Verdict.UNDECIDED
        assert rt.provider.asked == 1

    @pytest.mark.asyncio
    async def test_off_does_not_ask(self):
        rt = runtime(0.99, mode=DecisionMode.OFF)
        assert await decide(rt, user="I am vegan", assistant="") is Verdict.UNDECIDED
        assert rt.provider.asked == 0

    @pytest.mark.asyncio
    async def test_no_runtime_is_undecided(self):
        assert await decide(None, user="I am vegan", assistant="") is Verdict.UNDECIDED

    @pytest.mark.asyncio
    async def test_it_is_shown_only_the_last_exchange(self):
        rt = runtime(0.9)
        user, assistant = last_exchange(context())
        await decide(rt, user=user, assistant=assistant)
        assert "Rome" in rt.provider.state
        assert "hello" not in rt.provider.state


class TestTheLastExchange:
    def test_the_last_user_message_and_the_replies_after_it(self):
        assert last_exchange(context()) == (
            "I moved to Rome last month",
            "Nice, how do you like it?",
        )

    def test_the_turn_may_be_in_the_interaction_history(self):
        ctx = context()
        user = event(ExecutionEventType.USER_INPUT, "I have two cats")
        reply = event(ExecutionEventType.RESPONSE, "Lovely", "Assistant")
        ctx.interaction_history = [user, reply]
        assert last_exchange(ctx) == ("I have two cats", "Lovely")


class TestUpdateMemory:
    @pytest.mark.asyncio
    async def test_yes_has_the_llm_write_only(self, manager):
        agent = agent_whose_llm_says("The user lives in Rome", runtime(0.95))
        await manager.update_memory(agent, context())

        memories = await manager.recall_memories("u1")
        assert [m.memory for m in memories] == ["The user lives in Rome"]
        system = agent.llm_provider.generate.call_args.kwargs["messages"][0].content
        assert system == manager.memory_writing_prompt

    @pytest.mark.asyncio
    async def test_no_calls_no_llm(self, manager):
        agent = agent_whose_llm_says("StoreMemory: yes\nMemoryContent: x", runtime(0.05))
        await manager.update_memory(agent, context())

        agent.llm_provider.generate.assert_not_called()
        assert await manager.recall_memories("u1") == []

    @pytest.mark.asyncio
    async def test_the_writer_may_still_find_nothing(self, manager):
        agent = agent_whose_llm_says("NONE", runtime(0.95))
        await manager.update_memory(agent, context())
        assert await manager.recall_memories("u1") == []

    @pytest.mark.asyncio
    async def test_undecided_takes_the_path_without_a_decision_model(self, manager):
        agent = agent_whose_llm_says(
            "StoreMemory: yes\nMemoryContent: The user lives in Rome", runtime(None)
        )
        await manager.update_memory(agent, context())

        system = agent.llm_provider.generate.call_args.kwargs["messages"][0].content
        assert system == manager.memory_extraction_prompt
        assert [m.memory for m in await manager.recall_memories("u1")] == ["The user lives in Rome"]

    @pytest.mark.asyncio
    async def test_without_a_decision_model_nothing_changes(self, manager):
        agent = agent_whose_llm_says("StoreMemory: no\nMemoryContent: -")
        await manager.update_memory(agent, context())

        system = agent.llm_provider.generate.call_args.kwargs["messages"][0].content
        assert system == manager.memory_extraction_prompt
        assert await manager.recall_memories("u1") == []
