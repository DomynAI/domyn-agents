"""The model's whole reasoning goes back to it with its assistant messages.

The planner stores ``reasoning_content`` on the first call of a response, or on
its final answer when it calls nothing (``BaseEvent.reasoning_content``); the
context manager puts it on the assistant
message of that response, in every turn, for a provider that sends it
(``reasoning_key_in_use``). The short thought of each call is never added as the
message's content: it is already the call's own ``reasoning`` argument.
"""

# No `from __future__ import annotations`: builds real agents.
import json
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from pydantic import Field

from domyn.agents.agents import Agent
from domyn.agents.core import (
    BaseEvent,
    ExecutionEventType,
    InteractionContext,
    Part,
    ToolAction,
)
from domyn.agents.llm import LLMProviderFactory
from domyn.agents.llm.anthropic import AnthropicProvider
from domyn.agents.llm.base import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
    LLMToolCall,
    ToolCallFunction,
)
from domyn.agents.llm.openai import OpenAIProvider
from domyn.agents.planner_strategy import PlannerStrategyFactory
from domyn.agents.planner_strategy.context_management.budget import HeuristicEstimator
from domyn.agents.planner_strategy.context_management.compaction import (
    CompactionConfig,
    CompactionPolicy,
    Compactor,
    ConversationSummarizer,
    PruneOnlyCompaction,
    TwoPhaseCompaction,
    build_compactor,
)
from domyn.agents.planner_strategy.context_management.tool_use import (
    ToolUseWithThoughtContextManager,
)
from domyn.agents.runner import Runner
from domyn.agents.tools import FunctionTool

REASONING = "The user wants both files listed; I will look at a and b at once."


def look(path: str) -> str:
    """Look at a path."""
    return f"contents of {path}"


def _call(i: int, path: str, thought: str) -> LLMToolCall:
    return LLMToolCall(
        id=f"c{i}",
        type="function",
        function=ToolCallFunction(
            name="look", arguments=json.dumps({"path": path, "reasoning": thought})
        ),
    )


def _calls() -> LLMResponse:
    return LLMResponse(
        content="",
        generated_tokens=1,
        execution_time=0.0,
        tool_calls=[_call(0, "a", "look at a"), _call(1, "b", "look at b")],
        reasoning_content=REASONING,
    )


def _answer(text: str) -> LLMResponse:
    return LLMResponse(content=text, generated_tokens=1, execution_time=0.0)


class _Script(BaseLLMProvider):
    """Answers with ``replies`` in turn and records every request."""

    type: str = "script"
    name: str = "script"
    model_name: str = "script"
    supported_structured_outputs: list[Any] | None = None
    reasoning_content_key: ClassVar[str | None] = "reasoning_content"
    replies: list[LLMResponse] = Field(default_factory=list)
    requests: list[list[LLMMessage]] = Field(default_factory=list)

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        self.requests.append([m.model_copy(deep=True) for m in messages])
        return self.replies[len(self.requests) - 1]

    async def generate_streaming(self, messages: list[LLMMessage], **kwargs: Any) -> Any:
        self.requests.append([m.model_copy(deep=True) for m in messages])
        reply = self.replies[len(self.requests) - 1]
        # As vLLM streams it: the reasoning in a chunk of its own, then the
        # answer, then one chunk with the calls and the whole reasoning again.
        if reply.reasoning_content:
            yield _answer("").model_copy(update={"reasoning_content": reply.reasoning_content})
        if reply.content:
            yield _answer(reply.content)
        if reply.tool_calls:
            yield reply


class _Silent(_Script):
    reasoning_content_key: ClassVar[str | None] = None


def _agent(provider: BaseLLMProvider, stream: bool = False) -> Agent:
    return Agent(
        name="Solo",
        description="the agent",
        instruction="answer",
        llm_provider=provider,
        tools=[FunctionTool(func=look)],
        stream=stream,
        planner=PlannerStrategyFactory.from_config(
            {"type": "tool_use_thought"}, save_in_registry=False
        ),
    )


async def _turn(runner: Runner, agent: Agent, text: str, conversation: str) -> list[BaseEvent]:
    ask = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text=text)]
    )
    return [
        e
        async for e in runner.run(
            application_name="reasoning",
            user_id="u",
            conversation_id=conversation,
            root=agent,
            user_input=ask,
        )
    ]


def _assistants_with_calls(messages: list[LLMMessage]) -> list[LLMMessage]:
    return [m for m in messages if m.role == "assistant" and m.tool_calls]


def _thoughts_only_in_the_calls(message: LLMMessage) -> None:
    """No text beside the calls: each thought is in its call's arguments, once."""
    assert message.content is None
    assert [json.loads(c.function.arguments)["reasoning"] for c in message.tool_calls or []] == [
        "look at a",
        "look at b",
    ]


class TestAResponseWithoutReasoning:
    @pytest.mark.asyncio
    async def test_stores_none(self) -> None:
        bare = _calls().model_copy(update={"reasoning_content": None})
        provider = _Script(replies=[bare, _answer("done")])

        events = await _turn(Runner(), _agent(provider), "list a and b", "bare")

        assert all(e.reasoning_content is None for e in events)
        (message,) = _assistants_with_calls(provider.requests[1])
        _thoughts_only_in_the_calls(message)


class TestAResponseWithReasoning:
    @pytest.mark.asyncio
    async def test_carries_the_whole_reasoning_once_with_both_calls(self) -> None:
        provider = _Script(replies=[_calls(), _answer("done")])

        await _turn(Runner(), _agent(provider), "list a and b", "current")

        (message,) = _assistants_with_calls(provider.requests[1])
        assert [c.function.name for c in message.tool_calls or []] == ["look", "look"]
        assert message.reasoning_content == REASONING
        assert message.content is None
        assert message.to_dict(reasoning_key="reasoning_content")["reasoning_content"] == REASONING

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.asyncio
    async def test_streamed_or_not_the_same_reasoning_is_stored(self, stream: bool) -> None:
        provider = _Script(replies=[_calls(), _answer("done")])

        events = await _turn(Runner(), _agent(provider, stream), "list a and b", f"s{stream}")

        starts = [e for e in events if e.event_type == ExecutionEventType.TOOL_START]
        assert [s.reasoning_content for s in starts] == [REASONING, None]

    @pytest.mark.asyncio
    async def test_the_thought_shown_to_the_user_is_unchanged(self) -> None:
        provider = _Script(replies=[_calls(), _answer("done")])

        events = await _turn(Runner(), _agent(provider), "list a and b", "thought")

        starts = [e for e in events if e.event_type == ExecutionEventType.TOOL_START]
        assert [s.action.thought for s in starts if s.action] == ["look at a", "look at b"]


class TestAnEarlierTurn:
    @pytest.mark.asyncio
    async def test_carries_its_reasoning_too(self) -> None:
        provider = _Script(replies=[_calls(), _answer("done"), _answer("again")])
        runner, agent = Runner(), _agent(provider)

        await _turn(runner, agent, "list a and b", "earlier")
        await _turn(runner, agent, "thanks", "earlier")

        (message,) = _assistants_with_calls(provider.requests[2])
        assert message.reasoning_content == REASONING
        assert message.content is None

    @pytest.mark.asyncio
    async def test_without_reasoning_carries_only_the_calls(self) -> None:
        bare = _calls().model_copy(update={"reasoning_content": None})
        provider = _Script(replies=[bare, _answer("done"), _answer("again")])
        runner, agent = Runner(), _agent(provider)

        await _turn(runner, agent, "list a and b", "earlier, bare")
        await _turn(runner, agent, "thanks", "earlier, bare")

        (message,) = _assistants_with_calls(provider.requests[2])
        assert message.reasoning_content is None
        _thoughts_only_in_the_calls(message)


ANSWER_REASONING = "Both files are listed; I can answer now."


def _reasoned_answer(text: str) -> LLMResponse:
    return _answer(text).model_copy(update={"reasoning_content": ANSWER_REASONING})


def _answers(messages: list[LLMMessage]) -> list[LLMMessage]:
    return [m for m in messages if m.role == "assistant" and not m.tool_calls]


class TestAFinalAnswer:
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.asyncio
    async def test_carries_its_reasoning_on_the_next_turn(self, stream: bool) -> None:
        """Streamed, the closing chunk of an answer has no reasoning: it is taken
        from the chunk that carried it."""
        provider = _Script(replies=[_reasoned_answer("hello"), _answer("again")])
        runner, agent = Runner(), _agent(provider, stream)

        await _turn(runner, agent, "hi", f"answer {stream}")
        await _turn(runner, agent, "and now?", f"answer {stream}")

        (message,) = _answers(provider.requests[1])
        assert message.content == "hello"
        assert message.reasoning_content == ANSWER_REASONING
        assert message.to_dict(reasoning_key="reasoning_content")["reasoning_content"] == (
            ANSWER_REASONING
        )

    @pytest.mark.asyncio
    async def test_after_calls_carries_its_own_reasoning(self) -> None:
        provider = _Script(replies=[_calls(), _reasoned_answer("done"), _answer("again")])
        runner, agent = Runner(), _agent(provider)

        await _turn(runner, agent, "list a and b", "calls then answer")
        await _turn(runner, agent, "thanks", "calls then answer")

        (calls,) = _assistants_with_calls(provider.requests[2])
        (answer,) = _answers(provider.requests[2])
        assert calls.reasoning_content == REASONING
        assert (answer.content, answer.reasoning_content) == ("done", ANSWER_REASONING)

    @pytest.mark.asyncio
    async def test_without_reasoning_stores_none(self) -> None:
        provider = _Script(replies=[_answer("hello"), _answer("again")])
        runner, agent = Runner(), _agent(provider)

        events = await _turn(runner, agent, "hi", "bare answer")
        await _turn(runner, agent, "and now?", "bare answer")

        assert all(e.reasoning_content is None for e in events)
        (message,) = _answers(provider.requests[1])
        assert (message.content, message.reasoning_content) == ("hello", None)

    @pytest.mark.asyncio
    async def test_a_provider_that_does_not_send_it_gets_the_text_alone(self) -> None:
        provider = _Silent(replies=[_reasoned_answer("hello"), _answer("again")])
        runner, agent = Runner(), _agent(provider)

        events = await _turn(runner, agent, "hi", "silent answer")
        await _turn(runner, agent, "and now?", "silent answer")

        (message,) = _answers(provider.requests[1])
        assert (message.content, message.reasoning_content) == ("hello", None)
        (answer,) = [e for e in events if e.event_type == ExecutionEventType.RESPONSE]
        assert answer.reasoning_content == ANSWER_REASONING

    @pytest.mark.asyncio
    async def test_another_agents_answer_never_brings_its_reasoning(self) -> None:
        ask = BaseEvent(
            event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text="go")]
        )
        foreign = BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="Other",
            content=[Part(text="hello")],
            reasoning_content=ANSWER_REASONING,
        )
        context = InteractionContext(
            conversation_id="c", user_id="u", agent_input=ask, interaction_history=[ask, foreign]
        )

        messages = await ToolUseWithThoughtContextManager().get_context(context, _agent(_Script()))

        assert all(m.reasoning_content is None for m in messages)
        assert ANSWER_REASONING not in json.dumps([m.to_dict() for m in messages])


class TestAProviderThatDoesNotSendIt:
    @pytest.mark.asyncio
    async def test_gets_only_the_calls(self) -> None:
        provider = _Silent(replies=[_calls(), _answer("done")])

        await _turn(Runner(), _agent(provider), "list a and b", "silent")

        (message,) = _assistants_with_calls(provider.requests[1])
        assert message.reasoning_content is None
        _thoughts_only_in_the_calls(message)

    @pytest.mark.asyncio
    async def test_the_reasoning_is_stored_all_the_same(self) -> None:
        """Whether it is sent is decided per request, by the provider rendering it:
        a turn resumed on one that sends it gets it."""
        provider = _Silent(replies=[_calls(), _answer("done")])

        events = await _turn(Runner(), _agent(provider), "list a and b", "stored")

        starts = [e for e in events if e.event_type == ExecutionEventType.TOOL_START]
        assert [s.reasoning_content for s in starts] == [REASONING, None]
        ask = BaseEvent(
            event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text="go")]
        )
        context = InteractionContext(
            conversation_id="c", user_id="u", agent_input=ask, interaction_history=[ask, *starts]
        )
        resumed = await ToolUseWithThoughtContextManager().get_context(context, _agent(_Script()))
        (message,) = _assistants_with_calls(resumed)
        assert message.reasoning_content == REASONING

    def test_openai_and_anthropic_never_serialize_it(self) -> None:
        message = LLMMessage(
            role="assistant",
            content="look at a",
            tool_calls=[_call(0, "a", "look at a")],
            reasoning_content=REASONING,
        )
        assert "reasoning_content" not in message.to_dict()
        assert REASONING not in json.dumps(OpenAIProvider._serialize_messages([message]))
        anthropic = AnthropicProvider(model_name="claude", api_key="k")
        assert REASONING not in json.dumps(anthropic._convert_messages([message]))


class TestACallWithoutStoredReasoning:
    @pytest.mark.asyncio
    async def test_renders_the_call_alone(self) -> None:
        agent = _agent(_Script())
        ask = BaseEvent(
            event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text="go")]
        )
        start = BaseEvent(
            event_type=ExecutionEventType.TOOL_START,
            author="Solo",
            action=ToolAction(call_id="c0", name="look", parameters={"path": "a"}, thought="why"),
        )
        context = InteractionContext(
            conversation_id="c", user_id="u", agent_input=ask, interaction_history=[ask, start]
        )

        messages = await ToolUseWithThoughtContextManager().get_context(context, agent)

        (message,) = _assistants_with_calls(messages)
        assert message.reasoning_content is None
        assert message.content is None


class TestTheSwitch:
    @pytest.mark.asyncio
    async def test_off_sends_only_the_calls(self) -> None:
        provider = _Script(replies=[_calls(), _answer("done")], send_reasoning=False)

        events = await _turn(Runner(), _agent(provider), "list a and b", "switched off")

        (message,) = _assistants_with_calls(provider.requests[1])
        assert message.reasoning_content is None
        _thoughts_only_in_the_calls(message)
        starts = [e for e in events if e.event_type == ExecutionEventType.TOOL_START]
        assert starts[0].reasoning_content == REASONING

    def test_off_no_key_is_in_use(self) -> None:
        assert _Script().reasoning_key_in_use == "reasoning_content"
        assert _Script(send_reasoning=False).reasoning_key_in_use is None
        assert _Silent().reasoning_key_in_use is None

    def test_it_is_read_from_a_config(self) -> None:
        provider = LLMProviderFactory.from_config(
            {"type": "vllm", "model_name": "m", "url": "http://x", "send_reasoning": False}
        )

        assert provider.reasoning_key_in_use is None


def test_the_reasoning_goes_under_the_providers_key() -> None:
    message = LLMMessage(role="assistant", content=None, reasoning_content=REASONING)

    assert message.to_dict(reasoning_key="reasoning") == {
        "role": "assistant",
        "reasoning": REASONING,
    }
    assert "reasoning" not in LLMMessage(role="assistant", content="x").to_dict(
        reasoning_key="reasoning"
    )


def test_reasoning_counts_toward_the_estimate() -> None:
    estimator = HeuristicEstimator()
    bare = LLMMessage(role="assistant", tool_calls=[_call(0, "a", "t")])
    with_reasoning = bare.model_copy(update={"reasoning_content": "x" * 3000})

    assert estimator.estimate_messages([with_reasoning]) >= estimator.estimate_messages(
        [bare]
    ) + estimator.estimate_text("x" * 3000)


def test_reasoning_brings_compaction_to_its_threshold() -> None:
    """A conversation on the threshold without its reasoning is over it with."""
    bare = [
        LLMMessage(role="user", content="list a and b"),
        LLMMessage(role="assistant", tool_calls=[_call(0, "a", "t")]),
        LLMMessage(role="tool", tool_call_id="c0", name="look", content="contents of a"),
    ]
    with_reasoning = [m.model_copy(deep=True) for m in bare]
    with_reasoning[1].reasoning_content = "x" * 3000
    window = HeuristicEstimator().estimate_messages(bare) + 1_000
    agent = SimpleNamespace(name="Solo", llm_provider=SimpleNamespace(context_window=window))
    compactor = Compactor(
        strategy=PruneOnlyCompaction(), policy=CompactionPolicy(reserve_tokens=1_000)
    )

    assert compactor.assess(bare, agent).needed is False
    assert compactor.assess(with_reasoning, agent).needed is True


class TestTheSummarizer:
    """No message has the short thoughts as its text: the summarizer reads them
    from the calls' arguments, or the calls reach it with no reason behind them.
    The whole reasoning is not read."""

    @pytest.mark.asyncio
    async def test_reads_the_short_thoughts_not_the_reasoning(self) -> None:
        provider = _Script(replies=[_answer("summary")])
        calls = LLMMessage(
            role="assistant",
            tool_calls=[_call(0, "a", "look at a"), _call(1, "b", "look at b")],
            reasoning_content=REASONING,
        )

        await ConversationSummarizer().summarize([calls], llm_provider=provider)

        transcript = provider.requests[0][-1].content or ""
        assert "assistant: look at a\nlook at b (calls: look, look)" in transcript
        assert REASONING not in transcript

    @pytest.mark.parametrize("sent", [True, False])
    @pytest.mark.asyncio
    async def test_reads_the_same_whether_the_reasoning_was_sent_or_not(self, sent: bool) -> None:
        """Sent, the thoughts are only in the arguments; not sent, they are the
        message's text too, and are not read twice."""
        provider = _Script(replies=[_calls(), _answer("done")], send_reasoning=sent)
        await _turn(Runner(), _agent(provider), "list a and b", f"summary {sent}")
        (calls,) = _assistants_with_calls(provider.requests[1])
        summarizer = _Script(replies=[_answer("summary")])

        await ConversationSummarizer().summarize([calls], llm_provider=summarizer)

        transcript = summarizer.requests[0][-1].content or ""
        assert transcript.endswith("assistant: look at a\nlook at b (calls: look, look)")


class TestTheSummarizerReadingTheReasoning:
    """`read_full_reasoning` reads the whole reasoning in place of the short thoughts."""

    @pytest.mark.asyncio
    async def test_reads_it_in_place_of_the_thoughts(self) -> None:
        provider = _Script(replies=[_answer("summary")])
        calls = LLMMessage(
            role="assistant",
            tool_calls=[_call(0, "a", "look at a"), _call(1, "b", "look at b")],
            reasoning_content=REASONING,
        )

        await ConversationSummarizer(read_full_reasoning=True).summarize(
            [calls], llm_provider=provider
        )

        transcript = provider.requests[0][-1].content or ""
        assert transcript.endswith(f"assistant: (calls: look, look)\n(reasoning: {REASONING})")
        assert "look at a\nlook at b" not in transcript

    @pytest.mark.asyncio
    async def test_without_reasoning_on_the_message_reads_the_thoughts(self) -> None:
        provider = _Script(replies=[_answer("summary")])
        calls = LLMMessage(role="assistant", tool_calls=[_call(0, "a", "look at a")])

        await ConversationSummarizer(read_full_reasoning=True).summarize(
            [calls], llm_provider=provider
        )

        assert (provider.requests[0][-1].content or "").endswith("look at a (calls: look)")

    @pytest.mark.asyncio
    async def test_a_cut_takes_the_reasoning_before_the_text(self) -> None:
        provider = _Script(replies=[_answer("summary")])
        answer = LLMMessage(role="assistant", content="the answer", reasoning_content="x" * 500)

        await ConversationSummarizer(read_full_reasoning=True, max_chars_per_message=100).summarize(
            [answer], llm_provider=provider
        )

        transcript = provider.requests[0][-1].content or ""
        assert "assistant: the answer\n(reasoning: xxx" in transcript
        assert transcript.endswith("…(truncated)")

    @pytest.mark.parametrize("value", [True, False])
    def test_it_is_read_from_a_compactor_config(self, value: bool) -> None:
        compactor = build_compactor(CompactionConfig(read_full_reasoning=value))

        assert isinstance(compactor.strategy, TwoPhaseCompaction)
        assert compactor.strategy.summarizer.read_full_reasoning is value

    def test_off_by_default(self) -> None:
        compactor = build_compactor(CompactionConfig())

        assert isinstance(compactor.strategy, TwoPhaseCompaction)
        assert compactor.strategy.summarizer.read_full_reasoning is False
        assert "read_full_reasoning" not in CompactionConfig().model_dump(exclude_none=True)
