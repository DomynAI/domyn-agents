"""A summary carries the agent's line of thought, not only the conversation.

With `SummaryReasoning.CARRY` the summarizer writes, in the same request, a summary of
the reasoning of the stretch it replaces, and it is put, marked as a summary, before
the reasoning of the first assistant message after the summary: what the model's
next steps continue from. Only for a provider that sends reasoning back, the one
case where it reaches the model.
"""

from typing import Any

import pytest

from domyn.agents.llm import LLMMessage, LLMResponse
from domyn.agents.planner_strategy.context_management.compaction import (
    REASONING_SUMMARY_END,
    REASONING_SUMMARY_MARKER,
    SUMMARY_MARKER,
    CompactionPolicy,
    Compactor,
    ConversationSummarizer,
    SummaryReasoning,
    TwoPhaseCompaction,
    _parse_summary,
    _restore_summary,
    _with_reasoning_summary,
    read_record,
    record_summary,
)

AGENT = "Assistant"
TEXT = "The user asked for three files; all were read. Open: nothing."
THOUGHT = "I was checking the files one by one; next I meant to compare them."
TAGGED = f"<summary>\n{TEXT}\n</summary>\n<reasoning>\n{THOUGHT}\n</reasoning>"


class _Provider:
    """Records what it was asked; sends reasoning back when `reasoning_key_in_use`
    is set, as `BaseLLMProvider` does."""

    context_window = 8_000

    def __init__(self, answer: str = TAGGED, reasoning_key: str | None = "reasoning_content"):
        self.seen: list[list[LLMMessage]] = []
        self.answer = answer
        self.reasoning_key_in_use = reasoning_key

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        self.seen.append(list(messages))
        return LLMResponse(content=self.answer, generated_tokens=1, execution_time=0.0, usage={})


def _agent(provider: Any) -> Any:
    class _Agent:
        llm_provider = provider
        name = AGENT

    return _Agent()


def _compactor(reasoning: SummaryReasoning = SummaryReasoning.CARRY) -> Compactor:
    return Compactor(
        strategy=TwoPhaseCompaction(summarizer=ConversationSummarizer(reasoning=reasoning)),
        policy=CompactionPolicy(reserve_tokens=2_000, keep_recent_tokens=1_500),
    )


def _conversation(turns: int = 30) -> list[LLMMessage]:
    """Long questions and answers, each answer with its own reasoning: pruning
    finds nothing, so the second phase runs."""
    messages = [LLMMessage(role="system", content="sys")]
    for index in range(turns):
        messages.append(
            LLMMessage(
                role="user",
                content=f"question {index} " + "with a lot of detail. " * 40,
                source_event_id=f"q{index}",
            )
        )
        messages.append(
            LLMMessage(
                role="assistant",
                content=f"answer {index} " + "at considerable length. " * 60,
                reasoning_content=f"own reasoning {index}",
                source_event_id=f"a{index}",
            )
        )
    return messages


async def _compact(
    provider: _Provider, compactor: Compactor | None = None
) -> tuple[list[LLMMessage], dict[str, Any], Compactor]:
    compactor = compactor or _compactor()
    state: dict[str, Any] = {}
    messages = _conversation()
    plan = compactor.assess(messages, _agent(provider))
    out, _ = await compactor.run(messages, plan, state, AGENT)
    return out, state, compactor


def _after_summary(messages: list[LLMMessage]) -> list[LLMMessage]:
    position = next(i for i, m in enumerate(messages) if SUMMARY_MARKER in str(m.content))
    return messages[position + 1 :]


class TestTheReasoningIsCarried:
    @pytest.mark.asyncio
    async def test_it_opens_the_reasoning_of_the_first_assistant_after_the_summary(
        self,
    ) -> None:
        out, _, _ = await _compact(_Provider())

        first = next(m for m in _after_summary(out) if m.role == "assistant")
        assert first.reasoning_content is not None
        assert first.reasoning_content.startswith(
            f"{REASONING_SUMMARY_MARKER}\n{THOUGHT}\n{REASONING_SUMMARY_END}\n\n"
        )
        own = next(m for m in _conversation() if m.source_event_id == first.source_event_id)
        assert first.reasoning_content.endswith(f"\n\n{own.reasoning_content}")

    @pytest.mark.asyncio
    async def test_only_that_message_gets_it(self) -> None:
        out, _, _ = await _compact(_Provider())

        carrying = [m for m in out if REASONING_SUMMARY_MARKER in (m.reasoning_content or "")]
        assert len(carrying) == 1

    @pytest.mark.asyncio
    async def test_the_summary_message_holds_the_conversation_part_only(self) -> None:
        out, _, _ = await _compact(_Provider())

        (summary,) = [m for m in out if SUMMARY_MARKER in str(m.content)]
        assert summary.content == f"{SUMMARY_MARKER}\n{TEXT}"
        assert summary.reasoning_content is None

    @pytest.mark.asyncio
    async def test_one_request_asks_for_both_parts(self) -> None:
        provider = _Provider()
        await _compact(provider)

        assert len(provider.seen) == 1
        system = str(provider.seen[0][0].content)
        assert "<summary>" in system and "<reasoning>" in system

    @pytest.mark.asyncio
    async def test_every_later_rendering_gets_it_back_without_a_request(self) -> None:
        provider = _Provider()
        out, state, compactor = await _compact(provider)

        replayed = compactor.replay(_conversation(), state, AGENT)

        assert len(provider.seen) == 1
        assert read_record(state, AGENT).reasoning_summary == THOUGHT
        assert [m.reasoning_content for m in replayed] == [m.reasoning_content for m in out]

    @pytest.mark.asyncio
    async def test_replaying_twice_does_not_stack_it(self) -> None:
        _, state, compactor = await _compact(_Provider())

        once = compactor.replay(_conversation(), state, AGENT)
        twice = compactor.replay(once, state, AGENT)

        assert [m.reasoning_content for m in once] == [m.reasoning_content for m in twice]


class TestWhenItIsNotCarried:
    @pytest.mark.asyncio
    async def test_a_provider_that_sends_no_reasoning_is_not_asked_for_it(self) -> None:
        provider = _Provider(answer=TEXT, reasoning_key=None)
        out, state, _ = await _compact(provider)

        assert "<reasoning>" not in str(provider.seen[0][0].content)
        assert not any(REASONING_SUMMARY_MARKER in (m.reasoning_content or "") for m in out)
        assert read_record(state, AGENT).reasoning_summary is None

    @pytest.mark.parametrize("mode", [SummaryReasoning.IGNORE, SummaryReasoning.READ])
    @pytest.mark.asyncio
    async def test_the_other_modes_ask_for_the_summary_alone(self, mode: SummaryReasoning) -> None:
        provider = _Provider(answer=TEXT)
        out, _, _ = await _compact(provider, _compactor(mode))

        assert "<reasoning>" not in str(provider.seen[0][0].content)
        assert not any(REASONING_SUMMARY_MARKER in (m.reasoning_content or "") for m in out)

    @pytest.mark.asyncio
    async def test_an_answer_without_tags_is_the_summary_whole(self) -> None:
        out, state, _ = await _compact(_Provider(answer=TEXT))

        assert any(m.content == f"{SUMMARY_MARKER}\n{TEXT}" for m in out)
        assert read_record(state, AGENT).reasoning_summary is None
        assert not any(REASONING_SUMMARY_MARKER in (m.reasoning_content or "") for m in out)


class TestWhereItGoes:
    def _messages(self) -> list[LLMMessage]:
        return [
            LLMMessage(role="system", content="sys"),
            LLMMessage(role="user", content="old", source_event_id="old-q"),
            LLMMessage(role="assistant", content="old answer", source_event_id="old-a"),
            LLMMessage(role="user", content="the request", source_event_id="q"),
        ]

    def test_with_no_assistant_after_the_summary_nothing_is_attached(self) -> None:
        out = _restore_summary(self._messages(), TEXT, {"old-q", "old-a"}, THOUGHT)

        assert [m.role for m in out] == ["system", "user", "user"]
        assert all(m.reasoning_content is None for m in out)

    @pytest.mark.asyncio
    async def test_the_reply_that_follows_gets_it_on_the_next_rendering(self) -> None:
        state: dict[str, Any] = {}
        record_summary(state, AGENT, TEXT, ["old-q", "old-a"], THOUGHT)
        reply = LLMMessage(
            role="assistant", content="done", reasoning_content="mine", source_event_id="r"
        )

        out = _compactor().replay([*self._messages(), reply], state, AGENT)

        assert out[-1].reasoning_content == (
            f"{REASONING_SUMMARY_MARKER}\n{THOUGHT}\n{REASONING_SUMMARY_END}\n\nmine"
        )

    def test_a_message_without_reasoning_gets_the_summary_alone(self) -> None:
        message = LLMMessage(role="assistant", content="done")

        carried = _with_reasoning_summary(message, THOUGHT)

        assert carried.reasoning_content == (
            f"{REASONING_SUMMARY_MARKER}\n{THOUGHT}\n{REASONING_SUMMARY_END}"
        )
        assert message.reasoning_content is None, "the original was changed"

    def test_a_later_summary_replaces_the_earlier_one(self) -> None:
        message = _with_reasoning_summary(
            LLMMessage(role="assistant", content="done", reasoning_content="mine"), "first"
        )

        carried = _with_reasoning_summary(message, "second")

        assert carried.reasoning_content == (
            f"{REASONING_SUMMARY_MARKER}\nsecond\n{REASONING_SUMMARY_END}\n\nmine"
        )

    def test_a_later_summary_without_reasoning_clears_the_earlier_one(self) -> None:
        state: dict[str, Any] = {}
        record_summary(state, AGENT, "first", ["a"], THOUGHT)

        record = record_summary(state, AGENT, "second", ["b"])

        assert record.reasoning_summary is None


class TestReadingTheAnswer:
    def test_both_parts(self) -> None:
        parsed = _parse_summary(TAGGED)

        assert (parsed.text, parsed.reasoning) == (TEXT, THOUGHT)

    def test_unclosed_tags_end_at_the_next_part_or_the_end(self) -> None:
        parsed = _parse_summary(f"<summary>{TEXT}<reasoning>{THOUGHT}")

        assert (parsed.text, parsed.reasoning) == (TEXT, THOUGHT)

    def test_no_summary_tag_is_the_text_less_the_reasoning(self) -> None:
        parsed = _parse_summary(f"{TEXT}\n<reasoning>{THOUGHT}</reasoning>")

        assert (parsed.text, parsed.reasoning) == (TEXT, THOUGHT)

    def test_an_empty_reasoning_is_none(self) -> None:
        parsed = _parse_summary(f"<summary>{TEXT}</summary><reasoning>  </reasoning>")

        assert (parsed.text, parsed.reasoning) == (TEXT, None)

    @pytest.mark.asyncio
    async def test_a_reasoning_alone_is_no_summary(self) -> None:
        with pytest.raises(ValueError):
            await ConversationSummarizer().summarize(
                [LLMMessage(role="user", content="hi")],
                llm_provider=_Provider(answer=f"<reasoning>{THOUGHT}</reasoning>"),
                carry_reasoning=True,
            )
