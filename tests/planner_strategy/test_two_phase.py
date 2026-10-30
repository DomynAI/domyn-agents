"""Two phases: prune first, summarize only when pruning was not enough.

The order carries the whole design. Phase one is deterministic, free and instant,
and on an agent conversation it recovers most of the space because tool output is
what inflates a context. Phase two costs a request and is not reproducible, so it
has to run when the cheap answer has demonstrably failed — not on a schedule and
not on a guess. Half of these tests are therefore about phase two **not**
happening.
"""

from typing import Any

import pytest

from domyn.agents.llm import LLMMessage, LLMResponse, LLMToolCall, ToolCallFunction
from domyn.agents.planner_strategy.context_management.compaction import (
    SUMMARY_MARKER,
    CompactionPolicy,
    Compactor,
    ConversationSummarizer,
    TwoPhaseCompaction,
    read_record,
)

AGENT = "Assistant"
SUMMARY = "The user asked for three files; all were read. Open: nothing."


class _Provider:
    """Records what it was asked, so the prompt can be inspected."""

    context_window = 8_000

    def __init__(self, answer: str = SUMMARY, fail: bool = False) -> None:
        self.seen: list[list[LLMMessage]] = []
        self.answer = answer
        self.fail = fail

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        if self.fail:
            raise RuntimeError("model is down")
        self.seen.append(list(messages))
        return LLMResponse(content=self.answer, generated_tokens=1, execution_time=0.0, usage={})


def _agent(provider: Any) -> Any:
    class _Agent:
        llm_provider = provider
        name = AGENT

    return _Agent()


def _compactor(**policy: Any) -> Compactor:
    return Compactor(
        strategy=TwoPhaseCompaction(),
        policy=CompactionPolicy(reserve_tokens=2_000, keep_recent_tokens=1_500, **policy),
    )


def _tool_heavy(turns: int = 8) -> list[LLMMessage]:
    """Big tool output, little prose — pruning alone should be enough."""
    messages = [LLMMessage(role="system", content="sys")]
    for index in range(turns):
        messages.append(
            LLMMessage(role="user", content=f"read file {index}", source_event_id=f"e{index}u")
        )
        messages.append(
            LLMMessage(
                role="assistant",
                content=None,
                source_event_id=f"e{index}c",
                tool_calls=[
                    LLMToolCall(
                        id=f"c{index}",
                        type="function",
                        function=ToolCallFunction(name="read_file", arguments="{}"),
                    )
                ],
            )
        )
        messages.append(
            LLMMessage(
                role="tool",
                tool_call_id=f"c{index}",
                name="read_file",
                source_event_id=f"e{index}t",
                content="exit code: 0\n" + f"line of file {index}\n" * 200,
            )
        )
        messages.append(
            LLMMessage(
                role="assistant", content=f"file {index} read.", source_event_id=f"e{index}a"
            )
        )
    return messages


def _prose_heavy(turns: int = 30) -> list[LLMMessage]:
    """Little tool output, lots of conversation — pruning cannot be enough."""
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
                source_event_id=f"a{index}",
            )
        )
    return messages


class TestPhaseTwoStaysOutOfTheWay:
    @pytest.mark.asyncio
    async def test_no_model_is_called_when_pruning_suffices(self) -> None:
        """The expensive phase must not run just because the threshold was
        crossed: on an agent conversation the free phase usually wins."""
        provider = _Provider()
        compactor = _compactor()
        messages = _tool_heavy()

        plan = compactor.assess(messages, _agent(provider))
        assert plan.needed is True, plan.reason
        _, done = await compactor.run(messages, plan, {}, AGENT)

        assert provider.seen == [], "phase two ran when phase one was enough"
        assert done.metadata["sufficient"] is True
        assert "tool outputs dropped" in str(done.content[0].text)

    @pytest.mark.asyncio
    async def test_nothing_at_all_happens_below_the_threshold(self) -> None:
        provider = _Provider()
        compactor = _compactor()
        plan = compactor.assess([LLMMessage(role="user", content="hi")], _agent(provider))
        assert plan.needed is False
        assert provider.seen == []

    @pytest.mark.asyncio
    async def test_no_summary_is_recorded_when_phase_one_was_enough(self) -> None:
        state: dict[str, Any] = {}
        compactor = _compactor()
        messages = _tool_heavy()
        plan = compactor.assess(messages, _agent(_Provider()))
        await compactor.run(messages, plan, state, AGENT)
        assert read_record(state, AGENT).summary is None


class TestPhaseTwoWhenItIsNeeded:
    @pytest.mark.asyncio
    async def test_it_summarizes_and_gets_under_the_limit(self) -> None:
        provider = _Provider()
        compactor = _compactor()
        messages = _prose_heavy()

        plan = compactor.assess(messages, _agent(provider))
        _, done = await compactor.run(messages, plan, {}, AGENT)

        assert len(provider.seen) == 1, "exactly one request, not one per message"
        assert done.metadata["sufficient"] is True
        assert done.metadata["tokens_after"] < done.metadata["tokens_before"] / 4

    @pytest.mark.asyncio
    async def test_pruning_has_to_get_the_whole_request_under(self) -> None:
        """Tool definitions travel with every request and no phase can shrink
        them: a pruned conversation that would fit on its own is not enough."""
        provider = _Provider()
        compactor = _compactor()
        messages = _tool_heavy()
        tools = [{"type": "function", "function": {"name": "t", "description": "x" * 13_500}}]

        plan = compactor.assess(messages, _agent(provider), tool_defs=tools)
        await compactor.run(messages, plan, {}, AGENT)

        assert provider.seen, "pruning fit the conversation but not the request, and no summary"

    @pytest.mark.asyncio
    async def test_a_window_smaller_than_the_kept_tail_still_gets_under(self) -> None:
        """With a tail budget over the threshold the tail alone did not fit:
        nothing summarizable was left, and every request compacted again."""
        compactor = Compactor(
            strategy=TwoPhaseCompaction(),
            policy=CompactionPolicy(reserve_tokens=2_000, keep_recent_tokens=20_000),
        )
        messages = _prose_heavy()

        plan = compactor.assess(messages, _agent(_Provider()))
        _, done = await compactor.run(messages, plan, {}, AGENT)

        assert done.metadata["sufficient"] is True

    @pytest.mark.asyncio
    async def test_the_last_request_is_never_summarized(self) -> None:
        """An agent that lost the last request answers a summary of it. History
        before it is summarized; the request stays verbatim, even outside the
        tail."""
        provider = _Provider()
        compactor = _compactor()
        messages = _prose_heavy(10)
        messages.append(LLMMessage(role="user", content="THE REQUEST", source_event_id="request"))
        messages += [
            LLMMessage(
                role="assistant",
                content=f"step {index} " + "at considerable length. " * 60,
                source_event_id=f"step{index}",
            )
            for index in range(6)
        ]

        plan = compactor.assess(messages, _agent(provider))
        out, _ = await compactor.run(messages, plan, {}, AGENT)

        assert provider.seen, "the fixture must need a summary"
        assert any(m.role == "user" and m.content == "THE REQUEST" for m in out)

    @pytest.mark.asyncio
    async def test_the_summary_is_in_the_messages_and_marked(self) -> None:
        compactor = _compactor()
        messages = _prose_heavy()
        plan = compactor.assess(messages, _agent(_Provider()))
        out, _ = await compactor.run(messages, plan, {}, AGENT)

        injected = [m for m in out if SUMMARY_MARKER in str(m.content)]
        assert len(injected) == 1
        assert SUMMARY in str(injected[0].content)

    @pytest.mark.asyncio
    async def test_the_system_prompt_survives(self) -> None:
        """It has no originating event. Dropping it is a different failure
        entirely, and one that looks like the model losing its instructions."""
        compactor = _compactor()
        messages = _prose_heavy()
        plan = compactor.assess(messages, _agent(_Provider()))
        out, _ = await compactor.run(messages, plan, {}, AGENT)
        assert out[0].role == "system"

    @pytest.mark.asyncio
    async def test_the_summary_sits_where_the_history_was(self) -> None:
        """Order is what tells the model what came before what. A summary of the
        middle parked above the system prompt would read as instructions."""
        compactor = _compactor()
        messages = _prose_heavy()
        plan = compactor.assess(messages, _agent(_Provider()))
        out, _ = await compactor.run(messages, plan, {}, AGENT)

        position = next(i for i, m in enumerate(out) if SUMMARY_MARKER in str(m.content))
        assert position > 0, "the summary must not displace the system prompt"
        assert position < len(out) - 1, "and must not be the last thing the model reads"

    @pytest.mark.asyncio
    async def test_the_most_recent_exchange_is_not_summarized(self) -> None:
        compactor = _compactor()
        messages = _prose_heavy()
        plan = compactor.assess(messages, _agent(_Provider()))
        out, _ = await compactor.run(messages, plan, {}, AGENT)
        assert any("question 29" in str(m.content) for m in out)

    @pytest.mark.asyncio
    async def test_the_summarizer_is_shown_the_pruned_text_not_the_original(self) -> None:
        """Phase one already replaced those bodies with a line saying what they
        were. Paying to summarize the full text would undo the phase that just
        removed it."""
        provider = _Provider()
        compactor = _compactor()
        messages = _prose_heavy(20)
        # Inside a turn well into the middle. Position matters: the summarized
        # span begins at a user message, so anything before the middle's first
        # user turn is deliberately left out of it.
        messages.insert(
            8,
            LLMMessage(
                role="tool",
                tool_call_id="c",
                name="read_file",
                source_event_id="tool-1",
                content="exit code: 0\nSECRET_NEEDLE\n" + "filler\n" * 500,
            ),
        )
        plan = compactor.assess(messages, _agent(provider))
        await compactor.run(messages, plan, {}, AGENT)

        assert provider.seen, "phase two did not run"
        prompt = str(provider.seen[0][-1].content)
        assert "dropped to save context" in prompt
        assert "SECRET_NEEDLE" not in prompt


class TestWhenTheSummaryCannotBeHad:
    @pytest.mark.asyncio
    async def test_a_failing_summarizer_keeps_the_pruning(self) -> None:
        """`Compactor.apply` discards the whole outcome on an exception, so
        letting the failure escape would throw away the space phase one had
        already recovered for free."""
        compactor = _compactor()
        messages = _prose_heavy()
        messages.insert(
            3,
            LLMMessage(
                role="tool",
                tool_call_id="c",
                name="t",
                source_event_id="tool-1",
                content="exit code: 0\n" + "output\n" * 800,
            ),
        )
        plan = compactor.assess(messages, _agent(_Provider(fail=True)))
        _, done = await compactor.run(messages, plan, {}, AGENT)

        assert done.metadata["changed"] is True
        assert done.metadata["tokens_after"] < done.metadata["tokens_before"]
        assert "summary failed" in str(done.content[0].text)

    @pytest.mark.asyncio
    async def test_a_failed_summary_is_not_recorded(self) -> None:
        state: dict[str, Any] = {}
        compactor = _compactor()
        messages = _prose_heavy()
        plan = compactor.assess(messages, _agent(_Provider(fail=True)))
        await compactor.run(messages, plan, state, AGENT)
        assert read_record(state, AGENT).summary is None

    @pytest.mark.asyncio
    async def test_an_empty_summary_counts_as_a_failure(self) -> None:
        """An empty string would be recorded as a summary and replace the history
        with nothing at all."""
        compactor = _compactor()
        messages = _prose_heavy()
        plan = compactor.assess(messages, _agent(_Provider(answer="   ")))
        _, done = await compactor.run(messages, plan, {}, AGENT)
        assert "summary failed" in str(done.content[0].text)

    @pytest.mark.asyncio
    async def test_with_no_provider_it_returns_phase_one(self) -> None:
        compactor = _compactor()
        messages = _prose_heavy()

        class _NoProvider:
            name = AGENT
            llm_provider = None

        plan = compactor.assess(messages, _NoProvider())
        # No window on a `None` provider, so drive `apply` directly.
        from domyn.agents.planner_strategy import HeuristicEstimator
        from domyn.agents.planner_strategy.context_management.compaction import (
            Budget,
            CompactionPlan,
        )

        estimator = HeuristicEstimator()
        plan = CompactionPlan(
            needed=True,
            budget=Budget(
                context_window=8_000,
                reserve_tokens=2_000,
                current_tokens=estimator.estimate_messages(messages),
                estimator=estimator,
            ),
            llm_provider=None,
        )
        outcome = await compactor.apply(messages, plan, {})
        assert outcome.summary is None
        assert "not enough" in str(outcome.note)


class TestTheSummarySurvives:
    @pytest.mark.asyncio
    async def test_a_rerender_gets_the_summary_back_without_a_new_request(self) -> None:
        """The property the record exists for. Re-summarizing every turn would
        cost a request per turn and hand the model a different context each time
        for the same history."""
        provider = _Provider()
        state: dict[str, Any] = {}
        compactor = _compactor()
        messages = _prose_heavy()

        plan = compactor.assess(messages, _agent(provider))
        await compactor.run(messages, plan, state, AGENT)
        assert len(provider.seen) == 1

        # Next turn: the context manager rebuilds everything from the events.
        replayed = compactor.replay(messages, state, AGENT)
        assert len(provider.seen) == 1, "replay called the model again"
        assert any(SUMMARY_MARKER in str(m.content) for m in replayed)
        assert compactor.tokens_after(replayed) < compactor.tokens_after(messages) / 4

    @pytest.mark.asyncio
    async def test_replay_is_idempotent(self) -> None:
        state: dict[str, Any] = {}
        compactor = _compactor()
        messages = _prose_heavy()
        plan = compactor.assess(messages, _agent(_Provider()))
        await compactor.run(messages, plan, state, AGENT)

        once = compactor.replay(messages, state, AGENT)
        twice = compactor.replay(once, state, AGENT)
        assert [str(m.content) for m in once] == [str(m.content) for m in twice]

    @pytest.mark.asyncio
    async def test_a_second_summary_replaces_the_first(self) -> None:
        """The new summary was produced from a stretch that already contained the
        old one, so keeping both would state the same history twice."""
        provider = _Provider()
        state: dict[str, Any] = {}
        compactor = _compactor()

        plan = compactor.assess(_prose_heavy(30), _agent(provider))
        await compactor.run(_prose_heavy(30), plan, state, AGENT)
        first = read_record(state, AGENT)

        provider.answer = "A second, later summary."
        longer = _prose_heavy(60)
        plan = compactor.assess(compactor.replay(longer, state, AGENT), _agent(provider))
        await compactor.run(compactor.replay(longer, state, AGENT), plan, state, AGENT)
        second = read_record(state, AGENT)

        assert second.summary == "A second, later summary."
        assert len(second.summarized_event_ids) > len(first.summarized_event_ids), (
            "the second summary must cover at least what the first did"
        )

    @pytest.mark.asyncio
    async def test_another_agent_is_unaffected(self) -> None:
        state: dict[str, Any] = {}
        compactor = _compactor()
        messages = _prose_heavy()
        plan = compactor.assess(messages, _agent(_Provider()))
        await compactor.run(messages, plan, state, AGENT)
        assert compactor.replay(messages, state, "Researcher") is messages


class TestNotBreakingTheRequest:
    @pytest.mark.asyncio
    async def test_every_tool_result_still_has_its_call(self) -> None:
        """The failure mode that matters: a tool result whose call was summarized
        away makes the request malformed, and the provider rejects it outright."""
        compactor = _compactor()
        messages = _tool_heavy(40)
        plan = compactor.assess(messages, _agent(_Provider()))
        out, _ = await compactor.run(messages, plan, {}, AGENT)

        open_calls: set[str] = set()
        for message in out:
            for call in message.tool_calls or []:
                open_calls.add(call.id)
            if message.role == "tool":
                assert message.tool_call_id in open_calls, (
                    f"tool result {message.tool_call_id} lost its call"
                )

    @pytest.mark.asyncio
    async def test_an_event_spanning_the_boundary_is_left_alone(self) -> None:
        """One event can render into several messages and a cut can land between
        them. Dropping such an event by id would take a protected message with
        it — and if that message is half of a call pair, the request breaks."""
        compactor = _compactor()
        messages = _prose_heavy(30)
        # Same event id on both sides of any plausible cut.
        messages[2].source_event_id = "spanning"
        messages[-2].source_event_id = "spanning"

        plan = compactor.assess(messages, _agent(_Provider()))
        out, _ = await compactor.run(messages, plan, {}, AGENT)
        assert sum(1 for m in out if m.source_event_id == "spanning") == 2


class TestTheSummarizerItself:
    @pytest.mark.asyncio
    async def test_it_names_who_said_what(self) -> None:
        provider = _Provider()
        await ConversationSummarizer().summarize(
            [
                LLMMessage(role="user", content="find the bug"),
                LLMMessage(role="assistant", content="it is in parser.py"),
            ],
            llm_provider=provider,
        )
        prompt = str(provider.seen[0][-1].content)
        assert "user: find the bug" in prompt
        assert "assistant: it is in parser.py" in prompt

    @pytest.mark.asyncio
    async def test_tool_calls_are_named(self) -> None:
        provider = _Provider()
        await ConversationSummarizer().summarize(
            [
                LLMMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        LLMToolCall(
                            id="c",
                            type="function",
                            function=ToolCallFunction(name="delete_rows", arguments="{}"),
                        )
                    ],
                )
            ],
            llm_provider=provider,
        )
        assert "delete_rows" in str(provider.seen[0][-1].content)

    @pytest.mark.asyncio
    async def test_each_message_is_capped(self) -> None:
        """The input can be larger than the window we are trying to fit into, and
        a summarization request that is itself rejected leaves the turn with no
        way out at all."""
        provider = _Provider()
        summarizer = ConversationSummarizer(max_chars_per_message=100)
        await summarizer.summarize(
            [LLMMessage(role="user", content="x" * 10_000)], llm_provider=provider
        )
        assert len(str(provider.seen[0][-1].content)) < 1_000

    @pytest.mark.asyncio
    async def test_the_prompt_asks_for_what_matters(self) -> None:
        """Failures and open points are the two things a summary usually loses,
        and both cost real work when they go: the agent repeats a failed attempt
        or forgets what it was doing."""
        prompt = ConversationSummarizer().system_prompt.lower()
        assert "failed" in prompt
        assert "open" in prompt
        assert "invent" in prompt, "a summarizer told nothing about invention will invent"

    @pytest.mark.asyncio
    async def test_an_empty_answer_raises_rather_than_returning_nothing(self) -> None:
        with pytest.raises(ValueError):
            await ConversationSummarizer().summarize(
                [LLMMessage(role="user", content="hi")], llm_provider=_Provider(answer="")
            )


class TestItIsSelectableByName:
    def test_the_registry_knows_it(self) -> None:
        from domyn.agents.planner_strategy.context_management.compaction import build_strategy

        assert isinstance(build_strategy("two_phase"), TwoPhaseCompaction)

    def test_it_acts(self) -> None:
        assert TwoPhaseCompaction.acts is True

    def test_a_cheaper_model_can_be_pointed_at_the_summarizing(self) -> None:
        provider = _Provider()
        strategy = TwoPhaseCompaction(llm_provider=provider)
        assert strategy.llm_provider is provider


class TestTheSummarizedSpanFallsOnCleanBoundaries:
    """Both ends land where nothing that must stay together is separated.

    The one hard rule the wire protocol imposes is that an assistant message
    carrying `tool_calls` is followed by the results for all of them — providers
    reject a violation rather than degrading. Placing both ends on that boundary
    makes the property structural instead of something a test has to catch.

    It also means the span is complete blocks: it can begin at a user turn or at
    an assistant message, and it ends after a finished reply. An earlier version
    required whole *user* turns and refused a conversation containing one long
    exchange — a case that plainly wants compacting.
    """

    @pytest.mark.asyncio
    async def test_the_summary_replaces_a_run_that_began_with_a_user_message(self) -> None:
        compactor = _compactor()
        messages = _prose_heavy(30)
        plan = compactor.assess(messages, _agent(_Provider()))
        out, _ = await compactor.run(messages, plan, {}, AGENT)

        position = next(i for i, m in enumerate(out) if SUMMARY_MARKER in str(m.content))
        # What the summary displaced started with a user turn, so what precedes it
        # is whatever preceded that turn — never the middle of an exchange.
        assert out[position - 1].role in ("system", "user", "assistant")
        assert out[position].role == "user", "the replacement must occupy a user turn's place"

    @pytest.mark.asyncio
    async def test_no_assistant_message_is_left_without_its_question(self) -> None:
        """The visible symptom of a span starting mid-turn: an answer whose
        question was summarized away, which reads to the model as its own words
        about something nobody asked."""
        compactor = _compactor()
        messages = _prose_heavy(30)
        plan = compactor.assess(messages, _agent(_Provider()))
        out, _ = await compactor.run(messages, plan, {}, AGENT)

        for index, message in enumerate(out):
            if message.role == "assistant" and index:
                preceding = [m.role for m in out[:index]]
                assert "user" in preceding, "an answer with no question before it"

    @pytest.mark.asyncio
    async def test_a_single_long_exchange_is_summarizable(self) -> None:
        """One question and a very long answer: no second user turn anywhere, and
        it still compacts. The finished assistant blocks are boundaries."""
        compactor = _compactor()
        messages = [LLMMessage(role="system", content="sys")]
        messages.append(LLMMessage(role="user", content="the only question", source_event_id="q"))
        for index in range(60):
            messages.append(
                LLMMessage(
                    role="assistant",
                    content=f"part {index} " + "of a very long answer. " * 60,
                    source_event_id=f"a{index}",
                )
            )

        provider = _Provider()
        plan = compactor.assess(messages, _agent(provider))
        assert plan.needed is True, plan.reason
        _, done = await compactor.run(messages, plan, {}, AGENT)

        assert len(provider.seen) == 1, "it refused a conversation worth compacting"
        assert done.metadata["sufficient"] is True

    @pytest.mark.asyncio
    async def test_it_will_not_cut_between_a_call_and_its_results(self) -> None:
        """The case the rule exists for: a span ending right after the calls would
        leave them unanswered, and the provider rejects that outright."""
        from domyn.agents.planner_strategy.context_management.compaction import _clean_boundary

        sequence = [
            LLMMessage(role="user", content="go"),
            LLMMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    LLMToolCall(
                        id="c",
                        type="function",
                        function=ToolCallFunction(name="t", arguments="{}"),
                    )
                ],
            ),
            LLMMessage(role="tool", tool_call_id="c", name="t", content="done"),
            LLMMessage(role="assistant", content="finished"),
        ]
        assert _clean_boundary(sequence, 1) is True, "before the call is fine"
        assert _clean_boundary(sequence, 2) is False, "between the call and its result"
        assert _clean_boundary(sequence, 3) is True, "after the result is fine"
        assert _clean_boundary(sequence, 4) is True, "after a finished reply is fine"

    @pytest.mark.asyncio
    async def test_a_tool_call_and_its_result_are_never_separated(self) -> None:
        """Now a consequence of the boundary rather than of a retreat heuristic:
        both live inside a turn, and the span is bounded by turns."""
        compactor = _compactor()
        messages = _tool_heavy(40)
        plan = compactor.assess(messages, _agent(_Provider()))
        out, _ = await compactor.run(messages, plan, {}, AGENT)

        open_calls: set[str] = set()
        for message in out:
            for call in message.tool_calls or []:
                open_calls.add(call.id)
            if message.role == "tool":
                assert message.tool_call_id in open_calls
        # And the reverse: a call whose result was summarized away.
        results = {m.tool_call_id for m in out if m.role == "tool"}
        for message in out:
            for call in message.tool_calls or []:
                assert call.id in results, f"call {call.id} lost its result"
