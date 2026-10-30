"""Everything that can go wrong arrives as "I don't know", never as an exception.

This is the property the whole feature rests on: a caller that branches on
`known` behaves correctly without knowing which of the many ways to fail
happened, and a turn never dies because a decision model had a bad day.
"""

import asyncio

import pytest

from domyn.agents.decisions import Answer, Decision, DecisionProvider, NoDecisions, Provenance, noul


class Raises(DecisionProvider):
    type_name = "raises"

    async def _decide(self, state, questions):
        raise RuntimeError("the endpoint is on fire")


class Hangs(DecisionProvider):
    type_name = "hangs"

    async def _decide(self, state, questions):
        await asyncio.sleep(10)
        return Decision()


class Answers(DecisionProvider):
    type_name = "answers"
    provenance = Provenance.NATIVE

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = 0
        self.seen_state: str | None = None

    async def _decide(self, state, questions):
        self.calls += 1
        self.seen_state = state
        return Decision(
            answers={
                name: Answer(known=True, provenance=Provenance.NATIVE, value=1.0, confidence=1.0)
                for name in questions
            }
        )


QUESTION = {"q": noul("is it true?")}


class TestAFailureIsNotAnException:
    @pytest.mark.asyncio
    async def test_a_raising_backend_answers_nothing(self):
        decision = await Raises().decide({"x": 1}, QUESTION)
        assert decision.known is False
        assert decision.get("q").reason == "decision provider failed"

    @pytest.mark.asyncio
    async def test_a_hanging_backend_is_cut_off_at_the_timeout(self):
        provider = Hangs(timeout_s=0.05)
        decision = await provider.decide({"x": 1}, QUESTION)
        assert decision.get("q").reason == "decision provider timed out"

    @pytest.mark.asyncio
    async def test_the_timeout_is_the_only_protection_so_it_is_enforced_here(self):
        # The PRE_TOOL_APPROVAL phase has no timeout of its own: a hook that
        # hangs hangs the turn. So this must hold inside the provider.
        provider = Hangs(timeout_s=0.05)
        started = asyncio.get_running_loop().time()
        await provider.decide({"x": 1}, QUESTION)
        assert asyncio.get_running_loop().time() - started < 1.0


class TestTheBreakerStopsAskingADeadEndpoint:
    @pytest.mark.asyncio
    async def test_it_opens_after_the_configured_failures(self):
        provider = Raises(max_failures=2)
        assert not provider.breaker_open
        await provider.decide({"x": 1}, QUESTION)
        await provider.decide({"x": 1}, QUESTION)
        assert provider.breaker_open
        decision = await provider.decide({"x": 1}, QUESTION)
        assert "disabled" in (decision.get("q").reason or "")

    @pytest.mark.asyncio
    async def test_one_success_resets_it(self):
        provider = Answers(max_failures=2)
        provider._consecutive_failures = 1
        await provider.decide({"x": 1}, QUESTION)
        assert provider._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_max_failures_zero_never_opens_it(self):
        provider = Raises(max_failures=0)
        for _ in range(5):
            await provider.decide({"x": 1}, QUESTION)
        assert not provider.breaker_open


class TestTheClientCountsBecauseTheServerDoesNotTell:
    """Measured: a 200k and a 400k character state both returned 200 OK with
    `input_tokens: 73728` — truncated to the window, silently. An answer about
    a document the model only saw the start of is worse than no answer."""

    @pytest.mark.asyncio
    async def test_an_oversized_state_is_refused_before_sending(self):
        provider = Answers(max_state_tokens=100)
        decision = await provider.decide("x" * 100_000, QUESTION)
        assert decision.known is False
        assert decision.get("q").reason == "state longer than the model's context"
        assert provider.calls == 0

    @pytest.mark.asyncio
    async def test_a_state_that_fits_is_sent(self):
        provider = Answers(max_state_tokens=100)
        await provider.decide("x" * 30, QUESTION)
        assert provider.calls == 1

    @pytest.mark.asyncio
    async def test_no_limit_configured_sends_anything(self):
        provider = Answers()
        await provider.decide("x" * 100_000, QUESTION)
        assert provider.calls == 1

    @pytest.mark.asyncio
    async def test_an_oversized_state_does_not_trip_the_breaker(self):
        # Our own refusal is not the endpoint misbehaving.
        provider = Answers(max_state_tokens=10, max_failures=1)
        await provider.decide("x" * 10_000, QUESTION)
        assert not provider.breaker_open


class TestTheProtocolLimits:
    @pytest.mark.asyncio
    async def test_more_than_64_questions_is_refused_before_sending(self):
        provider = Answers()
        questions = {f"q{i}": noul("true?") for i in range(65)}
        decision = await provider.decide({"x": 1}, questions)
        assert decision.known is False
        assert provider.calls == 0

    @pytest.mark.asyncio
    async def test_exactly_64_is_allowed(self):
        provider = Answers()
        questions = {f"q{i}": noul("true?") for i in range(64)}
        assert (await provider.decide({"x": 1}, questions)).known
        assert provider.calls == 1

    @pytest.mark.asyncio
    async def test_no_questions_asks_nothing(self):
        provider = Answers()
        await provider.decide({"x": 1}, {})
        assert provider.calls == 0


class TestTheStateIsSerializedOnce:
    @pytest.mark.asyncio
    async def test_a_dict_becomes_json(self):
        provider = Answers()
        await provider.decide({"command": "rm -rf /"}, QUESTION)
        assert provider.seen_state == '{"command": "rm -rf /"}'

    @pytest.mark.asyncio
    async def test_a_string_is_passed_through(self):
        provider = Answers()
        await provider.decide("already text", QUESTION)
        assert provider.seen_state == "already text"

    @pytest.mark.asyncio
    async def test_non_ascii_is_not_escaped(self):
        # ensure_ascii would triple the size of an Italian state for nothing.
        provider = Answers()
        await provider.decide({"k": "perché"}, QUESTION)
        assert "perché" in (provider.seen_state or "")


class TestTheMissingAnswerIsFilledIn:
    @pytest.mark.asyncio
    async def test_a_question_the_backend_ignored_comes_back_unknown(self):
        class Partial(DecisionProvider):
            type_name = "partial"

            async def _decide(self, state, questions):
                return Decision(answers={"a": Answer(known=True, value=1.0, confidence=1.0)})

        decision = await Partial().decide({"x": 1}, {"a": noul("?"), "b": noul("?")})
        assert decision.get("a").known
        assert decision.get("b").known is False
        assert decision.get("b").reason == "not answered"


class TestNoDecisionsIsTheDefaultAndCostsNothing:
    @pytest.mark.asyncio
    async def test_it_does_not_act(self):
        assert NoDecisions().acts is False

    @pytest.mark.asyncio
    async def test_it_answers_nothing_without_touching_the_network(self):
        decision = await NoDecisions().decide({"x": 1}, QUESTION)
        assert decision.known is False
        assert decision.get("q").reason == "no decision provider configured"

    @pytest.mark.asyncio
    async def test_it_short_circuits_before_serializing_a_huge_state(self):
        # The point of `acts`: a caller may skip building the state, and the
        # provider must not pay for one that was built anyway.
        decision = await NoDecisions().decide("x" * 5_000_000, QUESTION)
        assert decision.known is False
