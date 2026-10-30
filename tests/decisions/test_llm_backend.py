"""An ordinary LLM as a decision backend, and the one way it lies.

The central test here is `TestTheReasoningTrap`. Measured on a real reasoning
model: reading the first token in the logprobs array that looks like an answer
picks it out of the model's *thinking* — "I should respond with one word:
allow" — and scores 7/13 with confidence 1.00 on every case. That result looks
excellent and is noise, which is exactly why it needs a test rather than a
comment.
"""

import math

import pytest

from domyn.agents.decisions import LLMDecisionProvider, Provenance, choice, noul
from domyn.agents.llm import LLMResponse

VERDICT = choice("what now?", {"allow": "run it", "ask": "confirm", "deny": "refuse"})


def _response(content: str, logprobs=None) -> LLMResponse:
    return LLMResponse(
        content=content,
        generated_tokens=1,
        execution_time=0.1,
        logprobs=logprobs,
        usage={},
    )


def _token(text: str, alternatives: dict[str, float]) -> dict:
    return {
        "token": text,
        "top_logprobs": [
            {"token": name, "logprob": math.log(p)} for name, p in alternatives.items()
        ],
    }


class FakeLLM:
    """Only what the decision provider touches."""

    model_name = "fake-model"

    def __init__(self, *responses: LLMResponse):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def generate(self, messages, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]


class NoLogprobsLLM(FakeLLM):
    """A provider whose `generate` does not accept the logprobs arguments."""

    async def generate(self, messages, temperature=0, max_tokens=None):
        self.calls.append({"temperature": temperature})
        return self._responses[0]


class TestTheReasoningTrap:
    @pytest.mark.asyncio
    async def test_the_answer_is_read_from_the_end_not_from_the_thinking(self):
        # "... respond with one word: allow" then the real answer: deny.
        logprobs = {
            "content": [
                _token("I", {"I": 0.9}),
                _token(" allow", {" allow": 0.8, " deny": 0.2}),  # inside the reasoning
                _token("</think>", {"</think>": 1.0}),
                _token("deny", {"deny": 0.91, "allow": 0.06, "ask": 0.03}),
            ]
        }
        provider = LLMDecisionProvider(llm=FakeLLM(_response("deny", logprobs)), mode="logprobs")
        answer = (await provider.decide({"cmd": "rm -rf /"}, {"v": VERDICT})).get("v")

        assert answer.value == "deny"
        assert answer.confidence == pytest.approx(0.91, abs=0.01)
        assert answer.provenance is Provenance.LOGPROBS

    @pytest.mark.asyncio
    async def test_a_single_observable_label_is_not_a_distribution(self):
        """Confidence 1.00 because nothing else was visible is not a
        measurement. Rather than report it, the provider asks again the honest
        way and stops claiming logprobs."""
        logprobs = {"content": [_token("deny", {"deny": 1.0})]}
        provider = LLMDecisionProvider(
            llm=FakeLLM(
                _response("deny", logprobs),
                _response('{"answer": "deny", "confidence": 0.6}'),
            ),
            mode="logprobs",
        )
        answer = (await provider.decide({"c": "x"}, {"v": VERDICT})).get("v")

        assert answer.value == "deny"
        assert answer.provenance is Provenance.STATED
        assert provider.provenance is Provenance.STATED

    @pytest.mark.asyncio
    async def test_a_downgraded_provider_stops_claiming_logprobs(self):
        logprobs = {"content": [_token("deny", {"deny": 1.0})]}
        provider = LLMDecisionProvider(
            llm=FakeLLM(
                _response("deny", logprobs),
                _response('{"answer": "deny", "confidence": 0.6}'),
            ),
            mode="logprobs",
        )
        assert provider.provenance is Provenance.LOGPROBS
        await provider.decide({"c": "x"}, {"v": VERDICT})
        assert provider.provenance is Provenance.STATED

    @pytest.mark.asyncio
    async def test_no_logprobs_at_all_asks_again_for_a_real_number(self):
        """Reading the bare word off the reply returned a *known* answer with
        confidence 0.0 — safe, since nothing acts on a zero, and dishonest.
        Seen live against a reasoning model, which is how it was caught."""
        provider = LLMDecisionProvider(
            llm=FakeLLM(_response("deny"), _response('{"answer": "deny", "confidence": 0.9}')),
            mode="logprobs",
        )
        answer = (await provider.decide({"c": "x"}, {"v": VERDICT})).get("v")
        assert answer.value == "deny"
        assert answer.provenance is Provenance.STATED
        assert answer.confidence == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_a_label_split_across_tokens_is_still_found(self):
        logprobs = {"content": [_token("al", {"al": 0.7, "de": 0.3})]}
        provider = LLMDecisionProvider(llm=FakeLLM(_response("allow", logprobs)), mode="logprobs")
        answer = (await provider.decide({"c": "x"}, {"v": VERDICT})).get("v")
        assert answer.value == "allow"
        assert answer.provenance is Provenance.LOGPROBS


class TestTheStatedMode:
    @pytest.mark.asyncio
    async def test_it_reads_the_json(self):
        provider = LLMDecisionProvider(
            llm=FakeLLM(_response('Sure: {"answer": "ask", "confidence": 0.74}')), mode="stated"
        )
        answer = (await provider.decide({"c": "x"}, {"v": VERDICT})).get("v")
        assert answer.value == "ask"
        assert answer.confidence == pytest.approx(0.74)
        assert answer.provenance is Provenance.STATED

    @pytest.mark.asyncio
    async def test_an_answer_outside_the_options_is_refused(self):
        provider = LLMDecisionProvider(
            llm=FakeLLM(_response('{"answer": "maybe", "confidence": 0.9}')), mode="stated"
        )
        assert (await provider.decide({"c": "x"}, {"v": VERDICT})).get("v").known is False

    @pytest.mark.asyncio
    async def test_no_json_is_not_an_answer(self):
        provider = LLMDecisionProvider(
            llm=FakeLLM(_response("I think you should ask")), mode="stated"
        )
        assert (await provider.decide({"c": "x"}, {"v": VERDICT})).get("v").known is False

    @pytest.mark.asyncio
    async def test_a_confidence_out_of_range_is_clamped(self):
        provider = LLMDecisionProvider(
            llm=FakeLLM(_response('{"answer": "deny", "confidence": 7}')), mode="stated"
        )
        assert (await provider.decide({"c": "x"}, {"v": VERDICT})).get("v").confidence == 1.0

    @pytest.mark.asyncio
    async def test_a_missing_confidence_is_zero_not_one(self):
        provider = LLMDecisionProvider(llm=FakeLLM(_response('{"answer": "deny"}')), mode="stated")
        assert (await provider.decide({"c": "x"}, {"v": VERDICT})).get("v").confidence == 0.0


class TestNoulAnswersReadTheSameWhicheverBackendAnswered:
    @pytest.mark.asyncio
    async def test_a_yes_becomes_a_probability(self):
        provider = LLMDecisionProvider(
            llm=FakeLLM(_response('{"answer": "yes", "confidence": 0.9}')), mode="stated"
        )
        answer = (await provider.decide({"c": "x"}, {"d": noul("dangerous?")})).get("d")
        assert answer.value == pytest.approx(0.9)
        assert answer.is_true(0.8)

    @pytest.mark.asyncio
    async def test_a_no_becomes_a_low_probability(self):
        provider = LLMDecisionProvider(
            llm=FakeLLM(_response('{"answer": "no", "confidence": 0.9}')), mode="stated"
        )
        answer = (await provider.decide({"c": "x"}, {"d": noul("dangerous?")})).get("d")
        assert answer.value == pytest.approx(0.1)
        assert not answer.is_true(0.5)
        assert answer.confidence == pytest.approx(0.9)


class TestItDoesNotShareOneGenerationBetweenQuestions:
    """A decision model answers every question in one pass; an LLM writes them
    one after another, and the last is not the quality of the first."""

    @pytest.mark.asyncio
    async def test_each_question_is_its_own_request(self):
        llm = FakeLLM(
            _response('{"answer": "deny", "confidence": 0.9}'),
            _response('{"answer": "yes", "confidence": 0.9}'),
        )
        provider = LLMDecisionProvider(llm=llm, mode="stated")
        await provider.decide({"c": "x"}, {"v": VERDICT, "d": noul("dangerous?")})
        assert len(llm.calls) == 2


class TestWhenThereIsNoLLM:
    @pytest.mark.asyncio
    async def test_it_answers_nothing_rather_than_raising(self):
        provider = LLMDecisionProvider(mode="stated")
        decision = await provider.decide({"c": "x"}, {"v": VERDICT})
        assert decision.known is False
        assert "no LLM" in (decision.get("v").reason or "")

    @pytest.mark.asyncio
    async def test_bind_lends_it_the_agents_provider(self):
        provider = LLMDecisionProvider(mode="stated")
        provider.bind(FakeLLM(_response('{"answer": "deny", "confidence": 0.9}')))
        assert (await provider.decide({"c": "x"}, {"v": VERDICT})).get("v").value == "deny"


class TestAProviderThatCannotDoLogprobs:
    @pytest.mark.asyncio
    async def test_it_degrades_to_stated_instead_of_failing(self):
        llm = NoLogprobsLLM(_response('{"answer": "deny", "confidence": 0.6}'))
        provider = LLMDecisionProvider(llm=llm, mode="logprobs")
        answer = (await provider.decide({"c": "x"}, {"v": VERDICT})).get("v")
        assert answer.value == "deny"
        assert answer.provenance is Provenance.STATED
        assert provider.provenance is Provenance.STATED


class TestFindingTheModel:
    def test_a_named_provider_comes_from_the_registry(self):
        # A real BaseLLMProvider: the registry instantiates anything that is
        # not one, so a stub would be called rather than returned.
        from domyn.agents.llm import BaseLLMProvider
        from domyn.agents.registry import get_llm_registry

        class Registered(BaseLLMProvider):
            type: str = "mock"
            name: str = "registered"
            model_name: str = "m"
            supported_structured_outputs: None = None

            async def generate(self, messages, **kwargs):
                pass

            async def generate_streaming(self, messages, **kwargs):
                yield

        llm = Registered()
        get_llm_registry().register_llm_provider("named_for_decisions", llm)
        provider = LLMDecisionProvider(provider_name="named_for_decisions", mode="stated")
        assert provider._resolve_llm() is llm

    def test_an_unregistered_name_resolves_to_nothing_rather_than_raising(self):
        provider = LLMDecisionProvider(provider_name="not_registered", mode="stated")
        assert provider._resolve_llm() is None

    def test_a_bound_provider_wins_over_a_name(self):
        llm = FakeLLM(_response("x"))
        provider = LLMDecisionProvider(llm=llm, provider_name="whatever", mode="stated")
        assert provider._resolve_llm() is llm

    def test_bind_does_not_overwrite_an_explicit_one(self):
        first = FakeLLM(_response("x"))
        provider = LLMDecisionProvider(llm=first, mode="stated")
        provider.bind(FakeLLM(_response("y")))
        assert provider._resolve_llm() is first

    def test_an_unknown_mode_falls_back_to_logprobs(self):
        assert LLMDecisionProvider(mode="vibes").mode == "logprobs"
