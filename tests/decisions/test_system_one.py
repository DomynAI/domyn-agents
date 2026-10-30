"""The wire protocol, pinned against the shape a Clef server returns.

Every payload in here is shaped like a `POST /v1/systemone` response. The
quirks it pins — a `noul` with no confidence field, zero-indexed score levels —
are the ones that would otherwise be found by reading answers wrong in
production.
"""

import httpx
import pytest
import respx

from domyn.agents.decisions import (
    Provenance,
    SystemOneProvider,
    WorkersAIProvider,
    choice,
    noul,
    score,
)

URL = "http://clef.test/v1/systemone"

# The shape of a response, for `state={"cmd": "rm -rf /"}` with one of each type.
SAMPLE_RESPONSE = {
    "model": "Cloudflare/clef",
    "answers": {
        "danger": {"type": "noul", "noul": 0.9815},
        "verdict": {
            "type": "choice",
            "choice": "deny",
            "confidence": 0.9341,
            "probabilities": {"allow": 0.0123, "ask": 0.0535, "deny": 0.9341},
        },
        "risk": {
            "type": "score",
            "score": 1.8575,
            "confidence": 0.9088,
            "legend": {"0": "none", "1": "some", "2": "lots"},
            "probabilities": {"0": 0.0513, "1": 0.0399, "2": 0.9088},
        },
    },
    "usage": {"input_tokens": 315, "output_tokens": 0},
}

QUESTIONS = {
    "danger": noul("is it destructive?"),
    "verdict": choice("what now?", {"allow": "run", "ask": "confirm", "deny": "refuse"}),
    "risk": score("how risky?", ["none", "some", "lots"]),
}


def provider(**kwargs) -> SystemOneProvider:
    return SystemOneProvider(
        base_url="http://clef.test", model="Cloudflare/clef", timeout_s=5.0, **kwargs
    )


class TestTheRequestIsWhatTheServerExpects:
    @pytest.mark.asyncio
    @respx.mock
    async def test_the_model_name_is_sent_verbatim(self):
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=SAMPLE_RESPONSE))
        await provider().decide({"cmd": "rm -rf /"}, QUESTIONS)
        # `clef` is rejected by a server that serves `Cloudflare/clef`. The name
        # is configuration, never derived from anything.
        assert route.calls.last.request.read().count(b'"Cloudflare/clef"') == 1

    @pytest.mark.asyncio
    @respx.mock
    async def test_questions_are_sent_by_name_with_their_criteria(self):
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=SAMPLE_RESPONSE))
        await provider().decide({"cmd": "x"}, QUESTIONS)
        body = route.calls.last.request.read().decode()
        assert '"questions"' in body
        assert '"criteria"' in body
        assert '"type":"noul"' in body
        # The state travels as text, which the server accepts (verified live):
        # the client has to serialize it anyway to count its tokens.
        assert '"state":"{' in body

    @pytest.mark.asyncio
    @respx.mock
    async def test_no_auth_header_without_a_key(self):
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=SAMPLE_RESPONSE))
        await provider().decide({"cmd": "x"}, QUESTIONS)
        assert "authorization" not in route.calls.last.request.headers

    @pytest.mark.asyncio
    @respx.mock
    async def test_a_key_becomes_a_bearer_header(self):
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=SAMPLE_RESPONSE))
        await provider(api_key="secret").decide({"cmd": "x"}, QUESTIONS)
        assert route.calls.last.request.headers["authorization"] == "Bearer secret"


class TestTheAnswersAreReadCorrectly:
    @pytest.mark.asyncio
    @respx.mock
    async def test_a_choice_carries_its_probabilities(self):
        respx.post(URL).mock(return_value=httpx.Response(200, json=SAMPLE_RESPONSE))
        answer = (await provider().decide({"cmd": "x"}, QUESTIONS)).get("verdict")
        assert answer.value == "deny"
        assert answer.confidence == pytest.approx(0.9341)
        assert answer.probabilities["allow"] == pytest.approx(0.0123)
        assert answer.provenance is Provenance.NATIVE

    @pytest.mark.asyncio
    @respx.mock
    async def test_a_noul_gets_a_derived_confidence(self):
        """There is no confidence field on a noul. 0.98 and 0.02 are both
        confident answers; 0.5 is the model saying it has no idea."""
        respx.post(URL).mock(return_value=httpx.Response(200, json=SAMPLE_RESPONSE))
        answer = (await provider().decide({"cmd": "x"}, QUESTIONS)).get("danger")
        assert answer.value == pytest.approx(0.9815)
        assert answer.confidence == pytest.approx(0.9815)
        assert answer.is_true(0.8)

    @pytest.mark.asyncio
    @respx.mock
    async def test_a_confidently_false_noul_is_confident(self):
        respx.post(URL).mock(
            return_value=httpx.Response(
                200, json={"answers": {"danger": {"type": "noul", "noul": 0.02}}}
            )
        )
        answer = (await provider().decide({"c": "x"}, {"danger": noul("?")})).get("danger")
        assert answer.confidence == pytest.approx(0.98)
        assert answer.is_true(0.8) is False

    @pytest.mark.asyncio
    @respx.mock
    async def test_an_undecided_noul_has_low_confidence(self):
        respx.post(URL).mock(
            return_value=httpx.Response(
                200, json={"answers": {"danger": {"type": "noul", "noul": 0.5}}}
            )
        )
        answer = (await provider().decide({"c": "x"}, {"danger": noul("?")})).get("danger")
        assert answer.confidence == pytest.approx(0.5)
        assert not answer.acts_above(0.8, minimum=Provenance.NATIVE)

    @pytest.mark.asyncio
    @respx.mock
    async def test_a_score_keeps_its_fractional_value(self):
        respx.post(URL).mock(return_value=httpx.Response(200, json=SAMPLE_RESPONSE))
        answer = (await provider().decide({"cmd": "x"}, QUESTIONS)).get("risk")
        assert answer.value == pytest.approx(1.8575)
        assert answer.probabilities["2"] == pytest.approx(0.9088)


class TestAMalformedResponseIsNotAnAnswer:
    @pytest.mark.asyncio
    @respx.mock
    async def test_a_missing_answers_object_answers_nothing(self):
        respx.post(URL).mock(return_value=httpx.Response(200, json={"usage": {}}))
        assert (await provider().decide({"c": "x"}, QUESTIONS)).known is False

    @pytest.mark.asyncio
    @respx.mock
    async def test_a_noul_without_a_number_is_unknown(self):
        respx.post(URL).mock(
            return_value=httpx.Response(
                200, json={"answers": {"danger": {"type": "noul", "noul": "very"}}}
            )
        )
        answer = (await provider().decide({"c": "x"}, {"danger": noul("?")})).get("danger")
        assert answer.known is False
        assert answer.reason == "malformed noul answer"

    @pytest.mark.asyncio
    @respx.mock
    async def test_an_http_error_answers_nothing(self):
        respx.post(URL).mock(
            return_value=httpx.Response(
                400, text="model 'clef' is not served here; available: 'Cloudflare/clef'"
            )
        )
        assert (await provider().decide({"c": "x"}, QUESTIONS)).known is False

    @pytest.mark.asyncio
    @respx.mock
    async def test_a_choice_outside_the_offered_options_is_still_reported(self):
        # Not our place to second-guess the server; the caller compares against
        # its own options and a threshold.
        respx.post(URL).mock(
            return_value=httpx.Response(
                200,
                json={"answers": {"verdict": {"type": "choice", "choice": "maybe"}}},
            )
        )
        answer = (await provider().decide({"c": "x"}, {"verdict": QUESTIONS["verdict"]})).get(
            "verdict"
        )
        assert answer.value == "maybe"
        assert answer.confidence == 0.0


class TestTheWorkersAIEnvelope:
    @pytest.mark.asyncio
    @respx.mock
    async def test_the_result_wrapper_is_unwrapped(self):
        url = "https://api.cloudflare.com/client/v4/accounts/acc1/ai/run/@cf/cloudflare/clef"
        respx.post(url).mock(
            return_value=httpx.Response(200, json={"result": SAMPLE_RESPONSE, "success": True})
        )
        workers = WorkersAIProvider(account_id="acc1", api_key="k", timeout_s=5.0)
        decision = await workers.decide({"cmd": "x"}, QUESTIONS)
        assert decision.get("verdict").value == "deny"

    @pytest.mark.asyncio
    @respx.mock
    async def test_a_bare_response_still_works(self):
        """The local server and the Jev API answer without the envelope."""
        respx.post(URL).mock(return_value=httpx.Response(200, json=SAMPLE_RESPONSE))
        assert (await provider().decide({"c": "x"}, QUESTIONS)).get("verdict").value == "deny"


class TestLatencyIsMeasured:
    @pytest.mark.asyncio
    @respx.mock
    async def test_the_decision_carries_its_own_timing(self):
        respx.post(URL).mock(return_value=httpx.Response(200, json=SAMPLE_RESPONSE))
        decision = await provider().decide({"c": "x"}, QUESTIONS)
        assert decision.latency_ms >= 0
        assert decision.model == "Cloudflare/clef"
        assert decision.provider_type == "system_one"


class TestTheClientItOwns:
    @pytest.mark.asyncio
    async def test_closing_twice_is_safe(self):
        built = provider()
        built._http()
        await built.aclose()
        await built.aclose()

    @pytest.mark.asyncio
    async def test_a_lent_client_is_not_closed(self):
        """A caller that passed its own client still owns it."""
        async with httpx.AsyncClient() as lent:
            built = SystemOneProvider(
                base_url="http://clef.test", model="m", client=lent, timeout_s=1.0
            )
            await built.aclose()
            assert not lent.is_closed

    @pytest.mark.asyncio
    @respx.mock
    async def test_a_non_object_response_answers_nothing(self):
        respx.post(URL).mock(return_value=httpx.Response(200, json=["not", "an", "object"]))
        assert (await provider().decide({"c": "x"}, QUESTIONS)).known is False

    def test_the_workers_ai_url_is_built_from_the_account(self):
        workers = WorkersAIProvider(account_id="acc1", api_key="k")
        assert workers.url.endswith("/accounts/acc1/ai/run/@cf/cloudflare/clef")
