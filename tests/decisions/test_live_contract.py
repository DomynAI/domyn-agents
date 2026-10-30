"""The contract against a real decision endpoint. Skipped unless you ask for it.

    DOMYN_DECISIONS_URL=http://localhost:28080 \
    DOMYN_DECISIONS_MODEL=Cloudflare/clef \
    uv run pytest tests/decisions/test_live_contract.py -q --no-cov

Opt-in, because it needs a live endpoint that is not always available: a suite
that fails when that endpoint goes away is a suite people learn to ignore. The
mocked tests pin the same wire format and run everywhere; this one answers the
question they cannot — whether the server still behaves the way they claim.
"""

import os

import pytest
import pytest_asyncio

from domyn.agents.decisions import Provenance, build_provider, choice, noul, score
from domyn.agents.decisions.approvals import VERDICT_QUESTION, build_state

URL = os.environ.get("DOMYN_DECISIONS_URL")
MODEL = os.environ.get("DOMYN_DECISIONS_MODEL", "Cloudflare/clef")

pytestmark = pytest.mark.skipif(
    not URL, reason="set DOMYN_DECISIONS_URL to run the live contract tests"
)


@pytest_asyncio.fixture
async def provider():
    built = build_provider(
        {"type": "system_one", "base_url": URL, "model": MODEL, "timeout_s": 30.0}
    )
    yield built
    await built.aclose()


class TestTheServerStillAnswersTheWayWeParse:
    @pytest.mark.asyncio
    async def test_all_three_question_types_in_one_request(self, provider):
        decision = await provider.decide(
            {"command": "rm -rf /"},
            {
                "danger": noul("Is this destructive?"),
                "verdict": choice(
                    "What now?", {"allow": "run", "ask": "confirm", "deny": "refuse"}
                ),
                "risk": score("How risky?", ["none", "some", "lots"]),
            },
        )

        assert decision.get("danger").known
        assert 0.0 <= float(decision.get("danger").value) <= 1.0
        assert decision.get("verdict").value in {"allow", "ask", "deny"}
        assert decision.get("verdict").probabilities
        assert decision.get("risk").known
        assert decision.get("verdict").provenance is Provenance.NATIVE

    @pytest.mark.asyncio
    async def test_it_agrees_with_us_about_a_dangerous_command(self, provider):
        """Not an accuracy test — one case, to catch a model swapped behind the
        same URL. `rm -rf /` is as unambiguous as this gets."""
        decision = await provider.decide(build_state_for("rm -rf /"), {"verdict": VERDICT_QUESTION})
        assert decision.get("verdict").value == "deny"

    @pytest.mark.asyncio
    async def test_and_about_a_harmless_one(self, provider):
        decision = await provider.decide(
            build_state_for("git status"), {"verdict": VERDICT_QUESTION}
        )
        assert decision.get("verdict").value == "allow"


class TestTheFailuresWeDesignedFor:
    @pytest.mark.asyncio
    async def test_a_wrong_model_name_answers_nothing_rather_than_raising(self):
        """`clef` is refused by a server serving `Cloudflare/clef`. It is the
        first mistake everyone makes, so it must degrade, not crash."""
        wrong = build_provider(
            {"type": "system_one", "base_url": URL, "model": "clef", "timeout_s": 30.0}
        )
        decision = await wrong.decide({"x": 1}, {"q": noul("true?")})
        assert decision.known is False
        await wrong.aclose()

    @pytest.mark.asyncio
    async def test_an_oversized_state_is_refused_by_us_not_truncated_by_them(self):
        """Measured: 200k and 400k characters both return 200 OK with
        `input_tokens: 73728`. The server truncates silently; we must not let
        it, or we get a confident answer about a document half seen."""
        guarded = build_provider(
            {
                "type": "system_one",
                "base_url": URL,
                "model": MODEL,
                "timeout_s": 30.0,
                "max_state_tokens": 73728,
            }
        )
        decision = await guarded.decide("x" * 400_000, {"q": noul("true?")})
        assert decision.known is False
        assert decision.get("q").reason == "state longer than the model's context"
        await guarded.aclose()

    @pytest.mark.asyncio
    async def test_it_is_fast_enough_to_sit_in_front_of_a_tool_call(self, provider):
        """The entire case for this feature. Measured at 117-162 ms; a second
        would mean the design needs revisiting, so the bar is deliberately
        loose and only catches a change of kind."""
        decision = await provider.decide(
            build_state_for("git status"), {"verdict": VERDICT_QUESTION}
        )
        assert decision.latency_ms < 2000


def build_state_for(command: str) -> dict:
    from domyn.agents.approvals import ApprovalRequest

    return build_state(
        ApprovalRequest(
            tool_name="run_command",
            subject=command,
            parts=[command],
            agent_name="Assistant",
        )
    )
