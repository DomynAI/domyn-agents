"""A decision model at the approval gate.

The tests that matter here are not the happy path: they are the ones showing
what the model *cannot* do. A hardline rule, a deny rule and a standing grant
all outrank it, and in shadow mode it changes nothing at all.
"""

import pytest

from domyn.agents.approvals import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalVerdict,
    InMemoryApprovalStore,
)
from domyn.agents.decisions import (
    Answer,
    Decision,
    DecisionMode,
    DecisionProvider,
    DecisionRuntime,
    Provenance,
    UseCasePolicy,
)
from domyn.agents.decisions.approvals import build_state, consult, event_metadata


class Says(DecisionProvider):
    """A backend that always answers the same thing."""

    type_name = "says"

    def __init__(self, value: str, confidence: float, provenance=Provenance.NATIVE, **kwargs):
        super().__init__(**kwargs)
        self._value = value
        self._confidence = confidence
        self._provenance = provenance
        self.states: list[str] = []

    @property
    def provenance(self):
        return self._provenance

    async def _decide(self, state, questions):
        self.states.append(state)
        return Decision(
            answers={
                name: Answer(
                    known=True,
                    provenance=self._provenance,
                    value=self._value,
                    confidence=self._confidence,
                )
                for name in questions
            },
            latency_ms=147,
        )


def request(subject: str = "rm -rf build/", parts=None) -> ApprovalRequest:
    return ApprovalRequest(
        tool_name="run_command",
        subject=subject,
        parts=parts,
        agent_name="Assistant",
        user_id="u1",
        conversation_id="c1",
    )


def runtime(value: str, confidence: float, mode=DecisionMode.ACTIVE, **kwargs) -> DecisionRuntime:
    return DecisionRuntime(
        Says(value, confidence, **kwargs), {"approvals": UseCasePolicy(mode, 0.8)}
    )


class TestTheDirectionRule:
    """Allowing removes a human confirmation; denying adds friction. They do
    not get the same bar, and that asymmetry is the point."""

    @pytest.mark.asyncio
    async def test_a_confident_allow_removes_the_question(self):
        result = await consult(runtime("allow", 0.95), request("ls -la"))
        assert result.verdict is not None
        assert result.verdict.decision is ApprovalDecision.ALLOW
        assert result.acted

    @pytest.mark.asyncio
    async def test_the_measured_mistake_does_not(self):
        """`rm -rf build/` was answered `allow` with 0.53 — the one error in the
        13-command eval. The threshold is what turns it into a question."""
        result = await consult(runtime("allow", 0.53), request())
        assert result.verdict is None
        assert result.acted is False

    @pytest.mark.asyncio
    async def test_an_llm_cannot_allow_however_sure_it_sounds(self):
        """Measured: a stated confidence put wrong answers at 0.78-0.85. The
        provenance check, not the threshold, is what stops this."""
        result = await consult(runtime("allow", 0.99, provenance=Provenance.STATED), request())
        assert result.verdict is None

    @pytest.mark.asyncio
    async def test_but_a_stated_backend_may_still_deny(self):
        result = await consult(runtime("deny", 0.9, provenance=Provenance.STATED), request())
        assert result.verdict is not None
        assert result.verdict.decision is ApprovalDecision.DENY

    @pytest.mark.asyncio
    async def test_an_unsure_deny_is_left_alone(self):
        assert (await consult(runtime("deny", 0.4), request())).verdict is None

    @pytest.mark.asyncio
    async def test_ask_produces_no_verdict_at_all(self):
        # Same outcome as an abstention, but distinguishable in the logs.
        result = await consult(runtime("ask", 0.95), request())
        assert result.verdict is None
        assert result.answer.value == "ask"


class TestShadowChangesNothing:
    @pytest.mark.asyncio
    async def test_a_confident_allow_is_recorded_and_ignored(self):
        result = await consult(runtime("allow", 0.99, mode=DecisionMode.SHADOW), request("ls"))
        assert result.verdict is None
        assert result.acted is False
        assert result.answer.known
        assert result.mode is DecisionMode.SHADOW

    @pytest.mark.asyncio
    async def test_it_still_asks_so_agreement_can_be_measured(self):
        rt = runtime("allow", 0.99, mode=DecisionMode.SHADOW)
        await consult(rt, request())
        assert rt.provider.states


class TestWithNoModelNothingHappens:
    @pytest.mark.asyncio
    async def test_no_runtime_is_silent(self):
        result = await consult(None, request())
        assert result.verdict is None
        assert result.mode is DecisionMode.OFF

    @pytest.mark.asyncio
    async def test_an_off_use_case_never_builds_a_state(self):
        rt = runtime("allow", 0.99, mode=DecisionMode.OFF)
        await consult(rt, request())
        assert rt.provider.states == []

    @pytest.mark.asyncio
    async def test_a_broken_backend_leaves_the_gate_alone(self):
        class Broken(DecisionProvider):
            type_name = "broken"
            provenance = Provenance.NATIVE

            async def _decide(self, state, questions):
                raise RuntimeError("down")

        rt = DecisionRuntime(Broken(), {"approvals": UseCasePolicy(DecisionMode.ACTIVE, 0.8)})
        result = await consult(rt, request())
        assert result.verdict is None
        assert result.answer.known is False


class TestTheModelCannotOutrankAnything:
    """The gate's order is the security model. These pin that a verdict from a
    model is consulted in the one position where it cannot do harm."""

    @pytest.mark.asyncio
    async def test_a_hardline_rule_beats_a_confident_allow(self):
        policy = ApprovalPolicy()
        verdict = await policy.resolve(
            request(subject="rm -rf /"),
            hook_verdicts=[ApprovalVerdict.allow(reason="model said so", rule="decision:allow")],
        )
        assert verdict.decision is ApprovalDecision.DENY
        assert (verdict.rule or "").startswith("hardline:")

    @pytest.mark.asyncio
    async def test_a_deny_rule_beats_a_confident_allow(self):
        # A subject only the deny rule catches: a hardline pattern would make
        # this test pass for the wrong reason.
        policy = ApprovalPolicy(deny_rules=["*npm publish*"])
        verdict = await policy.resolve(
            request(subject="npm publish --access public"),
            hook_verdicts=[ApprovalVerdict.allow(rule="decision:allow")],
        )
        assert verdict.decision is ApprovalDecision.DENY
        assert (verdict.rule or "").startswith("deny:")

    @pytest.mark.asyncio
    async def test_a_standing_grant_is_never_even_shown_to_the_model(self):
        """A grant returns before the hook verdicts are read, so a model's DENY
        cannot revoke a permission the user gave."""
        store = InMemoryApprovalStore()
        req = request(subject="git status")
        await store.grant(req, None)
        policy = ApprovalPolicy(stores=[store])
        verdict = await policy.resolve(
            req, hook_verdicts=[ApprovalVerdict.deny(rule="decision:deny")]
        )
        assert verdict.decision is ApprovalDecision.ALLOW
        assert (verdict.rule or "").startswith("grant:")

    @pytest.mark.asyncio
    async def test_with_nothing_in_the_way_the_model_decides(self):
        policy = ApprovalPolicy()
        verdict = await policy.resolve(
            request(subject="ls -la"),
            hook_verdicts=[ApprovalVerdict.allow(rule="decision:allow:0.95")],
        )
        assert verdict.decision is ApprovalDecision.ALLOW
        assert verdict.rule == "decision:allow:0.95"

    @pytest.mark.asyncio
    async def test_without_a_model_the_default_is_still_ask(self):
        verdict = await ApprovalPolicy().resolve(request(), hook_verdicts=[])
        assert verdict.decision is ApprovalDecision.ASK


class TestTheStateSentToTheModel:
    @pytest.mark.asyncio
    async def test_it_judges_the_same_subject_the_gate_judges(self):
        # Judging anything else would mean the model answers about a different
        # string than the one being approved and remembered.
        state = build_state(request(subject="rm -rf build/", parts=["rm -rf build/"]))
        assert state["command"] == "rm -rf build/"
        assert state["parts"] == ["rm -rf build/"]
        assert state["tool"] == "run_command"

    @pytest.mark.asyncio
    async def test_parts_are_left_out_when_there_are_none(self):
        assert "parts" not in build_state(request(parts=None))

    @pytest.mark.asyncio
    async def test_one_question_is_asked_alone(self):
        """Measured: four supporting `noul` questions beside the verdict flip
        `rm -rf build/` from ask (0.59) to allow (0.53). They share one forward
        pass, so a supporting question is not free."""
        rt = runtime("ask", 0.9)
        result = await consult(rt, request())
        assert len(result.decision.answers) == 1


class TestTheEventPayload:
    def test_it_carries_the_latency_and_not_the_command(self):
        async def run():
            return await consult(runtime("allow", 0.95), request(subject="cat ~/.ssh/id_rsa"))

        import asyncio

        result = asyncio.run(run())
        metadata = event_metadata("approvals", result)
        assert metadata["latency_ms"] == 147
        assert metadata["confidence"] == 0.95
        assert metadata["provenance"] == "native"
        assert metadata["acted"] is True
        # The command is customer content and this event is emitted at info.
        assert "id_rsa" not in str(metadata)
