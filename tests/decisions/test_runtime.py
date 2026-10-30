"""The rules about when an answer is allowed to matter.

These are the safety properties. The provider can be wrong; the runtime is what
decides whether being wrong costs anything.
"""

import pytest

from domyn.agents.decisions import (
    USE_CASES,
    Answer,
    Decision,
    DecisionMode,
    DecisionProvider,
    DecisionRuntime,
    NoDecisions,
    Provenance,
    UseCasePolicy,
    build_provider,
    noul,
)


class Fake(DecisionProvider):
    type_name = "fake"

    def __init__(self, provenance=Provenance.NATIVE, confidence=0.95, **kwargs):
        super().__init__(**kwargs)
        self._provenance = provenance
        self._confidence = confidence
        self.calls = 0

    @property
    def provenance(self):
        return self._provenance

    async def _decide(self, state, questions):
        self.calls += 1
        return Decision(
            answers={
                name: Answer(
                    known=True,
                    provenance=self._provenance,
                    value="allow",
                    confidence=self._confidence,
                )
                for name in questions
            }
        )


QUESTION = {"q": noul("true?")}


def runtime(mode=DecisionMode.ACTIVE, threshold=0.8, **kwargs) -> DecisionRuntime:
    return DecisionRuntime(Fake(**kwargs), {"approvals": UseCasePolicy(mode, threshold)})


class TestEverythingIsOffUntilSomeoneSaysOtherwise:
    def test_a_runtime_with_no_policies_is_off(self):
        assert DecisionRuntime(Fake()).mode("approvals") is DecisionMode.OFF

    def test_an_unconfigured_use_case_is_off_even_with_others_on(self):
        assert runtime().mode("skills") is DecisionMode.OFF

    def test_no_provider_means_off_whatever_the_policy_says(self):
        rt = DecisionRuntime(NoDecisions(), {"approvals": UseCasePolicy(DecisionMode.ACTIVE, 0.8)})
        assert rt.mode("approvals") is DecisionMode.OFF
        assert rt.enabled("approvals") is False

    @pytest.mark.asyncio
    async def test_an_off_use_case_does_not_call_the_provider(self):
        provider = Fake()
        rt = DecisionRuntime(provider, {})
        decision = await rt.ask("approvals", {"x": 1}, QUESTION)
        assert provider.calls == 0
        assert decision.known is False


class TestShadowAsksAndChangesNothing:
    @pytest.mark.asyncio
    async def test_it_calls_the_provider(self):
        provider = Fake()
        rt = DecisionRuntime(provider, {"approvals": UseCasePolicy(DecisionMode.SHADOW, 0.8)})
        await rt.ask("approvals", {"x": 1}, QUESTION)
        assert provider.calls == 1

    def test_but_never_authorises_acting(self):
        rt = DecisionRuntime(Fake(), {"approvals": UseCasePolicy(DecisionMode.SHADOW, 0.8)})
        confident = Answer(known=True, provenance=Provenance.NATIVE, value="allow", confidence=0.99)
        assert rt.may_act("approvals", confident) is False


class TestAUseCaseDeclaresTheQualityOfNumberItNeeds:
    """The minimum lives in the code, not the config: it is a safety property,
    not a preference, and approvals need it because acting means removing a
    human confirmation."""

    def test_a_stated_backend_can_never_remove_a_confirmation(self):
        """However sure it sounds. Measured: wrong answers at 0.78-0.85."""
        rt = runtime(provenance=Provenance.STATED)
        sure = Answer(known=True, provenance=Provenance.STATED, value="allow", confidence=0.99)
        assert rt.may_act("approvals", sure) is False

    def test_but_it_may_still_add_friction(self):
        """Approvals have two directions and they do not need the same
        evidence: denying costs an override, allowing costs a review."""
        rt = runtime(provenance=Provenance.STATED)
        assert rt.mode("approvals") is DecisionMode.ACTIVE
        sure = Answer(known=True, provenance=Provenance.STATED, value="deny", confidence=0.9)
        assert rt.may_act_cautiously("approvals", sure) is True

    def test_a_use_case_with_no_cautious_direction_downgrades_to_shadow(self):
        """The ladder can only hide, so a backend that cannot meet its bar has
        nothing safe left to do and stays in shadow."""
        rt = DecisionRuntime(
            Fake(provenance=Provenance.STATED),
            {"context_ladder": UseCasePolicy(DecisionMode.ACTIVE, 0.9)},
        )
        assert rt.mode("context_ladder") is DecisionMode.SHADOW
        assert (
            rt.may_act_cautiously(
                "context_ladder",
                Answer(known=True, provenance=Provenance.STATED, value="hide", confidence=0.99),
            )
            is False
        )

    def test_it_warns_once_not_every_turn(self, caplog):
        rt = DecisionRuntime(
            Fake(provenance=Provenance.STATED),
            {"context_ladder": UseCasePolicy(DecisionMode.ACTIVE, 0.9)},
        )
        for _ in range(5):
            rt.mode("context_ladder")
        warnings = [r for r in caplog.records if "staying in shadow" in r.getMessage()]
        assert len(warnings) <= 1

    def test_skills_are_happy_with_a_stated_backend(self):
        rt = DecisionRuntime(
            Fake(provenance=Provenance.STATED),
            {"skills": UseCasePolicy(DecisionMode.ACTIVE, 0.5)},
        )
        assert rt.mode("skills") is DecisionMode.ACTIVE

    def test_a_native_backend_satisfies_every_use_case(self):
        for name in USE_CASES:
            rt = DecisionRuntime(Fake(), {name: UseCasePolicy(DecisionMode.ACTIVE, 0.5)})
            assert rt.mode(name) is DecisionMode.ACTIVE

    def test_every_use_case_explains_its_minimum(self):
        # A future reader lowering one of these should have to read why.
        assert all(spec.why for spec in USE_CASES.values())


class TestTheThreshold:
    def test_below_it_nothing_acts(self):
        rt = runtime(threshold=0.8)
        assert (
            rt.may_act(
                "approvals",
                Answer(known=True, provenance=Provenance.NATIVE, value="allow", confidence=0.53),
            )
            is False
        )

    def test_at_it_the_answer_acts(self):
        rt = runtime(threshold=0.8)
        assert (
            rt.may_act(
                "approvals",
                Answer(known=True, provenance=Provenance.NATIVE, value="allow", confidence=0.8),
            )
            is True
        )

    def test_the_measured_mistake_is_rejected_by_the_default(self):
        """The 13-command eval: the one wrong verdict carried 0.53, every right
        one 0.83-0.98. This is that result, pinned."""
        rt = DecisionRuntime(Fake(), {"approvals": UseCasePolicy(DecisionMode.ACTIVE, 0.8)})
        wrong = Answer(known=True, provenance=Provenance.NATIVE, value="allow", confidence=0.53)
        right = Answer(known=True, provenance=Provenance.NATIVE, value="deny", confidence=0.83)
        assert rt.may_act("approvals", wrong) is False
        assert rt.may_act("approvals", right) is True

    def test_an_unconfigured_use_case_falls_back_to_its_default(self):
        assert DecisionRuntime(Fake()).threshold("approvals") == 0.8

    def test_an_unknown_use_case_can_never_act(self):
        """A typo in a config file must not inherit the threshold written next
        to it: that is an unreviewed use case running active on a number nobody
        chose for it."""
        rt = DecisionRuntime(Fake(), {"invented": UseCasePolicy(DecisionMode.ACTIVE, 0.1)})
        assert rt.mode("invented") is DecisionMode.OFF
        assert rt.enabled("invented") is False
        assert rt.threshold("invented") == 1.0
        assert (
            rt.may_act(
                "invented",
                Answer(known=True, provenance=Provenance.NATIVE, value="x", confidence=0.99),
            )
            is False
        )


class TestTheFactory:
    def test_an_unknown_type_raises_at_build_time(self):
        # House rule: a typo in a type name is a configuration error, like
        # LLMProviderFactory and SkillsRuntime. Degrading would hand someone a
        # silently disabled feature they believe they configured.
        with pytest.raises(ValueError, match="Unknown decision provider type"):
            build_provider({"type": "clef-ish"})

    def test_no_config_is_no_decisions(self):
        assert build_provider(None).acts is False
        assert build_provider({}).acts is False

    def test_it_builds_a_system_one_provider(self):
        provider = build_provider(
            {"type": "system_one", "base_url": "http://x", "model": "Cloudflare/clef"}
        )
        assert provider.provenance is Provenance.NATIVE
        assert provider.model == "Cloudflare/clef"

    def test_it_builds_an_llm_provider(self):
        provider = build_provider({"type": "llm", "mode": "stated"})
        assert provider.provenance is Provenance.STATED


class TestAttaching:
    """`attach_decisions` carries the guard against construction order: a
    skills selector is bound to the agent's facilities when the *skills*
    runtime is attached, which may happen before the decision runtime exists."""

    def _agent(self, skills=None):
        from domyn.agents import Agent
        from domyn.agents.llm import BaseLLMProvider

        class Mock(BaseLLMProvider):
            type: str = "mock"
            name: str = "m"
            model_name: str = "m"
            supported_structured_outputs: None = None

            async def generate(self, messages, **kwargs):
                pass

            async def generate_streaming(self, messages, **kwargs):
                yield

        return Agent(
            name="A",
            description="d",
            instruction="i",
            llm_provider=Mock(),
            skills=skills,
        )

    def test_it_sets_the_field(self):
        from domyn.agents.decisions import attach_decisions, runtime_of

        agent = self._agent()
        built = DecisionRuntime(Fake())
        assert runtime_of(agent) is None
        attach_decisions(agent, built)
        assert runtime_of(agent) is built

    def test_it_is_idempotent(self):
        from domyn.agents.decisions import attach_decisions, runtime_of

        agent = self._agent()
        built = DecisionRuntime(Fake())
        attach_decisions(attach_decisions(agent, built), built)
        assert runtime_of(agent) is built

    def test_an_agent_without_the_field_is_untouched(self):
        from domyn.agents.decisions import attach_decisions, runtime_of

        class Foreign:
            pass

        agent = Foreign()
        assert attach_decisions(agent, DecisionRuntime(Fake())) is agent
        assert runtime_of(agent) is None

    def test_it_rebinds_a_selector_attached_earlier(self):
        """The bug this exists for: a `decision` selector built before the
        runtime would silently fall back to the index forever."""
        from domyn.agents.decisions import attach_decisions
        from domyn.agents.skills import InMemorySkillSource, SkillsRuntime
        from domyn.agents.skills.selectors import DecisionSelector

        selector = DecisionSelector()
        agent = self._agent(
            skills=SkillsRuntime(source=InMemorySkillSource(skills={}), selector=selector)
        )
        assert selector._decisions is None

        built = DecisionRuntime(Fake())
        attach_decisions(agent, built)
        assert selector._decisions is built

    def test_runtime_of_ignores_a_field_holding_something_else(self):
        from domyn.agents.decisions import runtime_of

        agent = self._agent()
        agent.decisions = "not a runtime"
        assert runtime_of(agent) is None


class TestClosing:
    @pytest.mark.asyncio
    async def test_closing_the_runtime_closes_its_provider(self):
        closed: list[bool] = []

        class Closes(Fake):
            async def aclose(self):
                closed.append(True)

        await DecisionRuntime(Closes()).aclose()
        assert closed == [True]
