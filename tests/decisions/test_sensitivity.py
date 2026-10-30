"""The data boundary: which models a turn's data may reach.

This use case deliberately breaks the rule every other one follows. Elsewhere
"the model could not answer" degrades to today's behaviour, because today's
behaviour is safe. Here today's behaviour is *send it anyway*, which is the
thing being prevented — so the default on an unknown rating is to refuse.
"""

import pytest

from domyn.agents.decisions import (
    Answer,
    Decision,
    DecisionMode,
    DecisionProvider,
    DecisionRuntime,
    Provenance,
    UseCasePolicy,
)
from domyn.agents.decisions.sensitivity import (
    Rating,
    Sensitivity,
    allows,
    provider_ceiling,
    rate,
)


class Provider:
    """Only what the gate reads off a provider."""

    def __init__(self, max_sensitivity=None):
        self.max_sensitivity = max_sensitivity


class Rates(DecisionProvider):
    type_name = "rates"
    provenance = Provenance.NATIVE

    def __init__(self, level: float, confidence: float = 0.95, **kwargs):
        super().__init__(**kwargs)
        self._level = level
        self._confidence = confidence
        self.asked = 0

    async def _decide(self, state, questions):
        self.asked += 1
        return Decision(
            answers={
                name: Answer(
                    known=True,
                    provenance=Provenance.NATIVE,
                    value=self._level,
                    confidence=self._confidence,
                )
                for name in questions
            }
        )


def runtime(level: float, confidence: float = 0.95, mode=DecisionMode.ACTIVE) -> DecisionRuntime:
    return DecisionRuntime(Rates(level, confidence), {"sensitivity": UseCasePolicy(mode, 0.7)})


class TestReadingTheConfiguredCeiling:
    def test_a_name_is_accepted(self):
        assert provider_ceiling(Provider("internal")) is Sensitivity.INTERNAL
        assert provider_ceiling(Provider("REGULATED")) is Sensitivity.REGULATED

    def test_no_ceiling_means_no_check(self):
        assert provider_ceiling(Provider()) is None
        assert provider_ceiling(Provider("nonsense")) is None

    def test_a_bool_is_not_a_level(self):
        # `True` is an int in Python, and `max_sensitivity: yes` in YAML is a bool.
        assert provider_ceiling(Provider(True)) is None


class TestRating:
    @pytest.mark.asyncio
    async def test_a_clear_rating_comes_back(self):
        rating = await rate(runtime(0.1), task="fix a typo in the README")
        assert rating is not None
        assert rating.known
        assert rating.level is Sensitivity.INTERNAL

    @pytest.mark.asyncio
    async def test_a_fractional_score_rounds_up(self):
        """The score is a weighted mean over the levels. A turn sitting between
        internal and confidential is treated as confidential — the only safe
        direction to round in."""
        rating = await rate(runtime(1.2), task="x")
        assert rating is not None
        assert rating.level is Sensitivity.CONFIDENTIAL

    @pytest.mark.asyncio
    async def test_the_top_level_does_not_overflow(self):
        rating = await rate(runtime(3.0), task="x")
        assert rating is not None
        assert rating.level is Sensitivity.REGULATED

    @pytest.mark.asyncio
    async def test_off_means_no_rating_at_all(self):
        assert await rate(runtime(3.0, mode=DecisionMode.OFF), task="x") is None
        assert await rate(None, task="x") is None

    @pytest.mark.asyncio
    async def test_an_unsure_rating_is_not_a_rating(self):
        rating = await rate(runtime(0.0, confidence=0.3), task="x")
        assert rating is not None
        assert rating.known is False

    @pytest.mark.asyncio
    async def test_no_task_is_unknown_not_public(self):
        rating = await rate(runtime(0.0), task="")
        assert rating is not None
        assert rating.known is False


class TestTheGate:
    def test_a_turn_within_the_ceiling_runs(self):
        allowed, reason = allows(Provider("confidential"), Rating(Sensitivity.INTERNAL, known=True))
        assert allowed is True
        assert reason is None

    def test_a_turn_at_the_ceiling_runs(self):
        allowed, _ = allows(Provider("internal"), Rating(Sensitivity.INTERNAL, known=True))
        assert allowed is True

    def test_a_turn_above_it_is_refused_and_says_why(self):
        allowed, reason = allows(Provider("internal"), Rating(Sensitivity.REGULATED, known=True))
        assert allowed is False
        assert "regulated" in (reason or "")
        assert "internal" in (reason or "")

    def test_a_provider_with_no_ceiling_is_never_checked(self):
        allowed, _ = allows(Provider(), Rating(Sensitivity.REGULATED, known=True))
        assert allowed is True

    def test_no_rating_taken_is_not_a_refusal(self):
        """The use case is off, so the question never arose."""
        allowed, _ = allows(Provider("public"), None)
        assert allowed is True


class TestTheGuardrailFailsClosed:
    def test_an_unknown_rating_blocks_by_default(self):
        allowed, reason = allows(Provider("internal"), Rating(Sensitivity.REGULATED, known=False))
        assert allowed is False
        assert "could not be completed" in (reason or "")

    def test_but_an_operator_can_choose_otherwise(self):
        allowed, _ = allows(
            Provider("internal"),
            Rating(Sensitivity.REGULATED, known=False),
            on_unknown="allow",
        )
        assert allowed is True

    def test_the_choice_is_only_about_the_unknown_case(self):
        """`on_unknown: allow` must not also wave through a turn that was rated
        and rated too high."""
        allowed, _ = allows(
            Provider("internal"),
            Rating(Sensitivity.REGULATED, known=True),
            on_unknown="allow",
        )
        assert allowed is False


class TestTheRunnerRefusesTheTurn:
    @pytest.mark.asyncio
    async def test_a_turn_over_the_ceiling_produces_an_error_and_nothing_else(self):
        from domyn.agents import Agent, BaseEvent, ExecutionEventType, Part, Runner
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

        agent = Agent(
            name="A",
            description="d",
            instruction="i",
            llm_provider=Mock(max_sensitivity="public"),
            decisions=runtime(3.0),
        )
        events = [
            event
            async for event in Runner().run(
                conversation_id="sensitivity-refused",
                application_name="t",
                user_id="u",
                root=agent,
                user_input=BaseEvent(
                    event_type=ExecutionEventType.USER_INPUT,
                    author="user",
                    content=[Part(text="here is my AWS secret key")],
                    turn_id="turn_1",
                ),
            )
        ]
        errors = [e for e in events if e.event_type is ExecutionEventType.ERROR_OCCURRED]
        assert errors and errors[0].error_code == "SensitivityPolicy"
        assert ExecutionEventType.RESPONSE not in [e.event_type for e in events]

    @pytest.mark.asyncio
    async def test_a_provider_with_no_ceiling_runs_as_always(self):
        from domyn.agents import Agent, BaseEvent, ExecutionEventType, Part, Runner
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

        built = runtime(3.0)
        agent = Agent(
            name="A", description="d", instruction="i", llm_provider=Mock(), decisions=built
        )
        events = [
            event
            async for event in Runner().run(
                conversation_id="sensitivity-absent",
                application_name="t",
                user_id="u",
                root=agent,
                user_input=BaseEvent(
                    event_type=ExecutionEventType.USER_INPUT,
                    author="user",
                    content=[Part(text="hello")],
                    turn_id="turn_1",
                ),
            )
        ]
        # The mock provider fails on its own further down; what matters is
        # that the sensitivity gate let the turn start at all.
        codes = [getattr(e, "error_code", None) for e in events]
        assert "SensitivityPolicy" not in codes
        # Never even asked: no ceiling, no question.
        assert built.provider.asked == 0


class TestOnUnknownIsConfiguration:
    """The documented knob has to be a real one: a deployment that prefers
    availability over the boundary must be able to say so in the file, where a
    reviewer can see it."""

    def test_it_defaults_to_blocking(self):
        from domyn.agents.decisions.config import DecisionsConfig

        config = DecisionsConfig.from_dict({"provider": "p", "sensitivity": {"mode": "active"}})
        assert config.policies()["sensitivity"].on_unknown == "block"

    def test_an_operator_can_choose_allow(self):
        from domyn.agents.decisions.config import DecisionsConfig

        config = DecisionsConfig.from_dict(
            {"provider": "p", "sensitivity": {"mode": "active", "on_unknown": "allow"}}
        )
        assert config.policies()["sensitivity"].on_unknown == "allow"

    def test_anything_else_is_refused_at_load_time(self):
        from domyn.agents.decisions.config import UseCaseConfig

        with pytest.raises(ValueError):
            UseCaseConfig(mode="active", on_unknown="maybe")

    def test_the_runtime_reports_it(self):
        from domyn.agents.decisions import UseCasePolicy

        built = DecisionRuntime(
            Rates(0.0),
            {"sensitivity": UseCasePolicy(DecisionMode.ACTIVE, 0.7, on_unknown="allow")},
        )
        assert built.on_unknown("sensitivity") == "allow"
        assert built.on_unknown("approvals") == "block"

    @pytest.mark.asyncio
    async def test_it_reaches_the_runner(self):
        from domyn.agents import Agent, BaseEvent, ExecutionEventType, Part, Runner
        from domyn.agents.decisions import DecisionProvider, UseCasePolicy
        from domyn.agents.llm import BaseLLMProvider

        class Silent(DecisionProvider):
            """On, reachable, and unable to rate anything."""

            type_name = "silent"
            provenance = Provenance.NATIVE

            async def _decide(self, state, questions):
                raise RuntimeError("down")

        class Mock(BaseLLMProvider):
            type: str = "mock"
            name: str = "m"
            model_name: str = "m"
            supported_structured_outputs: None = None

            async def generate(self, messages, **kwargs):
                pass

            async def generate_streaming(self, messages, **kwargs):
                yield

        async def codes(on_unknown: str) -> list[str | None]:
            agent = Agent(
                name="A",
                description="d",
                instruction="i",
                llm_provider=Mock(max_sensitivity="public"),
                decisions=DecisionRuntime(
                    Silent(),
                    {"sensitivity": UseCasePolicy(DecisionMode.ACTIVE, 0.7, on_unknown=on_unknown)},
                ),
            )
            events = [
                event
                async for event in Runner().run(
                    conversation_id=f"unknown-{on_unknown}",
                    application_name="t",
                    user_id="u",
                    root=agent,
                    user_input=BaseEvent(
                        event_type=ExecutionEventType.USER_INPUT,
                        author="user",
                        content=[Part(text="something")],
                        turn_id="turn_1",
                    ),
                )
            ]
            return [getattr(e, "error_code", None) for e in events]

        assert "SensitivityPolicy" in await codes("block")
        assert "SensitivityPolicy" not in await codes("allow")
