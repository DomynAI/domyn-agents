"""Choosing skills with a decision model.

A selector may only decide what to *expand*; the index stays complete whatever
happens. That is what makes this the cheapest use case to get wrong: a mistake
costs one unopened file, and the agent can still reach everything through
`skill_view`.
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
from domyn.agents.skills import SkillDescriptor, SkillIndex, SkillQuery
from domyn.agents.skills.selectors import DecisionSelector

SKILLS = [
    ("presentations", "How to build a deck."),
    ("sql-review", "Checklist for reviewing SQL migrations."),
    ("charts", "House style for charts."),
]


def index(available: set[str] | None = None) -> SkillIndex:
    return SkillIndex(
        skills=[
            SkillDescriptor(
                name=name,
                description=description,
                available=available is None or name in available,
            )
            for name, description in SKILLS
        ]
    )


class Scores(DecisionProvider):
    """Answers each question with a probability taken from a list."""

    type_name = "scores"
    provenance = Provenance.NATIVE

    def __init__(self, probabilities: list[float], **kwargs):
        super().__init__(**kwargs)
        self._probabilities = probabilities
        self.questions: dict = {}

    async def _decide(self, state, questions):
        self.questions = questions
        answers = {}
        for position, name in enumerate(questions):
            probability = (
                self._probabilities[position] if position < len(self._probabilities) else 0.0
            )
            answers[name] = Answer(
                known=True,
                provenance=Provenance.NATIVE,
                value=probability,
                confidence=max(probability, 1.0 - probability),
            )
        return Decision(answers=answers, latency_ms=200)


def selector(probabilities: list[float], mode=DecisionMode.ACTIVE, **kwargs) -> DecisionSelector:
    built = DecisionSelector(**kwargs)
    built.bind_decisions(
        DecisionRuntime(Scores(probabilities), {"skills": UseCasePolicy(mode, 0.5)})
    )
    return built


class TestItExpandsWhatTheModelPicked:
    @pytest.mark.asyncio
    async def test_a_confident_yes_is_expanded(self):
        selection = await selector([0.95, 0.02, 0.01]).select(
            index(), SkillQuery(task="build me a deck")
        )
        assert [ref.name for ref in selection.expand] == ["presentations"]

    @pytest.mark.asyncio
    async def test_nothing_relevant_expands_nothing(self):
        """The common case: most turns need no skill, and the prompt says so."""
        selection = await selector([0.04, 0.02, 0.03]).select(
            index(), SkillQuery(task="what is the capital of France?")
        )
        assert selection.expand == []

    @pytest.mark.asyncio
    async def test_the_most_confident_come_first(self):
        selection = await selector([0.7, 0.99, 0.8], max_expansions=2).select(
            index(), SkillQuery(task="anything")
        )
        assert [ref.name for ref in selection.expand] == ["sql-review", "charts"]

    @pytest.mark.asyncio
    async def test_max_expansions_caps_it(self):
        selection = await selector([0.9, 0.9, 0.9], max_expansions=1).select(
            index(), SkillQuery(task="anything")
        )
        assert len(selection.expand) == 1


class TestTheIndexIsNeverTrimmed:
    """A selector that hid what it did not pick would turn its own mistakes
    into dead ends."""

    @pytest.mark.asyncio
    async def test_everything_stays_visible(self):
        selection = await selector([0.95, 0.0, 0.0]).select(index(), SkillQuery(task="deck"))
        assert len(selection.index) == len(SKILLS)

    @pytest.mark.asyncio
    async def test_even_when_the_model_says_nothing(self):
        class Silent(DecisionProvider):
            type_name = "silent"
            provenance = Provenance.NATIVE

            async def _decide(self, state, questions):
                raise RuntimeError("down")

        built = DecisionSelector()
        built.bind_decisions(
            DecisionRuntime(Silent(), {"skills": UseCasePolicy(DecisionMode.ACTIVE, 0.5)})
        )
        selection = await built.select(index(), SkillQuery(task="deck"))
        assert len(selection.index) == len(SKILLS)
        assert selection.expand == []


class TestItDegradesToTheIndex:
    @pytest.mark.asyncio
    async def test_with_no_runtime_bound(self):
        selection = await DecisionSelector().select(index(), SkillQuery(task="deck"))
        assert selection.expand == []
        assert "no decision model" in selection.rationale

    @pytest.mark.asyncio
    async def test_with_the_use_case_off(self):
        selection = await selector([0.99], mode=DecisionMode.OFF).select(
            index(), SkillQuery(task="deck")
        )
        assert selection.expand == []

    @pytest.mark.asyncio
    async def test_in_shadow_it_asks_but_expands_nothing(self):
        built = selector([0.99, 0.99, 0.99], mode=DecisionMode.SHADOW)
        selection = await built.select(index(), SkillQuery(task="deck"))
        assert selection.expand == []
        assert built._decisions.provider.questions  # it did ask

    @pytest.mark.asyncio
    async def test_with_no_task_it_does_not_ask(self):
        built = selector([0.99])
        await built.select(index(), SkillQuery(task=None))
        assert built._decisions.provider.questions == {}


class TestWhatItAsksAbout:
    @pytest.mark.asyncio
    async def test_one_question_per_available_skill(self):
        """Homogeneous and independent, so batching them is right here — unlike
        the approval verdict, which has to be asked alone."""
        built = selector([0.1, 0.1, 0.1])
        await built.select(index(), SkillQuery(task="anything"))
        assert len(built._decisions.provider.questions) == len(SKILLS)

    @pytest.mark.asyncio
    async def test_unavailable_skills_are_not_asked_about(self):
        built = selector([0.9, 0.9])
        await built.select(index(available={"charts"}), SkillQuery(task="anything"))
        assert len(built._decisions.provider.questions) == 1

    @pytest.mark.asyncio
    async def test_a_catalogue_over_the_protocol_limit_is_truncated_not_refused(self):
        big = SkillIndex(skills=[SkillDescriptor(name=f"s{i}", description="x") for i in range(80)])
        built = selector([0.0] * 80, max_questions=64)
        selection = await built.select(big, SkillQuery(task="anything"))
        assert len(built._decisions.provider.questions) == 64
        # The index is still whole, so the tail is reachable as it is today.
        assert len(selection.index) == 80

    @pytest.mark.asyncio
    async def test_the_task_is_the_state(self):
        built = selector([0.1, 0.1, 0.1])
        await built.select(index(), SkillQuery(task="review my SQL migration"))
        assert built._decisions.provider.questions


class TestTheThreshold:
    @pytest.mark.asyncio
    async def test_below_it_nothing_is_expanded(self):
        selection = await selector([0.45, 0.4, 0.3]).select(index(), SkillQuery(task="anything"))
        assert selection.expand == []

    @pytest.mark.asyncio
    async def test_the_selector_may_override_it(self):
        selection = await selector([0.45, 0.4, 0.3], min_confidence=0.4).select(
            index(), SkillQuery(task="anything")
        )
        assert {ref.name for ref in selection.expand} == {"presentations", "sql-review"}
