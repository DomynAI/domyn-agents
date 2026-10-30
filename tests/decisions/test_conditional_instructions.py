"""Standing instructions that only apply sometimes.

The direction is the safety property here: dropping a line the user wrote is a
change they did not ask for, so it takes a confident "no, this does not apply".
Everything uncertain keeps the text, which is what happens today.
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
from domyn.agents.instructions import AgentInstructions, Instruction, InstructionKind
from domyn.agents.instructions.conditional import (
    apply,
    resolve,
    split_sections,
)

TEXT = """\
You are a careful engineer.

## Commit messages
when: the task involves writing or amending a git commit

Write the why, not the what.

## SQL migrations
when: the task involves a database migration or a schema change

Always check the rollback path first.

## Tone
Be concise.
"""


class Answers(DecisionProvider):
    type_name = "answers"
    provenance = Provenance.NATIVE

    def __init__(self, probabilities: list[float], **kwargs):
        super().__init__(**kwargs)
        self._probabilities = probabilities
        self.asked = 0

    async def _decide(self, state, questions):
        self.asked += 1
        answers = {}
        for position, name in enumerate(questions):
            probability = (
                self._probabilities[position] if position < len(self._probabilities) else 1.0
            )
            answers[name] = Answer(
                known=True,
                provenance=Provenance.NATIVE,
                value=probability,
                confidence=max(probability, 1.0 - probability),
            )
        return Decision(answers=answers)


def runtime(probabilities: list[float], mode=DecisionMode.ACTIVE) -> DecisionRuntime:
    # 0.8 is the use case's own default, and it governs the *removal*: a
    # section goes only on a confident no.
    return DecisionRuntime(Answers(probabilities), {"instructions": UseCasePolicy(mode, 0.8)})


class TestReadingTheFile:
    def test_a_when_line_is_the_condition_and_leaves_the_body(self):
        _, sections = split_sections(TEXT)
        commit = sections[0]
        assert commit.heading == "Commit messages"
        assert commit.condition == "the task involves writing or amending a git commit"
        assert commit.body == "Write the why, not the what."

    def test_a_section_without_one_always_applies(self):
        _, sections = split_sections(TEXT)
        assert sections[-1].heading == "Tone"
        assert sections[-1].conditional is False

    def test_the_preamble_is_kept_apart(self):
        preamble, _ = split_sections(TEXT)
        assert preamble == "You are a careful engineer."

    def test_a_file_with_no_headings_is_one_block(self):
        preamble, sections = split_sections("just some text")
        assert preamble == "just some text"
        assert sections == []

    def test_a_when_further_down_is_prose_not_a_directive(self):
        """Only the first non-empty line counts; otherwise a sentence starting
        with 'when:' halfway down would silently change what a file means."""
        text = "## X\nSome rule.\n\nwhen: this is just prose\n"
        _, sections = split_sections(text)
        assert sections[0].conditional is False
        assert "when: this is just prose" in sections[0].body


class TestResolving:
    @pytest.mark.asyncio
    async def test_a_section_that_does_not_apply_is_dropped(self):
        resolved = await resolve(TEXT, "fix a typo in the README", runtime([0.02, 0.01]))
        assert "rollback path" not in resolved.text
        assert "Write the why" not in resolved.text
        assert set(resolved.skipped) == {"Commit messages", "SQL migrations"}

    @pytest.mark.asyncio
    async def test_a_section_that_applies_is_kept(self):
        resolved = await resolve(TEXT, "review this migration", runtime([0.02, 0.97]))
        assert "rollback path" in resolved.text
        assert "Write the why" not in resolved.text

    @pytest.mark.asyncio
    async def test_unconditional_sections_always_survive(self):
        resolved = await resolve(TEXT, "anything", runtime([0.0, 0.0]))
        assert "Be concise." in resolved.text
        assert "You are a careful engineer." in resolved.text


class TestEverythingUncertainKeepsTheText:
    @pytest.mark.asyncio
    async def test_with_no_runtime(self):
        resolved = await resolve(TEXT, "anything", None)
        assert resolved.text == TEXT
        assert resolved.skipped == ()

    @pytest.mark.asyncio
    async def test_with_the_use_case_off(self):
        resolved = await resolve(TEXT, "anything", runtime([0.0, 0.0], mode=DecisionMode.OFF))
        assert resolved.text == TEXT

    @pytest.mark.asyncio
    async def test_in_shadow_nothing_is_removed(self):
        built = runtime([0.0, 0.0], mode=DecisionMode.SHADOW)
        resolved = await resolve(TEXT, "anything", built)
        assert resolved.text == TEXT
        assert built.provider.asked == 1  # it did measure

    @pytest.mark.asyncio
    async def test_when_the_model_will_not_answer(self):
        class Broken(DecisionProvider):
            type_name = "broken"
            provenance = Provenance.NATIVE

            async def _decide(self, state, questions):
                raise RuntimeError("down")

        built = DecisionRuntime(Broken(), {"instructions": UseCasePolicy(DecisionMode.ACTIVE, 0.5)})
        resolved = await resolve(TEXT, "anything", built)
        assert resolved.text == TEXT

    @pytest.mark.asyncio
    async def test_a_coin_flip_keeps_it(self):
        """0.45 is neither yes nor no. Reading it as "not yes" would delete a
        line the user wrote on the strength of a coin flip — which is why the
        code asks `is_false` rather than `not is_true`."""
        resolved = await resolve(TEXT, "anything", runtime([0.45, 0.45]))
        assert "rollback path" in resolved.text
        assert resolved.skipped == ()

    @pytest.mark.asyncio
    async def test_only_a_confident_no_removes_anything(self):
        for probability in (0.3, 0.19, 0.21):
            resolved = await resolve(TEXT, "anything", runtime([probability, probability]))
            removed = resolved.skipped != ()
            assert removed is (1.0 - probability >= 0.8)

    @pytest.mark.asyncio
    async def test_a_file_with_no_conditions_is_never_even_asked_about(self):
        built = runtime([0.0])
        resolved = await resolve("## Tone\nBe concise.\n", "anything", built)
        assert built.provider.asked == 0
        assert "Be concise." in resolved.text


class TestApplyingToAWholeLoad:
    @pytest.mark.asyncio
    async def test_it_records_what_it_skipped_and_where(self):
        loaded = AgentInstructions(
            soul=(
                Instruction(
                    text=TEXT, instruction_kind=InstructionKind.SOUL, path="/home/me/SOUL.md"
                ),
            )
        )
        result = await apply(loaded, "fix a typo", runtime([0.01, 0.01]))

        assert result.skipped_sections == (
            "SOUL.md: Commit messages",
            "SOUL.md: SQL migrations",
        )
        assert "rollback path" not in result.soul[0].text

    @pytest.mark.asyncio
    async def test_an_unchanged_load_is_the_same_object(self):
        loaded = AgentInstructions(
            soul=(Instruction(text="## Tone\nBe nice.", instruction_kind=InstructionKind.SOUL),)
        )
        assert await apply(loaded, "anything", runtime([])) is loaded

    @pytest.mark.asyncio
    async def test_no_task_means_no_decision(self):
        loaded = AgentInstructions(
            soul=(Instruction(text=TEXT, instruction_kind=InstructionKind.SOUL),)
        )
        built = runtime([0.0, 0.0])
        assert await apply(loaded, "", built) is loaded
        assert built.provider.asked == 0
