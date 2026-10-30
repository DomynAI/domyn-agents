"""The vocabulary: what a question accepts, and what a probability is worth."""

import pytest

from domyn.agents.decisions import (
    Answer,
    Decision,
    Provenance,
    Question,
    choice,
    noul,
    score,
)


class TestAQuestionRefusesWhatTheProtocolRefuses:
    """Caught here rather than as a 422 from the server, two seconds later."""

    def test_a_choice_needs_at_least_two_options(self):
        with pytest.raises(ValueError, match="2-255"):
            choice("pick", {"only": "one"})

    def test_a_choice_refuses_more_than_255(self):
        with pytest.raises(ValueError, match="2-255"):
            choice("pick", {f"o{i}": "x" for i in range(256)})

    def test_a_score_needs_two_to_ten_levels(self):
        with pytest.raises(ValueError, match="2-10 levels"):
            score("how much", ["only one"])
        with pytest.raises(ValueError, match="2-10 levels"):
            score("how much", [f"l{i}" for i in range(11)])

    def test_a_noul_takes_no_criteria(self):
        with pytest.raises(ValueError, match="no criteria"):
            Question(type="noul", instructions="true?", criteria={"a": "b"})

    def test_a_choice_needs_a_mapping_not_a_list(self):
        with pytest.raises(ValueError, match="mapping"):
            Question(type="choice", instructions="pick", criteria=["a", "b"])


class TestScoreLevelsAreNumberedFromZero:
    """A Clef server returns `legend` and `probabilities` keyed "0", "1", "2".
    Numbering from one would misread every score answer."""

    def test_labels_start_at_zero(self):
        assert score("risk", ["none", "some", "lots"]).labels() == ["0", "1", "2"]

    def test_a_choice_keeps_its_own_names(self):
        assert choice("x", {"allow": "a", "deny": "d"}).labels() == ["allow", "deny"]


class TestProvenanceIsOrdered:
    """The ordering is the safety mechanism, so it is pinned."""

    def test_native_satisfies_everything(self):
        for minimum in Provenance:
            assert Provenance.NATIVE.at_least(minimum)

    def test_stated_does_not_satisfy_logprobs(self):
        assert not Provenance.STATED.at_least(Provenance.LOGPROBS)
        assert Provenance.STATED.at_least(Provenance.STATED)

    def test_none_satisfies_only_none(self):
        assert Provenance.NONE.at_least(Provenance.NONE)
        assert not Provenance.NONE.at_least(Provenance.STATED)


class TestActingNeedsBothHalves:
    """A high number from a source that does not make real probabilities is the
    case the whole design exists to refuse."""

    def test_a_confident_stated_answer_cannot_drive_a_logprobs_use_case(self):
        answer = Answer(known=True, provenance=Provenance.STATED, value="allow", confidence=0.99)
        assert not answer.acts_above(0.8, minimum=Provenance.LOGPROBS)
        assert answer.acts_above(0.8, minimum=Provenance.STATED)

    def test_a_native_answer_below_threshold_does_not_act(self):
        answer = Answer(known=True, provenance=Provenance.NATIVE, value="allow", confidence=0.53)
        assert not answer.acts_above(0.8, minimum=Provenance.NATIVE)

    def test_an_unknown_answer_never_acts(self):
        assert not Answer.unknown("nope").acts_above(0.0, minimum=Provenance.NONE)

    def test_is_true_is_false_for_an_unknown(self):
        assert not Answer.unknown("nope").is_true(0.5)

    def test_is_true_needs_the_probability_not_the_confidence(self):
        # A noul of 0.02 is a *confident* answer, and the confident answer is
        # "no". Reading confidence instead of the value would invert it.
        answer = Answer(known=True, provenance=Provenance.NATIVE, value=0.02, confidence=0.98)
        assert not answer.is_true(0.8)


class TestADecisionNeverMakesACallerCheckForKeys:
    def test_a_missing_key_comes_back_unknown(self):
        decision = Decision()
        assert decision.get("absent").known is False
        assert decision.get("absent").reason

    def test_nothing_attaches_the_reason_to_every_question(self):
        decision = Decision.nothing("endpoint down", ["a", "b"])
        assert [a.reason for a in decision.answers.values()] == ["endpoint down"] * 2
        assert decision.known is False
