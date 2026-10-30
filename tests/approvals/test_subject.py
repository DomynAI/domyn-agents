"""Tests for what a call gets reduced to before it is approved or matched.

The subject is the one string that has to mean the same thing in two places: the
gate computes it to look for a grant, and the approval computes it again to
store one. Everything here is about those two agreeing, because when they
disagree nothing fails loudly — the user approves, the grant is written, and the
very next identical call asks again as though they had never answered.
"""

from typing import Any

import pytest

from domyn.agents.approvals import (
    INJECTED_PARAMS,
    ApprovalPolicy,
    ApprovalScope,
    InMemoryApprovalStore,
    build_request,
    compute_subject,
    visible_parameters,
)


class _Tool:
    """Minimal stand-in for a FunctionTool: only what a subject reads."""

    def __init__(self, name: str = "web_search", approval_subject: Any = None) -> None:
        self.name = name
        self.approval_subject = approval_subject


class TestPlannerNoiseIsExcluded:
    """``reasoning`` is added to every tool's schema by the tool_use planner, so
    it arrives as an ordinary parameter and would otherwise be indistinguishable
    from one the user chose."""

    def test_reasoning_never_reaches_the_subject(self) -> None:
        subject = compute_subject(
            _Tool(), {"query": "weather in Bologna", "reasoning": "the user asked about weather"}
        )
        assert "reasoning" not in subject
        assert "query=weather in Bologna" in subject

    def test_the_same_call_reasoned_about_differently_is_one_subject(self) -> None:
        """The property that makes a remembered approval possible at all: the
        model's thought changes on every call, so a subject that contained it
        could never match twice."""
        first = compute_subject(_Tool(), {"query": "abc", "reasoning": "because they asked"})
        second = compute_subject(_Tool(), {"query": "abc", "reasoning": "following up on abc"})
        assert first == second

    def test_a_declared_subject_cannot_reintroduce_it(self) -> None:
        """A tool naming an injected parameter in ``approval_subject`` would put
        the per-call variance straight back, so the filter runs first."""
        subject = compute_subject(
            _Tool(approval_subject=["query", "reasoning"]),
            {"query": "abc", "reasoning": "whatever"},
        )
        assert "reasoning" not in subject

    def test_a_callable_subject_is_not_handed_it_either(self) -> None:
        seen: dict[str, Any] = {}

        def spec(**kwargs: Any) -> str:
            seen.update(kwargs)
            return "fixed"

        compute_subject(_Tool(approval_subject=spec), {"query": "abc", "reasoning": "whatever"})
        assert "reasoning" not in seen
        assert seen == {"query": "abc"}

    def test_a_call_that_is_only_noise_falls_back_to_the_tool_name(self) -> None:
        assert compute_subject(_Tool(), {"reasoning": "thinking"}) == "web_search"

    def test_visible_parameters_is_the_single_source_of_the_exclusion(self) -> None:
        """The runner filters the prompt through this same helper, which is what
        keeps the two sides from drifting apart again."""
        assert "reasoning" in INJECTED_PARAMS
        assert visible_parameters({"a": 1, "reasoning": "x"}) == {"a": 1}


class TestTheTwoSidesAgree:
    """The regression this whole change exists for: a grant stored from an
    approval must match the next identical call at the gate."""

    @pytest.mark.asyncio
    async def test_an_exact_grant_matches_the_next_identical_call(self) -> None:
        tool = _Tool()
        policy = ApprovalPolicy(stores=[InMemoryApprovalStore()])

        # What the approval sees: the user-facing parameters, which the runner
        # strips of planner noise before showing them.
        approved = build_request(tool, {"query": "abc"}, agent_name="Assistant", user_id="u1")
        await policy.remember(approved, ApprovalScope.USER)

        # What the gate sees on the next call: the raw action, noise included,
        # and a *different* thought because the model reasons anew every time.
        at_gate = build_request(
            tool,
            {"query": "abc", "reasoning": "a fresh thought"},
            agent_name="Assistant",
            user_id="u1",
        )
        assert at_gate.key() == approved.key()
        assert (await policy.resolve(at_gate)).is_allow

    @pytest.mark.asyncio
    async def test_a_different_argument_still_asks(self) -> None:
        """User scope is deliberately parameter-exact, so widening it is the
        user's decision and not a side effect of the filter above."""
        tool = _Tool()
        policy = ApprovalPolicy(stores=[InMemoryApprovalStore()])
        await policy.remember(
            build_request(tool, {"query": "abc"}, agent_name="Assistant", user_id="u1"),
            ApprovalScope.USER,
        )

        other = build_request(
            tool, {"query": "something else"}, agent_name="Assistant", user_id="u1"
        )
        assert not (await policy.resolve(other)).is_allow

    @pytest.mark.asyncio
    async def test_one_user_s_grant_does_not_answer_for_another(self) -> None:
        tool = _Tool()
        policy = ApprovalPolicy(stores=[InMemoryApprovalStore()])
        await policy.remember(
            build_request(tool, {"query": "abc"}, agent_name="Assistant", user_id="u1"),
            ApprovalScope.USER,
        )

        someone_else = build_request(tool, {"query": "abc"}, agent_name="Assistant", user_id="u2")
        assert not (await policy.resolve(someone_else)).is_allow
