"""The plan tool: what a call stores, what it refuses, and how the plan is shown."""

import pytest

from domyn.agents.llm import LLMMessage
from domyn.agents.planner_strategy.context_injection import append_plan
from domyn.agents.tools.update_plan import (
    PLAN_PART,
    PLANS_STATE_KEY,
    UPDATE_PLAN,
    UPDATE_PLAN_DEFINITION,
    PlanStatus,
    stored_plan,
)


async def _update(plan: list[dict[str, str]], state: dict | None = None, agent: str = "A"):
    return await UPDATE_PLAN.execute(plan=plan, state=state or {}, calling_agent=agent)


class TestACall:
    @pytest.mark.asyncio
    async def test_replaces_the_plan_and_says_where_it_is(self) -> None:
        result = await _update(
            [
                {"step": "Reproduce", "status": "completed"},
                {"step": "Fix", "status": "in_progress"},
                {"step": "Run the suite", "status": "pending"},
            ]
        )

        assert result.success
        assert result.data == "Plan updated: 1/3 steps completed; in progress: Fix."
        plan = stored_plan(result.state, "A")
        assert plan is not None
        assert [s.status for s in plan.steps] == [
            PlanStatus.COMPLETED,
            PlanStatus.IN_PROGRESS,
            PlanStatus.PENDING,
        ]

    @pytest.mark.asyncio
    async def test_keeps_the_plans_of_other_agents(self) -> None:
        other = {"steps": [{"step": "x", "status": "pending"}], "explanation": ""}
        result = await _update(
            [{"step": "y", "status": "pending"}], state={PLANS_STATE_KEY: {"B": other}}
        )

        assert set(result.state[PLANS_STATE_KEY]) == {"A", "B"}

    @pytest.mark.asyncio
    async def test_an_empty_plan_clears_it(self) -> None:
        first = await _update([{"step": "y", "status": "pending"}])
        result = await _update([], state=first.state)

        assert result.success and result.data == "Plan cleared."
        assert stored_plan(result.state, "A") is None

    @pytest.mark.asyncio
    async def test_refuses_two_steps_in_progress(self) -> None:
        result = await _update(
            [{"step": "a", "status": "in_progress"}, {"step": "b", "status": "in_progress"}]
        )

        assert not result.success
        assert "Only one step can be in_progress" in (result.error or "")

    @pytest.mark.asyncio
    async def test_names_the_step_it_cannot_read(self) -> None:
        result = await _update(
            [{"step": "a", "status": "pending"}, {"step": "b", "status": "done"}]
        )

        assert not result.success
        assert "Step 2" in (result.error or "")

    @pytest.mark.asyncio
    async def test_needs_to_know_the_agent(self) -> None:
        result = await UPDATE_PLAN.execute(plan=[{"step": "a", "status": "pending"}], state={})

        assert not result.success


class TestTheDefinition:
    def test_names_the_statuses_and_hides_the_runtime_arguments(self) -> None:
        parameters = UPDATE_PLAN_DEFINITION["function"]["parameters"]
        step = parameters["properties"]["plan"]["items"]

        assert step["properties"]["status"]["enum"] == ["pending", "in_progress", "completed"]
        assert set(parameters["properties"]) == {"plan", "explanation"}
        assert UPDATE_PLAN_DEFINITION["function"]["name"] == UPDATE_PLAN.name


class TestTheStoredPlan:
    def test_an_unreadable_one_is_none(self) -> None:
        assert stored_plan({PLANS_STATE_KEY: {"A": {"steps": "nope"}}}, "A") is None
        assert stored_plan(None, "A") is None

    @pytest.mark.asyncio
    async def test_is_open_until_every_step_is_completed(self) -> None:
        open_ = await _update([{"step": "a", "status": "in_progress"}])
        done = await _update([{"step": "a", "status": "completed"}])

        assert stored_plan(open_.state, "A").is_open
        assert not stored_plan(done.state, "A").is_open

    @pytest.mark.asyncio
    async def test_renders_with_marks(self) -> None:
        result = await _update(
            [{"step": "a", "status": "completed"}, {"step": "b", "status": "in_progress"}]
        )

        text = stored_plan(result.state, "A").render()
        assert "[x] 1. a" in text
        assert "[>] 2. b (in progress)" in text


def _plan_message() -> LLMMessage:
    return LLMMessage(role="user", content="<current_plan>…</current_plan>", part=PLAN_PART)


class TestAppendingThePlan:
    def test_goes_after_tool_results_as_a_message_of_its_own(self) -> None:
        messages = [
            LLMMessage(role="system", content="s"),
            LLMMessage(role="tool", content="r", tool_call_id="1"),
        ]

        append_plan(messages, _plan_message())

        assert messages[-1].part == PLAN_PART
        assert len(messages) == 3

    def test_joins_the_user_message_it_follows(self) -> None:
        # Never two user messages in a row of ours: a strict chat template refuses them.
        messages = [LLMMessage(role="system", content="s"), LLMMessage(role="user", content="go")]

        append_plan(messages, _plan_message())

        assert len(messages) == 2
        assert messages[-1].content == "go\n\n<current_plan>…</current_plan>"

    def test_none_changes_nothing(self) -> None:
        messages = [LLMMessage(role="user", content="go")]

        assert append_plan(messages, None) == [LLMMessage(role="user", content="go")]
