"""Keep a plan of the work in progress: ``update_plan``.

The model sends the whole plan on every call, as Codex's ``update_plan`` and
Claude Code's ``TodoWrite`` take it: no ids and no edit operations, so a call
cannot fail to match what is stored. The plan is kept per agent under
``context.state[PLANS_STATE_KEY]``, which the runner persists with the
conversation, so it survives compaction and ``--resume``.

The plan is not left to the history for the model to find again: while it has a
step left to do, the tool-use context manager shows it at the end of every request
(``plan_message``), after the history, so an update re-bills only the plan and the
rest of the request stays in the provider's prefix cache.

No agent lists this tool. The tool-use context manager offers it when its
``plan_tool`` is on (``ToolUseContextManager.offers_plan``), with the definition
written here (``UPDATE_PLAN_DEFINITION``: the function's own signature cannot say
what a step is); ``Agent.get_tool_by_name`` resolves its name, and the runner
hands it the name of the agent calling it through the reserved ``calling_agent``
argument.
"""

# No `from __future__ import annotations`: FunctionTool reads the annotations.
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from domyn.agents.tools.base import ToolResult
from domyn.agents.tools.function_tool import FunctionTool

#: Under ``context.state``: agent name -> its plan, as ``Plan.model_dump``.
PLANS_STATE_KEY = "plans"

#: The ``LLMMessage.part`` of the message showing the plan.
PLAN_PART = "plan"

UPDATE_PLAN_TOOL_NAME = "update_plan"


class PlanStatus(StrEnum):
    """Where a step is."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class PlanStep(BaseModel):
    """One step of a plan."""

    model_config = ConfigDict(extra="forbid")

    step: str = Field(min_length=1)
    status: PlanStatus


class Plan(BaseModel):
    """An agent's plan, as the last ``update_plan`` call left it."""

    steps: list[PlanStep]
    explanation: str = ""

    @property
    def is_open(self) -> bool:
        """Whether a step is still to do: only then is the plan shown."""
        return any(step.status is not PlanStatus.COMPLETED for step in self.steps)

    def summary(self) -> str:
        """One line: how far the plan is, and the step in progress."""
        done = sum(step.status is PlanStatus.COMPLETED for step in self.steps)
        current = next(
            (s.step for s in self.steps if s.status is PlanStatus.IN_PROGRESS),
            None,
        )
        line = f"{done}/{len(self.steps)} steps completed"
        return f"{line}; in progress: {current}" if current else line

    def render(self) -> str:
        """The plan as the model is shown it at the end of a request."""
        marks = {
            PlanStatus.COMPLETED: "[x]",
            PlanStatus.IN_PROGRESS: "[>]",
            PlanStatus.PENDING: "[ ]",
        }
        lines = [
            f"{marks[step.status]} {number}. {step.step}"
            + (" (in progress)" if step.status is PlanStatus.IN_PROGRESS else "")
            for number, step in enumerate(self.steps, start=1)
        ]
        return (
            "<current_plan>\n"
            "Your plan, as your last update_plan call left it. Keep it current with "
            "update_plan as you work.\n" + "\n".join(lines) + "\n</current_plan>"
        )


def stored_plan(state: dict[str, Any] | None, agent_name: str) -> Plan | None:
    """The plan *agent_name* keeps in *state*, or ``None`` without a valid one."""
    stored = ((state or {}).get(PLANS_STATE_KEY) or {}).get(agent_name)
    if not stored:
        return None
    try:
        return Plan.model_validate(stored)
    except ValidationError:
        # Written by hand or by another version: not shown rather than failing the turn.
        return None


def update_plan(
    plan: list[dict[str, Any]],
    explanation: str = "",
    state: dict[str, Any] | None = None,
    calling_agent: str = "",
) -> ToolResult:
    """Replace the plan with the one given; an empty list clears it."""
    if not calling_agent:
        raise ValueError("update_plan is not available to this agent.")
    steps: list[PlanStep] = []
    for number, item in enumerate(plan, start=1):
        try:
            steps.append(PlanStep.model_validate(item))
        except ValidationError as exc:
            errors = "; ".join(
                f"{'.'.join(str(p) for p in error['loc']) or 'step'}: {error['msg']}"
                for error in exc.errors()
            )
            raise ValueError(
                "Each step needs `step` (the text) and `status` (pending, in_progress or "
                f"completed). Step {number}: {errors}"
            ) from None
    if sum(step.status is PlanStatus.IN_PROGRESS for step in steps) > 1:
        raise ValueError(
            "Only one step can be in_progress at a time: mark the others pending or completed."
        )

    # `state` is a copy (a function tool's arguments are): the runner merges the
    # key returned below into the conversation's state, one level deep, so every
    # agent's plan goes back with it.
    plans = dict((state or {}).get(PLANS_STATE_KEY) or {})
    if not steps:
        plans.pop(calling_agent, None)
        return ToolResult(success=True, data="Plan cleared.", state={PLANS_STATE_KEY: plans})
    new = Plan(steps=steps, explanation=explanation.strip())
    plans[calling_agent] = new.model_dump(mode="json")
    return ToolResult(
        success=True,
        data=f"Plan updated: {new.summary()}.",
        state={PLANS_STATE_KEY: plans},
    )


UPDATE_PLAN = FunctionTool(func=update_plan)  # type: ignore[no-untyped-call]

_DESCRIPTION = (
    "Keep a short plan of the steps of a task with several steps, and update it "
    "as you work. Send the whole plan every time: it replaces the previous one. "
    "While a step is left to do, the plan is shown to you at the end of the "
    "conversation.\n"
    "- Skip it for a task you can do in one or two steps.\n"
    "- Steps are short (a few words each), in the order you will do them.\n"
    "- Exactly one step is in_progress while you work; mark a step completed as "
    "soon as it is done and checked, before starting the next, and do not mark "
    "a step completed that you have not done.\n"
    "- When the approach changes, change the plan: add, remove or rewrite steps.\n"
    "- Send an empty plan to clear one that no longer applies."
)

#: What the model is offered. Written out because a step is an object with a
#: status from a fixed set, which a function signature cannot express.
UPDATE_PLAN_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": UPDATE_PLAN_TOOL_NAME,
        "description": _DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "array",
                    "description": "Every step of the plan, in order.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {"type": "string", "description": "What the step does."},
                            "status": {
                                "type": "string",
                                "enum": [status.value for status in PlanStatus],
                            },
                        },
                        "required": ["step", "status"],
                    },
                },
                "explanation": {
                    "type": "string",
                    "description": "Optional: why the plan changed, in one line.",
                },
            },
            "required": ["plan"],
        },
    },
}
