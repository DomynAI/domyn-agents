from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator

from domyn.agents.approvals import ApprovalVerdict
from domyn.agents.core import BaseEvent, InteractionContext, Part
from domyn.agents.core.exceptions import RestHookError
from domyn.agents.core.utils import dict_to_part


class Hook(BaseModel):
    name: str | None = None
    type: HookType
    trigger: HookPhase
    silent: bool = False


class HookType(StrEnum):
    LISTENER = "listener"
    UPDATER = "updater"


class HookPhase(StrEnum):
    PRE_AGENT_CALL = "pre_agent_call"
    POST_AGENT_CALL = "post_agent_call"
    # Runs *before* the approval gate, and is the only phase that can influence
    # it. PRE_TOOL_CALL is too late: it runs inside tool execution, by which
    # point the decision to execute has already been made.
    PRE_TOOL_APPROVAL = "pre_tool_approval"
    PRE_TOOL_CALL = "pre_tool_call"
    POST_TOOL_CALL = "post_tool_call"
    ON_AGENT_EVENT = "on_agent_event"


class InputFields(StrEnum):
    AGENT_INPUT = "agent_input"
    AGENT_NAME = "agent_name"
    LAST_EVENT = "last_event"
    CURRENT_EVENT = "current_event"
    INTERACTION_HISTORY = "interaction_history"
    INTERACTION_CONTEXT = "interaction_context"
    TOOL_NAME = "tool_name"
    TOOL_ARGS = "tool_args"
    TOOL_RESULT = "tool_result"
    ON_AGENT_EVENT = "on_agent_event"


class HookError(BaseModel):
    error_code: str | int
    error_message: str

    @classmethod
    def from_exception(cls, exception: Exception) -> Self:
        if isinstance(exception, RestHookError):
            error_code = exception.error_code
            error_message = exception.error_message
        else:
            error_code = exception.__class__.__name__
            error_message = str(exception)
        return cls(error_code=error_code, error_message=error_message)

    def __repr__(self) -> str:
        return f"Error_code: {self.error_code}, error_message: {self.error_message}"


class HookResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    success: bool = True
    emitted_content: list[Part] | dict | str | None = []
    error: HookError | None = None
    is_update: bool = False
    name: str | None = None
    # agent hook fields
    new_context: InteractionContext | None = None
    break_execution: bool = False
    # Honoured only on POST_AGENT_CALL: when set, the runner re-enters the agent
    # loop with this event instead of ending the turn. That makes "you are not
    # done, keep going" policies (completion verifiers, constrained retries,
    # quality gates) expressible as hooks rather than as core features.
    #
    # ``break_execution`` wins if both are set: stopping is always the safer
    # interpretation of a hook that asks for two contradictory things.
    #
    # The runner — not the hook — caps how many times this may happen per turn.
    # A hook cannot be its own limit without the cap being circular.
    continue_with: BaseEvent | None = None
    # Honoured only on PRE_TOOL_APPROVAL: allow the call without asking, refuse
    # it without asking, or abstain. Abstaining (or returning nothing) is the
    # normal answer and leaves the decision to the next hook, then to the human.
    # A function hook may also return the verdict bare.
    #
    # The runner enforces the order these are weighed in, and a hook cannot
    # override it: a hardline rule or a user deny rule beats any ALLOW, whoever
    # returned it. Nor does an ALLOW here create a standing permission — only an
    # explicit human choice does that.
    approval: ApprovalVerdict | None = None
    # pre-tool hook fields
    new_tool_args: dict[str, Any] | None = None
    # post-tool hook fields
    new_result: Any = None  # ToolResult | None — type enforced at runtime
    # on_agent_event hook fields
    modified_event: BaseEvent | None = None

    @field_validator("new_result", mode="before")
    @classmethod
    def coerce_new_result(cls, v: Any) -> Any:
        if isinstance(v, dict):
            from domyn.agents.tools import ToolResult  # lazy to avoid circular import

            return ToolResult(**v)
        return v

    @field_validator("emitted_content", mode="before")
    @classmethod
    def process_content(cls, v: Any) -> list[Part]:
        if not v:
            return []
        if isinstance(v, dict):
            return [dict_to_part(v)]
        if isinstance(v, list):
            if all(isinstance(item, Part) for item in v):
                return list(v)
            return [Part(text=str(v))]
        return [Part(text=str(v))]


class HookRestAPIData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"]
    url: str
    headers: dict[str, str] | None = None
