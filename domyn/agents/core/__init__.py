import uuid
from datetime import datetime
from enum import Enum, StrEnum, auto
from functools import reduce
from importlib import import_module
from operator import or_
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationInfo, field_validator

from domyn.agents.core.visibility import VisibilityScope


# approval types feedback
class ApprovalType(Enum):
    USER_CONFIRM = "user_confirm"
    PARAMS_UPDATE = "params_update"


# Actions


class BaseAction(BaseModel):
    """Base class of the actions an event can carry."""

    # register subclasses
    _registry: ClassVar[dict[str, type["BaseAction"]]] = {}
    type: Literal["BaseAction"] = "BaseAction"

    name: str
    thought: str | None = None
    observation: Any | None = None
    message: str | None = None
    parameters: dict[str, Any] | None

    # auto-register sub types
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        tag = getattr(cls, "type", None)
        if tag is not None:
            BaseAction._registry[tag] = cls


class ToolAction(BaseAction):
    """Asks for a tool to be run."""

    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: Literal["ToolAction"] = "ToolAction"


class AgentAction(BaseAction):
    """Asks for a sub-agent to be called."""

    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parameters: dict[str, Any]
    type: Literal["AgentAction"] = "AgentAction"


class StopAction(BaseAction):
    """Ends the execution."""

    type: Literal["StopAction"] = "StopAction"


class FeedbackParameters(BaseModel):
    name: str = Field(..., description="Name of the parameter")
    value: Any = Field(..., description="Value of the parameter")
    type: str | None = Field(default=None, description="Type of the parameter")
    is_required: bool = Field(default=True, description="Whether this parameter is required")

    @field_validator("type", mode="before")
    @classmethod
    def val_x_before(cls, x):
        if isinstance(x, type):
            return x.__name__
        return x


class FeedbackAction(ToolAction):
    """Carries the user's answer to a question."""

    parameters: list[FeedbackParameters] = Field(default_factory=list)
    approval_type: ApprovalType
    approve: bool = True
    remember_scope: str | None = Field(
        default=None,
        description=(
            "How long this approval should stand: 'conversation', 'user', ... "
            "Only meaningful together with approve=True, and only ever set from "
            "an explicit human choice — a policy cannot create a standing "
            "permission on the user's behalf. The runner performs the storing, "
            "because it is the only side that holds the interaction context the "
            "grant belongs to."
        ),
    )
    remember_pattern: str | None = Field(
        default=None,
        description=(
            "Widen the remembered approval to everything matching this glob, "
            "instead of only this exact call. Meaningful only with "
            "remember_scope, and only ever set from a pattern the human saw and "
            "accepted: it grants more than the line they were shown."
        ),
    )
    type: Literal["FeedbackAction"] = "FeedbackAction"

    def to_tool_action(self) -> ToolAction:
        params = self.model_dump()
        params.pop("type")
        params.pop("remember_scope", None)
        params.pop("remember_pattern", None)
        params["parameters"] = {p.name: p.value for p in self.parameters}
        return ToolAction(**params)


class AuthenticationResult(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    USER_REFUSED = "user_refused"


class AuthenticationAction(ToolAction):
    """Carries the authentication credentials a tool asked for."""

    success: AuthenticationResult | None = None
    authentication_metadata: Any | None = None
    type: Literal["AuthenticationAction"] = "AuthenticationAction"


class LLMCallAction(BaseAction):
    """Carries the LLM call payload when the planner delegates execution to the runner.

    Used by DelegateLLMProvider: the planner yields LLM_START with this action
    instead of calling the provider directly. The runner intercepts it, executes
    the LLM call (optionally streaming), and sends back an LLMResponse via asend().
    """

    type: Literal["LLMCallAction"] = "LLMCallAction"
    name: str = "llm_call"
    parameters: dict[str, Any] | None = None
    messages: list[Any]  # list[LLMMessage] — typed Any to avoid circular import
    tools: list[Any] | None = None
    stream: bool = False


ActionsTypes = reduce(or_, (BaseAction, *BaseAction._registry.values()))
Action = TypeAdapter(Annotated[ActionsTypes, Field(discriminator="type")])


class ExecutionEventType(Enum):
    """The kinds of event an execution emits."""

    STARTED = "started"  # not emitted

    USER_INPUT = "user_input"

    # tools
    TOOL_START = "tool_start"
    TOOL_NEEDS_FEEDBACK = "tool_needs_feedback"  # the question to the user
    TOOL_FEEDBACK = "tool_feedback"  # the user's answer
    TOOL_ERROR = "tool_error"
    TOOL_END = "tool_end"
    TOOL_UPDATE = "tool_update"  # progress from a tool that streams updates while it runs

    # hooks
    HOOK_START = "hook_start"
    HOOK_UPDATE = "hook_update"
    HOOK_END = "hook_end"
    HOOK_ERROR = "hook_error"

    # agents
    AGENT_CALL = "agent_call"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    AGENT_NEEDS_FEEDBACK = "agent_needs_feedback"  # the question to the user
    AGENT_FEEDBACK = "agent_feedback"  # the user's answer

    # llms
    LLM_START = "llm_start"
    LLM_END = "llm_end"

    # response
    RESPONSE = "response"
    RESPONSE_TOKEN = "response_token"

    LLM_PROVIDER_ERROR = "llm_provider_error"  # Errors raised by the LLM provider.

    # context
    CONTEXT_COMPACTION_START = "context_compaction_start"
    """The conversation is about to be shortened to fit the window.

    Two events rather than one because the operation takes time: summarizing
    needs a model call, and a front-end that does not know what is going on shows
    an agent stopped for no reason. And because a silent compaction is
    indistinguishable from amnesia: whoever is watching must be able to connect
    "the agent forgot" to "it was compacted", or it looks like a defect."""

    CONTEXT_COMPACTED = "context_compacted"
    """Done. Metadata carries the tokens before and after."""

    # decisions
    DECISION_MADE = "decision_made"
    """A decision model was asked something, and this is what it said.

    Emitted in `shadow` as well as in `active` — in shadow it is the *only*
    output, since nothing else about the turn changes. Carries the use case, the
    answers with their confidences and provenance, whether the answer was acted
    on, and the latency. That last one matters more than it looks: the whole
    case for a decision model is that it costs ~150 ms instead of seconds, and
    this event is where one would read it from.

    Never carries the state that was sent. The state is customer content — a
    command, a conversation — and this event is emitted at info level.
    """

    COMPACTION_REQUESTED = "compaction_requested"
    """Someone asked for a compaction now, rather than waiting for the threshold.

    An input event, not an output: it is sent *to* the runner the way a user turn
    is, which is what makes the manual path work for any front-end instead of
    only the one that implements it. It is deliberately not a `USER_INPUT`, so it
    never enters the conversation history — asking to compact is not something the
    agent should later read as having been said to it.
    """

    # errors
    ERROR_OCCURRED = "error_occurred"
    END = "end"

    # external
    EXTERNAL_PART = "external_part"


class GuardrailResult(BaseModel):
    name: str
    passed: bool
    score: float | None = None
    reason: str | None = None
    error_occurred: bool = False


class Link(BaseModel):
    title: str | None = None
    url: str | None = None
    snippet: str | None = None
    favicon_url: str | None = None
    hostname: str | None = None
    name: str | None = None


class FileType(StrEnum):
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()
    DOCUMENT = auto()


class File(BaseModel):
    file_id: str
    type: FileType | None = None
    metadata: dict | None = None


class Part(BaseModel):
    text: str | None = Field(default=None, description="Text part")
    code: str | None = Field(default=None, description="Code part")
    rich_text: dict[str, Any] | None = Field(
        default=None, description="Enrichment for the text part, i.e. markdown for popover"
    )
    thought: str | None = Field(default=None, description="Thought part")
    html: str | None = Field(default=None, description="HTML part")
    chart: str | None = Field(default=None, description="Chart part")
    metadata: dict | None = Field(default=None, description="Structured data part")
    guardrail_result: GuardrailResult | None = Field(
        default=None, description="Result of a Guardrail check"
    )
    link: Link | None = Field(default=None, description="Link part")
    file: File | None = Field(default=None, description="File part")


# Well-known metadata keys for BaseEvent.metadata
METADATA_FORWARD_VERBATIM = "forward_verbatim"
METADATA_FORWARD_VERBATIM_SOURCE_ID = "forward_verbatim_source_id"
METADATA_FORWARD_VERBATIM_SOURCE_KEY = "forward_verbatim_source_key"
# Set on a subagent RESPONSE the runner converted into an AGENT_END: the
# event stays in history for auditability, but context managers must not
# render it as an assistant answer (the synthesized AGENT_END carries the
# content) and consumers must not show it to the user.
METADATA_CONVERTED_TO_AGENT_END = "converted_to_agent_end"
# Set on a TOOL_START/TOOL_END/TOOL_ERROR a delegated runtime emits to *report* a tool it ran
# itself, rather than to ask the local runner to run one. The local runner forwards these for
# observability and never executes them: the tool may well exist locally under the same name
# (see the DelegateAgent TOOL_START branch in `runner`), and running it would mean executing a
# call nobody made. Stamped by the hermes gateway on its built-in tools
# (`integrations/hermes_platform_gateway`).
METADATA_VISIBILITY_ONLY = "visibility_only"
# Set by a tool-use planner on each TOOL_START of a model response that asked for
# several tools at once: ``{"id": ..., "index": i, "size": n}``. The runner holds
# the calls until the last one arrives and then runs the batch, those that need
# no approval at the same time (see ``Runner._run_tool_calls``). A TOOL_START
# without it is run as soon as it arrives, as every call used to be.
METADATA_TOOL_BATCH = "tool_batch"
# Set on an EXTERNAL_PART event to say whether its content should be rendered
# into the messages sent to the model. Defaults to False when absent: an
# externally-injected part is for display/history only unless explicitly
# opted in, so a channel cannot silently smuggle content into the model's
# context by emitting an event a context manager doesn't otherwise recognize.
METADATA_EXTERNAL_PART_AGENT_VISIBLE = "agent_visible"

# Set on a USER_INPUT written by `Runner.push_parts()` — a message that arrived while a
# turn was already running. It is not how the runner *finds* those events (a
# USER_INPUT in neither conversation_history nor interaction_history is one by
# definition); it exists so `ContextBuilder.create_context` can tell a mid-turn
# message apart from the turn's own opening USER_INPUT, which carries the same
# interaction_id and which sub-agents are meant to see.
METADATA_STEERING = "steering"
# Generic agent-side failure surfaced through ERROR_OCCURRED events.
AGENT_ERROR_CODE = "AGENT_ERROR"
AGENT_ERROR_MESSAGE = "The agent encountered an error while responding."


class BaseEvent(BaseModel):
    event_type: ExecutionEventType
    author: str
    timestamp: datetime = Field(default_factory=datetime.now)
    event_id: str | None = Field(
        default_factory=lambda: f"event_{uuid.uuid4()}", description="Unique event ID"
    )
    conversation_id: str | None = Field(default=None, description="Conversation ID")
    interaction_id: str | None = Field(default=None, description="Interaction ID")
    turn_id: str | None = Field(
        default=None,
        description="Turn ID - uniquely identifies a single user-input turn across all agent transitions",
    )
    content: list[Part] = Field(default_factory=list, description="Event content parts")
    action: ActionsTypes | None = Field(default=None, description="Associated action")  # type: ignore

    # error info
    error_code: str | None = Field(default=None, description="Error code")
    error_message: str | None = Field(default=None, description="Error message")

    # Set on streamed responses: more events of the same answer follow.
    is_partial: bool = Field(
        default=False, description="If the event is partial. If True, more events will follow"
    )

    # The event waits for an answer from the user, about a response or a tool
    # call; False for a final answer.
    need_feedback: bool = Field(default=False, description="If the event needs feedback from user")

    # other info
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata stored in event."
    )

    def model_post_init(self, context: Any, /) -> None:
        # The field's default_factory covers an omitted event_id; this covers one
        # passed explicitly as None, which a caller forwarding an optional id does.
        self.event_id = self.event_id or f"event_{uuid.uuid4()}"

    def to_dict(self) -> dict[str, Any]:
        """Return the event as a dict, for serialization."""
        return self.model_dump()

    # Plain-text form of the event, fed to the summarizing context managers (summary.py).
    def prompt_print(self):
        content = " ".join([str(part) for part in self.content])
        return f"Author: {self.author} Content: {content} \n Action: {self.action!s}"

    def clone_with_author(self, author: str | None, interaction_id: str | None) -> "BaseEvent":
        e = self.model_copy()
        e.author = author if author else e.author
        e.interaction_id = interaction_id if interaction_id else e.interaction_id
        return e

    @field_validator("action", mode="before")
    def parse_action(cls, v: Any, info: ValidationInfo) -> BaseAction | None:
        if isinstance(v, str):
            return Action.validate_json(v)
        elif isinstance(v, dict):
            return Action.validate_python(v)
        return v


class AgentDescription(BaseModel):
    name: str = Field(..., description="Name of the agent")
    description: str = Field(..., description="Description of the agent")
    invocation_json_schema: dict[str, Any] | None = Field(
        ..., description="JSON schema for invoking the agent"
    )
    invocation_parameters: dict[str, Any] | None = Field(
        ..., description="Example parameters for invoking the agent"
    )


class RelayMessage(BaseModel):
    meta: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for delegation"
    )
    payload: BaseEvent = Field(..., description="The event to delegate")
    run_config: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Visibility-filtered caller run_config (PII) forwarded to a delegated agent. "
            "Kept out of the payload event on purpose; None when the delegating agent's "
            "visibility does not grant it. Serialize with exclude_none=True so unset "
            "frames stay byte-identical on the wire."
        ),
    )


__all__ = [
    "AgentDescription",
    "ApprovalType",
    "BaseAction",
    "BaseAgent",
    "BaseEvent",
    "ExecutionEventType",
    "InteractionContext",
    "LLMCallAction",
    "Link",
    "Part",
    "ToolAction",
    "VisibilityScope",
]


# Re-exported on first use: the defining modules import this package, so naming
# them here at import time would be a cycle.
_LAZY_EXPORTS = {
    "InteractionContext": "domyn.agents.core.context",
    "BaseAgent": "domyn.agents.core.interfaces",
}

if TYPE_CHECKING:
    from domyn.agents.core.context import InteractionContext as InteractionContext
    from domyn.agents.core.interfaces import BaseAgent as BaseAgent


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        value = getattr(import_module(_LAZY_EXPORTS[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
