"""Context management for agent interactions."""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from ulid import ULID

from domyn.agents.core import AgentDescription, BaseEvent


class InteractionContext(BaseModel):
    """Context for a single interaction session."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    conversation_id: str
    user_id: str
    interaction_id: str | None = Field(
        default=None, description="Unique interaction ID | Auto generated if not provided"
    )
    turn_id: str | None = Field(
        default=None,
        description="Turn ID - uniquely identifies a single user-input turn across all agent transitions",
    )
    agent_input: BaseEvent = Field(
        description=(
            "Current input/task for this interaction — the AGENT_START built by the "
            "runner for the entrypoint turn, the AGENT_START for sub-agents/after "
            "handoff, or the original event on tool-feedback resumes"
        )
    )
    is_subagent: bool = Field(
        default=False,
        description="True when this context was created for a sub-agent invoked via AGENT_CALL; False for the root/entrypoint agent.",
    )
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    interaction_history: list[BaseEvent] = Field(
        default_factory=list, description="All the events of the current interaction"
    )

    conversation_history: list[BaseEvent] = Field(
        default_factory=list, description="All the events of the conversation visible to this agent"
    )
    handoff_targets_description: list[AgentDescription] = Field(
        default_factory=list,
        description=(
            "Agents the current agent may invoke as handoff targets (callable set). "
            "Surfaced to the LLM in the system prompt and as tool definitions. "
            "Driven by the runner's handoff-targets resolution of the agent's "
            "`handoff_scope`: EXPLICIT honors the `sub_agents` list verbatim, "
            "DIRECT_PEERS/GLOBAL resolve against the live agent tree at runtime."
        ),
    )
    visible_agent_names: list[str] = Field(
        default_factory=list,
        description=(
            "Names of agents whose prior events are visible to the current agent. "
            "This READ scope is computed from Agent.context_visibility "
            "(GLOBAL/PEERS/STANDARD). It is distinct from handoff_targets_description, "
            "which is the callable EXECUTE set surfaced to the LLM."
        ),
    )

    # Free-form state, persisted per conversation by the runner
    state: dict[str, Any] = Field(default_factory=dict)

    # Read-only by convention: this is the runner's dict, not a copy, so writes leak.
    # Nothing enforces it.
    run_config: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, context: Any, /) -> None:
        if self.interaction_id is None:
            self.interaction_id = self.generate_interaction_id()

    @staticmethod
    def generate_interaction_id() -> str:
        return f"interaction_{uuid.uuid4().hex}"

    @staticmethod
    def generate_turn_id() -> str:
        """Generate a unique, sortable turn ID using ULID."""
        return str(ULID())

    @property
    def last_event(self) -> BaseEvent | None:
        """Get the last event in the interaction history."""
        return self.interaction_history[-1] if self.interaction_history else None
