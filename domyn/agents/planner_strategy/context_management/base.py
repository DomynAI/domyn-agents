from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Self

from pydantic import BaseModel, ConfigDict, model_validator

from domyn.agents.core import BaseAgent, BaseEvent, InteractionContext, Part
from domyn.agents.planner_strategy.context_management.budget import (
    HeuristicEstimator,
)
from domyn.agents.planner_strategy.prompting.base import BasePrompting
from domyn.agents.planner_strategy.prompting.json_prompting import JSONMessagePrompting
from domyn.agents.planner_strategy.prompting.messages_building import MessageBuilder
from domyn.agents.planner_strategy.prompting.structural_tags_prompting import (
    StructuralTagMessagePrompting,
)

if TYPE_CHECKING:
    from domyn.agents.tools.read_tool_result import ToolResults


class ContextManager(ABC, BaseModel):
    type: str
    name: str

    estimator: HeuristicEstimator = HeuristicEstimator()
    """How the tokens of what this manager produces are counted.

    It lives here rather than on the planner because this is the object that
    decides what enters the context: whoever builds the view is also who can
    measure it. Any future window policy leans on the same instrument."""

    model_config = ConfigDict(extra="ignore")  # or "forbid" / "allow"

    @staticmethod
    def is_forward_verbatim_enabled(agent: BaseAgent) -> bool:
        planner = getattr(agent, "planner", None)
        return bool(getattr(planner, "forward_verbatim_enabled", True))

    @abstractmethod
    async def get_context(
        self, context: InteractionContext, agent: BaseAgent, use_stop: bool, **kwargs
    ) -> Any:
        """Builds context to be passed to the model"""

    def tool_results(self, agent: BaseAgent, context: InteractionContext) -> "ToolResults | None":
        """The tool results *agent* sees in *context*, as this manager shows them:
        what ``read_tool_result`` reads. ``None`` from a manager that gives no call
        to read one back."""
        return None


class MessageContextManager(ContextManager, ABC):
    prompting: StructuralTagMessagePrompting | JSONMessagePrompting

    @model_validator(mode="after")
    def check_message_builder_types(self) -> Self:
        """Validates that message_builder is an instance of MessageBuilder and BasePrompting as python is missing intersection types"""
        if not isinstance(self.prompting, MessageBuilder):
            raise ValueError(
                "message_builder must be an instance of MessageBuilder and BasePrompting"
            )
        if not isinstance(self.prompting, BasePrompting):
            raise ValueError(
                "message_builder must be an instance of BasePrompting and MessageBuilder"
            )
        return self

    def add_default_agent_start_content(
        self, context: InteractionContext, event: BaseEvent
    ) -> BaseEvent:
        _e = event.model_copy(deep=True)
        _e.content = [Part(text=context.agent_input.content[0].text)]
        return _e
