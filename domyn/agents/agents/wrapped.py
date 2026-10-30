import inspect
from collections.abc import AsyncGenerator
from typing import Any

from pydantic import Field

from domyn.agents.agents.base import ContextVisibility, HandoffScope
from domyn.agents.core import AgentDescription, BaseAgent, BaseEvent, InteractionContext
from domyn.agents.hooks import FunctionHook


class WrappedAgent(BaseAgent):
    """A class that wraps another class like an Agent"""

    type: str = "WrappedAgent"

    name: str = Field(..., description="Name of the agent")
    description: str = Field(..., description="Description of the agent")

    wrapped_object: Any

    default_execute_agent_function: str = Field(
        default="execute", description="Default execute agent function"
    )

    max_iterations: int = Field(default=3, description="Maximum number of iterations")
    pre_hooks: list[FunctionHook] = Field(
        default_factory=list, description="Hooks to be executed before the agent execution"
    )
    post_hooks: list[FunctionHook] = Field(
        default_factory=list, description="Hooks to be executed after the agent execution"
    )
    on_event_hooks: list[FunctionHook] = Field(
        default_factory=list, description="Hooks run on every event emitted by or sent to the agent"
    )
    context_visibility: ContextVisibility = Field(
        default=ContextVisibility.STANDARD,
        description=(
            "Context/history visibility level for this agent (READ scope). Callable "
            "handoff targets are configured separately through handoff_scope."
        ),
    )
    handoff_scope: HandoffScope = Field(
        default=HandoffScope.EXPLICIT,
        description=(
            "EXECUTE scope for handoff targets, with the same semantics as "
            "Agent.handoff_scope. EXPLICIT honors `sub_agents` verbatim; DIRECT_PEERS/GLOBAL "
            "resolve the callable set against the live agent tree at runtime."
        ),
    )

    # Lets the wrapped object's methods be called on the agent directly
    def __getattr__(self, name):
        # Only delegate to wrapped_object if the attribute doesn't exist in this class
        if hasattr(self.wrapped_object, name):
            return getattr(self.wrapped_object, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    @property
    def invocations_parameters(self) -> dict[str, Any]:
        """Default parameters for agent invocation as function."""
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The specific task to perform by the agent",
                },
            },
            "required": ["task"],
        }

    @property
    def invocation_json_schema(self) -> dict:
        schema = {
            "properties": {
                self.name_string: {"const": self.name, "type": "string"},
                self.parameters_string: self.invocations_parameters,
            },
            "required": [self.parameters_string],
            "description": self.description,
        }
        return schema

    def get_agent_description(self) -> str:
        return AgentDescription(
            name=self.name,
            description=self.description,
            invocation_json_schema=self.invocation_json_schema,
            invocation_parameters=self.invocations_parameters,
        )

    async def execute(
        self, context: InteractionContext
    ) -> AsyncGenerator[tuple[BaseEvent, InteractionContext], None]:
        """Yields ``(event, context)``, the shape the runner reads from every agent."""
        if inspect.isfunction(self.wrapped_object):
            events = self.wrapped_object(context)
        else:
            fun = getattr(self.wrapped_object, self.default_execute_agent_function)
            arguments = {
                "interaction_context": context,
                "context": context,
                "run_config": context.run_config,
                "state": context.state,
            }
            accepted = inspect.signature(fun).parameters
            events = fun(**{k: v for k, v in arguments.items() if k in accepted})
        async for event in events:
            yield event, context

    @classmethod
    def from_config(cls, config: dict[str, Any], save_in_registry: bool = False) -> "WrappedAgent":
        """Not supported yet: raises NotImplementedError."""
        raise NotImplementedError("WrappedAgent.from_config not implemented yet")


WrappedAgent.model_rebuild()  # type: ignore
