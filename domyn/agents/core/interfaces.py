"""Base interfaces, kept apart to avoid circular imports.

This module holds only base classes and abstract interfaces, with no concrete
dependencies, so that any package can import it.
"""

from typing import Any, ClassVar

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class BaseAgent(BaseModel):
    """Base class of every agent."""

    # register subclasses by their `type` tag (same pattern as BaseTool)
    _registry: ClassVar[dict[str, type["BaseAgent"]]] = {}

    type: str = "BaseAgent"

    name: str = Field(..., description="Name of the agent")
    # declared after `name`: the validator reads it from the fields already validated
    id: str | None = Field(
        default=None,
        validate_default=True,
        description="Id of the agent; the agent's name when not given",
    )
    description: str = Field(..., description="Description of the agent")

    sub_agents: list["BaseAgent"] = []
    name_string: ClassVar[str] = "name"
    parameters_string: ClassVar[str] = "arguments"
    stream: bool = False
    tools: list[Any] = Field(default_factory=list, description="Available tools")

    @field_validator("id", mode="before")
    @classmethod
    def _default_id_to_name(cls, value: Any, info: ValidationInfo) -> Any:
        return info.data.get("name") if value is None else value

    # auto-register sub types so AgentFactory can resolve them by `type`
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        tag = getattr(cls, "type", None)
        if tag and tag != "BaseAgent":
            BaseAgent._registry[tag] = cls

    @classmethod
    def from_config(cls, config: dict[str, Any], save_in_registry: bool = False) -> "BaseAgent":
        """Create an agent instance from a configuration dict.

        Subclasses override this to provide their specific construction logic
        (same convention as BaseTool.from_config).
        """
        raise NotImplementedError(f"from_config not implemented for {cls.__name__}")


BaseAgent.model_rebuild()
