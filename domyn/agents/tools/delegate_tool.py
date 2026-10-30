"""Delegate tool — schema only, execution forwarded to the parent runner."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any, Literal

from pydantic import ConfigDict, Field

from domyn.agents.core import ApprovalType
from domyn.agents.tools.base import BaseTool, ToolResult


class DelegateTool(BaseTool):
    """A tool that is visible to the planner but has no local implementation.

    The tool's ``name``, ``description``, and ``parameters`` schema are
    provided explicitly so the planner can include the tool in LLM tool
    definitions.

    Execution modes (checked in order):

    1. ``wrapped`` is set: behaves like a normal local tool — the wrapped
       callable is invoked directly by the runner.  Mirrors the
       ``DelegateLLMProvider(wrapped=real_provider)`` pattern.

    2. ``relay`` is set: execution is delegated to a remote runtime (e.g.
       ``DomynAgentAdapter``).  The relay is wired at runtime startup so
       the tool stays serialisation-clean — the field is excluded from
       model dumps.  Concurrent requests are safe because each invocation
       registers its own per-invocation correlation state (a
       ``contextvars.ContextVar`` in ``DomynAgentAdapter``) so ``relay`` can
       pick up the correct correlation IDs.

    3. Neither set: raises ``RuntimeError`` (no implementation available).
    """

    type: Literal["delegate"] = "delegate"  # type: ignore[assignment]

    # Required fields — the planner needs these to build tool definitions
    tool_name: str = Field(..., description="Tool name exposed to the LLM")
    tool_description: str = Field(default="", description="Tool description exposed to the LLM")

    # OpenAI-style properties schema: {"param": {"type": "string", "description": "..."}}
    tool_parameters: dict[str, Any] = Field(default_factory=dict)
    required_params: list[str] = Field(default_factory=list)

    approval_type: ApprovalType | None = Field(default=None)

    # Optional local callable; when set the runner executes it normally
    wrapped: Callable[..., Any] | None = Field(default=None, exclude=True)

    # Optional relay; set by DomynAgentAdapter at startup to forward calls to the platform.
    # Excluded from serialisation so the tool model stays JSON-serialisable.
    relay: Callable[..., Any] | None = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ── Interface expected by Agent and Runner ───────────────────────────────

    @property
    def name(self) -> str:
        return self.tool_name

    @property
    def description(self) -> str:
        return self.tool_description

    @property
    def pre_hooks(self) -> list:
        return []

    @property
    def post_hooks(self) -> list:
        return []

    @property
    def tool_definition(self) -> dict[str, Any]:
        """OpenAI-compatible tool definition used by the planner context manager."""
        required = self.required_params or list(self.tool_parameters.keys())
        return {
            "type": "function",
            "function": {
                "name": self.tool_name,
                "description": self.tool_description,
                "parameters": {
                    "type": "object",
                    "properties": self.tool_parameters,
                    "required": required,
                },
            },
        }

    @property
    def invocation_json_schema(self) -> dict[str, Any]:
        required = self.required_params or list(self.tool_parameters.keys())
        return {
            "properties": {
                self.name_string: {"const": self.tool_name, "type": "string"},
                self.parameters_string: {
                    "properties": self.tool_parameters,
                    "required": required,
                },
            },
            "required": [self.parameters_string],
            "description": self.tool_description,
        }

    def required_param_names(self) -> set[str]:
        return set(self.required_params or self.tool_parameters.keys())

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool via ``wrapped``, ``relay``, or raise if neither is set."""
        if self.wrapped is None:
            if self.relay is not None:
                try:
                    # Strip runner-injected keys; only forward the declared tool parameters.
                    relay_kwargs = {k: v for k, v in kwargs.items() if k in self.tool_parameters}
                    result = await self.relay(self.tool_name, relay_kwargs)
                    if not isinstance(result, ToolResult):
                        result = ToolResult(success=True, data=result)
                except Exception as e:
                    result = ToolResult(success=False, error=str(e), error_code=type(e).__name__)
                return result
            raise RuntimeError(
                f"DelegateTool '{self.tool_name}' has no wrapped callable and no active relay."
            )
        try:
            # filter kwargs to only those accepted by the wrapped callable
            sig = inspect.signature(self.wrapped)
            accepted = set(sig.parameters.keys())
            filtered = {k: v for k, v in kwargs.items() if k in accepted}

            if inspect.iscoroutinefunction(self.wrapped):
                result = await self.wrapped(**filtered)
            else:
                result = await asyncio.to_thread(self.wrapped, **filtered)

            if not isinstance(result, ToolResult):
                result = ToolResult(success=True, data=result)
        except Exception as e:
            result = ToolResult(success=False, error=str(e), error_code=type(e).__name__)
        return result

    # ── Factory ─────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: dict[str, Any], save_in_registry: bool = False) -> DelegateTool:
        tool = cls(
            tool_name=config["name"],
            tool_description=config.get("description", ""),
            tool_parameters=config.get("parameters", {}),
            required_params=config.get("required_params", []),
            uid=config.get("uid"),
            approval_type=(
                ApprovalType(config["approval_type"]) if config.get("approval_type") else None
            ),
        )
        if save_in_registry:
            from domyn.agents.registry import get_tool_registry

            get_tool_registry().register_tool(tool.tool_name, tool)
        return tool
