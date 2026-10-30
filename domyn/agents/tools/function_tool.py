from __future__ import annotations

import asyncio
import importlib
import inspect
from collections.abc import Callable
from copy import deepcopy
from functools import cached_property
from typing import Any, Literal

from pydantic import ConfigDict, Field, PrivateAttr

from domyn.agents.core import ApprovalType
from domyn.agents.core.utils import SerializedCallable, load_function_from_file, serialize_callable
from domyn.agents.hooks.base import Hook, HookPhase
from domyn.agents.tools.base import (
    BaseTool,
    ConfigParam,
    FunctionDeclaration,
    ToolResult,
)
from domyn.agents.tools.schema import python_type_to_json_type


class FunctionTool(BaseTool):
    """Wrapper for Python functions as tools."""

    type: Literal["function"] = "function"  # type: ignore

    func: SerializedCallable | Callable[..., Any] = Field(
        ..., description="Callable function or serialized callable function metadata"
    )
    approval_type: ApprovalType | None = Field(
        default=None, description="Type of user approval required before executing the tool"
    )
    approval_pattern: Callable[..., str] | None = Field(
        default=None,
        description=(
            "Proposes a wider permission covering calls like this one — "
            "'$ curl *' for a shell command, a directory for a file write. "
            "Only ever a *proposal*: it is shown to the user and can be edited "
            "before it becomes a standing rule, because a pattern grants more "
            "than the line they read. Unset means only exact approvals are "
            "offered for this tool."
        ),
        exclude=True,
    )
    approval_parts: Callable[..., list[str]] | None = Field(
        default=None,
        description=(
            "Splits a call into the commands it runs, as subjects, so a pattern "
            "covers the call only if it covers each of them. An empty list means "
            "the call cannot be split safely and only an exact approval covers it. "
            "Unset means the subject is the one part. See domyn.agents.approvals."
        ),
        exclude=True,
    )
    approval_subject: list[str] | Callable[..., str] | None = Field(
        default=None,
        description=(
            "Which arguments carry the risk, so an approval can be remembered for "
            "them rather than for the whole tool. A list names the arguments; a "
            "callable receives them and returns the line to show and match on. "
            "Unset means every argument, which is safe and almost never matches "
            "twice — so the user keeps being asked. See domyn.agents.approvals."
        ),
        exclude=True,
    )
    config_params: list[ConfigParam] = Field(
        default_factory=list,
        description="Configuration parameters for the tool. These parameters are not shown to models.",
    )
    reference: str | None = Field(
        default=None, description="Reference tool name on which this tool is based."
    )
    # Pre/post tool hooks; the phase comes from each hook's `trigger`.
    hooks: list[Hook] = Field(default_factory=list, description="List of hooks applied to the tool")
    _actual_callable: Callable[..., Any] | None = PrivateAttr(default=None)
    _declaration: FunctionDeclaration | None = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, func=None, *args, **kwargs):
        if func is not None:
            kwargs["func"] = func

        field_names = [
            name
            for name, field_info in self.__class__.model_fields.items()
            if field_info.init is not False
        ]
        kwargs.update(dict(zip(field_names, args, strict=False)))

        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        return self.callable_function(*args, **kwargs)

    def model_post_init(self, context: Any, /) -> None:
        if not self._declaration:
            if callable(self.func):
                actual_callable = self.func
                serialized = serialize_callable(actual_callable)
                object.__setattr__(self, "func", serialized)
                # Store the original callable for cached_property access
                self._actual_callable = actual_callable
            elif isinstance(self.func, dict):
                # Try to reconstruct the function from serialized data
                try:
                    self._actual_callable = self._load_function_from_serialized(self.func)
                except Exception:
                    # If we can't load the function, we'll have to work without it
                    self._actual_callable = None
            else:
                self._actual_callable = None

            if self.reference is None:
                object.__setattr__(self, "reference", self.name)

            if self.uid is None:
                object.__setattr__(self, "uid", self.name)

    @property
    def pre_hooks(self) -> list[Hook]:
        return [hook for hook in self.hooks if hook.trigger == HookPhase.PRE_TOOL_CALL]

    @property
    def post_hooks(self) -> list[Hook]:
        return [hook for hook in self.hooks if hook.trigger == HookPhase.POST_TOOL_CALL]

    @property
    def approval_hooks(self) -> list[Hook]:
        """Hooks that may speak before the approval gate. Separate from
        ``pre_hooks`` because those run after it, inside execution."""
        return [hook for hook in self.hooks if hook.trigger == HookPhase.PRE_TOOL_APPROVAL]

    def _load_function_from_serialized(self, serialized: dict) -> Callable[..., Any] | None:
        """Try to load a function from serialized data."""
        if serialized.get("type") == "function":
            module_name = serialized.get("module")
            func_name = serialized.get("name")

            if module_name and func_name:
                try:
                    module = importlib.import_module(module_name)
                    return getattr(module, func_name)
                except (ImportError, AttributeError):
                    return None
        return None

    @property
    def declaration(self) -> FunctionDeclaration | None:
        if getattr(self, "_declaration", None) is not None:
            return self._declaration

        if self._actual_callable is not None:
            self._declaration = FunctionDeclaration.from_callable(
                self._actual_callable, ignore_params=[conf.name for conf in self.config_params]
            )
        return self._declaration

    def set_declaration(self, value: FunctionDeclaration | None):
        object.__setattr__(self, "_declaration", value)

    def required_param_names(self) -> set[str]:
        declaration = self.declaration
        if declaration is None:
            return set()
        return {p.name for p in declaration.parameters if p.is_required and p.name is not None}

    @cached_property
    def _is_async(self) -> bool:
        if self._actual_callable is not None:
            return inspect.iscoroutinefunction(self._actual_callable)
        return False

    @cached_property
    def _is_async_generator(self) -> bool:
        if self._actual_callable is not None:
            return inspect.isasyncgenfunction(self._actual_callable)
        return False

    @property
    def callable_function(self):
        if self._actual_callable is not None:
            return self._actual_callable
        else:
            raise ValueError(
                "Cannot convert serialized function back to callable. Original callable was lost."
            )

    @property
    def name(self) -> str:
        return self.declaration.name

    @name.setter
    def name(self, value: str) -> None:
        self.declaration.name = value

    @property
    def description(self) -> str:
        return self.declaration.description

    @description.setter
    def description(self, value: str) -> None:
        self.declaration.description = value

    @property
    def invocation_json_schema(self) -> dict[str, Any]:
        """Generate a JSON schema for the function parameters."""
        parameters_schema = {
            "properties": {},
            "required": [],
        }

        for param in self.declaration.parameters:
            param_schema = {
                "description": param.description,
                **python_type_to_json_type(param.type),
            }

            parameters_schema["properties"][param.name] = param_schema
            if param.is_required:
                parameters_schema["required"].append(param.name)
        schema = {
            "properties": {
                self.name_string: {"const": self.name, "type": "string"},
                self.parameters_string: parameters_schema,
            },
            "required": [self.parameters_string],
            "description": self.description,
        }
        return schema

    @property
    def tool_definition(self) -> dict[str, Any]:
        """Return OpenAI-compatible tool definition for native tool calling."""
        parameters_schema: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        for param in self.declaration.parameters:
            param_schema = {
                "description": param.description,
                **python_type_to_json_type(param.type),
            }
            parameters_schema["properties"][param.name] = param_schema
            if param.is_required:
                parameters_schema["required"].append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters_schema,
            },
        }

    @property
    def namespaced_invocation_json_schema(self) -> dict[str, Any]:
        """Generate a JSON schema for the function parameters with namespacing.
        Using '.' as namespace separator as per SEP-986 https://github.com/modelcontextprotocol/modelcontextprotocol/issues/986
        """
        schema = self.invocation_json_schema
        schema["properties"][self.name_string]["const"] = (
            f"{self.group or 'base-group'}.{self.name}"
        )
        return schema

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the function with given parameters."""
        try:
            kwargs = {
                k: deepcopy(
                    v
                )  # Ensure that a tool can not modify kwargs (e.g. state) in place without returning it
                for k, v in kwargs.items()
                if k in list(inspect.signature(self.callable_function).parameters.keys())
            }

            config_params = {param.name: param.value for param in self.config_params}

            if self._is_async_generator:
                result = "".join(
                    map(
                        str,
                        filter(
                            None,
                            [
                                item
                                async for item in self.callable_function(
                                    **{**kwargs, **config_params}
                                )
                            ],
                        ),
                    )
                )
            else:
                if self._is_async:
                    result = await self.callable_function(**{**kwargs, **config_params})
                else:
                    result = await asyncio.to_thread(
                        self.callable_function, **{**kwargs, **config_params}
                    )

            if not isinstance(result, ToolResult):
                # Cast a ToolResult-shaped dict (or a legacy "result" envelope)
                # directly instead of burying it in the `data` attribute.
                coerced = ToolResult.from_result_dict(result, result_key_fallback=True)
                result = coerced if coerced is not None else ToolResult(success=True, data=result)

        except Exception as e:
            result = ToolResult(success=False, error=str(e), error_code=type(e).__name__)

        return result

    # ---------------------------------------------------------------------
    # Factory helpers
    # ---------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: dict, save_in_registry: bool = False) -> FunctionTool:
        from domyn.agents.hooks import HookFactory, HookPhase

        hooks = []
        for hook in config.get("hooks", []):
            if isinstance(hook, dict):
                if hook["trigger"] == HookPhase.PRE_TOOL_CALL:
                    hooks.append(
                        HookFactory.from_config(hook, trigger=HookPhase.PRE_TOOL_CALL)
                        if isinstance(hook, dict)
                        else hook
                    )
                elif hook["trigger"] == HookPhase.POST_TOOL_CALL:
                    hooks.append(
                        HookFactory.from_config(hook, trigger=HookPhase.POST_TOOL_CALL)
                        if isinstance(hook, dict)
                        else hook
                    )
                elif hook["trigger"] == HookPhase.PRE_TOOL_APPROVAL:
                    hooks.append(
                        HookFactory.from_config(hook, trigger=HookPhase.PRE_TOOL_APPROVAL)
                        if isinstance(hook, dict)
                        else hook
                    )
                else:
                    raise ValueError(f"Unsupported hook trigger: {hook['trigger']}")
            else:
                hooks.append(hook)

        kwargs = {}
        if "path" in config:
            func = load_function_from_file(name=config["name"], path=config["path"])
        else:
            func = config.get("func")
            if func is None:
                raise ValueError("func or path must be provided")

            if isinstance(func, dict):
                func = load_function_from_file(
                    name=func["name"], path=func["path"] or func["module"]
                )

        # Only the values present, so pydantic applies its defaults to the rest
        kwargs.update(
            {
                "func": func,
                "hooks": hooks,
                "uid": config.get("uid"),
            }
        )

        kwargs["config_params"] = [
            ConfigParam(**param) for param in config.get("config_params", [])
        ]

        # approval_type only when the config sets it
        if "approval_type" in config:
            kwargs["approval_type"] = (
                ApprovalType(config["approval_type"]) if config["approval_type"] else None
            )

        tool = cls(**kwargs)
        if save_in_registry:
            from domyn.agents.registry import get_tool_registry

            get_tool_registry().register_tool(config.get("name", tool.name), tool)
        return tool

    @classmethod
    def read_from_registry_and_configure(cls, config: dict) -> FunctionTool:
        from domyn.agents.hooks import HookFactory, HookPhase
        from domyn.agents.registry import get_tool_registry

        hooks = []
        for hook in config.get("hooks", []):
            if isinstance(hook, dict):
                if hook["trigger"] == HookPhase.PRE_TOOL_CALL:
                    hooks.append(
                        HookFactory.from_config(hook, trigger=HookPhase.PRE_TOOL_CALL)
                        if isinstance(hook, dict)
                        else hook
                    )
                elif hook["trigger"] == HookPhase.POST_TOOL_CALL:
                    hooks.append(
                        HookFactory.from_config(hook, trigger=HookPhase.POST_TOOL_CALL)
                        if isinstance(hook, dict)
                        else hook
                    )
                elif hook["trigger"] == HookPhase.PRE_TOOL_APPROVAL:
                    hooks.append(
                        HookFactory.from_config(hook, trigger=HookPhase.PRE_TOOL_APPROVAL)
                        if isinstance(hook, dict)
                        else hook
                    )
                else:
                    raise ValueError(f"Unsupported hook trigger: {hook['trigger']}")
            else:
                hooks.append(hook)

        registered_tool = get_tool_registry().get_tool(config.get("reference") or config["name"])
        if registered_tool and isinstance(registered_tool, FunctionTool):
            cls.func = registered_tool.func

            _registered_callable = registered_tool._actual_callable
            registered_tool._actual_callable = (
                None  # Workaround: detach the live callable so deepcopy/pickle don't fail
            )
            tool_with_config_params = deepcopy(registered_tool)
            tool_with_config_params._actual_callable = _registered_callable
            registered_tool._actual_callable = _registered_callable
            if "config_params" in config:
                tool_with_config_params.config_params = [
                    ConfigParam(**param) for param in config.get("config_params", [])
                ]
            if "hooks" in config:
                tool_with_config_params.hooks = hooks
            if "approval_type" in config:
                tool_with_config_params.approval_type = config["approval_type"]

            if config.get("reference") and config["reference"] != config["name"]:
                tool_with_config_params.name = config["name"]

            if config.get("uid"):
                tool_with_config_params.uid = config["uid"]

            return tool_with_config_params
        else:
            raise ValueError(
                f"FunctionTool with name '{config['reference']}' not found in registry."
            )
