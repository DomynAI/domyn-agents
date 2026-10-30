"""IoC decorators for automatic component registration."""

import inspect
from collections.abc import Callable
from typing import Any, TypeVar

from domyn.agents.agents import Agent, WrappedAgent
from domyn.agents.core import ApprovalType, ExecutionEventType
from domyn.agents.hooks import FunctionHook
from domyn.agents.hooks.base import HookPhase, HookType, InputFields
from domyn.agents.llm import BaseLLMProvider
from domyn.agents.logger import get_logger
from domyn.agents.registry import (
    get_agent_registry,
    get_hook_registry,
    get_llm_registry,
    get_tool_registry,
)
from domyn.agents.tools import FunctionTool, Param, StreamableFunctionTool, ToolType

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T", bound=type)


def tool(
    name: str | None = None,
    description: str | None = None,
    type: ToolType = ToolType.FUNCTION,
    config_params: list[Param] | None = None,
    group: str | None = None,
    metadata: dict[str, Any] | None = None,
    approval_type: ApprovalType | None = None,
) -> Callable[[F], F]:
    """
    Decorator to register a function as a tool.

    Args:
        name: Override the tool name (defaults to function name)
        description: Override the tool description (defaults to docstring)
        type: FUNCTION, or STREAMABLE_FUNCTION for an async generator
        config_params: Parameters filled from the configuration rather than by the model
        group: Tool group
        metadata: Free-form tool metadata
        approval_type: Approval required before the tool runs, if any

    Returns:
        A decorator that wraps the function in a FunctionTool (or StreamableFunctionTool),
        registers it and returns the tool.
    """

    if config_params is None:
        config_params = []

    def decorator(func: F) -> FunctionTool | StreamableFunctionTool:
        # Create FunctionTool and register it
        match type:
            case ToolType.STREAMABLE_FUNCTION:
                if not inspect.isasyncgenfunction(func):
                    raise TypeError(
                        f"Function {func.__name__} must be an async generator for StreamableFunctionTool"
                    )
                function_tool = StreamableFunctionTool(
                    func=func,
                    config_params=config_params,
                    group=group,
                    metadata=metadata,
                    approval_type=approval_type,
                )
            case _:
                function_tool = FunctionTool(
                    func=func,
                    config_params=config_params,
                    group=group,
                    metadata=metadata,
                    approval_type=approval_type,
                )

        function_tool.name = name or function_tool.name
        function_tool.description = description or function_tool.description

        registry = get_tool_registry()
        registry.register_tool(function_tool)

        return function_tool

    return decorator


def agent(
    name: str | None = None,
    description: str | None = None,
    *,
    default_execute_agent_function: str = "execute",
    **config_kwargs: Any,
) -> Callable[[type[Agent]], type[Agent]]:
    """
    Decorator to register an agent class.

    Args:
        name: Override the agent name
        description: Override the agent description
        default_execute_agent_function: Override the default execute agent function.
        **config_kwargs: Agent fields, e.g. instruction, llm_provider, planner, sub_agents.
            On a function or a class that is not an ``Agent``, only ``WrappedAgent``
            fields are used; any other is ignored with a warning.

    Returns:
        A factory that builds the agent once and returns that instance; it is registered
        under the agent name.
    """

    def decorator(cls: type[T]) -> type[T]:
        registry = get_agent_registry()

        _name = name or cls.__name__

        # A wrapped agent calls no model and no tool (``tools`` is BaseAgent's, unused here).
        wrapped_kwargs: dict[str, Any] = {}
        if not (inspect.isclass(cls) and issubclass(cls, Agent)):
            wrapped_kwargs = {
                k: v
                for k, v in config_kwargs.items()
                if k in WrappedAgent.model_fields and k != "tools"
            }
            if ignored := sorted(set(config_kwargs) - set(wrapped_kwargs)):
                get_logger().warning(
                    f"@agent {_name!r} is not an Agent subclass: ignoring {', '.join(ignored)}"
                )

        built_instance = None

        def _factory_method(*args, **kwargs):
            nonlocal built_instance
            if built_instance is None:
                if inspect.isclass(cls) and issubclass(cls, Agent):
                    # Include name and description from decorator params
                    if name is not None:
                        config_kwargs["name"] = name
                    if description is not None:
                        config_kwargs["description"] = description
                    config_kwargs.update(kwargs)
                    built_instance = cls(**config_kwargs)
                else:
                    if inspect.isclass(cls):
                        built_instance = WrappedAgent(
                            name=_name,
                            description=description or _name,
                            wrapped_object=cls(),
                            default_execute_agent_function=default_execute_agent_function,
                            **wrapped_kwargs,
                        )
                    else:  # wrapped a function
                        built_instance = WrappedAgent(
                            name=_name,
                            description=description or _name,
                            wrapped_object=cls,
                            **wrapped_kwargs,
                        )
            return built_instance

        registry.register_agent(name=_name, agent=_factory_method)
        return _factory_method

    return decorator


def llm_provider(
    name: str | None = None,
    model_name: str | None = None,
    **config_kwargs: Any,
) -> Callable[[type[BaseLLMProvider]], type[BaseLLMProvider]]:
    """
    Decorator to register an LLM provider class.

    Args:
        name: Override the LLM provider name
        model_name: Override the model name
        **config_kwargs: Additional configuration for the LLM provider

    Returns:
        A factory that builds the provider once and returns that instance.
    """

    def decorator(cls: type[T]) -> type[T]:
        # Ensure the class inherits from BaseLLMProvider
        if not issubclass(cls, BaseLLMProvider):
            raise TypeError(f"Class {cls.__name__} must inherit from BaseLLMProvider")

        _name = name or cls.__name__
        _model_name = model_name or _name

        registry = get_llm_registry()

        built_instance = None

        config_kwargs["model_name"] = _model_name
        config_kwargs["name"] = _name

        def _factory_method(**kwargs) -> BaseLLMProvider:
            nonlocal built_instance
            if built_instance is None:
                config_kwargs.update(kwargs)
                built_instance = cls(**config_kwargs)
            return built_instance

        registry.register_llm_provider(name=_name, provider=_factory_method)
        return _factory_method

    return decorator


def hook(
    trigger: HookPhase,
    name: str | None = None,
    is_listener: bool = False,
    config_params: dict | None = None,
    input_fields: list[InputFields] | None = None,
    metadata: dict[str, Any] | None = None,
    on_event_triggers: list[ExecutionEventType] | None = None,
    break_on_error: bool = True,
    silent: bool = False,
) -> Callable[[Callable[..., Any]], FunctionHook]:
    if config_params is None:
        config_params = {}
    if metadata is None:
        metadata = {}

    def decorator(
        fun: Callable,
    ) -> FunctionHook:
        hook_name = name or fun.__name__
        hook_object = FunctionHook(
            fun=fun,
            name=hook_name,
            trigger=trigger,
            config_params=config_params,
            input_fields=input_fields,
            metadata=metadata,
            type=HookType.LISTENER if is_listener else HookType.UPDATER,
            on_event_triggers=on_event_triggers,
            break_on_error=break_on_error,
            silent=silent,
        )
        get_hook_registry().register(name=hook_name, hook=hook_object)
        return hook_object

    return decorator
