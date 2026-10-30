"""Domyn Agents.

Importing the package imports none of its subpackages, so a process that needs
one light module does not pay for the agent machinery. ``decorators`` and
``register_transport`` are imported on first use.
"""

import warnings
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

warnings.filterwarnings(
    "ignore", category=UserWarning, message=r"Field name .* shadows an attribute in parent"
)

try:
    __version__ = version("domyn")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "Agent",
    "AnthropicProvider",
    "ApprovalType",
    "BaseAgent",
    "BaseEvent",
    "ContextVisibility",
    "DelegateAgent",
    "ExecutionEventType",
    "FunctionTool",
    "HandoffScope",
    "InteractionContext",
    "MCPTool",
    "OpenAIProvider",
    "OpenRouterProvider",
    "Part",
    "Runner",
    "StreamableFunctionTool",
    "VLLMProvider",
    "__version__",
    "decorators",
    "load_application_from_file",
    "register_transport",
]

# Public names, resolved on first use from the subpackage that owns them, so that
# ``from domyn.agents import Agent`` works while importing the package imports none
# of its subpackages.
_LAZY_EXPORTS = {
    "Agent": "domyn.agents.agents",
    "ContextVisibility": "domyn.agents.agents",
    "HandoffScope": "domyn.agents.agents",
    "ApprovalType": "domyn.agents.core",
    "DelegateAgent": "domyn.agents.agents",
    "BaseAgent": "domyn.agents.core",
    "InteractionContext": "domyn.agents.core",
    "BaseEvent": "domyn.agents.core",
    "ExecutionEventType": "domyn.agents.core",
    "Part": "domyn.agents.core",
    "Runner": "domyn.agents.runner",
    "FunctionTool": "domyn.agents.tools",
    "StreamableFunctionTool": "domyn.agents.tools",
    "MCPTool": "domyn.agents.tools",
    "VLLMProvider": "domyn.agents.llm",
    "OpenAIProvider": "domyn.agents.llm",
    "OpenRouterProvider": "domyn.agents.llm",
    "AnthropicProvider": "domyn.agents.llm",
    "load_application_from_file": "domyn.agents.loader",
}

if TYPE_CHECKING:
    from domyn.agents.agents import Agent as Agent
    from domyn.agents.agents import ContextVisibility as ContextVisibility
    from domyn.agents.agents import DelegateAgent as DelegateAgent
    from domyn.agents.agents import HandoffScope as HandoffScope
    from domyn.agents.core import ApprovalType as ApprovalType
    from domyn.agents.core import BaseAgent as BaseAgent
    from domyn.agents.core import BaseEvent as BaseEvent
    from domyn.agents.core import ExecutionEventType as ExecutionEventType
    from domyn.agents.core import InteractionContext as InteractionContext
    from domyn.agents.core import Part as Part
    from domyn.agents.core import decorators as decorators
    from domyn.agents.llm import AnthropicProvider as AnthropicProvider
    from domyn.agents.llm import OpenAIProvider as OpenAIProvider
    from domyn.agents.llm import OpenRouterProvider as OpenRouterProvider
    from domyn.agents.llm import VLLMProvider as VLLMProvider
    from domyn.agents.loader import load_application_from_file as load_application_from_file
    from domyn.agents.runner import Runner as Runner
    from domyn.agents.runner.transport import register_transport as register_transport
    from domyn.agents.tools import FunctionTool as FunctionTool
    from domyn.agents.tools import MCPTool as MCPTool
    from domyn.agents.tools import StreamableFunctionTool as StreamableFunctionTool


def __getattr__(name: str) -> Any:
    if name == "decorators":
        return import_module("domyn.agents.decorators")
    if name == "register_transport":
        from domyn.agents.runner.transport import register_transport

        globals()[name] = register_transport
        return register_transport
    if name in _LAZY_EXPORTS:
        value = getattr(import_module(_LAZY_EXPORTS[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
