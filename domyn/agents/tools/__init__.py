"""Tools package for domyn agents framework."""

from enum import Enum
from functools import reduce
from operator import or_

from domyn.agents.tools.base import BaseTool, ConfigParam, Param, Tool, ToolResult
from domyn.agents.tools.delegate_tool import DelegateTool
from domyn.agents.tools.function_tool import FunctionTool
from domyn.agents.tools.mcp.mcp_tool import MCPTool
from domyn.agents.tools.rest_function_tool import RestFunctionTool
from domyn.agents.tools.streamable_function_tool import StreamableFunctionTool, StreamableToolResult


class ToolType(Enum):
    FUNCTION = "function"
    STREAMABLE_FUNCTION = "streamable_function"
    REST = "rest"
    MCP = "mcp"
    DELEGATE = "delegate"


ToolTypes = reduce(or_, (BaseTool, *BaseTool._registry.values()))


__all__ = [
    "ConfigParam",
    "DelegateTool",
    "FunctionTool",
    "MCPTool",
    "Param",
    "RestFunctionTool",
    "StreamableFunctionTool",
    "StreamableToolResult",
    "Tool",
    "ToolResult",
    "ToolType",
]
