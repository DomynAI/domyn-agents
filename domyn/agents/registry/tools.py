from domyn.agents.tools import FunctionTool


class ToolRegistry:
    """Registry for tool instances."""

    def __init__(self):
        self._tools: dict[str, FunctionTool] = {}

    def register_tool(self, tool: FunctionTool) -> None:
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> FunctionTool | None:
        _t = self._tools.get(name, None)
        if not _t:
            raise KeyError(f"Tool '{name}' not found in registry.")
        return _t

    def list_tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def list_tools(self) -> list[FunctionTool]:
        return list(self._tools.values())

    def clear(self) -> None:
        self._tools.clear()

    def __getitem__(self, name: str) -> FunctionTool:
        return self.get_tool(name)

    def __setitem__(self, name: str, tool: FunctionTool) -> None:
        """Register a tool instance."""
        self._tools[name] = tool


_tool_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    return _tool_registry
