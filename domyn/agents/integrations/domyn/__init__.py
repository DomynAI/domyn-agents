"""domyn integration: platform tool discovery for native domyn agents."""

from domyn.agents.integrations.domyn.domyn_platform import (
    fetch_delegate_tools,
    populate_delegate_tool_schemas,
)

__all__ = ["fetch_delegate_tools", "populate_delegate_tool_schemas"]
