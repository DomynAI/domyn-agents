"""MCP server configuration for Domyn agents."""

from typing import Any, Literal

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from mcp import Tool
from mcp.client.auth import OAuthClientProvider
from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema

from domyn.agents.logger import get_logger
from domyn.agents.tools.mcp.oauth_providers import (
    DEFAULT_OAUTH_PROVIDER_CONFIG,
    build_oauth_provider,
    eagerly_discover_oauth_metadata,
)


class MCPServerStreamableHTTPConfig(BaseModel):
    """Configuration for an MCP server connection. Only supports the Streamable HTTP protocol.

    This class handles connection details and authentication for MCP servers.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., description="Unique name for this MCP server")
    url: str = Field(..., description="URL of the MCP server endpoint")
    tool_autodiscovery: bool = Field(
        default=True,
        description="Whether to enable tool autodiscovery for this MCP server",
    )

    # Authentication options
    use_oauth: bool = Field(
        default=False,
        description="Whether to use OAuth authentication",
    )
    oauth_provider: SkipJsonSchema[OAuthClientProvider] | None = Field(
        default=None,
        description="OAuth client provider for authentication. Scopes, "
        "client_name, and a pre-registered client_id/client_secret are "
        "configured on the oauth_providers: entry this comes from, not here.",
    )
    headers: dict[str, str] | None = Field(
        default=None,
        description="HTTP headers for authentication (e.g., {'Authorization': 'Bearer token'})",
    )

    # Optional settings
    timeout: float = Field(
        default=30.0,
        description="Request timeout in seconds",
    )
    protocol_mode: Literal["legacy", "auto"] = Field(
        default="legacy",
        description="How the client negotiates the MCP protocol at connect time. 'legacy' is "
        "the classic initialize handshake. 'auto' probes for the 2026-07-28 revision first, "
        "which makes servers that enforce Mcp-Param-* headers (e.g. GitHub's) reject calls "
        "this client does not mirror into headers.",
    )

    async def create_client(self) -> Client[StreamableHttpTransport]:
        """Create a fastmcp Client instance from this configuration.

        Returns:
            Configured Client instance
        """
        # Create OAuth config if needed
        auth = None
        if self.use_oauth:
            # Normally already resolved (and possibly shared with other
            # servers/tools) by ConfigurationV1.from_dict() — this fallback
            # only fires when this config is built directly in Python with
            # use_oauth=True and no oauth_provider=, bypassing the loader.
            provider = self.oauth_provider or build_oauth_provider(
                DEFAULT_OAUTH_PROVIDER_CONFIG, self.url
            )
            if provider.context.oauth_metadata is None:
                # Unset means this instance hasn't been through discovery yet
                # (fresh, or shared but not yet used).
                await eagerly_discover_oauth_metadata(provider, self.url)
            auth = provider

        # Create client with appropriate config
        if self.headers and not self.use_oauth:
            # Use config format for headers
            config = {
                "mcpServers": {
                    self.name: {
                        "transport": "http",
                        "url": self.url,
                        "headers": self.headers,
                    }
                }
            }
            return Client(config, auth=auth, timeout=self.timeout, mode=self.protocol_mode)  # type: ignore
        else:
            # Simple URL-based connection (with optional OAuth)
            return Client(self.url, auth=auth, timeout=self.timeout, mode=self.protocol_mode)  # type: ignore

    async def discover_tools(self) -> list[Tool]:
        """Discover all available tools from this MCP server.

        Returns:
            List of tool schemas from the server

        Raises:
            RuntimeError: If connection or discovery fails
        """
        logger = get_logger()

        client = await self.create_client()

        try:
            async with client:
                tools = await client.list_tools()
                logger.info(f"Discovered {len(tools)} tools from MCP server '{self.name}'")
                return tools
        except Exception as e:
            logger.error(
                f"Failed to discover tools from MCP server '{self.name}': {type(e).__name__}"
            )
            logger.debug(f"Failed to discover tools from MCP server '{self.name}': {e}")
            raise RuntimeError(f"Failed to discover tools from MCP server '{self.name}'") from e

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "MCPServerStreamableHTTPConfig":
        """Create MCPServerStreamableHTTPConfig from a dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            MCPServerStreamableHTTPConfig instance
        """
        return cls(**config)
