from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio
from fastmcp import Client
from mcp.types import TextContent


@pytest_asyncio.fixture(autouse=True)
async def _clear_mcp_cache() -> None:
    from cashews import cache

    await cache.clear()


@pytest.fixture
def make_mock_response():
    """Factory fixture for MCP call_tool responses."""

    def _make(
        text: str = "Result",
        *,
        meta: dict[str, Any] | None = None,
        structured_content: Any = None,
        content: list | None = None,
    ) -> Mock:
        response = Mock()
        response.content = content if content is not None else [TextContent(type="text", text=text)]
        response.structured_content = structured_content
        response.meta = meta
        return response

    return _make


@pytest.fixture
def make_mock_client():
    """Factory fixture for async MCP clients.

    Returns a client with __aenter__/__aexit__ pre-wired.
    Accepts either a response object or a side_effect for call_tool.
    """

    def _make(
        response: Any = None,
        *,
        side_effect: Exception | None = None,
    ) -> AsyncMock:
        client = AsyncMock(spec=Client)
        if side_effect is not None:
            client.call_tool = AsyncMock(side_effect=side_effect)
        elif response is not None:
            client.call_tool = AsyncMock(return_value=response)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        return client

    return _make
