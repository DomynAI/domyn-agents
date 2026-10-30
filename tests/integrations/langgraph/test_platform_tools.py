import asyncio

import pytest

from domyn.agents.integrations.langgraph.domyn_platform import (
    aget_platform_tools,
    get_platform_tools,
)

_TOOL_DEFS = [
    {
        "name": "web_search",
        "description": "Search the web.",
        "parameters": [{"name": "query", "type": "str", "is_required": True}],
    }
]


def test_get_platform_tools_builds_a_registry_from_the_platform(platform):
    platform.serve(_TOOL_DEFS)

    registry = get_platform_tools(
        api_key="key", space_id="space", channel_id="chan", base_url="platform.example.com"
    )

    assert registry.list_tools() == ["web_search"]
    settings = platform.calls[-1]["settings"]
    assert (settings.api_key, settings.space_id, settings.channel_id) == ("key", "space", "chan")
    assert settings.base_url == "platform.example.com"


def test_get_platform_tools_works_from_inside_a_running_loop(platform):
    platform.serve(_TOOL_DEFS)

    async def main():
        return get_platform_tools(
            api_key="key", space_id="space", channel_id="chan", base_url="platform.example.com"
        )

    assert asyncio.run(main()).list_tools() == ["web_search"]


@pytest.mark.asyncio
async def test_aget_platform_tools_awaits_the_platform(platform):
    platform.serve(_TOOL_DEFS)

    registry = await aget_platform_tools(
        api_key="key", space_id="space", channel_id="chan", base_url="platform.example.com"
    )

    assert registry.list_tools() == ["web_search"]


def test_get_platform_tools_propagates_a_platform_failure(platform):
    platform.fail(PermissionError("403"))

    with pytest.raises(PermissionError):
        get_platform_tools(api_key="key", space_id="space", channel_id="chan", base_url="h")


@pytest.mark.parametrize("missing", ["api_key", "space_id", "channel_id"])
def test_get_platform_tools_needs_credentials(monkeypatch, missing):
    for var in ("DOMYN_API_KEY", "DOMYN_SPACE_ID", "DOMYN_CHANNEL_ID"):
        monkeypatch.delenv(var, raising=False)
    args = {"api_key": "k", "space_id": "s", "channel_id": "c", "base_url": "h"}
    del args[missing]

    with pytest.raises(RuntimeError, match="required"):
        get_platform_tools(**args)
