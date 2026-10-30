"""Tests for build_oauth_provider's token storage, callback port, the
serialization of concurrent authorization round trips sharing a port, and the
patches to the mcp SDK's token parsing and scope selection."""

import asyncio
from types import SimpleNamespace
from urllib.parse import parse_qsl, urlparse

import httpx2
import mcp.client.auth.oauth2 as mcp_oauth2
import pytest
from key_value.aio.stores.filetree import FileTreeStore
from key_value.aio.stores.memory import MemoryStore
from mcp.client.auth import AuthorizationCodeResult
from mcp.shared.auth import OAuthMetadata, OAuthToken
from pydantic import ValidationError

from domyn.agents.tools.mcp.oauth_providers import (
    DEFAULT_OAUTH_CALLBACK_PORT,
    OAuthProviderConfig,
    _lock_for_port,
    _redirect_port,
    build_oauth_provider,
)


def test_local_type_uses_filetree_storage_on_disk(tmp_path):
    storage_dir = tmp_path / "mcp_oauth"
    config = OAuthProviderConfig(name="p", type="local", storage_dir=str(storage_dir))

    provider = build_oauth_provider(config, "https://mcp.example.com/mcp")

    assert isinstance(provider.token_storage_adapter._key_value_store, FileTreeStore)
    assert storage_dir.is_dir()


def test_in_memory_type_uses_in_process_memory_storage(tmp_path):
    storage_dir = tmp_path / "mcp_oauth"
    config = OAuthProviderConfig(name="p", type="in_memory", storage_dir=str(storage_dir))

    provider = build_oauth_provider(config, "https://mcp.example.com/mcp")

    assert isinstance(provider.token_storage_adapter._key_value_store, MemoryStore)
    assert not storage_dir.exists()


def test_unknown_type_raises():
    with pytest.raises(ValidationError, match="'local' or 'in_memory'"):
        OAuthProviderConfig(name="p", type="bogus")


def test_callback_port_defaults_to_the_fixed_constant():
    config = OAuthProviderConfig(name="p")

    provider = build_oauth_provider(config, "https://mcp.example.com/mcp")

    assert provider.redirect_port == DEFAULT_OAUTH_CALLBACK_PORT


def test_explicit_callback_port_is_passed_through():
    config = OAuthProviderConfig(name="p", callback_port=51234)

    provider = build_oauth_provider(config, "https://mcp.example.com/mcp")

    assert provider.redirect_port == 51234


def test_redirect_port_reads_the_configured_redirect_uri():
    config = OAuthProviderConfig(name="p", callback_port=51299)
    provider = build_oauth_provider(config, "https://mcp.example.com/mcp")

    assert _redirect_port(provider.context) == 51299


def test_redirect_port_returns_none_without_redirect_uris():
    context = SimpleNamespace(client_metadata=SimpleNamespace(redirect_uris=None))

    assert _redirect_port(context) is None


@pytest.mark.asyncio
async def test_lock_for_port_serializes_concurrent_holders_of_the_same_port():
    active = 0
    max_active = 0
    order: list[str] = []

    async def holder(name: str) -> None:
        nonlocal active, max_active
        async with _lock_for_port(60000):
            order.append(f"{name}-start")
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.02)
            active -= 1
            order.append(f"{name}-end")

    await asyncio.gather(holder("a"), holder("b"))

    assert max_active == 1
    assert order in (
        ["a-start", "a-end", "b-start", "b-end"],
        ["b-start", "b-end", "a-start", "a-end"],
    )


@pytest.mark.asyncio
async def test_lock_for_port_does_not_serialize_different_ports():
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocking_holder() -> str:
        async with _lock_for_port(60001):
            started.set()
            await release.wait()
            return "blocked"

    async def other_holder() -> str:
        async with _lock_for_port(60002):
            return "other"

    blocking_task = asyncio.create_task(blocking_holder())
    await started.wait()

    result = await asyncio.wait_for(other_holder(), timeout=1)

    assert result == "other"
    release.set()
    await blocking_task


def _wire_fake_handlers(provider, name: str, active: dict, order: list[str]) -> None:
    """Replace a provider's redirect/callback handlers with fakes that record
    concurrency and thread the real ``state`` through, so the patched
    ``_perform_authorization_code_grant`` can be exercised end to end without
    a real browser or network call."""
    provider.context.client_info = SimpleNamespace(client_id="test-client")
    captured_state: dict[str, str] = {}

    async def redirect_handler(url: str) -> None:
        order.append(f"{name}-redirect")
        active["count"] += 1
        active["max"] = max(active["max"], active["count"])
        captured_state["state"] = dict(parse_qsl(urlparse(url).query))["state"]
        await asyncio.sleep(0.02)

    async def callback_handler() -> AuthorizationCodeResult:
        order.append(f"{name}-callback")
        await asyncio.sleep(0.02)
        active["count"] -= 1
        order.append(f"{name}-done")
        return AuthorizationCodeResult(code="test-code", state=captured_state["state"])

    provider.context.redirect_handler = redirect_handler
    provider.context.callback_handler = callback_handler


@pytest.mark.asyncio
async def test_authorization_round_trip_is_serialized_across_a_shared_port():
    """The bug this guards against: locking only callback_handler let two
    flows' browser tabs both be open at once, so whichever one the user
    completed first could land on the wrong flow's listener (a real
    ``state`` mismatch, observed live). The round trip — redirect_handler
    through callback_handler — must be one atomic unit per shared port."""
    config_a = OAuthProviderConfig(name="a", callback_port=51300)
    config_b = OAuthProviderConfig(name="b", callback_port=51300)
    provider_a = build_oauth_provider(config_a, "https://mcp-a.example.com/mcp")
    provider_b = build_oauth_provider(config_b, "https://mcp-b.example.com/mcp")

    active = {"count": 0, "max": 0}
    order: list[str] = []
    _wire_fake_handlers(provider_a, "a", active, order)
    _wire_fake_handlers(provider_b, "b", active, order)

    results = await asyncio.gather(
        provider_a._perform_authorization_code_grant(),
        provider_b._perform_authorization_code_grant(),
    )

    assert active["max"] == 1
    assert all(auth_code == "test-code" for auth_code, _code_verifier in results)
    assert order in (
        ["a-redirect", "a-callback", "a-done", "b-redirect", "b-callback", "b-done"],
        ["b-redirect", "b-callback", "b-done", "a-redirect", "a-callback", "a-done"],
    )


@pytest.mark.asyncio
async def test_authorization_round_trips_on_different_ports_are_not_serialized():
    config_a = OAuthProviderConfig(name="a", callback_port=51301)
    config_b = OAuthProviderConfig(name="b", callback_port=51302)
    provider_a = build_oauth_provider(config_a, "https://mcp-a.example.com/mcp")
    provider_b = build_oauth_provider(config_b, "https://mcp-b.example.com/mcp")

    active = {"count": 0, "max": 0}
    order: list[str] = []
    _wire_fake_handlers(provider_a, "a", active, order)
    _wire_fake_handlers(provider_b, "b", active, order)

    await asyncio.gather(
        provider_a._perform_authorization_code_grant(),
        provider_b._perform_authorization_code_grant(),
    )

    assert active["max"] == 2


# The patches below replace parts of the mcp SDK by name, so an SDK upgrade
# that renames or re-signs one of them leaves the patch unused or broken
# without failing anything else. These go through the SDK's own names.


def _token_response(body: bytes, status_code: int = 200) -> httpx2.Response:
    return httpx2.Response(
        status_code, content=body, request=httpx2.Request("POST", "https://as.example.com/token")
    )


@pytest.mark.asyncio
async def test_a_form_encoded_token_response_is_read():
    token = await mcp_oauth2.handle_token_response_scopes(
        _token_response(b"access_token=abc&token_type=bearer&scope=repo")
    )

    assert token.access_token == "abc"
    assert token.scope == "repo"


@pytest.mark.asyncio
async def test_a_form_encoded_refresh_keeps_what_it_leaves_out():
    provider = build_oauth_provider(
        OAuthProviderConfig(name="p", type="in_memory"), "https://mcp.example.com/mcp"
    )
    provider.context.current_tokens = OAuthToken(
        access_token="old", token_type="Bearer", refresh_token="keep", scope="read"
    )

    refreshed = await provider._handle_refresh_response(
        _token_response(b"access_token=new&token_type=bearer")
    )

    assert refreshed
    tokens = provider.context.current_tokens
    assert (tokens.access_token, tokens.refresh_token, tokens.scope) == ("new", "keep", "read")


@pytest.mark.asyncio
async def test_a_failed_refresh_clears_the_tokens():
    provider = build_oauth_provider(
        OAuthProviderConfig(name="p", type="in_memory"), "https://mcp.example.com/mcp"
    )
    provider.context.current_tokens = OAuthToken(access_token="old", token_type="Bearer")

    refreshed = await provider._handle_refresh_response(_token_response(b"", status_code=400))

    assert not refreshed
    assert provider.context.current_tokens is None


_SERVER_METADATA = OAuthMetadata(
    issuer="https://as.example.com",
    authorization_endpoint="https://as.example.com/authorize",
    token_endpoint="https://as.example.com/token",
    scopes_supported=["read", "write", "offline_access"],
)


class _CallSite:
    """Calls the scope selection as `async_auth_flow` does: from a method
    whose `self` has the provider's context."""

    def __init__(self, context: object) -> None:
        self.context = context

    def select(self) -> str | None:
        return mcp_oauth2.get_client_metadata_scopes(
            None, None, _SERVER_METADATA, ["authorization_code", "refresh_token"]
        )


def test_the_configured_scopes_win_over_the_servers():
    provider = build_oauth_provider(
        OAuthProviderConfig(name="p", type="in_memory", scopes=["read"]),
        "https://mcp.example.com/mcp",
    )

    assert _CallSite(provider.context).select() == "read"


def test_without_configured_scopes_the_sdk_selects_them():
    assert _CallSite(SimpleNamespace()).select() == "read write offline_access"
