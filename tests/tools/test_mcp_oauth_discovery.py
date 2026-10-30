"""Eager OAuth metadata discovery asks a server once, not once per client built.

Many servers publish nothing at their own address (the authorization server is
named by the protected-resource metadata, found on the first 401). The 404 is
then the answer, and used to come back with every tool initialized and every
call made.
"""

import asyncio
from unittest.mock import patch

import pytest

from domyn.agents.tools.mcp import oauth_providers as op
from domyn.agents.tools.mcp.oauth_providers import (
    DEFAULT_OAUTH_PROVIDER_CONFIG,
    build_oauth_provider,
    eagerly_discover_oauth_metadata,
)

URL = "https://mcp.example.com/mcp"


@pytest.fixture(autouse=True)
def forget_discoveries():
    op._discovery_retry_at.clear()
    op._discovery_locks.clear()


@pytest.fixture
def provider():
    return build_oauth_provider(DEFAULT_OAUTH_PROVIDER_CONFIG, URL)


class Asked:
    """Stands in for the network step, counting how often it is asked."""

    def __init__(self, finds: object = None, outcome: str = "empty") -> None:
        self.calls = 0
        self.finds = finds
        self.outcome = outcome

    async def __call__(self, provider, server_url) -> None:
        self.calls += 1
        await asyncio.sleep(0.01)
        if self.finds is not None:
            provider.context.oauth_metadata = self.finds
            return "found"
        return self.outcome


@pytest.mark.asyncio
async def test_an_empty_answer_is_not_asked_again(provider):
    asked = Asked()
    with patch.object(op, "_discover_oauth_metadata", asked):
        for _ in range(3):
            await eagerly_discover_oauth_metadata(provider, URL)
    assert asked.calls == 1


@pytest.mark.asyncio
async def test_callers_that_arrive_together_share_one_request(provider):
    asked = Asked()
    with patch.object(op, "_discover_oauth_metadata", asked):
        await asyncio.gather(*(eagerly_discover_oauth_metadata(provider, URL) for _ in range(10)))
    assert asked.calls == 1


@pytest.mark.asyncio
async def test_an_empty_answer_is_asked_again_after_the_retry_time(provider):
    asked = Asked()
    with (
        patch.object(op, "_discover_oauth_metadata", asked),
        patch.object(op, "DISCOVERY_RETRY_SECONDS", 0.0),
    ):
        await eagerly_discover_oauth_metadata(provider, URL)
        await eagerly_discover_oauth_metadata(provider, URL)
    assert asked.calls == 2


@pytest.mark.asyncio
async def test_what_was_found_is_kept_and_not_asked_again(provider):
    asked = Asked(finds=object())
    with patch.object(op, "_discover_oauth_metadata", asked):
        await eagerly_discover_oauth_metadata(provider, URL)
        await eagerly_discover_oauth_metadata(provider, URL)
    assert asked.calls == 1
    assert provider.context.oauth_metadata is not None


@pytest.mark.asyncio
async def test_another_address_is_asked_on_its_own(provider):
    other = build_oauth_provider(DEFAULT_OAUTH_PROVIDER_CONFIG, "https://other.example.com/mcp")
    asked = Asked()
    with patch.object(op, "_discover_oauth_metadata", asked):
        await eagerly_discover_oauth_metadata(provider, URL)
        await eagerly_discover_oauth_metadata(other, "https://other.example.com/mcp")
    assert asked.calls == 2


@pytest.mark.asyncio
async def test_a_new_provider_for_the_same_address_does_not_ask_again(provider):
    """A tool without an `oauth_provider` gets a new one for every client it builds."""
    asked = Asked()
    with patch.object(op, "_discover_oauth_metadata", asked):
        for _ in range(5):
            fresh = build_oauth_provider(DEFAULT_OAUTH_PROVIDER_CONFIG, URL)
            await eagerly_discover_oauth_metadata(fresh, URL)
    assert asked.calls == 1


@pytest.mark.asyncio
async def test_providers_arriving_together_for_one_address_share_one_request():
    asked = Asked()
    providers = [build_oauth_provider(DEFAULT_OAUTH_PROVIDER_CONFIG, URL) for _ in range(8)]
    with patch.object(op, "_discover_oauth_metadata", asked):
        await asyncio.gather(*(eagerly_discover_oauth_metadata(p, URL) for p in providers))
    assert asked.calls == 1


@pytest.mark.asyncio
async def test_a_failed_request_is_asked_again_sooner_than_an_empty_answer(provider):
    asked = Asked(outcome="failed")
    with (
        patch.object(op, "_discover_oauth_metadata", asked),
        patch.object(op, "DISCOVERY_FAILURE_RETRY_SECONDS", 0.0),
    ):
        await eagerly_discover_oauth_metadata(provider, URL)
        await eagerly_discover_oauth_metadata(provider, URL)
    assert asked.calls == 2
