"""Per-turn configuration sync driven from AGENT_START.

Approval mode is only consulted DURING a turn, so the adapter reconciles it
right before dispatching to hermes. The id comes from the turn itself
(``run_config.config_id``, lifted out by the relay client), which is what lets a
configuration activated *after* this pod spawned take effect without a restart.

The sync must be awaited (a re-enabled approval requirement must not land after
the turn already ran unguarded) but must never block a turn when the config
endpoint is slow or broken.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from domyn.agents.core import AgentAction, BaseEvent, ExecutionEventType
from domyn.agents.integrations.hermes_platform_gateway.adapter import DomynPlatformAdapter


def _make_adapter(config_syncer=None):
    mock_client = MagicMock()
    mock_client.send_event = AsyncMock()
    adapter = DomynPlatformAdapter(
        config=MagicMock(),
        channel_id="chan-1",
        relay_client_factory=lambda _cb: mock_client,
        config_syncer=config_syncer,
        tool_rename_mapping={},
    )
    adapter._client = mock_client
    # Swallow the hermes dispatch — we only care about the sync that precedes it.
    adapter.handle_message = AsyncMock()
    return adapter


def _agent_start(event_id: str = "evt-1") -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.AGENT_START,
        author="u1",
        event_id=event_id,
        conversation_id="conv-1",
        interaction_id="int-1",
        turn_id="turn-1",
        action=AgentAction(name="invoke", parameters={"input": "hello"}),
    )


@pytest.mark.asyncio
async def test_agent_start_syncs_with_turn_config_id():
    syncer = MagicMock()
    syncer.sync = AsyncMock()
    adapter = _make_adapter(syncer)

    await adapter._on_event(_agent_start(), None, "cfg-9")

    syncer.sync.assert_awaited_once_with("cfg-9")
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_start_without_config_id_still_syncs():
    """None is forwarded — ConfigSyncer resolves the pod's spawn-time default."""
    syncer = MagicMock()
    syncer.sync = AsyncMock()
    adapter = _make_adapter(syncer)

    await adapter._on_event(_agent_start(), None, None)

    syncer.sync.assert_awaited_once_with(None)
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_sync_happens_before_dispatch():
    """A re-enabled approval requirement must be on disk before the agent runs."""
    order: list[str] = []
    syncer = MagicMock()

    async def _sync(_config_id):
        order.append("sync")

    syncer.sync = _sync
    adapter = _make_adapter(syncer)

    async def _dispatch(_msg):
        order.append("dispatch")

    adapter.handle_message = _dispatch

    await adapter._on_event(_agent_start(), None, "cfg-9")

    assert order == ["sync", "dispatch"]


@pytest.mark.asyncio
async def test_turn_is_dispatched_even_if_sync_raises():
    """Fail-open: a broken config sync must not wedge the conversation.

    ``ConfigSyncer.sync`` swallows its own errors, so this covers the call-site
    backstop — the guarantee holds for any syncer, not just ours.
    """
    syncer = MagicMock()
    syncer.sync = AsyncMock(side_effect=RuntimeError("boom"))
    adapter = _make_adapter(syncer)

    await adapter._on_event(_agent_start(), None, "cfg-9")

    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_turn_cancellation_is_not_swallowed():
    """A genuine cancellation must propagate, unlike an ordinary error."""
    syncer = MagicMock()
    syncer.sync = AsyncMock(side_effect=asyncio.CancelledError())
    adapter = _make_adapter(syncer)

    with pytest.raises(asyncio.CancelledError):
        await adapter._on_event(_agent_start(), None, "cfg-9")

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_syncer_configured_is_a_noop():
    adapter = _make_adapter(None)

    await adapter._on_event(_agent_start(), None, "cfg-9")

    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_turn_state_is_registered_before_sync():
    """A cancel arriving mid-fetch must find a live turn to pop."""
    captured: list[bool] = []
    syncer = MagicMock()
    adapter = _make_adapter(syncer)

    async def _sync(_config_id):
        captured.append(bool(adapter._turn_by_session))

    syncer.sync = _sync

    await adapter._on_event(_agent_start(), None, "cfg-9")

    assert captured == [True]
