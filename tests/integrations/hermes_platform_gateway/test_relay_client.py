"""Receiving-side behaviour of DomynRelayClient._consume: per-connection dedup
plus advancing the persistent resume cursor."""

import uuid

import pytest

from domyn.agents.core import BaseEvent, ExecutionEventType, Part, RelayMessage
from domyn.agents.integrations.hermes_platform_gateway.relay_client import DomynRelayClient
from domyn.agents.integrations.utils import resume_headers


def _frame(event_id: str) -> str:
    """A RelayMessage-wrapped AGENT_START JSON frame with a fixed event_id."""
    event = BaseEvent(
        event_type=ExecutionEventType.AGENT_START,
        author="platform",
        event_id=event_id,
        conversation_id=str(uuid.uuid4()),
        content=[Part(text="hi")],
    )
    return RelayMessage(payload=event).model_dump_json()


class _FakeWS:
    """Async-iterable websocket that yields a fixed list of raw frames then stops."""

    def __init__(self, frames: list[str]) -> None:
        self._frames = frames

    async def __aiter__(self):
        for raw in self._frames:
            yield raw


@pytest.mark.asyncio
async def test_consume_drops_within_connection_duplicates_and_advances_cursor():
    seen: list[str] = []

    async def on_event(payload, meta, config_id=None) -> None:
        seen.append(payload.event_id)

    client = DomynRelayClient("ws://x", {"channel-id": "c"}, on_event=on_event)
    cursor: dict[str, str | None] = {"last": None}

    # e1 is replayed (appears twice) alongside a new e2.
    ws = _FakeWS([_frame("e1"), _frame("e1"), _frame("e2")])
    await client._consume(ws, cursor)

    # The duplicate e1 is dropped: on_event fires once per unique event_id.
    assert seen == ["e1", "e2"]
    # The cursor advanced to the last processed event.
    assert cursor["last"] == "e2"


@pytest.mark.asyncio
async def test_dedup_persists_across_reconnect():
    seen: list[str] = []

    async def on_event(payload, meta, config_id=None) -> None:
        seen.append(payload.event_id)

    client = DomynRelayClient("ws://x", {"channel-id": "c"}, on_event=on_event)
    # Cursor holder owned by the reconnect loop → survives reconnects.
    cursor: dict[str, str | None] = {"last": None}

    # Connection 1 processes e1.
    await client._consume(_FakeWS([_frame("e1")]), cursor)
    assert cursor["last"] == "e1"
    # The next connect would resume after e1.
    assert resume_headers({"channel-id": "c"}, cursor["last"])["last-event-id"] == "e1"

    # Connection 2 replays e1 (relay at-least-once) plus a new e2. The shared seen
    # window persists across reconnects, so e1 is dropped and only e2 dispatched.
    await client._consume(_FakeWS([_frame("e1"), _frame("e2")]), cursor)
    assert seen == ["e1", "e2"]  # e1 fired once, across both connections
    assert cursor["last"] == "e2"


@pytest.mark.asyncio
async def test_consume_drops_replayed_own_sent_event():
    seen: list[str] = []

    async def on_event(payload, meta, config_id=None) -> None:
        seen.append(payload.event_id)

    client = DomynRelayClient("ws://x", {"channel-id": "c"}, on_event=on_event)
    cursor: dict[str, str | None] = {"last": None}

    # We sent "mine"; the relay window replays it back to us alongside a real
    # inbound "other". The echo is dropped, the genuine inbound is dispatched.
    client._seen.add("mine")
    await client._consume(_FakeWS([_frame("mine"), _frame("other")]), cursor)
    assert seen == ["other"]
    assert cursor["last"] == "other"


# ---------------------------------------------------------------------------
# run_config.config_id projection
# ---------------------------------------------------------------------------


def _frame_with_run_config(event_id: str, run_config: dict | None) -> str:
    event = BaseEvent(
        event_type=ExecutionEventType.AGENT_START,
        author="platform",
        event_id=event_id,
        conversation_id=str(uuid.uuid4()),
        content=[Part(text="hi")],
    )
    return RelayMessage(payload=event, run_config=run_config).model_dump_json()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "run_config,expected",
    [
        ({"config_id": "cfg-9"}, "cfg-9"),
        ({"config_id": "cfg-9", "other": "pii"}, "cfg-9"),  # only the id is forwarded
        ({}, None),
        (None, None),  # visibility withheld run_config → adapter falls back to env
    ],
)
async def test_consume_forwards_run_config_config_id(run_config, expected):
    seen: list[str | None] = []

    async def on_event(payload, meta, config_id=None) -> None:
        seen.append(config_id)

    client = DomynRelayClient("ws://x", {"channel-id": "c"}, on_event=on_event)
    cursor: dict[str, str | None] = {"last": None}

    await client._consume(_FakeWS([_frame_with_run_config("e1", run_config)]), cursor)
    assert seen == [expected]
