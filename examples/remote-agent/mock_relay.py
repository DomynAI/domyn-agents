"""A minimal in-memory relay, enough for the remote example. Not for production.

A relay forwards messages between the clients connected to the same channel (the
``channel-id`` header): here, the runner of ``config.yaml`` and the agent ``remote_agent.py``
serves. Both sides close and reopen their connection during a turn, so the relay keeps a
log of each channel and sends a client that connects what it missed: the messages after
the event id in its ``last-event-id`` header, or all of them when it sends none. Both
sides drop the messages they have already seen.
"""

import asyncio
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field

import websockets.exceptions
from websockets.asyncio.server import ServerConnection


@dataclass
class _Channel:
    clients: list[ServerConnection] = field(default_factory=list)
    # (sender, event id, message). The sender is the conversation id when the client
    # sends one, so the runner reconnecting is not sent its own messages back.
    log: deque[tuple[str, str | None, str]] = field(default_factory=lambda: deque(maxlen=1000))
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


_channels: dict[str, _Channel] = defaultdict(_Channel)


def _event_id(message: str | bytes) -> str | None:
    """The ``event_id`` of an event, sent bare or inside a ``{"payload": ...}`` envelope."""
    try:
        data = json.loads(message)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    event = data["payload"] if isinstance(data.get("payload"), dict) else data
    event_id = event.get("event_id")
    return event_id if isinstance(event_id, str) else None


async def relay_handler(ws: ServerConnection) -> None:
    """Add the client to its channel, send it what it missed, then forward its messages."""
    headers = ws.request.headers if ws.request else {}
    channel = _channels[headers.get("channel-id", "default")]
    sender = headers.get("conversation-id") or f"client-{id(ws)}"
    cursor = headers.get("last-event-id")

    async with channel.lock:
        missed = [entry for entry in channel.log if entry[0] != sender]
        ids = [event_id for _, event_id, _ in missed]
        if cursor in ids:
            missed = missed[ids.index(cursor) + 1 :]
        for _, _, message in missed:
            await ws.send(message)
        channel.clients.append(ws)

    try:
        async for message in ws:
            async with channel.lock:
                channel.log.append((sender, _event_id(message), str(message)))
                peers = [client for client in channel.clients if client is not ws]
                # A peer closing mid-send must not end the sender's connection.
                await asyncio.gather(
                    *(peer.send(message) for peer in peers), return_exceptions=True
                )
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        channel.clients.remove(ws)
