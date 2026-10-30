"""InputChannel Protocol — abstracts the transport that delivers user input and receives agent output."""

from collections.abc import AsyncGenerator
from typing import Protocol

from domyn.agents.core import BaseEvent


class InputChannel(Protocol):
    """Bidirectional channel between the outside world and AgentServer.

    A channel owns the transport (WebSocket, HTTP, Telegram, …).  It converts
    raw inbound frames into ``(conversation_id, BaseEvent)`` tuples and accepts
    outbound ``BaseEvent`` instances to deliver back.

    Two send variants are required:

    * ``send_event`` — best-effort; may silently drop if the internal buffer is full.
      Use for trace events (RESPONSE partial, LLM_START, TOOL_UPDATE, …).
    * ``send_event_priority`` — blocking; must not be dropped.
      Use for TOOL_START (part of the platform-tool round-trip) and the final
      AGENT_END so the conversation always terminates cleanly.
    """

    async def send_event(self, conversation_id: str, event: BaseEvent) -> None:
        """Best-effort send — may drop if internal buffer is full."""
        ...

    async def send_event_priority(self, conversation_id: str, event: BaseEvent) -> None:
        """Guaranteed send — blocks until the event is accepted by the transport."""
        ...

    def receive(self) -> AsyncGenerator[tuple[str, BaseEvent], None]:
        """Yield ``(conversation_id, event)`` for every significant inbound event.

        Significant event types that must be yielded:
          * ``AGENT_START``   — new invocation request
          * ``TOOL_FEEDBACK`` — approval reply
          * ``TOOL_END``      — platform-tool result (resolves a pending DelegateTool Future)
          * ``TOOL_ERROR``    — platform-tool failure (rejects a pending DelegateTool Future)
          * ``AGENT_END``     — cancellation signal from the platform

        The channel MAY yield other event types; ``AgentServer`` ignores unknown ones.
        The generator must run indefinitely (reconnecting as needed) until ``close()``
        is called or the process exits.
        """
        ...

    async def close(self) -> None:
        """Gracefully stop the channel — no new events after this returns."""
        ...
