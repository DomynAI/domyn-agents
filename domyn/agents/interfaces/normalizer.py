"""InboundNormalizer and ResponseFormatter Protocols."""

from typing import Any, Protocol

from domyn.agents.core import BaseEvent


class InboundNormalizer(Protocol):
    """Translate a raw channel message into a normalised ``BaseEvent``.

    Each channel has its own native message format (Telegram ``Message``,
    HTTP JSON body, MIME email, …).  An ``InboundNormalizer`` maps that format
    to a ``BaseEvent(USER_INPUT)`` so ``AgentServer`` can hand it to the adapter
    without knowing anything about the channel.

    When the channel already delivers ``BaseEvent`` objects directly (e.g. the
    relay WebSocket channel), no normalizer is needed.
    """

    def normalize(self, raw: Any) -> BaseEvent:
        """Return a ``BaseEvent`` (typically ``USER_INPUT``) from *raw*.

        May raise ``ValueError`` when the raw message cannot be normalised.
        """
        ...


class ResponseFormatter(Protocol):
    """Format an outbound ``BaseEvent`` into text suitable for the channel.

    Different channels expect different text formats:

    * Markdown   — Telegram, Discord, Slack
    * HTML       — email
    * plain text — SMS, terminal
    * JSON       — REST API (the serialised ``BaseEvent``)

    Returning ``None`` suppresses the event (e.g. TOOL_START should not be
    shown verbatim on Telegram).
    """

    def format(self, event: BaseEvent) -> str | None:
        """Return a formatted string or ``None`` to suppress the event."""
        ...
