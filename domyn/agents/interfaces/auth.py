"""ChannelAuthenticator Protocol — validates inbound requests."""

from typing import Any, Protocol


class ChannelAuthenticator(Protocol):
    """Validate an inbound request before it reaches the adapter.

    The *request* object is channel-specific (a ``BaseEvent`` for relay channels,
    an HTTP ``Request`` for REST channels, etc.).  The authenticator returns
    ``True`` when the request is allowed and ``False`` to silently drop it.

    ``AgentServer`` calls this before normalisation; if ``False`` is returned
    the event is discarded and no response is sent.
    """

    async def authenticate(self, request: Any) -> bool:
        """Return ``True`` to allow, ``False`` to reject."""
        ...
