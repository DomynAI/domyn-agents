"""FrameworkAdapter Protocol — abstracts the runtime that executes agent logic."""

from collections.abc import Awaitable, Callable
from typing import Protocol

from domyn.agents.core import BaseEvent

# Callable that delivers a BaseEvent back to the channel (best-effort).
EmitFn = Callable[[BaseEvent], Awaitable[None]]

# Callable that delivers a BaseEvent back to the channel (must not drop).
EmitPriorityFn = Callable[[BaseEvent], Awaitable[None]]


class FrameworkAdapter(Protocol):
    """Runtime that executes agent logic for a single invocation.

    ``AgentServer`` calls ``invoke`` for every ``AGENT_START`` or ``TOOL_FEEDBACK``
    it receives from the channel, and routes ``TOOL_END`` / ``TOOL_ERROR`` /
    ``AGENT_END`` to the matching handler methods.

    All three handler methods must be safe to call from the event loop without
    ``await`` (``on_tool_response`` and ``on_cancel`` are synchronous dispatch
    callbacks).
    """

    async def invoke(
        self,
        event: BaseEvent,
        emit: EmitFn,
        emit_priority: EmitPriorityFn,
    ) -> None:
        """Handle one ``AGENT_START`` or ``TOOL_FEEDBACK`` event.

        Must fire-and-forget (return promptly after scheduling the work) so that
        ``AgentServer`` can continue processing other inbound events while this
        invocation is running.

        *emit* should be used for trace / streaming events (best-effort).
        *emit_priority* should be used for TOOL_START and the final AGENT_END.
        """
        ...

    def on_tool_response(self, event: BaseEvent) -> None:
        """Route an inbound ``TOOL_END`` or ``TOOL_ERROR`` to the waiting Future.

        Called synchronously by ``AgentServer`` from the event loop.  Must not
        block or raise.
        """
        ...

    def on_cancel(self, event: BaseEvent) -> None:
        """Cancel the in-flight invocation identified by ``event.conversation_id``.

        Called synchronously by ``AgentServer`` when an ``AGENT_END`` cancellation
        arrives from the platform.  Must not block or raise.
        """
        ...

    async def close(self) -> None:
        """Shut down: cancel in-flight tasks and reject pending Futures."""
        ...
