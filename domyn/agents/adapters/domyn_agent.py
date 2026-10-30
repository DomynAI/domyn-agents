"""DomynAgentAdapter — FrameworkAdapter that drives the native Domyn Runner.

WebSocket-specific code lives in the channel, not here.  The adapter:

  - Accepts ``(event, emit, emit_priority)`` per invocation.
  - Drives ``Runner.run()`` and forwards emitted ``BaseEvent`` objects via *emit*
    (best-effort) or *emit_priority* (guaranteed, used for TOOL_START and AGENT_END).
  - Wires ``DelegateTool.relay`` so platform tool round-trips work:
      DelegateTool.execute() → _make_tool_call() → emit_priority(TOOL_START)
      → suspend on Future → on_tool_response(TOOL_END) → Future resolved → return.
  - Manages in-flight task concurrency with a semaphore and a busy-buffer.
"""

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

from domyn.agents.adapters._invocation import InvocationCtx, invocation_ctx, spawn_invocation
from domyn.agents.agents import Agent
from domyn.agents.core import BaseEvent, ExecutionEventType, ToolAction
from domyn.agents.interfaces import EmitFn, EmitPriorityFn
from domyn.agents.logger import get_logger
from domyn.agents.runner import Runner
from domyn.agents.tools import DelegateTool

logger = get_logger()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_delegate_tools(agent: Agent) -> list[DelegateTool]:
    """Walk the agent tree and collect every DelegateTool without a wrapped callable."""
    tools: list[DelegateTool] = []
    visited: set[str] = set()

    def _walk(a: Agent) -> None:
        if a.name in visited:
            return
        visited.add(a.name)
        for tool in getattr(a, "tools", []):
            if isinstance(tool, DelegateTool) and tool.wrapped is None:
                tools.append(tool)
        for sub in getattr(a, "sub_agents", []):
            if isinstance(sub, Agent):
                _walk(sub)

    _walk(agent)
    return tools


# ---------------------------------------------------------------------------
# DomynAgentAdapter
# ---------------------------------------------------------------------------


class DomynAgentAdapter:
    """FrameworkAdapter that drives a native Domyn Agent via ``Runner.run()``.

    Parameters
    ----------
    root:
        Root agent of the agent tree to execute.
    application_name:
        Identifies this application in traces and conversation history.
    user_id:
        User identity forwarded to the Runner.
    max_concurrency:
        Maximum number of simultaneously active invocations.
    busy_buffer:
        Additional invocations queued behind the semaphore before WorkerBusy
        rejection kicks in.  Defaults to *max_concurrency*.
    """

    def __init__(
        self,
        root: Agent,
        application_name: str = "domyn-exposed",
        user_id: str = "platform",
        max_concurrency: int = 8,
        busy_buffer: int | None = None,
    ) -> None:
        self._root = root
        self._application_name = application_name
        self._user_id = user_id
        self._max_concurrency = max_concurrency
        self._busy_buffer = busy_buffer if busy_buffer is not None else max_concurrency

        self._runner = Runner()
        self._semaphore = asyncio.Semaphore(max_concurrency)

        # In-flight task registry (keyed by request_id == event_id of AGENT_START)
        self._in_flight: dict[str, asyncio.Task[None]] = {}
        # Separate index by conversation_id for cancel routing
        self._conversation_tasks: dict[str, asyncio.Task[None]] = {}

        # Platform-tool round-trip state
        self._platform_calls: dict[str, asyncio.Future[Any]] = {}

        # Wire relay onto every DelegateTool in the agent tree
        self._delegate_tools = _collect_delegate_tools(root)
        self._delegate_tool_names = frozenset(t.tool_name for t in self._delegate_tools)
        for tool in self._delegate_tools:
            tool.relay = self._make_tool_call
        if self._delegate_tools:
            logger.info(
                "DomynAgentAdapter: wired delegate tools: %s",
                ", ".join(t.tool_name for t in self._delegate_tools),
            )

        self.events_dropped: int = 0

    # ------------------------------------------------------------------
    # Periodic platform-tool rediscovery
    # ------------------------------------------------------------------

    async def refresh_delegate_tools(
        self,
        api_key: str | None,
        space_id: str | None,
        channel_id: str | None,
        base_url: str | None,
        configuration_id: str | None,
        attach_new_tools: bool = True,
    ) -> None:
        """Re-run platform tool discovery against the live agent tree.

        Unlike the one-time call at startup, a failed fetch here leaves the
        existing tool set untouched (``drop_on_fetch_failure=False``) — a
        transient network blip must not wipe out every platform tool from a
        running worker. Newly attached DelegateTool instances are wired with
        this adapter's relay exactly as ``__init__`` wires the initial set.
        """
        from domyn.agents.integrations.domyn import populate_delegate_tool_schemas

        await populate_delegate_tool_schemas(
            root=self._root,
            api_key=api_key,
            space_id=space_id,
            channel_id=channel_id,
            base_url=base_url,
            configuration_id=configuration_id,
            attach_new_tools=attach_new_tools,
            drop_on_fetch_failure=False,
        )

        fresh = _collect_delegate_tools(self._root)
        for tool in fresh:
            if tool.relay is None:
                tool.relay = self._make_tool_call
        self._delegate_tools = fresh
        self._delegate_tool_names = frozenset(t.tool_name for t in fresh)
        if fresh:
            logger.info(
                f"DomynAgentAdapter: refreshed delegate tools: "
                f"{', '.join(t.tool_name for t in fresh)}"
            )

    # ------------------------------------------------------------------
    # FrameworkAdapter contract
    # ------------------------------------------------------------------

    async def invoke(
        self,
        event: BaseEvent,
        emit: EmitFn,
        emit_priority: EmitPriorityFn,
    ) -> None:
        """Schedule execution of one AGENT_START event.

        Returns immediately after creating the asyncio.Task.
        """
        request_id = event.event_id or ""
        if not request_id:
            logger.warning("DomynAgentAdapter.invoke: AGENT_START has no event_id; skipping")
            return

        if len(self._in_flight) >= self._max_concurrency + self._busy_buffer:
            # event_id is per-frame unique; let BaseEvent auto-generate it. Stamping the
            # inbound request's id here makes the delegating transport dedup this frame as
            # a replay of the AGENT_START it already saw and drop it silently, so the turn
            # dies on a read timeout instead. The correlation rides in metadata.
            busy = BaseEvent(
                event_type=ExecutionEventType.ERROR_OCCURRED,
                author=event.author,
                timestamp=datetime.now(UTC).replace(tzinfo=None),
                interaction_id=event.interaction_id,
                error_code="WorkerBusy",
                error_message="max concurrency reached",
                metadata={"request_id": request_id},
            )
            await emit_priority(busy)
            return

        conversation_id = event.conversation_id or request_id

        async def _run_body() -> None:
            await self._run_agent(event, request_id, conversation_id, emit, emit_priority)

        spawn_invocation(
            request_id=request_id,
            conversation_id=conversation_id,
            ctx=InvocationCtx(
                emit_priority=emit_priority,
                author=event.author,
                conversation_id=conversation_id,
                turn_id=event.turn_id,
                interaction_id=event.interaction_id,
            ),
            semaphore=self._semaphore,
            conversation_tasks=self._conversation_tasks,
            in_flight=self._in_flight,
            run_body=_run_body,
        )

    def on_tool_response(self, event: BaseEvent) -> None:
        """Resolve (or reject) the Future suspended on a DelegateTool call."""
        call_id = getattr(event.action, "call_id", None) if event.action else None
        if call_id is None:
            return
        future = self._platform_calls.pop(call_id, None)
        if future is None or future.done():
            return
        if event.event_type == ExecutionEventType.TOOL_ERROR:
            future.set_exception(
                RuntimeError(event.error_message or f"platform tool error ({event.error_code})")
            )
        else:
            observation = getattr(event.action, "observation", None) if event.action else None
            future.set_result(observation)

    def on_cancel(self, event: BaseEvent) -> None:
        """Cancel the in-flight task for the given conversation_id."""
        conv_id = event.conversation_id
        if not conv_id:
            logger.debug("DomynAgentAdapter.on_cancel: no conversation_id — cancelling all")
            for t in list(self._in_flight.values()):
                t.cancel()
            return
        task: asyncio.Task[None] | None = self._conversation_tasks.get(conv_id)
        if task is None or task.done():
            logger.debug("DomynAgentAdapter.on_cancel: no in-flight task for conv=%s", conv_id)
            return
        logger.info("DomynAgentAdapter.on_cancel: cancelling conv=%s", conv_id)
        task.cancel()

    async def close(self) -> None:
        """Cancel all in-flight tasks and reject pending DelegateTool Futures."""
        for fut in self._platform_calls.values():
            if not fut.done():
                fut.set_exception(RuntimeError("DomynAgentAdapter shutting down"))
        self._platform_calls.clear()
        self._conversation_tasks.clear()
        for task in list(self._in_flight.values()):
            task.cancel()
        if self._in_flight:
            await asyncio.gather(*self._in_flight.values(), return_exceptions=True)

    # ------------------------------------------------------------------
    # Internal — invocation execution
    # ------------------------------------------------------------------

    async def _run_agent(
        self,
        agent_start: BaseEvent,
        request_id: str,
        conversation_id: str,
        emit: EmitFn,
        emit_priority: EmitPriorityFn,
    ) -> None:
        """Run the Runner for one invocation and forward events to the channel."""
        done: BaseEvent
        try:
            _saw_terminal = False
            async for ev in self._runner.run(
                application_name=self._application_name,
                user_id=self._user_id,
                user_input=agent_start,
                root=self._root,
                conversation_id=conversation_id,
            ):
                # Track whether the runner itself emitted a terminal event so we
                # skip the synthetic AGENT_END below.
                if (
                    (ev.event_type == ExecutionEventType.AGENT_END and ev.author == self._root.name)
                    or (ev.event_type == ExecutionEventType.RESPONSE and not ev.is_partial)
                    or ev.event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK
                ):
                    _saw_terminal = True

                # Suppress the runner's observability TOOL_START/TOOL_END for
                # DelegateTool names. _make_tool_call() runs the whole round trip
                # itself under a fresh call_id: it sends its own TOOL_START, and
                # the platform answers with the TOOL_END that resolves it. The
                # runner's pair carries the *LLM's* tool_call id instead — an id
                # the platform never issued — so forwarding them shows the relay a
                # duplicate TOOL_START and then a TOOL_END for a call it cannot
                # match. TOOL_ERROR is deliberately still forwarded: that one
                # reports a local failure (e.g. the relay round trip itself
                # failing), which is information the platform does not already have.
                if (
                    ev.event_type in (ExecutionEventType.TOOL_START, ExecutionEventType.TOOL_END)
                    and getattr(ev.action, "name", None) in self._delegate_tool_names
                ):
                    continue

                # Must never be dropped: losing it strands the turn forever — the
                # human never sees the approval request, and nothing else will
                # ever re-prompt them. emit() is best-effort (drops when the
                # channel's outbound queue is full); emit_priority() blocks
                # instead, the same guarantee already given to TOOL_START/AGENT_END.
                if ev.event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK:
                    await emit_priority(ev)
                    continue

                await emit(ev)

            if _saw_terminal:
                return

            done = BaseEvent(
                event_type=ExecutionEventType.AGENT_END,
                author=self._root.name,
                timestamp=datetime.now(UTC).replace(tzinfo=None),
                interaction_id=agent_start.interaction_id,
                conversation_id=conversation_id,
                turn_id=agent_start.turn_id,
                metadata={"request_id": request_id},
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "DomynAgentAdapter: invocation failed request_id=%s: %s",
                request_id,
                exc,
                exc_info=True,
            )
            done = BaseEvent(
                event_type=ExecutionEventType.AGENT_END,
                author=self._root.name,
                timestamp=datetime.now(UTC).replace(tzinfo=None),
                interaction_id=agent_start.interaction_id,
                conversation_id=conversation_id,
                turn_id=agent_start.turn_id,
                error_code=type(exc).__name__,
                error_message=str(exc),
                metadata={"request_id": request_id},
            )

        done = done.model_copy(update={"timestamp": datetime.now(UTC).replace(tzinfo=None)})
        await emit_priority(done)

    # ------------------------------------------------------------------
    # Internal — DelegateTool round-trip
    # ------------------------------------------------------------------

    async def _make_tool_call(self, name: str, kwargs: dict[str, Any]) -> Any:
        """Send TOOL_START to the relay and suspend until TOOL_END / TOOL_ERROR.

        Called by ``DelegateTool.execute()`` via ``tool.relay``.  Correlation IDs
        are read from the shared ``invocation_ctx`` ContextVar set by
        ``invoke()`` via ``spawn_invocation()``, not from
        ``asyncio.current_task()``: ``Runner._merge_with_injections`` drives
        every step of the turn (including the step that reaches this
        DelegateTool) in a fresh ``asyncio.Task`` via ``asyncio.ensure_future``,
        so the task executing here is never the one ``invoke()`` created. The
        ContextVar is still found because a ``contextvars.Context`` is copied
        into every child task spawned from a task carrying it, and concurrent
        invocations each get their own copy — see ``InvocationCtx``.
        """
        ctx = invocation_ctx.get()

        if ctx is None:
            raise RuntimeError(
                "DomynAgentAdapter._make_tool_call: no invocation context for current task — "
                "cannot send TOOL_START. Is the DelegateTool being called outside of invoke()?"
            )

        call_id = str(uuid.uuid4())
        tool_start = BaseEvent(
            event_type=ExecutionEventType.TOOL_START,
            author=ctx.author,
            timestamp=datetime.now(UTC).replace(tzinfo=None),
            conversation_id=ctx.conversation_id,
            interaction_id=ctx.interaction_id,
            turn_id=ctx.turn_id,
            action=ToolAction(name=name, parameters=kwargs, call_id=call_id),
        )

        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._platform_calls[call_id] = future

        try:
            await ctx.emit_priority(tool_start)
        except Exception:
            self._platform_calls.pop(call_id, None)
            raise

        try:
            return await future
        finally:
            self._platform_calls.pop(call_id, None)
