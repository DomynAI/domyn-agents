"""LangGraphAdapter — FrameworkAdapter that drives a compiled LangGraph runnable.

WebSocket-specific code lives in the channel, not here.  The adapter:

  - Accepts ``(event, emit, emit_priority)`` per invocation.
  - Applies an optional ``InputMapper`` to translate ``AGENT_START.action.parameters``
    into the graph's input shape.
  - Runs ``graph.ainvoke()`` with a ``PlatformCallbackHandler`` that translates
    LangChain callbacks into ``BaseEvent`` objects delivered via *emit*.
  - Supports platform tool round-trips via ``PlatformRelay``: the relay is bound
    when the adapter is constructed and ``call_platform_tool`` uses a ``ContextVar`` so
    concurrent invocations each carry their own *emit_priority*.
"""

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

from domyn.agents.adapters._invocation import InvocationCtx, invocation_ctx, spawn_invocation
from domyn.agents.core import BaseEvent, ExecutionEventType, Part, ToolAction
from domyn.agents.interfaces import EmitFn, EmitPriorityFn
from domyn.agents.logger import get_logger

logger = get_logger()


def _extract_result_text(result: Any) -> str:
    """Pull plain text from a LangGraph ainvoke result dict."""
    if isinstance(result, dict):
        messages = result.get("messages")
        if messages:
            for msg in reversed(messages):
                content = getattr(msg, "content", None)
                if isinstance(content, str) and content:
                    return content
        result_field = result.get("result")
        if isinstance(result_field, str) and result_field:
            return result_field
    import json

    return json.dumps(result, default=str)


def _result_content(result: Any) -> list[Part]:
    """Build the AGENT_END content from a LangGraph ainvoke result."""
    if result is None:
        return []
    if isinstance(result, list) and result and all(isinstance(p, Part) for p in result):
        return list(result)
    return [Part(text=_extract_result_text(result))]


class LangGraphAdapter:
    """FrameworkAdapter that drives a compiled LangGraph graph.

    Parameters
    ----------
    runnable:
        A compiled LangGraph graph (anything with ``ainvoke``).
    input_mapper:
        Optional callable that translates ``AGENT_START.action.parameters`` (a
        ``dict``) into the graph's input type.  When omitted, the parameters dict
        is passed to ``ainvoke`` as-is.
    platform_relay:
        Optional ``PlatformRelay`` instance used for DelegateTool round-trips.
        When provided, the relay is bound to this adapter so ``PlatformTool``
        instances wired to it can call back via ``call_platform_tool``.
    max_concurrency:
        Maximum simultaneously active invocations.
    busy_buffer:
        Extra invocations queued before WorkerBusy rejection.
    project_id:
        Identifier stamped on AGENT_END / ERROR_OCCURRED events.
    """

    def __init__(
        self,
        runnable: Any,
        *,
        input_mapper: Any | None = None,
        platform_relay: Any | None = None,
        max_concurrency: int = 8,
        busy_buffer: int | None = None,
        project_id: str = "domyn-relay",
    ) -> None:
        self._runnable = runnable
        self._input_mapper = input_mapper
        self._platform_relay = platform_relay
        self._max_concurrency = max_concurrency
        self._busy_buffer = busy_buffer if busy_buffer is not None else max_concurrency
        self._project_id = project_id

        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._in_flight: dict[str, asyncio.Task[None]] = {}
        self._conversation_tasks: dict[str, asyncio.Task[None]] = {}
        self._platform_calls: dict[str, asyncio.Future[Any]] = {}

        if self._platform_relay is not None:
            self._platform_relay.bind(self)

    # ------------------------------------------------------------------
    # FrameworkAdapter contract
    # ------------------------------------------------------------------

    async def invoke(
        self,
        event: BaseEvent,
        emit: EmitFn,
        emit_priority: EmitPriorityFn,
    ) -> None:
        """Schedule execution of one AGENT_START event."""
        request_id = event.event_id or ""
        if not request_id:
            logger.warning("LangGraphAdapter.invoke: AGENT_START has no event_id; skipping")
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

        action_params = getattr(event.action, "parameters", None)
        raw_input: dict[str, Any] = (
            dict(action_params) if action_params else event.model_dump(mode="json")
        )
        input_data: Any = self._input_mapper(raw_input) if self._input_mapper else raw_input

        metadata: dict[str, Any] = {
            **event.metadata,
            "user_id": event.metadata.get("user_id") or event.author,
            "session_id": event.interaction_id or event.metadata.get("session_id"),
            "conversation_id": event.conversation_id,
        }

        conversation_id = event.conversation_id or request_id

        async def _run_body() -> None:
            await self._run_graph(
                event,
                request_id,
                conversation_id,
                input_data,
                metadata,
                emit,
                emit_priority,
            )

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
        """Resolve the Future suspended on a platform tool call."""
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
            for t in list(self._in_flight.values()):
                t.cancel()
            return
        task: asyncio.Task[None] | None = self._conversation_tasks.get(conv_id)
        if task is None or task.done():
            return
        logger.info("LangGraphAdapter.on_cancel: cancelling conv=%s", conv_id)
        task.cancel()

    async def close(self) -> None:
        """Cancel in-flight tasks and reject pending tool Futures."""
        if self._platform_relay is not None:
            self._platform_relay.unbind(self)
        for fut in self._platform_calls.values():
            if not fut.done():
                fut.set_exception(RuntimeError("LangGraphAdapter shutting down"))
        self._platform_calls.clear()
        self._conversation_tasks.clear()
        for task in list(self._in_flight.values()):
            task.cancel()
        if self._in_flight:
            await asyncio.gather(*self._in_flight.values(), return_exceptions=True)

    # ------------------------------------------------------------------
    # PlatformRelay contract (called by PlatformTool._arun via the relay)
    # ------------------------------------------------------------------

    async def call_platform_tool(self, name: str, kwargs: dict[str, Any]) -> Any:
        """Send TOOL_START and suspend until TOOL_END / TOOL_ERROR.

        This method is called by ``PlatformTool._arun`` through the bound
        ``PlatformRelay``.  ``emit_priority`` is retrieved from a ``ContextVar``
        so concurrent invocations each use their own closure.
        """
        ctx = invocation_ctx.get()
        if ctx is None:
            raise RuntimeError(
                "LangGraphAdapter.call_platform_tool: no invocation context — "
                "cannot send TOOL_START. Is the tool being called outside of invoke()?"
            )
        emit_priority = ctx.emit_priority

        call_id = str(uuid.uuid4())
        tool_start = BaseEvent(
            event_type=ExecutionEventType.TOOL_START,
            author=ctx.author,
            timestamp=datetime.now(UTC).replace(tzinfo=None),
            conversation_id=ctx.conversation_id,
            turn_id=ctx.turn_id,
            interaction_id=ctx.interaction_id,
            action=ToolAction(name=name, parameters=kwargs, call_id=call_id),
        )

        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._platform_calls[call_id] = future

        try:
            await emit_priority(tool_start)
        except Exception:
            self._platform_calls.pop(call_id, None)
            raise

        try:
            return await future
        finally:
            self._platform_calls.pop(call_id, None)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _run_graph(
        self,
        event: BaseEvent,
        request_id: str,
        conversation_id: str,
        input_data: Any,
        metadata: dict[str, Any],
        emit: EmitFn,
        emit_priority: EmitPriorityFn,
    ) -> None:
        """Invoke the LangGraph runnable and stream events."""
        # Lazy import: langchain-core is optional
        try:
            from domyn.agents.integrations.langgraph import PlatformCallbackHandler
        except ImportError as exc:
            raise ImportError(
                "langchain-core is required for LangGraphAdapter. Run: pip install domyn[langgraph]"
            ) from exc

        handler: PlatformCallbackHandler | None = None
        done: BaseEvent
        try:
            handler = PlatformCallbackHandler(
                project_id=self._project_id,
                emit=emit,
                session_id=metadata.get("session_id"),
                conversation_id=conversation_id,
                turn_id=event.turn_id,
            )
            result = await self._runnable.ainvoke(
                input_data,
                config={
                    # LangGraph checkpointing keys all state by
                    # configurable.thread_id. Without this, a graph built with
                    # a checkpointer still has no way to know which
                    # conversation's state to load/save, so multi-turn history
                    # silently never persists across separate invoke() calls
                    # for the same platform conversation_id.
                    "configurable": {"thread_id": conversation_id},
                    "callbacks": [handler],
                    "metadata": {**metadata, "request_id": request_id},
                },
            )
            done = BaseEvent(
                event_type=ExecutionEventType.AGENT_END,
                author=event.author,
                timestamp=datetime.now(UTC).replace(tzinfo=None),
                interaction_id=event.interaction_id,
                conversation_id=conversation_id,
                turn_id=event.turn_id,
                content=_result_content(result),
                metadata={"request_id": request_id},
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "LangGraphAdapter: invocation failed request_id=%s: %s",
                request_id,
                exc,
                exc_info=True,
            )
            done = BaseEvent(
                event_type=ExecutionEventType.AGENT_END,
                author=event.author,
                timestamp=datetime.now(UTC).replace(tzinfo=None),
                interaction_id=event.interaction_id,
                conversation_id=conversation_id,
                turn_id=event.turn_id,
                error_code=type(exc).__name__,
                error_message=str(exc),
                metadata={"request_id": request_id},
            )
        finally:
            if handler is not None:
                await handler.aclose()

        done = done.model_copy(update={"timestamp": datetime.now(UTC).replace(tzinfo=None)})
        await emit_priority(done)
