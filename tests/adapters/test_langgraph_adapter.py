"""Tests for LangGraphAdapter.

Covered:
1. Regression: ``_run_graph`` must construct the *real* ``PlatformCallbackHandler``
   with a valid keyword signature. Passing ``user_id=...``, an argument the
   handler does not accept, fails *before* the try/except in ``_run_graph``, so
   the ``TypeError`` propagates out of the fire-and-forget ``asyncio.Task`` and
   only surfaces as an unretrieved-task traceback when the process shuts down —
   every single invocation fails silently.
2. Simple AGENT_START -> AGENT_END path with a stub runnable.
3. A runnable exception must produce an AGENT_END with error_code/error_message
   (not an unhandled task exception).
4. WorkerBusy rejection when max_concurrency + busy_buffer is exceeded.
5. on_cancel() cancels an in-flight task.
6. close() rejects pending platform-tool Futures.
7. Platform tool round-trip: call_platform_tool() sends TOOL_START via
   emit_priority and suspends until on_tool_response() resolves it.
8. A long, multi-turn conversation against a REAL (minimal) compiled LangGraph
   graph backed by an ``InMemorySaver`` checkpointer: many sequential
   AGENT_START events on the same conversation_id must accumulate real
   history, proving the adapter's ``config["configurable"]["thread_id"]``
   wiring actually round-trips through LangGraph's checkpointing (not just
   that the kwarg is present, which is covered by
   ``test_invoke_passes_thread_id_for_langgraph_checkpointing``).

These tests intentionally import the real ``PlatformCallbackHandler`` from
``domyn.agents.integrations.langgraph.handler`` (not a mock) so that a future
signature drift between the adapter and the handler is caught immediately by
the test suite, instead of silently breaking every real invocation again.
"""

import asyncio
import uuid
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from domyn.agents.adapters import LangGraphAdapter
from domyn.agents.core import BaseEvent, ExecutionEventType, ToolAction
from domyn.agents.integrations.langgraph import PlatformRelay

# ---------------------------------------------------------------------------
# Stub runnable (anything with an async ainvoke(input, config=...) is enough)
# ---------------------------------------------------------------------------


class _StubRunnable:
    """Minimal stand-in for a compiled LangGraph graph."""

    def __init__(
        self,
        result: Any = None,
        *,
        raises: Exception | None = None,
        on_ainvoke: Any = None,
    ) -> None:
        self._result = result
        self._raises = raises
        self._on_ainvoke = on_ainvoke
        self.last_config: dict[str, Any] | None = None

    async def ainvoke(self, input_data: Any, config: dict[str, Any] | None = None) -> Any:
        self.last_config = config
        if self._on_ainvoke is not None:
            await self._on_ainvoke()
        if self._raises is not None:
            raise self._raises
        return self._result


def _agent_start(conversation_id: str = "conv-1") -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.AGENT_START,
        author="platform",
        event_id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        turn_id="turn-1",
        interaction_id="interaction-1",
    )


def _make_emitters() -> tuple[list[BaseEvent], list[BaseEvent], Any, Any]:
    emitted: list[BaseEvent] = []
    priority: list[BaseEvent] = []

    async def emit(e: BaseEvent) -> None:
        emitted.append(e)

    async def emit_priority(e: BaseEvent) -> None:
        priority.append(e)

    return emitted, priority, emit, emit_priority


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invoke_constructs_real_handler_and_emits_agent_end() -> None:
    """Regression test for the PlatformCallbackHandler(user_id=...) TypeError.

    Uses the real PlatformCallbackHandler (imported for real inside
    LangGraphAdapter._run_graph via a lazy import) so any future kwarg
    mismatch between the adapter and the handler fails this test loudly,
    instead of failing silently in production.
    """
    runnable = _StubRunnable(result={"messages": [type("M", (), {"content": "hello"})()]})
    adapter = LangGraphAdapter(runnable)

    _emitted, priority, emit, emit_priority = _make_emitters()
    event = _agent_start()
    await adapter.invoke(event, emit, emit_priority)
    await asyncio.sleep(0.2)

    terminal = [e for e in priority if e.event_type == ExecutionEventType.AGENT_END]
    assert terminal, "Expected AGENT_END in priority-emitted events"
    done = terminal[0]
    assert done.error_code is None
    assert done.content and done.content[0].text == "hello"
    # Regression: the terminal AGENT_END must carry the inbound turn_id so the
    # platform can correlate/display the response with the right turn —
    # missing this made every response look like it belonged to no turn at all.
    assert done.turn_id == event.turn_id == "turn-1"
    await adapter.close()


@pytest.mark.asyncio
async def test_invoke_passes_thread_id_for_langgraph_checkpointing() -> None:
    """Regression: ainvoke's config must carry configurable.thread_id.

    LangGraph checkpointing keys all conversation state by
    config["configurable"]["thread_id"]. Without it, a graph built with a
    checkpointer has no way to know which conversation to load/save, so
    multi-turn history silently never persists across separate invoke()
    calls for the same platform conversation_id.
    """
    runnable = _StubRunnable(result={"result": "ok"})
    adapter = LangGraphAdapter(runnable)

    _emitted, _priority, emit, emit_priority = _make_emitters()
    event = _agent_start("conv-thread-42")
    await adapter.invoke(event, emit, emit_priority)
    await asyncio.sleep(0.2)

    assert runnable.last_config is not None
    assert runnable.last_config["configurable"]["thread_id"] == "conv-thread-42"
    await adapter.close()


@pytest.mark.asyncio
async def test_invoke_falls_back_to_request_id_when_conversation_id_missing() -> None:
    """Regression: thread_id must never be None even on a brand-new conversation.

    ``event.conversation_id`` can legitimately be empty/None on the very first
    turn (before the platform has assigned one). A checkpointer-backed graph
    raises/misbehaves if ``configurable.thread_id`` is ``None``, which silently
    dropped every single message once a checkpointer (e.g. ``InMemorySaver``)
    was attached to the graph. The adapter must fall back to ``request_id``
    (the AGENT_START's own event_id) instead of passing ``None`` through.
    """
    runnable = _StubRunnable(result={"result": "ok"})
    adapter = LangGraphAdapter(runnable)

    _emitted, priority, emit, emit_priority = _make_emitters()
    event = BaseEvent(
        event_type=ExecutionEventType.AGENT_START,
        author="platform",
        event_id="req-no-conv-id",
        conversation_id=None,
    )
    await adapter.invoke(event, emit, emit_priority)
    await asyncio.sleep(0.2)

    assert runnable.last_config is not None
    thread_id = runnable.last_config["configurable"]["thread_id"]
    assert thread_id, "thread_id must never be None/empty"
    assert thread_id == "req-no-conv-id"

    terminal = [e for e in priority if e.event_type == ExecutionEventType.AGENT_END]
    assert terminal, "Expected AGENT_END even without a conversation_id"
    assert terminal[0].error_code is None
    await adapter.close()


@pytest.mark.asyncio
async def test_runnable_exception_becomes_agent_end_error() -> None:
    """A raising runnable must produce an AGENT_END with error info, not a
    silently-dropped task exception."""
    runnable = _StubRunnable(raises=ValueError("boom"))
    adapter = LangGraphAdapter(runnable)

    _emitted, priority, emit, emit_priority = _make_emitters()
    event = _agent_start()
    await adapter.invoke(event, emit, emit_priority)
    await asyncio.sleep(0.2)

    terminal = [e for e in priority if e.event_type == ExecutionEventType.AGENT_END]
    assert terminal, "Expected AGENT_END even when the runnable raises"
    done = terminal[0]
    assert done.error_code == "ValueError"
    assert done.error_message == "boom"
    assert done.turn_id == event.turn_id == "turn-1"
    await adapter.close()


@pytest.mark.asyncio
async def test_workerbusy_when_full() -> None:
    """Adapter must emit ERROR_OCCURRED(WorkerBusy) when concurrency is exceeded."""

    async def _slow() -> None:
        await asyncio.sleep(1)

    runnable = _StubRunnable(result={"result": "ok"}, on_ainvoke=_slow)
    adapter = LangGraphAdapter(runnable, max_concurrency=1, busy_buffer=0)

    _emitted, priority, emit, emit_priority = _make_emitters()

    await adapter.invoke(_agent_start("conv-1"), emit, emit_priority)
    await asyncio.sleep(0.05)  # let the first task grab the semaphore
    await adapter.invoke(_agent_start("conv-2"), emit, emit_priority)
    await asyncio.sleep(0.05)

    busy = [e for e in priority if e.error_code == "WorkerBusy"]
    assert busy, "Expected a WorkerBusy ERROR_OCCURRED event"
    await adapter.close()


@pytest.mark.asyncio
async def test_on_cancel_cancels_inflight_task() -> None:
    """on_cancel() must cancel the task associated with the conversation_id."""
    started = asyncio.Event()
    released = asyncio.Event()

    async def _hang() -> None:
        started.set()
        await released.wait()

    runnable = _StubRunnable(result={"result": "ok"}, on_ainvoke=_hang)
    adapter = LangGraphAdapter(runnable)

    _emitted, _priority, emit, emit_priority = _make_emitters()
    event = _agent_start("conv-cancel")
    await adapter.invoke(event, emit, emit_priority)
    await asyncio.wait_for(started.wait(), timeout=1)

    cancel_event = BaseEvent(
        event_type=ExecutionEventType.AGENT_END,
        author="platform",
        conversation_id="conv-cancel",
        error_code="user_cancelled",
    )
    adapter.on_cancel(cancel_event)
    await asyncio.sleep(0.1)

    task = adapter._conversation_tasks.get("conv-cancel")
    assert task is None or task.cancelled() or task.done()
    released.set()
    await adapter.close()


@pytest.mark.asyncio
async def test_close_rejects_pending_platform_calls() -> None:
    """close() must reject any Future still suspended on a platform tool call."""
    runnable = _StubRunnable(result={"result": "ok"})
    adapter = LangGraphAdapter(runnable)

    loop = asyncio.get_running_loop()
    future: asyncio.Future[Any] = loop.create_future()
    adapter._platform_calls["call-1"] = future

    await adapter.close()

    assert future.done()
    with pytest.raises(RuntimeError):
        future.result()


@pytest.mark.asyncio
async def test_call_platform_tool_round_trip() -> None:
    """call_platform_tool() sends TOOL_START and suspends until TOOL_END resolves it."""
    relay = PlatformRelay()

    async def _call_tool_mid_invoke() -> None:
        # Simulate the graph invoking a PlatformTool mid-run.
        result = await relay.call_platform_tool("search", {"query": "hi"})
        assert result == "42"

    runnable = _StubRunnable(result={"result": "done"}, on_ainvoke=_call_tool_mid_invoke)
    adapter = LangGraphAdapter(runnable, platform_relay=relay)

    _emitted, priority, emit, emit_priority = _make_emitters()
    event = _agent_start("conv-tool")
    await adapter.invoke(event, emit, emit_priority)

    # Wait for TOOL_START to be emitted, then resolve it with TOOL_END.
    for _ in range(20):
        tool_starts = [e for e in priority if e.event_type == ExecutionEventType.TOOL_START]
        if tool_starts:
            break
        await asyncio.sleep(0.05)
    assert tool_starts, "Expected a TOOL_START event"
    call_id = getattr(tool_starts[0].action, "call_id", None)
    assert call_id is not None

    tool_end = BaseEvent(
        event_type=ExecutionEventType.TOOL_END,
        author="platform",
        action=ToolAction(name="search", parameters={}, call_id=call_id, observation="42"),
    )
    adapter.on_tool_response(tool_end)
    await asyncio.sleep(0.2)

    terminal = [e for e in priority if e.event_type == ExecutionEventType.AGENT_END]
    assert terminal, "Expected AGENT_END after the tool round-trip completed"
    assert terminal[0].error_code is None
    await adapter.close()


# ---------------------------------------------------------------------------
# Long multi-turn conversation against a REAL LangGraph graph + checkpointer
# ---------------------------------------------------------------------------


def _build_memory_graph() -> Any:
    """Minimal MessagesState graph whose reply reveals how much history it can
    see, so tests can assert on real accumulated state rather than mocks."""

    def _chat_node(state: MessagesState) -> dict[str, Any]:
        human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        first = human_messages[0].content if human_messages else ""
        reply = f"Turn {len(human_messages)}: you first said '{first}'."
        return {"messages": [AIMessage(content=reply)]}

    builder = StateGraph(MessagesState)
    builder.add_node("chat", _chat_node)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    return builder.compile(checkpointer=InMemorySaver())


@pytest.mark.asyncio
async def test_long_conversation_remembers_earlier_turns_via_checkpointer() -> None:
    """Many sequential AGENT_START turns on the SAME conversation_id against a
    real (minimal) LangGraph graph backed by InMemorySaver.

    Proves the adapter's thread_id wiring (``config["configurable"]["thread_id"]
    = conversation_id``) actually makes LangGraph's checkpointer persist and
    reload conversation history across turns -- not just that the kwarg is
    passed (that narrower claim is covered by
    ``test_invoke_passes_thread_id_for_langgraph_checkpointing``).
    """
    graph = _build_memory_graph()
    adapter = LangGraphAdapter(graph)
    conversation_id = "conv-long-history"

    n_turns = 12
    final_reply = ""
    for i in range(n_turns):
        _emitted, priority, emit, emit_priority = _make_emitters()
        event = BaseEvent(
            event_type=ExecutionEventType.AGENT_START,
            author="platform",
            event_id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            turn_id=f"turn-{i}",
            action=ToolAction(
                name="chat",
                parameters={"messages": [HumanMessage(content=f"message number {i}")]},
            ),
        )
        await adapter.invoke(event, emit, emit_priority)

        terminal: list[BaseEvent] = []
        for _ in range(20):
            terminal = [e for e in priority if e.event_type == ExecutionEventType.AGENT_END]
            if terminal:
                break
            await asyncio.sleep(0.05)
        assert terminal, f"Expected AGENT_END on turn {i}"
        assert terminal[0].error_code is None
        final_reply = terminal[0].content[0].text

    # After n_turns turns the graph must still recall the FIRST message -- proof
    # that the checkpointer accumulated real history under one stable thread_id
    # (not a fresh/empty state every call, and not a wrong/rotating thread_id).
    assert "message number 0" in final_reply
    assert f"Turn {n_turns}" in final_reply

    # Sanity: the checkpointer itself holds one growing thread with all messages
    # (one human + one AI message per turn).
    saved = await graph.aget_state({"configurable": {"thread_id": conversation_id}})
    assert len(saved.values["messages"]) == n_turns * 2

    await adapter.close()


@pytest.mark.asyncio
async def test_platform_tool_start_carries_invocation_correlation_ids() -> None:
    """TOOL_START must carry the invocation's conversation_id and turn_id.

    The relay routes frames to the platform by conversation_id and drops any
    frame whose value does not match the subscriber's, so a TOOL_START without
    one never reaches the platform-side runner and the tool call never
    completes. turn_id is read by the transports' cross-turn replay filter.
    """
    relay = PlatformRelay()

    async def _call_tool_mid_invoke() -> None:
        await relay.call_platform_tool("search", {"query": "hi"})

    runnable = _StubRunnable(result={"result": "done"}, on_ainvoke=_call_tool_mid_invoke)
    adapter = LangGraphAdapter(runnable, platform_relay=relay)

    _emitted, priority, emit, emit_priority = _make_emitters()
    await adapter.invoke(_agent_start("conv-tool"), emit, emit_priority)

    for _ in range(20):
        tool_starts = [e for e in priority if e.event_type == ExecutionEventType.TOOL_START]
        if tool_starts:
            break
        await asyncio.sleep(0.05)
    assert tool_starts, "Expected a TOOL_START event"

    assert tool_starts[0].conversation_id == "conv-tool"  # <- HERE
    assert tool_starts[0].turn_id == "turn-1"  # <- HERE
    # Regression: author/interaction_id must also match the invocation's
    # AGENT_START (same correlation set DomynAgentAdapter stamps on its own
    # TOOL_START) — a static author here made platform-side attribution wrong.
    assert tool_starts[0].author == "platform"
    assert tool_starts[0].interaction_id == "interaction-1"
    await adapter.close()
