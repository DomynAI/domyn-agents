"""Tests for DomynAgentAdapter.

Covered:
1. Simple AGENT_START → RESPONSE → synthetic AGENT_END path.
2. DelegateTool round-trip: adapter sends TOOL_START, on_tool_response() resolves the Future.
3. WorkerBusy rejection when max_concurrency + busy_buffer is exceeded.
4. on_cancel() cancels an in-flight task.
5. close() rejects pending tool Futures.
"""

import asyncio
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from domyn.agents import Runner
from domyn.agents.adapters import DomynAgentAdapter
from domyn.agents.agents import Agent, DelegateAgent
from domyn.agents.core import (
    AgentAction,
    ApprovalType,
    BaseEvent,
    ExecutionEventType,
    FeedbackAction,
    InteractionContext,
    Part,
    ToolAction,
)
from domyn.agents.llm import BaseLLMProvider, LLMMessage, LLMResponse
from domyn.agents.planner_strategy import ToolUsePlannerStrategy
from domyn.agents.runner import RunnerTransport
from domyn.agents.tools import DelegateTool, FunctionTool

_BASE_URL = "https://platform.example.com"

# ---------------------------------------------------------------------------
# Minimal mock LLM (mirrors _ScriptedLLM pattern from other integration tests)
# ---------------------------------------------------------------------------


class _FixedLLM(BaseLLMProvider):
    type: str = "mock"
    name: str = "mock"
    model_name: str = "mock"
    supported_structured_outputs: None = None

    def __init__(self, texts: list[str]) -> None:
        super().__init__(model_name="mock")
        object.__setattr__(self, "_texts", texts)
        object.__setattr__(self, "_idx", 0)

    async def generate(self, messages: Any, **kwargs: Any) -> LLMResponse:
        texts: list[str] = object.__getattribute__(self, "_texts")
        idx: int = object.__getattribute__(self, "_idx")
        object.__setattr__(self, "_idx", idx + 1)
        return LLMResponse(content=texts[min(idx, len(texts) - 1)])

    async def generate_streaming(
        self, messages: Any, **kwargs: Any
    ) -> AsyncGenerator[LLMResponse, None]:
        resp = await self.generate(messages, **kwargs)
        yield resp


class _ToolCallLLM(BaseLLMProvider):
    """First call requests a tool call; every call after returns final text.

    Mirrors _MockLLMProvider in tests/integrations/domyn/test_runtime_platform_integration.py.
    """

    type: str = "mock"
    name: str = "mock"
    model_name: str = "mock"
    supported_structured_outputs: None = None

    def __init__(self, tool_name: str, final_text: str) -> None:
        super().__init__(model_name="mock")
        object.__setattr__(self, "_tool_name", tool_name)
        object.__setattr__(self, "_final_text", final_text)
        object.__setattr__(self, "_call", 0)

    async def generate(self, messages: Any, **kwargs: Any) -> LLMResponse:
        call: int = object.__getattribute__(self, "_call")
        object.__setattr__(self, "_call", call + 1)
        if call == 0:
            return LLMResponse(
                content="",
                generated_tokens=4,
                execution_time=0.0,
                tool_calls=[
                    {
                        "id": "tc-1",
                        "type": "function",
                        "function": {
                            "name": object.__getattribute__(self, "_tool_name"),
                            "arguments": '{"query": "hi"}',
                        },
                    }
                ],
            )
        return LLMResponse(
            content=object.__getattribute__(self, "_final_text"),
            generated_tokens=8,
            execution_time=0.0,
        )

    async def generate_streaming(
        self, messages: Any, **kwargs: Any
    ) -> AsyncGenerator[LLMResponse, None]:
        yield await self.generate(messages, **kwargs)


def _make_agent(llm_responses: list[str], tools: list[Any] | None = None) -> Agent:
    llm = _FixedLLM(llm_responses)
    return Agent(
        name="TestAgent",
        description="test",
        instruction="You are a test agent.",
        llm_provider=llm,
        planner=ToolUsePlannerStrategy(),
        tools=tools or [],
    )


def _agent_start(conversation_id: str = "conv-1") -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.AGENT_START,
        author="platform",
        event_id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        action=AgentAction(name="TestAgent", parameters={"task": "hello", "user_request": "hi"}),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invoke_emits_agent_end() -> None:
    """Adapter must emit a terminal AGENT_END after a simple invocation."""
    agent = _make_agent(["Hello world"])
    adapter = DomynAgentAdapter(root=agent)

    emitted: list[BaseEvent] = []
    priority: list[BaseEvent] = []

    async def emit(e: BaseEvent) -> None:
        emitted.append(e)

    async def emit_priority(e: BaseEvent) -> None:
        priority.append(e)

    event = _agent_start()
    await adapter.invoke(event, emit, emit_priority)

    # Wait for the background task to finish
    await asyncio.sleep(0.2)

    terminal = [e for e in priority if e.event_type == ExecutionEventType.AGENT_END] + [
        e for e in emitted if e.event_type == ExecutionEventType.AGENT_END
    ]
    assert terminal, "Expected AGENT_END in emitted events"
    await adapter.close()


@pytest.mark.asyncio
async def test_synthetic_agent_end_carries_turn_id_on_exception() -> None:
    """Regression: the SYNTHETIC AGENT_END built in _run_agent's except branch
    must carry the inbound AGENT_START's turn_id, as every relay producer does. A platform that correlates/displays
    responses by turn_id would otherwise show every response under a `None`
    turn — looking exactly like "every request returns the same old answer"."""
    agent = _make_agent(["irrelevant"])
    adapter = DomynAgentAdapter(root=agent)

    class _RaisingRunner:
        async def run(self, *args: Any, **kwargs: Any) -> AsyncGenerator[BaseEvent, None]:
            raise RuntimeError("boom")
            yield  # pragma: no cover - make this an async generator

    adapter._runner = _RaisingRunner()  # type: ignore[assignment]

    priority: list[BaseEvent] = []

    async def emit(_: BaseEvent) -> None:
        pass

    async def emit_priority(e: BaseEvent) -> None:
        priority.append(e)

    event = _agent_start()
    event = event.model_copy(update={"turn_id": "turn-XYZ"})
    await adapter.invoke(event, emit, emit_priority)
    await asyncio.sleep(0.2)

    terminal = [e for e in priority if e.event_type == ExecutionEventType.AGENT_END]
    assert terminal, "Expected a synthetic AGENT_END on the exception path"
    assert terminal[0].error_code == "RuntimeError"
    assert terminal[0].turn_id == "turn-XYZ"
    await adapter.close()


@pytest.mark.asyncio
async def test_workerbusy_when_full() -> None:
    """Adapter must emit ERROR_OCCURRED(WorkerBusy) when concurrency is exceeded."""
    agent = _make_agent(["ok"])
    adapter = DomynAgentAdapter(root=agent, max_concurrency=1, busy_buffer=0)

    priority: list[BaseEvent] = []

    async def emit(_: BaseEvent) -> None:
        pass

    async def emit_priority(e: BaseEvent) -> None:
        priority.append(e)

    # Fill the semaphore artificially
    await adapter._semaphore.acquire()
    try:
        for _ in range(2):  # 1 in-flight + 1 over buffer
            event = _agent_start()
            adapter._in_flight[event.event_id or "x"] = asyncio.create_task(asyncio.sleep(100))  # type: ignore[arg-type]

        event2 = _agent_start("conv-busy")
        await adapter.invoke(event2, emit, emit_priority)

        busy_events = [
            e
            for e in priority
            if e.event_type == ExecutionEventType.ERROR_OCCURRED and e.error_code == "WorkerBusy"
        ]
        assert busy_events, "Expected WorkerBusy ERROR_OCCURRED"
    finally:
        adapter._semaphore.release()
        await adapter.close()


@pytest.mark.asyncio
async def test_on_tool_response_resolves_future() -> None:
    """on_tool_response must resolve the pending Future keyed by call_id."""
    agent = _make_agent(["ok"])
    adapter = DomynAgentAdapter(root=agent)

    call_id = str(uuid.uuid4())
    loop = asyncio.get_running_loop()
    future: asyncio.Future[Any] = loop.create_future()
    adapter._platform_calls[call_id] = future

    tool_end = BaseEvent(
        event_type=ExecutionEventType.TOOL_END,
        author="platform",
        action=ToolAction(name="my_tool", call_id=call_id, parameters={}, observation="result-42"),
    )
    adapter.on_tool_response(tool_end)

    result = await asyncio.wait_for(future, timeout=1.0)
    assert result == "result-42"
    await adapter.close()


@pytest.mark.asyncio
async def test_on_tool_error_rejects_future() -> None:
    """on_tool_response with TOOL_ERROR must reject the Future."""
    agent = _make_agent(["ok"])
    adapter = DomynAgentAdapter(root=agent)

    call_id = str(uuid.uuid4())
    loop = asyncio.get_running_loop()
    future: asyncio.Future[Any] = loop.create_future()
    adapter._platform_calls[call_id] = future

    tool_error = BaseEvent(
        event_type=ExecutionEventType.TOOL_ERROR,
        author="platform",
        action=ToolAction(name="my_tool", call_id=call_id, parameters={}),
        error_message="boom",
    )
    adapter.on_tool_response(tool_error)

    with pytest.raises(RuntimeError, match="boom"):
        await asyncio.wait_for(future, timeout=1.0)
    await adapter.close()


@pytest.mark.asyncio
async def test_delegate_tool_relay_round_trip_via_real_runner() -> None:
    """Regression for the asyncio.current_task() correlation bug.

    Drives a real DelegateTool (no `wrapped`, so `.relay` is used) through
    adapter.invoke() -> Runner.run() -> Runner._merge_with_injections() ->
    DelegateTool.execute() -> _make_tool_call(). _merge_with_injections wraps
    every agent-loop step in a *new* asyncio.Task via asyncio.ensure_future, so
    the step that reaches DelegateTool.execute() never runs on the task
    invoke() created. A task-identity-keyed lookup misses every time; this
    test fails against that implementation and passes against the
    ContextVar-based fix in domyn/agents/adapters/domyn_agent.py.
    """
    delegate_tool = DelegateTool(
        tool_name="search_kb",
        tool_description="search",
        tool_parameters={"query": {"type": "string"}},
        required_params=["query"],
    )
    llm = _ToolCallLLM(tool_name="search_kb", final_text="done: found it")
    agent = Agent(
        name="TestAgent",
        description="test",
        instruction="You are a test agent.",
        llm_provider=llm,
        planner=ToolUsePlannerStrategy(),
        tools=[delegate_tool],
    )
    adapter = DomynAgentAdapter(root=agent)

    emitted: list[BaseEvent] = []
    priority: list[BaseEvent] = []

    async def emit(e: BaseEvent) -> None:
        emitted.append(e)

    async def emit_priority(e: BaseEvent) -> None:
        priority.append(e)

    event = _agent_start()
    await adapter.invoke(event, emit, emit_priority)

    tool_starts: list[BaseEvent] = []
    for _ in range(20):
        tool_starts = [e for e in priority if e.event_type == ExecutionEventType.TOOL_START]
        if tool_starts:
            break
        await asyncio.sleep(0.05)
    assert tool_starts, "Expected TOOL_START from _make_tool_call (relay round-trip)"
    call_id = getattr(tool_starts[0].action, "call_id", None)
    assert call_id is not None

    tool_end = BaseEvent(
        event_type=ExecutionEventType.TOOL_END,
        author="platform",
        action=ToolAction(
            name="search_kb", call_id=call_id, parameters={}, observation="doc: found it"
        ),
    )
    adapter.on_tool_response(tool_end)

    terminal: list[BaseEvent] = []
    for _ in range(20):
        # A normal turn's AGENT_END is forwarded via `emit` (best-effort); only the
        # synthetic AGENT_END built on an error path goes through `emit_priority`.
        terminal = [e for e in priority if e.event_type == ExecutionEventType.AGENT_END] + [
            e for e in emitted if e.event_type == ExecutionEventType.AGENT_END
        ]
        if terminal:
            break
        await asyncio.sleep(0.05)
    assert terminal, "Expected AGENT_END after the DelegateTool relay round-trip completed"
    assert terminal[0].error_code is None

    # The runner's own TOOL_END for a delegate tool carries the LLM's tool_call
    # id, not the call_id _make_tool_call minted for the relay round trip — the
    # platform never issued it and cannot match it. Like the paired TOOL_START,
    # it must not reach the wire (the platform already produced this result).
    forwarded_tool_ends = [
        e
        for e in emitted + priority
        if e.event_type == ExecutionEventType.TOOL_END
        and getattr(e.action, "name", None) == "search_kb"
    ]
    assert not forwarded_tool_ends, (
        "the runner's delegate-tool TOOL_END (foreign call_id) must not be forwarded"
    )
    await adapter.close()


@pytest.mark.asyncio
async def test_platform_tool_start_carries_invocation_correlation_ids() -> None:
    """The relay TOOL_START must carry the INBOUND AGENT_START's correlation ids.

    The relay routes frames to the platform by conversation_id, turn_id feeds
    the transports' cross-turn replay filter, and author/interaction_id are the
    identity the platform matches the call against — send anything else and the
    platform never answers, so the tool call hangs forever.

    Regression: _InvocationCtx used to hold a reference to the inbound event and
    read these fields at tool-call time. By then Runner._setup_conversation_turn
    has rewritten the event in place with the runner's own freshly minted
    interaction_id, so the TOOL_START went out under an interaction the platform
    had never issued. Mirrors the sibling
    test_langgraph_adapter.py::test_platform_tool_start_carries_invocation_correlation_ids.
    """
    tool = DelegateTool(
        tool_name="search",
        tool_description="search",
        tool_parameters={"query": {"type": "string"}},
        required_params=["query"],
    )  # no wrapped= → relay-backed, so .relay (_make_tool_call) is used
    agent = Agent(
        name="TestAgent",
        description="test",
        instruction="You are a test agent.",
        llm_provider=_ToolCallLLM(tool_name="search", final_text="done"),
        planner=ToolUsePlannerStrategy(),
        tools=[tool],
    )
    adapter = DomynAgentAdapter(root=agent)

    priority: list[BaseEvent] = []

    async def emit(_: BaseEvent) -> None:
        pass

    async def emit_priority(e: BaseEvent) -> None:
        priority.append(e)

    inbound = BaseEvent(
        event_type=ExecutionEventType.AGENT_START,
        author="platform",
        event_id=str(uuid.uuid4()),
        conversation_id="conv-tool",
        turn_id="turn-1",
        interaction_id="interaction-1",
        action=AgentAction(name="TestAgent", parameters={"task": "hi", "user_request": "hi"}),
    )
    await adapter.invoke(inbound, emit, emit_priority)

    tool_starts: list[BaseEvent] = []
    for _ in range(20):
        tool_starts = [e for e in priority if e.event_type == ExecutionEventType.TOOL_START]
        if tool_starts:
            break
        await asyncio.sleep(0.05)
    assert tool_starts, "Expected a TOOL_START from _make_tool_call"

    assert tool_starts[0].conversation_id == "conv-tool"
    assert tool_starts[0].turn_id == "turn-1"
    assert tool_starts[0].author == "platform"
    assert tool_starts[0].interaction_id == "interaction-1"
    await adapter.close()


@pytest.mark.asyncio
async def test_gated_tool_feedback_round_trip_via_real_runner() -> None:
    """Regression for two bugs on the approval-gate path.

    1. TOOL_NEEDS_FEEDBACK must go through emit_priority (guaranteed delivery),
       not the best-effort emit — losing it strands the turn forever, since
       nothing else will ever re-prompt the human.
    2. _run_agent must not synthesize a bogus "done" AGENT_END right after a
       genuine pause for approval — that falsely signals turn completion while
       the approval request is actually still pending.

    Drives a gated DelegateTool (wrapped=, so execution is local — approval_type
    still routes it through the same Runner._pause_for_approval gate a
    relay-executed DelegateTool goes through) via adapter.invoke() twice: once
    to reach TOOL_NEEDS_FEEDBACK, and once more with a TOOL_FEEDBACK reply to
    resume and complete the turn.
    """
    executed: list[str] = []

    def search(query: str) -> str:
        executed.append(query)
        return "ok"

    tool = DelegateTool(
        tool_name="search",
        tool_description="search",
        tool_parameters={"query": {"type": "string"}},
        required_params=["query"],
        approval_type=ApprovalType.USER_CONFIRM,
        wrapped=search,
    )
    llm = _ToolCallLLM(tool_name="search", final_text="Done.")
    agent = Agent(
        name="TestAgent",
        description="test",
        instruction="You are a test agent.",
        llm_provider=llm,
        planner=ToolUsePlannerStrategy(),
        tools=[tool],
    )
    adapter = DomynAgentAdapter(root=agent)

    emitted: list[BaseEvent] = []
    priority: list[BaseEvent] = []

    async def emit(e: BaseEvent) -> None:
        emitted.append(e)

    async def emit_priority(e: BaseEvent) -> None:
        priority.append(e)

    event = _agent_start()
    await adapter.invoke(event, emit, emit_priority)

    needs: list[BaseEvent] = []
    for _ in range(20):
        needs = [e for e in priority if e.event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK]
        if needs:
            break
        await asyncio.sleep(0.05)
    assert needs, "Expected TOOL_NEEDS_FEEDBACK via emit_priority (guaranteed delivery)"
    assert not any(e.event_type == ExecutionEventType.AGENT_END for e in emitted + priority), (
        "No AGENT_END should be sent while the turn is paused for approval"
    )
    assert executed == [], "The gated tool must not run before approval"

    action = needs[0].action
    action.approve = True
    feedback = BaseEvent(
        event_type=ExecutionEventType.TOOL_FEEDBACK,
        author="user",
        event_id=str(uuid.uuid4()),
        conversation_id=event.conversation_id,
        action=action,
    )
    await adapter.invoke(feedback, emit, emit_priority)

    terminal: list[BaseEvent] = []
    for _ in range(20):
        terminal = [e for e in priority if e.event_type == ExecutionEventType.AGENT_END] + [
            e for e in emitted if e.event_type == ExecutionEventType.AGENT_END
        ]
        if terminal:
            break
        await asyncio.sleep(0.05)
    assert terminal, "Expected AGENT_END after the approval was granted and the turn resumed"
    assert terminal[0].error_code is None
    assert executed == ["hi"], "The gated tool must run exactly once, after approval"
    await adapter.close()


class _Wire(RunnerTransport):
    """The platform's end of `domyn expose`: events cross as JSON, and as on the relay
    a frame for another conversation is dropped."""

    def __init__(self, adapter: DomynAgentAdapter) -> None:
        self._adapter = adapter

    async def execute(
        self, agent_name: str, context: InteractionContext, run_config: dict[str, Any] | None = None
    ) -> AsyncGenerator[BaseEvent, None]:
        received: asyncio.Queue[BaseEvent] = asyncio.Queue()

        async def emit(event: BaseEvent) -> None:
            if event.conversation_id == context.conversation_id:
                received.put_nowait(BaseEvent.model_validate_json(event.model_dump_json()))

        sent = self._build_outbound_event(agent_name, context)
        await self._adapter.invoke(
            BaseEvent.model_validate_json(sent.model_dump_json()), emit, emit
        )
        while True:
            event = await asyncio.wait_for(received.get(), timeout=5)
            yield event.model_copy(update={"author": agent_name})
            if event.event_type is ExecutionEventType.AGENT_END or (
                event.event_type is ExecutionEventType.RESPONSE and not event.is_partial
            ):
                return

    async def send_result(self, event: BaseEvent, conversation_id: str | None = None) -> None:
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize("approve", [True, False], ids=["approved", "refused"])
async def test_approval_given_on_the_platform_resumes_the_exposed_agent(approve: bool) -> None:
    """Approving or refusing on the platform resumes the exposed agent.

    Regression: its events carried the interaction_id its own runner minted, so the
    platform could not find the request and dropped the reply; and the answered
    call's result went out without a conversation_id, so the relay dropped it.
    """
    executed: list[str] = []

    def search(query: str) -> str:
        """Search."""
        executed.append(query)
        return "ok"

    exposed = Agent(
        name="TestAgent",
        description="test",
        instruction="You are a test agent.",
        llm_provider=_ToolCallLLM(tool_name="search", final_text="Done."),
        planner=ToolUsePlannerStrategy(),
        tools=[FunctionTool(func=search, approval_type=ApprovalType.USER_CONFIRM)],
    )
    adapter = DomynAgentAdapter(root=exposed)
    platform = Runner()

    async def turn(user_input: BaseEvent) -> list[BaseEvent]:
        # The platform builds the configured agent again for every request.
        main = DelegateAgent(name="Main agent", description="external", transport=_Wire(adapter))
        return [
            e
            async for e in platform.run(
                application_name="platform",
                user_id="user",
                root=main,
                user_input=user_input,
                conversation_id="conv-1",
            )
        ]

    asked = await turn(
        BaseEvent(
            event_type=ExecutionEventType.USER_INPUT,
            author="user",
            content=[Part(text="search hi")],
            turn_id="turn-1",
        )
    )
    assert asked[-1].event_type is ExecutionEventType.TOOL_NEEDS_FEEDBACK
    assert executed == []

    # As the platform builds it from "Approve once" / "Not now": no call id of its own.
    reply = FeedbackAction(
        name=asked[-1].action.name,
        parameters=asked[-1].action.parameters,
        approval_type=ApprovalType.USER_CONFIRM,
        approve=approve,
    )
    resumed = await turn(
        BaseEvent(
            event_type=ExecutionEventType.TOOL_FEEDBACK,
            author="user",
            turn_id="turn-1",
            action=reply,
        )
    )

    assert executed == (["hi"] if approve else [])
    results = [e.action.name for e in resumed if e.event_type is ExecutionEventType.TOOL_END]
    assert results == ["search"]
    answers = [
        e.content[0].text
        for e in resumed
        if e.event_type is ExecutionEventType.RESPONSE and not e.is_partial
    ]
    assert answers == ["Done."]
    await adapter.close()


@pytest.mark.asyncio
async def test_close_rejects_pending_futures() -> None:
    """close() must reject all pending DelegateTool Futures."""
    agent = _make_agent(["ok"])
    adapter = DomynAgentAdapter(root=agent)

    call_id = str(uuid.uuid4())
    loop = asyncio.get_running_loop()
    future: asyncio.Future[Any] = loop.create_future()
    adapter._platform_calls[call_id] = future

    await adapter.close()

    assert future.done()
    assert isinstance(future.exception(), RuntimeError)


# ---------------------------------------------------------------------------
# refresh_delegate_tools — periodic platform-tool rediscovery
# ---------------------------------------------------------------------------

_TOOL_DEFS = [
    {
        "name": "web_search",
        "description": "Search the web.",
        "parameters": [
            {"name": "query", "type": "str", "description": "query", "is_required": True}
        ],
    },
]


@pytest.mark.asyncio
async def test_refresh_wires_newly_attached_tool(platform: Any) -> None:
    """A platform tool discovered only by refresh must be wired and callable."""
    platform.serve(_TOOL_DEFS)

    agent = _make_agent(["ok"], tools=[])
    adapter = DomynAgentAdapter(root=agent)
    assert adapter._delegate_tool_names == frozenset()

    await adapter.refresh_delegate_tools(
        api_key="key",
        space_id="space",
        channel_id="chan",
        base_url=_BASE_URL,
        configuration_id=None,
        attach_new_tools=True,
    )

    assert adapter._delegate_tool_names == frozenset({"web_search"})
    (attached,) = [t for t in agent.tools if isinstance(t, DelegateTool)]
    assert attached.relay == adapter._make_tool_call


@pytest.mark.asyncio
async def test_refresh_drops_tool_no_longer_on_platform(platform: Any) -> None:
    """A tool the platform no longer recognizes must disappear after refresh."""
    platform.serve([])

    stale = DelegateTool(tool_name="gone", tool_description="")
    agent = _make_agent(["ok"], tools=[stale])
    adapter = DomynAgentAdapter(root=agent)
    assert adapter._delegate_tool_names == frozenset({"gone"})

    await adapter.refresh_delegate_tools(
        api_key="key",
        space_id="space",
        channel_id="chan",
        base_url=_BASE_URL,
        configuration_id=None,
        attach_new_tools=True,
    )

    assert stale not in agent.tools
    assert adapter._delegate_tool_names == frozenset()


@pytest.mark.asyncio
async def test_refresh_keeps_existing_tools_on_fetch_failure(platform: Any) -> None:
    """A failed refresh must leave the live tool set (and wiring) untouched."""
    platform.fail()

    tool = DelegateTool(tool_name="web_search", tool_description="")
    agent = _make_agent(["ok"], tools=[tool])
    adapter = DomynAgentAdapter(root=agent)
    original_relay = tool.relay
    assert original_relay is not None

    await adapter.refresh_delegate_tools(
        api_key="key",
        space_id="space",
        channel_id="chan",
        base_url=_BASE_URL,
        configuration_id=None,
        attach_new_tools=True,
    )

    assert tool in agent.tools
    assert tool.relay is original_relay
    assert adapter._delegate_tool_names == frozenset({"web_search"})
