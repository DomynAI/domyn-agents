"""Integration tests: platform Runner + DelegateAgent interacting with a mock Domyn runtime.

The mock runtime is implemented as a stateful _MockTransport that yields a predetermined
realistic sequence of events — the same types a real Domyn runtime would emit over WebSocket
or SSE.  No HTTP layer is involved; tests exercise the real Runner and DelegateAgent code paths.

Realistic event sequences
-------------------------
A DomynAgentAdapter receives an AGENT_START from the platform and calls runner.run() with
that AGENT_START as user_input.  Because user_input.event_type == AGENT_START, the runner
does NOT re-emit a root AGENT_START; it goes straight to the agent loop.  The outbound event
stream therefore starts from the planner's first event (LLM_START).

Sub-agents end with AGENT_END (never a bare non-partial RESPONSE), because the runner converts
the planner's final RESPONSE to AGENT_END when exiting a sub-agent context.  If a sub-agent
non-partial RESPONSE were forwarded, the platform's DelegateAgent would correctly treat it as
the session's final answer and close the session — which is the intended semantics.

AGENT_START events for sub-agents carry the task in both content and action.parameters:
  content=[Part(text=task)], action=AgentAction(name=..., parameters={"task": task})

Covered scenarios
-----------------
1. Simple delegation — remote runtime emits LLM lifecycle then final RESPONSE.
2. Remote runtime has sub-agents — AGENT_CALL → sub AGENT_START (with task) →
   sub AGENT_END → root second LLM call → root RESPONSE.
3. Remote runtime calls a platform FunctionTool ("delegate tool of the platform"):
   TOOL_START → runner executes locally → TOOL_END sent back → remote resumes.
4. Remote runtime calls a platform DelegateTool (wrapped callable):
   same round-trip but via DelegateTool.wrapped instead of FunctionTool.
5. Platform root Agent delegates to DelegateAgent sub-agent:
   full stack — parent MockLLM → AGENT_CALL → DelegateAgent → remote events → AGENT_END.
"""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from domyn.agents import Runner
from domyn.agents.agents import Agent, DelegateAgent
from domyn.agents.core import (
    AgentAction,
    BaseEvent,
    ExecutionEventType,
    InteractionContext,
    Part,
    ToolAction,
)
from domyn.agents.llm import BaseLLMProvider, LLMMessage, LLMResponse, StructuredOutputType
from domyn.agents.planner_strategy import ToolUsePlannerStrategy
from domyn.agents.runner import RunnerTransport
from domyn.agents.tools import DelegateTool, FunctionTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user_input(text: str = "hello") -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text=text)],
    )


class _MockTransport(RunnerTransport):
    """In-process transport that drives one scripted conversation.

    *rounds* is a list of event batches.  The first batch is yielded immediately.
    Subsequent batches are yielded only after ``send_result()`` is called (i.e.
    after the runner sends a TOOL_END back), matching the real transport behaviour
    where the connection stays open and waits for the caller to push a result.
    """

    def __init__(self, rounds: list[list[BaseEvent]]) -> None:
        self._rounds = list(rounds)
        self._result_queue: asyncio.Queue[BaseEvent] = asyncio.Queue()
        self.results_received: list[BaseEvent] = []

    async def execute(
        self, agent_name: str, context: InteractionContext
    ) -> AsyncGenerator[BaseEvent, None]:
        for round_idx, events in enumerate(self._rounds):
            for event in events:
                yield event
            # Between rounds, block until the runner calls send_result()
            if round_idx < len(self._rounds) - 1:
                result = await self._result_queue.get()
                self.results_received.append(result)

    async def send_result(self, event: BaseEvent, conversation_id: str | None = None) -> None:
        await self._result_queue.put(event)


class _MockLLMProvider(BaseLLMProvider):
    type: str = "mock"
    name: str = "mock"
    model_name: str = "mock"
    supported_structured_outputs: None = None
    _responses: list[LLMResponse]
    _call: int = 0

    def __init__(self, responses: list[LLMResponse], **kwargs: Any) -> None:
        super().__init__(model_name="mock", **kwargs)
        object.__setattr__(self, "_responses", responses)
        object.__setattr__(self, "_call", 0)

    async def generate(self, messages: LLMMessage, **kwargs: Any) -> LLMResponse:
        idx = self._call
        object.__setattr__(self, "_call", idx + 1)
        return self._responses[min(idx, len(self._responses) - 1)]

    async def generate_streaming(
        self, messages: LLMMessage, **kwargs: Any
    ) -> AsyncGenerator[LLMResponse, None]:
        resp = await self.generate(messages, **kwargs)
        yield resp


def _make_agent(name: str, description: str, llm: Any, **kwargs: Any) -> Agent:
    """Create an Agent with ToolUsePlannerStrategy so tool_calls LLM responses work correctly."""
    return Agent(
        name=name,
        description=description,
        instruction="be helpful",
        llm_provider=llm,
        planner=ToolUsePlannerStrategy(),
        **kwargs,
    )


def _sub_agent_start(name: str, task: str) -> BaseEvent:
    """Build a realistic AGENT_START for a sub-agent (mirrors runner._handle_agent_transition)."""
    return BaseEvent(
        event_type=ExecutionEventType.AGENT_START,
        author=name,
        content=[Part(text=task)],
        action=AgentAction(name=name, parameters={"task": task}, thought=""),
    )


# ---------------------------------------------------------------------------
# Test 1: simple remote runtime response with realistic LLM lifecycle events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_platform_delegate_realistic_llm_events():
    """Remote emits a full LLM lifecycle (LLM_START → LLM_END → RESPONSE) — all surfaced.

    A real Domyn runtime using ToolUsePlannerStrategy without streaming emits exactly
    one LLM_START, one LLM_END, and one non-partial RESPONSE per turn.  No partial
    RESPONSE tokens are emitted in non-streaming mode.
    """
    transport = _MockTransport(
        rounds=[
            [
                BaseEvent(event_type=ExecutionEventType.LLM_START, author="RemoteAgent"),
                BaseEvent(event_type=ExecutionEventType.LLM_END, author="RemoteAgent"),
                BaseEvent(
                    event_type=ExecutionEventType.RESPONSE,
                    author="RemoteAgent",
                    content=[Part(text="Paris is the capital of France.")],
                    is_partial=False,
                ),
            ]
        ]
    )

    agent = DelegateAgent(name="RemoteAgent", description="geography expert", transport=transport)

    runner = Runner()
    events = [
        e
        async for e in runner.run(
            application_name="app",
            user_id="u1",
            root=agent,
            user_input=_user_input("What is the capital of France?"),
            conversation_id="conv-plat-1",
        )
    ]

    event_types = [e.event_type for e in events]
    assert ExecutionEventType.LLM_START in event_types
    assert ExecutionEventType.LLM_END in event_types
    final = [e for e in events if e.event_type == ExecutionEventType.RESPONSE and not e.is_partial]
    assert final
    text = " ".join(p.text or "" for p in final[-1].content if p.text)
    assert "France" in text


# ---------------------------------------------------------------------------
# Test 2: remote runtime has sub-agents (delegate agent has subagents scenario)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_platform_delegate_remote_has_subagents():
    """Remote runtime delegates to its own sub-agent; full event sequence surfaced to platform.

    Realistic sequence emitted by a remote Domyn runtime when the root agent calls a sub-agent:

      Root LLM_START → Root LLM_END → AGENT_CALL →
      Sub AGENT_START (task in content + parameters) →
      Sub LLM_START → Sub LLM_END →
      Sub AGENT_END (cast from sub's RESPONSE by the remote runner) →
      Root LLM_START (second call) → Root LLM_END →
      Root RESPONSE (final, non-partial)

    Note: sub-agents end with AGENT_END, never a bare non-partial RESPONSE.  The remote
    runner converts the planner's RESPONSE to AGENT_END when exiting sub-agent context.
    A forwarded sub-agent non-partial RESPONSE would correctly end the session immediately,
    so a well-behaved remote runtime converts it to AGENT_END to let the root continue.
    """
    transport = _MockTransport(
        rounds=[
            [
                # Remote root makes its first LLM call
                BaseEvent(event_type=ExecutionEventType.LLM_START, author="RemoteRoot"),
                BaseEvent(event_type=ExecutionEventType.LLM_END, author="RemoteRoot"),
                # Root decides to delegate to a specialist sub-agent
                BaseEvent(
                    event_type=ExecutionEventType.AGENT_CALL,
                    author="RemoteRoot",
                    content=[Part(text="dispatching to specialist")],
                    action=AgentAction(
                        name="RemoteSpecialist",
                        parameters={"task": "handle the details"},
                        thought="need specialist",
                    ),
                ),
                # Sub-agent starts: AGENT_START carries the task in both content and parameters
                _sub_agent_start("RemoteSpecialist", "handle the details"),
                # Sub-agent LLM call
                BaseEvent(event_type=ExecutionEventType.LLM_START, author="RemoteSpecialist"),
                BaseEvent(event_type=ExecutionEventType.LLM_END, author="RemoteSpecialist"),
                # Sub-agent ends with AGENT_END (the remote runner casts its RESPONSE to AGENT_END)
                BaseEvent(
                    event_type=ExecutionEventType.AGENT_END,
                    author="RemoteSpecialist",
                    content=[Part(text="specialist result: 42")],
                ),
                # Remote root resumes after sub-agent returns
                BaseEvent(event_type=ExecutionEventType.LLM_START, author="RemoteRoot"),
                BaseEvent(event_type=ExecutionEventType.LLM_END, author="RemoteRoot"),
                BaseEvent(
                    event_type=ExecutionEventType.RESPONSE,
                    author="RemoteRoot",
                    content=[Part(text="final answer incorporating specialist result: 42")],
                    is_partial=False,
                ),
            ]
        ]
    )

    agent = DelegateAgent(name="RemoteRoot", description="multi-agent remote", transport=transport)

    runner = Runner()
    events = [
        e
        async for e in runner.run(
            application_name="app",
            user_id="u1",
            root=agent,
            user_input=_user_input("complex query"),
            conversation_id="conv-plat-2",
        )
    ]

    event_types = [e.event_type for e in events]

    # Root LLM lifecycle events are surfaced
    assert ExecutionEventType.LLM_START in event_types
    assert ExecutionEventType.LLM_END in event_types

    # Sub-agent lifecycle events are forwarded for observability
    assert ExecutionEventType.AGENT_CALL in event_types
    assert ExecutionEventType.AGENT_START in event_types

    # RemoteSpecialist AGENT_END forwarded verbatim (different author from DelegateAgent)
    specialist_ends = [
        e
        for e in events
        if e.event_type == ExecutionEventType.AGENT_END and e.author == "RemoteSpecialist"
    ]
    assert specialist_ends, "RemoteSpecialist AGENT_END must be forwarded"

    # Session does NOT terminate at sub-agent AGENT_END — root's final RESPONSE arrives
    final = [e for e in events if e.event_type == ExecutionEventType.RESPONSE and not e.is_partial]
    assert final
    text = " ".join(p.text or "" for p in final[-1].content if p.text)
    assert "specialist" in text.lower() or "final" in text.lower()


# ---------------------------------------------------------------------------
# Test 3: remote calls a platform FunctionTool ("delegate tool of the platform")
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_platform_delegate_with_function_tool_roundtrip():
    """Remote runtime calls a FunctionTool exposed on the platform's DelegateAgent.

    Realistic round 0 (remote LLM calls the tool):
      LLM_START → LLM_END → TOOL_START

    Runner executes the tool locally and calls send_result(TOOL_END).

    Realistic round 1 (remote resumes after tool result):
      LLM_START → LLM_END → RESPONSE (final)
    """

    def lookup_exchange_rate(from_currency: str, to_currency: str) -> float:
        """Return a hard-coded exchange rate for testing."""
        rates = {("USD", "EUR"): 0.92, ("EUR", "USD"): 1.09}
        return rates.get((from_currency, to_currency), 1.0)

    platform_tool = FunctionTool(func=lookup_exchange_rate)

    transport = _MockTransport(
        rounds=[
            # Round 0: remote LLM decides to call the platform tool
            [
                BaseEvent(event_type=ExecutionEventType.LLM_START, author="RemoteAgent"),
                BaseEvent(event_type=ExecutionEventType.LLM_END, author="RemoteAgent"),
                BaseEvent(
                    event_type=ExecutionEventType.TOOL_START,
                    author="RemoteAgent",
                    action=ToolAction(
                        name="lookup_exchange_rate",
                        parameters={"from_currency": "USD", "to_currency": "EUR"},
                        call_id="call-fx-1",
                    ),
                ),
            ],
            # Round 1 (after TOOL_END sent back): remote produces final answer
            [
                BaseEvent(event_type=ExecutionEventType.LLM_START, author="RemoteAgent"),
                BaseEvent(event_type=ExecutionEventType.LLM_END, author="RemoteAgent"),
                BaseEvent(
                    event_type=ExecutionEventType.RESPONSE,
                    author="RemoteAgent",
                    content=[Part(text="1 USD = 0.92 EUR")],
                    is_partial=False,
                ),
            ],
        ]
    )

    agent = DelegateAgent(
        name="RemoteAgent",
        description="finance assistant",
        transport=transport,
        tools=[platform_tool],
    )

    runner = Runner()
    events = [
        e
        async for e in runner.run(
            application_name="app",
            user_id="u1",
            root=agent,
            user_input=_user_input("What is the USD to EUR rate?"),
            conversation_id="conv-plat-3",
        )
    ]

    event_types = [e.event_type for e in events]

    # Runner observed tool execution
    assert ExecutionEventType.TOOL_START in event_types
    assert ExecutionEventType.TOOL_END in event_types

    # TOOL_END was sent back to remote with the correct result
    assert len(transport.results_received) == 1
    result_event = transport.results_received[0]
    assert result_event.event_type == ExecutionEventType.TOOL_END
    assert result_event.action is not None
    assert result_event.action.observation == pytest.approx(0.92)

    # Final RESPONSE surfaced
    final = [e for e in events if e.event_type == ExecutionEventType.RESPONSE and not e.is_partial]
    assert final
    assert "0.92" in " ".join(p.text or "" for p in final[-1].content if p.text)


# ---------------------------------------------------------------------------
# Test 4: remote calls a DelegateTool with a wrapped callable on the platform
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_platform_delegate_with_delegate_tool_wrapped():
    """Remote runtime calls a platform DelegateTool whose `wrapped` callable runs locally.

    DelegateTool.wrapped acts like a FunctionTool from the runner's perspective:
    the runner executes wrapped(), gets the result, and sends TOOL_END back.

    Realistic round 0: LLM_START → LLM_END → TOOL_START
    Realistic round 1: LLM_START → LLM_END → RESPONSE (final)
    """
    _calls: list[dict] = []

    def search_knowledge_base(query: str, top_k: int = 3) -> list[str]:
        _calls.append({"query": query, "top_k": top_k})
        return [f"doc{i}: relevant content for '{query}'" for i in range(top_k)]

    platform_delegate_tool = DelegateTool(
        tool_name="search_knowledge_base",
        tool_description="Search internal knowledge base",
        tool_parameters={
            "query": {"type": "string", "description": "search query"},
            "top_k": {"type": "integer", "description": "number of results"},
        },
        required_params=["query"],
        wrapped=search_knowledge_base,
    )

    transport = _MockTransport(
        rounds=[
            # Round 0: remote LLM decides to call the platform tool
            [
                BaseEvent(event_type=ExecutionEventType.LLM_START, author="RemoteAgent"),
                BaseEvent(event_type=ExecutionEventType.LLM_END, author="RemoteAgent"),
                BaseEvent(
                    event_type=ExecutionEventType.TOOL_START,
                    author="RemoteAgent",
                    action=ToolAction(
                        name="search_knowledge_base",
                        parameters={"query": "machine learning basics", "top_k": 2},
                        call_id="call-kb-1",
                    ),
                ),
            ],
            # Round 1 (after TOOL_END sent back): remote produces final answer
            [
                BaseEvent(event_type=ExecutionEventType.LLM_START, author="RemoteAgent"),
                BaseEvent(event_type=ExecutionEventType.LLM_END, author="RemoteAgent"),
                BaseEvent(
                    event_type=ExecutionEventType.RESPONSE,
                    author="RemoteAgent",
                    content=[Part(text="Based on the knowledge base: doc0: relevant content")],
                    is_partial=False,
                ),
            ],
        ]
    )

    agent = DelegateAgent(
        name="RemoteAgent",
        description="knowledge agent",
        transport=transport,
        tools=[platform_delegate_tool],
    )

    runner = Runner()
    events = [
        e
        async for e in runner.run(
            application_name="app",
            user_id="u1",
            root=agent,
            user_input=_user_input("Tell me about machine learning"),
            conversation_id="conv-plat-4",
        )
    ]

    # Wrapped callable was invoked
    assert _calls, "wrapped callable must have been called"
    assert _calls[0]["query"] == "machine learning basics"
    assert _calls[0]["top_k"] == 2

    # TOOL_END sent back to remote
    assert transport.results_received, "TOOL_END must be sent back to remote"
    result_event = transport.results_received[0]
    assert result_event.action is not None
    obs = result_event.action.observation
    assert obs is not None

    # Final RESPONSE surfaced
    final = [e for e in events if e.event_type == ExecutionEventType.RESPONSE and not e.is_partial]
    assert final


# ---------------------------------------------------------------------------
# Test 5: platform root Agent delegates to DelegateAgent sub-agent
#         where the remote runtime itself has sub-agents
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_platform_root_agent_delegates_to_remote_with_subagents():
    """Platform root Agent (with MockLLM) hands off to a DelegateAgent sub-agent.

    The remote runtime emits a realistic sub-agent event chain; the DelegateAgent
    (running as sub-agent, is_subagent=True) converts the remote root's final RESPONSE
    to AGENT_END so the platform root can resume and produce a final synthesis.

    Remote event sequence (transport mock):
      Root LLM_START → Root LLM_END →
      AGENT_CALL → Sub AGENT_START (task in content+params) →
      Sub LLM_START → Sub LLM_END →
      Sub AGENT_END (cast from sub's RESPONSE) →
      Root LLM_START → Root LLM_END →
      Root RESPONSE (final) — DelegateAgent converts this to AGENT_END for the platform root.
    """
    transport = _MockTransport(
        rounds=[
            [
                # Remote root's first LLM call decides to delegate to a sub-agent
                BaseEvent(event_type=ExecutionEventType.LLM_START, author="RemoteRoot"),
                BaseEvent(event_type=ExecutionEventType.LLM_END, author="RemoteRoot"),
                BaseEvent(
                    event_type=ExecutionEventType.AGENT_CALL,
                    author="RemoteRoot",
                    content=[Part(text="calling remote sub")],
                    action=AgentAction(
                        name="RemoteSub",
                        parameters={"task": "fetch the data"},
                        thought="",
                    ),
                ),
                # Sub-agent starts: task in both content and action.parameters
                _sub_agent_start("RemoteSub", "fetch the data"),
                BaseEvent(event_type=ExecutionEventType.LLM_START, author="RemoteSub"),
                BaseEvent(event_type=ExecutionEventType.LLM_END, author="RemoteSub"),
                # Sub-agent ends with AGENT_END (cast from its RESPONSE by the remote runner)
                BaseEvent(
                    event_type=ExecutionEventType.AGENT_END,
                    author="RemoteSub",
                    content=[Part(text="data fetched successfully")],
                ),
                # Remote root resumes after sub-agent completes
                BaseEvent(event_type=ExecutionEventType.LLM_START, author="RemoteRoot"),
                BaseEvent(event_type=ExecutionEventType.LLM_END, author="RemoteRoot"),
                # Final remote RESPONSE — DelegateAgent (is_subagent=True) converts this
                # to AGENT_END(RemoteWorker) so the platform root can resume.
                BaseEvent(
                    event_type=ExecutionEventType.RESPONSE,
                    author="RemoteRoot",
                    content=[Part(text="remote task complete: data fetched successfully")],
                    is_partial=False,
                ),
            ]
        ]
    )

    delegate = DelegateAgent(
        name="RemoteWorker",
        description="handles data fetching tasks",
        transport=transport,
    )

    llm = _MockLLMProvider(
        responses=[
            # First call: hand off to RemoteWorker
            LLMResponse(
                content="",
                generated_tokens=4,
                execution_time=0.0,
                tool_calls=[
                    {  # type: ignore[list-item]
                        "id": "tc-1",
                        "type": "function",
                        "function": {
                            "name": "RemoteWorker",
                            "arguments": '{"task":"fetch the data"}',
                        },
                    }
                ],
            ),
            # Second call: after AGENT_END from delegate — final synthesis
            LLMResponse(
                content="Task completed. Data was fetched successfully by the remote worker.",
                generated_tokens=10,
                execution_time=0.0,
            ),
        ]
    )

    root = _make_agent(
        name="PlatformOrchestrator",
        description="orchestrates tasks",
        llm=llm,
        sub_agents=[delegate],
    )

    runner = Runner()
    events = [
        e
        async for e in runner.run(
            application_name="app",
            user_id="u1",
            root=root,
            user_input=_user_input("Please fetch the data"),
            conversation_id="conv-plat-5",
        )
    ]

    event_types = [e.event_type for e in events]

    # Delegate sub-agent lifecycle (runner emits AGENT_START when transitioning to RemoteWorker)
    assert ExecutionEventType.AGENT_START in event_types
    assert ExecutionEventType.AGENT_END in event_types

    # Remote sub-agent events are forwarded for observability
    remote_sub_ends = [
        e
        for e in events
        if e.event_type == ExecutionEventType.AGENT_END and e.author == "RemoteSub"
    ]
    assert remote_sub_ends, "RemoteSub AGENT_END must be forwarded"

    # Platform root produces final RESPONSE after delegate completes
    final = [e for e in events if e.event_type == ExecutionEventType.RESPONSE and not e.is_partial]
    assert final
    text = " ".join(p.text or "" for p in final[-1].content if p.text)
    assert (
        "remote worker" in text.lower() or "fetched" in text.lower() or "completed" in text.lower()
    )
