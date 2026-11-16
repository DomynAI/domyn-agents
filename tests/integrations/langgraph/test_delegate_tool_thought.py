"""The model's thought reaches a delegate tool's TOOL_START under LangGraph.

The callback handler does not emit a delegate tool's TOOL_START (the adapter
forwards it), so it used to return before taking the thought it had stashed for
that call: the TOOL_START was forwarded with thought=None, and the thought stayed in
``_pending_thoughts``. Now the handler keeps it under the tool's run id and
``PlatformTool._arun`` hands it to the relay.
"""

from typing import Any
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from domyn.agents.integrations.langgraph import PlatformCallbackHandler, PlatformRelay
from domyn.agents.integrations.langgraph.domyn_platform import PlatformToolRegistry

_DEFS = [
    {
        "name": "web_search",
        "description": "Search the web.",
        "parameters": [{"name": "query", "type": "str", "is_required": True}],
    }
]


class _Runtime:
    """Stands in for LangGraphAdapter: records what the relay passes."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], str | None]] = []

    async def call_platform_tool(
        self, name: str, kwargs: dict[str, Any], thought: str | None = None
    ) -> str:
        self.calls.append((name, kwargs, thought))
        return "result"


def _tool_and_runtime() -> tuple[Any, _Runtime]:
    relay = PlatformRelay()
    runtime = _Runtime()
    relay.bind(runtime)
    return PlatformToolRegistry(_DEFS, relay=relay).get_tool("web_search"), runtime


async def _handler_with_thought(thought: str, tool_call_id: str) -> PlatformCallbackHandler:
    async def emit(_: Any) -> None:
        pass

    handler = PlatformCallbackHandler(project_id="test", emit=emit, session_id="s")
    rid = uuid4()
    await handler.on_llm_start(None, ["p"], run_id=rid)
    message = AIMessage(
        content=thought,
        tool_calls=[{"name": "web_search", "args": {"query": "news"}, "id": tool_call_id}],
    )
    await handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=rid)
    return handler


@pytest.mark.asyncio
async def test_delegate_tool_call_carries_the_thought() -> None:
    tool, runtime = _tool_and_runtime()
    handler = await _handler_with_thought("Let me search for that.", "t1")

    await tool.ainvoke(
        {"type": "tool_call", "name": "web_search", "args": {"query": "news"}, "id": "t1"},
        config={"callbacks": [handler]},
    )

    assert runtime.calls == [("web_search", {"query": "news"}, "Let me search for that.")]
    # Consumed, not left behind for the handler's lifetime.
    assert handler._pending_thoughts == {}
    assert handler._delegate_thoughts == {}


@pytest.mark.asyncio
async def test_a_call_from_another_response_gets_no_thought() -> None:
    tool, runtime = _tool_and_runtime()
    handler = await _handler_with_thought("Let me search for that.", "t1")

    await tool.ainvoke(
        {"type": "tool_call", "name": "web_search", "args": {"query": "news"}, "id": "t2"},
        config={"callbacks": [handler]},
    )

    assert runtime.calls == [("web_search", {"query": "news"}, None)]


@pytest.mark.asyncio
async def test_a_delegate_tool_invoked_directly_still_works() -> None:
    tool, runtime = _tool_and_runtime()
    assert await tool.ainvoke({"query": "news"}) == "result"
    assert runtime.calls == [("web_search", {"query": "news"}, None)]


@pytest.mark.asyncio
async def test_relay_always_forwards_thought() -> None:
    relay = PlatformRelay()
    runtime = _Runtime()
    relay.bind(runtime)
    await relay.call_platform_tool("web_search", {"query": "q"})
    await relay.call_platform_tool("web_search", {"query": "q"}, thought="t")
    assert runtime.calls == [
        ("web_search", {"query": "q"}, None),
        ("web_search", {"query": "q"}, "t"),
    ]
