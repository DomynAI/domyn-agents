"""PlatformCallbackHandler.on_llm_end thought/text emission."""

import json
from typing import Annotated
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langgraph.graph.message import add_messages
from langgraph.types import Command
from pydantic import BaseModel

from domyn.agents import ExecutionEventType
from domyn.agents.integrations.langgraph import PlatformCallbackHandler


def _make_handler(turn_id=None):
    events = []

    async def emit(ev):
        events.append(ev)

    handler = PlatformCallbackHandler(project_id="test", emit=emit, session_id="s", turn_id=turn_id)
    return handler, events


def _llm_result(content, tool_calls=None, reasoning_content=None):
    extra = {"reasoning_content": reasoning_content} if reasoning_content else {}
    msg = AIMessage(content=content, tool_calls=tool_calls or [], additional_kwargs=extra)
    return LLMResult(generations=[[ChatGeneration(message=msg)]])


def _llm_end_parts(events):
    end = [e for e in events if e.event_type == ExecutionEventType.LLM_END]
    assert len(end) == 1, f"expected one LLM_END, got {len(end)}"
    parts = end[0].content
    return ([p.thought for p in parts if p.thought], [p.text for p in parts if p.text])


@pytest.mark.asyncio
async def test_preamble_with_tool_calls_emits_thought_only():
    handler, events = _make_handler()
    rid = uuid4()
    await handler.on_llm_start(None, ["p"], run_id=rid)
    await handler.on_llm_end(
        _llm_result(
            "Let me check both stocks.", tool_calls=[{"name": "fin", "args": {}, "id": "t1"}]
        ),
        run_id=rid,
    )
    thoughts, texts = _llm_end_parts(events)
    assert thoughts == ["Let me check both stocks."]
    assert texts == []  # NOT duplicated as text


@pytest.mark.asyncio
async def test_final_answer_no_tool_calls_emits_text_only():
    handler, events = _make_handler()
    rid = uuid4()
    await handler.on_llm_start(None, ["p"], run_id=rid)
    await handler.on_llm_end(_llm_result("The verdict is GOOGL."), run_id=rid)
    thoughts, texts = _llm_end_parts(events)
    assert thoughts == []
    assert texts == ["The verdict is GOOGL."]


@pytest.mark.asyncio
async def test_distinct_reasoning_and_text_both_emitted():
    handler, events = _make_handler()
    rid = uuid4()
    await handler.on_llm_start(None, ["p"], run_id=rid)
    await handler.on_llm_end(_llm_result("Answer.", reasoning_content="Reasoning."), run_id=rid)
    thoughts, texts = _llm_end_parts(events)
    assert thoughts == ["Reasoning."]
    assert texts == ["Answer."]  # de-dup only collapses the equal-string preamble case


def _tool_start_actions(events):
    return [
        e.action
        for e in events
        if e.event_type == ExecutionEventType.TOOL_START and e.action is not None
    ]


@pytest.mark.asyncio
async def test_tool_start_carries_thought_by_tool_call_id():
    """The platform renders thoughts from action.thought on TOOL_START, not from
    LLM_END Part(thought). The preamble is stashed against the originating
    tool-call id and popped when that tool starts (matched via tool_call_id)."""
    handler, events = _make_handler()
    rid = uuid4()
    await handler.on_llm_start(None, ["p"], run_id=rid)
    await handler.on_llm_end(
        _llm_result(
            "I'll check the financials.",
            tool_calls=[{"name": "financial_metrics_agent", "args": {}, "id": "t1"}],
        ),
        run_id=rid,
    )
    # The matching tool call (by tool_call_id) consumes the stashed thought.
    await handler.on_tool_start(
        {"name": "financial_metrics_agent"}, "AAPL", run_id=uuid4(), tool_call_id="t1"
    )
    actions = _tool_start_actions(events)
    assert actions[-1].thought == "I'll check the financials."

    # The same tool call starting again gets no thought (popped once consumed).
    await handler.on_tool_start(
        {"name": "financial_metrics_agent"}, "AAPL", run_id=uuid4(), tool_call_id="t1"
    )
    actions = _tool_start_actions(events)
    assert actions[-1].thought is None


@pytest.mark.asyncio
async def test_tool_start_thought_not_attributed_to_unrelated_tool():
    """A tool from a different turn (different tool_call_id) must not pick up
    another turn's preamble — the failure mode of the old single-slot carry."""
    handler, events = _make_handler()
    rid = uuid4()
    await handler.on_llm_start(None, ["p"], run_id=rid)
    await handler.on_llm_end(
        _llm_result(
            "I'll check the financials.",
            tool_calls=[{"name": "financial_metrics_agent", "args": {}, "id": "t1"}],
        ),
        run_id=rid,
    )
    # An unrelated tool (no matching tool_call_id) gets no thought.
    await handler.on_tool_start({"name": "stock_price"}, "AAPL", run_id=uuid4(), tool_call_id="t2")
    actions = _tool_start_actions(events)
    assert actions[-1].name == "stock_price"
    assert actions[-1].thought is None


def _thought_by_tool_name(events):
    return {a.name: a.thought for a in _tool_start_actions(events)}


@pytest.mark.asyncio
async def test_parallel_tools_one_turn_thought_on_first_only():
    """One LLM turn emitting several tool calls: the turn-level preamble attaches
    to the FIRST tool call only (mirrors the core tool_use_thought planner), and
    the sibling parallel tools carry no thought."""
    handler, events = _make_handler()
    rid = uuid4()
    await handler.on_llm_start(None, ["p"], run_id=rid)
    await handler.on_llm_end(
        _llm_result(
            "Checking both stocks.",
            tool_calls=[
                {"name": "fin", "args": {}, "id": "t1"},
                {"name": "price", "args": {}, "id": "t2"},
            ],
        ),
        run_id=rid,
    )
    await handler.on_tool_start({"name": "fin"}, "", run_id=uuid4(), tool_call_id="t1")
    await handler.on_tool_start({"name": "price"}, "", run_id=uuid4(), tool_call_id="t2")
    by_name = _thought_by_tool_name(events)
    assert by_name == {"fin": "Checking both stocks.", "price": None}


@pytest.mark.asyncio
async def test_parallel_tools_thought_attribution_is_start_order_independent():
    """Tools of one turn may start in any order (parallel execution). The preamble
    must still land on the first tool CALL, not on whichever tool starts first —
    the precise bug a single mutable slot would introduce."""
    handler, events = _make_handler()
    rid = uuid4()
    await handler.on_llm_start(None, ["p"], run_id=rid)
    await handler.on_llm_end(
        _llm_result(
            "Checking both stocks.",
            tool_calls=[
                {"name": "fin", "args": {}, "id": "t1"},
                {"name": "price", "args": {}, "id": "t2"},
            ],
        ),
        run_id=rid,
    )
    # Second tool call starts FIRST; a single-slot carry would hand it the preamble.
    await handler.on_tool_start({"name": "price"}, "", run_id=uuid4(), tool_call_id="t2")
    await handler.on_tool_start({"name": "fin"}, "", run_id=uuid4(), tool_call_id="t1")
    by_name = _thought_by_tool_name(events)
    assert by_name == {"price": None, "fin": "Checking both stocks."}


@pytest.mark.asyncio
async def test_concurrent_turns_no_cross_attribution_under_interleaving():
    """Two overlapping LLM turns (e.g. parallel graph branches) whose callbacks
    interleave: each tool must receive ITS OWN turn's preamble. The old single
    slot held only the most recent thought, so the earlier turn's tool would be
    mis-labelled with the later turn's preamble (or lose its own)."""
    handler, events = _make_handler()
    run_a, run_b = uuid4(), uuid4()
    await handler.on_llm_start(None, ["p"], run_id=run_a)
    await handler.on_llm_start(None, ["p"], run_id=run_b)
    # Both turns complete before either tool starts — last-writer-wins would clobber.
    await handler.on_llm_end(
        _llm_result("Thought A.", tool_calls=[{"name": "tool_a", "args": {}, "id": "a1"}]),
        run_id=run_a,
    )
    await handler.on_llm_end(
        _llm_result("Thought B.", tool_calls=[{"name": "tool_b", "args": {}, "id": "b1"}]),
        run_id=run_b,
    )
    # Interleaved starts: B before A.
    await handler.on_tool_start({"name": "tool_b"}, "", run_id=uuid4(), tool_call_id="b1")
    await handler.on_tool_start({"name": "tool_a"}, "", run_id=uuid4(), tool_call_id="a1")
    by_name = _thought_by_tool_name(events)
    assert by_name == {"tool_a": "Thought A.", "tool_b": "Thought B."}


@pytest.mark.asyncio
async def test_every_emitted_event_carries_turn_id():
    """The handler stamps its constructor turn_id onto every emitted event (same
    contract as conversation_id), so the platform can correlate the whole trace
    with the turn. Exercises LLM, tool, and error paths."""
    handler, events = _make_handler(turn_id="turn-XYZ")
    rid = uuid4()
    await handler.on_llm_start(None, ["p"], run_id=rid)
    await handler.on_llm_end(
        _llm_result("Checking.", tool_calls=[{"name": "fin", "args": {}, "id": "t1"}]),
        run_id=rid,
    )
    await handler.on_tool_start({"name": "fin"}, "AAPL", run_id=rid, tool_call_id="t1")
    await handler.on_tool_end("ok", run_id=rid)
    err_rid = uuid4()
    await handler.on_llm_start(None, ["p"], run_id=err_rid)
    await handler.on_llm_error(RuntimeError("boom"), run_id=err_rid)

    types = {e.event_type for e in events}
    assert ExecutionEventType.LLM_START in types
    assert ExecutionEventType.LLM_END in types
    assert ExecutionEventType.TOOL_START in types
    assert ExecutionEventType.TOOL_END in types
    assert ExecutionEventType.ERROR_OCCURRED in types
    assert events, "expected the handler to emit events"
    assert all(e.turn_id == "turn-XYZ" for e in events), (
        "every handler-emitted event must carry the turn_id: "
        f"{[(e.event_type, e.turn_id) for e in events if e.turn_id != 'turn-XYZ']}"
    )


# ---- a tool's own metadata reaching the TOOL_END event ----


async def _run_tool(handler, output, *, run_metadata=None):
    """Drive one full tool span and return the terminal TOOL_END / TOOL_ERROR."""
    rid = uuid4()
    await handler.on_tool_start({"name": "kg_nl_search"}, "q", run_id=rid, metadata=run_metadata)
    await handler.on_tool_end(output, run_id=rid)
    return rid


def _tool_end(events):
    ends = [
        e
        for e in events
        if e.event_type in (ExecutionEventType.TOOL_END, ExecutionEventType.TOOL_ERROR)
    ]
    assert len(ends) == 1, f"expected one terminal tool event, got {len(ends)}"
    return ends[0]


def _artifact_result(**kwargs):
    return {"success": True, **kwargs}


@pytest.mark.asyncio
async def test_command_wrapped_tool_message_is_parsed_as_tool_result():
    """A tool returning Command(update={"messages": [ToolMessage(artifact=...)]}).

    ``Command`` is a ToolOutputMixin, so langchain hands it to on_tool_end
    verbatim instead of wrapping it in a ToolMessage — the artifact has to be
    dug out of the state update or the whole result is lost.
    """
    handler, events = _make_handler()
    message = ToolMessage(
        content="the answer",
        tool_call_id="t1",
        artifact=_artifact_result(data="the answer", metadata={"token_usage": {"total": 12}}),
    )
    await _run_tool(handler, Command(update={"messages": [message]}))

    end = _tool_end(events)
    assert end.event_type == ExecutionEventType.TOOL_END
    assert end.action.observation == "the answer"
    assert end.metadata == {"token_usage": {"total": 12}}


@pytest.mark.asyncio
async def test_command_update_as_pairs_is_parsed():
    """LangGraph also accepts the ``[(key, value)]`` form of a state update."""
    handler, events = _make_handler()
    message = ToolMessage(
        content="ok", tool_call_id="t1", artifact=_artifact_result(data="ok", metadata={"a": 1})
    )
    await _run_tool(handler, Command(update=[("messages", [message])]))

    assert _tool_end(events).metadata == {"a": 1}


@pytest.mark.asyncio
async def test_last_message_wins_in_a_multi_message_update():
    handler, events = _make_handler()
    first = ToolMessage(
        content="stale", tool_call_id="t0", artifact=_artifact_result(data="stale", metadata={})
    )
    last = ToolMessage(
        content="fresh",
        tool_call_id="t1",
        artifact=_artifact_result(data="fresh", metadata={"which": "last"}),
    )
    await _run_tool(handler, Command(update={"messages": [first, last]}))

    end = _tool_end(events)
    assert end.action.observation == "fresh"
    assert end.metadata == {"which": "last"}


@pytest.mark.asyncio
async def test_annotated_state_object_update_is_unwrapped():
    """A graph whose state is a pydantic model updates with the model, not a dict.
    Normalisation is LangGraph's own, so this shape works like the others."""

    class GraphState(BaseModel):
        messages: Annotated[list, add_messages] = []

    handler, events = _make_handler()
    message = ToolMessage(
        content="ok", tool_call_id="t1", artifact=_artifact_result(data="ok", metadata={"s": 1})
    )
    await _run_tool(handler, Command(update=GraphState(messages=[message])))

    end = _tool_end(events)
    assert end.action.observation == "ok"
    assert end.metadata == {"s": 1}


@pytest.mark.asyncio
async def test_block_list_content_is_read_as_text():
    """``content`` may be a list of content blocks; the observation must be the
    text, not the repr of a list of dicts."""
    handler, events = _make_handler()
    message = ToolMessage(
        content=[{"type": "text", "text": "block a"}, {"type": "text", "text": " block b"}],
        tool_call_id="t1",
    )
    await _run_tool(handler, Command(update={"messages": [message]}))

    assert _tool_end(events).action.observation == "block a block b"


@pytest.mark.asyncio
async def test_unserializable_metadata_costs_the_key_not_the_event():
    """The runtime serializes the event outside its own try/except and the error
    surfaces inside a @_safe callback, so an unserializable value would take the
    whole TOOL_END with it. Only the offending key is dropped."""

    class NotSerializable:
        pass

    handler, events = _make_handler()
    message = ToolMessage(
        content="ok",
        tool_call_id="t1",
        artifact=_artifact_result(
            data="ok", metadata={"good": 1, "bad": NotSerializable(), "also_good": [2]}
        ),
    )
    await _run_tool(handler, Command(update={"messages": [message]}))

    end = _tool_end(events)
    assert end.metadata == {"good": 1, "also_good": [2]}
    end.model_dump_json()  # what the runtime does; must not raise


@pytest.mark.asyncio
async def test_metadata_only_artifact_falls_back_to_the_message_text():
    """Relaxing the ``data`` guard for artifacts must not cost the observation:
    the answer the model saw is still on the message."""
    handler, events = _make_handler()
    message = ToolMessage(
        content="the answer",
        tool_call_id="t1",
        artifact={"success": True, "metadata": {"context": ["chunk"]}},
    )
    await _run_tool(handler, Command(update={"messages": [message]}))

    end = _tool_end(events)
    assert end.action.observation == "the answer"
    assert end.metadata == {"context": ["chunk"]}


@pytest.mark.asyncio
async def test_falsy_data_is_not_overwritten_by_the_fallback():
    """A tool returning 0 / "" / [] as its data meant to — only a missing one
    falls back."""
    handler, events = _make_handler()
    message = ToolMessage(
        content="ignored", tool_call_id="t1", artifact=_artifact_result(data=0, metadata={})
    )
    await _run_tool(handler, Command(update={"messages": [message]}))

    assert _tool_end(events).action.observation == 0


@pytest.mark.asyncio
async def test_bare_tool_message_artifact_still_works():
    """The pre-existing content_and_artifact path keeps its behaviour, and now
    carries the metadata too."""
    handler, events = _make_handler()
    message = ToolMessage(
        content="unused",
        tool_call_id="t1",
        artifact=_artifact_result(data="the answer", metadata={"k": "v"}),
    )
    await _run_tool(handler, message)

    end = _tool_end(events)
    assert end.action.observation == "the answer"
    assert end.metadata == {"k": "v"}


@pytest.mark.asyncio
async def test_error_tool_result_carries_metadata_on_tool_error():
    handler, events = _make_handler()
    message = ToolMessage(
        content="boom",
        tool_call_id="t1",
        artifact={
            "success": False,
            "data": None,
            "error": "boom",
            "error_code": "500",
            "metadata": {"attempts": 3},
        },
    )
    await _run_tool(handler, Command(update={"messages": [message]}))

    end = _tool_end(events)
    assert end.event_type == ExecutionEventType.TOOL_ERROR
    assert end.error_message == "boom"
    assert end.metadata == {"attempts": 3}


@pytest.mark.asyncio
async def test_command_without_tool_result_falls_back_to_message_content():
    """No artifact: the observation is the content the model saw, never the
    ``Command(update={...})`` repr."""
    handler, events = _make_handler()
    message = ToolMessage(content="plain answer", tool_call_id="t1")
    await _run_tool(handler, Command(update={"messages": [message]}))

    end = _tool_end(events)
    assert end.event_type == ExecutionEventType.TOOL_END
    assert end.action.observation == "plain answer"
    assert end.metadata == {}


@pytest.mark.asyncio
async def test_non_tool_result_artifact_falls_through_to_content():
    """An artifact carrying domain data rather than a ToolResult is not a
    failure — the content is still read."""
    handler, events = _make_handler()
    message = ToolMessage(
        content=json.dumps({"success": True, "data": "from content", "metadata": {"src": "c"}}),
        tool_call_id="t1",
        artifact={"nodes": [], "edges": []},
    )
    await _run_tool(handler, Command(update={"messages": [message]}))

    end = _tool_end(events)
    assert end.action.observation == "from content"
    assert end.metadata == {"src": "c"}


@pytest.mark.asyncio
async def test_command_with_no_message_update_is_still_stringified():
    """A Command that routes without touching messages has no tool result and no
    content to fall back on; the old behaviour stands."""
    handler, events = _make_handler()
    await _run_tool(handler, Command(goto="next_node"))

    end = _tool_end(events)
    assert end.event_type == ExecutionEventType.TOOL_END
    assert "next_node" in end.action.observation
    assert end.metadata == {}


@pytest.mark.asyncio
async def test_plain_dict_output_is_not_mistaken_for_a_command():
    """``dict`` has an ``update`` method, so a dict output must be excluded from
    the Command unwrapping before the attribute is read, not after."""
    handler, events = _make_handler()
    await _run_tool(handler, {"success": True, "data": "d", "metadata": {"m": 1}})

    end = _tool_end(events)
    assert end.action.observation == "d"
    assert end.metadata == {"m": 1}


@pytest.mark.asyncio
async def test_plain_string_output_unchanged():
    handler, events = _make_handler()
    await _run_tool(handler, "just text")

    end = _tool_end(events)
    assert end.action.observation == "just text"
    assert end.metadata == {}


@pytest.mark.asyncio
async def test_run_metadata_never_reaches_the_wire():
    """The langchain run metadata carries the caller's run_config (PII) and
    langgraph's own plumbing. It resolves the session id and nothing else."""
    handler, events = _make_handler()
    run_metadata = {
        "session_id": "sess-1",
        "run_config": {"user_email": "someone@example.com"},
        "langgraph_node": "tools",
        "thread_id": "conv-1",
    }
    message = ToolMessage(
        content="ok", tool_call_id="t1", artifact=_artifact_result(data="ok", metadata={"k": "v"})
    )
    await _run_tool(handler, Command(update={"messages": [message]}), run_metadata=run_metadata)

    end = _tool_end(events)
    assert end.metadata == {"k": "v"}
    assert end.interaction_id == "sess-1"
    assert all("run_config" not in e.metadata for e in events)
