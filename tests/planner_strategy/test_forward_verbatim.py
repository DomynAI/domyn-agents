"""Tests for FORWARD_VERBATIM action in ReactPlannerStrategy.next_actions()."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from domyn.agents.core import (
    AGENT_ERROR_CODE,
    AGENT_ERROR_MESSAGE,
    METADATA_FORWARD_VERBATIM,
    BaseEvent,
    ExecutionEventType,
    InteractionContext,
    Part,
    ToolAction,
)
from domyn.agents.planner_strategy import ReactPlannerStrategy
from domyn.agents.planner_strategy.prompting.data import ParsedAction, ResponseType
from domyn.agents.planner_strategy.utils import extract_by_dot_path


def _make_tool_end_event(event_id: str, observation) -> BaseEvent:
    return BaseEvent(
        event_id=event_id,
        event_type=ExecutionEventType.TOOL_END,
        author="SomeAgent",
        content=[],
        action=ToolAction(name="some_tool", parameters={}, observation=observation),
    )


def _make_context(interaction_history: list[BaseEvent] | None = None) -> InteractionContext:
    return InteractionContext(
        conversation_id="conv-1",
        user_id="user-1",
        agent_input=BaseEvent(
            event_type=ExecutionEventType.USER_INPUT,
            author="user",
            content=[Part(text="test")],
        ),
        interaction_history=interaction_history or [],
    )


def _make_agent(stream: bool = False) -> MagicMock:
    agent = MagicMock()
    agent.name = "TestAgent"
    agent.stream = stream
    agent.tools = []
    agent.memory_manager = None
    agent.llm_provider = MagicMock()
    return agent


def _forward_verbatim_action(source_id: str, source_key: str | None = None) -> ParsedAction:
    args = {"source_id": source_id}
    if source_key is not None:
        args["source_key"] = source_key
    return ParsedAction(
        action_name="FORWARD_VERBATIM",
        response_type=ResponseType.FORWARD_VERBATIM,
        action_arguments=args,
        thought="forwarding tool result",
        message="forwarding",
    )


@pytest.fixture
def planner() -> ReactPlannerStrategy:
    mock_cm = MagicMock()
    mock_cm.get_context = AsyncMock(return_value=[])
    mock_cm.prompting.parse_llm_response = AsyncMock(return_value=[])
    return ReactPlannerStrategy.model_construct(
        type="react",
        name="test-react",
        context_manager=mock_cm,
        use_stop=False,
        stream=False,
        max_iterations=25,
        verbose=False,
        _disable_structured_outputs=False,
    )


async def _collect_events(async_gen) -> list[BaseEvent]:
    events = []
    async for e in async_gen:
        events.append(e)
    return events


async def _run_forward_test(planner, context, agent, parsed):
    """Common helper: mock LLM, set parsed response, collect events."""

    async def fake_invoke_llm(*args, **kwargs):
        yield "raw llm output"

    planner.context_manager.prompting.parse_llm_response = AsyncMock(return_value=[parsed])

    with (
        patch.object(ReactPlannerStrategy, "invoke_llm", side_effect=fake_invoke_llm),
        patch.object(ReactPlannerStrategy, "get_structured_outputs", return_value=None),
    ):
        return await _collect_events(planner.next_actions(agent, context))


@pytest.mark.asyncio
async def test_forward_verbatim_use_stop_emits_agent_end(planner):
    """Sub-agent path (use_stop=True): FORWARD_VERBATIM hit emits AGENT_END
    with STOP(info=text) + skip metadata, NOT a RESPONSE event. Required so
    the parent agent loop terminates properly."""
    tool_event = _make_tool_end_event("evt-sub", observation="verbatim output")
    context = _make_context(interaction_history=[tool_event])
    parsed = _forward_verbatim_action("evt-sub")

    planner.use_stop = True
    planner.context_manager.prompting.stop_action.name = "STOP"

    events = await _run_forward_test(planner, context, _make_agent(), parsed)

    assert not any(e.event_type == ExecutionEventType.RESPONSE for e in events)
    agent_end = next(e for e in events if e.event_type == ExecutionEventType.AGENT_END)
    assert agent_end.action.name == "STOP"
    assert agent_end.action.parameters == {"info": "verbatim output"}
    assert agent_end.content[0].text == "verbatim output"
    assert agent_end.metadata.get(METADATA_FORWARD_VERBATIM) is True


@pytest.mark.asyncio
async def test_forward_verbatim_uses_observation_from_history(planner):
    """FORWARD_VERBATIM action yields a RESPONSE with the referenced
    observation and METADATA_FORWARD_VERBATIM flag."""
    tool_event = _make_tool_end_event("evt-123", observation={"result": "chart data"})
    context = _make_context(interaction_history=[tool_event])
    parsed = _forward_verbatim_action("evt-123")

    events = await _run_forward_test(planner, context, _make_agent(), parsed)

    response_events = [e for e in events if e.event_type == ExecutionEventType.RESPONSE]
    assert len(response_events) == 1

    resp = response_events[0]
    assert resp.content[0].text == json.dumps({"result": "chart data"})
    assert resp.metadata.get(METADATA_FORWARD_VERBATIM) is True


@pytest.mark.asyncio
async def test_forward_verbatim_string_observation(planner):
    """String observations should be forwarded as-is via str()."""
    tool_event = _make_tool_end_event("evt-456", observation="plain text result")
    context = _make_context(interaction_history=[tool_event])
    parsed = _forward_verbatim_action("evt-456")

    events = await _run_forward_test(planner, context, _make_agent(), parsed)

    resp = next(e for e in events if e.event_type == ExecutionEventType.RESPONSE)
    assert resp.content[0].text == "plain text result"
    assert resp.metadata.get(METADATA_FORWARD_VERBATIM) is True


@pytest.mark.asyncio
async def test_forward_verbatim_not_found_yields_error_occurred(planner):
    """When source_id references a non-existent event, an ERROR_OCCURRED is
    yielded (planner-level failure) and the loop terminates — symmetric with
    the tool-use strategies' break-on-miss semantics."""
    context = _make_context(interaction_history=[])
    parsed = _forward_verbatim_action("evt-nonexistent")

    events = await _run_forward_test(planner, context, _make_agent(stream=False), parsed)

    response_events = [e for e in events if e.event_type == ExecutionEventType.RESPONSE]
    assert len(response_events) == 0
    tool_errors = [e for e in events if e.event_type == ExecutionEventType.TOOL_ERROR]
    assert len(tool_errors) == 0
    errors = [e for e in events if e.event_type == ExecutionEventType.ERROR_OCCURRED]
    assert len(errors) == 1
    assert errors[0].error_message == AGENT_ERROR_MESSAGE
    assert errors[0].error_code == AGENT_ERROR_CODE


@pytest.mark.asyncio
async def test_forward_verbatim_disabled_yields_error_occurred(planner):
    """When the planner disables FORWARD_VERBATIM, a parsed action is not forwarded."""
    tool_event = _make_tool_end_event("evt-disabled", observation="verbatim output")
    context = _make_context(interaction_history=[tool_event])
    parsed = _forward_verbatim_action("evt-disabled")

    planner.forward_verbatim_enabled = False

    events = await _run_forward_test(planner, context, _make_agent(stream=False), parsed)

    assert not any(
        e.event_type == ExecutionEventType.RESPONSE and e.metadata.get(METADATA_FORWARD_VERBATIM)
        for e in events
    )
    errors = [e for e in events if e.event_type == ExecutionEventType.ERROR_OCCURRED]
    assert len(errors) == 1
    assert errors[0].error_message == AGENT_ERROR_MESSAGE
    assert errors[0].error_code == AGENT_ERROR_CODE


@pytest.mark.asyncio
async def test_forward_verbatim_none_observation_yields_error_occurred(planner):
    """When the referenced event has observation=None, an ERROR_OCCURRED is yielded."""
    tool_event = _make_tool_end_event("evt-789", observation=None)
    context = _make_context(interaction_history=[tool_event])
    parsed = _forward_verbatim_action("evt-789")

    events = await _run_forward_test(planner, context, _make_agent(stream=False), parsed)

    response_events = [e for e in events if e.event_type == ExecutionEventType.RESPONSE]
    assert len(response_events) == 0
    errors = [e for e in events if e.event_type == ExecutionEventType.ERROR_OCCURRED]
    assert len(errors) == 1
    assert errors[0].error_message == AGENT_ERROR_MESSAGE
    assert errors[0].error_code == AGENT_ERROR_CODE


@pytest.mark.asyncio
async def test_forward_verbatim_empty_string_observation_is_forwarded(planner):
    """Empty string observation should be forwarded (is not None)."""
    tool_event = _make_tool_end_event("evt-empty", observation="")
    context = _make_context(interaction_history=[tool_event])
    parsed = _forward_verbatim_action("evt-empty")

    events = await _run_forward_test(planner, context, _make_agent(), parsed)

    resp = next(e for e in events if e.event_type == ExecutionEventType.RESPONSE)
    assert resp.content[0].text == ""
    assert resp.metadata.get(METADATA_FORWARD_VERBATIM) is True


# ---------------------------------------------------------------------------
# extract_by_dot_path (unit)
# ---------------------------------------------------------------------------

NESTED_OBS = {
    "status": "success",
    "result": {
        "documents": [{"name": "a.pdf"}, {"name": "b.pdf"}],
        "total": 2,
    },
    "metadata": {"request_id": "abc"},
}


class TestExtractByDotPath:
    def test_single_key(self):
        assert extract_by_dot_path(NESTED_OBS, "status") == "success"

    def test_nested_two_levels(self):
        assert extract_by_dot_path(NESTED_OBS, "result.documents") == [
            {"name": "a.pdf"},
            {"name": "b.pdf"},
        ]

    def test_nested_three_levels(self):
        assert extract_by_dot_path(NESTED_OBS, "metadata.request_id") == "abc"

    def test_missing_key(self):
        assert extract_by_dot_path(NESTED_OBS, "nonexistent") is None

    def test_missing_nested_key(self):
        assert extract_by_dot_path(NESTED_OBS, "result.missing") is None

    def test_path_through_non_dict(self):
        assert extract_by_dot_path(NESTED_OBS, "status.deeper") is None

    def test_non_dict_input(self):
        assert extract_by_dot_path("just a string", "key") is None

    def test_empty_path_segment(self):
        # "result..documents" has an empty segment
        assert extract_by_dot_path(NESTED_OBS, "result..documents") is None

    def test_list_index_at_root(self):
        """List-rooted observations (common for search-style MCP tools)
        navigate via numeric segments."""
        obs = [
            {"title": "Paper A", "summary": "abstract A"},
            {"title": "Paper B", "summary": "abstract B"},
        ]
        assert extract_by_dot_path(obs, "0.summary") == "abstract A"
        assert extract_by_dot_path(obs, "1.title") == "Paper B"

    def test_list_index_nested(self):
        """Numeric segments resolve list indices anywhere along the path."""
        obs = {"papers": [{"summary": "alpha"}, {"summary": "beta"}]}
        assert extract_by_dot_path(obs, "papers.0.summary") == "alpha"
        assert extract_by_dot_path(obs, "papers.1.summary") == "beta"

    def test_list_index_out_of_range(self):
        obs = [{"a": 1}]
        assert extract_by_dot_path(obs, "5.a") is None
        assert extract_by_dot_path(obs, "-1.a") is None

    def test_list_index_non_numeric(self):
        """Non-numeric segments against a list resolve to None."""
        obs = [{"a": 1}]
        assert extract_by_dot_path(obs, "first.a") is None

    def test_json_string_value_is_decoded(self):
        """When a value along the path is a JSON string, it should be
        transparently decoded so navigation can continue."""
        obs = {"result": json.dumps({"summary": "## ESG Score", "total": 42})}
        assert extract_by_dot_path(obs, "result.summary") == "## ESG Score"
        assert extract_by_dot_path(obs, "result.total") == 42

    def test_json_string_top_level(self):
        """A top-level JSON string should be decoded before applying the path."""
        data = json.dumps({"a": {"b": "deep"}})
        assert extract_by_dot_path(data, "a.b") == "deep"

    def test_non_json_string_returns_none(self):
        """A non-JSON string that can't be decoded returns None."""
        obs = {"result": "plain text, not json"}
        assert extract_by_dot_path(obs, "result.summary") is None

    def test_json_string_nested_two_levels(self):
        """JSON string wrapping can happen at any depth."""
        obs = {"outer": {"inner": json.dumps({"key": "value"})}}
        assert extract_by_dot_path(obs, "outer.inner.key") == "value"


# ---------------------------------------------------------------------------
# source_key integration with next_actions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_source_key_extracts_nested_value(planner):
    """source_key should extract a nested value from the observation."""
    obs = {"status": "success", "result": {"documents": ["a.pdf", "b.pdf"]}}
    tool_event = _make_tool_end_event("evt-key", observation=obs)
    context = _make_context(interaction_history=[tool_event])
    parsed = _forward_verbatim_action("evt-key", source_key="result.documents")

    events = await _run_forward_test(planner, context, _make_agent(), parsed)

    resp = next(e for e in events if e.event_type == ExecutionEventType.RESPONSE)
    assert resp.content[0].text == json.dumps(["a.pdf", "b.pdf"])
    assert resp.metadata.get(METADATA_FORWARD_VERBATIM) is True


@pytest.mark.asyncio
async def test_source_key_top_level_string(planner):
    """source_key on a top-level string value."""
    obs = {"result": "plain text answer", "status": "ok"}
    tool_event = _make_tool_end_event("evt-str", observation=obs)
    context = _make_context(interaction_history=[tool_event])
    parsed = _forward_verbatim_action("evt-str", source_key="result")

    events = await _run_forward_test(planner, context, _make_agent(), parsed)

    resp = next(e for e in events if e.event_type == ExecutionEventType.RESPONSE)
    assert resp.content[0].text == "plain text answer"
    assert resp.metadata.get(METADATA_FORWARD_VERBATIM) is True


@pytest.mark.asyncio
async def test_source_key_missing_yields_error_occurred(planner):
    """If source_key points to a missing path, an ERROR_OCCURRED is yielded
    (no RESPONSE) and the planner terminates."""
    obs = {"result": "data"}
    tool_event = _make_tool_end_event("evt-miss", observation=obs)
    context = _make_context(interaction_history=[tool_event])
    parsed = _forward_verbatim_action("evt-miss", source_key="nonexistent.path")

    events = await _run_forward_test(planner, context, _make_agent(stream=False), parsed)

    response_events = [e for e in events if e.event_type == ExecutionEventType.RESPONSE]
    assert len(response_events) == 0
    errors = [e for e in events if e.event_type == ExecutionEventType.ERROR_OCCURRED]
    assert len(errors) == 1
    assert errors[0].error_message == AGENT_ERROR_MESSAGE
    assert errors[0].error_code == AGENT_ERROR_CODE


@pytest.mark.asyncio
async def test_source_key_on_non_json_string_yields_error_occurred(planner):
    """If observation is a non-JSON string and source_key is set,
    extract_by_dot_path fails to decode → an ERROR_OCCURRED is yielded."""
    tool_event = _make_tool_end_event("evt-flat", observation="just text")
    context = _make_context(interaction_history=[tool_event])
    parsed = _forward_verbatim_action("evt-flat", source_key="result")

    events = await _run_forward_test(planner, context, _make_agent(stream=False), parsed)

    response_events = [e for e in events if e.event_type == ExecutionEventType.RESPONSE]
    assert len(response_events) == 0
    errors = [e for e in events if e.event_type == ExecutionEventType.ERROR_OCCURRED]
    assert len(errors) == 1


@pytest.mark.asyncio
async def test_source_key_on_json_string_wrapper_yields_error_occurred(planner):
    """When obs is a JSON string wrapper and the key doesn't exist at the top
    level of the decoded JSON, an ERROR_OCCURRED is yielded.
    Reproduces the AlphaVantage/earnings transcript bug."""
    wrapper = json.dumps(
        {
            "preview": True,
            "data_type": "csv",
            "sample_data": json.dumps({"transcript": [{"speaker": "A", "content": "..."}]}),
        }
    )
    tool_event = _make_tool_end_event("evt-wrap", observation=wrapper)
    context = _make_context(interaction_history=[tool_event])
    parsed = _forward_verbatim_action("evt-wrap", source_key="transcript")

    events = await _run_forward_test(planner, context, _make_agent(stream=False), parsed)

    response_events = [e for e in events if e.event_type == ExecutionEventType.RESPONSE]
    assert len(response_events) == 0
    errors = [e for e in events if e.event_type == ExecutionEventType.ERROR_OCCURRED]
    assert len(errors) == 1


@pytest.mark.asyncio
async def test_source_key_on_json_string_obs_extracts_top_level(planner):
    """When obs is a JSON string and the key exists at the top level of the
    decoded JSON, extraction succeeds and the value is forwarded."""
    obs_str = json.dumps({"summary": "## ESG Results", "count": 42})
    tool_event = _make_tool_end_event("evt-json-top", observation=obs_str)
    context = _make_context(interaction_history=[tool_event])
    parsed = _forward_verbatim_action("evt-json-top", source_key="summary")

    events = await _run_forward_test(planner, context, _make_agent(), parsed)

    resp = next(e for e in events if e.event_type == ExecutionEventType.RESPONSE)
    assert resp.content[0].text == "## ESG Results"
    assert resp.metadata.get(METADATA_FORWARD_VERBATIM) is True


@pytest.mark.asyncio
async def test_source_key_on_logical_mcp_json_result(planner):
    """After MCP transport unwrapping, source_key is relative to the logical result."""
    logical_result = json.dumps(
        {
            "case_id": "THINK-9",
            "q4-ledger": {
                "sec-7": {
                    "final answer": {
                        "slot_02": "### Sales pulse\n\n`FV-CANARY:THINK-9:nested-markdown`"
                    },
                    "nearly right": {"slot_02": "Internal note"},
                }
            },
        }
    )
    tool_event = _make_tool_end_event("evt-logical-mcp", observation=logical_result)
    context = _make_context(interaction_history=[tool_event])
    parsed = _forward_verbatim_action(
        "evt-logical-mcp", source_key="q4-ledger.sec-7.final answer.slot_02"
    )

    events = await _run_forward_test(planner, context, _make_agent(), parsed)

    resp = next(e for e in events if e.event_type == ExecutionEventType.RESPONSE)
    assert resp.content[0].text == "### Sales pulse\n\n`FV-CANARY:THINK-9:nested-markdown`"
    assert resp.metadata.get(METADATA_FORWARD_VERBATIM) is True


@pytest.mark.asyncio
async def test_source_key_through_json_string_value(planner):
    """When an observation wraps nested data as a JSON string, source_key
    should still navigate into it."""
    inner = json.dumps({"summary": "## ESG Score Summary", "results": [1, 2, 3]})
    obs = {"result": inner}
    tool_event = _make_tool_end_event("evt-json-str", observation=obs)
    context = _make_context(interaction_history=[tool_event])
    parsed = _forward_verbatim_action("evt-json-str", source_key="result.summary")

    events = await _run_forward_test(planner, context, _make_agent(), parsed)

    resp = next(e for e in events if e.event_type == ExecutionEventType.RESPONSE)
    assert resp.content[0].text == "## ESG Score Summary"
    assert resp.metadata.get(METADATA_FORWARD_VERBATIM) is True


@pytest.mark.asyncio
async def test_forward_verbatim_without_key_sends_whole_observation(planner):
    """Without source_key, the entire observation dict is forwarded."""
    obs = {"result": "data", "extra": "stuff"}
    tool_event = _make_tool_end_event("evt-all", observation=obs)
    context = _make_context(interaction_history=[tool_event])
    parsed = _forward_verbatim_action("evt-all")

    events = await _run_forward_test(planner, context, _make_agent(), parsed)

    resp = next(e for e in events if e.event_type == ExecutionEventType.RESPONSE)
    assert resp.content[0].text == json.dumps(obs)
    assert resp.metadata.get(METADATA_FORWARD_VERBATIM) is True
