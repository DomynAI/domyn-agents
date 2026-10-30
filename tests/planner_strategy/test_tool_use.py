"""Tests for ToolUsePlannerStrategy and ToolUseContextManager."""

import json
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from domyn.agents.agents import ContextVisibility
from domyn.agents.core import (
    AGENT_ERROR_CODE,
    AGENT_ERROR_MESSAGE,
    METADATA_CONVERTED_TO_AGENT_END,
    METADATA_FORWARD_VERBATIM,
    METADATA_FORWARD_VERBATIM_SOURCE_ID,
    AgentAction,
    BaseEvent,
    ExecutionEventType,
    InteractionContext,
    Part,
    ToolAction,
)
from domyn.agents.llm import LLMMessage, LLMResponse, LLMToolCall, ToolCallFunction
from domyn.agents.planner_strategy import ToolUseWithThoughtPlannerStrategy
from domyn.agents.planner_strategy.context_management.tool_use import (
    _REACT_THOUGHT_INSTRUCTION,
    ToolUseContextManager,
    ToolUseWithThoughtContextManager,
)
from domyn.agents.planner_strategy.tool_use import (
    MAX_TOOL_CALL_ARGS_RETRIES,
    ToolUsePlannerStrategy,
)
from domyn.agents.planner_strategy.utils import (
    reorder_messages_around_tool_calls,
    resolve_verbatim,
)

# --- Fixtures ---


def make_event(event_type, author="test_agent", content_text=None, action=None, event_id=None):
    return BaseEvent(
        event_type=event_type,
        author=author,
        event_id=event_id or str(uuid.uuid4()),
        timestamp=datetime.now(),
        content=[Part(text=content_text)] if content_text else [],
        action=action,
    )


def make_tool_action(name="search", call_id=None, parameters=None, thought=None, observation=None):
    return ToolAction(
        call_id=call_id or str(uuid.uuid4()),
        name=name,
        parameters=parameters or {},
        thought=thought,
        observation=observation,
    )


def make_mock_agent(tools=None, sub_agents=None, instruction="You are helpful."):
    agent = MagicMock()
    agent.name = "test_agent"
    agent.description = "A test agent"
    agent.instruction = instruction
    agent.tools = tools or []
    agent.sub_agents = sub_agents or []
    agent.memory_manager = None

    def get_tool_by_name(name):
        for t in agent.tools:
            if t.name == name:
                return t
        return None

    agent.get_tool_by_name = get_tool_by_name
    agent.stream = False
    return agent


def make_mock_tool(name="search", description="Search the web"):
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.tool_definition = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
    }
    return tool


def make_context(conversation_history=None, interaction_history=None):
    ctx = MagicMock(spec=InteractionContext)
    ctx.conversation_id = "conv_1"
    ctx.user_id = "user_1"
    ctx.interaction_id = "int_1"
    ctx.conversation_history = conversation_history or []
    ctx.interaction_history = interaction_history or []
    ctx.handoff_targets_description = []
    ctx.state = {}
    ctx.run_config = {}
    ctx.agent_input = make_event(ExecutionEventType.USER_INPUT, content_text="Hello")
    return ctx


# --- ToolUseContextManager Tests ---


class TestToolUseContextManager:
    def test_instantiation(self):
        ctx = ToolUseContextManager()
        assert ctx.type == "tool_use"
        assert ctx.max_tool_result_chars == 20_000

    @pytest.mark.asyncio
    async def test_get_context_empty_history(self):
        ctx_mgr = ToolUseContextManager()
        agent = make_mock_agent()
        context = make_context()

        messages = await ctx_mgr.get_context(context=context, agent=agent)

        assert len(messages) == 1
        assert messages[0].role == "system"
        assert "You are helpful" in messages[0].content

    @pytest.mark.asyncio
    async def test_get_context_with_user_input(self):
        ctx_mgr = ToolUseContextManager()
        agent = make_mock_agent()
        user_event = make_event(ExecutionEventType.USER_INPUT, content_text="What is AI?")
        context = make_context(conversation_history=[user_event])

        messages = await ctx_mgr.get_context(context=context, agent=agent)

        assert len(messages) == 2
        assert messages[1].role == "user"
        assert messages[1].content == "USER INPUT: What is AI?"

    @pytest.mark.asyncio
    async def test_get_context_with_tool_roundtrip(self):
        ctx_mgr = ToolUseContextManager()
        agent = make_mock_agent()

        call_id = "call_123"
        tool_action_start = make_tool_action(
            name="search", call_id=call_id, parameters={"query": "AI"}, thought="Let me search"
        )
        tool_action_end = make_tool_action(
            name="search",
            call_id=call_id,
            parameters={"query": "AI"},
            observation={"result": "AI is..."},
        )

        events = [
            make_event(ExecutionEventType.USER_INPUT, content_text="What is AI?"),
            make_event(ExecutionEventType.TOOL_START, action=tool_action_start),
            make_event(ExecutionEventType.TOOL_END, action=tool_action_end),
        ]
        context = make_context(conversation_history=events)

        messages = await ctx_mgr.get_context(context=context, agent=agent)

        # system + user + assistant(tool_call) + tool(result)
        assert len(messages) == 4
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert messages[2].role == "assistant"
        assert messages[2].tool_calls is not None
        assert len(messages[2].tool_calls) == 1
        assert messages[2].tool_calls[0].id == call_id
        assert messages[2].tool_calls[0].function.name == "search"
        assert messages[3].role == "tool"
        assert messages[3].tool_call_id == call_id
        assert "AI is..." in messages[3].content

    @pytest.mark.asyncio
    async def test_event_id_header_injected_on_tool_observations(self):
        """The event_id header is injected on every TOOL_END observation so
        the LLM can reference observations by event_id in a later
        FORWARD_VERBATIM call."""
        agent = make_mock_agent()
        call_id = "call_hdr"
        start = make_tool_action(name="search", call_id=call_id, parameters={"q": "x"})
        end = make_tool_action(
            name="search", call_id=call_id, parameters={"q": "x"}, observation={"result": "data"}
        )
        events = [
            make_event(
                ExecutionEventType.TOOL_START,
                action=start,
                event_id="evt_hdr_start",
                author=agent.name,
            ),
            make_event(
                ExecutionEventType.TOOL_END,
                action=end,
                event_id="evt_hdr_end",
                author=agent.name,
            ),
        ]
        context = make_context(conversation_history=events)

        ctx_mgr = ToolUseContextManager()
        messages = await ctx_mgr.get_context(context=context, agent=agent)
        tool_msg = next(m for m in messages if m.role == "tool")
        assert tool_msg.content.startswith("[event_id: ")
        assert "evt_hdr_end" in tool_msg.content

    @pytest.mark.asyncio
    async def test_event_id_header_omitted_when_forward_verbatim_disabled(self):
        """The observation header is omitted when the planner disables FORWARD_VERBATIM."""
        agent = make_mock_agent()
        agent.planner = MagicMock()
        agent.planner.forward_verbatim_enabled = False
        call_id = "call_no_hdr"
        start = make_tool_action(name="search", call_id=call_id, parameters={"q": "x"})
        end = make_tool_action(
            name="search", call_id=call_id, parameters={"q": "x"}, observation={"result": "data"}
        )
        events = [
            make_event(ExecutionEventType.TOOL_START, action=start, event_id="evt_no_hdr_start"),
            make_event(ExecutionEventType.TOOL_END, action=end, event_id="evt_no_hdr_end"),
        ]
        context = make_context(conversation_history=events)

        ctx_mgr = ToolUseContextManager()
        messages = await ctx_mgr.get_context(context=context, agent=agent)
        tool_msg = next(m for m in messages if m.role == "tool")
        assert "data" in tool_msg.content
        assert "evt_no_hdr_end" not in tool_msg.content

    @pytest.mark.asyncio
    async def test_get_context_renders_paired_tool_call(self):
        """A TOOL_START/TOOL_END pair becomes a single assistant(tool_calls)
        message followed by one role=tool message, honoring the native
        tool-use assistant→tool invariant.
        """
        ctx_mgr = ToolUseContextManager()
        agent = make_mock_agent()

        call_id = "call_paired"
        start = make_tool_action(name="search", call_id=call_id, parameters={"q": "x"})
        end = make_tool_action(
            name="search", call_id=call_id, parameters={"q": "x"}, observation="done"
        )
        events = [
            make_event(ExecutionEventType.TOOL_START, action=start),
            make_event(ExecutionEventType.TOOL_END, action=end),
        ]
        context = make_context(conversation_history=events)

        messages = await ctx_mgr.get_context(context=context, agent=agent)

        # system + one assistant + one tool
        assistants = [m for m in messages if m.role == "assistant"]
        assert len(assistants) == 1
        assert assistants[0].tool_calls is not None
        assert len(assistants[0].tool_calls) == 1
        assert assistants[0].tool_calls[0].id == call_id

    @pytest.mark.asyncio
    async def test_tool_result_truncation(self):
        ctx_mgr = ToolUseContextManager(max_tool_result_chars=50)
        agent = make_mock_agent()

        call_id = "call_456"
        long_observation = {"result": "x" * 100}
        tool_action = make_tool_action(name="read", call_id=call_id, observation=long_observation)

        events = [
            make_event(ExecutionEventType.TOOL_END, action=tool_action),
        ]
        context = make_context(conversation_history=events)

        messages = await ctx_mgr.get_context(context=context, agent=agent)

        tool_msg = next(m for m in messages if m.role == "tool")
        assert "...(truncated)" in tool_msg.content
        assert "x" * 100 not in tool_msg.content
        # Total content (header + truncated body + marker) must stay within budget
        assert len(tool_msg.content) <= ctx_mgr.max_tool_result_chars + len("\n...(truncated)")

    @pytest.mark.asyncio
    async def test_tool_result_truncation_when_header_exceeds_budget(self, caplog):
        """When max_tool_result_chars is smaller than the event_id header,
        the header is dropped so observation slicing stays within budget
        (instead of body_budget going negative and slicing from the end).
        A warning is logged because dropping the header breaks FORWARD_VERBATIM
        for this observation."""
        import logging

        ctx_mgr = ToolUseContextManager(max_tool_result_chars=10)
        agent = make_mock_agent()

        call_id = "call_789"
        observation = "y" * 200
        tool_action = make_tool_action(name="read", call_id=call_id, observation=observation)

        event = make_event(ExecutionEventType.TOOL_END, action=tool_action, author=agent.name)
        # Ensure event_id is a long-enough string to overflow max_tool_result_chars.
        event.event_id = "abcdef0123456789"
        context = make_context(conversation_history=[event])

        with caplog.at_level(logging.WARNING, logger="domyn.agents"):
            messages = await ctx_mgr.get_context(context=context, agent=agent)

        tool_msg = next(m for m in messages if m.role == "tool")
        # Header dropped, so the event_id should NOT appear in the content.
        assert event.event_id not in tool_msg.content
        # Body must come from the head of the observation, not the tail.
        assert tool_msg.content.startswith("y" * ctx_mgr.max_tool_result_chars)
        assert len(tool_msg.content) <= ctx_mgr.max_tool_result_chars + len("\n...(truncated)")
        # Warning surfaced so a misconfigured budget does not silently break FV.
        assert any(
            "FORWARD_VERBATIM" in r.message and event.event_id in r.message
            for r in caplog.records
            if r.levelno == logging.WARNING
        )

    @pytest.mark.asyncio
    async def test_a_cut_result_keeps_its_end(self):
        """The end is where tools put the last error or the way to the rest: a
        cut keeps it, as well as the start."""
        ctx_mgr = ToolUseContextManager(max_tool_result_chars=2_000)
        agent = make_mock_agent()
        observation = "".join(f"line {i}\n" for i in range(1000)) + "LAST LINE"
        tool_action = make_tool_action(name="read", observation=observation)
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.TOOL_END, action=tool_action)]
        )

        messages = await ctx_mgr.get_context(context=context, agent=agent)

        content = next(m for m in messages if m.role == "tool").content
        assert "line 0\n" in content
        assert content.endswith("LAST LINE")
        assert len(content) <= 2_000

    def test_get_tool_definitions(self):
        ctx_mgr = ToolUseContextManager()
        tool = make_mock_tool("search", "Search the web")
        agent = make_mock_agent(tools=[tool])
        agent.planner = MagicMock()
        agent.planner.use_stop = False
        context = make_context()

        defs = ctx_mgr.get_tool_definitions(agent, context)

        tool_names = [d["function"]["name"] for d in defs]
        # use_stop=False → no special tools, only regular tools
        assert "search" in tool_names
        assert ctx_mgr.respond_tool_name not in tool_names
        assert ctx_mgr.stop_tool_name not in tool_names
        assert ctx_mgr.ask_user_tool_name not in tool_names

    def test_get_tool_definitions_with_sub_agents(self):
        ctx_mgr = ToolUseContextManager()
        agent = make_mock_agent()
        agent.planner = MagicMock()
        agent.planner.use_stop = False
        context = make_context()

        agent_desc = MagicMock()
        agent_desc.name = "helper_agent"
        agent_desc.description = "A helper"
        agent_desc.invocation_json_schema = {
            "type": "object",
            "properties": {"task": {"type": "string"}},
        }
        context.handoff_targets_description = [agent_desc]

        defs = ctx_mgr.get_tool_definitions(agent, context)

        tool_names = [d["function"]["name"] for d in defs]
        # use_stop=False → no special tools, only sub-agent tool
        assert "helper_agent" in tool_names
        assert ctx_mgr.respond_tool_name not in tool_names

    def test_get_tool_definitions_does_not_mutate_source_schemas(self):
        """`get_tool_definitions` must leave `tool.tool_definition` and
        `agent_desc.invocation_parameters` untouched, even when called many
        times. The reasoning injection runs every turn; in-place mutation
        of the source would compound across turns and eventually produce
        schemas that OpenAI rejects with `non-unique elements` in `required`.

        This asserts the "no mutation" contract in two ways:

        - **Content** unchanged: the original ``required`` / ``properties``
          carry the same keys after repeated calls.
        - **Identity** preserved on the source container: the
          ``parameters`` dict reachable from ``tool.tool_definition`` /
          ``agent_desc.invocation_parameters`` is still the same Python
          object as before. This catches a subtler bug where the source
          dict is *replaced* by a new dict (e.g.
          ``tool["function"]["parameters"] = copy.deepcopy(...)`` inside
          ``get_tool_definitions``): the old reference would stay
          untouched and the content assertion would pass, but the source
          schema would in fact have been mutated at the container level.
        """
        ctx_mgr = ToolUseContextManager()

        tool = make_mock_tool("search", "Search the web")
        source_tool_params = tool.tool_definition["function"]["parameters"]
        original_tool_required = list(source_tool_params["required"])
        original_tool_properties = set(source_tool_params["properties"].keys())

        agent_desc = MagicMock()
        agent_desc.name = "helper_agent"
        agent_desc.description = "A helper"
        agent_desc.invocation_parameters = {
            "type": "object",
            "properties": {"task": {"type": "string"}},
            "required": ["task"],
        }
        source_sub_params = agent_desc.invocation_parameters
        original_sub_required = list(source_sub_params["required"])
        original_sub_properties = set(source_sub_params["properties"].keys())

        agent = make_mock_agent(tools=[tool])
        agent.planner = MagicMock()
        agent.planner.use_stop = False
        context = make_context()
        context.handoff_targets_description = [agent_desc]

        for _ in range(3):
            defs = ctx_mgr.get_tool_definitions(agent, context)
            for d in defs:
                required = d["function"]["parameters"].get("required", [])
                assert required.count("reasoning") == 1, (
                    f"`reasoning` duplicated in tool `{d['function']['name']}`: {required}"
                )

        # content invariant
        assert source_tool_params["required"] == original_tool_required
        assert set(source_tool_params["properties"].keys()) == original_tool_properties
        assert source_sub_params["required"] == original_sub_required
        assert set(source_sub_params["properties"].keys()) == original_sub_properties

        # identity invariant on the source container — the source dict slot
        # must still point to the same Python object we captured before any
        # ``get_tool_definitions`` call. If this fails, the source schema
        # was replaced even though its old content stayed intact.
        assert tool.tool_definition["function"]["parameters"] is source_tool_params, (
            "`get_tool_definitions` replaced the `parameters` dict on "
            "`tool.tool_definition` instead of operating on a copy."
        )
        assert agent_desc.invocation_parameters is source_sub_params, (
            "`get_tool_definitions` replaced `agent_desc.invocation_parameters` "
            "instead of operating on a copy."
        )

    def test_forward_verbatim_registered_on_top_level(self):
        """use_stop=False (top-level agent): FORWARD_VERBATIM is registered by default."""
        ctx_mgr = ToolUseContextManager()
        agent = make_mock_agent(tools=[make_mock_tool("search")])
        agent.planner = MagicMock()
        agent.planner.use_stop = False
        context = make_context()

        defs = ctx_mgr.get_tool_definitions(agent, context)
        tool_names = [d["function"]["name"] for d in defs]
        assert ctx_mgr.forward_verbatim_tool_name in tool_names

    def test_forward_verbatim_registered_alongside_stop(self):
        """use_stop=True (sub-agent): FORWARD_VERBATIM coexists with STOP/ASK_USER by default."""
        ctx_mgr = ToolUseContextManager()
        agent = make_mock_agent(tools=[make_mock_tool("search")])
        agent.planner = MagicMock()
        agent.planner.use_stop = True
        context = make_context()

        defs = ctx_mgr.get_tool_definitions(agent, context)
        tool_names = [d["function"]["name"] for d in defs]
        assert ctx_mgr.forward_verbatim_tool_name in tool_names
        assert ctx_mgr.stop_tool_name in tool_names
        assert ctx_mgr.ask_user_tool_name in tool_names

    def test_forward_verbatim_not_registered_when_disabled(self):
        """A planner can hide FORWARD_VERBATIM without affecting normal tools."""
        ctx_mgr = ToolUseContextManager()
        agent = make_mock_agent(tools=[make_mock_tool("search")])
        agent.planner = MagicMock()
        agent.planner.use_stop = False
        agent.planner.forward_verbatim_enabled = False
        context = make_context()

        defs = ctx_mgr.get_tool_definitions(agent, context)
        tool_names = [d["function"]["name"] for d in defs]
        assert "search" in tool_names
        assert ctx_mgr.forward_verbatim_tool_name not in tool_names

    def test_forward_verbatim_disabled_keeps_stop_tools(self):
        """Disabling FORWARD_VERBATIM does not remove STOP/ASK_USER for sub-agents."""
        ctx_mgr = ToolUseContextManager()
        agent = make_mock_agent(tools=[make_mock_tool("search")])
        agent.planner = MagicMock()
        agent.planner.use_stop = True
        agent.planner.forward_verbatim_enabled = False
        context = make_context()

        defs = ctx_mgr.get_tool_definitions(agent, context)
        tool_names = [d["function"]["name"] for d in defs]
        assert ctx_mgr.forward_verbatim_tool_name not in tool_names
        assert ctx_mgr.stop_tool_name in tool_names
        assert ctx_mgr.ask_user_tool_name in tool_names

    def test_forward_verbatim_inherited_by_thought_manager(self):
        """ToolUseWithThoughtContextManager registers FORWARD_VERBATIM like the parent."""
        ctx_mgr = ToolUseWithThoughtContextManager()
        agent = make_mock_agent(tools=[make_mock_tool("search")])
        agent.planner = MagicMock()
        agent.planner.use_stop = False
        context = make_context()

        tool_names = [d["function"]["name"] for d in ctx_mgr.get_tool_definitions(agent, context)]
        assert ctx_mgr.forward_verbatim_tool_name in tool_names

    def test_forward_verbatim_disable_inherited_by_thought_manager(self):
        """ToolUseWithThoughtContextManager honors the same planner-level FV toggle."""
        ctx_mgr = ToolUseWithThoughtContextManager()
        agent = make_mock_agent(tools=[make_mock_tool("search")])
        agent.planner = MagicMock()
        agent.planner.use_stop = False
        agent.planner.forward_verbatim_enabled = False
        context = make_context()

        tool_names = [d["function"]["name"] for d in ctx_mgr.get_tool_definitions(agent, context)]
        assert ctx_mgr.forward_verbatim_tool_name not in tool_names

    @pytest.mark.asyncio
    async def test_deduplicates_events(self):
        ctx_mgr = ToolUseContextManager()
        agent = make_mock_agent()
        event = make_event(ExecutionEventType.USER_INPUT, content_text="Hello", event_id="dup_1")
        # Same event in both conversation and interaction history
        context = make_context(conversation_history=[event], interaction_history=[event])

        messages = await ctx_mgr.get_context(context=context, agent=agent)

        user_msgs = [m for m in messages if m.role == "user"]
        assert len(user_msgs) == 1

    @pytest.mark.asyncio
    async def test_get_context_reorders_injected_response_wedged_in_tool_call(self):
        """A partial RESPONSE injected mid-turn (e.g. via ``Runner.push_parts``)
        can land, in event order, between TOOL_START and its TOOL_END. Native
        tool-calling providers (OpenAI) reject a request where the assistant's
        tool_calls message isn't immediately followed by the tool response, so
        ``get_context`` must move it to right after the tool observation."""
        ctx_mgr = ToolUseContextManager()
        agent = make_mock_agent()

        call_id = "call_injected"
        tool_action_start = make_tool_action(
            name="search", call_id=call_id, parameters={"query": "AI"}
        )
        tool_action_end = make_tool_action(
            name="search",
            call_id=call_id,
            parameters={"query": "AI"},
            observation={"result": "AI is..."},
        )

        events = [
            make_event(ExecutionEventType.USER_INPUT, content_text="What is AI?"),
            make_event(ExecutionEventType.TOOL_START, action=tool_action_start),
            make_event(
                ExecutionEventType.RESPONSE,
                content_text="progress update",
                author=agent.name,  # push_parts defaults to the entrypoint, i.e. this agent
            ),
            make_event(ExecutionEventType.TOOL_END, action=tool_action_end),
            make_event(ExecutionEventType.RESPONSE, content_text="final answer"),
        ]
        context = make_context(conversation_history=events)

        messages = await ctx_mgr.get_context(context=context, agent=agent)

        # system + user + assistant(tool_call) + tool(result) + assistant(injected) + assistant(final)
        roles = [m.role for m in messages]
        assert roles == ["system", "user", "assistant", "tool", "assistant", "assistant"]
        assert messages[2].tool_calls is not None
        assert messages[3].tool_call_id == call_id
        assert messages[4].content == "progress update"
        assert messages[5].content == "final answer"


class TestReorderMessagesAroundToolCalls:
    """Unit tests for the standalone ``reorder_messages_around_tool_calls`` helper."""

    def test_moves_single_wedged_message_after_tool_response(self):
        tool_call = LLMToolCall(
            id="call_1", type="function", function=ToolCallFunction(name="search", arguments="{}")
        )
        messages = [
            LLMMessage(role="assistant", tool_calls=[tool_call]),
            LLMMessage(role="assistant", content="injected"),
            LLMMessage(role="tool", tool_call_id="call_1", name="search", content="observation"),
            LLMMessage(role="assistant", content="final"),
        ]

        result = reorder_messages_around_tool_calls(messages)

        assert [m.content for m in result] == [None, "observation", "injected", "final"]
        assert result[0].tool_calls[0].id == "call_1"
        assert result[1].role == "tool"

    def test_leaves_already_correct_order_unchanged(self):
        tool_call = LLMToolCall(
            id="call_1", type="function", function=ToolCallFunction(name="search", arguments="{}")
        )
        messages = [
            LLMMessage(role="system", content="sys"),
            LLMMessage(role="assistant", tool_calls=[tool_call]),
            LLMMessage(role="tool", tool_call_id="call_1", name="search", content="observation"),
            LLMMessage(role="assistant", content="final"),
        ]

        result = reorder_messages_around_tool_calls(messages)

        assert result == messages

    def test_defers_multiple_wedged_messages_for_parallel_tool_calls(self):
        tool_call_1 = LLMToolCall(
            id="call_1", type="function", function=ToolCallFunction(name="a", arguments="{}")
        )
        tool_call_2 = LLMToolCall(
            id="call_2", type="function", function=ToolCallFunction(name="b", arguments="{}")
        )
        messages = [
            LLMMessage(role="assistant", tool_calls=[tool_call_1, tool_call_2]),
            LLMMessage(role="assistant", content="injected_1"),
            LLMMessage(role="tool", tool_call_id="call_1", name="a", content="obs_a"),
            LLMMessage(role="assistant", content="injected_2"),
            LLMMessage(role="tool", tool_call_id="call_2", name="b", content="obs_b"),
        ]

        result = reorder_messages_around_tool_calls(messages)

        assert [m.tool_call_id or m.content for m in result[1:]] == [
            "call_1",
            "call_2",
            "injected_1",
            "injected_2",
        ]

    def test_leaves_unresolved_tool_call_untouched(self):
        """When the matching tool response is not yet in the list (the tool
        call is genuinely still in flight), there is nothing to move the
        intervening message after, so it stays where it is."""
        tool_call = LLMToolCall(
            id="call_1", type="function", function=ToolCallFunction(name="search", arguments="{}")
        )
        messages = [
            LLMMessage(role="assistant", tool_calls=[tool_call]),
            LLMMessage(role="assistant", content="injected"),
        ]

        result = reorder_messages_around_tool_calls(messages)

        assert result == messages


class TestSubAgentEndHeaderExposure:
    """The event_id header on a sub-agent's AGENT_END message is the only
    handle the parent has to call FORWARD_VERBATIM(source_id=...) on that
    sub's output. Exposure requires the parent to have the FV tool registered
    and the event to already be in that parent's visible context. The sub's
    own FV flag only controls whether it can call FORWARD_VERBATIM in its own
    planner; once the parent can see an event, the parent owns the decision to
    forward that visible result or adapt it.
    """

    @staticmethod
    def _build(parent_fv: bool, sub_fv: bool, attach_sub: bool = True):
        """Build a parent agent + AGENT_END context exercising header rules.

        Parameters
        ----------
        parent_fv:
            Value for the *parent* planner's ``forward_verbatim_enabled``.
        sub_fv:
            Value for the *sub*-agent planner's ``forward_verbatim_enabled``.
            This does not gate AGENT_END header exposure.
        attach_sub:
            When False, the emitting agent is NOT registered in
            ``parent.sub_agents`` (simulates deep cascade where a deeper
            descendant's AGENT_END is visible via GLOBAL/PEERS). Since this
            event is already in the parent's visible context, the header is
            still exposed when the parent FV flag is on.
        """
        parent_agent = make_mock_agent()
        parent_agent.planner = MagicMock()
        parent_agent.planner.forward_verbatim_enabled = parent_fv

        sub_agent = MagicMock()
        sub_agent.name = "SubAgent"
        sub_agent.planner = MagicMock()
        sub_agent.planner.forward_verbatim_enabled = sub_fv
        if attach_sub:
            parent_agent.sub_agents = [sub_agent]
        else:
            parent_agent.sub_agents = []

        sub_end = make_event(
            ExecutionEventType.AGENT_END,
            author="SubAgent",
            event_id="evt_sub_end_1",
            content_text="sub output text",
            action=ToolAction(name="STOP", parameters=None),
        )
        context = make_context(conversation_history=[sub_end])
        return parent_agent, context, sub_end

    @pytest.mark.asyncio
    async def test_header_shown_when_both_flags_on(self):
        parent_agent, context, sub_end = self._build(parent_fv=True, sub_fv=True)
        ctx_mgr = ToolUseContextManager()

        messages = await ctx_mgr.get_context(context=context, agent=parent_agent)

        sub_end_msg = next(m for m in messages if m.role == "tool" and "Agent:" in m.content)
        assert f"[event_id: {sub_end.event_id}]" in sub_end_msg.content
        # Information body still present so the parent LLM can reason on the
        # sub's output even when it chooses not to forward verbatim.
        assert "sub output text" in sub_end_msg.content

    @pytest.mark.asyncio
    async def test_header_shown_when_sub_flag_off(self):
        # The sub-agent FV flag controls whether the sub can use FV internally,
        # not whether the parent can forward the sub's already-returned output.
        parent_agent, context, sub_end = self._build(parent_fv=True, sub_fv=False)
        ctx_mgr = ToolUseContextManager()

        messages = await ctx_mgr.get_context(context=context, agent=parent_agent)

        sub_end_msg = next(m for m in messages if m.role == "tool" and "Agent:" in m.content)
        assert sub_end_msg.content.startswith("Agent: SubAgent ended ")
        assert f"[event_id: {sub_end.event_id}]" in sub_end_msg.content
        assert "sub output text" in sub_end_msg.content

    @pytest.mark.asyncio
    async def test_no_header_when_parent_flag_off(self):
        parent_agent, context, sub_end = self._build(parent_fv=False, sub_fv=True)
        ctx_mgr = ToolUseContextManager()

        messages = await ctx_mgr.get_context(context=context, agent=parent_agent)

        sub_end_msg = next(m for m in messages if m.role == "tool" and "SubAgent" in m.content)
        assert sub_end_msg.content.startswith("Agent: SubAgent ended.")
        assert "[event_id:" not in sub_end_msg.content
        assert sub_end.event_id not in sub_end_msg.content
        assert "sub output text" in sub_end_msg.content

    @pytest.mark.asyncio
    async def test_header_shown_when_sub_not_in_parent_sub_agents(self):
        # Deep cascade simulation: the AGENT_END is from an agent visible via
        # GLOBAL/PEERS handoff but is not a direct child of the current parent.
        # The event is already in the parent's context, so the parent FV flag
        # alone controls whether it can be used as a source.
        parent_agent, context, sub_end = self._build(parent_fv=True, sub_fv=True, attach_sub=False)
        ctx_mgr = ToolUseContextManager()

        messages = await ctx_mgr.get_context(context=context, agent=parent_agent)

        sub_end_msg = next(m for m in messages if m.role == "tool" and "SubAgent" in m.content)
        assert sub_end_msg.content.startswith("Agent: SubAgent ended ")
        assert f"[event_id: {sub_end.event_id}]" in sub_end_msg.content
        assert "sub output text" in sub_end_msg.content

    @pytest.mark.asyncio
    async def test_depth_three_global_grandchild_header_exposed(self):
        """Depth-3 hierarchy: parent → child → grandchild with GLOBAL handoff
        visibility. Both child's and grandchild's AGENT_END events appear in
        the parent's ``conversation_history`` (visibility allows it). Since
        both events are already visible, both event_ids are surfaced when the
        parent FV flag is on.
        """
        parent_agent = make_mock_agent()
        parent_agent.planner = MagicMock()
        parent_agent.planner.forward_verbatim_enabled = True

        child_agent = MagicMock()
        child_agent.name = "ChildAgent"
        child_agent.planner = MagicMock()
        child_agent.planner.forward_verbatim_enabled = True
        parent_agent.sub_agents = [child_agent]

        # Grandchild not in parent.sub_agents — visible via GLOBAL only.
        # Since it is visible, the parent's FV flag still exposes its id.
        grandchild_end = make_event(
            ExecutionEventType.AGENT_END,
            author="GrandchildAgent",
            event_id="evt_grandchild_end",
            content_text="grandchild prose",
            action=ToolAction(name="STOP", parameters=None),
        )
        child_end = make_event(
            ExecutionEventType.AGENT_END,
            author="ChildAgent",
            event_id="evt_child_end",
            content_text="child prose",
            action=ToolAction(name="STOP", parameters=None),
        )
        context = make_context(conversation_history=[grandchild_end, child_end])

        ctx_mgr = ToolUseContextManager()
        messages = await ctx_mgr.get_context(context=context, agent=parent_agent)

        tool_msgs = [m for m in messages if m.role == "tool"]
        child_msg = next(m for m in tool_msgs if "ChildAgent" in m.content)
        grandchild_msg = next(m for m in tool_msgs if "GrandchildAgent" in m.content)

        assert child_msg.content.startswith("Agent: ChildAgent ended ")
        assert f"[event_id: {child_end.event_id}]" in child_msg.content
        assert "child prose" in child_msg.content

        assert grandchild_msg.content.startswith("Agent: GrandchildAgent ended ")
        assert f"[event_id: {grandchild_end.event_id}]" in grandchild_msg.content
        assert "grandchild prose" in grandchild_msg.content

    @pytest.mark.asyncio
    async def test_header_independent_of_sub_metadata_forward_verbatim(self):
        """Whether the sub itself flagged its own AGENT_END with
        METADATA_FORWARD_VERBATIM (i.e., the sub called FORWARD_VERBATIM in
        its own scope) is irrelevant to header exposure. The parent planner flag
        and visible context drive exposure — an explicit per-event hint from the
        sub does NOT gate the parent's view.
        """
        parent_agent, context, sub_end = self._build(parent_fv=True, sub_fv=True)
        # No metadata set on sub_end → header still must appear because
        # the eligibility decision is taken at flag-level, not event-level.
        assert sub_end.metadata.get(METADATA_FORWARD_VERBATIM) is None

        ctx_mgr = ToolUseContextManager()
        messages = await ctx_mgr.get_context(context=context, agent=parent_agent)

        sub_end_msg = next(m for m in messages if m.role == "tool" and "Agent:" in m.content)
        assert f"[event_id: {sub_end.event_id}]" in sub_end_msg.content


class TestToolEndHeaderExposure:
    """The event_id header on a TOOL_END/TOOL_ERROR rendered to the parent
    is controlled by the current parent planner's FV flag. If the event has
    already reached the parent's visible context, the sub-agent's FV flag does
    not gate whether the parent can forward it.
    """

    @staticmethod
    def _build(parent_fv: bool, emitter_fv: bool, attach_emitter: bool = True):
        parent_agent = make_mock_agent()
        parent_agent.name = "Parent"
        parent_agent.planner = MagicMock()
        parent_agent.planner.forward_verbatim_enabled = parent_fv

        worker = MagicMock()
        worker.name = "Worker"
        worker.planner = MagicMock()
        worker.planner.forward_verbatim_enabled = emitter_fv
        parent_agent.sub_agents = [worker] if attach_emitter else []

        tool_end = make_event(
            ExecutionEventType.TOOL_END,
            author="Worker",
            event_id="evt_worker_tool_end",
            action=make_tool_action(
                name="sales_brief",
                observation={"case_id": "X", "result": {"summary": "...prose..."}},
            ),
        )
        context = make_context(conversation_history=[tool_end])
        return parent_agent, context, tool_end

    @pytest.mark.asyncio
    async def test_header_shown_when_emitter_fv_on(self):
        parent, context, tool_end = self._build(parent_fv=True, emitter_fv=True)
        ctx_mgr = ToolUseContextManager()

        messages = await ctx_mgr.get_context(context=context, agent=parent)

        tool_msg = next(m for m in messages if m.role == "user" and "called tool" in m.content)
        assert f"[event_id: {tool_end.event_id}]" in tool_msg.content
        assert tool_msg.content.startswith("Agent: Worker called tool sales_brief")

    @pytest.mark.asyncio
    async def test_header_shown_when_emitter_fv_off(self):
        # Worker has FV off, parent has FV on. The worker cannot call FV in its
        # own planner, but the parent may still forward what reached its
        # visible context.
        parent, context, tool_end = self._build(parent_fv=True, emitter_fv=False)
        ctx_mgr = ToolUseContextManager()

        messages = await ctx_mgr.get_context(context=context, agent=parent)

        tool_msg = next(m for m in messages if m.role == "user" and "called tool" in m.content)
        assert f"[event_id: {tool_end.event_id}]" in tool_msg.content
        assert "case_id" in tool_msg.content

    @pytest.mark.asyncio
    async def test_header_hidden_when_parent_fv_off(self):
        parent, context, _tool_end = self._build(parent_fv=False, emitter_fv=True)
        ctx_mgr = ToolUseContextManager()

        messages = await ctx_mgr.get_context(context=context, agent=parent)

        tool_msg = next(m for m in messages if m.role == "user" and "called tool" in m.content)
        assert "[event_id:" not in tool_msg.content

    @pytest.mark.asyncio
    async def test_header_shown_when_emitter_not_in_direct_sub_agents(self):
        # Emitter visible via GLOBAL/PEERS handoff but not registered in
        # parent.sub_agents. Since the event reached the parent's context, the
        # parent FV flag is enough to expose its source id.
        parent, context, tool_end = self._build(
            parent_fv=True, emitter_fv=True, attach_emitter=False
        )
        ctx_mgr = ToolUseContextManager()

        messages = await ctx_mgr.get_context(context=context, agent=parent)

        tool_msg = next(m for m in messages if m.role == "user" and "called tool" in m.content)
        assert f"[event_id: {tool_end.event_id}]" in tool_msg.content

    @pytest.mark.asyncio
    async def test_header_shown_for_self_emitted_tool_end(self):
        # The parent's own TOOL_END (author == agent.name) must keep the
        # header even when no sub_agents are configured: the agent itself
        # is its own emitter and is always FV-eligible if its planner says so.
        parent_agent = make_mock_agent()
        parent_agent.name = "Parent"
        parent_agent.planner = MagicMock()
        parent_agent.planner.forward_verbatim_enabled = True
        parent_agent.sub_agents = []

        tool_end = make_event(
            ExecutionEventType.TOOL_END,
            author="Parent",
            event_id="evt_self_tool_end",
            action=make_tool_action(name="search", observation="prose result"),
        )
        context = make_context(conversation_history=[tool_end])

        ctx_mgr = ToolUseContextManager()
        messages = await ctx_mgr.get_context(context=context, agent=parent_agent)

        tool_msg = next(m for m in messages if m.role == "tool")
        assert f"[event_id: {tool_end.event_id}]" in tool_msg.content


class TestPerspectiveAwareRendering:
    """Events authored by a different agent than the consumer ("foreign"
    events) must not be rendered as the consumer's own actionable history.
    Rendering a foreign tool/handoff call as an assistant ``tool_call`` makes
    the consumer LLM believe it can call tools/agents it does not have
    (the ToolNotFound carousel); foreign results are rendered as attributed
    narration instead.
    """

    @staticmethod
    def _consumer(name: str = "Parent", use_stop: bool = False, fv: bool = True):
        agent = make_mock_agent()
        agent.name = name
        agent.planner = MagicMock()
        agent.planner.use_stop = use_stop
        agent.planner.forward_verbatim_enabled = fv
        return agent

    @pytest.mark.asyncio
    async def test_foreign_tool_start_is_skipped(self):
        consumer = self._consumer()
        tool_start = make_event(
            ExecutionEventType.TOOL_START,
            author="Worker",
            action=make_tool_action(name="search", call_id="c1", parameters={"q": "x"}),
        )
        context = make_context(conversation_history=[tool_start])

        messages = await ToolUseContextManager().get_context(context=context, agent=consumer)

        # No assistant tool_call is produced for a foreign TOOL_START.
        assert not any(m.role == "assistant" and m.tool_calls for m in messages)

    @pytest.mark.asyncio
    async def test_own_tool_start_renders_assistant_tool_call(self):
        consumer = self._consumer()
        tool_start = make_event(
            ExecutionEventType.TOOL_START,
            author="Parent",
            action=make_tool_action(name="search", call_id="c1", parameters={"q": "x"}),
        )
        context = make_context(conversation_history=[tool_start])

        messages = await ToolUseContextManager().get_context(context=context, agent=consumer)

        assistant = next(m for m in messages if m.role == "assistant" and m.tool_calls)
        assert assistant.tool_calls[0].function.name == "search"

    @pytest.mark.asyncio
    async def test_foreign_tool_end_renders_user_narration(self):
        consumer = self._consumer()
        tool_end = make_event(
            ExecutionEventType.TOOL_END,
            author="Worker",
            action=make_tool_action(name="search", parameters={"q": "x"}, observation="RESULT"),
        )
        context = make_context(conversation_history=[tool_end])

        messages = await ToolUseContextManager().get_context(context=context, agent=consumer)

        # Foreign tool result → attributed user-role narration. Never a tool-role
        # message: its TOOL_START was skipped, so a tool-role reply would orphan.
        assert not any(m.role == "tool" for m in messages)
        narration = next(m for m in messages if m.role == "user" and "called tool" in m.content)
        assert narration.content.startswith("Agent: Worker called tool search(")
        assert '"q": "x"' in narration.content
        assert "Result: RESULT" in narration.content

    @pytest.mark.asyncio
    async def test_own_tool_end_renders_tool_role_message(self):
        consumer = self._consumer()
        tool_end = make_event(
            ExecutionEventType.TOOL_END,
            author="Parent",
            action=make_tool_action(name="search", call_id="c1", observation="RESULT"),
        )
        context = make_context(conversation_history=[tool_end])

        messages = await ToolUseContextManager().get_context(context=context, agent=consumer)

        tool_msg = next(m for m in messages if m.role == "tool")
        assert tool_msg.tool_call_id == "c1"
        assert tool_msg.name == "search"
        assert "RESULT" in tool_msg.content

    @pytest.mark.asyncio
    async def test_foreign_tool_error_renders_user_narration(self):
        # TOOL_ERROR shares the TOOL_END case arm, so a foreign tool error is
        # also rendered as attributed user narration.
        consumer = self._consumer()
        tool_error = make_event(
            ExecutionEventType.TOOL_ERROR,
            author="Worker",
            action=make_tool_action(name="search", parameters={"q": "x"}, observation="boom"),
        )
        context = make_context(conversation_history=[tool_error])

        messages = await ToolUseContextManager().get_context(context=context, agent=consumer)

        assert not any(m.role == "tool" for m in messages)
        narration = next(m for m in messages if m.role == "user" and "called tool" in m.content)
        assert narration.content.startswith("Agent: Worker called tool search(")
        assert "Result: boom" in narration.content

    @pytest.mark.asyncio
    async def test_foreign_tool_end_keeps_event_id_header_when_fv_enabled(self):
        consumer = self._consumer(fv=True)
        tool_end = make_event(
            ExecutionEventType.TOOL_END,
            author="Worker",
            event_id="evt_worker_end",
            action=make_tool_action(name="search", observation="RESULT"),
        )
        context = make_context(conversation_history=[tool_end])

        messages = await ToolUseContextManager().get_context(context=context, agent=consumer)

        narration = next(m for m in messages if m.role == "user" and "called tool" in m.content)
        # Header stays INSIDE the narration so the consumer can FORWARD_VERBATIM it.
        assert "[event_id: evt_worker_end]" in narration.content

    @pytest.mark.asyncio
    async def test_foreign_agent_call_is_skipped(self):
        consumer = self._consumer()
        agent_call = make_event(
            ExecutionEventType.AGENT_CALL,
            author="Worker",
            action=make_tool_action(name="Grandchild", call_id="c1", parameters={"task": "t"}),
        )
        context = make_context(conversation_history=[agent_call])

        messages = await ToolUseContextManager().get_context(context=context, agent=consumer)

        # A foreign delegation must not become an assistant tool_call (that would
        # imply the consumer can invoke that agent itself).
        assert not any(m.role == "assistant" and m.tool_calls for m in messages)

    @pytest.mark.asyncio
    async def test_own_agent_call_renders_assistant_tool_call(self):
        consumer = self._consumer()
        agent_call = make_event(
            ExecutionEventType.AGENT_CALL,
            author="Parent",
            action=make_tool_action(name="Child", call_id="c1", parameters={"task": "t"}),
        )
        context = make_context(conversation_history=[agent_call])

        messages = await ToolUseContextManager().get_context(context=context, agent=consumer)

        assistant = next(m for m in messages if m.role == "assistant" and m.tool_calls)
        assert assistant.tool_calls[0].function.name == "Child"

    @pytest.mark.asyncio
    async def test_deep_handoff_foreign_agent_start_is_not_dropped(self):
        # Regression for "subagents not getting task". Consumer "Parent" watches a
        # two-level handoff chain (5 events):
        #   1. AGENT_CALL  Parent -> Child       (own delegation)
        #   2. AGENT_START Child                 (Child receives its task; closes the pair)
        #   3. TOOL_END    Child                 (Child runs a tool)
        #   4. AGENT_CALL  Child -> Grandchild   (foreign delegation, skipped)
        #   5. AGENT_START Grandchild            (foreign task assignment)
        # The deeper AGENT_CALL (4) is foreign and skipped, so Child's tool result
        # (3) renders as a user-role narration immediately before Grandchild's
        # AGENT_START (5). That foreign AGENT_START must still be appended or the
        # grandchild's task silently disappears from Parent's view.
        consumer = self._consumer()  # name="Parent"

        parent_calls_child = make_event(
            ExecutionEventType.AGENT_CALL,
            author="Parent",
            action=make_tool_action(
                name="Child", call_id="cc1", parameters={"task": "investigate"}
            ),
        )
        child_start = make_event(
            ExecutionEventType.AGENT_START,
            author="Child",
            content_text="investigate",
        )
        child_tool_end = make_event(
            ExecutionEventType.TOOL_END,
            author="Child",
            action=make_tool_action(name="search", parameters={"q": "x"}, observation="RESULT"),
        )
        child_calls_grandchild = make_event(
            ExecutionEventType.AGENT_CALL,
            author="Child",
            action=make_tool_action(
                name="Grandchild", call_id="cc2", parameters={"task": "dig deeper"}
            ),
        )
        grandchild_start = make_event(
            ExecutionEventType.AGENT_START,
            author="Grandchild",
            content_text="dig deeper",
        )
        context = make_context(
            conversation_history=[
                parent_calls_child,
                child_start,
                child_tool_end,
                child_calls_grandchild,
                grandchild_start,
            ]
        )

        messages = await ToolUseContextManager().get_context(context=context, agent=consumer)

        # Only Parent's OWN delegation becomes an assistant tool_call; the foreign
        # Child -> Grandchild delegation is skipped (consumer can't invoke
        # Grandchild itself).
        assistant_tool_calls = [
            tc.function.name
            for m in messages
            if m.role == "assistant" and m.tool_calls
            for tc in m.tool_calls
        ]
        assert assistant_tool_calls == ["Child"]

        # Grandchild's task assignment survives as a user-role message.
        task_msg = next(
            m for m in messages if m.role == "user" and m.content.startswith("Agent: Grandchild")
        )
        assert "Task assigned: dig deeper" in task_msg.content

    @pytest.mark.asyncio
    async def test_own_agent_start_after_user_input_is_skipped(self):
        # A start built from a USER_INPUT (same interaction, same text) duplicates it.
        consumer = self._consumer()
        user_input = make_event(ExecutionEventType.USER_INPUT, author="user", content_text="hello")
        agent_start = make_event(
            ExecutionEventType.AGENT_START,
            author="Parent",
            content_text="hello",
        )
        context = make_context(conversation_history=[user_input, agent_start])

        messages = await ToolUseContextManager().get_context(context=context, agent=consumer)

        assert not any("Task assigned:" in (m.content or "") for m in messages)


class TestReCalledSubAgentGetsItsTask:
    """A sub-agent called again in a later turn must see the new task, whatever its
    visibility and ``use_stop``; the root must see each user request once.

    Histories are filtered the way ``ContextBuilder.create_context`` does: GLOBAL
    sees every event, STANDARD/PEERS (no siblings here) only the sub's own.
    """

    ROOT, SUB = "Support", "OrderAgent"

    @staticmethod
    def _agent(name: str, use_stop: bool):
        agent = make_mock_agent()
        agent.name = name
        agent.planner = MagicMock()
        agent.planner.use_stop = use_stop
        agent.planner.forward_verbatim_enabled = False
        return agent

    @staticmethod
    def _event(event_type, author, interaction_id, text=None, action=None):
        event = make_event(event_type, author=author, content_text=text, action=action)
        event.interaction_id = interaction_id
        return event

    def _turn(self, n: int, order: str, use_stop: bool, verbatim: bool = False) -> dict:
        """One root turn that delegates to the sub-agent, which calls a tool and ends."""
        ev, root, sub = self._event, self.ROOT, self.SUB
        request = f"what about order {order}?"
        task = request if verbatim else f"check order {order}"
        r_id, s_id = f"root_{n}", f"sub_{n}"
        call = make_tool_action(name=sub, call_id=f"ac{n}", parameters={"task": task})
        tool = make_tool_action(
            name="get_order_status", call_id=f"tc{n}", parameters={"order_id": order}
        )
        done = make_tool_action(
            name="get_order_status", call_id=f"tc{n}", observation=f"{order}: shipped"
        )
        sub_answer = f"Order {order} is shipped"

        def start(name: str, text: str) -> AgentAction:
            return AgentAction(name=name, parameters={"task": text})

        end_action = (
            make_tool_action(name="STOP", parameters={"info": sub_answer}) if use_stop else None
        )
        converted = []
        if not use_stop:
            response = ev(ExecutionEventType.RESPONSE, sub, s_id, sub_answer)
            response.metadata[METADATA_CONVERTED_TO_AGENT_END] = True
            converted = [response]
        return {
            "user_input": ev(ExecutionEventType.USER_INPUT, "user", r_id, request),
            "root_start": ev(
                ExecutionEventType.AGENT_START, root, r_id, request, start(root, request)
            ),
            "call": ev(ExecutionEventType.AGENT_CALL, root, r_id, action=call),
            "sub_start": ev(ExecutionEventType.AGENT_START, sub, s_id, task, start(sub, task)),
            "sub_work": [
                ev(ExecutionEventType.TOOL_START, sub, s_id, action=tool),
                ev(ExecutionEventType.TOOL_END, sub, s_id, action=done),
                *converted,
                ev(ExecutionEventType.AGENT_END, sub, s_id, sub_answer, action=end_action),
            ],
            "root_answer": ev(ExecutionEventType.RESPONSE, root, r_id, f"Done: {sub_answer}"),
        }

    @staticmethod
    def _sub_context(history, start, visibility):
        if visibility != ContextVisibility.GLOBAL:
            history = [e for e in history if e.author == TestReCalledSubAgentGetsItsTask.SUB]
        context = make_context(conversation_history=history, interaction_history=[start])
        context.agent_input = start
        return context

    @staticmethod
    def _texts(messages) -> list[str]:
        return [m.content or "" for m in messages]

    @pytest.mark.parametrize("use_stop", [False, True])
    @pytest.mark.parametrize("visibility", list(ContextVisibility))
    @pytest.mark.asyncio
    async def test_second_call_sees_the_new_task(self, visibility, use_stop):
        t1, t2 = self._turn(1, "1234", use_stop), self._turn(2, "5678", use_stop)
        history = [
            t1["user_input"],
            t1["root_start"],
            t1["call"],
            t1["sub_start"],
            *t1["sub_work"],
            t1["root_answer"],
            t2["user_input"],
            t2["call"],
        ]
        context = self._sub_context(history, t2["sub_start"], visibility)

        messages = await ToolUseWithThoughtContextManager().get_context(
            context, self._agent(self.SUB, use_stop)
        )

        texts = self._texts(messages)
        new_task = [i for i, t in enumerate(texts) if "Task assigned: check order 5678" in t]
        first_end = next(i for i, t in enumerate(texts) if f"Agent: {self.SUB} ended" in t)
        assert len(new_task) == 1 and new_task[0] > first_end
        assert sum("Task assigned: check order 1234" in t for t in texts) == 1

    @pytest.mark.asyncio
    async def test_global_sub_keeps_a_task_equal_to_the_request(self):
        # Same text as the user's request, but the sub's own interaction: not a duplicate.
        t1, t2 = self._turn(1, "1234", False, True), self._turn(2, "5678", False, True)
        history = [t1["user_input"], t1["root_start"], t1["call"], t1["sub_start"]]
        history += [*t1["sub_work"], t1["root_answer"], t2["user_input"], t2["call"]]
        context = self._sub_context(history, t2["sub_start"], ContextVisibility.GLOBAL)

        messages = await ToolUseContextManager().get_context(context, self._agent(self.SUB, False))

        texts = self._texts(messages)
        for order in ("1234", "5678"):
            task = f"Agent: {self.SUB}\nTask assigned: what about order {order}?"
            assert sum(task in t for t in texts) == 1

    @pytest.mark.parametrize("use_stop", [False, True])
    @pytest.mark.asyncio
    async def test_global_sub_sees_the_root_task_on_first_call(self, use_stop):
        t1 = self._turn(1, "1234", use_stop)
        history = [t1["user_input"], t1["root_start"], t1["call"]]
        context = self._sub_context(history, t1["sub_start"], ContextVisibility.GLOBAL)

        messages = await ToolUseContextManager().get_context(
            context, self._agent(self.SUB, use_stop)
        )

        texts = self._texts(messages)
        assert sum("Task assigned: check order 1234" in t for t in texts) == 1
        # The root's own start duplicates the user's request: not rendered.
        assert not any(f"Agent: {self.ROOT}\nTask assigned" in t for t in texts)

    @pytest.mark.asyncio
    async def test_global_sub_does_not_take_the_root_answer_as_its_own(self):
        t1, t2 = self._turn(1, "1234", True), self._turn(2, "5678", True)
        history = [t1["user_input"], t1["call"], t1["sub_start"], *t1["sub_work"]]
        history += [t1["root_answer"], t2["user_input"], t2["call"]]
        context = self._sub_context(history, t2["sub_start"], ContextVisibility.GLOBAL)

        messages = await ToolUseContextManager().get_context(context, self._agent(self.SUB, True))

        answer = next(m for m in messages if "Done: Order 1234" in (m.content or ""))
        assert answer.role == "user"
        assert answer.content.startswith(f"Agent: {self.ROOT} replied to the user:")

    @pytest.mark.parametrize("use_stop", [False, True])
    @pytest.mark.asyncio
    async def test_root_sees_each_request_once(self, use_stop):
        t1, t2 = self._turn(1, "1234", use_stop), self._turn(2, "5678", use_stop)
        history = [
            t1["user_input"],
            t1["root_start"],
            t1["call"],
            t1["sub_start"],
            *t1["sub_work"],
            t1["root_answer"],
        ]
        # Turn 2 as the runner builds it: the start is agent_input, the request is
        # in interaction_history.
        context = make_context(conversation_history=history, interaction_history=[t2["user_input"]])
        context.agent_input = t2["root_start"]

        messages = await ToolUseContextManager().get_context(
            context, self._agent(self.ROOT, use_stop)
        )

        texts = self._texts(messages)
        assert sum("what about order 5678?" in t for t in texts) == 1
        assert texts[-1] == "USER INPUT: what about order 5678?"
        assert sum("what about order 1234?" in t for t in texts) == 1
        # The root's own past answer stays its own.
        answer = next(m for m in messages if "Done: Order 1234" in (m.content or ""))
        assert answer.role == "assistant"


class TestAgentEndUseStopRendering:
    """AGENT_END renders perspective-aware on the CONSUMER's relationship with
    the STOP tool: a self-contained STOP tool-call pair when the consumer has
    ``use_stop``, plain user-role narration when it does not (a STOP tool_use
    in a non-STOP agent's history baits it into calling an undeclared tool).
    """

    @staticmethod
    def _agent_end(parameters=None, content_text="sub answer", event_id="evt_sub_end"):
        return make_event(
            ExecutionEventType.AGENT_END,
            author="SubAgent",
            event_id=event_id,
            content_text=content_text,
            action=ToolAction(name="STOP", parameters=parameters),
        )

    @staticmethod
    def _consumer(use_stop: bool, fv: bool = True):
        agent = make_mock_agent()
        agent.name = "Parent"
        agent.planner = MagicMock()
        agent.planner.use_stop = use_stop
        agent.planner.forward_verbatim_enabled = fv
        return agent

    @pytest.mark.asyncio
    async def test_use_stop_consumer_renders_stop_tool_call_pair(self):
        consumer = self._consumer(use_stop=True)
        context = make_context(conversation_history=[self._agent_end()])
        ctx_mgr = ToolUseContextManager()

        messages = await ctx_mgr.get_context(context=context, agent=consumer)

        assistant = next(m for m in messages if m.role == "assistant" and m.tool_calls)
        assert assistant.tool_calls[0].function.name == ctx_mgr.stop_tool_name
        stop_id = assistant.tool_calls[0].id
        # The paired tool-role reply closes the synthetic STOP call.
        tool_msg = next(m for m in messages if m.role == "tool" and m.tool_call_id == stop_id)
        assert tool_msg.name == ctx_mgr.stop_tool_name
        assert "Information from this agent's execution: sub answer" in tool_msg.content

    @pytest.mark.asyncio
    async def test_use_stop_replays_recorded_args_faithfully(self):
        consumer = self._consumer(use_stop=True)
        context = make_context(
            conversation_history=[self._agent_end(parameters={"info": "x", "extra": 1})]
        )
        ctx_mgr = ToolUseContextManager()

        messages = await ctx_mgr.get_context(context=context, agent=consumer)

        assistant = next(m for m in messages if m.role == "assistant" and m.tool_calls)
        # Recorded args replayed verbatim — a hardcoded "{}" would falsify the
        # in-context STOP examples the LLM imitates.
        assert json.loads(assistant.tool_calls[0].function.arguments) == {"info": "x", "extra": 1}

    @pytest.mark.asyncio
    async def test_use_stop_reconstructs_args_when_none_recorded(self):
        consumer = self._consumer(use_stop=True)
        context = make_context(conversation_history=[self._agent_end(parameters=None)])
        ctx_mgr = ToolUseContextManager()

        messages = await ctx_mgr.get_context(context=context, agent=consumer)

        assistant = next(m for m in messages if m.role == "assistant" and m.tool_calls)
        # No recorded args (framework-fabricated prose promotion) → reconstruct
        # {"info": <content>} so every STOP example in history is well-formed.
        assert json.loads(assistant.tool_calls[0].function.arguments) == {"info": "sub answer"}

    @pytest.mark.asyncio
    async def test_non_stop_consumer_renders_user_narration(self):
        consumer = self._consumer(use_stop=False)
        context = make_context(conversation_history=[self._agent_end()])
        ctx_mgr = ToolUseContextManager()

        messages = await ctx_mgr.get_context(context=context, agent=consumer)

        # No STOP tool_call anywhere — that would bait an undeclared tool call.
        assert not any(m.role == "assistant" and m.tool_calls for m in messages)
        assert not any(m.role == "tool" for m in messages)
        narration = next(m for m in messages if m.role == "user" and "ended" in m.content)
        assert narration.content.startswith("Agent: SubAgent ended")
        assert "Information from this agent's execution: sub answer" in narration.content

    @pytest.mark.asyncio
    async def test_empty_content_agent_end_is_skipped(self):
        for use_stop in (True, False):
            consumer = self._consumer(use_stop=use_stop)
            agent_end = make_event(
                ExecutionEventType.AGENT_END,
                author="SubAgent",
                action=ToolAction(name="STOP", parameters=None),
            )
            context = make_context(conversation_history=[agent_end])

            messages = await ToolUseContextManager().get_context(context=context, agent=consumer)

            # Only the system message — the empty-content AGENT_END produced nothing.
            assert [m.role for m in messages] == ["system"]

    @pytest.mark.asyncio
    async def test_no_double_turn_with_preceding_response(self):
        # A RESPONSE followed by an empty-content AGENT_END must not yield two
        # assistant turns; the empty-content guard drops the AGENT_END.
        consumer = self._consumer(use_stop=False)
        response = make_event(
            ExecutionEventType.RESPONSE, author="Parent", content_text="final answer"
        )
        agent_end = make_event(
            ExecutionEventType.AGENT_END,
            author="Parent",
            action=ToolAction(name="STOP", parameters=None),
        )
        context = make_context(conversation_history=[response, agent_end])

        messages = await ToolUseContextManager().get_context(context=context, agent=consumer)

        assistant_msgs = [m for m in messages if m.role == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].content == "final answer"

    @pytest.mark.asyncio
    async def test_header_in_narration_when_fv_enabled(self):
        consumer = self._consumer(use_stop=False, fv=True)
        context = make_context(conversation_history=[self._agent_end(event_id="evt_h")])

        messages = await ToolUseContextManager().get_context(context=context, agent=consumer)

        narration = next(m for m in messages if m.role == "user" and "ended" in m.content)
        assert "[event_id: evt_h]" in narration.content

    @pytest.mark.asyncio
    async def test_no_header_in_narration_when_fv_disabled(self):
        consumer = self._consumer(use_stop=False, fv=False)
        context = make_context(conversation_history=[self._agent_end(event_id="evt_h")])

        messages = await ToolUseContextManager().get_context(context=context, agent=consumer)

        narration = next(m for m in messages if m.role == "user" and "ended" in m.content)
        assert "[event_id:" not in narration.content


# --- ToolUseWithThoughtContextManager Tests ---


class TestToolUseWithThoughtContextManager:
    def test_instantiation(self):
        ctx = ToolUseWithThoughtContextManager()
        assert ctx.type == "thought_tool_use"

    def test_is_subclass_of_tool_use_context_manager(self):
        assert issubclass(ToolUseWithThoughtContextManager, ToolUseContextManager)

    @pytest.mark.asyncio
    async def test_system_prompt_contains_thought_instruction(self):
        """ToolUseWithThoughtContextManager must inject the reasoning instruction into the system prompt."""
        ctx_mgr = ToolUseWithThoughtContextManager()
        agent = make_mock_agent(instruction="You are a helpful assistant.")
        context = make_context()

        messages = await ctx_mgr.get_context(context=context, agent=agent)

        system_msg = messages[0]
        assert system_msg.role == "system"
        assert "You are a helpful assistant." in system_msg.content
        assert _REACT_THOUGHT_INSTRUCTION in system_msg.content

    @pytest.mark.asyncio
    async def test_system_prompt_thought_instruction_absent_in_plain_tool_use(self):
        """Plain ToolUseContextManager must NOT include the reasoning instruction."""
        ctx_mgr = ToolUseContextManager()
        agent = make_mock_agent(instruction="You are a helpful assistant.")
        context = make_context()

        messages = await ctx_mgr.get_context(context=context, agent=agent)

        assert _REACT_THOUGHT_INSTRUCTION not in messages[0].content

    @pytest.mark.asyncio
    async def test_thought_instruction_appended_after_agent_instruction(self):
        """The reasoning instruction must come after the agent's own instruction."""
        ctx_mgr = ToolUseWithThoughtContextManager()
        agent = make_mock_agent(instruction="Custom agent instruction.")
        context = make_context()

        messages = await ctx_mgr.get_context(context=context, agent=agent)

        system_content = messages[0].content
        instruction_pos = system_content.index("Custom agent instruction.")
        thought_pos = system_content.index(_REACT_THOUGHT_INSTRUCTION)
        assert instruction_pos < thought_pos

    @pytest.mark.asyncio
    async def test_tool_definitions_identical_to_base(self):
        """ToolUseWithThoughtContextManager must expose the same tool definitions as the base."""
        base_ctx = ToolUseContextManager()
        react_ctx = ToolUseWithThoughtContextManager()
        tool = make_mock_tool("calc", "A calculator")
        agent = make_mock_agent(tools=[tool])
        agent.planner = MagicMock()
        agent.planner.use_stop = False
        context = make_context()

        base_defs = base_ctx.get_tool_definitions(agent, context)
        react_defs = react_ctx.get_tool_definitions(agent, context)

        assert base_defs == react_defs

    @pytest.mark.asyncio
    async def test_react_strategy_uses_react_context_manager_by_default(self):
        """ToolUseWithThoughtPlannerStrategy must default to ToolUseWithThoughtContextManager."""
        from domyn.agents.planner_strategy import ToolUseWithThoughtPlannerStrategy

        strategy = ToolUseWithThoughtPlannerStrategy()
        assert isinstance(strategy.context_manager, ToolUseWithThoughtContextManager)

    @pytest.mark.asyncio
    async def test_react_strategy_doesnt_emit_thought_in_llm_end(self):
        """ToolUseWithThoughtPlannerStrategy must not emit Part(thought=...) in LLM_END event."""
        from domyn.agents.planner_strategy import ToolUseWithThoughtPlannerStrategy

        strategy = ToolUseWithThoughtPlannerStrategy(use_stop=False)
        tool = make_mock_tool("search")
        agent = make_mock_agent(tools=[tool])
        agent.planner = MagicMock()
        agent.planner.use_stop = False

        # First call: LLM returns text + tool call (text is the thought)
        # Second call: LLM returns final RESPOND special tool
        ctx_mgr = ToolUseWithThoughtContextManager()
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            side_effect=[
                LLMResponse(
                    content="Let me search for that first.",
                    generated_tokens=10,
                    execution_time=0.1,
                    tool_calls=[
                        LLMToolCall(
                            id="call_1",
                            function=ToolCallFunction(
                                name="search",
                                arguments='{"query": "x", "reasoning": "Let me search for that first."}',
                            ),
                        )
                    ],
                ),
                LLMResponse(
                    content="",
                    generated_tokens=5,
                    execution_time=0.1,
                    tool_calls=[
                        LLMToolCall(
                            id="call_2",
                            function=ToolCallFunction(
                                name=ctx_mgr.respond_tool_name,
                                arguments='{"text": "Here is the answer.", "need_feedback": false}',
                            ),
                        )
                    ],
                ),
            ]
        )

        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Find x")]
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        # Find the first LLM_END event
        llm_end_events = [e for e in events if e.event_type == ExecutionEventType.LLM_END]
        first_llm_end = llm_end_events[0]

        # It must carry the thought
        assert len(first_llm_end.content) == 1
        assert first_llm_end.content[0].thought is None

        # The TOOL_START action must also carry the thought
        tool_start = next(e for e in events if e.event_type == ExecutionEventType.TOOL_START)
        assert tool_start.action.thought == "Let me search for that first."


# --- ToolUseWithThoughtPlannerStrategy Tests ---


class TestToolUseWithThoughtPlannerStrategy:
    """Tests for ToolUseWithThoughtPlannerStrategy and _extract_thought."""

    # ── helpers ──────────────────────────────────────────────────────────────

    def _make_strategy(self, use_stop: bool = False) -> "ToolUseWithThoughtPlannerStrategy":
        return ToolUseWithThoughtPlannerStrategy(use_stop=use_stop)

    def _make_agent(self, tools=None, use_stop: bool = False) -> MagicMock:
        agent = make_mock_agent(tools=tools or [])
        agent.planner = MagicMock()
        agent.planner.use_stop = use_stop
        return agent

    def _llm_response(
        self, content="", tool_calls=None, reasoning_content=None, thinking_blocks=None
    ):
        return LLMResponse(
            content=content,
            generated_tokens=5,
            execution_time=0.1,
            tool_calls=tool_calls or [],
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks,
        )

    def _tool_call(self, name: str, arguments: dict, call_id: str = "call_1") -> LLMToolCall:
        return LLMToolCall(
            id=call_id,
            function=ToolCallFunction(name=name, arguments=json.dumps(arguments)),
        )

    # ── _extract_thought tests ────────────────────────────────────────────────

    def test_extract_thought_from_reasoning_content(self):
        from domyn.agents.planner_strategy.tool_use_thought import _extract_thought

        llm_call = self._tool_call("search", {})
        response = self._llm_response(
            content="some text",
            reasoning_content="deep reasoning",
            tool_calls=[llm_call],
        )
        assert _extract_thought(llm_call, response, True) == "deep reasoning"

    def test_extract_thought_from_thinking_blocks(self):
        from domyn.agents.planner_strategy.tool_use_thought import _extract_thought

        llm_call = self._tool_call("search", {})
        response = self._llm_response(
            content="",
            thinking_blocks=[
                {"type": "thinking", "thinking": "block reasoning"},
                {"type": "text", "text": "something else"},
            ],
            tool_calls=[llm_call],
        )
        assert _extract_thought(llm_call, response, True) == "block reasoning"

    def test_extract_thought_from_content_preamble(self):
        from domyn.agents.planner_strategy.tool_use_thought import _extract_thought

        llm_call = self._tool_call("search", {})
        response = self._llm_response(
            content="I should call search.",
            tool_calls=[llm_call],
        )
        assert _extract_thought(llm_call, response, True) == "I should call search."

    def test_extract_thought_returns_none_when_no_thought(self):
        from domyn.agents.planner_strategy.tool_use_thought import _extract_thought

        # content only, no tool calls → final answer, not a preamble thought
        response = self._llm_response(content="Final answer.")
        assert _extract_thought(None, response, True) is None

    def test_extract_thought_reasoning_content_takes_priority_over_content(self):
        from domyn.agents.planner_strategy.tool_use_thought import _extract_thought

        response = self._llm_response(
            content="plain preamble",
            reasoning_content="reasoning wins",
            tool_calls=[self._tool_call("search", {})],
        )
        llm_call = self._tool_call("search", {})
        assert _extract_thought(llm_call, response, True) == "reasoning wins"

    def test_extract_thought_reasoning_content_takes_priority_over_thinking_blocks(self):
        from domyn.agents.planner_strategy.tool_use_thought import _extract_thought

        response = self._llm_response(
            content="",
            reasoning_content="reasoning wins",
            thinking_blocks=[{"type": "thinking", "thinking": "block"}],
            tool_calls=[self._tool_call("search", {})],
        )
        llm_call = self._tool_call("search", {})
        assert _extract_thought(llm_call, response, True) == "reasoning wins"

    def test_extract_thought_tolerates_malformed_arguments(self):
        """Malformed tool-call arguments must not raise — fall through to the
        other thought sources. Regression: a raise here produced an orphan
        TOOL_ERROR (empty call_id, no paired assistant tool_call) that strict
        providers (Google/Baidu/SiliconFlow via OpenRouter) reject with 400."""
        from domyn.agents.planner_strategy.tool_use_thought import _extract_thought

        # missing comma between fields → invalid JSON
        bad_call = LLMToolCall(
            id="call_1",
            function=ToolCallFunction(name="search", arguments='{"query": "x" "k": 1}'),
        )
        # reasoning is unreadable from the broken args, but a fallback source exists
        response = self._llm_response(
            content="", reasoning_content="fallback reasoning", tool_calls=[bad_call]
        )
        assert _extract_thought(bad_call, response, True) == "fallback reasoning"

        # with no fallback source it returns None — still no raise
        assert _extract_thought(bad_call, self._llm_response(tool_calls=[bad_call]), False) is None

    # ── instantiation ─────────────────────────────────────────────────────────

    def test_instantiation(self):
        from domyn.agents.planner_strategy import (
            ToolUseWithThoughtContextManager,
            ToolUseWithThoughtPlannerStrategy,
        )

        strategy = ToolUseWithThoughtPlannerStrategy()
        assert strategy.type == "tool_use_thought"
        assert strategy.name == "tool_use_thought"
        assert isinstance(strategy.context_manager, ToolUseWithThoughtContextManager)
        assert strategy.use_stop is False

    def test_is_subclass_of_tool_use_planner_strategy(self):
        from domyn.agents.planner_strategy import ToolUseWithThoughtPlannerStrategy

        assert issubclass(ToolUseWithThoughtPlannerStrategy, ToolUsePlannerStrategy)

    # ── LLM_END without thought ───────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_llm_end_without_thought_emits_fallback_text(self):
        """When there is no thought, LLM_END must carry plain text, not Part(thought=...)."""
        strategy = self._make_strategy()
        agent = self._make_agent()
        ctx_mgr = strategy.context_manager
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            return_value=self._llm_response(
                content="",
                tool_calls=[
                    self._tool_call(
                        ctx_mgr.respond_tool_name, {"text": "Hi", "need_feedback": False}
                    )
                ],
            )
        )
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Hello")]
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        llm_end = next(e for e in events if e.event_type == ExecutionEventType.LLM_END)
        assert llm_end.content[0].thought is None
        assert llm_end.content[0].text is not None

    # ── thought propagation into actions ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_thought_propagated_to_tool_action(self):
        """ToolAction.thought must carry the extracted thought."""
        strategy = self._make_strategy()
        tool = make_mock_tool("search")
        agent = self._make_agent(tools=[tool])
        agent.llm_provider = AsyncMock()
        ctx_mgr = strategy.context_manager

        agent.llm_provider.generate = AsyncMock(
            side_effect=[
                self._llm_response(
                    content="Let me search.",
                    tool_calls=[self._tool_call("search", {"query": "x"}, "c1")],
                ),
                self._llm_response(
                    content="",
                    tool_calls=[
                        self._tool_call(
                            ctx_mgr.respond_tool_name,
                            {"text": "Done.", "need_feedback": False},
                            "c2",
                        )
                    ],
                ),
            ]
        )
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Search")]
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        tool_start = next(e for e in events if e.event_type == ExecutionEventType.TOOL_START)
        assert tool_start.action.thought == "Let me search."
        assert tool_start.action.call_id == "c1"

    @pytest.mark.asyncio
    async def test_malformed_arguments_emit_no_orphan_tool_error(self):
        """A tool call with persistently malformed JSON arguments must NOT produce
        an orphan TOOL_ERROR/TOOL_START (empty call_id, no paired assistant
        tool_call). After the planner retries and exhausts MAX_TOOL_CALL_ARGS_RETRIES,
        it must give up with a bare ERROR_OCCURRED rather than ever committing
        a TOOL_START/TOOL_ERROR for the unresolved call_id — the assistant's
        malformed tool_calls were never exposed downstream, so nothing needs a
        paired tool-role message. Regression for the deepseek/mistral/nex-agi 400
        'missing field tool_call_id' cascade."""
        strategy = self._make_strategy()
        tool = make_mock_tool("search")
        agent = self._make_agent(tools=[tool])
        agent.llm_provider = AsyncMock()

        # missing comma between fields → invalid JSON arguments
        bad_call = LLMToolCall(
            id="c1",
            function=ToolCallFunction(name="search", arguments='{"query": "x" "k": 1}'),
        )
        agent.llm_provider.generate = AsyncMock(
            return_value=self._llm_response(content="", tool_calls=[bad_call])
        )
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Search")]
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        assert agent.llm_provider.generate.call_count == 1 + MAX_TOOL_CALL_ARGS_RETRIES
        # no orphan tool-error (empty call_id) emitted
        orphans = [
            e
            for e in events
            if e.event_type == ExecutionEventType.TOOL_ERROR and not e.action.call_id
        ]
        assert orphans == []
        # c1's tool_call is never committed to history at all (no TOOL_START/TOOL_ERROR)
        assert not any(e.event_type == ExecutionEventType.TOOL_START for e in events)
        assert not any(e.event_type == ExecutionEventType.TOOL_ERROR for e in events)
        assert any(e.event_type == ExecutionEventType.ERROR_OCCURRED for e in events)

    @pytest.mark.asyncio
    async def test_thought_propagated_to_agent_handoff(self):
        """AgentAction.thought and .message must carry the extracted thought on handoff."""
        strategy = self._make_strategy()
        agent = self._make_agent()
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            return_value=self._llm_response(
                content="Delegating to helper.",
                tool_calls=[self._tool_call("helper_agent", {"task": "do it"}, "c_ag")],
            )
        )
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Do it")]
        )
        sub_desc = MagicMock()
        sub_desc.name = "helper_agent"
        sub_desc.description = "helper"
        sub_desc.invocation_parameters = {
            "type": "object",
            "properties": {"task": {"type": "string"}},
        }
        context.handoff_targets_description = [sub_desc]

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        agent_start = next(e for e in events if e.event_type == ExecutionEventType.AGENT_CALL)
        assert agent_start.action.thought == "Delegating to helper."
        assert agent_start.action.message == "Delegating to helper."
        assert agent_start.action.parameters == {"task": "do it"}

    @pytest.mark.asyncio
    async def test_thought_propagated_to_stop_action(self):
        """AGENT_END action.thought must carry the thought when use_stop=True."""
        strategy = self._make_strategy(use_stop=True)
        agent = self._make_agent(use_stop=True)
        agent.llm_provider = AsyncMock()
        ctx_mgr = strategy.context_manager
        agent.llm_provider.generate = AsyncMock(
            return_value=self._llm_response(
                content="I am done thinking.",
                tool_calls=[self._tool_call(ctx_mgr.stop_tool_name, {"info": "result"}, "c_stop")],
            )
        )
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Go")]
        )
        context.handoff_targets_description = []

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        agent_end = next(e for e in events if e.event_type == ExecutionEventType.AGENT_END)
        assert agent_end.action.thought == "I am done thinking."
        assert agent_end.action.parameters == {"info": "result"}
        # AGENT_END content is normalized at creation to carry the STOP info text.
        assert agent_end.content == [Part(text="result")]

    @pytest.mark.asyncio
    async def test_thought_propagated_to_ask_user_response(self):
        """RESPONSE action.thought must carry the thought for ASK_USER tool (use_stop=True)."""
        strategy = self._make_strategy(use_stop=True)
        agent = self._make_agent(use_stop=True)
        agent.llm_provider = AsyncMock()
        ctx_mgr = strategy.context_manager
        agent.llm_provider.generate = AsyncMock(
            return_value=self._llm_response(
                content="I need to ask.",
                tool_calls=[
                    self._tool_call(
                        ctx_mgr.ask_user_tool_name, {"text": "Can you clarify?"}, "c_ask"
                    )
                ],
            )
        )
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Go")]
        )
        context.handoff_targets_description = []

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        response_event = next(e for e in events if e.event_type == ExecutionEventType.RESPONSE)
        assert response_event.action.thought == "I need to ask."
        assert response_event.content[0].text == "Can you clarify?"
        assert response_event.is_partial is False
        assert response_event.need_feedback is True

    # ── factory registration ──────────────────────────────────────────────────

    def test_factory_creates_react_tool_use_strategy(self):
        from domyn.agents.planner_strategy import (
            PlannerStrategyFactory,
            ToolUseWithThoughtContextManager,
            ToolUseWithThoughtPlannerStrategy,
        )

        strategy = PlannerStrategyFactory.from_config({"type": "tool_use_thought", "name": "test"})
        assert isinstance(strategy, ToolUseWithThoughtPlannerStrategy)
        assert isinstance(strategy.context_manager, ToolUseWithThoughtContextManager)

    def test_factory_react_tool_use_with_custom_max_iterations(self):
        from domyn.agents.planner_strategy import PlannerStrategyFactory

        strategy = PlannerStrategyFactory.from_config(
            {"type": "tool_use_thought", "name": "test", "max_iterations": 10}
        )
        assert strategy.max_iterations == 10

    # ── error handling ────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_llm_error_emits_error_occurred(self):
        strategy = self._make_strategy()
        agent = self._make_agent()
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(side_effect=Exception("timeout"))
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Hi")]
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        assert any(e.event_type == ExecutionEventType.ERROR_OCCURRED for e in events)

    @pytest.mark.asyncio
    async def test_malformed_tool_call_args_retries_then_recovers(self):
        """A malformed tool-call arguments JSON should trigger a verbatim retry."""
        strategy = self._make_strategy()
        tool = make_mock_tool("search")
        agent = self._make_agent(tools=[tool])
        agent.llm_provider = AsyncMock()

        malformed_response = self._llm_response(
            tool_calls=[
                LLMToolCall(
                    id="call_1",
                    function=ToolCallFunction(name="search", arguments='{"query": "test"'),
                )
            ]
        )
        valid_response = self._llm_response(
            tool_calls=[
                LLMToolCall(
                    id="call_1",
                    function=ToolCallFunction(name="search", arguments='{"query": "test"}'),
                )
            ]
        )
        # A 3rd, terminal response for the next planner iteration (which runs
        # after the tool call from `valid_response` is processed) — otherwise
        # the outer loop would call generate() again with nothing left to return.
        final_response = self._llm_response(content="Here are the results.")
        agent.llm_provider.generate = AsyncMock(
            side_effect=[malformed_response, valid_response, final_response]
        )

        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Hi")]
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        assert agent.llm_provider.generate.call_count == 3
        # The first two calls (malformed, then retry) are both for the same
        # planner iteration and must be a verbatim resend — same `messages`.
        first_call_messages = agent.llm_provider.generate.call_args_list[0].kwargs["messages"]
        second_call_messages = agent.llm_provider.generate.call_args_list[1].kwargs["messages"]
        assert first_call_messages == second_call_messages

        event_types = [e.event_type for e in events]
        assert ExecutionEventType.ERROR_OCCURRED not in event_types
        tool_start = next(e for e in events if e.event_type == ExecutionEventType.TOOL_START)
        assert tool_start.action.parameters == {"query": "test"}

    @pytest.mark.asyncio
    async def test_malformed_tool_call_args_gives_up_after_max_retries(self):
        """Persistent malformed JSON should error out after MAX_TOOL_CALL_ARGS_RETRIES."""
        strategy = self._make_strategy()
        tool = make_mock_tool("search")
        agent = self._make_agent(tools=[tool])
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            return_value=self._llm_response(
                tool_calls=[
                    LLMToolCall(
                        id="call_1",
                        function=ToolCallFunction(name="search", arguments='{"query": "test"'),
                    )
                ]
            )
        )

        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Hi")]
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        assert agent.llm_provider.generate.call_count == 1 + MAX_TOOL_CALL_ARGS_RETRIES
        error_events = [e for e in events if e.event_type == ExecutionEventType.ERROR_OCCURRED]
        assert len(error_events) == 1
        assert not any(e.event_type == ExecutionEventType.TOOL_START for e in events)
        assert not any(e.event_type == ExecutionEventType.RESPONSE for e in events)

    @pytest.mark.asyncio
    async def test_max_iterations_emits_response(self):
        strategy = self._make_strategy()
        strategy.max_iterations = 2
        tool = make_mock_tool("loop_tool")
        agent = self._make_agent(tools=[tool])
        agent.llm_provider = AsyncMock()
        # always return a tool call → never terminates naturally
        agent.llm_provider.generate = AsyncMock(
            return_value=self._llm_response(
                content="thinking...",
                tool_calls=[self._tool_call("loop_tool", {})],
            )
        )
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Loop")]
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        final = next(e for e in events if e.event_type == ExecutionEventType.RESPONSE)
        assert "maximum iterations" in final.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_max_iterations_use_stop_emits_agent_end(self):
        strategy = self._make_strategy(use_stop=True)
        strategy.max_iterations = 2
        tool = make_mock_tool("loop_tool")
        agent = self._make_agent(tools=[tool], use_stop=True)
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            return_value=self._llm_response(
                content="thinking...",
                tool_calls=[self._tool_call("loop_tool", {})],
            )
        )
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Loop")]
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        agent_end = next(e for e in events if e.event_type == ExecutionEventType.AGENT_END)
        assert agent_end.action.name == strategy.context_manager.stop_tool_name
        assert agent_end.action.parameters is None
        assert agent_end.content[0].text == "Reached maximum iterations (2)"
        assert not any(e.event_type == ExecutionEventType.RESPONSE for e in events)
        assert agent.llm_provider.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_plain_response_with_use_stop_emits_agent_end_content_only(self):
        """Content-only response under ``use_stop`` terminates via AGENT_END with
        ``parameters=None`` and the text in ``content``."""
        strategy = self._make_strategy(use_stop=True)
        agent = self._make_agent(use_stop=True)
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            return_value=self._llm_response(content="Here is my answer.")
        )
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Go")]
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        agent_end = next(e for e in events if e.event_type == ExecutionEventType.AGENT_END)
        assert agent_end.action.name == strategy.context_manager.stop_tool_name
        assert agent_end.action.parameters is None
        assert agent_end.content[0].text == "Here is my answer."
        assert not any(e.event_type == ExecutionEventType.RESPONSE for e in events)

    @pytest.mark.asyncio
    async def test_unknown_tool_emits_tool_start_before_tool_error(self):
        """Tool-not-found emits a paired TOOL_START immediately before the
        TOOL_ERROR (same call_id) so the failed call renders as a complete pair."""
        strategy = self._make_strategy()
        agent = self._make_agent(tools=[])
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            side_effect=[
                self._llm_response(tool_calls=[self._tool_call("nonexistent", {}, "call_bad")]),
                self._llm_response(content="Sorry."),
            ]
        )
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Go")]
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        types = [e.event_type for e in events]
        ts_idx = types.index(ExecutionEventType.TOOL_START)
        te_idx = types.index(ExecutionEventType.TOOL_ERROR)
        assert te_idx == ts_idx + 1
        assert events[ts_idx].action.name == "nonexistent"
        assert events[ts_idx].action.call_id == "call_bad"
        assert events[te_idx].action.call_id == "call_bad"

    # ── agent.stream = True ───────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_stream_true_calls_generate_streaming(self):
        """When agent.stream=True, generate_streaming must be called instead of generate."""
        strategy = self._make_strategy()
        agent = self._make_agent()
        agent.stream = True
        agent.llm_provider = AsyncMock()

        # generate_streaming yields an LLMResponse with a tool call that ends the loop
        ctx_mgr = strategy.context_manager
        terminal_response = LLMResponse(
            content="",
            generated_tokens=3,
            execution_time=0.1,
            tool_calls=[
                LLMToolCall(
                    id="c_stream",
                    function=ToolCallFunction(
                        name=ctx_mgr.respond_tool_name,
                        arguments='{"text": "streaming done", "need_feedback": false}',
                    ),
                )
            ],
        )

        async def fake_streaming(*args, **kwargs):
            yield terminal_response

        agent.llm_provider.generate_streaming = fake_streaming
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Hi")]
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        # generate_streaming was used (generate was never called)
        agent.llm_provider.generate.assert_not_called()
        # The loop still reaches LLM_END and produces a RESPONSE
        assert any(e.event_type == ExecutionEventType.LLM_END for e in events)
        assert any(e.event_type == ExecutionEventType.RESPONSE for e in events)

    @pytest.mark.asyncio
    async def test_stream_true_yields_response_token_for_content_chunks(self):
        """Content chunks from generate_streaming must be emitted as RESPONSE_TOKEN events."""
        strategy = self._make_strategy()
        agent = self._make_agent()
        agent.stream = True
        agent.llm_provider = AsyncMock()

        async def fake_streaming(*args, **kwargs):
            yield LLMResponse(content="Hello ", generated_tokens=1, execution_time=0.0)
            yield LLMResponse(content="world", generated_tokens=2, execution_time=0.0)

        agent.llm_provider.generate_streaming = fake_streaming
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Hi")]
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        token_events = [e for e in events if e.event_type == ExecutionEventType.RESPONSE_TOKEN]
        assert len(token_events) == 2
        assert token_events[0].content[0].text == "Hello "
        assert token_events[1].content[0].text == "world"
        assert all(e.is_partial is True for e in token_events)

        # Final RESPONSE is canonical history/observability; the runner
        # suppresses it from live stream output when RESPONSE_TOKENs already
        # carried the text.
        response_event = next(e for e in events if e.event_type == ExecutionEventType.RESPONSE)
        assert response_event.content[0].text == "Hello world"

    @pytest.mark.asyncio
    async def test_stream_true_llm_error_emits_error_occurred(self):
        """An exception from generate_streaming must emit ERROR_OCCURRED and stop the loop."""
        strategy = self._make_strategy()
        agent = self._make_agent()
        agent.stream = True
        agent.llm_provider = AsyncMock()

        async def failing_streaming(*args, **kwargs):
            raise RuntimeError("stream broken")
            yield  # make it an async generator

        agent.llm_provider.generate_streaming = failing_streaming
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Hi")]
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        assert any(e.event_type == ExecutionEventType.ERROR_OCCURRED for e in events)

    @pytest.mark.asyncio
    async def test_stream_true_non_str_llm_response_chunk_not_emitted_as_token(self):
        """LLMResponse chunks (non-str) from generate_streaming must NOT produce RESPONSE_TOKEN."""
        strategy = self._make_strategy()
        agent = self._make_agent()
        agent.stream = True
        agent.llm_provider = AsyncMock()

        ctx_mgr = strategy.context_manager

        async def fake_streaming(*args, **kwargs):
            # Only LLMResponse objects — no str chunks
            yield LLMResponse(
                content="",
                generated_tokens=1,
                execution_time=0.1,
                tool_calls=[
                    LLMToolCall(
                        id="c_llm",
                        function=ToolCallFunction(
                            name=ctx_mgr.respond_tool_name,
                            arguments='{"text": "ok", "need_feedback": false}',
                        ),
                    )
                ],
            )

        agent.llm_provider.generate_streaming = fake_streaming
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Hi")]
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        token_events = [e for e in events if e.event_type == ExecutionEventType.RESPONSE_TOKEN]
        assert token_events == [], "LLMResponse chunks must not produce RESPONSE_TOKEN events"

    @pytest.mark.xfail(
        reason=(
            "known gap in tool_use_thought: pre-tool content "
            "is still treated as thought and may also stream as RESPONSE_TOKEN."
        ),
        strict=True,
    )
    @pytest.mark.asyncio
    async def test_stream_true_buffers_thought_before_tool_calls(self):
        """Streaming thought before a tool call must not leak as user-visible tokens."""
        strategy = self._make_strategy()
        tool = make_mock_tool("search")
        agent = self._make_agent(tools=[tool])
        agent.stream = True
        agent.llm_provider = AsyncMock()

        stream_calls = 0

        async def fake_streaming(*args, **kwargs):
            nonlocal stream_calls
            stream_calls += 1
            if stream_calls == 1:
                yield LLMResponse(
                    content="I should search. ", generated_tokens=3, execution_time=0.0
                )
                yield LLMResponse(
                    content="",
                    generated_tokens=4,
                    execution_time=0.1,
                    tool_calls=[
                        LLMToolCall(
                            id="c_search",
                            function=ToolCallFunction(
                                name="search",
                                arguments='{"query": "answer"}',
                            ),
                        )
                    ],
                )
            else:
                yield LLMResponse(content="Done", generated_tokens=1, execution_time=0.0)

        agent.llm_provider.generate_streaming = fake_streaming
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Search")]
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        token_texts = [
            e.content[0].text for e in events if e.event_type == ExecutionEventType.RESPONSE_TOKEN
        ]
        assert "I should search. " not in token_texts

        tool_start_index = next(
            i for i, e in enumerate(events) if e.event_type == ExecutionEventType.TOOL_START
        )
        terminal_response_indexes = [
            i
            for i, e in enumerate(events)
            if e.event_type == ExecutionEventType.RESPONSE and e.is_partial is False
        ]
        assert all(i > tool_start_index for i in terminal_response_indexes)

    @pytest.mark.xfail(
        reason=(
            "known gap in tool_use_thought: pre-tool content "
            "is still treated as thought and may also stream as RESPONSE_TOKEN."
        ),
        strict=True,
    )
    @pytest.mark.asyncio
    async def test_stream_true_split_content_chunks_surface_as_thought(self):
        """Streaming + tool calls: split pre-tool content must surface as
        thought both on ``LLM_END`` and the following ``TOOL_START`` action.
        """
        strategy = self._make_strategy()
        tool = make_mock_tool("search")
        agent = self._make_agent(tools=[tool])
        agent.stream = True
        agent.llm_provider = AsyncMock()

        stream_calls = 0

        async def fake_streaming(*args, **kwargs):
            nonlocal stream_calls
            stream_calls += 1
            if stream_calls == 1:
                yield LLMResponse(content="Reasoning ", generated_tokens=1, execution_time=0.0)
                yield LLMResponse(content="step ", generated_tokens=1, execution_time=0.0)
                yield LLMResponse(content="one.", generated_tokens=1, execution_time=0.0)
                yield LLMResponse(
                    content="",
                    generated_tokens=1,
                    execution_time=0.0,
                    tool_calls=[
                        LLMToolCall(
                            id="c_split_thought",
                            function=ToolCallFunction(name="search", arguments='{"query": "x"}'),
                        )
                    ],
                )
            else:
                yield LLMResponse(content="Done", generated_tokens=1, execution_time=0.0)

        agent.llm_provider.generate_streaming = fake_streaming
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Go")]
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        llm_end = next(e for e in events if e.event_type == ExecutionEventType.LLM_END)
        assert llm_end.content[0].thought == "Reasoning step one."

        tool_start = next(e for e in events if e.event_type == ExecutionEventType.TOOL_START)
        assert tool_start.action.thought == "Reasoning step one."

        leaked_tokens = [
            e.content[0].text
            for e in events
            if e.event_type == ExecutionEventType.RESPONSE_TOKEN
            and e.content[0].text in {"Reasoning ", "step ", "one."}
        ]
        assert leaked_tokens == []


# --- ToolUsePlannerStrategy Tests ---


class TestToolUsePlannerStrategy:
    def test_instantiation(self):
        strategy = ToolUsePlannerStrategy(
            context_manager=ToolUseContextManager(),
        )
        assert strategy.type == "tool_use"
        assert strategy.max_iterations == 25

    def test_instantiation_from_dict(self):
        strategy = ToolUsePlannerStrategy(
            context_manager={"type": "tool_use", "name": "tool_use"},
        )
        assert strategy.context_manager.type == "tool_use"

    def test_instantiation_with_none_context(self):
        strategy = ToolUsePlannerStrategy(context_manager=None)
        assert strategy.context_manager.type == "tool_use"

    @pytest.mark.asyncio
    async def test_execute_no_tools_direct_response(self):
        """LLM returns text with no tool calls — should emit RESPONSE."""
        strategy = ToolUsePlannerStrategy(
            context_manager=ToolUseContextManager(),
        )

        agent = make_mock_agent()
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            return_value=LLMResponse(
                content="Hello! I'm here to help.",
                generated_tokens=10,
                execution_time=0.5,
                tool_calls=[],
            )
        )

        context = make_context(
            conversation_history=[
                make_event(ExecutionEventType.USER_INPUT, content_text="Hi"),
            ]
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        event_types = [e.event_type for e in events]
        assert ExecutionEventType.LLM_START in event_types
        assert ExecutionEventType.LLM_END in event_types
        assert ExecutionEventType.RESPONSE in event_types

        response_event = next(e for e in events if e.event_type == ExecutionEventType.RESPONSE)
        assert response_event.content[0].text == "Hello! I'm here to help."
        assert response_event.is_partial is False

    @pytest.mark.asyncio
    async def test_execute_with_tool_call(self):
        """LLM calls a tool — should emit TOOL_START."""
        tool = make_mock_tool("search")
        strategy = ToolUsePlannerStrategy(
            context_manager=ToolUseContextManager(),
        )
        agent = make_mock_agent(tools=[tool])

        # First call: LLM calls tool. Second call: LLM gives final response.
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            side_effect=[
                LLMResponse(
                    content="Let me search for that.",
                    generated_tokens=5,
                    execution_time=0.3,
                    tool_calls=[
                        LLMToolCall(
                            id="call_1",
                            function=ToolCallFunction(
                                name="search",
                                arguments='{"query": "test"}',
                            ),
                        )
                    ],
                ),
                LLMResponse(
                    content="Here are the results.",
                    generated_tokens=5,
                    execution_time=0.3,
                    tool_calls=[],
                ),
            ]
        )

        context = make_context(
            conversation_history=[
                make_event(ExecutionEventType.USER_INPUT, content_text="Search for test"),
            ]
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        event_types = [e.event_type for e in events]

        # First iteration: LLM_START, LLM_END, TOOL_START
        assert ExecutionEventType.TOOL_START in event_types

        tool_start = next(e for e in events if e.event_type == ExecutionEventType.TOOL_START)
        assert tool_start.action.name == "search"
        assert tool_start.action.call_id == "call_1"
        assert tool_start.action.parameters == {"query": "test"}

        # Second iteration: LLM_START, LLM_END, RESPONSE
        assert ExecutionEventType.RESPONSE in event_types

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        """LLM calls a tool that doesn't exist — should emit TOOL_ERROR."""
        strategy = ToolUsePlannerStrategy(
            context_manager=ToolUseContextManager(),
        )
        agent = make_mock_agent(tools=[])

        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            side_effect=[
                LLMResponse(
                    content="",
                    generated_tokens=5,
                    execution_time=0.3,
                    tool_calls=[
                        LLMToolCall(
                            id="call_bad",
                            function=ToolCallFunction(
                                name="nonexistent",
                                arguments="{}",
                            ),
                        )
                    ],
                ),
                LLMResponse(
                    content="Sorry about that.",
                    generated_tokens=5,
                    execution_time=0.3,
                    tool_calls=[],
                ),
            ]
        )

        context = make_context(
            conversation_history=[
                make_event(ExecutionEventType.USER_INPUT, content_text="Do something"),
            ]
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        event_types = [e.event_type for e in events]
        assert ExecutionEventType.TOOL_ERROR in event_types

        error_event = next(e for e in events if e.event_type == ExecutionEventType.TOOL_ERROR)
        assert "nonexistent" in error_event.action.observation

    @pytest.mark.asyncio
    async def test_unknown_tool_emits_tool_start_before_tool_error(self):
        """Tool-not-found now emits a paired TOOL_START immediately before the
        TOOL_ERROR so the failed call renders as a complete pair downstream."""
        strategy = ToolUsePlannerStrategy(context_manager=ToolUseContextManager())
        agent = make_mock_agent(tools=[])
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            side_effect=[
                LLMResponse(
                    content="",
                    generated_tokens=5,
                    execution_time=0.1,
                    tool_calls=[
                        LLMToolCall(
                            id="call_bad",
                            function=ToolCallFunction(name="nonexistent", arguments="{}"),
                        )
                    ],
                ),
                LLMResponse(
                    content="Sorry.", generated_tokens=5, execution_time=0.1, tool_calls=[]
                ),
            ]
        )
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Go")]
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        types = [e.event_type for e in events]
        ts_idx = types.index(ExecutionEventType.TOOL_START)
        te_idx = types.index(ExecutionEventType.TOOL_ERROR)
        assert te_idx == ts_idx + 1
        assert events[ts_idx].action.name == "nonexistent"
        assert events[ts_idx].action.call_id == "call_bad"
        assert events[te_idx].action.call_id == "call_bad"

    @pytest.mark.asyncio
    async def test_stop_tool_call_emits_agent_end_with_args_and_content(self):
        """An explicit STOP tool call records its args on the AGENT_END action
        and carries the ``info`` payload in ``content``."""
        strategy = ToolUsePlannerStrategy(context_manager=ToolUseContextManager(), use_stop=True)
        agent = make_mock_agent(tools=[])
        ctx_mgr = strategy.context_manager
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            return_value=LLMResponse(
                content="",
                generated_tokens=5,
                execution_time=0.1,
                tool_calls=[
                    LLMToolCall(
                        id="c_stop",
                        function=ToolCallFunction(
                            name=ctx_mgr.stop_tool_name,
                            arguments='{"info": "final result"}',
                        ),
                    )
                ],
            )
        )
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Go")]
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        agent_end = next(e for e in events if e.event_type == ExecutionEventType.AGENT_END)
        assert agent_end.action.name == ctx_mgr.stop_tool_name
        assert agent_end.action.parameters == {"info": "final result"}
        assert agent_end.content[0].text == "final result"

    @pytest.mark.asyncio
    async def test_plain_response_with_use_stop_emits_agent_end_content_only(self):
        """A content-only response (no tool call) under ``use_stop`` terminates
        via AGENT_END with ``parameters=None`` and the text in ``content`` (not
        a RESPONSE event)."""
        strategy = ToolUsePlannerStrategy(context_manager=ToolUseContextManager(), use_stop=True)
        agent = make_mock_agent(tools=[])
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            return_value=LLMResponse(
                content="Here is my answer.",
                generated_tokens=5,
                execution_time=0.1,
                tool_calls=[],
            )
        )
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Go")]
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        agent_end = next(e for e in events if e.event_type == ExecutionEventType.AGENT_END)
        assert agent_end.action.name == strategy.context_manager.stop_tool_name
        assert agent_end.action.parameters is None
        assert agent_end.content[0].text == "Here is my answer."
        assert not any(e.event_type == ExecutionEventType.RESPONSE for e in events)

    @pytest.mark.asyncio
    async def test_execute_agent_handoff(self):
        """LLM calls a sub-agent — should emit AGENT_START."""
        strategy = ToolUsePlannerStrategy(
            context_manager=ToolUseContextManager(),
        )
        agent = make_mock_agent(tools=[])

        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            return_value=LLMResponse(
                content="Delegating to helper.",
                generated_tokens=5,
                execution_time=0.3,
                tool_calls=[
                    LLMToolCall(
                        id="call_agent",
                        function=ToolCallFunction(
                            name="helper_agent",
                            arguments='{"task": "do stuff"}',
                        ),
                    )
                ],
            )
        )

        context = make_context(
            conversation_history=[
                make_event(ExecutionEventType.USER_INPUT, content_text="Help me"),
            ]
        )
        agent_desc = MagicMock()
        agent_desc.name = "helper_agent"
        agent_desc.description = "A helper"
        agent_desc.invocation_json_schema = None
        context.handoff_targets_description = [agent_desc]

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        event_types = [e.event_type for e in events]
        assert ExecutionEventType.AGENT_CALL in event_types

        agent_start = next(e for e in events if e.event_type == ExecutionEventType.AGENT_CALL)
        assert agent_start.action.name == "helper_agent"

    @pytest.mark.asyncio
    async def test_execute_ask_user_marks_feedback_terminal(self):
        """ASK_USER must be emitted as a persisted feedback response, not a partial thought."""
        strategy = ToolUsePlannerStrategy(
            context_manager=ToolUseContextManager(),
            use_stop=True,
        )
        agent = make_mock_agent()
        agent.planner = MagicMock()
        agent.planner.use_stop = True
        agent.llm_provider = AsyncMock()
        ctx_mgr = strategy.context_manager
        agent.llm_provider.generate = AsyncMock(
            return_value=LLMResponse(
                content="",
                generated_tokens=5,
                execution_time=0.3,
                tool_calls=[
                    LLMToolCall(
                        id="call_ask",
                        function=ToolCallFunction(
                            name=ctx_mgr.ask_user_tool_name,
                            arguments='{"text": "Can you clarify?"}',
                        ),
                    )
                ],
            )
        )
        context = make_context(
            conversation_history=[
                make_event(ExecutionEventType.USER_INPUT, content_text="Help me"),
            ]
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        response_event = next(e for e in events if e.event_type == ExecutionEventType.RESPONSE)
        assert response_event.content[0].text == "Can you clarify?"
        assert response_event.action.name == ctx_mgr.ask_user_tool_name
        assert response_event.is_partial is False
        assert response_event.need_feedback is True

    @pytest.mark.asyncio
    async def test_stream_true_does_not_emit_terminal_response_before_tool_calls(self):
        """Streaming content plus tool calls emits preamble tokens, then executes tools."""
        tool = make_mock_tool("search")
        strategy = ToolUsePlannerStrategy(
            context_manager=ToolUseContextManager(),
        )
        agent = make_mock_agent(tools=[tool])
        agent.stream = True
        agent.llm_provider = AsyncMock()

        stream_calls = 0

        async def fake_streaming(*args, **kwargs):
            nonlocal stream_calls
            stream_calls += 1
            if stream_calls == 1:
                yield LLMResponse(content="Let me search. ", generated_tokens=3, execution_time=0.0)
                yield LLMResponse(
                    content="",
                    generated_tokens=4,
                    execution_time=0.1,
                    tool_calls=[
                        LLMToolCall(
                            id="call_search",
                            function=ToolCallFunction(
                                name="search",
                                arguments='{"query": "answer"}',
                            ),
                        )
                    ],
                )
            else:
                yield LLMResponse(content="Done", generated_tokens=1, execution_time=0.0)

        agent.llm_provider.generate_streaming = fake_streaming
        context = make_context(
            conversation_history=[
                make_event(ExecutionEventType.USER_INPUT, content_text="Search for answer"),
            ]
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        tool_start_index = next(
            i for i, e in enumerate(events) if e.event_type == ExecutionEventType.TOOL_START
        )
        terminal_response_indexes = [
            i
            for i, e in enumerate(events)
            if e.event_type == ExecutionEventType.RESPONSE and e.is_partial is False
        ]
        assert terminal_response_indexes
        assert all(i > tool_start_index for i in terminal_response_indexes)

        token_texts = [
            e.content[0].text for e in events if e.event_type == ExecutionEventType.RESPONSE_TOKEN
        ]
        assert "Let me search. " in token_texts
        tool_start = events[tool_start_index]
        assert tool_start.action.thought is None
        assert tool_start.action.message is None

    @pytest.mark.asyncio
    async def test_execute_max_iterations(self):
        """Should stop after max_iterations."""
        strategy = ToolUsePlannerStrategy(
            context_manager=ToolUseContextManager(),
            max_iterations=2,
        )
        tool = make_mock_tool("search")
        agent = make_mock_agent(tools=[tool])

        # Always return tool calls — should hit max_iterations
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            return_value=LLMResponse(
                content="",
                generated_tokens=5,
                execution_time=0.1,
                tool_calls=[
                    LLMToolCall(
                        id="call_loop",
                        function=ToolCallFunction(name="search", arguments='{"query": "loop"}'),
                    )
                ],
            )
        )

        context = make_context(
            conversation_history=[
                make_event(ExecutionEventType.USER_INPUT, content_text="Loop"),
            ]
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        response_events = [e for e in events if e.event_type == ExecutionEventType.RESPONSE]
        assert len(response_events) == 1
        assert "maximum iterations" in response_events[0].content[0].text.lower()

    @pytest.mark.asyncio
    async def test_execute_max_iterations_use_stop_emits_agent_end(self):
        """Sub-agents must hand control back to the parent when max_iterations is hit."""
        strategy = ToolUsePlannerStrategy(
            context_manager=ToolUseContextManager(),
            max_iterations=2,
            use_stop=True,
        )
        tool = make_mock_tool("search")
        agent = make_mock_agent(tools=[tool])

        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            return_value=LLMResponse(
                content="",
                generated_tokens=5,
                execution_time=0.1,
                tool_calls=[
                    LLMToolCall(
                        id="call_loop",
                        function=ToolCallFunction(name="search", arguments='{"query": "loop"}'),
                    )
                ],
            )
        )

        context = make_context(
            conversation_history=[
                make_event(ExecutionEventType.USER_INPUT, content_text="Loop"),
            ]
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        agent_end_events = [e for e in events if e.event_type == ExecutionEventType.AGENT_END]
        assert len(agent_end_events) == 1
        assert agent_end_events[0].action.name == strategy.context_manager.stop_tool_name
        assert agent_end_events[0].action.parameters is None
        assert agent_end_events[0].content[0].text == "Reached maximum iterations (2)"
        assert not any(e.event_type == ExecutionEventType.RESPONSE for e in events)
        assert agent.llm_provider.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_llm_error(self):
        """Should handle LLM errors gracefully."""
        strategy = ToolUsePlannerStrategy(
            context_manager=ToolUseContextManager(),
        )
        agent = make_mock_agent()
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(side_effect=Exception("API timeout"))

        context = make_context(
            conversation_history=[
                make_event(ExecutionEventType.USER_INPUT, content_text="Hi"),
            ]
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        event_types = [e.event_type for e in events]
        assert ExecutionEventType.ERROR_OCCURRED in event_types

    @pytest.mark.asyncio
    async def test_malformed_tool_call_args_retries_then_recovers(self):
        """A malformed tool-call arguments JSON should trigger a verbatim retry."""
        strategy = ToolUsePlannerStrategy(
            context_manager=ToolUseContextManager(),
        )
        tool = make_mock_tool("search")
        agent = make_mock_agent(tools=[tool])
        agent.llm_provider = AsyncMock()

        malformed_response = LLMResponse(
            content="",
            generated_tokens=5,
            execution_time=0.1,
            tool_calls=[
                LLMToolCall(
                    id="call_1",
                    function=ToolCallFunction(name="search", arguments='{"query": "test"'),
                )
            ],
        )
        valid_response = LLMResponse(
            content="",
            generated_tokens=5,
            execution_time=0.1,
            tool_calls=[
                LLMToolCall(
                    id="call_1",
                    function=ToolCallFunction(name="search", arguments='{"query": "test"}'),
                )
            ],
        )
        # A 3rd, terminal response for the next planner iteration (which runs
        # after the tool call from `valid_response` is processed) — otherwise
        # the outer loop would call generate() again with nothing left to return.
        final_response = LLMResponse(
            content="Here are the results.",
            generated_tokens=5,
            execution_time=0.1,
            tool_calls=[],
        )
        agent.llm_provider.generate = AsyncMock(
            side_effect=[malformed_response, valid_response, final_response]
        )

        context = make_context(
            conversation_history=[
                make_event(ExecutionEventType.USER_INPUT, content_text="Hi"),
            ]
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        assert agent.llm_provider.generate.call_count == 3
        # Verbatim resend: the same `messages` list on both attempts, no injected
        # corrective message (unlike the empty-response retry path).
        first_call_messages = agent.llm_provider.generate.call_args_list[0].kwargs["messages"]
        second_call_messages = agent.llm_provider.generate.call_args_list[1].kwargs["messages"]
        assert first_call_messages == second_call_messages

        event_types = [e.event_type for e in events]
        assert ExecutionEventType.ERROR_OCCURRED not in event_types
        tool_start = next(e for e in events if e.event_type == ExecutionEventType.TOOL_START)
        assert tool_start.action.parameters == {"query": "test"}

    @pytest.mark.asyncio
    async def test_malformed_tool_call_args_gives_up_after_max_retries(self):
        """Persistent malformed JSON should error out after MAX_TOOL_CALL_ARGS_RETRIES."""
        strategy = ToolUsePlannerStrategy(
            context_manager=ToolUseContextManager(),
        )
        tool = make_mock_tool("search")
        agent = make_mock_agent(tools=[tool])
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            return_value=LLMResponse(
                content="",
                generated_tokens=5,
                execution_time=0.1,
                tool_calls=[
                    LLMToolCall(
                        id="call_1",
                        function=ToolCallFunction(name="search", arguments='{"query": "test"'),
                    )
                ],
            )
        )

        context = make_context(
            conversation_history=[
                make_event(ExecutionEventType.USER_INPUT, content_text="Hi"),
            ]
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        assert agent.llm_provider.generate.call_count == 1 + MAX_TOOL_CALL_ARGS_RETRIES
        error_events = [e for e in events if e.event_type == ExecutionEventType.ERROR_OCCURRED]
        assert len(error_events) == 1
        assert not any(e.event_type == ExecutionEventType.TOOL_START for e in events)
        # The item-3 fix: an early break from the outer loop must not also
        # trigger the unrelated "Reached maximum iterations" fallback.
        assert not any(e.event_type == ExecutionEventType.RESPONSE for e in events)

    @pytest.mark.asyncio
    async def test_nostream_content_plus_tool_calls_emits_partial_response(self):
        """Non-streaming: when an LLM turn has both content and tool calls,
        the preamble content never reached the consumer through
        ``RESPONSE_TOKEN`` (no streaming pipeline ran), so the planner must
        emit it as a single ``RESPONSE(is_partial=True)`` before the
        ``TOOL_START`` event. The content must still NOT be reclassified as
        internal thought on the tool event.

        Regression guard: dropping this emission on the non-streaming path
        silently loses any user-visible preamble the model returns alongside
        a tool call.
        """
        tool = make_mock_tool("search")
        strategy = ToolUsePlannerStrategy(context_manager=ToolUseContextManager())
        agent = make_mock_agent(tools=[tool])
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            side_effect=[
                LLMResponse(
                    content="Let me search for that.",
                    generated_tokens=5,
                    execution_time=0.1,
                    tool_calls=[
                        LLMToolCall(
                            id="call_1",
                            function=ToolCallFunction(name="search", arguments='{"query": "x"}'),
                        )
                    ],
                ),
                LLMResponse(content="Done.", generated_tokens=2, execution_time=0.1, tool_calls=[]),
            ]
        )
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Search")]
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        tool_start_idx = next(
            i for i, e in enumerate(events) if e.event_type == ExecutionEventType.TOOL_START
        )
        events_before_tool = events[:tool_start_idx]
        responses_before_tool = [
            e for e in events_before_tool if e.event_type == ExecutionEventType.RESPONSE
        ]
        assert len(responses_before_tool) == 1, (
            f"Exactly one partial RESPONSE must precede TOOL_START on a "
            f"non-streaming content+tool_calls turn; found {len(responses_before_tool)}"
        )
        preamble = responses_before_tool[0]
        assert preamble.is_partial is True
        assert preamble.content[0].text == "Let me search for that."
        tool_start = events[tool_start_idx]
        assert tool_start.action.thought is None
        assert tool_start.action.message is None

    @pytest.mark.asyncio
    async def test_stream_true_content_only_tokens_flushed_contiguously(self):
        """Streaming, content-only turn: every chunk's content lands in a
        ``RESPONSE_TOKEN`` event; the emitted tokens form a contiguous block
        (no foreign event types between them) and arrive after ``LLM_START``.

        Buffer-then-decide invariant: while the provider stream is consumed
        the planner accumulates content; once the stream ends without
        ``tool_calls``, every accumulated chunk is flushed in order. Pinning
        contiguity prevents regressions where flushing got interleaved with
        other planner work (and shifts the partial vs final RESPONSE
        boundary).
        """
        strategy = ToolUsePlannerStrategy(context_manager=ToolUseContextManager())
        agent = make_mock_agent()
        agent.stream = True
        agent.llm_provider = AsyncMock()

        async def fake_streaming(*args, **kwargs):
            yield LLMResponse(content="Hello ", generated_tokens=1, execution_time=0.0)
            yield LLMResponse(content="world", generated_tokens=2, execution_time=0.0)

        agent.llm_provider.generate_streaming = fake_streaming
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Hi")]
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        token_indexes = [
            i for i, e in enumerate(events) if e.event_type == ExecutionEventType.RESPONSE_TOKEN
        ]
        assert len(token_indexes) == 2
        assert token_indexes == list(range(token_indexes[0], token_indexes[-1] + 1)), (
            "RESPONSE_TOKEN events must be flushed contiguously, with no other "
            "event types interleaved between them."
        )
        llm_start_idx = next(
            i for i, e in enumerate(events) if e.event_type == ExecutionEventType.LLM_START
        )
        llm_end_idx = next(
            i for i, e in enumerate(events) if e.event_type == ExecutionEventType.LLM_END
        )
        assert all(i > llm_start_idx for i in token_indexes)
        assert all(i < llm_end_idx for i in token_indexes)
        token_texts = [events[i].content[0].text for i in token_indexes]
        assert token_texts == ["Hello ", "world"]
        terminal_response = next(
            e
            for e in events
            if e.event_type == ExecutionEventType.RESPONSE and e.is_partial is False
        )
        assert terminal_response.content[0].text == "Hello world", (
            "Terminal RESPONSE on streaming path must carry the canonical full text "
            "for history/observability; the runner suppresses it from live output."
        )

    @pytest.mark.asyncio
    async def test_stream_true_content_plus_tool_calls_streams_preamble_without_thought(self):
        """Streaming + tool calls: content chunks before ``tool_calls`` are
        user-visible preamble tokens on the plain tool_use path, not internal
        thought on the resulting ``TOOL_START``.
        """
        tool = make_mock_tool("search")
        strategy = ToolUsePlannerStrategy(context_manager=ToolUseContextManager())
        agent = make_mock_agent(tools=[tool])
        agent.stream = True
        agent.llm_provider = AsyncMock()

        stream_calls = 0

        async def fake_streaming(*args, **kwargs):
            nonlocal stream_calls
            stream_calls += 1
            if stream_calls == 1:
                yield LLMResponse(content="Reasoning ", generated_tokens=1, execution_time=0.0)
                yield LLMResponse(content="step ", generated_tokens=1, execution_time=0.0)
                yield LLMResponse(content="one. ", generated_tokens=1, execution_time=0.0)
                yield LLMResponse(
                    content="",
                    generated_tokens=1,
                    execution_time=0.0,
                    tool_calls=[
                        LLMToolCall(
                            id="c_split",
                            function=ToolCallFunction(name="search", arguments='{"query": "x"}'),
                        )
                    ],
                )
            else:
                yield LLMResponse(content="Done", generated_tokens=1, execution_time=0.0)

        agent.llm_provider.generate_streaming = fake_streaming
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Go")]
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        tool_start = next(e for e in events if e.event_type == ExecutionEventType.TOOL_START)
        assert tool_start.action.thought is None
        assert tool_start.action.message is None

        token_texts = [
            e.content[0].text
            for e in events
            if e.event_type == ExecutionEventType.RESPONSE_TOKEN
            and e.content[0].text in {"Reasoning ", "step ", "one. "}
        ]
        assert token_texts == ["Reasoning ", "step ", "one. "]

        tool_start_idx = next(
            i for i, e in enumerate(events) if e.event_type == ExecutionEventType.TOOL_START
        )
        responses_before_tool = [
            e for e in events[:tool_start_idx] if e.event_type == ExecutionEventType.RESPONSE
        ]
        assert responses_before_tool == [], (
            "Streaming path must not duplicate the preamble as a partial "
            "RESPONSE; tokens already flowed via RESPONSE_TOKEN"
        )


# --- FORWARD_VERBATIM tests ---


class TestResolveVerbatimAgentEndFallback:
    """Regression tests for the AGENT_END → content fallback in
    ``resolve_verbatim``.

    Sub-agents terminating via STOP emit AGENT_END events whose ``action``
    has ``observation=None`` and the forwarded payload in ``content``.
    These tests lock in that the fallback returns the content text and not
    ``str(None)``.
    """

    def test_agent_end_with_content_is_forwarded(self):
        agent_end = make_event(
            ExecutionEventType.AGENT_END,
            author="sub_agent",
            event_id="evt_sub",
            content_text="sub-agent verbatim output",
            action=make_tool_action(name="STOP", parameters=None, observation=None),
        )
        context = make_context(interaction_history=[agent_end])

        text, found = resolve_verbatim(context, source_id="evt_sub")

        assert found is True
        assert text == "sub-agent verbatim output"

    def test_agent_end_multi_part_content_is_joined(self):
        """An AGENT_END with several content parts forwards the concatenation of
        all text parts, not just the first."""
        agent_end = BaseEvent(
            event_type=ExecutionEventType.AGENT_END,
            author="sub_agent",
            event_id="evt_multi",
            timestamp=datetime.now(),
            content=[Part(text="part one "), Part(text="part two")],
            action=make_tool_action(name="STOP", parameters=None, observation=None),
        )
        context = make_context(interaction_history=[agent_end])

        text, found = resolve_verbatim(context, source_id="evt_multi")

        assert found is True
        assert text == "part one part two"

    def test_agent_end_without_content_returns_not_found(self):
        agent_end = make_event(
            ExecutionEventType.AGENT_END,
            author="sub_agent",
            event_id="evt_empty",
            action=make_tool_action(name="STOP", parameters=None, observation=None),
        )
        context = make_context(interaction_history=[agent_end])

        text, found = resolve_verbatim(context, source_id="evt_empty")

        assert found is False
        assert text == ""

    def test_tool_end_with_observation_is_forwarded(self):
        """Baseline: TOOL_END path (observation set) must still work after the
        AGENT_END fallback refactor."""
        tool_end = make_event(
            ExecutionEventType.TOOL_END,
            event_id="evt_tool",
            action=make_tool_action(name="search", observation="tool output"),
        )
        context = make_context(interaction_history=[tool_end])

        text, found = resolve_verbatim(context, source_id="evt_tool")

        assert found is True
        assert text == "tool output"

    def test_agent_end_with_observation_set_takes_precedence(self):
        """If an AGENT_END happens to have observation set, it is used directly
        (the fallback to content only triggers when observation is None)."""
        agent_end = make_event(
            ExecutionEventType.AGENT_END,
            event_id="evt_direct",
            content_text="should not be used",
            action=make_tool_action(
                name="STOP",
                parameters=None,
                observation="direct observation",
            ),
        )
        context = make_context(interaction_history=[agent_end])

        text, found = resolve_verbatim(context, source_id="evt_direct")

        assert found is True
        assert text == "direct observation"

    def test_source_id_selects_correct_event_among_multiple(self):
        """With multiple events, source_id matches the correct one by event_id."""
        first = make_event(
            ExecutionEventType.TOOL_END,
            event_id="evt_first",
            action=make_tool_action(name="search", observation="old result"),
        )
        second = make_event(
            ExecutionEventType.TOOL_END,
            event_id="evt_second",
            action=make_tool_action(name="search", observation="most recent result"),
        )
        context = make_context(interaction_history=[first, second])

        text, found = resolve_verbatim(context, source_id="evt_first")
        assert found is True
        assert text == "old result"

        text2, found2 = resolve_verbatim(context, source_id="evt_second")
        assert found2 is True
        assert text2 == "most recent result"


class TestResolveVerbatimLogging:
    """Logging on every failure / success path of ``resolve_verbatim``."""

    def test_warning_when_source_id_not_found(self, caplog):
        context = make_context(interaction_history=[])

        with caplog.at_level("WARNING", logger="domyn.agents"):
            text, found = resolve_verbatim(context, source_id="missing_call_id")

        assert found is False
        assert text == ""
        assert any(
            "forward_verbatim" in r.message and "missing_call_id" in r.message
            for r in caplog.records
        )

    def test_warning_when_observation_is_none(self, caplog):
        tool_end = make_event(
            ExecutionEventType.TOOL_END,
            event_id="evt_none",
            action=make_tool_action(name="search", observation=None),
        )
        context = make_context(interaction_history=[tool_end])

        with caplog.at_level("WARNING", logger="domyn.agents"):
            text, found = resolve_verbatim(context, source_id="evt_none")

        assert found is False
        assert text == ""
        assert any("observation is None" in r.message for r in caplog.records)

    def test_warning_when_source_key_does_not_resolve(self, caplog):
        tool_end = make_event(
            ExecutionEventType.TOOL_END,
            event_id="evt_key",
            action=make_tool_action(name="search", observation={"result": "ok"}),
        )
        context = make_context(interaction_history=[tool_end])

        with caplog.at_level("WARNING", logger="domyn.agents"):
            text, found = resolve_verbatim(context, source_id="evt_key", source_key="missing.path")

        assert found is False
        assert text == ""
        assert any(
            "source_key='missing.path'" in r.message and "did not resolve" in r.message
            for r in caplog.records
        )

    def test_info_on_successful_forward(self, caplog):
        tool_end = make_event(
            ExecutionEventType.TOOL_END,
            event_id="evt-success",
            action=make_tool_action(name="search", observation="final answer"),
        )
        context = make_context(interaction_history=[tool_end])

        with caplog.at_level("INFO", logger="domyn.agents"):
            text, found = resolve_verbatim(context, source_id="evt-success")

        assert found is True
        assert text == "final answer"
        assert any(
            "forwarding observation" in r.message and "evt-success" in r.message
            for r in caplog.records
        )


# --- LLMMessage / LLMResponse backward compatibility ---


class TestForwardVerbatimDispatch:
    """End-to-end dispatch tests for FORWARD_VERBATIM in execute().

    These tests verify the full path: LLM returns a FORWARD_VERBATIM tool
    call → resolve_verbatim looks up the observation → a RESPONSE event
    is yielded with METADATA_FORWARD_VERBATIM.
    """

    def _forward_tool_call(self, source_id=None, source_key=None, call_id="call_fwd"):
        args = {}
        if source_id is not None:
            args["source_id"] = source_id
        if source_key is not None:
            args["source_key"] = source_key
        return LLMToolCall(
            id=call_id,
            function=ToolCallFunction(
                name="FORWARD_VERBATIM",
                arguments=json.dumps(args),
            ),
        )

    @pytest.mark.asyncio
    async def test_forward_tool_observation_by_source_id(self):
        """LLM calls FORWARD_VERBATIM with source_id matching a TOOL_END's event_id
        → RESPONSE carries the observation and METADATA_FORWARD_VERBATIM."""
        strategy = ToolUsePlannerStrategy()
        agent = make_mock_agent()
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            return_value=LLMResponse(
                content="",
                generated_tokens=5,
                execution_time=0.1,
                tool_calls=[self._forward_tool_call(source_id="evt_search_end")],
            )
        )

        tool_end = make_event(
            ExecutionEventType.TOOL_END,
            event_id="evt_search_end",
            action=make_tool_action(
                name="search",
                call_id="call_search",
                observation="## Search Results\n\nFound 3 items.",
            ),
        )
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Search")],
            interaction_history=[tool_end],
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        response = next(e for e in events if e.event_type == ExecutionEventType.RESPONSE)
        assert response.content[0].text == "## Search Results\n\nFound 3 items."
        assert response.metadata.get(METADATA_FORWARD_VERBATIM) is True
        assert response.is_partial is False

    @pytest.mark.asyncio
    async def test_forward_not_found_exhausts_retries_then_emits_error_occurred(self):
        """When source_id doesn't match, the planner offers the LLM a
        bounded number of soft-miss retries followed by a single
        last-ditch hard-fallback nudge before falling back to
        ``ERROR_OCCURRED``. Each soft retry replays the LLM with a
        tool-role diagnostic appended; the hard-fallback message tells
        the LLM to abandon FORWARD_VERBATIM and respond directly. If
        the LLM still emits FV after that, the planner terminates with
        the legacy hard-miss event."""
        from domyn.agents.planner_strategy.tool_use import MAX_FV_SOFT_MISS_RETRIES

        strategy = ToolUsePlannerStrategy()
        agent = make_mock_agent()
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            return_value=LLMResponse(
                content="",
                generated_tokens=5,
                execution_time=0.1,
                tool_calls=[self._forward_tool_call(source_id="nonexistent_id")],
            )
        )
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Hi")],
            interaction_history=[],
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        # No RESPONSE event: planner exhausts retries then terminates on hard miss.
        response_events = [e for e in events if e.event_type == ExecutionEventType.RESPONSE]
        assert len(response_events) == 0
        # One AGENT_END carrying a generic user-facing agent error in
        # error_code/error_message. SDK consumers (runner / platform)
        # decide whether to surface as graceful response, retry, or hard error.
        agent_ends = [e for e in events if e.event_type == ExecutionEventType.AGENT_END]
        assert len(agent_ends) == 1
        assert agent_ends[0].error_message == AGENT_ERROR_MESSAGE
        assert agent_ends[0].error_code == AGENT_ERROR_CODE
        # No legacy ERROR_OCCURRED — top-level hard miss now uses AGENT_END.
        legacy_errors = [e for e in events if e.event_type == ExecutionEventType.ERROR_OCCURRED]
        assert len(legacy_errors) == 0
        # No TOOL_ERROR — FORWARD_VERBATIM is a special action, not a tool.
        tool_errors = [e for e in events if e.event_type == ExecutionEventType.TOOL_ERROR]
        assert len(tool_errors) == 0
        # LLM call count: 1 initial + MAX_FV_SOFT_MISS_RETRIES soft retries
        # + 1 hard-fallback retry (last-ditch nudge to abandon FV).
        assert agent.llm_provider.generate.call_count == 1 + MAX_FV_SOFT_MISS_RETRIES + 1

    @pytest.mark.asyncio
    async def test_hard_fallback_recovers_when_llm_responds_directly(self):
        """After the soft-miss budget is spent, the planner injects a
        last-ditch fallback message asking the LLM to abandon
        FORWARD_VERBATIM and respond directly. If the LLM honours the
        nudge (returns plain content with no tool_calls), the planner
        emits a normal RESPONSE and the turn ends without ERROR_OCCURRED."""
        from domyn.agents.planner_strategy.tool_use import MAX_FV_SOFT_MISS_RETRIES

        strategy = ToolUsePlannerStrategy()
        agent = make_mock_agent()
        agent.llm_provider = AsyncMock()
        # 1 initial + MAX_FV_SOFT_MISS_RETRIES soft retries: all FV miss.
        # Then on the hard-fallback retry, LLM returns plain content (no tool_calls).
        miss_response = LLMResponse(
            content="",
            generated_tokens=5,
            execution_time=0.1,
            tool_calls=[self._forward_tool_call(source_id="nonexistent_id")],
        )
        recovery_response = LLMResponse(
            content="Sorry, I couldn't fetch the source. Here's a brief answer instead.",
            generated_tokens=12,
            execution_time=0.1,
            tool_calls=[],
        )
        agent.llm_provider.generate = AsyncMock(
            side_effect=[miss_response] * (1 + MAX_FV_SOFT_MISS_RETRIES) + [recovery_response]
        )
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Hi")],
            interaction_history=[],
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        # No ERROR_OCCURRED: the hard-fallback nudge worked.
        errors = [e for e in events if e.event_type == ExecutionEventType.ERROR_OCCURRED]
        assert len(errors) == 0
        # Exactly one terminal RESPONSE with the recovery content.
        responses = [e for e in events if e.event_type == ExecutionEventType.RESPONSE]
        assert len(responses) == 1
        assert "Here's a brief answer" in responses[0].content[0].text
        # 1 initial + MAX_FV_SOFT_MISS_RETRIES soft + 1 hard-fallback = total LLM calls.
        assert agent.llm_provider.generate.call_count == 1 + MAX_FV_SOFT_MISS_RETRIES + 1

    @pytest.mark.asyncio
    async def test_forward_with_source_key_extracts_nested(self):
        """source_key navigates into the observation dict via dot-path."""
        strategy = ToolUsePlannerStrategy()
        agent = make_mock_agent()
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            return_value=LLMResponse(
                content="",
                generated_tokens=5,
                execution_time=0.1,
                tool_calls=[
                    self._forward_tool_call(source_id="evt_data_end", source_key="result.summary")
                ],
            )
        )

        obs = {"status": "ok", "result": {"summary": "## Revenue Report", "data": [1, 2, 3]}}
        tool_end = make_event(
            ExecutionEventType.TOOL_END,
            event_id="evt_data_end",
            action=make_tool_action(name="query", call_id="call_data", observation=obs),
        )
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Report")],
            interaction_history=[tool_end],
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        response = next(e for e in events if e.event_type == ExecutionEventType.RESPONSE)
        assert response.content[0].text == "## Revenue Report"
        assert response.metadata.get(METADATA_FORWARD_VERBATIM) is True

    @pytest.mark.asyncio
    async def test_forward_agent_end_via_content(self):
        """FORWARD_VERBATIM resolves AGENT_END events via content
        (the path used when sub-agents STOP)."""
        strategy = ToolUsePlannerStrategy()
        agent = make_mock_agent()
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            return_value=LLMResponse(
                content="",
                generated_tokens=5,
                execution_time=0.1,
                tool_calls=[self._forward_tool_call(source_id="evt_sub")],
            )
        )

        agent_end = make_event(
            ExecutionEventType.AGENT_END,
            author="sub_agent",
            event_id="evt_sub",
            content_text="Sub-agent analysis complete.",
            action=make_tool_action(name="STOP", parameters=None, observation=None),
        )
        context = make_context(
            conversation_history=[
                make_event(ExecutionEventType.USER_INPUT, content_text="Analyze")
            ],
            interaction_history=[agent_end],
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        response = next(e for e in events if e.event_type == ExecutionEventType.RESPONSE)
        assert response.content[0].text == "Sub-agent analysis complete."
        assert response.metadata.get(METADATA_FORWARD_VERBATIM) is True

    @pytest.mark.asyncio
    async def test_forward_terminates_loop(self):
        """After FORWARD_VERBATIM the loop must return — LLM is called only once."""
        strategy = ToolUsePlannerStrategy()
        agent = make_mock_agent()
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            return_value=LLMResponse(
                content="",
                generated_tokens=5,
                execution_time=0.1,
                tool_calls=[self._forward_tool_call(source_id="evt_x")],
            )
        )

        tool_end = make_event(
            ExecutionEventType.TOOL_END,
            event_id="evt_x",
            action=make_tool_action(name="search", call_id="call_x", observation="result"),
        )
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Go")],
            interaction_history=[tool_end],
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        assert agent.llm_provider.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_forward_dict_observation_is_stringified_whole(self):
        """When no source_key and observation is a dict, the full dict is
        JSON-stringified — no implicit auto-extraction of 'result'."""
        strategy = ToolUsePlannerStrategy()
        agent = make_mock_agent()
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            return_value=LLMResponse(
                content="",
                generated_tokens=5,
                execution_time=0.1,
                tool_calls=[self._forward_tool_call(source_id="evt_q")],
            )
        )

        obs = {"result": "## Key Findings\n\nRevenue grew 15%.", "metadata": {"id": "abc"}}
        tool_end = make_event(
            ExecutionEventType.TOOL_END,
            event_id="evt_q",
            action=make_tool_action(name="query", call_id="call_q", observation=obs),
        )
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Report")],
            interaction_history=[tool_end],
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        response = next(e for e in events if e.event_type == ExecutionEventType.RESPONSE)
        forwarded = json.loads(response.content[0].text)
        assert forwarded["result"] == "## Key Findings\n\nRevenue grew 15%."
        assert forwarded["metadata"] == {"id": "abc"}
        assert response.metadata.get(METADATA_FORWARD_VERBATIM) is True


class TestForwardVerbatimMiss:
    """Tests for FORWARD_VERBATIM miss handling.

    Top-level agents (``use_stop=False``): the planner injects a tool-role
    diagnostic and re-iterates the LLM up to ``MAX_FV_SOFT_MISS_RETRIES``
    times so the model can recover; once the budget is exhausted, the
    planner yields ``ERROR_OCCURRED``. The retry carry lives outside the
    event log (assistant ``tool_calls`` + paired ``tool`` reply) so the
    native tool-use invariant holds without polluting history.

    Sub-agents (``use_stop=True``): no soft retry — the planner yields
    ``AGENT_END`` with the STOP envelope so the parent decides recovery."""

    def _forward_tool_call(self, source_id=None, call_id="call_fwd"):
        return LLMToolCall(
            id=call_id,
            function=ToolCallFunction(
                name="FORWARD_VERBATIM",
                arguments=json.dumps({"source_id": source_id} if source_id else {}),
            ),
        )

    @pytest.mark.asyncio
    async def test_miss_replays_lllm_until_retries_exhausted(self):
        """Miss path issues ``MAX_FV_SOFT_MISS_RETRIES`` retries plus one
        last-ditch hard-fallback retry before terminating. Each replay
        brackets one LLM call with its own ``LLM_START``/``LLM_END`` pair;
        if the LLM keeps emitting the same miss, the planner finally
        yields ``ERROR_OCCURRED``."""
        from domyn.agents.planner_strategy.tool_use import MAX_FV_SOFT_MISS_RETRIES

        strategy = ToolUsePlannerStrategy()
        agent = make_mock_agent()
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            return_value=LLMResponse(
                content="",
                generated_tokens=5,
                execution_time=0.1,
                tool_calls=[self._forward_tool_call(source_id="bad_id")],
            )
        )
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Hi")],
            interaction_history=[],
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        # 1 initial + MAX_FV_SOFT_MISS_RETRIES soft retries + 1 hard-fallback
        expected_calls = 1 + MAX_FV_SOFT_MISS_RETRIES + 1
        event_types = [e.event_type for e in events]
        assert event_types.count(ExecutionEventType.LLM_START) == expected_calls
        assert event_types.count(ExecutionEventType.LLM_END) == expected_calls
        assert agent.llm_provider.generate.call_count == expected_calls

    @pytest.mark.asyncio
    async def test_miss_emits_generic_agent_error(self):
        """Miss emits a single ERROR_OCCURRED event (no TOOL_START, TOOL_END
        or TOOL_ERROR) whose ``error_message`` stays user-facing and does not
        expose planner internals."""
        strategy = ToolUsePlannerStrategy()
        agent = make_mock_agent()
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            return_value=LLMResponse(
                content="",
                generated_tokens=5,
                execution_time=0.1,
                tool_calls=[self._forward_tool_call(source_id="missing_id")],
            )
        )
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Hi")],
            interaction_history=[],
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        tool_starts = [e for e in events if e.event_type == ExecutionEventType.TOOL_START]
        tool_ends = [e for e in events if e.event_type == ExecutionEventType.TOOL_END]
        tool_errors = [e for e in events if e.event_type == ExecutionEventType.TOOL_ERROR]
        # Hard miss now surfaces as AGENT_END with error_code/error_message
        # populated; SDK consumers decide UX. Legacy ERROR_OCCURRED stays
        # reserved for executive-level failures (LLM provider errors, etc.).
        agent_ends = [e for e in events if e.event_type == ExecutionEventType.AGENT_END]
        legacy_errors = [e for e in events if e.event_type == ExecutionEventType.ERROR_OCCURRED]
        assert len(tool_starts) == 0
        assert len(tool_ends) == 0
        assert len(tool_errors) == 0
        assert len(legacy_errors) == 0
        assert len(agent_ends) == 1
        assert agent_ends[0].error_message == AGENT_ERROR_MESSAGE
        assert agent_ends[0].error_code == AGENT_ERROR_CODE


class TestForwardVerbatimSoftMissRecovery:
    """Soft-miss path: when ``FORWARD_VERBATIM`` misses on a top-level agent,
    the planner injects a tool-role diagnostic and re-iterates so the LLM
    can recover. Only after retries are exhausted does the planner emit
    ``ERROR_OCCURRED``. Validates the four miss reasons surface different
    LLM-facing feedback strings, and that the LLM actually receives the
    feedback as a tool reply paired with the failed tool_call assistant
    message (native tool-use invariant).
    """

    @staticmethod
    def _fv_call(source_id, source_key=None, call_id="call_fwd"):
        args = {"source_id": source_id} if source_id is not None else {}
        if source_key is not None:
            args["source_key"] = source_key
        return LLMToolCall(
            id=call_id,
            function=ToolCallFunction(
                name="FORWARD_VERBATIM",
                arguments=json.dumps(args),
            ),
        )

    @pytest.mark.asyncio
    async def test_recovery_via_plain_response_after_one_soft_miss(self):
        """LLM hallucinates a phrase-shaped source_id, sees the format
        diagnostic via tool reply, then responds with plain content. The
        planner must yield the recovered RESPONSE, no ERROR_OCCURRED."""
        strategy = ToolUsePlannerStrategy()
        agent = make_mock_agent()
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            side_effect=[
                LLMResponse(
                    content="",
                    generated_tokens=5,
                    execution_time=0.1,
                    tool_calls=[self._fv_call(source_id="Agent: Worker ended.")],
                ),
                LLMResponse(
                    content="Recovered answer in my own words.",
                    generated_tokens=5,
                    execution_time=0.1,
                    tool_calls=[],
                ),
            ]
        )
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Hi")],
            interaction_history=[],
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        responses = [e for e in events if e.event_type == ExecutionEventType.RESPONSE]
        assert len(responses) == 1
        assert responses[0].content[0].text == "Recovered answer in my own words."
        assert responses[0].metadata.get(METADATA_FORWARD_VERBATIM) is None
        assert not any(e.event_type == ExecutionEventType.ERROR_OCCURRED for e in events)
        assert agent.llm_provider.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_invalid_format_feedback_reaches_llm_in_retry(self):
        """The phrase-shaped source_id triggers the format-invalid feedback.
        Inspect the messages passed to the second LLM call: it must contain
        the assistant tool_call + tool-role reply with the invalid-format
        diagnostic — that is the channel the LLM uses to recover."""
        strategy = ToolUsePlannerStrategy()
        agent = make_mock_agent()
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            side_effect=[
                LLMResponse(
                    content="",
                    generated_tokens=5,
                    execution_time=0.1,
                    tool_calls=[self._fv_call(source_id="Agent: Worker ended.")],
                ),
                LLMResponse(
                    content="recovered",
                    generated_tokens=5,
                    execution_time=0.1,
                    tool_calls=[],
                ),
            ]
        )
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Hi")],
            interaction_history=[],
        )

        async for _ in strategy.execute(agent, context):
            pass

        # Second call carries the FV miss exchange in its messages.
        second_call = agent.llm_provider.generate.call_args_list[1]
        messages = second_call.kwargs["messages"]
        assistant_fv_calls = [m for m in messages if m.role == "assistant" and m.tool_calls]
        tool_replies = [m for m in messages if m.role == "tool"]
        assert assistant_fv_calls, "second LLM call must see the prior FV tool_call"
        assert tool_replies, "second LLM call must see a tool reply for the FV call"
        # Format-invalid diagnostic + reference to the bracketed-event_id format.
        assert any("not a valid event_id" in m.content for m in tool_replies)
        assert any("[event_id:" in m.content for m in tool_replies)

    @pytest.mark.asyncio
    async def test_not_found_feedback_when_format_valid_but_history_misses(self):
        """source_id has identifier shape but doesn't match any event in
        history → distinct ``not_found`` feedback (different from format-
        invalid). LLM still gets a chance to recover."""
        from domyn.agents.planner_strategy.utils import (
            FORWARD_VERBATIM_FEEDBACK_NOT_FOUND,
        )

        strategy = ToolUsePlannerStrategy()
        agent = make_mock_agent()
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            side_effect=[
                LLMResponse(
                    content="",
                    generated_tokens=5,
                    execution_time=0.1,
                    tool_calls=[self._fv_call(source_id="event_does_not_exist")],
                ),
                LLMResponse(
                    content="answered without forwarding",
                    generated_tokens=5,
                    execution_time=0.1,
                    tool_calls=[],
                ),
            ]
        )
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Hi")],
            interaction_history=[],
        )

        async for _ in strategy.execute(agent, context):
            pass

        second_call = agent.llm_provider.generate.call_args_list[1]
        messages = second_call.kwargs["messages"]
        tool_replies = [m for m in messages if m.role == "tool"]
        expected = FORWARD_VERBATIM_FEEDBACK_NOT_FOUND.format(source_id="event_does_not_exist")
        assert any(m.content == expected for m in tool_replies)

    @pytest.mark.asyncio
    async def test_agent_end_source_key_feedback_specific(self):
        """When source_key is set on an AGENT_END source the planner emits
        the AGENT_END-specific feedback (not the generic not-found message),
        guiding the LLM to either drop source_key or pick a TOOL_END source."""
        from domyn.agents.planner_strategy.utils import (
            FORWARD_VERBATIM_FEEDBACK_AGENT_END_SOURCE_KEY,
        )

        agent_end = make_event(
            ExecutionEventType.AGENT_END,
            event_id="evt_subagent_end",
            action=ToolAction(name="STOP", parameters={"info": "sub answer"}),
        )

        strategy = ToolUsePlannerStrategy()
        agent = make_mock_agent()
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            side_effect=[
                LLMResponse(
                    content="",
                    generated_tokens=5,
                    execution_time=0.1,
                    tool_calls=[self._fv_call(source_id="evt_subagent_end", source_key="info")],
                ),
                LLMResponse(
                    content="ok, retrying without source_key",
                    generated_tokens=5,
                    execution_time=0.1,
                    tool_calls=[],
                ),
            ]
        )
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Hi")],
            interaction_history=[agent_end],
        )

        async for _ in strategy.execute(agent, context):
            pass

        second_call = agent.llm_provider.generate.call_args_list[1]
        messages = second_call.kwargs["messages"]
        tool_replies = [m for m in messages if m.role == "tool"]
        expected = FORWARD_VERBATIM_FEEDBACK_AGENT_END_SOURCE_KEY.format(source_key="info")
        assert any(m.content == expected for m in tool_replies)

    @pytest.mark.asyncio
    async def test_source_key_path_miss_feedback_does_not_blame_event_id(self):
        """If source_id resolves but source_key misses, retry feedback must not
        claim the event_id was wrong."""
        strategy = ToolUsePlannerStrategy()
        agent = make_mock_agent()
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            side_effect=[
                LLMResponse(
                    content="",
                    generated_tokens=5,
                    execution_time=0.1,
                    tool_calls=[
                        self._fv_call(
                            source_id="evt_vendor",
                            source_key="narrative.z9.slot_02",
                        )
                    ],
                ),
                LLMResponse(
                    content="recovered",
                    generated_tokens=5,
                    execution_time=0.1,
                    tool_calls=[],
                ),
            ]
        )
        tool_end = make_event(
            ExecutionEventType.TOOL_END,
            event_id="evt_vendor",
            action=make_tool_action(
                name="vendor",
                observation={
                    "bundle": {
                        "narrative": {
                            "z9": {
                                "slot_02": "## Vendor answer\n\nReady to show.",
                            }
                        }
                    },
                    "metadata": {"run_id": "r1"},
                },
            ),
        )
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Hi")],
            interaction_history=[tool_end],
        )

        async for _ in strategy.execute(agent, context):
            pass

        second_call = agent.llm_provider.generate.call_args_list[1]
        messages = second_call.kwargs["messages"]
        tool_replies = [m.content for m in messages if m.role == "tool"]

        assert any("source_id='evt_vendor' resolved correctly" in r for r in tool_replies)
        assert any("source_key='narrative.z9.slot_02' did not resolve" in r for r in tool_replies)
        assert not any("did not match any event_id" in r for r in tool_replies)
        assert not any("Do not change source_id" in r for r in tool_replies)
        assert not any("Observation key tree" in r for r in tool_replies)
        assert not any("Candidate leaf paths" in r for r in tool_replies)

    @pytest.mark.asyncio
    async def test_sub_agent_miss_skips_soft_retry(self):
        """Sub-agents (use_stop=True) do not soft-retry on miss: the parent
        is the recovery point, so the planner emits AGENT_END error
        immediately and only one LLM call happens."""
        strategy = ToolUsePlannerStrategy(use_stop=True)
        agent = make_mock_agent()
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            return_value=LLMResponse(
                content="",
                generated_tokens=5,
                execution_time=0.1,
                tool_calls=[self._fv_call(source_id="missing_sub_id")],
            )
        )
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Hi")],
            interaction_history=[],
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        agent_end = next(e for e in events if e.event_type == ExecutionEventType.AGENT_END)
        assert agent_end.action.name == strategy.context_manager.stop_tool_name
        assert agent_end.content[0].text == AGENT_ERROR_MESSAGE
        assert not any(e.event_type == ExecutionEventType.ERROR_OCCURRED for e in events)
        # Sub-agent path must not soft-retry: exactly one LLM call.
        assert agent.llm_provider.generate.call_count == 1


class TestForwardVerbatimDispatchUseStop:
    """Dispatch tests for FORWARD_VERBATIM on the use_stop=True path
    (sub-agents). Terminal event must be AGENT_END with STOP action
    instead of RESPONSE."""

    def _forward_tool_call(self, source_id=None, source_key=None, call_id="call_fwd"):
        args = {}
        if source_id is not None:
            args["source_id"] = source_id
        if source_key is not None:
            args["source_key"] = source_key
        return LLMToolCall(
            id=call_id,
            function=ToolCallFunction(
                name="FORWARD_VERBATIM",
                arguments=json.dumps(args),
            ),
        )

    @pytest.mark.asyncio
    async def test_found_emits_agent_end_with_stop_action(self):
        """Sub-agent FORWARD_VERBATIM → AGENT_END with STOP(info=text) + skip metadata."""
        strategy = ToolUsePlannerStrategy(use_stop=True)
        agent = make_mock_agent()
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            return_value=LLMResponse(
                content="",
                generated_tokens=5,
                execution_time=0.1,
                tool_calls=[self._forward_tool_call(source_id="evt_search_end")],
            )
        )

        tool_end = make_event(
            ExecutionEventType.TOOL_END,
            event_id="evt_search_end",
            action=make_tool_action(
                name="search",
                call_id="call_search",
                observation="## Results\n\nFound 3 items.",
            ),
        )
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Go")],
            interaction_history=[tool_end],
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        agent_end = next(e for e in events if e.event_type == ExecutionEventType.AGENT_END)
        assert agent_end.action.name == strategy.context_manager.stop_tool_name
        assert agent_end.action.parameters is None
        assert agent_end.content[0].text == "## Results\n\nFound 3 items."
        assert agent_end.metadata.get(METADATA_FORWARD_VERBATIM) is True
        assert agent_end.metadata[METADATA_FORWARD_VERBATIM_SOURCE_ID] == "evt_search_end"
        assert not any(e.event_type == ExecutionEventType.RESPONSE for e in events)

    @pytest.mark.asyncio
    async def test_not_found_returns_error_to_parent_with_agent_end(self):
        """Sub-agent (use_stop=True) path: miss returns control to the parent."""
        strategy = ToolUsePlannerStrategy(use_stop=True)
        agent = make_mock_agent()
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            return_value=LLMResponse(
                content="",
                generated_tokens=5,
                execution_time=0.1,
                tool_calls=[self._forward_tool_call(source_id="nonexistent")],
            )
        )

        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Hi")],
            interaction_history=[],
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        assert not any(e.event_type == ExecutionEventType.ERROR_OCCURRED for e in events)
        agent_end = next(e for e in events if e.event_type == ExecutionEventType.AGENT_END)
        assert agent_end.action.name == strategy.context_manager.stop_tool_name
        assert agent_end.action.parameters is None
        assert agent_end.content[0].text == AGENT_ERROR_MESSAGE
        assert agent_end.metadata.get(METADATA_FORWARD_VERBATIM) is None
        assert agent.llm_provider.generate.call_count == 1


class TestForwardVerbatimDispatchAcrossStrategies:
    """Smoke tests verifying FORWARD_VERBATIM dispatch works identically
    in all three tool-use strategy variants."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "strategy_cls",
        [
            ToolUsePlannerStrategy,
            ToolUseWithThoughtPlannerStrategy,
            # ReactToolUsePlannerStrategy skipped: ReactToolUseContextManager
            # not yet defined in context_management/tool_use.py
        ],
    )
    async def test_forward_verbatim_smoke(self, strategy_cls):
        strategy = strategy_cls(use_stop=False)
        agent = make_mock_agent()
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            return_value=LLMResponse(
                content="",
                generated_tokens=5,
                execution_time=0.1,
                tool_calls=[
                    LLMToolCall(
                        id="call_fwd",
                        function=ToolCallFunction(
                            name="FORWARD_VERBATIM",
                            arguments=json.dumps({"source_id": "evt_tool_end"}),
                        ),
                    )
                ],
            )
        )

        tool_end = make_event(
            ExecutionEventType.TOOL_END,
            event_id="evt_tool_end",
            action=make_tool_action(
                name="search", call_id="call_tool", observation="verbatim content"
            ),
        )
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Go")],
            interaction_history=[tool_end],
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        response = next(e for e in events if e.event_type == ExecutionEventType.RESPONSE)
        assert response.content[0].text == "verbatim content"
        assert response.metadata.get(METADATA_FORWARD_VERBATIM) is True
        assert agent.llm_provider.generate.call_count == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "strategy_cls",
        [
            ToolUsePlannerStrategy,
            ToolUseWithThoughtPlannerStrategy,
        ],
    )
    async def test_forward_verbatim_use_stop_smoke(self, strategy_cls):
        """Sub-agent variant: FORWARD_VERBATIM emits AGENT_END instead of RESPONSE."""
        strategy = strategy_cls(use_stop=True)
        agent = make_mock_agent()
        agent.llm_provider = AsyncMock()
        agent.llm_provider.generate = AsyncMock(
            return_value=LLMResponse(
                content="",
                generated_tokens=5,
                execution_time=0.1,
                tool_calls=[
                    LLMToolCall(
                        id="call_fwd",
                        function=ToolCallFunction(
                            name="FORWARD_VERBATIM",
                            arguments=json.dumps({"source_id": "evt_tool_end"}),
                        ),
                    )
                ],
            )
        )

        tool_end = make_event(
            ExecutionEventType.TOOL_END,
            event_id="evt_tool_end",
            action=make_tool_action(
                name="search", call_id="call_tool", observation="verbatim content"
            ),
        )
        context = make_context(
            conversation_history=[make_event(ExecutionEventType.USER_INPUT, content_text="Go")],
            interaction_history=[tool_end],
        )

        events = []
        async for event in strategy.execute(agent, context):
            events.append(event)

        agent_end = next(e for e in events if e.event_type == ExecutionEventType.AGENT_END)
        assert agent_end.action.name == strategy.context_manager.stop_tool_name
        assert agent_end.action.parameters is None
        assert agent_end.content[0].text == "verbatim content"
        assert agent_end.metadata.get(METADATA_FORWARD_VERBATIM) is True
        assert agent_end.metadata[METADATA_FORWARD_VERBATIM_SOURCE_ID] == "evt_tool_end"
        assert not any(e.event_type == ExecutionEventType.RESPONSE for e in events)


# --- LLMMessage / LLMResponse backward compatibility ---


class TestLLMMessageBackwardCompat:
    def test_tool_call_message(self):
        msg = LLMMessage(
            role="assistant",
            content="Searching...",
            tool_calls=[
                LLMToolCall(
                    id="call_1",
                    function=ToolCallFunction(name="search", arguments='{"q": "test"}'),
                )
            ],
        )
        assert msg.tool_calls[0].function.name == "search"


class TestLLMResponseBackwardCompat:
    def test_response_with_tool_calls(self):
        resp = LLMResponse(
            content="",
            generated_tokens=5,
            execution_time=0.1,
            tool_calls=[
                LLMToolCall(
                    id="tc_1",
                    function=ToolCallFunction(name="test", arguments="{}"),
                )
            ],
        )
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].id == "tc_1"


# --- Factory / Registration ---


class TestToolUseMultiAgentFlow:
    """Tests for ToolUsePlannerStrategy with main agent + two sub-agents.

    These tests verify the sequence of events emitted by the strategy
    for each individual agent in two handoff scenarios.
    """

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _make_agent(self, name: str, use_stop: bool, tools=None) -> MagicMock:
        """Create a mock agent with the given name and use_stop setting."""
        agent = make_mock_agent(tools=tools or [], instruction=f"You are {name}.")
        agent.name = name
        agent.planner = MagicMock()
        agent.planner.use_stop = use_stop
        return agent

    def _make_sub_agent_desc(self, name: str) -> MagicMock:
        desc = MagicMock()
        desc.name = name
        desc.description = f"Sub-agent {name}"
        desc.invocation_parameters = {
            "type": "object",
            "properties": {"task": {"type": "string"}},
            "required": ["task"],
        }
        return desc

    def _make_strategy(self, use_stop: bool) -> ToolUsePlannerStrategy:
        return ToolUsePlannerStrategy(
            context_manager=ToolUseContextManager(),
            use_stop=use_stop,
        )

    def _make_llm_tool_call(self, call_id: str, tool_name: str, arguments: dict) -> LLMToolCall:
        return LLMToolCall(
            id=call_id,
            function=ToolCallFunction(
                name=tool_name,
                arguments=json.dumps(arguments),
            ),
        )

    def _llm_response(self, content: str, tool_calls=None):
        from domyn.agents.llm import LLMResponse

        return LLMResponse(
            content=content,
            generated_tokens=10,
            execution_time=0.1,
            tool_calls=tool_calls or [],
        )

    # ── Test 1: cascata main → sub_A → sub_B → sub_B stops → sub_A stops → main responds ──

    @pytest.mark.asyncio
    async def test_cascade_handoff_flow(self):
        """
        Scenario:
          main_agent  calls sub_agent_A  (AGENT_START, return)
          sub_agent_A calls sub_agent_B  (AGENT_START, return)
          sub_agent_B completes with STOP("info from B")
          sub_agent_A receives B's info then completes with STOP("info from A+B")
          main_agent  receives A's info then emits final RESPOND
        """
        ctx_mgr = ToolUseContextManager()

        # ── main_agent: calls sub_agent_A ─────────────────────────────────────
        main_agent = self._make_agent("main_agent", use_stop=False)
        main_context = make_context(
            conversation_history=[
                make_event(ExecutionEventType.USER_INPUT, content_text="Get info from A then B")
            ]
        )
        sub_a_desc = self._make_sub_agent_desc("sub_agent_A")
        main_context.handoff_targets_description = [sub_a_desc]

        main_strategy = self._make_strategy(use_stop=False)
        main_agent.llm_provider = AsyncMock()

        # first call: handoff to sub_agent_A
        main_agent.llm_provider.generate = AsyncMock(
            return_value=self._llm_response(
                content="I'll delegate to sub_agent_A.",
                tool_calls=[
                    self._make_llm_tool_call("call_main_to_A", "sub_agent_A", {"task": "get info"})
                ],
            )
        )

        main_events: list[BaseEvent] = []
        async for event in main_strategy.execute(main_agent, main_context):
            main_events.append(event)

        main_event_types = [e.event_type for e in main_events]
        assert main_event_types == [
            ExecutionEventType.LLM_START,
            ExecutionEventType.LLM_END,
            ExecutionEventType.RESPONSE,
            ExecutionEventType.AGENT_CALL,
        ], f"main_agent first run events: {main_event_types}"

        # Plain tool_use treats content emitted alongside tool calls as a
        # user-visible preamble: in non-streaming mode it surfaces as a
        # partial RESPONSE before AGENT_CALL, never as internal thought.
        preamble = main_events[-2]
        assert preamble.is_partial is True
        assert preamble.content[0].text == "I'll delegate to sub_agent_A."
        agent_start = main_events[-1]
        assert agent_start.action.name == "sub_agent_A"
        assert agent_start.action.parameters == {"task": "get info"}
        assert agent_start.action.thought is None
        assert agent_start.action.message is None

        # ── sub_agent_A: calls sub_agent_B ────────────────────────────────────
        sub_a_agent = self._make_agent("sub_agent_A", use_stop=True)
        sub_a_context = make_context(
            conversation_history=[
                make_event(ExecutionEventType.USER_INPUT, content_text="get info")
            ]
        )
        sub_b_desc = self._make_sub_agent_desc("sub_agent_B")
        sub_a_context.handoff_targets_description = [sub_b_desc]

        sub_a_strategy = self._make_strategy(use_stop=True)
        sub_a_agent.llm_provider = AsyncMock()

        # first call: handoff to sub_agent_B
        sub_a_agent.llm_provider.generate = AsyncMock(
            return_value=self._llm_response(
                content="Delegating to B.",
                tool_calls=[
                    self._make_llm_tool_call("call_A_to_B", "sub_agent_B", {"task": "get detail"})
                ],
            )
        )

        sub_a_events: list[BaseEvent] = []
        async for event in sub_a_strategy.execute(sub_a_agent, sub_a_context):
            sub_a_events.append(event)

        sub_a_event_types = [e.event_type for e in sub_a_events]
        assert sub_a_event_types == [
            ExecutionEventType.LLM_START,
            ExecutionEventType.LLM_END,
            ExecutionEventType.AGENT_CALL,
        ], f"sub_agent_A first run events: {sub_a_event_types}"

        preamble_a = sub_a_events[-1]
        assert preamble_a.is_partial is False
        agent_start_b = sub_a_events[-1]
        assert agent_start_b.action.name == "sub_agent_B"
        assert agent_start_b.action.parameters == {"task": "get detail"}
        assert agent_start_b.action.thought is None
        assert agent_start_b.action.message is None

        # ── sub_agent_B: uses STOP to return info ─────────────────────────────
        sub_b_agent = self._make_agent("sub_agent_B", use_stop=True)
        sub_b_context = make_context(
            conversation_history=[
                make_event(ExecutionEventType.USER_INPUT, content_text="get detail")
            ]
        )
        sub_b_context.handoff_targets_description = []  # no further sub-agents

        sub_b_strategy = self._make_strategy(use_stop=True)
        sub_b_agent.llm_provider = AsyncMock()

        stop_tool_name = ctx_mgr.stop_tool_name  # "STOP"
        sub_b_agent.llm_provider.generate = AsyncMock(
            return_value=self._llm_response(
                content="",
                tool_calls=[
                    self._make_llm_tool_call(
                        "call_B_stop", stop_tool_name, {"info": "detail from B"}
                    )
                ],
            )
        )

        sub_b_events: list[BaseEvent] = []
        async for event in sub_b_strategy.execute(sub_b_agent, sub_b_context):
            sub_b_events.append(event)

        sub_b_event_types = [e.event_type for e in sub_b_events]
        assert sub_b_event_types == [
            ExecutionEventType.LLM_START,
            ExecutionEventType.LLM_END,
            ExecutionEventType.AGENT_END,
        ], f"sub_agent_B events: {sub_b_event_types}"

        agent_end_b = sub_b_events[-1]
        assert agent_end_b.action.name == stop_tool_name
        assert agent_end_b.action.parameters == {"info": "detail from B"}

        # ── sub_agent_A: after receiving B's result, uses STOP ────────────────
        sub_a_agent_2 = self._make_agent("sub_agent_A", use_stop=True)
        sub_a_context_2 = make_context(
            conversation_history=[
                make_event(ExecutionEventType.USER_INPUT, content_text="get info"),
                make_event(
                    ExecutionEventType.AGENT_END,
                    author="sub_agent_B",
                    content_text="detail from B",
                    action=ToolAction(name=stop_tool_name, parameters={"info": "detail from B"}),
                ),
            ]
        )
        sub_a_context_2.handoff_targets_description = []

        sub_a_strategy_2 = self._make_strategy(use_stop=True)
        sub_a_agent_2.llm_provider = AsyncMock()
        sub_a_agent_2.llm_provider.generate = AsyncMock(
            return_value=self._llm_response(
                content="",
                tool_calls=[
                    self._make_llm_tool_call(
                        "call_A_stop", stop_tool_name, {"info": "info from A+B"}
                    )
                ],
            )
        )

        sub_a_events_2: list[BaseEvent] = []
        async for event in sub_a_strategy_2.execute(sub_a_agent_2, sub_a_context_2):
            sub_a_events_2.append(event)

        sub_a_event_types_2 = [e.event_type for e in sub_a_events_2]
        assert sub_a_event_types_2 == [
            ExecutionEventType.LLM_START,
            ExecutionEventType.LLM_END,
            ExecutionEventType.AGENT_END,
        ], f"sub_agent_A second run events: {sub_a_event_types_2}"

        agent_end_a = sub_a_events_2[-1]
        assert agent_end_a.action.name == stop_tool_name
        assert agent_end_a.action.parameters == {"info": "info from A+B"}

        # ── main_agent: after receiving A's result, emits final response via content ──
        main_agent_2 = self._make_agent("main_agent", use_stop=False)
        main_context_2 = make_context(
            conversation_history=[
                make_event(ExecutionEventType.USER_INPUT, content_text="Get info from A then B"),
                make_event(
                    ExecutionEventType.AGENT_END,
                    author="sub_agent_A",
                    content_text="info from A+B",
                    action=ToolAction(name=stop_tool_name, parameters={"info": "info from A+B"}),
                ),
            ]
        )
        main_context_2.handoff_targets_description = []

        main_strategy_2 = self._make_strategy(use_stop=False)
        main_agent_2.llm_provider = AsyncMock()
        main_agent_2.llm_provider.generate = AsyncMock(
            return_value=self._llm_response(
                content="Here is all the info: info from A+B",
                tool_calls=[],
            )
        )

        main_events_2: list[BaseEvent] = []
        async for event in main_strategy_2.execute(main_agent_2, main_context_2):
            main_events_2.append(event)

        main_event_types_2 = [e.event_type for e in main_events_2]
        assert main_event_types_2 == [
            ExecutionEventType.LLM_START,
            ExecutionEventType.LLM_END,
            ExecutionEventType.RESPONSE,
        ], f"main_agent final run events: {main_event_types_2}"

        response_event = main_events_2[-1]
        assert response_event.is_partial is False
        assert response_event.content[0].text == "Here is all the info: info from A+B"

    # ── Test 2: sequenziale main → sub_A (STOP) → main → sub_B (STOP) → main responds ──

    @pytest.mark.asyncio
    async def test_sequential_handoff_flow(self):
        """
        Scenario:
          main_agent  calls sub_agent_A  (AGENT_START, return)
          sub_agent_A has STANDARD visibility (no further sub-agents), STOPs with info
          main_agent  receives A's info, then calls sub_agent_B  (AGENT_START, return)
          sub_agent_B STOPs with info
          main_agent  receives B's info, then emits final RESPOND
        """
        ctx_mgr = ToolUseContextManager()
        stop_tool_name = ctx_mgr.stop_tool_name  # "STOP"

        sub_a_desc = self._make_sub_agent_desc("sub_agent_A")
        sub_b_desc = self._make_sub_agent_desc("sub_agent_B")

        # ── Step 1: main_agent → calls sub_agent_A ────────────────────────────
        main_agent_1 = self._make_agent("main_agent", use_stop=False)
        main_context_1 = make_context(
            conversation_history=[
                make_event(ExecutionEventType.USER_INPUT, content_text="Process with A then B")
            ]
        )
        main_context_1.handoff_targets_description = [sub_a_desc, sub_b_desc]
        main_strategy = self._make_strategy(use_stop=False)
        main_agent_1.llm_provider = AsyncMock()
        main_agent_1.llm_provider.generate = AsyncMock(
            return_value=self._llm_response(
                content="First calling A.",
                tool_calls=[
                    self._make_llm_tool_call(
                        "call_main_to_A", "sub_agent_A", {"task": "step A task"}
                    )
                ],
            )
        )

        step1_events: list[BaseEvent] = []
        async for event in main_strategy.execute(main_agent_1, main_context_1):
            step1_events.append(event)

        step1_types = [e.event_type for e in step1_events]
        assert step1_types == [
            ExecutionEventType.LLM_START,
            ExecutionEventType.LLM_END,
            ExecutionEventType.RESPONSE,
            ExecutionEventType.AGENT_CALL,
        ], f"Step 1 main_agent events: {step1_types}"

        # Plain tool_use treats content emitted alongside tool calls as a
        # user-visible preamble: in non-streaming mode it surfaces as a
        # partial RESPONSE before AGENT_CALL, never as internal thought.
        preamble_step1 = step1_events[-2]
        assert preamble_step1.is_partial is True
        assert preamble_step1.content[0].text == "First calling A."
        agent_start_a = step1_events[-1]
        assert agent_start_a.action.name == "sub_agent_A"
        assert agent_start_a.action.parameters == {"task": "step A task"}
        assert agent_start_a.action.thought is None
        assert agent_start_a.action.message is None

        # ── Step 2: sub_agent_A (STANDARD visibility) → STOPs ─────────────────
        sub_a_agent = self._make_agent("sub_agent_A", use_stop=True)
        sub_a_context = make_context(
            conversation_history=[
                make_event(ExecutionEventType.USER_INPUT, content_text="step A task")
            ]
        )
        sub_a_context.handoff_targets_description = []  # STANDARD: no sub-agents visible

        sub_a_strategy = self._make_strategy(use_stop=True)
        sub_a_agent.llm_provider = AsyncMock()
        sub_a_agent.llm_provider.generate = AsyncMock(
            return_value=self._llm_response(
                content="",
                tool_calls=[
                    self._make_llm_tool_call(
                        "call_A_stop", stop_tool_name, {"info": "result from A"}
                    )
                ],
            )
        )

        step2_events: list[BaseEvent] = []
        async for event in sub_a_strategy.execute(sub_a_agent, sub_a_context):
            step2_events.append(event)

        step2_types = [e.event_type for e in step2_events]
        assert step2_types == [
            ExecutionEventType.LLM_START,
            ExecutionEventType.LLM_END,
            ExecutionEventType.AGENT_END,
        ], f"Step 2 sub_agent_A events: {step2_types}"

        agent_end_a = step2_events[-1]
        assert agent_end_a.action.name == stop_tool_name
        assert agent_end_a.action.parameters == {"info": "result from A"}

        # ── Step 3: main_agent → receives A's result, calls sub_agent_B ───────
        main_agent_2 = self._make_agent("main_agent", use_stop=False)
        main_context_2 = make_context(
            conversation_history=[
                make_event(ExecutionEventType.USER_INPUT, content_text="Process with A then B"),
                make_event(
                    ExecutionEventType.AGENT_END,
                    author="sub_agent_A",
                    content_text="result from A",
                    action=ToolAction(name=stop_tool_name, parameters={"info": "result from A"}),
                ),
            ]
        )
        main_context_2.handoff_targets_description = [sub_b_desc]

        main_strategy_2 = self._make_strategy(use_stop=False)
        main_agent_2.llm_provider = AsyncMock()
        main_agent_2.llm_provider.generate = AsyncMock(
            return_value=self._llm_response(
                content="Now calling B.",
                tool_calls=[
                    self._make_llm_tool_call(
                        "call_main_to_B", "sub_agent_B", {"task": "step B task"}
                    )
                ],
            )
        )

        step3_events: list[BaseEvent] = []
        async for event in main_strategy_2.execute(main_agent_2, main_context_2):
            step3_events.append(event)

        step3_types = [e.event_type for e in step3_events]
        assert step3_types == [
            ExecutionEventType.LLM_START,
            ExecutionEventType.LLM_END,
            ExecutionEventType.RESPONSE,
            ExecutionEventType.AGENT_CALL,
        ], f"Step 3 main_agent events: {step3_types}"

        # Plain tool_use treats content emitted alongside tool calls as a
        # user-visible preamble: in non-streaming mode it surfaces as a
        # partial RESPONSE before AGENT_CALL, never as internal thought.
        preamble_step3 = step3_events[-2]
        assert preamble_step3.is_partial is True
        assert preamble_step3.content[0].text == "Now calling B."
        agent_start_b = step3_events[-1]
        assert agent_start_b.action.name == "sub_agent_B"
        assert agent_start_b.action.parameters == {"task": "step B task"}
        assert agent_start_b.action.thought is None
        assert agent_start_b.action.message is None

        # ── Step 4: sub_agent_B → STOPs with info ─────────────────────────────
        sub_b_agent = self._make_agent("sub_agent_B", use_stop=True)
        sub_b_context = make_context(
            conversation_history=[
                make_event(ExecutionEventType.USER_INPUT, content_text="step B task")
            ]
        )
        sub_b_context.handoff_targets_description = []

        sub_b_strategy = self._make_strategy(use_stop=True)
        sub_b_agent.llm_provider = AsyncMock()
        sub_b_agent.llm_provider.generate = AsyncMock(
            return_value=self._llm_response(
                content="",
                tool_calls=[
                    self._make_llm_tool_call(
                        "call_B_stop", stop_tool_name, {"info": "result from B"}
                    )
                ],
            )
        )

        step4_events: list[BaseEvent] = []
        async for event in sub_b_strategy.execute(sub_b_agent, sub_b_context):
            step4_events.append(event)

        step4_types = [e.event_type for e in step4_events]
        assert step4_types == [
            ExecutionEventType.LLM_START,
            ExecutionEventType.LLM_END,
            ExecutionEventType.AGENT_END,
        ], f"Step 4 sub_agent_B events: {step4_types}"

        agent_end_b = step4_events[-1]
        assert agent_end_b.action.name == stop_tool_name
        assert agent_end_b.action.parameters == {"info": "result from B"}

        # ── Step 5: main_agent → receives B's result, emits final response via content ──
        main_agent_3 = self._make_agent("main_agent", use_stop=False)
        main_context_3 = make_context(
            conversation_history=[
                make_event(ExecutionEventType.USER_INPUT, content_text="Process with A then B"),
                make_event(
                    ExecutionEventType.AGENT_END,
                    author="sub_agent_A",
                    content_text="result from A",
                    action=ToolAction(name=stop_tool_name, parameters={"info": "result from A"}),
                ),
                make_event(
                    ExecutionEventType.AGENT_END,
                    author="sub_agent_B",
                    content_text="result from B",
                    action=ToolAction(name=stop_tool_name, parameters={"info": "result from B"}),
                ),
            ]
        )
        main_context_3.handoff_targets_description = []

        main_strategy_3 = self._make_strategy(use_stop=False)
        main_agent_3.llm_provider = AsyncMock()
        main_agent_3.llm_provider.generate = AsyncMock(
            return_value=self._llm_response(
                content="Done: result from A + result from B",
                tool_calls=[],
            )
        )

        step5_events: list[BaseEvent] = []
        async for event in main_strategy_3.execute(main_agent_3, main_context_3):
            step5_events.append(event)

        step5_types = [e.event_type for e in step5_events]
        assert step5_types == [
            ExecutionEventType.LLM_START,
            ExecutionEventType.LLM_END,
            ExecutionEventType.RESPONSE,
        ], f"Step 5 main_agent final events: {step5_types}"

        response_event = step5_events[-1]
        assert response_event.is_partial is False
        assert response_event.content[0].text == "Done: result from A + result from B"


class TestPlannerStrategyFactory:
    def test_create_tool_use_from_config(self):
        from domyn.agents.planner_strategy import PlannerStrategyFactory

        strategy = PlannerStrategyFactory.from_config(
            {
                "type": "tool_use",
                "name": "test_tool_use",
                "max_iterations": 15,
            }
        )
        assert isinstance(strategy, ToolUsePlannerStrategy)
        assert strategy.max_iterations == 15
        assert strategy.context_manager.type == "tool_use"

    def test_create_tool_use_with_context_manager_config(self):
        from domyn.agents.planner_strategy import PlannerStrategyFactory

        strategy = PlannerStrategyFactory.from_config(
            {
                "type": "tool_use",
                "name": "test_tool_use",
                "context_manager": {
                    "type": "tool_use",
                    "name": "custom_ctx",
                    "max_tool_result_chars": 8000,
                    "useless_extra_args_to_ignore": "ignored_value",
                },
            }
        )
        assert strategy.context_manager.max_tool_result_chars == 8000
