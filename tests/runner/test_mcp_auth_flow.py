"""Runner-level tests for MCP tool authentication checks at TOOL_START time."""

from collections.abc import AsyncGenerator
from unittest.mock import patch

import pytest

from domyn.agents import Runner
from domyn.agents.agents import Agent
from domyn.agents.core import (
    AgentAction,
    ApprovalType,
    AuthenticationAction,
    AuthenticationResult,
    BaseEvent,
    ExecutionEventType,
    FeedbackAction,
    FeedbackParameters,
    Part,
    ToolAction,
)
from domyn.agents.llm import BaseLLMProvider, LLMMessage
from domyn.agents.tools import ToolResult
from domyn.agents.tools.mcp import MCPCredentialsCheckResult, MCPTool

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

_MCP_TOOL_NAME = "my_tool"
_SERVER_URL = "https://example.com/mcp"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_mcp_tool() -> MCPTool:
    return MCPTool(tool_name=_MCP_TOOL_NAME, server_url=_SERVER_URL, use_oauth=True)


def _tool_start_event(agent_name: str) -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.TOOL_START,
        author=agent_name,
        action=ToolAction(name=_MCP_TOOL_NAME, parameters={}, thought="calling my_tool"),
    )


def _response_event(agent_name: str) -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.RESPONSE,
        author=agent_name,
        content=[Part(text="done")],
    )


async def _fail_auth(self) -> MCPCredentialsCheckResult:
    return MCPCredentialsCheckResult(
        credentials_valid=False,
        tool_uid=self.uid,
        tool_name=self.tool_name,
        server_url=self.server_url,
    )


async def _pass_auth(self) -> MCPCredentialsCheckResult:
    return MCPCredentialsCheckResult(
        credentials_valid=True,
        tool_uid=self.uid,
        tool_name=self.tool_name,
        server_url=self.server_url,
    )


async def _mock_execute(self, **kwargs) -> AsyncGenerator:
    yield ToolResult(success=True, data="ok")


def async_gen_from_list(values):
    _iter = iter(values)
    finished = False

    async def _gen(*args, **kwargs) -> AsyncGenerator:
        nonlocal finished
        if finished:
            raise StopAsyncIteration
        for v in _iter:
            yield v
        finished = True

    return _gen


@pytest.fixture
def runner() -> Runner:
    return Runner()


@pytest.fixture
def llm_provider() -> BaseLLMProvider:
    class MockLLM(BaseLLMProvider):
        type: str = "mock"
        name: str = "mock-llm"
        model_name: str = "mock-model"
        supported_structured_outputs: None = None

        async def generate(self, messages: LLMMessage, **kwargs):
            pass

        async def generate_streaming(self, messages: LLMMessage, **kwargs):
            yield

    return MockLLM()


def _user_input() -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="hello")],
    )


# ---------------------------------------------------------------------------
# root-agent path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_root_agent_mcp_auth_failure_interrupts_runner(
    runner: Runner, llm_provider: BaseLLMProvider
):
    """Auth failure on an MCPTool at TOOL_START must yield TOOL_NEEDS_FEEDBACK and stop."""
    mcp_tool = _make_mcp_tool()
    agent = Agent(
        name="RootAgent",
        description="desc",
        instruction="instr",
        llm_provider=llm_provider,
        tools=[mcp_tool],
    )
    agent._execute = async_gen_from_list([_tool_start_event("RootAgent")])

    with patch.object(MCPTool, "check_auth_credentials", _fail_auth):
        events = [
            e
            async for e in runner.run(
                application_name="app",
                user_id="user",
                root=agent,
                user_input=_user_input(),
                conversation_id="conv_root_fail",
            )
        ]

    tnf = next((e for e in events if e.event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK), None)
    assert tnf is not None, "Expected TOOL_NEEDS_FEEDBACK"
    assert isinstance(tnf.action, AuthenticationAction)
    assert tnf.action.name == _MCP_TOOL_NAME
    assert tnf.action.authentication_metadata.server_url == _SERVER_URL
    assert events.index(tnf) == len(events) - 1

    # TOOL_START (carrying its thought) must be forwarded immediately before
    # TOOL_NEEDS_FEEDBACK — previously it was swallowed entirely.
    tool_start = events[events.index(tnf) - 1]
    assert tool_start.event_type == ExecutionEventType.TOOL_START
    assert tool_start.action.thought == "calling my_tool"


@pytest.mark.asyncio
async def test_root_agent_mcp_auth_success_continues_normally(
    runner: Runner, llm_provider: BaseLLMProvider
):
    """Successful auth must not interrupt the runner; the agent response must be reached."""
    mcp_tool = _make_mcp_tool()
    agent = Agent(
        name="RootAgent",
        description="desc",
        instruction="instr",
        llm_provider=llm_provider,
        tools=[mcp_tool],
    )
    agent._execute = async_gen_from_list(
        [_tool_start_event("RootAgent"), _response_event("RootAgent")]
    )

    with (
        patch.object(MCPTool, "check_auth_credentials", _pass_auth),
        patch.object(MCPTool, "execute", _mock_execute),
    ):
        events = [
            e
            async for e in runner.run(
                application_name="app",
                user_id="user",
                root=agent,
                user_input=_user_input(),
                conversation_id="conv_root_ok",
            )
        ]

    event_types = [e.event_type for e in events]
    assert ExecutionEventType.TOOL_NEEDS_FEEDBACK not in event_types
    assert any(
        e.event_type == ExecutionEventType.RESPONSE and e.author == "RootAgent" for e in events
    )


@pytest.mark.asyncio
async def test_root_agent_no_mcp_tools_continues_normally(
    runner: Runner, llm_provider: BaseLLMProvider
):
    """An agent with no MCPTools must not be interrupted."""
    agent = Agent(
        name="RootAgent",
        description="desc",
        instruction="instr",
        llm_provider=llm_provider,
    )
    agent._execute = async_gen_from_list([_response_event("RootAgent")])

    events = [
        e
        async for e in runner.run(
            application_name="app",
            user_id="user",
            root=agent,
            user_input=_user_input(),
            conversation_id="conv_no_tools",
        )
    ]

    assert ExecutionEventType.TOOL_NEEDS_FEEDBACK not in [e.event_type for e in events]
    assert any(e.event_type == ExecutionEventType.RESPONSE for e in events)


# ---------------------------------------------------------------------------
# sub-agent path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sub_agent_mcp_auth_failure_interrupts_runner(
    runner: Runner, llm_provider: BaseLLMProvider
):
    """Auth failure on a sub-agent's MCPTool must yield TOOL_NEEDS_FEEDBACK as the last event."""
    mcp_tool = _make_mcp_tool()
    sub_agent = Agent(
        name="SubAgent",
        description="sub desc",
        instruction="sub instr",
        llm_provider=llm_provider,
        tools=[mcp_tool],
    )
    main_agent = Agent(
        name="MainAgent",
        description="main desc",
        instruction="main instr",
        llm_provider=llm_provider,
        sub_agents=[sub_agent],
    )
    main_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.AGENT_CALL,
                author="MainAgent",
                content=[],
                action=AgentAction(
                    name="SubAgent",
                    parameters={"task": "do something"},
                    thought="delegating",
                ),
            )
        ]
    )
    sub_agent._execute = async_gen_from_list([_tool_start_event("SubAgent")])

    with patch.object(MCPTool, "check_auth_credentials", _fail_auth):
        events = [
            e
            async for e in runner.run(
                application_name="app",
                user_id="user",
                root=main_agent,
                user_input=_user_input(),
                conversation_id="conv_sub_fail",
            )
        ]

    event_types = [e.event_type for e in events]
    assert ExecutionEventType.TOOL_NEEDS_FEEDBACK in event_types

    tnf = next(e for e in events if e.event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK)
    assert isinstance(tnf.action, AuthenticationAction)
    assert tnf.action.name == _MCP_TOOL_NAME
    assert events.index(tnf) == len(events) - 1

    # TOOL_START (carrying its thought) must be forwarded immediately before
    # TOOL_NEEDS_FEEDBACK — previously it was swallowed entirely.
    tool_start = events[events.index(tnf) - 1]
    assert tool_start.event_type == ExecutionEventType.TOOL_START
    assert tool_start.action.thought == "calling my_tool"


@pytest.mark.asyncio
async def test_sub_agent_mcp_auth_success_continues_normally(
    runner: Runner, llm_provider: BaseLLMProvider
):
    """Successful auth on the sub-agent's MCPTool must allow its response to be reached."""
    mcp_tool = _make_mcp_tool()
    sub_agent = Agent(
        name="SubAgent",
        description="sub desc",
        instruction="sub instr",
        llm_provider=llm_provider,
        tools=[mcp_tool],
    )
    main_agent = Agent(
        name="MainAgent",
        description="main desc",
        instruction="main instr",
        llm_provider=llm_provider,
        sub_agents=[sub_agent],
    )
    main_agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.AGENT_CALL,
                author="MainAgent",
                content=[],
                action=AgentAction(
                    name="SubAgent",
                    parameters={"task": "do something"},
                    thought="delegating",
                ),
            )
        ]
    )
    sub_agent._execute = async_gen_from_list(
        [_tool_start_event("SubAgent"), _response_event("SubAgent")]
    )

    with (
        patch.object(MCPTool, "check_auth_credentials", _pass_auth),
        patch.object(MCPTool, "execute", _mock_execute),
    ):
        events = [
            e
            async for e in runner.run(
                application_name="app",
                user_id="user",
                root=main_agent,
                user_input=_user_input(),
                conversation_id="conv_sub_ok",
            )
        ]

    event_types = [e.event_type for e in events]
    assert ExecutionEventType.TOOL_NEEDS_FEEDBACK not in event_types
    # The sub-agent's final RESPONSE is converted to AGENT_END by the runner
    # (subagents never respond to the user directly); reaching it proves the
    # auth success let execution continue past the tool call.
    assert not any(
        e.event_type == ExecutionEventType.RESPONSE and e.author == "SubAgent" for e in events
    )
    assert any(
        e.event_type == ExecutionEventType.AGENT_END and e.author == "SubAgent" for e in events
    )


# ---------------------------------------------------------------------------
# auth error raised during tool execution (401 inside execute())
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_tool_auth_error_during_execution_yields_tool_needs_feedback(
    runner: Runner, llm_provider: BaseLLMProvider
):
    """If the pre-auth check passes but the tool itself returns error_code='401',
    the runner must still emit TOOL_NEEDS_FEEDBACK with AuthenticationAction."""
    mcp_tool = _make_mcp_tool()
    agent = Agent(
        name="RootAgent",
        description="desc",
        instruction="instr",
        llm_provider=llm_provider,
        tools=[mcp_tool],
    )
    agent._execute = async_gen_from_list([_tool_start_event("RootAgent")])

    async def _401_execute(self, **kwargs):
        yield ToolResult(
            success=False,
            error="Authentication required: OAuth token is invalid or expired. Please re-authenticate.",
            error_code="401",
        )

    with (
        patch.object(MCPTool, "check_auth_credentials", _pass_auth),
        patch.object(MCPTool, "execute", _401_execute),
    ):
        events = [
            e
            async for e in runner.run(
                application_name="app",
                user_id="user",
                root=agent,
                user_input=_user_input(),
                conversation_id="conv_exec_401",
            )
        ]

    tnf = next((e for e in events if e.event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK), None)
    assert tnf is not None, "Expected TOOL_NEEDS_FEEDBACK due to 401 during tool execution"
    assert isinstance(tnf.action, AuthenticationAction)
    assert tnf.action.name == _MCP_TOOL_NAME
    assert tnf.action.authentication_metadata.server_url == _SERVER_URL


# ---------------------------------------------------------------------------
# TOOL_FEEDBACK resumption
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_feedback_resumes_tool_execution(runner: Runner, llm_provider: BaseLLMProvider):
    """After TOOL_NEEDS_FEEDBACK, sending TOOL_FEEDBACK with auth now passing must
    re-execute the tool and resume the agent."""
    conv_id = "conv_feedback_resume"
    mcp_tool = _make_mcp_tool()
    agent = Agent(
        name="RootAgent",
        description="desc",
        instruction="instr",
        llm_provider=llm_provider,
        tools=[mcp_tool],
    )
    agent._execute = async_gen_from_list([_tool_start_event("RootAgent")])

    # First run: auth fails → TOOL_NEEDS_FEEDBACK
    with patch.object(MCPTool, "check_auth_credentials", _fail_auth):
        first_run = [
            e
            async for e in runner.run(
                application_name="app",
                user_id="user",
                root=agent,
                user_input=_user_input(),
                conversation_id=conv_id,
            )
        ]

    assert first_run[-1].event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK
    tnf = first_run[-1]

    # Second run: client sends TOOL_FEEDBACK; auth now succeeds and tool executes
    agent._execute = async_gen_from_list([_response_event("RootAgent")])

    tool_feedback = BaseEvent(
        event_type=ExecutionEventType.TOOL_FEEDBACK,
        author="user",
        content=[],
        action=AuthenticationAction(
            call_id=tnf.action.call_id,
            name=tnf.action.name,
            parameters=None,
            success=AuthenticationResult.SUCCESS,
        ),
    )

    with patch.object(MCPTool, "execute", _mock_execute):
        second_run = [
            e
            async for e in runner.run(
                application_name="app",
                user_id="user",
                root=agent,
                user_input=tool_feedback,
                conversation_id=conv_id,
            )
        ]

    event_types = [e.event_type for e in second_run]
    assert ExecutionEventType.TOOL_NEEDS_FEEDBACK not in event_types
    assert any(
        e.event_type == ExecutionEventType.RESPONSE and e.author == "RootAgent" for e in second_run
    )


@pytest.mark.asyncio
async def test_tool_feedback_not_success_yields_tool_error(
    runner: Runner, llm_provider: BaseLLMProvider
):
    """TOOL_FEEDBACK with success=FAILURE must yield TOOL_ERROR instead of re-executing the tool."""
    conv_id = "conv_feedback_skipped"
    mcp_tool = _make_mcp_tool()
    agent = Agent(
        name="RootAgent",
        description="desc",
        instruction="instr",
        llm_provider=llm_provider,
        tools=[mcp_tool],
    )
    agent._execute = async_gen_from_list([_tool_start_event("RootAgent")])

    # First run: auth fails → TOOL_NEEDS_FEEDBACK
    with patch.object(MCPTool, "check_auth_credentials", _fail_auth):
        first_run = [
            e
            async for e in runner.run(
                application_name="app",
                user_id="user",
                root=agent,
                user_input=_user_input(),
                conversation_id=conv_id,
            )
        ]

    assert first_run[-1].event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK
    tnf = first_run[-1]

    # Second run: client skips authentication
    agent._execute = async_gen_from_list([_response_event("RootAgent")])

    tool_feedback = BaseEvent(
        event_type=ExecutionEventType.TOOL_FEEDBACK,
        author="user",
        content=[],
        action=AuthenticationAction(
            call_id=tnf.action.call_id,
            name=tnf.action.name,
            parameters=None,
            success=AuthenticationResult.FAILURE,
        ),
    )

    second_run = [
        e
        async for e in runner.run(
            application_name="app",
            user_id="user",
            root=agent,
            user_input=tool_feedback,
            conversation_id=conv_id,
        )
    ]

    event_types = [e.event_type for e in second_run]
    assert ExecutionEventType.TOOL_ERROR in event_types
    assert ExecutionEventType.TOOL_NEEDS_FEEDBACK not in event_types

    tool_error = next(e for e in second_run if e.event_type == ExecutionEventType.TOOL_ERROR)
    assert "Authentication failed" in tool_error.action.observation
    assert _MCP_TOOL_NAME in tool_error.action.observation


@pytest.mark.asyncio
async def test_tool_feedback_user_refused_yields_tool_end_with_refusal_observation(
    runner: Runner, llm_provider: BaseLLMProvider
):
    """TOOL_FEEDBACK with success=USER_REFUSED must yield a TOOL_END carrying a refusal
    observation instead of a TOOL_ERROR, and must not re-execute the tool."""
    conv_id = "conv_feedback_user_refused"
    mcp_tool = _make_mcp_tool()
    agent = Agent(
        name="RootAgent",
        description="desc",
        instruction="instr",
        llm_provider=llm_provider,
        tools=[mcp_tool],
    )
    agent._execute = async_gen_from_list([_tool_start_event("RootAgent")])

    # First run: auth fails → TOOL_NEEDS_FEEDBACK
    with patch.object(MCPTool, "check_auth_credentials", _fail_auth):
        first_run = [
            e
            async for e in runner.run(
                application_name="app",
                user_id="user",
                root=agent,
                user_input=_user_input(),
                conversation_id=conv_id,
            )
        ]

    assert first_run[-1].event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK
    tnf = first_run[-1]

    # Second run: user declines to authenticate at all
    agent._execute = async_gen_from_list([_response_event("RootAgent")])

    tool_feedback = BaseEvent(
        event_type=ExecutionEventType.TOOL_FEEDBACK,
        author="user",
        content=[],
        action=AuthenticationAction(
            call_id=tnf.action.call_id,
            name=tnf.action.name,
            parameters=None,
            success=AuthenticationResult.USER_REFUSED,
        ),
    )

    execute_calls: list[dict] = []

    async def _tracking_execute(self, **kwargs) -> AsyncGenerator:
        execute_calls.append(kwargs)
        yield ToolResult(success=True, data="ok")

    with patch.object(MCPTool, "execute", _tracking_execute):
        second_run = [
            e
            async for e in runner.run(
                application_name="app",
                user_id="user",
                root=agent,
                user_input=tool_feedback,
                conversation_id=conv_id,
            )
        ]

    assert execute_calls == [], "Tool must not be re-executed after a user-refused auth"

    event_types = [e.event_type for e in second_run]
    assert ExecutionEventType.TOOL_END in event_types
    assert ExecutionEventType.TOOL_ERROR not in event_types
    assert ExecutionEventType.TOOL_NEEDS_FEEDBACK not in event_types

    tool_end = next(e for e in second_run if e.event_type == ExecutionEventType.TOOL_END)
    assert "User refused to authenticate" in tool_end.action.observation
    assert _MCP_TOOL_NAME in tool_end.action.observation


# ---------------------------------------------------------------------------
# combined parameter feedback + auth feedback
# ---------------------------------------------------------------------------


def _make_mcp_tool_with_approval() -> MCPTool:
    return MCPTool(
        tool_name=_MCP_TOOL_NAME,
        server_url=_SERVER_URL,
        use_oauth=True,
        approval_type=ApprovalType.PARAMS_UPDATE,
    )


def _make_capturing_execute():
    captured: dict = {}

    async def _execute(self, **kwargs):
        captured.update(kwargs)
        yield ToolResult(success=True, data="ok")

    return _execute, captured


@pytest.mark.asyncio
async def test_param_feedback_then_auth_failure_preserves_modified_params(
    runner: Runner, llm_provider: BaseLLMProvider
):
    """Modified parameters from a parameter-approval TOOL_FEEDBACK must be used for tool
    execution even when an auth failure interrupts between the two feedback rounds."""
    conv_id = "conv_param_auth_3turn"
    mcp_tool = _make_mcp_tool_with_approval()
    agent = Agent(
        name="RootAgent",
        description="desc",
        instruction="instr",
        llm_provider=llm_provider,
        tools=[mcp_tool],
    )

    # Turn 1: agent emits TOOL_START with original params → runner pauses for param approval
    agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.TOOL_START,
                author="RootAgent",
                action=ToolAction(
                    name=_MCP_TOOL_NAME,
                    parameters={"query": "original_value"},
                    thought="calling my_tool",
                ),
            )
        ]
    )
    turn1 = [
        e
        async for e in runner.run(
            application_name="app",
            user_id="user",
            root=agent,
            user_input=_user_input(),
            conversation_id=conv_id,
        )
    ]
    assert turn1[-1].event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK
    assert isinstance(turn1[-1].action, FeedbackAction)

    # TOOL_START (carrying its thought) must be forwarded immediately before
    # the approval TOOL_NEEDS_FEEDBACK.
    assert turn1[-2].event_type == ExecutionEventType.TOOL_START
    assert turn1[-2].action.thought == "calling my_tool"

    # Turn 2: client sends modified params; auth check fails → TOOL_NEEDS_FEEDBACK (AuthenticationAction)
    param_feedback = BaseEvent(
        event_type=ExecutionEventType.TOOL_FEEDBACK,
        author="user",
        content=[],
        action=FeedbackAction(
            name=_MCP_TOOL_NAME,
            parameters=[FeedbackParameters(name="query", value="modified_value")],
            approval_type=ApprovalType.PARAMS_UPDATE,
            approve=True,
        ),
    )
    with patch.object(MCPTool, "check_auth_credentials", _fail_auth):
        turn2 = [
            e
            async for e in runner.run(
                application_name="app",
                user_id="user",
                root=agent,
                user_input=param_feedback,
                conversation_id=conv_id,
            )
        ]

    assert turn2[-1].event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK
    tnf_auth = turn2[-1]
    assert isinstance(tnf_auth.action, AuthenticationAction)

    # TOOL_START was already delivered on turn 1 — the resume path must not
    # forward (and thus duplicate) it a second time.
    assert ExecutionEventType.TOOL_START not in [e.event_type for e in turn2]

    # Turn 3: client resolves auth; tool must execute with the MODIFIED params from Turn 2
    agent._execute = async_gen_from_list([_response_event("RootAgent")])
    auth_feedback = BaseEvent(
        event_type=ExecutionEventType.TOOL_FEEDBACK,
        author="user",
        content=[],
        action=AuthenticationAction(
            call_id=tnf_auth.action.call_id,
            name=tnf_auth.action.name,
            parameters=None,
            success=AuthenticationResult.SUCCESS,
        ),
    )
    capturing_execute, captured_kwargs = _make_capturing_execute()
    with patch.object(MCPTool, "execute", capturing_execute):
        turn3 = [
            e
            async for e in runner.run(
                application_name="app",
                user_id="user",
                root=agent,
                user_input=auth_feedback,
                conversation_id=conv_id,
            )
        ]

    assert any(
        e.event_type == ExecutionEventType.RESPONSE and e.author == "RootAgent" for e in turn3
    )
    assert captured_kwargs.get("query") == "modified_value", (
        f"Tool was called with {captured_kwargs!r}, expected query='modified_value'"
    )
    assert ExecutionEventType.TOOL_START not in [e.event_type for e in turn3]


@pytest.mark.asyncio
async def test_param_feedback_then_auth_success_uses_modified_params(
    runner: Runner, llm_provider: BaseLLMProvider
):
    """When auth succeeds immediately after param approval, the tool must still use
    the user-modified parameters, not the original ones from TOOL_START."""
    conv_id = "conv_param_auth_2turn"
    mcp_tool = _make_mcp_tool_with_approval()
    agent = Agent(
        name="RootAgent",
        description="desc",
        instruction="instr",
        llm_provider=llm_provider,
        tools=[mcp_tool],
    )

    # Turn 1: agent emits TOOL_START with original params → runner pauses for param approval
    agent._execute = async_gen_from_list(
        [
            BaseEvent(
                event_type=ExecutionEventType.TOOL_START,
                author="RootAgent",
                action=ToolAction(
                    name=_MCP_TOOL_NAME,
                    parameters={"query": "original_value"},
                    thought="calling my_tool",
                ),
            )
        ]
    )
    turn1 = [
        e
        async for e in runner.run(
            application_name="app",
            user_id="user",
            root=agent,
            user_input=_user_input(),
            conversation_id=conv_id,
        )
    ]
    assert turn1[-1].event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK
    assert isinstance(turn1[-1].action, FeedbackAction)

    # TOOL_START (carrying its thought) must be forwarded immediately before
    # the approval TOOL_NEEDS_FEEDBACK.
    assert turn1[-2].event_type == ExecutionEventType.TOOL_START
    assert turn1[-2].action.thought == "calling my_tool"

    # Turn 2: client sends modified params; auth succeeds → tool executes with modified params
    agent._execute = async_gen_from_list([_response_event("RootAgent")])
    param_feedback = BaseEvent(
        event_type=ExecutionEventType.TOOL_FEEDBACK,
        author="user",
        content=[],
        action=FeedbackAction(
            name=_MCP_TOOL_NAME,
            parameters=[FeedbackParameters(name="query", value="modified_value")],
            approval_type=ApprovalType.PARAMS_UPDATE,
            approve=True,
        ),
    )
    capturing_execute, captured_kwargs = _make_capturing_execute()
    with (
        patch.object(MCPTool, "check_auth_credentials", _pass_auth),
        patch.object(MCPTool, "execute", capturing_execute),
    ):
        turn2 = [
            e
            async for e in runner.run(
                application_name="app",
                user_id="user",
                root=agent,
                user_input=param_feedback,
                conversation_id=conv_id,
            )
        ]

    assert ExecutionEventType.TOOL_NEEDS_FEEDBACK not in [e.event_type for e in turn2]
    assert any(
        e.event_type == ExecutionEventType.RESPONSE and e.author == "RootAgent" for e in turn2
    )
    assert captured_kwargs.get("query") == "modified_value", (
        f"Tool was called with {captured_kwargs!r}, expected query='modified_value'"
    )
    # TOOL_START was already delivered on turn 1 — the resume path must not
    # forward (and thus duplicate) it a second time.
    assert ExecutionEventType.TOOL_START not in [e.event_type for e in turn2]
