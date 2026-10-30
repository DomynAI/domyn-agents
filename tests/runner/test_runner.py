from unittest.mock import MagicMock

import pytest

from domyn.agents.agents import Agent, ContextVisibility
from domyn.agents.conversation_manager import BaseConversationManager, ConversationInfo
from domyn.agents.core import AgentAction, BaseEvent, ExecutionEventType, Part, ToolAction
from domyn.agents.hooks import FunctionHook, HookPhase, HookType
from domyn.agents.llm import BaseLLMProvider
from domyn.agents.runner import ContextBuilder, Runner, RunnerState


@pytest.mark.asyncio
async def test_init_runner_state_empty_root(runner: Runner) -> None:
    """Test the initialization of RunnerState."""

    with pytest.raises(ValueError):
        await runner._get_state(
            application_name="test_app",
            conversation_id="test_convo",
            user_id="test_user",
            root=None,
            entrypoint=None,
        )


@pytest.mark.asyncio
async def test_init_runner_state_with_root(runner: Runner, llm_provider: BaseLLMProvider):
    """Test the initialization of RunnerState."""

    agent_a = Agent(
        name="agent_a",
        description="agent_a description",
        instruction="agent_a instruction",
        llm_provider=llm_provider,
    )

    state: RunnerState = await runner._get_state(
        application_name="test_app",
        conversation_id="test_convo",
        user_id="test_user",
        root=agent_a,
        entrypoint=None,
    )

    assert state
    assert state.application_name == "test_app"
    assert state.conversation_info is not None
    assert isinstance(state.conversation_info, ConversationInfo)
    assert state.interaction_id is None
    assert state.entrypoint_agent == agent_a
    assert state.available_agents
    assert state.current_agent == agent_a


@pytest.mark.asyncio
async def test__setup_conversation_turn_and_build_agent_start_event(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    application_name = "test_app"
    conversation_id = "test_convo"
    user_id = "test_user"

    agent_a = Agent(
        name="agent_a",
        description="agent_a description",
        instruction="agent_a instruction",
        llm_provider=llm_provider,
    )

    state: RunnerState = await runner._get_state(
        application_name=application_name,
        conversation_id=conversation_id,
        user_id=user_id,
        root=agent_a,
        entrypoint=None,
    )

    user_input = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        content=[Part(text="Test user input")],
        author="user",
    )
    store_agent_start, _ = await runner._setup_conversation_turn(
        state=state,
        user_input=user_input,
    )
    start_event = runner._build_agent_start_event(state, user_input, store_agent_start)

    assert start_event.event_type == ExecutionEventType.AGENT_START
    assert start_event.author == "agent_a"
    assert start_event.action.name == "agent_a"

    last_stack = await runner.conversation_manager.get_last_agent_stack(
        application_name=application_name, user_id=user_id, conversation_id=conversation_id
    )
    assert last_stack.agent_name == "agent_a"


@pytest.mark.asyncio
async def test__handle_agent_transition_from_none(runner: Runner, llm_provider: BaseLLMProvider):
    """Test the _handle_agent_transition method of the Runner."""

    agent_a = Agent(
        name="agent_a",
        description="agent_a description",
        instruction="agent_a instruction",
        llm_provider=llm_provider,
    )

    state: RunnerState = await runner._get_state(
        application_name="test_app",
        conversation_id="test_convo",
        user_id="test_user",
        root=agent_a,
        entrypoint=None,
    )

    event = BaseEvent(
        event_type=ExecutionEventType.AGENT_START,
        content=[],
        author="user",
        action=AgentAction(name="agent_a", parameters={}, thought="Starting agent_a"),
    )

    context = ContextBuilder.create_context(
        state=state,
        user_input=BaseEvent(
            event_type=ExecutionEventType.USER_INPUT,
            content=[Part(text="Test user input")],
            author="user",
        ),
    )

    new_agent, new_context, _ = await runner._handle_agent_transition(
        state=state, event=event, current_context=context
    )
    assert new_agent == agent_a
    assert new_context is not None
    assert new_context.agent_input.event_type == ExecutionEventType.AGENT_START
    assert new_context.conversation_id == "test_convo"
    assert new_context.interaction_id is not None


@pytest.mark.asyncio
async def test__handle_agent_transition_wrong_event_type(
    runner: Runner, llm_provider: BaseLLMProvider
):
    """Test the _handle_agent_transition method of the Runner."""

    agent_a = Agent(
        name="agent_a",
        description="agent_a description",
        instruction="agent_a instruction",
        llm_provider=llm_provider,
    )

    state: RunnerState = await runner._get_state(
        application_name="test_app",
        conversation_id="test_convo",
        user_id="test_user",
        root=agent_a,
        entrypoint=None,
    )

    new_event = BaseEvent(
        event_type=ExecutionEventType.TOOL_START,
        content=[],
        author="user",
        action=ToolAction(name="agent_a", parameters={}, thought="Starting agent_a"),
    )

    context = ContextBuilder.create_context(
        state=state,
        user_input=BaseEvent(
            event_type=ExecutionEventType.USER_INPUT,
            content=[Part(text="Test user input")],
            author="user",
        ),
    )

    with pytest.raises(ValueError):
        _, __, ___ = await runner._handle_agent_transition(
            state=state, event=new_event, current_context=context
        )


@pytest.mark.asyncio
async def test__handle_agent_transition_from_agent(runner: Runner, llm_provider: BaseLLMProvider):
    """Test the _handle_agent_transition method of the Runner."""

    agent_b = Agent(
        name="agent_b",
        description="agent_b description",
        instruction="agent_b instruction",
        llm_provider=llm_provider,
    )
    agent_a = Agent(
        name="agent_a",
        description="agent_a description",
        instruction="agent_a instruction",
        llm_provider=llm_provider,
        sub_agents=[agent_b],
    )

    application_name = "test_app"
    conversation_id = "test_convo"
    user_id = "test_user"

    state: RunnerState = await runner._get_state(
        application_name=application_name,
        conversation_id=conversation_id,
        user_id=user_id,
        root=agent_a,
        entrypoint=None,
    )

    assert state.available_agents == [agent_b, agent_a]

    event = BaseEvent(
        event_type=ExecutionEventType.AGENT_START,
        content=[],
        author="user",
        action=AgentAction(name="agent_b", parameters={}, thought="Starting agent_b"),
    )

    context = ContextBuilder.create_context(
        state=state,
        user_input=BaseEvent(
            event_type=ExecutionEventType.USER_INPUT,
            content=[Part(text="Test user input")],
            author="user",
        ),
    )

    new_agent, new_context, _ = await runner._handle_agent_transition(
        state=state, event=event, current_context=context
    )
    assert new_agent == agent_b
    assert new_context is not None
    assert new_context.agent_input.event_type == ExecutionEventType.AGENT_START
    assert new_context.conversation_id == "test_convo"
    assert new_context.interaction_id is not None

    # check new agent is in the stack
    last_stack = await runner.conversation_manager.get_last_agent_stack(
        application_name=application_name, user_id=user_id, conversation_id=conversation_id
    )
    assert last_stack.agent_name == "agent_b"


@pytest.mark.asyncio
async def test_start_and__handle_agent_transition_from_agent(
    runner: Runner, llm_provider: BaseLLMProvider
):
    """Test the _handle_agent_transition method of the Runner."""

    agent_b = Agent(
        name="agent_b",
        description="agent_b description",
        instruction="agent_b instruction",
        llm_provider=llm_provider,
    )
    agent_a = Agent(
        name="agent_a",
        description="agent_a description",
        instruction="agent_a instruction",
        llm_provider=llm_provider,
        sub_agents=[agent_b],
    )

    application_name = "test_app"
    conversation_id = "test_convo"
    user_id = "test_user"

    state: RunnerState = await runner._get_state(
        application_name=application_name,
        conversation_id=conversation_id,
        user_id=user_id,
        root=agent_a,
        entrypoint=None,
    )

    assert state.available_agents == [agent_b, agent_a]

    user_input_for_start = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        content=[Part(text="Test user input")],
        author="user",
    )
    store_agent_start, _ = await runner._setup_conversation_turn(
        state=state,
        user_input=user_input_for_start,
    )
    start_event = runner._build_agent_start_event(state, user_input_for_start, store_agent_start)
    assert start_event.event_type == ExecutionEventType.AGENT_START
    last_stack = await runner.conversation_manager.get_last_agent_stack(
        application_name=application_name, user_id=user_id, conversation_id=conversation_id
    )
    assert last_stack.agent_name == "agent_a"

    change_event = BaseEvent(
        event_type=ExecutionEventType.AGENT_START,
        content=[],
        author="agent_a",
        action=AgentAction(name="agent_b", parameters={}, thought="Starting agent_b"),
    )

    context = ContextBuilder.create_context(
        state=state,
        user_input=BaseEvent(
            event_type=ExecutionEventType.USER_INPUT,
            content=[Part(text="Test user input")],
            author="user",
        ),
    )

    new_agent, new_context, _ = await runner._handle_agent_transition(
        state=state, event=change_event, current_context=context
    )

    assert new_agent == agent_b
    assert new_context is not None
    assert new_context.agent_input.event_type == ExecutionEventType.AGENT_START
    assert new_context.conversation_id == conversation_id
    assert new_context.interaction_id is not None

    # check new agent is in the stack
    last_stack = await runner.conversation_manager.get_last_agent_stack(
        application_name=application_name, user_id=user_id, conversation_id=conversation_id
    )
    assert last_stack.agent_name == "agent_b"

    # check stack
    conversation_info: ConversationInfo = await runner.conversation_manager.get_conversation(
        application_name=application_name,
        user_id=user_id,
        conversation_id=conversation_id,
    )
    assert len(conversation_info.agents_name_stack) == 2
    assert conversation_info.agents_name_stack[0].agent_name == "agent_a"


@pytest.mark.asyncio
async def test__handle_agent_transition_propagates_input_metadata_not_run_config(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """A sub-agent AGENT_CALL transition copies the parent's input metadata (e.g.
    user_info) onto the sub-agent AGENT_START — and does NOT pull from run_config."""
    agent_b = Agent(
        name="agent_b",
        description="agent_b description",
        instruction="agent_b instruction",
        llm_provider=llm_provider,
    )
    agent_a = Agent(
        name="agent_a",
        description="agent_a description",
        instruction="agent_a instruction",
        llm_provider=llm_provider,
        sub_agents=[agent_b],
    )

    state: RunnerState = await runner._get_state(
        application_name="test_app",
        conversation_id="test_convo",
        user_id="test_user",
        root=agent_a,
        entrypoint=None,
    )
    # A secret in run_config must NOT leak onto the sub-agent start.
    state.run_config = {"x-domyn-authorization": "Bearer SECRET"}

    # Parent context whose agent_input (the root AGENT_START) carries caller metadata.
    parent_input = BaseEvent(
        event_type=ExecutionEventType.AGENT_START,
        author="agent_a",
        content=[Part(text="parent task")],
        metadata={"user_info": {"user_id": "u-42"}},
    )
    context = ContextBuilder.create_context(state=state, user_input=parent_input)

    call_event = BaseEvent(
        event_type=ExecutionEventType.AGENT_CALL,
        author="agent_a",
        content=[],
        action=AgentAction(name="agent_b", parameters={"task": "do it"}, thought="handoff"),
    )

    new_agent, _new_context, agent_start_event = await runner._handle_agent_transition(
        state=state, event=call_event, current_context=context
    )

    assert new_agent == agent_b
    assert agent_start_event is not None
    assert agent_start_event.event_type == ExecutionEventType.AGENT_START
    # user_info propagated from the parent's input...
    assert agent_start_event.metadata.get("user_info") == {"user_id": "u-42"}
    # ...but the run_config secret did not.
    assert "x-domyn-authorization" not in agent_start_event.metadata


# ===========================================================================
# run_on_event_hooks — dict modified_event casting
# ===========================================================================


def _make_original_event(
    author: str = "agent",
    interaction_id: str = "iid_original",
    turn_id: str = "tid_original",
) -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author=author,
        interaction_id=interaction_id,
        turn_id=turn_id,
    )


def _make_hook(fn) -> FunctionHook:
    return FunctionHook(fun=fn, trigger=HookPhase.ON_AGENT_EVENT, type=HookType.UPDATER)


async def _collect_on_event(runner: Runner, hook: FunctionHook, event: BaseEvent):
    results = []
    noop_state = MagicMock(metadata={})
    async for execution in runner.run_on_event_hooks([hook], event, context=None, state=noop_state):
        results.append(execution)
    return results


@pytest.mark.asyncio
async def test_on_event_dict_all_fields_explicit(runner: Runner) -> None:
    """Dict with all three defaulted fields explicit — those values are used, not the originals."""

    def hook_fn(event: BaseEvent, context, **_):
        return {
            "event_type": ExecutionEventType.RESPONSE,
            "author": "new_author",
            "turn_id": "new_turn",
            "interaction_id": "new_iid",
        }

    original = _make_original_event()
    executions = await _collect_on_event(runner, _make_hook(hook_fn), original)

    final_event = executions[-1].current_event
    assert isinstance(final_event, BaseEvent)
    assert final_event.event_type == ExecutionEventType.RESPONSE
    assert final_event.author == "new_author"
    assert final_event.turn_id == "new_turn"
    assert final_event.interaction_id == "new_iid"


@pytest.mark.asyncio
async def test_on_event_dict_missing_author_defaults_to_original(runner: Runner) -> None:
    """Dict without 'author' inherits it from the original event."""

    def hook_fn(event: BaseEvent, context, **_):
        return {"event_type": ExecutionEventType.RESPONSE}

    original = _make_original_event(author="original_agent", interaction_id="iid1", turn_id="tid1")
    executions = await _collect_on_event(runner, _make_hook(hook_fn), original)

    final_event = executions[-1].current_event
    assert isinstance(final_event, BaseEvent)
    assert final_event.author == "original_agent"
    assert final_event.interaction_id == "iid1"
    assert final_event.turn_id == "tid1"


@pytest.mark.asyncio
async def test_on_event_dict_missing_turn_id_defaults_to_original(runner: Runner) -> None:
    """Dict with explicit author but no turn_id/interaction_id inherits them from original event."""

    def hook_fn(event: BaseEvent, context, **_):
        return {"event_type": ExecutionEventType.RESPONSE, "author": "hook_author"}

    original = _make_original_event(interaction_id="iid_orig", turn_id="tid_orig")
    executions = await _collect_on_event(runner, _make_hook(hook_fn), original)

    # Conversion and default-filling happens in process_result; runner just sees a BaseEvent
    final_event = executions[-1].current_event
    assert isinstance(final_event, BaseEvent)
    assert final_event.author == "hook_author"
    assert final_event.interaction_id == "iid_orig"
    assert final_event.turn_id == "tid_orig"


@pytest.mark.asyncio
async def test_on_event_dict_produces_base_event_instance(runner: Runner) -> None:
    """After runner processing, current_event is always a BaseEvent, never a raw dict."""

    def hook_fn(event: BaseEvent, context, **_):
        return {"event_type": ExecutionEventType.LLM_END, "author": "llm"}

    original = _make_original_event()
    executions = await _collect_on_event(runner, _make_hook(hook_fn), original)

    final_event = executions[-1].current_event
    assert type(final_event) is BaseEvent


@pytest.mark.asyncio
async def test_on_event_none_return_preserves_original_event(runner: Runner) -> None:
    """Hook returning None passes the original event through unchanged."""

    def hook_fn(event: BaseEvent, context, **_):
        return None

    original = _make_original_event(author="keep_me", turn_id="keep_turn")
    executions = await _collect_on_event(runner, _make_hook(hook_fn), original)

    final_event = executions[-1].current_event
    assert final_event.author == "keep_me"
    assert final_event.turn_id == "keep_turn"
    assert final_event.event_type == ExecutionEventType.USER_INPUT


@pytest.mark.asyncio
async def test_on_event_base_event_return_used_as_is(runner: Runner) -> None:
    """Hook returning a BaseEvent directly uses its fields without applying defaults."""

    def hook_fn(event: BaseEvent, context, **_):
        return BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="direct_author",
            turn_id="direct_turn",
            interaction_id="direct_iid",
        )

    original = _make_original_event(author="original", turn_id="orig_turn")
    executions = await _collect_on_event(runner, _make_hook(hook_fn), original)

    final_event = executions[-1].current_event
    assert final_event.author == "direct_author"
    assert final_event.turn_id == "direct_turn"
    assert final_event.interaction_id == "direct_iid"


# ===========================================================================
# on_agent_event hook — redaction persistence across the execution pipeline
# ===========================================================================


@pytest.mark.asyncio
async def test_on_event_user_input_redaction_persists(runner: Runner) -> None:
    """A USER_INPUT redaction hook yields a redacted current_event, which is what
    runner.run() reassigns as user_input before storing and passing to the agent loop."""

    def redact_fn(event: BaseEvent, context, **_) -> BaseEvent | None:
        if event.event_type != ExecutionEventType.USER_INPUT:
            return None
        modified = event.model_copy(deep=True)
        modified.content = [
            Part(text=p.text.replace("secret_token", "[REDACTED]")) if p.text else p
            for p in modified.content
        ]
        return modified

    hook = _make_hook(redact_fn)
    original = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="my secret_token is abc123")],
    )

    executions = await _collect_on_event(runner, hook, original)

    # current_event is what runner.run() assigns back as user_input via `user_input = on_exec.current_event`
    redacted = executions[-1].current_event
    assert redacted.event_type == ExecutionEventType.USER_INPUT
    assert "[REDACTED]" in redacted.content[0].text
    assert "secret_token" not in redacted.content[0].text
    # Original event must not be mutated
    assert original.content[0].text == "my secret_token is abc123"


@pytest.mark.asyncio
async def test_on_event_agent_call_task_redaction_persists_across_handoff(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """A hook on the calling agent that redacts the task in an AGENT_CALL event produces a
    modified event that is passed to _handle_agent_transition, so the sub-agent's
    context.agent_input carries the redacted task — not the original one.

    After the AGENT_CALL/AGENT_START split, the calling agent emits AGENT_CALL (with the task
    in action.parameters); _handle_agent_transition creates the runner-generated AGENT_START
    from those parameters, so the hook must modify action.parameters to affect what the
    child agent receives.
    """

    def redact_task_fn(event: BaseEvent, context, **_) -> BaseEvent | None:
        if event.event_type != ExecutionEventType.AGENT_CALL:
            return None
        modified = event.model_copy(deep=True)
        if modified.action and modified.action.parameters:
            modified.action.parameters = {
                k: v.replace("confidential", "[REDACTED]") if isinstance(v, str) else v
                for k, v in modified.action.parameters.items()
            }
        return modified

    hook = _make_hook(redact_task_fn)

    child_agent = Agent(
        name="child_agent",
        description="child",
        instruction="child instruction",
        llm_provider=llm_provider,
    )
    parent_agent = Agent(
        name="parent_agent",
        description="parent",
        instruction="parent instruction",
        llm_provider=llm_provider,
        sub_agents=[child_agent],
        hooks=[hook],
    )

    state = await runner._get_state(
        application_name="test_app",
        conversation_id="test_conv_agent_call_redact",
        user_id="test_user",
        root=parent_agent,
        entrypoint=None,
    )

    # AGENT_CALL event the parent emits when delegating to child (task is in action.parameters)
    agent_call_event = BaseEvent(
        event_type=ExecutionEventType.AGENT_CALL,
        author="parent_agent",
        content=[],
        action=AgentAction(
            name="child_agent",
            thought="Delegating task",
            parameters={"task": "Perform the confidential analysis task"},
        ),
    )

    # Run the hook as the runner does before the match statement
    executions = []
    async for on_exec in runner.run_on_event_hooks(
        parent_agent.on_event_hooks,
        agent_call_event,
        context=None,
        agent_name=parent_agent.name,
        state=state,
    ):
        executions.append(on_exec)

    modified_event = executions[-1].current_event
    assert "[REDACTED]" in modified_event.action.parameters["task"]
    assert "confidential" not in modified_event.action.parameters["task"]

    user_input = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="original user request")],
    )
    context = ContextBuilder.create_context(state=state, user_input=user_input)

    # Transition to child with the modified event — this is what runner._execute_agent_loop does
    new_agent, new_context, agent_start_event = await runner._handle_agent_transition(
        state=state, event=modified_event, current_context=context
    )

    assert new_agent == child_agent
    # _handle_agent_transition creates a runner-generated AGENT_START from action.parameters
    assert agent_start_event is not None
    assert agent_start_event.event_type == ExecutionEventType.AGENT_START
    assert new_context.agent_input.event_type == ExecutionEventType.AGENT_START
    # The task the child agent receives has the redacted parameters
    assert "[REDACTED]" in new_context.agent_input.content[0].text
    assert "confidential" not in new_context.agent_input.content[0].text


@pytest.mark.asyncio
async def test_on_event_agent_end_use_stop_redaction_returned_to_calling_agent(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """A hook on the sub-agent that redacts an AGENT_END event (emitted when the sub-agent
    stops via use_stop) produces a modified event whose redacted content is what the calling
    (parent) agent receives as its new context.agent_input."""

    def redact_result_fn(event: BaseEvent, context, **_) -> BaseEvent | None:
        if event.event_type != ExecutionEventType.AGENT_END:
            return None
        modified = event.model_copy(deep=True)
        modified.content = [
            Part(text=p.text.replace("sensitive_result", "[REDACTED]")) if p.text else p
            for p in modified.content
        ]
        return modified

    hook = _make_hook(redact_result_fn)

    child_agent = Agent(
        name="child_agent",
        description="child",
        instruction="child instruction",
        llm_provider=llm_provider,
        hooks=[hook],  # hook on child fires when child emits AGENT_END via use_stop
    )
    parent_agent = Agent(
        name="parent_agent",
        description="parent",
        instruction="parent instruction",
        llm_provider=llm_provider,
        sub_agents=[child_agent],
    )

    state = await runner._get_state(
        application_name="test_app",
        conversation_id="test_conv_agent_end_redact",
        user_id="test_user",
        root=parent_agent,
        entrypoint=None,
    )

    user_input = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="do something")],
    )

    # Simulate conversation start — pushes parent onto stack
    await runner._setup_conversation_turn(state, user_input)

    # Simulate parent → child handoff by pushing child onto stack
    await runner.conversation_manager.add_agent_to_stack(
        application_name=state.application_name,
        user_id=state.user_id,
        conversation_id=state.conversation_id,
        agent_name=child_agent.name,
        interaction_id="child_interaction_1",
        turn_id="turn_1",
    )
    state.current_agent = child_agent

    # AGENT_END emitted by child via use_stop with a sensitive result in its content
    agent_end_event = BaseEvent(
        event_type=ExecutionEventType.AGENT_END,
        author="child_agent",
        content=[Part(text="The sensitive_result is: top secret data")],
        action=ToolAction(name="stop", parameters={}, thought="Done"),
    )

    # Run on_event hooks of child — as runner._execute_agent_loop does for AGENT_END
    executions = []
    async for on_exec in runner.run_on_event_hooks(
        child_agent.on_event_hooks,
        agent_end_event,
        context=None,
        agent_name=child_agent.name,
        state=state,
    ):
        executions.append(on_exec)

    modified_event = executions[-1].current_event
    assert "[REDACTED]" in modified_event.content[0].text
    assert "sensitive_result" not in modified_event.content[0].text

    # Perform the agent transition — runner returns control to parent with the modified event
    context = ContextBuilder.create_context(state=state, user_input=user_input)
    new_agent, new_context, _ = await runner._handle_agent_transition(
        state=state, event=modified_event, current_context=context
    )

    # Parent agent resumes; its context.agent_input is the redacted AGENT_END
    assert new_agent == parent_agent
    assert new_context.agent_input.event_type == ExecutionEventType.AGENT_END
    assert "[REDACTED]" in new_context.agent_input.content[0].text
    assert "sensitive_result" not in new_context.agent_input.content[0].text


@pytest.mark.asyncio
async def test_on_event_agent_start_modified_agent_call_unchanged_standard_visibility(
    runner: Runner, llm_provider: BaseLLMProvider
) -> None:
    """A hook on a STANDARD-visibility sub-agent modifies the runner-created AGENT_START
    (removing a private message from its content) while leaving the original AGENT_CALL event
    untouched.  Because the sub-agent has STANDARD visibility, the AGENT_CALL authored by the
    parent never appears in the sub-agent's conversation_history regardless of redaction.

    This verifies two independent guarantees:
    1. on_agent_event hooks fire on AGENT_START, not on AGENT_CALL.
    2. STANDARD visibility excludes parent-authored events from the sub-agent's history.
    """

    private = "private_message_xyz"

    def strip_private_from_start(event: BaseEvent, context, **_) -> BaseEvent | None:
        if event.event_type != ExecutionEventType.AGENT_START:
            return None
        modified = event.model_copy(deep=True)
        modified.content = [
            Part(text=p.text.replace(private, "[REMOVED]")) if p.text else p
            for p in modified.content
        ]
        return modified

    hook = _make_hook(strip_private_from_start)

    sub_agent = Agent(
        name="sub_agent",
        description="sub",
        instruction="sub instruction",
        llm_provider=llm_provider,
        hooks=[hook],
        # STANDARD is the default, but make it explicit for clarity
        context_visibility=ContextVisibility.STANDARD,
    )
    parent_agent = Agent(
        name="parent_agent",
        description="parent",
        instruction="parent instruction",
        llm_provider=llm_provider,
        sub_agents=[sub_agent],
    )

    assert sub_agent.context_visibility == ContextVisibility.STANDARD

    state = await runner._get_state(
        application_name="test_app",
        conversation_id="test_conv_start_redact_visibility",
        user_id="test_user",
        root=parent_agent,
        entrypoint=None,
    )

    user_input = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="start")],
    )
    await runner._setup_conversation_turn(state, user_input)

    # AGENT_CALL the parent emits — task carries the private message
    agent_call_event = BaseEvent(
        event_type=ExecutionEventType.AGENT_CALL,
        author="parent_agent",
        content=[],
        action=AgentAction(
            name="sub_agent",
            thought="Delegating",
            parameters={"task": f"Please handle this {private} carefully"},
        ),
    )

    # The runner stores every non-partial event before the match statement (see
    # `_execute_agent_loop`).
    # Add the AGENT_CALL to history so we can later verify visibility filtering.
    await runner.conversation_manager.add_conversation_event(
        state.application_name, state.user_id, state.conversation_id, agent_call_event
    )
    await runner._refresh_state(state)

    context = ContextBuilder.create_context(state=state, user_input=user_input)

    # _handle_agent_transition creates the runner-generated AGENT_START from action.parameters
    new_agent, new_context, agent_start_event = await runner._handle_agent_transition(
        state=state, event=agent_call_event, current_context=context
    )

    assert new_agent == sub_agent
    assert agent_start_event is not None
    assert agent_start_event.event_type == ExecutionEventType.AGENT_START
    # Runner encodes parameters into the AGENT_START content
    assert private in agent_start_event.content[0].text

    # Run the sub-agent's on_event hooks on the AGENT_START (as the runner does after transition)
    executions = []
    async for on_exec in runner.run_on_event_hooks(
        sub_agent.on_event_hooks,
        agent_start_event,
        new_context,
        agent_name=sub_agent.name,
        state=state,
    ):
        executions.append(on_exec)
        agent_start_event = on_exec.current_event

    # 1. Hook modified AGENT_START — private message is gone
    assert private not in agent_start_event.content[0].text
    assert "[REMOVED]" in agent_start_event.content[0].text

    # 2. Hook did NOT touch the original AGENT_CALL
    assert private in agent_call_event.action.parameters["task"]

    # 3. STANDARD visibility: sub_agent's conversation_history excludes events authored by parent.
    #    The AGENT_CALL (author="parent_agent") must not be present, even though it is in the
    #    full events_history that the runner persists.
    assert not any(
        e.event_type == ExecutionEventType.AGENT_CALL and e.author == "parent_agent"
        for e in new_context.conversation_history
    )
