"""Tests for `FullMessageContext` history filtering.

Cover the visibility/handoff split: history events from agents in the
broader visibility scope (driven by ``context_visibility``) must remain
visible in the prompt context even when those agents are not in the
current agent's callable set (``sub_agents``).
"""

import pytest

from domyn.agents.agents import Agent, ContextVisibility
from domyn.agents.conversation_manager import ConversationInfo
from domyn.agents.core import (
    METADATA_CONVERTED_TO_AGENT_END,
    AgentAction,
    BaseEvent,
    ExecutionEventType,
    Part,
    ToolAction,
)
from domyn.agents.llm import BaseLLMProvider
from domyn.agents.planner_strategy.context_management.full_messages import FullMessageContext
from domyn.agents.planner_strategy.prompting.structural_tags_prompting import (
    StructuralTagMessagePrompting,
)
from domyn.agents.runner import ContextBuilder, RunnerState


def _user_input(text: str, interaction_id: str | None = None) -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        content=[Part(text=text)],
        author="user",
        interaction_id=interaction_id,
    )


def _tool_end(author: str, observation: str, interaction_id: str | None = None) -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.TOOL_END,
        author=author,
        content=[],
        action=ToolAction(
            name="some_tool",
            parameters={},
            observation=observation,
            thought=None,
            message=None,
        ),
        interaction_id=interaction_id,
    )


def _agent_start(
    callee: str, task: str = "do the thing", interaction_id: str | None = None
) -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.AGENT_START,
        content=[Part(text=task)],
        author=callee,
        action=AgentAction(name=callee, parameters={}, thought=None, message=None),
        interaction_id=interaction_id,
    )


def _make_state(
    *,
    current_agent: Agent,
    entrypoint_agent: Agent,
    events_history: list[BaseEvent],
    interaction_id: str | None,
) -> RunnerState:
    return RunnerState(
        application_name="app",
        user_id="user",
        conversation_id="conv",
        conversation_info=ConversationInfo(
            application_name="app",
            user_id="user",
            conversation_id="conv",
            events_history=events_history,
        ),
        current_agent=current_agent,
        available_agents=[current_agent, entrypoint_agent],
        entrypoint_agent=entrypoint_agent,
        interaction_id=interaction_id,
    )


@pytest.mark.asyncio
async def test_global_subagent_sees_tool_end_from_siblings(llm_provider: BaseLLMProvider):
    """An agent with ``context_visibility=GLOBAL`` and empty ``sub_agents``
    should still see TOOL_END events emitted by other agents in the
    entrypoint's subtree (visibility scope) when its
    ``FullMessageContext`` filter is in narrow mode.

    Regression guard for the visibility/handoff split: handoff_targets
    (callable set) is narrower than the visibility scope used for
    history filtering.
    """

    code_agent = Agent(
        name="Code",
        description="code agent",
        instruction="i",
        llm_provider=llm_provider,
    )
    presentation = Agent(
        name="Presentation",
        description="presentation agent",
        instruction="i",
        llm_provider=llm_provider,
        context_visibility=ContextVisibility.GLOBAL,
        sub_agents=[],
    )
    main = Agent(
        name="Main",
        description="main",
        instruction="i",
        llm_provider=llm_provider,
        sub_agents=[presentation, code_agent],
    )

    interaction_id = "interaction-1"
    code_tool_end = _tool_end(
        author="Code",
        observation="CLEANED_DATA_MARKER",
        interaction_id=interaction_id,
    )
    presentation_start = _agent_start(callee="Presentation", interaction_id=interaction_id)

    state = _make_state(
        current_agent=presentation,
        entrypoint_agent=main,
        events_history=[code_tool_end, presentation_start],
        interaction_id=interaction_id,
    )

    context = ContextBuilder.create_context(state=state, user_input=presentation_start)

    # GLOBAL visibility → Code is in the visibility scope
    assert "Code" in context.visible_agent_names
    # sub_agents=[] → Code is NOT a handoff target (not in callable set)
    assert all(d.name != "Code" for d in context.handoff_targets_description)

    ctx_mgr = FullMessageContext(
        narrow_non_visible_agents=True,
        prompting=StructuralTagMessagePrompting(),
        name="full_message_context",
    )
    messages = await ctx_mgr.get_context(context=context, agent=presentation)

    rendered = "\n".join(m.content for m in messages if m.content)
    assert "CLEANED_DATA_MARKER" in rendered, (
        "Code's TOOL_END must remain visible to Presentation (GLOBAL visibility), "
        "even though Code is not a callable handoff target."
    )


@pytest.mark.asyncio
async def test_standard_subagent_drops_tool_end_from_siblings(llm_provider: BaseLLMProvider):
    """An agent with ``context_visibility=STANDARD`` and empty ``sub_agents``
    has no visible peers; TOOL_END events from siblings must be filtered
    out of the prompt context when ``narrow_non_visible_agents=True``."""

    code_agent = Agent(
        name="Code",
        description="code agent",
        instruction="i",
        llm_provider=llm_provider,
    )
    presentation = Agent(
        name="Presentation",
        description="presentation agent",
        instruction="i",
        llm_provider=llm_provider,
        context_visibility=ContextVisibility.STANDARD,
        sub_agents=[],
    )
    main = Agent(
        name="Main",
        description="main",
        instruction="i",
        llm_provider=llm_provider,
        sub_agents=[presentation, code_agent],
    )

    interaction_id = "interaction-2"
    code_tool_end = _tool_end(
        author="Code",
        observation="CLEANED_DATA_MARKER",
        interaction_id=interaction_id,
    )
    presentation_start = _agent_start(callee="Presentation", interaction_id=interaction_id)

    state = _make_state(
        current_agent=presentation,
        entrypoint_agent=main,
        events_history=[code_tool_end, presentation_start],
        interaction_id=interaction_id,
    )

    context = ContextBuilder.create_context(state=state, user_input=presentation_start)

    assert "Code" not in context.visible_agent_names

    ctx_mgr = FullMessageContext(
        narrow_non_visible_agents=True,
        prompting=StructuralTagMessagePrompting(),
        name="full_message_context",
    )
    messages = await ctx_mgr.get_context(context=context, agent=presentation)

    rendered = "\n".join(m.content for m in messages if m.content)
    assert "CLEANED_DATA_MARKER" not in rendered, (
        "STANDARD visibility excludes Code from the visible scope, so its "
        "TOOL_END must be filtered out by the context manager."
    )


@pytest.mark.asyncio
async def test_narrow_false_keeps_sibling_tool_end_outside_visible_names(
    llm_provider: BaseLLMProvider,
):
    """``narrow_non_visible_agents=False`` short-circuits the
    ``visible_agent_names`` filter and renders every TOOL_END unconditionally
    — even from authors outside the visibility scope.

    Manual ``InteractionContext`` construction is required to isolate the
    flag: the runner-level filter in ``ContextBuilder.create_context``
    would otherwise strip sibling events from ``conversation_history``
    before they reach ``FullMessageContext``."""

    presentation = Agent(
        name="Presentation",
        description="presentation agent",
        instruction="i",
        llm_provider=llm_provider,
    )

    interaction_id = "interaction-narrow-false"
    code_tool_end = _tool_end(
        author="Code",
        observation="CLEANED_DATA_MARKER",
        interaction_id=interaction_id,
    )
    presentation_start = _agent_start(callee="Presentation", interaction_id=interaction_id)

    from domyn.agents.core import InteractionContext

    # Code's event survives in conversation_history, but Code is NOT in
    # `visible_agent_names` — the scenario that exercises the flag.
    context = InteractionContext(
        conversation_id="conv",
        user_id="user",
        interaction_id=interaction_id,
        agent_input=presentation_start,
        conversation_history=[code_tool_end, presentation_start],
        interaction_history=[],
        visible_agent_names=[],
    )

    ctx_mgr_narrow = FullMessageContext(
        narrow_non_visible_agents=True,
        prompting=StructuralTagMessagePrompting(),
        name="full_message_context",
    )
    rendered_narrow = "\n".join(
        m.content
        for m in await ctx_mgr_narrow.get_context(context=context, agent=presentation)
        if m.content
    )
    assert "CLEANED_DATA_MARKER" not in rendered_narrow

    ctx_mgr_wide = FullMessageContext(
        narrow_non_visible_agents=False,
        prompting=StructuralTagMessagePrompting(),
        name="full_message_context",
    )
    rendered_wide = "\n".join(
        m.content
        for m in await ctx_mgr_wide.get_context(context=context, agent=presentation)
        if m.content
    )
    assert "CLEANED_DATA_MARKER" in rendered_wide, (
        "narrow_non_visible_agents=False must disable the author filter and "
        "render every TOOL_END regardless of the visibility scope."
    )


@pytest.mark.asyncio
async def test_sibling_agent_start_not_filtered_by_visibility(
    llm_provider: BaseLLMProvider,
):
    """The ``visible_agent_names`` filter in ``FullMessageContext`` only
    gates ``TOOL_END`` events. ``AGENT_START`` events from authors outside
    the visibility scope are surfaced unconditionally, documenting the
    asymmetry of the filter (and acting as a regression guard if the gate
    is ever widened by mistake).

    Constructed manually for the same reason as
    ``test_narrow_false_keeps_sibling_tool_end_outside_visible_names``:
    the upstream filter would otherwise strip the sibling event."""

    presentation = Agent(
        name="Presentation",
        description="presentation agent",
        instruction="i",
        llm_provider=llm_provider,
    )

    interaction_id = "interaction-agent-start"
    code_start = _agent_start(
        callee="Code", task="SIBLING_START_MARKER", interaction_id=interaction_id
    )
    presentation_start = _agent_start(callee="Presentation", interaction_id=interaction_id)

    from domyn.agents.core import InteractionContext

    context = InteractionContext(
        conversation_id="conv",
        user_id="user",
        interaction_id=interaction_id,
        agent_input=presentation_start,
        conversation_history=[code_start, presentation_start],
        interaction_history=[],
        visible_agent_names=[],
    )

    ctx_mgr = FullMessageContext(
        narrow_non_visible_agents=True,
        prompting=StructuralTagMessagePrompting(),
        name="full_message_context",
    )
    messages = await ctx_mgr.get_context(context=context, agent=presentation)

    rendered = "\n".join(m.content for m in messages if m.content)
    assert "SIBLING_START_MARKER" in rendered, (
        "AGENT_START events are not gated by `visible_agent_names`; the "
        "sibling's start event must remain in Presentation's prompt context."
    )


@pytest.mark.asyncio
async def test_empty_visible_agent_names_drops_all_non_self_tool_end(
    llm_provider: BaseLLMProvider,
):
    """Edge case: when ``InteractionContext`` is built directly with the
    default ``visible_agent_names=[]`` (e.g. legacy callers, manual
    construction in tests), ``FullMessageContext`` only retains TOOL_END
    events authored by the current agent. Documents the default-empty
    contract introduced by the visibility/handoff split."""

    presentation = Agent(
        name="Presentation",
        description="presentation agent",
        instruction="i",
        llm_provider=llm_provider,
    )

    interaction_id = "interaction-empty-visible"
    own_tool_end = _tool_end(
        author="Presentation",
        observation="OWN_TOOL_MARKER",
        interaction_id=interaction_id,
    )
    sibling_tool_end = _tool_end(
        author="Code",
        observation="SIBLING_TOOL_MARKER",
        interaction_id=interaction_id,
    )
    presentation_start = _agent_start(callee="Presentation", interaction_id=interaction_id)

    # Manually construct InteractionContext with default empty
    # visible_agent_names (bypassing ContextBuilder).
    from domyn.agents.core import InteractionContext

    context = InteractionContext(
        conversation_id="conv",
        user_id="user",
        interaction_id=interaction_id,
        agent_input=presentation_start,
        conversation_history=[own_tool_end, sibling_tool_end, presentation_start],
        interaction_history=[],
    )

    assert context.visible_agent_names == []

    ctx_mgr = FullMessageContext(
        narrow_non_visible_agents=True,
        prompting=StructuralTagMessagePrompting(),
        name="full_message_context",
    )
    messages = await ctx_mgr.get_context(context=context, agent=presentation)

    rendered = "\n".join(m.content for m in messages if m.content)
    assert "OWN_TOOL_MARKER" in rendered
    assert "SIBLING_TOOL_MARKER" not in rendered, (
        "With default empty visible_agent_names, only the current agent's "
        "own TOOL_END events survive the filter."
    )


@pytest.mark.asyncio
async def test_converted_subagent_response_skipped(llm_provider: BaseLLMProvider):
    """A RESPONSE marked with METADATA_CONVERTED_TO_AGENT_END was converted by
    the runner into a synthesized AGENT_END: the marked event must not render
    as an assistant answer (the AGENT_END carries the content instead)."""

    presentation = Agent(
        name="Presentation",
        description="presentation agent",
        instruction="i",
        llm_provider=llm_provider,
    )

    interaction_id = "interaction-converted-response"
    presentation_start = _agent_start(callee="Presentation", interaction_id=interaction_id)

    def _response(marked: bool) -> BaseEvent:
        return BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="Presentation",
            content=[Part(text="BOUNCED_ANSWER_MARKER")],
            interaction_id=interaction_id,
            metadata={METADATA_CONVERTED_TO_AGENT_END: True} if marked else {},
        )

    from domyn.agents.core import InteractionContext

    ctx_mgr = FullMessageContext(
        narrow_non_visible_agents=True,
        prompting=StructuralTagMessagePrompting(),
        name="full_message_context",
    )

    def _context(response: BaseEvent) -> InteractionContext:
        return InteractionContext(
            conversation_id="conv",
            user_id="user",
            interaction_id=interaction_id,
            agent_input=presentation_start,
            conversation_history=[presentation_start, response],
            interaction_history=[],
            visible_agent_names=[],
        )

    rendered_marked = "\n".join(
        m.content
        for m in await ctx_mgr.get_context(
            context=_context(_response(marked=True)), agent=presentation
        )
        if m.content
    )
    assert "BOUNCED_ANSWER_MARKER" not in rendered_marked, (
        "A converted sub-agent RESPONSE must be skipped by the context manager."
    )

    rendered_unmarked = "\n".join(
        m.content
        for m in await ctx_mgr.get_context(
            context=_context(_response(marked=False)), agent=presentation
        )
        if m.content
    )
    assert "BOUNCED_ANSWER_MARKER" in rendered_unmarked, (
        "An unmarked RESPONSE must keep rendering as before."
    )
