"""Tests for ContextBuilder scoping rules that prevent cross-scope leakage."""

import pytest

from domyn.agents.agents import Agent, ContextVisibility, HandoffScope
from domyn.agents.conversation_manager import ConversationInfo
from domyn.agents.core import (
    METADATA_EXTERNAL_PART_AGENT_VISIBLE,
    AgentAction,
    BaseEvent,
    ExecutionEventType,
    Part,
)
from domyn.agents.llm import BaseLLMProvider
from domyn.agents.registry import get_agent_registry
from domyn.agents.runner import ContextBuilder, RunnerState


def _user_input(text: str, interaction_id: str | None = None) -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        content=[Part(text=text)],
        author="user",
        interaction_id=interaction_id,
    )


def _agent_start(
    callee: str,
    *,
    task: str = "",
    interaction_id: str | None = None,
) -> BaseEvent:
    """Build an AGENT_START as the runner emits it after handling an AGENT_CALL:
    author is the called agent, content carries the task, action.parameters is empty."""
    return BaseEvent(
        event_type=ExecutionEventType.AGENT_START,
        content=[Part(text=task)] if task else [],
        author=callee,
        action=AgentAction(
            name=callee,
            parameters={},
            thought=None,
            message=None,
        ),
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
async def test_entrypoint_sees_user_inputs_across_interactions(
    llm_provider: BaseLLMProvider,
):
    """On a new turn the entrypoint gets a fresh interaction_id but must still
    see prior USER_INPUT events from earlier interactions (multi-turn memory)."""

    entrypoint = Agent(
        name="MainAgent",
        description="main",
        instruction="main",
        llm_provider=llm_provider,
    )

    old_interaction = "interaction-old"
    new_interaction = "interaction-new"
    old_user_input = _user_input("question from turn 1", old_interaction)
    new_user_input = _user_input("question from turn 2", new_interaction)

    state = _make_state(
        current_agent=entrypoint,
        entrypoint_agent=entrypoint,
        events_history=[old_user_input, new_user_input],
        interaction_id=new_interaction,
    )

    context = ContextBuilder.create_context(state=state, user_input=new_user_input)

    user_inputs = [
        e for e in context.conversation_history if e.event_type == ExecutionEventType.USER_INPUT
    ]
    assert old_user_input in user_inputs
    assert new_user_input in user_inputs


@pytest.mark.asyncio
async def test_global_subagent_sees_user_inputs_across_interactions(
    llm_provider: BaseLLMProvider,
):
    """A GLOBAL-visibility sub-agent should not have its user-input context
    scoped to the current interaction_id."""

    global_sub = Agent(
        name="GlobalSub",
        description="global sub",
        instruction="global sub",
        llm_provider=llm_provider,
        context_visibility=ContextVisibility.GLOBAL,
    )
    entrypoint = Agent(
        name="MainAgent",
        description="main",
        instruction="main",
        llm_provider=llm_provider,
        sub_agents=[global_sub],
    )

    old_interaction = "interaction-old"
    new_interaction = "interaction-new"
    old_user_input = _user_input("question from turn 1", old_interaction)
    new_user_input = _user_input("question from turn 2", new_interaction)
    callee_start = _agent_start(
        callee="GlobalSub",
        task="t",
        interaction_id=new_interaction,
    )

    state = _make_state(
        current_agent=global_sub,
        entrypoint_agent=entrypoint,
        events_history=[old_user_input, new_user_input, callee_start],
        interaction_id=new_interaction,
    )

    context = ContextBuilder.create_context(state=state, user_input=callee_start)

    user_inputs = [
        e for e in context.conversation_history if e.event_type == ExecutionEventType.USER_INPUT
    ]
    assert old_user_input in user_inputs
    assert new_user_input in user_inputs


@pytest.mark.asyncio
async def test_standard_subagent_user_inputs_scoped_to_current_interaction(
    llm_provider: BaseLLMProvider,
):
    """A STANDARD-visibility sub-agent must only see USER_INPUT events
    belonging to the current interaction."""

    sub = Agent(
        name="SubAgent",
        description="sub",
        instruction="sub",
        llm_provider=llm_provider,
        context_visibility=ContextVisibility.STANDARD,
    )
    entrypoint = Agent(
        name="MainAgent",
        description="main",
        instruction="main",
        llm_provider=llm_provider,
        sub_agents=[sub],
    )

    old_interaction = "interaction-old"
    new_interaction = "interaction-new"
    old_user_input = _user_input("question from turn 1", old_interaction)
    new_user_input = _user_input("question from turn 2", new_interaction)
    callee_start = _agent_start(
        callee="SubAgent",
        task="t",
        interaction_id=new_interaction,
    )

    state = _make_state(
        current_agent=sub,
        entrypoint_agent=entrypoint,
        events_history=[old_user_input, new_user_input, callee_start],
        interaction_id=new_interaction,
    )

    context = ContextBuilder.create_context(state=state, user_input=callee_start)

    user_inputs = [
        e for e in context.conversation_history if e.event_type == ExecutionEventType.USER_INPUT
    ]
    assert old_user_input not in user_inputs
    assert new_user_input in user_inputs


@pytest.mark.asyncio
async def test_global_with_narrow_explicit_sub_agents(
    llm_provider: BaseLLMProvider,
):
    """``sub_agents`` is honored verbatim as the callable set even when the
    agent has ``context_visibility=GLOBAL`` and the subtree contains more
    agents.

    Documents the asymmetry introduced by the visibility/handoff split:
    visibility scope can be broader than the callable set, and an agent
    can deliberately narrow what it may invoke without changing what it
    sees in the conversation."""

    chart = Agent(
        name="Chart",
        description="chart agent",
        instruction="i",
        llm_provider=llm_provider,
    )
    code = Agent(
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
        sub_agents=[chart],
    )
    entrypoint = Agent(
        name="MainAgent",
        description="main",
        instruction="main",
        llm_provider=llm_provider,
        sub_agents=[presentation, code],
    )

    state = _make_state(
        current_agent=presentation,
        entrypoint_agent=entrypoint,
        events_history=[],
        interaction_id="interaction-narrow",
    )
    context = ContextBuilder.create_context(
        state=state,
        user_input=_user_input("draw it", interaction_id="interaction-narrow"),
    )

    # Callable set surfaced to the LLM matches sub_agents exactly.
    callable_names = {d.name for d in context.handoff_targets_description}
    assert callable_names == {"Chart"}

    # Visibility scope reaches the broader subtree (Code is a sibling).
    assert "Code" in context.visible_agent_names
    assert "Chart" in context.visible_agent_names

    # The visibility scope is a strict superset of the callable set,
    # guarding the intent of the split (sub_agents ⊆ visibility scope).
    assert set(context.visible_agent_names) > callable_names


# ---------------------------------------------------------------------------
# `_get_handoff_targets` unit tests — the callable (EXECUTE) set is driven by
# ``handoff_scope`` (default EXPLICIT = the agent's explicit ``sub_agents``
# list). DIRECT_PEERS/GLOBAL resolve the callable set against the live
# entrypoint tree at runtime. Always orthogonal to ``context_visibility``
# (READ scope).
# ---------------------------------------------------------------------------


def _agent(name: str, llm_provider: BaseLLMProvider, **kwargs) -> Agent:
    return Agent(
        name=name,
        description=name,
        instruction="i",
        llm_provider=llm_provider,
        **kwargs,
    )


@pytest.mark.parametrize(
    "visibility",
    [ContextVisibility.GLOBAL, ContextVisibility.PEERS, ContextVisibility.STANDARD],
)
def test_handoff_targets_ignores_visibility_when_subagents_explicit(
    llm_provider: BaseLLMProvider, visibility: ContextVisibility
):
    """The callable set is the agent's explicit ``sub_agents`` for every
    ``context_visibility`` value. Regression guard against the legacy
    behaviour where GLOBAL/PEERS expanded the callable set to the subtree."""

    chart = _agent("Chart", llm_provider)
    code = _agent("Code", llm_provider)
    current = _agent(
        "Presentation",
        llm_provider,
        context_visibility=visibility,
        sub_agents=[chart],
    )
    entrypoint = _agent("Main", llm_provider, sub_agents=[current, code])

    targets = ContextBuilder._get_handoff_targets(current, entrypoint)

    assert [a.name for a in targets] == ["Chart"]


@pytest.mark.parametrize(
    "visibility",
    [ContextVisibility.GLOBAL, ContextVisibility.PEERS, ContextVisibility.STANDARD],
)
def test_handoff_targets_empty_when_subagents_empty(
    llm_provider: BaseLLMProvider, visibility: ContextVisibility
):
    """An agent with ``sub_agents=[]`` has no callable targets, regardless
    of its ``context_visibility``. Pre-refactor, GLOBAL/PEERS implicitly
    expanded targets to the (peers of the) entrypoint subtree; this test
    locks the new explicit semantics."""

    code = _agent("Code", llm_provider)
    current = _agent(
        "Presentation",
        llm_provider,
        context_visibility=visibility,
        sub_agents=[],
    )
    entrypoint = _agent("Main", llm_provider, sub_agents=[current, code])

    targets = ContextBuilder._get_handoff_targets(current, entrypoint)

    assert targets == []


def test_handoff_targets_resolves_string_refs_via_registry(
    llm_provider: BaseLLMProvider,
):
    """``sub_agents`` may contain string names resolved through the global
    agent registry. ``_get_handoff_targets`` must materialise them to the
    actual agent instances."""

    registry = get_agent_registry()
    chart = _agent("Chart", llm_provider)
    registry.register_agent("Chart", chart)
    try:
        current = _agent("Presentation", llm_provider, sub_agents=["Chart"])
        entrypoint = _agent("Main", llm_provider, sub_agents=[current])

        targets = ContextBuilder._get_handoff_targets(current, entrypoint)

        assert targets == [chart]
    finally:
        registry.clear()


def test_handoff_targets_explicit_default_does_not_reach_siblings(
    llm_provider: BaseLLMProvider,
):
    """The default ``handoff_scope=EXPLICIT`` keeps the callable set bound to
    ``sub_agents`` — a sibling in the tree is NOT reachable. Regression guard
    for the visibility/handoff split: broadening must be opt-in."""

    sibling = _agent("Sibling", llm_provider)
    chart = _agent("Chart", llm_provider)
    current = _agent("Curr", llm_provider, sub_agents=[chart])  # handoff_scope defaults EXPLICIT
    entrypoint = _agent("Main", llm_provider, sub_agents=[current, sibling])

    targets = ContextBuilder._get_handoff_targets(current, entrypoint)

    assert [a.name for a in targets] == ["Chart"]


def test_handoff_targets_direct_peers_resolved_at_runtime(
    llm_provider: BaseLLMProvider,
):
    """``handoff_scope=DIRECT_PEERS`` resolves to DIRECT peers only: the agent's own
    children plus its direct siblings — one level down and sideways, with NO
    recursion into a sibling's subtree. This is the built-in (presentation/
    code) use case: reach user-defined siblings appended per-request, while
    handing off to a peer's entrypoint rather than into its internals."""

    leaf = _agent("Leaf", llm_provider)
    own_child = _agent("OwnChild", llm_provider)
    sibling = _agent("Sibling", llm_provider, sub_agents=[leaf])
    current = _agent(
        "Curr", llm_provider, handoff_scope=HandoffScope.DIRECT_PEERS, sub_agents=[own_child]
    )
    entrypoint = _agent("Main", llm_provider, sub_agents=[current, sibling])

    targets = ContextBuilder._get_handoff_targets(current, entrypoint)

    # Own child + direct siblings (incl. self, excluded downstream). Sibling's
    # child "Leaf" is NOT reachable — no recursion into a sibling's subtree.
    assert {a.name for a in targets} == {"OwnChild", "Curr", "Sibling"}
    assert "Leaf" not in {a.name for a in targets}


def test_handoff_targets_global_returns_entire_subtree(
    llm_provider: BaseLLMProvider,
):
    """``handoff_scope=GLOBAL`` resolves the callable set to the entire
    entrypoint subtree at runtime (including the entrypoint and self)."""

    leaf = _agent("Leaf", llm_provider)
    mid = _agent("Mid", llm_provider, sub_agents=[leaf])
    current = _agent("Curr", llm_provider, handoff_scope=HandoffScope.GLOBAL, sub_agents=[])
    entrypoint = _agent("Main", llm_provider, sub_agents=[current, mid])

    targets = ContextBuilder._get_handoff_targets(current, entrypoint)

    assert {a.name for a in targets} == {"Main", "Curr", "Mid", "Leaf"}


def test_handoff_targets_direct_peers_orphan_falls_back_to_own_children(
    llm_provider: BaseLLMProvider,
):
    """DIRECT_PEERS handoff on an agent not reachable from the entrypoint (no parent)
    falls back to the agent's own direct children — there are no siblings to
    add, and the direct-peers rule does not recurse."""

    child = _agent("Child", llm_provider)
    orphan = _agent(
        "Orphan", llm_provider, handoff_scope=HandoffScope.DIRECT_PEERS, sub_agents=[child]
    )
    unrelated_entrypoint = _agent("Main", llm_provider)

    targets = ContextBuilder._get_handoff_targets(orphan, unrelated_entrypoint)

    assert {a.name for a in targets} == {"Child"}


# ---------------------------------------------------------------------------
# `_get_context_visibility` unit tests — the READ scope is driven by
# ``context_visibility`` (GLOBAL/PEERS/STANDARD) and is orthogonal to
# ``sub_agents``.
# ---------------------------------------------------------------------------


def test_context_visibility_global_returns_entire_subtree(
    llm_provider: BaseLLMProvider,
):
    leaf = _agent("Leaf", llm_provider)
    mid = _agent("Mid", llm_provider, sub_agents=[leaf])
    current = _agent("Curr", llm_provider, context_visibility=ContextVisibility.GLOBAL)
    entrypoint = _agent("Main", llm_provider, sub_agents=[current, mid])

    scope = ContextBuilder._get_context_visibility(current, entrypoint)

    # GLOBAL → entire subtree rooted at the entrypoint, including the
    # entrypoint itself and the current agent.
    assert {a.name for a in scope} == {"Main", "Curr", "Mid", "Leaf"}


def test_context_visibility_peers_returns_sibling_subtrees(
    llm_provider: BaseLLMProvider,
):
    leaf = _agent("Leaf", llm_provider)
    sibling = _agent("Sibling", llm_provider, sub_agents=[leaf])
    current = _agent("Curr", llm_provider, context_visibility=ContextVisibility.PEERS)
    entrypoint = _agent("Main", llm_provider, sub_agents=[current, sibling])

    scope = ContextBuilder._get_context_visibility(current, entrypoint)

    # PEERS → each child of the parent (current's siblings, INCLUDING self
    # since self is also a child of the parent) plus their subtrees.
    assert {a.name for a in scope} == {"Curr", "Sibling", "Leaf"}


def test_context_visibility_standard_returns_own_subtree(
    llm_provider: BaseLLMProvider,
):
    leaf = _agent("Leaf", llm_provider)
    sibling = _agent("Sibling", llm_provider)
    current = _agent(
        "Curr",
        llm_provider,
        context_visibility=ContextVisibility.STANDARD,
        sub_agents=[leaf],
    )
    entrypoint = _agent("Main", llm_provider, sub_agents=[current, sibling])

    scope = ContextBuilder._get_context_visibility(current, entrypoint)

    # STANDARD → the agent's own subtree: sub_agents (recursively) plus self
    # (self is excluded downstream by create_context). No sibling.
    assert {a.name for a in scope} == {"Leaf", "Curr"}


def test_context_visibility_standard_includes_nested_grandchildren(
    llm_provider: BaseLLMProvider,
):
    """STANDARD now returns the agent's full recursive subtree, not just direct
    children: a grandchild is reachable too."""
    grandchild = _agent("Grandchild", llm_provider)
    child = _agent("Child", llm_provider, sub_agents=[grandchild])
    current = _agent(
        "Curr",
        llm_provider,
        context_visibility=ContextVisibility.STANDARD,
        sub_agents=[child],
    )
    entrypoint = _agent("Main", llm_provider, sub_agents=[current])

    scope = ContextBuilder._get_context_visibility(current, entrypoint)

    # STANDARD → recursive subtree rooted at the current agent (plus self).
    assert {a.name for a in scope} == {"Curr", "Child", "Grandchild"}


def test_context_visibility_peers_orphan_falls_back_to_self_subagents(
    llm_provider: BaseLLMProvider,
):
    """When the current agent is not reachable from the entrypoint (e.g.
    standalone/orphan agent reused without re-attachment), the PEERS
    branch in ``_get_context_visibility`` falls back to the current
    agent's own subtree to avoid returning an empty scope."""

    child = _agent("Child", llm_provider)
    orphan = _agent(
        "Orphan",
        llm_provider,
        context_visibility=ContextVisibility.PEERS,
        sub_agents=[child],
    )
    unrelated_entrypoint = _agent("Main", llm_provider)

    scope = ContextBuilder._get_context_visibility(orphan, unrelated_entrypoint)

    # Fallback: subtree rooted at the orphan (includes self + sub_agents).
    assert {a.name for a in scope} == {"Orphan", "Child"}


# ---------------------------------------------------------------------------
# PEERS visibility — integration via ContextBuilder.create_context.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_peers_subagent_sees_sibling_in_visible_names(
    llm_provider: BaseLLMProvider,
):
    """A PEERS-visibility sub-agent must surface its siblings (and their
    subtrees) in ``visible_agent_names`` while keeping the callable set
    bound to its own ``sub_agents``."""

    leaf = _agent("Leaf", llm_provider)
    sibling = _agent("Sibling", llm_provider, sub_agents=[leaf])
    presentation = _agent(
        "Presentation",
        llm_provider,
        context_visibility=ContextVisibility.PEERS,
        sub_agents=[],
    )
    entrypoint = _agent("Main", llm_provider, sub_agents=[presentation, sibling])

    state = _make_state(
        current_agent=presentation,
        entrypoint_agent=entrypoint,
        events_history=[],
        interaction_id="interaction-peers",
    )
    context = ContextBuilder.create_context(
        state=state,
        user_input=_agent_start(callee="Presentation", interaction_id="interaction-peers"),
    )

    # Visibility scope reaches each sibling subtree (current excluded).
    assert set(context.visible_agent_names) == {"Sibling", "Leaf"}
    # Callable set follows sub_agents=[] — no implicit expansion.
    assert context.handoff_targets_description == []


@pytest.mark.asyncio
async def test_direct_peers_handoff_subagent_can_call_sibling_via_create_context(
    llm_provider: BaseLLMProvider,
):
    """A built-in-style agent with ``handoff_scope=DIRECT_PEERS`` and ``sub_agents=[]``
    gets its callable targets resolved to its DIRECT siblings at runtime
    through ``create_context`` (self excluded; no recursion into the sibling's
    subtree). With the default STANDARD ``context_visibility`` it can CALL the
    sibling without SEEING its events — the EXECUTE and READ axes stay
    orthogonal."""

    leaf = _agent("Leaf", llm_provider)
    sibling = _agent("Sibling", llm_provider, sub_agents=[leaf])
    presentation = _agent(
        "Presentation",
        llm_provider,
        handoff_scope=HandoffScope.DIRECT_PEERS,
        sub_agents=[],
    )
    entrypoint = _agent("Main", llm_provider, sub_agents=[presentation, sibling])

    state = _make_state(
        current_agent=presentation,
        entrypoint_agent=entrypoint,
        events_history=[],
        interaction_id="interaction-peers-handoff",
    )
    context = ContextBuilder.create_context(
        state=state,
        user_input=_agent_start(callee="Presentation", interaction_id="interaction-peers-handoff"),
    )

    # EXECUTE scope resolved to the DIRECT sibling at runtime (self excluded).
    # "Leaf" (the sibling's child) is NOT reachable — no recursion.
    assert {d.name for d in context.handoff_targets_description} == {"Sibling"}
    # READ scope stays narrow: STANDARD visibility + empty sub_agents → nothing.
    assert context.visible_agent_names == []


def _external_part(
    text: str,
    *,
    agent_visible: bool | None,
    interaction_id: str | None = None,
    tool_name: str = "StoreMessage",
) -> BaseEvent:
    """Build an EXTERNAL_PART as the platform writes it: authored by the tool the push
    token was minted for, so never by a name in the agent tree.

    ``agent_visible=None`` leaves the metadata out entirely, which is what a caller that
    predates the flag produces."""
    metadata = (
        {} if agent_visible is None else {METADATA_EXTERNAL_PART_AGENT_VISIBLE: agent_visible}
    )
    return BaseEvent(
        event_type=ExecutionEventType.EXTERNAL_PART,
        content=[Part(text=text)],
        author=tool_name,
        interaction_id=interaction_id,
        metadata=metadata,
    )


def test_visible_external_part_survives_the_author_filter(llm_provider: BaseLLMProvider):
    """An external part is authored by a tool, not an agent, so the author filter used to
    drop it before any context manager could apply its visibility metadata."""

    entrypoint = _agent("Main", llm_provider)
    part = _external_part("pushed by an external tool", agent_visible=True)

    state = _make_state(
        current_agent=entrypoint,
        entrypoint_agent=entrypoint,
        events_history=[part],
        interaction_id="interaction-1",
    )

    context = ContextBuilder.create_context(
        state=state, user_input=_user_input("q", "interaction-1")
    )

    assert part in context.conversation_history


@pytest.mark.parametrize("agent_visible", [False, None])
def test_non_visible_external_part_never_reaches_the_context(
    llm_provider: BaseLLMProvider, agent_visible: bool | None
):
    """Parts pushed for the user's eyes only are dropped here, so a context manager with no
    EXTERNAL_PART branch cannot leak them. A missing flag counts as not visible."""

    entrypoint = _agent("Main", llm_provider)
    part = _external_part("for the user only", agent_visible=agent_visible)

    state = _make_state(
        current_agent=entrypoint,
        entrypoint_agent=entrypoint,
        events_history=[part],
        interaction_id="interaction-1",
    )

    context = ContextBuilder.create_context(
        state=state, user_input=_user_input("q", "interaction-1")
    )

    assert part not in context.conversation_history


def test_visible_external_part_outlives_its_own_interaction(llm_provider: BaseLLMProvider):
    """Unlike USER_INPUT, external parts are not scoped by interaction_id: a part pushed in
    an earlier turn still belongs to the conversation when a sub-agent reads it later."""

    sub = _agent("Sub", llm_provider)
    entrypoint = _agent("Main", llm_provider, sub_agents=[sub])
    old_part = _external_part(
        "pushed during turn 1", agent_visible=True, interaction_id="interaction-old"
    )

    state = _make_state(
        current_agent=sub,
        entrypoint_agent=entrypoint,
        events_history=[old_part],
        interaction_id="interaction-new",
    )

    context = ContextBuilder.create_context(
        state=state,
        user_input=_agent_start(callee="Sub", task="t", interaction_id="interaction-new"),
    )

    assert old_part in context.conversation_history
