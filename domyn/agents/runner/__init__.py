import asyncio
import contextlib
import copy
import inspect
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from enum import StrEnum
from importlib import import_module
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from domyn.agents.agents.agent import Agent
from domyn.agents.agents.base import ContextVisibility, HandoffScope
from domyn.agents.agents.delegate_agent import DelegateAgent
from domyn.agents.approvals import (
    ApprovalDecision,
    ApprovalGrant,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalScope,
    ApprovalVerdict,
    ConversationApprovalStore,
    build_request,
    is_already_approved,
    visible_parameters,
)
from domyn.agents.conversation_manager import (
    BaseConversationManager,
    ConversationInfo,
    InMemoryConversationManager,
)
from domyn.agents.conversation_manager.base import StackAgentEntry
from domyn.agents.core import (
    METADATA_CONVERTED_TO_AGENT_END,
    METADATA_EXTERNAL_PART_AGENT_VISIBLE,
    METADATA_FORWARD_VERBATIM,
    METADATA_FORWARD_VERBATIM_SOURCE_ID,
    METADATA_FORWARD_VERBATIM_SOURCE_KEY,
    METADATA_STEERING,
    METADATA_TOOL_BATCH,
    METADATA_VISIBILITY_ONLY,
    AgentAction,
    ApprovalType,
    AuthenticationAction,
    AuthenticationResult,
    BaseAction,
    BaseEvent,
    ExecutionEventType,
    FeedbackAction,
    FeedbackParameters,
    LLMCallAction,
    Part,
    ToolAction,
)
from domyn.agents.core.context import InteractionContext
from domyn.agents.core.interfaces import BaseAgent
from domyn.agents.decisions import runtime_of
from domyn.agents.decisions.approvals import consult as consult_decision_for_approval
from domyn.agents.decisions.approvals import event_metadata as decision_event_metadata
from domyn.agents.decisions.context import decide as decide_context_relevance
from domyn.agents.decisions.events import event as decision_event
from domyn.agents.decisions.events import metadata as decision_metadata
from domyn.agents.decisions.runtime import DecisionMode
from domyn.agents.decisions.sensitivity import Rating, Sensitivity, provider_ceiling
from domyn.agents.decisions.sensitivity import allows as sensitivity_allows
from domyn.agents.decisions.sensitivity import rate as rate_sensitivity
from domyn.agents.decisions.tools import decide as decide_tool_visibility
from domyn.agents.hooks.base import HookError, HookResult, HookType
from domyn.agents.hooks.function_hook import FunctionHook
from domyn.agents.instructions import AgentInstructions, InstructionRuntime
from domyn.agents.instructions.conditional import apply as apply_conditional_instructions
from domyn.agents.llm.base import BaseLLMProvider, LLMResponse
from domyn.agents.llm.context_window import resolve_context_window
from domyn.agents.llm.delegate import DelegateLLMProvider
from domyn.agents.logger import get_logger
from domyn.agents.planner_strategy.context_management.base import ContextManager
from domyn.agents.registry.agents import get_agent_registry
from domyn.agents.skills.models import LoadedSkills
from domyn.agents.skills.runtime import (
    SkillRequiredError,
    SkillsRuntime,
    raise_if_required_unavailable,
)
from domyn.agents.skills.turn import skill_query_for
from domyn.agents.tools.base import ToolResult, UpdateToolResult
from domyn.agents.tools.delegate_tool import DelegateTool
from domyn.agents.tools.function_tool import FunctionTool
from domyn.agents.tools.mcp.mcp_tool import MCPCredentialsCheckResult, MCPTool
from domyn.agents.tools.streamable_function_tool import (
    StreamableToolResult,
)

NO_COPY_KEYS = {"tool", "result", "agent"}


class PushTiming(StrEnum):
    """When content handed to ``push_parts`` reaches the turn.

    What separates them is which part of the runner handles them, and therefore what
    they can do.

    ``RACING`` is raced against the agent loop by ``_merge_with_injections``, which
    runs outside it. The event goes out to the consumer wherever the loop happens to
    be — mid-model-call, mid-tool — and is persisted, but enters no
    ``interaction_history``: that generator cannot know which agent's history is the
    live one, so the model reads the content on a later turn, built from the
    conversation.

    ``SCHEDULED`` is held for the loop, which takes it in at the next of
    ``steering_triggers`` — a boundary. That costs the wait, and buys what racing
    cannot: the content enters the running agent's history, and a turn's final answer
    is itself a boundary, so a message absorbed there re-enters the turn rather than
    ending it, and the agent answers what was just said.

    Content to *show* wants the first. Content the turn must *answer* wants the
    second.
    """

    RACING = "racing"
    SCHEDULED = "scheduled"


class PushResult(BaseModel):
    """What ``push_parts`` answers.

    ``delivered`` says the event reached a live run, which is the only thing this
    process can promise: there is no run to hand it to otherwise, and nothing is
    written on the caller's behalf. It does **not** promise the event was consumed —
    a run that ends before reaching it discards it — so a caller that cannot lose the
    content owns the fallback. Both in-tree callers already have one.

    ``event`` is the event as built, carrying the caller's ``event_id`` when one was
    given, so the id in a caller's own records is the id that reaches the stream and
    the conversation history.

    Deliberately no ``__bool__``: a model is truthy whatever ``delivered`` says, so
    the result has to be read through ``delivered`` and testing it directly is a bug
    that would always pass.
    """

    delivered: bool
    event: BaseEvent


class InjectionsQueue(BaseModel):
    """Where content pushed into a running turn waits until the turn takes it.

    Registered by ``run()`` — before the conversation setup, so a message typed
    during it still has somewhere to go — and looked up by ``push_parts``. Its
    presence in ``Runner.injection_queue`` is the whole of what ``delivered`` means.

    Two queues, because they are consumed by different code at different times
    rather than because the items differ:

    - ``racing_queue`` is raced against the agent loop by
      ``_merge_with_injections`` and goes out wherever the loop happens to be.
    - ``scheduled_queue`` is picked up by the loop itself at the next of
      ``steering_triggers``, where the planner is about to rebuild its messages.

    It holds no identifiers and no run state. An event's ``interaction_id`` and
    ``turn_id`` are stamped by ``_add_event`` when the turn takes it in, which names
    the interaction that read the content rather than whichever one was current when
    it was pushed. ``default_author`` is the one thing that has to come from the run,
    and it does not change during one.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    racing_queue: asyncio.Queue[BaseEvent]
    scheduled_queue: asyncio.Queue[BaseEvent]
    default_author: str

    async def take_scheduled(self) -> list[BaseEvent]:
        """Everything waiting for a boundary, oldest first, removed as it is taken.

        A plain drain, and it stays one only because nothing decides *per event*
        whether it may be handed over: whether a message reaches this agent at all is
        answered before the queue is touched, by
        ``Runner.steering_reaches_subagents``. A per-event rule would mean inspecting
        each item and putting back what this agent may not have — which is atomic here
        only by accident of a single event loop, and would need a lock or a script the
        moment the queue is shared between replicas.

        ``async`` so a shared implementation can replace this one without changing a
        caller. The contract is just FIFO: each event handed over exactly once, in
        arrival order.
        """
        taken: list[BaseEvent] = []
        while not self.scheduled_queue.empty():
            taken.append(self.scheduled_queue.get_nowait())
        return taken


# Observation carried back to the caller when a subagent asks the user for
# missing information (RESPONSE with need_feedback=True). Subagents never
# speak to the user directly, so the caller must either supply the answer
# from context or relay the question with its own need_feedback response.
SUBAGENT_NEEDS_INFO_TEMPLATE = (
    "Agent {agent_name} needs more information from the user before it can "
    "complete its task:\n\n{question}\n\n"
    "If the missing information is already available in the conversation, "
    "call {agent_name} again including it in the task. Otherwise ask the "
    "user for it (need_feedback=true) and, once answered, call {agent_name} "
    "again passing the answer along."
)

# Ten minutes, chosen to sit above every timeout the package configures itself
# (30s for an MCP client, 120s before a shell command is left to run on) so a
# tool that has its own limit still fails with its own message. What this
# catches is the call that has no limit at all and never returns.
DEFAULT_TOOL_TIMEOUT_S = 600.0


class ToolTimeoutError(Exception):
    """A tool held the turn past its ceiling.

    Raised rather than returned so it lands in the one handler that already
    turns anything a tool does into a TOOL_ERROR the model reads and acts on.
    """


async def _bounded(awaitable: Any, ceiling: float, name: str) -> Any:
    if not ceiling:
        return await awaitable
    try:
        return await asyncio.wait_for(awaitable, ceiling)
    except TimeoutError:
        raise ToolTimeoutError(
            _TIMEOUT_OBSERVATION.format(name=name, seconds=int(ceiling))
        ) from None


async def _bounded_iter(agen: Any, ceiling: float, name: str) -> AsyncGenerator[Any, None]:
    """Bound the wait for each chunk, not the whole call.

    A streaming tool that is still reporting is still working, and cutting it
    off at a total would punish exactly the tools that say what they are doing.
    """
    if not ceiling:
        async for item in agen:
            yield item
        return
    iterator = agen.__aiter__()
    try:
        while True:
            try:
                item = await asyncio.wait_for(iterator.__anext__(), ceiling)
            except StopAsyncIteration:
                return
            except TimeoutError:
                raise ToolTimeoutError(
                    _TIMEOUT_OBSERVATION.format(name=name, seconds=int(ceiling))
                ) from None
            yield item
    finally:
        await agen.aclose()


_TIMEOUT_OBSERVATION = (
    "Tool {name} did not return within {seconds}s and was given up on. Anything it "
    "started may still be running. Do not simply retry it unchanged: either run it "
    "over less work, or use a tool that reports progress while it runs."
)


@dataclass
class _ParentTransition:
    """Outcome of Runner._return_to_parent.

    ``iterator`` is the resumed caller's iterator, or None when the agent
    loop must stop (root reached, hook break, or should_continue already
    False). ``context`` and ``should_continue`` carry hook-updated state.
    """

    context: InteractionContext
    should_continue: bool = True
    iterator: AsyncGenerator[Any, Any] | None = None


# Under ``context.state``: the calls of a model response still to run when an
# approval (or an authentication) paused it — the call waiting for the answer and
# the ones after it — so the resume runs them before the model is asked again.
HELD_TOOL_CALLS_KEY = "held_tool_calls"

# What the model is told about a call of its response that never ran, because a
# hook stopped the turn before it or the user wrote instead of answering.
_STOPPED = "the turn was stopped before it ran."
_SUPERSEDED = (
    "the user sent a new message instead of answering the approval it was waiting "
    "for, so it did not run. Ask again if it is still needed."
)

# Appended to a tool result when the user refused to run the tool, or to authenticate
# for it: the refusal covers the related calls in the model's response too.
_USER_REFUSED_TOOL_CALLS = (
    "The user decided not to execute it; acknowledge that for related tool calls as well."
)

# How many calls of one response may run at the same time.
DEFAULT_MAX_PARALLEL_TOOL_CALLS = 8


@dataclass
class _ToolCallsOutcome:
    """How ``Runner._run_tool_calls`` ended, for the arm that called it.

    ``paused``: a call waits for a human, so the run ends here (``close_iterator``
    when the planner's iterator must be closed, as for an authentication).
    ``should_continue``: False when an ON_AGENT_EVENT hook stopped the turn.
    ``last``: the result of the last call, stored but not yet yielded, so the loop
    yields it where it yields every event and absorbs steering after it.
    """

    paused: bool = False
    close_iterator: bool = False
    should_continue: bool = True
    last: BaseEvent | None = None


@dataclass
class RunnerState:
    """Encapsulates the state of a runner execution."""

    application_name: str
    user_id: str
    conversation_id: str
    conversation_info: ConversationInfo
    current_agent: Agent | DelegateAgent
    available_agents: list[Agent | DelegateAgent]
    entrypoint_agent: Agent | DelegateAgent
    interaction_id: str | None = None
    turn_id: str | None = None  # stable across agent transitions within a single user-input turn
    run_config: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    # How many times a post_agent_call hook has forced the turn to continue.
    # Lives on the run state, never on the Agent: one Agent instance is shared
    # across concurrent conversations, so a counter there would leak between them.
    continue_count: int = 0
    # Copied from the Runner at the top of the turn, because the two places that
    # enforce it are on opposite sides of a staticmethod: `_absorb_pending` has the
    # Runner, `ContextBuilder.create_context` has only this.
    steering_reaches_subagents: bool = False
    # Skills loaded once for this turn, keyed by agent name. On the run state for
    # the same reason as above, and because a remote source serves a different
    # catalogue per user. A key is present for every agent that declares a
    # runtime — empty when its source failed — so a *missing* key means "nobody
    # loaded", which is what tells the agent to load for itself.
    skills: dict[str, LoadedSkills] = field(default_factory=dict)
    # Standing instructions loaded once for this turn, keyed by agent name, on
    # the same terms as `skills`: present for every agent that declares a
    # runtime, empty when its source failed.
    instructions: dict[str, AgentInstructions] = field(default_factory=dict)


class ContextBuilder:
    """Helper class to build InteractionContext instances."""

    @staticmethod
    def create_context(
        state: RunnerState,
        user_input: BaseEvent,
        agent_stack_entry: StackAgentEntry | None = None,
    ) -> InteractionContext:
        """Create an InteractionContext with proper visibility rules."""
        context_agents = ContextBuilder._get_context_visibility(
            state.current_agent, state.entrypoint_agent
        )
        handoff_targets = ContextBuilder._get_handoff_targets(
            state.current_agent, state.entrypoint_agent
        )

        # filter events_history
        visible_authors = [state.current_agent.name] + [a.name for a in context_agents]
        user_event_types = [ExecutionEventType.USER_INPUT, ExecutionEventType.TOOL_FEEDBACK]
        restrict_user_events_by_interaction_id = (
            state.current_agent is not state.entrypoint_agent
            and state.current_agent.context_visibility != ContextVisibility.GLOBAL
        )

        def _steering_is_hidden(event: BaseEvent) -> bool:
            """A mid-turn message this agent is not the intended audience for.

            The companion to the gate in ``Runner._absorb_pending``, and needed
            separately because that one only governs an agent that is already
            running. A sub-agent started *after* the message was taken in builds its
            history from here instead, and a steering event carries the turn's own
            interaction_id — so without this it would arrive through the door the
            absorb gate closed. Hidden, not deleted: the event stays in the
            conversation, and the agent the user is talking to still gets it.

            The turn's opening USER_INPUT is indistinguishable from a mid-turn one at
            this point — same type, same interaction_id — which is the whole reason
            ``METADATA_STEERING`` exists. Sub-agents must keep seeing the former.
            """
            return (
                restrict_user_events_by_interaction_id
                and not state.steering_reaches_subagents
                and bool(event.metadata.get(METADATA_STEERING, False))
            )

        events_history = [
            event
            for event in state.conversation_info.events_history
            if event.author in visible_authors
            or (
                event.event_type in user_event_types
                and (
                    not restrict_user_events_by_interaction_id
                    or event.interaction_id == state.interaction_id
                )
                and not _steering_is_hidden(event)
            )
            # Authored by an external source that inject visible part in conversation, never by an agent.
            or (
                event.event_type == ExecutionEventType.EXTERNAL_PART
                and event.metadata.get(METADATA_EXTERNAL_PART_AGENT_VISIBLE, False)
            )
        ]

        if user_input.interaction_id is None:
            user_input.interaction_id = state.interaction_id

        # build interaction_history. For TOOL_FEEDBACK resume we must seed it from the
        # persisted history (the in-progress agent's prior tool calls live there).
        # For other inputs the history is populated incrementally via _add_event(...,
        # context=context) as USER_INPUT and AGENT_START are stored.
        interaction_history: list[BaseEvent] = []
        if (
            user_input.event_type == ExecutionEventType.TOOL_FEEDBACK
            and state.conversation_info.events_history
        ):
            # recover the AGENT_START that called the current agent plus all its interaction events
            agents_calls = list(
                filter(
                    lambda x: (
                        x.event_type == ExecutionEventType.AGENT_START
                        and x.action.name == state.current_agent.name
                    ),
                    state.conversation_info.events_history,
                )
            )
            if agents_calls:
                agents_calls = [agents_calls[-1]]

            interaction_history = agents_calls + list(
                filter(
                    lambda x: (
                        x.interaction_id == state.interaction_id
                        and x.event_type != ExecutionEventType.USER_INPUT
                    ),
                    state.conversation_info.events_history,
                )
            )

            if (
                state.conversation_info.events_history[-1].event_type
                == ExecutionEventType.AGENT_FEEDBACK
            ):
                # Agent resuming after tool re-authentication: seed interaction_history with
                # the most recent AGENT_START for the current agent (its original task).
                # No prior reasoning trace exists since execution never started.
                agents_calls = [
                    x
                    for x in state.conversation_info.events_history
                    if x.event_type == ExecutionEventType.AGENT_START
                    and x.action.name == state.current_agent.name
                ]
                if agents_calls:
                    interaction_history = [agents_calls[-1]]

        return InteractionContext(
            conversation_id=state.conversation_id,
            user_id=state.user_id,
            agent_input=user_input,
            interaction_id=state.interaction_id,
            turn_id=state.turn_id,
            conversation_history=events_history,
            interaction_history=interaction_history,
            handoff_targets_description=[
                agent.get_agent_description()
                for agent in handoff_targets
                if agent != state.current_agent
            ],
            visible_agent_names=[
                agent.name for agent in context_agents if agent != state.current_agent
            ],
            state=state.conversation_info.state or {},
            run_config=state.run_config,
            is_subagent=state.current_agent is not state.entrypoint_agent,
        )

    @staticmethod
    def _get_context_visibility(
        current_agent: Agent | DelegateAgent, entrypoint_agent: Agent | DelegateAgent
    ) -> list[Agent | DelegateAgent]:
        """Determine which agents are visible in the prompt context history.

        This controls READ scope only: prior events from these agents may be
        retained by context managers. See ``_get_handoff_targets`` for the
        separate EXECUTE scope that defines which agents the current agent may
        actually invoke via handoff.

        Driven by ``context_visibility``:
          - GLOBAL: entire subtree from entrypoint
          - PEERS: siblings (subtree of each child of the parent)
          - STANDARD: own subtree (``sub_agents`` recursively, plus self)
        """

        if current_agent.context_visibility == ContextVisibility.GLOBAL:
            return _return_subagents(entrypoint_agent)
        elif current_agent.context_visibility == ContextVisibility.PEERS:
            return _collect_peers(current_agent, entrypoint_agent)
        else:  # ContextVisibility.STANDARD
            return _return_subagents(current_agent)

    @staticmethod
    def _get_handoff_targets(
        current_agent: Agent | DelegateAgent, entrypoint_agent: Agent | DelegateAgent
    ) -> list[Agent | DelegateAgent]:
        """Determine which agents the current agent can actually invoke via
        handoff. Controls EXECUTE scope only — orthogonal to context
        visibility.

        Driven by ``handoff_scope``:
          - EXPLICIT (default): the explicit ``sub_agents`` list, verbatim.
          - DIRECT_PEERS: the agent's own direct children plus its direct siblings
            (the parent's other children) — one level down and sideways, with
            NO recursion into a sibling's subtree. This intentionally differs
            from the READ-scope PEERS (``_collect_peers``, which is recursive):
            for EXECUTE you hand off to a peer's entrypoint, not into another
            branch's internals (that would skip the peer's orchestration).
          - GLOBAL: entire subtree from the entrypoint, resolved at runtime.

        The DIRECT_PEERS/GLOBAL modes let built-in agents (e.g. PresentationAgent,
        CodeAgent) hand off to user-defined agents that are not known when the
        built-in is defined — they are appended to the tree per-request and
        resolved here at ``create_context`` time. EXPLICIT keeps a curated
        callable set decoupled from what the agent may read
        (``context_visibility``), e.g. read GLOBAL but only call ``chart_agent``.
        """

        if current_agent.handoff_scope == HandoffScope.GLOBAL:
            return _return_subagents(entrypoint_agent)
        elif current_agent.handoff_scope == HandoffScope.DIRECT_PEERS:
            return _collect_direct_peers(current_agent, entrypoint_agent)
        else:  # HandoffScope.EXPLICIT
            return [_return_agent(agent) for agent in current_agent.sub_agents]


def _return_agent(agent: Agent | DelegateAgent | str) -> Agent | DelegateAgent:
    """Return the agent instance, resolving string references via the registry."""
    if isinstance(agent, str):
        return get_agent_registry().get_agent(agent)
    return agent


def _find_parent(
    root: Agent | DelegateAgent, target: Agent | DelegateAgent
) -> Agent | DelegateAgent | None:
    """Find the direct parent of `target` in the agent tree rooted at `root`."""
    for sub_agent in getattr(root, "sub_agents", []):
        resolved = _return_agent(sub_agent)
        if resolved == target:
            return root
        parent = _find_parent(resolved, target)
        if parent is not None:
            return parent
    return None


def _collect_peers(
    current_agent: Agent | DelegateAgent, entrypoint_agent: Agent | DelegateAgent
) -> list[Agent | DelegateAgent]:
    """Return the current agent's sibling subtrees (each child of its parent,
    plus their descendants). Includes the current agent itself, since it is
    also a child of the parent; callers exclude self downstream.

    When the current agent is not reachable from the entrypoint (orphan), it
    falls back to the current agent's own subtree to avoid an empty scope.
    Used only by the READ-scope PEERS branch (context visibility); the
    EXECUTE-scope DIRECT_PEERS branch is deliberately non-recursive and does
    not share this helper.
    """
    parent = _find_parent(entrypoint_agent, current_agent)
    if parent is None:
        return _return_subagents(current_agent)
    peers: list[Agent | DelegateAgent] = []
    for sibling in parent.sub_agents:
        peers.extend(_return_subagents(_return_agent(sibling)))
    return peers


def _collect_direct_peers(
    current_agent: Agent | DelegateAgent, entrypoint_agent: Agent | DelegateAgent
) -> list[Agent | DelegateAgent]:
    """Return the current agent's own direct children plus the parent's
    children (siblings, incl. self — callers exclude self downstream).

    Non-recursive by design, unlike the READ-scope ``_collect_peers``: you
    hand off to a peer's entrypoint, not into its subtree, so a sibling's
    internal sub-agents are not reachable. Orphans (no parent) fall back to
    own children. Used only by the EXECUTE-scope DIRECT_PEERS branch.
    """
    parent = _find_parent(entrypoint_agent, current_agent)
    direct_siblings = parent.sub_agents if parent is not None else []
    targets: list[Agent | DelegateAgent] = []
    for ref in (*current_agent.sub_agents, *direct_siblings):
        resolved = _return_agent(ref)
        if resolved not in targets:
            targets.append(resolved)
    return targets


def _return_subagents(agent: Agent | DelegateAgent) -> list[Agent | DelegateAgent]:
    """Return all agents in the subtree rooted at `agent`, including itself."""
    agents: list[Agent | DelegateAgent] = []

    for sub_agent in getattr(agent, "sub_agents", []):
        _sub_agent = _return_agent(sub_agent)
        agents.extend(_return_subagents(_sub_agent))

    if agent not in agents:
        agents.append(_return_agent(agent))
    return list(filter(None, agents))


class AgentHookExecution(BaseModel):
    event: BaseEvent
    context: InteractionContext
    break_execution: bool
    continue_with: BaseEvent | None = None


class OnEventHookExecution(BaseModel):
    hook_event: BaseEvent
    current_event: BaseEvent
    break_execution: bool = False


def _proposed_pattern(tool: Any, request: ApprovalRequest) -> str | None:
    """The wider permission a tool suggests for this call, if it suggests one.

    A suggestion only: it is shown to the user and can be edited before it
    becomes a rule, because a pattern grants more than the line they read. A
    tool that raises here loses the offer, not the prompt, and one that returns
    ``None`` is declining to offer one — not proposing the pattern ``"None"``.

    Never one that misses this call: such a rule would never apply.
    """
    proposer = getattr(tool, "approval_pattern", None)
    if not callable(proposer):
        return None
    try:
        proposal = proposer(**request.parameters)
    except Exception as exc:
        get_logger().debug(f"approval_pattern for '{request.tool_name}' failed: {exc}")
        return None
    if proposal is None:
        return None
    proposal = str(proposal)
    if not is_already_approved(
        request, [ApprovalGrant(value=request.pattern_key(proposal), kind="pattern")]
    ):
        get_logger().debug(f"approval_pattern for '{request.tool_name}' misses its own call")
        return None
    return proposal


class Runner:
    """Orchestrates agent execution and conversation management."""

    steering_triggers: tuple[ExecutionEventType, ...] = (
        ExecutionEventType.TOOL_END,
        ExecutionEventType.TOOL_ERROR,
        ExecutionEventType.TOOL_NEEDS_FEEDBACK,
        ExecutionEventType.AGENT_END,
        ExecutionEventType.RESPONSE,
    )
    """Which events are followed by a check for user messages that arrived mid-turn.

    The first three are points where the planner is about to rebuild its message
    list, so a message taken in there reaches the next LLM call. ``TOOL_END`` and
    ``TOOL_ERROR`` are a tool cycle finishing; ``AGENT_END`` is a parent resuming
    after a sub-agent, where the message may have been waiting for the whole of the
    sub-agent's work.

    ``TOOL_NEEDS_FEEDBACK`` is the *last* event of its run: an approval gate ends the
    generator, and the caller starts the other half of the turn with a
    ``TOOL_FEEDBACK``. Everything registered for delivery — the queue included —
    goes with that generator, so a message not taken in here is one this runner
    accepted and then discarded, and the front-end, which was told it was delivered,
    waits for an echo that never comes. Taking it in *before* the pause is what makes
    the queue's per-run lifetime harmless. With ``/auto`` every gated call is such a
    pause, so on the CLI's own configuration (``files`` and ``shell``, both gated)
    this is the difference between mid-turn messages working and none of them
    reaching the model at all.

    It writes the message into history between a tool call and its result, which the
    wire protocol rejects. That is already handled one layer down, by
    ``reorder_messages_around_tool_calls``, which moves anything wedged there to
    after the result — the same position it would have had at ``TOOL_END``.

    ``RESPONSE`` is different in kind: it is the point where the turn is about to
    *end*, so there is no next call to reach — taking a message in there re-enters
    the agent and creates one. Without it, a message that arrives after the last
    tool call, i.e. during the final model call, has no boundary left to be taken in
    at and is only read by whatever turn comes next. That window is where a user
    watching an answer stream past is most likely to type, so it is on by default;
    see ``_absorb_for_steering`` and ``_reenter_after_steering``.

    A class attribute so a subclass or a single instance can widen or narrow it —
    an embedder whose agents rarely call tools may want another point entirely, and
    an empty tuple switches the feature off. Dropping ``RESPONSE`` alone restores
    the behaviour of ending the turn on the answer.

    Widening it is free: messages are handed to the turn by ``push_parts``, so a
    boundary with nothing waiting does no work at all. A **partial** event never takes
    a message in, because a boundary after it feeds the same model call and catches a
    superset of what it could — which makes ``RESPONSE_TOKEN`` a no-op here, every
    token being partial.
    """

    steering_reaches_subagents: bool = False
    """Whether a mid-turn message reaches a sub-agent, or only the agent the user is
    talking to.

    Off by default: a sub-agent is midway through a task its parent specified, and a
    user message that silently redirects it hands the parent back a result that does
    not answer what it asked. The message is not lost by that — it stays queued, and
    the parent takes it in on the ``AGENT_END`` that returns control.

    A class attribute, like ``steering_triggers``, so a deployment whose sub-agents
    are long-running enough that waiting for the parent is too slow can turn it on.

    It is deliberately a property of the deployment rather than of each message. The
    difference is what the queue has to support: a per-message rule means a boundary
    must look at every waiting event and put back the ones it may not have, which is
    only atomic by accident in one process and needs a lock or a script once the queue
    is shared across replicas. This rule is answerable before the queue is touched, so
    a boundary either drains it or leaves it alone.
    """

    def __init__(
        self,
        conversation_manager: BaseConversationManager | None = None,
        runner_id: str | None = None,
        max_turn_continuations: int = 3,
        approval_policy: ApprovalPolicy | None = None,
        tool_timeout_s: float = DEFAULT_TOOL_TIMEOUT_S,
        max_parallel_tool_calls: int = DEFAULT_MAX_PARALLEL_TOOL_CALLS,
    ):
        logger = get_logger()
        self.runner_id = runner_id or f"runner_{uuid.uuid4().hex[:8]}"
        self.conversation_manager = conversation_manager or InMemoryConversationManager()
        # Decides whether a tool marked `approval_type` still needs a human this
        # time. With no stored grants and no policy hooks it answers ASK for
        # everything, which is exactly the behaviour that existed before it — so
        # a caller that ignores approvals entirely sees no change.
        self.approval_policy = approval_policy or ApprovalPolicy(
            stores=[ConversationApprovalStore()]
        )
        # Upper bound on how many times post_agent_call hooks may force a turn to
        # continue via HookResult.continue_with. Enforced by the runner because a
        # hook cannot be trusted to bound itself.
        self._max_turn_continuations = max_turn_continuations
        # How long any one tool may hold the turn. A backstop against a wedged
        # call, not a policy on how long work may take: it sits above every
        # specific timeout so those still produce their own, better, error.
        self._tool_timeout_s = tool_timeout_s
        # How many calls of one model response may run at the same time. Whether
        # they do at all is the planner's: `parallel_tool_calls` on the tool-use
        # strategies, which marks them as a batch (see `_run_tool_calls`).
        self.max_parallel_tool_calls = max(1, max_parallel_tool_calls)
        # The turns this runner is executing right now, keyed by
        # (application_name, conversation_id). An entry exists only while a run() is
        # in flight for that conversation, and its presence is the whole answer to
        # "is there anywhere to deliver this": content handed to push_parts() with no
        # entry here is reported undelivered rather than stored on the caller's
        # behalf. A run that ends at an approval gate takes its queue with it, which
        # is why the gate is a boundary — see `steering_triggers`.
        # See push_parts() / _merge_with_injections().
        self.injection_queue: dict[tuple[str, str], InjectionsQueue] = {}
        logger.info(
            f"Runner initialized with conversation manager: {type(self.conversation_manager).__name__}, runner_id: {self.runner_id}"
        )

    async def push_parts(
        self,
        application_name: str,
        conversation_id: str,
        parts: list[Part],
        *,
        author: str | None = None,
        event_type: ExecutionEventType = ExecutionEventType.RESPONSE,
        timing: PushTiming = PushTiming.RACING,
        event_id: str | None = None,
        is_partial: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PushResult:
        """Hand externally-produced parts to the turn running for this conversation.

        The one way content that did not come from the agent gets into one. Steering
        — a user message typed while the agent is still working — is this call with
        ``event_type=USER_INPUT`` and ``timing=SCHEDULED``; an external tool pushing
        something to display is this call with ``EXTERNAL_PART`` and the default
        ``RACING``. The parts are wrapped in an event of ``event_type`` and
        delivered according to ``timing``:

        - ``RACING`` is raced against the agent loop, so it streams out of the active
          ``run()`` generator wherever the loop happens to be — an LLM call, a tool
          execution — without waiting for a natural yield point.
        - ``SCHEDULED`` is held until the turn reaches one of ``steering_triggers``:
          a finished tool cycle, a sub-agent returning, or the answer the turn was
          about to end on, which it then continues rather than ending. That is where
          the planner is about to rebuild its messages, so it is the only timing at
          which content reliably reaches the model.

        **Content the model must answer belongs on ``SCHEDULED``.** A ``USER_INPUT``
        delivered ``RACING`` lands in the history *behind* the messages already
        built for the call in flight, so the turn can end on an answer written
        without it. Racing buys nothing for such content anyway: mid-tool-call, the
        earliest position the wire protocol allows *is* the scheduled one.

        A ``USER_INPUT`` pushed here is marked ``METADATA_STEERING`` automatically —
        by definition it arrived mid-turn. Who it reaches is not a property of the
        message but of the deployment: see ``Runner.steering_reaches_subagents``.

        ``event_type`` defaults to ``RESPONSE`` for backwards compatibility, but
        content that did not come from the agent should be pushed as
        ``EXTERNAL_PART``: a ``RESPONSE`` is unconditionally part of the agent's
        own history, whereas an ``EXTERNAL_PART`` re-enters the model's context
        only when ``metadata`` marks it visible (see
        ``METADATA_EXTERNAL_PART_AGENT_VISIBLE``). Passing the metadata is
        therefore what makes that choice expressible at all — without it the
        content defaults to invisible to the model.

        ``event_id`` lets a caller name the event it is pushing, so the id in its own
        records is the id that reaches the stream and the conversation history. Left
        unset, the runner mints one.

        The result's ``delivered`` is ``False`` — and nothing else happens — when no
        ``run()`` is in flight for that conversation. **Nothing is written on the
        caller's behalf in that case**: a runner with no turn to deliver to has no
        business deciding a conversation's contents, and the caller that knows why it
        is pushing is the one that can decide whether to persist, retry or drop. Both
        the CLI and callers that push external parts have such a fallback.

        ``delivered=True`` means the event reached a live run, not that the run
        consumed it: a turn that ends before reaching the item discards it. The
        window is small and deliberately not papered over — the turn deregisters as
        soon as it knows it is ending, so a push after that point is reported
        ``False`` rather than accepted and dropped.
        """

        injections = self.injection_queue.get((application_name, conversation_id))
        metadata = dict(metadata or {})
        if event_type is ExecutionEventType.USER_INPUT:
            # A user message arriving through here is by definition one that turned up
            # mid-turn, which is what `METADATA_STEERING` says. Stamped rather than
            # asked for: a caller that forgot would have its message leak into every
            # sub-agent started afterwards — see `ContextBuilder`.
            metadata.setdefault(METADATA_STEERING, True)

        event = BaseEvent(
            event_type=event_type,
            author=author or (injections.default_author if injections else "external"),
            content=parts,
            event_id=event_id,
            conversation_id=conversation_id,
            # Partial by default for racing content, which is a fragment of
            # something still being produced; never for a message that is whole in
            # itself, which is what a scheduled one is.
            is_partial=timing is PushTiming.RACING if is_partial is None else is_partial,
            metadata=metadata,
        )
        if injections is None:
            return PushResult(delivered=False, event=event)

        # `interaction_id` and `turn_id` are deliberately left unset. `_add_event`
        # fills them when the turn takes the event in, which is both later and more
        # accurate: they then name the interaction that actually read the content
        # rather than whichever one was current when it was pushed.
        #
        # No await between the lookup above and the hand-off below, so a turn tearing
        # down cannot interleave and leave the content somewhere nobody will read.
        if timing is PushTiming.SCHEDULED:
            injections.scheduled_queue.put_nowait(event)
        else:
            injections.racing_queue.put_nowait(event)
        return PushResult(delivered=True, event=event)

    async def _pause_for_approval(
        self,
        tool: FunctionTool | DelegateTool,
        event: BaseEvent,
        state: RunnerState,
        context: InteractionContext,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Ask for approval, and take in what is waiting: this run ends here.

        The two halves are one step, so they are one call: the queue is registered
        per ``run()`` and an approval gate ends one, which makes the gate the last
        chance a waiting message gets. A site that yielded the event without the
        absorb would accept a message and then discard it — see
        ``steering_triggers`` — and a gate is added to this loop often enough that
        the pairing should not be something to remember.

        Taken in after the event rather than before it, so a message typed while the
        gate was being resolved still counts, and conditioned on
        ``steering_triggers`` like every other boundary, so narrowing that tuple
        switches this off with the rest.
        """
        yield self._build_tool_needs_feedback(tool, event, state, context)
        if ExecutionEventType.TOOL_NEEDS_FEEDBACK in self.steering_triggers:
            for steered_event in await self._absorb_pending(state, context):
                yield steered_event

    async def _pause_for_auth(
        self,
        event: BaseEvent,
        state: RunnerState,
        context: InteractionContext,
        auth_metadata: MCPCredentialsCheckResult,
    ) -> AsyncGenerator[BaseEvent, None]:
        """The same pause, for a tool whose credentials have to be renewed first.

        The condition is spelled out again rather than shared: two lines of it, and
        a reader of either arm can see what the gate does to a waiting message
        without following a third call.
        """
        yield self._build_tool_needs_feedback_auth(event, state, context, auth_metadata)
        if ExecutionEventType.TOOL_NEEDS_FEEDBACK in self.steering_triggers:
            for steered_event in await self._absorb_pending(state, context):
                yield steered_event

    async def _merge_with_injections(
        self,
        state: RunnerState,
        source: AsyncGenerator[BaseEvent, None],
    ) -> AsyncGenerator[BaseEvent, None]:
        """Interleave events from the agent loop with racing injections.

        Races the next event from ``source`` (the agent loop) against the next
        ``RACING`` item handed to ``push_parts`` for this conversation, yielding
        whichever is ready first. The event arrives here already built by
        ``push_parts``; this persists it to the conversation and yields it, and
        deliberately puts it in no ``interaction_history`` — so the model reads it on
        a later turn, built from the conversation, rather than this one.

        That is the difference between the two timings, and it is why anything the
        model must read *this* turn goes on ``SCHEDULED``: those items never come
        through here at all. The loop takes them itself, at a point where it has the
        running agent's context in hand and the planner is about to rebuild its
        messages from it.

        The ``InjectionsQueue`` is registered by ``run()``, not here, so that a message
        arriving during the conversation setup is not told there is no turn. Its
        presence *is* ``push_parts``'s promise, which is why it is taken away as
        early as possible: by the end-of-turn arms, which know the turn is ending
        before this generator finds out, and by this ``finally`` for every other way
        a turn can stop.

        ``source.__anext__()`` is only ever called once per step, and the next
        call is only made *after* the current event has been yielded to the
        caller — i.e. this never lets the agent loop advance (and apply a
        later step's side effects, e.g. conversation-manager/agent-stack
        updates) further than what has actually been consumed. A
        producer/consumer queue design was tried and rejected here: even with
        a bounded queue, the background task computes event N+1 (including
        its side effects) before it can hand off event N to the consumer,
        which broke callers that inspect state immediately after receiving a
        given event.
        """
        key = (state.application_name, state.conversation_id)
        # Registered by run() before any of the conversation setup, so anything
        # pushed while that was happening is already waiting here. A fresh one only
        # if some caller drives the loop directly.
        injections = self.injection_queue.setdefault(
            key,
            InjectionsQueue(
                racing_queue=asyncio.Queue(),
                scheduled_queue=asyncio.Queue(),
                default_author=state.entrypoint_agent.name,
            ),
        )
        queue = injections.racing_queue

        source_task: asyncio.Task[BaseEvent] | None = asyncio.ensure_future(source.__anext__())
        queue_task: asyncio.Task[BaseEvent] | None = asyncio.ensure_future(queue.get())
        try:
            while source_task is not None:
                pending = {t for t in (source_task, queue_task) if t is not None}
                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

                if queue_task is not None and queue_task in done:
                    injected = queue_task.result()
                    # Persisted, and deliberately in no `interaction_history`: this
                    # generator runs outside the agent loop and cannot know which
                    # agent's history is the one about to be rendered. The content is
                    # in the conversation, and a later turn builds its history there.
                    await self._add_event(state, injected, context=None)
                    yield injected
                    queue_task = asyncio.ensure_future(queue.get())

                if source_task in done:
                    try:
                        event = source_task.result()
                        yield event
                        source_task = asyncio.ensure_future(source.__anext__())
                    except StopAsyncIteration:
                        source_task = None
        finally:
            # Usually already gone: the end-of-turn arms deregister as soon as they
            # know the turn is ending, so a push arriving during the several awaits
            # that follow is told so rather than accepted and dropped. This covers
            # every other way a turn can finish — an error, a hook break, a suspension
            # for approval, a cancellation.
            self.injection_queue.pop(key, None)
            if queue_task is not None:
                queue_task.cancel()
            if source_task is not None:
                source_task.cancel()
            with contextlib.suppress(Exception):
                await source.aclose()

    async def run(
        self,
        application_name: str,
        user_id: str,
        user_input: BaseEvent,
        root: Agent,
        entrypoint_name: str | None = None,
        entrypoint_context_visibility: ContextVisibility | None = None,
        conversation_id: str | None = None,
        run_config: dict | None = None,
        metadata: dict | None = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Run the agent framework with the given parameters.

        A thin wrapper so that the turn's registration in ``injection_queue`` — which is
        what makes ``push_parts`` answer "delivered" — is removed however this ends:
        normally, by an error, by a suspension for approval, or by the consumer
        walking away. The registration itself happens inside, once there is a
        conversation id to key it by.
        """
        registered: list[tuple[str, str]] = []
        try:
            async for event in self._run(
                application_name,
                user_id,
                user_input,
                root,
                entrypoint_name,
                entrypoint_context_visibility,
                conversation_id,
                run_config,
                metadata,
                registered,
            ):
                yield event
        finally:
            for key in registered:
                self.injection_queue.pop(key, None)

    async def _run(
        self,
        application_name: str,
        user_id: str,
        user_input: BaseEvent,
        root: Agent,
        entrypoint_name: str | None,
        entrypoint_context_visibility: ContextVisibility | None,
        conversation_id: str | None,
        run_config: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
        registered: list[tuple[str, str]],
    ) -> AsyncGenerator[BaseEvent, None]:
        if entrypoint_name is None:
            entrypoint_name = root.name

        entrypoint: Agent | DelegateAgent | None = next(
            filter(lambda agent: agent.name == entrypoint_name, _return_subagents(root)), None
        )

        if entrypoint is None:
            raise ValueError(
                f"Entrypoint agent '{entrypoint_name}' not found in the agent hierarchy."
            )

        entrypoint.context_visibility = (
            entrypoint_context_visibility or entrypoint.context_visibility
        )

        # The turn is in flight from the moment the conversation is named, not from
        # where the agent loop starts: setting one up is several store round trips,
        # and against a real database those are real awaits. A message typed in that
        # window would otherwise be told nothing was running while a turn
        # demonstrably was. `run()` removes the registration again whatever happens
        # after, so registering before the work is safe.
        if conversation_id:
            early = (application_name, conversation_id)
            self.injection_queue[early] = InjectionsQueue(
                racing_queue=asyncio.Queue(),
                scheduled_queue=asyncio.Queue(),
                default_author=root.name,
            )
            registered.append(early)

        state = await self._get_state(
            application_name=application_name,
            user_id=user_id,
            root=root,
            entrypoint=entrypoint,
            conversation_id=conversation_id,
        )
        state.steering_reaches_subagents = self.steering_reaches_subagents

        # Only when the store answered with an id the caller did not supply — one
        # it generated — is there a second key to register. Re-registering the one
        # above would discard anything pushed into it in the meantime.
        key = (state.application_name, state.conversation_id)
        if key not in self.injection_queue:
            self.injection_queue[key] = InjectionsQueue(
                racing_queue=asyncio.Queue(),
                scheduled_queue=asyncio.Queue(),
                default_author=state.entrypoint_agent.name,
            )
            registered.append(key)

        # pass run_config to state
        state.run_config = run_config or {}

        # metadata could not be empty
        if metadata is None:
            metadata = {}

        state.metadata = metadata

        # Set up IDs; AGENT_START is built after hooks so hooks can modify user_input first
        is_new_conversation, _ = await self._setup_conversation_turn(state, user_input)
        # Build the InteractionContext now so on_agent_event hooks fired on the initial
        # USER_INPUT receive a real context. interaction_history starts
        # empty for non-TOOL_FEEDBACK inputs and is populated incrementally below by
        # _add_event(..., context=context) calls.
        await self._refresh_state(state)
        stack_entry = await self.conversation_manager.get_last_agent_stack(
            state.application_name, state.user_id, state.conversation_id
        )
        context = ContextBuilder.create_context(state, user_input, stack_entry)

        # A request to compact is not a turn: no agent runs, no model is called,
        # and nothing is added to the history. Handled here, before the
        # agent-start machinery, because everything below this point exists to
        # start an agent and none of it applies.
        if user_input.event_type == ExecutionEventType.COMPACTION_REQUESTED:
            async for event in self._handle_compaction_request(state, context):
                yield event
            return

        should_continue = True
        # Run ON_AGENT_EVENT hooks on the incoming user input before storing it
        async for on_exec in self.run_on_event_hooks(
            state.current_agent.on_event_hooks,
            user_input,
            context=context,
            agent_name=state.current_agent.name,
            state=state,
        ):
            yield on_exec.hook_event
            user_input = on_exec.current_event
            should_continue = not on_exec.break_execution and (
                on_exec.current_event.event_type != ExecutionEventType.RESPONSE
                or on_exec.current_event.is_partial
            )
            if not should_continue:
                break
        if not should_continue:
            if not on_exec.break_execution:
                await self._add_event(state, user_input, context=context)
                yield user_input
            await self.conversation_manager.empty_agents_stack(
                state.application_name, state.user_id, state.conversation_id
            )
            return

        # Store the user input (TOOL_FEEDBACK events will be saved later on)
        if user_input.event_type == ExecutionEventType.USER_INPUT:
            await self._add_event(state, user_input, context=context)

        # Build AGENT_START from the (possibly hook-modified) user_input, then store/yield it.
        # Hooks do NOT run on AGENT_START — they already fired on USER_INPUT above.
        # Side effect: on_agent_event hooks with on_event_triggers = AGENT_START will not fire on the initial AGENT_START
        event_start = self._build_agent_start_event(state, user_input, is_new_conversation)
        if event_start and event_start.event_type != user_input.event_type:
            if is_new_conversation:
                await self._add_event(state, event_start, context=context)
            yield event_start
        elif is_new_conversation and event_start:
            # TOOL_FEEDBACK resuming: event_start is user_input (already stored above), store only
            await self._add_event(state, event_start, context=context)

        # Point agent_input at the built AGENT_START so its metadata/action reach
        # the planner, context managers and (for a DelegateAgent) the transport.
        # For TOOL_FEEDBACK/AGENT_FEEDBACK resumes event_start IS user_input, so
        # this keeps that event. DelegateAgent distinguishes root vs sub-agent via
        # context.is_subagent (not agent_input.event_type), so the root path is
        # unaffected by agent_input now being an AGENT_START.
        context.agent_input = event_start or user_input

        # A new message where the answer to an approval was expected: the call that
        # was waiting, and the ones held after it, will not run. Each gets a result
        # saying so, or the next request carries calls without results.
        if user_input.event_type == ExecutionEventType.USER_INPUT:
            await self._drop_held_tool_calls(state, context)

        # Handle feedback resumptions before entering the agent loop
        abort: list[bool] = [False]
        if user_input.event_type == ExecutionEventType.TOOL_FEEDBACK:
            resumed: BaseEvent | None = None
            async for event in self._handle_tool_feedback(state, user_input, context, abort):
                yield event
                if event.event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK:
                    return
                if event.event_type in (
                    ExecutionEventType.TOOL_END,
                    ExecutionEventType.TOOL_ERROR,
                ):
                    resumed = event
            if abort[0]:
                return
            # The calls of the same response after the one that was waiting: run
            # now, before the model is asked again, as they would have been had it
            # not stopped to ask. One of them may need an answer in its turn.
            _, still_held = self._held_tool_calls(context)
            context.state.pop(HELD_TOOL_CALLS_KEY, None)
            if still_held:
                outcome = _ToolCallsOutcome()
                async for event in self._run_tool_calls(still_held, state, context, outcome):
                    yield event
                if outcome.paused:
                    return
                if outcome.last is not None:
                    yield outcome.last
                    resumed = outcome.last
                if not outcome.should_continue:
                    return
            # The approved call finishing is a tool-cycle boundary like any other —
            # it just happens out here, because a resume runs the tool before the
            # agent loop is entered. Without this the first boundary of a resumed
            # run is whatever comes *after* the planner has already been given its
            # messages, so a message waiting through the approval reaches the model
            # a whole tool call late. Same rule as the loop's, read from the same
            # seam: narrowing `steering_triggers` has to switch this off too.
            if resumed is not None and resumed.event_type in self.steering_triggers:
                for steered_event in await self._absorb_pending(state, context):
                    yield steered_event

        try:
            state, turn_decision_events = await self._load_turn_resources(state, context)
        except SkillRequiredError as exc:
            refusal = BaseEvent(
                event_type=ExecutionEventType.ERROR_OCCURRED,
                author=state.current_agent.name,
                error_code="RequiredSkillUnavailable",
                error_message=f"Cannot run this turn: {exc}",
            )
            await self._add_event(state, refusal, context=context)
            yield refusal
            # Nothing ran to pop the entrypoint _setup_conversation_turn pushed, and a
            # stale entry makes the next turn a resumption: its AGENT_START is not stored.
            await self.conversation_manager.empty_agents_stack(
                state.application_name, state.user_id, state.conversation_id
            )
            return

        # What the decision model decided while the turn was being prepared.
        # Yielded here rather than where each decision happens, because those
        # run together inside an `asyncio.gather` in a plain coroutine, which
        # cannot yield — and *before* the turn proceeds, so a consumer sees why
        # a tool vanished before it notices that it has.
        for decision_made in turn_decision_events:
            await self._add_event(state, decision_made, context=context)
            yield decision_made

        # The data boundary, when one is configured. Refused here, before the
        # agent produces anything, for the same reason a required skill is: a
        # check the turn can route around by taking another path is not a check.
        blocked = await self._refuse_over_sensitivity(state, context)
        if blocked is not None:
            await self._add_event(state, blocked, context=context)
            yield blocked
            await self.conversation_manager.empty_agents_stack(
                state.application_name, state.user_id, state.conversation_id
            )
            return

        async for event in self._merge_with_injections(
            state, self._execute_agent_loop(state, context)
        ):
            yield event

    async def _load_turn_resources(
        self, state: RunnerState, context: InteractionContext
    ) -> tuple[RunnerState, list[BaseEvent]]:
        """Load the turn's skills and standing instructions, and learn the models'
        context windows, concurrently.

        Independent sources with independent failure modes, so neither waits on
        the other. Both are waited for even when one fails — a refused turn must
        not leave the other load running unobserved — and only then is the first
        failure raised: :class:`SkillRequiredError` is the one the caller turns
        into a refusal, anything else is a bug and propagates.

        Returns the events the decision model produced along the way, for the
        caller to yield. ``gather`` already collects what its coroutines return,
        so they are carried out the way every other value here is rather than
        through a list handed down to be filled in.
        """
        results = await asyncio.gather(
            self._load_turn_skills(state, context),
            self._load_turn_instructions(state, context),
            self._resolve_turn_context_windows(state),
            # Alongside the others: it talks to a different service, it can
            # only ever narrow the tool list, and it must not delay a turn that
            # would otherwise be ready.
            self._decide_tool_visibility(state, context),
            self._decide_context_relevance(state, context),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result
        _, instructions, _context_windows, visibility, relevance = results
        decision_events = [
            *(instructions[1] if isinstance(instructions, tuple) else []),
            *(visibility if isinstance(visibility, list) else []),
            *(relevance if isinstance(relevance, list) else []),
        ]
        return state, decision_events

    async def _load_turn_skills(
        self, state: RunnerState, context: InteractionContext
    ) -> RunnerState:
        """Load every declaring agent's skills once, before any of them runs.

        Once per turn.

        Every agent in the tree is loaded, not only the one starting: which agents
        a turn will reach is not known until it reaches them, and loading at that
        point is what this exists to stop.

        A source that is down degrades to nothing for that agent and is not retried
        on re-entry. An advisory skill must never take a turn down, and a required
        one that never became visible cannot enforce anything either.

        Raises :class:`SkillRequiredError` when an agent's catalogue advertises a
        ``required`` skill whose requirements are unmet — see the call site, which
        turns that into one refused turn rather than a dead session.
        """
        targets: dict[str, tuple[Agent | DelegateAgent, SkillsRuntime]] = {}
        for agent in state.available_agents:
            runtime = getattr(agent, "skills", None)
            # isinstance, not `is not None`: BaseAgent declares no `skills` field,
            # and an agent built as a MagicMock answers every getattr with a
            # truthy child mock. Keyed by name because _return_subagents dedupes
            # only the root, so a sub-agent under two parents appears twice.
            if not isinstance(runtime, SkillsRuntime) or agent.name in targets:
                continue
            targets[agent.name] = (agent, runtime)

        if not targets:
            return state

        logger = get_logger()
        results = await asyncio.gather(
            *(runtime.load(skill_query_for(agent, context)) for agent, runtime in targets.values()),
            return_exceptions=True,
        )
        for name, result in zip(targets, results, strict=True):
            if isinstance(result, asyncio.CancelledError):
                # return_exceptions=True hands a child's cancellation back as a
                # value instead of propagating it; swallowing it would leave the
                # turn running inside a cancelled task.
                raise result
            if isinstance(result, BaseException):
                logger.warning(f"skills: could not load for '{name}': {type(result).__name__}")
                logger.debug(f"skills load detail for '{name}': {result}")
                state.skills[name] = LoadedSkills()
                continue

            state.skills[name] = result
            logger.info(
                f"skills: loaded {len(result.index.skills)} for '{name}' "
                f"from {targets[name][1].source.describe()}"
            )
            # Checked here — after the failed-load `continue` above, never before
            # it. The two look alike and are not: a source that could not be read
            # never told us what its skills declare, so a file that failed to load
            # cannot be known to have said `required`. Enforcement is a property
            # of something we successfully read.
            raise_if_required_unavailable(result.index, agent_name=name)
        return state

    async def _load_turn_instructions(
        self,
        state: RunnerState,
        context: InteractionContext | None = None,
    ) -> tuple[RunnerState, list[BaseEvent]]:
        """Load every declaring agent's standing instructions, once per turn.

        Per turn, like skills, so an edited ``SOUL.md`` reaches the next message
        and a new conversation needs nothing special. Unlike skills nothing here
        can refuse a turn: a source that fails gives that agent no standing
        instructions for this turn, with a warning.

        Returns the state and whatever the conditional-section decision had to
        say, which the caller passes on to whoever can yield it.
        """
        targets: dict[str, InstructionRuntime] = {}
        for agent in state.available_agents:
            runtime = getattr(agent, "standing_instructions", None)
            # isinstance for the same MagicMock reason as `_load_turn_skills`.
            if isinstance(runtime, InstructionRuntime) and agent.name not in targets:
                targets[agent.name] = runtime
        if not targets:
            return state, []

        results = await asyncio.gather(
            *(runtime.load() for runtime in targets.values()), return_exceptions=True
        )
        logger = get_logger()
        for name, result in zip(targets, results, strict=True):
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, BaseException):
                logger.warning(
                    f"instructions: could not load for '{name}': {type(result).__name__}"
                )
                logger.debug(f"instructions load detail for '{name}': {result}")
                state.instructions[name] = AgentInstructions()
                continue
            state.instructions[name] = result

        # Conditional `##` sections, resolved once per user turn — never inside
        # the loop, because this text is the cached part of the prompt.
        return state, await self._apply_conditional_instructions(state, targets, context)

    async def _resolve_turn_context_windows(self, state: RunnerState) -> RunnerState:
        """Ask every provider in the tree for the window it does not declare, once
        per turn — before any request, so the first one is already measured against
        it. Answers are cached across turns (`resolve_context_window`), and a provider
        that cannot answer keeps an unknown window: nothing here can fail a turn."""
        providers = {
            id(provider): provider
            for agent in state.available_agents
            # isinstance for the MagicMock reason given in `_load_turn_skills`.
            if isinstance(provider := getattr(agent, "llm_provider", None), BaseLLMProvider)
        }
        await asyncio.gather(*(resolve_context_window(p) for p in providers.values()))
        return state

    async def _decide_tool_visibility(
        self,
        state: RunnerState,
        context: InteractionContext | None,
    ) -> list[BaseEvent]:
        """Let a decision model hide tools this turn plainly does not need.

        Once per user turn, and stashed on ``context.state``: tool definitions
        are at the front of the prompt, so deciding again mid-turn would throw
        away the prefix cache on every iteration. Only the agent starting the
        turn is asked — hiding across a whole tree would pay for agents the turn
        may never reach.

        Returns what it decided, for a caller that can yield it; nothing to say
        is an empty list, which is also every path that declines to ask.
        """
        if context is None:
            return []
        agent = state.current_agent
        runtime = runtime_of(agent)
        if runtime is None or not runtime.enabled("tools"):
            return []
        task = _turn_task(context)
        if not task:
            return []
        try:
            visibility = await decide_tool_visibility(
                runtime,
                tools=list(getattr(agent, "tools", None) or []),
                task=task,
                context=context,
            )
            if visibility is None:
                return []
            return [
                decision_event(
                    agent.name,
                    decision_metadata(
                        "tools",
                        mode=runtime.mode("tools"),
                        acted=True,
                        summary=f"{len(visibility.hidden)} tool(s) hidden this turn",
                        hidden=list(visibility.hidden),
                    ),
                )
            ]
        except Exception as exc:
            get_logger().warning(f"decisions: tool visibility failed: {type(exc).__name__}")
            get_logger().debug(f"tool visibility detail: {exc}")
            return []

    async def _refuse_over_sensitivity(
        self, state: RunnerState, context: InteractionContext | None
    ) -> BaseEvent | None:
        """The refusal this turn earns on data sensitivity, or ``None`` to proceed.

        Only an agent whose provider declares ``max_sensitivity`` is checked at
        all, so nothing changes for a configuration that does not use this.
        """
        if context is None:
            return None
        agent = state.current_agent
        provider = getattr(agent, "llm_provider", None)
        if provider_ceiling(provider) is None:
            return None

        runtime = runtime_of(agent)
        try:
            rating = await rate_sensitivity(runtime, task=_turn_task(context), state=context.state)
        except Exception as exc:
            get_logger().warning(f"decisions: sensitivity check failed: {type(exc).__name__}")
            get_logger().debug(f"sensitivity detail: {exc}")
            rating = Rating(level=Sensitivity.REGULATED, known=False)

        allowed, reason = sensitivity_allows(
            provider,
            rating,
            on_unknown=runtime.on_unknown("sensitivity") if runtime is not None else "block",
        )
        if allowed:
            return None
        get_logger().warning(f"decisions[sensitivity]: turn refused — {reason}")
        return BaseEvent(
            event_type=ExecutionEventType.ERROR_OCCURRED,
            author=agent.name,
            error_code="SensitivityPolicy",
            error_message=reason or "Blocked by the data-sensitivity policy.",
        )

    async def _decide_context_relevance(
        self,
        state: RunnerState,
        context: InteractionContext | None,
    ) -> list[BaseEvent]:
        """Judge which old tool outputs this turn still needs.

        The verdict is left on ``context.state`` for compaction to use if it
        runs. Taken here, once per turn, rather than inside the compactor: the
        compactor is on the critical path of a request that is already too big,
        and it has no idea what the user asked for.

        Returns what it decided, for a caller that can yield it.
        """
        if context is None:
            return []
        runtime = runtime_of(state.current_agent)
        if runtime is None or not runtime.enabled("context_ladder"):
            return []
        task = _turn_task(context)
        if not task:
            return []
        try:
            marked = await decide_context_relevance(
                runtime,
                history=list(getattr(context, "conversation_history", None) or []),
                task=task,
                state=context.state,
            )
            if not marked:
                return []
            return [
                decision_event(
                    state.current_agent.name,
                    decision_metadata(
                        "context_ladder",
                        mode=runtime.mode("context_ladder"),
                        acted=True,
                        summary=f"{len(marked)} old tool output(s) marked as no longer needed",
                    ),
                )
            ]
        except Exception as exc:
            get_logger().warning(f"decisions: context relevance failed: {type(exc).__name__}")
            get_logger().debug(f"context relevance detail: {exc}")
            return []

    async def _apply_conditional_instructions(
        self,
        state: RunnerState,
        targets: dict[str, InstructionRuntime],
        context: InteractionContext | None,
    ) -> list[BaseEvent]:
        """Keep the sections that apply to this turn, for agents that can tell.

        Best-effort in every direction: an agent with no decision model, a model
        that will not answer, or an instruction with no `when:` anywhere all
        keep the full text, which is the behaviour without any of this.

        Returns one event per agent whose text was narrowed — several, unlike
        the other two, because every agent in the tree declaring instructions
        is resolved here.
        """
        if context is None:
            return []
        task = _turn_task(context)
        if not task:
            return []

        events: list[BaseEvent] = []
        by_name = {agent.name: agent for agent in state.available_agents}
        for name in targets:
            runtime = runtime_of(by_name.get(name))
            if runtime is None or not runtime.enabled("instructions"):
                continue
            loaded = state.instructions.get(name)
            if loaded is None:
                continue
            try:
                resolved = await apply_conditional_instructions(loaded, task, runtime)
                state.instructions[name] = resolved
                if resolved.skipped_sections:
                    events.append(
                        decision_event(
                            name,
                            decision_metadata(
                                "instructions",
                                mode=runtime.mode("instructions"),
                                acted=True,
                                summary=(
                                    f"{len(resolved.skipped_sections)} instruction "
                                    "section(s) did not apply"
                                ),
                                skipped=list(resolved.skipped_sections),
                            ),
                        )
                    )
            except Exception as exc:
                get_logger().warning(
                    f"instructions: conditional sections failed for '{name}': {type(exc).__name__}"
                )
                get_logger().debug(f"conditional instructions detail: {exc}")
        return events

    def _execute_kwargs(self, state: RunnerState, agent: Agent | DelegateAgent) -> dict[str, Any]:
        """The ``skills=`` / ``instructions=`` keywords for this agent's
        ``execute``, or nothing at all.

        Empty when there is nothing to hand over, so an agent that declares
        neither — every ``DelegateAgent`` and ``WrappedAgent``, and any
        ``execute`` written before these arguments — is called exactly as it was
        before.

        ``Agent`` is the only class with these fields, so it is the only one whose
        ``execute`` accepts the keywords. That is checked rather than assumed: the
        maps are keyed by name, and a name is not a type.

        A helper rather than four copies of the condition: this is called from four
        places across four methods, and a fifth that forgot it would silently run
        an agent without its skills and standing instructions.
        """
        if not isinstance(agent, Agent):
            return {}
        kwargs: dict[str, Any] = {}
        if (skills := state.skills.get(agent.name)) is not None:
            kwargs["skills"] = skills
        if (instructions := state.instructions.get(agent.name)) is not None:
            kwargs["instructions"] = instructions
        return kwargs

    async def _handle_compaction_request(
        self, state: RunnerState, context: InteractionContext
    ) -> AsyncGenerator[BaseEvent, None]:
        """Compact now, because someone asked.

        Works on the conversation history rather than on rendered messages, which
        is what keeps this out of the two planner loops — they are copies of each
        other, and a feature added to one of them has already shipped missing
        from the other once.

        The result is a record, not a rewritten history: the events are untouched
        and the dropping is applied when the next request is rendered. That is
        also why it survives — a compaction of a transient message list would be
        undone by the next re-render.
        """
        agent = state.current_agent
        planner = getattr(agent, "planner", None)
        compactor = getattr(planner, "compactor", None)

        if planner is None or compactor is None:
            # Some planners have no compactor at all. Saying so beats a silent
            # no-op, which is indistinguishable from a compaction that worked.
            yield BaseEvent(
                event_type=ExecutionEventType.CONTEXT_COMPACTED,
                author=agent.name,
                content=[Part(text="this agent's planner does not support compaction")],
                metadata={"changed": False, "requested": True},
            )
            return

        yield BaseEvent(
            event_type=ExecutionEventType.CONTEXT_COMPACTION_START,
            author=agent.name,
            content=[Part(text="Compacting the conversation on request")],
            metadata={"requested": True},
        )

        # Rendered as the next request will be, so the saving the event reports
        # is counted on the text the model reads rather than guessed from events.
        rendered = await planner.context_manager.get_context(context=context, agent=agent)
        done = compactor.request_from_history(
            context.conversation_history, context.state, agent.name, rendered=rendered
        )

        # Persisted explicitly: the usual write-back rides along with events
        # stored during an agent loop, and there is no agent loop here.
        await self.conversation_manager.update_conversation_state(
            state.application_name,
            state.user_id,
            state.conversation_id,
            context.state,
        )
        state.conversation_info.state = context.state

        yield done

    async def _get_state(
        self,
        application_name: str,
        user_id: str,
        root: Agent | DelegateAgent,
        entrypoint: Agent | DelegateAgent,
        conversation_id: str | None,
    ) -> RunnerState:
        """Retrieve or create runner state with conversation and agent setup."""
        if root is None:
            raise ValueError("Root agent must be provided.")

        if entrypoint is None:
            entrypoint = root

        available_agents = _return_subagents(root)

        if not await self.conversation_manager.check_conversation_exists(
            application_name, user_id, conversation_id
        ):
            conversation_info = await self.conversation_manager.create_conversation(
                application_name, user_id, conversation_id
            )
        else:
            conversation_info = await self.conversation_manager.get_conversation(
                application_name, user_id, conversation_id
            )
        conversation_id = conversation_info.conversation_id

        # Determine current agent (resume from stack or use entrypoint)
        current_agent = await self._get_current_agent(
            application_name, user_id, conversation_id, entrypoint, available_agents
        )

        return RunnerState(
            application_name=application_name,
            user_id=user_id,
            conversation_id=conversation_id,
            conversation_info=conversation_info,
            current_agent=current_agent,
            available_agents=available_agents,
            entrypoint_agent=root,
        )

    async def _get_current_agent(
        self,
        application_name: str,
        user_id: str,
        conversation_id: str,
        entrypoint: Agent | DelegateAgent,
        available_agents: list[Agent | DelegateAgent],
    ) -> Agent | DelegateAgent:
        """Get the current agent from stack or return entrypoint."""
        stack_entry = await self.conversation_manager.get_last_agent_stack(
            application_name, user_id, conversation_id
        )

        if stack_entry:
            return next(
                (agent for agent in available_agents if agent.name == stack_entry.agent_name),
                entrypoint,
            )
        return entrypoint

    async def _setup_conversation_turn(
        self, state: RunnerState, user_input: BaseEvent
    ) -> tuple[bool, str | None]:
        """Set up IDs for a new/resumed turn. Must be called before ON_AGENT_EVENT hooks run.

        Returns (is_new_conversation, stack_agent_name) where:
        - is_new_conversation is True only for new conversations whose AGENT_START must be persisted.
        - stack_agent_name is the agent name from the existing stack entry (resumed conversations only).
        """
        stack_entry = await self.conversation_manager.get_last_agent_stack(
            state.application_name, state.user_id, state.conversation_id
        )

        if not stack_entry:  # New conversation
            state.interaction_id = InteractionContext.generate_interaction_id()
            state.turn_id = user_input.turn_id or InteractionContext.generate_turn_id()

            user_input.conversation_id = state.conversation_id
            user_input.interaction_id = state.interaction_id
            user_input.turn_id = state.turn_id

            await self.conversation_manager.add_agent_to_stack(
                application_name=state.application_name,
                user_id=state.user_id,
                conversation_id=state.conversation_id,
                interaction_id=state.interaction_id,
                turn_id=state.turn_id,
                agent_name=state.current_agent.name,
            )
            return True, None

        else:  # Resuming conversation
            state.interaction_id = stack_entry.interaction_id
            if user_input.event_type == ExecutionEventType.USER_INPUT:
                state.turn_id = user_input.turn_id or InteractionContext.generate_turn_id()
            else:
                state.turn_id = (
                    user_input.turn_id
                    or stack_entry.turn_id
                    or InteractionContext.generate_turn_id()
                )

            user_input.conversation_id = state.conversation_id
            user_input.interaction_id = state.interaction_id
            user_input.turn_id = state.turn_id

            return False, stack_entry.agent_name

    def _build_agent_start_event(
        self,
        state: RunnerState,
        user_input: BaseEvent,
        is_new_conversation: bool,
    ) -> BaseEvent:
        """Build the AGENT_START event from the (possibly hook-modified) user_input.

        Must be called after ON_AGENT_EVENT hooks have run on user_input so that any
        content modifications made by those hooks are reflected in the AGENT_START event.

        The AGENT_START inherits the user_input's ``metadata`` (e.g. ``user_info``) — it
        does NOT pull from ``run_config``. ``run_config`` stays the tool channel
        (``context.run_config``); copying it into events would leak caller secrets (e.g.
        auth tokens) to delegated agents. Whoever builds the user_input owns event metadata.
        """
        # Delegate agent path: the incoming AGENT_START is already the real start event.
        # Return it as-is so context.agent_input points to the original and the runner
        # does not yield a duplicate (same object → elif branch stores without yielding).
        if user_input.event_type == ExecutionEventType.AGENT_START:
            return user_input

        if is_new_conversation:  # New conversation
            task_text = " ".join(p.text for p in (user_input.content or []) if p.text)
            event = BaseEvent(
                event_type=ExecutionEventType.AGENT_START,
                author=state.current_agent.name,
                content=user_input.content,
                conversation_id=state.conversation_id,
                interaction_id=state.interaction_id,
                turn_id=state.turn_id,
                action=AgentAction(
                    name=state.current_agent.name,
                    parameters={"task": task_text},
                    thought="",
                ),
            )
            event.metadata.update(copy.deepcopy(user_input.metadata))
            return event

        # Resuming conversation
        if user_input.event_type == ExecutionEventType.USER_INPUT:
            task_text = " ".join(p.text for p in (user_input.content or []) if p.text)
            event = BaseEvent(
                event_type=ExecutionEventType.AGENT_START,
                author=state.current_agent.name,
                content=user_input.content,
                conversation_id=state.conversation_id,
                interaction_id=state.interaction_id,
                turn_id=state.turn_id,
                action=AgentAction(
                    name=state.current_agent.name,
                    parameters={"task": task_text},
                    thought="",
                ),
            )
            event.metadata.update(copy.deepcopy(user_input.metadata))
            return event

        # TOOL_FEEDBACK / AGENT_FEEDBACK resumption: treat user_input itself as the start event
        return user_input

    async def _handle_tool_feedback(
        self,
        state: RunnerState,
        tool_feedback_input: BaseEvent,
        context: InteractionContext,
        abort: list[bool],
    ) -> AsyncGenerator[BaseEvent, None]:
        logger = get_logger()

        # The call the answer is for: the one recorded as waiting when the run
        # paused, which is not the latest TOOL_START when the response asked for
        # several (they are all stored before any runs); else, as before, the latest.
        waiting_id, _ = self._held_tool_calls(context)
        starts = [
            x
            for x in reversed(context.interaction_history)
            if x.event_type == ExecutionEventType.TOOL_START
        ]
        previous_event: BaseEvent | None = next(
            (x for x in starts if waiting_id and getattr(x.action, "call_id", None) == waiting_id),
            starts[0] if starts else None,
        )
        if previous_event is None:
            logger.error("TOOL_FEEDBACK received but no prior TOOL_START found in history")
            abort[0] = True
            return

        # Any: mypy reads `BaseEvent.action`, an unresolved union, as None.
        reply_action: Any = tool_feedback_input.action
        waiting_action: Any = previous_event.action
        tool_name = waiting_action.name

        # The delegate's own approval request (no tool of ours): it gets the reply.
        if (
            isinstance(state.current_agent, DelegateAgent)
            and state.current_agent.get_tool_by_name(reply_action.name) is None
        ):
            return
        # What runs is decided by the waiting call, never by the reply. A reply for
        # another call is ignored, not refused: a stale one must not cancel this call.
        if self._answers_another_call(reply_action, waiting_action, context):
            logger.warning(
                f"approval: ignored a reply that is not for the waiting '{tool_name}' call"
            )
            abort[0] = True
            return
        tool: FunctionTool | DelegateTool | None = state.current_agent.get_tool_by_name(tool_name)
        if tool_feedback_input.action.type == "AuthenticationAction":
            # Auth completed. Restore the original ToolAction from TOOL_START so that
            # _execute_tool() receives the original call_id and parameters unchanged.
            if tool is None:
                logger.error(
                    f"Authentication received for unknown tool '{tool_feedback_input.action.name}'"
                )
                abort[0] = True
                return
            if tool_feedback_input.action.success == AuthenticationResult.FAILURE:
                error_action = previous_event.action.model_copy(
                    update={
                        "observation": f"Authentication failed for tool {tool_feedback_input.action.name}. It is not possible to execute the tool."
                    }
                )
                last_tool_event = self._build_tool_error_event(state, context, error_action)
            elif tool_feedback_input.action.success == AuthenticationResult.USER_REFUSED:
                refusal_action = previous_event.action.model_copy(
                    update={
                        "observation": f"User refused to authenticate for tool {tool_feedback_input.action.name}. {_USER_REFUSED_TOOL_CALLS}"
                    }
                )
                last_tool_event = BaseEvent(
                    event_type=ExecutionEventType.TOOL_END,
                    author=state.current_agent.name,
                    conversation_id=state.conversation_id,
                    interaction_id=context.interaction_id,
                    turn_id=state.turn_id,
                    action=refusal_action,
                )
            else:
                tool_feedback_input.action = previous_event.action

                # If the user modified parameters before auth interrupted execution, a TOOL_FEEDBACK
                # event was saved after the TOOL_START. Recover it and apply its parameters so the
                # re-execution uses what the user actually approved.
                _history = context.interaction_history
                _tool_start_idx = next(
                    (
                        len(_history) - 1 - i
                        for i, x in enumerate(reversed(_history))
                        if x.event_type == ExecutionEventType.TOOL_START
                    ),
                    None,
                )
                _latest_tool_feedback = None
                if _tool_start_idx is not None:
                    _latest_tool_feedback = next(
                        (
                            x
                            for x in reversed(_history[_tool_start_idx + 1 :])
                            if x.event_type == ExecutionEventType.TOOL_FEEDBACK
                        ),
                        None,
                    )

                auth_params_changed_msg = None
                if _latest_tool_feedback is not None:
                    tool_feedback_input.action = tool_feedback_input.action.model_copy(
                        update={"parameters": _latest_tool_feedback.action.parameters}
                    )
                    auth_params_changed_msg = self._params_changed_message(
                        tool_feedback_input.action.name, tool_feedback_input.action.parameters
                    )
                captured: list[BaseEvent] = []
                async for ev in self._stream_tool_events(
                    tool, state, context, tool_feedback_input, captured
                ):
                    yield ev
                last_tool_event = self._finalize_tool_event(
                    captured[0] if captured else None,
                    previous_event.action,
                    "authentication",
                    state,
                    context,
                    auth_params_changed_msg,
                )

        elif (
            refusal := await self._refuse_edited_call(
                tool, previous_event, tool_feedback_input, state, context
            )
        ) is not None:
            last_tool_event = refusal
        elif tool_feedback_input.action.approve:
            if tool is None:
                logger.error(
                    f"Approval received for unknown tool '{tool_feedback_input.action.name}'"
                )
                abort[0] = True
                return

            new_approve_action = tool_feedback_input.action.to_tool_action()
            # Restore the original call_id from the TOOL_START so the remote
            # runner can match the TOOL_END to its pending tool-call future.
            # The client's TOOL_FEEDBACK carries a fresh UUID by default.
            new_approve_action.call_id = previous_event.action.call_id
            requested_parameters = previous_event.action.parameters
            # USER_CONFIRM cards are read-only: the user can only approve or refuse,
            # never edit. The values that come back are therefore a round-trip of what
            # was *displayed*, not user intent, so the TOOL_START is the authority on
            # what to execute. This keeps the display layer free to rewrite a value for
            # readability (e.g. a front-end prettifying opaque commands/queries) without
            # ever changing the executed arguments.
            # PARAMS_UPDATE is untouched: there the incoming values *are* the intent.
            if tool.approval_type == ApprovalType.USER_CONFIRM and requested_parameters is not None:
                new_approve_action.parameters = copy.deepcopy(requested_parameters)
            last_tool_event = None

            # Need to save the tool_feedback_input event before executing the tool in case the execution is stopped
            # for an auth event. The auth event handling will need the information on the tool parameters to re-execute the tool after the auth is completed.
            params_changed_msg = None
            if self.parameters_changed(
                new_approve_action.parameters, previous_event.action.parameters
            ):
                params_changed_msg = self._params_changed_message(
                    new_approve_action.name, new_approve_action.parameters
                )
                new_approve_action.observation = params_changed_msg
                tool_feedback_input.action.observation = params_changed_msg

            # A standing grant is recorded here, not by the front-end: the
            # interaction context the grant lives on only exists on this side.
            # It uses the *approved* parameters, which may have been edited in
            # the prompt — remembering the proposal instead would grant
            # permission for something the user never saw.
            requested_scope = getattr(tool_feedback_input.action, "remember_scope", None)
            if requested_scope and requested_scope != ApprovalScope.ONCE:
                grant_request = build_request(
                    tool,
                    dict(new_approve_action.parameters or {}),
                    agent_name=state.current_agent.name,
                    user_id=state.user_id,
                    conversation_id=state.conversation_id,
                )
                stored = await self.approval_policy.remember(
                    grant_request,
                    requested_scope,
                    context=context,
                    pattern=getattr(tool_feedback_input.action, "remember_pattern", None),
                )
                if stored:
                    remembered = (
                        getattr(tool_feedback_input.action, "remember_pattern", None)
                        or grant_request.subject
                    )
                    logger.info(f"approval: remembered '{remembered}' for {requested_scope}")
                else:
                    logger.warning(
                        f"approval: no store for scope '{requested_scope}'; "
                        "the approval applies to this call only"
                    )

            tool_feedback_input.action = new_approve_action
            await self._add_event(state, tool_feedback_input, context=context)

            if isinstance(tool, MCPTool) and tool.use_oauth:
                auth_metadata = await tool.check_auth_credentials()
                if not auth_metadata.credentials_valid:
                    async for paused in self._pause_for_auth(
                        tool_feedback_input, state, context, auth_metadata
                    ):
                        yield paused
                    return

            approval_captured: list[BaseEvent] = []
            async for ev in self._stream_tool_events(
                tool, state, context, tool_feedback_input, approval_captured
            ):
                yield ev
            last_tool_event = self._finalize_tool_event(
                approval_captured[0] if approval_captured else None,
                tool_feedback_input.action,
                "approval",
                state,
                context,
                params_changed_msg,
            )

        else:
            refusal_message = f"User refused to execute the tool {tool_feedback_input.action.name}. {_USER_REFUSED_TOOL_CALLS}"
            new_refuse_action = tool_feedback_input.action.to_tool_action()
            # Restore the original call_id so the remote runner can resolve its future.
            new_refuse_action.call_id = previous_event.action.call_id
            new_refuse_action.observation = refusal_message
            last_tool_event = BaseEvent(
                event_type=ExecutionEventType.TOOL_END,
                author=previous_event.author,
                interaction_id=context.interaction_id,
                action=new_refuse_action,
            )
            tool_feedback_input.action.observation = refusal_message
            await self._add_event(state, tool_feedback_input, context=context)

        if tool:
            # Stored before it is yielded, like every other tool result: storing stamps its ids.
            await self._add_event(state, last_tool_event, context=context)  # type: ignore[possibly-undefined]
            yield last_tool_event

        # For DelegateAgent: relay the result back so the remote runner can resume.
        # The delegate agent never sees TOOL_NEEDS_FEEDBACK; it only ever receives
        # TOOL_END (with result or refusal) as if HITL never happened.
        if (
            isinstance(state.current_agent, DelegateAgent)
            and state.current_agent.transport is not None
        ):
            await state.current_agent.transport.send_result(
                last_tool_event,  # type: ignore[possibly-undefined]
                conversation_id=state.conversation_id,
            )

        # Recover the AGENT_START that seeded the current agent's task
        context.agent_input = next(
            filter(
                lambda x: (
                    x.event_type == ExecutionEventType.AGENT_START
                    and getattr(x.action, "name", None) == state.current_agent.name
                ),
                reversed(context.conversation_history),
            ),
            context.agent_input,
        )

    async def _execute_agent_loop(
        self,
        state: RunnerState,
        context: InteractionContext,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Execute the main agent loop with proper event handling."""
        logger = get_logger()

        logger.info("Runner: starting agent loop")

        # Run pre-hooks of the current agent before starting the planner loop
        should_continue = True
        async for hook_exec in self._apply_agent_hooks(
            hooks=state.current_agent.pre_hooks,
            state=state,
            context=context,
            agent=state.current_agent,
        ):
            context = hook_exec.context
            if should_continue:
                should_continue = not hook_exec.break_execution
            yield hook_exec.event

        agent_iterator = state.current_agent.execute(
            context, **self._execute_kwargs(state, state.current_agent)
        )
        # Pending LLM response to feed back to the planner via asend()
        # when a DelegateLLMProvider is in use.
        pending_llm_response: LLMResponse | None = None
        # Calls of one response held until the last of them arrives: see the
        # TOOL_START arm.
        held_tool_calls: list[BaseEvent] = []

        try:
            while state.current_agent is not None and should_continue:
                try:
                    event: BaseEvent
                    yielded = await agent_iterator.asend(pending_llm_response)
                    event, context = yielded

                    # Compatibility: some tests/legacy mocks yield (event, context)
                    # from the inner execute generator. In that case the wrapped
                    # iterator returns ((event, context), context).
                    if (
                        isinstance(event, tuple)
                        and len(event) == 2
                        and isinstance(event[0], BaseEvent)
                    ):
                        nested_context = event[1]
                        event = event[0]
                        if isinstance(nested_context, InteractionContext):
                            context = nested_context

                    pending_llm_response = None
                except StopAsyncIteration:
                    break

                event = self._ensure_event_identifiers(state=state, event=event, context=context)

                # Run ON_AGENT_EVENT hooks of the current agent on each emitted event
                if agent_on_event_hooks := state.current_agent.on_event_hooks:
                    async for on_exec in self.run_on_event_hooks(
                        agent_on_event_hooks,
                        event,
                        context,
                        agent_name=state.current_agent.name,
                        state=state,
                    ):
                        yield on_exec.hook_event
                        event = on_exec.current_event
                        should_continue = not on_exec.break_execution
                        if not should_continue:
                            break
                    if not should_continue:
                        break

                if (
                    event.event_type == ExecutionEventType.RESPONSE
                    and context.is_subagent
                    and (event.need_feedback or not event.is_partial)
                ):
                    # This response is converted to an AGENT_END below (see the
                    # RESPONSE arm): mark the persisted event so context managers
                    # don't render it as a second assistant answer next to the
                    # synthesized AGENT_END, and consumers don't show it.
                    event.metadata[METADATA_CONVERTED_TO_AGENT_END] = True

                if event.event_type != ExecutionEventType.RESPONSE_TOKEN:
                    await self._add_event(state, event, context=context)
                    await self.conversation_manager.update_conversation_state(
                        state.application_name,
                        state.user_id,
                        state.conversation_id,
                        context.state,
                    )

                # Sync in-memory state to prevent loss on next context refresh
                # Without this, custom keys in context.state are lost when creating next InteractionContext
                state.conversation_info.state = context.state

                # Handle response events (end of conversation)
                match event.event_type:
                    case ExecutionEventType.RESPONSE:
                        if context.is_subagent:
                            # Delegation semantics: subagents never speak to the user.
                            # Partials are persisted (line above) but not yielded;
                            # terminal and need_feedback responses are converted into
                            # an AGENT_END so control returns to the caller with the
                            # content as the observation (parity with the planner's
                            # use_stop flow and DelegateAgent's remote conversion).
                            if event.is_partial and not event.need_feedback:
                                continue
                            event = self._build_subagent_end_event(event, state)
                            await self._add_event(state, event, context=context)
                            transition = _ParentTransition(context=context)
                            async for ev in self._return_to_parent(
                                state, event, context, agent_iterator, should_continue, transition
                            ):
                                yield ev
                            context = transition.context
                            should_continue = transition.should_continue
                            if transition.iterator is None:
                                break
                            agent_iterator = transition.iterator
                            await self.conversation_manager.update_conversation_state(
                                state.application_name,
                                state.user_id,
                                state.conversation_id,
                                context.state,
                            )
                            # Yield the synthesized AGENT_END for observability,
                            # mirroring the bottom-of-loop yield of a planner-emitted one.
                            yield event
                            continue

                        steered: list[BaseEvent] = []
                        if not event.need_feedback and not event.is_partial:
                            # The turn ends on this answer — unless a message arrived
                            # during the last model call, which has no later boundary
                            # left to be taken in at.
                            #
                            # Taken in here, before the answer is yielded, because a
                            # consumer receiving the answer of a turn that really is
                            # ending has always found the agents stack already
                            # emptied. Echoed after it (below), because the user has
                            # to see the answer before the message that follows it.
                            steered = await self._absorb_for_steering(state, context)
                            if not steered:
                                # Nothing owed an answer, so the turn really is
                                # ending. Deregistering *here* rather than leaving it
                                # to the wrapper's teardown is the difference between
                                # telling a pusher "too late" and accepting content
                                # into a run that is about to discard it: everything
                                # below — emptying the stack, the answer itself, the
                                # post-agent hooks, the synthesized AGENT_END — is a
                                # suspension point a push could land in. No await
                                # between the check above and this line, so nothing
                                # can slip between them.
                                self.injection_queue.pop(
                                    (state.application_name, state.conversation_id), None
                                )
                                await self.conversation_manager.empty_agents_stack(
                                    state.application_name, state.user_id, state.conversation_id
                                )
                        # Yield RESPONSE when:
                        # - agent is not streaming (no RESPONSE_TOKEN sequence to replace it), OR
                        # - the event carries FORWARD_VERBATIM metadata from the active
                        #   user-facing agent, OR
                        # - the event needs feedback from the user:
                        #   feedback responses are emitted as a single direct yield,
                        #   not preceded by RESPONSE_TOKEN streaming, so dropping
                        #   them under stream=True loses the user-facing question.
                        if (
                            not state.current_agent.stream
                            or event.metadata.get(METADATA_FORWARD_VERBATIM)
                            or event.need_feedback
                        ):
                            yield event
                        if event.need_feedback:
                            return
                        if steered:
                            for steered_event in steered:
                                yield steered_event
                            agent_iterator = await self._reenter_after_steering(
                                state, context, agent_iterator, len(steered)
                            )
                            continue
                        if not event.is_partial:
                            # Final response - close agent and execute post-hooks before exiting
                            try:
                                await agent_iterator.aclose()
                            finally:
                                # Execute post-hooks of the ending agent (must run even if aclose fails)
                                async for hook_exec in self._apply_agent_hooks(
                                    hooks=state.current_agent.post_hooks,
                                    state=state,
                                    context=context,
                                    agent=state.current_agent,
                                ):
                                    context = hook_exec.context
                                    if should_continue:
                                        should_continue = not hook_exec.break_execution
                                    yield hook_exec.event

                            # Yield AGENT_END event and exit
                            event = BaseEvent(
                                event_type=ExecutionEventType.AGENT_END,
                                author=event.author,
                                interaction_id=event.interaction_id,
                                turn_id=state.turn_id,
                            )
                            if agent_on_event_hooks := state.current_agent.on_event_hooks:
                                async for on_exec in self.run_on_event_hooks(
                                    agent_on_event_hooks,
                                    event,
                                    context,
                                    agent_name=state.current_agent.name,
                                    state=state,
                                ):
                                    yield on_exec.hook_event
                                    event = on_exec.current_event
                            event = self._ensure_event_identifiers(
                                state=state, event=event, context=context
                            )
                            await self._add_event(state, event, context)
                            yield event
                            break

                    # Handle agent call (strategy emits AGENT_CALL when delegating to a sub-agent)
                    case ExecutionEventType.AGENT_CALL:
                        if isinstance(state.current_agent, DelegateAgent):
                            # AGENT_CALL from a remote delegate. The remote runner gated this
                            # call against the REMOTE agent's handoff_scope; locally we must
                            # still enforce THIS delegate's own handoff_scope before
                            # transitioning into a locally-known agent. A target that is
                            # unknown locally OR outside the delegate's local handoff_scope is
                            # handled entirely on the remote side — yield the event for
                            # observability and let the runner continue consuming the same
                            # DelegateAgent iterator (mirrors the TOOL_START pattern for
                            # unknown remote tools).
                            target_name = (
                                event.action.name
                                if isinstance(event.action, AgentAction) and event.action.name
                                else None
                            )
                            local_agent = (
                                next(
                                    (a for a in state.available_agents if a.name == target_name),
                                    None,
                                )
                                if target_name
                                else None
                            )
                            local_handoff_targets = {
                                agent.name
                                for agent in ContextBuilder._get_handoff_targets(
                                    state.current_agent, state.entrypoint_agent
                                )
                                if agent.name != state.current_agent.name
                            }
                            if local_agent is None or target_name not in local_handoff_targets:
                                # A locally-KNOWN target that is off-scope is a denied handoff
                                # (security-relevant): audit it. An unknown target is genuinely
                                # handled on the remote side — no denial, stay quiet.
                                if local_agent is not None:
                                    logger.warning(
                                        f"Relayed AGENT_CALL denied: delegate "
                                        f"{state.current_agent.name!r} cannot hand off to local "
                                        f"agent {target_name!r} (outside its "
                                        f"handoff_scope={state.current_agent.handoff_scope})"
                                    )
                                event = self._ensure_event_identifiers(
                                    state=state, event=event, context=context
                                )
                                yield event
                                continue
                            # Target is locally known AND within the delegate's local
                            # handoff_scope — fall through to normal handoff handling.

                        try:
                            await agent_iterator.aclose()
                        except Exception as exc:
                            logger.warning(
                                "agent_iterator.aclose() failed during agent transition: "
                                f"{type(exc).__name__}"
                            )
                            logger.debug(f"agent_iterator.aclose() detail: {exc}", exc_info=True)

                        stopped_by_hook = False
                        async for hook_exec in self._apply_agent_hooks(
                            hooks=state.current_agent.post_hooks,
                            state=state,
                            context=context,
                            agent=state.current_agent,
                        ):
                            context = hook_exec.context
                            yield hook_exec.event
                            if hook_exec.break_execution:
                                stopped_by_hook = True
                        if stopped_by_hook:
                            await self.conversation_manager.empty_agents_stack(
                                state.application_name, state.user_id, state.conversation_id
                            )
                            break
                        if should_continue is False:
                            yield event
                            await self.conversation_manager.empty_agents_stack(
                                state.application_name, state.user_id, state.conversation_id
                            )
                            break

                        # Yield AGENT_CALL first so consumers see the calling agent's action
                        yield event

                        (
                            new_agent,
                            new_context,
                            agent_start_event,
                        ) = await self._handle_agent_transition(state, event, context)

                        if new_agent is None:
                            break

                        state.current_agent = new_agent
                        context = new_context

                        # Then yield the runner-created AGENT_START (called agent receiving its task).
                        # Apply the called agent's on_event hooks first so they can modify the event.
                        if agent_start_event is not None:
                            if called_agent_on_event_hooks := new_agent.on_event_hooks:
                                async for on_exec in self.run_on_event_hooks(
                                    called_agent_on_event_hooks,
                                    agent_start_event,
                                    context,
                                    agent_name=new_agent.name,
                                    state=state,
                                ):
                                    yield on_exec.hook_event
                                    agent_start_event = on_exec.current_event
                                    should_continue = not on_exec.break_execution and (
                                        on_exec.current_event.event_type
                                        != ExecutionEventType.RESPONSE
                                        or on_exec.current_event.is_partial
                                    )
                                    if not should_continue:
                                        break
                                if not should_continue:
                                    if not on_exec.break_execution:
                                        await self._add_event(
                                            state, agent_start_event, context=context
                                        )
                                        yield agent_start_event
                                    await self.conversation_manager.empty_agents_stack(
                                        state.application_name, state.user_id, state.conversation_id
                                    )
                                    break
                            await self._add_event(state, agent_start_event, context=context)
                            yield agent_start_event

                        # Sync context.agent_input so the transport (and any downstream code)
                        # sees the hook-processed AGENT_START, not the original pre-hook version.
                        context.agent_input = agent_start_event

                        async for hook_exec in self._apply_agent_hooks(
                            hooks=state.current_agent.pre_hooks,
                            state=state,
                            context=context,
                            agent=state.current_agent,
                        ):
                            context = hook_exec.context
                            if should_continue:
                                should_continue = not hook_exec.break_execution
                            yield hook_exec.event
                        agent_iterator = new_agent.execute(
                            context, **self._execute_kwargs(state, new_agent)
                        )
                        continue  # skip bottom yield — already yielded AGENT_CALL above

                    # AGENT_START events proxied from the remote are forwarded as-is.
                    # The runner must NOT re-enter the transition path for these events.
                    case ExecutionEventType.AGENT_START:
                        if isinstance(state.current_agent, DelegateAgent):
                            yield event
                            continue

                    # Handle agent end (sub-agent returning to caller)
                    case ExecutionEventType.AGENT_END:
                        if isinstance(state.current_agent, DelegateAgent) and (
                            event.author != state.current_agent.name
                        ):
                            # AGENT_END from a remote delegate: the transport normalises the
                            # root session-end author to the DelegateAgent name; sub-agent
                            # AGENT_END events keep their original remote author.
                            # If the author is not the DelegateAgent itself, treat it as a
                            # remote sub-agent end — yield for observability and continue
                            # (mirrors the AGENT_CALL / TOOL_START unknown-agent pattern).
                            event = self._ensure_event_identifiers(
                                state=state, event=event, context=context
                            )
                            yield event
                            continue
                        # Author matches the DelegateAgent: this is the root session end.
                        # Fall through to normal AGENT_END handling.

                        transition = _ParentTransition(context=context)
                        async for ev in self._return_to_parent(
                            state, event, context, agent_iterator, should_continue, transition
                        ):
                            yield ev
                        context = transition.context
                        should_continue = transition.should_continue
                        if transition.iterator is None:
                            # The root has nowhere to return to, so this AGENT_END is
                            # the end of the turn — the same window as the final
                            # RESPONSE above, for a `use_stop` planner that carries
                            # its answer in an AGENT_END instead. One asymmetry,
                            # accepted: `_return_to_parent` runs the post_agent_call
                            # hooks and yields this AGENT_END before it can be known
                            # that the root was reached, so unlike the RESPONSE arm
                            # both have already happened. Unifying them means
                            # restructuring a function the sub-agent path shares.
                            if should_continue:
                                steered = await self._absorb_for_steering(state, context)
                                if steered:
                                    for steered_event in steered:
                                        yield steered_event
                                    agent_iterator = await self._reenter_after_steering(
                                        state, context, agent_iterator, len(steered)
                                    )
                                    continue
                            # As in the RESPONSE arm: say "too late" from here on
                            # rather than accept content the teardown will discard.
                            # The window is narrower — only the loop's own closing
                            # `aclose()` sits between this and the wrapper's finally —
                            # but the promise should not depend on how narrow it is.
                            self.injection_queue.pop(
                                (state.application_name, state.conversation_id), None
                            )
                            break
                        agent_iterator = transition.iterator

                    case ExecutionEventType.LLM_START:
                        # Handle delegate LLM execution when the agent uses DelegateLLMProvider.
                        # The planner yielded LLM_START with an LLMCallAction payload instead of
                        # calling the provider directly. Execute here and set pending_llm_response
                        # so the next asend() feeds the result back to the planner.
                        if (
                            not isinstance(state.current_agent, DelegateAgent)
                            and isinstance(event.action, LLMCallAction)
                            and isinstance(
                                getattr(state.current_agent, "llm_provider", None),
                                DelegateLLMProvider,
                            )
                        ):
                            provider: DelegateLLMProvider = state.current_agent.llm_provider
                            if provider.wrapped is not None:
                                if event.action.stream:
                                    chunks = []
                                    async for chunk in provider.wrapped.generate_streaming(
                                        messages=event.action.messages,
                                        tools=event.action.tools,
                                    ):
                                        chunks.append(chunk)
                                        if chunk.content and not context.is_subagent:
                                            yield self._ensure_event_identifiers(
                                                state=state,
                                                context=context,
                                                event=BaseEvent(
                                                    event_type=ExecutionEventType.RESPONSE_TOKEN,
                                                    content=[Part(text=chunk.content)],
                                                    author=state.current_agent.name,
                                                    interaction_id=context.interaction_id,
                                                    turn_id=state.turn_id,
                                                    is_partial=True,
                                                ),
                                            )
                                    accumulated = "".join(c.content or "" for c in chunks)
                                    last = chunks[-1] if chunks else None
                                    pending_llm_response = (
                                        last.model_copy(update={"content": accumulated})
                                        if last
                                        else LLMResponse(
                                            content=accumulated,
                                            generated_tokens=0,
                                            execution_time=0.0,
                                        )
                                    )
                                else:
                                    pending_llm_response = await provider.wrapped.generate(
                                        messages=event.action.messages,
                                        tools=event.action.tools,
                                    )
                            # Yield LLM_START for observability (without large payload)
                            observability_text = (
                                "Delegated LLM call — executing via wrapped provider"
                                if provider.wrapped is not None
                                else "Delegated LLM call — no wrapped provider, runner-to-runner transport not yet implemented"
                            )
                            yield self._ensure_event_identifiers(
                                state=state,
                                context=context,
                                event=BaseEvent(
                                    event_type=ExecutionEventType.LLM_START,
                                    content=[Part(text=observability_text)],
                                    author=event.author,
                                    interaction_id=context.interaction_id,
                                    turn_id=state.turn_id,
                                ),
                            )
                            continue

                    case ExecutionEventType.TOOL_NEEDS_FEEDBACK:
                        if isinstance(state.current_agent, DelegateAgent):
                            yield event
                            await agent_iterator.aclose()
                            break

                    case ExecutionEventType.TOOL_START:
                        if isinstance(state.current_agent, DelegateAgent):
                            event = self._ensure_event_identifiers(
                                state=state, event=event, context=context
                            )
                            known_tool = (
                                state.current_agent.get_tool_by_name(event.action.name)
                                if isinstance(event.action, ToolAction)
                                else None
                            )
                            if (
                                known_tool is None
                                or event.metadata.get(METADATA_VISIBILITY_ONLY)
                                # Compatibility: older delegated runtimes mark
                                # visibility-only calls with a `visibility-` call_id
                                # prefix instead of the metadata flag above.
                                or event.action.call_id.startswith("visibility")
                            ):
                                # Unknown tool or non-ToolAction: yield for observability,
                                # external agent handles it.
                                # A delegated agent may report a tool it ran itself that shares
                                # a name with a local one; it already ran remotely, so not here.
                                yield event
                                continue

                            # Approval gate: HITL is handled entirely by the local runner;
                            # the delegate agent never sees TOOL_NEEDS_FEEDBACK.
                            # TOOL_START is forwarded below for observability; it is also
                            # already saved to history by the main loop so the TOOL_FEEDBACK
                            # resume can find it independently.
                            yield event  # forward TOOL_START (with its thought) before any feedback gate

                            # Same gate as the local path. Leaving it out here would
                            # make approvals bypassable by delegating to an external
                            # agent, which is a hole rather than an omission.
                            if known_tool.approval_type:
                                verdict, decision_events = await self._resolve_approval(
                                    known_tool, event, state, context
                                )
                                for decision_event in decision_events:
                                    yield decision_event
                                if verdict.is_deny:
                                    refusal = self._build_policy_refusal(
                                        event, state, context, verdict
                                    )
                                    await self._add_event(state, refusal, context=context)
                                    # The external agent is waiting on a result for
                                    # this call; without one it hangs on a future
                                    # that is never resolved.
                                    if state.current_agent.transport is not None:
                                        await state.current_agent.transport.send_result(
                                            refusal, conversation_id=state.conversation_id
                                        )
                                    yield refusal
                                    continue
                                if not verdict.is_allow:
                                    async for paused in self._pause_for_approval(
                                        known_tool, event, state, context
                                    ):
                                        yield paused
                                    # Close the transport connection cleanly. On
                                    # TOOL_FEEDBACK a fresh iterator will be created;
                                    # send_result() pre-sets the TOOL_END on the new
                                    # transport so execute() skips AGENT_START.
                                    await agent_iterator.aclose()
                                    break  # pause loop; resume when TOOL_FEEDBACK arrives

                            if isinstance(known_tool, MCPTool) and known_tool.use_oauth:
                                auth_metadata = await known_tool.check_auth_credentials()
                                if not auth_metadata.credentials_valid:
                                    async for paused in self._pause_for_auth(
                                        event, state, context, auth_metadata
                                    ):
                                        yield paused

                                    await agent_iterator.aclose()
                                    break
                            # Known tool, no approval: execute locally and send the result
                            # back so the external agent can continue.
                            # Signal the transport that we're about to block executing a
                            # local tool and will stop reading it. Lets the transport
                            # release its receive connection so events the remote emits
                            # meanwhile (e.g. a background-delegation wrap-up) aren't lost
                            # on the connected-but-unread socket. No-op for transports that
                            # don't need it.
                            if state.current_agent.transport is not None:
                                await state.current_agent.transport.pause_receiving()
                            tool_events = self._execute_tool(known_tool, state, context, event)
                            async for tool_event in tool_events:
                                tool_event = self._ensure_event_identifiers(
                                    state=state,
                                    event=tool_event,
                                    context=context,
                                )
                                tool_event.turn_id = state.turn_id
                                tool_event.interaction_id = context.interaction_id
                                if agent_on_event_hooks := state.current_agent.on_event_hooks:
                                    async for on_exec in self.run_on_event_hooks(
                                        agent_on_event_hooks,
                                        tool_event,
                                        context,
                                        agent_name=state.current_agent.name,
                                        state=state,
                                    ):
                                        yield on_exec.hook_event
                                        tool_event = on_exec.current_event
                                        if on_exec.break_execution:
                                            should_continue = False
                                if tool_event.event_type not in [
                                    ExecutionEventType.TOOL_END,
                                    ExecutionEventType.TOOL_ERROR,
                                ]:
                                    yield tool_event
                            event = tool_event
                            await self._add_event(state, event, context=context)
                            if state.current_agent.transport is not None:
                                # Resolve the delegate's pending Future unconditionally,
                                # even if a hook set break_execution: the external agent
                                # is blocked waiting for this result, and withholding it
                                # would hang the remote side instead of stopping the run.
                                await state.current_agent.transport.send_result(
                                    event, conversation_id=state.conversation_id
                                )
                            yield event  # TOOL_END / TOOL_ERROR for observability
                            if not should_continue:
                                break
                            continue

                        # skip if tool start action not provided
                        # A TOOL_START without a ToolAction (e.g. from a custom agent) has nothing
                        # to run.
                        if not isinstance(event.action, ToolAction):
                            continue

                        # A call from a batch (several tools asked for in one
                        # response) is held until the last of it arrives, and then
                        # the batch runs: see `_run_tool_calls`. It is stored
                        # already, at the top of this iteration, and drawn when the
                        # batch runs.
                        batch = (event.metadata or {}).get(METADATA_TOOL_BATCH) or {}
                        if batch.get("index", 0) < batch.get("size", 1) - 1:
                            held_tool_calls.append(event)
                            continue
                        calls, held_tool_calls = [*held_tool_calls, event], []
                        outcome = _ToolCallsOutcome()
                        async for out in self._run_tool_calls(calls, state, context, outcome):
                            yield out
                        if outcome.paused:
                            if outcome.close_iterator:
                                await agent_iterator.aclose()
                            break  # wait for the answer before resuming
                        if not outcome.should_continue:
                            should_continue = False
                            break
                        if outcome.last is None:
                            continue
                        # The last result, stored already: yielded below like every
                        # other event, and steering is taken in after it.
                        event = outcome.last
                # normal event yield
                # RESPONSE was yielded (or suppressed when streamed) in its arm above.

                # Last point before asend() resumes the planner into its next
                # get_context(), i.e. the last chance to change what the next LLM
                # call sees. On AGENT_END `context` has already been swapped back to
                # the parent's by _return_to_parent, so a message taken in here
                # lands in the history of the agent that is about to run rather than
                # the one that just finished.
                #
                # The guard is about the conversation *state* dict, not the event:
                # the event was stored at the top of this iteration along with the
                # first write of state, and this second write exists because the arm
                # in between can change it — compaction records, summaries, context
                # usage. An answer is excluded because its arms have already ended
                # the turn, continued it, or written state themselves; a streamed
                # token because this loop turns once per token, and neither a store
                # call nor the coroutine needed to make one is a price an answer
                # should pay a thousand times over.
                #
                # Whether to also *absorb* is `steering_triggers`, minus partials: a
                # partial is by definition followed by more, so the boundary after it
                # feeds the same model call and catches a superset of what this one
                # could. Absorbing at both cannot change an outcome. The case in
                # practice is a non-streamed preamble — content alongside tool calls,
                # emitted as a partial RESPONSE before the tools run, with the
                # TOOL_END moments behind it.
                steered = []
                if event.event_type not in (
                    ExecutionEventType.RESPONSE,
                    ExecutionEventType.RESPONSE_TOKEN,
                ):
                    await self._write_conversation_state(state, context)
                    if event.event_type in self.steering_triggers and not event.is_partial:
                        steered = await self._absorb_pending(state, context)

                if event.event_type != ExecutionEventType.RESPONSE:
                    if event.event_type != ExecutionEventType.RESPONSE_TOKEN:
                        yield event
                    elif not context.is_subagent:
                        # Subagent streaming tokens are internal: the content reaches
                        # the caller grouped in the converted AGENT_END instead.
                        yield event

                for steered_event in steered:
                    yield steered_event
        except asyncio.CancelledError:
            # Outside asend() the iterator sits at its own yield, so a cancellation
            # never reaches it; aclose() would look like a deliberate close instead.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await agent_iterator.athrow(asyncio.CancelledError())
            raise
        finally:
            # The break paths that don't close the iterator themselves would
            # otherwise leave it un-finalized until garbage collection.
            with contextlib.suppress(Exception):
                await agent_iterator.aclose()

        # Safety: if a pre-hook set should_continue=False and exited via `continue`
        # (which exits the while condition rather than an explicit break), the stack
        # was never cleared by the explicit break paths above. Clear it now.
        # All other hook-break paths already called empty_agents_stack before their
        # break, so this is a no-op for those cases.
        if not should_continue:
            await self.conversation_manager.empty_agents_stack(
                state.application_name, state.user_id, state.conversation_id
            )

    async def _run_tool_calls(
        self,
        calls: list[BaseEvent],
        state: RunnerState,
        context: InteractionContext,
        outcome: _ToolCallsOutcome,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Run the calls of one model response, or the one call it made.

        In order, except that a stretch of consecutive calls needing no approval runs
        at the same time: reading three files, querying two
        MCP servers. A call with ``approval_type`` (or an MCP tool behind OAuth) runs
        alone, after everything before it and before everything after it: what needs
        a human's say is the kind of call another one may depend on (write, then run
        the tests), and a question must be asked about a call that has not started.

        When a call waits for an answer, the run ends there and the calls after it
        are held under ``HELD_TOOL_CALLS_KEY``: the resume runs them before the model
        is asked again, instead of leaving it with calls it made that never ran.

        Every event is yielded except the last result, which goes to
        ``outcome.last``: the agent loop yields it like any event and takes steering
        in after it, as it did for a single call.
        """
        held: BaseEvent | None = None
        async for kind, item in self._tool_call_items(calls, state, context, outcome):
            if held is not None:
                yield held
                held = None
            if kind == "result":
                held = item
            else:
                yield item
        if held is not None and not outcome.should_continue:
            # The call that stopped the turn yields nothing of its own: the result
            # held here is an earlier call's, and was due.
            yield held
            held = None
        outcome.last = held

    async def _tool_call_items(
        self,
        calls: list[BaseEvent],
        state: RunnerState,
        context: InteractionContext,
        outcome: _ToolCallsOutcome,
    ) -> AsyncGenerator[tuple[str, BaseEvent], None]:
        """``_run_tool_calls`` as ``(kind, event)``: ``"result"`` for a call's final
        TOOL_END/TOOL_ERROR, stored already; ``"event"`` for everything else."""
        group: list[tuple[BaseEvent, FunctionTool | DelegateTool]] = []
        for position, call in enumerate(calls):
            if not isinstance(call.action, ToolAction):
                continue
            tool: FunctionTool | DelegateTool | None = state.current_agent.get_tool_by_name(
                call.action.name
            )
            gated = tool is not None and (
                bool(tool.approval_type) or (isinstance(tool, MCPTool) and tool.use_oauth)
            )
            if tool is not None and not gated:
                group.append((call, tool))
                continue

            async for item in self._run_tool_group(group, state, context, outcome):
                yield item
            group = []
            if not outcome.should_continue:
                await self._close_tool_calls(calls[position:], state, context, _STOPPED)
                return

            if tool is None:
                action_copy = call.action.model_copy()
                action_copy.observation = (
                    f"Tool '{call.action.name}' not found on agent '{state.current_agent.name}'"
                )
                yield (
                    "event",
                    self._ensure_event_identifiers(
                        state=state,
                        context=context,
                        event=BaseEvent(
                            event_type=ExecutionEventType.TOOL_ERROR,
                            author=state.current_agent.name,
                            action=action_copy,
                            error_code="ToolNotFound",
                            error_message=action_copy.observation,
                        ),
                    ),
                )
                continue

            yield "event", call  # forward TOOL_START (with its thought) before any feedback gate

            if tool.approval_type:
                verdict, decision_events = await self._resolve_approval(tool, call, state, context)
                for decision_event in decision_events:
                    yield "event", decision_event
                if verdict.is_deny:
                    # No human to resume this turn, so suspending would deadlock.
                    # Produce the refusal the model would have seen and carry on.
                    refusal = self._build_policy_refusal(call, state, context, verdict)
                    await self._add_event(state, refusal, context=context)
                    yield "event", refusal
                    continue
                if not verdict.is_allow:
                    await self._hold_tool_calls(call, calls[position + 1 :], state, context)
                    async for paused in self._pause_for_approval(tool, call, state, context):
                        yield "event", paused
                    outcome.paused = True
                    return

            if isinstance(tool, MCPTool) and tool.use_oauth:
                auth_metadata = await tool.check_auth_credentials()
                if not auth_metadata.credentials_valid:
                    await self._hold_tool_calls(call, calls[position + 1 :], state, context)
                    async for paused in self._pause_for_auth(call, state, context, auth_metadata):
                        yield "event", paused
                    outcome.paused = True
                    outcome.close_iterator = True
                    return

            async for item in self._run_one_tool(tool, call, state, context, outcome):
                yield item
            if not outcome.should_continue:
                await self._close_tool_calls(calls[position + 1 :], state, context, _STOPPED)
                return

        async for item in self._run_tool_group(group, state, context, outcome):
            yield item

    async def _run_tool_group(
        self,
        group: list[tuple[BaseEvent, FunctionTool | DelegateTool]],
        state: RunnerState,
        context: InteractionContext,
        outcome: _ToolCallsOutcome,
    ) -> AsyncGenerator[tuple[str, BaseEvent], None]:
        """Consecutive calls that need no approval: one runs as it always has, two
        or more at the same time, at most ``max_parallel_tool_calls`` at once.

        Their TOOL_STARTs go out first, then every event as it comes, so results
        arrive in the order the calls finish; each call's result is stored when it
        finishes. A hook stopping the turn on one call lets the others finish: they
        are already running, and stopping them halfway leaves work half done.
        """
        if not group:
            return
        if len(group) == 1:
            call, tool = group[0]
            yield "event", call
            async for item in self._run_one_tool(tool, call, state, context, outcome):
                yield item
            return

        for call, _ in group:
            yield "event", call
        queue: asyncio.Queue[tuple[str, BaseEvent] | None] = asyncio.Queue()
        limit = asyncio.Semaphore(self.max_parallel_tool_calls)
        outcomes = [_ToolCallsOutcome() for _ in group]

        async def pump(
            tool: FunctionTool | DelegateTool, call: BaseEvent, own: _ToolCallsOutcome
        ) -> None:
            try:
                async with limit:
                    async for item in self._run_one_tool(tool, call, state, context, own):
                        await queue.put(item)
            finally:
                await queue.put(None)

        tasks = [
            asyncio.ensure_future(pump(tool, call, own))
            for (call, tool), own in zip(group, outcomes, strict=True)
        ]
        running = len(tasks)
        try:
            while running:
                arrived = await queue.get()
                if arrived is None:
                    running -= 1
                    continue
                yield arrived
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        for task in tasks:
            if not task.cancelled() and task.exception() is not None:
                raise task.exception()  # type: ignore[misc]
        if not all(own.should_continue for own in outcomes):
            outcome.should_continue = False

    async def _run_one_tool(
        self,
        tool: FunctionTool | DelegateTool,
        call: BaseEvent,
        state: RunnerState,
        context: InteractionContext,
        outcome: _ToolCallsOutcome,
    ) -> AsyncGenerator[tuple[str, BaseEvent], None]:
        """Execute one call that is cleared to run, with the agent's ON_AGENT_EVENT
        hooks on each of its events, and store its result."""
        should_continue = True
        on_exec: Any = None
        tool_event: BaseEvent = call
        async for tool_event in self._execute_tool(tool, state, context, call):
            tool_event = self._ensure_event_identifiers(
                state=state,
                event=tool_event,
                context=context,
            )
            tool_event.turn_id = state.turn_id
            tool_event.interaction_id = context.interaction_id
            if agent_on_event_hooks := state.current_agent.on_event_hooks:
                async for on_exec in self.run_on_event_hooks(
                    agent_on_event_hooks,
                    tool_event,
                    context,
                    agent_name=state.current_agent.name,
                    state=state,
                ):
                    yield "event", on_exec.hook_event
                    tool_event = on_exec.current_event
                    should_continue = not on_exec.break_execution and (
                        on_exec.current_event.event_type != ExecutionEventType.RESPONSE
                        or on_exec.current_event.is_partial
                    )
                    if not should_continue:
                        break
                if not should_continue:
                    if not on_exec.break_execution:
                        yield "event", tool_event
                    break
            if tool_event.event_type not in [
                ExecutionEventType.TOOL_END,
                ExecutionEventType.TOOL_ERROR,
            ]:
                yield "event", tool_event
        # The tool's last event is the one returned to the user, and saved
        await self._add_event(state, tool_event, context=context)
        if should_continue:
            yield "result", tool_event
        else:
            outcome.should_continue = False

    async def _hold_tool_calls(
        self,
        waiting: BaseEvent,
        rest: list[BaseEvent],
        state: RunnerState,
        context: InteractionContext,
    ) -> None:
        """Remember, before the run ends at a pause, which call is waiting and which
        come after it in the same response."""
        context.state[HELD_TOOL_CALLS_KEY] = {
            "waiting": getattr(waiting.action, "call_id", None),
            "calls": [
                call.model_dump(mode="json") for call in rest if isinstance(call.action, ToolAction)
            ],
        }
        await self._write_conversation_state(state, context)

    async def _drop_held_tool_calls(self, state: RunnerState, context: InteractionContext) -> None:
        """Close the calls of a pause the user did not answer: see ``run()``."""
        if HELD_TOOL_CALLS_KEY not in context.state:
            return
        waiting_id, held = self._held_tool_calls(context)
        context.state.pop(HELD_TOOL_CALLS_KEY, None)
        answered = {
            getattr(x.action, "call_id", None)
            for x in (*context.conversation_history, *context.interaction_history)
            if x.event_type in (ExecutionEventType.TOOL_END, ExecutionEventType.TOOL_ERROR)
        }
        waiting = next(
            (
                x
                for x in reversed([*context.conversation_history, *context.interaction_history])
                if x.event_type == ExecutionEventType.TOOL_START
                and waiting_id
                and getattr(x.action, "call_id", None) == waiting_id
            ),
            None,
        )
        unanswered = [
            call
            for call in ([waiting] if waiting is not None else []) + held
            if getattr(call.action, "call_id", None) not in answered
        ]
        await self._close_tool_calls(unanswered, state, context, _SUPERSEDED)
        await self._write_conversation_state(state, context)

    @staticmethod
    def _held_tool_calls(context: InteractionContext) -> tuple[str | None, list[BaseEvent]]:
        """The call waiting at the last pause and the calls held after it."""
        held = context.state.get(HELD_TOOL_CALLS_KEY) or {}
        calls = []
        for raw in held.get("calls") or []:
            with contextlib.suppress(Exception):
                calls.append(BaseEvent.model_validate(raw))
        return held.get("waiting"), calls

    async def _close_tool_calls(
        self,
        calls: list[BaseEvent],
        state: RunnerState,
        context: InteractionContext,
        reason: str,
    ) -> None:
        """Give each call that will never run a result saying so.

        Every call a model made needs a result in the next request, or the provider
        refuses it; and a model that is not told a call did not run assumes it did.
        Stored, not yielded: nothing was shown for these calls, and there is nothing
        to show now.
        """
        for call in calls:
            # Any: mypy reads `BaseEvent.action`, an unresolved union, as None.
            call_action: Any = call.action
            if not isinstance(call_action, ToolAction):
                continue
            action = call_action.model_copy(deep=True)
            action.observation = f"The tool {action.name} was not executed: {reason}"
            await self._add_event(
                state,
                self._ensure_event_identifiers(
                    state=state,
                    context=context,
                    event=BaseEvent(
                        event_type=ExecutionEventType.TOOL_END,
                        author=call.author or state.current_agent.name,
                        conversation_id=state.conversation_id,
                        interaction_id=context.interaction_id,
                        turn_id=state.turn_id,
                        action=action,
                        metadata={"not_executed": True},
                    ),
                ),
                context=context,
            )

    async def _resolve_approval(
        self,
        tool: FunctionTool | DelegateTool,
        event: BaseEvent,
        state: RunnerState,
        context: InteractionContext,
    ) -> tuple[ApprovalVerdict, list[BaseEvent]]:
        """Decide whether this call still needs a human.

        Only ever called for tools that declared ``approval_type``: a tool that
        never wanted a gate does not get one, and cannot acquire one through a
        policy. Widening what needs approval is the tool author's call.

        Returns the verdict and the events the resolution produced — today only
        ``DECISION_MADE`` — for the two call sites that are generators and can
        yield them. The third (``_refuse_edited_call``) is not, and discards
        them at the call rather than being handed a list to leave empty: the
        decision is in the log either way.
        """
        parameters = dict(getattr(event.action, "parameters", None) or {})
        request = build_request(
            tool,
            parameters,
            agent_name=state.current_agent.name,
            user_id=state.user_id,
            conversation_id=state.conversation_id,
        )

        hook_verdicts: list[ApprovalVerdict] = []
        for hook in getattr(tool, "approval_hooks", []):
            async for result in self._collect_hook_results(
                hook,
                f"Approval hook {hook.name} did not return a result.",
                tool=tool,
                tool_args=parameters,
            ):
                if result.success and result.approval is not None:
                    hook_verdicts.append(result.approval)
                elif not result.success:
                    # A hook that failed has no opinion. Treating an error as
                    # permission is the one mistake this whole subsystem exists
                    # to prevent.
                    get_logger().warning(
                        f"approval hook '{hook.name}' failed; falling back to asking"
                    )

        # The agent's own decision model, if it has one and it is switched on.
        # Its verdict joins the hook verdicts rather than bypassing them, so it
        # lands in the one place the ordering is defined: after hardline rules,
        # after deny rules and after a stored grant, never before.
        consultation = await consult_decision_for_approval(runtime_of(state.current_agent), request)
        if consultation.verdict is not None:
            hook_verdicts.append(consultation.verdict)
        decision_events: list[BaseEvent] = []
        if consultation.mode is not DecisionMode.OFF:
            decision_events.append(
                BaseEvent(
                    event_type=ExecutionEventType.DECISION_MADE,
                    author=state.current_agent.name,
                    metadata={
                        **decision_event_metadata("approvals", consultation),
                        "tool": request.tool_name,
                    },
                )
            )

        verdict = await self.approval_policy.resolve(
            request, context=context, hook_verdicts=hook_verdicts
        )
        if verdict.decision is not ApprovalDecision.ASK:
            get_logger().info(
                f"approval: {verdict.decision} for '{request.tool_name}' "
                f"via {verdict.rule or 'policy'}"
            )
        return verdict, decision_events

    @staticmethod
    def _answers_another_call(
        reply: ToolAction, waiting: ToolAction, context: InteractionContext
    ) -> bool:
        """Whether a reply is for a call other than the one waiting.

        The tool must match, and the call id must not be an earlier call's. A fresh
        id identifies nothing: some clients send one.
        """
        if reply.name != waiting.name:
            return True
        if not reply.call_id or reply.call_id == waiting.call_id:
            return False
        return any(
            event.event_type == ExecutionEventType.TOOL_START
            and getattr(event.action, "call_id", None) == reply.call_id
            for event in (*context.conversation_history, *context.interaction_history)
        )

    async def _refuse_edited_call(
        self,
        tool: FunctionTool | DelegateTool | None,
        waiting: BaseEvent,
        reply: BaseEvent,
        state: RunnerState,
        context: InteractionContext,
    ) -> BaseEvent | None:
        """The TOOL_END refusing an edited call, or None to run it.

        An edit is a new call, so hardline, deny rules and hooks judge it again.
        ALLOW or ASK change nothing: a human has just approved it.
        """
        # Any: see `_handle_tool_feedback`.
        action: Any = reply.action
        waiting_action: Any = waiting.action
        if (
            tool is None
            or tool.approval_type != ApprovalType.PARAMS_UPDATE
            or not isinstance(action, FeedbackAction)
            or not action.approve
        ):
            return None
        edited = action.to_tool_action()
        if not self.parameters_changed(edited.parameters, waiting_action.parameters):
            return None
        edited.call_id = waiting_action.call_id
        call = waiting.model_copy(update={"action": edited})
        # Not a generator, so the decision stays in the log rather than the stream.
        verdict, _ = await self._resolve_approval(tool, call, state, context)
        return (
            self._build_policy_refusal(call, state, context, verdict) if verdict.is_deny else None
        )

    def _build_policy_refusal(
        self,
        event: BaseEvent,
        state: RunnerState,
        context: InteractionContext,
        verdict: ApprovalVerdict,
    ) -> BaseEvent:
        """The TOOL_END a denied call produces when no human was involved.

        Same shape as a human refusal, so the model's experience of "that was
        not allowed" is identical either way — but worded so it does not claim
        the user refused. The model reads this text and repeats it back; telling
        it a person said no, when a rule did, puts a lie in front of the user.
        """
        action = event.action.model_copy(deep=True)
        detail = verdict.reason or "It was blocked by policy."
        action.observation = (
            f"The tool {action.name} was not executed: {detail} "
            "This was an automatic policy decision, not a choice by the user. "
            "Do not retry it; find another way or explain the limitation."
        )
        return self._ensure_event_identifiers(
            state=state,
            context=context,
            event=BaseEvent(
                event_type=ExecutionEventType.TOOL_END,
                author=state.current_agent.name,
                conversation_id=state.conversation_id,
                interaction_id=context.interaction_id,
                turn_id=state.turn_id,
                action=action,
                metadata={"denied_by": verdict.rule or "policy"},
            ),
        )

    def _build_tool_needs_feedback(
        self,
        tool: FunctionTool | DelegateTool,
        event: BaseEvent,
        state: RunnerState,
        context: InteractionContext,
    ) -> BaseEvent:

        request = build_request(
            tool,
            dict(getattr(event.action, "parameters", None) or {}),
            agent_name=state.current_agent.name,
            user_id=state.user_id,
            conversation_id=state.conversation_id,
        )
        return self._ensure_event_identifiers(
            state=state,
            context=context,
            event=BaseEvent(
                # The subject travels with the request so the front-end shows the
                # same string the policy will match a grant against. Two
                # descriptions of one call is how you approve one thing and run
                # another.
                metadata={
                    "approval_subject": request.subject,
                    # Lets a client check an edited pattern the way the policy will.
                    "approval_parts": request.parts,
                    "approval_scopes": [str(s) for s in self.approval_policy.available_scopes()],
                    "approval_pattern": _proposed_pattern(tool, request),
                },
                event_type=ExecutionEventType.TOOL_NEEDS_FEEDBACK,
                author=state.current_agent.name,
                conversation_id=state.conversation_id,
                interaction_id=context.interaction_id,
                turn_id=state.turn_id,
                action=FeedbackAction(
                    call_id=event.action.call_id,
                    approval_type=tool.approval_type,
                    name=event.action.name,
                    # Filtered through the same helper the subject is built from,
                    # so what the user approves and what a grant matches on cannot
                    # drift apart. Hardcoding the exclusion here (as this once did)
                    # is what made a remembered approval unmatchable: the gate's
                    # subject carried `reasoning`, the approved one did not.
                    parameters=[
                        FeedbackParameters(
                            name=k,
                            value=v,
                            type=type(v).__name__,
                            is_required=k in tool.required_param_names(),
                        )
                        for k, v in visible_parameters(event.action.parameters).items()
                    ],
                ),
            ),
        )

    @staticmethod
    def _params_changed_message(tool_name: str, params: Any) -> str:
        return (
            f"IMPORTANT: The user updated the tool parameters for tool: {tool_name} with parameters: "
            f"{params}. Continue the execution with these new parameters. The user changed the "
            f"parameters because they changed their mind about the tool execution; from now on use "
            f"the updated parameters instead of the old ones, also for related tool calls."
        )

    def _build_tool_error_event(
        self,
        state: RunnerState,
        context: InteractionContext,
        action: BaseAction,
    ) -> BaseEvent:
        return BaseEvent(
            event_type=ExecutionEventType.TOOL_ERROR,
            author=state.current_agent.name,
            conversation_id=state.conversation_id,
            interaction_id=context.interaction_id,
            turn_id=state.turn_id,
            action=action,
        )

    async def _stream_tool_events(
        self,
        tool: FunctionTool | DelegateTool,
        state: RunnerState,
        context: InteractionContext,
        event: BaseEvent,
        captured: list[BaseEvent],
    ) -> AsyncGenerator[BaseEvent, None]:
        async for tool_event in self._execute_tool(tool, state, context, event):
            tool_event.turn_id = state.turn_id
            tool_event.interaction_id = context.interaction_id
            if tool_event.event_type not in [
                ExecutionEventType.TOOL_END,
                ExecutionEventType.TOOL_ERROR,
            ]:
                yield tool_event
            captured.clear()
            captured.append(tool_event)

    def _finalize_tool_event(
        self,
        last_tool_event: BaseEvent | None,
        fallback_action: BaseAction,
        error_suffix: str,
        state: RunnerState,
        context: InteractionContext,
        params_changed_msg: str | None,
    ) -> BaseEvent:
        if last_tool_event is None:
            error_action = fallback_action.model_copy(
                update={
                    "observation": f"Tool {fallback_action.name} execution did not yield any events after {error_suffix}."
                }
            )
            return self._build_tool_error_event(state, context, error_action)
        last_tool_event.interaction_id = context.interaction_id
        last_tool_event.turn_id = state.turn_id
        if (
            params_changed_msg is not None
            and last_tool_event.event_type == ExecutionEventType.TOOL_END
            and isinstance(state.current_agent, DelegateAgent)
        ):
            last_tool_event.action.observation = (
                f"{last_tool_event.action.observation}\n{params_changed_msg}"
            )
        return last_tool_event

    def _build_tool_needs_feedback_auth(
        self,
        event: BaseEvent,
        state: RunnerState,
        context: InteractionContext,
        auth_metadata: MCPCredentialsCheckResult,
    ) -> BaseEvent:

        return self._ensure_event_identifiers(
            state=state,
            context=context,
            event=BaseEvent(
                event_type=ExecutionEventType.TOOL_NEEDS_FEEDBACK,
                author=state.current_agent.name,
                conversation_id=state.conversation_id,
                interaction_id=context.interaction_id,
                turn_id=state.turn_id,
                action=AuthenticationAction(
                    call_id=event.action.call_id,
                    name=event.action.name,
                    thought=event.action.thought,
                    parameters=None,
                    authentication_metadata=auth_metadata,
                ),
            ),
        )

    def _ensure_event_identifiers(
        self,
        state: RunnerState,
        event: BaseEvent,
        context: InteractionContext | None = None,
    ) -> BaseEvent:
        """Fill event identifiers from runner/context when missing."""
        event.event_id = event.event_id or f"event_{uuid.uuid4()}"
        event.conversation_id = event.conversation_id or state.conversation_id
        event.interaction_id = (
            event.interaction_id or getattr(context, "interaction_id", None) or state.interaction_id
        )
        event.turn_id = event.turn_id or getattr(context, "turn_id", None) or state.turn_id
        return event

    def _tool_ceiling(self, tool: FunctionTool | DelegateTool) -> float:
        """The tool's own limit, else the runner's. 0 anywhere means none.

        A delegate is exempt unless configured: it is not a call but a whole
        nested turn, bounded by its transport, and a shared ceiling would cut
        off a sub-agent that is working exactly as intended.
        """
        own = getattr(tool, "timeout_s", None)
        if own is not None:
            return float(own)
        return 0.0 if isinstance(tool, DelegateTool) else self._tool_timeout_s

    async def _execute_tool(
        self,
        tool: FunctionTool | DelegateTool,
        state: RunnerState,
        context: InteractionContext,
        event: BaseEvent,
    ) -> AsyncGenerator[BaseEvent, None]:
        manager = getattr(getattr(state.current_agent, "planner", None), "context_manager", None)
        tool_args = {
            "run_config": context.run_config,
            "state": context.state,
            # Reserved, so it never reaches the model. A tool that needs to keep
            # something per conversation — a working directory, an environment —
            # has no other way to know which one it is in.
            "conversation_id": state.conversation_id,
            # Per-call metadata (e.g. an auth token minted by a PRE_TOOL_CALL hook),
            # always forwarded to REST headers / MCP meta — unlike run_config it is
            # never gated by VisibilityScope. Seeded empty; hooks populate it.
            "call_metadata": {},
            **event.action.parameters,
        }
        # What only the tool gets, merged in after its pre-hooks. Both are live
        # objects: a REST hook puts its `tool_args` in its request body, where they
        # cannot be serialized, so the request was never sent and the tool ran as if
        # the hook did not exist. No hook has a use for them. Merged last, so neither
        # the model's arguments nor what a hook returns can stand in for them.
        tool_only_args = {
            # The skills available to the current agent: `skill_view` reads them
            # instead of fetching them again.
            "loaded_skills": state.skills.get(state.current_agent.name),
            # For `read_tool_result`: the tool results this agent sees, as its
            # model is shown them, and how much of one a read may return.
            "tool_results": (
                manager.tool_results(state.current_agent, context)
                if isinstance(manager, ContextManager)
                else None
            ),
        }
        """Execute a streamable tool and yield events."""
        tool_result: ToolResult | None = None
        # An agent suppresses partial RESPONSE events (and surfaces the content via
        # the observation instead) when it is not allowed to display content
        # directly to the user: either it is a delegated, non-entrypoint agent, or
        # its planner runs with use_stop=True. In those cases we accumulate the
        # streamed chunks here (local to this single tool execution) and fold them
        # into the terminal TOOL_END below.
        suppress_partials = context.is_subagent or getattr(
            getattr(state.current_agent, "planner", None), "use_stop", False
        )
        streamed_parts: list[Part] = []
        try:
            # Execute pre-hooks before executing the tool
            async for hook_result, updated_tool_args in self.run_tool_pre_hooks(
                context=context, tool_args=tool_args, state=state, tool=tool
            ):
                tool_args = updated_tool_args
                yield hook_result
            tool_args = {**tool_args, **tool_only_args}
            # Execute the tool

            ceiling = self._tool_ceiling(tool)
            if not inspect.isasyncgenfunction(tool.execute):
                tool_result = await _bounded(tool.execute(**tool_args), ceiling, event.action.name)
            else:
                _s: ToolResult | StreamableToolResult | UpdateToolResult
                async for _s in _bounded_iter(
                    tool.execute(**tool_args), ceiling, event.action.name
                ):
                    if isinstance(_s, StreamableToolResult):
                        chunk_parts = _s.data if isinstance(_s.data, list) else [_s.data]
                        if suppress_partials:
                            # Suppress the partial RESPONSE event and buffer the
                            # content so it rides along with the TOOL_END instead.
                            streamed_parts.extend(chunk_parts)
                            continue
                        partial_event = BaseEvent(
                            event_type=ExecutionEventType.RESPONSE,
                            author=state.current_agent.name,
                            action=event.action,
                            is_partial=True,
                            content=chunk_parts,
                        )
                        await self._add_event(state, partial_event, context=context)
                        yield partial_event
                    elif isinstance(_s, UpdateToolResult):
                        partial_event = BaseEvent(
                            event_type=ExecutionEventType.TOOL_UPDATE,
                            author=state.current_agent.name,
                            action=event.action,
                            is_partial=True,
                            content=[Part(metadata=_s.data)],
                        )
                        await self._add_event(state, partial_event, context=context)
                        yield partial_event
                    else:
                        # Here we have a ToolResult as last event
                        tool_result = _s
            if tool_result is None:
                raise RuntimeError("Streamable tool did not yield a result.")
            if tool_result.state is not None:
                # Inplace update of context state
                context.state.clear()
                context.state.update(tool_result.state)
            # update tool_result.state with context.state for post hooks --> so the post hooks receive the updated state
            tool_result.state = context.state
            # Execute post hooks before yielding the tool end if the execution is successful
            async for hook_execution, tool_result in self.run_tool_post_hooks(  # noqa: B007, B020
                context=context, tool_result=tool_result, state=state, tool=tool
            ):
                yield hook_execution

            # When partials are suppressed, the tool's emitted_content would
            # otherwise surface to the user via the event content; fold it into
            # streamed_parts as well so everything the agent produced only informs
            # the parent (via the observation) and nothing leaks out through the
            # event content.
            if suppress_partials:
                streamed_parts.extend(tool_result.emitted_content or [])

            # only last event become observation
            action_copy = event.action.model_copy()
            # When partials are suppressed the streamed chunks were not yielded as
            # partial RESPONSEs; surface their text as the observation so the agent
            # reasons over what was streamed. streamed_parts is empty otherwise (and
            # for non-streaming tools), so those keep tool_result.data.
            streamed_text = (
                "NOTE: This tool produced display-ready content. The content HAS NOT BEEN SHOWN.\n"
                "In your response, describe what the tool produced and ALWAYS state "
                "that it HAS NOT BEEN presented, but that it is ready to if needed.\n"
                "Captured content (for your reference only):\n"
                + "\n".join(p.model_dump_json(exclude_none=True) for p in streamed_parts)
                if streamed_parts
                else ""
            )
            if tool_result.error:
                action_copy.observation = tool_result.error
            elif streamed_text:
                action_copy.observation = streamed_text
            else:
                action_copy.observation = tool_result.data
            auth_metadata = (
                tool.format_auth_credentials_result_failed_execution()
                if isinstance(tool, MCPTool) and tool.use_oauth and tool_result.error_code == "401"
                else None
            )
            if auth_metadata is not None and not auth_metadata.credentials_valid:
                tool_event = self._build_tool_needs_feedback_auth(
                    auth_metadata=auth_metadata,
                    event=event,
                    state=state,
                    context=context,
                )
            else:
                tool_event = BaseEvent(
                    event_type=ExecutionEventType.TOOL_END
                    if not tool_result.error
                    else ExecutionEventType.TOOL_ERROR,
                    author=state.current_agent.name,
                    action=action_copy,
                    # When partials are suppressed, everything (streamed chunks +
                    # emitted_content) was folded into streamed_parts / the observation,
                    # so the event content stays empty — nothing leaks to the user.
                    # Otherwise the agent keeps its emitted_content here.
                    content=[] if suppress_partials else [*(tool_result.emitted_content or [])],
                    metadata=tool_result.metadata if tool_result.metadata else {},
                    error_code=tool_result.error_code,
                    error_message=tool_result.error,
                )
            yield tool_event
        except Exception as e:
            action_copy = event.action.model_copy()
            action_copy.observation = f"Error executing tool {event.action.name}: {e}"
            error_code: str | None
            error_message: str | None
            if isinstance(e, ToolTimeoutError):
                action_copy.observation = str(e)
                error_code, error_message = "tool_timeout", str(e)
            elif tool_result is None:
                error_code = "Missing tool result due to exception"
                error_message = str(e)
            else:
                # Use error info from _s if available, otherwise use defaults
                error_code = tool_result.error_code
                error_message = tool_result.error
            tool_event = BaseEvent(
                event_type=ExecutionEventType.TOOL_ERROR,
                author=state.current_agent.name,
                action=action_copy,
                error_code=error_code,
                error_message=error_message,
            )
            yield tool_event

    def _build_subagent_end_event(self, event: BaseEvent, state: RunnerState) -> BaseEvent:
        """Convert a subagent's terminal RESPONSE into an AGENT_END.

        Subagents never speak to the user directly: the response content is
        carried back to the caller as the agent-end observation, mirroring the
        planner's use_stop stop event and DelegateAgent's remote conversion.
        A need_feedback response becomes a request for the caller to gather
        the missing information (from context or from the user).
        """
        text = " ".join(p.text for p in (event.content or []) if p.text)
        content = event.content
        if event.need_feedback:
            text = SUBAGENT_NEEDS_INFO_TEMPLATE.format(agent_name=event.author, question=text)
            content = [Part(text=text)]

        # Shape the action like the planner's use_stop stop event so the
        # prompting layer renders it identically to a real stop.
        planner = getattr(state.current_agent, "planner", None)
        prompting = getattr(getattr(planner, "context_manager", None), "prompting", None)
        stop_name = getattr(getattr(prompting, "stop_action", None), "name", None) or "stop"

        forwarded_metadata = {
            key: event.metadata[key]
            for key in (
                METADATA_FORWARD_VERBATIM,
                METADATA_FORWARD_VERBATIM_SOURCE_ID,
                METADATA_FORWARD_VERBATIM_SOURCE_KEY,
            )
            if key in event.metadata
        }
        return BaseEvent(
            event_type=ExecutionEventType.AGENT_END,
            author=event.author,
            action=ToolAction(
                name=stop_name,
                parameters={"info": text},
                thought=event.action.thought if event.action else None,
            ),
            content=content,
            interaction_id=event.interaction_id,
            turn_id=state.turn_id,
            metadata=forwarded_metadata,
        )

    async def _return_to_parent(
        self,
        state: RunnerState,
        event: BaseEvent,
        context: InteractionContext,
        agent_iterator: AsyncGenerator[Any, Any],
        should_continue: bool,
        transition: _ParentTransition,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Close the current agent and resume its caller (AGENT_END semantics).

        Shared by the AGENT_END arm and the subagent RESPONSE conversion in
        ``_execute_agent_loop``. Yields hook/observability events; the outcome
        is written into ``transition``: ``iterator`` is the resumed caller's
        iterator, or None when the loop must stop (root reached, hook break,
        or should_continue already False — the stop paths yield ``event``
        themselves where the original inline code did).
        """
        logger = get_logger()
        try:
            await agent_iterator.aclose()
        except Exception as exc:
            logger.warning(
                f"agent_iterator.aclose() failed during agent transition: {type(exc).__name__}"
            )
            logger.debug(f"agent_iterator.aclose() detail: {exc}", exc_info=True)

        stopped_by_hook = False
        continue_with: BaseEvent | None = None
        continue_requested_by: str | None = None
        async for hook_exec in self._apply_agent_hooks(
            hooks=state.current_agent.post_hooks,
            state=state,
            context=context,
            agent=state.current_agent,
        ):
            context = hook_exec.context
            yield hook_exec.event
            if hook_exec.break_execution:
                stopped_by_hook = True
            if hook_exec.continue_with is not None:
                continue_with = hook_exec.continue_with
                continue_requested_by = (
                    hook_exec.event.action.name if hook_exec.event.action else None
                )
        transition.context = context
        transition.should_continue = should_continue
        if stopped_by_hook:
            # break_execution wins over continue_with: when a hook asks for two
            # contradictory things, stopping is the safer reading.
            await self.conversation_manager.empty_agents_stack(
                state.application_name, state.user_id, state.conversation_id
            )
            return

        if continue_with is not None and should_continue is not False:
            async for ev in self._continue_turn(
                state, context, continue_with, continue_requested_by, transition
            ):
                yield ev
            if transition.iterator is not None:
                return
            # Cap reached: fall through and end the turn normally.

        if should_continue is False:
            yield event
            await self.conversation_manager.empty_agents_stack(
                state.application_name, state.user_id, state.conversation_id
            )
            return

        new_agent, new_context, _ = await self._handle_agent_transition(state, event, context)

        if new_agent is None:
            yield event
            return

        state.current_agent = new_agent
        context = new_context
        transition.context = context

        async for hook_exec in self._apply_agent_hooks(
            hooks=state.current_agent.pre_hooks,
            state=state,
            context=context,
            agent=state.current_agent,
        ):
            context = hook_exec.context
            if should_continue:
                should_continue = not hook_exec.break_execution
            yield hook_exec.event
        transition.context = context
        transition.should_continue = should_continue
        transition.iterator = new_agent.execute(context, **self._execute_kwargs(state, new_agent))

    async def _continue_turn(
        self,
        state: RunnerState,
        context: InteractionContext,
        continue_with: BaseEvent,
        requested_by: str | None,
        transition: _ParentTransition,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Re-enter the current agent because a post_agent_call hook asked to.

        Sets ``transition.iterator`` when the turn continues, and leaves it as
        ``None`` when the per-turn cap has been reached, so the caller ends the
        turn normally.

        The cap lives here and not in the hook on purpose: a hook cannot be its
        own limit without the limit being circular, and a buggy one would
        otherwise burn tokens until the context window ran out.
        """
        logger = get_logger()
        limit = self._max_turn_continuations

        if state.continue_count >= limit:
            logger.warning(
                f"turn continuation cap of {limit} reached for agent "
                f"'{state.current_agent.name}'; ending the turn"
            )
            # Made visible as an event, not just logged: a turn that quietly
            # stopped continuing is indistinguishable from one that finished.
            capped = BaseEvent(
                event_type=ExecutionEventType.ERROR_OCCURRED,
                author=state.current_agent.name,
                error_code="ContinuationCapReached",
                error_message=(
                    f"A hook asked to continue the turn more than {limit} times; stopping."
                ),
            )
            await self._add_event(state, capped, context=context)
            yield capped
            return

        state.continue_count += 1
        by = requested_by or "hook"
        logger.info(
            f"turn continued by {by} "
            f"({state.continue_count}/{limit}) for agent '{state.current_agent.name}'"
        )

        # Persisted so the model can read *why* it is continuing; without the
        # reason in history it just sees an unexplained extra turn.
        continue_with.metadata.setdefault("continuation", True)
        continue_with.metadata.setdefault("continuation_index", state.continue_count)
        continue_with.metadata.setdefault("requested_by", by)
        await self._add_event(state, continue_with, context=context)
        yield continue_with

        context.agent_input = continue_with
        transition.context = context
        transition.should_continue = True
        transition.iterator = self._fresh_agent_iterator(state, context)

    def _fresh_agent_iterator(
        self, state: RunnerState, context: InteractionContext
    ) -> AsyncGenerator[Any, Any]:
        """A *fresh* iterator for the agent that is already running.

        Fresh rather than an ``asend()`` into the existing one, because a planner is
        not reliably resumable once it has produced a final answer:
        ``ToolUsePlannerStrategy`` and ``ToolUseWithThoughtPlannerStrategy`` both
        ``return`` immediately after yielding it, so ``asend()`` raises
        ``StopAsyncIteration`` and the turn ends anyway. Only ``react`` loops.
        Calling ``execute()`` again is the one mechanism that works for all three,
        and it costs nothing: the planner rebuilds its messages from ``context``,
        which is why appending to a history is all it takes to be read.
        """
        return state.current_agent.execute(
            context, **self._execute_kwargs(state, state.current_agent)
        )

    async def _absorb_for_steering(
        self, state: RunnerState, context: InteractionContext
    ) -> list[BaseEvent]:
        """Take in messages that arrived during the turn's final model call.

        Returned rather than yielded, because the caller has to place them: they are
        taken in before the answer is handed to the consumer and echoed after it.
        Empty means the turn ends: either nothing was waiting, or ``RESPONSE`` has
        been taken out of ``steering_triggers``, which is how this is switched off.

        ``RESPONSE`` gates the ``AGENT_END`` end-of-turn case as well: what it names
        is the turn's final answer, and a ``use_stop`` planner carries that in an
        ``AGENT_END``. Gating that on ``AGENT_END`` instead would make it
        un-disableable without also losing the absorb when a sub-agent returns.
        """
        if ExecutionEventType.RESPONSE not in self.steering_triggers:
            return []
        return await self._absorb_pending(state, context)

    async def _write_conversation_state(
        self, state: RunnerState, context: InteractionContext
    ) -> None:
        """Push this turn's state to the store.

        The second of the two writes a loop iteration makes. The first went out with
        the event at the top of the iteration; this one exists because the arm in
        between can change ``context.state`` — compaction records, summaries, context
        usage — and none of that would otherwise survive the turn.

        A failure is swallowed deliberately: a turn that cannot reach its store still
        has an answer to finish.
        """
        try:
            await self.conversation_manager.update_conversation_state(
                state.application_name,
                state.user_id,
                state.conversation_id,
                context.state,
            )
        except Exception as e:
            get_logger().warning(f"could not write conversation state: {type(e).__name__}")
            get_logger().debug(f"could not write conversation state, detail: {e}")

    async def _absorb_pending(
        self, state: RunnerState, context: InteractionContext
    ) -> list[BaseEvent]:
        """Take what this agent may see out of the messages waiting for this turn.

        No store read: the message is handed to the turn, so a boundary costs one
        store *write* for what it takes and nothing at all when nothing is waiting.

        That write is this method's job. ``push_parts`` persists nothing, so the turn
        that takes a message in is what commits it — through ``_add_event``, which
        also appends it to ``interaction_history`` after everything already there,
        which is where a message that arrived after three tool calls has to render.

        **A sub-agent does not look at the queue at all**, unless
        ``steering_reaches_subagents`` says otherwise: it is midway through a task its
        parent specified, and a user message that silently redirects it hands the
        parent back a result that does not answer what it asked. Leaving the queue
        untouched is also what keeps the message: the parent takes it in on the
        ``AGENT_END`` that returns control, and nothing else holds a copy.

        Whether *this* is a point a message may be taken in has already been decided
        by the caller — ``event.event_type in steering_triggers`` — and narrowing that
        tuple has to keep working.
        """
        if context.is_subagent and not state.steering_reaches_subagents:
            return []
        injections = self.injection_queue.get((state.application_name, state.conversation_id))
        if injections is None:
            return []

        taken: list[BaseEvent] = []
        for event in await injections.take_scheduled():
            # Everything scheduled is persisted and rendered; only a user message is
            # something the turn *answers*, and only the answered ones are returned,
            # because the caller uses them to decide whether the turn continues.
            await self._add_event(state, event, context=context)
            if event.event_type is ExecutionEventType.USER_INPUT:
                taken.append(event)
        return taken

    async def _reenter_after_steering(
        self,
        state: RunnerState,
        context: InteractionContext,
        agent_iterator: AsyncGenerator[Any, Any],
        absorbed: int,
    ) -> AsyncGenerator[Any, Any]:
        """Keep the turn alive: close the finished planner and hand back a new one.

        Three things this deliberately does **not** do:

        - **Rewrite ``context.agent_input``.** ``_continue_turn`` does, because its
          continuation event *is* the new input. Here it would be destructive: for a
          root turn ``agent_input`` is the ``AGENT_START`` carrying the turn's task,
          persisted only on a new conversation, so on every later turn it exists
          nowhere else — and ``ToolUseContextManager`` renders it. Replacing it with
          an action-less ``USER_INPUT`` would drop the task from the next call's
          messages. The absorbed event is in ``interaction_history``, which is
          rendered straight after, so it is read either way.
        - **Stamp the absorbed event.** The writer persisted it and the store holds
          the un-stamped copy; ``continuation`` metadata added here would exist only
          in memory and disagree with the next read. The message is its own reason.
        - **Run the post_agent_call hooks, or emit the synthesized AGENT_END.** The
          agent has not ended. Those hooks firing once per answer would double-write
          any updater that summarises or remembers, and the AGENT_END is persisted
          and rendered as a finished stop-tool pair — a fake "the agent stopped here"
          in the middle of a turn. (``_return_to_parent`` runs the hooks before
          ``_continue_turn`` only because a post hook is what asked to continue.)

        There is no cap. Each continuation needs a *new* event from a writer, and
        absorption is idempotent, so the loop advances only on genuinely new input:
        the bound is whoever is typing. A client that writes ``USER_INPUT`` in a loop
        can therefore hold a turn open, which is why every continuation is logged.
        """
        get_logger().info(
            f"turn continued by steering ({absorbed} message(s)) "
            f"for agent '{state.current_agent.name}'"
        )
        # The planner is suspended at its final yield, so its own `return` has not
        # run and the loop's `finally` will close the replacement rather than this
        # one. Closing it here is what finalizes it.
        with contextlib.suppress(Exception):
            await agent_iterator.aclose()
        return self._fresh_agent_iterator(state, context)

    async def _handle_agent_transition(
        self,
        state: RunnerState,
        event: BaseEvent,
        current_context: InteractionContext,
    ) -> tuple[Agent | DelegateAgent | None, InteractionContext, BaseEvent | None]:
        """Handle agent call/end transitions.

        Returns (new_agent, new_context, agent_start_event) where agent_start_event is the
        runner-created AGENT_START event when an AGENT_CALL was handled, else None.
        """
        new_interaction_id = InteractionContext.generate_interaction_id()
        agent_start_event: BaseEvent | None = None

        if event.event_type == ExecutionEventType.AGENT_CALL:
            # Callee resolved against the WHOLE tree (``available_agents``), not
            # the caller's resolved handoff targets. This is safe because both
            # entry paths are already scope-gated upstream:
            #   - LLM path: a planner only emits an AGENT_CALL when the name is in
            #     ``handoff_targets_description`` (the offered set, driven by
            #     ``handoff_scope``); an off-scope name parses as an unknown tool
            #     (TOOL_ERROR) and never becomes an AGENT_CALL.
            #   - DelegateAgent relay: the AGENT_CALL case enforces the delegate's
            #     local ``handoff_scope`` (via ``_get_handoff_targets``) before
            #     reaching this transition, so a relayed call to an off-scope local
            #     agent is handled remotely and never arrives here.
            new_agent = next(
                (agent for agent in state.available_agents if agent.name == event.action.name), None
            )
            if new_agent is not None:
                # Create the AGENT_START that represents the called agent receiving its task.
                # author = called agent so context managers can identify the called agent's start.
                # Task is stored in content (mirroring USER_INPUT), not in action.parameters.
                task_text = (
                    event.action.parameters.get("task")
                    if event.action.parameters and "task" in event.action.parameters
                    else (event.action.message or "")
                    if event.action and event.action.message
                    else " ".join(p.text for p in event.content if p.text)
                    if event.content
                    else ""
                )
                # Fall back to the parent's current user input when the LLM omits the task.
                if not task_text and current_context.agent_input is not None:
                    inp = current_context.agent_input
                    task_text = " ".join(p.text for p in (inp.content or []) if p.text)
                    if not task_text:
                        params = getattr(getattr(inp, "action", None), "parameters", None) or {}
                        task_text = params.get("task", "") or params.get("user_request", "") or ""

                # Delegates with a custom invocation schema (pushed by the remote
                # runner) expect their own parameter shape — not the default
                # `{"task": ...}`. Forward the LLM-produced parameters verbatim
                # in that case.
                if (
                    isinstance(new_agent, DelegateAgent)
                    and new_agent.invocation_parameters_override is not None
                    and event.action.parameters
                ):
                    start_parameters = dict(event.action.parameters)
                else:
                    start_parameters = {"task": task_text} if task_text else {}

                agent_start_event = BaseEvent(
                    event_type=ExecutionEventType.AGENT_START,
                    author=new_agent.name,
                    content=[Part(text=task_text)],
                    action=AgentAction(
                        name=new_agent.name,
                        parameters=start_parameters,
                        thought=None,
                        message=None,
                    ),
                    conversation_id=state.conversation_id,
                    interaction_id=new_interaction_id,
                    turn_id=state.turn_id,
                )
                state.current_agent = new_agent
                state.interaction_id = new_interaction_id
                await self._refresh_state(state)
        elif event.event_type == ExecutionEventType.AGENT_START:
            new_agent = next(
                (agent for agent in state.available_agents if agent.name == event.action.name), None
            )
        elif event.event_type == ExecutionEventType.AGENT_END:
            await self.conversation_manager.pop_agent_from_stack(
                state.application_name, state.user_id, state.conversation_id
            )

            stack_entry = await self.conversation_manager.get_last_agent_stack(
                state.application_name, state.user_id, state.conversation_id
            )

            new_agent = None
            if stack_entry:
                new_agent = next(
                    (
                        agent
                        for agent in state.available_agents
                        if agent.name == stack_entry.agent_name
                    ),
                    state.entrypoint_agent,
                )
                new_interaction_id = stack_entry.interaction_id or new_interaction_id
        else:
            raise ValueError(f"Unexpected event type {event.event_type} for agent transition.")

        if new_agent is None:
            return None, current_context, None

        # For AGENT_CALL, state was already updated above; update for other event types here
        if event.event_type != ExecutionEventType.AGENT_CALL:
            state.current_agent = new_agent
            state.interaction_id = new_interaction_id
            await self._refresh_state(state)

        stack_entry = await self.conversation_manager.get_last_agent_stack(
            state.application_name, state.user_id, state.conversation_id
        )

        context_event = agent_start_event if agent_start_event is not None else event
        new_context = ContextBuilder.create_context(state, context_event, stack_entry)
        if event.event_type in (ExecutionEventType.AGENT_CALL, ExecutionEventType.AGENT_START):
            new_context.is_subagent = True
            await self.conversation_manager.add_agent_to_stack(
                state.application_name,
                state.user_id,
                state.conversation_id,
                agent_name=new_agent.name,
                interaction_id=new_context.interaction_id,
                turn_id=state.turn_id,
            )
        # Propagate the caller's metadata (e.g. user_info) down the handoff chain from the
        # parent's input event — NOT from run_config (which carries secrets and is the tool
        # channel). current_context.agent_input is the parent's USER_INPUT-derived root
        # AGENT_START (or a parent sub-agent's start) and is always present.
        if agent_start_event is not None and current_context.agent_input is not None:
            agent_start_event.metadata.update(copy.deepcopy(current_context.agent_input.metadata))
        return new_agent, new_context, agent_start_event

    async def _apply_agent_hooks(
        self,
        hooks: list[FunctionHook],
        state: RunnerState,
        context: InteractionContext,
        agent: Agent | DelegateAgent,
    ) -> AsyncGenerator[AgentHookExecution, None]:
        """Run agent hooks and yield each execution result."""
        async for hook_exec in self.run_agent_transition_hooks(
            hooks=hooks, context=context, agent=agent, state=state
        ):
            yield hook_exec

    async def _refresh_state(self, state: RunnerState) -> None:
        """Refresh conversation info in state."""
        state.conversation_info = await self.conversation_manager.get_conversation(
            state.application_name, state.user_id, state.conversation_id
        )

    async def _add_event(
        self,
        state: RunnerState,
        event: BaseEvent,
        context: InteractionContext | None,
        silent: bool = False,
    ) -> None:
        """Add event to conversation and interaction history."""
        event = self._ensure_event_identifiers(state=state, event=event, context=context)
        event.metadata.update(state.metadata)
        if silent:
            return
        await self.conversation_manager.add_conversation_event(
            state.application_name, state.user_id, state.conversation_id, event
        )
        # Add event also to interaction history if context is provided
        if context is not None:
            context.interaction_history.append(event)

    @staticmethod
    async def _collect_hook_results(
        hook: FunctionHook,
        fallback_error_msg: str,
        **kwargs: Any,
    ) -> AsyncGenerator[HookResult, None]:
        """Run one hook and yield all results (is_update partials then final).
        If the hook never yields a final result, synthesises an error HookResult."""
        has_final = False
        async for result in hook.run(**kwargs):
            yield result
            if not result.is_update:
                has_final = True
        if not has_final:
            yield HookResult(
                success=False,
                error=HookError(error_code=404, error_message=fallback_error_msg),
                name=hook.name,
            )

    @staticmethod
    async def _drain_listener_queue(
        listener_hooks: list[FunctionHook],
        **kwargs: Any,
    ) -> AsyncGenerator[HookResult, None]:
        """Run listener hooks concurrently via a queue and yield every HookResult."""
        queue: asyncio.Queue = asyncio.Queue()
        tasks = [
            asyncio.create_task(h.drain_to_queue(queue=queue, **kwargs)) for h in listener_hooks
        ]
        pending = len(tasks)
        while pending > 0:
            item = await queue.get()
            if not item.is_update:
                pending -= 1
            yield item
        await asyncio.gather(*tasks, return_exceptions=True)

    async def run_agent_transition_hooks(
        self,
        hooks: list[FunctionHook],
        context: InteractionContext,
        agent: BaseAgent,
        state: RunnerState,
    ) -> AsyncGenerator[AgentHookExecution, None]:
        break_loop_response = (
            None  # Overwrite in case more hooks break the execution only one will be used
        )
        silent_by_name = {hook.name: hook.silent for hook in hooks}
        if hooks:
            updater_hooks = [hook for hook in hooks if hook.type == HookType.UPDATER]
            # Run updater hooks sequentially
            context = copy.deepcopy(context)
            for hook in updater_hooks:
                event = BaseEvent(
                    event_type=ExecutionEventType.HOOK_START,
                    author=agent.name,
                    metadata={"hook_name": hook.name},
                    interaction_id=context.interaction_id,
                    turn_id=context.turn_id,
                )
                await self._add_event(
                    state, event, context=context, silent=silent_by_name.get(hook.name, False)
                )
                yield AgentHookExecution(event=event, context=context, break_execution=False)

                # All updater hooks are now async generators
                async for result in self._collect_hook_results(
                    hook,
                    f"Updater hook {hook.name} did not return a result.",
                    context=copy.deepcopy(context),
                    agent=agent,
                ):
                    if result.is_update:
                        hook_event = self.create_hook_event(
                            result, author=agent.name, context=context
                        )
                        await self._add_event(
                            state,
                            hook_event,
                            context=context,
                            silent=silent_by_name.get(hook.name, False),
                        )
                        yield AgentHookExecution(
                            event=hook_event,
                            context=context,
                            break_execution=False,
                        )
                    else:
                        hook_result = result  # AgentHookResult (final)
                if hook_result is None:
                    # Generator yielded nothing useful; treat as failure
                    hook_result = HookResult(
                        new_context=context,
                        emitted_content=None,
                        success=False,
                        error=HookError(
                            error_code=404,
                            error_message=f"Updater hook {hook.name} did not return a result.",
                        ),
                        name=hook.name,
                    )

                if hook_result.break_execution:
                    break_loop_response = self.break_loop_response_event(
                        hook_result, author=agent.name
                    )
                # Skip updating context if the hook failed.
                if not hook_result.success:
                    error = hook_result.error
                    # `error_code` is the exception class name for a hook that raised, the
                    # role `type(e).__name__` plays elsewhere.
                    code = error.error_code if error is not None else "no error reported"
                    get_logger().warning(f"hook {hook.name} failed: {code}")
                    get_logger().debug(f"hook {hook.name} failed, detail: {error}")
                new_context = hook_result.new_context if hook_result.success else context
                hook_end_event = self.create_hook_event(
                    hook_result, author=agent.name, context=context
                )
                await self._add_event(
                    state,
                    hook_end_event,
                    context=context,
                    silent=silent_by_name.get(hook.name, False),
                )
                yield AgentHookExecution(
                    event=hook_end_event,
                    context=new_context,
                    break_execution=hook_result.break_execution,
                    continue_with=hook_result.continue_with,
                )

                context = new_context  # update context for next updater hook
            # RUN LISTENER HOOKS CONCURRENTLY ON THE UPDATED CONTEXT
            listener_hooks = [hook for hook in hooks if hook.type == HookType.LISTENER]

            if listener_hooks:
                for hook in listener_hooks:
                    hook_start = BaseEvent(
                        event_type=ExecutionEventType.HOOK_START,
                        author=agent.name,
                        metadata={"hook_name": hook.name},
                        interaction_id=context.interaction_id,
                        turn_id=context.turn_id,
                    )
                    await self._add_event(
                        state,
                        hook_start,
                        context=context,
                        silent=silent_by_name.get(hook.name, False),
                    )
                    yield AgentHookExecution(
                        event=hook_start,
                        context=context,
                        break_execution=False,
                    )

                async for item in self._drain_listener_queue(
                    listener_hooks, context=copy.deepcopy(context), agent=agent
                ):
                    if item.break_execution and not item.is_update:
                        break_loop_response = self.break_loop_response_event(
                            item, author=agent.name
                        )
                    hook_event = self.create_hook_event(item, author=agent.name, context=context)
                    await self._add_event(
                        state,
                        hook_event,
                        context=context,
                        silent=silent_by_name.get(item.name, False),
                    )
                    yield AgentHookExecution(
                        event=hook_event,
                        context=context,
                        break_execution=item.break_execution and not item.is_update,
                    )

        if break_loop_response:
            break_loop_response.turn_id = context.turn_id
            await self._add_event(state, break_loop_response, context=context)
            yield AgentHookExecution(
                event=break_loop_response,
                context=context,
                break_execution=True,
            )

    async def run_on_event_hooks(
        self,
        hooks: list[FunctionHook],
        event: BaseEvent,
        context: InteractionContext | None,
        state: RunnerState,
        agent_name: str = "runner",
    ) -> AsyncGenerator[OnEventHookExecution, None]:
        current_event = event

        hooks = [h for h in hooks if event.event_type in h.on_event_triggers]
        if not hooks:
            return

        silent_by_name = {hook.name: hook.silent for hook in hooks}

        def _hook_start(hook_name: str) -> BaseEvent:
            return BaseEvent(
                event_type=ExecutionEventType.HOOK_START,
                author=agent_name,
                metadata={"hook_name": hook_name},
                interaction_id=context.interaction_id if context else current_event.interaction_id,
                turn_id=context.turn_id if context else current_event.turn_id,
            )

        def _make_hook_event(hook_result: HookResult) -> BaseEvent:
            return self.create_hook_event(
                hook_result,
                author=agent_name,
                context=context,
                interaction_id=current_event.interaction_id,
                turn_id=current_event.turn_id,
            )

        break_loop_response: BaseEvent | None = None

        updater_hooks = [h for h in hooks if (h.type == HookType.UPDATER)]
        for hook in updater_hooks:
            hook_start = _hook_start(hook.name)
            await self._add_event(
                state, hook_start, context=context, silent=silent_by_name.get(hook.name, False)
            )
            yield OnEventHookExecution(
                hook_event=hook_start,
                current_event=current_event,
            )
            async for result in self._collect_hook_results(
                hook,
                f"Updater on_agent_event hook {hook.name} did not return a result.",
                event=current_event,
                context=context,
            ):
                if result.is_update:
                    hook_event = _make_hook_event(result)
                    await self._add_event(
                        state,
                        hook_event,
                        context=context,
                        silent=silent_by_name.get(hook.name, False),
                    )
                    yield OnEventHookExecution(
                        hook_event=hook_event,
                        current_event=current_event,
                    )
                else:
                    if result.break_execution and break_loop_response is None:
                        br = self.break_loop_response_event(result, author=agent_name)
                        br.turn_id = current_event.turn_id
                        br.interaction_id = current_event.interaction_id
                        break_loop_response = br
                    if result.success and result.modified_event is not None:
                        current_event = result.modified_event
                    hook_event = _make_hook_event(result)
                    await self._add_event(
                        state,
                        hook_event,
                        context=context,
                        silent=silent_by_name.get(hook.name, False),
                    )
                    yield OnEventHookExecution(
                        hook_event=hook_event,
                        current_event=current_event,
                    )

        listener_hooks = [h for h in hooks if h.type == HookType.LISTENER]
        if listener_hooks:
            for hook in listener_hooks:
                hook_start = _hook_start(hook.name)
                await self._add_event(
                    state,
                    hook_start,
                    context=context,
                    silent=silent_by_name.get(hook.name, False),
                )
                yield OnEventHookExecution(
                    hook_event=hook_start,
                    current_event=current_event,
                )

            async for item in self._drain_listener_queue(
                listener_hooks, event=current_event, context=context
            ):
                if item.break_execution and not item.is_update and break_loop_response is None:
                    br = self.break_loop_response_event(item, author=agent_name)
                    br.turn_id = current_event.turn_id
                    br.interaction_id = current_event.interaction_id
                    break_loop_response = br
                hook_event = _make_hook_event(item)
                await self._add_event(
                    state,
                    hook_event,
                    context=context,
                    silent=silent_by_name.get(item.name, False),
                )
                yield OnEventHookExecution(
                    hook_event=hook_event,
                    current_event=current_event,
                )

        if break_loop_response is not None:
            await self._add_event(state, break_loop_response, context=context)
            yield OnEventHookExecution(
                hook_event=break_loop_response,
                current_event=current_event,
                break_execution=True,
            )

    async def run_tool_pre_hooks(
        self,
        tool: FunctionTool | DelegateTool,
        tool_args: dict,
        context: InteractionContext,
        state: RunnerState,
    ) -> AsyncGenerator[tuple[BaseEvent, dict], None]:
        # RUN UPDATER HOOKS SEQUENTIALLY
        updater_hooks: list[FunctionHook] = [
            hook for hook in tool.pre_hooks if hook.type == HookType.UPDATER
        ]
        silent_by_name = {hook.name: hook.silent for hook in tool.pre_hooks}
        tool_args = {
            k: copy.deepcopy(v) if k not in NO_COPY_KEYS else v for k, v in tool_args.items()
        }
        for hook in updater_hooks:
            hook_start = BaseEvent(
                event_type=ExecutionEventType.HOOK_START,
                author=state.current_agent.name,
                metadata={"hook_name": hook.name},
                interaction_id=context.interaction_id,
            )
            await self._add_event(
                state, hook_start, context=context, silent=silent_by_name.get(hook.name, False)
            )
            yield hook_start, tool_args
            async for result in self._collect_hook_results(
                hook,
                f"Updater Tool pre-hook {hook.name} did not return a result.",
                tool=tool,
                tool_args=copy.deepcopy(tool_args),
            ):
                if result.is_update:
                    hook_event = self.create_hook_event(
                        result, author=state.current_agent.name, context=context
                    )
                    await self._add_event(
                        state,
                        hook_event,
                        context=context,
                        silent=silent_by_name.get(hook.name, False),
                    )
                    yield hook_event, tool_args
                else:
                    tool_args = (
                        result.new_tool_args
                        if result.success and result.new_tool_args is not None
                        else tool_args
                    )
                    hook_event = self.create_hook_event(
                        result, author=state.current_agent.name, context=context
                    )
                    await self._add_event(
                        state,
                        hook_event,
                        context=context,
                        silent=silent_by_name.get(hook.name, False),
                    )
                    yield hook_event, tool_args
                    if result.new_tool_args is not None and "state" in result.new_tool_args:
                        context.state.clear()
                        context.state.update(result.new_tool_args.get("state", {}))
        # RUN LISTENER HOOKS CONCURRENTLY ON THE NEW KWARGS
        listener_hooks: list[FunctionHook] = [
            hook for hook in tool.pre_hooks if hook.type == HookType.LISTENER
        ]
        if listener_hooks:
            for hook in listener_hooks:
                hook_start = BaseEvent(
                    event_type=ExecutionEventType.HOOK_START,
                    author=state.current_agent.name,
                    metadata={"hook_name": hook.name},
                    interaction_id=context.interaction_id,
                )
                await self._add_event(
                    state,
                    hook_start,
                    context=context,
                    silent=silent_by_name.get(hook.name, False),
                )
                yield hook_start, tool_args
            async for item in self._drain_listener_queue(
                listener_hooks, tool=tool, tool_args=copy.deepcopy(tool_args)
            ):
                hook_event = self.create_hook_event(
                    item, author=state.current_agent.name, context=context
                )
                await self._add_event(
                    state,
                    hook_event,
                    context=context,
                    silent=silent_by_name.get(item.name, False),
                )
                yield hook_event, tool_args

    async def run_tool_post_hooks(
        self,
        tool: FunctionTool | DelegateTool,
        tool_result: ToolResult,
        context: InteractionContext,
        state: RunnerState,
    ) -> AsyncGenerator[tuple[BaseEvent, ToolResult], None]:
        # RUN UPDATER HOOKS SEQUENTIALLY
        tool_result = copy.deepcopy(tool_result)
        updater_hooks: list[FunctionHook] = [
            hook for hook in tool.post_hooks if hook.type == HookType.UPDATER
        ]
        silent_by_name = {hook.name: hook.silent for hook in tool.post_hooks}
        for hook in updater_hooks:
            hook_start = BaseEvent(
                event_type=ExecutionEventType.HOOK_START,
                author=state.current_agent.name,
                metadata={"hook_name": hook.name},
                interaction_id=context.interaction_id,
            )
            await self._add_event(
                state, hook_start, context=context, silent=silent_by_name.get(hook.name, False)
            )
            yield hook_start, tool_result
            async for result in self._collect_hook_results(
                hook,
                f"Updater Tool post-hook {hook.name} did not return a result.",
                tool=tool,
                result=copy.deepcopy(tool_result),
            ):
                if result.is_update:
                    hook_event = self.create_hook_event(
                        result, author=state.current_agent.name, context=context
                    )
                    await self._add_event(
                        state,
                        hook_event,
                        context=context,
                        silent=silent_by_name.get(hook.name, False),
                    )
                    yield hook_event, tool_result
                else:
                    tool_result = (
                        result.new_result
                        if result.success and result.new_result is not None
                        else tool_result
                    )
                    hook_event = self.create_hook_event(
                        result, author=state.current_agent.name, context=context
                    )
                    await self._add_event(
                        state,
                        hook_event,
                        context=context,
                        silent=silent_by_name.get(hook.name, False),
                    )
                    yield hook_event, tool_result
                    if result.new_result is not None and result.new_result.state is not None:
                        context.state.clear()
                        context.state.update(result.new_result.state)

        # RUN LISTENER HOOKS CONCURRENTLY ON THE NEW RESULT
        listener_hooks: list[FunctionHook] = [
            hook for hook in tool.post_hooks if hook.type == HookType.LISTENER
        ]
        if listener_hooks:
            for hook in listener_hooks:
                hook_start = BaseEvent(
                    event_type=ExecutionEventType.HOOK_START,
                    author=state.current_agent.name,
                    metadata={"hook_name": hook.name},
                    interaction_id=context.interaction_id,
                )
                await self._add_event(
                    state,
                    hook_start,
                    context=context,
                    silent=silent_by_name.get(hook.name, False),
                )
                yield hook_start, tool_result
            async for item in self._drain_listener_queue(
                listener_hooks, tool=tool, result=copy.deepcopy(tool_result)
            ):
                hook_event = self.create_hook_event(
                    item, author=state.current_agent.name, context=context
                )
                await self._add_event(
                    state,
                    hook_event,
                    context=context,
                    silent=silent_by_name.get(item.name, False),
                )
                yield hook_event, tool_result

    @staticmethod
    def create_hook_event(
        hook_result: HookResult,
        author: str,
        context: InteractionContext | None,
        interaction_id: str | None = None,
        turn_id: str | None = None,
    ) -> BaseEvent:
        _interaction_id = context.interaction_id if context else interaction_id
        _turn_id = context.turn_id if context else turn_id
        if hook_result.is_update:
            event = BaseEvent(
                event_type=ExecutionEventType.HOOK_UPDATE,
                author=author,
                metadata={"hook_name": hook_result.name} if hook_result.name else {},
                content=hook_result.emitted_content,
                interaction_id=_interaction_id,
                turn_id=_turn_id,
            )
        elif hook_result.success:
            event = BaseEvent(
                event_type=ExecutionEventType.HOOK_END,
                author=author,
                metadata={"hook_name": hook_result.name},
                content=hook_result.emitted_content,
                interaction_id=_interaction_id,
                turn_id=_turn_id,
            )
        else:
            if hook_result.error:
                error_code = hook_result.error.error_code or "unknown_error"
                error_message = (
                    hook_result.error.error_message or "Unknown error occurred while running hook."
                )
            else:
                error_code = "unknown_error"
                error_message = "Unknown error occurred while running hook."
            event = BaseEvent(
                event_type=ExecutionEventType.HOOK_ERROR,
                author=author,
                error_code=str(error_code),
                error_message=error_message,
                content=hook_result.emitted_content if hook_result.emitted_content else [],
                metadata={"hook_name": hook_result.name},
                interaction_id=_interaction_id,
                turn_id=_turn_id,
            )
        return event

    @staticmethod
    def break_loop_response_event(hook_result: HookResult, author: str) -> BaseEvent:
        response_content = None
        if hook_result.emitted_content:  # Check reason of guardrail that didn't pass first
            for part in hook_result.emitted_content:
                if (
                    part.guardrail_result
                    and (not part.guardrail_result.passed)
                    and not part.guardrail_result.error_occurred
                    and part.guardrail_result.reason
                ):
                    response_content = part.guardrail_result.reason
                    break
            if (
                response_content is None
            ):  # If no guardrail reason, check if there is a text to return
                for part in hook_result.emitted_content:
                    if part.text:
                        response_content = part.text
                        break
            if response_content is None:  # If there is still no text, return default message
                response_content = f"Execution stopped by hook {hook_result.name}"
        response_event = BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            content=[Part(text=response_content)],
            author=author,
            action=BaseAction(name=hook_result.name, parameters={"break_execution": True}),
        )
        return response_event

    @staticmethod
    def parameters_changed(current: Any, previous: Any) -> bool:
        exclude = {"reasoning"}
        return {k: v for k, v in current.items() if k not in exclude} != {
            k: v for k, v in previous.items() if k not in exclude
        }


def _turn_task(context: Any) -> str:
    """The user's text for this turn, as the decision use cases see it.

    The whole input, not its first part: a model routing on the task should see
    everything that was said. Mirrors `skill_query_for`, which does the same for
    skills — kept separate rather than imported because that one builds a whole
    SkillQuery and this needs only the text.
    """
    agent_input = getattr(context, "agent_input", None)
    if agent_input is None or not getattr(agent_input, "content", None):
        return ""
    return " ".join(part.text or "" for part in agent_input.content).strip()


# Re-exported on first use: the defining modules import this package, so naming
# them here at import time would be a cycle.
_LAZY_EXPORTS = {
    "RunnerTransport": "domyn.agents.runner.transport",
}

if TYPE_CHECKING:
    from domyn.agents.runner.transport import RunnerTransport as RunnerTransport


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        value = getattr(import_module(_LAZY_EXPORTS[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
