"""DelegateAgent — an agent whose execution is handled by a remote runner.

A DelegateAgent appears in the local agent hierarchy (so the parent agent's
planner can hand off to it), but its actual execution is delegated via a
RunnerTransport to another runner instance, which may be in-process or remote.

The agent is a thin wrapper: it streams every event from the transport to the
runner, under this delegation's ids and the local time.  The only other
transformation is that a final (non-partial) ``RESPONSE`` is converted to
``AGENT_END`` when the agent is running as a sub-agent, so the parent
orchestrator can resume with the result in history.

Tool events (``TOOL_START`` / ``TOOL_END``) from the external agent are
forwarded for observability.  The runner already skips local tool execution
for ``DelegateAgent`` instances, so no special routing is needed here.

Configuration example (programmatic)::

    from domyn.agents.agents.delegate_agent import DelegateAgent
    from domyn.agents.runner.websocket_transport import WebSocketTransport

    transport = WebSocketTransport(ws_url="ws://runner-b/ws")
    specialist = DelegateAgent(
        name="Specialist",
        description="Expert in domain X",
        transport=transport,
    )
    root_agent.sub_agents.append(specialist)
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Coroutine
from datetime import datetime
from typing import Any, ClassVar, cast

from pydantic import ConfigDict, Field, model_validator

from domyn.agents.agents.base import ContextVisibility, HandoffScope
from domyn.agents.core import (
    AgentDescription,
    BaseAgent,
    BaseEvent,
    ExecutionEventType,
    InteractionContext,
    Part,
    VisibilityScope,
)
from domyn.agents.decisions import DecisionRuntime
from domyn.agents.hooks import FunctionHook, HookPhase
from domyn.agents.logger import get_logger
from domyn.agents.tools import DelegateTool, FunctionTool, MCPTool

logger = get_logger()


class DelegateAgent(BaseAgent):
    """Agent whose execution is fully delegated to a remote runner."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_by_name=True, validate_by_alias=True
    )

    # Strong references to the fire-and-forget cancel signals below: a task with
    # no live reference can be collected before it has run. Shared by every
    # instance because the signal has to outlive the agent that emits it.
    _cancel_tasks: ClassVar[set[asyncio.Task[None]]] = set()

    type: str = "delegate"
    stream: bool = False  # runner must not suppress the promoted RESPONSE event

    decisions: DecisionRuntime | None = Field(
        default=None,
        exclude=True,
        description=(
            "The same optional decision model `Agent` takes, for the same reason it is "
            "needed here: a tool call arriving from the delegated runtime is approved "
            "by the *local* gate (`Runner._execute_agent_loop`, the `DelegateAgent` "
            "branch), because leaving that out would make approvals bypassable by "
            "delegating. Without this field the gate is there and the model behind it "
            "never is, so an `approvals: active` policy would hold for local calls and "
            "quietly not for delegated ones. `None` — the default — is today's "
            "behaviour: the call falls through to the hooks and then to the user."
        ),
    )

    invocation_parameters_override: dict[str, Any] | None = Field(
        default=None,
        alias="invocation_parameters",
        description=(
            "Optional JSON Schema overriding the default 'task' invocation shape — "
            "typically pushed by the remote runner on startup so the parent "
            "orchestrator knows what arguments the delegated agent expects."
        ),
    )

    runner_url: str | None = Field(
        default=None,
        description="URL of the remote runner. Used by HTTP-based transports.",
    )
    transport: Any | None = Field(
        default=None,
        description="RunnerTransport instance that handles communication with the remote runner.",
    )
    context_visibility: ContextVisibility = Field(
        default=ContextVisibility.STANDARD,
        description=(
            "Context/history visibility level for this delegated agent, with the same "
            "READ-scope semantics as Agent.context_visibility."
        ),
    )
    handoff_scope: HandoffScope = Field(
        default=HandoffScope.EXPLICIT,
        description=(
            "EXECUTE scope for handoff targets, with the same semantics as "
            "Agent.handoff_scope. EXPLICIT honors `sub_agents` verbatim; DIRECT_PEERS/GLOBAL "
            "resolve the callable set against the live agent tree at runtime."
        ),
    )

    visibility: VisibilityScope = Field(
        default=VisibilityScope.ISOLATED,
        description=(
            "Controls whether the caller's run_config (PII: user_id, space_id, auth "
            "tokens, …) is forwarded to the external agent, using the same trust ladder "
            "as MCP tools. ISOLATED/SCOPED forward nothing; TRUSTED/GLOBAL forward the "
            "run_config as a separate envelope argument (never inside the event). Grant "
            "PII-carrying scopes from the host application, not from untrusted config."
        ),
    )

    tools: list[FunctionTool | DelegateTool] = Field(
        default_factory=list,
        description=(
            "Tools the external agent may call. FunctionTools are executed locally "
            "and the result is sent back; DelegateTools are observed only (the "
            "external agent executes them itself)."
        ),
    )

    hooks: list[FunctionHook] = Field(
        default_factory=list, description="Hooks to be executed during the agent execution"
    )

    @property
    def pre_hooks(self) -> list[FunctionHook]:
        return [h for h in self.hooks if h.trigger == HookPhase.PRE_AGENT_CALL]

    @property
    def post_hooks(self) -> list[FunctionHook]:
        return [h for h in self.hooks if h.trigger == HookPhase.POST_AGENT_CALL]

    @property
    def on_event_hooks(self) -> list[FunctionHook]:
        return [h for h in self.hooks if h.trigger == HookPhase.ON_AGENT_EVENT]

    @classmethod
    def from_config(cls, config: dict[str, Any], save_in_registry: bool = False) -> DelegateAgent:
        """Build a DelegateAgent (or subclass) from a config dict.

        Inherited by subclasses so that ``cls`` is the concrete type and
        ``model_validate`` yields an instance of it. Registration in the
        agent registry is handled by ``AgentFactory.from_config``.
        """
        from domyn.agents.agents.base import normalize_legacy_visibility

        return cls.model_validate(normalize_legacy_visibility(config))

    @model_validator(mode="before")
    @classmethod
    def _build_transport_from_dict(cls, values: Any) -> Any:
        """If ``transport`` is a plain dict, build the RunnerTransport from it.

        This lets callers (including the config loader) pass raw dicts and get
        a fully wired DelegateAgent back from ``model_validate``.
        """
        if not isinstance(values, dict):
            return values

        transport = values.get("transport")
        if transport is None or not isinstance(transport, dict):
            return values

        from domyn.agents.runner.transport import build_transport

        built = build_transport(transport)
        values["transport"] = built
        # Preserve runner_url as a convenience attribute for HTTP-based transports.
        values.setdefault("runner_url", transport.get("url") or None)
        # Tools stay in values["tools"] so they are held on the agent and
        # looked up by the runner when the external agent calls them.

        return values

    @property
    def invocation_parameters(self) -> dict[str, Any]:
        """Parameters for agent invocation as function.

        Returns the override pushed by the remote runner if set, otherwise the
        default `{"task": "<string>"}` schema.
        """
        if self.invocation_parameters_override is not None:
            return self.invocation_parameters_override
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The specific task to perform by the agent. This is the more specific request for the agent to perform, with all the necessary details.",
                },
            },
            "required": ["task"],
        }

    @property
    def invocation_json_schema(self) -> dict:
        schema = {
            "properties": {
                self.name_string: {"const": self.name, "type": "string"},
                self.parameters_string: self.invocation_parameters,
            },
            "required": [self.parameters_string],
            "description": self.description,
        }
        return schema

    def get_tool_by_name(self, tool_name: str) -> FunctionTool | DelegateTool | None:
        return next((t for t in self.tools if t.name == tool_name), None)

    def get_agent_description(self) -> AgentDescription:
        return AgentDescription(
            name=self.name,
            description=self.description,
            invocation_json_schema=self.invocation_json_schema,
            invocation_parameters=self.invocation_parameters,
        )

    def __hash__(self) -> int:
        return hash(self.name)

    async def execute(
        self, context: InteractionContext
    ) -> AsyncGenerator[tuple[BaseEvent, InteractionContext], None]:
        if self.transport is None:
            raise RuntimeError(
                f"DelegateAgent '{self.name}' has no transport configured. "
                "Set a RunnerTransport before execution."
            )

        originating_action = context.agent_input.action if context.agent_input is not None else None
        agent_end_emitted = False
        _cancelled = False

        # run_config carries PII; forward it only when this delegate's visibility grants
        # it, as a separate argument (never inside the wire event). See VisibilityScope.
        # Pass the kwarg only when there is something to forward, so transports that
        # predate this argument (and the default ISOLATED path) keep working unchanged.
        filtered_run_config = self.visibility.filter_run_config(context.run_config)
        execute_kwargs: dict[str, Any] = (
            {"run_config": filtered_run_config} if filtered_run_config else {}
        )

        try:
            async for event in self.transport.execute(self.name, context, **execute_kwargs):
                # Ids and time are this delegation's, not the remote runner's: it mints its
                # own interaction_id, under which a reply to its approval request finds no
                # request, and its clock can be hours off, which puts history out of order.
                event.conversation_id = context.conversation_id or event.conversation_id
                event.turn_id = context.turn_id or event.turn_id
                event.interaction_id = context.interaction_id or event.interaction_id
                event.timestamp = datetime.now()

                # Some external agents signal completion via AGENT_END (with content)
                # instead of RESPONSE. Promote it to a final RESPONSE only on the
                # root-agent path. For sub-agent path, keep AGENT_END semantics.
                # An AGENT_END with no content is a pure wire-level termination
                # signal — end the generator cleanly.

                if event.event_type == ExecutionEventType.AGENT_END:
                    if event.author != self.name:
                        # Remote sub-agent AGENT_END: the transport preserved the original
                        # remote author to distinguish it from the root session end.  Yield
                        # verbatim so the runner can log it for observability without
                        # treating it as this DelegateAgent's completion.
                        yield event, context
                        continue
                    # No content is passed on as it is. The delegate may have streamed its
                    # answer as tokens and have nothing to add, so no text is invented for
                    # the user to read; the final RESPONSE is still sent because it is what
                    # closes the turn (the runner ends it with an AGENT_END only after one).
                    if context.is_subagent:
                        # Ensure the AGENT_END carries the originating action so the parent
                        # context manager can reconstruct the assistant message.
                        # Preserve error_code/error_message so the parent planner's
                        # OBSERVATION message surfaces the failure to the parent LLM.
                        agent_end_emitted = True
                        yield (
                            BaseEvent(
                                event_type=ExecutionEventType.AGENT_END,
                                author=self.name,
                                content=event.content,
                                action=originating_action,
                                error_code=event.error_code,
                                error_message=event.error_message,
                            ),
                            context,
                        )
                    else:
                        logger.info(
                            f"[DelegateAgent '{self.name}'] Promoting AGENT_END with content "
                            "to final RESPONSE"
                        )
                        yield (
                            BaseEvent(
                                event_type=ExecutionEventType.RESPONSE,
                                author=self.name,
                                content=event.content,
                                is_partial=False,
                                error_code=event.error_code,
                                error_message=event.error_message,
                            ),
                            context,
                        )
                    continue

                if (
                    event.event_type == ExecutionEventType.RESPONSE
                    and not event.is_partial
                    and context.is_subagent
                ):
                    # Sub-agent path: the parent orchestrator expects AGENT_END to resume.
                    # Carry the originating action so the parent context manager can
                    # reconstruct the assistant message.
                    logger.info(
                        f"[DelegateAgent '{self.name}'] Converting final RESPONSE to AGENT_END "
                        "for parent orchestrator"
                    )
                    agent_end_emitted = True
                    yield (
                        BaseEvent(
                            event_type=ExecutionEventType.AGENT_END,
                            author=self.name,
                            content=event.content,
                            action=originating_action,
                        ),
                        context,
                    )
                else:
                    yield event, context

        except asyncio.CancelledError:
            _cancelled = True
            raise
        finally:
            if _cancelled:
                # Fire-and-forget: schedule cancel signal independently so it is not
                # itself cancelled by the propagating CancelledError.
                task = asyncio.ensure_future(self.transport.cancel(context.conversation_id or ""))
                DelegateAgent._cancel_tasks.add(task)
                task.add_done_callback(DelegateAgent._cancel_tasks.discard)

        # Fallback: if running as sub-agent and the transport ended without emitting
        # a proper AGENT_END (e.g. connection drop, timeout, or partial-only stream),
        # synthesize one so the parent runner can pop the stack and resume correctly.
        if context.is_subagent and not agent_end_emitted:
            logger.warning(
                f"[DelegateAgent '{self.name}'] Transport ended without AGENT_END — "
                "emitting fallback"
            )
            yield (
                BaseEvent(
                    event_type=ExecutionEventType.AGENT_END,
                    author=self.name,
                    content=[
                        Part(thought=f"Agent {self.name} completed without explicit response")
                    ],
                    action=originating_action,
                ),
                context,
            )

    async def initialize_mcp_tools(self) -> None:
        """Initialize and fetch tools from configured MCP servers.

        This method should be called after agent creation to automatically
        discover and add tools from configured MCP servers.
        """

        tasks: list[Coroutine[Any, Any, Any]] = []

        async def safe_tool_init(t: MCPTool) -> None:
            try:
                await t.initialize()
            except Exception as e:
                get_logger().warning(
                    f"Could not initialize MCP tool {t.name} from {t.server_url}: "
                    f"{type(e).__name__}"
                )
                get_logger().debug(f"MCP tool {t.name} initialization detail: {e}")

        mcp_tools = filter(lambda x: isinstance(x, MCPTool), cast(list[MCPTool], self.tools))
        init_tasks = map(safe_tool_init, mcp_tools)
        tasks.extend(init_tasks)
        await asyncio.gather(*tasks)
