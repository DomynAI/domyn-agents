from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Coroutine
from typing import Any, cast

from pydantic import ConfigDict, Field, field_serializer, field_validator

from domyn.agents.agents.base import ContextVisibility, HandoffScope, normalize_legacy_visibility
from domyn.agents.agents.delegate_agent import DelegateAgent
from domyn.agents.agents.wrapped import WrappedAgent
from domyn.agents.core import (
    AgentDescription,
    BaseAgent,
    BaseEvent,
    ExecutionEventType,
    InteractionContext,
)
from domyn.agents.decisions import DecisionRuntime
from domyn.agents.hooks import FunctionHook, HookFactory, HookPhase, HookType
from domyn.agents.instructions import AgentInstructions, InstructionRuntime
from domyn.agents.llm import BaseLLMProvider, LLMProviderFactory
from domyn.agents.logger import get_logger
from domyn.agents.memory.managers import BaseMemoryManager, MemoryExecutionMode
from domyn.agents.planner_strategy import BasePlannerStrategy, PlannerStrategyFactory
from domyn.agents.skills import LoadedSkills, SkillRequiredError, SkillsRuntime
from domyn.agents.skills.integration import skill_tools
from domyn.agents.skills.turn import build_skills_payload
from domyn.agents.tools import (
    ConfigParam,
    DelegateTool,
    FunctionTool,
    MCPTool,
    StreamableFunctionTool,
    Tool,
)
from domyn.agents.tools.builtin import BuiltinTools
from domyn.agents.tools.mcp import MCPServerStreamableHTTPConfig
from domyn.agents.tools.read_tool_result import READ_TOOL_RESULT


def _default_planner_factory() -> BasePlannerStrategy:
    """The planner an agent gets when none is given: ``tool_use_thought``.

    Native tool calling, with the model's reasoning surfaced as a thought before
    each call — the planner the CLI runs on. Not ``react``: that one needs the
    model to answer in a ``<thought>/<action>`` text format, and a model that
    drifts from it fails the turn with ``UNPARSABLE_RESPONSE``.
    """
    from domyn.agents.planner_strategy import (
        ToolUseWithThoughtPlannerStrategy,
    )

    return ToolUseWithThoughtPlannerStrategy()


MAX_AGENT_NAME_LENGTH = 30


class Agent(BaseAgent):
    """Base class for all agents - serves as a resource container."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: str = "Agent"

    name: str = Field(
        ..., description="Name of the agent", min_length=1, max_length=MAX_AGENT_NAME_LENGTH
    )
    planner: BasePlannerStrategy = Field(
        default_factory=_default_planner_factory,
        description=(
            "Planning strategy for the agent. Defaults to `tool_use_thought` "
            "(native tool calling with the model's reasoning surfaced)."
        ),
    )
    description: str = Field(..., description="Description of the agent", max_length=1000)
    instruction: str | Awaitable[str] = Field(..., description="Instruction for the agent")
    """The agent's own instruction, written by its author.

    **Deliberately uncapped.** It carried ``max_length=2000``, a limit that
    validated nothing — this value is never serialized, never sent to a peer
    agent, and never reaches a delegate — and had bent two designs around
    itself: the skills index went into a tool description to escape it
    (``skills/integration.py``), and standing instructions nearly got a second
    field of their own. Standing instructions (``standing_instructions``) are
    merged after it at render time and never written into it.
    """
    llm_provider: BaseLLMProvider = Field(..., description="LLM provider")
    tools: list[FunctionTool | DelegateTool] = Field(
        default_factory=list, description="Available tools"
    )
    sub_agents: list[Agent | DelegateAgent | WrappedAgent | str] = Field(
        default_factory=list, description="Sub-agents"
    )  # type: ignore[assignment]
    max_iterations: int = Field(default=3, description="Maximum number of iterations")
    mcp_servers: list[MCPServerStreamableHTTPConfig] = Field(
        default_factory=list,
        description="MCP server configurations for auto-fetching tools",
    )
    hooks: list[FunctionHook] = Field(
        default_factory=list, description="Hooks to be executed during the agent execution"
    )
    context_visibility: ContextVisibility = Field(
        default=ContextVisibility.STANDARD,
        description=(
            "Context/history visibility level for this agent (READ scope). Callable "
            "handoff targets are configured separately through handoff_scope."
        ),
    )
    handoff_scope: HandoffScope = Field(
        default=HandoffScope.EXPLICIT,
        description=(
            "EXECUTE scope: which agents this agent may invoke via handoff. "
            "EXPLICIT (default) honors the explicit `sub_agents` list verbatim. "
            "DIRECT_PEERS/GLOBAL resolve the callable set against the live agent tree at "
            "runtime (siblings / entire entrypoint subtree) — useful for built-in "
            "agents that must reach user-defined agents unknown at definition time. "
            "Orthogonal to context_visibility (READ scope)."
        ),
    )
    memory_manager: BaseMemoryManager | None = None
    skills: SkillsRuntime | None = Field(
        default=None,
        description=(
            "Skills available to this agent. Declared inline so each agent in a "
            "multi-agent setup can carry its own set. Configuring it attaches the "
            "skill_view and skill_search tools automatically; the catalogue is loaded once per turn "
            "by the runner and selected from on each entry by execute()."
        ),
    )

    standing_instructions: InstructionRuntime | None = Field(
        default=None,
        description=(
            "Standing instructions (SOUL.md / AGENTS.md) for this agent. Loaded once "
            "per turn by the runner and merged after `instruction` in the system "
            "prompt; `instruction` itself is never rewritten."
        ),
    )

    decisions: DecisionRuntime | None = Field(
        default=None,
        exclude=True,
        description=(
            "An optional decision model (Jev/Clef, or an LLM standing in) this agent "
            "may consult, plus the per-use-case policy saying where it is allowed to "
            "matter. `None` — the default — means every caller behaves exactly as it "
            "does today: nothing is asked and nothing changes. Declared per agent so "
            "a reviewer agent can run approvals active while the rest of a tree does "
            "not. Stateless, and shared between the agents that declare it."
        ),
    )

    default_execute_agent_function: str = Field(
        default="_execute", description="Default execute agent function"
    )
    builtin_tools: list[BuiltinTools] = Field(
        default_factory=list,
        exclude=True,
        description=(
            "Built-in tool bundles (files/shell/browser) mounted on this agent, one per "
            "`tool_bundles` entry it declares — kept here (rather than only in "
            "`tools`) because each carries a Workspace and, for shell, a "
            "CommandBackend that outlives a single call and must be closed. "
            "Excluded from serialization: runtime infrastructure, not user config."
        ),
    )

    def model_post_init(self, __context: Any) -> None:
        """Called after model initialization to add custom tools."""
        if self.skills is not None:
            self._attach_skills()
        if self.memory_manager:
            if self.memory_manager.execution_mode == MemoryExecutionMode.AGENTIC:
                memory_tool = FunctionTool(
                    name="remember",
                    description="Store relevant information to the agent's memory for future use. Use this proactively whenever the user reveals information about themselves, their preferences, interests, or opinions — even if mentioned in passing.",
                    func=self.memory_manager.remember,  # type: ignore
                    config_params=[
                        ConfigParam(name="agent", value=self),
                    ],
                )
                self.tools.append(memory_tool)
            elif self.memory_manager.execution_mode == MemoryExecutionMode.AUTOMATIC:
                memory_post_hook = FunctionHook(
                    name="update_memory",
                    fun=self.memory_manager.update_memory,  # type: ignore
                    trigger=HookPhase.POST_AGENT_CALL,
                    type=HookType.LISTENER,
                )
                self.hooks.append(memory_post_hook)

    def _attach_skills(self) -> None:
        """Wire the lookup tools, ``skill_view`` and ``skill_search``.

        Only the parts that need no I/O happen here. Building the tools is
        pure construction; reading the catalogue touches disk or network and
        depends on who is asking, so the runner loads it once per turn and hands
        it to :meth:`execute`, which selects from it into a local.
        """

        if self.skills is None:  # pragma: no cover - guarded by the caller
            return
        # A selector that routes with a model gets one for free, so declaring
        # `selector: {type: llm}` needs no extra configuration. Bound once here
        # rather than per turn: the provider is a property of the agent, not of
        # the conversation.
        self.skills.bind_llm(self.llm_provider)
        # Same deal for a selector that routes with a decision model. Offered
        # even when it is None, so a selector can tell "no model configured"
        # from "never bound" and degrade to the index either way.
        self.skills.selector.bind_decisions(self.decisions)
        present = {getattr(t, "name", "") for t in self.tools}
        self.tools.extend(t for t in skill_tools(self.skills) if t.name not in present)

    async def close(self, *, _visited: set[int] | None = None) -> None:
        """Release what this agent's built-in tool bundles hold, and recurse into
        sub-agents that carry their own. Best-effort and idempotent — a single
        bundle failing to close must not stop the rest of the tree from being
        torn down, and nothing upstream can act on the failure anyway.

        ``_visited`` (by ``id()``, since these models aren't hashable) guards
        against closing a sub-agent twice when it's shared by more than one
        parent — the loader rejects true cycles, but not that.
        """
        visited = _visited if _visited is not None else set()
        if id(self) in visited:
            return
        visited.add(id(self))

        for built in self.builtin_tools:
            try:
                await built.close()
            except Exception as exc:
                get_logger().warning(f"could not release tool resources: {type(exc).__name__}")
                get_logger().debug(f"tool teardown detail: {exc}")

        for sub in self.sub_agents:
            if isinstance(sub, Agent):
                await sub.close(_visited=visited)

    @property
    def pre_hooks(self) -> list[FunctionHook]:
        return [h for h in self.hooks if h.trigger == HookPhase.PRE_AGENT_CALL]

    @property
    def post_hooks(self) -> list[FunctionHook]:
        return [h for h in self.hooks if h.trigger == HookPhase.POST_AGENT_CALL]

    @property
    def on_event_hooks(self) -> list[FunctionHook]:
        return [h for h in self.hooks if h.trigger == HookPhase.ON_AGENT_EVENT]

    @field_validator("llm_provider", mode="wrap")
    @classmethod
    def validate_llm_provider(cls, v: Any, handler: Any, info: Any) -> Any:
        """Validate and convert llm_provider from dict to concrete implementation using type field."""

        if isinstance(v, BaseLLMProvider):
            return v

        # Handle dict deserialization using type field
        if isinstance(v, dict):
            provider_type = v.get("type", None)
            if provider_type:
                try:
                    return LLMProviderFactory.from_config(v, save_in_registry=False)

                except Exception as e:
                    get_logger().warning(
                        f"Warning: Could not deserialize LLM provider type '{provider_type}': "
                        f"{type(e).__name__}"
                    )
                    get_logger().debug(f"LLM provider deserialization detail: {e}")

        try:
            return handler(v)
        except Exception:
            return v

    @field_serializer("llm_provider")
    def serialize_llm_provider(self, value: Any) -> Any:
        """Serialize llm_provider to dict with automatic type detection."""
        if hasattr(value, "model_dump"):
            serialized = value.model_dump()
            if hasattr(value, "__class__"):
                serialized["type"] = value.__class__.__name__
            return serialized
        return value

    @field_validator("planner", mode="wrap")
    @classmethod
    def validate_planner(cls, v: Any, handler: Any, info: Any) -> Any:
        """Validate and convert planner from dict to concrete implementation using type field."""

        if isinstance(v, BasePlannerStrategy):
            return v

        # Handle dict deserialization using type field
        if isinstance(v, dict):
            planner_type = v.get("type")
            if planner_type:
                try:
                    return PlannerStrategyFactory.from_config(v, save_in_registry=False)

                except Exception as e:
                    get_logger().warning(
                        f"Warning: Could not deserialize planner type '{planner_type}': "
                        f"{type(e).__name__}"
                    )
                    get_logger().debug(f"Planner deserialization detail: {e}")

        try:
            return handler(v)
        except Exception:
            return v

    @field_serializer("planner")
    def serialize_planner(self, value: Any) -> Any:
        """Serialize planner to dict with automatic type detection."""
        if hasattr(value, "model_dump"):
            serialized = value.model_dump()
            if hasattr(value, "__class__"):
                serialized["type"] = value.__class__.__name__
            return serialized
        return value

    @field_validator("memory_manager", mode="wrap")
    @classmethod
    def validate_memory_manager(cls, v: Any, handler: Any, info: Any) -> BaseMemoryManager | None:
        """Validate and reconstruct memory_manager from dict if needed."""
        logger = get_logger()
        if v is None:
            return None

        if isinstance(v, BaseMemoryManager):
            return v

        if isinstance(v, dict):
            manager_type = v.get("type")
            if manager_type is None:
                logger.warning("Memory manager config missing 'type' field")
                return None

            manager_class = BaseMemoryManager._registry.get(manager_type)
            if manager_class is None:
                logger.warning(f"Unknown memory manager type '{manager_type}'")
                return None

            try:
                # Use from_config pattern (consistent with BaseTool)
                return manager_class.from_config(v)
            except Exception as e:
                logger.warning(
                    f"Could not deserialize memory manager type '{manager_type}': "
                    f"{type(e).__name__}"
                )
                logger.debug(f"Memory manager deserialization detail: {e}")
                return None

        try:
            return handler(v)
        except Exception:
            return None

    @field_serializer("memory_manager")
    def serialize_memory_manager(self, value) -> dict[str, Any] | None:
        """Serialize memory_manager to dict with automatic type detection."""
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            return value.model_dump()
        return None

    @property
    def invocations_parameters(self) -> dict[str, Any]:
        """Default parameters for agent invocation as function."""
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
                self.parameters_string: self.invocations_parameters,
            },
            "required": [self.parameters_string],
            "description": self.description,
        }
        return schema

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

        for s in filter(lambda s: s.tool_autodiscovery, self.mcp_servers):
            # Remove existing MCPTools from this server
            new_tools = [
                t
                for t in self.tools
                if not (isinstance(t, MCPTool) and t.server_url == s.url and t.group == s.name)
            ]

            try:
                discovered_tools = await s.discover_tools()
                for tool_schema in discovered_tools:
                    mcp_tool = MCPTool(
                        server_url=s.url,
                        tool_name=tool_schema.name,
                        group=s.name,
                        use_oauth=s.use_oauth,
                        oauth_provider=s.oauth_provider,
                        headers=s.headers,
                        protocol_mode=s.protocol_mode,
                        metadata={
                            # Only what the server declares, no name fallback:
                            # consumers apply their own display fallback (e.g.
                            # prettifying the tool name).
                            **({"title": tool_schema.title} if tool_schema.title else {}),
                            **(
                                {"description": tool_schema.description}
                                if tool_schema.description
                                else {}
                            ),
                        },
                    )
                    new_tools.append(mcp_tool)
                    tasks.append(safe_tool_init(mcp_tool))

                self.tools = new_tools
            except Exception:
                get_logger().warning(f"Could not discover tools from MCP server {s.url}")

        await asyncio.gather(*tasks)

    async def execute(
        self,
        context: InteractionContext,
        *,
        skills: LoadedSkills | None = None,
        instructions: AgentInstructions | None = None,
    ) -> AsyncGenerator[tuple[BaseEvent, InteractionContext], None]:
        """Bidirectional wrapper: propagates asend() values from runner to _execute.

        ``skills`` and ``instructions`` are what the runner loaded for this agent
        at the top of the turn. ``None`` means nobody preloaded — a direct call
        outside a runner — and they are then loaded on demand.
        """
        func_name = getattr(self, "default_execute_agent_function", "_execute")
        # Forwarded only when there is something to forward, so a custom dispatch
        # target written before these arguments keeps working untouched. Same
        # idiom as DelegateAgent.execute uses for run_config.
        extra: dict[str, Any] = {}
        if skills is not None:
            extra["skills"] = skills
        if instructions is not None:
            extra["instructions"] = instructions
        inner = getattr(self, func_name)(context, **extra)
        sent: Any = None
        while True:
            try:
                event = await inner.asend(sent)
                sent = yield event, context
            except StopAsyncIteration:
                break

    async def _execute(
        self,
        context: InteractionContext,
        *,
        skills: LoadedSkills | None = None,
        instructions: AgentInstructions | None = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Bidirectional wrapper: propagates asend() values from execute() to the planner."""
        skills_payload = ""
        if self.skills is not None:
            try:
                skills_payload = await build_skills_payload(self, context, skills)
            except SkillRequiredError as exc:
                # A guardrail that could not load.
                yield BaseEvent(
                    event_type=ExecutionEventType.ERROR_OCCURRED,
                    author=self.name,
                    error_code="RequiredSkillUnavailable",
                    error_message=f"Cannot run this turn: {exc}",
                )
                return

        # Passed only when there is something to pass, so a planner written
        # before these arguments keeps working for every agent that declares
        # neither skills nor standing instructions.
        extra: dict[str, Any] = {"skills_payload": skills_payload} if skills_payload else {}
        instruction = await self._merged_instruction(instructions)
        if instruction is not None:
            extra["instruction"] = instruction
        inner = self.planner.execute(self, context, **extra)
        sent: Any = None
        while True:
            try:
                event = await inner.asend(sent)
                sent = yield event
            except StopAsyncIteration:
                break

    async def _merged_instruction(self, loaded: AgentInstructions | None) -> str | None:
        """``instruction`` with this turn's standing instructions merged in, or
        ``None`` when there are none and the planner should use ``instruction``.

        Into a local, never onto ``self.instruction``: the agent is shared across
        concurrent conversations, and a remote source may serve each user a
        different soul.
        """
        runtime = self.standing_instructions
        if runtime is None:
            return None
        if loaded is None:
            try:
                loaded = await runtime.load()
            except Exception as exc:
                get_logger().warning(
                    f"instructions: could not load for '{self.name}': {type(exc).__name__}"
                )
                get_logger().debug(f"instructions load detail for '{self.name}': {exc}")
                return None
        if not loaded:
            return None
        if not isinstance(self.instruction, str):
            # A coroutine cannot be concatenated, and awaiting it here would
            # consume it before whatever else reads the field.
            get_logger().warning(
                f"instructions: {self.name} builds its instruction asynchronously; "
                "standing instructions were not applied"
            )
            return None
        return runtime.merge(self.instruction, loaded)

    def get_tool_by_name(self, tool_name: str) -> FunctionTool | DelegateTool | None:

        matching_tools = [t for t in self.tools if t.name == tool_name]

        # if not found, try to check if it was included with a namespace. Using '.' as separator as per SEP-986
        # https://github.com/modelcontextprotocol/modelcontextprotocol/issues/986
        matching_tool: FunctionTool | DelegateTool | None = None
        if len(matching_tools) == 1:
            matching_tool = matching_tools[0]
        else:
            tool_name_parts = tool_name.split(".")
            tool_name_short = tool_name_parts[-1]
            tool_namespaces = ".".join(tool_name_parts[:-1])
            matching_tool = next(
                (t for t in self.tools if t.name == tool_name_short and t.group == tool_namespaces),
                None,
            )
        if matching_tool is None and tool_name == READ_TOOL_RESULT.name:
            # Offered to every agent by the tool-use context manager, never
            # listed in `tools`: see `domyn.agents.tools.read_tool_result`.
            return READ_TOOL_RESULT
        return matching_tool

    def get_sub_agent_by_name(self, agent_name: str) -> Agent | None:
        return next((agent for agent in self.sub_agents if agent.name == agent_name), None)

    def remove_agent_by_name(self, agent_name: str) -> None:
        """Remove a sub-agent by name."""
        agent = self.get_sub_agent_by_name(agent_name)
        if not agent:
            raise ValueError(f"Agent {agent_name} not found as sub agent of {self.name}")

        self.sub_agents.remove(agent)

    def get_agent_description(self) -> AgentDescription:
        return AgentDescription(
            name=self.name,
            description=self.description,
            invocation_json_schema=self.invocation_json_schema,
            invocation_parameters=self.invocations_parameters,
        )

    def __hash__(self):
        return self.name.__hash__()

    @classmethod
    def from_config(cls, config: dict[str, Any], save_in_registry: bool = False) -> Agent:
        """Create an Agent from config.

        Config keys:
          - name: name of the agent
          - description: description of the agent
          - instruction: system / agent instruction
          - llm_provider: provider config or instance (expects key 'type' for dict)
          - planner_strategy: planner instance, or dict with a 'type' key
                (react|tool_use|tool_use_thought). Omitted: tool_use_thought,
                the same default as constructing an Agent directly.
          - tools: list of tool specifications. Each item can be:
                * FunctionTool / StreamableFunctionTool instance
                * dict with at least: name, type (function|streamable_function) and path or callable
          - subagents: list of Agent instances OR dict configs (recursive)
          - hooks: list of hook specs (instances or dicts)
          - context_visibility: one of global|peers|standard
          - handoff_scope: one of explicit|direct_peers|global (default explicit)
          - max_iterations: override agent level iterations (optional)
          - memory_manager: dict with 'type' key, BaseMemoryManager instance, or None.
                Deserialization is handled by the field_validator (registry lookup + from_config).
          - builtin_tools: list of already-built BuiltinTools instances (the loader
                expands `tool_bundles` into these before calling here) — passed
                straight through, not built from this method.

        """
        # Defensive copy to avoid mutating caller data; translates the
        # pre-split `handoff_visibility` key from stored legacy configs.
        cfg = dict(normalize_legacy_visibility(config))

        required = ["name", "description", "instruction", "llm_provider"]
        missing = [k for k in required if k not in cfg]
        if missing:
            raise ValueError(f"Missing required agent config keys: {missing}")

        # ------------------------------------------------------------------
        # Build LLM provider
        # ------------------------------------------------------------------
        llm_provider_cfg = cfg.get("llm_provider")
        if isinstance(llm_provider_cfg, BaseLLMProvider):
            llm_provider = llm_provider_cfg
        elif isinstance(llm_provider_cfg, dict):
            llm_provider = LLMProviderFactory.from_config(
                llm_provider_cfg, save_in_registry=save_in_registry
            )
        else:
            raise ValueError("Invalid llm_provider specification")

        # ------------------------------------------------------------------
        # Planner strategy
        # ------------------------------------------------------------------
        planner_cfg = cfg.get("planner_strategy")
        if planner_cfg is None:
            planner = _default_planner_factory()
        elif isinstance(planner_cfg, BasePlannerStrategy):
            planner = planner_cfg
        elif isinstance(planner_cfg, dict):
            planner = PlannerStrategyFactory.from_config(
                planner_cfg, save_in_registry=save_in_registry
            )
        else:
            raise ValueError("Invalid planner_strategy specification")

        # ------------------------------------------------------------------
        # Hooks (agent level)
        # ------------------------------------------------------------------

        def _build_agent_hooks(
            items: list[Any],
        ) -> list[FunctionHook]:
            built: list[FunctionHook] = []
            for h in items:
                if isinstance(h, FunctionHook):
                    built.append(h)
                elif isinstance(h, dict):
                    built.append(HookFactory.from_config(h, save_in_registry=save_in_registry))
                else:
                    raise ValueError(f"Unsupported agent hook spec: {h}")
            return built

        hooks_cfg = cfg.get("hooks", []) or []
        hooks = _build_agent_hooks(hooks_cfg)

        # ------------------------------------------------------------------
        # Tools
        # ------------------------------------------------------------------

        tool_specs = cfg.get("tools", []) or []
        built_tools: list[FunctionTool | DelegateTool] = []
        for spec in tool_specs:
            if isinstance(spec, FunctionTool | StreamableFunctionTool | MCPTool):
                tool_obj = spec
            elif isinstance(spec, dict):
                tool_obj = Tool.from_config(spec, save_in_registry=save_in_registry)
            else:
                raise ValueError(f"Unsupported tool spec: {spec}")

            built_tools.append(tool_obj)

        # ------------------------------------------------------------------
        # Sub agents (recursive)
        # ------------------------------------------------------------------
        sub_agents_cfg = cfg.get("subagents") or cfg.get("sub_agents") or []
        built_sub_agents: list[Agent] = []
        for sub in sub_agents_cfg:
            if isinstance(sub, str):
                built_sub_agents.append(sub)
            elif isinstance(sub, BaseAgent):
                built_sub_agents.append(sub)  # type: ignore[arg-type]
            elif isinstance(sub, dict):
                built_sub_agents.append(cls.from_config(sub, save_in_registry=save_in_registry))
            else:
                raise ValueError(f"Unsupported subagent spec: {sub}")

        # ------------------------------------------------------------------
        # MCP Servers
        # ------------------------------------------------------------------
        mcp_servers_cfg = cfg.get("mcp_servers", []) or []
        built_mcp_servers: list[MCPServerStreamableHTTPConfig] = []
        for server_spec in mcp_servers_cfg:
            if isinstance(server_spec, MCPServerStreamableHTTPConfig):
                built_mcp_servers.append(server_spec)
            elif isinstance(server_spec, dict):
                built_mcp_servers.append(MCPServerStreamableHTTPConfig.from_dict(server_spec))
            else:
                raise ValueError(f"Unsupported MCP server spec: {server_spec}")

        # ------------------------------------------------------------------
        # Context visibility
        # ------------------------------------------------------------------
        context_visibility = cfg.get("context_visibility", ContextVisibility.STANDARD)
        if isinstance(context_visibility, str):
            context_visibility = ContextVisibility(context_visibility.lower())

        # ------------------------------------------------------------------
        # Handoff scope (EXECUTE)
        # ------------------------------------------------------------------
        handoff_scope = cfg.get("handoff_scope", HandoffScope.EXPLICIT)
        if isinstance(handoff_scope, str):
            handoff_scope = HandoffScope(handoff_scope.lower())

        agent = cls(
            id=cfg.get("id"),
            name=cfg["name"],
            description=cfg["description"],
            instruction=cfg["instruction"],
            llm_provider=llm_provider,
            planner=planner,
            tools=built_tools,
            sub_agents=built_sub_agents,
            hooks=hooks,
            mcp_servers=built_mcp_servers,
            context_visibility=context_visibility,  # type: ignore[arg-type]
            handoff_scope=handoff_scope,  # type: ignore[arg-type]
            max_iterations=cfg.get(
                "max_iterations",
                planner.max_iterations if hasattr(planner, "max_iterations") else 3,
            ),
            memory_manager=cfg.get("memory_manager"),
            skills=cfg.get("skills"),
            standing_instructions=cfg.get("standing_instructions"),
            decisions=cfg.get("decisions"),
            stream=cfg.get("stream", False),
            builtin_tools=cfg.get("builtin_tools", []),
        )
        return agent


Agent.model_rebuild()  # type: ignore
