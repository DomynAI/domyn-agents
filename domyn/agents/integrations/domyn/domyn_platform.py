"""Platform tool discovery for native domyn agents.

Fetches tool definitions from the Domyn platform and converts them to
``DelegateTool`` instances (or populates schemas on existing ones).

The definitions come from ``domyn.agents.integrations.domyn_platform_sdk``, the same source the
LangGraph integration (``domyn.agents.integrations.langgraph.domyn_platform``) uses.  This
module provides the equivalent for native domyn agents (``DomynAgentAdapter``).
"""

from typing import Any

from domyn.agents.agents import Agent
from domyn.agents.integrations.domyn_platform_sdk import (
    PlatformSettings,
    ToolCatalog,
    default_catalog,
)
from domyn.agents.logger import get_logger
from domyn.agents.tools import DelegateTool

logger = get_logger()


def _collect_delegate_tools_with_owners(agent: Agent) -> list[tuple[Agent, DelegateTool]]:
    """Walk agent tree, return (owner, tool) pairs for every DelegateTool without a wrapped callable."""
    pairs: list[tuple[Agent, DelegateTool]] = []
    visited: set[str] = set()

    def _walk(a: Agent) -> None:
        if a.name in visited:
            return
        visited.add(a.name)
        for tool in getattr(a, "tools", []):
            if isinstance(tool, DelegateTool) and tool.wrapped is None:
                pairs.append((a, tool))
        for sub in getattr(a, "sub_agents", []):
            if isinstance(sub, Agent):
                _walk(sub)

    _walk(agent)
    return pairs


def _collect_all_tool_names(agent: Agent) -> set[str]:
    """Walk agent tree, return every tool name present anywhere (any tool type)."""
    names: set[str] = set()
    visited: set[str] = set()

    def _walk(a: Agent) -> None:
        if a.name in visited:
            return
        visited.add(a.name)
        for tool in getattr(a, "tools", []):
            name = getattr(tool, "name", None)
            if name:
                names.add(name)
        for sub in getattr(a, "sub_agents", []):
            if isinstance(sub, Agent):
                _walk(sub)

    _walk(agent)
    return names


# Maps the platform's Python-style type strings to JSON Schema types.
_JSON_TYPE_MAP: dict[str, str] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
}


def _params_to_schema(
    parameters: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Convert platform parameter specs to (tool_parameters, required_params)."""
    tool_parameters: dict[str, Any] = {}
    required_params: list[str] = []

    for p in parameters:
        name = p["name"]
        json_type = _JSON_TYPE_MAP.get(p.get("type", "str"), "string")
        prop: dict[str, Any] = {"type": json_type}
        if p.get("description"):
            prop["description"] = p["description"]
        if "default" in p:
            prop["default"] = p["default"]
        tool_parameters[name] = prop

        has_default = "default" in p
        is_required = p.get("is_required", True)
        if not has_default and is_required is not False:
            required_params.append(name)

    return tool_parameters, required_params


async def fetch_delegate_tools(
    api_key: str | None = None,
    space_id: str | None = None,
    channel_id: str | None = None,
    base_url: str | None = None,
    configuration_id: str | None = None,
    catalog: ToolCatalog | None = None,
) -> list[DelegateTool]:
    """Fetch platform tool definitions and return them as ``DelegateTool`` instances.

    All arguments fall back to the corresponding ``DOMYN_*`` environment variables
    when not supplied explicitly. ``catalog`` replaces the platform SDK as the source of
    the definitions.
    """
    settings = PlatformSettings.from_env(api_key, space_id, channel_id, base_url, configuration_id)

    if not all([settings.api_key, settings.space_id, settings.channel_id, settings.base_url]):
        raise ValueError(
            "api_key, space_id, channel_id, and base_url are all required "
            "(set them explicitly or via DOMYN_API_KEY / DOMYN_SPACE_ID / "
            "DOMYN_CHANNEL_ID / DOMYN_BASE_URL env vars)"
        )

    tool_defs = await (catalog or default_catalog(settings)).list_tools()

    tools: list[DelegateTool] = []
    for td in tool_defs:
        name = td.get("name")
        if not name:
            continue
        params = td.get("parameters") or []
        tool_parameters, required_params = _params_to_schema(params)
        tools.append(
            DelegateTool(
                tool_name=name,
                tool_description=td.get("description", ""),
                tool_parameters=tool_parameters,
                required_params=required_params,
            )
        )

    logger.info(
        f"Fetched {len(tools)} platform tool definition(s)",
        payload={"tool_names": [t.tool_name for t in tools]},
    )
    return tools


async def populate_delegate_tool_schemas(
    root: Agent,
    api_key: str | None = None,
    space_id: str | None = None,
    channel_id: str | None = None,
    base_url: str | None = None,
    configuration_id: str | None = None,
    attach_new_tools: bool = True,
    drop_on_fetch_failure: bool = True,
    catalog: ToolCatalog | None = None,
) -> None:
    """Verify delegate tools against the platform and drop any that cannot be fetched.

    Every DelegateTool in the agent tree must be confirmed on the platform or it is
    removed from its owning agent's tool list.  Tools are dropped regardless of whether
    they already carry a local schema.  For tools that are found, the schema is
    populated only when the tool does not already have one.  Called once by
    ``_build_adapter()`` (``cli/commands/serve/command.py``) before a worker's
    event loop starts, and again on each tick of its optional periodic refresh
    (``DomynAgentAdapter.refresh_delegate_tools``).

    When ``attach_new_tools`` is true (the default), every platform tool not already
    present anywhere in the agent tree (by name, any tool type) is additionally
    attached to ``root`` as a new ``DelegateTool`` — this is what lets ``domyn
    serve``/``domyn expose`` expose the platform's full tool catalog without each
    tool being hand-declared. Pass ``attach_new_tools=False`` to keep the narrower
    schema-verification-only behavior.

    When ``drop_on_fetch_failure`` is true (the default), a failed fetch drops every
    existing delegate tool — the right fail-closed behavior for the one-time startup
    call. Pass ``drop_on_fetch_failure=False`` for a periodic refresh against an
    already-running worker: a transient network blip must not wipe out every
    platform tool from an otherwise-healthy agent, so the existing tool set is left
    untouched and the next scheduled attempt simply tries again.
    """
    pairs = _collect_delegate_tools_with_owners(root)
    if not pairs and not attach_new_tools:
        logger.debug("No DelegateTool instances found in agent tree; nothing to fetch.")
        return

    try:
        fetched = await fetch_delegate_tools(
            api_key=api_key,
            space_id=space_id,
            channel_id=channel_id,
            base_url=base_url,
            configuration_id=configuration_id,
            catalog=catalog,
        )
    except Exception as exc:
        if drop_on_fetch_failure:
            logger.warning(
                f"Failed to fetch platform tool schemas: {type(exc).__name__} — "
                f"dropping all {len(pairs)} delegate tool(s)."
            )
            for owner, tool in pairs:
                owner.tools = [t for t in owner.tools if t is not tool]
        else:
            logger.warning(
                f"Failed to refresh platform tool schemas: {type(exc).__name__} — "
                f"keeping existing {len(pairs)} delegate tool(s)."
            )
        logger.debug(f"Platform tool schema fetch detail: {exc}")
        return

    fetched_by_name = {t.tool_name: t for t in fetched}
    for owner, tool in pairs:
        fetched_tool = fetched_by_name.get(tool.tool_name)
        if fetched_tool is None:
            logger.warning(
                f"DelegateTool '{tool.tool_name}' was not found on the platform; dropping it."
            )
            owner.tools = [t for t in owner.tools if t is not tool]
            continue
        tool.tool_parameters = fetched_tool.tool_parameters
        tool.required_params = fetched_tool.required_params
        if fetched_tool.tool_description:
            tool.tool_description = fetched_tool.tool_description
        logger.info(f"Populated schema for DelegateTool '{tool.tool_name}' from platform.")

    if attach_new_tools:
        existing_names = _collect_all_tool_names(root)
        new_tools = [t for t in fetched if t.tool_name not in existing_names]
        if new_tools:
            root.tools = [*root.tools, *new_tools]
            logger.info(
                f"Attached {len(new_tools)} new platform tool(s) to '{root.name}'",
                payload={"tool_names": [t.tool_name for t in new_tools]},
            )
