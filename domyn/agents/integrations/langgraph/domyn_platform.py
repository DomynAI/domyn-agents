"""Platform tool integration: lets a langgraph runnable call tools hosted on the Domyn platform.

Tools fetched here look like ordinary langchain tools to the graph, but their
execution is delegated: ``_arun`` ships the call over the bound adapter's
transport as a TOOL_START BaseEvent and suspends until a matching TOOL_END /
TOOL_ERROR arrives back, keyed by ``ToolAction.call_id``.
"""

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr, create_model

from domyn.agents.integrations.domyn_platform_sdk import (
    PlatformSettings,
    ToolCatalog,
    default_catalog,
    run_sync,
)
from domyn.agents.logger import get_logger
from domyn.agents.tools.schema import python_type_str_to_python_type

logger = get_logger()

# Marker placed on every PlatformTool's ``tags`` list. The langchain callback
# handler reads this to suppress local trace emission for platform tools — the
# bound adapter already sends TOOL_START / TOOL_END for these calls itself, so
# the in-process callbacks would produce duplicates.
PLATFORM_TOOL_TAG = "domyn:platform_tool"


def _build_args_schema(tool_name: str, parameters: list[dict[str, Any]]) -> type[BaseModel]:
    """Build a Pydantic args model from a platform tool's parameter spec.

    Optionality is inferred from two keys (either may appear):
      - ``default``: when present, the field is optional with that value;
      - ``is_required``: when explicitly ``False``, the field is optional and
        defaults to ``None`` (the annotation becomes ``T | None``).

    Fields with neither key are required.
    """
    fields: dict[str, Any] = {}
    for p in parameters:
        name = p["name"]
        py_type = python_type_str_to_python_type(p.get("type", "str"))
        description = p.get("description", "")
        if "default" in p:
            fields[name] = (py_type, Field(default=p["default"], description=description))
        elif p.get("is_required") is False:
            fields[name] = (py_type | None, Field(default=None, description=description))
        else:
            fields[name] = (py_type, Field(..., description=description))
    return create_model(f"{tool_name}_args", **fields)


class PlatformRelay:
    """Bridge between PlatformTool calls and the bound adapter's transport.

    A user's module typically calls :func:`get_platform_tools` at import time
    (before the adapter exists), which binds tools to a shared default relay.
    The adapter (e.g. ``LangGraphAdapter``) later binds itself to that relay
    on startup via ``bind()``/``unbind()``; calls before binding raise.

    Any object exposing an async ``call_platform_tool(name, kwargs)`` method
    can be bound here (duck-typed on purpose, so this relay works with both
    ``LangGraphAdapter`` and any future adapter with the same contract).
    """

    def __init__(self) -> None:
        self._runtime: Any | None = None

    def bind(self, runtime: Any) -> None:
        self._runtime = runtime

    def unbind(self, runtime: Any) -> None:
        if self._runtime is runtime:
            self._runtime = None

    async def call_platform_tool(self, name: str, kwargs: dict[str, Any]) -> Any:
        if self._runtime is None:
            raise RuntimeError(f"PlatformRelay is not bound to a Runtime; cannot invoke {name!r}")
        return await self._runtime.call_platform_tool(name, kwargs)


_default_relay = PlatformRelay()


def get_default_relay() -> PlatformRelay:
    return _default_relay


class PlatformTool(BaseTool):
    name: str
    description: str
    args_schema: type[BaseModel]
    _relay: PlatformRelay = PrivateAttr()

    def __init__(self, relay: PlatformRelay, **kwargs: Any) -> None:
        tags = list(kwargs.pop("tags", None) or [])
        if PLATFORM_TOOL_TAG not in tags:
            tags.append(PLATFORM_TOOL_TAG)
        super().__init__(tags=tags, **kwargs)
        self._relay = relay

    def _run(self, **kwargs: Any) -> Any:
        raise NotImplementedError("Platform tools must be invoked asynchronously")

    async def _arun(self, **kwargs: Any) -> Any:
        # Optional params the LLM leaves unset are still present here (Pydantic
        # fills unfilled optional fields with their None default per
        # _build_args_schema), so without this filter every omitted optional
        # param would be sent over the wire as an explicit `null`. Platform/MCP
        # tool schemas generally treat "optional" as "may be omitted", not
        # "may be null", and reject the latter (e.g. a strict MCP server
        # rejecting `"cloudId": null` with "expected: string, received: null").
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return await self._relay.call_platform_tool(self.name, kwargs)


class PlatformToolRegistry:
    def __init__(self, tool_defs: list[dict[str, Any]], relay: PlatformRelay) -> None:
        self._registry: dict[str, PlatformTool] = {}
        for t in tool_defs:
            params = t.get("parameters")
            if params is None:
                logger.warning(
                    "platform tool %r has no 'parameters' in its definition — "
                    "skipping it (it will not be available to the agent). "
                    "Full definition: %r",
                    t.get("name"),
                    t,
                )
                continue
            schema = _build_args_schema(t["name"], params)
            self._registry[t["name"]] = PlatformTool(
                relay=relay,
                name=t["name"],
                description=t.get("description", ""),
                args_schema=schema,
            )

    def get_tool(self, name: str) -> PlatformTool:
        if name not in self._registry:
            raise KeyError(f"Platform tool {name!r} not found. Available: {list(self._registry)}")
        return self._registry[name]

    def list_tools(self) -> list[str]:
        return list(self._registry)


def _resolve_platform_args(
    api_key: str | None,
    space_id: str | None,
    channel_id: str | None,
    base_url: str | None,
    relay: PlatformRelay | None,
    configuration_id: str | None = None,
) -> tuple[PlatformSettings, PlatformRelay]:
    # when no configuration id is given, the platform falls back to the last-released config
    settings = PlatformSettings.from_env(api_key, space_id, channel_id, base_url, configuration_id)
    if not settings.api_key:
        raise RuntimeError("DOMYN_API_KEY is required to fetch platform tools")
    if not settings.space_id:
        raise RuntimeError("DOMYN_SPACE_ID is required to fetch platform tools")
    if not settings.channel_id:
        raise RuntimeError("DOMYN_CHANNEL_ID is required to fetch platform tools")
    return settings, relay or _default_relay


def get_platform_tools(
    api_key: str | None = None,
    space_id: str | None = None,
    channel_id: str | None = None,
    base_url: str | None = None,
    relay: PlatformRelay | None = None,
    configuration_id: str | None = None,
    catalog: ToolCatalog | None = None,
) -> PlatformToolRegistry:
    """Fetch the platform's tool catalog and return a registry of PlatformTool wrappers.

    Blocks until the fetch is done, also when called from inside a running event loop; use
    :func:`aget_platform_tools` from async code. ``catalog`` replaces the platform SDK as
    the source of the definitions.
    """
    settings, relay = _resolve_platform_args(
        api_key=api_key,
        space_id=space_id,
        channel_id=channel_id,
        base_url=base_url,
        relay=relay,
        configuration_id=configuration_id,
    )
    defs = run_sync((catalog or default_catalog(settings)).list_tools())
    registry = PlatformToolRegistry(defs, relay=relay)
    logger.info(
        f"loaded {len(registry.list_tools())} platform tools",
        payload={"tools": registry.list_tools()},
    )
    return registry


async def aget_platform_tools(
    api_key: str | None = None,
    space_id: str | None = None,
    channel_id: str | None = None,
    base_url: str | None = None,
    relay: PlatformRelay | None = None,
    configuration_id: str | None = None,
    catalog: ToolCatalog | None = None,
) -> PlatformToolRegistry:
    """Async version of :func:`get_platform_tools`.

    Same behavior, but doesn't block the event loop when called from inside
    async code (e.g. graph startup hooks).
    """
    settings, relay = _resolve_platform_args(
        api_key=api_key,
        space_id=space_id,
        channel_id=channel_id,
        base_url=base_url,
        relay=relay,
        configuration_id=configuration_id,
    )
    defs = await (catalog or default_catalog(settings)).list_tools()
    registry = PlatformToolRegistry(defs, relay=relay)
    logger.info(
        f"loaded {len(registry.list_tools())} platform tools",
        payload={"tools": registry.list_tools()},
    )
    return registry
