"""MCP tool wrapper that integrates remote MCP tools with Domyn agents.

This module provides MCPTool, which wraps MCP server tools using fastmcp.Client,
allowing them to be used seamlessly within Domyn agents with full hook support.
"""

import asyncio
import contextlib
import hashlib
import inspect
import uuid
from collections.abc import AsyncGenerator, Callable, Coroutine
from datetime import timedelta
from typing import Any, Literal, cast

from cashews import cache
from fastmcp import Client
from fastmcp.client import StreamableHttpTransport
from fastmcp.exceptions import ClientError, ToolError, ValidationError
from mcp.client.auth import OAuthClientProvider
from mcp.types import TextContent, Tool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from pydantic.json_schema import SkipJsonSchema

from domyn.agents.core import ApprovalType, VisibilityScope
from domyn.agents.hooks import HookPhase
from domyn.agents.logger import get_logger
from domyn.agents.tools.base import (
    _MISSING,
    FunctionDeclaration,
    Param,
    ToolResult,
    UpdateToolResult,
    _parse_napoleon_docstring,
)
from domyn.agents.tools.function_tool import FunctionTool
from domyn.agents.tools.mcp.oauth_providers import (
    DEFAULT_OAUTH_PROVIDER_CONFIG,
    build_oauth_provider,
    eagerly_discover_oauth_metadata,
)
from domyn.agents.tools.schema import (
    as_tool_parameters,
    inline_schema_refs,
    json_type_to_python_type,
)
from domyn.agents.tools.streamable_function_tool import StreamableToolResult
from domyn.agents.tracing import inject_trace_headers

cache.setup("mem://")


class MCPAuthRedirectRequiredError(Exception):
    """Raise this exception in your oauth provider redirect handler to indicate that the user needs to re-authenticate with the MCP server."""

    def __init__(self, url: str) -> None:
        super().__init__(url)
        self.url = url


def _is_oauth_redirect_error(e: Exception) -> bool:
    if isinstance(e, MCPAuthRedirectRequiredError):
        return True
    return isinstance(e.__cause__, MCPAuthRedirectRequiredError)


_LIST_TOOLS_FAILURE_SENTINEL = "__mcp_list_tools_failure__"
_LIST_TOOLS_SUCCESS_TTL = timedelta(seconds=5)
_LIST_TOOLS_FAILURE_TTL = timedelta(seconds=5)


async def _compute_oauth_cache_key(oauth_provider: OAuthClientProvider) -> str:
    tokens = await oauth_provider.context.storage.get_tokens()
    token_str = tokens.access_token if tokens else ""
    metadata_str = oauth_provider.context.client_metadata.model_dump_json()
    return hashlib.sha256(f"{metadata_str}:{token_str}".encode()).hexdigest()[:16]


class MCPCredentialsCheckResult(BaseModel):
    """Result of an MCPTool authentication attempt."""

    credentials_valid: bool = Field(
        description="Whether authentication credentials check succeeded"
    )
    tool_uid: str | None = Field(description="Unique identifier of the tool")
    tool_name: str = Field(description="Name of the tool on the MCP server")
    server_url: str = Field(description="URL of the MCP server endpoint")


class MCPTool(FunctionTool):
    """Wrapper for MCP (Model Context Protocol) remote tools.

    This class extends FunctionTool to provide integration with remote MCP servers
    using the fastmcp library. It handles tool discovery, schema conversion, and
    execution via the MCP protocol over HTTP/SSE transport.
    """

    type: Literal["mcp"] = "mcp"  # type: ignore
    model_config = ConfigDict(arbitrary_types_allowed=True)
    server_url: str = Field(..., description="URL of the MCP server endpoint")
    tool_name: str = Field(..., description="Name of the tool on the MCP server")
    display_name: str | None = Field(
        default=None,
        description="Optional override for the name shown to the model. Falls back to "
        "tool_name when unset. Distinct from tool_name so renaming for the model's "
        "benefit (e.g. disambiguating tools with the same name across servers) never "
        "changes which tool is actually invoked on the MCP server.",
    )
    visibility: VisibilityScope = VisibilityScope.ISOLATED
    timeout: float = Field(
        0,
        description="Maximum time in seconds to wait for tool execution. 0 means no limit, "
        "leaving the runner's own ceiling as the only backstop.",
    )

    protocol_mode: Literal["legacy", "auto"] = Field(
        default="legacy",
        description="How the client negotiates the MCP protocol at connect time. See "
        "MCPServerStreamableHTTPConfig.protocol_mode.",
    )

    # OAuth configuration
    use_oauth: bool = Field(
        default=False,
        description="Whether to use OAuth authentication",
    )
    oauth_provider: SkipJsonSchema[OAuthClientProvider] | None = Field(
        default=None,
        description="OAuth client provider for authentication. Scopes, "
        "client_name, and a pre-registered client_id/client_secret are "
        "configured on the oauth_providers: entry this comes from, not here.",
    )

    # Additional headers for non-OAuth authentication
    headers: dict[str, str | Callable[[], Coroutine[Any, Any, str]]] | None = Field(
        default=None,
        description="HTTP headers for authentication (e.g., {'Authorization': 'Bearer token'})",
    )

    # Override func to be optional for MCP tools
    func: Any = None

    # Private attributes
    _client: Client | None = PrivateAttr(default=None)  # type: ignore
    _tool_schema: dict[str, Any] | None = PrivateAttr(default=None)
    _timeout_ping: float = PrivateAttr(
        default=5.0
    )  # Interval for healthcheck pings during execution

    @staticmethod
    async def _cached_list_tools(
        server_url: str, client: "Client", oauth_provider: OAuthClientProvider | None = None
    ) -> list[Tool]:
        suffix = f":{await _compute_oauth_cache_key(oauth_provider)}" if oauth_provider else ""
        key = f"mcp:list_tools:{server_url}{suffix}"
        cached = await cache.get(key)
        if cached is not None:
            if cached == _LIST_TOOLS_FAILURE_SENTINEL:
                raise ClientError(f"MCP server {server_url}: list_tools recently failed")
            return cast(list[Tool], cached)
        async with cache.lock(f"{key}:lock", expire=30):
            cached = await cache.get(key)
            if cached is not None:
                if cached == _LIST_TOOLS_FAILURE_SENTINEL:
                    raise ClientError(f"MCP server {server_url}: list_tools recently failed")
                return cast(list[Tool], cached)
            try:
                async with client as c:
                    tools = await c.list_tools()
                await cache.set(key, tools, expire=_LIST_TOOLS_SUCCESS_TTL)
                return tools
            except Exception:
                await cache.set(key, _LIST_TOOLS_FAILURE_SENTINEL, expire=_LIST_TOOLS_FAILURE_TTL)
                raise

    async def _build_client(self) -> Client:
        """Create an MCP client with the current authentication configuration."""
        # Create OAuth config if needed
        auth = None
        if self.use_oauth:
            # Normally already resolved (and possibly shared with other
            # tools) by ConfigurationV1.from_dict() — this fallback only
            # fires when an MCPTool is built directly in Python with
            # use_oauth=True and no oauth_provider=, bypassing the loader.
            provider = self.oauth_provider or build_oauth_provider(
                DEFAULT_OAUTH_PROVIDER_CONFIG, self.server_url
            )
            if provider.context.oauth_metadata is None:
                # Unset means this instance hasn't been through discovery yet
                # (fresh, or shared but not yet used) — see
                # eagerly_discover_oauth_metadata's docstring for why this
                # must happen before any request, not just on a 401.
                await eagerly_discover_oauth_metadata(provider, self.server_url)
            auth = provider

        if self.headers:
            self.headers = {
                k: str(await v()) if inspect.iscoroutinefunction(v) else v
                for k, v in self.headers.items()
            }

        transport = StreamableHttpTransport(
            url=self.server_url,
            headers=cast(dict[str, str], self.headers),
            auth=auth,
        )

        return Client(transport, timeout=self.timeout or None, mode=self.protocol_mode)

    def model_post_init(self, __context: Any) -> None:
        """Initialize after Pydantic validation.

        Skip FunctionTool's model_post_init since we don't have a callable function.
        """
        logger = get_logger()
        # Don't call super().model_post_init() - we manage our own callable
        self._actual_callable = None

        if self.uid is None:
            object.__setattr__(self, "uid", f"{self.server_url}.{self.tool_name}")

        if self.visibility in (VisibilityScope.SCOPED, VisibilityScope.TRUSTED) and not self.group:
            logger.warning(
                f"MCPTool '{self.tool_name}' has {self.visibility.value} visibility but no group set. "
                "State will be keyed by tool_name, equivalent to ISOLATED.",
            )

    async def initialize(self) -> Client:  # type: ignore
        """Ensure fastmcp client is created and connected.

        Returns:
            Client instance

        Raises:
            RuntimeError: If connection fails
        """
        if self._client is None:
            self._client = await self._build_client()

        await self.get_tool_schema()

        if self._tool_schema is None:
            raise RuntimeError(f"Tool '{self.tool_name}' not found on MCP server {self.server_url}")

        return self._client

    async def check_auth_credentials(self) -> MCPCredentialsCheckResult:
        """Base implementation: always reports valid credentials; override to check for real."""

        return MCPCredentialsCheckResult(
            credentials_valid=True,
            tool_uid=self.uid,
            tool_name=self.tool_name,
            server_url=self.server_url,
        )

    def format_auth_credentials_result_failed_execution(self) -> MCPCredentialsCheckResult:
        """Returns an MCPCredentialsCheckResult indicating failed execution due to authentication issues during tool execution.
        Used to retrieve extra metadata from extended functions.
        """
        return MCPCredentialsCheckResult(
            credentials_valid=False,
            tool_uid=self.uid,
            tool_name=self.tool_name,
            server_url=self.server_url,
        )

    async def get_tool_schema(self) -> None:
        logger = get_logger()
        if self._tool_schema is None:
            tools = await self._cached_list_tools(
                server_url=self.server_url, client=self._client, oauth_provider=self.oauth_provider
            )
            for tool in tools:
                if tool.name == self.tool_name:
                    # Server-provided human-readable title (MCP spec 2025-06-18);
                    # an explicit title from config takes precedence.
                    if tool.title and isinstance(self.metadata, dict):
                        self.metadata.setdefault("title", tool.title)
                    napoleon = (
                        _parse_napoleon_docstring(tool.description)
                        if isinstance(tool.description, str)
                        else None
                    )
                    server_schema = tool.input_schema or {}
                    # A self-contained copy: `tool.input_schema` is shared through the
                    # `_cached_list_tools` cache, and the enrichment below writes into it.
                    input_schema = inline_schema_refs(
                        server_schema,
                        on_unresolved=lambda ref: logger.warning(
                            f"MCP tool '{self.tool_name}' on {self.server_url}: schema "
                            f"reference {ref} does not resolve — dropped from the schema "
                            f"sent to the model"
                        ),
                        on_truncated=lambda cut: logger.warning(
                            f"MCP tool '{self.tool_name}' on {self.server_url}: schema too "
                            f"large to inline in full — {cut} references left as "
                            f"placeholders"
                        ),
                    )
                    napoleon_args_by_name = {}
                    if napoleon and napoleon.get("args"):
                        napoleon_args_by_name = {
                            arg["name"]: arg for arg in napoleon["args"] if "name" in arg
                        }
                    # Asked of the server's schema: inlining gives a bare `$ref` parameter
                    # its definition's description, which `Args:` should still override.
                    server_properties = server_schema.get("properties", {})
                    for param_name, param_schema in input_schema.get("properties", {}).items():
                        if "description" in server_properties.get(param_name, {}):
                            continue
                        napoleon_arg = napoleon_args_by_name.get(param_name)
                        if napoleon_arg and "description" in napoleon_arg:
                            param_schema["description"] = napoleon_arg["description"]
                        else:
                            param_schema.setdefault("description", f"Parameter {param_name}")
                    self._tool_schema = input_schema
                    napoleon_desc = napoleon.get("description", None) if napoleon else None
                    self._tool_schema["description"] = napoleon_desc or tool.description
                    break

    @property
    def invocation_json_schema(self) -> dict[str, Any]:
        if self._tool_schema is None:
            return {
                "properties": {
                    self.name_string: {"const": self.name, "type": "string"},
                    self.parameters_string: {},
                },
                "required": [self.parameters_string],
                "description": self.name,
            }

        return {
            "properties": {
                self.name_string: {"const": self.name, "type": "string"},
                self.parameters_string: {
                    "properties": self._tool_schema.get("properties", {}),
                    "required": self._tool_schema.get("required", []),
                },
            },
            "required": [self.parameters_string],
            "description": self._tool_schema.get("description", ""),
        }

    @property
    def tool_definition(self) -> dict[str, Any]:
        """Native tool definition carrying the server's JSON Schema as is.

        ``FunctionTool`` rebuilds it from ``declaration``, whose ``Param`` types flatten
        nested structure away; ``declaration`` stays for the consumers that want ``Param``s.
        """
        if self._tool_schema is None:
            return super().tool_definition

        # Drop the tool's description, parked on the schema by `get_tool_schema`.
        parameters = as_tool_parameters(
            {k: v for k, v in self._tool_schema.items() if k != "description"}
        )

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    @property
    def declaration(self) -> FunctionDeclaration:
        """Convert MCP tool schema to FunctionDeclaration.

        Returns:
            FunctionDeclaration compatible with Domyn's tool system
        """
        if self._tool_schema is None:
            # Return minimal declaration if schema not yet discovered
            return FunctionDeclaration(
                name=self.name,
                description=self.name,
                parameters=[],
            )

        # Extract input schema
        input_schema = self._tool_schema
        properties = input_schema.get("properties", {})
        required = set(input_schema.get("required", []))

        parameters = []
        for param_name, param_schema in properties.items():
            param_desc = param_schema.get("description", f"Parameter {param_name}")
            py_type = json_type_to_python_type(param_schema.get("type"))

            has_default = "default" in param_schema
            parameters.append(
                Param(
                    name=param_name,
                    type=py_type,
                    description=param_desc,
                    is_required=param_name in required,
                    default=param_schema["default"] if has_default else _MISSING,
                )
            )

        description = self._tool_schema.get("description", "")
        if not description:
            description = self._tool_schema.get("title", self.tool_name)

        return FunctionDeclaration(
            name=self.name,
            description=description,
            parameters=parameters,
        )

    @property
    def name(self) -> str:
        """Get the name shown to the model (display_name if set, else tool_name)."""
        return self.display_name or self.tool_name

    @name.setter
    def name(self, value: str) -> None:
        """Set the name shown to the model. Does not affect tool_name, the identifier
        used to invoke the tool on the MCP server."""
        object.__setattr__(self, "display_name", value)

    @staticmethod
    def convert_content_to_string(result_data: Any) -> str:
        return "\n".join(result_data) if result_data else ""

    @staticmethod
    def _unwrap_fastmcp_result(data: Any, metadata: dict[str, Any]) -> Any:
        """Remove FastMCP's transport wrapper from wrapped tool results.

        FastMCP may expose a tool return value as structured content shaped
        like ``{"result": <actual payload>}`` and mark it with
        ``meta.fastmcp.wrap_result``.  Downstream agents should reason about
        the logical tool payload, not the transport envelope.
        """
        fastmcp_meta = metadata.get("fastmcp")
        wrap_result = isinstance(fastmcp_meta, dict) and fastmcp_meta.get("wrap_result") is True
        if wrap_result and isinstance(data, dict) and set(data) == {"result"}:
            return data["result"]
        return data

    async def execute(  # type: ignore[override]
        self, **kwargs: Any
    ) -> AsyncGenerator[StreamableToolResult | UpdateToolResult | ToolResult, None]:
        """Execute the MCP tool remotely.

        Yields intermediate objects for each MCP progress notification received during the call,
        then yields the final result once the remote tool completes.

        Args:
            **kwargs: Tool arguments matching the tool's input schema

        Yields:
            UpdateToolResult: zero or more update notifications
            ToolResult: the final result of the tool call

        Errors are returned as ToolResult(success=False, ...), not raised.
        """
        # retrieve logger
        logger = get_logger()
        result: ToolResult

        try:
            # Filter kwargs to only include expected parameters
            _state = kwargs.get("state", {})
            _run_config = kwargs.get("run_config", {})

            # State scope:
            #   ISOLATED → per tool (tool_name key), strictest sandbox.
            #   SCOPED   → per group (MCP server URL key), sibling tools share state.
            #   TRUSTED  → per group (like SCOPED) + run_config.
            #   GLOBAL   → full state dict + run_config.
            if self.visibility == VisibilityScope.ISOLATED:
                filtered_state = _state.get(self.tool_name, {})
            elif self.visibility in (VisibilityScope.SCOPED, VisibilityScope.TRUSTED):
                state_key = self.group or self.tool_name
                filtered_state = _state.get(state_key, {})
            else:
                filtered_state = _state

            # run_config: TRUSTED and GLOBAL receive it (contains PII: user_id, space_id, etc.).
            # The gate lives on VisibilityScope so MCP tools and delegated agents share it.
            # TODO: support an optional meta_fields: list[str] allowlist for granular control.
            # call_metadata is never gated: it's an explicit per-call channel (e.g. an
            # auth token minted by a PRE_TOOL_CALL hook), not PII forwarded by default —
            # merged last so it can't be silently shadowed by run_config/state.
            call_metadata = kwargs.get("call_metadata") or {}
            meta = (
                self.visibility.filter_run_config(_run_config)
                | {"state": filtered_state}
                | call_metadata
            )

            meta = inject_trace_headers(meta)  # Add tracing headers if enabled

            if self._tool_schema:
                expected_params = set(self._tool_schema.get("properties", {}).keys())
                kwargs = {k: v for k, v in kwargs.items() if k in expected_params}

            logger.debug(f"Executing MCP tool {self.tool_name}", payload={"args": kwargs})

            # Unique ID shared by all progress notifications for this call.
            call_id = str(uuid.uuid4())

            # Queue used to put progress responses into the async generator.
            progress_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

            async def _progress_handler(
                progress: float, total: float | None, message: str | None
            ) -> None:
                """Handle progress updates from the MCP tool call."""
                await progress_queue.put(
                    {"call_id": call_id, "progress": progress, "total": total, "message": message}
                )

            # Shared result container — TaskGroup doesn't return values directly.
            call_result: list[Any] = []
            # Event to signal that call_task is done (either success or failure) so that other tasks can clean up.
            # Also used by _healthcheck_loop.
            call_done = asyncio.Event()

            async def _call_tool() -> None:
                """Run the remote tool call and store the result."""
                try:
                    response = await self._client.call_tool(
                        name=self.tool_name,
                        arguments=kwargs,
                        meta=meta,
                        progress_handler=_progress_handler,
                    )
                    call_result.append(response)
                finally:
                    # Also on failure or cancellation: the health check stops on this
                    # flag even when its own cancellation does not reach it.
                    call_done.set()

            async def _healthcheck_loop() -> None:
                """Ping the MCP server every 5 s while the remote call is in-flight.

                Uses a single dedicated client connection (not shared with call_task),
                opened only when the first ping is due, so a call that finishes sooner
                costs no connection, and stops as soon as the call finishes.
                """
                try:
                    async with contextlib.AsyncExitStack() as stack:
                        probe_client: Client | None = None  # type: ignore[type-arg]
                        while True:
                            # Woken by the call finishing, not by a sleep the TaskGroup
                            # would otherwise sit out in full. Not asyncio.wait_for: on
                            # Python 3.11 it drops a cancellation that arrives as the wait
                            # completes, and this loop then ran on forever with the
                            # TaskGroup waiting for it.
                            try:
                                async with asyncio.timeout(self._timeout_ping):
                                    await call_done.wait()
                                return
                            except TimeoutError:
                                pass

                            # A call cancelled as the wait ran out sets the flag a turn
                            # later: let it, or a ping goes out for a call that is over.
                            await asyncio.sleep(0)
                            if call_done.is_set():
                                return

                            if probe_client is None:
                                probe_client = await stack.enter_async_context(
                                    await self._build_client()
                                )
                            async with asyncio.timeout(60.0):
                                await probe_client.ping()
                except Exception as exc:
                    raise RuntimeError(
                        f"MCP server at {self.server_url} stopped responding during tool execution"
                    ) from exc

            async def _run_call_and_healthcheck() -> None:
                """Run the call and its health check in one TaskGroup."""
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(_call_tool())
                    tg.create_task(_healthcheck_loop())

            async with self._client:
                # The runner resumes this generator from a new task on every step, so the
                # TaskGroup gets a task of its own that outlives each step and sees its errors.
                group_task = asyncio.create_task(_run_call_and_healthcheck())
                next_item: asyncio.Future[dict[str, Any]] | None = None
                try:
                    while True:
                        next_item = asyncio.ensure_future(progress_queue.get())
                        await asyncio.wait(
                            {next_item, group_task}, return_when=asyncio.FIRST_COMPLETED
                        )
                        if not next_item.done():
                            next_item.cancel()
                            break
                        item = next_item.result()
                        yield UpdateToolResult(
                            data={
                                "call_id": item.get("call_id"),
                                "progress": item.get("progress"),
                                "total": item.get("total", None),
                                "message": item.get("message", None),
                            }
                        )
                except BaseException:
                    if next_item is not None:
                        next_item.cancel()
                    group_task.cancel()
                    await asyncio.gather(group_task, return_exceptions=True)
                    raise

                try:
                    await group_task
                except* RuntimeError as eg:
                    # Unwrap the first RuntimeError from the ExceptionGroup.
                    raise eg.exceptions[0] from eg.exceptions[0].__cause__
                except* ToolError as eg:
                    # Unwrap ToolError so the outer except handler catches it correctly.
                    raise eg.exceptions[0] from eg.exceptions[0].__cause__
                except* ClientError as eg:
                    # Unwrap ClientError so the outer except handler catches it correctly.
                    raise eg.exceptions[0] from eg.exceptions[0].__cause__
                except* ValidationError as eg:
                    # Unwrap ValidationError so the outer except handler catches it correctly.
                    raise eg.exceptions[0] from eg.exceptions[0].__cause__
                except* MCPAuthRedirectRequiredError as eg:
                    raise eg.exceptions[0] from None

            # Flush any notifications that arrived just before the tasks finished.
            while not progress_queue.empty():
                item = progress_queue.get_nowait()
                yield UpdateToolResult(
                    data={
                        "call_id": item.get("call_id"),
                        "progress": item.get("progress"),
                        "total": item.get("total", None),
                        "message": item.get("message", None),
                    }
                )

            # Retrieve the result stored by _call_tool.
            response = call_result[0]

        except ToolError as e:
            logger.error(f"Error executing MCP tool {self.tool_name}: {type(e).__name__}")
            logger.debug(f"Error executing MCP tool {self.tool_name}: {e}")
            result = ToolResult(success=False, error=str(e) + " - Is the MCP server running?")
        except ValidationError as e:
            logger.error(f"Validation error for MCP tool {self.tool_name}: {type(e).__name__}")
            logger.debug(f"Validation error for MCP tool {self.tool_name}: {e}")
            result = ToolResult(success=False, error=str(e))
        except ClientError as e:
            logger.error(
                f"Error with the Client during MCP tool execution {self.tool_name}: "
                f"{type(e).__name__}"
            )
            logger.debug(f"Error with the Client during MCP tool execution {self.tool_name}: {e}")
            result = ToolResult(success=False, error=str(e) + " - Is the MCP server running?")
        except Exception as e:
            if isinstance(e, ExceptionGroup):
                # Unwrap the first exception from the group for logging.
                e = e.exceptions[0]
            if _is_oauth_redirect_error(e):
                logger.warning(
                    f"MCP tool {self.tool_name}: OAuth token invalid or expired — re-authentication required"
                )
                result = ToolResult(
                    success=False,
                    error="Authentication required: OAuth token is invalid or expired. Please re-authenticate.",
                    error_code="401",
                )
            else:
                err_msg = str(e).strip() or type(e).__name__
                logger.error(f"Error executing MCP tool {self.tool_name}: {type(e).__name__}")
                logger.debug(f"Error executing MCP tool {self.tool_name}: {err_msg}")
                result = ToolResult(success=False, error=err_msg + " - Is the MCP server running?")
        else:
            # If no errors
            result_data: Any = []
            response_meta = dict(getattr(response, "meta", None) or {})

            if hasattr(response, "structured_content") and response.structured_content is not None:
                result_data = response.structured_content
            elif hasattr(response, "content"):
                for content in response.content:
                    # Extract text from TextContent objects
                    if isinstance(content, TextContent):
                        result_data.append(content.text)
                    else:
                        result_data.append(str(content))
                result_data = self.convert_content_to_string(result_data)
            elif isinstance(response, list):
                for content in response:
                    if isinstance(content, TextContent):
                        result_data.append(content.text)
                    else:
                        result_data.append(str(content))
                result_data = self.convert_content_to_string(result_data)
            else:
                # Fallback to string representation
                result_data.append(str(response))

            result_data = self._unwrap_fastmcp_result(result_data, response_meta)
            meta_state = response_meta.pop("state", {})

            mcp_metadata = {
                "mcp_server": self.server_url,
                "tool_name": self.tool_name,
            } | response_meta

            # If the server returned a ToolResult-shaped dict (or a legacy "result"
            # envelope) cast it directly and merge the MCP envelope so no MCP info is
            # lost. Unified with FunctionTool/StreamableFunctionTool. A state the tool
            # returns inside its result is folded into meta_state so both are scoped
            # together below (otherwise the scoped keys would clobber each other).
            coerced = ToolResult.from_result_dict(result_data, result_key_fallback=True)
            if coerced is not None and coerced.state:
                meta_state = meta_state | coerced.state

            # Write-back mirrors read scope:
            #   ISOLATED → nested under tool_name key
            #   SCOPED/TRUSTED → nested under group key
            #   GLOBAL   → merged at top level
            if not meta_state:
                state_addition = {}
            elif self.visibility == VisibilityScope.ISOLATED:
                state_addition = {self.tool_name: meta_state}
            elif self.visibility in (VisibilityScope.SCOPED, VisibilityScope.TRUSTED):
                state_addition = {(self.group or self.tool_name): meta_state}
            else:
                state_addition = meta_state

            merged_state = _state | state_addition

            if coerced is not None:
                result = coerced
                result.metadata = mcp_metadata | result.metadata
                result.state = merged_state
            else:
                result = ToolResult(
                    success=True,
                    data=result_data,
                    state=merged_state,
                    metadata=mcp_metadata,
                )

        yield result

    @classmethod
    def from_config(cls, config: dict[str, Any], save_in_registry: bool = False) -> "MCPTool":
        """Create MCPTool from configuration dict.

        Args:
            config: Configuration dictionary with keys:
                - server_url: URL of MCP server
                - tool_name: Name of tool on server
                - name: Optional override for tool name (defaults to tool_name)
                - use_oauth: Optional OAuth authentication flag
                - oauth_provider: Resolved OAuth provider instance (built by
                  ConfigurationV1.from_dict() from an oauth_providers: entry,
                  named or default — not meant to be set directly from raw
                  config)
                - headers: Optional HTTP headers dict
                - hooks: Optional list of hook configs or instances (trigger
                  pre_tool_call/post_tool_call)
                - visibility: Optional VisibilityScope (defaults to ISOLATED)
                - uid: Optional unique identifier
                - metadata: Optional metadata dict
                - approval_type: Optional approval type
            save_in_registry: Whether to save tool in registry

        Returns:
            MCPTool instance
        """
        from domyn.agents.hooks import HookFactory

        hooks = []
        for hook in config.get("hooks", []):
            if isinstance(hook, dict):
                if hook["trigger"] == HookPhase.PRE_TOOL_CALL:
                    hooks.append(
                        HookFactory.from_config(hook, trigger=HookPhase.PRE_TOOL_CALL)
                        if isinstance(hook, dict)
                        else hook
                    )
                elif hook["trigger"] == HookPhase.POST_TOOL_CALL:
                    hooks.append(
                        HookFactory.from_config(hook, trigger=HookPhase.POST_TOOL_CALL)
                        if isinstance(hook, dict)
                        else hook
                    )
                elif hook["trigger"] == HookPhase.PRE_TOOL_APPROVAL:
                    hooks.append(
                        HookFactory.from_config(hook, trigger=HookPhase.PRE_TOOL_APPROVAL)
                        if isinstance(hook, dict)
                        else hook
                    )
                else:
                    raise ValueError(f"Unsupported hook trigger: {hook['trigger']}")
            else:
                hooks.append(hook)

        # GLOBAL controls run_config (PII) access — set programmatically by the
        # host application (e.g. URL allowlist), not expected from user config.
        visibility = VisibilityScope(config.get("visibility", VisibilityScope.ISOLATED))

        kwargs = {
            "server_url": config["server_url"],
            "tool_name": config.get("name") or config["tool_name"],
            "use_oauth": config.get("use_oauth", False),
            "oauth_provider": config.get("oauth_provider"),
            "headers": config.get("headers"),
            "protocol_mode": config.get("protocol_mode", "legacy"),
            "hooks": hooks,
            "visibility": visibility,
            "uid": config.get("uid"),
            "metadata": config.get("metadata", {}),
        }

        if "approval_type" in config:
            kwargs["approval_type"] = ApprovalType(config["approval_type"])

        tool = cls(**kwargs)

        if save_in_registry:
            from domyn.agents.registry import get_tool_registry

            get_tool_registry().register_tool(tool)

        return tool

    async def disconnect(self) -> None:
        """Disconnect from the MCP server.

        This should be called when the tool is no longer needed to clean up resources.
        """
        if self._client:
            # FastMCP Client handles cleanup in async context manager
            # No explicit disconnect needed
            self._client = None
