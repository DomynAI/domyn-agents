"""Named OAuth provider configurations for MCP tools and servers.

An ``oauth_providers:`` entry in a YAML config is a *template*, not a live
instance: unlike an LLM provider, an ``OAuth``/``OAuthClientProvider`` is
bound to one MCP server URL at construction (its token storage is keyed by
that URL), so it cannot be shared as-is across different servers the way an
LLM provider can be shared across agents. The
loader builds one live instance per ``(provider_name, server_url)`` pair
actually referenced — see ``OAuthProviderInstanceCache`` — so tools naming
the same provider *and* pointing at the same server genuinely share one
instance, while tools naming the same provider but different servers each
get their own, correctly-bound one.

Two ``type`` values exist: ``local`` (tokens persisted to disk via
``FileTreeStore``, under ``storage_dir``) and ``in_memory`` (tokens kept
only in-process, gone on restart — fastmcp's own ``OAuth`` default when no
``token_storage`` is given). Both are plain branches in one function rather
than a full Factory + registry, matching the reasoning in
``tools/builtin/backends.py``: an abstraction with only a couple of
implementations usually has the wrong seams. An unknown type raises rather
than silently falling back — same precedent as ``build_backend()``.

The local OAuth callback port (``callback_port``) defaults to
``DEFAULT_OAUTH_CALLBACK_PORT``, one fixed, known constant, rather than being
computed from ``server_url`` — so ``http://localhost:<port>/callback`` is
knowable and registrable with an external OAuth app before any config is
written. Because that means every provider/server shares the same port by
default, and MCP tool/server initialization runs concurrently
(``asyncio.gather``), a whole authorization round trip — opening the
browser tab and waiting for *its own* redirect — is serialized per port by
patching ``OAuthClientProvider._perform_authorization_code_grant`` (see the
patch below, and ``_lock_for_port``). Locking only the local server's bind
was tried first and found insufficient: ``_perform_authorization_code_grant``
opens the browser tab (``redirect_handler``) *before* waiting for the
callback (``callback_handler``), so with only the bind locked, multiple
tabs could still be open at once — and whichever one the user completed
first would hit whatever flow's local server happened to be listening at
that moment, not necessarily its own, producing a genuine
``state`` mismatch (observed live across three concurrently-authenticating
MCP servers) rather than the intended clean crash-prevention. Locking the
whole round trip means a second flow sharing the port doesn't even open its
tab until the first one has finished — first-time auth across servers
sharing the default port happens one at a time, not concurrently, which is
the necessary trade-off for a shared, pre-registrable port.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qsl, urlparse

import httpx2
import mcp.client.auth.oauth2 as _mcp_oauth2
import mcp.client.auth.utils as _mcp_oauth_utils
from fastmcp.client.auth import OAuth
from key_value.aio.stores.filetree import (
    FileTreeStore,
    FileTreeV1CollectionSanitizationStrategy,
    FileTreeV1KeySanitizationStrategy,
)
from mcp.client.auth import OAuthClientProvider
from mcp.client.auth.exceptions import OAuthTokenError
from mcp.client.auth.utils import (
    build_oauth_authorization_server_metadata_discovery_urls,
    create_oauth_metadata_request,
    handle_auth_metadata_response,
)
from mcp.client.auth.utils import (
    get_client_metadata_scopes as _original_get_client_metadata_scopes,
)
from mcp.shared._httpx_utils import redirect_note
from mcp.shared.auth import OAuthMetadata, OAuthToken, ProtectedResourceMetadata
from pydantic import BaseModel, Field
from pydantic import ValidationError as PydanticValidationError

from domyn.agents.logger import get_logger

DEFAULT_OAUTH_PROVIDER_NAME = "__default__"
"""Reserved name for the provider auto-injected when a tool/server sets
``use_oauth: true`` but names no ``oauth_provider:`` — never a valid name for
a user-declared entry (enforced by ``OAuthProviderConfig``)."""

DEFAULT_OAUTH_CALLBACK_PORT = 45678
"""Fixed local OAuth callback port used unless a provider overrides it.

A single known constant, not a value derived from server_url, so
``http://localhost:<port>/callback`` is knowable and registrable with an
OAuth app before any config is written. Shared across providers/servers by
default — see ``_lock_for_port`` / ``build_oauth_provider`` for why that's
safe rather than a race."""


class OAuthProviderConfig(BaseModel):
    """One ``oauth_providers:`` entry — configuration, not a live instance."""

    name: str
    type: Literal["local", "in_memory"] = "local"
    # Ignored for type="in_memory" — that variant never touches disk.
    storage_dir: str = str(Path.home() / ".domyn" / "mcp_oauth")
    scopes: list[str] = Field(default_factory=lambda: ["read"])
    client_name: str = "Domyn Agent"
    client_id: str | None = Field(
        default=None,
        description="Pre-registered OAuth client ID. Skips Dynamic Client "
        "Registration when set, using this static client instead.",
    )
    client_secret: str | None = Field(
        default=None,
        description="Secret for client_id, when the server requires one.",
    )
    callback_port: int = Field(
        default=DEFAULT_OAUTH_CALLBACK_PORT,
        description="Local OAuth callback port to register as "
        "http://localhost:<port>/callback with the OAuth app. Shared "
        "across providers by default; override per-provider only if that "
        "default is already in use on the host.",
    )

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> OAuthProviderConfig:
        """Build from a user-supplied ``oauth_providers:`` entry.

        Rejects the reserved default-provider name here, rather than in a
        validator — that name is only ever used for
        ``DEFAULT_OAUTH_PROVIDER_CONFIG`` below, constructed directly, not
        through this method.
        """
        if config.get("name") == DEFAULT_OAUTH_PROVIDER_NAME:
            raise ValueError(f"'{DEFAULT_OAUTH_PROVIDER_NAME}' is a reserved oauth provider name")
        return cls(**config)


DEFAULT_OAUTH_PROVIDER_CONFIG = OAuthProviderConfig(
    name=DEFAULT_OAUTH_PROVIDER_NAME, type="in_memory"
)


#: How long a discovery that found nothing is not asked again. Many servers publish
#: no metadata at their own address (the authorization server is named by the
#: protected-resource metadata, found on the first 401), so the 404 is the answer
#: and would otherwise come back with every client built: once per tool, once per call.
DISCOVERY_RETRY_SECONDS = 600.0
#: The same for a discovery that failed (timeout, connection refused): shorter,
#: since that says nothing about the server and may pass.
DISCOVERY_FAILURE_RETRY_SECONDS = 30.0

# Keyed by the address asked, not by provider: a tool built without an
# ``oauth_provider`` gets a new one for every client, which would never see the
# answer a previous one got. The address is all the empty answer depends on.
_discovery_retry_at: dict[str, float] = {}
_discovery_locks: dict[str, asyncio.Lock] = {}


def _discovery_key(server_url: str) -> str:
    parsed = urlparse(server_url)
    return f"{parsed.scheme}://{parsed.netloc}".lower()


async def eagerly_discover_oauth_metadata(provider: OAuthClientProvider, server_url: str) -> None:
    """Populate ``oauth_metadata`` before any request goes out.

    Typed against ``OAuthClientProvider`` (the base class), not ``OAuth``
    (fastmcp's subclass) — ``context.oauth_metadata`` lives on the shared
    base ``OAuthContext``, and callers pass whatever `MCPTool.oauth_provider`
    /`MCPServerStreamableHTTPConfig.oauth_provider` actually holds, which is
    itself typed as the base class.

    ``OAuthClientProvider._refresh_token()`` only uses the server's real
    (path-aware) token endpoint when ``context.oauth_metadata`` is already
    set; otherwise it guesses a path-stripped URL that 404s. Normally that
    field is only populated reactively, after a 401 — but the refresh
    shortcut runs *before* any request is sent, so a brand-new instance that
    loads a cached refresh_token from disk always guesses wrong. Fetching it
    proactively here, with the same public helpers the SDK uses internally,
    closes that gap.

    An address that gave nothing is not asked again for
    ``DISCOVERY_RETRY_SECONDS`` (``DISCOVERY_FAILURE_RETRY_SECONDS`` when the
    request failed), whichever provider asks, and callers that arrive meanwhile
    wait for the one request in flight instead of making their own.
    """
    key = _discovery_key(server_url)
    lock = _discovery_locks.setdefault(key, asyncio.Lock())
    async with lock:
        if provider.context.oauth_metadata is not None:
            return
        retry_at = _discovery_retry_at.get(key)
        if retry_at is not None and time.monotonic() < retry_at:
            return
        outcome = await _discover_oauth_metadata(provider, server_url)
        if provider.context.oauth_metadata is None:
            wait = (
                DISCOVERY_FAILURE_RETRY_SECONDS if outcome == "failed" else DISCOVERY_RETRY_SECONDS
            )
            _discovery_retry_at[key] = time.monotonic() + wait


async def _discover_oauth_metadata(
    provider: OAuthClientProvider, server_url: str
) -> Literal["found", "empty", "failed"]:
    logger = get_logger()
    try:
        # httpx2, not httpx: the requests below are built by the SDK, which
        # moved to httpx2 with mcp 2.
        async with httpx2.AsyncClient() as client:
            for url in build_oauth_authorization_server_metadata_discovery_urls(None, server_url):
                response = await client.send(create_oauth_metadata_request(url))
                ok, metadata = await handle_auth_metadata_response(response)
                if metadata is not None:
                    provider.context.oauth_metadata = metadata
                    return "found"
                if not ok:
                    return "empty"
    except Exception as exc:
        # Best-effort: if discovery fails, fall back to the SDK's own
        # (reactive, 401-triggered) behavior rather than blocking startup.
        logger.warning(f"eager OAuth metadata discovery failed: {type(exc).__name__}")
        logger.debug(f"eager OAuth metadata discovery detail: {exc}")
        return "failed"
    return "empty"


def build_oauth_provider(config: OAuthProviderConfig, server_url: str) -> OAuth:
    """Construct an ``OAuth`` instance bound to ``server_url`` from a template.

    Synchronous and network-free — the actual `.well-known` discovery is a
    separate, later, async step (``eagerly_discover_oauth_metadata``), kept
    apart so this can be called from ``ConfigurationV1.from_dict()``, which
    is itself synchronous.
    """
    token_storage = None
    if config.type == "local":
        storage_dir = Path(config.storage_dir).expanduser()
        storage_dir.mkdir(parents=True, exist_ok=True)
        token_storage = FileTreeStore(
            data_directory=str(storage_dir),
            # Unsanitized keys/collections are literal path segments — an
            # MCP server URL's "/" would otherwise be read as nested
            # directories that are never created.
            key_sanitization_strategy=FileTreeV1KeySanitizationStrategy(storage_dir),
            collection_sanitization_strategy=FileTreeV1CollectionSanitizationStrategy(storage_dir),
        )
    # in_memory: token_storage stays None — OAuth's own default
    # (fastmcp's OAuth._bind: `self._token_storage or MemoryStore()`) is an
    # in-process store, never written to disk.

    provider = OAuth(
        mcp_url=server_url,
        scopes=config.scopes,
        client_name=config.client_name,
        client_id=config.client_id,
        client_secret=config.client_secret,
        callback_port=config.callback_port,
        token_storage=token_storage,
    )
    # See _patched_get_client_metadata_scopes below: the SDK's own scope
    # selection would otherwise discard config.scopes entirely during a
    # normal 401-triggered auth flow. This is what that patch looks for.
    provider.context._scope_override = " ".join(config.scopes)  # type: ignore[attr-defined]
    return provider


class OAuthProviderInstanceCache:
    """One ``OAuth`` instance per ``(provider_name, server_url)`` pair.

    Scoped to a single ``ConfigurationV1.from_dict()`` call — its only job is
    deduplicating construction *within one config load*, so tools/servers
    that name the same provider and point at the same server share one live
    instance instead of each independently building (and independently
    authenticating) their own. Persistence across separate loads/processes
    already comes from the shared token storage on disk, independent of
    Python object identity.
    """

    def __init__(self) -> None:
        self._instances: dict[tuple[str, str], OAuth] = {}

    def get_or_build(self, config: OAuthProviderConfig, server_url: str) -> OAuth:
        key = (config.name, server_url)
        if key not in self._instances:
            self._instances[key] = build_oauth_provider(config, server_url)
        return self._instances[key]


# --- Workaround for a real gap in the `mcp` SDK's OAuth token parsing ------
#
# RFC 6749 §5.1 says token responses should be JSON. GitHub's classic OAuth
# token endpoint (https://github.com/login/oauth/access_token) only complies
# when the request explicitly sends `Accept: application/json` — otherwise it
# replies in `application/x-www-form-urlencoded`
# (`access_token=...&scope=&token_type=bearer`). Confirmed live: identical
# request, only the Accept header differed, and the response content-type
# flipped between text/plain and application/json accordingly.
#
# Neither `_exchange_token_authorization_code` nor `_refresh_token`
# (`mcp/client/auth/oauth2.py`) ever send that header, so
# `OAuthToken.model_validate_json()` chokes on the form-encoded body — a
# crash on the initial exchange (`handle_token_response_scopes` raises,
# uncaught further up), and a silent "refresh never works, always falls back
# to a full browser re-auth" on the refresh path (`_handle_refresh_response`
# already catches the parse failure, but gives up rather than trying the
# other format).
#
# Patched at import time rather than forking the library. Two things need
# reassigning, not one: `oauth2.py` imports `handle_token_response_scopes` by
# name (`from mcp.client.auth.utils import (..., handle_token_response_scopes, ...)`),
# which binds a separate reference into its own module namespace — patching
# `mcp.client.auth.utils.handle_token_response_scopes` alone would not affect
# the name `oauth2.py` actually calls.


def _parse_oauth_token(content: bytes) -> OAuthToken:
    """Parse a token endpoint response body as JSON, falling back to
    application/x-www-form-urlencoded when that fails."""
    try:
        return OAuthToken.model_validate_json(content)
    except PydanticValidationError as json_error:
        try:
            form = dict(parse_qsl(content.decode("utf-8")))
        except (UnicodeDecodeError, ValueError):
            form = {}
        if not form.get("access_token"):
            # Not a token response in either format — the JSON error is the
            # more informative one to surface.
            raise json_error from None
        return OAuthToken.model_validate(form)


async def _patched_handle_token_response_scopes(response: httpx2.Response) -> OAuthToken:
    content = await response.aread()
    try:
        return _parse_oauth_token(content)
    except PydanticValidationError as e:
        raise OAuthTokenError(f"Invalid token response: {e}") from e


async def _patched_handle_refresh_response(
    self: OAuthClientProvider, response: httpx2.Response
) -> bool:
    # The SDK's own method with the parse swapped for `_parse_oauth_token`:
    # keep the rest in step with it on every `mcp` upgrade.
    if response.status_code != 200:
        _mcp_oauth2.logger.warning(
            f"Token refresh failed: {response.status_code}{redirect_note(response)}"
        )
        self.context.clear_tokens()
        return False
    try:
        content = await response.aread()
        token_response = _parse_oauth_token(content)
        # RFC 6749 §6: a refresh response may omit scope and refresh_token,
        # which then stay what they were.
        prior = self.context.current_tokens
        if token_response.scope is None and prior is not None:
            token_response.scope = prior.scope
        if token_response.refresh_token is None and prior is not None:
            token_response.refresh_token = prior.refresh_token
        self.context.current_tokens = token_response
        self.context.update_token_expiry(token_response)
        await self.context.storage.set_tokens(token_response)
        return True
    except PydanticValidationError:
        _mcp_oauth2.logger.exception("Invalid refresh response")
        self.context.clear_tokens()
        return False


_mcp_oauth_utils.handle_token_response_scopes = _patched_handle_token_response_scopes
_mcp_oauth2.handle_token_response_scopes = _patched_handle_token_response_scopes  # type: ignore[attr-defined]
_mcp_oauth2.OAuthClientProvider._handle_refresh_response = (  # type: ignore[method-assign]
    _patched_handle_refresh_response
)


# --- Workaround for the `mcp` SDK discarding a caller-configured scope -----
#
# `get_client_metadata_scopes()` (`mcp/client/auth/utils.py`) is called at
# Step 3 of the 401-triggered auth flow and *unconditionally* overwrites
# `self.context.client_metadata.scope` — the same field our `scopes` config
# set at construction — with, per its own docstring, "the MCP spec's scope
# selection strategy": the WWW-Authenticate header's scope if given, else
# *every* scope the server's Protected Resource Metadata advertises. Slack's
# 401 carries no WWW-Authenticate scope, so it falls to the second case and
# requests its entire scopes_supported list — confirmed live, by inspecting
# the actual authorization URL logged, which listed every Slack scope except
# the one actually configured.
#
# There is no clean extension point for this: the function receives only
# (www_authenticate_scope, protected_resource_metadata,
# authorization_server_metadata) — no reference to which OAuthClientProvider
# is asking, so a drop-in replacement has no way to know what was originally
# configured for *this* flow. Pre-populating protected_resource_metadata the
# way oauth_metadata is eagerly pre-populated above doesn't help either —
# async_auth_flow re-discovers and overwrites it unconditionally, with no
# "already set" guard.
#
# So this reaches across the stack frame to the caller instead: at both call
# site (`async_auth_flow`, on a 401 and on a 403 step-up), the result goes to
# `self.context.client_metadata.scope` — `self` is a local variable in that
# immediate caller frame. If
# `build_oauth_provider()` recorded a `_scope_override` on that provider's
# context, return it directly instead of computing the SDK's own answer.
# Verified in isolation before implementing: a fake object exposing
# `.context._scope_override` is correctly found and preferred; without it,
# falls through to the original function unchanged.


def _patched_get_client_metadata_scopes(
    www_authenticate_scope: str | None,
    protected_resource_metadata: ProtectedResourceMetadata | None,
    authorization_server_metadata: OAuthMetadata | None = None,
    client_grant_types: list[str] | None = None,
) -> str | None:
    frame = inspect.currentframe()
    caller_frame = frame.f_back if frame is not None else None
    caller_self = caller_frame.f_locals.get("self") if caller_frame is not None else None
    override = getattr(getattr(caller_self, "context", None), "_scope_override", None)
    if override:
        return str(override)
    return _original_get_client_metadata_scopes(
        www_authenticate_scope,
        protected_resource_metadata,
        authorization_server_metadata,
        client_grant_types,
    )


_mcp_oauth_utils.get_client_metadata_scopes = _patched_get_client_metadata_scopes
_mcp_oauth2.get_client_metadata_scopes = _patched_get_client_metadata_scopes  # type: ignore[attr-defined]


# --- Serialize concurrent authorization round trips sharing a callback port -
#
# `callback_port` defaults to the same `DEFAULT_OAUTH_CALLBACK_PORT` for
# every provider, and MCP tool/server initialization runs concurrently
# (`asyncio.gather`, both within `Agent.initialize_mcp_tools()` and across
# sub-agents in the loader). `_perform_authorization_code_grant()` opens the
# browser tab (`context.redirect_handler`) and only afterwards waits for the
# one request the local callback server accepts (`context.callback_handler`)
# — two calls in the same method, back to back, with nothing serializing
# either of them by default.
#
# A first attempt locked only the local server's bind (wrapping
# `context.callback_handler` in `build_oauth_provider()`). That stops the
# most acute failure — uvicorn turns a bind collision into an uncaught
# `SystemExit` rather than a catchable `OSError`, since `Server.startup()`
# calls `sys.exit(1)` on failure, which can abort the whole process — but it
# was still wrong: multiple browser tabs could be open at once (nothing
# guarded `redirect_handler`), and whichever tab the user completed first hit
# whatever flow's local server happened to be listening at that moment, not
# necessarily its own. Confirmed live across three concurrently-authenticating
# MCP servers (Atlassian, Slack, GitHub): the callback intended for one
# provider's `state` landed on a different provider's listener, correctly
# detected as `OAuthFlowError: State parameter mismatch`, but at the cost of
# one flow never receiving its own callback at all.
#
# So the whole round trip — open the tab, wait for its own redirect — is
# serialized instead, by patching the method that runs both steps. Keyed by
# the port in this flow's own registered `redirect_uri`, not by
# `OAuthProviderConfig.callback_port` directly, so it matches whatever
# `OAuth._bind()` actually built the callback server on regardless of
# whether that came from the shared default or a per-provider override.
# The consequence: first-time (or refresh-failed) auth across servers
# sharing one port happens strictly one at a time — a second flow doesn't
# even open its tab until the first has finished, errored, or hit
# `callback_timeout` (default 300s) — which is the necessary trade-off for a
# fixed, pre-registrable, shared-by-default redirect_uri.

_callback_port_locks: dict[int, asyncio.Lock] = {}


def _lock_for_port(port: int) -> asyncio.Lock:
    return _callback_port_locks.setdefault(port, asyncio.Lock())


def _redirect_port(context: Any) -> int | None:
    """The port from this flow's own registered redirect_uri."""
    redirect_uris = context.client_metadata.redirect_uris
    if not redirect_uris:
        return None
    return urlparse(str(redirect_uris[0])).port


_original_perform_authorization_code_grant = (
    _mcp_oauth2.OAuthClientProvider._perform_authorization_code_grant
)


async def _patched_perform_authorization_code_grant(
    self: OAuthClientProvider,
) -> tuple[str, str]:
    port = _redirect_port(self.context)
    if port is None:
        return await _original_perform_authorization_code_grant(self)
    async with _lock_for_port(port):
        return await _original_perform_authorization_code_grant(self)


_mcp_oauth2.OAuthClientProvider._perform_authorization_code_grant = (  # type: ignore[method-assign]
    _patched_perform_authorization_code_grant
)
