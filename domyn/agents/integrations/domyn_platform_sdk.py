"""The one place domyn-agents reaches the Domyn platform, and only through its SDK.

domyn-agents runs without the platform. Everything that does talk to it (tool discovery for a delegate
agent, the stored configuration, the relay to connect to, the language models of a space) goes through
``domyn-platform-sdk``, which owns the endpoints, headers and URL rules. The SDK is not a dependency of this package (it is distributed with
the platform, so the lock cannot resolve it): it is installed next to it, at ``MIN_SDK_VERSION`` or later,
and imported here and nowhere else.

Callers depend on :class:`ToolCatalog`, not on the SDK, so a test or a third party can supply its own.
"""

import asyncio
import importlib.metadata
import importlib.util
import os
import re
import threading
from collections.abc import AsyncIterator, Coroutine
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

# One tool as the platform describes it: ``name``, ``description`` and ``parameters``, each parameter a
# dict with ``name`` and optionally ``type``, ``description``, ``default`` and ``is_required``.
ToolDefinition = dict[str, Any]

_T = TypeVar("_T")

# The oldest SDK that has what this module calls (``api.agents``, ``api.relay``). The one place the
# version is written down: it is not in ``pyproject.toml``, since the lock could not resolve it.
MIN_SDK_VERSION = "0.37.0"
SDK_REQUIREMENT = f"domyn-platform-sdk>={MIN_SDK_VERSION}"


class PlatformSDKNotInstalledError(ImportError):
    """The platform was asked for, but ``domyn-platform-sdk`` is missing or older than needed."""

    def __init__(self, installed: str | None = None) -> None:
        found = f"version {installed} is installed" if installed else "it is not installed"
        super().__init__(
            f"Using the Domyn platform needs {SDK_REQUIREMENT}, and {found}. "
            f'It is distributed with the Domyn platform; install it with: uv pip install "{SDK_REQUIREMENT}"'
        )


def sdk_installed() -> bool:
    """Whether ``domyn-platform-sdk`` can be imported, without importing it."""
    return importlib.util.find_spec("domyn_platform_sdk") is not None


def _release(version: str) -> tuple[int, ...]:
    """The numeric part of a version, so ``0.37.0rc1`` reads as ``(0, 37, 0)``."""
    match = re.match(r"\d+(?:\.\d+)*", version)
    return tuple(int(part) for part in match.group().split(".")) if match else ()


def _installed_version() -> str | None:
    try:
        return importlib.metadata.version("domyn-platform-sdk")
    except importlib.metadata.PackageNotFoundError:
        return None  # importable without being installed as a distribution (a source checkout)


def _require_sdk() -> None:
    """Raise :class:`PlatformSDKNotInstalledError` unless the SDK is there and recent enough."""
    if not sdk_installed():
        raise PlatformSDKNotInstalledError()
    installed = _installed_version()
    if installed is not None and _release(installed) < _release(MIN_SDK_VERSION):
        raise PlatformSDKNotInstalledError(installed)


def platform_tools_enabled(requested: bool | None) -> bool:
    """Whether a worker should attach the platform's tools: what was asked for, else whether it can.

    ``None`` is "not asked", which means yes when the platform SDK is installed, so installing it is
    all it takes. An explicit ``True`` is kept even without the SDK, so the missing SDK is reported
    when the tools are fetched instead of being silently skipped.
    """
    return sdk_installed() if requested is None else requested


class ToolCatalog(Protocol):
    """Where a delegate agent learns which tools it can call."""

    async def list_tools(self, *, agent_name: str | None = None) -> list[ToolDefinition]: ...


@dataclass(frozen=True)
class PlatformSettings:
    """What identifies a delegate agent to the platform. Every field falls back to its ``DOMYN_*`` variable."""

    api_key: str = ""
    space_id: str = ""
    channel_id: str = ""
    base_url: str = ""
    configuration_id: str | None = None

    @classmethod
    def from_env(
        cls,
        api_key: str | None = None,
        space_id: str | None = None,
        channel_id: str | None = None,
        base_url: str | None = None,
        configuration_id: str | None = None,
    ) -> "PlatformSettings":
        return cls(
            api_key=api_key or os.environ.get("DOMYN_API_KEY", ""),
            space_id=space_id or os.environ.get("DOMYN_SPACE_ID", ""),
            channel_id=channel_id or os.environ.get("DOMYN_CHANNEL_ID", ""),
            base_url=base_url or os.environ.get("DOMYN_BASE_URL", ""),
            configuration_id=configuration_id or os.environ.get("DOMYN_CONFIGURATION_ID") or None,
        )


def _with_scheme(base_url: str) -> str:
    """A bare host becomes an https URL; which host the platform serves is the SDK's business."""
    base_url = base_url.strip().rstrip("/")
    if base_url and not base_url.startswith(("http://", "https://")):
        return f"https://{base_url}"
    return base_url


def _new_sdk(settings: PlatformSettings) -> Any:
    _require_sdk()
    try:
        from domyn_platform_sdk import PlatformSDK
    except ImportError as exc:
        raise PlatformSDKNotInstalledError() from exc
    api: dict[str, Any] = {}
    if settings.base_url:
        api["base_url"] = _with_scheme(settings.base_url)
    return PlatformSDK({"api": api})


def _auth(settings: PlatformSettings) -> Any:
    _require_sdk()
    try:
        from domyn_platform_sdk.auth import AuthContext
    except ImportError as exc:
        raise PlatformSDKNotInstalledError() from exc
    return AuthContext(api_key=settings.api_key)


async def _close(sdk: Any) -> None:
    # A client belongs to the event loop it was built on, so each call builds and closes its own.
    await sdk.http_client.build_client().aclose()


class SdkToolCatalog:
    """A :class:`ToolCatalog` backed by ``domyn-platform-sdk``."""

    def __init__(self, settings: PlatformSettings) -> None:
        self._settings = settings

    async def list_tools(self, *, agent_name: str | None = None) -> list[ToolDefinition]:
        s = self._settings
        sdk = _new_sdk(s)
        try:
            tools = await sdk.api.agents.list_delegate_tools_for_channel(
                _auth(s),
                space_id=s.space_id,
                channel_id=s.channel_id,
                configuration_id=s.configuration_id,
                agent_name=agent_name,
            )
        finally:
            await _close(sdk)
        return list(tools)


def default_catalog(settings: PlatformSettings) -> ToolCatalog:
    return SdkToolCatalog(settings)


async def sandbox_auto_approve(settings: PlatformSettings, configuration_id: str) -> bool | None:
    """Whether the stored configuration lets a sandbox agent skip approval; ``None`` if it cannot be read."""
    sdk = _new_sdk(settings)
    try:
        from domyn_platform_sdk.api.agents import sandbox_auto_approve as read_toggle
    except ImportError as exc:
        raise PlatformSDKNotInstalledError() from exc
    try:
        configuration = await sdk.api.agents.get_configuration(
            _auth(settings), configuration_id=configuration_id, space_id=settings.space_id
        )
    finally:
        await _close(sdk)
    toggle: bool | None = read_toggle(configuration)
    return toggle


async def list_models(settings: PlatformSettings) -> list[dict[str, Any]]:
    """The models the space's configuration offers: ``id`` is the name to give a provider."""
    sdk = _new_sdk(settings)
    try:
        models: list[dict[str, Any]] = await sdk.api.llm.list_models(
            _auth(settings), settings.space_id, configuration_id=settings.configuration_id
        )
    finally:
        await _close(sdk)
    return list(models)


async def chat_completion(settings: PlatformSettings, payload: dict[str, Any]) -> dict[str, Any]:
    """One chat completion through the platform: an OpenAI-style request, the provider's response."""
    sdk = _new_sdk(settings)
    try:
        completion: dict[str, Any] = await sdk.api.llm.chat_completions(
            _auth(settings), settings.space_id, payload, configuration_id=settings.configuration_id
        )
    finally:
        await _close(sdk)
    return completion


async def stream_chat_completion(
    settings: PlatformSettings, payload: dict[str, Any]
) -> AsyncIterator[dict[str, Any]]:
    """The chunks of one chat completion through the platform, as they arrive."""
    sdk = _new_sdk(settings)
    try:
        async for chunk in sdk.api.llm.stream_chat_completions(
            _auth(settings), settings.space_id, payload, configuration_id=settings.configuration_id
        ):
            yield chunk
    finally:
        await _close(sdk)


def relay_connection(settings: PlatformSettings) -> tuple[str, dict[str, str]]:
    """The relay's WebSocket URL and the headers to connect to it with."""
    connection = _new_sdk(settings).api.relay.connection(
        _auth(settings),
        channel_id=settings.channel_id,
        space_id=settings.space_id,
        configuration_id=settings.configuration_id,
    )
    return connection["url"], {k: v for k, v in connection["headers"].items() if v}


def run_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run a coroutine from synchronous code, including code already inside a running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    outcome: list[Any] = []

    def target() -> None:
        try:
            outcome.append((True, asyncio.run(coro)))
        except BaseException as exc:  # handed back to the caller's thread
            outcome.append((False, exc))

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join()
    ok, value = outcome[0]
    if not ok:
        raise value
    return value  # type: ignore[no-any-return]
