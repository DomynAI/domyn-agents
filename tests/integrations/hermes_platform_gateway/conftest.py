"""Stub the hermes ``gateway`` / ``hermes_cli`` packages — not installed in the test env.

``adapter.py`` imports several runtime symbols from ``gateway.*`` and
``client.py`` imports from ``hermes_cli.*`` at module load. The plugin uses
these packages only when hosted inside hermes, so they aren't pyproject
dependencies. Tests that import or patch these modules need the names to
resolve — we install minimal stand-ins into ``sys.modules`` before any test
module is collected.
"""

import sys
import types
from typing import Any
from unittest.mock import patch

import pytest


def _ensure_module(name: str) -> types.ModuleType:
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
    return sys.modules[name]


def _install_gateway_stub() -> None:
    try:
        import gateway

        _ = gateway  # real package present — don't shadow it
        return
    except ImportError:
        pass

    _ensure_module("gateway")
    _ensure_module("gateway.platforms")

    config = _ensure_module("gateway.config")

    class _Platform:
        def __init__(self, name: str) -> None:
            self.name = name

        def __eq__(self, other: Any) -> bool:
            return isinstance(other, _Platform) and self.name == other.name

        def __hash__(self) -> int:
            return hash(self.name)

    class _PlatformConfig:
        pass

    config.Platform = _Platform
    config.PlatformConfig = _PlatformConfig

    pr = _ensure_module("gateway.platform_registry")

    class _PlatformEntry:
        def __init__(self, **kw: Any) -> None:
            self.__dict__.update(kw)

    class _PlatformRegistry:
        def is_registered(self, _name: str) -> bool:
            return True  # short-circuit adapter.py's _ensure_domyn_registered

        def register(self, _entry: Any) -> None:
            pass

    pr.PlatformEntry = _PlatformEntry
    pr.platform_registry = _PlatformRegistry()

    base = _ensure_module("gateway.platforms.base")

    class _BasePlatformAdapter:
        def __init__(self, *, config: Any = None, platform: Any = None) -> None:
            # Mirror the real base's public attributes (gateway/platforms/base.py
            # sets ``self.config`` / ``self.platform``).
            self.config = config
            self.platform = platform

        def _mark_connected(self) -> None:
            pass

        def _mark_disconnected(self) -> None:
            pass

        async def handle_message(self, _msg: Any) -> None:
            pass

        # Attempts per logical send, mirroring the real base's retry loop. Tests that
        # exercise retry behaviour raise this; a successful first attempt returns
        # immediately, so the default is indistinguishable from a single send().
        max_send_attempts = 2

        async def _send_with_retry(
            self,
            chat_id: str,
            content: str,
            reply_to: Any = None,
            metadata: Any = None,
            **_: Any,
        ) -> Any:
            # Mirror the real base: deliver the final response via send(), retrying
            # while the result says the failure is retryable.
            result = None
            for _attempt in range(self.max_send_attempts):
                result = await self.send(
                    chat_id=chat_id, content=content, reply_to=reply_to, metadata=metadata
                )
                if getattr(result, "success", True) or not getattr(result, "retryable", False):
                    return result
            return result

        async def interrupt_session_activity(self, _session_key: str, _chat_id: str) -> None:
            pass

        async def on_processing_start(self, _event: Any) -> None:
            pass

        async def on_processing_complete(self, _event: Any, _outcome: Any) -> None:
            pass

    class _MessageEvent:
        def __init__(self, **kw: Any) -> None:
            self.__dict__.update(kw)

    class _MessageType:
        TEXT = "text"

    class _SendResult:
        def __init__(
            self,
            *,
            success: bool = True,
            error: str | None = None,
            retryable: bool = False,
            message_id: str | None = None,
        ) -> None:
            self.success = success
            self.error = error
            self.retryable = retryable
            self.message_id = message_id

    base.BasePlatformAdapter = _BasePlatformAdapter
    base.MessageEvent = _MessageEvent
    base.MessageType = _MessageType
    base.SendResult = _SendResult

    session = _ensure_module("gateway.session")

    class _SessionSource:
        def __init__(self, **kw: Any) -> None:
            self.__dict__.update(kw)

    def _build_session_key(
        source: Any,
        group_sessions_per_user: bool = True,
        thread_sessions_per_user: bool = False,
        profile: Any = None,
    ) -> str:
        # Mirror the real gateway's DM key shape: agent:main:<platform>:dm:<chat_id>
        platform = getattr(getattr(source, "platform", None), "name", "") or ""
        return f"agent:main:{platform}:{getattr(source, 'chat_type', 'dm')}:{getattr(source, 'chat_id', '')}"

    session.SessionSource = _SessionSource
    session.build_session_key = _build_session_key


def _install_hermes_cli_stub() -> None:
    """Stub ``hermes_cli.config`` / ``hermes_cli.managed_scope``.

    ``client.py`` imports ``is_managed`` / ``load_config`` / ``save_config`` and
    ``managed_scope`` at module load. Tests drive the approval-mode apply via an
    injected callback, so these stand-ins only need to exist (and behave as an
    unmanaged, empty config) for the import to resolve.
    """
    try:
        import hermes_cli

        _ = hermes_cli  # real package present — don't shadow it
        return
    except ImportError:
        pass

    _ensure_module("hermes_cli")

    config = _ensure_module("hermes_cli.config")

    def _is_managed() -> bool:
        return False

    def _load_config() -> dict[str, Any]:
        return {}

    def _save_config(_config: Any, **_: Any) -> None:
        pass

    config.is_managed = _is_managed
    config.load_config = _load_config
    config.save_config = _save_config

    managed_scope = _ensure_module("hermes_cli.managed_scope")

    def _is_key_managed(_key: str) -> bool:
        return False

    managed_scope.is_key_managed = _is_key_managed


_install_gateway_stub()
_install_hermes_cli_stub()


_PLUGIN_PKG = "domyn.agents.integrations.hermes_platform_gateway"


@pytest.fixture(autouse=True)
def boot_config_sync():
    """Keep ``register()`` off the network.

    ``register()`` reconciles approval mode once at boot, which is a real HTTP
    GET. ``ConfigSyncer`` swallows its own fetch errors, so an un-stubbed call
    wouldn't fail the test — it would just sit there until the request timed out.
    Stub it by default; tests that assert on the boot sync take this fixture and
    inspect the mock.
    """
    with patch(f"{_PLUGIN_PKG}._sync_config_at_boot") as mock:
        yield mock


WS_URL = "wss://relay.example.com/ws"
HEADERS = {"channel-id": "chan-1", "space-id": "space-1", "api-key": "key-1"}


@pytest.fixture(autouse=True)
def relay(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Where the platform SDK says to connect, without the SDK.

    ``register()`` asks the SDK for the relay's URL and headers, and a test environment without
    the optional ``domyn-platform-sdk`` would make it skip the registration. Returns the settings
    it was asked about.
    """
    calls: list[Any] = []

    def relay_connection(settings: Any) -> tuple[str, dict[str, str]]:
        calls.append(settings)
        return WS_URL, dict(HEADERS)

    monkeypatch.setattr(f"{_PLUGIN_PKG}.relay_connection", relay_connection)
    return calls
