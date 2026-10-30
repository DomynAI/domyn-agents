"""Unit tests for the plugin's ``register()`` entrypoint.

The post-refactor plugin uses a factory pattern: ``register()`` calls
``ctx.register_platform(adapter_factory=...)`` and the gateway invokes
the factory later to build the ``DomynPlatformAdapter``. Tests capture
the factory and exercise it with stubbed adapter / relay-client classes.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from domyn.agents.integrations.domyn_platform_sdk import PlatformSettings

from .conftest import HEADERS as _HEADERS
from .conftest import WS_URL as _WS_URL

_PKG = "domyn.agents.integrations.hermes_platform_gateway"


def _set_env(monkeypatch, **extra):
    monkeypatch.setenv("DOMYN_API_KEY", "key-1")
    monkeypatch.setenv("DOMYN_BASE_URL", "api.example.com")
    monkeypatch.setenv("DOMYN_SPACE_ID", "space-1")
    monkeypatch.setenv("DOMYN_CHANNEL_ID", "chan-1")
    for k, v in extra.items():
        monkeypatch.setenv(k, v)


# ---------------------------------------------------------------------------
# Env-var gating
# ---------------------------------------------------------------------------


def test_register_skips_when_any_env_var_missing(monkeypatch):
    for var in ["DOMYN_API_KEY", "DOMYN_BASE_URL", "DOMYN_SPACE_ID", "DOMYN_CHANNEL_ID"]:
        monkeypatch.delenv(var, raising=False)
    ctx = MagicMock()
    from domyn.agents.integrations.hermes_platform_gateway import register

    register(ctx)
    ctx.register_tool.assert_not_called()
    ctx.register_platform.assert_not_called()


def test_register_skips_when_fetch_tools_fails(monkeypatch):
    """fetch_tools failing must NOT abort registration — the adapter
    factory + platform registration still happen so the gateway can
    boot; only tool registration is skipped."""
    _set_env(monkeypatch)
    with (
        patch(f"{_PKG}.client.fetch_tools", side_effect=Exception("unreachable")),
        patch(f"{_PKG}._start_refresh_loop"),
    ):
        ctx = MagicMock()
        from domyn.agents.integrations.hermes_platform_gateway import register

        register(ctx)
        ctx.register_tool.assert_not_called()
        ctx.register_platform.assert_called_once()


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def test_register_skips_tool_with_missing_name(monkeypatch):
    _set_env(monkeypatch)
    tool_defs = [
        {"parameters": []},  # no name
        {"name": "ok_tool", "parameters": []},
    ]
    with (
        patch(f"{_PKG}.client.fetch_tools", return_value=tool_defs),
        patch(f"{_PKG}._start_refresh_loop"),
    ):
        ctx = MagicMock()
        from domyn.agents.integrations.hermes_platform_gateway import register

        register(ctx)
        assert ctx.register_tool.call_count == 1
        assert ctx.register_tool.call_args.kwargs["name"] == "ok_tool"


def test_register_calls_ctx_register_tool_with_correct_kwargs(monkeypatch):
    _set_env(monkeypatch)
    tool_defs = [
        {
            "name": "send_email",
            "description": "Send an email",
            "parameters": [
                {"name": "to", "type": "str", "is_required": True, "description": "Recipient"},
            ],
        },
        {"name": "ping", "description": "Ping", "parameters": []},
    ]
    with (
        patch(f"{_PKG}.client.fetch_tools", return_value=tool_defs),
        patch(f"{_PKG}._start_refresh_loop"),
    ):
        ctx = MagicMock()
        from domyn.agents.integrations.hermes_platform_gateway import register

        register(ctx)

    assert ctx.register_tool.call_count == 2

    by_name = {c.kwargs["name"]: c.kwargs for c in ctx.register_tool.call_args_list}

    assert by_name["send_email"]["toolset"] == "platform"
    # Platform tool handlers are async — adapter.call_tool is awaited.
    assert by_name["send_email"]["is_async"] is True
    assert by_name["send_email"]["schema"]["name"] == "send_email"
    assert by_name["send_email"]["schema"]["description"] == "Send an email"
    props = by_name["send_email"]["schema"]["parameters"]["properties"]
    assert props["to"]["type"] == "string"
    assert by_name["send_email"]["schema"]["parameters"]["required"] == ["to"]

    assert by_name["ping"]["schema"]["parameters"]["properties"] == {}


def test_register_handler_delegates_to_adapter_call_tool(monkeypatch):
    """The tool handler routes (task_id → chat_id → session_key) and then
    awaits ``adapter.call_tool``. We drive the registered factory with a
    stubbed adapter so the slot is filled before invoking the handler."""
    _set_env(monkeypatch)
    tool_defs = [{"name": "echo", "parameters": []}]

    mock_adapter = MagicMock()
    mock_adapter._chat_id_by_task = {"task-1": "chat-1"}
    mock_adapter.session_key_for_chat.return_value = "domyn:chan-1:chat-1"
    mock_adapter.thought_for_task.return_value = "thinking"
    mock_adapter.call_tool = AsyncMock(return_value='{"result": "pong"}')

    with (
        patch(f"{_PKG}.client.fetch_tools", return_value=tool_defs),
        patch(f"{_PKG}.adapter.DomynPlatformAdapter", return_value=mock_adapter),
        patch(f"{_PKG}.relay_client.DomynRelayClient"),
        patch(f"{_PKG}._start_refresh_loop"),
    ):
        ctx = MagicMock()
        from domyn.agents.integrations.hermes_platform_gateway import register

        register(ctx)

        # Trigger the factory so adapter_slot[0] is populated.
        factory = ctx.register_platform.call_args.kwargs["adapter_factory"]
        factory(MagicMock())  # PlatformConfig stand-in

        handler = ctx.register_tool.call_args.kwargs["handler"]
        result = asyncio.run(handler({"msg": "ping"}, task_id="task-1"))

    mock_adapter.call_tool.assert_awaited_once()
    kwargs = mock_adapter.call_tool.await_args.kwargs
    assert kwargs["session_key"] == "domyn:chan-1:chat-1"
    assert kwargs["tool_name"] == "echo"
    assert kwargs["args"] == {"msg": "ping"}
    assert kwargs["thought"] == "thinking"
    assert result == '{"result": "pong"}'


def test_handler_returns_error_when_adapter_not_ready(monkeypatch):
    """If the gateway never built the adapter, the handler short-circuits
    with a JSON error rather than NPE'ing."""
    import json

    _set_env(monkeypatch)
    tool_defs = [{"name": "echo", "parameters": []}]

    with (
        patch(f"{_PKG}.client.fetch_tools", return_value=tool_defs),
        patch(f"{_PKG}._start_refresh_loop"),
    ):
        ctx = MagicMock()
        from domyn.agents.integrations.hermes_platform_gateway import register

        register(ctx)
        handler = ctx.register_tool.call_args.kwargs["handler"]
        result = asyncio.run(handler({}, task_id="task-1"))

    assert "adapter not ready" in json.loads(result)["error"]


# ---------------------------------------------------------------------------
# Refresh loop
# ---------------------------------------------------------------------------


def test_register_starts_refresh_loop_by_default(monkeypatch):
    _set_env(monkeypatch)
    tool_defs = [{"name": "t", "parameters": []}]

    with (
        patch(f"{_PKG}.client.fetch_tools", return_value=tool_defs),
        patch(f"{_PKG}._start_refresh_loop") as mock_start,
    ):
        from domyn.agents.integrations.hermes_platform_gateway import register

        register(MagicMock())

    mock_start.assert_called_once()
    assert mock_start.call_args.kwargs["interval"] == 60.0


def test_register_skips_refresh_loop_when_interval_zero(monkeypatch):
    _set_env(monkeypatch, PLATFORM_TOOL_REFRESH_INTERVAL="0")
    tool_defs = [{"name": "t", "parameters": []}]

    with (
        patch(f"{_PKG}.client.fetch_tools", return_value=tool_defs),
        patch(f"{_PKG}._start_refresh_loop") as mock_start,
    ):
        from domyn.agents.integrations.hermes_platform_gateway import register

        register(MagicMock())

    mock_start.assert_not_called()


# ---------------------------------------------------------------------------
# Config syncer wiring
# ---------------------------------------------------------------------------


def test_register_builds_config_syncer_from_env(monkeypatch):
    _set_env(monkeypatch, DOMYN_CONFIGURATION_ID="cfg-env")

    with (
        patch(f"{_PKG}.client.fetch_tools", return_value=[]),
        patch(f"{_PKG}._start_refresh_loop"),
        patch(f"{_PKG}._build_config_syncer") as mock_build,
    ):
        from domyn.agents.integrations.hermes_platform_gateway import register

        register(MagicMock())

    kwargs = mock_build.call_args.kwargs
    assert kwargs["settings"] == PlatformSettings(
        api_key="key-1",
        space_id="space-1",
        channel_id="chan-1",
        base_url="api.example.com",
        configuration_id="cfg-env",
    )
    assert kwargs["default_config_id"] == "cfg-env"


def test_register_skips_when_there_is_no_relay_connection(monkeypatch):
    """No relay to connect to means no platform to register, and no crash."""
    _set_env(monkeypatch)

    def no_relay(settings):
        raise ImportError("domyn-platform-sdk")

    monkeypatch.setattr(f"{_PKG}.relay_connection", no_relay)
    with (
        patch(f"{_PKG}.client.fetch_tools", return_value=[]),
        patch(f"{_PKG}._start_refresh_loop"),
    ):
        ctx = MagicMock()
        from domyn.agents.integrations.hermes_platform_gateway import register

        register(ctx)

    ctx.register_platform.assert_not_called()


def test_register_connects_without_the_configuration_id(monkeypatch, relay):
    _set_env(monkeypatch, DOMYN_CONFIGURATION_ID="cfg-env")
    with (
        patch(f"{_PKG}.client.fetch_tools", return_value=[]),
        patch(f"{_PKG}._start_refresh_loop"),
    ):
        from domyn.agents.integrations.hermes_platform_gateway import register

        register(MagicMock())

    assert relay[0].configuration_id is None
    assert relay[0].channel_id == "chan-1"


def test_register_runs_boot_config_sync(monkeypatch, boot_config_sync):
    """The pod must converge before serving its first turn."""
    _set_env(monkeypatch)

    with (
        patch(f"{_PKG}.client.fetch_tools", return_value=[]),
        patch(f"{_PKG}._start_refresh_loop"),
    ):
        from domyn.agents.integrations.hermes_platform_gateway import register

        register(MagicMock())

    boot_config_sync.assert_called_once()


def test_register_boot_sync_runs_even_without_refresh_loop(monkeypatch, boot_config_sync):
    """Boot sync is unconditional — not gated on the tool-refresh interval."""
    _set_env(monkeypatch, PLATFORM_TOOL_REFRESH_INTERVAL="0")

    with patch(f"{_PKG}.client.fetch_tools", return_value=[]):
        from domyn.agents.integrations.hermes_platform_gateway import register

        register(MagicMock())

    boot_config_sync.assert_called_once()


def test_register_passes_syncer_to_adapter(monkeypatch):
    _set_env(monkeypatch)
    sentinel = MagicMock(name="syncer")

    with (
        patch(f"{_PKG}.client.fetch_tools", return_value=[]),
        patch(f"{_PKG}._start_refresh_loop"),
        patch(f"{_PKG}._build_config_syncer", return_value=sentinel),
    ):
        from domyn.agents.integrations.hermes_platform_gateway import register

        ctx = MagicMock()
        register(ctx)
        factory = ctx.register_platform.call_args.kwargs["adapter_factory"]

        with patch(f"{_PKG}.adapter.DomynPlatformAdapter") as mock_adapter:
            factory(MagicMock())

    assert mock_adapter.call_args.kwargs["config_syncer"] is sentinel


# ---------------------------------------------------------------------------
# Hook registration
# ---------------------------------------------------------------------------


def test_register_registers_hooks(monkeypatch):
    """The plugin registers tool hooks (chat_id stashing / visibility) and
    LLM hooks (LLM_START / LLM_END emission)."""
    _set_env(monkeypatch)
    tool_defs = [{"name": "t", "parameters": []}]

    with (
        patch(f"{_PKG}.client.fetch_tools", return_value=tool_defs),
        patch(f"{_PKG}._start_refresh_loop"),
    ):
        ctx = MagicMock()
        from domyn.agents.integrations.hermes_platform_gateway import register

        register(ctx)

    hook_names = [call.args[0] for call in ctx.register_hook.call_args_list]
    assert "pre_tool_call" in hook_names
    assert "post_tool_call" in hook_names
    assert "pre_llm_call" in hook_names
    assert "post_llm_call" in hook_names


# ---------------------------------------------------------------------------
# _chat_id_for_session_scan
# ---------------------------------------------------------------------------


def test_chat_id_for_session_scan_matches_by_session_id(monkeypatch):
    """Finds the agent whose session_id attribute matches and returns its _chat_id."""
    from domyn.agents.integrations.hermes_platform_gateway import _chat_id_for_session_scan

    mock_agent = MagicMock()
    mock_agent.session_id = "s1"
    mock_agent._chat_id = "conv-99"
    mock_runner = MagicMock()
    mock_runner._running_agents = {"some-key": mock_agent}

    with patch(
        "domyn.agents.integrations.hermes_platform_gateway._runner_ref_or_none",
        return_value=mock_runner,
    ):
        result = _chat_id_for_session_scan("s1")

    assert result == "conv-99"


def test_chat_id_for_session_scan_returns_none_when_no_match(monkeypatch):
    from domyn.agents.integrations.hermes_platform_gateway import _chat_id_for_session_scan

    mock_agent = MagicMock()
    mock_agent.session_id = "other"
    mock_runner = MagicMock()
    mock_runner._running_agents = {"some-key": mock_agent}

    with patch(
        "domyn.agents.integrations.hermes_platform_gateway._runner_ref_or_none",
        return_value=mock_runner,
    ):
        result = _chat_id_for_session_scan("s1")

    assert result is None


def test_chat_id_for_session_scan_returns_none_when_runner_none(monkeypatch):
    from domyn.agents.integrations.hermes_platform_gateway import _chat_id_for_session_scan

    with patch(
        "domyn.agents.integrations.hermes_platform_gateway._runner_ref_or_none",
        return_value=None,
    ):
        result = _chat_id_for_session_scan("s1")

    assert result is None


# ---------------------------------------------------------------------------
# LLM hooks
# ---------------------------------------------------------------------------


def _extract_hook(ctx: MagicMock, name: str):
    """Return the callable registered under *name* via ctx.register_hook."""
    for call in ctx.register_hook.call_args_list:
        if call.args[0] == name:
            return call.args[1]
    raise AssertionError(f"Hook {name!r} was not registered")


def _make_registered_ctx(monkeypatch, tool_defs=None):
    """Run register() and return the ctx mock."""
    _set_env(monkeypatch)
    with (
        patch(f"{_PKG}.client.fetch_tools", return_value=tool_defs or []),
        patch(f"{_PKG}._start_refresh_loop"),
    ):
        ctx = MagicMock()
        from domyn.agents.integrations.hermes_platform_gateway import register

        register(ctx)
    return ctx


def test_pre_llm_call_no_op_when_adapter_none(monkeypatch):
    ctx = _make_registered_ctx(monkeypatch)
    hook = _extract_hook(ctx, "pre_llm_call")
    # adapter_slot[0] is None (factory never called) — must not raise
    hook(
        session_id="s1",
        model="gpt-4",
        user_message="hi",
        conversation_history=[],
        is_first_turn=True,
        platform="domyn",
        sender_id="",
    )


def test_post_llm_call_no_op_when_no_stash(monkeypatch):
    ctx = _make_registered_ctx(monkeypatch)
    hook = _extract_hook(ctx, "post_llm_call")
    # adapter_slot[0] is None — must not raise
    hook(
        session_id="s1",
        model="gpt-4",
        assistant_response="done",
        user_message="hi",
        conversation_history=[],
        platform="domyn",
    )


def test_pre_llm_call_no_op_when_chat_id_not_found(monkeypatch):
    ctx = _make_registered_ctx(monkeypatch)
    hook = _extract_hook(ctx, "pre_llm_call")

    mock_adapter = MagicMock()
    # Trigger factory so adapter_slot is filled
    factory = ctx.register_platform.call_args.kwargs["adapter_factory"]
    with (
        patch(f"{_PKG}.adapter.DomynPlatformAdapter", return_value=mock_adapter),
        patch(f"{_PKG}.relay_client.DomynRelayClient"),
    ):
        factory(MagicMock())

    mock_adapter._chat_id_by_hermes_session = {}
    mock_adapter._turn_by_session = {}
    hook(
        session_id="s1",
        model="gpt-4",
        user_message="hi",
        conversation_history=[],
        is_first_turn=True,
        platform="domyn",
        sender_id="",
    )

    mock_adapter.record_llm_call.assert_not_called()


def test_pre_llm_call_records_and_schedules_llm_start(monkeypatch):
    _set_env(monkeypatch)
    with (
        patch(f"{_PKG}.client.fetch_tools", return_value=[]),
        patch(f"{_PKG}._start_refresh_loop"),
        patch(f"{_PKG}.adapter.DomynPlatformAdapter") as mock_cls,
        patch(f"{_PKG}.relay_client.DomynRelayClient"),
    ):
        mock_adapter = MagicMock()
        mock_cls.return_value = mock_adapter

        ctx = MagicMock()
        from domyn.agents.integrations.hermes_platform_gateway import register

        register(ctx)
        factory = ctx.register_platform.call_args.kwargs["adapter_factory"]
        factory(MagicMock())

    hook = _extract_hook(ctx, "pre_llm_call")

    # Simulate on_session_start having stashed the chat_id already.
    mock_adapter._chat_id_by_hermes_session = {"s1": "conv-1"}
    mock_adapter._turn_by_session = {}

    with patch(f"{_PKG}._schedule_on_gateway_loop") as mock_schedule:
        hook(
            session_id="s1",
            model="claude-3",
            user_message="hi",
            conversation_history=[],
            is_first_turn=True,
            platform="domyn",
            sender_id="",
        )

    mock_adapter.record_llm_call.assert_called_once()
    kwargs = mock_adapter.record_llm_call.call_args.kwargs
    assert kwargs["session_id"] == "s1"
    assert kwargs["chat_id"] == "conv-1"
    assert kwargs["call_id"].startswith("llm-")
    mock_schedule.assert_called_once()
    assert mock_schedule.call_args.kwargs["label"].startswith("LLM_START")


def test_post_llm_call_pops_stash_and_schedules_llm_end(monkeypatch):
    _set_env(monkeypatch)
    with (
        patch(f"{_PKG}.client.fetch_tools", return_value=[]),
        patch(f"{_PKG}._start_refresh_loop"),
        patch(f"{_PKG}.adapter.DomynPlatformAdapter") as mock_cls,
        patch(f"{_PKG}.relay_client.DomynRelayClient"),
    ):
        mock_adapter = MagicMock()
        mock_adapter.pop_llm_call.return_value = ("conv-1", "llm-abc")
        mock_cls.return_value = mock_adapter

        ctx = MagicMock()
        from domyn.agents.integrations.hermes_platform_gateway import register

        register(ctx)
        factory = ctx.register_platform.call_args.kwargs["adapter_factory"]
        factory(MagicMock())

    hook = _extract_hook(ctx, "post_llm_call")

    with patch(f"{_PKG}._schedule_on_gateway_loop") as mock_schedule:
        hook(
            session_id="s1",
            model="claude-3",
            assistant_response="done",
            user_message="hi",
            conversation_history=[],
            platform="domyn",
        )

    mock_adapter.pop_llm_call.assert_called_once_with("s1")
    mock_schedule.assert_called_once()
    assert mock_schedule.call_args.kwargs["label"].startswith("LLM_END")


def test_post_llm_call_no_op_when_pop_returns_none(monkeypatch):
    _set_env(monkeypatch)
    with (
        patch(f"{_PKG}.client.fetch_tools", return_value=[]),
        patch(f"{_PKG}._start_refresh_loop"),
        patch(f"{_PKG}.adapter.DomynPlatformAdapter") as mock_cls,
        patch(f"{_PKG}.relay_client.DomynRelayClient"),
    ):
        mock_adapter = MagicMock()
        mock_adapter.pop_llm_call.return_value = None
        mock_cls.return_value = mock_adapter

        ctx = MagicMock()
        from domyn.agents.integrations.hermes_platform_gateway import register

        register(ctx)
        factory = ctx.register_platform.call_args.kwargs["adapter_factory"]
        factory(MagicMock())

    hook = _extract_hook(ctx, "post_llm_call")

    with patch(f"{_PKG}._schedule_on_gateway_loop") as mock_schedule:
        hook(
            session_id="s1",
            model="claude-3",
            assistant_response="done",
            user_message="hi",
            conversation_history=[],
            platform="domyn",
        )

    mock_schedule.assert_not_called()


# ---------------------------------------------------------------------------
# pre_tool_call: platform tool vs hermes built-in
#
# register() hands its registered-name set to RefreshLoop and keeps reading it
# to classify invoked tools. A platform tool must take the relay path (the
# handler's own TOOL_START/TOOL_END round-trip); only hermes built-ins get a
# visibility-only pair. If the set were copied rather than shared, a tool
# RefreshLoop registers after startup would be misclassified as a built-in and
# emit visibility events on top of the canonical ones.
# ---------------------------------------------------------------------------


def _register_capturing_refresh(monkeypatch, tool_defs):
    """Run register(), returning (ctx, adapter, the set handed to RefreshLoop)."""
    _set_env(monkeypatch)
    with (
        patch(f"{_PKG}.client.fetch_tools", return_value=tool_defs),
        patch(f"{_PKG}._start_refresh_loop") as mock_start,
        patch(f"{_PKG}.adapter.DomynPlatformAdapter") as mock_cls,
        patch(f"{_PKG}.relay_client.DomynRelayClient"),
    ):
        mock_adapter = MagicMock()
        mock_cls.return_value = mock_adapter

        ctx = MagicMock()
        from domyn.agents.integrations.hermes_platform_gateway import register

        register(ctx)
        factory = ctx.register_platform.call_args.kwargs["adapter_factory"]
        factory(MagicMock())

    # One active turn lets _on_pre_tool_call resolve a chat_id without a
    # hermes runner present.
    mock_adapter._turn_by_session = {"sess": SimpleNamespace(conversation_id="conv-1")}
    return ctx, mock_adapter, mock_start.call_args.kwargs["initial_names"]


def test_pre_tool_call_emits_visibility_for_hermes_builtin(monkeypatch):
    """Control for the test below: a name we never registered is a hermes
    built-in and does get a visibility TOOL_START."""
    ctx, adapter, _registered = _register_capturing_refresh(
        monkeypatch, [{"name": "startup_tool", "parameters": []}]
    )
    hook = _extract_hook(ctx, "pre_tool_call")

    with patch(f"{_PKG}._schedule_on_gateway_loop") as mock_schedule:
        hook(tool_name="read_file", args={}, task_id="task-1")

    adapter.record_visibility_call.assert_called_once()
    mock_schedule.assert_called_once()
    assert mock_schedule.call_args.kwargs["label"].startswith("visibility TOOL_START")


def test_pre_tool_call_treats_startup_tool_as_platform(monkeypatch):
    ctx, adapter, _registered = _register_capturing_refresh(
        monkeypatch, [{"name": "startup_tool", "parameters": []}]
    )
    hook = _extract_hook(ctx, "pre_tool_call")

    with patch(f"{_PKG}._schedule_on_gateway_loop") as mock_schedule:
        hook(tool_name="startup_tool", args={}, task_id="task-1")

    adapter.record_visibility_call.assert_not_called()
    mock_schedule.assert_not_called()


def test_pre_tool_call_treats_refresh_added_tool_as_platform(monkeypatch):
    """The regression this guards: RefreshLoop registering a tool after startup
    must be visible to the hook, which only holds if the set is shared."""
    ctx, adapter, registered = _register_capturing_refresh(
        monkeypatch, [{"name": "startup_tool", "parameters": []}]
    )
    hook = _extract_hook(ctx, "pre_tool_call")

    assert registered == {"startup_tool"}
    # Stand in for RefreshLoop._refresh() picking up a newly attached connector.
    registered.add("refresh_added_tool")

    with patch(f"{_PKG}._schedule_on_gateway_loop") as mock_schedule:
        hook(tool_name="refresh_added_tool", args={}, task_id="task-1")

    adapter.record_visibility_call.assert_not_called()
    mock_schedule.assert_not_called()


def test_post_tool_call_skips_visibility_end_for_refresh_added_tool(monkeypatch):
    """The post-hook guard reads the same set, so it must agree with the pre-hook."""
    ctx, adapter, registered = _register_capturing_refresh(
        monkeypatch, [{"name": "startup_tool", "parameters": []}]
    )
    hook = _extract_hook(ctx, "post_tool_call")
    registered.add("refresh_added_tool")
    # A stale visibility call_id must not be paired with a platform tool.
    adapter.pop_visibility_call.return_value = "visibility-stale"

    with patch(f"{_PKG}._schedule_on_gateway_loop") as mock_schedule:
        hook(tool_name="refresh_added_tool", task_id="task-1", result="{}")

    mock_schedule.assert_not_called()


# ---------------------------------------------------------------------------
# on_session_start hook
# ---------------------------------------------------------------------------


def test_on_session_start_stashes_chat_id(monkeypatch):
    _set_env(monkeypatch)
    with (
        patch(f"{_PKG}.client.fetch_tools", return_value=[]),
        patch(f"{_PKG}._start_refresh_loop"),
        patch(f"{_PKG}.adapter.DomynPlatformAdapter") as mock_cls,
        patch(f"{_PKG}.relay_client.DomynRelayClient"),
    ):
        mock_adapter = MagicMock()
        mock_cls.return_value = mock_adapter

        ctx = MagicMock()
        from domyn.agents.integrations.hermes_platform_gateway import register

        register(ctx)
        factory = ctx.register_platform.call_args.kwargs["adapter_factory"]
        factory(MagicMock())

    hook = _extract_hook(ctx, "on_session_start")

    with patch(f"{_PKG}._chat_id_for_session_scan", return_value="conv-1") as mock_scan:
        hook(session_id="s1", platform="domyn")

    mock_scan.assert_called_once_with("s1")
    mock_adapter.record_hermes_session.assert_called_once_with(session_id="s1", chat_id="conv-1")


def test_on_session_start_ignores_non_domyn_platform(monkeypatch):
    _set_env(monkeypatch)
    with (
        patch(f"{_PKG}.client.fetch_tools", return_value=[]),
        patch(f"{_PKG}._start_refresh_loop"),
        patch(f"{_PKG}.adapter.DomynPlatformAdapter") as mock_cls,
        patch(f"{_PKG}.relay_client.DomynRelayClient"),
    ):
        mock_adapter = MagicMock()
        mock_cls.return_value = mock_adapter

        ctx = MagicMock()
        from domyn.agents.integrations.hermes_platform_gateway import register

        register(ctx)
        factory = ctx.register_platform.call_args.kwargs["adapter_factory"]
        factory(MagicMock())

    hook = _extract_hook(ctx, "on_session_start")

    with patch(f"{_PKG}._chat_id_for_session_scan") as mock_scan:
        hook(session_id="s1", platform="slack")

    mock_scan.assert_not_called()
    mock_adapter.record_hermes_session.assert_not_called()


def test_on_session_start_no_op_when_scan_returns_none(monkeypatch):
    _set_env(monkeypatch)
    with (
        patch(f"{_PKG}.client.fetch_tools", return_value=[]),
        patch(f"{_PKG}._start_refresh_loop"),
        patch(f"{_PKG}.adapter.DomynPlatformAdapter") as mock_cls,
        patch(f"{_PKG}.relay_client.DomynRelayClient"),
    ):
        mock_adapter = MagicMock()
        mock_cls.return_value = mock_adapter

        ctx = MagicMock()
        from domyn.agents.integrations.hermes_platform_gateway import register

        register(ctx)
        factory = ctx.register_platform.call_args.kwargs["adapter_factory"]
        factory(MagicMock())

    hook = _extract_hook(ctx, "on_session_start")

    with patch(f"{_PKG}._chat_id_for_session_scan", return_value=None):
        hook(session_id="s1", platform="domyn")

    mock_adapter.record_hermes_session.assert_not_called()


# ---------------------------------------------------------------------------
# Platform registration / relay-client wiring
# ---------------------------------------------------------------------------


def test_register_registers_platform_with_factory(monkeypatch):
    _set_env(monkeypatch)
    with (
        patch(f"{_PKG}.client.fetch_tools", return_value=[]),
        patch(f"{_PKG}._start_refresh_loop"),
    ):
        ctx = MagicMock()
        from domyn.agents.integrations.hermes_platform_gateway import register

        register(ctx)

    ctx.register_platform.assert_called_once()
    kwargs = ctx.register_platform.call_args.kwargs
    assert kwargs["name"] == "domyn"
    assert kwargs["label"] == "Domyn"
    assert callable(kwargs["adapter_factory"])
    assert callable(kwargs["check_fn"])
    assert kwargs["required_env"] == [
        "DOMYN_SPACE_ID",
        "DOMYN_CHANNEL_ID",
    ]


def test_register_passes_correct_ws_url_and_headers(monkeypatch):
    """The factory wires the relay client with the right ws_url + auth
    headers. Captured by intercepting ``DomynRelayClient.__init__``."""
    _set_env(monkeypatch)
    captured: dict = {}

    class CapturingRelayClient:
        def __init__(self, *, ws_url, headers, on_event):
            captured["ws_url"] = ws_url
            captured["headers"] = headers

    with (
        patch(f"{_PKG}.client.fetch_tools", return_value=[]),
        patch(f"{_PKG}.adapter.DomynPlatformAdapter") as mock_adapter_cls,
        patch(f"{_PKG}.relay_client.DomynRelayClient", CapturingRelayClient),
        patch(f"{_PKG}._start_refresh_loop"),
    ):
        # The adapter ctor calls relay_client_factory(self._on_event); we
        # need that factory to actually run so CapturingRelayClient is
        # exercised. Route the patched adapter through a side_effect that
        # invokes the supplied factory.
        def _build_adapter(**kw):
            kw["relay_client_factory"](lambda _e: None)
            return MagicMock()

        mock_adapter_cls.side_effect = _build_adapter

        ctx = MagicMock()
        from domyn.agents.integrations.hermes_platform_gateway import register

        register(ctx)
        factory = ctx.register_platform.call_args.kwargs["adapter_factory"]
        factory(MagicMock())  # trigger the relay-client factory

    assert captured["ws_url"] == _WS_URL
    assert captured["headers"] == _HEADERS


# ---------------------------------------------------------------------------
# _extract_user_text — now lives in adapter.py
# ---------------------------------------------------------------------------


def test_extract_user_text_from_input_param():
    from domyn.agents.core import AgentAction, BaseEvent, ExecutionEventType
    from domyn.agents.integrations.hermes_platform_gateway.adapter import _extract_user_text

    event = BaseEvent(
        event_type=ExecutionEventType.AGENT_START,
        author="platform",
        action=AgentAction(name="invoke", parameters={"input": "hello world"}),
    )
    assert _extract_user_text(event) == "hello world"


def test_extract_user_text_from_content_part():
    from domyn.agents import BaseEvent, ExecutionEventType, Part
    from domyn.agents.integrations.hermes_platform_gateway.adapter import _extract_user_text

    event = BaseEvent(
        event_type=ExecutionEventType.AGENT_START,
        author="platform",
        content=[Part(text="hello from content")],
    )
    assert _extract_user_text(event) == "hello from content"


def test_extract_user_text_falls_back_to_json():
    import json

    from domyn.agents.core import AgentAction, BaseEvent, ExecutionEventType
    from domyn.agents.integrations.hermes_platform_gateway.adapter import _extract_user_text

    event = BaseEvent(
        event_type=ExecutionEventType.AGENT_START,
        author="platform",
        action=AgentAction(
            name="invoke", parameters={"messages": [{"role": "user", "content": "hi"}]}
        ),
    )
    result = _extract_user_text(event)
    assert json.loads(result)["messages"][0]["content"] == "hi"


def test_extract_user_text_returns_empty_for_no_content():
    from domyn.agents import BaseEvent, ExecutionEventType
    from domyn.agents.integrations.hermes_platform_gateway.adapter import _extract_user_text

    event = BaseEvent(event_type=ExecutionEventType.AGENT_START, author="platform")
    assert _extract_user_text(event) == ""
