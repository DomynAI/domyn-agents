"""Tests for domyn/agents/cli/commands/serve/"""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import click
import pytest
import typer.exceptions
from fastapi.testclient import TestClient

from domyn.agents import BaseEvent, ExecutionEventType, Part
from domyn.agents.agents import Agent
from domyn.agents.cli.commands.serve.command import _tool_refresh_loop, serve_command
from domyn.agents.cli.commands.serve.server import create_app

_ExitExc = (SystemExit, click.exceptions.Exit, typer.Exit)


def _exit_code(exc_info: Any) -> int:
    """Return the exit code from either SystemExit or click.exceptions.Exit."""
    v = exc_info.value
    return int(v.code if hasattr(v, "code") else v.exit_code)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_asyncio_run(side_effect: Any = None, return_value: Any = None) -> Any:
    kwargs: dict[str, Any] = {}
    if side_effect is not None:
        kwargs["side_effect"] = side_effect
    else:
        kwargs["return_value"] = return_value
    return patch("domyn.agents.cli.commands.serve.command.asyncio.run", **kwargs)


def _patch_server_interface() -> Any:
    return patch("domyn.agents.cli.commands.serve.command.ServerInterface")


def _patch_py_loader(agent: Agent) -> tuple[Any, Any, Any]:
    """Return patches for load_module_from_path, collect_all_agents, find_main_agent."""
    return (
        patch("domyn.agents.cli.commands.serve.command.load_module_from_path"),
        patch(
            "domyn.agents.cli.commands.serve.command.collect_all_agents",
            return_value={agent.name: agent},
        ),
        patch(
            "domyn.agents.cli.commands.serve.command.find_main_agent",
            return_value=agent,
        ),
    )


def _fake_runner(*events: BaseEvent) -> Any:
    """Return a context manager that patches Runner to yield the given events."""

    async def _run(*args: Any, **kwargs: Any) -> Any:
        for ev in events:
            yield ev

    mock_instance = MagicMock()
    mock_instance.run = _run
    return patch("domyn.agents.cli.commands.serve.server.Runner", return_value=mock_instance)


# ---------------------------------------------------------------------------
# serve_command — channel validation
# ---------------------------------------------------------------------------


class TestServeCommandChannelValidation:
    def test_unknown_channel_exits_with_1(self, tmp_path: Path) -> None:
        fake_file = tmp_path / "agent.py"
        fake_file.write_text("# agent")
        with pytest.raises(_ExitExc) as exc_info:
            serve_command(entrypoint=str(fake_file), channel="grpc")
        assert _exit_code(exc_info) == 1

    def test_channel_matching_is_case_insensitive(self, tmp_path: Path, agent: Agent) -> None:
        py_file = tmp_path / "agent.py"
        py_file.write_text("# agent")
        patch_module, patch_collect, patch_find = _patch_py_loader(agent)
        with (
            patch_module,
            patch_collect,
            patch_find,
            _patch_asyncio_run(),
            _patch_server_interface() as mock_iface,
        ):
            serve_command(entrypoint=str(py_file), channel="REST")
        mock_iface.assert_called_once()

    def test_unknown_channel_does_not_start_server(self, tmp_path: Path) -> None:
        fake_file = tmp_path / "agent.py"
        fake_file.write_text("# agent")
        with _patch_server_interface() as mock_iface, pytest.raises(_ExitExc):
            serve_command(entrypoint=str(fake_file), channel="websocket")
        mock_iface.assert_not_called()


# ---------------------------------------------------------------------------
# --channel multi
# ---------------------------------------------------------------------------


class TestReadServeChannelsConfig:
    def test_reads_channels_list(self, tmp_path: Path) -> None:
        from domyn.agents.cli.commands.serve.command import _read_serve_channels_config

        yaml_file = tmp_path / "agent.yaml"
        yaml_file.write_text(
            "entrypoint: MyAgent\n"
            "serve:\n"
            "  channels:\n"
            "    - type: telegram\n"
            "      token_env: TELEGRAM_BOT_TOKEN\n"
            "    - type: slack\n"
            "      bot_token_env: SLACK_BOT_TOKEN\n"
            "      app_token_env: SLACK_APP_TOKEN\n"
        )

        channels = _read_serve_channels_config(yaml_file)

        assert channels == [
            {"type": "telegram", "token_env": "TELEGRAM_BOT_TOKEN"},
            {
                "type": "slack",
                "bot_token_env": "SLACK_BOT_TOKEN",
                "app_token_env": "SLACK_APP_TOKEN",
            },
        ]

    def test_missing_serve_section_exits_with_1(self, tmp_path: Path) -> None:
        from domyn.agents.cli.commands.serve.command import _read_serve_channels_config

        yaml_file = tmp_path / "agent.yaml"
        yaml_file.write_text("entrypoint: MyAgent\n")

        with pytest.raises(_ExitExc) as exc_info:
            _read_serve_channels_config(yaml_file)
        assert _exit_code(exc_info) == 1

    def test_empty_channels_list_exits_with_1(self, tmp_path: Path) -> None:
        from domyn.agents.cli.commands.serve.command import _read_serve_channels_config

        yaml_file = tmp_path / "agent.yaml"
        yaml_file.write_text("entrypoint: MyAgent\nserve:\n  channels: []\n")

        with pytest.raises(_ExitExc) as exc_info:
            _read_serve_channels_config(yaml_file)
        assert _exit_code(exc_info) == 1

    def test_malformed_yaml_exits_with_1(self, tmp_path: Path) -> None:
        from domyn.agents.cli.commands.serve.command import _read_serve_channels_config

        yaml_file = tmp_path / "agent.yaml"
        yaml_file.write_text("entrypoint: [unterminated\n")

        with pytest.raises(_ExitExc) as exc_info:
            _read_serve_channels_config(yaml_file)
        assert _exit_code(exc_info) == 1


class TestServeTranscriber:
    """`serve.transcriber` reaches every channel that names none of its own."""

    def _read(self, tmp_path: Path, text: str) -> list[dict[str, Any]]:
        from domyn.agents.cli.commands.serve.command import _read_serve_channels_config

        yaml_file = tmp_path / "agent.yaml"
        yaml_file.write_text(text)
        return _read_serve_channels_config(yaml_file)

    def test_the_shared_one_is_given_to_each_channel(self, tmp_path: Path) -> None:
        channels = self._read(
            tmp_path,
            "serve:\n"
            "  transcriber: {type: local}\n"
            "  channels:\n"
            "    - type: telegram\n"
            "    - type: slack\n"
            "      transcriber: {type: openai, url: http://asr.test}\n"
            "    - type: whatsapp\n"
            "      transcriber: null\n",
        )
        assert [c.get("transcriber") for c in channels] == [
            {"type": "local"},
            {"type": "openai", "url": "http://asr.test"},
            None,
        ]

    def test_without_one_entries_are_unchanged(self, tmp_path: Path) -> None:
        channels = self._read(tmp_path, "serve:\n  channels:\n    - type: telegram\n")
        assert channels == [{"type": "telegram"}]

    def test_a_channel_is_built_with_its_transcriber(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from domyn.agents.audio import LocalTranscriber
        from domyn.agents.cli.commands.serve.command import _build_channel_from_config

        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tg-token")
        with patch("domyn.agents.channels.telegram.TelegramChannel") as mock_cls:
            _build_channel_from_config(
                {"type": "telegram", "transcriber": {"type": "local", "language": "it"}}
            )
        transcriber = mock_cls.call_args.kwargs["transcriber"]
        assert isinstance(transcriber, LocalTranscriber) and transcriber.language == "it"

    def test_an_invalid_transcriber_stops_the_server_before_it_starts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from domyn.agents.cli.commands.serve.command import _build_channel_from_config

        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tg-token")
        with pytest.raises(_ExitExc) as exc_info:
            _build_channel_from_config({"type": "telegram", "transcriber": {"type": "nope"}})
        assert _exit_code(exc_info) == 1


class TestBuildChannelFromConfig:
    def test_relay_channel_reads_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from domyn.agents.cli.commands.serve.command import _build_channel_from_config

        monkeypatch.setenv("DOMYN_API_KEY", "key-123")
        with patch("domyn.agents.channels.domyn_relay.DomynRelayChannel.from_env") as mock_from_env:
            _build_channel_from_config(
                {"type": "relay", "channel_id": "chan-1", "space_id": "space-1"}
            )
        mock_from_env.assert_called_once_with(
            api_key="key-123",
            channel_id="chan-1",
            space_id="space-1",
            base_url="",
            configuration_id=None,
        )

    def test_relay_channel_missing_api_key_exits_with_1(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from domyn.agents.cli.commands.serve.command import _build_channel_from_config

        monkeypatch.delenv("DOMYN_API_KEY", raising=False)
        with pytest.raises(_ExitExc) as exc_info:
            _build_channel_from_config({"type": "relay"})
        assert _exit_code(exc_info) == 1

    def test_telegram_channel_reads_custom_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from domyn.agents.cli.commands.serve.command import _build_channel_from_config

        monkeypatch.setenv("MY_BOT_TOKEN", "tg-token")
        with patch("domyn.agents.channels.telegram.TelegramChannel") as mock_cls:
            _build_channel_from_config({"type": "telegram", "token_env": "MY_BOT_TOKEN"})
        mock_cls.assert_called_once_with(token="tg-token", poll_timeout=30, transcriber=None)

    def test_telegram_channel_missing_token_exits_with_1(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from domyn.agents.cli.commands.serve.command import _build_channel_from_config

        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        with pytest.raises(_ExitExc) as exc_info:
            _build_channel_from_config({"type": "telegram"})
        assert _exit_code(exc_info) == 1

    def test_slack_channel_reads_both_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from domyn.agents.cli.commands.serve.command import _build_channel_from_config

        monkeypatch.setenv("SLACK_BOT_TOKEN", "bot-tok")
        monkeypatch.setenv("SLACK_APP_TOKEN", "app-tok")
        with patch("domyn.agents.channels.slack.SlackChannel") as mock_cls:
            _build_channel_from_config({"type": "slack"})
        mock_cls.assert_called_once_with(bot_token="bot-tok", app_token="app-tok", transcriber=None)

    def test_whatsapp_channel_reads_all_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from domyn.agents.cli.commands.serve.command import _build_channel_from_config

        monkeypatch.setenv("WHATSAPP_TOKEN", "wa-token")
        monkeypatch.setenv("WHATSAPP_PHONE_ID", "phone-1")
        monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "verify-1")
        with patch("domyn.agents.channels.whatsapp.WhatsAppChannel") as mock_cls:
            _build_channel_from_config({"type": "whatsapp"})
        mock_cls.assert_called_once_with(
            token="wa-token",
            phone_number_id="phone-1",
            verify_token="verify-1",
            webhook_host="0.0.0.0",
            webhook_port=8080,
            transcriber=None,
        )

    def test_unknown_channel_type_exits_with_1(self) -> None:
        from domyn.agents.cli.commands.serve.command import _build_channel_from_config

        with pytest.raises(_ExitExc) as exc_info:
            _build_channel_from_config({"type": "irc"})
        assert _exit_code(exc_info) == 1


class TestServeMulti:
    def test_rejects_non_yaml_entrypoint(self, tmp_path: Path) -> None:
        from domyn.agents.cli.commands.serve.command import _serve_multi

        py_file = tmp_path / "agent.py"
        py_file.write_text("# agent")

        with pytest.raises(_ExitExc) as exc_info:
            _serve_multi(str(py_file), max_concurrency=8)
        assert _exit_code(exc_info) == 1

    def test_builds_one_adapter_and_one_server_per_channel(
        self, tmp_path: Path, agent: Agent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from domyn.agents.cli.commands.serve.command import _serve_multi

        monkeypatch.delenv("DOMYN_API_KEY", raising=False)
        yaml_file = tmp_path / "agent.yaml"
        yaml_file.write_text(
            "entrypoint: MyAgent\n"
            "serve:\n"
            "  channels:\n"
            "    - type: telegram\n"
            "      token_env: TELEGRAM_BOT_TOKEN\n"
            "    - type: slack\n"
            "      bot_token_env: SLACK_BOT_TOKEN\n"
            "      app_token_env: SLACK_APP_TOKEN\n"
        )

        fake_channels = [MagicMock(name="telegram_channel"), MagicMock(name="slack_channel")]
        for ch in fake_channels:
            ch.close = AsyncMock()

        fake_servers = [MagicMock(serve=AsyncMock()) for _ in fake_channels]

        with (
            patch(
                "domyn.agents.cli.commands.serve.command.load_agent_from_yaml",
                new_callable=AsyncMock,
                return_value=agent,
            ),
            patch(
                "domyn.agents.cli.commands.serve.command._build_adapter",
                return_value=MagicMock(name="adapter"),
            ) as mock_build_adapter,
            patch(
                "domyn.agents.cli.commands.serve.command._build_channel_from_config",
                side_effect=fake_channels,
            ) as mock_build_channel,
            patch(
                "domyn.agents.server.AgentServer",
                side_effect=fake_servers,
            ) as mock_agent_server,
        ):
            _serve_multi(str(yaml_file), max_concurrency=4)

        assert mock_build_channel.call_count == 2
        mock_build_adapter.assert_called_once()
        assert mock_agent_server.call_count == 2
        for server in fake_servers:
            server.serve.assert_awaited_once()
        for ch in fake_channels:
            ch.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# --channel relay — repeated --target
# ---------------------------------------------------------------------------


class TestServeRelayTargets:
    def _patch_relay_deps(
        self, channels: list[Any], servers: list[Any], adapter: Any = None
    ) -> tuple[Any, ...]:
        return (
            patch(
                "domyn.agents.cli.commands.serve.command._load_symbol",
                return_value=object(),
            ),
            patch(
                "domyn.agents.channels.domyn_relay.DomynRelayChannel.from_env",
                side_effect=channels,
            ),
            patch(
                "domyn.agents.cli.commands.serve.command._build_adapter",
                return_value=adapter if adapter is not None else MagicMock(name="adapter"),
            ),
            patch("domyn.agents.server.AgentServer", side_effect=servers),
        )

    def test_multiple_targets_share_one_adapter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from domyn.agents.cli.commands.serve.command import _serve_relay

        monkeypatch.setenv("DOMYN_API_KEY", "key-123")

        fake_channels = [MagicMock(name="chan-a"), MagicMock(name="chan-b")]
        for ch in fake_channels:
            ch._ws_url = "wss://relay.example.com/relay/v1/ws"
            ch.close = AsyncMock()
        fake_servers = [MagicMock(serve=AsyncMock()) for _ in fake_channels]

        patch_load, patch_from_env, patch_build_adapter, patch_agent_server = (
            self._patch_relay_deps(fake_channels, fake_servers)
        )
        with (
            patch_load,
            patch_from_env as mock_from_env,
            patch_build_adapter as mock_build_adapter,
            patch_agent_server as mock_agent_server,
        ):
            _serve_relay(
                "mod:sym",
                channel_id=None,
                space_id="sp",
                configuration_id=None,
                base_url="relay.example.com",
                target=["channel_id=chan-a", "channel_id=chan-b"],
                max_concurrency=8,
            )

        assert [c.kwargs["channel_id"] for c in mock_from_env.call_args_list] == [
            "chan-a",
            "chan-b",
        ]
        mock_build_adapter.assert_called_once()
        # first target's channel-id is used for one-time DelegateTool schema population
        assert mock_build_adapter.call_args.args[3] == "chan-a"
        assert mock_agent_server.call_count == 2
        for server in fake_servers:
            server.serve.assert_awaited_once()
        for ch in fake_channels:
            ch.close.assert_awaited_once()

    def test_no_target_falls_back_to_legacy_single_channel_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from domyn.agents.cli.commands.serve.command import _serve_relay

        monkeypatch.setenv("DOMYN_API_KEY", "key-123")

        fake_channel = MagicMock(name="chan-a")
        fake_channel._ws_url = "wss://relay.example.com/relay/v1/ws"
        fake_channel.close = AsyncMock()
        fake_server = MagicMock(serve=AsyncMock())

        patch_load, patch_from_env, patch_build_adapter, patch_agent_server = (
            self._patch_relay_deps([fake_channel], [fake_server])
        )
        with patch_load, patch_from_env as mock_from_env, patch_build_adapter, patch_agent_server:
            _serve_relay(
                "mod:sym",
                channel_id="chan-a",
                space_id="sp",
                configuration_id=None,
                base_url="relay.example.com",
                target=None,
                max_concurrency=8,
            )

        mock_from_env.assert_called_once()
        assert mock_from_env.call_args.kwargs["channel_id"] == "chan-a"
        fake_server.serve.assert_awaited_once()

    def test_no_channel_id_and_no_target_falls_back_to_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from domyn.agents.cli.commands.serve.command import _serve_relay

        monkeypatch.setenv("DOMYN_API_KEY", "key-123")
        monkeypatch.setenv("DOMYN_CHANNEL_ID", "env-chan")

        fake_channel = MagicMock(name="chan-env")
        fake_channel._ws_url = "wss://relay.example.com/relay/v1/ws"
        fake_channel.close = AsyncMock()
        fake_server = MagicMock(serve=AsyncMock())

        patch_load, patch_from_env, patch_build_adapter, patch_agent_server = (
            self._patch_relay_deps([fake_channel], [fake_server])
        )
        with patch_load, patch_from_env as mock_from_env, patch_build_adapter, patch_agent_server:
            _serve_relay(
                "mod:sym",
                channel_id=None,
                space_id="sp",
                configuration_id=None,
                base_url="relay.example.com",
                target=None,
                max_concurrency=8,
            )

        assert mock_from_env.call_args.kwargs["channel_id"] == "env-chan"

    def test_target_space_and_base_url_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """'channel_id=...,space_id=...,base_url=...' scopes those fields to just that
        one target, letting one process straddle several spaces/relay deployments.
        """
        from domyn.agents.cli.commands.serve.command import _serve_relay

        monkeypatch.setenv("DOMYN_API_KEY", "key-123")

        fake_channels = [MagicMock(name="chan-a"), MagicMock(name="chan-b")]
        for ch in fake_channels:
            ch._ws_url = "wss://relay.example.com/relay/v1/ws"
            ch.close = AsyncMock()
        fake_servers = [MagicMock(serve=AsyncMock()) for _ in fake_channels]

        patch_load, patch_from_env, patch_build_adapter, patch_agent_server = (
            self._patch_relay_deps(fake_channels, fake_servers)
        )
        with patch_load, patch_from_env as mock_from_env, patch_build_adapter, patch_agent_server:
            _serve_relay(
                "mod:sym",
                channel_id=None,
                space_id="default-space",
                configuration_id="default-cfg",
                base_url="relay1.example.com",
                target=[
                    "channel_id=chan-a,space_id=space-1",
                    "channel_id=chan-b,space_id=space-2,base_url=relay2.example.com",
                ],
                max_concurrency=8,
            )

        first_call, second_call = mock_from_env.call_args_list
        assert first_call.kwargs == {
            "api_key": "key-123",
            "channel_id": "chan-a",
            "space_id": "space-1",
            "base_url": "relay1.example.com",
            # NOT "default-cfg": that configuration_id belongs to the shared
            # default space, not to "space-1" this target pinned itself to.
            "configuration_id": None,
        }
        assert second_call.kwargs == {
            "api_key": "key-123",
            "channel_id": "chan-b",
            "space_id": "space-2",
            "base_url": "relay2.example.com",
            "configuration_id": None,
        }

    def test_configuration_id_override_wins_even_with_space_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from domyn.agents.cli.commands.serve.command import _serve_relay

        monkeypatch.setenv("DOMYN_API_KEY", "key-123")

        fake_channel = MagicMock(name="chan-a")
        fake_channel._ws_url = "wss://relay.example.com/relay/v1/ws"
        fake_channel.close = AsyncMock()
        fake_server = MagicMock(serve=AsyncMock())

        patch_load, patch_from_env, patch_build_adapter, patch_agent_server = (
            self._patch_relay_deps([fake_channel], [fake_server])
        )
        with patch_load, patch_from_env as mock_from_env, patch_build_adapter, patch_agent_server:
            _serve_relay(
                "mod:sym",
                channel_id=None,
                space_id="default-space",
                configuration_id="default-cfg",
                base_url="relay.example.com",
                target=["channel_id=chan-a,space_id=space-1,configuration_id=cfg-1"],
                max_concurrency=8,
            )

        assert mock_from_env.call_args.kwargs["space_id"] == "space-1"
        assert mock_from_env.call_args.kwargs["configuration_id"] == "cfg-1"

    def test_target_without_space_id_inherits_shared_configuration_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No space_id field -> the target IS the shared/default space, so the
        shared --configuration-id legitimately applies to it.
        """
        from domyn.agents.cli.commands.serve.command import _serve_relay

        monkeypatch.setenv("DOMYN_API_KEY", "key-123")

        fake_channel = MagicMock(name="chan-a")
        fake_channel._ws_url = "wss://relay.example.com/relay/v1/ws"
        fake_channel.close = AsyncMock()
        fake_server = MagicMock(serve=AsyncMock())

        patch_load, patch_from_env, patch_build_adapter, patch_agent_server = (
            self._patch_relay_deps([fake_channel], [fake_server])
        )
        with patch_load, patch_from_env as mock_from_env, patch_build_adapter, patch_agent_server:
            _serve_relay(
                "mod:sym",
                channel_id=None,
                space_id="default-space",
                configuration_id="default-cfg",
                base_url="relay.example.com",
                target=["channel_id=chan-a"],
                max_concurrency=8,
            )

        assert mock_from_env.call_args.kwargs["space_id"] == "default-space"
        assert mock_from_env.call_args.kwargs["configuration_id"] == "default-cfg"

    def test_invalid_target_field_exits_with_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from domyn.agents.cli.commands.serve.command import _serve_relay

        monkeypatch.setenv("DOMYN_API_KEY", "key-123")

        with pytest.raises(_ExitExc) as exc_info:
            _serve_relay(
                "mod:sym",
                channel_id=None,
                space_id="sp",
                configuration_id=None,
                base_url="relay.example.com",
                target=["channel_id=chan-a,bogus_key=x"],
                max_concurrency=8,
            )
        assert _exit_code(exc_info) == 1

    def test_target_missing_channel_id_exits_with_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from domyn.agents.cli.commands.serve.command import _serve_relay

        monkeypatch.setenv("DOMYN_API_KEY", "key-123")

        with pytest.raises(_ExitExc) as exc_info:
            _serve_relay(
                "mod:sym",
                channel_id=None,
                space_id="sp",
                configuration_id=None,
                base_url="relay.example.com",
                target=["space_id=space-1"],
                max_concurrency=8,
            )
        assert _exit_code(exc_info) == 1

    def _run_with_failing_channel(self, monkeypatch: pytest.MonkeyPatch, error: Exception) -> Any:
        from domyn.agents.cli.commands.serve.command import _serve_relay

        monkeypatch.setenv("DOMYN_API_KEY", "key-123")
        with (
            patch(
                "domyn.agents.cli.commands.serve.command._load_symbol",
                return_value=object(),
            ),
            patch(
                "domyn.agents.channels.domyn_relay.DomynRelayChannel.from_env",
                side_effect=error,
            ),
            patch("domyn.agents.cli.commands.serve.command.console") as mock_console,
            pytest.raises(_ExitExc) as exc_info,
        ):
            _serve_relay(
                "mod:sym",
                channel_id="chan-a",
                space_id="sp",
                configuration_id=None,
                base_url="relay.example.com",
                target=None,
                max_concurrency=8,
            )
        assert _exit_code(exc_info) == 1
        return mock_console

    def test_missing_sdk_prints_its_message_and_exits_with_1(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from domyn.agents.integrations.domyn_platform_sdk import PlatformSDKNotInstalledError

        console = self._run_with_failing_channel(
            monkeypatch, PlatformSDKNotInstalledError("install domyn-platform-sdk")
        )

        printed = console.print.call_args.args[0]
        assert "install domyn-platform-sdk" in printed

    def test_other_setup_failure_names_only_the_error_type(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        console = self._run_with_failing_channel(monkeypatch, ValueError("secret-detail"))

        printed = console.print.call_args.args[0]
        assert "ValueError" in printed
        assert "secret-detail" not in printed

    def test_each_agent_server_gets_its_own_channel_but_shared_adapter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each --target must get its own AgentServer/channel pair, all backed by the
        SAME adapter instance — this is what keeps a reply for a conversation started
        on one channel from ever being routed through another one, since
        AgentServer.send routes exclusively through the ``channel`` it was built with.
        """
        from domyn.agents.cli.commands.serve.command import _serve_relay

        monkeypatch.setenv("DOMYN_API_KEY", "key-123")

        fake_channels = [MagicMock(name="chan-a"), MagicMock(name="chan-b")]
        for ch in fake_channels:
            ch._ws_url = "wss://relay.example.com/relay/v1/ws"
            ch.close = AsyncMock()
        fake_servers = [MagicMock(serve=AsyncMock()) for _ in fake_channels]
        shared_adapter = MagicMock(name="adapter")

        patch_load, patch_from_env, patch_build_adapter, patch_agent_server = (
            self._patch_relay_deps(fake_channels, fake_servers, adapter=shared_adapter)
        )
        with (
            patch_load,
            patch_from_env,
            patch_build_adapter,
            patch_agent_server as mock_agent_server,
        ):
            _serve_relay(
                "mod:sym",
                channel_id=None,
                space_id="sp",
                configuration_id=None,
                base_url="relay.example.com",
                target=["channel_id=chan-a", "channel_id=chan-b"],
                max_concurrency=8,
            )

        assert [c.kwargs["channel"] for c in mock_agent_server.call_args_list] == fake_channels
        assert all(c.kwargs["adapter"] is shared_adapter for c in mock_agent_server.call_args_list)

    def test_reply_for_one_target_is_never_sent_on_another(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end (real AgentServer, not mocked): an AGENT_START on channel A must
        only ever produce a reply on channel A, never on channel B.
        """
        from domyn.agents.cli.commands.serve.command import _serve_relay

        monkeypatch.setenv("DOMYN_API_KEY", "key-123")

        start_event = BaseEvent(
            event_type=ExecutionEventType.AGENT_START,
            author="user",
            conversation_id="conv-1",
        )

        def make_channel(events: list[Any]) -> Any:
            ch = MagicMock()
            ch._ws_url = "wss://relay.example.com/relay/v1/ws"
            ch.close = AsyncMock()
            ch.send_event = AsyncMock()
            ch.send_event_priority = AsyncMock()

            async def receive() -> Any:
                for e in events:
                    yield e

            ch.receive = receive
            return ch

        chan_a = make_channel([("conv-1", start_event)])
        chan_b = make_channel([])

        class FakeAdapter:
            async def invoke(self, event: BaseEvent, emit: Any, emit_priority: Any) -> None:
                done = BaseEvent(
                    event_type=ExecutionEventType.AGENT_END,
                    author="agent",
                    conversation_id=event.conversation_id,
                )
                await emit_priority(done)

            def on_tool_response(self, event: BaseEvent) -> None: ...
            def on_cancel(self, event: BaseEvent) -> None: ...
            async def close(self) -> None: ...

        patch_load, patch_from_env, patch_build_adapter, _unused_agent_server_patch = (
            self._patch_relay_deps([chan_a, chan_b], [], adapter=FakeAdapter())
        )
        with patch_load, patch_from_env, patch_build_adapter:
            _serve_relay(
                "mod:sym",
                channel_id=None,
                space_id="sp",
                configuration_id=None,
                base_url="relay.example.com",
                target=["channel_id=chan-a", "channel_id=chan-b"],
                max_concurrency=8,
            )

        chan_a.send_event_priority.assert_awaited_once()
        chan_b.send_event_priority.assert_not_awaited()
        chan_b.send_event.assert_not_awaited()

    def test_serve_command_forwards_repeated_target(self) -> None:
        from domyn.agents.cli.commands import serve_command

        with patch("domyn.agents.cli.commands.serve.command._serve_relay") as mock_serve_relay:
            serve_command(
                entrypoint="mod:sym",
                channel="relay",
                target=["channel_id=chan-a", "channel_id=chan-b"],
            )

        mock_serve_relay.assert_called_once()
        assert mock_serve_relay.call_args.kwargs["target"] == [
            "channel_id=chan-a",
            "channel_id=chan-b",
        ]


class TestParseTargetSpec:
    def test_channel_id_only(self) -> None:
        from domyn.agents.cli.commands.serve.command import _parse_target_spec

        assert _parse_target_spec("channel_id=chan-a") == {"channel_id": "chan-a"}

    def test_channel_id_plus_space_id(self) -> None:
        from domyn.agents.cli.commands.serve.command import _parse_target_spec

        assert _parse_target_spec("channel_id=chan-a,space_id=space-1") == {
            "channel_id": "chan-a",
            "space_id": "space-1",
        }

    def test_all_fields(self) -> None:
        from domyn.agents.cli.commands.serve.command import _parse_target_spec

        fields = _parse_target_spec(
            "channel_id=chan-b,space_id=space-2,base_url=relay2.example.com,configuration_id=cfg-1"
        )
        assert fields == {
            "channel_id": "chan-b",
            "space_id": "space-2",
            "base_url": "relay2.example.com",
            "configuration_id": "cfg-1",
        }

    def test_base_url_with_port_is_preserved(self) -> None:
        """base_url values may themselves contain ':' — only ',' splits fields."""
        from domyn.agents.cli.commands.serve.command import _parse_target_spec

        fields = _parse_target_spec("channel_id=chan-a,base_url=relay.example.com:8443")
        assert fields == {"channel_id": "chan-a", "base_url": "relay.example.com:8443"}

    def test_field_order_does_not_matter(self) -> None:
        from domyn.agents.cli.commands.serve.command import _parse_target_spec

        assert _parse_target_spec("space_id=space-1,channel_id=chan-a") == {
            "channel_id": "chan-a",
            "space_id": "space-1",
        }

    def test_missing_channel_id_raises_value_error(self) -> None:
        from domyn.agents.cli.commands.serve.command import _parse_target_spec

        with pytest.raises(ValueError, match="channel_id"):
            _parse_target_spec("space_id=space-1")

    def test_unknown_key_raises_value_error(self) -> None:
        from domyn.agents.cli.commands.serve.command import _parse_target_spec

        with pytest.raises(ValueError, match="bogus"):
            _parse_target_spec("channel_id=chan-a,bogus=1")

    def test_missing_equals_raises_value_error(self) -> None:
        from domyn.agents.cli.commands.serve.command import _parse_target_spec

        with pytest.raises(ValueError):
            _parse_target_spec("channel_id=chan-a,space_id")


# ---------------------------------------------------------------------------
# serve_command — agent loading
# ---------------------------------------------------------------------------


class TestServeCommandAgentLoading:
    def test_yaml_file_triggers_yaml_loader(self, tmp_path: Path, agent: Agent) -> None:
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("version: '1.0'")

        with (
            patch(
                "domyn.agents.cli.commands.serve.command.load_agent_from_yaml",
                new_callable=AsyncMock,
                return_value=agent,
            ) as mock_yaml,
            _patch_asyncio_run(return_value=agent),
            _patch_server_interface(),
        ):
            serve_command(entrypoint=str(yaml_file), channel="rest")

        mock_yaml.assert_called_once_with(str(yaml_file))

    def test_yml_extension_triggers_yaml_loader(self, tmp_path: Path, agent: Agent) -> None:
        yml_file = tmp_path / "config.yml"
        yml_file.write_text("version: '1.0'")

        with (
            patch(
                "domyn.agents.cli.commands.serve.command.load_agent_from_yaml",
                new_callable=AsyncMock,
                return_value=agent,
            ) as mock_yaml,
            _patch_asyncio_run(return_value=agent),
            _patch_server_interface(),
        ):
            serve_command(entrypoint=str(yml_file), channel="rest")

        mock_yaml.assert_called_once()

    def test_python_file_uses_load_module_from_path(self, tmp_path: Path, agent: Agent) -> None:
        py_file = tmp_path / "agent.py"
        py_file.write_text("# agent")
        patch_module, patch_collect, patch_find = _patch_py_loader(agent)

        with (
            patch_module as mock_load,
            patch_collect,
            patch_find,
            _patch_asyncio_run(),
            _patch_server_interface(),
        ):
            serve_command(entrypoint=str(py_file), channel="rest")

        mock_load.assert_called_once_with(str(py_file))

    def test_agent_name_forwarded_to_find_main_agent(self, tmp_path: Path, agent: Agent) -> None:
        py_file = tmp_path / "agent.py"
        py_file.write_text("# agent")
        patch_module, patch_collect, patch_find = _patch_py_loader(agent)

        with (
            patch_module,
            patch_collect,
            patch_find as mock_find,
            _patch_asyncio_run(),
            _patch_server_interface(),
        ):
            serve_command(entrypoint=str(py_file), channel="rest", agent_name="SpecificAgent")

        mock_find.assert_called_once_with({agent.name: agent}, "SpecificAgent")

    def test_loading_error_exits_with_1(self, tmp_path: Path) -> None:
        py_file = tmp_path / "agent.py"
        py_file.write_text("# agent")

        with (
            patch(
                "domyn.agents.cli.commands.serve.command.load_module_from_path",
                side_effect=ImportError("bad import"),
            ),
            pytest.raises(_ExitExc) as exc_info,
        ):
            serve_command(entrypoint=str(py_file), channel="rest")

        assert _exit_code(exc_info) == 1

    def test_loading_error_does_not_start_server(self, tmp_path: Path) -> None:
        py_file = tmp_path / "agent.py"
        py_file.write_text("# agent")

        with (
            patch(
                "domyn.agents.cli.commands.serve.command.load_module_from_path",
                side_effect=ValueError("no agent found"),
            ),
            _patch_server_interface() as mock_iface,
            pytest.raises(_ExitExc),
        ):
            serve_command(entrypoint=str(py_file), channel="rest")

        mock_iface.assert_not_called()


# ---------------------------------------------------------------------------
# serve_command — server lifecycle
# ---------------------------------------------------------------------------


class TestServeCommandServerLifecycle:
    def test_server_interface_created_with_main_and_all_agents(
        self, tmp_path: Path, agent: Agent
    ) -> None:
        py_file = tmp_path / "agent.py"
        py_file.write_text("# agent")
        all_agents = {agent.name: agent}
        patch_module, patch_collect, patch_find = _patch_py_loader(agent)

        with (
            patch_module,
            patch_collect,
            patch_find,
            _patch_asyncio_run(),
            _patch_server_interface() as mock_iface,
        ):
            serve_command(entrypoint=str(py_file), channel="rest")

        mock_iface.assert_called_once_with(agent, all_agents)

    def test_keyboard_interrupt_handled_gracefully(self, tmp_path: Path, agent: Agent) -> None:
        py_file = tmp_path / "agent.py"
        py_file.write_text("# agent")
        patch_module, patch_collect, patch_find = _patch_py_loader(agent)

        with (
            patch_module,
            patch_collect,
            patch_find,
            _patch_asyncio_run(side_effect=KeyboardInterrupt),
            _patch_server_interface(),
        ):
            # must not propagate
            serve_command(entrypoint=str(py_file), channel="rest")

    def test_server_start_error_exits_with_1(self, tmp_path: Path, agent: Agent) -> None:
        py_file = tmp_path / "agent.py"
        py_file.write_text("# agent")
        patch_module, patch_collect, patch_find = _patch_py_loader(agent)

        with (
            patch_module,
            patch_collect,
            patch_find,
            _patch_asyncio_run(side_effect=RuntimeError("port already in use")),
            _patch_server_interface(),
            pytest.raises(_ExitExc) as exc_info,
        ):
            serve_command(entrypoint=str(py_file), channel="rest")

        assert _exit_code(exc_info) == 1


# ---------------------------------------------------------------------------
# create_app — /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_returns_200(self, agent: Agent) -> None:
        client = TestClient(create_app(agent, {agent.name: agent}))
        assert client.get("/health").status_code == 200

    def test_status_is_healthy(self, agent: Agent) -> None:
        client = TestClient(create_app(agent, {agent.name: agent}))
        assert client.get("/health").json()["status"] == "healthy"

    def test_reports_main_agent_name(self, agent: Agent, child_agent: Agent) -> None:
        client = TestClient(create_app(child_agent, {child_agent.name: child_agent}))
        assert client.get("/health").json()["agent"] == child_agent.name


# ---------------------------------------------------------------------------
# create_app — /agents
# ---------------------------------------------------------------------------


class TestAgentsListEndpoint:
    def test_returns_200(self, agent: Agent) -> None:
        client = TestClient(create_app(agent, {agent.name: agent}))
        assert client.get("/agents").status_code == 200

    def test_returns_all_registered_agents(self, agent: Agent, child_agent: Agent) -> None:
        all_agents = {agent.name: agent, child_agent.name: child_agent}
        client = TestClient(create_app(agent, all_agents))
        names = {a["name"] for a in client.get("/agents").json()}
        assert names == {agent.name, child_agent.name}

    def test_each_entry_has_name_and_description(self, agent: Agent) -> None:
        client = TestClient(create_app(agent, {agent.name: agent}))
        entry = client.get("/agents").json()[0]
        assert "name" in entry
        assert "description" in entry

    def test_description_matches_agent(self, agent: Agent) -> None:
        client = TestClient(create_app(agent, {agent.name: agent}))
        entry = client.get("/agents").json()[0]
        assert entry["description"] == agent.description


# ---------------------------------------------------------------------------
# create_app — /agents/{name}
# ---------------------------------------------------------------------------


class TestAgentInfoEndpoint:
    def test_returns_200_for_known_agent(self, agent: Agent) -> None:
        client = TestClient(create_app(agent, {agent.name: agent}))
        assert client.get(f"/agents/{agent.name}").status_code == 200

    def test_returns_404_for_unknown_agent(self, agent: Agent) -> None:
        client = TestClient(create_app(agent, {agent.name: agent}))
        assert client.get("/agents/Ghost").status_code == 404

    def test_body_contains_name_description_tools(self, agent: Agent) -> None:
        client = TestClient(create_app(agent, {agent.name: agent}))
        body = client.get(f"/agents/{agent.name}").json()
        assert body["name"] == agent.name
        assert body["description"] == agent.description
        assert "tools" in body

    def test_lists_tool_names(self, agent: Agent) -> None:
        mock_tool = MagicMock()
        mock_tool.name = "calculator"
        agent.tools = [mock_tool]
        client = TestClient(create_app(agent, {agent.name: agent}))
        body = client.get(f"/agents/{agent.name}").json()
        assert body["tools"] == ["calculator"]

    def test_empty_tools_when_agent_has_none(self, agent: Agent) -> None:
        agent.tools = []
        client = TestClient(create_app(agent, {agent.name: agent}))
        body = client.get(f"/agents/{agent.name}").json()
        assert body["tools"] == []


# ---------------------------------------------------------------------------
# create_app — POST /run
# ---------------------------------------------------------------------------


class TestRunEndpoint:
    def test_returns_200(self, agent: Agent) -> None:
        response_event = BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author=agent.name,
            content=[Part(text="Done!")],
            is_partial=False,
        )
        client = TestClient(create_app(agent, {agent.name: agent}))
        with _fake_runner(response_event):
            resp = client.post("/run", json={"text_input": "hello"})
        assert resp.status_code == 200

    def test_response_contains_agent_text(self, agent: Agent) -> None:
        response_event = BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author=agent.name,
            content=[Part(text="Hello from agent!")],
            is_partial=False,
        )
        client = TestClient(create_app(agent, {agent.name: agent}))
        with _fake_runner(response_event):
            body = client.post("/run", json={"text_input": "hi"}).json()
        assert body["response"] == "Hello from agent!"

    def test_response_includes_agent_name(self, agent: Agent) -> None:
        response_event = BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author=agent.name,
            content=[Part(text="ok")],
            is_partial=False,
        )
        client = TestClient(create_app(agent, {agent.name: agent}))
        with _fake_runner(response_event):
            body = client.post("/run", json={"text_input": "ping"}).json()
        assert body["agent_name"] == agent.name

    def test_partial_events_are_excluded_from_response(self, agent: Agent) -> None:
        partial_event = BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author=agent.name,
            content=[Part(text="partial chunk")],
            is_partial=True,
        )
        final_event = BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author=agent.name,
            content=[Part(text="final answer")],
            is_partial=False,
        )
        client = TestClient(create_app(agent, {agent.name: agent}))
        with _fake_runner(partial_event, final_event):
            body = client.post("/run", json={"text_input": "hi"}).json()
        assert body["response"] == "final answer"

    def test_empty_response_when_no_response_events(self, agent: Agent) -> None:
        tool_event = BaseEvent(
            event_type=ExecutionEventType.TOOL_START,
            author=agent.name,
        )
        client = TestClient(create_app(agent, {agent.name: agent}))
        with _fake_runner(tool_event):
            body = client.post("/run", json={"text_input": "hi"}).json()
        assert body["response"] == ""

    def test_conversation_id_echoed_back(self, agent: Agent) -> None:
        response_event = BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author=agent.name,
            content=[Part(text="ok")],
            is_partial=False,
        )
        client = TestClient(create_app(agent, {agent.name: agent}))
        with _fake_runner(response_event):
            body = client.post(
                "/run", json={"text_input": "hi", "conversation_id": "conv_abc123"}
            ).json()
        assert body["conversation_id"] == "conv_abc123"


# ---------------------------------------------------------------------------
# create_app — POST /execute (SSE)
# ---------------------------------------------------------------------------


class TestExecuteEndpoint:
    def test_returns_200_with_event_stream_content_type(self, agent: Agent) -> None:
        response_event = BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author=agent.name,
            content=[Part(text="streaming!")],
            is_partial=False,
        )
        client = TestClient(create_app(agent, {agent.name: agent}))
        with _fake_runner(response_event):
            resp = client.post("/execute", json={"text_input": "stream me"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_response_body_contains_sse_data_frames(self, agent: Agent) -> None:
        response_event = BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author=agent.name,
            content=[Part(text="chunk")],
            is_partial=False,
        )
        client = TestClient(create_app(agent, {agent.name: agent}))
        with _fake_runner(response_event):
            body = client.post("/execute", json={"text_input": "go"}).text
        assert "data:" in body

    def test_stream_ends_with_end_event(self, agent: Agent) -> None:
        client = TestClient(create_app(agent, {agent.name: agent}))
        with _fake_runner():
            body = client.post("/execute", json={"text_input": "go"}).text
        assert "event: end" in body

    def test_stream_contains_error_event_on_runner_exception(self, agent: Agent) -> None:
        async def _run_raises(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("runner failed")
            yield  # make it an async generator

        mock_instance = MagicMock()
        mock_instance.run = _run_raises

        client = TestClient(create_app(agent, {agent.name: agent}))
        with patch("domyn.agents.cli.commands.serve.server.Runner", return_value=mock_instance):
            body = client.post("/execute", json={"text_input": "fail"}).text

        assert ExecutionEventType.ERROR_OCCURRED.value in body


# ---------------------------------------------------------------------------
# _tool_refresh_loop
# ---------------------------------------------------------------------------


async def _until_calls(mock: AsyncMock, n: int, timeout: float = 5.0) -> None:
    """Wait until ``mock`` has been awaited ``n`` times, rather than for a fixed time.

    A failed refresh logs through rich, and the first print in a process loads
    ``rich._emoji_codes``: ~2 ms normally, ~240 ms under coverage, all of it
    blocking the loop. A fixed 50 ms window ended inside it on any worker that
    had not printed yet — with ``-n auto`` on a 14-core machine, most of them.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while mock.await_count < n:
        assert asyncio.get_running_loop().time() < deadline, f"{mock.await_count} calls"
        await asyncio.sleep(0.01)


class TestToolRefreshLoop:
    @pytest.mark.asyncio
    async def test_calls_refresh_with_expected_kwargs(self) -> None:
        adapter = MagicMock()
        adapter.refresh_delegate_tools = AsyncMock(return_value=None)

        task = asyncio.create_task(
            _tool_refresh_loop(
                adapter,
                0.01,
                api_key="key",
                space_id="space",
                channel_id="chan",
                base_url="https://platform.example.com",
                configuration_id="cfg",
                attach_new_tools=True,
            )
        )
        try:
            await _until_calls(adapter.refresh_delegate_tools, 1)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        adapter.refresh_delegate_tools.assert_awaited_with(
            api_key="key",
            space_id="space",
            channel_id="chan",
            base_url="https://platform.example.com",
            configuration_id="cfg",
            attach_new_tools=True,
        )

    @pytest.mark.asyncio
    async def test_survives_refresh_failure(self) -> None:
        """A failing refresh tick must not stop the loop from trying again."""
        adapter = MagicMock()
        adapter.refresh_delegate_tools = AsyncMock(side_effect=RuntimeError("boom"))

        task = asyncio.create_task(
            _tool_refresh_loop(
                adapter,
                0.01,
                api_key="key",
                space_id=None,
                channel_id=None,
                base_url=None,
                configuration_id=None,
                attach_new_tools=True,
            )
        )
        try:
            # Still running and still calling despite every call raising.
            await _until_calls(adapter.refresh_delegate_tools, 2)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


class TestPlatformToolsDefault:
    """Platform tools are on when the platform SDK is installed, unless the flag says otherwise."""

    @staticmethod
    def _expose(attach: bool | None) -> bool:
        from domyn.agents.cli.commands import expose_command

        with patch("domyn.agents.cli.commands.serve.command._serve_relay") as serve_relay:
            expose_command(
                "mod:sym",
                channel_id="c",
                space_id="s",
                configuration_id=None,
                base_url="h",
                target=None,
                max_concurrency=8,
                framework=None,
                attach_platform_tools=attach,
                tool_refresh_interval=0.0,
            )
        return bool(serve_relay.call_args.kwargs["attach_platform_tools"])

    @pytest.mark.parametrize(
        "installed,asked,expected",
        [
            (True, None, True),
            (False, None, False),
            (True, False, False),
            (False, True, True),
            (True, True, True),
            (False, False, False),
        ],
    )
    def test_expose(self, installed: bool, asked: bool | None, expected: bool) -> None:
        with patch(
            "domyn.agents.integrations.domyn_platform_sdk.sdk_installed", return_value=installed
        ):
            assert self._expose(asked) is expected
