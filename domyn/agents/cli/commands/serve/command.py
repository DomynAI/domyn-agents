"""domyn serve — serve an agent via REST or relay a runnable to the Domyn relay.

Channel values
--------------
  rest   — Start a local FastAPI/uvicorn server (default when a .py/.yaml file is given).
  relay  — Connect outbound to the Domyn relay WebSocket and process invoke requests.
           Requires DOMYN_API_KEY, --channel-id, --space-id, --base-url (or env vars).
           Accepts either a .py/.yaml agent file OR a 'module:symbol' runnable entrypoint.
           --target may be repeated to serve the SAME agent (one shared adapter/
           conversation state) on several relay channels — even across different
           spaces/relay deployments — at once; one WebSocket connection is opened per
           --target. Each value is a comma-separated 'channel_id=...,space_id=...,
           base_url=...,configuration_id=...' set, e.g.::

               --target "channel_id=a,space_id=s1,base_url=relay1.example.com"
               --target "channel_id=b,space_id=s2,base_url=relay2.example.com,configuration_id=cfg2"

           'channel_id' is required; 'space_id'/'base_url' fall back to
           --space-id/--base-url when omitted. 'configuration_id' is per-space: it is
           NOT inherited from --configuration-id unless the target also omits
           'space_id' (the shared --configuration-id belongs to the shared/default
           space). When --target is omitted entirely, exactly one relay connection is
           built from the plain --channel-id/--space-id/--base-url/--configuration-id
           options (or their env vars) — the original single-target behavior.
           The adapter is auto-detected:
             * Agent instance          → DomynAgentAdapter
             * object with .ainvoke    → LangGraphAdapter
             * zero-arg async callable → LangGraphAdapter (called once at startup)
  multi  — Run several InputChannel implementations concurrently against ONE shared
           adapter (same agent, same Runner/conversation state, whichever channel an
           AGENT_START arrives on gets the reply). Requires a .yaml/.yml agent file with
           a top-level ``serve.channels`` list, e.g.::

               entrypoint: MyAgent
               serve:
                 channels:
                   - type: relay
                     channel_id: my-channel   # optional, falls back to DOMYN_CHANNEL_ID
                     space_id: my-space       # optional, falls back to DOMYN_SPACE_ID
                     base_url: relay.example.com  # optional, falls back to DOMYN_BASE_URL
                   - type: telegram
                     token_env: TELEGRAM_BOT_TOKEN   # env var holding the bot token
                     transcriber: {type: local}       # voice notes to text (audio extra)
                   - type: slack
                     bot_token_env: SLACK_BOT_TOKEN
                     app_token_env: SLACK_APP_TOKEN
                   - type: whatsapp
                     token_env: WHATSAPP_TOKEN
                     phone_number_id_env: WHATSAPP_PHONE_ID
                     verify_token_env: WHATSAPP_VERIFY_TOKEN

               agents:
                 - name: MyAgent
                   ...

           ``serve.transcriber`` (beside ``channels``) gives every channel that names
           none of its own a transcriber for voice notes; without one, the agent is
           told a voice message arrived that it cannot read.

           Then simply: ``domyn serve agent.yaml --channel multi``.
"""

import asyncio
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, cast

import typer
import yaml
from rich.console import Console

from domyn.agents.cli.commands.run.command import (
    collect_all_agents,
    find_main_agent,
    load_agent_from_yaml,
    load_module_from_path,
)
from domyn.agents.cli.commands.serve.server import ServerInterface
from domyn.agents.integrations.domyn_platform_sdk import (
    PlatformSDKNotInstalledError,
    platform_tools_enabled,
)
from domyn.agents.logger import get_logger

logger = get_logger()
console = Console()

# ---------------------------------------------------------------------------
# Helpers shared with expose_command (which becomes an alias below)
# ---------------------------------------------------------------------------


_TARGET_SPEC_KEYS = {"channel_id", "space_id", "base_url", "configuration_id"}


def _parse_target_spec(spec: str) -> dict[str, str]:
    """Parse a --target value: comma-separated 'key=value' pairs.

    'channel_id' is required; 'space_id', 'base_url', 'configuration_id' are
    optional and, when omitted, fall back to the shared --space-id/--base-url/
    --configuration-id, e.g.::

        channel_id=a                                              shared space/base-url
        channel_id=a,space_id=s1                                  own space, shared base-url
        channel_id=b,space_id=s2,base_url=relay2.example.com      own space AND base-url
    """
    fields: dict[str, str] = {}
    for part in spec.split(","):
        key, sep, value = part.partition("=")
        if not sep or key not in _TARGET_SPEC_KEYS:
            raise ValueError(
                f"invalid --target field {part!r} in {spec!r}; expected "
                f"'key=value' with key in {sorted(_TARGET_SPEC_KEYS)}"
            )
        fields[key] = value
    if "channel_id" not in fields:
        raise ValueError(f"--target {spec!r} is missing the required 'channel_id=...' field")
    return fields


def _load_symbol(entrypoint: str) -> Any:
    """Load 'module:symbol' or 'path/to/file.py:symbol'."""
    if ":" not in entrypoint:
        raise ValueError(f"entrypoint must be 'module:symbol', got: {entrypoint!r}")

    module_path, _, symbol = entrypoint.rpartition(":")

    if module_path.endswith(".py") or os.sep in module_path or "/" in module_path:
        abs_path = os.path.abspath(module_path)
        spec = importlib.util.spec_from_file_location("_domyn_serve_module", abs_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from path: {abs_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_domyn_serve_module"] = mod
        spec.loader.exec_module(mod)
    else:
        cwd = os.getcwd()
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
        mod = importlib.import_module(module_path)

    obj = mod
    for attr in symbol.split("."):
        obj = getattr(obj, attr)
    return obj


# ---------------------------------------------------------------------------
# serve command
# ---------------------------------------------------------------------------


def serve_command(
    entrypoint: str = typer.Argument(
        ...,
        help=(
            "For --channel rest: path to the .py or .yaml agent file.  "
            "For --channel relay: 'module:symbol' or 'path/file.py:symbol'."
        ),
        metavar="FILE_OR_ENTRYPOINT",
    ),
    channel: str = typer.Option(
        "rest",
        "--channel",
        "-c",
        help="Transport channel: rest | relay | multi",
    ),
    # ---- REST options ----
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host (rest only, default: 0.0.0.0)"),
    port: int = typer.Option(8000, "--port", help="Bind port (rest only, default: 8000)"),
    agent_name: str | None = typer.Option(
        None,
        "--agent-name",
        help="Agent to serve; auto-detected when omitted (rest only)",
    ),
    # ---- relay options ----
    channel_id: str | None = typer.Option(
        None, "--channel-id", envvar="DOMYN_CHANNEL_ID", help="Relay channel ID (relay only)"
    ),
    space_id: str | None = typer.Option(
        None, "--space-id", envvar="DOMYN_SPACE_ID", help="Relay space ID (relay only)"
    ),
    configuration_id: str | None = typer.Option(
        None,
        "--configuration-id",
        envvar="DOMYN_CONFIGURATION_ID",
        help="Relay configuration ID (relay only, optional)",
    ),
    base_url: str | None = typer.Option(
        None, "--base-url", envvar="DOMYN_BASE_URL", help="Relay server hostname (relay only)"
    ),
    target: list[str] = typer.Option(
        [],
        "--target",
        help=(
            "Relay target (relay only). Repeatable — each one is a comma-separated "
            "'channel_id=...,space_id=...,base_url=...,configuration_id=...' set, "
            "letting one process serve several relay channels (even across different "
            "spaces/relay deployments) at once. 'channel_id' is required; 'space_id'/"
            "'base_url' fall back to --space-id/--base-url when omitted. "
            "'configuration_id' is optional and per-space — NOT inherited from "
            "--configuration-id unless the target also omits 'space_id'. When omitted "
            "entirely, falls back to a single target built from "
            "--channel-id/--space-id/--base-url/--configuration-id. Example: "
            "--target 'channel_id=a,space_id=s1,base_url=relay1.example.com'."
        ),
    ),
    max_concurrency: int = typer.Option(
        8,
        "--max-concurrency",
        help="Max concurrent in-flight invocations (relay only, default: 8)",
        min=1,
    ),
    attach_platform_tools: bool | None = typer.Option(
        None,
        "--platform-tools/--no-platform-tools",
        help=(
            "Attach every undeclared platform tool to the entrypoint agent "
            "(relay/multi only; default: on when domyn-platform-sdk is installed, off otherwise)"
        ),
    ),
    tool_refresh_interval: float = typer.Option(
        0.0,
        "--tool-refresh-interval",
        help=(
            "Re-run platform tool discovery every N seconds while the process "
            "runs (relay/multi only; requires platform tools; default: 0 = disabled)"
        ),
        min=0.0,
    ),
) -> None:
    r"""Serve an agent via a REST API or the Domyn relay.

    Examples::

        domyn serve agent.py                          # REST on :8000
        domyn serve agent.py --port 9000              # REST on :9000
        domyn serve agent.py:my_agent --channel relay \
            --channel-id <id> --space-id <id> --base-url <host>
    """
    ch = channel.lower()
    attach_platform_tools = platform_tools_enabled(attach_platform_tools)

    if ch == "rest":
        _serve_rest(entrypoint, host=host, port=port, agent_name=agent_name)
    elif ch == "relay":
        _serve_relay(
            entrypoint,
            channel_id=channel_id,
            space_id=space_id,
            configuration_id=configuration_id,
            base_url=base_url,
            target=target,
            max_concurrency=max_concurrency,
            attach_platform_tools=attach_platform_tools,
            tool_refresh_interval=tool_refresh_interval,
        )
    elif ch == "multi":
        _serve_multi(
            entrypoint,
            max_concurrency=max_concurrency,
            attach_platform_tools=attach_platform_tools,
            tool_refresh_interval=tool_refresh_interval,
        )
    else:
        typer.echo(f"Error: Unknown channel '{channel}'. Choose: rest, relay, multi", err=True)
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# connect_command — `serve --channel relay` (`expose` is kept as an alias)
# ---------------------------------------------------------------------------


def connect_command(
    entrypoint: str = typer.Argument(
        ...,
        help="Runnable entrypoint as 'module:symbol' or 'path/to/file.py:symbol'",
        metavar="MODULE:SYMBOL",
    ),
    channel_id: str | None = typer.Option(
        None, "--channel-id", envvar="DOMYN_CHANNEL_ID", help="Relay channel ID"
    ),
    space_id: str | None = typer.Option(
        None, "--space-id", envvar="DOMYN_SPACE_ID", help="Relay space ID"
    ),
    configuration_id: str | None = typer.Option(
        None,
        "--configuration-id",
        envvar="DOMYN_CONFIGURATION_ID",
        help="Relay configuration ID",
    ),
    base_url: str | None = typer.Option(
        None, "--base-url", envvar="DOMYN_BASE_URL", help="Relay server hostname"
    ),
    target: list[str] = typer.Option(
        [],
        "--target",
        help=(
            "Relay target. Repeatable — each one is a comma-separated "
            "'channel_id=...,space_id=...,base_url=...,configuration_id=...' set, "
            "letting one process serve several relay channels (even across different "
            "spaces/relay deployments) at once. 'channel_id' is required; 'space_id'/"
            "'base_url' fall back to --space-id/--base-url when omitted. "
            "'configuration_id' is optional and per-space — NOT inherited from "
            "--configuration-id unless the target also omits 'space_id'. When omitted "
            "entirely, falls back to a single target built from "
            "--channel-id/--space-id/--base-url/--configuration-id. Example: "
            "--target 'channel_id=a,space_id=s1,base_url=relay1.example.com'."
        ),
    ),
    max_concurrency: int = typer.Option(
        8,
        "--max-concurrency",
        help="Max concurrent in-flight invocations before WorkerBusy rejection",
        min=1,
    ),
    # --framework kept for backward compat but ignored (auto-detected)
    framework: str | None = typer.Option(
        None,
        "--framework",
        hidden=True,
        help="Deprecated — adapter is auto-detected from the entrypoint symbol.",
    ),
    attach_platform_tools: bool | None = typer.Option(
        None,
        "--platform-tools/--no-platform-tools",
        help=(
            "Attach every undeclared platform tool to the entrypoint agent "
            "(default: on when domyn-platform-sdk is installed, off otherwise)"
        ),
    ),
    tool_refresh_interval: float = typer.Option(
        0.0,
        "--tool-refresh-interval",
        help=(
            "Re-run platform tool discovery every N seconds while the process "
            "runs (requires --platform-tools; default: 0 = disabled)"
        ),
        min=0.0,
    ),
) -> None:
    """Connect a runnable to the Domyn platform as a relay worker (same as serve --channel relay).

    Also available as `domyn expose`.

    The DOMYN_API_KEY environment variable must be set.
    """
    if framework is not None:
        console.print(
            "[yellow]--framework is deprecated and ignored; the adapter is now "
            "auto-detected from the entrypoint symbol.[/yellow]"
        )
    _serve_relay(
        entrypoint,
        channel_id=channel_id,
        space_id=space_id,
        configuration_id=configuration_id,
        base_url=base_url,
        target=target,
        max_concurrency=max_concurrency,
        attach_platform_tools=platform_tools_enabled(attach_platform_tools),
        tool_refresh_interval=tool_refresh_interval,
    )


expose_command = connect_command


# ---------------------------------------------------------------------------
# Internal implementations
# ---------------------------------------------------------------------------


def _serve_rest(
    file: str,
    *,
    host: str,
    port: int,
    agent_name: str | None,
) -> None:
    path = Path(file)
    try:
        if path.suffix in (".yaml", ".yml"):
            main_agent = asyncio.run(load_agent_from_yaml(str(path)))
            all_agents = {main_agent.name: main_agent}
        else:
            module = load_module_from_path(str(path))
            all_agents = collect_all_agents(module)
            main_agent = find_main_agent(all_agents, agent_name)
    except Exception as e:
        typer.echo(f"Error loading agent from {file}: {e}", err=True)
        raise typer.Exit(1) from e

    typer.echo(f"Starting REST server for agent '{main_agent.name}' on {host}:{port}")
    try:
        server = ServerInterface(main_agent, all_agents)
        asyncio.run(server.start_async(host=host, port=port))
    except KeyboardInterrupt:
        typer.echo("\nServer stopped.")
    except Exception as e:
        typer.echo(f"Error starting server: {e}", err=True)
        raise typer.Exit(1) from e


async def _tool_refresh_loop(
    adapter: Any,
    interval: float,
    *,
    api_key: str,
    space_id: str | None,
    channel_id: str | None,
    base_url: str | None,
    configuration_id: str | None,
    attach_new_tools: bool,
) -> None:
    """Periodically re-run platform tool discovery against a live adapter."""
    while True:
        await asyncio.sleep(interval)
        try:
            await adapter.refresh_delegate_tools(
                api_key=api_key,
                space_id=space_id,
                channel_id=channel_id,
                base_url=base_url,
                configuration_id=configuration_id,
                attach_new_tools=attach_new_tools,
            )
        except Exception as exc:
            console.print(
                f"[yellow]WARNING:[/yellow] scheduled platform tool refresh "
                f"failed: {type(exc).__name__}"
            )


def _serve_relay(
    entrypoint: str,
    *,
    channel_id: str | None,
    space_id: str | None,
    configuration_id: str | None,
    base_url: str | None,
    target: list[str] | None,
    max_concurrency: int,
    attach_platform_tools: bool = False,
    tool_refresh_interval: float = 0.0,
) -> None:
    api_key = os.environ.get("DOMYN_API_KEY")
    if not api_key:
        console.print("[red]ERROR:[/red] DOMYN_API_KEY environment variable is not set.")
        raise typer.Exit(1)

    # Propagate CLI values into env so DelegateTool / platform helpers can read them.
    if channel_id is not None:
        os.environ["DOMYN_CHANNEL_ID"] = channel_id
    if space_id is not None:
        os.environ["DOMYN_SPACE_ID"] = space_id
    if configuration_id is not None:
        os.environ["DOMYN_CONFIGURATION_ID"] = configuration_id
    if base_url is not None:
        os.environ["DOMYN_BASE_URL"] = base_url

    eff_channel_id = channel_id or os.environ.get("DOMYN_CHANNEL_ID")
    eff_space_id = space_id or os.environ.get("DOMYN_SPACE_ID")
    eff_base_url = base_url or os.environ.get("DOMYN_BASE_URL", "")
    eff_cfg_id = configuration_id or os.environ.get("DOMYN_CONFIGURATION_ID")

    try:
        targets = [_parse_target_spec(spec) for spec in target] if target else None
    except ValueError as exc:
        console.print(f"[red]ERROR:[/red] {exc}")
        raise typer.Exit(1) from exc

    # No --target at all: fall back to the original single-target behavior, built
    # entirely from --channel-id/--space-id/--base-url/--configuration-id.
    if not targets:
        targets = [{"channel_id": eff_channel_id or ""}]

    from domyn.agents.channels import DomynRelayChannel
    from domyn.agents.server import AgentServer

    # Each target resolves its own space_id/base_url/configuration_id: a field set
    # on the target wins, otherwise it falls back to the shared value — EXCEPT
    # configuration_id, which is a per-space concept: the shared --configuration-id
    # belongs to the shared/default space, so it must not leak onto a target that
    # pins a *different* space_id (that target gets None — the platform's latest
    # release for its own space — unless it also sets its own configuration_id).
    resolved = [
        (
            fields["channel_id"],
            fields.get("space_id") or eff_space_id or "",
            fields.get("base_url") or eff_base_url,
            fields.get("configuration_id") or (eff_cfg_id if "space_id" not in fields else None),
        )
        for fields in targets
    ]

    try:
        channel_objs = [
            DomynRelayChannel.from_env(
                api_key=api_key, channel_id=cid, space_id=sp, base_url=bu, configuration_id=cfg
            )
            for cid, sp, bu, cfg in resolved
        ]
    except Exception as exc:
        # The message of a missing SDK already says what to install; anything
        # else is named by type, with the detail left to debug.
        if isinstance(exc, PlatformSDKNotInstalledError):
            console.print(f"[red]ERROR:[/red] {exc}", soft_wrap=True)
        else:
            console.print(
                f"[red]ERROR:[/red] could not set up the relay connection: {type(exc).__name__}"
            )
            logger.debug(f"relay connection setup failed, detail: {exc}")
        raise typer.Exit(1) from None

    console.print(f"[bold]domyn connect[/bold] {entrypoint}")
    for (cid, sp, bu, cfg), channel_obj in zip(resolved, channel_objs, strict=True):
        console.print(
            f"  relay           : {channel_obj._ws_url}  "
            f"(channel-id: {cid}, space-id: {sp}, base-url: {bu}, configuration-id: {cfg})"
        )
    console.print(f"  max-concurrency : {max_concurrency}")
    console.print("Press Ctrl+C to stop.")

    # Load object — can be a .py/.yaml file (Agent) or module:symbol (anything)
    path = Path(entrypoint)
    obj: Any

    if ":" in entrypoint:
        try:
            obj = _load_symbol(entrypoint)
        except Exception as exc:
            console.print(f"[red]ERROR:[/red] failed to load entrypoint: {exc}")
            raise typer.Exit(1) from exc
    elif path.suffix in (".yaml", ".yml"):
        try:
            obj = asyncio.run(load_agent_from_yaml(str(path)))
        except Exception as exc:
            console.print(f"[red]ERROR:[/red] failed to load YAML: {exc}")
            raise typer.Exit(1) from exc
    else:
        try:
            module = load_module_from_path(str(path))
            all_agents = collect_all_agents(module)
            obj = find_main_agent(all_agents, None)
        except Exception as exc:
            console.print(f"[red]ERROR:[/red] failed to load file: {exc}")
            raise typer.Exit(1) from exc

    # Auto-detect adapter — ONE shared adapter/conversation state across all channels.
    first_cid, first_space, first_base_url, first_cfg = resolved[0]
    adapter = _build_adapter(
        obj,
        max_concurrency,
        api_key,
        first_cid,
        first_space,
        first_base_url,
        first_cfg,
        attach_new_tools=attach_platform_tools,
    )
    servers = [AgentServer(channel=ch, adapter=adapter) for ch in channel_objs]

    if tool_refresh_interval > 0 and not attach_platform_tools:
        console.print(
            "[yellow]WARNING:[/yellow] --tool-refresh-interval has no effect "
            "without --platform-tools; ignoring."
        )
    refresh_enabled = (
        attach_platform_tools
        and tool_refresh_interval > 0
        and hasattr(adapter, "refresh_delegate_tools")
    )
    if attach_platform_tools and tool_refresh_interval > 0 and not refresh_enabled:
        console.print(
            "[yellow]WARNING:[/yellow] --tool-refresh-interval has no effect for this adapter type."
        )
    if refresh_enabled:
        console.print(f"  tool-refresh    : every {tool_refresh_interval:g}s")

    async def _run_all() -> None:
        tasks = [s.serve() for s in servers]
        if refresh_enabled:
            tasks.append(
                _tool_refresh_loop(
                    adapter,
                    tool_refresh_interval,
                    api_key=api_key,
                    space_id=first_space,
                    channel_id=first_cid,
                    base_url=first_base_url,
                    configuration_id=first_cfg,
                    attach_new_tools=attach_platform_tools,
                )
            )
        try:
            await asyncio.gather(*tasks)
        finally:
            for ch in channel_objs:
                await ch.close()

    try:
        asyncio.run(_run_all())
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/dim]")


def _read_serve_channels_config(path: Path) -> list[dict[str, Any]]:
    """Read the ``serve.channels`` list from a raw YAML agent config file.

    This is independent of ``ConfigurationV1`` (which ignores unknown top-level
    keys like ``serve``) — it just re-reads the same file as plain YAML.
    """
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except Exception as exc:
        console.print(f"[red]ERROR:[/red] failed to parse YAML {path}: {exc}")
        raise typer.Exit(1) from exc

    serve = (raw or {}).get("serve") or {}
    channels = serve.get("channels")
    if not channels or not isinstance(channels, list):
        console.print(
            f"[red]ERROR:[/red] {path} has no 'serve.channels' list. "
            "--channel multi requires a YAML file with a top-level:\n\n"
            "serve:\n  channels:\n    - type: telegram\n      token_env: TELEGRAM_BOT_TOKEN\n    - type: slack\n      ..."
        )
        raise typer.Exit(1)
    # `serve.transcriber` is every channel's, unless an entry names its own
    # (or `transcriber: null` for none).
    shared = serve.get("transcriber")
    if shared is not None:
        channels = [
            entry if "transcriber" in entry else {**entry, "transcriber": shared}
            for entry in channels
            if isinstance(entry, dict)
        ]
    return cast(list[dict[str, Any]], channels)


def _channel_transcriber(entry: dict[str, Any]) -> Any:
    """The entry's ``transcriber``, built, or None; a bad one stops the server
    before it starts rather than at the first voice note."""
    config = entry.get("transcriber")
    if config is None:
        return None
    from domyn.agents.audio import resolve_transcriber

    try:
        return resolve_transcriber(config)
    except (ValueError, TypeError) as exc:
        console.print(
            f"[red]ERROR:[/red] channel '{entry.get('type')}': invalid transcriber: {exc}"
        )
        raise typer.Exit(1) from exc


def _build_channel_from_config(entry: dict[str, Any]) -> Any:
    """Build one ``InputChannel`` from a ``serve.channels[i]`` YAML entry.

    Secrets are read from the environment variable *named* by the entry (e.g.
    ``token_env: TELEGRAM_BOT_TOKEN``), never inlined in the YAML itself.
    """
    channel_type = str(entry.get("type", "")).lower()

    def _env(key: str, default_var: str, required: bool = True) -> str:
        var_name = entry.get(key, default_var)
        value = os.environ.get(var_name, "")
        if required and not value:
            console.print(
                f"[red]ERROR:[/red] channel '{channel_type}': environment variable "
                f"'{var_name}' (from '{key}') is not set."
            )
            raise typer.Exit(1)
        return value

    if channel_type == "relay":
        from domyn.agents.channels import DomynRelayChannel

        api_key = _env("api_key_env", "DOMYN_API_KEY")
        channel_id = entry.get("channel_id") or os.environ.get("DOMYN_CHANNEL_ID", "")
        space_id = entry.get("space_id") or os.environ.get("DOMYN_SPACE_ID", "")
        base_url = entry.get("base_url") or os.environ.get("DOMYN_BASE_URL", "")
        configuration_id = entry.get("configuration_id") or os.environ.get("DOMYN_CONFIGURATION_ID")
        return DomynRelayChannel.from_env(
            api_key=api_key,
            channel_id=channel_id,
            space_id=space_id,
            base_url=base_url,
            configuration_id=configuration_id,
        )

    if channel_type == "telegram":
        try:
            from domyn.agents.channels.telegram import TelegramChannel
        except ImportError as exc:
            console.print(
                "[red]ERROR:[/red] Telegram channel not available. "
                "Install with: pip install 'domyn[telegram]'"
            )
            raise typer.Exit(1) from exc
        token = _env("token_env", "TELEGRAM_BOT_TOKEN")
        return TelegramChannel(
            token=token,
            poll_timeout=entry.get("poll_timeout", 30),
            transcriber=_channel_transcriber(entry),
        )

    if channel_type == "slack":
        try:
            from domyn.agents.channels.slack import SlackChannel
        except ImportError as exc:
            console.print(
                "[red]ERROR:[/red] Slack channel not available. "
                "Install with: pip install 'domyn[slack]'"
            )
            raise typer.Exit(1) from exc
        bot_token = _env("bot_token_env", "SLACK_BOT_TOKEN")
        app_token = _env("app_token_env", "SLACK_APP_TOKEN")
        return SlackChannel(
            bot_token=bot_token, app_token=app_token, transcriber=_channel_transcriber(entry)
        )

    if channel_type == "whatsapp":
        from domyn.agents.channels.whatsapp import WhatsAppChannel

        token = _env("token_env", "WHATSAPP_TOKEN")
        phone_number_id = _env("phone_number_id_env", "WHATSAPP_PHONE_ID")
        verify_token = _env("verify_token_env", "WHATSAPP_VERIFY_TOKEN")
        return WhatsAppChannel(
            token=token,
            phone_number_id=phone_number_id,
            verify_token=verify_token,
            webhook_host=entry.get("webhook_host", "0.0.0.0"),
            webhook_port=entry.get("webhook_port", 8080),
            transcriber=_channel_transcriber(entry),
        )

    console.print(
        f"[red]ERROR:[/red] unknown channel type '{channel_type}' in serve.channels. "
        "Choose: relay, telegram, slack, whatsapp"
    )
    raise typer.Exit(1)


def _serve_multi(
    entrypoint: str,
    *,
    max_concurrency: int,
    attach_platform_tools: bool = False,
    tool_refresh_interval: float = 0.0,
) -> None:
    """Run several channels concurrently against ONE shared adapter.

    Same agent, same ``Runner``/conversation state — whichever channel an
    ``AGENT_START`` arrives on gets the reply on that same channel. See the
    module docstring for the expected ``serve.channels`` YAML shape.
    """
    path = Path(entrypoint)
    if path.suffix not in (".yaml", ".yml"):
        console.print(
            "[red]ERROR:[/red] --channel multi requires a .yaml/.yml agent file "
            "with a top-level 'serve.channels' list (see 'domyn serve --help')."
        )
        raise typer.Exit(1)

    channel_configs = _read_serve_channels_config(path)

    try:
        obj = asyncio.run(load_agent_from_yaml(str(path)))
    except Exception as exc:
        console.print(f"[red]ERROR:[/red] failed to load YAML: {exc}")
        raise typer.Exit(1) from exc

    # Relay credentials (if a 'relay' entry is present) are also used for the
    # one-time DelegateTool schema population below; harmless/no-op otherwise
    # since _build_adapter skips it when api_key is empty.
    relay_entry = next(
        (c for c in channel_configs if str(c.get("type", "")).lower() == "relay"), None
    )
    api_key = os.environ.get("DOMYN_API_KEY", "")
    eff_channel_id = (relay_entry or {}).get("channel_id") or os.environ.get("DOMYN_CHANNEL_ID", "")
    eff_space_id = (relay_entry or {}).get("space_id") or os.environ.get("DOMYN_SPACE_ID", "")
    eff_base_url = (relay_entry or {}).get("base_url") or os.environ.get("DOMYN_BASE_URL", "")
    eff_cfg_id = (relay_entry or {}).get("configuration_id") or os.environ.get(
        "DOMYN_CONFIGURATION_ID"
    )

    adapter = _build_adapter(
        obj,
        max_concurrency,
        api_key,
        eff_channel_id,
        eff_space_id,
        eff_base_url,
        eff_cfg_id,
        attach_new_tools=attach_platform_tools,
    )

    channels = [_build_channel_from_config(entry) for entry in channel_configs]

    console.print(f"[bold]domyn serve --channel multi[/bold] {entrypoint}")
    for entry, ch in zip(channel_configs, channels, strict=True):
        console.print(f"  channel         : {entry.get('type')} ({type(ch).__name__})")
        transcriber = getattr(ch, "transcriber", None)
        if transcriber is not None:
            ready = "" if transcriber.is_ready() else ", model downloaded on the first voice note"
            console.print(f"    voice notes   : {transcriber.describe()}{ready}")
    console.print(f"  max-concurrency : {max_concurrency}")
    console.print("Press Ctrl+C to stop.")

    from domyn.agents.server import AgentServer

    servers = [AgentServer(channel=ch, adapter=adapter) for ch in channels]

    if tool_refresh_interval > 0 and not attach_platform_tools:
        console.print(
            "[yellow]WARNING:[/yellow] --tool-refresh-interval has no effect "
            "without --platform-tools; ignoring."
        )
    refresh_enabled = (
        attach_platform_tools
        and tool_refresh_interval > 0
        and hasattr(adapter, "refresh_delegate_tools")
    )
    if attach_platform_tools and tool_refresh_interval > 0 and not refresh_enabled:
        console.print(
            "[yellow]WARNING:[/yellow] --tool-refresh-interval has no effect for this adapter type."
        )
    if refresh_enabled:
        console.print(f"  tool-refresh    : every {tool_refresh_interval:g}s")

    async def _run_all() -> None:
        tasks = [s.serve() for s in servers]
        if refresh_enabled:
            tasks.append(
                _tool_refresh_loop(
                    adapter,
                    tool_refresh_interval,
                    api_key=api_key,
                    space_id=eff_space_id,
                    channel_id=eff_channel_id,
                    base_url=eff_base_url,
                    configuration_id=eff_cfg_id,
                    attach_new_tools=attach_platform_tools,
                )
            )
        try:
            await asyncio.gather(*tasks)
        finally:
            for ch in channels:
                close = getattr(ch, "close", None)
                if close is not None:
                    await close()

    try:
        asyncio.run(_run_all())
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/dim]")


def _build_adapter(
    obj: Any,
    max_concurrency: int,
    api_key: str,
    channel_id: str | None,
    space_id: str | None,
    base_url: str,
    configuration_id: str | None,
    attach_new_tools: bool = True,
) -> Any:
    """Auto-detect and build the right FrameworkAdapter for *obj*."""
    from domyn.agents.agents import Agent

    if isinstance(obj, Agent):
        from domyn.agents.adapters import DomynAgentAdapter

        if api_key:
            from domyn.agents.integrations.domyn import populate_delegate_tool_schemas

            asyncio.run(
                populate_delegate_tool_schemas(
                    root=obj,
                    api_key=api_key,
                    space_id=space_id,
                    channel_id=channel_id,
                    base_url=base_url,
                    configuration_id=configuration_id,
                    attach_new_tools=attach_new_tools,
                )
            )
        console.print(f"  adapter         : DomynAgentAdapter (agent={obj.name!r})")
        return DomynAgentAdapter(
            root=obj, application_name=obj.name, max_concurrency=max_concurrency
        )

    # LangGraph: has .ainvoke or is a zero-arg callable that produces something with .ainvoke
    if callable(obj) and not hasattr(obj, "ainvoke"):
        # factory — call once to get the compiled graph
        try:
            obj = obj()
        except Exception as exc:
            console.print(f"[red]ERROR:[/red] calling factory {obj!r} raised: {exc}")
            raise typer.Exit(1) from exc

    if hasattr(obj, "ainvoke"):
        try:
            from domyn.agents.adapters import LangGraphAdapter
            from domyn.agents.integrations.langgraph import get_default_relay
            from domyn.agents.integrations.langgraph.utils import get_input_mapper
        except ImportError as exc:
            console.print(
                "[red]ERROR:[/red] LangGraph integration not available. "
                "Install with: pip install 'domyn[langgraph]'"
            )
            raise typer.Exit(1) from exc

        input_mapper = get_input_mapper(obj)
        relay = get_default_relay()
        console.print("  adapter         : LangGraphAdapter")
        return LangGraphAdapter(
            runnable=obj,
            input_mapper=input_mapper,
            platform_relay=relay,
            max_concurrency=max_concurrency,
        )

    console.print(
        f"[red]ERROR:[/red] cannot auto-detect adapter for {type(obj).__name__!r}. "
        "The symbol must be an Agent instance or a LangGraph compiled graph (has .ainvoke)."
    )
    raise typer.Exit(1)
