"""Platform relay client: tool discovery and WebSocket connection management."""

import asyncio
import re
import threading
import time
from collections.abc import Callable
from typing import Any

from hermes_cli import managed_scope
from hermes_cli.config import is_managed, load_config, save_config

from domyn.agents.integrations.domyn_platform_sdk import (
    PlatformSettings,
    SdkToolCatalog,
    run_sync,
    sandbox_auto_approve,
)
from domyn.agents.logger import get_logger

logger = get_logger()

# Toolset our own canvas tools are registered under (``ctx.register_tool``).
_PLATFORM_TOOLSET = "platform"


def hermes_tool_names(exclude_toolset: str = _PLATFORM_TOOLSET) -> set[str]:
    """Return the tool names hermes claims — the conflict set for canvas tools.

    Two sources, unioned:

    * ``toolsets.TOOLSETS`` — the static built-in declaration, and the only one
      that carries weight at plugin-load time. Built-ins self-register when
      ``model_tools`` is imported, but the gateway runs plugin discovery
      *before* it ever imports ``model_tools`` (it lazy-imports ``run_agent``
      per request), so the live registry is still empty of built-ins here.
      ``toolsets`` is pure data — importing it triggers no tool-module imports.
    * the live registry — MCP servers and other plugins, which appear after
      boot and so matter on refresh ticks rather than at startup.

    Names in *exclude_toolset* (our own canvas tools) are dropped. Once we have
    registered a canvas tool, reporting its name back as reserved would make the
    platform rename it on the next refresh — and again on every tick after that.

    The static half is deliberately conservative: it reserves names hermes *will*
    claim once the tool modules import, not only the ones registered right now.

    Never raises: a missing hermes package yields an empty set, i.e. the same
    behavior as before the platform learned about reserved names.
    """
    names: set[str] = set()

    try:
        from toolsets import get_all_toolsets, resolve_toolset

        for toolset_name in get_all_toolsets():
            names.update(resolve_toolset(toolset_name))
    except Exception as exc:
        logger.warning(
            "platform-gateway: could not enumerate hermes toolsets - %s", type(exc).__name__
        )
        logger.debug("platform-gateway: toolset enumeration detail: %s", exc)

    tool_to_toolset: dict[str, str] = {}
    try:
        from tools.registry import registry

        tool_to_toolset = registry.get_tool_to_toolset_map()
    except Exception as exc:
        logger.debug("platform-gateway: tool registry unavailable: %s", exc)

    names.update(tool_to_toolset)
    names.difference_update(
        name for name, toolset in tool_to_toolset.items() if toolset == exclude_toolset
    )
    return names


def _sanitized_group_prefix(group_name: str | None) -> str:
    """Sanitize *group_name* into a safe tool-name prefix; ``"domyn"`` if empty/unusable."""
    prefix = re.sub(r"[^0-9a-zA-Z_-]", "_", group_name or "").strip("_")
    return prefix or "domyn"


def rename_colliding_tools(
    raw_tools: list[dict[str, Any]],
    reserved_tool_names: set[str],
    previous_mapping: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Rename tools whose names are already claimed, and report what was renamed.

    Returns copies of the tools with their names applied, plus a mapping of
    exposed name -> original platform name; *raw_tools* is left untouched. A name
    is "claimed" if hermes reserves it, if an earlier tick handed it out to a
    different tool, or if a sibling in this batch holds it — the algorithm makes
    no distinction between the three.

    Pass *previous_mapping* — the mapping returned by the last call — to keep
    assignments stable across refresh ticks. Without it, an unrelated tool
    joining the batch can take a name a live tool is already exposed under,
    silently repointing a name the model has been calling. Assignments are keyed
    on the original name, which identifies a tool within a tick because the
    platform disambiguates duplicates before serialising the batch.

    Deliberate asymmetry: a tool that once yielded keeps its suffixed name even
    if the name it lost later frees up. Stability beats prettiness — a rename is
    only ever forced by a genuine collision.

    The passes below read from *originals* and write to *renamed*, so each one
    sees the platform's name for a tool rather than a name an earlier pass
    rewrote — the ordering between them is load-bearing (see pass 2).
    """
    originals = [t for t in raw_tools if t.get("name")]
    # Shallow: only the top-level "name" is ever reassigned, and nothing downstream
    # writes through the nested "parameters" (``convert_schema`` only reads it).
    renamed = [dict(t) for t in originals]

    # original name -> the exposed name previously issued for it.
    prior = {original: exposed for exposed, original in (previous_mapping or {}).items()}

    taken = set(reserved_tool_names)
    claimed: set[int] = set()
    rename_mapping: dict[str, str] = {}

    # Pass 1 — honour names already handed out on an earlier tick.
    for i, tool in enumerate(originals):
        exposed = prior.get(tool["name"])
        # ``not in taken``: hermes may have reserved the name since we issued it.
        if exposed is not None and exposed not in taken:
            renamed[i]["name"] = exposed
            rename_mapping[exposed] = tool["name"]
            taken.add(exposed)
            claimed.add(i)

    # Pass 2 — tools that can simply keep their own name; the copy already has it.
    # ``taken`` is seeded with the reserved names, so this one membership test covers
    # both "hermes owns it" and "something in this batch has claimed it". Must finish
    # before pass 3 starts: otherwise a rename candidate computed for an early tool
    # could take a name that a tool further down the batch is entitled to keep.
    for i, tool in enumerate(originals):
        if i not in claimed and tool["name"] not in taken:
            taken.add(tool["name"])
            claimed.add(i)

    # Pass 3 — whatever is left needs a fresh, unique name.
    for i, tool in enumerate(originals):
        if i in claimed:
            continue
        name = tool["name"]
        # Group-prefix, unless the name already carries that prefix — in which case
        # the numeric suffix below is the only thing that can disambiguate it.
        prefix = _sanitized_group_prefix(tool.get("group_name"))
        base = name if name.startswith(f"{prefix}_") else f"{prefix}_{name}"
        new_name = base
        suffix = 1
        while new_name in taken:
            new_name = f"{base}_{suffix}"
            suffix += 1
        renamed[i]["name"] = new_name
        rename_mapping[new_name] = name
        taken.add(new_name)
        claimed.add(i)

    return renamed, rename_mapping


def fetch_tools(
    settings: PlatformSettings,
    agent_name: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch the canvas tool list from the platform, blocking.

    ``settings`` carries the space, channel, API key and optional configuration id;
    ``agent_name`` narrows the list to that agent's own tools.
    """
    return run_sync(SdkToolCatalog(settings).list_tools(agent_name=agent_name))


def fetch_approval_mode(settings: PlatformSettings, config_id: str) -> str | None:
    """Derive the hermes ``approvals.mode`` from the stored configuration *config_id*.

    Reads the configuration's ``sandbox_auto_approve`` toggle and maps it to a hermes mode:

      * ``True``  → ``"off"``    (approvals bypassed)
      * absent/``False`` → ``"manual"`` (approvals required)

    A missing flag resolves to ``"manual"``, mirroring the platform's own default, so a
    missing setting can never silently bypass approvals. Returns ``None`` only when the
    response isn't a usable configuration, so a malformed/empty fetch leaves the pod's
    current mode untouched instead of clobbering it.
    """
    logger.debug("platform-gateway: fetching configuration %s", config_id)
    auto_approve = run_sync(sandbox_auto_approve(settings, config_id))
    if auto_approve is None:
        return None
    return "off" if auto_approve else "manual"


def apply_approval_mode(mode: str) -> None:
    """Persist ``approvals.mode`` to the pod's config, read live on the next check.

    Written via ``load_config``/``save_config`` (not ``set_config_value``, which
    ``sys.exit``s on a managed key — that would kill the pod). Managed installs
    and admin-pinned keys are guarded explicitly and skipped with a warning.
    """
    if is_managed() or managed_scope.is_key_managed("approvals.mode"):
        logger.warning("platform-gateway: approvals.mode is managed — skipping approval-mode sync")
        return

    try:
        config = load_config()
        approvals = config.setdefault("approvals", {})
        if not isinstance(approvals, dict):
            approvals = config["approvals"] = {}
        approvals["mode"] = mode
        save_config(config)
        logger.warning("platform-gateway: applied approvals.mode=%s", mode)
    except Exception as exc:
        logger.warning("platform-gateway: failed to apply approvals.mode - %s", type(exc).__name__)
        logger.debug("platform-gateway: apply approvals.mode detail: %s", exc)


class ConfigSyncer:
    """Reconciles the pod's approval mode against a platform configuration.

    One instance per pod, shared by the boot sync (at plugin register) and the
    per-turn sync (every AGENT_START). Owns the edge-trigger memo so those two
    callers can't double-write ``config.yaml``.

    ``config_id`` is resolved per call: the id carried by the turn wins, falling
    back to the pod's spawn-time ``DOMYN_CONFIGURATION_ID``. That is what lets a
    freshly-activated configuration (a NEW uid, never seen by this pod) take
    effect without a restart.

    Thread-safety: the whole fetch → compare → apply sequence runs under a lock.
    Concurrent AGENT_STARTs (the gateway multiplexes conversations) would
    otherwise race both the memo and the read-modify-write of ``config.yaml``.

    Note: the ``approvals.mode`` write is pod-GLOBAL while ``config_id`` is
    per-turn. Harmless under one-pod-per-conversation; two conversations in one
    pod with opposing toggles would flip each other's setting.
    """

    def __init__(
        self,
        *,
        settings: PlatformSettings,
        default_config_id: str | None = None,
        _fetch: Callable[[PlatformSettings, str], str | None] | None = None,
        _apply: Callable[[str], None] | None = None,
    ) -> None:
        self._settings = settings
        self._default_config_id = default_config_id
        self._fetch_fn = _fetch if _fetch is not None else fetch_approval_mode
        self._apply_fn = _apply if _apply is not None else apply_approval_mode
        # Last mode written — edge-trigger so an unchanged value doesn't rewrite
        # config.yaml (and thrash hermes' mtime-keyed load_config cache).
        self._applied_mode: str | None = None
        self._lock = threading.Lock()

    def sync_blocking(self, config_id: str | None = None) -> None:
        """Fetch *config_id* and apply its approval mode. Never raises.

        Blocking (sync HTTP + a possible config write) — call via :meth:`sync`
        from async contexts so the event loop isn't stalled.
        """
        resolved = config_id or self._default_config_id
        if not resolved:
            return
        if not config_id and self._default_config_id:
            # The turn carried no id (older platform, or run_config withheld) —
            # the pinned spawn-time config can be stale, so make it diagnosable.
            logger.debug(
                "platform-gateway: no config id on turn, falling back to %s",
                self._default_config_id,
            )
        with self._lock:
            try:
                mode = self._fetch_fn(self._settings, resolved)
            except Exception as exc:
                logger.warning("platform-gateway: config fetch failed - %s", type(exc).__name__)
                logger.debug("platform-gateway: config fetch detail: %s", exc)
                return

            if mode is None or mode == self._applied_mode:
                return
            self._apply_fn(mode)
            self._applied_mode = mode

    async def sync(self, config_id: str | None = None, *, timeout: float = 3.0) -> None:
        """Await :meth:`sync_blocking` off the event loop, bounded by *timeout*.

        Fails open: on timeout or error the caller proceeds with whatever mode is
        currently on disk. A config-endpoint blip must never wedge a turn — and
        the safe default (flag absent ⇒ ``manual``) means a pod that has never
        reached the platform errs toward REQUIRING approvals.
        """
        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, self.sync_blocking, config_id),
                timeout=timeout,
            )
        except TimeoutError:
            logger.warning("platform-gateway: config sync timed out after %.1fs", timeout)
        except Exception as exc:
            logger.warning("platform-gateway: config sync failed - %s", type(exc).__name__)
            logger.debug("platform-gateway: config sync detail: %s", exc)


def _deregister_tool(name: str) -> None:
    """Remove a tool from the hermes tool registry."""
    try:
        from tools.registry import registry

        registry.deregister(name)
        logger.info("platform-gateway: deregistered tool '%s'", name)
    except ImportError:
        logger.debug("platform-gateway: tools.registry unavailable, cannot deregister '%s'", name)


class RefreshLoop:
    """Periodically re-fetches the canvas tool list and syncs the hermes registry.

    Runs in a daemon thread. On each interval it diffs the live tool list against
    the currently-registered set: new tools are registered, removed tools are
    deregistered. Unchanged tools are left alone.

    ``initial_names`` is *adopted, not copied*: the caller keeps a reference to the
    same set and we mutate it in place, so ``register()``'s view of which tools are
    ours stays correct as tools come and go. That set drives the ``is_platform``
    branch in the tool hooks, and a stale copy there means a platform tool is
    mistaken for a hermes built-in (spurious visibility events on top of the
    canonical relay round-trip).

    Pass ``_deregister`` in tests to avoid the hermes registry import.
    """

    def __init__(
        self,
        ctx: Any,
        handler_factory: Callable[[str], Callable],
        settings: PlatformSettings,
        interval: float,
        initial_names: set[str],
        tool_rename_mapping: dict[str, str],
        agent_name: str | None = None,
        _deregister: Callable[[str], None] | None = None,
    ) -> None:
        self._ctx = ctx
        self._handler_factory = handler_factory
        self._settings = settings
        self._interval = interval
        self._agent_name = agent_name
        self._tool_rename_mapping = tool_rename_mapping
        # Adopted, not copied — see the class docstring. Only this thread writes to
        # it (add/discard, both atomic); the hooks only do membership tests, so no
        # lock is needed.
        self._registered: set[str] = initial_names
        self._deregister_fn = _deregister if _deregister is not None else _deregister_tool

    def start(self) -> None:
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

    def _run(self) -> None:
        while True:
            time.sleep(self._interval)
            self._refresh()

    def _refresh(self) -> None:
        from domyn.agents.integrations.hermes_platform_gateway.schema import convert_schema

        try:
            raw_tools = fetch_tools(self._settings, agent_name=self._agent_name)
        except Exception as exc:
            logger.warning("platform-gateway: refresh fetch failed - %s", type(exc).__name__)
            logger.debug("platform-gateway: refresh fetch detail: %s", exc)
            return
        reserved_tool_names = hermes_tool_names(exclude_toolset=_PLATFORM_TOOLSET)
        # ``previous_mapping``: keeps a live tool's exposed name stable when the batch
        # composition changes. Safe to pass the shared dict — the callee only reads it,
        # and fully before the in-place diff below mutates it.
        raw_tools, new_tool_rename_mapping = rename_colliding_tools(
            raw_tools, reserved_tool_names, previous_mapping=self._tool_rename_mapping
        )
        # Swap atomically: readers (adapter visibility events) do unsynchronised
        # .get() on this dict from the asyncio thread.  A .clear()-.update()
        # sequence would leave the dict momentarily empty, causing a reader in
        # that window to miss the mapping and send the wrong (renamed) tool name
        # to the platform.  Building a full replacement dict first and then doing
        # a single .clear()-.update() pair with no yield point in between is
        # still two bytecodes, so instead we compute the minimal diff and apply
        # only individual key insertions/deletions — each is atomic under the GIL.
        stale_keys = self._tool_rename_mapping.keys() - new_tool_rename_mapping.keys()
        for k in stale_keys:
            self._tool_rename_mapping.pop(k, None)
        self._tool_rename_mapping.update(new_tool_rename_mapping)
        new_defs = {t["name"]: t for t in raw_tools if t.get("name")}
        new_names = set(new_defs.keys())

        added = new_names - self._registered
        removed = self._registered - new_names

        if not added and not removed:
            return

        # ``_registered`` is mutated one name at a time rather than rebound (or
        # cleared and refilled): the set is shared with ``register()`` and read
        # concurrently by the tool hooks, so it must never be momentarily empty
        # or replaced by a different object.
        for name in removed:
            self._deregister_fn(name)
            self._registered.discard(name)

        for name in added:
            try:
                schema = convert_schema(new_defs[name])
            except Exception as exc:
                logger.warning(
                    "platform-gateway: skipping new tool '%s': %s", name, type(exc).__name__
                )
                logger.debug("platform-gateway: new tool '%s' schema error detail: %s", name, exc)
                continue
            self._ctx.register_tool(
                name=name,
                toolset=_PLATFORM_TOOLSET,
                schema=schema,
                handler=self._handler_factory(name),
                is_async=True,
            )
            # Recorded only once hermes has it, so a tool whose schema failed to
            # convert is retried on the next tick instead of being written off.
            self._registered.add(name)

        logger.info(
            "platform-gateway: canvas refresh - +%d added, -%d removed",
            len(added),
            len(removed),
        )
