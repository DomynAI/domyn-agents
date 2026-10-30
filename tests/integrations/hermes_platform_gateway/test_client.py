import time
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from domyn.agents.integrations.domyn_platform_sdk import PlatformSettings
from domyn.agents.integrations.hermes_platform_gateway.client import (
    ConfigSyncer,
    RefreshLoop,
    fetch_approval_mode,
    fetch_tools,
    rename_colliding_tools,
)

_SCHEMA_MOD = "domyn.agents.integrations.hermes_platform_gateway.schema"


SETTINGS = PlatformSettings(
    api_key="k1", space_id="s1", channel_id="c1", base_url="example.com", configuration_id="cfg-1"
)


def test_fetch_tools_returns_the_catalog_tools(platform):
    platform.serve([{"name": "echo", "description": "Echo", "parameters": []}])
    tools = fetch_tools(SETTINGS)
    assert [t["name"] for t in tools] == ["echo"]


def test_fetch_tools_passes_settings_and_agent_name(platform):
    platform.serve([])
    fetch_tools(SETTINGS, agent_name="worker")
    assert platform.calls[-1] == {"settings": SETTINGS, "agent_name": "worker"}


def test_fetch_tools_propagates_a_platform_failure(platform):
    platform.fail(PermissionError("401"))
    with pytest.raises(PermissionError):
        fetch_tools(SETTINGS)


@pytest.mark.asyncio
async def test_fetch_tools_works_inside_a_running_loop(platform):
    platform.serve([{"name": "t", "parameters": []}])
    assert len(fetch_tools(SETTINGS)) == 1


# ---------------------------------------------------------------------------
# rename_colliding_tools
# ---------------------------------------------------------------------------


def test_rename_colliding_tools_no_collision_leaves_names_untouched():
    tools = [{"name": "echo", "group_name": "gitlab"}]
    renamed, mapping = rename_colliding_tools(tools, reserved_tool_names=set())
    assert [t["name"] for t in renamed] == ["echo"]
    assert mapping == {}


def test_rename_colliding_tools_renames_with_group_prefix():
    tools = [{"name": "search", "group_name": "github"}]
    renamed, mapping = rename_colliding_tools(tools, reserved_tool_names={"search"})
    assert [t["name"] for t in renamed] == ["github_search"]
    assert mapping == {"github_search": "search"}


def test_rename_colliding_tools_renames_with_domyn_prefix_when_no_group():
    tools = [{"name": "search", "group_name": None}]
    renamed, mapping = rename_colliding_tools(tools, reserved_tool_names={"search"})
    assert [t["name"] for t in renamed] == ["domyn_search"]
    assert mapping == {"domyn_search": "search"}


def test_rename_colliding_tools_sanitizes_group_name_with_invalid_chars():
    tools = [{"name": "search", "group_name": "My Group!"}]
    renamed, mapping = rename_colliding_tools(tools, reserved_tool_names={"search"})
    assert [t["name"] for t in renamed] == ["My_Group_search"]
    assert mapping == {"My_Group_search": "search"}


def test_rename_colliding_tools_empty_group_name_falls_back_to_domyn():
    tools = [{"name": "search", "group_name": ""}]
    renamed, mapping = rename_colliding_tools(tools, reserved_tool_names={"search"})
    assert [t["name"] for t in renamed] == ["domyn_search"]
    assert mapping == {"domyn_search": "search"}


def test_rename_colliding_tools_avoids_collision_with_other_raw_tool_name():
    # "github_search" already exists among the raw tools, so the renamed
    # candidate must not shadow it.
    tools = [
        {"name": "github_search", "group_name": "github"},
        {"name": "search", "group_name": "github"},
    ]
    renamed, mapping = rename_colliding_tools(tools, reserved_tool_names={"search"})
    assert [t["name"] for t in renamed] == ["github_search", "github_search_1"]
    assert mapping == {"github_search_1": "search"}


def test_rename_colliding_tools_does_not_steal_a_later_tools_own_name():
    # The renaming tool is listed FIRST: its candidate "github_search" must not be
    # claimed ahead of the tool further down the batch that really is named that.
    # This is what forces pass 2 (keepers) to finish before pass 3 (renames) starts;
    # the test above happens to list them in the order that survives a merge.
    tools = [
        {"name": "search", "group_name": "github"},
        {"name": "github_search", "group_name": "other"},
    ]
    renamed, mapping = rename_colliding_tools(tools, reserved_tool_names={"search"})
    assert [t["name"] for t in renamed] == ["github_search_1", "github_search"]
    assert mapping == {"github_search_1": "search"}


def test_rename_colliding_tools_leaves_the_input_list_untouched():
    tools = [{"name": "search", "group_name": "github"}]
    renamed, _ = rename_colliding_tools(tools, reserved_tool_names={"search"})
    assert tools == [{"name": "search", "group_name": "github"}]
    assert renamed[0]["name"] == "github_search"


def test_rename_colliding_tools_avoids_within_batch_collision():
    # Two distinct colliding tools that compute the same candidate name must
    # not clobber each other.
    tools = [
        {"name": "search", "group_name": "github", "uid": "tool-1"},
        {"name": "search", "group_name": "github", "uid": "tool-2"},
    ]
    renamed, mapping = rename_colliding_tools(tools, reserved_tool_names={"search"})
    assert [t["name"] for t in renamed] == ["github_search", "github_search_1"]
    assert mapping == {"github_search": "search", "github_search_1": "search"}


def test_rename_colliding_tools_skips_tool_without_name():
    tools = [{"group_name": "github"}]
    renamed, mapping = rename_colliding_tools(tools, reserved_tool_names={"search"})
    assert renamed == []
    assert mapping == {}


def test_rename_colliding_tools_mapping_is_new_name_to_original_name():
    tools = [{"name": "search", "group_name": "github"}]
    _, mapping = rename_colliding_tools(tools, reserved_tool_names={"search"})
    assert mapping.get("github_search") == "search"
    assert "search" not in mapping


def test_rename_colliding_tools_numeric_suffix_when_name_already_carries_group_prefix():
    # Prefixing would yield "github_github_search"; the suffix is the only thing
    # that can disambiguate a name that is already group-prefixed.
    tools = [{"name": "github_search", "group_name": "github"}]
    renamed, mapping = rename_colliding_tools(tools, reserved_tool_names={"github_search"})
    assert [t["name"] for t in renamed] == ["github_search_1"]
    assert mapping == {"github_search_1": "github_search"}


# ---------------------------------------------------------------------------
# rename_colliding_tools — stability via previous_mapping
# ---------------------------------------------------------------------------


def test_rename_colliding_tools_reclaims_previously_issued_name():
    tools = [{"name": "search", "group_name": "github"}]
    renamed, mapping = rename_colliding_tools(
        tools, reserved_tool_names={"search"}, previous_mapping={"github_search": "search"}
    )
    assert [t["name"] for t in renamed] == ["github_search"]
    assert mapping == {"github_search": "search"}


def test_rename_colliding_tools_newcomer_yields_to_previously_issued_name():
    # The incumbent holds "github_search"; the newcomer's real name is that very
    # string, so the newcomer is the one that has to move.
    tools = [
        {"name": "search", "group_name": "github"},
        {"name": "github_search", "group_name": "github"},
    ]
    renamed, mapping = rename_colliding_tools(
        tools, reserved_tool_names={"search"}, previous_mapping={"github_search": "search"}
    )
    assert [t["name"] for t in renamed] == ["github_search", "github_search_1"]
    assert mapping == {"github_search": "search", "github_search_1": "github_search"}


def test_rename_colliding_tools_is_idempotent_when_fed_its_own_output():
    tools = [
        {"name": "search", "group_name": "github"},
        {"name": "github_search", "group_name": "github"},
    ]
    settled = {"github_search": "search", "github_search_1": "github_search"}
    renamed, mapping = rename_colliding_tools(
        tools, reserved_tool_names={"search"}, previous_mapping=settled
    )
    assert [t["name"] for t in renamed] == ["github_search", "github_search_1"]
    assert mapping == settled


def test_rename_colliding_tools_does_not_reclaim_name_hermes_has_since_reserved():
    # "github_search" was ours last tick, but hermes now owns it — take a new one
    # rather than shadowing a built-in.
    tools = [{"name": "search", "group_name": "github"}]
    renamed, mapping = rename_colliding_tools(
        tools,
        reserved_tool_names={"search", "github_search"},
        previous_mapping={"github_search": "search"},
    )
    assert [t["name"] for t in renamed] == ["github_search_1"]
    assert mapping == {"github_search_1": "search"}


def test_rename_colliding_tools_drops_previous_entry_for_absent_tool():
    # A tool that left the batch must not linger in the mapping — RefreshLoop
    # diffs the returned keys to decide what to prune.
    tools = [{"name": "search", "group_name": "github"}]
    _, mapping = rename_colliding_tools(
        tools,
        reserved_tool_names={"search"},
        previous_mapping={"github_search": "search", "github_list": "list"},
    )
    assert mapping == {"github_search": "search"}


# ---------------------------------------------------------------------------
# RefreshLoop
# ---------------------------------------------------------------------------


def _make_refresh_loop(initial_names, deregister_fn=None, config_id="cfg-1"):
    ctx = MagicMock()
    deregister = deregister_fn or MagicMock()
    loop = RefreshLoop(
        ctx=ctx,
        handler_factory=lambda name: lambda args, **kw: f"handled {name}",
        settings=replace(SETTINGS, configuration_id=config_id),
        interval=60,
        # Passed through as-is, not copied: RefreshLoop adopts the caller's set and
        # mutates it in place, and some tests assert on that shared object.
        initial_names=initial_names,
        tool_rename_mapping={},
        _deregister=deregister,
    )
    return loop, ctx, deregister


def test_refresh_loop_registers_new_tool(platform):
    platform.serve(
        [
            {"name": "existing", "parameters": []},
            {"name": "new_tool", "description": "Brand new", "parameters": []},
        ]
    )
    loop, ctx, deregister = _make_refresh_loop({"existing"})
    loop._refresh()

    ctx.register_tool.assert_called_once()
    assert ctx.register_tool.call_args.kwargs["name"] == "new_tool"
    deregister.assert_not_called()
    assert loop._registered == {"existing", "new_tool"}


def test_refresh_loop_deregisters_removed_tool(platform):
    platform.serve(
        [
            {"name": "remaining", "parameters": []},
        ]
    )
    loop, ctx, deregister = _make_refresh_loop({"remaining", "gone"})
    loop._refresh()

    deregister.assert_called_once_with("gone")
    ctx.register_tool.assert_not_called()
    assert loop._registered == {"remaining"}


def test_refresh_loop_no_change_makes_no_calls(platform):
    platform.serve([{"name": "tool_a", "parameters": []}])
    loop, ctx, deregister = _make_refresh_loop({"tool_a"})
    loop._refresh()

    ctx.register_tool.assert_not_called()
    deregister.assert_not_called()


def test_refresh_loop_fetch_failure_leaves_registered_set_unchanged(platform):
    platform.fail()
    loop, ctx, deregister = _make_refresh_loop({"tool_a"})
    loop._refresh()

    ctx.register_tool.assert_not_called()
    deregister.assert_not_called()
    assert loop._registered == {"tool_a"}


def test_refresh_loop_skips_new_tool_with_missing_name(platform):
    platform.serve(
        [
            {"parameters": []},  # no name — should be skipped
            {"name": "good", "parameters": []},
        ]
    )
    loop, ctx, _deregister = _make_refresh_loop(set())
    loop._refresh()

    assert ctx.register_tool.call_count == 1
    assert ctx.register_tool.call_args.kwargs["name"] == "good"


# ---------------------------------------------------------------------------
# RefreshLoop shares its registered-name set with register()
#
# register() hands the same set to RefreshLoop and keeps reading it from the
# pre/post_tool_call hooks to decide whether an invoked tool is one of ours
# (``is_platform``). If RefreshLoop copies or rebinds it, the hooks go stale the
# first time the tool list changes and a refresh-added platform tool is mistaken
# for a hermes built-in — emitting visibility events on top of the canonical
# relay round-trip.
# ---------------------------------------------------------------------------


def test_refresh_loop_adopts_caller_set_without_copying(platform):
    platform.serve([{"name": "existing", "parameters": []}])
    shared: set[str] = {"existing"}
    loop, _ctx, _deregister = _make_refresh_loop(shared)

    assert loop._registered is shared


def test_refresh_loop_adds_new_tool_to_shared_set_in_place(platform):
    platform.serve(
        [
            {"name": "existing", "parameters": []},
            {"name": "new_tool", "parameters": []},
        ]
    )
    shared: set[str] = {"existing"}
    loop, _ctx, _deregister = _make_refresh_loop(shared)
    loop._refresh()

    # The caller's own reference sees the addition — no rebind.
    assert shared == {"existing", "new_tool"}
    assert loop._registered is shared


def test_refresh_loop_discards_removed_tool_from_shared_set_in_place(platform):
    platform.serve([{"name": "remaining", "parameters": []}])
    shared: set[str] = {"remaining", "gone"}
    loop, _ctx, deregister = _make_refresh_loop(shared)
    loop._refresh()

    deregister.assert_called_once_with("gone")
    assert shared == {"remaining"}
    assert loop._registered is shared


# ---------------------------------------------------------------------------
# Rename stability across refresh ticks
#
# Spec: once a colliding tool has been exposed under a disambiguated name, a
# later tick must not reassign that name to a different tool just because an
# unrelated tool happens to join the batch. RefreshLoop carries its mapping
# forward into rename_colliding_tools for exactly this reason — recomputing
# from the current batch alone would silently repoint a name the model has
# already been calling.
# ---------------------------------------------------------------------------


def test_refresh_loop_rename_is_stable_when_sibling_tool_appears(platform, monkeypatch):
    monkeypatch.setattr(
        "domyn.agents.integrations.hermes_platform_gateway.client.hermes_tool_names",
        lambda exclude_toolset=None: {"search"},
    )
    # Tick 1: only the colliding tool is present — it claims "github_search".
    platform.serve([{"name": "search", "group_name": "github"}])
    loop, _ctx, _deregister = _make_refresh_loop(set())
    loop._refresh()

    assert loop._registered == {"github_search"}
    assert loop._tool_rename_mapping == {"github_search": "search"}

    # Tick 2: a genuinely different tool literally named "github_search" shows
    # up in the same batch. The already-disambiguated "search" tool must keep
    # its name — the newcomer is the one that should yield, not the tool that
    # was here first.
    platform.serve(
        [
            {"name": "search", "group_name": "github"},
            {"name": "github_search", "group_name": "github"},
        ]
    )
    loop._refresh()

    # "search" keeps its established name across the tick...
    assert loop._tool_rename_mapping == {
        "github_search": "search",
        "github_search_1": "github_search",
    }
    # ...and the newcomer is the one disambiguated instead.
    assert loop._registered == {"github_search", "github_search_1"}


def test_refresh_loop_does_not_record_tool_whose_schema_fails(platform):
    """A tool we failed to register must not be recorded as ours, so it is retried
    on the next tick rather than written off — and never deregistered out from
    under whoever does own the name."""
    platform.serve(
        [
            {"name": "good", "parameters": []},
            {"name": "broken", "parameters": []},
        ]
    )
    shared: set[str] = set()
    loop, ctx, _deregister = _make_refresh_loop(shared)

    def _flaky_convert(tool_def):
        if tool_def["name"] == "broken":
            raise ValueError("bad schema")
        return {"type": "object", "properties": {}}

    # _refresh does `from .schema import convert_schema`, so patch it at the source.
    with patch(f"{_SCHEMA_MOD}.convert_schema", side_effect=_flaky_convert):
        loop._refresh()

    assert shared == {"good"}
    assert ctx.register_tool.call_count == 1

    # Second tick: "broken" is still absent from the set, so it is retried.
    with patch(f"{_SCHEMA_MOD}.convert_schema", return_value={"type": "object"}):
        loop._refresh()

    assert shared == {"good", "broken"}


# ---------------------------------------------------------------------------
# Configuration fetch + approval-mode mapping
# ---------------------------------------------------------------------------


def _patch_toggle(monkeypatch, value):
    async def toggle(settings, config_id):
        toggle.seen.append((settings, config_id))
        if isinstance(value, Exception):
            raise value
        return value

    toggle.seen = []
    monkeypatch.setattr(
        "domyn.agents.integrations.hermes_platform_gateway.client.sandbox_auto_approve", toggle
    )
    return toggle


@pytest.mark.parametrize(
    "toggle,expected",
    [
        (True, "off"),
        (False, "manual"),  # flag absent/false → require (safe default)
        (None, None),  # not a usable config object → leave mode untouched
    ],
)
def test_fetch_approval_mode(monkeypatch, toggle, expected):
    seen = _patch_toggle(monkeypatch, toggle).seen
    assert fetch_approval_mode(SETTINGS, "cfg-1") == expected
    assert seen == [(SETTINGS, "cfg-1")]


def test_fetch_approval_mode_propagates_a_failure(monkeypatch):
    _patch_toggle(monkeypatch, RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        fetch_approval_mode(SETTINGS, "cfg-1")


# ---------------------------------------------------------------------------
# ConfigSyncer
# ---------------------------------------------------------------------------


def _mode(bypass):
    return "off" if bypass else "manual"


def _make_syncer(*, default_config_id="cfg-env", fetch=None, apply_fn=None):
    apply_fn = apply_fn or MagicMock()
    syncer = ConfigSyncer(
        settings=SETTINGS,
        default_config_id=default_config_id,
        _fetch=fetch,
        _apply=apply_fn,
    )
    return syncer, apply_fn


def test_syncer_applies_mode_on_change():
    fetch = MagicMock(return_value=_mode(True))
    syncer, apply_fn = _make_syncer(fetch=fetch)
    syncer.sync_blocking("cfg-9")
    apply_fn.assert_called_once_with("off")
    # The turn's id wins over the pod's spawn-time default.
    assert fetch.call_args.args[1] == "cfg-9"


def test_syncer_falls_back_to_default_config_id():
    fetch = MagicMock(return_value=_mode(False))
    syncer, apply_fn = _make_syncer(fetch=fetch)
    syncer.sync_blocking(None)
    assert fetch.call_args.args[1] == "cfg-env"
    apply_fn.assert_called_once_with("manual")


def test_syncer_noop_without_any_config_id():
    fetch = MagicMock()
    syncer, apply_fn = _make_syncer(default_config_id=None, fetch=fetch)
    syncer.sync_blocking(None)
    fetch.assert_not_called()
    apply_fn.assert_not_called()


def test_syncer_edge_triggers_no_reapply_when_unchanged():
    fetch = MagicMock(return_value=_mode(True))
    syncer, apply_fn = _make_syncer(fetch=fetch)
    syncer.sync_blocking("cfg-9")
    syncer.sync_blocking("cfg-9")
    apply_fn.assert_called_once()  # unchanged mode — no rewrite
    assert fetch.call_count == 2  # ...but we still re-fetch every turn


def test_syncer_reapplies_when_mode_flips():
    fetch = MagicMock(side_effect=[_mode(True), _mode(False)])
    syncer, apply_fn = _make_syncer(fetch=fetch)
    syncer.sync_blocking("cfg-9")
    syncer.sync_blocking("cfg-9")
    assert [c.args[0] for c in apply_fn.call_args_list] == ["off", "manual"]


def test_syncer_skips_when_config_unusable():
    syncer, apply_fn = _make_syncer(fetch=MagicMock(return_value=None))
    syncer.sync_blocking("cfg-9")
    apply_fn.assert_not_called()


def test_syncer_fetch_failure_is_swallowed():
    syncer, apply_fn = _make_syncer(fetch=MagicMock(side_effect=RuntimeError("boom")))
    syncer.sync_blocking("cfg-9")  # must not raise
    apply_fn.assert_not_called()


def test_syncer_uses_real_fetch_by_default(monkeypatch):
    seen = _patch_toggle(monkeypatch, True).seen
    apply_fn = MagicMock()
    syncer, _ = _make_syncer(apply_fn=apply_fn)
    syncer.sync_blocking("cfg-9")
    apply_fn.assert_called_once_with("off")
    assert seen == [(SETTINGS, "cfg-9")]


@pytest.mark.asyncio
async def test_syncer_sync_awaits_apply_off_loop():
    fetch = MagicMock(return_value=_mode(True))
    syncer, apply_fn = _make_syncer(fetch=fetch)
    await syncer.sync("cfg-9")
    apply_fn.assert_called_once_with("off")


@pytest.mark.asyncio
async def test_syncer_sync_fails_open_on_error():
    syncer, apply_fn = _make_syncer(fetch=MagicMock(side_effect=RuntimeError("boom")))
    await syncer.sync("cfg-9")  # must not raise — caller proceeds with the turn
    apply_fn.assert_not_called()


@pytest.mark.asyncio
async def test_syncer_sync_fails_open_on_timeout():
    def _slow(*_args, **_kwargs):
        time.sleep(0.5)
        return _mode(True)

    syncer, apply_fn = _make_syncer(fetch=_slow)
    await syncer.sync("cfg-9", timeout=0.01)  # must not raise
    apply_fn.assert_not_called()
