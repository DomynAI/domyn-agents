"""What a configuration's ``resources:`` mean, checked when it loads.

The rules, every one an error naming the key to change (a missing local folder is
the one warning):

1. resource ids are unique, and none equals an agent's name;
2. every id an agent lists or a consumer names is declared;
3. an execution bundle (``shell``, ``browser``) is bound to a resource it can run
   commands in, never one that is not on this machine;
4. a resource's own check passes (``Resource.check``); a local resource's folders
   that do not exist are a warning, since a consumer that needs one fails on use;
5. a consumer that needs a resource names one: a bundle entry on an agent says
   ``{use: <bundle>, workspace: <id>}``, a ``type: resource`` source says
   ``resource:``.

Nothing is read or created: a resource does something only when a consumer uses it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from domyn.agents.logger import get_logger
from domyn.agents.resources.base import Resource, ResourceError, resolve_resource

EXECUTION_BUNDLES: frozenset[str] = frozenset({"shell", "browser"})
"""Bundles that run a process in their workspace, and so need a local folder."""


def parse_resources(entries: Iterable[Any]) -> dict[str, Resource]:
    """The ``resources:`` list by id, each one checked (rules 1 and 4)."""
    declared: dict[str, Resource] = {}
    for index, entry in enumerate(entries or []):
        resource = resolve_resource(entry)
        if resource.id in declared:
            raise ResourceError(
                f"Resource id {resource.id!r} is declared twice under 'resources' "
                f"(entry #{index + 1}): ids must be unique"
            )
        for warning in resource.check():
            get_logger().warning(warning)
        declared[resource.id] = resource
    return declared


def check_agent_names(declared: Mapping[str, Resource], agent_names: Iterable[str]) -> None:
    """Rule 1, second half: an agent with no resource is keyed by its name on the
    platform, so the two namespaces must not meet."""
    clashing = sorted(set(declared) & set(agent_names))
    if clashing:
        raise ResourceError(
            f"Resource id {', '.join(repr(c) for c in clashing)} is also an agent's name: "
            "rename the resource"
        )


@dataclass(frozen=True)
class BundleUse:
    """One entry of an agent's ``tool_bundles:``: which bundle, bound to which resource."""

    bundle: str
    workspace: str


def parse_bundle_use(entry: Any, agent_name: str) -> BundleUse:
    """``{use: <bundle>, workspace: <id>}``, or an error saying how to write it (rule 5).

    A bare bundle name is refused: it names no resource, and there is no default
    one to fall back to.
    """
    if isinstance(entry, str):
        raise ResourceError(
            f"Agent {agent_name!r} uses tool bundle {entry!r} without a workspace: write "
            f"`{{use: {entry}, workspace: <resource id>}}` and declare the resource under "
            "'resources:' (for example `{id: workspace, type: local_dir, paths: [.]}`), "
            "or run `domyn setup` again"
        )
    if not isinstance(entry, dict) or not isinstance(entry.get("use"), str):
        raise ResourceError(
            f"Agent {agent_name!r}: a 'tool_bundles' entry must be "
            f"`{{use: <bundle>, workspace: <resource id>}}`, got {entry!r}"
        )
    unknown = sorted(set(entry) - {"use", "workspace"})
    if unknown:
        raise ResourceError(
            f"Agent {agent_name!r}, tool bundle {entry['use']!r}: unknown key "
            f"{', '.join(repr(k) for k in unknown)} (expected 'use' and 'workspace')"
        )
    workspace = entry.get("workspace")
    if not isinstance(workspace, str) or not workspace:
        raise ResourceError(
            f"Agent {agent_name!r} uses tool bundle {entry['use']!r} without a workspace: "
            f"add `workspace: <resource id>` to the entry"
        )
    return BundleUse(bundle=entry["use"], workspace=workspace)


def check_bundle_resource(
    use: BundleUse, bundles: Iterable[str], resource: Resource, agent_name: str
) -> None:
    """Rule 3: what a bundle does must be possible in the resource it is bound to."""
    names = set(bundles)
    where = f"Agent {agent_name!r}, tool bundle {use.bundle!r}"
    if resource.local_roots() is None:
        running = sorted(names & EXECUTION_BUNDLES)
        if running:
            raise ResourceError(
                f"{where}: {', '.join(running)} runs commands in its workspace, and "
                f"resource {resource.id!r} ({resource.type}) is not a folder on this "
                "machine. Bind the bundle to a local_dir resource."
            )
        if "files" in names:
            # Valid in the design; the file tools work on local paths until they are
            # moved onto the file-store view and tested there.
            raise ResourceError(
                f"{where}: the files bundle cannot work on resource {resource.id!r} "
                f"({resource.type}) yet. Bind it to a local_dir resource; this resource "
                "can be read through skills and instructions."
            )


def resolve_ids(ids: Iterable[str], declared: Mapping[str, Resource], where: str) -> list[Resource]:
    """Rule 2: the resources ``ids`` name, in order."""
    found: list[Resource] = []
    for one in ids:
        if not isinstance(one, str) or one not in declared:
            raise ResourceError(
                f"{where} names resource {one!r}, which no 'resources:' entry declares"
            )
        found.append(declared[one])
    return found


def effective_resources(*groups: Iterable[Resource]) -> list[Resource]:
    """The union of an agent's list and every reference, first appearance first."""
    seen: dict[str, Resource] = {}
    for group in groups:
        for one in group:
            seen.setdefault(one.id, one)
    return list(seen.values())
