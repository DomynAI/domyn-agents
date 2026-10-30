"""Tools the framework ships but never mounts by itself.

The distinction is the whole point. ``Agent`` gets no tools automatically, so a
customer-facing agent does not silently acquire shell access because someone
added a default somewhere. Mounting is always an explicit act: the CLI does it
for its own agent, and a YAML config does it by naming a bundle. Every comparable
tool converged on the same separation — in pi the tools live in the coding agent,
not in the agent core; in Hermes they are toolsets you enable.

Bundles rather than individual names because the useful unit is a capability,
not a function: "this agent can work with files" is a decision someone can
reason about, "this agent has ``glob_files``" is not.

    from domyn.agents.tools.builtin import bundle
    agent.tools.extend(bundle("files", "shell", root="/path/to/project"))

Everything is bound to a :class:`~domyn.agents.tools.builtin.workspace.Workspace`
at construction, so a tool physically cannot reach outside it — the root is not
an argument the model can pass.

Where commands run is a separate axis, the
:class:`~domyn.agents.tools.builtin.backends.CommandBackend`. Only the local one
exists today; :func:`bundle` accepts one and returns it via :class:`BuiltinTools`
so the caller can shut it down, because the backend that will eventually need
shutting down is a container and finding the end of a session later, under
pressure, is how teardown gets forgotten.

The ``browser`` bundle runs on that same backend — every browser command is a
shell line — so it goes wherever the shell goes. It is known but never on by
default (:data:`DEFAULT_BUNDLES`): browsing reads untrusted pages, and an agent
should not start doing that because a config entry left its list empty.
"""

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel
from pydantic import Field as PydanticField

from domyn.agents.logger import get_logger
from domyn.agents.tools.builtin.backends import (
    BackendUnavailableError,
    CommandBackend,
    CommandResult,
    LocalBackend,
    build_backend,
)
from domyn.agents.tools.builtin.browser import browser_tools
from domyn.agents.tools.builtin.browser_sessions import BrowserOptions, BrowserSessions
from domyn.agents.tools.builtin.files import file_tools
from domyn.agents.tools.builtin.shell import run_command_tool, shell_tools
from domyn.agents.tools.builtin.workspace import Workspace, WorkspaceError, is_ignored

# Bundle name -> what it contains. Kept small and coarse on purpose.
BUNDLES: tuple[str, ...] = ("files", "shell", "browser")

# What a bundle entry gets when it does not list its bundles. Not `BUNDLES`:
# adding a bundle must not switch it on for every existing config.
DEFAULT_BUNDLES: tuple[str, ...] = ("files", "shell")

# What each bundle mounts, by tool name. Static because a describe call (`/config`)
# must not build a workspace and a command backend just to list names;
# `TestBundles` fails if this drifts from what `bundle()` actually builds.
BUNDLE_TOOL_NAMES: dict[str, tuple[str, ...]] = {
    "files": (
        "read_file",
        "list_dir",
        "glob_files",
        "grep",
        "write_file",
        "append_file",
        "edit_file",
    ),
    "shell": ("run_command", "check_command", "list_commands", "stop_command"),
    "browser": (
        "browser_open",
        "browser_snapshot",
        "browser_read",
        "browser_scroll",
        "browser_back",
        "browser_click",
        "browser_type",
        "browser_press",
    ),
}

# What each bundle is for, in one line — what `domyn setup` shows beside its name
# when asking which to mount. A new bundle needs an entry here, in `BUNDLES` and in
# `BUNDLE_TOOL_NAMES`; `TestBundles` fails if they disagree.
BUNDLE_DESCRIPTIONS: dict[str, str] = {
    "files": "read, search and edit files in the workspace (writes ask first)",
    "shell": "run commands in the workspace (always asks first)",
    "browser": "browse the web with agent-browser (reads untrusted pages)",
}


@dataclass
class BuiltinTools:
    """The tools, plus the things behind them that outlive a single call.

    A plain list would be enough today and wrong tomorrow: a container backend
    has to be stopped, and nothing in a list of tools can be asked to do that.
    Iterating still yields the tools, so callers that only want to mount them
    do not have to know any of this.
    """

    tools: list[Any] = field(default_factory=list)
    workspace: Workspace | None = None
    backend: CommandBackend | None = None
    browser: BrowserSessions | None = None

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.tools)

    def __len__(self) -> int:
        return len(self.tools)

    async def close(self) -> None:
        """Release anything the tools hold. Safe to call more than once, and
        safe when nothing was ever started."""
        # The browser first: closing it is a command, and needs the backend.
        if self.browser is not None:
            await self.browser.close()
        if self.backend is not None:
            await self.backend.close()


def bundle(
    *names: str,
    root: str | os.PathLike[str] | None = None,
    workspace: Workspace | None = None,
    backend: CommandBackend | str | None = None,
    allowed_env: Iterable[str] = (),
    browser: BrowserOptions | None = None,
) -> BuiltinTools:
    """Build the tools in the named bundles, bound to one workspace.

    Args:
        names: Bundle names — ``files``, ``shell``, ``browser``. An unknown name is logged
            and skipped: losing a capability is recoverable, and a typo in a
            config file should not stop an agent from starting.
        root: Workspace root. Defaults to the current directory.
        workspace: An already-built workspace, when several bundles share one.
        backend: Where commands run — a backend, or a name to resolve. An
            unknown *backend* name raises, unlike an unknown bundle name:
            handing back a local backend to someone who asked to be sandboxed
            gives them the illusion of a boundary, which is worse than none.
        allowed_env: Extra environment variable names ``run_command``'s
            subprocess may see, beyond the fixed safe set. Only used when
            *backend* is a name or omitted — a backend passed in already has
            its own environment policy.
        browser: How the ``browser`` bundle launches and fences its browser.

    Returns:
        A :class:`BuiltinTools`, which iterates as the list of tools.
    """
    space = workspace or Workspace.at(root)
    resolved: CommandBackend | None = None
    if isinstance(backend, str):
        resolved = build_backend(backend, space, allowed_env=allowed_env)
    elif backend is not None:
        resolved = backend

    tools: list[Any] = []
    sessions: BrowserSessions | None = None
    for name in names:
        if name == "files":
            tools.extend(file_tools(space))
        elif name == "shell":
            resolved = resolved or LocalBackend(space, allowed_env=allowed_env)
            tools.extend(shell_tools(space, backend=resolved))
        elif name == "browser":
            # The same backend as the shell, so the browser is wherever the
            # commands are — on this machine, or in the sandbox.
            resolved = resolved or LocalBackend(space, allowed_env=allowed_env)
            browser_list, sessions = browser_tools(backend=resolved, options=browser)
            tools.extend(browser_list)
        else:
            get_logger().warning(
                f"unknown tool bundle '{name}' (known: {', '.join(BUNDLES)}); skipping"
            )

    get_logger().info(
        f"built {len(tools)} built-in tool(s) from {list(names)} rooted at {space.root}"
        + (f" running on the {resolved.name} backend" if resolved else "")
    )
    return BuiltinTools(tools=tools, workspace=space, backend=resolved, browser=sessions)


class BuiltinBundleConfig(BaseModel):
    """A named, declarative request for built-in tool bundles.

    Exists because :func:`bundle` returns many tools from one call — ``files``
    alone is several ``FunctionTool``s — while a ``ConfigurationV1`` ``tools:``
    entry maps to exactly one ``BaseTool`` (``Tool.from_config`` is 1:1). A
    ``tool_bundles:`` list sidesteps that the same way ``mcp_servers:`` already
    does: declared once at the top level, referenced by name per agent, and
    expanded before the agent is built rather than resolved through the
    single-tool registry path.
    """

    name: str
    bundles: list[str] = PydanticField(default_factory=lambda: list(DEFAULT_BUNDLES))
    workspace: str | None = None
    sandbox: str = "local"
    isolate_environment: bool = True
    allowed_env: list[str] = PydanticField(default_factory=list)
    """Extra environment variable names ``run_command``'s subprocess may see,
    beyond the fixed safe set (``PATH``, ``HOME``, etc.). Empty by default —
    secrets loaded into the CLI's own environment (from ``.env`` or
    ``~/.domyn/credentials``) are not visible to shell commands unless named
    here."""
    browser: BrowserOptions = PydanticField(default_factory=BrowserOptions)
    """Options for the ``browser`` bundle; ignored unless it is listed."""

    def build(self, *, extra_roots: Iterable[str | os.PathLike[str]] = ()) -> BuiltinTools:
        """Build the bundle's tools.

        ``extra_roots`` widens the workspace the same way the CLI's own
        ``skill_dirs`` does today (see ``Workspace.extra_roots``) — passed in
        rather than declared in the YAML, because which directories those are
        is resolved at run time (the CLI's configured skill directories), not
        something a persisted file should freeze.
        """
        space = Workspace.at(self.workspace, also=extra_roots)
        backend = build_backend(
            self.sandbox,
            space,
            isolate_environment=self.isolate_environment,
            allowed_env=self.allowed_env,
        )
        return bundle(*self.bundles, workspace=space, backend=backend, browser=self.browser)


__all__ = [
    "BUNDLES",
    "BUNDLE_DESCRIPTIONS",
    "BUNDLE_TOOL_NAMES",
    "DEFAULT_BUNDLES",
    "BackendUnavailableError",
    "BrowserOptions",
    "BrowserSessions",
    "BuiltinBundleConfig",
    "BuiltinTools",
    "CommandBackend",
    "CommandResult",
    "LocalBackend",
    "Workspace",
    "WorkspaceError",
    "browser_tools",
    "build_backend",
    "bundle",
    "file_tools",
    "is_ignored",
    "run_command_tool",
    "shell_tools",
]
