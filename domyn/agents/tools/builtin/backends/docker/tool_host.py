"""What a container runs to host a tool body: one request on stdin, one answer on stdout.

This is a script, like ``egress_proxy.py`` beside it, run by its path
(``python -P tool_host.py``) by the Docker image and never imported, and that is the point. The body is a file tool (``builtin/files.py``, ``main``) and needs none of
what ``domyn.agents.tools`` imports when it is imported as a package, the MCP client
above all (about 0.5 s, and a call is one process: the whole cost of a read). Python runs
a package's ``__init__`` before any module in it, so even this module, were it imported,
would pay for it. Run by path it pays nothing, and puts the two packages above the file
tools in place as bare names with a path, so ``files`` is imported through them without
their ``__init__`` being run.

``-P`` keeps this folder off ``sys.path``: its ``cli.py`` and ``state.py`` would stand in
for any module of those names. Only for a process that does one thing and exits: in one that also uses the
agent machinery the stubs would hide the real packages.
"""

import sys
import types
from pathlib import Path

# .../tools/builtin/backends/docker/tool_host.py
_BUILTIN = Path(__file__).resolve().parents[2]
_TOOLS = _BUILTIN.parent


def _skip_package_inits() -> None:
    for name, folder in (
        ("domyn.agents.tools", _TOOLS),
        ("domyn.agents.tools.builtin", _BUILTIN),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(folder)]
        sys.modules[name] = package


def main() -> int:
    _skip_package_inits()
    from domyn.agents.tools.builtin.files import main as serve_one

    return serve_one()


if __name__ == "__main__":
    raise SystemExit(main())
