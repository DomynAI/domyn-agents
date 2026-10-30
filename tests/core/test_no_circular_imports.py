"""Tests that verify no circular imports exist in any domyn.agents module."""

import ast
import importlib.util
import pkgutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest


def _discover_modules() -> list[str]:
    """Walk the domyn.agents package and collect every importable module name."""
    modules: list[str] = ["domyn.agents"]
    spec = importlib.util.find_spec("domyn.agents")
    if spec is None or spec.submodule_search_locations is None:
        return modules
    for info in pkgutil.walk_packages(
        path=list(spec.submodule_search_locations),
        prefix="domyn.agents.",
        onerror=lambda _: None,
    ):
        modules.append(info.name)
    return sorted(modules)


def _resolve_relative_import(mod: str, level: int, module: str | None, is_package: bool) -> str:
    """Resolve a relative import to an absolute module name."""
    parts = mod.split(".")
    # For a package __init__, level=1 means "this package"; for a module, level=1 means "parent package"
    adjustment = level - 1 if is_package else level
    base_parts = parts[: len(parts) - adjustment] if adjustment > 0 else parts
    if module:
        return ".".join(base_parts + module.split("."))
    return ".".join(base_parts)


def _find_root_modules(modules: list[str]) -> list[str]:
    """Return only modules not imported by any other module in the package.

    Importing a root exercises all its transitive dependencies, so testing roots
    alone gives the same circular-import coverage with far fewer subprocesses.
    """
    modules_set = set(modules)
    imported_by_others: set[str] = set()

    for mod in modules:
        spec = importlib.util.find_spec(mod)
        if spec is None or spec.origin is None:
            continue
        is_package = spec.submodule_search_locations is not None
        try:
            source = Path(spec.origin).read_text()
            tree = ast.parse(source)
        except Exception:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in modules_set:
                        imported_by_others.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0:
                    if node.module and node.module in modules_set:
                        imported_by_others.add(node.module)
                else:
                    resolved = _resolve_relative_import(mod, node.level, node.module, is_package)
                    if resolved in modules_set:
                        imported_by_others.add(resolved)

    return [m for m in modules if m not in imported_by_others]


MODULES_TO_CHECK = _find_root_modules(_discover_modules())


def _import_in_subprocess(module: str) -> subprocess.CompletedProcess[str]:
    script = dedent(f"""
        import importlib
        importlib.import_module("{module}")
    """)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("module", MODULES_TO_CHECK)
def test_no_circular_import(module: str) -> None:
    result = _import_in_subprocess(module)
    assert result.returncode == 0, (
        f"Importing '{module}' raised an error (likely a circular import):\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
