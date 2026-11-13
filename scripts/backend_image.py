#!/usr/bin/env python3
"""Print the names the backend image is published under, as ``NAME=value`` lines.

They are defined once, in ``domyn/agents/tools/builtin/backends/docker/images.py``,
where the CLI pulls the image by them. CI reads them from there (without importing it:
no dependency is installed to do so) instead of repeating them, so the two cannot drift.
Used by .github/workflows/docker-backend-image.yml: ``python3 scripts/backend_image.py >> "$GITHUB_ENV"``.
"""

import ast
from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[1] / "domyn/agents/tools/builtin/backends/docker/images.py"
)
NAMES = ("IMAGE_REPOSITORY", "LATEST_TAG")


def read() -> dict[str, str]:
    found: dict[str, str] = {}
    for node in ast.parse(SOURCE.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in NAMES:
                    found[target.id] = str(node.value.value)
    missing = [name for name in NAMES if name not in found]
    if missing:
        raise SystemExit(f"{SOURCE} no longer defines {', '.join(missing)}")
    return found


if __name__ == "__main__":
    for name, value in read().items():
        print(f"{name}={value}")
