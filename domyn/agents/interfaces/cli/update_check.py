"""Tell the user, at most once a day, that a newer domyn is out.

`update_notice` is called by ``domyn run`` under the banner. It asks PyPI for the
latest stable release only when the last answer is older than a day, and keeps the
answer in ``~/.domyn/update-check.json`` with whether the check is on
(``/update-check on|off``). The file is always the global one, never a project's:
there is one installation to update, whichever project a session uses.
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
from packaging.version import InvalidVersion, Version

import domyn.agents
from domyn.agents.interfaces.cli import config as cli_config
from domyn.agents.logger import get_logger

UPDATE_CHECKS_FILENAME = "update-check.json"
PYPI_URL = "https://pypi.org/pypi/domyn/json"
CHECK_INTERVAL_SECONDS = 24 * 60 * 60
TIMEOUT_SECONDS = 1.5

RELEASES_URL = "https://github.com/domynai/domyn-agents/releases/latest/download"
# Written by install.sh and install.ps1 in the folder that holds the runtime.
STANDALONE_MARKER = ".domyn-runtime"


def _update_checks_path() -> Path:
    # Read at call time, so the tests' redirect of GLOBAL_DIR applies.
    return cli_config.GLOBAL_DIR / UPDATE_CHECKS_FILENAME


def _read_update_checks() -> dict[str, Any]:
    try:
        update_checks = json.loads(_update_checks_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return update_checks if isinstance(update_checks, dict) else {}


def _write_update_checks(update_checks: dict[str, Any]) -> None:
    path = _update_checks_path()
    tmp: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(update_checks, handle)
        os.replace(tmp, path)
    except OSError as exc:
        if tmp is not None:
            Path(tmp).unlink(missing_ok=True)
        get_logger().debug(f"cannot save {UPDATE_CHECKS_FILENAME}: {type(exc).__name__}: {exc}")


def is_enabled() -> bool:
    return _read_update_checks().get("enabled", True) is not False


def set_enabled(enabled: bool) -> None:
    _write_update_checks({**_read_update_checks(), "enabled": enabled})


def latest_version() -> str | None:
    """The latest stable release on PyPI, or None when it cannot be known."""
    try:
        response = httpx.get(PYPI_URL, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        latest = response.json()["info"]["version"]
    except Exception as exc:
        get_logger().debug(f"update check failed: {type(exc).__name__}: {exc}")
        return None
    return latest if isinstance(latest, str) else None


def update_command() -> str:
    """The command that updates domyn the way it was installed."""
    runtime = Path(sys.prefix)
    if (runtime.parent / STANDALONE_MARKER).is_file():
        if sys.platform == "win32":
            return f"irm {RELEASES_URL}/install.ps1 | iex"
        return f"curl -fsSL {RELEASES_URL}/install.sh | sh"
    if (runtime / "uv-receipt.toml").is_file():
        return "uv tool upgrade domyn"
    return 'pip install -U "domyn[cli]"'


def update_notice() -> str | None:
    """The line to show when a newer release is out, else None."""
    installed = domyn.agents.__version__
    # An editable install or a build between releases has nothing to update to.
    if installed == "unknown" or ".dev" in installed or "+" in installed:
        return None
    update_checks = _read_update_checks()
    if update_checks.get("enabled", True) is False:
        return None

    checked_at = update_checks.get("checked_at")
    if (
        not isinstance(checked_at, (int, float))
        or time.time() - checked_at > CHECK_INTERVAL_SECONDS
    ):
        latest = latest_version()
        # Stored even when the check failed, so an offline machine asks once a day.
        update_checks = {
            **update_checks,
            "checked_at": time.time(),
            "latest": latest or update_checks.get("latest"),
        }
        _write_update_checks(update_checks)

    latest = update_checks.get("latest")
    try:
        if not isinstance(latest, str) or Version(latest) <= Version(installed):
            return None
    except InvalidVersion:
        return None
    return f"A new version of domyn ({latest}) is available, update with: {update_command()}"
