"""Where the browser tools find their programs, and the Chrome they download.

Two programs are needed. ``agent-browser`` ships inside the standalone build
(``<runtime>/bin``), found here when it is not on ``PATH``. Chrome is not
shipped: the first browser tool call that finds none downloads Chrome for Testing
(Google's channel for automation, the one ``agent-browser install`` uses) into
``~/.domyn/browsers/``, once, and the browser is told where it is.

Only for commands that run on this machine. A sandbox backend would need the
download inside the sandbox, which this module cannot do from here.
"""

import os
import platform
import shutil
import stat
import sys
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from domyn.agents.logger import get_logger

CHROME_VERSIONS_URL = (
    "https://googlechromelabs.github.io/chrome-for-testing/"
    "last-known-good-versions-with-downloads.json"
)
_DOWNLOAD_HOST = "storage.googleapis.com"
_APPROX_SIZE = "about 200 MB"

# Chrome for Testing platform -> where Chrome is inside the unpacked archive.
_EXECUTABLES = {
    "mac-arm64": "chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
    "mac-x64": "chrome-mac-x64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
    "linux64": "chrome-linux64/chrome",
    "linux-arm64": "chrome-linux-arm64/chrome",
}

# Where a Chrome the user already has is usually found. agent-browser looks for
# one itself; this only decides whether to download another, so missing one costs
# a download and no more.
_SYSTEM_CHROME = {
    "darwin": (
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    ),
    "linux": (),
}
_SYSTEM_NAMES = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome")


class BrowserInstallError(Exception):
    """Chrome could not be downloaded or unpacked."""


def chrome_platform() -> str | None:
    """The Chrome for Testing platform of this machine, or ``None`` when it has none."""
    machine = platform.machine().lower()
    arm = machine in ("arm64", "aarch64")
    system: str = sys.platform
    if system == "darwin":
        return "mac-arm64" if arm else "mac-x64"
    if system.startswith("linux"):
        return "linux-arm64" if arm else ("linux64" if machine in ("x86_64", "amd64") else None)
    return None


def default_browsers_dir() -> Path:
    # Looked up when asked, not bound at import, so tests can point it elsewhere.
    return Path.home() / ".domyn" / "browsers"


def bundled_binary(name: str = "agent-browser") -> str | None:
    """The copy of ``name`` that ships with the standalone build, if there is one."""
    candidate = Path(sys.prefix) / "bin" / name
    return str(candidate) if candidate.is_file() and os.access(candidate, os.X_OK) else None


def resolve_binary(name: str) -> str:
    """``name`` as it will be run: itself when it is on ``PATH`` or is a path, else
    the bundled copy, else ``name`` so the failure is the usual one."""
    if os.sep in name or shutil.which(name):
        return name
    return bundled_binary(name) or name


def _version_key(directory: Path) -> tuple[int, ...]:
    version = directory.name.split("-")[1] if directory.name.count("-") >= 1 else ""
    try:
        return tuple(int(part) for part in version.split("."))
    except ValueError:
        return ()


def downloaded_chrome(browsers_dir: Path | None = None) -> str | None:
    """The newest Chrome this module downloaded, or ``None``."""
    key = chrome_platform()
    if key is None:
        return None
    root = (browsers_dir or default_browsers_dir()).expanduser()
    if not root.is_dir():
        return None
    found = [
        d for d in root.glob(f"chrome-*-{key}") if d.is_dir() and (d / _EXECUTABLES[key]).is_file()
    ]
    if not found:
        return None
    return str(max(found, key=_version_key) / _EXECUTABLES[key])


def system_chrome() -> str | None:
    """A Chrome or Chromium the user already has, or ``None``."""
    for path in _SYSTEM_CHROME.get("darwin" if sys.platform == "darwin" else "linux", ()):
        if os.access(path, os.X_OK):
            return path
    for name in _SYSTEM_NAMES:
        found = shutil.which(name)
        if found:
            return found
    return None


def _extract(archive: Path, dest: Path) -> None:
    """Unpack a zip with the permissions and symbolic links it records.

    ``ZipFile.extractall`` drops both, and a macOS app is made of links. Nothing
    is written outside ``dest``, a link included.
    """
    root = dest.resolve()

    def inside(path: Path) -> bool:
        return path == root or root in path.parents

    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            target = (root / info.filename).resolve()
            if not inside(target):
                raise BrowserInstallError(
                    f"the archive names a path outside itself: {info.filename}"
                )
            mode = info.external_attr >> 16
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif stat.S_ISLNK(mode):
                link_to = bundle.read(info).decode()
                if not inside((target.parent / link_to).resolve()):
                    raise BrowserInstallError(f"a link leaves the archive: {info.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(link_to, target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info) as source, open(target, "wb") as out:
                    shutil.copyfileobj(source, out)
                if mode & 0o111:
                    target.chmod(target.stat().st_mode | 0o111)


def _latest_stable(key: str) -> tuple[str, str]:
    response = httpx.get(CHROME_VERSIONS_URL, timeout=30.0, follow_redirects=True)
    response.raise_for_status()
    stable: dict[str, Any] = response.json()["channels"]["Stable"]
    for entry in stable["downloads"]["chrome"]:
        if entry["platform"] == key:
            url = str(entry["url"])
            if httpx.URL(url).host != _DOWNLOAD_HOST:
                raise BrowserInstallError(f"unexpected download address: {url}")
            return str(stable["version"]), url
    raise BrowserInstallError(f"Chrome for Testing has no build for {key}")


def download_chrome(
    browsers_dir: Path | None = None,
    on_notice: Callable[[str], None] | None = None,
) -> str:
    """Download the current stable Chrome for Testing and return its executable.

    Blocking: run it off the event loop. Unpacked in a temporary folder beside the
    destination and renamed, so a download cut short leaves nothing that looks
    installed.
    """
    key = chrome_platform()
    if key is None:
        raise BrowserInstallError(
            f"no Chrome for Testing build for {sys.platform}/{platform.machine()}"
        )
    root = (browsers_dir or default_browsers_dir()).expanduser()
    try:
        version, url = _latest_stable(key)
        final = root / f"chrome-{version}-{key}"
        if (final / _EXECUTABLES[key]).is_file():
            return str(final / _EXECUTABLES[key])

        if on_notice is not None:
            on_notice(f"Downloading Chrome {version} ({_APPROX_SIZE}) to {root}: only this once.")
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root, prefix=".download-") as scratch:
            archive = Path(scratch) / "chrome.zip"
            with (
                httpx.stream(
                    "GET", url, timeout=httpx.Timeout(30.0, read=120.0), follow_redirects=True
                ) as response,
                open(archive, "wb") as out,
            ):
                response.raise_for_status()
                for chunk in response.iter_bytes(1 << 20):
                    out.write(chunk)
            unpacked = Path(scratch) / "unpacked"
            unpacked.mkdir()
            _extract(archive, unpacked)
            if not (unpacked / _EXECUTABLES[key]).is_file():
                raise BrowserInstallError("the download does not contain Chrome where expected")
            if final.exists():
                shutil.rmtree(final)
            unpacked.rename(final)
    except (httpx.HTTPError, OSError, zipfile.BadZipFile, KeyError, ValueError) as exc:
        get_logger().debug(f"browser: chrome download detail: {exc}")
        raise BrowserInstallError(f"{type(exc).__name__}") from exc
    get_logger().info(f"browser: Chrome {version} installed in {final}")
    return str(final / _EXECUTABLES[key])
