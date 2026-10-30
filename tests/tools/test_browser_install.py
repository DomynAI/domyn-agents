"""Chrome downloaded on first use, and the agent-browser that ships with the build.

What can go wrong: unpacking a macOS app without its links and permissions,
a path in the archive that leaves the folder, a download that fails half way
and looks installed, downloading when Chrome is already there, and running
anywhere but this machine.
"""

import io
import os
import stat
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from domyn.agents.tools.builtin import browser_install as bi
from domyn.agents.tools.builtin.browser_sessions import (
    BrowserError,
    BrowserOptions,
    BrowserSessions,
)

KEY = "linux64"
EXE = bi._EXECUTABLES[KEY]


def chrome_zip(
    *, extra: dict[str, bytes] | None = None, link: tuple[str, str] | None = None
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        info = zipfile.ZipInfo(EXE)
        info.external_attr = (stat.S_IFREG | 0o755) << 16
        z.writestr(info, b"#!/bin/sh\n")
        for name, data in (extra or {}).items():
            plain = zipfile.ZipInfo(name)
            plain.external_attr = (stat.S_IFREG | 0o644) << 16
            z.writestr(plain, data)
        if link is not None:
            symlink = zipfile.ZipInfo(link[0])
            symlink.external_attr = (stat.S_IFLNK | 0o755) << 16
            z.writestr(symlink, link[1])
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def on_linux64(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bi, "chrome_platform", lambda: KEY)


def listing(version: str = "156.0.1.2") -> dict:
    return {
        "channels": {
            "Stable": {
                "version": version,
                "downloads": {
                    "chrome": [
                        {
                            "platform": KEY,
                            "url": f"https://storage.googleapis.com/chrome-for-testing-public/{version}/{KEY}/chrome-linux64.zip",
                        }
                    ]
                },
            }
        }
    }


class TestThePlatform:
    @pytest.mark.parametrize(
        ("system", "machine", "expected"),
        [
            ("darwin", "arm64", "mac-arm64"),
            ("darwin", "x86_64", "mac-x64"),
            ("linux", "x86_64", "linux64"),
            ("linux", "aarch64", "linux-arm64"),
            ("linux", "riscv64", None),
            ("win32", "AMD64", None),
        ],
    )
    def test_it_names_the_build_to_download(self, monkeypatch, system, machine, expected):
        monkeypatch.undo()
        monkeypatch.setattr(bi.sys, "platform", system)
        monkeypatch.setattr(bi.platform, "machine", lambda: machine)
        assert bi.chrome_platform() == expected


class TestFindingPrograms:
    def test_the_binary_on_path_wins(self, monkeypatch):
        monkeypatch.setattr(bi.shutil, "which", lambda name: "/usr/bin/agent-browser")
        assert bi.resolve_binary("agent-browser") == "agent-browser"

    def test_the_bundled_copy_is_used_when_nothing_is_on_path(self, monkeypatch, tmp_path):
        (tmp_path / "bin").mkdir()
        binary = tmp_path / "bin" / "agent-browser"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        monkeypatch.setattr(bi.shutil, "which", lambda name: None)
        monkeypatch.setattr(bi.sys, "prefix", str(tmp_path))
        assert bi.resolve_binary("agent-browser") == str(binary)

    def test_a_missing_binary_keeps_its_name_so_the_usual_error_follows(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(bi.shutil, "which", lambda name: None)
        monkeypatch.setattr(bi.sys, "prefix", str(tmp_path))
        assert bi.resolve_binary("agent-browser") == "agent-browser"

    def test_a_path_is_taken_as_given(self):
        assert bi.resolve_binary("/opt/ab/agent-browser") == "/opt/ab/agent-browser"

    def test_the_newest_downloaded_chrome_is_found(self, tmp_path):
        for version in ("9.0.0.1", "10.0.0.1"):
            path = tmp_path / f"chrome-{version}-{KEY}" / EXE
            path.parent.mkdir(parents=True)
            path.write_text("")
        assert bi.downloaded_chrome(tmp_path) == str(tmp_path / f"chrome-10.0.0.1-{KEY}" / EXE)

    def test_nothing_downloaded_is_none(self, tmp_path):
        assert bi.downloaded_chrome(tmp_path) is None
        assert bi.downloaded_chrome(tmp_path / "absent") is None

    def test_a_folder_without_chrome_in_it_is_not_one(self, tmp_path):
        (tmp_path / f"chrome-10.0.0.1-{KEY}").mkdir()
        assert bi.downloaded_chrome(tmp_path) is None


class TestUnpacking:
    def test_permissions_and_links_are_kept(self, tmp_path):
        archive = tmp_path / "a.zip"
        archive.write_bytes(
            chrome_zip(
                extra={"chrome-linux64/lib/x.so": b"x"}, link=("chrome-linux64/lib/y.so", "x.so")
            )
        )
        dest = tmp_path / "out"
        dest.mkdir()
        bi._extract(archive, dest)
        assert os.access(dest / EXE, os.X_OK)
        assert not os.access(dest / "chrome-linux64/lib/x.so", os.X_OK)
        assert (dest / "chrome-linux64/lib/y.so").is_symlink()
        assert (dest / "chrome-linux64/lib/y.so").read_bytes() == b"x"

    def test_a_path_leaving_the_folder_is_refused(self, tmp_path):
        archive = tmp_path / "a.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("../escape", b"x")
        dest = tmp_path / "out"
        dest.mkdir()
        with pytest.raises(bi.BrowserInstallError):
            bi._extract(archive, dest)
        assert not (tmp_path / "escape").exists()

    def test_a_link_leaving_the_folder_is_refused(self, tmp_path):
        archive = tmp_path / "a.zip"
        archive.write_bytes(chrome_zip(link=("chrome-linux64/evil", "../../../etc/passwd")))
        dest = tmp_path / "out"
        dest.mkdir()
        with pytest.raises(bi.BrowserInstallError):
            bi._extract(archive, dest)


class TestDownloading:
    @respx.mock
    def test_it_downloads_unpacks_and_announces(self, tmp_path):
        respx.get(bi.CHROME_VERSIONS_URL).mock(return_value=httpx.Response(200, json=listing()))
        respx.get(url__regex=r"https://storage\.googleapis\.com/.*").mock(
            return_value=httpx.Response(200, content=chrome_zip())
        )
        said: list[str] = []
        path = bi.download_chrome(tmp_path, said.append)
        assert path == str(tmp_path / f"chrome-156.0.1.2-{KEY}" / EXE)
        assert os.access(path, os.X_OK)
        assert len(said) == 1 and "156.0.1.2" in said[0] and str(tmp_path) in said[0]
        assert not [p for p in tmp_path.iterdir() if p.name.startswith(".download-")]

    @respx.mock
    def test_an_installed_version_is_not_downloaded_again(self, tmp_path):
        respx.get(bi.CHROME_VERSIONS_URL).mock(return_value=httpx.Response(200, json=listing()))
        existing = tmp_path / f"chrome-156.0.1.2-{KEY}" / EXE
        existing.parent.mkdir(parents=True)
        existing.write_text("")
        archive = respx.get(url__regex=r"https://storage\.googleapis\.com/.*")
        said: list[str] = []
        assert bi.download_chrome(tmp_path, said.append) == str(existing)
        assert not archive.called and said == []

    @respx.mock
    def test_a_download_cut_short_leaves_nothing_that_looks_installed(self, tmp_path):
        respx.get(bi.CHROME_VERSIONS_URL).mock(return_value=httpx.Response(200, json=listing()))
        respx.get(url__regex=r"https://storage\.googleapis\.com/.*").mock(
            return_value=httpx.Response(200, content=b"not a zip")
        )
        with pytest.raises(bi.BrowserInstallError):
            bi.download_chrome(tmp_path)
        assert bi.downloaded_chrome(tmp_path) is None
        assert list(tmp_path.iterdir()) == []

    @respx.mock
    def test_an_address_outside_google_storage_is_refused(self, tmp_path):
        bad = listing()
        bad["channels"]["Stable"]["downloads"]["chrome"][0]["url"] = (
            "https://evil.example.com/c.zip"
        )
        respx.get(bi.CHROME_VERSIONS_URL).mock(return_value=httpx.Response(200, json=bad))
        with pytest.raises(bi.BrowserInstallError):
            bi.download_chrome(tmp_path)

    @respx.mock
    def test_an_unreachable_server_is_an_install_error(self, tmp_path):
        respx.get(bi.CHROME_VERSIONS_URL).mock(side_effect=httpx.ConnectError("down"))
        with pytest.raises(bi.BrowserInstallError):
            bi.download_chrome(tmp_path)


class Backend:
    """Stands in for a command backend: only what the browser runtime asks of it."""

    def __init__(self, name: str = "local") -> None:
        self.name = name

    async def start(self) -> None: ...

    async def run(self, *args, **kwargs):
        raise AssertionError("not reached")


def sessions(name: str = "local", **options) -> BrowserSessions:
    return BrowserSessions(Backend(name), BrowserOptions(**options))  # type: ignore[arg-type]


class TestFirstUse:
    @pytest.mark.asyncio
    async def test_chrome_is_downloaded_when_there_is_none_and_the_browser_is_told_where(self):
        s = sessions()
        said: list[str] = []
        s.on_notice = said.append
        with (
            patch.object(bi, "downloaded_chrome", return_value=None),
            patch.object(bi, "system_chrome", return_value=None),
            patch.object(
                bi, "download_chrome", side_effect=lambda d, n: (n("downloading"), "/x/chrome")[1]
            ),
        ):
            await s._ensure_chrome()
        assert said == ["downloading"]
        assert "--executable-path /x/chrome" in s.command("c1", "open", "https://example.com")

    @pytest.mark.asyncio
    async def test_it_is_asked_once(self):
        s = sessions()
        with (
            patch.object(bi, "downloaded_chrome", return_value="/x/chrome") as found,
            patch.object(bi, "download_chrome") as download,
        ):
            await s._ensure_chrome()
            await s._ensure_chrome()
        assert found.call_count == 1
        download.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_chrome_the_user_has_is_left_alone(self):
        s = sessions()
        with (
            patch.object(bi, "downloaded_chrome", return_value=None),
            patch.object(bi, "system_chrome", return_value="/Applications/Chrome"),
            patch.object(bi, "download_chrome") as download,
        ):
            await s._ensure_chrome()
        download.assert_not_called()
        assert "--executable-path" not in s.command("c1", "open", "https://example.com")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"executable_path": "/my/chrome"},
            {"cdp_url": "ws://localhost:9222"},
            {"engine": "lightpanda"},
        ],
    )
    async def test_nothing_is_downloaded_when_the_options_name_a_browser(self, kwargs):
        s = sessions(**kwargs)
        with patch.object(bi, "download_chrome") as download:
            await s._ensure_chrome()
        download.assert_not_called()

    @pytest.mark.asyncio
    async def test_nothing_is_downloaded_for_commands_that_run_elsewhere(self):
        s = sessions(name="sandbox")
        with patch.object(bi, "download_chrome") as download:
            await s._ensure_chrome()
        download.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_environment_can_name_a_browser(self, monkeypatch):
        monkeypatch.setenv("AGENT_BROWSER_EXECUTABLE_PATH", "/env/chrome")
        s = sessions()
        with patch.object(bi, "download_chrome") as download:
            await s._ensure_chrome()
        download.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_failed_download_says_what_to_do_and_is_tried_again_next_time(self):
        s = sessions()
        with (
            patch.object(bi, "downloaded_chrome", return_value=None),
            patch.object(bi, "system_chrome", return_value=None),
            patch.object(bi, "download_chrome", side_effect=bi.BrowserInstallError("ConnectError")),
        ):
            with pytest.raises(BrowserError, match="agent-browser install"):
                await s._ensure_chrome()
            assert not s._chrome_checked

    @pytest.mark.asyncio
    async def test_call_makes_sure_first(self):
        s = sessions()
        s._ensure_chrome = AsyncMock(side_effect=BrowserError("no chrome"))  # type: ignore[method-assign]
        with pytest.raises(BrowserError, match="no chrome"):
            await s.call("c1", "open", "https://example.com")
        s._ensure_chrome.assert_awaited_once()

    def test_a_missing_shared_library_is_explained(self):
        s = sessions()
        from domyn.agents.tools.builtin.browser_sessions import _Reply

        error = s.failure(
            _Reply(
                ok=False, data={}, error="chrome: error while loading shared libraries: libnss3.so"
            ),
            "opening",
        )
        assert "libnss3" in str(error) and "apt install" in str(error)


class TestThePinnedBinary:
    """What the standalone build downloads: a version and a checksum for each target."""

    def test_every_target_has_a_checksum(self):
        lock = Path(__file__).parents[2] / "scripts" / "agent_browser.lock"
        values = dict(
            line.split("=", 1)
            for line in lock.read_text().splitlines()
            if line and not line.startswith("#")
        )
        assert values["AGENT_BROWSER_VERSION"].count(".") == 2
        for target in (
            "aarch64_apple_darwin",
            "x86_64_unknown_linux_gnu",
            "aarch64_unknown_linux_gnu",
        ):
            digest = values[f"AGENT_BROWSER_SHA256_{target}"]
            assert len(digest) == 64 and set(digest) <= set("0123456789abcdef")
