"""The browser bundle against the real agent-browser, on a local test site.

Skipped when agent-browser is not installed. What it checks is what the
scripted tests in ``test_browser_tools.py`` cannot: that the JSON shapes, the
ref behaviour and the config precedence the module was built on are still what
the installed binary does.
"""

# No `from __future__ import annotations`: builds real tools.
import http.server
import json
import shutil
import threading
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import pytest

from domyn.agents.tools.builtin import BrowserOptions, bundle
from domyn.agents.tools.builtin.browser_sessions import BrowserSessions, session_name

pytestmark = pytest.mark.skipif(
    shutil.which("agent-browser") is None, reason="agent-browser is not installed"
)

PAGES = {
    "/": '<title>Home</title><h1>Test home</h1><a href="/form">Go to form</a>',
    "/form": (
        '<title>Form</title><form action="/done"><label>Name <input name="name"></label>'
        '<button type="submit">Send</button></form>'
    ),
    "/names": '<title>Names</title><a href="/card">Apple&nbsp;Card</a>',
}


class _Site(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        body = PAGES.get(self.path.split("?")[0], f"<title>{self.path}</title>{self.path}").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def site() -> Iterator[str]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Site)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


async def _call(tools: dict[str, Any], cid: str, name: str, /, **kwargs: Any) -> str:
    result = await tools[name].execute(conversation_id=cid, **kwargs)
    assert result.success, result.error
    return str(result.data)


async def _fail(tools: dict[str, Any], cid: str, name: str, /, **kwargs: Any) -> str:
    result = await tools[name].execute(conversation_id=cid, **kwargs)
    assert not result.success, result.data
    assert result.error_code == "BrowserError"
    return str(result.error)


def _ref(snapshot: str, name: str) -> str:
    line = next(line for line in snapshot.splitlines() if f'"{name}"' in line)
    return line.split("ref=")[1].split("]")[0]


@pytest.mark.asyncio
async def test_open_click_type_submit_and_close(site: str, tmp_path: Path) -> None:
    # A project config that would cap every output at 20 characters, were it
    # applied: it must not be.
    (tmp_path / "agent-browser.json").write_text(json.dumps({"maxOutput": 20}))
    built = bundle("browser", root=tmp_path, browser=BrowserOptions(allow_private_urls=True))
    tools = {t.name: t for t in built}
    cid = f"test-{uuid.uuid4()}"
    try:
        opened = await _call(tools, cid, "browser_open", url=f"{site}/")
        assert '"Go to form"' in opened and "truncated" not in opened

        go = {"url": f"{site}/", "ref": _ref(opened, "Go to form"), "name": "Go to form"}
        # A ref the model got wrong would "succeed" in agent-browser; here the
        # name it gave does not match, and nothing is clicked.
        wrong = await _fail(tools, cid, "browser_click", **{**go, "name": "Test home"})
        assert "Nothing was clicked" in wrong

        assert "refs have changed" in await _call(tools, cid, "browser_click", **go)
        # The old page's ref would still "succeed" in agent-browser; here the
        # page it names is not the one the browser is on.
        assert "Nothing was clicked" in await _fail(tools, cid, "browser_click", **go)

        form = await _call(tools, cid, "browser_snapshot")
        field = {"url": f"{site}/form", "ref": _ref(form, "Name"), "name": "Name"}
        typed = await _call(tools, cid, "browser_type", **field, text="Ada; rm -rf ~", submit=True)
        assert "/done?name=Ada%3B+rm+-rf+%7E" in typed

        read = await _call(tools, cid, "browser_read")
        assert "/done?name=" in read and "untrusted" in read
    finally:
        await built.close()

    run_dir = Path.home() / ".agent-browser" / "namespaces" / BrowserSessions.NAMESPACE / "run"
    sid = session_name(cid)
    assert not (run_dir / f"{sid}.target").exists()
    assert not (run_dir / f"{sid}.config").exists()


@pytest.mark.asyncio
async def test_text_with_a_space_is_typed_even_when_it_reads_like_an_option(
    site: str, tmp_path: Path
) -> None:
    """``browser_type`` refuses only one word starting with ``-``, because
    agent-browser reads an argument as one of its options only when the whole
    argument is one: ``--version`` is, `` --version`` is typed. An upgrade that
    trims arguments, or reads ``--flag=value`` out of a longer one, fails here."""
    built = bundle("browser", root=tmp_path, browser=BrowserOptions(allow_private_urls=True))
    tools = {t.name: t for t in built}
    cid = f"test-{uuid.uuid4()}"
    try:
        for text in (" --version", "--version ", "--headed=true for now"):
            form = await _call(tools, cid, "browser_open", url=f"{site}/form")
            field = {"url": f"{site}/form", "ref": _ref(form, "Name"), "name": "Name"}
            typed = await _call(tools, cid, "browser_type", **field, text=text, submit=True)
            assert f"/done?name={quote_plus(text)}" in typed, typed
    finally:
        await built.close()


@pytest.mark.asyncio
async def test_a_non_breaking_space_is_matched_as_the_snapshot_shows_it(
    site: str, tmp_path: Path
) -> None:
    """The JSON name keeps the non-breaking space; the text the model copies
    shows a plain one, and that must verify."""
    built = bundle("browser", root=tmp_path, browser=BrowserOptions(allow_private_urls=True))
    tools = {t.name: t for t in built}
    cid = f"test-{uuid.uuid4()}"
    try:
        opened = await _call(tools, cid, "browser_open", url=f"{site}/names")
        card = {"url": f"{site}/names", "ref": _ref(opened, "Apple Card"), "name": "Apple Card"}
        assert f"The page is now {site}/card" in await _call(tools, cid, "browser_click", **card)
    finally:
        await built.close()


@pytest.mark.asyncio
async def test_a_local_address_is_refused_by_default(site: str, tmp_path: Path) -> None:
    built = bundle("browser", root=tmp_path)
    tools = {t.name: t for t in built}
    try:
        reply = await _fail(tools, f"test-{uuid.uuid4()}", "browser_open", url=f"{site}/")
        assert reply.startswith("Cannot open") and "private or local" in reply
    finally:
        await built.close()
