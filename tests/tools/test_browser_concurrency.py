"""Browser calls that run at the same time on one conversation's page.

Reproductions, not yet fixes: every test here is ``xfail(strict=True)`` and
documents a race the runner can produce today.

The runner runs consecutive calls with no ``approval_type`` concurrently
(``Runner._run_tool_group``), and sub-agents named in one response as branches
of the same conversation (``runner/branches.py``). Every browser tool resolves
its session from ``conversation_id`` alone (``session_name``), and
``BrowserSessions`` holds no lock, so all of them drive one page, and each tool
is several agent-browser commands that interleave with the other's.
"""

import asyncio
from typing import Any

import pytest

from domyn.agents.tools.builtin import CommandResult

from .test_browser_tools import CID, GO, SITE, ScriptedBrowser, _call, _opened


class InterleavingBrowser(ScriptedBrowser):
    """The scripted browser, yielding to the event loop before each answer, as
    a real ``backend.run`` (a subprocess, a ``docker exec``) does. It records
    the page every command acted on."""

    def __init__(self) -> None:
        super().__init__()
        self.acted_on: list[tuple[str, str]] = []

    async def run(
        self, command: str, *, timeout: float, conversation_id: str = "", stdin: str | None = None
    ) -> CommandResult:
        await asyncio.sleep(0)
        before, calls = self.url, len(self.calls)
        result = await super().run(
            command, timeout=timeout, conversation_id=conversation_id, stdin=stdin
        )
        if len(self.calls) > calls:
            self.acted_on.append((self.calls[-1][0], before))
        return result


@pytest.fixture
def fake() -> InterleavingBrowser:
    return InterleavingBrowser()


class TestOnePagePerConversation:
    @pytest.mark.asyncio
    @pytest.mark.xfail(
        strict=True,
        reason="verify() and the click are two commands; a concurrent navigation "
        "lands between them",
    )
    async def test_a_click_runs_only_on_the_page_it_verified(
        self, fake: InterleavingBrowser, tmp_path: Any
    ) -> None:
        """A sub-agent's approved click while a sibling branch goes back: the
        click was checked on the page it names, then sent to another one,
        where agent-browser clicks whatever has that ref."""
        tools = await _opened(fake, tmp_path)
        fake.acted_on.clear()
        await asyncio.gather(
            _call(tools, "browser_back"),
            _call(tools, "browser_click", **GO),
        )
        clicked_on = [url for verb, url in fake.acted_on if verb == "click"]
        assert clicked_on == [f"{SITE}/"]

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        strict=True,
        reason="ungated calls of one response run concurrently, so their order "
        "on the page is not the order the model wrote them in",
    )
    async def test_calls_of_one_response_act_in_the_order_they_were_made(
        self, fake: InterleavingBrowser, tmp_path: Any
    ) -> None:
        """`browser_back` then `browser_read`, in one response: both are
        ungated, so the runner gathers them, and the read describes the page
        before the back, or a mix of the two."""
        tools = await _opened(fake, tmp_path)
        await _call(tools, "browser_open", url=f"{SITE}/form")
        went_back, page = await asyncio.gather(
            _call(tools, "browser_back"),
            _call(tools, "browser_read"),
        )
        assert went_back.startswith(f"Went back. The page is now {SITE}/")
        assert "Welcome." in page
