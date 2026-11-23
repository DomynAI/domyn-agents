"""Browser calls that run at the same time.

The runner runs sub-agents named in one response as branches of the same
conversation, at the same time (``runner/branches.py``), and a tool is several
agent-browser commands (verify, act, see where it landed). So:

- every call to a sub-agent has a browser of its own (``Session``), and the
  agent the conversation is addressed to keeps the conversation's: two agents
  never share a page;
- on one page, a tool runs whole before the next starts (``exclusive``), so an
  approved action runs on the page it verified, or not at all;
- a sub-agent's browser is closed when its call ends (``end_agent_call``).

The order of the calls of one response is the runner's (``parallel_safe``),
tested in ``tests/runner/test_tool_agent_call.py``.
"""

import asyncio
from typing import Any

import pytest

from domyn.agents.tools.builtin import BuiltinTools, CommandResult, bundle
from domyn.agents.tools.builtin.browser_sessions import (
    BrowserOptions,
    Session,
    session_name,
)

from .test_browser_tools import (  # `_no_dns`: an autouse fixture
    CID,
    GO,
    SITE,
    ScriptedBrowser,
    _no_dns,
    _parse_line,
)


class PerSessionBrowser(ScriptedBrowser):
    """The scripted browser with a page per agent-browser session, as the real
    one has, yielding to the event loop before each answer as a real
    ``backend.run`` (a subprocess, a ``docker exec``) does. It records the
    session and the page every command acted on."""

    def __init__(self) -> None:
        super().__init__()
        self.pages_of: dict[str, tuple[str, list[str]]] = {}
        self.acted_on: list[tuple[str, str, str]] = []

    async def run(
        self, command: str, *, timeout: float, conversation_id: str = "", stdin: str | None = None
    ) -> CommandResult:
        await asyncio.sleep(0)
        if command.startswith("cat > ") or self.missing:
            return await super().run(
                command, timeout=timeout, conversation_id=conversation_id, stdin=stdin
            )
        sid = _session_of(command)
        self.url, self.history = self.pages_of.get(sid, ("about:blank", []))
        before, calls = self.url, len(self.calls)
        result = await super().run(
            command, timeout=timeout, conversation_id=conversation_id, stdin=stdin
        )
        self.pages_of[sid] = (self.url, self.history)
        if len(self.calls) > calls:
            self.acted_on.append((sid, self.calls[-1][0], before))
        return result

    def page(self, session: Session | str) -> str:
        return self.pages_of.get(Session.of(session).name, ("about:blank", []))[0]

    def closed_sessions(self) -> list[str]:
        return [sid for sid, verb, _ in self.acted_on if verb == "close"]


def _session_of(command: str) -> str:
    tokens = command.split()
    return tokens[tokens.index("--session") + 1]


@pytest.fixture
def fake() -> PerSessionBrowser:
    return PerSessionBrowser()


def _built(fake: PerSessionBrowser, tmp_path: Any) -> BuiltinTools:
    return bundle("browser", root=tmp_path, backend=fake, browser=BrowserOptions())


def _by_name(built: BuiltinTools) -> dict[str, Any]:
    return {t.name: t for t in built}


async def _do(tools: dict[str, Any], name: str, agent_call: str = "", /, **kwargs: Any) -> Any:
    """A call as the runner makes it: the reserved arguments, then the model's."""
    return await tools[name].execute(conversation_id=CID, agent_call_id=agent_call, **kwargs)


async def _ok(tools: dict[str, Any], name: str, agent_call: str = "", /, **kwargs: Any) -> str:
    result = await _do(tools, name, agent_call, **kwargs)
    assert result.success, result.error
    return str(result.data)


class TestOnePagePerAgentCall:
    def test_the_conversation_keeps_the_name_it_always_had(self) -> None:
        """No sub-agent call: the session of an existing conversation is found
        again after an upgrade."""
        assert Session(CID).name == session_name(CID)
        assert len({session_name(CID), session_name(CID, "a"), session_name(CID, "b")}) == 3

    @pytest.mark.asyncio
    async def test_a_sub_agent_does_not_move_the_callers_page(
        self, fake: PerSessionBrowser, tmp_path: Any
    ) -> None:
        tools = _by_name(_built(fake, tmp_path))
        await _ok(tools, "browser_open", url=f"{SITE}/")

        await _ok(tools, "browser_open", "call-a", url=f"{SITE}/form")

        assert fake.page(Session(CID)) == f"{SITE}/"
        assert fake.page(Session(CID, "call-a")) == f"{SITE}/form"

    @pytest.mark.asyncio
    async def test_a_sub_agent_starts_on_no_page(
        self, fake: PerSessionBrowser, tmp_path: Any
    ) -> None:
        """It does not inherit the caller's page, nor what the caller is logged in to."""
        tools = _by_name(_built(fake, tmp_path))
        await _ok(tools, "browser_open", url=f"{SITE}/")

        result = await _do(tools, "browser_read", "call-a")

        assert not result.success
        assert "No page is open in your browser" in str(result.error)

    @pytest.mark.asyncio
    async def test_sub_agents_at_the_same_time_each_read_their_own_page(
        self, fake: PerSessionBrowser, tmp_path: Any
    ) -> None:
        tools = _by_name(_built(fake, tmp_path))

        async def browse(agent_call: str, url: str) -> str:
            await _ok(tools, "browser_open", agent_call, url=url)
            await _ok(tools, "browser_scroll", agent_call, direction="down")
            return await _ok(tools, "browser_read", agent_call)

        home, form = await asyncio.gather(
            browse("call-a", f"{SITE}/"), browse("call-b", f"{SITE}/form")
        )

        assert f"url={SITE}/ " in home and "Welcome." in home
        assert f"url={SITE}/form " in form and "Welcome." not in form

    @pytest.mark.asyncio
    async def test_two_copies_of_one_agent_are_two_calls(
        self, fake: PerSessionBrowser, tmp_path: Any
    ) -> None:
        """Copies of one agent called in one response share its tools, not a page."""
        tools = _by_name(_built(fake, tmp_path))

        await asyncio.gather(
            _ok(tools, "browser_open", "copy-1", url=f"{SITE}/"),
            _ok(tools, "browser_open", "copy-2", url=f"{SITE}/form"),
        )

        assert fake.page(Session(CID, "copy-1")) == f"{SITE}/"
        assert fake.page(Session(CID, "copy-2")) == f"{SITE}/form"

    @pytest.mark.asyncio
    async def test_commands_still_run_where_the_conversation_runs(
        self, fake: PerSessionBrowser, tmp_path: Any
    ) -> None:
        """A container is per conversation: a sub-agent's browser is in it too."""
        seen: list[str] = []
        original = fake.run

        async def run(command: str, **kwargs: Any) -> CommandResult:
            seen.append(kwargs.get("conversation_id", ""))
            return await original(command, **kwargs)

        fake.run = run  # type: ignore[method-assign]
        tools = _by_name(_built(fake, tmp_path))
        await _ok(tools, "browser_open", "call-a", url=f"{SITE}/")

        assert seen and set(seen) == {CID}


class TestOneToolAtATimeOnAPage:
    @pytest.mark.asyncio
    async def test_an_approved_click_runs_only_on_the_page_it_verified(
        self, fake: PerSessionBrowser, tmp_path: Any
    ) -> None:
        """A navigation started first runs whole; the click then verifies the page
        it lands on, and refuses, rather than being checked on one page and sent
        to another, where agent-browser clicks whatever has that ref."""
        tools = _by_name(_built(fake, tmp_path))
        await _ok(tools, "browser_open", url=f"{SITE}/form")
        await _ok(tools, "browser_open", url=f"{SITE}/")
        fake.acted_on.clear()

        went_back, clicked = await asyncio.gather(
            _do(tools, "browser_back"), _do(tools, "browser_click", **GO)
        )

        assert went_back.success
        assert not clicked.success
        assert str(clicked.error).startswith(
            f"The page is {SITE}/form, not {SITE}/. Nothing was clicked."
        )
        assert [verb for _, verb, _ in fake.acted_on if verb == "click"] == []

    @pytest.mark.asyncio
    async def test_the_lock_holds_across_the_agents_tools(
        self, fake: PerSessionBrowser, tmp_path: Any
    ) -> None:
        """Every agent that declares the browser builds its own tools: the lock is
        the page's, not one agent's."""
        first, second = _by_name(_built(fake, tmp_path)), _by_name(_built(fake, tmp_path))
        await _ok(first, "browser_open", url=f"{SITE}/")
        fake.acted_on.clear()

        await asyncio.gather(_do(second, "browser_back"), _do(first, "browser_click", **GO))

        assert [page for _, verb, page in fake.acted_on if verb == "click"] in ([], [f"{SITE}/"])

    @pytest.mark.asyncio
    async def test_calls_on_one_page_act_in_the_order_they_were_made(
        self, fake: PerSessionBrowser, tmp_path: Any
    ) -> None:
        """`browser_back` then `browser_read`: the read describes the page after
        the back, never the one before, nor a mix of the two."""
        tools = _by_name(_built(fake, tmp_path))
        await _ok(tools, "browser_open", url=f"{SITE}/")
        await _ok(tools, "browser_open", url=f"{SITE}/form")

        went_back, page = await asyncio.gather(
            _ok(tools, "browser_back"), _ok(tools, "browser_read")
        )

        assert went_back.startswith(f"Went back. The page is now {SITE}/")
        assert "Welcome." in page

    @pytest.mark.asyncio
    async def test_other_pages_are_not_held_up(
        self, fake: PerSessionBrowser, tmp_path: Any
    ) -> None:
        """The lock is per page: a sub-agent busy on its page holds up no other."""
        built = _built(fake, tmp_path)
        tools = _by_name(built)
        assert built.browser is not None

        async with built.browser.exclusive(Session(CID, "call-a")):
            opened = await asyncio.wait_for(
                _ok(tools, "browser_open", "call-b", url=f"{SITE}/form"), timeout=5
            )

        assert opened.startswith(f"Opened {SITE}/form")


class TestReleasedWhenTheCallEnds:
    @pytest.mark.asyncio
    async def test_the_sub_agents_browser_is_closed_and_the_callers_is_not(
        self, fake: PerSessionBrowser, tmp_path: Any
    ) -> None:
        built = _built(fake, tmp_path)
        tools = _by_name(built)
        await _ok(tools, "browser_open", url=f"{SITE}/")
        await _ok(tools, "browser_open", "call-a", url=f"{SITE}/form")

        await built.end_agent_call(CID, "call-a")

        assert fake.closed_sessions() == [session_name(CID, "call-a")]
        leftovers = [line for line in fake.lines if "rm -f" in line]
        assert leftovers and session_name(CID, "call-a") in leftovers[-1]
        # The caller's page is where it was, and the next close closes only it.
        assert fake.page(Session(CID)) == f"{SITE}/"
        await built.close()
        assert fake.closed_sessions() == [session_name(CID, "call-a"), session_name(CID)]

    @pytest.mark.asyncio
    async def test_a_call_that_never_browsed_sends_nothing(
        self, fake: PerSessionBrowser, tmp_path: Any
    ) -> None:
        built = _built(fake, tmp_path)

        await built.end_agent_call(CID, "call-a")
        await built.end_agent_call(CID, "")

        assert fake.lines == []

    @pytest.mark.asyncio
    async def test_the_end_of_the_callers_own_call_closes_nothing(
        self, fake: PerSessionBrowser, tmp_path: Any
    ) -> None:
        """The runner reports every AGENT_END, the root's included: the root's
        browser is the conversation's, and outlives the turn."""
        built = _built(fake, tmp_path)
        await _ok(_by_name(built), "browser_open", url=f"{SITE}/")

        await built.end_agent_call(CID, "the-roots-interaction")

        assert fake.closed_sessions() == []

    @pytest.mark.asyncio
    async def test_closing_waits_for_a_call_still_on_the_page(
        self, fake: PerSessionBrowser, tmp_path: Any
    ) -> None:
        """A close sent during a command would be undone by it (agent-browser
        starts the session again), so it waits its turn."""
        built = _built(fake, tmp_path)
        tools = _by_name(built)
        await _ok(tools, "browser_open", "call-a", url=f"{SITE}/")

        await asyncio.gather(
            _ok(tools, "browser_scroll", "call-a", direction="down"),
            built.end_agent_call(CID, "call-a"),
        )

        verbs = [verb for sid, verb, _ in fake.acted_on if sid == session_name(CID, "call-a")]
        assert verbs[-1] == "close"

    def test_the_session_flag_is_the_one_parsed(self) -> None:
        """The fake reads the session from the command line the module builds."""
        line = f"agent-browser --session {session_name(CID, 'x')} get url"
        assert _session_of(line) == session_name(CID, "x")
        assert _parse_line(line) == ("get", ["url"])
