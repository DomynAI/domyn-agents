r"""Browsing, by driving the ``agent-browser`` CLI through a command backend.

Every browser command is a shell line handed to the same
:class:`~domyn.agents.tools.builtin.backends.CommandBackend` ``run_command``
uses, so the browser runs wherever commands run: on this machine today, in a
sandbox once a backend for one exists. Nothing here talks to a browser
directly, and nothing has to change when the backend does.

``agent-browser`` (Vercel Labs, Apache-2.0) keeps a daemon per session that
holds the page between calls, labels elements with refs (``@e5``) in an
accessibility snapshot, and drives Chrome or Lightpanda. What it does not do is
decide what a model may reach, and that is most of this module.

**Measured, not assumed** — agent-browser 0.38.1, Chrome for Testing 154,
Lightpanda 1.0.0-nightly, macOS arm64, every command through
``LocalBackend.run``. The design follows from these, so they are kept here:

* **The daemon detaches.** The first call returns once the browser is up (2.2 s
  for Chrome, 0.3 s for Lightpanda), later ones in 10-20 ms, and the daemon
  survives the process-group kill a timeout triggers. No workaround needed.
* **Nothing bounds a navigation.** ``AGENT_BROWSER_DEFAULT_TIMEOUT`` covers
  actions, not ``open``: a 20 s page took 20.7 s with it at 3 s. After *our*
  timeout the navigation carries on and the next command queues behind it —
  17 s, measured — and a ``close`` sent meanwhile is undone by the queued
  ``open`` starting the session again. So a timeout is reported as "may still
  be busy", and never answered with ``close``.
* **A wrong ref succeeds silently.** ``click @e5`` on the page after a form
  submit answered ``clicked`` and did nothing; ``click @e5`` meant for ``@e6``
  clicks ``@e5``. Within one document refs are stable: snapshot after snapshot
  (interactive or full) numbered the same elements the same way, and an
  element inserted at the top got the next free number rather than shifting
  the rest. Hence the gated actions: the model names the page, the ref *and*
  the element's name, the approval prompt is built from those claims alone, and
  the action takes a fresh snapshot and refuses unless all three still hold.
* **The snapshot text prints a non-breaking space as a plain one**; the JSON
  ``refs`` keep it (apple.com: 13 of 158 refs). The model copies the text, so
  names are compared with every whitespace character read as a plain space.
* **Clicks do not report where they went.** ``open`` returns the final URL
  after redirects; ``click``/``fill``/``press`` return ``{clicked}`` and the
  like, without waiting for a navigation they start (0.02 s for a link to a
  20 s page). The *next* command waits for it, so the ``get url`` after an
  action is accurate. A navigation a page script starts later (a ``setTimeout``
  after the click) is invisible at that moment, which is why the URL is also
  checked at the start of every call.
* **It opens anything.** ``file:///etc/passwd`` and ``data:`` URLs load;
  ``javascript:`` is refused. A URL without a scheme gets ``https://``, and
  Python's ``urlparse("localhost:3000")`` reads ``localhost`` as the scheme —
  so the policy normalises first, then checks, then passes on the normalised
  string, so the checker and the browser read the same URL. Hosts still part
  ways: Chrome decodes a %-escape in one (``127%2e0%2e0%2e1``), maps non-ASCII
  characters with a newer Unicode table than Python's IDNA codec, and ends the
  host at a backslash (``127.0.0.1:80\@example.com``, where ``urlsplit`` sees
  ``example.com``). Six such URLs passed the check and reached a loopback
  server; the policy refuses all three forms.
* **Configuration files are code.** Both ``./agent-browser.json`` in the
  working directory (the workspace root) and ``~/.agent-browser/config.json``
  are applied, and either can name an executable, extensions, init scripts or
  plugins. A cloned repository would get code run on the first, ungated
  ``browser_open``. ``--config`` replaces both, so every command passes
  ``/dev/stdin`` with ``{}`` piped in. A missing or invalid file is an error
  (``/dev/null`` is one), and a file written in the workspace would be one a
  repository can replace with a symlink. The daemon reads it once, at launch.
* **The allowlist contains the page.** ``--allowed-domains`` blocked a page's
  own ``fetch`` and image requests to an address outside it; without it both
  reached the server. It sticks to the session once set, and cannot be
  combined with ``--cdp``. Without it, requests a page makes are not seen by
  the URL policy here, which checks navigations only.
* **Sockets have a 103-byte path budget**, under
  ``$HOME/.agent-browser/namespaces/<ns>/run/<session>.sock``: a 37-character
  session name failed. Sessions are therefore named by a short hash.
* **``close`` leaves the last URL on disk**, in ``<session>.target``, in plain
  text. Teardown removes it.
* **JSON output is not truncated** by ``--max-output`` (365k characters came
  back with it at 300), and ``--content-boundaries`` only adds a field to it.
  Truncation and framing are done here.
* **A Chrome session costs about 1 GB** of resident memory across 14
  processes; Lightpanda about 58 MB. One conversation, one session — hence a
  short idle timeout.
"""

# No `from __future__ import annotations` here: FunctionTool reads the signature
# of each tool at construction time and cannot resolve string annotations.

import re
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

from domyn.agents.core import ApprovalType
from domyn.agents.tools.builtin.backends import CommandBackend
from domyn.agents.tools.builtin.browser_sessions import (
    BrowserOptions,
    BrowserSessions,
    UrlPolicy,
    _ref_key,
)
from domyn.agents.tools.builtin.shell import flag

# What an origin shown in an approval prompt may contain: a host (or bracketed
# IPv6 address) and a port. Anything else — a space, a second "::" — could make
# a claimed URL read like another site's prompt line.
_PLAIN_ORIGIN = re.compile(r"^https?://(?:[a-z0-9.-]+|\[[0-9a-f:.]+\])(?::\d{1,5})?$")
# Field names whose typed text is not repeated in the approval prompt. The
# prompt line is also the key an exact grant is stored under.
_SECRET_FIELD = re.compile(r"pass|secret|token|key|otp|pin|cvv|cvc|card|ssn", re.IGNORECASE)


def claimed_origin(url: str) -> str | None:
    """The site a model's ``url`` argument names, for an approval prompt, or
    ``None`` when it names none that can be shown safely.

    ``https://github.com`` for any page on it. Built from the argument alone,
    before anything runs. That is only safe because the action refuses to run
    anywhere else; and only readable when the origin is a plain host, so a
    crafted URL cannot open a prompt line with a granted site's prefix.
    """
    try:
        parts = urlsplit(UrlPolicy.normalize(str(url)))
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None
    origin = f"{parts.scheme}://{parts.netloc.lower()}"
    return origin if _PLAIN_ORIGIN.match(origin) else None


# --- Approval --------------------------------------------------------------
#
# Built from the model's arguments alone: what is approved is what was asked
# for, and the action refuses to run if the page does not match it (`verify`).
# The site goes first, so a grant for it is a pattern anchored on it: an element
# named "https://github.com :: x" on another site cannot make that site's line
# look like github's, and `claimed_origin` keeps a crafted URL from doing it.


def _site(url: str) -> str:
    return claimed_origin(url) or "(invalid url)"


def _target(ref: str, name: str) -> str:
    key = _ref_key(ref) or ref
    return f"{name!r} [@{key}]"


def click_subject(url: str = "", ref: str = "", name: str = "", **_: Any) -> str:
    return f"{_site(url)} :: click {_target(ref, name)}"


def type_subject(
    url: str = "", ref: str = "", name: str = "", text: str = "", submit: bool = False, **_: Any
) -> str:
    typed = (
        f"({len(str(text))} characters, not shown)"
        if _SECRET_FIELD.search(str(name))
        else repr(text)
    )
    enter = " and press Enter" if flag(submit) else ""
    return f"{_site(url)} :: type {typed} into {_target(ref, name)}{enter}"


def press_subject(url: str = "", key: str = "", **_: Any) -> str:
    return f"{_site(url)} :: press {key!r}"


def site_pattern(url: str = "", **_: Any) -> str | None:
    origin = claimed_origin(url)
    return f"{origin} :: *" if origin else None


# --- Tools -----------------------------------------------------------------


def browser_tools(
    backend: CommandBackend, options: BrowserOptions | None = None
) -> tuple[list[Any], BrowserSessions]:
    """The browser tools, and the sessions behind them so a caller can close them.

    Split by risk, like the file tools: reading the page is ungated, acting on
    it is gated per site. Opening a page is ungated and fenced by the URL
    policy instead — asking before every site would be asking before reading.
    """
    from domyn.agents.tools.function_tool import FunctionTool

    sessions = BrowserSessions(backend, options or BrowserOptions())

    async def browser_open(url: str, conversation_id: str = "") -> str:
        """Open a web page in this conversation's browser and show its interactive elements.

        The page stays open between calls, with its cookies, so you can act on it
        with the other browser tools. The result lists the elements you can click
        or type into, each with a ref (e5). To read the page's text, use
        browser_read. Only http and https pages on public addresses can be
        opened.

        Args:
            url: The address to open, e.g. https://example.com/docs.
        """
        result = await sessions.open(url=url, conversation_id=conversation_id)
        return result

    async def browser_snapshot(full: bool = False, conversation_id: str = "") -> str:
        """Show the elements on the current page, each with a ref to act on.

        Call it after anything that may have changed the page - a click that
        navigated, a form submit - because refs from an older snapshot are no
        longer valid. The default lists only interactive elements (links,
        buttons, fields); for the page's text use browser_read.

        Args:
            full: Include every element, not only interactive ones. Much longer.
        """
        result = await sessions.take_snapshot(conversation_id, full=flag(full))
        return result

    async def browser_read(url: str = "", ref: str = "", conversation_id: str = "") -> str:
        """Read the current page's text, as markdown, or the text of one element.

        Use it to find out what a page says; use browser_snapshot to find out what
        you can click.

        Args:
            url: The page's url, as the latest browser_snapshot shows it. Needed
                with ref, which only means something on that page.
            ref: An element ref from the latest snapshot, to read only that
                element. Empty reads the whole page.
        """
        result = await sessions.read(conversation_id=conversation_id, url=url, ref=ref)
        return result

    async def browser_scroll(direction: str, conversation_id: str = "") -> str:
        """Scroll the page, to load content that only appears when scrolled to.

        Args:
            direction: up or down.
        """
        result = await sessions.scroll(direction=direction, conversation_id=conversation_id)
        return result

    async def browser_back(conversation_id: str = "") -> str:
        """Go back to the previous page in this conversation's browser."""
        result = await sessions.back(conversation_id=conversation_id)
        return result

    async def browser_click(url: str, ref: str, name: str, conversation_id: str = "") -> str:
        """Click an element on the current page: a link, a button, a checkbox.

        All three arguments come from the latest browser_snapshot, and all three
        are checked before clicking: if the page is not at that url, or the ref
        is not an element with exactly that name, nothing is clicked.

        Args:
            url: The page's url, as the snapshot shows it.
            ref: The element's ref, e.g. e5.
            name: The element's name, exactly as the snapshot shows it.
        """
        result = await sessions.click(url=url, ref=ref, name=name, conversation_id=conversation_id)
        return result

    async def browser_type(
        url: str, ref: str, name: str, text: str, submit: bool = False, conversation_id: str = ""
    ) -> str:
        """Type text into a field on the current page, replacing what is in it.

        url, ref and name come from the latest browser_snapshot, and are checked
        before typing: if the page is not at that url, or the ref is not a field
        with exactly that name, nothing is typed.

        Args:
            url: The page's url, as the snapshot shows it.
            ref: The field's ref, e.g. e3.
            name: The field's name, exactly as the snapshot shows it.
            text: What to type.
            submit: Press Enter afterwards, to submit the form.
        """
        result = await sessions.type(
            url=url,
            ref=ref,
            name=name,
            text=text,
            submit=flag(submit),
            conversation_id=conversation_id,
        )
        return result

    async def browser_press(url: str, key: str, conversation_id: str = "") -> str:
        """Press one of these keys on the current page: Enter, Tab, Escape, ArrowDown, Control+a.

        Args:
            url: The page's url, as the latest browser_snapshot shows it. If the
                page is not at that url, nothing is pressed.
            key: The key to press, spelt exactly as listed above.
        """
        result = await sessions.press(url=url, key=key, conversation_id=conversation_id)
        return result

    gated: dict[str, Callable[..., str]] = {
        "browser_click": click_subject,
        "browser_type": type_subject,
        "browser_press": press_subject,
    }
    tools: list[Any] = []
    for func in (
        browser_open,
        browser_snapshot,
        browser_read,
        browser_scroll,
        browser_back,
        browser_click,
        browser_type,
        browser_press,
    ):
        subject = gated.get(func.__name__)
        tools.append(
            FunctionTool(
                func=func,
                approval_type=ApprovalType.USER_CONFIRM if subject else None,
                approval_subject=subject,
                approval_pattern=site_pattern if subject else None,
            )
        )
    return tools, sessions


__all__ = [
    "browser_tools",
    "claimed_origin",
    "click_subject",
    "press_subject",
    "site_pattern",
    "type_subject",
]
