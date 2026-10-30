"""The model's context window, asked of whoever serves it.

First answer wins:

1. ``context_window`` declared on the provider.
2. The provider's own API, asked by :meth:`BaseLLMProvider.fetch_context_window`:
   vLLM's ``/v1/models`` (``max_model_len``), a gateway's (``max_input_tokens``),
   OpenRouter's (``top_provider.context_length``), Anthropic's ``/v1/models/{id}``
   (``max_input_tokens``).
3. For hosted APIs that report nothing — OpenAI's ``/v1/models`` has no window — the
   catalogue generated from models.dev (``context_windows.json``), by exact model id,
   for the providers that name one (``context_window_catalog``).

What none of them knows is learned from the provider's rejection
(``compaction._adopt_window``), and so is a window one of them put too high. Never from
a model name on a self-hosted endpoint: the same weights served with
``--max-model-len 32768`` or ``131072`` have different windows, which is why vLLM and
custom endpoints have no catalogue.

Answers, including "no answer", are kept for :data:`CACHE_SECONDS` per provider class,
endpoint and model, so a host that builds its providers for every request asks once per
model, not once per turn. A question that failed (refused, timed out, unreachable) says
nothing about the window, and is asked again after :data:`RETRY_SECONDS`. Only by a
provider that still has no window, though: the answer is written on the provider as a
declaration would be, so one that keeps its provider (the CLI, for a whole session)
asks once, and keeps whatever it got, the CLI's fallback included, until it builds
another. A window that later shrinks is still learned from the rejection. A
window stated in a rejection is kept under the same key until the process exits
(:func:`remember_stated_window`) and caps whatever the steps above say: a provider built
again for the next request would otherwise have forgotten it, and the next long request
would be refused again. It does not expire like an answer because it cannot be checked
like one: an answer is asked again with one GET, a stated window only by another
refused request, after which the conversation is compacted toward half its size instead
of at the threshold. Nothing is written to disk.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from domyn.agents.logger import get_logger

if TYPE_CHECKING:
    from domyn.agents.llm.base import BaseLLMProvider

CACHE_SECONDS = 300.0
"""How long an answer is trusted. Five minutes, LiteLLM's own refresh period for the
limits it reads upstream: a gateway that learns a window is believed within one more."""

RETRY_SECONDS = 60.0
"""How long a failed question waits before it is asked again, by a provider built after
it: one that already holds a window does not ask. A failure says nothing about the
window: kept as long as an answer, one dropped request would leave the window unknown,
and nothing compacted ahead of time, for five minutes."""

REQUEST_TIMEOUT_SECONDS = 5.0

_CATALOGUE = Path(__file__).with_name("context_windows.json")
# Dated snapshots (`gpt-4o-2024-08-06`, `claude-3-5-sonnet-20241022`) share the window of
# the alias models.dev lists them under.
_DATE_SUFFIX = re.compile(r"-\d{4}-?\d{2}-?\d{2}$")

_answers: dict[tuple[str, str, str], tuple[float, int | None]] = {}

_stated: dict[tuple[str, str, str], int] = {}
"""The window each provider class, endpoint and model stated when it refused a request
as too long, kept until the process exits.

Only ever lowered: a smaller statement replaces a larger one, and it caps a window from
any other source without ever raising one. A server later given a larger window keeps
the old one here until this process restarts: compaction starts earlier than it needs
to, but no request is refused."""


def positive_int(value: Any) -> int | None:
    """``value`` when it is a usable window. ``bool`` is an ``int`` and is not one."""
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def models_url(endpoint: str | None) -> str | None:
    """The ``/models`` listing beside an OpenAI-style endpoint, or ``None`` when the URL
    has another shape — never a guess."""
    if not endpoint:
        return None
    endpoint = endpoint.rstrip("/")
    for suffix in ("/chat/completions", "/responses", "/completions"):
        if endpoint.endswith(suffix):
            return endpoint[: -len(suffix)] + "/models"
    return None


def window_from_model_entry(entry: Mapping[str, Any]) -> int | None:
    """The window one ``/models`` entry states, in the spellings servers use: LiteLLM's
    ``max_input_tokens``, vLLM's ``max_model_len`` (the whole window), OpenRouter's
    ``top_provider.context_length`` (where a request is routed by default) and its
    ``context_length`` (the largest among its providers).

    The prompt limit comes first because compaction measures the prompt: an entry
    stating both describes a window that includes the output, which would size the
    threshold too large."""
    top_provider = entry.get("top_provider")
    candidates = (
        entry.get("max_input_tokens"),
        entry.get("max_model_len"),
        top_provider.get("context_length") if isinstance(top_provider, Mapping) else None,
        entry.get("context_length"),
    )
    return next((window for value in candidates if (window := positive_int(value))), None)


async def get_json(url: str, headers: Mapping[str, str]) -> Any:
    """One GET, no retries: this runs before a turn, and a slow answer is no answer."""
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        response = await client.get(url, headers=dict(headers))
    response.raise_for_status()
    return response.json()


async def window_from_models_listing(
    url: str, headers: Mapping[str, str], model: str
) -> int | None:
    """The window ``url`` (an OpenAI-style ``/models``) states for ``model``."""
    listing = await get_json(url, headers)
    entries = listing.get("data") if isinstance(listing, Mapping) else None
    for entry in entries or ():
        if isinstance(entry, Mapping) and entry.get("id") == model:
            return window_from_model_entry(entry)
    return None


@cache
def _catalogue() -> Mapping[str, Mapping[str, int]]:
    windows: Mapping[str, Mapping[str, int]] = json.loads(_CATALOGUE.read_text())["windows"]
    return windows


def catalogue_window(catalogue: str, model: str) -> int | None:
    """The window models.dev lists for ``model`` under ``catalogue``, by exact id."""
    models = _catalogue().get(catalogue, {})
    return models.get(model) or models.get(_DATE_SUFFIX.sub("", model))


async def resolve_context_window(provider: BaseLLMProvider) -> int | None:
    """Give ``provider`` its window and return it: the one it declares, else the one it
    is asked for, capped by a window a rejection stated for the same endpoint and model.

    Never raises: a provider that cannot be asked — or is not what it claims, like a
    test double — keeps the window it had. Sets ``context_window`` exactly as a
    declaration would, so a rejection stating a smaller window still lowers it.
    """
    try:
        return await _resolve(provider)
    except Exception as exc:
        get_logger().debug(f"context window not resolved: {type(exc).__name__}: {exc}")
        return None


def remember_stated_window(provider: Any, window: int) -> None:
    """Keep the window ``provider`` stated in a rejection for every provider built later
    for the same endpoint and model. Only ever lowered, like the window it adopts."""
    key = deployment_key(provider)
    if key is None or positive_int(window) is None:
        return
    known = _stated.get(key)
    if known is None or window < known:
        _stated[key] = window


def deployment_key(provider: Any) -> tuple[str, str, str] | None:
    """The deployment ``provider`` talks to — its class, endpoint and model — or ``None``
    for one that names no model. What a window, and a token count, belong to."""
    model = getattr(provider, "model_name", None)
    if not isinstance(model, str) or not model:
        return None
    return (type(provider).__name__, str(getattr(provider, "url", "") or ""), model)


async def _resolve(provider: BaseLLMProvider) -> int | None:
    key = deployment_key(provider)
    window = provider.context_window
    if window is None and key is not None:
        window = await _asked(provider, key)
    stated = _stated.get(key) if key is not None else None
    if stated is not None and (window is None or stated < window):
        window = stated
    if window is not None and window != provider.context_window:
        provider.context_window = window
    return window


async def _asked(provider: BaseLLMProvider, key: tuple[str, str, str]) -> int | None:
    now = time.monotonic()
    known = _answers.get(key)
    if known is not None and known[0] > now:
        return known[1]
    window, source, trusted_for = await _ask(provider)
    _answers[key] = (now + trusted_for, window)
    if window is not None and (known is None or known[1] != window):
        get_logger().info(
            f"context window of {provider.model_name}: {window} tokens, from {source}"
        )
    return window


async def _ask(provider: BaseLLMProvider) -> tuple[int | None, str, float]:
    """The window, where it came from, and for how long the answer holds."""
    trusted_for = CACHE_SECONDS
    try:
        window = positive_int(await provider.fetch_context_window())
    except Exception as exc:
        # A gateway that refuses the route, a server that is down, a reply in another
        # shape: all mean "not reported", and none of them may fail the turn.
        get_logger().debug(
            f"context window of {provider.model_name} not reported: {type(exc).__name__}: {exc}"
        )
        window, trusted_for = None, RETRY_SECONDS
    if window is not None:
        return window, "the provider", trusted_for
    catalogue = provider.context_window_catalog
    if catalogue and (window := catalogue_window(catalogue, provider.model_name)):
        return window, "the models.dev catalogue", trusted_for
    return None, "", trusted_for
