"""Which LLM failures are worth asking again, and how long to wait before it.

A turn is one long conversation: the planner sends the whole history on every call, and a single
transient failure — a 429 from a shared gateway, a 502 while an upstream restarts, a stream the peer
closes half-way — used to end it with ``ERROR_OCCURRED``. The work done so far survived, but the turn
did not, and in an unattended run nobody asks again.

The retry repeats the *request*, not the turn: the failed attempt is thrown away and the same
messages go out again. Nothing of it reaches the conversation — streamed tokens are partial events,
which the runner does not store — so a retry costs latency, never correctness.
"""

from __future__ import annotations

import email.utils
import random
from datetime import UTC, datetime

import httpx

# Statuses a retry can fix: the provider is busy, out of capacity, or behind a gateway with no
# upstream. A 4xx that is not a rate limit is the request's own fault — wrong key, wrong model, a
# prompt the provider refused — and would fail again exactly the same way.
TRANSIENT_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

# The transport's failures: no connection at all, one lost mid-stream (``RemoteProtocolError``), a
# read that timed out. Every one of them was seen in production, and every one is retried by the
# harnesses this one is compared with.
TRANSIENT_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.WriteError,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
)

# The longest a turn will wait for one retry. A provider that asks for more (``Retry-After: 300``)
# is right about itself and wrong for an interactive turn: the caller would rather fail, or retry
# later, than sit for five minutes. The wait actually taken is logged.
MAX_WAIT_S = 120.0


def status_of(error: BaseException) -> int | None:
    """The HTTP status behind an error, when it has one."""
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    detail = getattr(error, "status_code", None)
    return detail if isinstance(detail, int) else None


def is_transient(error: BaseException) -> bool:
    """Whether asking the same request again could plausibly succeed."""
    status = status_of(error)
    if status is not None:
        return status in TRANSIENT_STATUS
    return isinstance(error, TRANSIENT_ERRORS)


def retry_after_seconds(error: BaseException) -> float | None:
    """The provider's own wait, from ``Retry-After``: seconds, or an HTTP date.

    Read from the error's response when it has one. Anything unparseable is ignored, which leaves
    the caller's own backoff in charge.
    """
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if not raw:
        return None
    raw = str(raw).strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


def backoff_delay(attempt: int, *, base: float = 1.0, max_wait: float = MAX_WAIT_S) -> float:
    """Seconds to wait before retry number ``attempt`` (1 for the first).

    Doubles per attempt up to ``max_wait``, with jitter added rather than subtracted: several
    agents failing together should not retry together, and waiting a little longer is harmless.
    """
    if attempt < 1:
        attempt = 1
    delay = min(max_wait, base * (2 ** (attempt - 1)))
    return min(max_wait, delay * (1.0 + random.uniform(0.0, 0.25)))


def retry_delay(error: BaseException, used: int, limit: int) -> float | None:
    """How long to wait before retrying ``error``, or ``None`` to let it through.

    ``used`` counts the retries already spent on this call and ``limit`` is the planner's
    ``max_llm_retries``; at the limit the error is the caller's to handle. A provider's own
    ``Retry-After`` wins over the exponential backoff, still within ``MAX_WAIT_S``.
    """
    if used >= limit or not is_transient(error):
        return None
    asked = retry_after_seconds(error)
    if asked is not None:
        return min(asked, MAX_WAIT_S)
    return backoff_delay(used + 1)
