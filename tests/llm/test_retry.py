"""Which failures are worth asking again, and how long the wait is.

The rule the planner follows is ``retry_delay``: it returns a wait for a failure a retry could fix,
and ``None`` for everything else — a bad key, a refused prompt, or a retry budget already spent.
"""

import email.utils
from datetime import UTC, datetime, timedelta, timezone

import httpx
import pytest

from domyn.agents.llm.retry import (
    MAX_WAIT_S,
    backoff_delay,
    is_transient,
    retry_after_seconds,
    retry_delay,
    status_of,
)


def _status_error(code: int, headers: dict[str, str] | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx.Response(code, request=request, headers=headers or {})
    return httpx.HTTPStatusError(f"{code}", request=request, response=response)


class TestWhatIsTransient:
    @pytest.mark.parametrize("code", [408, 425, 429, 500, 502, 503, 504])
    def test_overload_and_rate_limits_are(self, code: int) -> None:
        assert is_transient(_status_error(code))

    @pytest.mark.parametrize("code", [400, 401, 403, 404, 413, 422])
    def test_the_requests_own_faults_are_not(self, code: int) -> None:
        """Retrying a refused request just spends the turn's time twice."""
        assert not is_transient(_status_error(code))

    @pytest.mark.parametrize(
        "error",
        [
            httpx.ConnectError("no route"),
            httpx.ConnectTimeout("slow"),
            httpx.ReadTimeout("slow"),
            httpx.ReadError("reset"),
            httpx.RemoteProtocolError(
                "peer closed connection without sending complete message body"
            ),
        ],
    )
    def test_a_broken_transport_is(self, error: Exception) -> None:
        """The case measured on our gateway: a stream the peer closes half-way."""
        assert is_transient(error)

    def test_an_unrelated_exception_is_not(self) -> None:
        assert not is_transient(ValueError("bad argument"))
        assert status_of(ValueError("x")) is None


class TestHowLongToWait:
    def test_the_provider_may_say_how_long(self) -> None:
        assert retry_after_seconds(_status_error(429, {"Retry-After": "7"})) == 7.0

    def test_an_http_date_is_read_as_a_wait(self) -> None:
        when = datetime.now(UTC) + timedelta(seconds=30)
        header = email.utils.format_datetime(when)
        asked = retry_after_seconds(_status_error(503, {"Retry-After": header}))

        assert asked is not None and 25 <= asked <= 31

    def test_a_date_in_the_past_is_no_wait(self) -> None:
        header = email.utils.format_datetime(datetime.now(UTC) - timedelta(seconds=60))

        assert retry_after_seconds(_status_error(503, {"Retry-After": header})) == 0.0

    def test_nonsense_is_ignored(self) -> None:
        assert retry_after_seconds(_status_error(503, {"Retry-After": "soon"})) is None
        assert retry_after_seconds(httpx.ConnectError("x")) is None

    def test_backoff_doubles_and_stops_at_the_cap(self) -> None:
        waits = [backoff_delay(n) for n in range(1, 9)]

        assert all(w >= 1.0 for w in waits)
        assert waits[0] < waits[3], waits
        assert max(waits) <= MAX_WAIT_S * 1.25, waits

    def test_the_wait_is_capped_even_when_the_provider_asks_for_more(self) -> None:
        """A five-minute pause inside a turn is worse than failing it."""
        delay = retry_delay(_status_error(429, {"Retry-After": "300"}), used=0, limit=3)

        assert delay == MAX_WAIT_S


class TestWhenNotToRetry:
    def test_a_refusal_has_no_delay(self) -> None:
        assert retry_delay(_status_error(400), used=0, limit=3) is None

    def test_a_spent_budget_has_none(self) -> None:
        assert retry_delay(_status_error(429), used=3, limit=3) is None

    def test_a_transient_failure_with_budget_gets_one(self) -> None:
        assert retry_delay(httpx.ConnectError("x"), used=0, limit=3) is not None
