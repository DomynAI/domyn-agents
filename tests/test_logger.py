"""Tests for the logger seam: printf-arg folding and per-request context.

The gateway (and other library call sites) use stdlib printf-style logging:
``logger.warning("... failed - %s", exc)``. When a structured logger is injected
via ``set_logger`` those positional args are not interpolated — and a logger
whose first parameter is a data field (a structured logger with signature
``(message, payload=None)``) silently absorbs a lone positional arg, leaving a
literal ``%s`` in the message.
These tests pin ``_safe_log`` folding so that never regresses, plus the
``bind_log_context`` contextvar seam introduced alongside it.
"""

import logging

import pytest

import domyn.agents.logger as logger_module
from domyn.agents.logger import (
    _safe_log,
    bind_log_context,
    get_log_context,
    get_logger,
    set_logger,
)


class PayloadFirstLogger:
    """Structured logger with signature ``(message, payload=None, **kwargs)``.

    A lone positional arg binds to ``payload`` rather than interpolating into
    ``message`` — the exact signature that made the old ``_safe_log`` leave a
    literal ``%s`` in the rendered line.
    """

    def __init__(self) -> None:
        self.records: list[tuple[str, str, object]] = []

    def _log(self, level: str, message, payload=None, **kwargs):
        self.records.append((level, message, payload))

    def info(self, message, payload=None, **kwargs):
        self._log("info", message, payload)

    def warning(self, message, payload=None, **kwargs):
        self._log("warning", message, payload)

    def error(self, message, payload=None, **kwargs):
        self._log("error", message, payload)

    def debug(self, message, payload=None, **kwargs):
        self._log("debug", message, payload)

    def critical(self, message, payload=None, **kwargs):
        self._log("critical", message, payload)


@pytest.fixture
def payload_logger():
    """Inject a payload-first logger for the duration of the test."""
    prev = logger_module._library_logger
    rec = PayloadFirstLogger()
    set_logger(rec)
    yield rec
    logger_module._library_logger = prev


class TestSafeLogFolding:
    """``_safe_log`` must fold printf args into the message before calling ``fn``."""

    def test_single_positional_arg_is_folded_not_absorbed(self, payload_logger):
        # Regression: the dominant gateway pattern. The old implementation only
        # folded on TypeError, so a lone arg landed in ``payload`` and the
        # message kept a literal ``%s``.
        get_logger().warning("send failed - %s", "boom")

        level, message, payload = payload_logger.records[-1]
        assert level == "warning"
        assert message == "send failed - boom"
        assert payload is None

    def test_multiple_positional_args_are_folded(self, payload_logger):
        get_logger().info("schedule %s failed - %s", "retry", "timeout")

        _, message, payload = payload_logger.records[-1]
        assert message == "schedule retry failed - timeout"
        assert payload is None

    def test_no_args_message_passes_through_unchanged(self, payload_logger):
        get_logger().debug("just a milestone")

        _, message, payload = payload_logger.records[-1]
        assert message == "just a milestone"
        assert payload is None

    def test_mismatched_placeholders_do_not_raise(self, payload_logger):
        # A message with no placeholder but a stray arg must not crash; the raw
        # message is logged rather than blowing up the call site.
        get_logger().warning("no placeholder here", "stray")

        _, message, _ = payload_logger.records[-1]
        assert message == "no placeholder here"

    def test_unsupported_kwargs_are_dropped_on_retry(self):
        # A logger accepting only ``(message)`` must still receive the folded
        # line; kwargs it cannot take are dropped on the retry path.
        seen: list[str] = []

        def strict_only_message(message):
            seen.append(message)

        _safe_log(strict_only_message, "hello %s", "world", exc_info=True)
        assert seen == ["hello world"]


class TestSafeLogStdlib:
    """The default stdlib logger keeps rendering printf args correctly."""

    def test_stdlib_interpolation(self, caplog):
        stdlib = logging.getLogger("domyn.agents.test_safe_log")
        with caplog.at_level(logging.WARNING, logger=stdlib.name):
            _safe_log(stdlib.warning, "value=%s count=%d", "x", 3)

        assert caplog.records[-1].getMessage() == "value=x count=3"


class TestBindLogContext:
    """Per-request contextvar seam: bind, merge, drop-None, and restore."""

    def test_bound_fields_are_visible_inside_scope(self):
        assert get_log_context() == {}
        with bind_log_context(conversation_id="c1", turn_id="t1"):
            assert get_log_context() == {"conversation_id": "c1", "turn_id": "t1"}
        assert get_log_context() == {}

    def test_none_values_are_dropped(self):
        with bind_log_context(conversation_id="c1", interaction_id=None):
            ctx = get_log_context()
            assert ctx == {"conversation_id": "c1"}
            assert "interaction_id" not in ctx

    def test_nested_scopes_merge_and_restore(self):
        with bind_log_context(conversation_id="c1"):
            with bind_log_context(turn_id="t2"):
                assert get_log_context() == {"conversation_id": "c1", "turn_id": "t2"}
            # Inner scope unwound; outer context restored intact.
            assert get_log_context() == {"conversation_id": "c1"}
        assert get_log_context() == {}

    def test_inner_scope_overrides_same_key(self):
        with bind_log_context(turn_id="outer"):
            with bind_log_context(turn_id="inner"):
                assert get_log_context()["turn_id"] == "inner"
            assert get_log_context()["turn_id"] == "outer"
