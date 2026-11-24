"""Captured log records split at debug: a host embedding the library logs at info, so what is
above debug is what reaches its log store."""

import logging

import pytest


def split_at_debug(caplog: pytest.LogCaptureFixture) -> tuple[str, str]:
    """The records above debug and those at debug, each as one text, tracebacks included."""
    formatter = logging.Formatter()
    above: list[str] = []
    debug: list[str] = []
    for record in caplog.records:
        (above if record.levelno > logging.DEBUG else debug).append(formatter.format(record))
    return "\n".join(above), "\n".join(debug)
