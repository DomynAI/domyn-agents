import contextlib
import contextvars
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol, runtime_checkable

logging.getLogger("domyn.agents").addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Per-request log context
#
# A dependency-free seam (stdlib contextvars only) that lets the library bind
# per-request fields (conversation_id, interaction_id, ...) to the current
# execution context. An injected structured logger (see ``set_logger``) can read
# ``get_log_context()`` and merge it into every emitted record. When the default
# stdlib logger is in use these fields are simply ignored, so the core library
# keeps working without any structured-logging dependency.
# ---------------------------------------------------------------------------

_log_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "domyn_log_context", default=None
)


def get_log_context() -> dict[str, Any]:
    """Return the fields bound to the current execution context (may be empty)."""
    return _log_context.get() or {}


@contextmanager
def bind_log_context(**fields: Any) -> Iterator[None]:
    """Bind *fields* to the current context for the duration of the block.

    ``None`` values are dropped so callers can pass optional IDs unconditionally.
    Restores the previous context on exit (contextvar token reset), so nested
    scopes and cross-loop re-binding compose correctly.
    """
    merged = {**get_log_context(), **{k: v for k, v in fields.items() if v is not None}}
    token = _log_context.set(merged)
    try:
        yield
    finally:
        _log_context.reset(token)


@runtime_checkable
class LoggerProtocol(Protocol):
    """Minimal interface compatible with logging.Logger, loguru, structlog, etc."""

    def info(self, message: Any, *args: Any, **kwargs: Any) -> None: ...
    def warning(self, message: Any, *args: Any, **kwargs: Any) -> None: ...
    def error(self, message: Any, *args: Any, **kwargs: Any) -> None: ...
    def debug(self, message: Any, *args: Any, **kwargs: Any) -> None: ...
    def critical(self, message: Any, *args: Any, **kwargs: Any) -> None: ...


def _safe_log(fn: Any, message: Any, *args: Any, **kwargs: Any) -> None:
    """Call the log method, degrading gracefully for loggers with a narrower API.

    Positional ``*args`` follow the stdlib printf contract (``msg % args``). An
    injected structured logger (e.g. loguru, structlog) does not
    interpolate them — worse, one whose first parameter is a data field (e.g. a
    logger with signature ``(message, payload=None, **kwargs)``) silently absorbs a lone
    positional arg without raising, leaving a literal ``%s`` in the message. So
    we fold the args into the message *up front* rather than relying on a
    ``TypeError`` that never comes — this keeps existing
    ``logger.warning("... %s ...", x)`` call sites rendering correctly under any
    injected logger. ``**kwargs`` unsupported by the target are dropped on retry.
    """
    if args:
        # Mismatched/absent placeholders: fall back to the raw message.
        with contextlib.suppress(Exception):
            message = message % args
    try:
        fn(message, **kwargs)
    except TypeError:
        fn(message)


class SafeLoggerWrapper:
    """Wraps any logger injected via set_logger(), dropping unsupported kwargs."""

    def __init__(self, logger: LoggerProtocol) -> None:
        self._logger = logger

    def info(self, message: Any, *args: Any, **kwargs: Any) -> None:
        _safe_log(self._logger.info, message, *args, **kwargs)

    def warning(self, message: Any, *args: Any, **kwargs: Any) -> None:
        _safe_log(self._logger.warning, message, *args, **kwargs)

    def error(self, message: Any, *args: Any, **kwargs: Any) -> None:
        _safe_log(self._logger.error, message, *args, **kwargs)

    def debug(self, message: Any, *args: Any, **kwargs: Any) -> None:
        _safe_log(self._logger.debug, message, *args, **kwargs)

    def critical(self, message: Any, *args: Any, **kwargs: Any) -> None:
        _safe_log(self._logger.critical, message, *args, **kwargs)


# The library's default logger
_library_logger: LoggerProtocol = SafeLoggerWrapper(logging.getLogger("domyn.agents"))


def set_logger(logger: LoggerProtocol) -> None:
    """Inject globally a logger instance to be used across the library. The logger should be compatible with LoggerProtocol."""
    global _library_logger
    _library_logger = SafeLoggerWrapper(logger)


def get_logger() -> LoggerProtocol:
    return _library_logger
