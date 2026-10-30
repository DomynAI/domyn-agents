import os
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from typing import Any

from opentelemetry.trace import Tracer

# -------------------------------
# Whether OpenTelemetry is installed
# -------------------------------
try:
    from opentelemetry import trace
    from opentelemetry.propagate import inject

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

# -------------------------------
# Flag enable tracing
# -------------------------------
ENABLE_TRACING = os.getenv("ENABLE_TRACING", "true").lower() in ("true", "1", "yes")


# -------------------------------
# Dummy tracer
# -------------------------------
class DummySpan:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def set_attribute(self, key, value):
        pass

    def record_exception(self, exc):
        pass

    def set_status(self, status):
        pass


class DummyTracer:
    def start_as_current_span(self, name):
        return DummySpan()


def get_safe_tracer(name: str) -> Tracer | DummyTracer:
    if OPENTELEMETRY_AVAILABLE and ENABLE_TRACING:
        return trace.get_tracer(name)
    else:
        return DummyTracer()


# -------------------------------
# Decorator for optional instrumentation
# -------------------------------
def instrumented(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracer = get_safe_tracer(func.__module__)
        span = None
        if OPENTELEMETRY_AVAILABLE and ENABLE_TRACING:
            span = trace.get_current_span()
            # No active span: run untraced rather than start a new root span.
            if span is None or not span.get_span_context().is_valid:
                return func(*args, **kwargs)

        with tracer.start_as_current_span(func.__name__):
            return func(*args, **kwargs)

    return wrapper


def inject_trace_headers(headers: dict[str, Any]) -> dict[str, Any]:
    if OPENTELEMETRY_AVAILABLE and ENABLE_TRACING:
        inject(headers)
    return headers


def add_trace_headers(headers: dict[str, Any]) -> dict[str, Any]:
    _h = deepcopy(headers)
    if OPENTELEMETRY_AVAILABLE and ENABLE_TRACING:
        inject(_h)
    return _h


def extract_context(carrier: dict[str, Any] | None) -> Any | None:
    """Extract a W3C trace context (``traceparent``/``tracestate``) from *carrier*.

    Returns an OpenTelemetry ``Context`` to use as a span parent, or ``None``
    when tracing is disabled/unavailable or the carrier holds no trace context.
    """
    if not (OPENTELEMETRY_AVAILABLE and ENABLE_TRACING) or not carrier:
        return None
    from opentelemetry.propagate import extract

    return extract(carrier)


@contextmanager
def request_span(name: str, carrier: dict[str, Any] | None = None) -> Iterator[None]:
    """Run the block inside a span parented on the trace context in *carrier*.

    Starting the span as current means any trace-aware logger emits records
    carrying the same ``trace`` id as the caller that sent *carrier* (e.g. an
    upstream gateway), joining this process into the distributed trace. Re-derives the
    parent from *carrier* on every call so it works across event loops / tasks
    (outbound emits are scheduled cross-loop and cannot inherit an ambient
    context). No-op passthrough when OpenTelemetry is unavailable/disabled.
    """
    if not (OPENTELEMETRY_AVAILABLE and ENABLE_TRACING):
        yield
        return
    from opentelemetry import context as otel_context

    parent = extract_context(carrier)
    token = otel_context.attach(parent) if parent is not None else None
    try:
        with trace.get_tracer("domyn.agents.hermes_gateway").start_as_current_span(name):
            yield
    finally:
        if token is not None:
            otel_context.detach(token)
