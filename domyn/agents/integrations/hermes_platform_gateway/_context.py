"""Per-request logging/trace scope for the platform gateway.

Composes two dependency-free seams so an injected structured logger (see
``domyn.agents.logger.set_logger``) can emit request-correlated, trace-joined
records:

* ``domyn.agents.logger.bind_log_context`` — binds the turn's correlation IDs
  (conversation_id, interaction_id, ...) to the current context so the injected
  logger can merge them into every record.
* ``domyn.agents.tracing.request_span`` — re-parents a span on the platform's
  W3C trace context (carried in the relay envelope ``meta.traceparent``) so
  records carry the same ``trace`` id as the Domyn platform and relay.

Tracing is optional: ``domyn.agents.tracing`` hard-imports OpenTelemetry, so we
import it lazily and degrade to an ID-only scope when the ``tracing`` extra is
not installed. This keeps the gateway importable in a plain (PyPI-only) install.
"""

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

from domyn.agents.logger import bind_log_context

try:  # OpenTelemetry is an optional extra; degrade gracefully without it.
    from domyn.agents.tracing import request_span

    _TRACING_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without the tracing extra
    _TRACING_AVAILABLE = False


@contextmanager
def request_log_scope(carrier: dict[str, Any] | None = None, **ids: Any) -> Iterator[None]:
    """Scope a block with the turn's correlation IDs and distributed trace.

    *carrier* is the relay envelope ``meta`` (holding ``traceparent``); *ids* are
    the correlation fields to bind. Re-derives the span parent from *carrier* on
    every call, so it is safe to (re)enter from cross-loop outbound coroutines.
    """
    span_cm = request_span("hermes.gateway", carrier) if _TRACING_AVAILABLE else nullcontext()
    with span_cm, bind_log_context(**ids):
        yield
