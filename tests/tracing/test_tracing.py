from copy import deepcopy
from unittest.mock import patch

import domyn.agents.tracing as tracing_module
from domyn.agents.tracing import add_trace_headers, inject_trace_headers

# ---------------------------------------------------------------------------
# inject_trace_headers
# ---------------------------------------------------------------------------


class TestInjectTraceHeaders:
    def test_injects_headers_when_tracing_enabled(self):
        """inject_trace_headers calls opentelemetry inject when tracing is enabled."""

        def fake_inject(carrier):
            carrier["traceparent"] = "00-abc123-def456-01"

        with (
            patch.object(tracing_module, "OPENTELEMETRY_AVAILABLE", True),
            patch.object(tracing_module, "ENABLE_TRACING", True),
            patch.object(tracing_module, "inject", fake_inject),
        ):
            headers = {"content-type": "application/json"}
            inject_trace_headers(headers)

        assert "traceparent" in headers

    def test_does_not_inject_headers_when_tracing_disabled(self, monkeypatch):
        """inject_trace_headers leaves the headers alone when ENABLE_TRACING is False."""
        monkeypatch.setenv("ENABLE_TRACING", "false")

        with (
            patch.object(tracing_module, "OPENTELEMETRY_AVAILABLE", True),
            patch.object(tracing_module, "ENABLE_TRACING", False),
        ):
            headers = {"content-type": "application/json"}
            original = deepcopy(headers)
            inject_trace_headers(headers)

        assert headers == original


# ---------------------------------------------------------------------------
# add_trace_headers
# ---------------------------------------------------------------------------


class TestAddTraceHeaders:
    def test_returns_new_dict_with_injected_headers_when_tracing_enabled(self):
        """add_trace_headers returns a copy with the trace headers added."""

        def fake_inject(carrier):
            carrier["traceparent"] = "00-abc123-def456-01"

        with (
            patch.object(tracing_module, "OPENTELEMETRY_AVAILABLE", True),
            patch.object(tracing_module, "ENABLE_TRACING", True),
            patch.object(tracing_module, "inject", fake_inject),
        ):
            original = {"content-type": "application/json"}
            result = add_trace_headers(original)

        assert "traceparent" in result
        # the original is left unchanged
        assert "traceparent" not in original

    def test_returns_copy_without_trace_headers_when_tracing_disabled(self, monkeypatch):
        """add_trace_headers returns an unchanged copy when ENABLE_TRACING is False."""
        monkeypatch.setenv("ENABLE_TRACING", "false")

        with (
            patch.object(tracing_module, "OPENTELEMETRY_AVAILABLE", True),
            patch.object(tracing_module, "ENABLE_TRACING", False),
        ):
            original = {"content-type": "application/json"}
            result = add_trace_headers(original)

        assert result == original
        assert result is not original  # still a copy
