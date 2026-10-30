"""Unit tests for the shared VisibilityScope trust ladder.

VisibilityScope owns the single run_config (PII) gate used by both MCP tools and
delegated agents: ISOLATED/SCOPED grant nothing; TRUSTED/GLOBAL grant the config.
"""

import pytest

from domyn.agents.core import VisibilityScope

RUN_CONFIG = {"user_id": "u-1", "space_id": "s-9", "token": "secret"}


@pytest.mark.parametrize(
    ("scope", "grants"),
    [
        (VisibilityScope.ISOLATED, False),
        (VisibilityScope.SCOPED, False),
        (VisibilityScope.TRUSTED, True),
        (VisibilityScope.GLOBAL, True),
    ],
)
def test_grants_run_config(scope: VisibilityScope, grants: bool) -> None:
    assert scope.grants_run_config is grants


@pytest.mark.parametrize("scope", [VisibilityScope.ISOLATED, VisibilityScope.SCOPED])
def test_filter_run_config_denied_scopes_return_empty(scope: VisibilityScope) -> None:
    assert scope.filter_run_config(RUN_CONFIG) == {}


@pytest.mark.parametrize("scope", [VisibilityScope.TRUSTED, VisibilityScope.GLOBAL])
def test_filter_run_config_granted_scopes_return_copy(scope: VisibilityScope) -> None:
    out = scope.filter_run_config(RUN_CONFIG)
    assert out == RUN_CONFIG
    # A copy, not the same object — mutating the result must not touch the source.
    assert out is not RUN_CONFIG
    out["injected"] = True
    assert "injected" not in RUN_CONFIG


def test_filter_run_config_handles_none() -> None:
    assert VisibilityScope.TRUSTED.filter_run_config(None) == {}
    assert VisibilityScope.ISOLATED.filter_run_config(None) == {}
