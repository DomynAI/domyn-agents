"""Declaring a decision model in a configuration file.

Two halves, mirroring how ``tool_bundles`` works: a named provider at the top
level, and a per-agent block that points at one and says where it is allowed to
matter.

.. code-block:: yaml

    decision_providers:
      - name: local_clef
        type: system_one
        base_url: http://localhost:28080
        model: Cloudflare/clef          # the name the SERVER exposes
        max_state_tokens: 73728

    agents:
      - name: Reviewer
        decisions:
          provider: local_clef
          approvals: {mode: active, min_confidence: 0.85}

What raises and what degrades follows the house rule, and the split is
deliberate: a **configuration** mistake raises at load time (an unknown type, an
unknown use case, a provider name nothing declares), because a silently
disabled feature someone believes they configured is worse than a failed start.
A **runtime** failure degrades (the endpoint is down, the model is slow), because
by then there is a turn in flight and the user's work is worth more than our
feature.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from domyn.agents.decisions.runtime import (
    USE_CASES,
    DecisionMode,
    DecisionRuntime,
    UseCasePolicy,
)


class UseCaseConfig(BaseModel):
    """Where one use case stands: off, measuring, or acting."""

    model_config = ConfigDict(extra="forbid")

    mode: DecisionMode = DecisionMode.OFF
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    """The bar an answer must clear. ``None`` takes the use case's default.

    In the config rather than the code because the measurement shows the
    threshold *is* the safety mechanism, and one buried in the source is
    neither inspectable nor tunable per installation."""

    on_unknown: Literal["block", "allow"] = "block"
    """What happens when the model cannot answer. Only ``sensitivity`` reads it:
    every other use case treats an unanswerable question as "behave exactly as
    today", which is safe by construction. The data boundary cannot do that —
    today's behaviour there is to send the data — so the choice is explicit,
    defaults to the inconvenient option, and sits in the file where a reviewer
    can see whose data it was chosen for."""


class DecisionProviderConfig(BaseModel):
    """A named decision backend."""

    model_config = ConfigDict(extra="allow")

    name: str
    type: str = "none"

    def build(self) -> Any:
        from domyn.agents.decisions import build_provider

        settings = self.model_dump(exclude_none=True)
        settings.pop("name", None)
        return build_provider(settings)


class DecisionsConfig(BaseModel):
    """What one agent declares under ``decisions:``."""

    model_config = ConfigDict(extra="forbid")

    provider: str | None = None
    """The name of an entry in the top-level ``decision_providers:``. Omitted,
    the agent gets no decision model at all — which is the default everywhere."""

    use_cases: dict[str, UseCaseConfig] = Field(default_factory=dict)

    @field_validator("use_cases")
    @classmethod
    def _known_use_cases(cls, value: dict[str, UseCaseConfig]) -> dict[str, UseCaseConfig]:
        unknown = sorted(set(value) - set(USE_CASES))
        if unknown:
            raise ValueError(
                f"Unknown decision use case(s): {', '.join(unknown)}. "
                f"Known: {', '.join(sorted(USE_CASES))}"
            )
        return value

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DecisionsConfig:
        """Read the flat YAML form, where each use case is its own key.

        ``{provider: x, approvals: {...}}`` rather than
        ``{provider: x, use_cases: {approvals: {...}}}``: the nesting buys
        nothing and every other block in this file is flat.
        """
        settings = dict(raw)
        provider = settings.pop("provider", None)
        return cls(
            provider=provider,
            use_cases={name: UseCaseConfig(**(value or {})) for name, value in settings.items()},
        )

    def policies(self) -> dict[str, UseCasePolicy]:
        return {
            name: UseCasePolicy(
                mode=use_case.mode,
                threshold=(
                    use_case.min_confidence
                    if use_case.min_confidence is not None
                    else USE_CASES[name].default_threshold
                ),
                on_unknown=use_case.on_unknown,
            )
            for name, use_case in self.use_cases.items()
        }

    def build(self, providers: dict[str, Any]) -> DecisionRuntime | None:
        """The runtime this config describes, or ``None`` for "behave as today".

        A ``provider:`` naming something nothing declares raises: it is the same
        class of mistake as an unknown tool bundle, which this loader already
        refuses.
        """
        if self.provider is None:
            return None
        if self.provider not in providers:
            raise ValueError(
                f"Decision provider '{self.provider}' not found in configuration. "
                f"Declared: {', '.join(sorted(providers)) or 'none'}"
            )
        return DecisionRuntime(providers[self.provider], self.policies())
