"""Estimating what a request costs in context.

It exists for one reason: to know how close we are to the model's limit, because
today we have no idea. Without that number any context policy is blind — and a
long session does not degrade, it breaks on a rejected request.

**Why an estimate and not the real tokenizer.** The right tokenizer is the one
belonging to the model we serve, which on vLLM changes with the deployment and
cannot be queried from here. ``tiktoken`` is OpenAI's tokenizer: it would apply
the wrong segmentation and add a dependency for a number we only use as a
threshold. All three harnesses we compared against (Claude Code, Hermes-Agent,
pi) use heuristics.

**The error is not symmetric, and that decides the defaults.** Overestimating
costs an early compaction; underestimating costs a rejected request, i.e. a lost
turn. So rounding is always up, and ``chars_per_token`` is deliberately low —
see the field for the reasoning behind the value.

**No calibration against the provider's count.** The count arrives one request
late, a ratio learned from it lags whenever the content changes, and the reserve
above the threshold leaves little room for an error in the unsafe direction. The
heuristic overestimates code and prose, which is the prudent side. Providers
still return their count in ``LLMResponse.usage``, for logs and telemetry.
"""

import json
import math
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from domyn.agents.llm import LLMMessage

CONTEXT_USAGE_METADATA_KEY = "context_usage"
"""Key the measurement travels under in ``LLM_START`` metadata.

No ``_`` prefix: this is not a secret and must not be stripped on the way out —
it is telemetry a front-end has to be able to read.
"""


@runtime_checkable
class TokenEstimator(Protocol):
    """Anything that can price a piece of text or a list of messages."""

    def estimate_text(self, text: str) -> int: ...

    def estimate_messages(self, messages: list[LLMMessage]) -> int: ...

    def estimate_tool_definitions(self, tool_defs: list[dict[str, Any]]) -> int: ...


class HeuristicEstimator(BaseModel):
    """Character-based estimate, with an allowance for protocol envelopes.

    It is not accurate and does not try to be: it only has to avoid
    underestimating.
    """

    chars_per_token: float = Field(
        default=3.0,
        gt=0,
        description=(
            "Characters per token. The canonical 4.0 comes from English; Italian "
            "sits nearer 3.5 and code nearer 3.0. We keep 3.0 because the token "
            "reserve does not cover estimation error on large windows: on 200K a "
            "15% error is ~30K tokens, more than the reserve itself. The margin "
            "is reclaimed by tuning this against a specific model's real "
            "tokenizer."
        ),
    )
    per_message_overhead: int = Field(
        default=4,
        ge=0,
        description="Role delimiters every chat template adds per message.",
    )
    per_tool_call_overhead: int = Field(
        default=16,
        ge=0,
        description="A tool call's envelope beyond its JSON (id, type, name).",
    )
    image_tokens: int = Field(
        default=1_600,
        ge=0,
        description=(
            "Flat cost of one image. It varies a lot by model and resolution; "
            "deliberately high, for the same asymmetry."
        ),
    )

    def estimate_text(self, text: str) -> int:
        if not text:
            return 0
        return math.ceil(len(text) / self.chars_per_token)

    def estimate_messages(self, messages: list[LLMMessage]) -> int:
        return sum(self._estimate_message(message) for message in messages)

    def estimate_tool_definitions(self, tool_defs: list[dict[str, Any]]) -> int:
        """Tool definitions, which travel outside the messages.

        They cost as much as the conversation and no compaction can touch them:
        they are a request parameter, not a message. Measuring them separately is
        what makes that cost visible.
        """
        if not tool_defs:
            return 0
        return self.estimate_text(_stable_json(tool_defs))

    # --- internals --------------------------------------------------------

    def _estimate_message(self, message: LLMMessage) -> int:
        total = self.per_message_overhead
        total += self.estimate_text(message.name or "")
        total += self._estimate_content(message.content)

        for call in message.tool_calls or []:
            total += self.per_tool_call_overhead
            total += self.estimate_text(_stable_json(call.model_dump()))

        return total

    def _estimate_content(self, content: str | list[dict[str, Any]] | None) -> int:
        if content is None:
            return 0
        if isinstance(content, str):
            return self.estimate_text(content)

        # Multimodal content: non-text parts are charged a flat rate, because
        # their size in characters says nothing about their cost. No defensive
        # type branch: pydantic validates `content` at construction, so a part
        # here is always a dict.
        total = 0
        for part in content:
            if part.get("type") in ("image", "image_url"):
                total += self.image_tokens
            else:
                total += self.estimate_text(_stable_json(part))
        return total


class ContextUsage(BaseModel):
    """What a request cost, broken into its parts.

    Tool definitions get their own line rather than being folded into the
    conversation, and that is not pedantry: they travel in a different request
    parameter, they are paid on every turn from the first, and **no compaction
    can touch them**. Summing them into the conversation would hide the one cost
    that cannot be compressed.
    """

    message_tokens: int = Field(ge=0)
    tool_definition_tokens: int = Field(default=0, ge=0)
    context_window: int | None = Field(
        default=None, description="`None` when the provider does not declare it."
    )
    threshold_tokens: int | None = Field(
        default=None,
        description=(
            "Threshold beyond which the conversation will be shortened. `None` "
            "when it will not happen: usage can still be shown, but there is no "
            "distance to an event worth announcing. Must also stay `None` when a "
            "policy is configured but its strategy is inert — announcing a "
            "threshold that never fires is worse than saying nothing, because "
            "the reader plans around it."
        ),
    )
    strategy: str | None = Field(
        default=None,
        description="Name of the strategy that will act, when there is one.",
    )
    correction_tokens: int = Field(
        default=0,
        description=(
            "The provider's count of the last request minus our estimate of it: "
            "negative on prose, which the estimate overcounts, positive on token-"
            "dense text. After a compaction, the share that belongs to the text it "
            "kept. 0 without a count, or when the count came from another deployment "
            "(provider class, URL or model)."
        ),
    )

    @property
    def total_tokens(self) -> int:
        """The whole request: the number the compaction threshold is checked against."""
        return max(0, self.message_tokens + self.tool_definition_tokens + self.correction_tokens)

    @property
    def fraction_used(self) -> float | None:
        """Share of the window in use, or `None` if the window is unknown.

        `None` is not an error to paper over with a zero: it is the difference
        between "I am using nothing" and "I do not know how much I am using".
        """
        if not self.context_window:
            return None
        return self.total_tokens / self.context_window

    @property
    def tokens_until_threshold(self) -> int | None:
        """How much is left before a policy acts, never negative."""
        if self.threshold_tokens is None:
            return None
        return max(0, self.threshold_tokens - self.total_tokens)


def measure(
    messages: list[LLMMessage],
    tool_defs: list[dict[str, Any]] | None = None,
    *,
    estimator: TokenEstimator | None = None,
    context_window: int | None = None,
    threshold_tokens: int | None = None,
    strategy: str | None = None,
    correction_tokens: int = 0,
) -> ContextUsage:
    """Measure a request. Changes nothing: it reads and counts."""
    estimator = estimator or HeuristicEstimator()
    return ContextUsage(
        message_tokens=estimator.estimate_messages(messages),
        tool_definition_tokens=estimator.estimate_tool_definitions(tool_defs or []),
        context_window=context_window,
        threshold_tokens=threshold_tokens,
        strategy=strategy,
        correction_tokens=correction_tokens,
    )


def _stable_json(value: Any) -> str:
    """Deterministic JSON, so the same structure always estimates the same."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
