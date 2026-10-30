"""Making the conversation fit the model's window.

Deciding **whether** to act is separate from deciding **what to do**: the first
is ``Compactor``, which a planner asks; the second is a ``CompactionStrategy``,
which is replaceable. The reason is that "which compaction runs" is what changes
between deployments, while "when it is needed" does not.

Two properties hold for every strategy and are not negotiable:

* **Additive.** ``InteractionContext`` is never modified. What gets shortened is
  the message projection of *one request*; the events stay whole and next turn
  the projection is rebuilt from scratch.
* **Ignorable.** If the whole history fits *this* model's window, the whole
  history is used. Compressing when it is not needed is a free loss.
"""

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from domyn.agents.core import BaseEvent, ExecutionEventType, Part
from domyn.agents.llm.base import LLMMessage, context_limit_from_error, reported_prompt_tokens
from domyn.agents.llm.context_window import deployment_key, remember_stated_window
from domyn.agents.logger import get_logger
from domyn.agents.planner_strategy.context_management.budget import (
    CONTEXT_USAGE_METADATA_KEY,
    HeuristicEstimator,
    TokenEstimator,
    measure,
)

_RECORDS_KEY = "compaction_records"
"""Where the durable decisions live inside ``InteractionContext.state``.

`state` is persisted per conversation by the runner and reloaded into every new
`InteractionContext`, which is what makes a compaction outlive the turn that
performed it. Without that, a manual compaction would evaporate: the next turn
re-renders the whole history, finds itself below the threshold, prunes nothing,
and the text the user asked to drop is back.

Keyed **by agent** rather than by conversation, and that is correctness rather
than tidiness: `context_visibility` means two agents can see different slices of
the same history, so "what has been dropped" is not a property they share.
"""

_ANCHORS_KEY = "context_anchors"
"""Where the provider's count of the last request lives, per agent, in
``InteractionContext.state``: ``{"reported", "estimated", "generation", "deployment"}``.

A count describes one rendering of the history, as one deployment tokenizes it
(``deployment_key``: provider class, URL, model). ``generation`` is the record's at the
time. A compaction since carries it over to the rendering it produced, with the share of
the correction that belongs to the text it kept (``_rescale``); a record changed any
other way retires it, and so does another deployment, whose tokenizer counts the same
text differently."""

_ANCHOR_RATIO = (0.25, 4.0)
"""Bounds on reported/estimated beyond which a count is ignored. Measured on real
providers: 0.56 for English prose, 2.4 for UUIDs and hex. Outside these a provider is
counting on another basis — a cache-excluded `input_tokens` reads 4 tokens on a
prompt of 8,561 — and trusting it would mean never compacting at all."""


class CompactionPolicy(BaseModel):
    """When to act, and how much to keep."""

    enabled: bool = True

    reserve_tokens: int = Field(
        default=16_384,
        gt=0,
        description=(
            "Space reserved for the reply. The threshold is `window - reserve`. "
            "Absolute rather than a percentage: it reserves what *one* reply "
            "needs, which does not scale with the window. A percentage would need "
            "a second parameter for small windows."
        ),
    )
    keep_recent_tokens: int = Field(
        default=20_000,
        gt=0,
        description=(
            "Budget for the tail kept verbatim, capped at half the room under the "
            "threshold (`Budget.tail_tokens`) so a small window can still compact."
        ),
    )
    protect_first_n: int = Field(
        default=1,
        ge=0,
        description=(
            "Head messages protected beyond the system prompt: without the first "
            "exchange the conversation loses its framing."
        ),
    )
    keep_last_user_messages: int = Field(
        default=1,
        ge=0,
        description=(
            "Real user messages a summary never replaces: the last N stay "
            "verbatim, wherever the tail is cut. An agent that lost the last "
            "request answers a summary of the request. Pruning never touches them: "
            "it only shortens tool output."
        ),
    )
    keep_recent_tool_results: int = Field(
        default=3,
        ge=0,
        description=(
            "How many of the most recent tool results a manual compaction leaves "
            "alone. The manual path counts in events rather than tokens, because "
            "it works on the history before anything is rendered — so the tail is "
            "protected by count. Three because the last couple of results are "
            "usually what the agent is actively reasoning about, and dropping "
            "those is how a compaction turns into amnesia."
        ),
    )
    prune_tool_results_over_chars: int = Field(
        default=200,
        ge=0,
        description="Below this length a tool result is not worth pruning.",
    )


class CompactionRecord(BaseModel):
    """What has already been dropped from one agent's view of the conversation.

    Holds event ids, not message positions: positions shift as the conversation
    grows, an event id does not. It is also the only handle that survives the
    re-render, since the message list is rebuilt from events every turn.
    """

    pruned_event_ids: list[str] = Field(default_factory=list)
    summary: str | None = None
    summarized_event_ids: list[str] = Field(default_factory=list)
    generation: int = Field(
        default=0,
        description=(
            "Bumped on every change. Not a lock: two concurrent turns for the "
            "same agent would each read, add, and write, and the loser's ids "
            "would be lost. Pruning is idempotent and additive, so losing an "
            "entry costs one re-prune rather than a corrupted history — a "
            "counter that makes the overwrite detectable is the proportionate "
            "answer, a lock is not."
        ),
    )


def read_record(state: dict[str, Any] | None, agent_name: str) -> CompactionRecord:
    """Never raises and never returns `None`: an absent record is an empty one."""
    if not state:
        return CompactionRecord()
    raw = (state.get(_RECORDS_KEY) or {}).get(agent_name)
    if not raw:
        return CompactionRecord()
    try:
        return CompactionRecord.model_validate(raw)
    except Exception:
        # A malformed record must not end the session. Losing it costs a
        # re-prune; raising here would make the conversation unusable.
        get_logger().warning(f"discarding an unreadable compaction record for {agent_name}")
        return CompactionRecord()


def record_summary(
    state: dict[str, Any] | None, agent_name: str, summary: str, event_ids: Iterable[str]
) -> CompactionRecord:
    """Store a summary and what it stands for.

    Replaces any previous summary rather than accumulating one: the new summary
    was produced from a stretch that already contained the old one, so keeping
    both would state the same history twice.
    """
    record = read_record(state, agent_name)
    if state is None:
        return record
    covered = list(dict.fromkeys([*record.summarized_event_ids, *[e for e in event_ids if e]]))
    if not covered:
        return record
    record.summary = summary
    record.summarized_event_ids = covered
    record.generation += 1
    state.setdefault(_RECORDS_KEY, {})[agent_name] = record.model_dump()
    return record


def add_to_record(
    state: dict[str, Any] | None, agent_name: str, event_ids: Iterable[str]
) -> CompactionRecord:
    """Add ids to an agent's record and persist it. Returns the new record."""
    record = read_record(state, agent_name)
    if state is None:
        return record
    known = set(record.pruned_event_ids)
    fresh = [event_id for event_id in event_ids if event_id and event_id not in known]
    if not fresh:
        return record
    record.pruned_event_ids.extend(fresh)
    record.generation += 1
    state.setdefault(_RECORDS_KEY, {})[agent_name] = record.model_dump()
    return record


def _rescale(correction: int, base: int, dropped: int) -> tuple[int, int]:
    """``(correction, base)`` once ``dropped`` estimated tokens of the text measured as
    ``base`` are gone.

    The correction is the provider's count of that text minus our estimate of it, so it
    belongs to the text: dropping part of it drops the same share. Not the whole of it,
    which charges the text kept for what was dropped (a summary was bought, and "still
    over the limit" reported, for a request the provider counted 11,893 against a 16,384
    threshold); and not none of it, since what is kept is usually just as dense (a pruned
    request estimated at 9,981 was counted 22,647). Both measured on hex tool output.
    Still an estimate, and it undercounts when the text kept is denser than the text
    dropped: after a summary that took prose and kept fresh hex output verbatim, a
    request was put at 15,160 and counted 22,609, about 5,400 of it a tool output added
    since the last count (estimated only, see ``Compactor.observe``).
    Without a base there is nothing to scale against, and the correction is kept whole.
    """
    if base <= 0:
        return correction, base
    kept = max(0, base - max(0, dropped))
    return round(correction * kept / base), kept


def _deployment(provider: Any) -> list[str] | None:
    """The deployment a count belongs to, as stored: a list, which is what a tuple
    becomes once the conversation state has been saved as JSON and read back."""
    key = deployment_key(provider)
    return list(key) if key is not None else None


class Budget(BaseModel):
    """Everything needed to decide, computed once by the caller."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    context_window: int
    reserve_tokens: int
    current_tokens: int
    """The whole request: conversation, turn context and tool definitions, with the
    correction measured on them."""
    fixed_tokens: int = 0
    """The part of ``current_tokens`` no compaction can reduce: tool definitions and the
    context injected for this turn, as estimated. Counted, never touched."""
    message_tokens: int = 0
    """The conversation, as estimated: what a compaction can shorten."""
    correction_tokens: int = 0
    """The provider's count of the last request minus our estimate of it
    (``Compactor.observe``), included in ``current_tokens``. It belongs to the text it
    was measured on, so a compaction that drops some of that text drops its share of it
    (``tokens_for``)."""
    correction_base_tokens: int = 0
    """Our estimate of the text the correction was measured on."""
    estimator: Any

    @property
    def threshold_tokens(self) -> int:
        return _threshold(self.context_window, self.reserve_tokens)

    @property
    def over_threshold(self) -> bool:
        return self.current_tokens > self.threshold_tokens

    def tokens_for(self, messages: list[LLMMessage]) -> int:
        """The whole request built on ``messages`` instead: estimated, plus the share of
        the correction that belongs to the text they kept. ``current_tokens`` on the
        conversation measured, and the number the next request starts from (the count
        is carried over the compaction, see ``Compactor.run``)."""
        conversation = int(self.estimator.estimate_messages(messages))
        correction, _ = _rescale(
            self.correction_tokens,
            self.correction_base_tokens,
            self.message_tokens - conversation,
        )
        return conversation + self.fixed_tokens + correction

    def fits(self, messages: list[LLMMessage]) -> bool:
        """Whether a request built on ``messages`` would be under the threshold."""
        return self.tokens_for(messages) <= self.threshold_tokens

    def tail_tokens(self, keep_recent_tokens: int) -> int:
        """How much recent conversation a compaction keeps verbatim, in estimated tokens.

        At most half the room the conversation has under the threshold: on a
        small window ``keep_recent_tokens`` alone can exceed it, and then the
        tail swallows everything, no compaction gets under the threshold, and
        every request tries again. The room is in the provider's tokens and the tail
        is measured text, so each estimated token of it costs what the provider counted
        per estimated token. Charged as a fixed cost instead, a correction larger than
        the threshold left no tail at all, and the summary took everything up to the
        last request.
        """
        room = max(0, self.threshold_tokens - self.fixed_tokens)
        measured = self.correction_base_tokens + self.correction_tokens
        if self.correction_tokens and self.correction_base_tokens > 0 and measured > 0:
            room = room * self.correction_base_tokens // measured
        return min(keep_recent_tokens, room // 2)


class CompactionOutcome(BaseModel):
    """What a strategy did."""

    messages: list[LLMMessage]
    changed: bool = False
    summary: str | None = Field(
        default=None,
        description="Text standing in for the events named in `summarized_event_ids`.",
    )
    summarized_event_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Events the summary replaces. Recorded so the summary is paid for "
            "once: re-summarizing on every render would cost a request per turn "
            "and give a different context each time for the same history, which "
            "is the surest way to make a session impossible to reproduce."
        ),
    )
    pruned_event_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Events whose text this compaction dropped, when they could be "
            "identified. Recorded so the decision is re-applied on later turns "
            "instead of being recomputed — which is what makes it stick below "
            "the threshold. Messages carrying no `source_event_id` are still "
            "pruned, just not remembered."
        ),
    )
    note: str | None = Field(
        default=None,
        description="Readable line about what was done, for the user and the logs.",
    )


class CompactionPlan(BaseModel):
    """The decision, before acting.

    Separate from the action because deciding is synchronous and cheap (estimate
    only) while acting can take seconds and a model call. It is that separation
    which lets a planner **announce** a compaction before performing it, instead
    of making context vanish in silence.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    needed: bool
    budget: Budget | None = None
    reason: str | None = None
    rejected: bool = Field(
        default=False,
        description=(
            "The provider rejected the request as too long. Said with the "
            "announcement, or a compaction far below the threshold looks arbitrary."
        ),
    )
    llm_provider: Any = Field(
        default=None,
        exclude=True,
        description=(
            "The agent's provider, for a strategy that needs to call a model. "
            "Carried on the plan because `assess()` is the only step that sees "
            "the agent, and threading it through every planner call site instead "
            "would mean editing two duplicated loops for every strategy that "
            "ever needs something new from the agent."
        ),
    )

    @property
    def current_tokens(self) -> int:
        return self.budget.current_tokens if self.budget else 0

    @property
    def threshold_tokens(self) -> int | None:
        return self.budget.threshold_tokens if self.budget else None


@dataclass(frozen=True)
class Split:
    """Protected head, compressible middle, verbatim tail."""

    head: list[LLMMessage]
    middle: list[LLMMessage]
    tail: list[LLMMessage]

    @property
    def has_middle(self) -> bool:
        return bool(self.middle)


def _context_window(agent: Any) -> int | None:
    """The window the agent's provider declares, or `None` if it declares none.

    Only a positive integer counts, because the value comes from an attribute on
    a foreign object: ``bool`` subclasses ``int``, and several objects let
    themselves be converted to one — a test double silently becomes ``1``, and a
    one-token window makes every conversation too long. An unknown window gets no
    invented threshold: better to stay inert than to infer.
    """
    window = getattr(getattr(agent, "llm_provider", None), "context_window", None)
    if isinstance(window, int) and not isinstance(window, bool) and window > 0:
        return window
    return None


def _adopt_window(agent: Any, window: int | None) -> None:
    """Make a window the provider stated in a rejection the agent's own.

    Set on the provider rather than in the conversation: a window belongs to the
    deployment, and every conversation on it benefits — and remembered for that
    deployment (``remember_stated_window``), so a provider built again for the next
    turn starts from it. Only ever lowered — a declared window larger than the real
    one is the mistake the rejection just proved; a smaller one is a choice.
    """
    provider = getattr(agent, "llm_provider", None)
    if provider is None or not window or window <= 0:
        return
    remember_stated_window(provider, window)
    declared = _context_window(agent)
    if declared is not None and declared <= window:
        return
    try:
        provider.context_window = window
    except Exception as exc:
        # A provider that refuses the attribute keeps working as it did.
        get_logger().debug(f"context window not adopted: {type(exc).__name__}: {exc}")
        return
    get_logger().info(
        f"context window of {getattr(provider, 'model_name', 'the model')} learned from "
        f"the provider's rejection: {window} tokens (declared: {declared or 'none'})"
    )


def _threshold(context_window: int, reserve_tokens: int) -> int:
    """Request size beyond which we act. Never negative: on a window smaller than
    the reserve the threshold is zero, i.e. "always", which is the right reading."""
    return max(0, context_window - reserve_tokens)


def _history_text_size(event: Any) -> int:
    """How much text an event contributes, measured where it actually lives.

    A tool result keeps its payload in ``action.observation``, not in ``content``
    — measuring ``content`` would find almost nothing and the manual compaction
    would decide there was never anything worth dropping.
    """
    action = getattr(event, "action", None)
    observation = getattr(action, "observation", None) if action else None
    if observation:
        return len(str(observation))
    parts = getattr(event, "content", None) or []
    return sum(len(getattr(part, "text", "") or "") for part in parts)


_DROPPED_MARKER = "to save context"
"""Substring that identifies a placeholder this module wrote.

Needed so pruning can recognise its own work. Without it a second pass replaces
the placeholder with a placeholder *about the placeholder* — "1 lines", the file
name gone, the exit code gone — quietly destroying the very summary that made
the drop tolerable.
"""


def _already_dropped(message: LLMMessage) -> bool:
    content = message.content
    return isinstance(content, str) and content.startswith("[") and _DROPPED_MARKER in content


def _is_tool_result(message: LLMMessage) -> bool:
    return message.role == "tool" or message.tool_call_id is not None


def _carries_tool_calls(message: LLMMessage) -> bool:
    return bool(message.tool_calls)


class TokenBudgetSplitter(BaseModel):
    """Splits by walking backwards from the end until the budget runs out.

    One constraint dominates all the others: **never cut between a tool call and
    its result**. A model handed the result of a call it cannot see, or a call
    whose result has vanished, receives a malformed conversation — and providers
    reject it.
    """

    def split(
        self,
        messages: list[LLMMessage],
        *,
        policy: CompactionPolicy,
        estimator: TokenEstimator,
        keep_recent_tokens: int | None = None,
    ) -> Split:
        """``keep_recent_tokens`` replaces the policy's tail budget: what a
        strategy passes once it knows the window (``Budget.tail_tokens``)."""
        if not messages:
            return Split([], [], [])

        head_size = self._head_size(messages, policy)
        head = messages[:head_size]
        rest = messages[head_size:]
        if not rest:
            return Split(head, [], [])

        if keep_recent_tokens is None:
            keep_recent_tokens = policy.keep_recent_tokens
        cut = self._cut_point(rest, estimator=estimator, keep_recent_tokens=keep_recent_tokens)
        return Split(head, rest[:cut], rest[cut:])

    # --- internals --------------------------------------------------------

    @staticmethod
    def _head_size(messages: list[LLMMessage], policy: CompactionPolicy) -> int:
        """The system prompt plus the first ``protect_first_n``."""
        size = 1 if messages and messages[0].role == "system" else 0
        return min(len(messages), size + policy.protect_first_n)

    def _cut_point(
        self,
        rest: list[LLMMessage],
        *,
        estimator: TokenEstimator,
        keep_recent_tokens: int,
    ) -> int:
        """Index in ``rest`` where the tail begins: by budget alone.

        The last request is protected where it can be lost — from the summary
        (`TwoPhaseCompaction`); pruning only ever shortens tool output — and not
        by stretching the tail back to it. That stretch put every output of a
        long turn in the tail, and compaction found nothing to do exactly when
        one turn had filled the window.
        """
        spent = 0
        cut = len(rest)
        for index in range(len(rest) - 1, -1, -1):
            # Checked before consuming: the message that crosses the budget is kept.
            if spent > keep_recent_tokens:
                break
            spent += estimator.estimate_messages([rest[index]])
            cut = index
        return self._retreat_off_tool_boundary(rest, cut)

    @staticmethod
    def _retreat_off_tool_boundary(rest: list[LLMMessage], cut: int) -> int:
        """Retreat until the boundary no longer splits a call from its result.

        The cut lands badly in two ways: the tail starts with a result (its call
        would be in the middle), or the middle ends with a call (its result would
        be in the tail). Either way we retreat, pulling more into the tail:
        keeping too much always beats sending a malformed conversation.
        """
        while 0 < cut < len(rest) and (
            _is_tool_result(rest[cut]) or _carries_tool_calls(rest[cut - 1])
        ):
            cut -= 1
        return cut


class CompactionStrategy(ABC, BaseModel):
    """A way of making the conversation fit the window."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: ClassVar[str] = "abstract"

    acts: ClassVar[bool] = True
    """Whether this strategy can change anything.

    It exists because a front-end has to be able to tell the truth about *when*
    the conversation will be shortened, and "never" is a legitimate answer.
    Without it, with the inert strategy mounted by default, a CLI would announce
    a compaction that will not happen — and the reader would plan their work
    around an imaginary threshold."""

    @abstractmethod
    async def compact(
        self,
        messages: list[LLMMessage],
        *,
        budget: Budget,
        policy: CompactionPolicy,
        state: dict[str, Any],
        **kwargs: Any,
    ) -> CompactionOutcome:
        """Return the list to send to the model.

        Must never raise: the caller treats a failure as "do not compress", and
        an oversized request is a smaller problem than a turn that dies inside
        context management.
        """


class NoOpCompaction(CompactionStrategy):
    """Does nothing. Useful to switch off without dismantling the
    configuration, and as a baseline in tests."""

    type: ClassVar[str] = "noop"
    acts: ClassVar[bool] = False

    async def compact(
        self,
        messages: list[LLMMessage],
        *,
        budget: Budget,
        policy: CompactionPolicy,
        state: dict[str, Any],
        **kwargs: Any,
    ) -> CompactionOutcome:
        return CompactionOutcome(messages=messages, changed=False)


class Compactor(BaseModel):
    """Measures, decides, and on request shortens.

    It is **not** a ``ContextManager`` wrapping another one, and the first draft
    was. The flaw in that shape: ``get_context`` is not a generator, so a
    compaction hidden inside it cannot emit events — and the only way to let a
    front-end know was to pass a note through ``context.state``, i.e. a service
    channel instead of the framework's own. For an operation that loses
    information and will take seconds, being invisible is a defect.

    So a planner drives it in five lines: ask whether it is needed, announce,
    apply, announce the result. The six ``ContextManager`` classes stay untouched.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    policy: CompactionPolicy = CompactionPolicy()
    strategy: CompactionStrategy = Field(default_factory=lambda: TwoPhaseCompaction())
    """Prune, and summarize only if pruning was not enough.

    It was inert once, on the reasoning that a behaviour change should be opted
    into. That reasoning was wrong: at the threshold the alternative is not
    "leave things alone" — it is a rejected request and a lost turn.

    `two_phase` rather than `prune_only` because it *is* `prune_only` with a
    fallback, and the fallback only runs once the free, deterministic phase has
    demonstrably failed to get under the limit. The extra request is spent at the
    one moment when not spending it means losing the conversation. Both
    `prune_only` and `noop` remain selectable for anyone who wants no model call
    on this path, or nothing at all.
    """
    estimator: Any = Field(default_factory=lambda: HeuristicEstimator())
    """Replaceable: the heuristic by default, or anything with the same three methods."""

    @property
    def active(self) -> bool:
        """Whether a compaction can actually happen.

        Being enabled is not enough: with an inert strategy the answer to "how
        much is left" is "it will not happen", and it has to be said that way."""
        return self.policy.enabled and self.strategy.acts

    @property
    def strategy_name(self) -> str | None:
        return type(self.strategy).type if self.active else None

    def assess(
        self,
        messages: list[LLMMessage],
        agent: Any,
        *,
        tool_defs: list[dict[str, Any]] | None = None,
        turn_context: list[LLMMessage] | None = None,
        state: dict[str, Any] | None = None,
    ) -> CompactionPlan:
        """Synchronous and cheap: it calls nobody.

        Decides on the whole request, the total ``usage_metadata`` reports:
        ``messages`` is the conversation a strategy may shorten; ``tool_defs`` and
        ``turn_context`` travel with every request and are only counted. Deciding
        on the conversation alone is how a CLI announced "compaction in ~0
        tokens" over requests that then went out uncompacted.
        """
        if not self.policy.enabled:
            return CompactionPlan(needed=False, reason="disabled")
        if not self.strategy.acts:
            # Named before the window check on purpose: without a strategy,
            # configuring a window would change nothing, so the message must
            # name the first thing to fix rather than the first thing checked.
            return CompactionPlan(needed=False, reason="no compaction strategy")

        window = _context_window(agent)
        if window is None:
            return CompactionPlan(needed=False, reason="model context window unknown")

        budget = self._budget(messages, tool_defs, turn_context, state, agent, window=window)
        provider = getattr(agent, "llm_provider", None)
        if not budget.over_threshold:
            return CompactionPlan(
                needed=False, budget=budget, reason="below threshold", llm_provider=provider
            )
        return CompactionPlan(
            needed=True,
            budget=budget,
            llm_provider=provider,
            reason=(
                f"{budget.current_tokens} estimated tokens over the "
                f"{budget.threshold_tokens} threshold"
            ),
        )

    def overflow_plan(
        self,
        messages: list[LLMMessage],
        agent: Any,
        *,
        tool_defs: list[dict[str, Any]] | None = None,
        turn_context: list[LLMMessage] | None = None,
        state: dict[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> CompactionPlan | None:
        """The compaction to run after the provider rejected a request as too long.

        Needs no declared window: the rejection is the signal, and a stronger one
        than any number in a configuration. The estimate counted less than the
        provider did, so a threshold built on it cannot be trusted either; the
        target is half the estimate, reached only by actually shortening the
        conversation. And a provider nobody declared a window for is the one
        most likely to be rejected. `None` when no compaction can happen at all:
        switched off, or an inert strategy.

        When the rejection states the model's window, the provider adopts it (see
        ``_adopt_window``): the next requests are then compacted before they are
        rejected.
        """
        if not self.active:
            return None
        if error is not None:
            _adopt_window(agent, context_limit_from_error(error))
        budget = self._budget(messages, tool_defs, turn_context, state, agent, window=None)
        return CompactionPlan(
            needed=True,
            budget=budget,
            reason="the provider rejected the request as too long",
            rejected=True,
            llm_provider=getattr(agent, "llm_provider", None),
        )

    def _budget(
        self,
        messages: list[LLMMessage],
        tool_defs: list[dict[str, Any]] | None,
        turn_context: list[LLMMessage] | None,
        state: dict[str, Any] | None,
        agent: Any,
        *,
        window: int | None,
    ) -> Budget:
        """The request measured: estimated, plus the correction the provider's last
        count gives it. ``window`` ``None`` after a rejection: not the model's window
        then, but the one whose threshold is half the request, which is all a strategy
        reads from it."""
        conversation = int(self.estimator.estimate_messages(messages))
        fixed = int(self.estimator.estimate_messages(turn_context or []))
        fixed += int(self.estimator.estimate_tool_definitions(tool_defs or []))
        correction, base = self._anchor(state, agent)
        current = conversation + fixed + correction
        reserve = self.policy.reserve_tokens
        return Budget(
            context_window=window if window is not None else reserve + current // 2,
            reserve_tokens=reserve,
            current_tokens=current,
            fixed_tokens=fixed,
            message_tokens=conversation,
            correction_tokens=correction,
            correction_base_tokens=base,
            estimator=self.estimator,
        )

    def observe(
        self,
        usage: dict[str, Any] | None,
        measured: dict[str, Any] | None,
        state: dict[str, Any] | None,
        agent_name: str,
        provider: Any = None,
    ) -> None:
        """Remember what the provider counted for the request just answered.

        ``measured`` is the ``usage_metadata`` that went out with that request. From
        then on the size of a request is the provider's count of the last one plus
        an estimate of what was added since: the error of the estimate stays on the
        addition, instead of on the whole conversation. Arithmetically it is the
        estimate plus a fixed correction, reported minus estimated, kept for the
        rendering it was counted on (carried over a compaction, see ``run``) and for
        ``provider``'s deployment alone.

        Not the calibration this replaces: that multiplied every estimate by a
        ratio learned on other text, so prose made dense text look small.
        """
        reported = reported_prompt_tokens(usage)
        raw = (measured or {}).get(CONTEXT_USAGE_METADATA_KEY) or {}
        estimated = int(raw.get("message_tokens") or 0) + int(
            raw.get("tool_definition_tokens") or 0
        )
        if state is None or reported is None or estimated <= 0:
            return
        low, high = _ANCHOR_RATIO
        if not low <= reported / estimated <= high:
            get_logger().warning(
                f"provider token count ignored for {agent_name}: {reported} reported "
                f"against {estimated} estimated"
            )
            return
        state.setdefault(_ANCHORS_KEY, {})[agent_name] = {
            "reported": reported,
            "estimated": estimated,
            "generation": read_record(state, agent_name).generation,
            "deployment": _deployment(provider),
        }

    def _anchor(self, state: dict[str, Any] | None, agent: Any) -> tuple[int, int]:
        """``(correction, the estimate it was measured on)`` for the history as it
        renders now. ``(0, 0)`` without a count, after a change to the record that
        did not carry it, or when another deployment counted it: a ``/model``, a
        conversation resumed on another model, a model the gateway replaced."""
        name = getattr(agent, "name", None)
        if not state or not name:
            return 0, 0
        anchor = (state.get(_ANCHORS_KEY) or {}).get(name)
        if not isinstance(anchor, dict):
            return 0, 0
        if anchor.get("generation") != read_record(state, name).generation:
            return 0, 0
        if anchor.get("deployment") != _deployment(getattr(agent, "llm_provider", None)):
            return 0, 0
        try:
            reported, estimated = int(anchor["reported"]), int(anchor["estimated"])
        except (KeyError, TypeError, ValueError):
            return 0, 0
        return reported - estimated, estimated

    def _carry_anchor(
        self, state: dict[str, Any] | None, agent_name: str, generation: int, dropped: int
    ) -> int:
        """Carry the last count over a compaction that dropped ``dropped`` estimated
        tokens of the conversation; return how much of the correction went with them.

        Without it a compaction retired the count, and the request it produced went
        out measured on the estimate alone, while the compaction had judged it with
        the whole correction: two bases for one request. Carried, the request that
        goes out, a recovery from its rejection and the result reported all read the
        same number until the next response counts again. Only a count that described
        the rendering before the compaction (``generation``) is carried.
        """
        anchors = (state or {}).get(_ANCHORS_KEY) or {}
        anchor = anchors.get(agent_name)
        now = read_record(state, agent_name).generation
        if not isinstance(anchor, dict) or anchor.get("generation") != generation:
            return 0
        if now == generation:
            return 0
        try:
            reported, estimated = int(anchor["reported"]), int(anchor["estimated"])
        except (KeyError, TypeError, ValueError):
            return 0
        correction, base = _rescale(reported - estimated, estimated, dropped)
        anchors[agent_name] = {
            **anchor,
            "reported": base + correction,
            "estimated": base,
            "generation": now,
        }
        return reported - estimated - correction

    async def apply(
        self,
        messages: list[LLMMessage],
        plan: CompactionPlan,
        state: dict[str, Any] | None = None,
    ) -> CompactionOutcome:
        """Apply the strategy. Never raises.

        A failure means "do not compress": the oversized request goes out anyway
        and the provider will reject it with a readable error, which is a smaller
        problem than a turn dying inside context management.
        """
        if not plan.needed or plan.budget is None:
            return CompactionOutcome(messages=messages, changed=False)
        try:
            outcome = await self.strategy.compact(
                messages,
                budget=plan.budget,
                policy=self.policy,
                state=state or {},
                llm_provider=plan.llm_provider,
            )
        except Exception as exc:
            get_logger().warning(f"compaction failed, sending uncompacted: {type(exc).__name__}")
            get_logger().debug(f"compaction failure detail: {exc}")
            return CompactionOutcome(messages=messages, changed=False)

        if outcome.changed:
            after = plan.budget.tokens_for(outcome.messages)
            get_logger().info(
                f"context compacted: {plan.budget.current_tokens} -> {after} estimated tokens"
            )
        return outcome

    def replay(
        self, messages: list[LLMMessage], state: dict[str, Any] | None, agent_name: str
    ) -> list[LLMMessage]:
        """Re-apply what has already been dropped for this agent.

        Called on every render, before the threshold is even considered, because
        the message list is rebuilt from events each turn and would otherwise
        arrive with the full text restored. Cheap when the record is empty, which
        is the common case.
        """
        record = read_record(state, agent_name)
        if not record.pruned_event_ids and not record.summary:
            return messages

        out = messages
        if record.summary and record.summarized_event_ids:
            # The summary goes back first. It removes messages, so pruning what
            # survives afterwards is both cheaper and correct — pruning messages
            # that are about to disappear would be work thrown away.
            out = _restore_summary(out, record.summary, set(record.summarized_event_ids))

        if record.pruned_event_ids:
            # The strategy's own pruner when it has one, so the placeholder text
            # stays identical to what the threshold path writes. A fallback when
            # it does not — including `noop`: a decision already taken stands
            # regardless of which strategy is mounted now, and un-pruning on a
            # configuration change would restore text the agent has already been
            # told is gone.
            pruner = getattr(self.strategy, "pruner", None) or ToolResultPruner()
            out, pruned, _ = pruner.prune_recorded(out, set(record.pruned_event_ids))
            if pruned:
                get_logger().debug(
                    f"replayed {pruned} previously dropped tool outputs for {agent_name}"
                )
        return out

    def request_from_history(
        self,
        events: list[Any],
        state: dict[str, Any] | None,
        agent_name: str,
        rendered: list[LLMMessage] | None = None,
    ) -> BaseEvent:
        """Compact on request, working on the history rather than on messages.

        Deliberately not the threshold path. It runs before anything is rendered,
        so it selects by event: the most recent `keep_recent_tool_results` are
        left alone and the older large ones are recorded as dropped. The effect
        lands on the next request, when the record is replayed — which is also why
        it lasts, whereas a compaction of a transient message list would not.

        ``rendered`` is the conversation as the next request would render it.
        Given, the event carries ``tokens_saved``: the record replayed onto it
        before and after, so the saving is counted on the text the model reads, plus
        the share of the measured correction that went with it (``_carry_anchor``).

        Returns the event to show, always: "nothing to compact" is an answer, and
        a command that stays silent looks broken.
        """
        candidates = [
            event
            for event in events
            if getattr(event, "event_type", None)
            in (ExecutionEventType.TOOL_END, ExecutionEventType.AGENT_END)
            and getattr(event, "event_id", None)
        ]
        already = set(read_record(state, agent_name).pruned_event_ids)
        older = candidates[: -self.policy.keep_recent_tool_results or None]
        chosen = [
            event.event_id
            for event in older
            if event.event_id not in already
            and _history_text_size(event) > self.policy.prune_tool_results_over_chars
        ]

        def size() -> int | None:
            if rendered is None:
                return None
            return self.tokens_after(self.replay(rendered, state, agent_name))

        generation = read_record(state, agent_name).generation
        before = size()
        if chosen:
            add_to_record(state, agent_name, chosen)
            note = f"context compacted: {len(chosen)} old tool outputs dropped"
        elif already:
            note = "nothing further to compact: the older tool outputs are already dropped"
        else:
            note = "nothing to compact yet: no old tool output large enough to be worth dropping"

        metadata: dict[str, Any] = {
            "changed": bool(chosen),
            "events_dropped": len(chosen),
            "requested": True,
        }
        after = size()
        if before is not None and after is not None:
            dropped = before - after
            metadata["tokens_saved"] = dropped + self._carry_anchor(
                state, agent_name, generation, dropped
            )

        get_logger().info(f"manual compaction for {agent_name}: {len(chosen)} events recorded")
        return BaseEvent(
            event_type=ExecutionEventType.CONTEXT_COMPACTED,
            author=agent_name,
            content=[Part(text=note)],
            metadata=metadata,
        )

    def tokens_after(self, messages: list[LLMMessage]) -> int:
        return int(self.estimator.estimate_messages(messages))

    # --- for a planner to use ---------------------------------------------
    #
    # The `yield` has to live in the planner's generator, but everything else
    # lives here. The reason is concrete: `ToolUseWithThoughtPlannerStrategy` is
    # a full copy of its parent's loop, so any logic written inside one of them
    # is silently missing from the other — which is exactly how this measurement
    # came to not work on the planner the CLI actually uses.

    def start_event(self, plan: CompactionPlan, author: str) -> BaseEvent:
        """The announcement, to emit **before** acting.

        Before and not after because with a summarizing strategy there is a
        multi-second pause here, and an agent stopped without explanation looks
        stuck.
        """
        return BaseEvent(
            event_type=ExecutionEventType.CONTEXT_COMPACTION_START,
            author=author,
            content=[Part(text="Compacting the conversation to fit the context window")],
            metadata={
                "tokens_before": plan.current_tokens,
                "threshold_tokens": plan.threshold_tokens,
                "rejected": plan.rejected,
            },
        )

    async def run(
        self,
        messages: list[LLMMessage],
        plan: CompactionPlan,
        state: dict[str, Any] | None,
        author: str,
    ) -> tuple[list[LLMMessage], BaseEvent]:
        """Apply and return ``(messages, closing event)``."""
        before = plan.current_tokens
        outcome = await self.apply(messages, plan, state)
        after = (
            plan.budget.tokens_for(outcome.messages)
            if plan.budget is not None
            else self.tokens_after(outcome.messages)
        )
        generation = read_record(state, author).generation
        if outcome.pruned_event_ids:
            # Recorded so the automatic path is monotonic too: without this the
            # same decision is recomputed every turn and reverses itself the
            # moment the conversation dips back below the threshold.
            add_to_record(state, author, outcome.pruned_event_ids)
        if outcome.summary and outcome.summarized_event_ids:
            # The expensive one. Recorded so it is paid for once: re-summarizing
            # every render would cost a request per turn and hand the model a
            # different context each time for the same history.
            record_summary(state, author, outcome.summary, outcome.summarized_event_ids)
        if plan.budget is not None:
            # The next request is measured on the rendering this produced: carried,
            # the count keeps the share of its correction that belongs to the text
            # kept, which is the share `after` was computed with.
            self._carry_anchor(
                state,
                author,
                generation,
                plan.budget.message_tokens - self.tokens_after(outcome.messages),
            )

        # Did it actually get under the limit? "I pruned something" and "the
        # request will now fit" are different claims, and only the second one is
        # what anybody wanted. Reporting the first as success is how a session
        # dies on a rejected request right after being told the context was
        # compacted.
        threshold = plan.threshold_tokens
        sufficient = threshold is None or after <= threshold
        if not sufficient:
            get_logger().warning(
                f"compaction did not get under the limit for {author}: {after} tokens "
                f"against a {threshold} threshold — dropping tool output is no longer "
                "enough, the conversation itself is now the bulk of the context"
            )
        done = BaseEvent(
            event_type=ExecutionEventType.CONTEXT_COMPACTED,
            author=author,
            content=[Part(text=outcome.note or "context compacted")],
            metadata={
                "tokens_before": before,
                "tokens_after": after,
                # False when the strategy found nothing to do: saying so avoids
                # implying a recovery that did not happen, which is how people
                # stop investigating a session that is still too large.
                "changed": outcome.changed,
                # False when the result is still over the limit. Distinct from
                # `changed` on purpose: a compaction can prune plenty and still
                # not be enough, and past a certain point `prune_only` always
                # finds one fresh tool output to drop while the total keeps
                # climbing. Whoever reads this is the one who can act — by
                # summarizing, or by starting a new conversation.
                "sufficient": sufficient,
                "threshold_tokens": threshold,
            },
        )
        return outcome.messages, done

    def usage_metadata(
        self,
        messages: list[LLMMessage],
        tool_defs: list[dict[str, Any]] | None,
        agent: Any,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """The context measurement, ready for ``LLM_START`` metadata.

        Computed like ``assess`` computes its total, so the usage a front-end
        shows and the number the threshold is checked against are one number.
        Threshold and strategy name only when a compaction will actually happen:
        a threshold announced and never reached is misinformation.
        """
        window = _context_window(agent)
        usage = measure(
            messages,
            tool_defs,
            estimator=self.estimator,
            context_window=window,
            threshold_tokens=(
                _threshold(window, self.policy.reserve_tokens)
                if self.active and window is not None
                else None
            ),
            strategy=self.strategy_name,
            correction_tokens=self._anchor(state, agent)[0],
        )
        get_logger().debug(
            f"context usage: {usage.total_tokens} tokens "
            f"({usage.message_tokens} messages + {usage.tool_definition_tokens} "
            f"definitions of {len(tool_defs or [])} tools, estimated; "
            f"{usage.correction_tokens:+d} measured), window "
            f"{usage.context_window or 'unknown'}"
        )
        return {CONTEXT_USAGE_METADATA_KEY: usage.model_dump()}


# ---------------------------------------------------------------------------
# The deterministic phase
# ---------------------------------------------------------------------------

_EXIT_CODE = re.compile(r"exit code:\s*(-?\d+)", re.IGNORECASE)


class ToolResultPruner(BaseModel):
    """Replaces old tool results with a line saying what they were.

    This is the phase that recovers most of the space in all three harnesses we
    compared against, and it costs no model call. In an agent it is tool output
    that inflates the context: the conversation text is comparatively small.

    The placeholder is **not** a bare `[removed]`. Without knowing what was there
    and how it went, the model would re-run the command — and the pruning would
    have cost time instead of buying space. For a tool's result it also gives the
    call that reads it back (`LLMMessage.read_back`, set by the context manager),
    for when the model needs what it said.

    Trap found by accident: the placeholder starts with ``[``, which rich reads
    as a style tag. Not a problem today — this text goes to the model, and the
    CLI renders events, not the compacted list — but anyone printing it through
    rich with markup on would watch it disappear.
    """

    def prune(self, messages: list[LLMMessage], *, over_chars: int) -> tuple[list[LLMMessage], int]:
        """Return ``(messages, how many pruned)``. Originals are not modified."""
        out, count, _ = self.prune_reporting(messages, over_chars=over_chars)
        return out, count

    def prune_reporting(
        self, messages: list[LLMMessage], *, over_chars: int
    ) -> tuple[list[LLMMessage], int, list[str]]:
        """As `prune`, and also says *which events* it dropped.

        The count and the id list are not the same number: a message rendered
        from no identifiable event is still pruned, it just cannot be remembered.
        Returning both keeps that difference visible instead of implying the
        record covers everything.
        """
        return self._apply(messages, keep=lambda message: len(str(message.content)) <= over_chars)

    def prune_recorded(
        self, messages: list[LLMMessage], event_ids: set[str]
    ) -> tuple[list[LLMMessage], int, list[str]]:
        """Re-apply a decision already taken, whatever the size of the text.

        Size is irrelevant here: the question is no longer "is this big enough to
        be worth dropping" but "was this already dropped". Asking the first
        question again would quietly restore anything that had shrunk.
        """
        return self._apply(
            messages,
            keep=lambda message: getattr(message, "source_event_id", None) not in event_ids,
        )

    def _apply(
        self, messages: list[LLMMessage], *, keep: Callable[[LLMMessage], bool]
    ) -> tuple[list[LLMMessage], int, list[str]]:
        pruned = 0
        event_ids: list[str] = []
        out: list[LLMMessage] = []
        for message in messages:
            if (
                not _is_tool_result(message)
                or not isinstance(message.content, str)
                or _already_dropped(message)
                or keep(message)
            ):
                out.append(message)
                continue
            replacement = message.model_copy(deep=True)
            replacement.content = self._placeholder(message)
            out.append(replacement)
            pruned += 1
            source = getattr(message, "source_event_id", None)
            if source:
                event_ids.append(source)
        return out, pruned, event_ids

    def _placeholder(self, message: LLMMessage) -> str:
        content = message.content if isinstance(message.content, str) else ""
        lines = content.count("\n") + 1
        name = message.name or "tool"

        outcome = ""
        found = _EXIT_CODE.search(content)
        if found:
            outcome = f", exit {found.group(1)}"

        recovery = f" Read it again with {message.read_back}" if message.read_back else ""

        return f"[{name} output dropped {_DROPPED_MARKER} — {lines} lines{outcome}.{recovery}]"


def _prune_irrelevant_first(
    pruner: ToolResultPruner, messages: list[LLMMessage], state: dict[str, Any]
) -> tuple[list[LLMMessage], int, list[str]]:
    """Drop what a decision model judged no longer needed, before size is considered.

    `decisions/context.py` leaves its verdict on `context.state`; this re-uses
    `prune_recorded`, which already exists to re-apply a decision by event id.
    An empty verdict — no model, nothing confident — returns the input and the
    size rule does all the work, exactly as before.
    """
    from domyn.agents.decisions.context import irrelevant_ids

    ids = irrelevant_ids(state)
    if not ids:
        return messages, 0, []
    return pruner.prune_recorded(messages, ids)


class PruneOnlyCompaction(CompactionStrategy):
    """The deterministic phase only: no model call.

    Not a stepping stone towards something better. It is the right answer for
    anyone who does not want extra spend or non-determinism on the critical path,
    and it will be the fallback of a summarizing strategy when the summary fails.
    """

    type: ClassVar[str] = "prune_only"

    splitter: TokenBudgetSplitter = TokenBudgetSplitter()
    pruner: ToolResultPruner = ToolResultPruner()

    async def compact(
        self,
        messages: list[LLMMessage],
        *,
        budget: Budget,
        policy: CompactionPolicy,
        state: dict[str, Any],
        **kwargs: Any,
    ) -> CompactionOutcome:
        split = self.splitter.split(
            messages,
            policy=policy,
            estimator=budget.estimator,
            keep_recent_tokens=budget.tail_tokens(policy.keep_recent_tokens),
        )
        if not split.has_middle:
            # Everything is protected or in the tail: nothing to act on, and
            # saying so beats returning a "changed" list identical to the input.
            return CompactionOutcome(messages=messages, changed=False)

        earlier, by_relevance, relevance_ids = _prune_irrelevant_first(
            self.pruner, split.head + split.middle, state
        )
        head_and_middle, by_size, size_ids = self.pruner.prune_reporting(
            earlier, over_chars=policy.prune_tool_results_over_chars
        )
        pruned = by_relevance + by_size
        event_ids = relevance_ids + size_ids
        if not pruned:
            return CompactionOutcome(messages=messages, changed=False)

        return CompactionOutcome(
            messages=head_and_middle + split.tail,
            changed=True,
            note=f"context compacted: {pruned} old tool outputs dropped",
            pruned_event_ids=event_ids,
        )


SUMMARY_MARKER = "[conversation summary]"
"""Prefix identifying an injected summary.

So the model, and anyone reading a request, can tell a reconstruction from
something that was actually said. It carries no logic — a later summarization
folds an earlier summary into the new one on purpose, since the new one is meant
to subsume it, and the record keeps a single summary rather than a chain.
"""


class ConversationSummarizer(BaseModel):
    """Turns a stretch of conversation into one paragraph, by asking a model.

    Separate from the strategy so it can be replaced on its own: the prompt is
    the part most likely to need tuning per deployment, and it should not require
    reimplementing the phase logic around it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    system_prompt: str = (
        "You are compacting a conversation between a user and an AI agent so it "
        "fits in a smaller context window. Write a summary that lets the agent "
        "carry on without the original text.\n\n"
        "Preserve, in this order of priority:\n"
        "1. What the user asked for, including constraints and preferences they "
        "stated, and any decision they made.\n"
        "2. What has been established as fact: values found, files touched, "
        "results obtained, conclusions reached.\n"
        "3. What was tried and failed, and why — so it is not tried again.\n"
        "4. What is still open.\n\n"
        "Be specific: keep names, paths, identifiers and numbers verbatim. Do "
        "not add advice, opinions or a preamble. Do not invent anything that is "
        "not in the text. Write prose, not a transcript."
    )
    user_preamble: str = "Summarize this conversation:\n\n"
    max_chars_per_message: int = Field(
        default=2_000,
        gt=0,
        description="Cap on each message fed to the summarizer.",
    )
    max_transcript_chars: int = Field(
        default=120_000,
        gt=0,
        description=(
            "Cap on the whole transcript, which the per-message cap does not "
            "give: three hundred messages of two thousand characters is 200k "
            "tokens, so the summarization request would be rejected too — and "
            "then the turn has no way out at all. ~40k tokens at three "
            "characters per token, which fits any window we would be compacting "
            "for."
        ),
    )

    async def summarize(self, messages: list[LLMMessage], *, llm_provider: Any) -> str:
        """Ask for the summary. Returns the text, or raises for the caller to handle."""
        transcript = self._transcript(messages)
        response = await llm_provider.generate(
            [
                LLMMessage(role="system", content=self.system_prompt),
                LLMMessage(role="user", content=self.user_preamble + transcript),
            ]
        )
        text = (getattr(response, "content", "") or "").strip()
        if not text:
            raise ValueError("the summarizer returned nothing")
        return text

    def _transcript(self, messages: list[LLMMessage]) -> str:
        """The stretch as text, kept under the cap by dropping the *oldest* part.

        Oldest and not newest: what happened last is what the agent is still
        working on, and a summary missing the recent half is worse than one
        missing the distant half. What was cut is stated rather than silently
        absent, so the summary does not present itself as complete.
        """
        rendered = [text for text in (self._render(message) for message in messages) if text]
        kept: list[str] = []
        budget = self.max_transcript_chars
        for text in reversed(rendered):
            if len(text) > budget:
                break
            kept.append(text)
            budget -= len(text)
        kept.reverse()

        dropped = len(rendered) - len(kept)
        if dropped:
            get_logger().info(
                f"summarizer input capped: {dropped} of {len(rendered)} messages left out"
            )
            kept.insert(0, f"[{dropped} earlier messages omitted, too long to include]")
        return "\n\n".join(kept)

    def _render(self, message: LLMMessage) -> str:
        """One message as plain text, for the summarizer to read."""
        body = message.content if isinstance(message.content, str) else ""
        if message.tool_calls:
            names = ", ".join(call.function.name for call in message.tool_calls)
            body = f"{body} (calls: {names})".strip()
        if not body:
            return ""
        if len(body) > self.max_chars_per_message:
            body = body[: self.max_chars_per_message] + " …(truncated)"
        who = message.name or message.role
        return f"{who}: {body}"


class TwoPhaseCompaction(CompactionStrategy):
    """Prune first, summarize only if pruning was not enough.

    The order is the point. Phase one is deterministic, free and instant, and on
    an agent conversation it recovers most of the space, because tool output is
    what inflates a context. Phase two costs a request and is not reproducible,
    so it runs when the cheap answer has demonstrably failed — not on a schedule
    and not on a guess.

    Measured on a heavy session: pruning alone stays sufficient for roughly two
    hundred turns and then stops being enough for good, because the conversation
    text keeps growing and no amount of pruning touches it. That crossing is
    exactly when phase two earns its cost.
    """

    type: ClassVar[str] = "two_phase"

    splitter: TokenBudgetSplitter = TokenBudgetSplitter()
    pruner: ToolResultPruner = ToolResultPruner()
    summarizer: ConversationSummarizer = ConversationSummarizer()
    llm_provider: Any = Field(
        default=None,
        description=(
            "Model used for summarizing. `None` means the agent's own, which is "
            "the sane default: it already has credentials and a window. Set it "
            "to send summarization to something cheaper."
        ),
    )

    async def compact(
        self,
        messages: list[LLMMessage],
        *,
        budget: Budget,
        policy: CompactionPolicy,
        state: dict[str, Any],
        **kwargs: Any,
    ) -> CompactionOutcome:
        split = self.splitter.split(
            messages,
            policy=policy,
            estimator=budget.estimator,
            keep_recent_tokens=budget.tail_tokens(policy.keep_recent_tokens),
        )
        if not split.has_middle:
            return CompactionOutcome(messages=messages, changed=False)

        # --- phase one: deterministic, free -------------------------------
        earlier, by_relevance, relevance_ids = _prune_irrelevant_first(
            self.pruner, split.head + split.middle, state
        )
        head_and_middle, by_size, size_ids = self.pruner.prune_reporting(
            earlier, over_chars=policy.prune_tool_results_over_chars
        )
        pruned = by_relevance + by_size
        pruned_ids = relevance_ids + size_ids
        after_pruning = head_and_middle + split.tail
        if budget.fits(after_pruning):
            if not pruned:
                return CompactionOutcome(messages=messages, changed=False)
            return CompactionOutcome(
                messages=after_pruning,
                changed=True,
                note=f"context compacted: {pruned} old tool outputs dropped",
                pruned_event_ids=pruned_ids,
            )

        # --- phase two: ask a model ---------------------------------------
        provider = self.llm_provider or kwargs.get("llm_provider")
        if provider is None:
            # Nothing to summarize with. Returning phase one rather than nothing
            # keeps the space it did recover; the caller reports that the result
            # is still over the limit.
            get_logger().warning("pruning was not enough and no provider is available to summarize")
            return CompactionOutcome(
                messages=after_pruning,
                changed=bool(pruned),
                note="context compacted, but pruning alone was not enough",
                pruned_event_ids=pruned_ids,
            )

        # The last requests are never summarized: the span ends before the first
        # of them. What precedes it can be summarized; it, and what followed it,
        # stay — pruned of old tool output, not summarized.
        kept = {id(m) for m in _last_requests(messages, policy.keep_last_user_messages)}
        stop = next((i for i, m in enumerate(split.middle) if id(m) in kept), len(split.middle))
        span = _summarizable_span(
            split.head, split.middle[:stop], [*split.middle[stop:], *split.tail]
        )
        if span is None:
            return CompactionOutcome(
                messages=after_pruning,
                changed=bool(pruned),
                note="context compacted: no complete block to summarize",
                pruned_event_ids=pruned_ids,
            )
        begin, end = span

        # Summarized from the pruned middle, not the original: the placeholders
        # already say what the dropped output was, and paying for the full text
        # here would defeat the phase that just removed it.
        pruned_middle = head_and_middle[len(split.head) :][begin:end]
        event_ids = _exclusive_event_ids(
            split.middle[begin:end],
            keep=[*split.head, *split.middle[:begin], *split.middle[end:], *split.tail],
        )
        if not event_ids:
            return CompactionOutcome(
                messages=after_pruning,
                changed=bool(pruned),
                note="context compacted: nothing summarizable in the middle",
                pruned_event_ids=pruned_ids,
            )
        phase_one = CompactionOutcome(
            messages=after_pruning,
            changed=bool(pruned),
            note="context compacted, but the summary failed",
            pruned_event_ids=pruned_ids,
        )
        try:
            summary = await self.summarizer.summarize(pruned_middle, llm_provider=provider)
        except Exception as exc:
            # Caught here rather than left to `Compactor.apply`, which discards
            # the whole outcome on an exception: that would throw away the space
            # phase one had already recovered for free. A failed summary means
            # "you get the deterministic part", not "you get nothing".
            get_logger().warning(f"summarization failed: {type(exc).__name__}")
            get_logger().debug(f"summarization failure detail: {exc}")
            return phase_one

        return CompactionOutcome(
            # Built with the very function `replay` uses, so this request and every
            # later re-render of it agree by construction. Dropping the middle
            # wholesale instead looked equivalent and was not: an event rendering
            # on both sides of the cut is excluded from the record, so replay kept
            # its messages while this path removed them — and if one of those was
            # half of a tool-call pair, only the immediate request was malformed.
            messages=_restore_summary(after_pruning, summary, set(event_ids)),
            changed=True,
            note=f"context compacted: {len(event_ids)} exchanges summarized",
            pruned_event_ids=pruned_ids,
            summary=summary,
            summarized_event_ids=event_ids,
        )


def _restore_summary(
    messages: list[LLMMessage], summary: str, event_ids: set[str]
) -> list[LLMMessage]:
    """Put the summary back where the events it replaced used to be.

    In their place rather than at the front: order is what tells the model what
    came before what, and a summary of the middle parked above the system prompt
    would read as instructions.

    Messages carrying no `source_event_id` are always kept. The system prompt is
    the one that matters — it has no originating event, and dropping it would be
    a different failure entirely.
    """
    out: list[LLMMessage] = []
    inserted = False
    for message in messages:
        source = message.source_event_id
        if source and source in event_ids:
            if not inserted:
                out.append(_summary_message(summary))
                inserted = True
            continue
        out.append(message)
    return out


def _summary_message(summary: str) -> LLMMessage:
    """The summary as a message.

    Sent with the `user` role because that is the only role every chat template
    accepts mid-conversation: a `system` message in the middle is dropped or
    rejected by some servers, and putting words in the assistant's mouth invites
    it to treat a reconstruction as something it actually said. The marker makes
    the difference explicit instead of relying on the role to carry it.
    """
    return LLMMessage(role="user", content=f"{SUMMARY_MARKER}\n{summary}")


def _clean_boundary(sequence: list[LLMMessage], index: int) -> bool:
    """Whether a cut at `index` separates nothing that must stay together.

    One rule, and it is the only hard one the wire protocol imposes: an assistant
    message carrying `tool_calls` must be followed by the `tool` results for every
    one of them. Providers do not degrade on a violation, they reject the request.

    So a boundary is clean when it does not fall immediately after a message that
    made calls, and does not fall immediately before a result.
    """
    if index > 0 and _carries_tool_calls(sequence[index - 1]):
        return False
    return not (index < len(sequence) and _is_tool_result(sequence[index]))


def _last_requests(messages: list[LLMMessage], count: int) -> list[LLMMessage]:
    """The last `count` user messages that are requests: not a tool result, not
    a summary standing in for older history."""
    if count <= 0:
        return []
    requests = [
        message
        for message in messages
        if message.role == "user"
        and not _is_tool_result(message)
        and not (isinstance(message.content, str) and message.content.startswith(SUMMARY_MARKER))
    ]
    return requests[-count:]


def _summarizable_span(
    head: list[LLMMessage], middle: list[LLMMessage], tail: list[LLMMessage]
) -> tuple[int, int] | None:
    """The widest stretch of the middle that can be replaced, or `None`.

    Both ends are placed on clean boundaries, which is what makes correctness
    structural rather than something a test has to catch. It also means the span
    is made of complete blocks: it can begin at a user turn or at an assistant
    message, and it ends after a reply that is finished — so replacing it leaves
    the surrounding sequence reading exactly as it did.

    Deliberately not "whole user turns", which was the first rule tried here: it
    refused to summarize a conversation with a single long exchange in it, which
    is precisely a case worth compacting. Cutting on call boundaries admits that
    case and gives up nothing, because a completed reply is a boundary too.
    """
    whole = [*head, *middle, *tail]
    offset = len(head)

    begin = next(
        (index for index in range(len(middle)) if _clean_boundary(whole, offset + index)),
        None,
    )
    if begin is None:
        return None

    end = next(
        (
            index
            for index in range(len(middle), begin, -1)
            if _clean_boundary(whole, offset + index)
        ),
        None,
    )
    return (begin, end) if end is not None and end > begin else None


def _exclusive_event_ids(middle: list[LLMMessage], *, keep: list[LLMMessage]) -> list[str]:
    """Event ids present in the middle and nowhere else.

    One event can render into several messages, and a cut can land between them.
    Dropping such an event by id would take a message out of the protected head
    or tail with it — and if that message is half of a tool-call pair, the request
    becomes malformed. Excluding shared events costs a little compression and
    removes the failure entirely.
    """
    kept = {message.source_event_id for message in keep if message.source_event_id}
    seen: list[str] = []
    for message in middle:
        source = message.source_event_id
        if source and source not in kept and source not in seen:
            seen.append(source)
    return seen


STRATEGIES: dict[str, type[CompactionStrategy]] = {
    NoOpCompaction.type: NoOpCompaction,
    PruneOnlyCompaction.type: PruneOnlyCompaction,
    TwoPhaseCompaction.type: TwoPhaseCompaction,
}
"""Known strategies by name.

An open registry: whoever writes one adds it here (or via the decorator) and
names it in configuration, without touching the planner or the six context
managers.
"""


def compaction_strategy(name: str) -> Any:
    """Register a strategy under a name usable in configuration."""

    def register(cls: type[CompactionStrategy]) -> type[CompactionStrategy]:
        STRATEGIES[name] = cls
        return cls

    return register


def build_strategy(name: str, **kwargs: Any) -> CompactionStrategy:
    """Instantiate a strategy by name.

    An unknown name **raises**: silently falling back to `noop` would give
    someone who mistyped a line of configuration the illusion of an active
    compaction.
    """
    try:
        cls = STRATEGIES[name]
    except KeyError:
        known = ", ".join(sorted(STRATEGIES))
        raise ValueError(f"unknown compaction strategy {name!r}; known: {known}") from None
    return cls(**kwargs)


class CompactionConfig(BaseModel):
    """How the session shortens itself when it approaches the model's limit."""

    strategy: str = Field(
        default="two_phase",
        description=(
            "`two_phase` (prune, then summarize if pruning was not enough), "
            "`prune_only` (never calls a model), or `noop` (do nothing and let "
            "the request be rejected). Default is `two_phase` because it *is* "
            "`prune_only` plus a fallback that only runs once the free phase has "
            "demonstrably failed — at which point the alternative is a lost turn."
        ),
    )
    reserve_tokens: int | None = Field(
        default=None,
        description="Headroom kept below the window. `None` uses the framework default.",
    )
    keep_recent_tokens: int | None = Field(
        default=None,
        description="Recent conversation left untouched. `None` uses the framework default.",
    )


def build_compactor(config: CompactionConfig) -> Compactor:
    """Build the compactor described by the config.

    An unknown strategy name falls back to the default with a warning rather than
    killing the session: a typo in one configuration line should not stop someone
    from working, and the warning is what gets it fixed.
    """
    try:
        strategy = build_strategy(config.strategy)
    except ValueError as exc:
        get_logger().warning(f"{exc}; using the default compaction strategy instead")
        strategy = build_strategy(CompactionConfig().strategy)

    # Built field by field rather than from a dict of overrides: `**kwargs` into a
    # model whose first field is a bool defeats type checking on every other one.
    policy = CompactionPolicy()
    if config.reserve_tokens is not None:
        policy.reserve_tokens = config.reserve_tokens
    if config.keep_recent_tokens is not None:
        policy.keep_recent_tokens = config.keep_recent_tokens
    return Compactor(strategy=strategy, policy=policy)
