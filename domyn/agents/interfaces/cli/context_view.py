"""Showing how much context the session is using.

Two different things, and the second is the one that actually gets used.

``render`` answers an explicit question (``/context``): the breakdown of the last
request that went out. ``indicator`` answers nothing — it is the line you see
without asking, and its job is to be noticed **before** something happens, not to
explain what happened afterwards.

The constraint that shapes the indicator: in a fifty-turn session anything
printed every turn becomes noise and stops being read. So it stays quiet until
there is something to say, and when it speaks it says one thing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from domyn.agents.planner_strategy.context_management.budget import ContextUsage

if TYPE_CHECKING:
    from pathlib import Path

    from rich.console import Console

    from domyn.agents.interfaces.cli.theme import Theme

# Below this share the indicator stays quiet: the session has room, and saying so
# every turn would be noise. The value is chosen to leave time to react —
# noticing at 95% is noticing too late.
ATTENTION_FRACTION = 0.70

# Above this share the tone changes: it is no longer information, it is a warning.
WARNING_FRACTION = 0.90


def _thousands(tokens: int) -> str:
    """Tokens in readable form: nobody reads 184213 as a quantity."""
    if tokens < 1_000:
        return str(tokens)
    return f"{tokens / 1_000:.1f}k".replace(".0k", "k")


def indicator(usage: ContextUsage | None) -> tuple[str, str] | None:
    """``(text, style name)`` to show without anyone asking for it.

    Returns the style *name* rather than a resolved style: semantics are decided
    here ("this is a warning" vs "this is information"), appearance is decided by
    whoever prints. That way this function needs no theme, and the difference
    between informing and warning stays visible to a colourless theme too — which
    is to say, testable.

    Returns ``None`` in three cases, all of them "nothing to say": no request has
    gone out yet, the model window is unknown, or the session is still
    comfortably below the attention threshold.
    """
    if usage is None:
        return None

    fraction = usage.fraction_used
    if fraction is None:
        # Unknown window: showing an invented percentage would be worse than
        # saying nothing, because it would look like data.
        return None
    if fraction < ATTENTION_FRACTION:
        return None

    share = f"{fraction:.0%}"

    remaining = usage.tokens_until_threshold
    if remaining is not None:
        # A compaction will happen: the useful thing is the distance to it.
        text = f"context {share} · compaction in ~{_thousands(remaining)} tokens"
    else:
        # No compaction coming, so no distance to promise. The share remains,
        # which is what needs to be known anyway: this window will run out and
        # the request will be rejected.
        text = f"context {share} of {_thousands(usage.context_window or 0)}"

    return text, "warning" if fraction >= WARNING_FRACTION else "meta"


def render(
    console: Console,
    theme: Theme,
    usage: ContextUsage | None,
    *,
    paths: dict[str, Path] | None = None,
) -> None:
    """The breakdown, on request.

    Tool definitions are a separate line from the conversation because they
    travel in a different request parameter, they are paid on every turn from the
    first, and **no compaction can reduce them**. Folding them in would hide the
    one cost that cannot be compressed — and it is the number that says whether
    it is worth working on the tools instead of the conversation.
    """
    meta = theme.style("meta")

    if usage is None:
        console.print(
            "  No request in this conversation yet, so nothing to measure.",
            style=meta,
            highlight=False,
        )
        return

    console.print()
    console.print("  context of the last request", style=theme.style("accent_bold"))

    rows = [
        ("conversation", usage.message_tokens),
        ("tool definitions", usage.tool_definition_tokens),
    ]
    # Shares of the estimate: with a measured correction the total is no longer
    # the sum of these two rows, and a share of it would read past 100%.
    estimated = usage.message_tokens + usage.tool_definition_tokens
    for label, tokens in rows:
        share = f"{tokens / estimated:.0%}" if estimated else "—"
        console.print(
            f"   {label:<22} {_thousands(tokens):>7}  {share:>4}",
            style=meta,
            highlight=False,
        )
    if usage.correction_tokens:
        # The provider's count of the last request against our estimate of it:
        # the rows above are estimates, the total is anchored to the count.
        sign = "+" if usage.correction_tokens > 0 else "-"
        console.print(
            f"   {'measured correction':<22} {sign + _thousands(abs(usage.correction_tokens)):>7}",
            style=meta,
            highlight=False,
        )
    console.print(
        f"   {'total':<22} {_thousands(usage.total_tokens):>7}",
        style=meta,
        highlight=False,
    )

    if usage.context_window:
        fraction = usage.fraction_used or 0.0
        console.print(
            f"   {'model window':<22} {_thousands(usage.context_window):>7}  {fraction:>4.0%} used",
            style=theme.style("warning" if fraction >= WARNING_FRACTION else meta),
            highlight=False,
        )
        remaining = usage.tokens_until_threshold
        if remaining is not None:
            console.print(
                f"   {'until compaction':<22} {_thousands(remaining):>7}   ({usage.strategy})",
                style=meta,
                highlight=False,
            )
        else:
            # The field the user came looking for. Staying silent would leave
            # them wondering whether the number is missing or the compaction is:
            # two different situations, one of which is fixed by a line of
            # configuration.
            console.print(
                f"   {'until compaction':<22} {'—':>7}   no compaction configured",
                style=meta,
                highlight=False,
            )
    else:
        # Short lines, each with its own indent: one long sentence wrapping on
        # its own loses the indentation and aligns to the margin, which on a
        # narrow terminal reads as a printing error.
        for line in (
            "",
            "  Model window unknown: not declared, and",
            "  the provider does not report it, so there",
            "  is no way to say how much room is left.",
            "  Set `context_window` on the provider.",
        ):
            console.print(line, style=meta, highlight=False, no_wrap=True, crop=False)

    if usage.tool_definition_tokens > usage.message_tokens:
        # Worth saying because it is counter-intuitive and changes where it is
        # worth intervening.
        console.print(
            "\n  Tool definitions outweigh the conversation: they are a fixed cost"
            "\n  of every request, and no compaction reduces them.",
            style=theme.style("warning"),
            highlight=False,
        )
    console.print()
