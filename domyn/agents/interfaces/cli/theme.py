"""Visual theme for the interactive CLI.

The palette is **semantic, not decorative**: every colour carries meaning about
the kind of event being rendered, so a user can parse a transcript at a glance.

Two deliberate choices:

* The agent's own answer is rendered **without colour**, but for one tint on
  the code and links inside it and the syntax colours of its code blocks. It is
  the content the user actually came for; colouring it would make it compete
  with the trace. Code and links are the exception because they are what the
  eye goes looking for in it.
* Every colour is picked to stay legible on **both light and dark** terminals.
  That rules out bright yellow, plain white and plain black, which each vanish
  against one of the two backgrounds.

Capability detection is centralised here (``Capabilities``) so no other module
has to guess whether it may emit escape codes or Unicode.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar


@dataclass(frozen=True)
class Capabilities:
    """What the current terminal can actually do.

    Resolved once at startup. ``color`` and ``animation`` are independent: a
    redirected stream keeps colour off *and* animation off, but a user may keep
    colour while disabling animation via configuration.
    """

    color: bool
    unicode: bool
    animation: bool

    @classmethod
    def detect(cls, *, stream: object | None = None, allow_animation: bool = True) -> Capabilities:
        out = stream if stream is not None else sys.stdout

        is_tty = bool(getattr(out, "isatty", lambda: False)())
        term = os.environ.get("TERM", "")
        dumb = term in ("", "dumb")

        # NO_COLOR is honoured whenever it is set to anything at all, per the
        # no-color.org convention.
        no_color = "NO_COLOR" in os.environ

        color = is_tty and not dumb and not no_color

        encoding = (getattr(out, "encoding", None) or "").lower()
        unicode_ok = "utf" in encoding

        return cls(
            color=color,
            unicode=unicode_ok,
            animation=allow_animation and color,
        )


class Symbols:
    """Glyphs with ASCII fallbacks.

    Chosen so the fallback keeps the same visual grammar (direction and
    nesting) rather than degrading to interchangeable punctuation.
    """

    def __init__(self, *, unicode: bool) -> None:
        if unicode:
            self.tool_call = "→"
            self.tool_result = "←"
            self.handoff_in = "↪"
            self.handoff_out = "↩"
            self.approved = "✓"
            self.denied = "✗"
            self.prompt = "\u203a"  # single right-pointing angle quote
            self.agent = "\u23fa"  # black circle for record: the agent speaking
            self.bullet = "•"
            self.rule = "─"
            self.quote = "│"
            self.separator = "·"
            self.auto_mode = "◆"
        else:
            self.tool_call = "->"
            self.tool_result = "<-"
            self.handoff_in = ">>"
            self.handoff_out = "<<"
            self.approved = "[ok]"
            self.denied = "[no]"
            self.prompt = ">"
            self.agent = "*"
            self.bullet = "*"
            self.rule = "-"
            self.quote = "|"
            self.separator = "-"
            self.auto_mode = "!"


class Styles:
    """Rich style strings, one per semantic role.

    Keeping them in one place means a future ``--theme`` flag only has to swap
    this object, and no call site changes.
    """

    # Brand accent: a real tint sampled from the logo gradient, which runs from
    # a deep violet (#210d6e) to a light blue (#36b7ff).
    #
    # This one is picked by measuring, not by eye. Of the 220 tints in the
    # gradient it is the one with the best *worst-case* contrast — 4.7:1 on
    # black, 4.5:1 on white — so it holds up on either background. The cyan it
    # replaces managed 7.8:1 on black but only 2.7:1 on white, which is why it
    # looked washed out in a light terminal. Terminals without truecolor
    # downgrade it to their nearest blue, which is still the right hue.
    brand = "#1f6dff"
    accent = brand
    accent_bold = f"bold {brand}"

    # The answer itself: no colour on purpose (see module docstring).
    response = ""

    # Reasoning is secondary to the answer, never competing with it.
    thought = "dim italic"

    # Tools: the call is the actionable part, the result is supporting detail.
    tool_call = "yellow"
    tool_name = "bold yellow"
    tool_result = "dim"
    # An edit waiting for approval: named colours, as the syntax colours below, so
    # the terminal's palette keeps them legible; the `-`/`+` marks carry the same
    # meaning without colour.
    diff_removed = "red"
    diff_added = "green"
    diff_kept = "dim"

    # Control flow between agents.
    handoff = "magenta"

    # Diagnostics.
    #
    # Something made that the user acts on next (a new project). The green of
    # `md_accent`, measured to hold 4.5:1 on black and on white; the brand blue
    # reads too dark on some dark palettes for a line that has to be noticed.
    success = "bold #00875f"
    error = "bold red"
    warning = "yellow"
    # Between the two: nothing failed, but what follows may not work as it did
    # where it was made (a project exported by another version). The red of
    # `diff_removed`, not bold, so it is not read as an error.
    caution = "red"
    info = "dim"

    # Anything awaiting a human decision must be impossible to miss.
    approval = "bold yellow"
    approval_border = "yellow"

    # Auto-approve: orange, so "nobody is being asked" never reads as the
    # yellow of "someone is being asked". xterm 166 exactly, so a 256-colour
    # terminal draws the same hue, and 5.5:1 on black, 3.8:1 on white — enough
    # for the bold text it is used for on either background.
    auto = "#d75f00"
    auto_mode = f"bold {auto}"

    # A decision model changing something about the turn. Green and red for the
    # two verdicts that act on the approval gate, the same pair and the same
    # reason as `diff_added`/`diff_removed`: named ANSI colours, so the
    # terminal's own palette picks a shade that stays legible on light and dark
    # alike. Nothing else in the trace is green, so the hue is free to mean
    # "this was let through without asking".
    #
    # These are *not* `meta`: a decision that acted is the only record that the
    # gate behaved differently from its default, and it is the line a reader
    # goes looking for after "why didn't it ask me". Dimmed, it reads as
    # background noise — which is exactly what it was reported as being.
    decision = "bold"
    decision_allow = "green"
    decision_deny = "red"

    # Chrome.
    #
    # The user's prompt carries the brand colour, and it is the only thing that
    # does inside a transcript. That makes it the anchor the eye scrolls back
    # to: everything between two blue markers is one exchange.
    user_prompt = f"bold {brand}"
    # The marker in front of the agent's answer. Same hue as the prompt so the
    # pair reads as one conversation, not bold so it does not compete with it.
    agent_marker = brand
    rule = "dim"
    meta = "dim"
    # Names in a listing: what the eye scans for, against dim detail.
    emphasis = "bold"

    # Markdown inside the answer.
    #
    # Structure is carried by attributes, not hue: the answer stays uncoloured
    # (see the module docstring), and every hue of the trace already means
    # something. The exception is one tint for the spans the eye goes looking
    # for — code and links — measured the way `brand` was: #00875f holds 4.6:1
    # on black and 4.5:1 on white, the only green or teal of the xterm-256 cube
    # to clear 4.5:1 on both. Being in the cube, it reaches a terminal without
    # truecolor unchanged.
    md_accent = "#00875f"
    md_code = md_accent
    md_link = f"underline {md_accent}"
    md_link_url = "dim"
    md_title = "bold underline"
    md_heading = "bold"
    md_quote = "italic"
    # Bullets, numbers, quote bars, code fences, rules, table borders: the
    # scaffolding, dimmed so that what it holds is what reads.
    md_marker = "dim"

    # Syntax colours in code blocks, by Pygments token type (a parent type covers
    # its children). Named ANSI colours, so the terminal's own palette picks the
    # shade, which is what keeps them legible on a light background and a dark
    # one alike. No bright variants, no yellow and no blue: those are the ones
    # that wash out on white or sink into black.
    syntax: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "Comment": "dim italic",
            "Keyword": "magenta",
            "Operator.Word": "magenta",
            "Name.Builtin": "cyan",
            "Name.Function": "bold",
            "Name.Class": "bold",
            "Name.Decorator": "magenta",
            "Name.Exception": "bold",
            "Name.Tag": "cyan",
            "Name.Attribute": "cyan",
            "Name.Variable": "cyan",
            "Literal.String": "green",
            "Literal.Number": "cyan",
            "Generic.Inserted": "green",
            "Generic.Deleted": "red",
            "Generic.Heading": "bold",
            "Generic.Subheading": "bold",
            "Generic.Prompt": "dim",
        }
    )


@dataclass(frozen=True)
class Theme:
    """Bundle of capabilities, symbols and styles handed to the renderer."""

    capabilities: Capabilities
    symbols: Symbols
    styles: type[Styles]

    @classmethod
    def build(cls, *, allow_animation: bool = True, stream: object | None = None) -> Theme:
        caps = Capabilities.detect(stream=stream, allow_animation=allow_animation)
        return cls(
            capabilities=caps,
            symbols=Symbols(unicode=caps.unicode),
            styles=Styles,
        )

    def style(self, name: str) -> str:
        """Return a style string, or "" when colour is disabled.

        Call sites can pass the result straight to rich; an empty style is a
        no-op, so no branching is needed at the call site.
        """
        if not self.capabilities.color:
            return ""
        return str(getattr(self.styles, name, ""))
