"""Skills runtime: ties source, selector and budgets together.

This is the object a front-end or an agent hook holds. It owns the two things
that must not be scattered around: the token budget, and the rule that dropping
something is always reported.

Nothing here mutates the ``Agent``. A single ``Agent`` instance is shared across
concurrent conversations, so per-session skill state kept on it would leak
between users. State belongs to the caller (the REPL, or the interaction
context).
"""

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from domyn.agents.logger import get_logger
from domyn.agents.skills.models import (
    LoadedSkills,
    Selection,
    SkillContent,
    SkillDescriptor,
    SkillIndex,
    SkillQuery,
    SkillRef,
)
from domyn.agents.skills.search import SearchResult, SkillSearch
from domyn.agents.skills.selectors import IndexSelector, SkillSelector
from domyn.agents.skills.sources import SkillNotFoundError, SkillSource, narrow_to_section

if TYPE_CHECKING:
    from domyn.agents.llm import BaseLLMProvider


def _resolve_by_type(value: Any, base: Any, label: str) -> Any:
    """Turn a ``{"type": ..., ...}`` mapping into the registered subclass.

    Raising on an unknown type — rather than falling back to a default — is
    deliberate: a typo in ``type:`` would otherwise produce an agent that runs
    with silently different behaviour from the one that was configured.

    Anything that is not a mapping is returned untouched for pydantic to judge:
    the callers are ``mode="before"`` validators, so the field's own type check
    still runs on the result — an instance passes, ``source: local_dir`` is
    rejected at load time rather than stored and failing on first use.
    """
    if not isinstance(value, dict):
        return value
    tag = value.get("type")
    if not tag:
        raise ValueError(f"{label} configuration is missing 'type'")
    subclass = base._registry.get(tag)
    if subclass is None:
        known = ", ".join(sorted(base._registry)) or "none registered"
        raise ValueError(f"unknown {label} type '{tag}' (known: {known})")
    return subclass.model_validate(value)


class SkillRequiredError(RuntimeError):
    """A skill marked ``required`` could not be loaded.

    Raised instead of degrading: a required skill is a guardrail, and running
    without a guardrail is worse than not running at all.
    """


def raise_if_required_unavailable(index: SkillIndex, *, agent_name: str | None = None) -> None:
    """Refuse an index that advertises a ``required`` skill it cannot serve.

    Checked on the **index**, not at expansion: every selector drops an
    unavailable skill before it can be expanded — ``AllSkillsSelector`` filters
    on ``available``, ``LLMSelector`` routes only over available candidates, and
    ``IndexSelector`` expands nothing at all — so a check downstream of selection
    is a check that never fires. The descriptor is also the only place carrying
    both the enforcement and the reason.

    ``agent_name`` is optional because callers know different amounts: the runner
    loads per agent and always has one, a hand-built ``SkillQuery`` may not.
    """
    blocked = [
        skill for skill in index.skills if skill.enforcement == "required" and not skill.available
    ]
    if not blocked:
        return

    whose = f" for agent '{agent_name}'" if agent_name else ""
    # Same "(unavailable: <reason>)" shape the index line and /skills already
    # use, so the refusal reads as the same fact the catalogue was showing.
    detail = "; ".join(
        f"{skill.name} (unavailable: {skill.unavailable_reason or 'requirements unmet'})"
        for skill in blocked
    )
    raise SkillRequiredError(f"required skill(s) unavailable{whose}: {detail}")


class SkillBudget(BaseModel):
    """Limits on how much skill material may enter the context.

    Defaults are deliberately modest. An index over ``max_index_chars`` is shown
    in a shorter form rather than cut (see :meth:`SkillsRuntime.layout`), and
    whatever still does not fit is counted in the index itself — a silently
    trimmed index is the difference between "the skill did not apply" and "the
    skill was never shown", and those need different fixes.
    """

    max_index_chars: int = 10_000
    max_description_chars: int = 120
    """How long a description may be once the index is shortened to fit."""
    max_body_chars: int = 20_000
    max_expansions_per_turn: int = 5


IndexTier = Literal["full", "short", "names"]
"""The form advisory skills take in the index, in order of preference: whole
descriptions, descriptions cut to ``max_description_chars``, names only."""


class IndexLayout(BaseModel):
    """How an index is shown within its budget.

    One tier for every advisory skill, never a mix: which skill kept its
    description would otherwise come down to where its file sorts. ``required``
    skills are always shown whole, whatever the tier.
    """

    tier: IndexTier = "full"
    listed: list[SkillDescriptor] = Field(default_factory=list)
    """Every skill shown, in index order."""
    omitted: list[str] = Field(default_factory=list)
    """Advisory skills that did not fit even by name. Counted in the index, so the
    model knows they exist."""

    @property
    def shortened(self) -> bool:
        return self.tier != "full" or bool(self.omitted)


class SkillsRuntime(BaseModel):
    """Resolves skills for a turn.

    A pydantic model rather than a plain dataclass so it can be an ``Agent`` field
    and be built straight from YAML, with ``source`` and ``selector`` resolved by
    their ``type`` tag the same way memory managers and tools are.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: SkillSource
    selector: SkillSelector = Field(default_factory=IndexSelector)
    budget: SkillBudget = Field(default_factory=SkillBudget)
    search: SkillSearch = Field(default_factory=SkillSearch)
    """How ``skill_search`` searches when the source does not."""
    index_sections: bool = False
    """Whether each index line lists the skill's sections. Off by default: see
    :meth:`SkillDescriptor.index_line`. Turn it on when the model should be able to
    ask for one section of a skill without opening the whole file first."""

    @field_validator("source", mode="before")
    @classmethod
    def _resolve_source(cls, value: Any) -> Any:
        return _resolve_by_type(value, SkillSource, "skill source")

    @field_validator("selector", mode="before")
    @classmethod
    def _resolve_selector(cls, value: Any) -> Any:
        return _resolve_by_type(value, SkillSelector, "skill selector")

    async def load(self, query: SkillQuery) -> LoadedSkills:
        """Everything the source can give up front: the whole catalogue.

        This is the half that does I/O, and the reason it is separate: the runner
        calls it **once per turn**, while :meth:`select` runs once per agent
        entry.

        No budget here: it is applied where the index is rendered
        (:meth:`build_payload`). Applied here, it hid skills from the selector and
        from ``skill_view`` as well as from the index, so a router could never pick
        a skill that did not fit.

        Bodies come back only from a source that can produce them cheaply. Every
        local one can — it parses them to build the index anyway — while a source
        that would need a round-trip per body returns none, and expansion fetches
        what it needs.
        """
        return await self.source.load(query)

    async def index(self, query: SkillQuery) -> SkillIndex:
        """L0 index: the whole catalogue, budget not applied."""
        return await self.source.index(query)

    async def fetch(self, ref: SkillRef, query: SkillQuery) -> SkillContent:
        """L1/L2 content for one skill."""
        return await self.source.fetch(ref, query)

    def bind_llm(self, provider: "BaseLLMProvider | None") -> None:
        """Offer the owning agent's provider to everything here that may ask a model."""
        self.selector.bind_llm(provider)
        self.search.bind_llm(provider)

    async def find(
        self, text: str, limit: int, loaded: LoadedSkills, query: SkillQuery
    ) -> SearchResult:
        """Search the whole catalogue: the source's own search, or else :attr:`search`.

        ``loaded`` is this turn's catalogue, budget not applied, so a skill the
        index shows by name only, or not at all, is found like any other.
        """
        hits = await self.source.search(text, limit + 1, query)
        if hits is not None:
            return SearchResult(hits=hits[:limit], more=len(hits) > limit, method="source")
        return await self.search.find(text, loaded.index.skills, loaded.bodies, limit)

    async def resolve(self, query: SkillQuery) -> tuple[Selection, list[SkillContent]]:
        """Full pipeline: load, select, expand."""
        return await self.select(await self.load(query), query)

    async def select(
        self, loaded: LoadedSkills, query: SkillQuery
    ) -> tuple[Selection, list[SkillContent]]:
        """Choose and expand, against skills already loaded for this turn.

        Returns the selection (what the model may see) and the bodies that were
        pulled in. Two things make a ``required`` skill raise here: one whose
        requirements are unmet, refused on the index below before anything is
        selected, and one whose *body* cannot be fetched during expansion. The
        same failures on an ``advisory`` skill are logged and skipped.

        Runs once per agent entry rather than once per turn: the selector routes on
        *this* entry's task, which for a sub-agent is the task it was handed, not
        the text the user typed.
        """
        # Before selection, because selection is where the evidence disappears:
        # every selector drops an unavailable skill, so by the time anything
        # reaches expansion the descriptor is gone and the fetch-failure handler
        # below can never have seen it.
        raise_if_required_unavailable(loaded.index, agent_name=query.agent_name)

        selection = await self.selector.select(loaded.index, query)

        refs = selection.expand[: self.budget.max_expansions_per_turn]
        if len(selection.expand) > len(refs):
            get_logger().warning(
                f"skills: expansion capped at {self.budget.max_expansions_per_turn}, "
                f"skipped {[r.name for r in selection.expand[len(refs) :]]}"
            )

        contents: list[SkillContent] = []
        used = 0
        for ref in refs:
            descriptor = loaded.index.by_name(ref.name)
            try:
                # A preloaded body answers both L1 and L2 with no second read:
                # narrowing to a section is the same slice the source would do.
                preloaded = loaded.body(ref.name)
                content = (
                    narrow_to_section(preloaded, ref)
                    if preloaded is not None
                    else await self.fetch(ref, query)
                )
            except (SkillNotFoundError, OSError) as exc:
                if descriptor is not None and descriptor.enforcement == "required":
                    raise SkillRequiredError(
                        f"required skill '{ref.name}' could not be loaded: {type(exc).__name__}"
                    ) from exc
                get_logger().warning(f"skills: skipping '{ref.name}' — {type(exc).__name__}")
                get_logger().debug(f"skills: fetch detail for '{ref.name}': {exc}")
                continue

            if used + len(content.body) > self.budget.max_body_chars:
                get_logger().warning(
                    f"skills: body budget reached, '{content.name}' not injected "
                    f"({used}/{self.budget.max_body_chars} chars used)"
                )
                continue
            used += len(content.body)
            contents.append(content)

        return selection, contents

    def build_payload(self, selection: Selection, contents: list[SkillContent]) -> str:
        """Render the block handed to the model.

        Delivered as a message rather than folded into the system prompt: it keeps
        the prompt prefix stable across the iterations of a turn.
        """
        blocks: list[str] = []

        if selection.index:
            layout = self.layout(selection.index)
            self._log_layout(layout)
            blocks.append(self.render_index(layout))

        for content in contents:
            blocks.append(f"{content.header()}\n{content.body}")

        return "\n\n".join(blocks)

    def layout(self, skills: list[SkillDescriptor]) -> IndexLayout:
        """Fit ``skills`` into ``max_index_chars``, shortening rather than dropping.

        Advisory skills take the first tier at which all of them fit: whole
        descriptions, then descriptions cut to ``max_description_chars``, then names
        only. Past that, names are listed in index order while they fit and the rest
        are counted, not hidden: the model is told how many more there are.

        A ``required`` skill is always shown whole. Shortening or dropping one
        disarms it: a guardrail the model cannot read is one it does not follow.

        Their cost is **charged before the rest**, not as each one is reached.
        Charging in place would make the result depend on where the required entry
        sorts; pre-charging makes the outcome position-independent. It is charged
        rather than exempted because exempting would make the index unbounded —
        ``max_index_chars`` plus the required total — and a budget that grows with
        its input is not a budget. Over-budget required skills are kept and logged,
        never raised on: this runs on every agent entry, and the refusal belongs to
        the ``raise_if_required_unavailable`` call sites.
        """
        budget = self.budget.max_index_chars
        required = [skill for skill in skills if skill.enforcement == "required"]
        advisory = [skill for skill in skills if skill.enforcement != "required"]

        room = budget - sum(len(self._full_line(skill)) for skill in required)
        if room < 0:
            get_logger().warning(
                f"skills: required skills alone need {budget - room} chars, over the "
                f"index budget of {budget}; kept anyway"
            )

        if not advisory or sum(len(self._full_line(skill)) for skill in advisory) <= room:
            return IndexLayout(tier="full", listed=list(skills))

        short = self.budget.max_description_chars
        if sum(len(skill.index_line(max_description_chars=short)) for skill in advisory) <= room:
            return IndexLayout(tier="short", listed=list(skills))

        named: set[str] = set()
        used = len(_NAMES_HEADER)
        for skill in advisory:
            cost = len(_name_entry(skill)) + len(_NAMES_SEPARATOR)
            if used + cost > room:
                break
            used += cost
            named.add(skill.name)
        listed = [s for s in skills if s.enforcement == "required" or s.name in named]
        omitted = [s.name for s in advisory if s.name not in named]
        return IndexLayout(tier="names", listed=listed, omitted=omitted)

    def render_index(self, layout: IndexLayout) -> str:
        """The index block, in the form ``layout`` decided."""
        header = "Available skills (call skill_view to read one in full):"
        lines: list[str] = []
        names: list[str] = []
        for skill in layout.listed:
            if skill.enforcement == "required" or layout.tier == "full":
                lines.append(self._full_line(skill))
            elif layout.tier == "short":
                lines.append(
                    skill.index_line(max_description_chars=self.budget.max_description_chars)
                )
            else:
                names.append(_name_entry(skill))
        if names:
            lines.append(_NAMES_HEADER + _NAMES_SEPARATOR.join(names))
        if layout.omitted:
            lines.append(
                f"+{len(layout.omitted)} more skills, not listed for lack of space: "
                "find them with skill_search."
            )
        return header + "\n" + "\n".join(lines)

    def _full_line(self, skill: SkillDescriptor) -> str:
        return skill.index_line(sections=self.index_sections)

    @staticmethod
    def _log_layout(layout: IndexLayout) -> None:
        if layout.omitted:
            get_logger().warning(
                f"skills: index budget reached, {len(layout.omitted)} skill(s) not listed: "
                f"{layout.omitted}"
            )
        elif layout.tier != "full":
            get_logger().info(f"skills: index over budget, shown as '{layout.tier}'")


# The names-only tier: one line, so a hundred skills cost a hundred names and
# nothing else. An unavailable skill says so, as the full line does.
_NAMES_HEADER = "Also available, by name only: "
_NAMES_SEPARATOR = ", "


def _name_entry(skill: SkillDescriptor) -> str:
    return skill.name if skill.available else f"{skill.name} (unavailable)"
