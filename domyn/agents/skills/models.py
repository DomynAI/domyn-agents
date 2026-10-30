"""Skill data model.

Three levels of disclosure, mirroring what Claude Code, Hermes and pi converged
on independently:

* **L0** — :class:`SkillDescriptor`: name and description only. Always in context,
  cheap enough that dozens of skills cost a few hundred tokens.
* **L1** — :class:`SkillContent` with ``section=None``: the full body, loaded when
  a skill is actually needed.
* **L2** — :class:`SkillContent` with a ``section``: one addressable slice.

``description`` is required, not optional. Without it there is no index, and
without an index there is no progressive disclosure — a skill nobody can see the
purpose of is a skill the model cannot choose. This is also what makes skills
portable to other agentskills.io-compatible tools.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# Enforcement decides what happens when a skill cannot be loaded.
#
# ``advisory`` skills are enhancements: losing one degrades quality, so the turn
# proceeds with a warning. ``required`` skills are guardrails — compliance rules,
# constraints on what may be said. Running without one is worse than failing, so
# the turn fails loudly.
Enforcement = Literal["advisory", "required"]


class SkillRequirements(BaseModel):
    """Preconditions for a skill to be usable.

    Evaluated locally (binaries, env vars, platform, files) or declared by a remote
    source. A skill whose requirements are unmet stays visible but is marked
    unavailable: hiding it silently makes debugging impossible.
    """

    bins: list[str] = Field(default_factory=list)
    env: list[str] = Field(default_factory=list)
    os: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    external_files: list[str] = Field(default_factory=list)
    """Files the skill needs besides its ``SKILL.md``, as written: paths relative to it.
    External to the file, not to the folder: ``scripts/`` is an entry like
    ``../shared.py``. Unlike the other four they are checked only for a skill with a
    ``path``, so a remote source may declare them but nothing checks them there.
    ``domyn setup`` copies them for a built-in skill. Shown to the model only as the
    reason a skill is unavailable."""

    def is_empty(self) -> bool:
        return not (self.bins or self.env or self.os or self.tools or self.external_files)


class SkillRef(BaseModel):
    """Points at a skill, optionally at one section of it.

    Carries no source identifier: a single source is active per agent, so there
    is nothing to disambiguate. If several sources are ever combined, that is
    implemented as a composite source which resolves refs internally — the ref
    shape does not need to change.
    """

    name: str
    section: str | None = None


class SkillDescriptor(BaseModel):
    """L0 entry: what the model sees before deciding to open a skill."""

    name: str
    description: str
    version: str | None = None
    etag: str | None = None
    source: str = "unknown"
    sections: list[str] = Field(default_factory=list)
    requires: SkillRequirements = Field(default_factory=SkillRequirements)
    enforcement: Enforcement = "advisory"
    available: bool = True
    unavailable_reason: str | None = None
    path: Path | None = None

    def index_line(
        self, *, sections: bool = False, max_description_chars: int | None = None
    ) -> str:
        """One-line form used to build the index handed to the model.

        ``sections`` appends the ``##`` headings. Off by default: the list is
        often the longest part of the line, it is paid for on every request of
        every turn, and the model sees the headings anyway once it opens the
        skill — which is when it can use them.

        ``max_description_chars`` shortens the description, for an index that
        does not fit its budget with the whole of every one.

        The file's location is not part of it. It was, and it was half of every
        line; it is given where it is used instead — with the skill's body, by
        ``skill_view`` and by expansion (:meth:`SkillContent.header`).
        """
        description = self.description
        if max_description_chars is not None:
            description = clip_text(description, max_description_chars)
        suffix = ""
        if not self.available:
            reason = self.unavailable_reason or "unavailable"
            suffix = f" [unavailable: {reason}]"
        elif sections and self.sections:
            suffix = f" [sections: {', '.join(self.sections)}]"
        return f"- {self.name}: {description}{suffix}"


def clip_text(text: str, limit: int) -> str:
    """``text`` cut to at most ``limit`` characters, at a word boundary, marked with ``…``.

    Cut at the last space when there is one in the second half of the allowance,
    so a word is not split; otherwise mid-word, since one long token would leave
    nothing at all.
    """
    if len(text) <= limit:
        return text
    if limit < 1:
        return ""
    cut = text[: limit - 1]
    space = cut.rfind(" ")
    if space >= limit // 2:
        cut = cut[:space]
    return cut.rstrip(" ,;:.-") + "…"


class SkillContent(BaseModel):
    """L1/L2 payload: the body, or one section of it."""

    name: str
    body: str
    version: str | None = None
    section: str | None = None
    partial: bool = False
    path: Path | None = None

    def header(self) -> str:
        """The lines that introduce the body, wherever a body reaches the model.

        Carries the file's location, which the index no longer does. Without it an
        agent asked to change a skill has to guess where it lives, and what it
        guesses is the current directory — wrong whenever skills come from anywhere
        else. Nor is it derivable from the name: both ``<dir>/<name>/SKILL.md`` and
        ``<dir>/<name>.md`` are supported layouts. Builtin skills also name their
        files relative to it.
        """
        header = f"Skill: {self.name}"
        if self.section:
            header += f" (section: {self.section})"
        if self.path is not None:
            header += (
                f"\n[file: {self.path}] — a real file you can read and edit with your file"
                " tools when asked to change this skill."
            )
        return header


class SkillIndex(BaseModel):
    """Result of an ``index()`` call: every skill the source has, budget or not.

    The budget is applied where the index is rendered for the model
    (:meth:`SkillsRuntime.layout`), never here, so a selector, ``skill_view`` and
    ``/skills`` all see the whole catalogue.
    """

    skills: list[SkillDescriptor] = Field(default_factory=list)

    def by_name(self, name: str) -> SkillDescriptor | None:
        for skill in self.skills:
            if skill.name == name:
                return skill
        return None


class SkillQuery(BaseModel):
    """What a source is told about the situation.

    ``user_id`` plus an open-ended ``context`` bag: enough for per-user filtering
    today, extensible to roles or tenants later without changing the interface.
    The bag is expected to be pre-filtered by the caller according to the
    visibility rules — a source must never be handed data it is not cleared for.
    """

    user_id: str | None = None
    agent_name: str | None = None
    task: str | None = None
    available_tools: list[str] = Field(default_factory=list)
    context: dict[str, object] = Field(default_factory=dict)
    max_skills: int | None = None


class Selection(BaseModel):
    """What the selector decided for this turn.

    ``index`` is what the model can see; ``expand`` is what gets its body pulled
    in immediately. Keeping them separate is what makes progressive disclosure a
    policy choice rather than a hardcoded behaviour.
    """

    index: list[SkillDescriptor] = Field(default_factory=list)
    expand: list[SkillRef] = Field(default_factory=list)
    rationale: str | None = None


class LoadedSkills(BaseModel):
    """Everything a source could hand over up front, for one turn.

    ``bodies`` is keyed by the *parsed* descriptor name — the key
    :meth:`SkillIndex.by_name` and :attr:`SkillRef.name` already use, never a
    filename or a mapping key. The two differ whenever frontmatter declares a
    ``name:``, and a body filed under the wrong one is a body nobody can find.

    Partial by design: a source whose bodies cost a round-trip each returns none
    of them, and expansion falls back to ``fetch``. An empty instance is also
    what a failed load leaves behind — which is why "loaded, and there was
    nothing" and "nobody loaded" must stay distinguishable, the latter being
    ``None``.
    """

    index: SkillIndex = Field(default_factory=SkillIndex)
    bodies: dict[str, SkillContent] = Field(default_factory=dict)

    def body(self, name: str) -> SkillContent | None:
        return self.bodies.get(name)
