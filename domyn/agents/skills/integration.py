"""Wiring skills into an agent without touching the framework core.

The catalogue reaches the model through the planner: the runner loads each
agent's skills once per turn and ``skills.turn.build_skills_payload`` renders
them on every agent entry. This module provides the other half — the
``skill_view`` tool the model calls for a skill's body, the ``skill_search`` tool
it calls to find a skill the index shows by name only or not at all, and
``attach_skills``, which gives an agent a runtime and those tools.

Two constraints discovered while wiring ``skill_view``, both worth knowing before
adding another synthetic tool:

* ``FunctionTool.__init__`` introspects the callable immediately (via
  ``model_post_init`` → ``name`` → ``declaration``), and the introspector cannot
  read **string** annotations. So this module must not use
  ``from __future__ import annotations``, or every tool callable defined here
  fails to construct.
* Pre-setting ``_declaration`` is not a workaround: the whole body of
  ``model_post_init`` is guarded by ``if not self._declaration``, so doing that
  would also skip binding the real callable. The declaration is therefore
  overridden with ``set_declaration`` *after* construction.

``get_tool_definitions`` also injects a mandatory ``reasoning`` parameter into
every tool, so the callable has to accept it.
"""

from typing import Any

from domyn.agents.logger import get_logger
from domyn.agents.skills.models import LoadedSkills, SkillQuery, SkillRef
from domyn.agents.skills.runtime import SkillsRuntime
from domyn.agents.skills.sources import SkillNotFoundError, narrow_to_section
from domyn.agents.tools import FunctionTool
from domyn.agents.tools.base import FunctionDeclaration, Param

_VIEW_DESCRIPTION = (
    "Read one of the available skills in full. The list of skills, with a short "
    "description of each, is provided in the conversation. Call this when a skill "
    "looks relevant to the task, before acting. Pass 'section' to read only one "
    "section of a long skill."
)


def build_skill_view_tool(
    runtime: SkillsRuntime,
    query_factory: Any = None,
) -> FunctionTool:
    """Build the ``skill_view`` tool bound to a runtime.

    Args:
        runtime: the skills runtime to read from.
        query_factory: optional zero-argument callable returning a
            :class:`SkillQuery`; lets a caller pass user identity through to the
            source. Defaults to an empty query.
    """

    async def skill_view(
        name: str, section: str = "", reasoning: str = "", loaded_skills: LoadedSkills | None = None
    ) -> str:
        """Return a skill body, or a readable error the model can act on.

        Async so the fallback fetch runs on the runner's own loop. A sync tool would have to
        bridge with ``asyncio.run`` on a fresh loop, which breaks any source holding loop-bound
        resources — a shared ``httpx.AsyncClient``, a cached task — with "bound to a different
        event loop".
        """

        try:
            skill_body = loaded_skills.body(name=name) if loaded_skills is not None else None
            if skill_body is not None:
                skill_ref = SkillRef(name=name, section=section or None)
                content = narrow_to_section(skill_body, ref=skill_ref)
            else:
                get_logger().debug(f"skill_view: fetching skill '{name}' with section '{section}'")
                query = query_factory() if callable(query_factory) else SkillQuery()
                content = await runtime.fetch(SkillRef(name=name, section=section or None), query)
        except SkillNotFoundError:
            target = f"{name}#{section}" if section else name
            # Returned to the model rather than raised: a missing skill is
            # information it can recover from, not a crash.
            return f"Skill not found: {target}"
        except OSError as exc:
            get_logger().warning(f"skill_view: cannot read '{name}': {type(exc).__name__}")
            get_logger().debug(f"skill_view detail for '{name}': {exc}")
            return f"Skill '{name}' is currently unavailable."

        return f"{content.header()}\n{content.body}"

    # FunctionTool.__init__ is untyped in the framework; the call is correct.
    tool = FunctionTool(func=skill_view, name="skill_view", description="")  # type: ignore[no-untyped-call]
    tool.set_declaration(
        FunctionDeclaration(
            name="skill_view",
            description=_VIEW_DESCRIPTION,
            parameters=[
                Param(name="name", type="str", description="Name of the skill to read."),
                Param(
                    name="section",
                    type="str",
                    description="Optional: read only this section of the skill.",
                    is_required=False,
                    default="",
                ),
            ],
        )
    )
    return tool


_SEARCH_DESCRIPTION = (
    "Search all skills, including those the skills list shows by name only or not at "
    "all. Returns the best matches with what each is for; open one with skill_view. "
    "Write the query as keywords; skills may be in any language, so when the first "
    "search finds nothing, try synonyms or the words in another language."
)

# A search answers with a few lines whatever the catalogue holds: past this, the
# results stop being the best matches and start being a listing.
MAX_SEARCH_RESULTS = 10


def build_skill_search_tool(
    runtime: SkillsRuntime,
    query_factory: Any = None,
) -> FunctionTool:
    """Build the ``skill_search`` tool bound to a runtime.

    It never lists the catalogue: a query is required and the answer is capped at
    :data:`MAX_SEARCH_RESULTS`, so its cost does not grow with the number of
    skills — the problem it exists to get around in the index.
    """

    async def skill_search(
        query: str,
        limit: int = 5,
        reasoning: str = "",
        loaded_skills: LoadedSkills | None = None,
    ) -> str:
        """Return the best matches, or a readable reason there are none."""
        text = query.strip()
        if not text:
            return "A query is required: describe the task or name its subject in a few words."
        limit = max(1, min(int(limit), MAX_SEARCH_RESULTS))
        skill_query = query_factory() if callable(query_factory) else SkillQuery()
        try:
            loaded = loaded_skills if loaded_skills is not None else await runtime.load(skill_query)
            result = await runtime.find(text, limit, loaded, skill_query)
        except Exception as exc:
            # Returned to the model rather than raised, as for skill_view.
            get_logger().warning(f"skill_search: cannot search: {type(exc).__name__}")
            get_logger().debug(f"skill_search detail: {exc}")
            return "Skill search is currently unavailable."

        get_logger().info(f"skill_search: {len(result.hits)} hit(s), by {result.method}")
        get_logger().debug(f"skill_search: query {text!r}")
        if not result.hits:
            return (
                f'No skill matches "{text}". Try other keywords, synonyms, '
                "or the words in another language."
            )
        lines = [f'Skills matching "{text}" (open one with skill_view):']
        lines += [
            hit.index_line(max_description_chars=runtime.search.description_chars)
            for hit in result.hits
        ]
        if result.more:
            lines.append("More skills match: narrow the query to see them.")
        if result.method == "words":
            lines.append(
                "Matched by shared words: a skill written in another language may be "
                "missing, so search again in that language if nothing here fits."
            )
        return "\n".join(lines)

    # FunctionTool.__init__ is untyped in the framework; the call is correct.
    tool = FunctionTool(func=skill_search, name="skill_search", description="")  # type: ignore[no-untyped-call]
    tool.set_declaration(
        FunctionDeclaration(
            name="skill_search",
            description=_SEARCH_DESCRIPTION,
            parameters=[
                Param(name="query", type="str", description="What to look for, as keywords."),
                Param(
                    name="limit",
                    type="int",
                    description=f"How many matches to return, at most {MAX_SEARCH_RESULTS}.",
                    is_required=False,
                    default=5,
                ),
            ],
        )
    )
    return tool


SKILL_TOOL_NAMES = ("skill_view", "skill_search")


def skill_tools(runtime: SkillsRuntime) -> list[FunctionTool]:
    """The tools an agent with skills gets, built against one runtime."""
    return [build_skill_view_tool(runtime), build_skill_search_tool(runtime)]


def attach_skills(agent: Any, runtime: SkillsRuntime) -> Any:
    """The runtime and the tool are set together because they are read by two
    different consumers: the planner's per-turn skills block reads
    ``agent.skills``, while ``skill_view`` reads the runtime it was built with.
    Replacing one without the other leaves the model shown skills by one runtime
    and fetching them from another.

    Pure wiring, no I/O: loading the catalogue and refusing a turn over an
    unavailable ``required`` skill are the runner's job, once per turn
    (``Runner._load_turn_skills``).

    Idempotent: calling it twice replaces the previous skill tools rather than
    stacking duplicates.

    An agent with no ``skills`` field (``DelegateAgent``, ``WrappedAgent``) is
    returned untouched: the catalogue reaches the model only through that field,
    so a ``skill_view`` on such an agent would be a tool for skills no model is
    ever shown.
    """
    if not hasattr(agent, "skills"):
        return agent

    agent.skills = runtime
    if llm_provider := getattr(agent, "llm_provider", None):
        runtime.bind_llm(llm_provider)

    existing = [
        t
        for t in (getattr(agent, "tools", None) or [])
        if getattr(t, "name", "") not in SKILL_TOOL_NAMES
    ]
    agent.tools = [*existing, *skill_tools(runtime)]

    get_logger().info(
        f"skills: attached skill(s) to agent "
        f"'{getattr(agent, 'name', '?')}' from {runtime.source.describe()}"
    )
    return agent
