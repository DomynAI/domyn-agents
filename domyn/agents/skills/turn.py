"""One turn's worth of skills, for one agent entry.

This module sits on the split between the two halves of the pipeline. The runner
**loads** each agent's skills once per turn (:meth:`SkillsRuntime.load`); the
agent **selects** from what was loaded, on every entry (:meth:`SkillsRuntime.select`).
Loading is I/O and does not depend on which sub-task an agent was handed;
selection does, and a single turn may enter the same agent more than once.

Nothing derived here is cached anywhere. The payload is a function of *(agent,
user, this entry's text)*: :class:`SkillQuery` carries ``user_id`` and ``task``,
a remote source serves a different catalogue per user, and ``LLMSelector`` routes
on the task. One ``Agent`` instance serves concurrent conversations, and one
planner instance is shared by every agent that names it in YAML, so a field on
either would hand one user another user's skills. ``context.state`` is worse
still: it is merged into the persisted conversation and reseeded into every
context the runner builds, sub-agents' included — which is exactly how a parent's
payload once reached a sub-agent that has no ``skill_view`` tool.
"""

from domyn.agents.core import BaseAgent, InteractionContext
from domyn.agents.logger import get_logger
from domyn.agents.skills.models import LoadedSkills, SkillQuery
from domyn.agents.skills.runtime import SkillRequiredError, SkillsRuntime


def skill_query_for(agent: BaseAgent, context: InteractionContext) -> SkillQuery:
    """Describe this agent, this user and this input to a source or a selector.

    One function, two callers: the runner builds it at load time, when
    ``agent_input`` is the user's turn, and the agent at select time, when it is
    the task this agent was handed. That the ``task`` differs between the two is
    the intended behaviour of the split, not an accident of duplication.

    The whole input text, not just its first part: a selector routing on the task
    should see everything that was said.
    """
    agent_input = getattr(context, "agent_input", None)
    task = ""
    if agent_input is not None and getattr(agent_input, "content", None):
        task = " ".join(part.text or "" for part in agent_input.content).strip()

    return SkillQuery(
        user_id=getattr(context, "user_id", None),
        agent_name=getattr(agent, "name", None),
        task=task or None,
        available_tools=[
            tool.name
            for tool in (getattr(agent, "tools", None) or [])
            if getattr(tool, "name", None)
        ],
    )


async def build_skills_payload(
    agent: BaseAgent,
    context: InteractionContext,
    loaded: LoadedSkills | None = None,
) -> str:
    """Select from this turn's skills and render the block handed to the model.

    ``loaded`` is what the runner loaded for this agent at the top of the turn.
    ``None`` means nobody preloaded — a direct ``agent.execute(context)`` outside
    a runner — and the runtime then loads for itself, which is what happened on
    every entry before the split.

    Returns ``""`` when the agent declares no skills, which is the common case. A
    failure to resolve an *advisory* skill is logged and degrades to no skills; a
    *required* one raises :class:`SkillRequiredError`, because a guardrail that
    silently fails to load is worse than a refusal to run.
    """
    runtime = getattr(agent, "skills", None)
    # isinstance, not `is None`: BaseAgent declares no `skills` attribute at all,
    # and an agent built as a MagicMock answers every getattr with a truthy child
    # mock, which would then blow up on the first await.
    if not isinstance(runtime, SkillsRuntime):
        return ""

    query = skill_query_for(agent, context)

    try:
        if loaded is None:
            loaded = await runtime.load(query)
        selection, contents = await runtime.select(loaded, query)
    except SkillRequiredError:
        raise
    except Exception as exc:
        # Advisory skills must never take a turn down.
        get_logger().warning(f"skills: could not resolve for this turn: {type(exc).__name__}")
        get_logger().debug(f"skills resolve detail: {exc}")
        return ""

    payload = runtime.build_payload(selection, contents)
    if payload:
        get_logger().info(
            f"skills: {len(selection.index)} indexed, {len(contents)} expanded"
            + (f" ({selection.rationale})" if selection.rationale else "")
        )
        # Names only at info level; bodies are user content and stay at debug.
        get_logger().debug(f"skills: payload is {len(payload)} chars")
    return payload
