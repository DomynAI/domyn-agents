"""Decision models: typed questions, probability answers, one pass.

A decision model (Jev, Clef) is not an LLM. It takes a state and a set of typed
questions and answers all of them at once with probability distributions, in
roughly 150 ms. The framework uses it where it currently either asks a human,
asks a whole LLM for a yes/no, or guesses with a hand-written heuristic.

Nothing here is on by default. With no configuration an agent gets
:class:`~domyn.agents.decisions.none.NoDecisions`, every call returns "I don't
know", and every caller does exactly what it does today.

    from domyn.agents.decisions import DecisionRuntime, build_provider, choice

    runtime = DecisionRuntime(build_provider({
        "type": "system_one",
        "base_url": "http://localhost:28080",
        "model": "Cloudflare/clef",      # the name the SERVER exposes
    }))
    decision = await runtime.ask("approvals", state, {"verdict": choice(...)})
    if runtime.may_act("approvals", decision.get("verdict")):
        ...
"""

from typing import Any

from domyn.agents.decisions.base import DecisionProvider
from domyn.agents.decisions.llm import LLMDecisionProvider
from domyn.agents.decisions.models import (
    Answer,
    Decision,
    Provenance,
    Question,
    choice,
    noul,
    score,
)
from domyn.agents.decisions.none import NoDecisions
from domyn.agents.decisions.runtime import (
    USE_CASES,
    DecisionMode,
    DecisionRuntime,
    UseCase,
    UseCasePolicy,
)
from domyn.agents.decisions.system_one import SystemOneProvider, WorkersAIProvider

PROVIDER_TYPES: dict[str, type[DecisionProvider]] = {
    "system_one": SystemOneProvider,
    "workers_ai": WorkersAIProvider,
    "llm": LLMDecisionProvider,
    "none": NoDecisions,
}


def build_provider(config: dict[str, Any] | None) -> DecisionProvider:
    """Build the provider a configuration describes.

    An unknown ``type`` **raises**, and that is the house rule rather than an
    oversight: a typo in a type name is a configuration error, and the codebase
    is consistent that those surface at load time (``LLMProviderFactory``,
    ``SkillsRuntime._resolve_by_type``, ``CommandBackend``). Degrading quietly
    here would hand someone a silently disabled feature they believe they
    configured. What degrades at *run* time is a provider that cannot be
    reached — see :class:`~domyn.agents.decisions.base.DecisionProvider.decide`.
    """
    if not config:
        return NoDecisions()
    settings = dict(config)
    type_name = settings.pop("type", "none")
    provider_class = PROVIDER_TYPES.get(type_name)
    if provider_class is None:
        raise ValueError(
            f"Unknown decision provider type: {type_name!r}. "
            f"Known types: {', '.join(sorted(PROVIDER_TYPES))}"
        )
    return provider_class(**settings)


def attach_decisions(agent: Any, runtime: DecisionRuntime) -> Any:
    """Give ``agent`` a decision runtime. Pure wiring, no I/O.

    Idempotent, and only the agent it is given — never its sub-agents. The same
    rule as ``attach_instructions``, for the same reason: every use case lives
    on the path of the agent that talks to the user, and attaching down a tree
    would multiply the calls without anyone asking for it. A sub-agent that
    wants its own declares it, and the loader builds it.

    An agent with no ``decisions`` field (``WrappedAgent``, ``DelegateAgent``,
    anything behind an adapter) is returned untouched.
    """
    if not hasattr(agent, "decisions"):
        return agent
    agent.decisions = runtime
    # A skills selector is bound to the agent's facilities when the skills
    # runtime is attached, which may have happened before this. Re-offering is
    # the difference between a `decision` selector that works and one that
    # silently falls back to the index depending on construction order.
    selector = getattr(getattr(agent, "skills", None), "selector", None)
    if selector is not None and hasattr(selector, "bind_decisions"):
        selector.bind_decisions(runtime)
    return agent


def runtime_of(agent: Any) -> DecisionRuntime | None:
    """The agent's decision runtime, if it has a usable one.

    The single accessor every use case goes through, so "no runtime", "a field
    holding something else" and "a runtime whose provider does nothing" all
    produce one answer: ``None``, meaning do what you do today.
    """
    runtime = getattr(agent, "decisions", None)
    if not isinstance(runtime, DecisionRuntime):
        return None
    return runtime


__all__ = [
    "USE_CASES",
    "Answer",
    "Decision",
    "DecisionMode",
    "DecisionProvider",
    "DecisionRuntime",
    "LLMDecisionProvider",
    "NoDecisions",
    "Provenance",
    "Question",
    "SystemOneProvider",
    "UseCase",
    "UseCasePolicy",
    "WorkersAIProvider",
    "attach_decisions",
    "build_provider",
    "choice",
    "noul",
    "runtime_of",
    "score",
]
