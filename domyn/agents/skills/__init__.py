"""Skills: reusable instructions the agent can consult on demand.

Progressive disclosure by default: the agent always sees a cheap index of what is
available, and pulls a skill's body in only when it needs it. Format is
agentskills.io-compatible, so skills are portable to and from other tools.

Opt-in: an agent with no skills configured behaves exactly as before.
"""

from domyn.agents.skills.models import (
    Enforcement,
    LoadedSkills,
    Selection,
    SkillContent,
    SkillDescriptor,
    SkillIndex,
    SkillQuery,
    SkillRef,
    SkillRequirements,
    clip_text,
)
from domyn.agents.skills.parser import (
    SkillParseError,
    extract_section,
    parse_skill_file,
    parse_skill_text,
    section_names,
)
from domyn.agents.skills.remote import RemoteSkillSource
from domyn.agents.skills.runtime import (
    IndexLayout,
    IndexTier,
    SkillBudget,
    SkillRequiredError,
    SkillsRuntime,
    raise_if_required_unavailable,
)
from domyn.agents.skills.selectors import (
    AllSkillsSelector,
    ExplicitSelector,
    IndexSelector,
    LLMSelector,
    SkillSelector,
)
from domyn.agents.skills.sources import (
    InMemorySkillSource,
    LocalDirSkillSource,
    SkillNotFoundError,
    SkillSource,
    evaluate_requirements,
)

__all__ = [
    "AllSkillsSelector",
    "Enforcement",
    "ExplicitSelector",
    "InMemorySkillSource",
    "IndexLayout",
    "IndexSelector",
    "IndexTier",
    "LLMSelector",
    "LoadedSkills",
    "LocalDirSkillSource",
    "RemoteSkillSource",
    "Selection",
    "SkillBudget",
    "SkillContent",
    "SkillDescriptor",
    "SkillIndex",
    "SkillNotFoundError",
    "SkillParseError",
    "SkillQuery",
    "SkillRef",
    "SkillRequiredError",
    "SkillRequirements",
    "SkillSelector",
    "SkillSource",
    "SkillsRuntime",
    "clip_text",
    "evaluate_requirements",
    "extract_section",
    "parse_skill_file",
    "parse_skill_text",
    "raise_if_required_unavailable",
    "section_names",
]
