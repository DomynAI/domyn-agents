"""Skill source backed by an HTTP service.

This is a placeholder implementation that raises `NotImplementedError` for all operations.
"""

from domyn.agents.skills.models import SkillContent, SkillIndex, SkillQuery, SkillRef
from domyn.agents.skills.sources import SkillSource


class RemoteSkillSource(SkillSource):
    """Placeholder for an HTTP-backed skill source."""

    type: str = "remote"

    async def index(self, query: SkillQuery) -> SkillIndex:
        raise NotImplementedError("RemoteSkillSource is not implemented")

    async def fetch(self, ref: SkillRef, query: SkillQuery) -> SkillContent:
        raise NotImplementedError("RemoteSkillSource is not implemented")
