"""Instructions runtime: a source plus the budget, loaded by the runner every turn.

The counterpart of ``SkillsRuntime``. Nothing here mutates the ``Agent``: one
instance is shared across concurrent conversations, and a remote source may serve
a different soul per user, so what was loaded lives on the run state for one turn
and reaches the prompt as an argument.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from domyn.agents.instructions.models import AgentInstructions, Instruction
from domyn.agents.instructions.sources import InstructionsSource
from domyn.agents.logger import get_logger
from domyn.agents.skills.runtime import _resolve_by_type

MAX_INSTRUCTION_BYTES = 32_768
"""Total budget across every instruction, in bytes of UTF-8.

The same ceiling Codex puts on ``AGENTS.md``. The cost is paid on **every**
request and no compaction can reduce it: a 200 KB file pasted into ``~/.domyn``
would not fail, it would quietly halve the usable window.
"""


class InstructionRuntime(BaseModel):
    """Resolves an agent's standing instructions for a turn."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: InstructionsSource
    max_bytes: int = MAX_INSTRUCTION_BYTES

    @field_validator("source", mode="before")
    @classmethod
    def _resolve_source(cls, value: Any) -> Any:
        return _resolve_by_type(value, InstructionsSource, "instructions source")

    async def load(self) -> AgentInstructions:
        """What the source holds now, with the budget applied."""
        loaded = self._apply_budget(await self.source.load())
        if loaded:
            get_logger().debug(
                f"instructions: loaded {len(loaded.parts)} from "
                f"{self.source.describe()}, {loaded.chars} chars"
            )
        return loaded

    def merge(self, instruction: str, loaded: AgentInstructions) -> str:
        """The agent's instruction with the standing instructions after it.

        Here rather than on the source: a source says where instructions come
        from, and how they are combined with the agent's own does not depend on
        that — two sources returning the same text merge the same way.

        **Appended, never substituted.** The existing value is the author's: for
        an agent loaded from a ``.py`` or ``.yaml`` it is that agent's job, and for
        the CLI's default it is what ``domyn setup`` asked the user to type.
        Replacing it on the mere presence of a file would silently turn a
        customer-service agent into somebody's personal assistant. After, not
        before, because of two conflicting lines the later one is the one a model
        follows, and these are the user's.
        """
        block = loaded.text
        if not block:
            return instruction
        return f"{instruction}\n\n{block}" if instruction.strip() else block

    def _apply_budget(self, loaded: AgentInstructions) -> AgentInstructions:
        """Cut what exceeds ``max_bytes``, in precedence order, and say so.

        The instruction that crosses the line is truncated on a character
        boundary; anything after it is recorded in ``dropped``. **What was cut is
        always said**, at warning level and on the returned object.
        """
        budget = max(self.max_bytes, 0)
        kept: list[Instruction] = []
        dropped = list(loaded.dropped)
        for one in loaded.parts:
            encoded = one.text.encode("utf-8")
            if len(encoded) <= budget:
                budget -= len(encoded)
                kept.append(one)
            elif budget <= 0:
                dropped.append(one.path or one.name)
            else:
                # `errors="ignore"` drops the partial codepoint at the end rather
                # than putting a replacement character in the model's context.
                text = encoded[:budget].decode("utf-8", errors="ignore")
                budget = 0
                kept.append(one.model_copy(update={"text": text, "truncated": True}))

        result = AgentInstructions.from_parts(
            kept, dropped=tuple(dropped), superseded=loaded.superseded
        )
        if result.truncated:
            cut = [one.path or one.name for one in result.parts if one.truncated]
            get_logger().warning(
                f"instructions: over the {self.max_bytes}-byte budget, content was cut "
                f"from: {', '.join([*cut, *result.dropped])}"
            )
        return result
