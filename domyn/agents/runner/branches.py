"""Sub-agents of one response running at the same time, as branches of one turn.

A hand-off moves the turn to one agent at a time: the conversation keeps a stack
of who is running, the agent on top runs, and when it ends the one below resumes.
Under ``parallel_handoffs`` a response may name several sub-agents, and each runs
as a **branch**: the same conversation, the same history and visibility rules,
but a stack of its own, kept in memory, so the branches can start, hand off and
end without knowing about each other. The agent that called them stays on top of
the conversation's stack (the trunk) and resumes once every branch has ended.

A branch that waits for an answer stops; the others go on. Once all have
stopped, the first one waiting has its agents put on top of the caller's stack,
and from there it is a sub-agent paused like any other: the answer resumes it,
and its end returns to the caller. The others waiting are kept, under
``WAITING_BRANCHES_KEY`` in the conversation state, and put on top in turn when
the one before returns. So only the batch's run needs anything of its own: a
pause, its answer and a restart in between go the way they always went.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from domyn.agents.conversation_manager.base import BaseConversationManager, StackAgentEntry

# Under ``context.state``, keyed by the interaction of the agent that ran a batch:
# the branches of that batch still waiting for an answer after the first one,
# ``{"branches": [record, ...], "held": [call, ...]}``, the calls being those of
# the same response after the batch. See `Runner._next_waiting_branch`.
WAITING_BRANCHES_KEY = "waiting_branches"

# The status of a branch in its record.
RUNNING = "running"
PAUSED = "paused"
DONE = "done"


class TrunkStack:
    """The conversation's own stack, as its conversation manager keeps it."""

    def __init__(
        self,
        manager: BaseConversationManager,
        application_name: str,
        user_id: str,
        conversation_id: str,
    ) -> None:
        self._manager = manager
        self._key = (application_name, user_id, conversation_id)

    async def add(self, agent_name: str, interaction_id: str | None, turn_id: str | None) -> None:
        await self._manager.add_agent_to_stack(
            *self._key, agent_name=agent_name, interaction_id=interaction_id, turn_id=turn_id
        )

    async def pop(self) -> None:
        await self._manager.pop_agent_from_stack(*self._key)

    async def last(self) -> StackAgentEntry | None:
        return await self._manager.get_last_agent_stack(*self._key)

    async def empty(self) -> None:
        await self._manager.empty_agents_stack(*self._key)


class BranchStack:
    """A branch's stack, kept in its record: what goes on top of its caller's
    stack if it stops to wait for an answer. The record also holds the calls the
    branch holds at a pause, moved with its agents."""

    def __init__(self, record: dict[str, Any]) -> None:
        self.record = record

    @property
    def entries(self) -> list[StackAgentEntry]:
        return [StackAgentEntry.model_validate(raw) for raw in self.record["stack"]]

    async def add(self, agent_name: str, interaction_id: str | None, turn_id: str | None) -> None:
        self.record["stack"].append(
            StackAgentEntry(
                agent_name=agent_name, interaction_id=interaction_id, turn_id=turn_id
            ).model_dump()
        )
        if interaction_id:
            self.record["interactions"].append(interaction_id)

    async def pop(self) -> None:
        if self.record["stack"]:
            self.record["stack"].pop()

    async def last(self) -> StackAgentEntry | None:
        entries = self.entries
        return entries[-1] if entries else None

    async def empty(self) -> None:
        self.record["stack"].clear()


AgentStack = TrunkStack | BranchStack


def new_batch(calls: list[tuple[str, str]]) -> dict[str, Any]:
    """The record of a batch about to start: one branch per ``(call_id, agent)``,
    in the order of the calls."""
    return {
        "branches": [
            {
                "call_id": call_id,
                "agent": agent,
                "status": RUNNING,
                "stack": [],
                "interactions": [],
                "question": None,
            }
            for call_id, agent in calls
        ],
    }


def branch_of(batch: dict[str, Any], call_id: str | None) -> dict[str, Any] | None:
    return next((b for b in batch["branches"] if b["call_id"] == call_id), None)


def hidden_from(
    batch: dict[str, Any], call_id: str, outer: Callable[[], set[str]] | None = None
) -> Callable[[], set[str]]:
    """The interactions a branch must not see: those of its siblings that have not
    ended, and whatever its caller does not see (``outer``, for a branch whose
    caller is a branch). Read when asked, so a sibling that ends is visible from
    then on."""

    def hidden() -> set[str]:
        mine = {
            interaction
            for branch in batch["branches"]
            if branch["call_id"] != call_id and branch["status"] != DONE
            for interaction in branch["interactions"]
        }
        return mine | outer() if outer is not None else mine

    return hidden
