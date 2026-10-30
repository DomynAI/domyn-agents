"""RunState tracks the lifecycle of a single langchain run; RunRegistry stores them.

The registry exists for two reasons:
  - root detection: ``state.parent_run_id is None`` tells the handler whether
    a finished run is the trace root (and therefore should emit a terminal
    RESPONSE event);
  - subtree cleanup: ``pop_trajectory`` drops a finished tree from the
    registry so completed runs don't leak.

Ownership contract: `RunState` instances live inside `RunRegistry` and are
mutated ONLY by registry methods (under the registry's lock). The registry's
public mutators return the `RunState` after their critical section so callers
can *read* its fields outside the lock. Callers MUST NOT write to a returned
`RunState` — the registry is the single writer. Reads after `pop_trajectory`
are safe because the state is no longer reachable from the registry.
"""

import threading
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import UUID

RunStatus = Literal["open", "ok", "error"]


@dataclass
class RunState:
    run_id: UUID
    parent_run_id: UUID | None
    span_type: str
    start_time: float
    end_time: float | None = None
    inputs: Any = None
    output: Any = None
    error: BaseException | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    children: list[UUID] = field(default_factory=list)
    status: RunStatus = "open"


class RunRegistry:
    """Thread-safe registry of in-flight langchain runs.

    Lock discipline: every public method holds `self._lock` only long enough to
    read/mutate the internal dicts. The recursive subtree walk in
    `pop_trajectory` collects + deletes inside a single short lock block.

    Parent linkage is stable: `parent_run_id` is written exactly once (at
    `start()`) and never changes, so walking parents cannot form a cycle.
    """

    def __init__(self) -> None:
        self._runs: dict[UUID, RunState] = {}
        # Children whose parent hasn't been registered yet. Under RunnableParallel
        # or similar executors, a child's on_*_start may arrive before the
        # parent's on_chain_start. We park them here and link once the parent
        # arrives. Keyed by parent_run_id → list of child run_ids.
        self._pending_children: dict[UUID, list[UUID]] = {}
        self._lock = threading.Lock()

    def start(
        self,
        *,
        run_id: UUID,
        parent_run_id: UUID | None,
        span_type: str,
        start_time: float,
        inputs: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> RunState:
        with self._lock:
            state = RunState(
                run_id=run_id,
                parent_run_id=parent_run_id,
                span_type=span_type,
                start_time=start_time,
                inputs=inputs,
                metadata=metadata or {},
            )
            self._runs[run_id] = state
            # Link to parent if already registered; otherwise park as pending.
            if parent_run_id is not None:
                parent = self._runs.get(parent_run_id)
                if parent is not None:
                    parent.children.append(run_id)
                else:
                    self._pending_children.setdefault(parent_run_id, []).append(run_id)
            # Adopt any children that arrived before this run did.
            deferred = self._pending_children.pop(run_id, None)
            if deferred:
                state.children.extend(deferred)
            return state

    def end(
        self,
        *,
        run_id: UUID,
        output: Any,
        end_time: float,
    ) -> tuple[RunState, bool]:
        with self._lock:
            state = self._runs.get(run_id)
            if state is None:
                raise KeyError(f"run_id {run_id} not found in registry (end before start?)")
            state.end_time = end_time
            state.output = output
            state.status = "ok"
            is_root = state.parent_run_id is None
            return state, is_root

    def error(
        self,
        *,
        run_id: UUID,
        error: BaseException,
        end_time: float,
    ) -> tuple[RunState | None, bool]:
        """Record an error against ``run_id``.

        Returns ``(None, False)`` instead of raising when ``run_id`` was never
        registered - this is the normal case for chain-level runs under
        create_agent()-style graphs (see on_chain_start's `ls_integration ==
        "langchain_create_agent"` early-return), where an error propagating up
        through the graph has no matching registry entry to update. Raising
        here used to abort on_chain_error before it could emit the
        ERROR_OCCURRED event at all, silently dropping error visibility.
        """
        with self._lock:
            state = self._runs.get(run_id)
            if state is None:
                return None, False
            state.end_time = end_time
            state.error = error
            state.status = "error"
            is_root = state.parent_run_id is None
            return state, is_root

    def get(self, run_id: UUID) -> RunState | None:
        with self._lock:
            return self._runs.get(run_id)

    def pop_trajectory(self, root_id: UUID) -> None:
        """Delete ``root_id`` and the entire subtree rooted at it.

        Called by the handler when a root run completes, to release all the
        per-run state held under it. Done inside one short lock block.
        """
        with self._lock:
            self._collect_and_delete(root_id)

    def _collect_and_delete(self, run_id: UUID) -> None:
        state = self._runs.get(run_id)
        if state is None:
            return
        for child_id in list(state.children):
            self._collect_and_delete(child_id)
        self._runs.pop(run_id, None)
