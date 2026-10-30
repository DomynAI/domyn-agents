"""Shared per-invocation plumbing for FrameworkAdapter implementations.

LangGraphAdapter and DomynAgentAdapter each drive a fresh asyncio.Task for
their delegate/platform-tool callback (``call_platform_tool`` /
``_make_tool_call``) — LangGraph's ``ainvoke()`` and
``Runner._merge_with_injections`` both schedule every step of a turn via
``asyncio.ensure_future``/``asyncio.create_task``, so ``asyncio.current_task()``
inside that callback is never the task ``invoke()``'s ``_runner_task()``
registered anything under. A ``contextvars.ContextVar`` survives this where a
``dict[asyncio.Task, ...]`` would not: a ``contextvars.Context`` is copied into
every child task at the moment it is spawned from a task carrying it, so
``.get()`` finds the right value regardless of which disposable child task is
executing.

``InvocationCtx`` fields are *snapshots* taken at ``invoke()`` time, not a live
reference to the inbound ``BaseEvent``: the native Runner's
``_setup_conversation_turn`` rewrites that event's
``interaction_id``/``turn_id``/``conversation_id`` in place (it always mints
its own ``interaction_id``), so reading these fields later off a stored
reference yields the runner's regenerated ids instead of the platform's
originals — and the platform then can't correlate, or never answers, the tool
call. Both adapters hit the same failure mode, so both snapshot at
``invoke()`` time via ``spawn_invocation()`` below rather than re-reading the
event later.
"""

import asyncio
import contextvars
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from domyn.agents.interfaces import EmitPriorityFn


@dataclass(frozen=True, slots=True)
class InvocationCtx:
    """Per-invocation data needed by a platform/delegate tool round trip.

    author/conversation_id/turn_id/interaction_id ride along with
    emit_priority: the relay routes frames to the platform by
    conversation_id and drops any frame whose value doesn't match, turn_id
    is read by the transports' cross-turn replay filter, and
    author/interaction_id are the correlation identity the platform matches
    a TOOL_START against. Without them, TOOL_START events never reach the
    platform-side runner and the tool call never completes.
    """

    emit_priority: EmitPriorityFn
    author: str
    conversation_id: str | None
    turn_id: str | None
    interaction_id: str | None


# Shared by every FrameworkAdapter implementation. Safe across adapter types:
# each adapter's own _runner_task (built by spawn_invocation below) sets and
# resets its own value inside its own task's context, and InvocationCtx
# carries the invocation's own emit_priority closure rather than depending on
# which adapter subclass is running — nothing reads adapter identity off it.
invocation_ctx: contextvars.ContextVar[InvocationCtx | None] = contextvars.ContextVar(
    "adapter_invocation_ctx", default=None
)


def spawn_invocation(
    *,
    request_id: str,
    conversation_id: str,
    ctx: InvocationCtx,
    semaphore: asyncio.Semaphore,
    conversation_tasks: dict[str, asyncio.Task[None]],
    in_flight: dict[str, asyncio.Task[None]],
    run_body: Callable[[], Awaitable[None]],
) -> None:
    """Schedule *run_body* under the invocation bookkeeping both adapters share.

    Registers the task in *conversation_tasks* (read by on_cancel) and
    *in_flight* (read by close() / the busy-buffer check), sets *ctx* on the
    shared ``invocation_ctx`` ContextVar for the duration of the call, and
    bounds concurrency with *semaphore* — all reset/popped in a ``finally``
    so a cancelled or failing *run_body* still cleans up.
    """

    async def _runner_task() -> None:
        current = asyncio.current_task()
        conversation_tasks[conversation_id] = current  # type: ignore[assignment]
        token = invocation_ctx.set(ctx)
        try:
            async with semaphore:
                await run_body()
        finally:
            invocation_ctx.reset(token)
            conversation_tasks.pop(conversation_id, None)

    task = asyncio.create_task(_runner_task(), name=f"invoke:{request_id}")
    in_flight[request_id] = task
    task.add_done_callback(lambda _: in_flight.pop(request_id, None))
