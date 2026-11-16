"""The hook of an ``AUTOMATIC`` memory: ``update_memory``, after every agent call.

The counterpart of :mod:`domyn.agents.tools.builtin.memory`: an ``AGENTIC`` memory
gets tools and the agent decides what to keep, an ``AUTOMATIC`` one gets this hook
and the manager decides after each turn (``Agent.model_post_init`` mounts one or the
other). It is a listener that starts the update in the background and returns, so
the turn ends without waiting for the model calls that decide and write it.
"""

from domyn.agents.hooks.base import HookPhase, HookType
from domyn.agents.hooks.function_hook import FunctionHook
from domyn.agents.memory.managers import BaseMemoryManager, MemoryAsset


def memory_update_hook(manager: BaseMemoryManager) -> FunctionHook:
    """The post-agent listener that starts *manager*'s memory update."""
    return FunctionHook(
        name=MemoryAsset.UPDATE_HOOK,
        fun=manager.start_memory_update,  # type: ignore[arg-type]
        trigger=HookPhase.POST_AGENT_CALL,
        type=HookType.LISTENER,
    )
