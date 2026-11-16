"""The tools of an ``AGENTIC`` memory: ``remember``, ``recall`` and ``forget``.

They go together. An agent that decides what to keep also reads it back a page at a
time and drops what stopped being true, so the three are built by one function, and
the agent mounts them whenever its memory manager runs in ``AGENTIC`` mode
(``Agent.model_post_init``). They are not a ``tool_bundles`` entry: a bundle is a
capability a config switches on by itself and is bound to a workspace, while these
exist only with a memory manager, which is what declares them.

Each is a thin wrapper over a :class:`BaseMemoryManager` method, which reads the
user from the call's ``run_config`` and the agent from the ``agent`` config param.
"""

from typing import Any

from domyn.agents.memory.managers import BaseMemoryManager, MemoryAsset
from domyn.agents.tools.function_tool import ConfigParam, FunctionTool


def memory_tools(manager: BaseMemoryManager, agent: Any) -> list[FunctionTool]:
    """``remember``, ``recall`` and ``forget`` over *manager*, bound to *agent*.

    *agent* is typed ``Any`` because ``agents`` imports ``tools``: naming ``Agent``
    here would close the cycle.
    """
    bound_to_agent = [ConfigParam(name="agent", value=agent)]
    return [
        FunctionTool(
            name=MemoryAsset.REMEMBER_TOOL,
            description=(
                "Store relevant information to the agent's memory for future use. Use this "
                "proactively whenever the user reveals information about themselves, their "
                "preferences, interests, or opinions — even if mentioned in passing."
            ),
            func=manager.remember,  # type: ignore[arg-type]
            config_params=bound_to_agent,
        ),
        FunctionTool(
            name=MemoryAsset.RECALL_TOOL,
            description=(
                "Read what is remembered about the user. Without arguments it returns the "
                "most recent facts; with a query, the ones that match it; with the cursor "
                "from a previous answer, the next page."
            ),
            func=manager.recall,  # type: ignore[arg-type]
            config_params=bound_to_agent,
        ),
        FunctionTool(
            name=MemoryAsset.FORGET_TOOL,
            description=(
                "Remove one remembered fact, by the id shown with it. Use it when the user "
                "asks you to forget something or says a fact is no longer true."
            ),
            func=manager.forget,  # type: ignore[arg-type]
            config_params=bound_to_agent,
        ),
    ]
