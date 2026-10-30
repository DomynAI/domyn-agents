import json

from domyn.agents.agents import Agent
from domyn.agents.core import InteractionContext
from domyn.agents.hooks import FunctionHook, HookPhase, HookType


def listener_agent_hook(context: InteractionContext, agent: Agent) -> None:
    pass


CONFIG_PARAMS = {"param1": "value1", "param2": 2.0, "param3": True}


def test_listener_function_agent_hook_serialization():
    original_hook = FunctionHook(
        fun=listener_agent_hook,
        config_params=CONFIG_PARAMS,
        trigger=HookPhase.PRE_AGENT_CALL,
        type=HookType.LISTENER,
    )
    serialized = original_hook.model_dump_json()
    loaded_hook = FunctionHook(**json.loads(serialized))
    assert loaded_hook.model_dump() == original_hook.model_dump()


def test_updater_function_agent_hook_serialization():
    original_hook = FunctionHook(
        fun=listener_agent_hook,
        config_params=CONFIG_PARAMS,
        trigger=HookPhase.POST_AGENT_CALL,
        type=HookType.UPDATER,
    )
    serialized = original_hook.model_dump_json()
    loaded_hook = FunctionHook(**json.loads(serialized))
    assert loaded_hook.model_dump() == original_hook.model_dump()
