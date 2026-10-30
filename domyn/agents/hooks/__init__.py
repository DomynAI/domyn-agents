from httpx import AsyncClient

from domyn.agents.core import ExecutionEventType
from domyn.agents.core.utils import load_function_from_file
from domyn.agents.hooks.base import HookError, HookPhase, HookRestAPIData, HookResult, HookType
from domyn.agents.hooks.function_hook import FunctionHook


def _parse_on_event_triggers(
    raw: list[str] | list[ExecutionEventType] | None,
) -> list[ExecutionEventType] | None:
    """Convert a list of string event-type names/values to ExecutionEventType, or return None."""
    if raw is None:
        return None
    result = []
    for item in raw:
        if isinstance(item, ExecutionEventType):
            result.append(item)
        else:
            try:
                result.append(ExecutionEventType(item))
            except ValueError as err:
                valid = [e.value for e in ExecutionEventType]
                raise ValueError(
                    f"Invalid on_event_triggers value '{item}'. Valid values: {valid}"
                ) from err
    return result


class HookFactory:
    @staticmethod
    def from_config(
        config: dict,
        trigger: HookPhase | None = None,
        save_in_registry: bool = False,
        client: AsyncClient | None = None,
    ) -> FunctionHook:
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")

        if config.get("reference"):
            obj = HookFactory._lookup_reference(
                config["reference"],
                trigger or (HookPhase(config["trigger"]) if config.get("trigger") else None),
            )
            alias = config.get("name")
            updates: dict = {}
            if alias and alias != obj.name:
                updates["name"] = alias
            on_event_triggers = _parse_on_event_triggers(config.get("on_event_triggers"))
            if on_event_triggers is not None:
                updates["on_event_triggers"] = on_event_triggers
            if "silent" in config:
                updates["silent"] = config["silent"]
            if updates:
                obj = obj.model_copy(update=updates)
            return obj

        resolved_trigger = trigger or HookPhase(config.get("trigger"))
        hook_type = config.get("type")
        if not hook_type:
            raise ValueError("Hook type must be specified in the config")

        config_params = config.get("config_params", {})
        on_event_triggers = _parse_on_event_triggers(config.get("on_event_triggers"))
        extra: dict = {}
        if on_event_triggers is not None:
            extra["on_event_triggers"] = on_event_triggers
        if "break_on_error" in config:
            extra["break_on_error"] = config["break_on_error"]
        if "silent" in config:
            extra["silent"] = config["silent"]

        if "path" in config:
            fun = load_function_from_file(name=config["name"], path=config["path"])
            obj = FunctionHook(
                fun=fun,
                name=config["name"],
                config_params=config_params,
                trigger=resolved_trigger,
                type=HookType(hook_type),
                input_fields=config.get("input_fields"),
                **extra,
            )
        elif "url" in config:
            if client is None:
                client = AsyncClient(timeout=config.get("timeout", 300.0))
            rest_api_data = HookRestAPIData(
                headers=config.get("headers"),
                method=config.get("method", "POST"),
                url=config["url"],
            )
            obj = FunctionHook(
                fun=None,
                name=config.get("name"),
                rest_api=rest_api_data,
                httpx_client=client,
                config_params=config_params,
                trigger=resolved_trigger,
                type=HookType(hook_type),
                input_fields=config.get("input_fields"),
                **extra,
            )
        else:
            raise ValueError(
                "Either a 'reference' to registry, a 'path' to a local function, "
                "or a 'url' to a REST API must be specified in the config"
            )

        obj.name = config.get("name") or obj.name
        if obj.name is None:
            raise ValueError("Hook 'name' is required when it cannot be derived automatically")

        if save_in_registry:
            HookFactory._get_registry(resolved_trigger)().register(obj.name, obj)

        return obj

    @staticmethod
    def _lookup_reference(name: str, trigger: HookPhase | None) -> FunctionHook:
        from domyn.agents.registry import get_hook_registry

        obj = get_hook_registry().get_hook(name)
        if obj is not None:
            return obj
        raise ValueError(f"Hook '{name}' not found in registry")

    @staticmethod
    def _get_registry(trigger: HookPhase):
        from domyn.agents.registry import get_hook_registry

        return get_hook_registry


__all__ = [
    "FunctionHook",
    "HookError",
    "HookFactory",
    "HookPhase",
    "HookRestAPIData",
    "HookResult",
    "HookType",
]
