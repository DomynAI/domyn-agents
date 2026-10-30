"""Helpers for adapting Domyn invoke payloads to a langgraph runnable's input shape."""

import contextlib
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

InputMapper = Callable[[dict[str, Any]], Any]

INPUT_MAPPER_ATTR = "__domyn_input_mapper__"

T = TypeVar("T")


def _try_set(obj: Any, value: Any) -> None:
    with contextlib.suppress(AttributeError, TypeError):
        setattr(obj, INPUT_MAPPER_ATTR, value)


def input_mapper(map_fn: InputMapper) -> Callable[[T], T]:
    """Attach an input mapper to a langgraph factory or compiled runnable.

    Domyn AGENT_START events carry an ``action.parameters`` dict (typically
    ``{"task": ..., "user_request": ...}``) that does not match a graph's
    declared state schema. Decorating either the factory that produces the
    graph or the compiled graph itself attaches a translation function that
    the runtime applies before calling ``ainvoke``.

    When applied to a factory, the mapper is recorded both on the factory
    itself and on whatever it returns when called, so loading the entrypoint
    by either symbol resolves the mapping.

    Example::

        @input_mapper(lambda d: GraphState(question=d["task"], route="", answer=""))
        def build_graph() -> Any:
            ...
    """

    def decorator(target: T) -> T:
        _try_set(target, map_fn)

        if callable(target) and not hasattr(target, "ainvoke"):

            @wraps(target)  # type: ignore[arg-type]
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                result = target(*args, **kwargs)  # type: ignore[operator]
                _try_set(result, map_fn)
                return result

            _try_set(wrapper, map_fn)
            return wrapper  # type: ignore[return-value]

        return target

    return decorator


def get_input_mapper(obj: Any) -> InputMapper | None:
    return getattr(obj, INPUT_MAPPER_ATTR, None)
