from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncGenerator, Callable
from typing import Any, ClassVar, Literal

import httpx
from pydantic import ConfigDict, Field, PrivateAttr, field_validator, model_validator

from domyn.agents.approvals import ApprovalVerdict
from domyn.agents.core import ExecutionEventType, GuardrailResult, InteractionContext, Part
from domyn.agents.core.exceptions import RestHookError
from domyn.agents.core.utils import (
    SerializedCallable,
    serialize_callable,
    validate_parameter_keys,
    wrap_as_async_generator,
)
from domyn.agents.hooks.base import (
    Hook,
    HookError,
    HookPhase,
    HookRestAPIData,
    HookResult,
    HookType,
    InputFields,
)
from domyn.agents.tools import ToolResult
from domyn.agents.tracing import add_trace_headers


class FunctionHook(Hook):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    phase_params: ClassVar[dict] = {
        HookPhase.PRE_AGENT_CALL: {"context", "agent"},
        HookPhase.POST_AGENT_CALL: {"context", "agent"},
        HookPhase.PRE_TOOL_APPROVAL: {"tool", "tool_args"},
        HookPhase.PRE_TOOL_CALL: {"tool", "tool_args"},
        HookPhase.POST_TOOL_CALL: {"tool", "result"},
        HookPhase.ON_AGENT_EVENT: {"event", "context"},
    }
    valid_input_fields: ClassVar[dict[HookPhase, frozenset[InputFields]]] = {
        HookPhase.PRE_AGENT_CALL: frozenset(
            {
                InputFields.AGENT_INPUT,
                InputFields.AGENT_NAME,
                InputFields.LAST_EVENT,
                InputFields.INTERACTION_HISTORY,
                InputFields.INTERACTION_CONTEXT,
            }
        ),
        HookPhase.POST_AGENT_CALL: frozenset(
            {
                InputFields.AGENT_INPUT,
                InputFields.AGENT_NAME,
                InputFields.LAST_EVENT,
                InputFields.INTERACTION_HISTORY,
                InputFields.INTERACTION_CONTEXT,
            }
        ),
        HookPhase.PRE_TOOL_APPROVAL: frozenset(
            {
                InputFields.TOOL_NAME,
                InputFields.TOOL_ARGS,
            }
        ),
        HookPhase.PRE_TOOL_CALL: frozenset(
            {
                InputFields.TOOL_NAME,
                InputFields.TOOL_ARGS,
            }
        ),
        HookPhase.POST_TOOL_CALL: frozenset(
            {
                InputFields.TOOL_NAME,
                InputFields.TOOL_RESULT,
            }
        ),
        HookPhase.ON_AGENT_EVENT: frozenset(
            {
                InputFields.LAST_EVENT,
                InputFields.INTERACTION_HISTORY,
                InputFields.CURRENT_EVENT,
                InputFields.INTERACTION_CONTEXT,
            }
        ),
    }

    fun: SerializedCallable | Any = Field(
        ..., description="Serialized callable function metadata or callable"
    )

    _actual_callable: Any = PrivateAttr(default=None)
    _async_generator: Any = PrivateAttr(default=None)

    rest_api: HookRestAPIData | None = None

    config_params: dict = Field(default_factory=dict)
    input_fields: list[InputFields] | None = None
    httpx_client: httpx.AsyncClient | dict | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    on_event_triggers: list[ExecutionEventType] | None = Field(default=None, validate_default=True)
    break_on_error: bool = Field(
        default=True, description="Whether to break execution on error in this hook."
    )

    @field_validator("on_event_triggers", mode="before")
    @classmethod
    def default_on_event_triggers(cls, v) -> list[ExecutionEventType]:
        if v is None:
            v = [
                ExecutionEventType.USER_INPUT,
                ExecutionEventType.TOOL_FEEDBACK,
                ExecutionEventType.AGENT_CALL,
                ExecutionEventType.AGENT_START,
                ExecutionEventType.AGENT_END,
                ExecutionEventType.TOOL_START,
                ExecutionEventType.TOOL_UPDATE,
                ExecutionEventType.TOOL_NEEDS_FEEDBACK,
                ExecutionEventType.TOOL_END,
                ExecutionEventType.RESPONSE_TOKEN,
                ExecutionEventType.RESPONSE,
            ]
        return v

    def model_post_init(self, context: Any, /) -> None:
        if self.rest_api is not None:
            if self.httpx_client is None:
                self.httpx_client = httpx.AsyncClient()
            elif isinstance(self.httpx_client, dict):
                self.httpx_client = httpx.AsyncClient(**self.httpx_client)
            if self.rest_api.headers is None:
                self.rest_api.headers = {}
            self.fun = self._make_rest_call_wrapper(
                method=self.rest_api.method,
                url=self.rest_api.url,
                headers=add_trace_headers(self.rest_api.headers),
                httpx_client=self.httpx_client,
            )
        if callable(self.fun):
            self._actual_callable = self.fun
            self._async_generator = wrap_as_async_generator(self._actual_callable)
            object.__setattr__(self, "fun", serialize_callable(self.fun))
        else:
            self._actual_callable = None
        self.httpx_client = None  # Remove client after use to avoid serialization issues

    @model_validator(mode="after")
    def set_name(self) -> FunctionHook:
        if self.name is None and self._actual_callable is not None:
            self.name = self._actual_callable.__name__
        elif self.name is None and isinstance(self.fun, dict):
            self.name = self.fun.get("name", "unknown")
        return self

    @model_validator(mode="after")
    def validate_phase_params(self) -> FunctionHook:
        if self._actual_callable is None or self.rest_api is not None:
            return self
        if self.input_fields is None:
            mandatory_keys = self.phase_params[self.trigger]
        else:
            mandatory_keys = {f.value for f in self.input_fields}
        validate_parameter_keys(self._actual_callable, mandatory_keys=mandatory_keys)
        return self

    @model_validator(mode="after")
    def validate_input_fields(self) -> FunctionHook:
        if self.input_fields is None:
            return self
        if self.trigger not in self.valid_input_fields:
            raise ValueError(f"Unknown hook phase: {self.trigger}")
        valid_fields = self.valid_input_fields[self.trigger]
        if not set(self.input_fields).issubset(valid_fields):
            raise ValueError(
                f"Invalid input fields {self.input_fields} for trigger {self.trigger}. "
                f"Valid fields are: {set(valid_fields)}"
            )
        return self

    def _filter_args(self, **kwargs: Any) -> dict[str, Any]:
        """Project canonical phase kwargs into a dict keyed by selected InputFields.

        Returns native Python objects (BaseEvent, ToolResult, ...) — REST callers must
        JSON-serialize on top.
        """
        agent = kwargs.get("agent")
        context = kwargs.get("context")
        tool = kwargs.get("tool")
        tool_args = kwargs.get("tool_args")
        result = kwargs.get("result")
        current_event = kwargs.get("event")
        filtered: dict[str, Any] = {}
        for field in self.input_fields or []:
            match field:
                case InputFields.AGENT_NAME:
                    filtered[field.value] = agent.name if agent else ""
                case InputFields.LAST_EVENT:
                    filtered[field.value] = context.last_event if context else None
                case InputFields.INTERACTION_HISTORY:
                    filtered[field.value] = list(context.interaction_history) if context else []
                case InputFields.AGENT_INPUT:
                    text = ""
                    ev = context.agent_input if context else None
                    if ev is not None:
                        if ev.event_type == ExecutionEventType.USER_INPUT and ev.content:
                            text = getattr(ev.content[0], "text", None) or ""
                        elif ev.event_type == ExecutionEventType.AGENT_START and isinstance(
                            getattr(ev.action, "parameters", None), dict
                        ):
                            text = ev.action.parameters.get("task") or ""
                    filtered[field.value] = text
                case InputFields.CURRENT_EVENT:
                    filtered[field.value] = current_event if current_event else {}
                case InputFields.TOOL_NAME:
                    filtered[field.value] = tool.name if tool else ""
                case InputFields.TOOL_ARGS:
                    filtered[field.value] = tool_args if tool_args else {}
                case InputFields.TOOL_RESULT:
                    filtered[field.value] = result if result else {}
                case InputFields.INTERACTION_CONTEXT:
                    filtered[field.value] = context if context else {}
                case _:
                    raise NotImplementedError(f"Input field {field} is not implemented yet.")
        return filtered

    def _create_body(self, **kwargs: Any) -> dict[str, Any]:
        def to_jsonable(value: Any) -> Any:
            if value is None:
                return {}
            if hasattr(value, "model_dump"):
                return value.model_dump(mode="json")
            if isinstance(value, list):
                return [
                    item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                    for item in value
                ]
            return value

        # TODO: make input_fields non-nullable (default []). None is the legacy mode: all phase
        # params plus config_params.
        if self.input_fields is not None:
            # kwargs already projected by `run()` via `_filter_args`.
            keys, extras = list(kwargs.keys()), {}
        else:
            keys = list(self.phase_params[self.trigger])
            extras = {"config_params": {k: v for k, v in kwargs.items() if k not in keys}}
        return {k: to_jsonable(kwargs[k]) for k in keys} | extras

    def _make_rest_call_wrapper(
        self,
        method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"],
        url: str,
        headers: dict[str, str] | None,
        httpx_client: httpx.AsyncClient,
    ) -> Callable[..., Any]:
        async def _inner(**kwargs: Any) -> dict:
            body = self._create_body(**kwargs)
            response = await httpx_client.request(
                method=method, url=url, json=body, headers=add_trace_headers(headers)
            )
            if response.status_code != 200:
                raise RestHookError(error_code=response.status_code, error_message=response.text)
            return response.json()

        return _inner

    @property
    def async_generator(self) -> Callable[..., AsyncGenerator]:
        if self._async_generator is not None:
            return self._async_generator
        else:
            raise ValueError(
                "Cannot convert serialized function back to callable. Original callable was lost."
            )

    @property
    def original_callable_function(self) -> Any:
        if self._actual_callable is not None:
            return self._actual_callable
        else:
            raise ValueError(
                "Cannot convert serialized function back to callable. Original callable was lost."
            )

    def validate_result(self, hook_result: HookResult) -> None:
        if not isinstance(hook_result, HookResult):
            raise TypeError(f"Expected HookResult, got {type(hook_result).__name__}")
        match self.trigger:
            case HookPhase.PRE_AGENT_CALL | HookPhase.POST_AGENT_CALL:
                if hook_result.new_tool_args is not None or hook_result.new_result is not None:
                    raise ValueError(
                        "Agent hook functions cannot modify or return tool arguments or results."
                    )
                match self.type:
                    case HookType.LISTENER:
                        if (not hook_result.is_update) and hook_result.new_context is not None:
                            raise ValueError(
                                "The listener agent hook function cannot modify context."
                            )
                    case HookType.UPDATER:
                        if (not hook_result.is_update) and hook_result.new_context is None:
                            raise ValueError(
                                "The updater agent hook function must return the new context."
                            )
                    case _:
                        raise ValueError(f"Invalid hook type: {self.type}")
            case HookPhase.PRE_TOOL_APPROVAL:
                # An approval hook says allow, deny or nothing. It cannot edit
                # arguments or results: changing what runs is a different
                # decision from deciding whether it may run, and mixing them
                # would let a policy approve one call and execute another.
                if (
                    (hook_result.new_context is not None)
                    or (hook_result.new_result is not None)
                    or (hook_result.new_tool_args is not None)
                    or hook_result.break_execution
                ):
                    raise ValueError(
                        "Pre-tool-approval hooks may only return an approval verdict: "
                        "not a context, tool arguments, a result, or break_execution."
                    )
            case HookPhase.PRE_TOOL_CALL:
                if (
                    (hook_result.new_context is not None)
                    or (hook_result.new_result is not None)
                    or hook_result.break_execution
                ):
                    raise ValueError(
                        "Pre-tool hook functions cannot modify or return the context or tool result, nor they can break the execution."
                    )
                match self.type:
                    case HookType.LISTENER:
                        if hook_result.new_tool_args is not None:
                            raise ValueError(
                                "Listener pre-hook cannot modify or return the tool arguments."
                            )
                    case HookType.UPDATER:
                        if (not hook_result.is_update) and (hook_result.new_tool_args is None):
                            raise ValueError("Updater pre-hook must return the tool arguments.")
                        if hook_result.new_tool_args is not None:
                            if not isinstance(hook_result.new_tool_args.get("state", {}), dict):
                                raise ValueError(
                                    "If present, the 'state' in tool arguments returned by the updater pre-hook must be a dictionary."
                                )
                            if not isinstance(
                                hook_result.new_tool_args.get("run_config", {}), dict
                            ):
                                raise ValueError(
                                    "If present, the 'run_config' in tool arguments returned by the updater pre-hook must be a dictionary."
                                )
                            if not isinstance(
                                hook_result.new_tool_args.get("call_metadata", {}), dict
                            ):
                                raise ValueError(
                                    "If present, the 'call_metadata' in tool arguments returned by the updater pre-hook must be a dictionary."
                                )
                    case _:
                        raise ValueError(f"Invalid hook type: {self.type}")
            case HookPhase.POST_TOOL_CALL:
                if (
                    (hook_result.new_context is not None)
                    or (hook_result.new_tool_args is not None)
                    or hook_result.break_execution
                ):
                    raise ValueError(
                        "Post-tool hook functions cannot modify or return the context or the tool args, nor they can break the execution."
                    )
                match self.type:
                    case HookType.LISTENER:
                        if hook_result.new_result is not None:
                            raise ValueError(
                                "Listener post-hook cannot modify or return the tool result."
                            )
                    case HookType.UPDATER:
                        if (not hook_result.is_update) and hook_result.new_result is None:
                            raise ValueError("Updater post-hook must return the new tool result.")
                    case _:
                        raise ValueError(f"Invalid hook type: {self.type}")
            case HookPhase.ON_AGENT_EVENT:
                if (
                    hook_result.new_context is not None
                    or hook_result.new_tool_args is not None
                    or hook_result.new_result is not None
                ):
                    raise ValueError(
                        "ON_AGENT_EVENT hooks cannot modify context, tool arguments, or tool results."
                    )
                match self.type:
                    case HookType.LISTENER:
                        if hook_result.modified_event is not None:
                            raise ValueError(
                                "Listener on_agent_event hook cannot modify the event."
                            )
                    case HookType.UPDATER:
                        pass  # new_event can be None (pass through) or BaseEvent (replace)
                    case _:
                        raise ValueError(f"Invalid hook type: {self.type}")
            case _:
                raise ValueError(f"Unknown trigger: {self.trigger}")

    def process_result(self, result: Any, original_event: Any = None) -> HookResult:
        from domyn.agents.core import BaseEvent as _BaseEvent  # lazy to avoid circular import

        if isinstance(result, HookResult):
            if result.name is None:
                result.name = self.name
        elif isinstance(result, InteractionContext):
            result = HookResult(new_context=result, success=True, name=self.name)
        elif isinstance(result, ToolResult):
            result = HookResult(new_result=result, success=True, name=self.name)
        elif isinstance(result, _BaseEvent):
            result = HookResult(modified_event=result, success=True, name=self.name)
        elif isinstance(result, ApprovalVerdict):
            result = HookResult(approval=result, success=True, name=self.name)
        elif isinstance(result, dict):
            if result.keys() & set(HookResult.model_fields):
                result = HookResult(**{"success": True, "name": self.name, **result})
            elif self.trigger == HookPhase.ON_AGENT_EVENT:
                event_dict = result.copy()
                if original_event is not None:
                    event_dict.setdefault("turn_id", original_event.turn_id)
                    event_dict.setdefault("interaction_id", original_event.interaction_id)
                    event_dict.setdefault("author", original_event.author)
                result = HookResult(
                    modified_event=_BaseEvent(**event_dict), success=True, name=self.name
                )
            else:
                result = HookResult(emitted_content=result, success=True, name=self.name)
        elif result is None:
            result = HookResult(emitted_content=None, success=True, name=self.name)
        else:
            result = HookResult(emitted_content=str(result), success=True, name=self.name)
        return result

    def _make_error_guardrail_content(self) -> list[Part]:
        name = "Can't complete your request"
        reason = f"The {name} check didn't work as expected, so I'll stop here."
        text_options = [
            "Out of caution, I can't complete your request at this time. Please try a different one.",
            "For safety reasons, I won't move forward with the answer. Feel free to ask me something else.",
            "To prevent unchecked content, I can't proceed with your request right now. Please ask me something else.",
        ]
        text = text_options[random.randint(0, len(text_options) - 1)]
        return [
            Part(
                text=text,
                guardrail_result=GuardrailResult(
                    name=name,
                    passed=False,
                    error_occurred=True,
                    reason=reason,
                ),
            )
        ]

    async def run(self, *args: Any, **kwargs: Any) -> AsyncGenerator[HookResult, None]:
        try:
            original_event = (
                kwargs.get("event") if self.trigger == HookPhase.ON_AGENT_EVENT else None
            )
            if self.input_fields is not None:
                kwargs = self._filter_args(**kwargs)
            async for result in self.async_generator(*args, **kwargs, **self.config_params):
                hook_result = self.process_result(result, original_event=original_event)
                self.validate_result(hook_result)
                if not hook_result.success and self.break_on_error:
                    hook_result = hook_result.model_copy(
                        update={
                            "break_execution": True,
                            "emitted_content": hook_result.emitted_content
                            or self._make_error_guardrail_content(),
                        }
                    )
                yield hook_result

        except Exception as e:
            error = HookError.from_exception(e)
            guardrail_content = (
                self._make_error_guardrail_content() if self.break_on_error else None
            )
            yield HookResult(
                emitted_content=guardrail_content,
                success=False,
                error=error,
                name=self.name,
                break_execution=self.break_on_error,
            )

    async def drain_to_queue(
        self,
        queue: asyncio.Queue,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Drain this hook's async generator into a queue for parallel execution.

        Intermediate results (is_update=True) and the final result (is_update=False) are
        put directly onto the queue. If the hook never yields a final result, a synthetic
        successful HookResult is appended so the consumer always sees a terminal item.
        """
        had_final = False
        async for item in self.run(*args, **kwargs):
            await queue.put(item)
            if not item.is_update:
                had_final = True
        if not had_final:
            await queue.put(
                HookResult(
                    emitted_content=None,
                    success=True,
                    error=None,
                    name=self.name,
                )
            )
