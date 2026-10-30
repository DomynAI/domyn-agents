import inspect
from collections.abc import Callable
from functools import cached_property
from typing import Any, Literal

import httpx
from pydantic import Field, PrivateAttr

from domyn.agents.core.utils import serialize_callable
from domyn.agents.hooks import HookPhase
from domyn.agents.tools.base import FunctionDeclaration, Param
from domyn.agents.tools.function_tool import FunctionTool
from domyn.agents.tracing import add_trace_headers


class RestFunctionTool(FunctionTool):
    """A tool that makes REST API calls."""

    type: Literal["rest"] = "rest"
    url: str
    name: str
    description: str = ""
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH", "UPDATE"]
    headers: dict[str, str | dict] = Field(
        default={}, description="Optional headers to include in the request"
    )
    body: list[Param] | None = Field(
        default_factory=list, description="The body of the request (for POST, PUT, PATCH)"
    )
    _httpx_client: httpx.AsyncClient | None = PrivateAttr(default=None)

    func: Callable[..., Any] | None = None

    def __call__(self, *args, **kwargs):
        return self.callable_function(*args, **kwargs)

    @property
    def url_params(self) -> dict[str, Any]:
        # extract url params from url
        parsed_url = httpx.URL(self.url)
        url_params = dict(parsed_url.params.items())
        return url_params

    def model_post_init(self, context: Any, /) -> None:
        if self._httpx_client is None:
            self._httpx_client = httpx.AsyncClient()

        def _wrap_rest_call():
            async def _inner(*args: Any, **kwargs: Any) -> str:
                # Per-call metadata (e.g. an auth token minted by a PRE_TOOL_CALL hook),
                # forwarded as extra headers — never part of the request body.
                call_metadata = kwargs.pop("call_metadata", None) or {}

                request_params = {}
                if self.body:
                    # Also the only filter when the tool is called directly (tool(...)), bypassing
                    # execute().
                    for param in self.body:
                        if param.name in kwargs:
                            request_params[param.name] = kwargs[param.name]

                config_params = {param.name: param.value for param in self.config_params}
                request_params = {**request_params, **config_params}
                headers = add_trace_headers(self.headers)
                for key, value in call_metadata.items():
                    # httpx requires string header values; drop blank ones rather
                    # than send a header some servers/proxies treat inconsistently.
                    str_value = str(value).strip()
                    if str_value:
                        headers[key] = str_value
                if self.method in {"POST", "PUT", "PATCH"}:
                    response = await self._httpx_client.request(
                        method=self.method,
                        url=self.url,
                        json=request_params,
                        headers=headers,
                        params=self.url_params,
                    )
                else:
                    response = await self._httpx_client.request(
                        method=self.method,
                        url=self.url,
                        headers=headers,
                        params=self.url_params | request_params,
                    )
                response.raise_for_status()
                # A non-JSON body (text, empty 204) raises here, so the call is reported as
                # failed; reading `response.text` instead would report the body.
                return response.json()

            # update signature
            sig_params = [param.name for param in self.body] if self.body else []
            _inner.__signature__ = inspect.Signature(
                parameters=[
                    inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                    for name in sig_params
                ]
                + [inspect.Parameter("call_metadata", inspect.Parameter.KEYWORD_ONLY, default=None)]
            )

            return _inner

        self._actual_callable = _wrap_rest_call()

        serialized = serialize_callable(self._actual_callable)
        object.__setattr__(self, "func", serialized)

        if self.uid is None:
            object.__setattr__(self, "uid", self.name)

    @cached_property
    def declaration(self):
        return FunctionDeclaration(
            name=self.__dict__["name"],  # avoid recursion
            description=self.__dict__["description"],
            parameters=self.body,
        )

    # ---------------------------------------------------------------------
    # Factory helpers
    # ---------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: dict, save_in_registry: bool = False) -> "RestFunctionTool":
        from domyn.agents.hooks import HookFactory

        hooks = []
        for hook in config.get("hooks", []):
            if isinstance(hook, dict):
                if hook["trigger"] == HookPhase.PRE_TOOL_CALL:
                    hooks.append(
                        HookFactory.from_config(hook, trigger=HookPhase.PRE_TOOL_CALL)
                        if isinstance(hook, dict)
                        else hook
                    )
                elif hook["trigger"] == HookPhase.POST_TOOL_CALL:
                    hooks.append(
                        HookFactory.from_config(hook, trigger=HookPhase.POST_TOOL_CALL)
                        if isinstance(hook, dict)
                        else hook
                    )
                elif hook["trigger"] == HookPhase.PRE_TOOL_APPROVAL:
                    hooks.append(
                        HookFactory.from_config(hook, trigger=HookPhase.PRE_TOOL_APPROVAL)
                        if isinstance(hook, dict)
                        else hook
                    )
                else:
                    raise ValueError(f"Unsupported hook trigger: {hook['trigger']}")
            else:
                hooks.append(hook)

        config.update(
            {
                "func": None,
                "hooks": hooks,
            }
        )

        tool = cls(**config)
        if save_in_registry:
            from domyn.agents.registry import get_tool_registry

            get_tool_registry().register_tool(config.get("name", tool.name), tool)
        return tool


__all__ = ["RestFunctionTool"]
