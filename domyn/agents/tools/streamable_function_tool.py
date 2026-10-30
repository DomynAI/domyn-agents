import inspect
from collections.abc import AsyncGenerator
from typing import Any, Literal

from pydantic import BaseModel, Field

from domyn.agents.core import Part
from domyn.agents.tools.base import ToolResult, UpdateToolResult
from domyn.agents.tools.function_tool import FunctionTool


class StreamableToolResult(BaseModel):
    """Streamable tool result"""

    data: Part | list[Part] = Field(..., description="The part data for streaming")


class StreamableFunctionTool(FunctionTool):
    """Wrapper for async generator that can stream directly output"""

    type: Literal["StreamableFunctionTool"] = "StreamableFunctionTool"

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        if not self._is_async_generator:
            raise ValueError("Function must be an async generator function")

    async def execute(
        self, **kwargs: Any
    ) -> AsyncGenerator[StreamableToolResult | UpdateToolResult | ToolResult, ToolResult]:
        """Call the generator function."""

        state = kwargs.get("state")

        kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in list(inspect.signature(self.callable_function).parameters.keys())
        }

        config_params = {param.name: param.value for param in self.config_params}

        async for result in self.callable_function(**{**kwargs, **config_params}):
            if isinstance(result, StreamableToolResult | UpdateToolResult):
                yield result
            elif not isinstance(result, ToolResult):
                # Cast a ToolResult-shaped dict (or a legacy "result" envelope)
                # directly instead of burying it in the `data` attribute.
                coerced = ToolResult.from_result_dict(
                    result, result_key_fallback=True, default_state=state
                )
                # A non-string result is stringified with str() (Python repr); FunctionTool
                # keeps it as data.
                result = (
                    coerced
                    if coerced is not None
                    else ToolResult(data=str(result), success=True, state=state)
                )

        if not isinstance(result, ToolResult):
            result = ToolResult(
                data=[{k: v for k, v in p.model_dump().items() if v} for p in result.data]
                if isinstance(result.data, list)
                else {k: v for k, v in result.data.model_dump().items() if v},
                success=True,
            )

        yield result
