"""DelegateLLMProvider — signals that LLM execution should be handled by the runner.

When a planner detects a DelegateLLMProvider it uses the bidirectional generator
protocol: it yields LLM_START with an LLMCallAction payload instead of calling
the provider directly, and receives the LLMResponse back from the runner via asend().

Two modes:

* **Wrapped provider** (single-runner, runner controls LLM):
  ``DelegateLLMProvider(wrapped=real_provider)``
  The local runner intercepts LLM_START and executes via ``wrapped``.
  Useful for centralising retry logic, rate limiting, or tracing.

* **Pure delegate** (runner-to-runner):
  ``DelegateLLMProvider()``
  The local runner forwards LLM_START to the parent runner via transport.
  The parent runner executes with its own provider and sends back LLM_END.
  (Transport back-delegation is implemented in Step 4.)
"""

from collections.abc import AsyncGenerator
from typing import Any, ClassVar

from pydantic import Field

from domyn.agents.llm.base import BaseLLMProvider, LLMMessage, LLMResponse, StructuredOutputType


class DelegateLLMProvider(BaseLLMProvider):
    """LLM provider that delegates execution to the runner."""

    type: str = "delegate"
    model_name: str = "delegate"
    supported_structured_outputs: list[StructuredOutputType] | None = Field(default=None)

    default_generation_params: ClassVar[dict[str, Any]] = {}

    # Optional: wraps a real provider for single-runner delegation.
    # If None, the runner forwards the call to a parent runner via transport.
    wrapped: BaseLLMProvider | None = Field(
        default=None,
        description="Real provider used when the local runner handles the LLM call.",
    )

    async def generate(
        self,
        messages: list[LLMMessage],
        structured_outputs: Any | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        raise RuntimeError(
            "DelegateLLMProvider.generate() must not be called directly. "
            "The runner intercepts LLM_START and executes via the wrapped provider "
            "or forwards to a parent runner."
        )

    async def generate_streaming(
        self, messages: list[LLMMessage], **kwargs: Any
    ) -> AsyncGenerator[LLMResponse, None]:
        raise RuntimeError("DelegateLLMProvider.generate_streaming() must not be called directly.")
        yield  # make Python treat this as a generator
