"""Base logic implementation for planner strategies.

This module purposefully avoids importing the concrete *Agent* implementation at
runtime to prevent circular import issues (Agent -> planner_strategy -> Agent).
Only lightweight core/llm types are imported at import time; Agent types are
referenced via forward annotations and imported under TYPE_CHECKING.
"""

from abc import abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

from pydantic import BaseModel

from domyn.agents.core import BaseAgent, BaseEvent, InteractionContext
from domyn.agents.llm import BaseLLMProvider, LLMMessage
from domyn.agents.planner_strategy.context_management.base import MessageContextManager


class BasePlannerStrategy(BaseModel):
    """Decides the next actions while the agent runs.

    It calls the LLM provider for inference and emits events, possibly several
    actions per step.
    """

    type: str = "base"
    name: str
    context_manager: MessageContextManager
    max_iterations: int = 25
    verbose: bool = False
    forward_verbatim_enabled: bool = True

    @abstractmethod
    async def execute(
        self,
        agent: BaseAgent,
        context: InteractionContext,
        *,
        skills_payload: str = "",
        instruction: str | None = None,
    ) -> AsyncGenerator[BaseEvent, Any]:
        """Execute the agent logic, yielding events and optionally receiving LLM responses.

        ``skills_payload`` is the rendered skills block the *agent* resolved for
        this entry, or ``""``. ``instruction`` is the agent's instruction with its
        standing instructions merged in, rendered into the system prompt in place
        of ``agent.instruction``; ``None`` means use ``agent.instruction``.

        When the agent uses a DelegateLLMProvider, the planner yields LLM_START
        with an LLMCallAction payload instead of calling the provider directly.
        The runner intercepts it, executes the LLM call, and sends back an
        LLMResponse via asend(). The planner receives it as the return value of
        ``response = yield llm_start_event``.

        For all other event types, the sent value is None and can be ignored.
        """
        ...

    async def invoke_llm(
        self, llm_provider: BaseLLMProvider, prompt: str, context: InteractionContext, **kwargs: Any
    ) -> str:
        """Call the LLM provider with a single prompt."""
        messages = [LLMMessage(role="user", content=prompt)]
        response = await llm_provider.generate(messages, **kwargs)
        return response
