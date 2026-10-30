"""LLM provider system for domyn agents framework."""

import re
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from enum import StrEnum, auto
from typing import Any, ClassVar

from pydantic import BaseModel, Field, model_validator

from domyn.agents.llm.context_window import models_url, window_from_models_listing


class StructuredOutputType(StrEnum):
    JSON_SCHEMA = auto()
    LARK = auto()
    GBNF = auto()
    LLGUIDANCE = auto()


class StructuredOutput(BaseModel):
    type: StructuredOutputType
    rules: dict | str


class ToolCallFunction(BaseModel):
    """Function details within a tool call."""

    name: str = Field(..., description="Name of the function to call")
    arguments: str = Field(..., description="JSON-encoded arguments for the function")


class LLMToolCall(BaseModel):
    """A tool call request from the LLM."""

    id: str = Field(..., description="Unique identifier for this tool call")
    type: str = Field(default="function", description="Type of tool call")
    function: ToolCallFunction = Field(..., description="Function to call")


class LLMMessage(BaseModel):
    """Message structure for LLM communication."""

    role: str = Field(..., description="Role of the message sender (system, user, assistant, tool)")
    content: str | list[dict[str, Any]] | None = Field(
        default=None, description="Content of the message"
    )
    tool_calls: list[LLMToolCall] | None = Field(
        default=None, description="Tool calls made by the assistant"
    )
    tool_call_id: str | None = Field(
        default=None, description="ID of the tool call this message is a result for"
    )
    name: str | None = Field(default=None, description="Tool name for tool result messages")

    source_event_id: str | None = Field(
        default=None,
        exclude=True,
        description=(
            "Event this message was rendered from, when the renderer knows it. "
            "Excluded from serialization: it is our bookkeeping, not part of the "
            "wire protocol, and a provider receiving an unknown field rejects it. "
            "Needed because a context policy that shortens the rendered history "
            "has to record *where* it cut, and a message otherwise carries no "
            "link back to the event that produced it. `None` is fine — a "
            "renderer that does not set it simply cannot be cut precisely."
        ),
    )
    read_back: str | None = Field(
        default=None,
        exclude=True,
        description=(
            "The call that reads this message's tool result back, set by the "
            "context manager when the agent can make it: what compaction offers "
            "in place of a result it drops. Excluded from serialization, like "
            "`source_event_id`. `None` on every other message, including the "
            "tool-role messages that carry a sub-agent's task or report: no call "
            "reads those."
        ),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert message to a dictionary format suitable for LLM input."""
        result: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            result["content"] = self.content
        if self.tool_calls:
            result["tool_calls"] = [call.model_dump() for call in self.tool_calls]
        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            result["name"] = self.name
        return result


class LLMResponse(BaseModel):
    """Response from LLM provider."""

    content: str = Field(..., description="Generated content")
    usage: dict[str, Any] = Field(default_factory=dict, description="Usage statistics")
    generated_tokens: int
    execution_time: float  # in seconds
    tool_calls: list[LLMToolCall] = Field(
        default_factory=list, description="Tool calls requested by the LLM"
    )
    reasoning_content: str | None = Field(
        default=None, description="Reasoning content (DeepSeek R1, Kimi)"
    )
    thinking_blocks: list[dict[str, Any]] | None = Field(
        default=None, description="Thinking blocks (Anthropic extended thinking)"
    )
    logprobs: dict[str, Any] | None = Field(
        default=None, description="Token log-probabilities, when requested"
    )
    finish_reason: str | None = Field(
        default=None,
        description=(
            "Why generation stopped, as the provider reported it: 'stop', "
            "'length', 'tool_calls'. Worth carrying because 'length' is the "
            "difference between a model that made a mistake and a model that "
            "was cut off mid-sentence — and those need opposite responses."
        ),
    )

    @property
    def truncated(self) -> bool:
        """Generation hit a limit rather than finishing.

        Anything built from a truncated response is incomplete by definition: a
        half-written JSON argument, a sentence that stops. Reading it as a
        malformed answer sends the caller looking for a mistake that was never
        made.
        """
        return self.finish_reason == "length"


def log_inference_request(
    logger: Any, url: str, payload: dict[str, Any], *, stream: bool = False
) -> None:
    """Log an outbound LLM API request.

    Request metadata goes to info; the full payload (messages, system prompt,
    tool schemas) only to debug so conversation content never lands in
    default-level logs.
    """
    marker = "[Inference Stream Request]" if stream else "[Inference Request]"
    messages = payload.get("messages") or payload.get("input") or []
    logger.info(
        f"{marker}: {url}",
        payload={
            "model": payload.get("model"),
            "messages_count": len(messages),
            "tools_count": len(payload.get("tools") or []),
            "stream": stream,
        },
    )
    logger.debug(f"{marker} full payload: {url}", payload=payload)


def log_inference_response(logger: Any, url: str, payload: dict[str, Any]) -> None:
    """Log an LLM API response.

    Usage metadata — the provider's own token counts included — goes to info;
    the full payload (response text, reasoning, tool-call arguments) only to
    debug so generated content never lands in default-level logs.
    """
    tool_calls = payload.get("tool_calls") or []
    logger.info(
        f"[Inference Response]: {url}",
        payload={
            "generated_tokens": payload.get("generated_tokens"),
            "execution_time": payload.get("execution_time"),
            "response_length": len(payload.get("response") or ""),
            "tool_call_names": [
                (tc.get("function") or {}).get("name") for tc in tool_calls if isinstance(tc, dict)
            ],
            "usage": payload.get("usage") or {},
        },
    )
    logger.debug(f"[Inference Response] full payload: {url}", payload=payload)


_VLLM_ENGINE_TOO_LONG = r"decoder prompt \(length \d+\)[^.]*maximum model length"
"""vLLM's engine, for a prompt that got past the "maximum context length" check —
multimodal tokens, or exactly the window's length: "The decoder prompt (length M) …
maximum model length of N". The gateway passes it on as a plain ``BadRequestError``.
Its encoder twin is left out: the number there is the encoder's own budget
(``mm_encoder_cache_size``), not the window, and its input is not the conversation
compaction shortens."""

_CONTEXT_OVERFLOW = re.compile(
    r"maximum context length|context_length_exceeded|contextwindowexceeded"
    rf"|exceeds the context window|prompt is too long|{_VLLM_ENGINE_TOO_LONG}",
    re.IGNORECASE,
)
"""How providers word "the prompt does not fit": vLLM and OpenRouter ("maximum
context length is N tokens"), vLLM's engine (`_VLLM_ENGINE_TOO_LONG`), the LiteLLM
gateway (``ContextWindowExceededError``), OpenAI (``context_length_exceeded``,
"exceeds the context window"), Anthropic ("prompt is too long")."""


def _rejection_body(error: BaseException) -> str | None:
    """The body of a provider's 400 or 413, or `None` when there is none to read."""
    response = getattr(error, "response", None)
    if response is None or getattr(response, "status_code", None) not in (400, 413):
        return None
    try:
        return str(response.text)
    except Exception:
        # A streamed body nobody read: there is nothing to go on.
        return None


def is_context_overflow(error: BaseException) -> bool:
    """Whether the provider rejected a request because the prompt did not fit.

    Read off the HTTP error every provider raises: a 400 or 413 whose body says
    so. A request for more output than the model allows is a different error,
    and no compaction can fix it.
    """
    body = _rejection_body(error)
    return body is not None and bool(_CONTEXT_OVERFLOW.search(body))


_CONTEXT_LIMIT = (
    re.compile(r"maximum context length is (\d+) tokens", re.IGNORECASE),
    re.compile(r"maximum context length \((\d+)\)", re.IGNORECASE),
    re.compile(r"prompt is too long: \d+ tokens > (\d+) maximum", re.IGNORECASE),
    re.compile(rf"{_VLLM_ENGINE_TOO_LONG} of (\d+)", re.IGNORECASE),
    re.compile(r"Max Input Tokens=(\d+)", re.IGNORECASE),
)
"""Where a rejection states the window: vLLM, OpenAI and the OpenAI-compatible
family ("maximum context length is N tokens", also through the LiteLLM gateway;
OpenRouter says "This endpoint's" where vLLM says "This model's"), vLLM's later
check ("Input length (M) exceeds model's maximum context length (N)": the only one a
gpt-oss prompt meets, since vLLM renders Harmony without the first), Anthropic
("prompt is too long: M tokens > N maximum"), vLLM's engine ("… maximum model length
of N"), and the LiteLLM gateway's own pre-call check ("Max Input Tokens=N", once for
each deployment of the group it ruled out)."""


def context_limit_from_error(error: BaseException) -> int | None:
    """The model's window, when a rejection as too long states it.

    The last source of a window, for one nobody declared and the provider did not
    report (`llm/context_window.py`): a gateway that drops the server's
    `max_model_len`, an endpoint with no models listing.

    The widest one the body states: LiteLLM's pre-call check lists every deployment
    of the group it ruled out, and a request that fits the widest is routed there.
    Adoption only ever lowers a window, so one read too wide is corrected by the
    next rejection; one read too narrow never is.
    """
    body = _rejection_body(error)
    if body is None or not _CONTEXT_OVERFLOW.search(body):
        return None
    for pattern in _CONTEXT_LIMIT:
        if windows := pattern.findall(body):
            return max(int(window) for window in windows)
    return None


def reported_prompt_tokens(usage: dict[str, Any] | None) -> int | None:
    """The whole prompt as the provider counted it, or `None` when it did not say.

    `prompt_tokens` (chat completions: vLLM, OpenAI, Cerebras, OpenRouter) already
    includes cached tokens, and so does `input_tokens` on OpenAI's Responses API.
    Anthropic's `input_tokens` does not: with a cache in play it is only the
    uncached tail — measured, 4 tokens of a prompt of 8,561 — and the two cache
    counts are the rest.
    """
    if not usage:
        return None
    if prompt := _token_count(usage.get("prompt_tokens")):
        return prompt
    total = sum(
        _token_count(usage.get(key))
        for key in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    )
    return total or None


def _token_count(value: Any) -> int:
    """A positive count, else 0. `bool` is an `int`, and is not a count."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


class BaseLLMProvider(BaseModel, ABC):
    """Abstract base class for LLM providers."""

    type: str = "base"
    name: str | None = Field(default=None, description="Unique name of the LLM provider instance")
    model_name: str = Field(..., description="Name of the LLM model")

    # Subclasses should define this as a class variable with default generation params
    default_generation_params: ClassVar[dict[str, Any]] = {}
    supported_structured_outputs: list[StructuredOutputType] | None

    context_window: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Total token window of the model. `None` means not declared: the runner "
            "then asks the provider (`fetch_context_window`) and, for hosted APIs that "
            "report nothing, the models.dev catalogue (`context_window_catalog`); a "
            "rejection stating the window teaches it too. Unknown stays usable — every "
            "context-window policy is inert without it."
        ),
    )

    context_window_catalog: ClassVar[str | None] = None
    """The models.dev provider whose catalogue knows this provider's windows by model
    id. Only for hosted APIs: on a self-hosted endpoint the window is how the server was
    started, and the name of the weights says nothing about it."""

    max_sensitivity: str | None = Field(
        default=None,
        description=(
            "Highest data-sensitivity level this provider may be sent: "
            "`public`, `internal`, `confidential` or `regulated`. `None` — the "
            "default — means no ceiling and no check, so an existing config is "
            "unaffected. Declaring one opts this provider into the sensitivity "
            "gate (`decisions/sensitivity.py`), which refuses a turn rated "
            "above it. It is a deployment's data boundary expressed as "
            "configuration rather than as discipline."
        ),
    )

    append_text_system_prompt: str | None = Field(
        default=None, description="System prompt text to append to each request"
    )
    append_text_last_message: str | None = Field(
        default=None, description="Text to append to the last user message"
    )

    @model_validator(mode="before")
    @classmethod
    def merge_generation_params(cls, data: Any) -> Any:
        """Merge provided generation_params with default values from the class."""
        if isinstance(data, dict) and "generation_params" in data:
            # Get the default generation params from the class
            defaults = cls.default_generation_params.copy()
            # Merge with provided params (provided params take precedence)
            provided_params = data.get("generation_params")
            if provided_params is not None:
                defaults.update(provided_params)
            data["generation_params"] = defaults
        return data

    @abstractmethod
    async def generate(
        self,
        messages: list[LLMMessage],
        structured_outputs: StructuredOutput | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse: ...

    @abstractmethod
    async def generate_streaming(
        self, messages: list[LLMMessage], **kwargs: Any
    ) -> AsyncGenerator[LLMResponse, None]:
        """Asynchronously generate content from messages."""

    async def fetch_context_window(self) -> int | None:
        """The window this provider's endpoint reports for ``model_name``, or ``None``.

        By default the ``/models`` listing beside an OpenAI-style ``url``: what vLLM,
        LiteLLM proxies, OpenRouter and Cerebras serve. May raise on a failed request;
        ``resolve_context_window`` reads any failure as "not reported".
        """
        url = models_url(getattr(self, "url", None))
        if url is None:
            return None
        return await window_from_models_listing(
            url, self._context_window_headers(), self.model_name
        )

    def _context_window_headers(self) -> dict[str, str]:
        """Credentials for the ``/models`` request: the bearer key, when there is one."""
        api_key = getattr(self, "api_key", None)
        return (
            {"Authorization": f"Bearer {api_key}"} if isinstance(api_key, str) and api_key else {}
        )


__all__ = [
    "BaseLLMProvider",
    "LLMMessage",
    "LLMResponse",
    "LLMToolCall",
    "StructuredOutputType",
    "ToolCallFunction",
]
