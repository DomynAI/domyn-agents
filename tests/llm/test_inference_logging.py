"""Anti-regression tests for inference logging.

Guards the privacy contract of log_inference_request / log_inference_response:
conversation content (messages, system prompt, response text, tool-call
arguments) must never appear at info level — only usage metadata does. The
full payload is available at debug only.
"""

import httpx
import pytest
import respx

import domyn.agents.logger as logger_module
from domyn.agents.llm import VLLMProvider
from domyn.agents.llm.base import LLMMessage, log_inference_request, log_inference_response

URL = "https://llm.test/v1/chat/completions"
USER_SECRET = "super-secret-user-question"
ASSISTANT_SECRET = "super-secret-assistant-answer"
TOOL_ARG_SECRET = "super-secret-tool-argument"


class RecordingLogger:
    """Minimal LoggerProtocol impl that records (level, message, payload)."""

    def __init__(self):
        self.records: list[tuple[str, str, dict | None]] = []

    def _log(self, level, message, payload=None, **kwargs):
        self.records.append((level, message, payload))

    def info(self, message, payload=None, **kwargs):
        self._log("info", message, payload)

    def warning(self, message, payload=None, **kwargs):
        self._log("warning", message, payload)

    def error(self, message, payload=None, **kwargs):
        self._log("error", message, payload)

    def debug(self, message, payload=None, **kwargs):
        self._log("debug", message, payload)

    def critical(self, message, payload=None, **kwargs):
        self._log("critical", message, payload)

    def dump(self, level):
        return " | ".join(f"{m} {p}" for lv, m, p in self.records if lv == level)


@pytest.fixture
def recording_logger():
    prev = logger_module._library_logger
    rec = RecordingLogger()
    logger_module.set_logger(rec)
    yield rec
    logger_module._library_logger = prev


class TestLogInferenceRequest:
    def test_info_has_metadata_but_no_content(self, recording_logger):
        payload = {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "system prompt secret"},
                {"role": "user", "content": USER_SECRET},
            ],
            "tools": [{"type": "function", "function": {"name": "t1"}}],
        }
        log_inference_request(logger_module.get_logger(), URL, payload)

        info = recording_logger.dump("info")
        assert "[Inference Request]" in info
        assert USER_SECRET not in info
        assert "system prompt secret" not in info
        assert "'messages_count': 2" in info
        assert "'tools_count': 1" in info
        assert "'model': 'test-model'" in info

    def test_debug_carries_full_payload(self, recording_logger):
        payload = {"model": "m", "messages": [{"role": "user", "content": USER_SECRET}]}
        log_inference_request(logger_module.get_logger(), URL, payload)

        debug = recording_logger.dump("debug")
        assert "[Inference Request] full payload" in debug
        assert USER_SECRET in debug

    def test_stream_marker(self, recording_logger):
        log_inference_request(logger_module.get_logger(), URL, {"messages": []}, stream=True)
        assert "[Inference Stream Request]" in recording_logger.dump("info")

    def test_openai_responses_api_input_key_counts(self, recording_logger):
        payload = {"model": "m", "input": [{"role": "user", "content": USER_SECRET}]}
        log_inference_request(logger_module.get_logger(), URL, payload)
        assert "'messages_count': 1" in recording_logger.dump("info")


class TestLogInferenceResponse:
    def test_info_has_usage_but_no_content(self, recording_logger):
        payload = {
            "response": ASSISTANT_SECRET,
            "generated_tokens": 7,
            "execution_time": 0.5,
            "tool_calls": [
                {"id": "1", "function": {"name": "search", "arguments": TOOL_ARG_SECRET}}
            ],
        }
        log_inference_response(logger_module.get_logger(), URL, payload)

        info = recording_logger.dump("info")
        assert "[Inference Response]" in info
        assert ASSISTANT_SECRET not in info
        assert TOOL_ARG_SECRET not in info
        assert "'generated_tokens': 7" in info
        assert f"'response_length': {len(ASSISTANT_SECRET)}" in info
        assert "'tool_call_names': ['search']" in info

    def test_debug_carries_full_payload(self, recording_logger):
        payload = {"response": ASSISTANT_SECRET, "tool_calls": []}
        log_inference_response(logger_module.get_logger(), URL, payload)

        debug = recording_logger.dump("debug")
        assert "[Inference Response] full payload" in debug
        assert ASSISTANT_SECRET in debug


class TestProviderWiring:
    """End-to-end through a real provider call: content stays out of info."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_vllm_generate_leaks_nothing_at_info(self, recording_logger):
        respx.post(URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": ASSISTANT_SECRET}}],
                    "usage": {"completion_tokens": 3},
                },
            )
        )
        provider = VLLMProvider(url=URL, model_name="test-model", api_key="k")
        await provider.generate([LLMMessage(role="user", content=USER_SECRET)])

        info = recording_logger.dump("info")
        assert "[Inference Request]" in info
        assert "[Inference Response]" in info
        assert USER_SECRET not in info
        assert ASSISTANT_SECRET not in info

        debug = recording_logger.dump("debug")
        assert USER_SECRET in debug
        assert ASSISTANT_SECRET in debug
