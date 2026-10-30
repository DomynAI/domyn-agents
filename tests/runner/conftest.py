import pytest

from domyn.agents import Runner
from domyn.agents.llm import BaseLLMProvider, LLMMessage, LLMResponse, StructuredOutputType


@pytest.fixture
def runner() -> Runner:
    return Runner()


@pytest.fixture
def llm_provider() -> BaseLLMProvider:
    """Fixture for a mock LLM provider."""

    class MockLLMProvider(BaseLLMProvider):
        type: str = "mock"
        name: str = "mock-llm"
        model_name: str = "mock-model"
        supported_structured_outputs: None = None

        async def generate(self, messages: LLMMessage, **kwargs):
            pass

        async def generate_streaming(self, messages: LLMMessage, **kwargs):
            yield

    return MockLLMProvider()
