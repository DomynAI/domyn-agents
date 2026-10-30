"""Fixtures for the runner-level decision tests.

Deliberate copies of what `tests/runner/conftest.py` provides: a mock provider
that returns nothing (the agent's `_execute` is replaced wholesale, so the LLM
is never called) and the generator helper the flow tests use. Importing across
test packages would couple these to a conftest they do not own.
"""

from collections.abc import AsyncGenerator

import pytest

from domyn.agents import Runner
from domyn.agents.llm import BaseLLMProvider, LLMMessage


@pytest.fixture
def runner() -> Runner:
    return Runner()


@pytest.fixture
def llm_provider() -> BaseLLMProvider:
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


def async_gen_from_list(values):
    """Replay a fixed list of events as an agent's `_execute`, once."""
    _iter = iter(values)
    finished = False

    async def _gen(*args, **kwargs) -> AsyncGenerator:
        nonlocal finished
        if finished:
            raise StopAsyncIteration
        for value in _iter:
            yield value
        finished = True
        return

    return _gen
