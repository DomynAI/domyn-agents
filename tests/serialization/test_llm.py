from domyn.agents.llm import LLMMessage, LLMResponse, OpenAIProvider, VLLMProvider


def test_vllm_serialization():
    provider = VLLMProvider(
        model_name="test-model",
        generation_params={"max_length": 1000},
        url="https://llm.example.com/v1/chat/completions",
    )
    serialized = provider.model_dump()
    deserialized = VLLMProvider.from_config(serialized)
    assert deserialized == provider
    assert deserialized.name == "test-model"
    assert deserialized.generation_params["max_length"] == 1000


def test_openai_serialization():
    provider = OpenAIProvider(
        model_name="gpt-4", api_key="test-key", generation_params={"temperature": 0.5}
    )
    serialized = provider.model_dump()
    deserialized = OpenAIProvider(**serialized)
    assert deserialized == provider
    assert deserialized.api_key == "test-key"
    assert deserialized.generation_params["temperature"] == 0.5
    assert deserialized.name == "gpt-4"
