"""A tool entry the loader cannot build is skipped, and says so."""

import logging

from domyn.agents.loader import ConfigurationV1

_LLM = {
    "name": "test-provider",
    "type": "vllm",
    "model_name": "test-model",
    "url": "http://localhost:8000/v1/chat/completions",
}


def _config(tool_type: str) -> dict:
    return {
        "version": "1.0",
        "llm_providers": [_LLM],
        "tools": [
            {
                "name": "get_order_info",
                "type": tool_type,
                "path": "tests/loader/customer_service_tools.py",
            }
        ],
        "agents": [
            {
                "name": "TestAgent",
                "description": "Test agent",
                "instruction": "You are a test agent",
                "llm_provider": "test-provider",
                "tools": ["get_order_info"],
            }
        ],
    }


def test_an_unknown_tool_type_is_skipped_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING):
        result = ConfigurationV1.from_dict(_config("Function"))

    assert result.tools == []
    assert result.agents[0].tools == []
    warning = next(r for r in caplog.records if r.levelno == logging.WARNING)
    assert "get_order_info" in warning.getMessage()
    assert "Function" in warning.getMessage()


def test_a_known_tool_type_logs_nothing(caplog):
    with caplog.at_level(logging.WARNING):
        result = ConfigurationV1.from_dict(_config("function"))

    assert [t.name for t in result.tools] == ["get_order_info"]
    assert not [r for r in caplog.records if "Skipping tool" in r.getMessage()]
