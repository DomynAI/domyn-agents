from pathlib import Path

import pytest
import yaml

from domyn.agents.loader import load_from_file


def _write_yaml(tmp_path: Path, data: dict) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


def test_resolves_a_single_env_var_in_a_scalar_field(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "secret-value")
    file_path = _write_yaml(tmp_path, {"api_key": "${VLLM_API_KEY}"})

    config = load_from_file(file_path)

    assert config["api_key"] == "secret-value"


def test_resolves_an_env_var_embedded_within_a_larger_string(tmp_path, monkeypatch):
    monkeypatch.setenv("HOST", "llm.internal")
    file_path = _write_yaml(tmp_path, {"url": "http://${HOST}:8080/v1"})

    config = load_from_file(file_path)

    assert config["url"] == "http://llm.internal:8080/v1"


def test_resolves_env_vars_nested_in_dicts_and_lists(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "secret-value")
    monkeypatch.setenv("GH_TOKEN", "gh-secret")
    file_path = _write_yaml(
        tmp_path,
        {
            "llm_providers": [{"name": "p", "api_key": "${VLLM_API_KEY}"}],
            "tools": [{"name": "t", "headers": {"Authorization": "Bearer ${GH_TOKEN}"}}],
        },
    )

    config = load_from_file(file_path)

    assert config["llm_providers"][0]["api_key"] == "secret-value"
    assert config["tools"][0]["headers"]["Authorization"] == "Bearer gh-secret"


def test_raises_when_referenced_env_var_is_not_set(tmp_path, monkeypatch):
    monkeypatch.delenv("MISSING_VAR", raising=False)
    file_path = _write_yaml(tmp_path, {"api_key": "${MISSING_VAR}"})

    with pytest.raises(ValueError, match="MISSING_VAR"):
        load_from_file(file_path)


def test_non_string_leaf_values_pass_through_unchanged(tmp_path):
    file_path = _write_yaml(
        tmp_path,
        {"temperature": 0.7, "min_events_for_summary": 5, "stream": True, "reference": None},
    )

    config = load_from_file(file_path)

    assert config["temperature"] == 0.7
    assert config["min_events_for_summary"] == 5
    assert config["stream"] is True
    assert config["reference"] is None


def test_a_literal_dollar_sign_without_braces_is_left_alone(tmp_path):
    file_path = _write_yaml(tmp_path, {"description": "costs $5"})

    config = load_from_file(file_path)

    assert config["description"] == "costs $5"
