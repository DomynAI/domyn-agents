"""Which of models.dev's limits the catalogue keeps (`scripts/update_context_windows.py`)."""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "update_context_windows.py"
_spec = importlib.util.spec_from_file_location("update_context_windows", _SCRIPT)
assert _spec is not None and _spec.loader is not None
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)


@pytest.mark.parametrize(
    ("limit", "kept"),
    [
        ({"context": 400_000, "input": 272_000, "output": 128_000}, 272_000),
        ({"context": 128_000, "output": 16_384}, 128_000),
        ({"context": 0, "output": 0}, None),
    ],
    ids=["prompt-limit", "whole-window", "no-window"],
)
def test_the_prompt_limit_is_kept_where_models_dev_states_one(
    limit: dict, kept: int | None
) -> None:
    """gpt-5: 400k window, 272k of prompt. Compaction measures the prompt."""
    assert script.window(limit) == kept
