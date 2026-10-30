"""Tests for JSON escape-sequence handling in partial_pattern_matching."""

from collections.abc import AsyncGenerator

import pytest

from domyn.agents.llm import LLMResponse
from domyn.agents.planner_strategy.prompting.utils import partial_pattern_matching

# Simple 1-level pattern: extract the value of a JSON "text" key.
_START = [r'"text"\s*:\s*"']
_END = ['"']
_NEG = ["\\"]


async def _stream(chunks: list[str]) -> AsyncGenerator[LLMResponse, None]:
    for c in chunks:
        yield LLMResponse(content=c, generated_tokens=1, execution_time=0)


async def _collect(chunks: list[str]) -> str:
    result = ""
    async for item in partial_pattern_matching(
        _stream(chunks),
        start_pattern_conditions=_START,
        end_pattern_conditions=_END,
        end_pattern_negative_lookaheads=_NEG,
    ):
        if isinstance(item, str):
            result += item
    return result


# ---------------------------------------------------------------------------
# Parametrised cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "chunks, expected",
    [
        # Plain text — no escapes
        (['{"text": "hello"}'], "hello"),
        # Standard single-character escapes in one chunk
        (['{"text": "line1\\nline2"}'], "line1\nline2"),
        (['{"text": "col1\\tcol2"}'], "col1\tcol2"),
        (['{"text": "back\\\\slash"}'], "back\\slash"),
        # Unicode escape in one chunk
        (['{"text": "caf\\u00e9"}'], "café"),
        # \n split: backslash in chunk 1, 'n' in chunk 2
        (['"text": "line1\\', 'nline2"'], "line1\nline2"),
        # \t split: backslash in chunk 1, 't' in chunk 2
        (['"text": "col1\\', 'tcol2"'], "col1\tcol2"),
        # \\ split: first backslash in chunk 1, second in chunk 2
        (['"text": "back\\', '\\slash"'], "back\\slash"),
        # \u00e9 split at every position
        (['"text": "caf\\', 'u00e9"'], "café"),
        (['"text": "caf\\u', '00e9"'], "café"),
        (['"text": "caf\\u0', '0e9"'], "café"),
        (['"text": "caf\\u00', 'e9"'], "café"),
        (['"text": "caf\\u00e', '9"'], "café"),
        # Multiple escapes, all split across chunks
        (['"text": "line1\\', "n\\", 'nline2"'], "line1\n\nline2"),
        # Escaped quote: \" should not end the match
        (['"text": "say \\"hi\\"!"'], 'say "hi"!'),
        # Escaped quote split: backslash arrives before the quote
        (['"text": "say \\', '"hi\\', '"!"'], 'say "hi"!'),
        # Complex mixed escapes, each \ and " in its own token:
        #   JSON body:  print(\"He said \\\"Hi\\\"\") and this is it
        #   decoded:    print("He said \"Hi\"") and this is it
        (
            [
                '"text": "print(',
                "\\",
                '"',  # \" → "
                "He said ",
                "\\",
                "\\",
                "\\",
                '"',  # \\\" → \"
                "Hi",
                "\\",
                "\\",
                "\\",
                '"',  # \\\" → \"
                "\\",
                '"',  # \" → "
                ') and this is it"',
            ],
            'print("He said \\"Hi\\"") and this is it',
        ),
    ],
)
@pytest.mark.asyncio
async def test_escape_sequence_handling(chunks: list[str], expected: str) -> None:
    assert await _collect(chunks) == expected
