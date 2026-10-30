from collections.abc import AsyncGenerator

import pytest

from domyn.agents.llm import LLMResponse


async def list_async_gen(lst: list[str]) -> AsyncGenerator[LLMResponse, None]:
    for item in lst:
        yield LLMResponse(content=item, generated_tokens=1, execution_time=0)


_STRUCTURAL_TAGS_START_PATTERNS = [
    "<action>",
    r'"name"\s*:\s*("RESPOND"|"ASK_USER_MISSING_INFO")',
    r'"text"\s*:\s*"',
]
_STRUCTURAL_TAGS_END_PATTERNS = ["{action_end_pattern}", "}}", '"']
_STRUCTURAL_TAGS_END_PATTERNS_NEGATIVE_LOOKAHEADS = ["", "", "\\"]

structural_tags_test_data = [
    # Base case start and end patterns arrive in same chunk
    (
        [
            '<action>{"name":"RESPOND", "arguments":{"text":"some text with \\"escaped quotes\\" in same chunk"}, "extra": "trailing content"}}'
        ],
        _STRUCTURAL_TAGS_START_PATTERNS,
        _STRUCTURAL_TAGS_END_PATTERNS,
        _STRUCTURAL_TAGS_END_PATTERNS_NEGATIVE_LOOKAHEADS,
        'some text with "escaped quotes" in same chunk',
        '<action>{"name":"RESPOND", "arguments":{"text":"some text with \\"escaped quotes\\" in same chunk"}, "extra": "trailing content"}}',
    ),
    # Base case: token-split chunks — keys and values split mid-character as an LLM tokenizer would
    (
        [
            "<act",
            "ion>",
            '{"na',
            'me":"',
            "RES",
            'POND", ',
            '"argu',
            "ments",
            '":{"te',
            'xt":"tok',
            "en-spl",
            "it hel",
            "lo wor",
            'ld"}}',
        ],
        _STRUCTURAL_TAGS_START_PATTERNS,
        _STRUCTURAL_TAGS_END_PATTERNS,
        _STRUCTURAL_TAGS_END_PATTERNS_NEGATIVE_LOOKAHEADS,
        "token-split hello world",
        '<action>{"name":"RESPOND", "arguments":{"text":"token-split hello world"}}',
    ),
    (
        [
            "<act",
            "ion>",
            '{"na',
            'me":"',
            "RES",
            'POND", ',
            '"argu',
            "ments",
            '":{"te',
            'xt":"tok',
            "en-spl",
            "it \\n hel",
            "lo wor",
            'ld\\n"}}',
        ],
        _STRUCTURAL_TAGS_START_PATTERNS,
        _STRUCTURAL_TAGS_END_PATTERNS,
        _STRUCTURAL_TAGS_END_PATTERNS_NEGATIVE_LOOKAHEADS,
        "token-split \n hello world\n",
        '<action>{"name":"RESPOND", "arguments":{"text":"token-split \\n hello world\\n"}}',
    ),
    # Base case: content split across chunks, escaped quotes in value
    (
        [
            "<thought> agdgd",
            "</thought>",
            "as<act",
            "ion>{",
            '"name":"RESPOND",',
            '"arguments:{',
            '"need_feedback": "yes"',
            ', "text":"this',
            " has to be ",
            "streamed",
            ' including \\" and',
            ' \\" other text"}}',
        ],
        _STRUCTURAL_TAGS_START_PATTERNS,
        _STRUCTURAL_TAGS_END_PATTERNS,
        _STRUCTURAL_TAGS_END_PATTERNS_NEGATIVE_LOOKAHEADS,
        'this has to be streamed including " and " other text',
        '<thought> agdgd</thought>as<action>{"name":"RESPOND","arguments:{"need_feedback": "yes", "text":"this has to be streamed including \\" and \\" other text"}}',
    ),
    (
        [
            "<thought> agdgd",
            "</thought>",
            "as<act",
            "ion>{",
            '"name":"RESPOND",',
            '"arguments:{',
            '"need_feedback": "yes"',
            ', "text":"this',
            " has to be ",
            "streamed",
            " including \\",
            '" and',
            " \\",
            '" other text"}}',
        ],
        _STRUCTURAL_TAGS_START_PATTERNS,
        _STRUCTURAL_TAGS_END_PATTERNS,
        _STRUCTURAL_TAGS_END_PATTERNS_NEGATIVE_LOOKAHEADS,
        'this has to be streamed including " and " other text',
        '<thought> agdgd</thought>as<action>{"name":"RESPOND","arguments:{"need_feedback": "yes", "text":"this has to be streamed including \\" and \\" other text"}}',
    ),
    # Base case: action name is "TOOL" but pattern expects "RESPONSE" — nothing should be yielded
    (
        [
            '<action>{"name":"TOOL", "arguments":{"text":"this should not be streamed"}}',
            " extra trailing content",
        ],
        _STRUCTURAL_TAGS_START_PATTERNS,
        _STRUCTURAL_TAGS_END_PATTERNS,
        _STRUCTURAL_TAGS_END_PATTERNS_NEGATIVE_LOOKAHEADS,
        "",
        '<action>{"name":"TOOL", "arguments":{"text":"this should not be streamed"}} extra trailing content',
    ),
    # Edge case: empty string value (start and end pattern arrive in same chunk)
    (
        ['<action>{"text":""}', " extra trailing content"],
        ["<action>", '"text":"'],
        ["<action>", '"'],
        ["", "\\"],
        "",
        '<action>{"text":""} extra trailing content',
    ),
    # Edge case: no start pattern present — nothing should be yielded
    (
        ["<thought>some thought</thought>", " no action here"],
        _STRUCTURAL_TAGS_START_PATTERNS,
        _STRUCTURAL_TAGS_END_PATTERNS,
        _STRUCTURAL_TAGS_END_PATTERNS_NEGATIVE_LOOKAHEADS,
        "",
        "<thought>some thought</thought> no action here",
    ),
    # Edge case: text before start pattern contains end-pattern characters (quoted text),
    # verifying the function doesn't crash or incorrectly match when pattern_level_matched is -1
    (
        [
            '<thought>some "quoted" thought</thought>',
            " no action here",
        ],
        _STRUCTURAL_TAGS_START_PATTERNS,
        _STRUCTURAL_TAGS_END_PATTERNS,
        _STRUCTURAL_TAGS_END_PATTERNS_NEGATIVE_LOOKAHEADS,
        "",
        '<thought>some "quoted" thought</thought> no action here',
    ),
    # Edge case: quoted text (matching end pattern) before start pattern, followed by a valid
    # action block — verifies no crash on pattern_level_matched == -1 AND correct streaming
    (
        [
            '<thought>some "quoted" thought</thought>',
            '<action>{"name":"RESPOND", "arguments":{"text":"hello streamed world"}}',
            " trailing",
        ],
        _STRUCTURAL_TAGS_START_PATTERNS,
        _STRUCTURAL_TAGS_END_PATTERNS,
        _STRUCTURAL_TAGS_END_PATTERNS_NEGATIVE_LOOKAHEADS,
        "hello streamed world",
        '<thought>some "quoted" thought</thought><action>{"name":"RESPOND", "arguments":{"text":"hello streamed world"}} trailing',
    ),
    # Regex in start pattern: action name matched with \s* (whitespace flexibility), RESPOND action — token-split
    (
        [
            "<thou",
            "ght>thi",
            "nking</",
            "thought>",
            "<act",
            "ion>",
            '{"na',
            'me"  ',
            ':  "RE',
            'SPOND", ',
            '"argu',
            'ments": {',
            '"text": "stre',
            "amed wi",
            "th reg",
            'ex"}}',
            "</act",
            "ion>",
        ],
        _STRUCTURAL_TAGS_START_PATTERNS,
        _STRUCTURAL_TAGS_END_PATTERNS,
        _STRUCTURAL_TAGS_END_PATTERNS_NEGATIVE_LOOKAHEADS,
        "streamed with regex",
        '<thought>thinking</thought><action>{"name"  :  "RESPOND", "arguments": {"text": "streamed with regex"}}</action>',
    ),
    # Regex in start pattern: ASK_USER_MISSING_INFO action name matched via alternation — token-split
    (
        [
            "<tho",
            "ught>thi",
            "nking</tho",
            "ught>",
            "<act",
            "ion>",
            '{"na',
            'me": "AS',
            "K_USER",
            "_MISS",
            'ING_INFO", ',
            '"arguments": {',
            '"text": "par',
            "tial str",
            "eamed con",
            'tent"}}',
            "</act",
            "ion>",
        ],
        _STRUCTURAL_TAGS_START_PATTERNS,
        _STRUCTURAL_TAGS_END_PATTERNS,
        _STRUCTURAL_TAGS_END_PATTERNS_NEGATIVE_LOOKAHEADS,
        "partial streamed content",
        '<thought>thinking</thought><action>{"name": "ASK_USER_MISSING_INFO", "arguments": {"text": "partial streamed content"}}</action>',
    ),
    # Regex in start pattern: neither RESPOND nor ASK_USER_MISSING_INFO — nothing should be streamed
    (
        [
            "<action>",
            '{"name": "TOOL_CALL", "arguments": {"text": "this should not be streamed"}}',
            "</action>",
        ],
        _STRUCTURAL_TAGS_START_PATTERNS,
        _STRUCTURAL_TAGS_END_PATTERNS,
        _STRUCTURAL_TAGS_END_PATTERNS_NEGATIVE_LOOKAHEADS,
        "",
        '<action>{"name": "TOOL_CALL", "arguments": {"text": "this should not be streamed"}}</action>',
    ),
    # Regex in start pattern: extra whitespace around colon and value split across chunks — token-split
    (
        [
            "<thought>I need",
            " to respond</tho",
            "ught>",
            "<action>",
            '{"na',
            'me"',
            "   :",
            '   "RES',
            'POND"',
            ', "arg',
            'uments": {',
            '"text"',
            "   :",
            '   "hel',
            "lo fro",
            "m split",
            ' chunks"',
            "}}",
            "</action>",
        ],
        _STRUCTURAL_TAGS_START_PATTERNS,
        _STRUCTURAL_TAGS_END_PATTERNS,
        _STRUCTURAL_TAGS_END_PATTERNS_NEGATIVE_LOOKAHEADS,
        "hello from split chunks",
        '<thought>I need to respond</thought><action>{"name"   :   "RESPOND", "arguments": {"text"   :   "hello from split chunks"}}</action>',
    ),
    # Regex in start pattern: content with escaped quotes split across chunks — token-split
    (
        [
            "<thou",
            "ght>car",
            "eful</th",
            "ought>",
            '<action>{"na',
            'me" : "RES',
            'POND", "arg',
            'uments": {',
            '"text": "val',
            'ue with \\"',
            "esc",
            'aped\\"',
            " and more",
            ' \\" quo',
            'tes"}}',
            "</action>",
        ],
        _STRUCTURAL_TAGS_START_PATTERNS,
        _STRUCTURAL_TAGS_END_PATTERNS,
        _STRUCTURAL_TAGS_END_PATTERNS_NEGATIVE_LOOKAHEADS,
        'value with "escaped" and more " quotes',
        '<thought>careful</thought><action>{"name" : "RESPOND", "arguments": {"text": "value with \\"escaped\\" and more \\" quotes"}}</action>',
    ),
    # Edge case: a first action block closes without triggering a response inside the thought,
    # then a second action block opens with RESPOND and should be streamed correctly
    (
        [
            "<thought>first I call a tool",
            "<action>action inside thought",
            "</action>",
            "now I respond</thought>",
            '<action>{"name": "RESPOND", "arguments": {"text": "response after tool action"}}',
            "</action>",
        ],
        _STRUCTURAL_TAGS_START_PATTERNS,
        _STRUCTURAL_TAGS_END_PATTERNS,
        _STRUCTURAL_TAGS_END_PATTERNS_NEGATIVE_LOOKAHEADS,
        "response after tool action",
        '<thought>first I call a tool<action>action inside thought</action>now I respond</thought><action>{"name": "RESPOND", "arguments": {"text": "response after tool action"}}</action>',
    ),
]

_JSON_START_PATTERNS = [
    r'"action"\s*:\s*',
    r'"name"\s*:\s*("RESPOND"|"ASK_USER_MISSING_INFO")',
    r'"text"\s*:\s*"',
]
_JSON_END_PATTERNS = ["}", "}}", '"']
_JSON_NEGATIVE_LOOKAHEADS = ["", "", "\\"]

json_prompting_test_data = [
    # Base case: RESPOND action, text in same chunk
    (
        [
            '{"thought": "I should respond", "action": {"name": "RESPOND", "arguments": {"text": "hello world", "other": "no streaming"}}}'
        ],
        _JSON_START_PATTERNS,
        _JSON_END_PATTERNS,
        _JSON_NEGATIVE_LOOKAHEADS,
        "hello world",
        '{"thought": "I should respond", "action": {"name": "RESPOND", "arguments": {"text": "hello world", "other": "no streaming"}}}',
    ),
    # Base case: RESPOND action, all keys and values token-split mid-character
    (
        [
            '{"tho',
            'ught":',
            ' "I sh',
            "ould res",
            'pond",',
            ' "act',
            'ion":',
            ' {"na',
            'me":',
            ' "RES',
            'POND",',
            ' "arg',
            'uments":',
            ' {"te',
            'xt":',
            ' "hel',
            "lo wor",
            'ld",',
            ' "ot',
            'her":',
            ' "no str',
            "eam",
            'ing"}}}',
        ],
        _JSON_START_PATTERNS,
        _JSON_END_PATTERNS,
        _JSON_NEGATIVE_LOOKAHEADS,
        "hello world",
        '{"thought": "I should respond", "action": {"name": "RESPOND", "arguments": {"text": "hello world", "other": "no streaming"}}}',
    ),
    # Base case: ASK_USER_MISSING_INFO action, text in same chunk
    (
        [
            '{"thought": "I need more info", "action": {"name": "ASK_USER_MISSING_INFO", "arguments": {"text": "what is your name?"}}}'
        ],
        _JSON_START_PATTERNS,
        _JSON_END_PATTERNS,
        _JSON_NEGATIVE_LOOKAHEADS,
        "what is your name?",
        '{"thought": "I need more info", "action": {"name": "ASK_USER_MISSING_INFO", "arguments": {"text": "what is your name?"}}}',
    ),
    # Base case: ASK_USER_MISSING_INFO action, all keys and values token-split mid-character
    (
        [
            '{"tho',
            'ught":',
            ' "I ne',
            "ed mo",
            "re in",
            'fo",',
            ' "act',
            'ion":',
            ' {"na',
            'me":',
            ' "ASK',
            "_USER",
            "_MISS",
            "ING_I",
            'NFO",',
            ' "arg',
            'uments":',
            ' {"te',
            'xt":',
            ' "wha',
            "t is",
            " your",
            " name",
            '?"}}}',
        ],
        _JSON_START_PATTERNS,
        _JSON_END_PATTERNS,
        _JSON_NEGATIVE_LOOKAHEADS,
        "what is your name?",
        '{"thought": "I need more info", "action": {"name": "ASK_USER_MISSING_INFO", "arguments": {"text": "what is your name?"}}}',
    ),
    # Non-matching action name (e.g. TOOL_CALL) — nothing should be streamed
    (
        [
            '{"thought": "calling a tool", "action": {"name": "TOOL_CALL", "arguments": {"text": "this should not be streamed"}}}'
        ],
        _JSON_START_PATTERNS,
        _JSON_END_PATTERNS,
        _JSON_NEGATIVE_LOOKAHEADS,
        "",
        '{"thought": "calling a tool", "action": {"name": "TOOL_CALL", "arguments": {"text": "this should not be streamed"}}}',
    ),
    # Content split across chunks, RESPOND action — finer token-split mid-word
    (
        [
            '{"tho',
            'ught":',
            ' "res',
            "pond",
            "ing t",
            "o use",
            'r",',
            ' "act',
            'ion":',
            ' {"na',
            'me":',
            ' "RES',
            'POND",',
            ' "arg',
            'uments":',
            ' {"te',
            'xt":',
            ' "thi',
            "s is",
            " spli",
            "t ac",
            "ross",
            " chu",
            'nks",',
            ' "ot',
            'her":',
            ' "ign',
            'ored"',
            "}}}",
        ],
        _JSON_START_PATTERNS,
        _JSON_END_PATTERNS,
        _JSON_NEGATIVE_LOOKAHEADS,
        "this is split across chunks",
        '{"thought": "responding to user", "action": {"name": "RESPOND", "arguments": {"text": "this is split across chunks", "other": "ignored"}}}',
    ),
    # Content split across chunks, ASK_USER_MISSING_INFO action
    (
        [
            '{"thought": "need clarification",',
            ' "action": {"name": "ASK_USER_MISSING_INFO",',
            ' "arguments": {"text": "can you',
            ' clarify please?"}}}',
        ],
        _JSON_START_PATTERNS,
        _JSON_END_PATTERNS,
        _JSON_NEGATIVE_LOOKAHEADS,
        "can you clarify please?",
        '{"thought": "need clarification", "action": {"name": "ASK_USER_MISSING_INFO", "arguments": {"text": "can you clarify please?"}}}',
    ),
    # Escaped quotes inside text value, RESPOND action — token-split mid-word
    (
        [
            '{"tho',
            'ught":',
            ' "car',
            "eful w",
            "ith esc",
            'apes",',
            ' "act',
            'ion":',
            ' {"na',
            'me":',
            ' "RES',
            'POND",',
            ' "arg',
            'uments":',
            ' {"te',
            'xt":',
            ' "val',
            "ue wi",
            'th \\"',
            "esc",
            'aped\\"',
            " and",
            " more",
            ' \\" q',
            'uotes",',
            ' "ot',
            'her":',
            ' "no s',
            "tream",
            'ing"',
            "}}}",
        ],
        _JSON_START_PATTERNS,
        _JSON_END_PATTERNS,
        _JSON_NEGATIVE_LOOKAHEADS,
        'value with "escaped" and more " quotes',
        '{"thought": "careful with escapes", "action": {"name": "RESPOND", "arguments": {"text": "value with \\"escaped\\" and more \\" quotes", "other": "no streaming"}}}',
    ),
    # Extra whitespace around colons in name and text keys (regex flexibility), RESPOND action — token-split
    (
        [
            '{"tho',
            'ught":',
            ' "whi',
            "tespa",
            "ce te",
            'st",',
            ' "act',
            'ion":',
            ' {"na',
            'me"   ',
            ':   "R',
            "ESPON",
            'D",',
            ' "arg',
            'uments":',
            ' {"te',
            'xt"   ',
            ':   "s',
            "paced",
            " out",
            " cont",
            'ent",',
            ' "ot',
            'her":',
            ' "ign',
            'ored"',
            "}}}",
        ],
        _JSON_START_PATTERNS,
        _JSON_END_PATTERNS,
        _JSON_NEGATIVE_LOOKAHEADS,
        "spaced out content",
        '{"thought": "whitespace test", "action": {"name"   :   "RESPOND", "arguments": {"text"   :   "spaced out content", "other": "ignored"}}}',
    ),
    # Thought contains quoted text that could confuse pattern matching — nothing should be misstreamed
    (
        [
            '{"thought": "I think \\"this\\" is a \\"quoted\\" thought",',
            ' "action": {"name": "TOOL_CALL", "arguments": {"param": "value"}}}',
        ],
        _JSON_START_PATTERNS,
        _JSON_END_PATTERNS,
        _JSON_NEGATIVE_LOOKAHEADS,
        "",
        '{"thought": "I think \\"this\\" is a \\"quoted\\" thought", "action": {"name": "TOOL_CALL", "arguments": {"param": "value"}}}',
    ),
    # Thought with quoted text followed by valid RESPOND action — token-split
    (
        [
            '{"tho',
            'ught":',
            ' "the',
            " user",
            " said",
            ' \\"he',
            'llo\\"',
            " earl",
            'ier",',
            ' "act',
            'ion":',
            ' {"na',
            'me":',
            ' "RES',
            'POND",',
            ' "arg',
            'uments":',
            ' {"te',
            'xt":',
            ' "str',
            "eame",
            "d aft",
            "er qu",
            "oted",
            " tho",
            'ught"',
            "}}}",
        ],
        _JSON_START_PATTERNS,
        _JSON_END_PATTERNS,
        _JSON_NEGATIVE_LOOKAHEADS,
        "streamed after quoted thought",
        '{"thought": "the user said \\"hello\\" earlier", "action": {"name": "RESPOND", "arguments": {"text": "streamed after quoted thought"}}}',
    ),
    # Action name and arguments split across many small chunks
    (
        [
            '{"thought":',
            ' "step by step",',
            ' "action": {',
            '"name"',
            ': "RESPOND"',
            ', "arguments": {',
            '"text": "',
            "hello from",
            " many chunks",
            '", "other": "no"}}}',
        ],
        _JSON_START_PATTERNS,
        _JSON_END_PATTERNS,
        _JSON_NEGATIVE_LOOKAHEADS,
        "hello from many chunks",
        '{"thought": "step by step", "action": {"name": "RESPOND", "arguments": {"text": "hello from many chunks", "other": "no"}}}',
    ),
    # Whitespace flexibility: extra spaces around "action" colon, RESPOND action in same chunk
    (
        [
            '{"thought": "whitespace around action key", "action"   :   {"name": "RESPOND", "arguments": {"text": "spaced action key"}}}'
        ],
        _JSON_START_PATTERNS,
        _JSON_END_PATTERNS,
        _JSON_NEGATIVE_LOOKAHEADS,
        "spaced action key",
        '{"thought": "whitespace around action key", "action"   :   {"name": "RESPOND", "arguments": {"text": "spaced action key"}}}',
    ),
    # Whitespace flexibility: "action" key with spaces, content split across chunks
    (
        [
            '{"thought": "split with spaced action key",',
            ' "action"  :  {"name": "RESPOND",',
            ' "arguments": {"text": "streamed with spaced action',
            ' key"}}}',
        ],
        _JSON_START_PATTERNS,
        _JSON_END_PATTERNS,
        _JSON_NEGATIVE_LOOKAHEADS,
        "streamed with spaced action key",
        '{"thought": "split with spaced action key", "action"  :  {"name": "RESPOND", "arguments": {"text": "streamed with spaced action key"}}}',
    ),
    # Whitespace flexibility: "action" key with spaces, non-matching action name — nothing streamed
    (
        [
            '{"thought": "tool call with spaced action key",',
            ' "action"   :   {"name": "TOOL_CALL", "arguments": {"text": "this should not be streamed"}}}',
        ],
        _JSON_START_PATTERNS,
        _JSON_END_PATTERNS,
        _JSON_NEGATIVE_LOOKAHEADS,
        "",
        '{"thought": "tool call with spaced action key", "action"   :   {"name": "TOOL_CALL", "arguments": {"text": "this should not be streamed"}}}',
    ),
    # Whitespace flexibility: "action" key with spaces, ASK_USER_MISSING_INFO action
    (
        [
            '{"thought": "asking user",',
            ' "action" : {"name": "ASK_USER_MISSING_INFO", "arguments": {"text": "please provide more details"}}}',
        ],
        _JSON_START_PATTERNS,
        _JSON_END_PATTERNS,
        _JSON_NEGATIVE_LOOKAHEADS,
        "please provide more details",
        '{"thought": "asking user", "action" : {"name": "ASK_USER_MISSING_INFO", "arguments": {"text": "please provide more details"}}}',
    ),
    # Model generate multiple emtpy strings(regex flexibility), RESPOND action — token-split
    (
        [
            '{"tho',
            'ught":',
            ' "whi',
            "tespa",
            "ce te",
            'st",',
            ' "act',
            'ion":',
            ' {"na',
            'me"   ',
            ':   "R',
            "ESPON",
            'D",',
            "",
            "",
            ' "arg',
            'uments":',
            ' {"te',
            'xt"   ',
            ':   "s',
            "paced",
            " out",
            " cont",
            'ent",',
            ' "ot',
            'her":',
            ' "ign',
            'ored"',
            "}",
            "}",
            "}",
            "",
            "",
        ],
        _JSON_START_PATTERNS,
        _JSON_END_PATTERNS,
        _JSON_NEGATIVE_LOOKAHEADS,
        "spaced out content",
        '{"thought": "whitespace test", "action": {"name"   :   "RESPOND", "arguments": {"text"   :   "spaced out content", "other": "ignored"}}}',
    ),
    # model generate json inside text
    (
        [
            '{"thought": "asking user",',
            r' "action" : {"name": "ASK_USER_MISSING_INFO", "arguments": {"text": "{\"key\": \"value\", \"text\": \"nested text\"}"}}}',
        ],
        _JSON_START_PATTERNS,
        _JSON_END_PATTERNS,
        _JSON_NEGATIVE_LOOKAHEADS,
        '{"key": "value", "text": "nested text"}',
        r'{"thought": "asking user", "action" : {"name": "ASK_USER_MISSING_INFO", "arguments": {"text": "{\"key\": \"value\", \"text\": \"nested text\"}"}}}',
    ),
]


@pytest.mark.parametrize(
    "generator_list, start_patterns, end_patterns, end_patterns_negative_lookaheads, expected_streaming, expected_full_response",
    structural_tags_test_data + json_prompting_test_data,
)
@pytest.mark.asyncio
async def test_partial_pattern_matching(
    generator_list,
    start_patterns: list[str],
    end_patterns: list[str],
    end_patterns_negative_lookaheads: list[str],
    expected_streaming: str,
    expected_full_response: str,
):
    from domyn.agents.planner_strategy.prompting.utils import partial_pattern_matching

    streamed = ""
    streamed_response = None
    full_response = None
    async for matched_item in partial_pattern_matching(
        list_async_gen(generator_list),
        start_pattern_conditions=start_patterns,
        end_pattern_conditions=end_patterns,
        end_pattern_negative_lookaheads=end_patterns_negative_lookaheads,
    ):
        if isinstance(matched_item, str):
            streamed += matched_item
        elif isinstance(matched_item, dict):
            streamed_response = matched_item.get("streamed_response", None)
            full_response = matched_item.get("full_response", None)
    assert streamed == expected_streaming
    assert streamed_response == expected_streaming
    assert full_response == expected_full_response


@pytest.mark.parametrize(
    "generator_list, start_patterns, end_patterns, end_patterns_negative_lookaheads, expected_streaming, expected_full_response",
    structural_tags_test_data,
)
@pytest.mark.asyncio
async def test_structural_tags_prompting_pattern_matching(
    generator_list,
    start_patterns: list[str],
    end_patterns: list[str],
    end_patterns_negative_lookaheads: list[str],
    expected_streaming: str,
    expected_full_response: str,
):
    from domyn.agents.planner_strategy.prompting.structural_tags_prompting import (
        StructuralTagsPrompting,
    )

    streamed = ""
    streamed_response = None
    full_response = None
    prompting = StructuralTagsPrompting()
    async for matched_item in prompting.parse_streaming_response(
        generator=list_async_gen(generator_list)
    ):
        if isinstance(matched_item, str):
            streamed += matched_item
        elif isinstance(matched_item, dict):
            streamed_response = matched_item.get("streamed_response", None)
            full_response = matched_item.get("full_response", None)
    assert streamed == expected_streaming
    assert streamed_response == expected_streaming
    assert full_response == expected_full_response


@pytest.mark.parametrize(
    "generator_list, start_patterns, end_patterns, end_patterns_negative_lookaheads, expected_streaming, expected_full_response",
    json_prompting_test_data,
)
@pytest.mark.asyncio
async def test_json_prompting_pattern_matching(
    generator_list: list[str],
    start_patterns: list[str],
    end_patterns: list[str],
    end_patterns_negative_lookaheads: list[str],
    expected_streaming: str,
    expected_full_response: str,
):
    from domyn.agents.planner_strategy import JSONPrompting

    streamed = ""
    streamed_response = None
    full_response = None
    prompting = JSONPrompting()
    async for matched_item in prompting.parse_streaming_response(
        generator=list_async_gen(generator_list)
    ):
        if isinstance(matched_item, str):
            streamed += matched_item
        elif isinstance(matched_item, dict):
            streamed_response = matched_item.get("streamed_response", None)
            full_response = matched_item.get("full_response", None)
    assert streamed == expected_streaming
    assert streamed_response == expected_streaming
    assert full_response == expected_full_response


json_unescape_test_data = [
    # Newline escape (\n) in single chunk — the original bug scenario
    (
        [
            '{"thought": "thinking", "action": {"name": "RESPOND", "arguments": {"text": "line1\\nline2"}}}'
        ],
        _JSON_START_PATTERNS,
        _JSON_END_PATTERNS,
        _JSON_NEGATIVE_LOOKAHEADS,
        "line1\nline2",
        '{"thought": "thinking", "action": {"name": "RESPOND", "arguments": {"text": "line1\\nline2"}}}',
    ),
    # Newline escape split across two chunks (\ in one chunk, n in the next)
    (
        [
            '{"thought": "t", "action": {"name": "RESPOND", "arguments": {"text": "line1\\',
            'nline2"}}}',
        ],
        _JSON_START_PATTERNS,
        _JSON_END_PATTERNS,
        _JSON_NEGATIVE_LOOKAHEADS,
        "line1\nline2",
        '{"thought": "t", "action": {"name": "RESPOND", "arguments": {"text": "line1\\nline2"}}}',
    ),
    # Tab and newline escapes mixed in same chunk
    (
        [
            '{"thought": "t", "action": {"name": "RESPOND", "arguments": {"text": "col1\\tcol2\\nrow2"}}}'
        ],
        _JSON_START_PATTERNS,
        _JSON_END_PATTERNS,
        _JSON_NEGATIVE_LOOKAHEADS,
        "col1\tcol2\nrow2",
        '{"thought": "t", "action": {"name": "RESPOND", "arguments": {"text": "col1\\tcol2\\nrow2"}}}',
    ),
    # Double newline inside the text (regression)
    (
        [
            '{"thought": "t", "action": {"name": "RESPOND", "arguments": {"text": "Here are three films:\\n\\n1. Film A\\n2. Film B"}}}'
        ],
        _JSON_START_PATTERNS,
        _JSON_END_PATTERNS,
        _JSON_NEGATIVE_LOOKAHEADS,
        "Here are three films:\n\n1. Film A\n2. Film B",
        '{"thought": "t", "action": {"name": "RESPOND", "arguments": {"text": "Here are three films:\\n\\n1. Film A\\n2. Film B"}}}',
    ),
    # Unicode escape (\u00e9 = é) in same chunk
    (
        ['{"thought": "t", "action": {"name": "RESPOND", "arguments": {"text": "caf\\u00e9"}}}'],
        _JSON_START_PATTERNS,
        _JSON_END_PATTERNS,
        _JSON_NEGATIVE_LOOKAHEADS,
        "café",
        '{"thought": "t", "action": {"name": "RESPOND", "arguments": {"text": "caf\\u00e9"}}}',
    ),
    # Unicode escape split across chunks (\u00e in chunk 1, 9 in chunk 2)
    (
        [
            '{"thought": "t", "action": {"name": "RESPOND", "arguments": {"text": "caf\\u00e',
            '9"}}}',
        ],
        _JSON_START_PATTERNS,
        _JSON_END_PATTERNS,
        _JSON_NEGATIVE_LOOKAHEADS,
        "café",
        '{"thought": "t", "action": {"name": "RESPOND", "arguments": {"text": "caf\\u00e9"}}}',
    ),
    # Backslash literal (\\) in same chunk
    (
        ['{"thought": "t", "action": {"name": "RESPOND", "arguments": {"text": "path\\\\file"}}}'],
        _JSON_START_PATTERNS,
        _JSON_END_PATTERNS,
        _JSON_NEGATIVE_LOOKAHEADS,
        "path\\file",
        '{"thought": "t", "action": {"name": "RESPOND", "arguments": {"text": "path\\\\file"}}}',
    ),
    # Hex escape (\x41 = 'A', \x0A = newline) in same chunk
    (
        [
            '{"thought": "t", "action": {"name": "RESPOND", "arguments": {"text": "\\x41BC\\x0Anewline"}}}'
        ],
        _JSON_START_PATTERNS,
        _JSON_END_PATTERNS,
        _JSON_NEGATIVE_LOOKAHEADS,
        "ABC\nnewline",
        '{"thought": "t", "action": {"name": "RESPOND", "arguments": {"text": "\\x41BC\\x0Anewline"}}}',
    ),
    # Hex escape split across chunks (\x0 in chunk 1, A in chunk 2) — exercises the boundary-hold logic
    (
        [
            '{"thought": "t", "action": {"name": "RESPOND", "arguments": {"text": "before\\x0',
            'Aafter"}}}',
        ],
        _JSON_START_PATTERNS,
        _JSON_END_PATTERNS,
        _JSON_NEGATIVE_LOOKAHEADS,
        "before\nafter",
        '{"thought": "t", "action": {"name": "RESPOND", "arguments": {"text": "before\\x0Aafter"}}}',
    ),
    (
        [
            '{"thought": "t", "action": {"name": "RESPOND", "arguments": {"text": "`print(\\"He said \\\\\\"Hi\\\\\\"\\',
            '")`"}}}',
        ],
        _JSON_START_PATTERNS,
        _JSON_END_PATTERNS,
        _JSON_NEGATIVE_LOOKAHEADS,
        '`print("He said \\"Hi\\"")`',
        '{"thought": "t", "action": {"name": "RESPOND", "arguments": {"text": "`print(\\"He said \\\\\\"Hi\\\\\\"\\")`"}}}',
    ),
]


@pytest.mark.parametrize(
    "generator_list, start_patterns, end_patterns, end_patterns_negative_lookaheads, expected_streaming, expected_full_response",
    json_unescape_test_data,
)
@pytest.mark.asyncio
async def test_json_prompting_unescape(
    generator_list: list[str],
    start_patterns: list[str],
    end_patterns: list[str],
    end_patterns_negative_lookaheads: list[str],
    expected_streaming: str,
    expected_full_response: str,
) -> None:
    r"""JSON escape sequences (\n, \t, \uXXXX, \\) must be decoded when streamed."""
    from domyn.agents.planner_strategy import JSONPrompting

    streamed = ""
    streamed_response = None
    full_response = None
    prompting = JSONPrompting()
    async for matched_item in prompting.parse_streaming_response(
        generator=list_async_gen(generator_list)
    ):
        if isinstance(matched_item, str):
            streamed += matched_item
        elif isinstance(matched_item, dict):
            streamed_response = matched_item.get("streamed_response", None)
            full_response = matched_item.get("full_response", None)
    assert streamed == expected_streaming
    assert streamed_response == expected_streaming
    assert full_response == expected_full_response


@pytest.mark.asyncio
async def test_json_prompting_unescape_disabled() -> None:
    """With unescape_json_strings=False, raw JSON escape sequences are passed through as-is."""
    from domyn.agents.planner_strategy.prompting.utils import partial_pattern_matching

    chunks = [
        '{"thought": "t", "action": {"name": "RESPOND", "arguments": {"text": "line1\\nline2"}}}'
    ]
    streamed = ""
    async for matched_item in partial_pattern_matching(
        list_async_gen(chunks),
        start_pattern_conditions=_JSON_START_PATTERNS,
        end_pattern_conditions=_JSON_END_PATTERNS,
        end_pattern_negative_lookaheads=_JSON_NEGATIVE_LOOKAHEADS,
        unescape_streamed_chunks=False,
    ):
        if isinstance(matched_item, str):
            streamed += matched_item
    assert streamed == "line1\\nline2"


@pytest.mark.asyncio
async def test_raise_partial_pattern_matching():
    from domyn.agents.planner_strategy.prompting.utils import partial_pattern_matching

    with pytest.raises(ValueError, match="All pattern condition lists must have the same length"):
        async for _ in partial_pattern_matching(
            list_async_gen(['<action>{"name":"RESPOND", "arguments":{"text":"some text"}}}']),
            start_pattern_conditions=[
                "<action>",
                '"name"\\s*:\\s*("RESPOND"|"ASK_USER_MISSING_INFO")',
            ],
            end_pattern_conditions=["}}"],
            end_pattern_negative_lookaheads=["\\"],
        ):
            pass


@pytest.mark.asyncio
async def test_json_prompting_unescape_disabled_split_escape() -> None:
    r"""With unescape_streamed_chunks=False, a \" split across tokens must not fire as end-pattern."""
    from domyn.agents.planner_strategy.prompting.utils import partial_pattern_matching

    chunks = [
        '{"thought": "t", "action": {"name": "RESPOND", "arguments": {"text": "`print(\\"He said \\\\\\"Hi\\\\\\"\\',
        '")`"}}}',
    ]
    streamed = ""
    async for matched_item in partial_pattern_matching(
        list_async_gen(chunks),
        start_pattern_conditions=_JSON_START_PATTERNS,
        end_pattern_conditions=_JSON_END_PATTERNS,
        end_pattern_negative_lookaheads=_JSON_NEGATIVE_LOOKAHEADS,
        unescape_streamed_chunks=False,
    ):
        if isinstance(matched_item, str):
            streamed += matched_item
    assert streamed == '`print(\\"He said \\\\\\"Hi\\\\\\"\\")`'
