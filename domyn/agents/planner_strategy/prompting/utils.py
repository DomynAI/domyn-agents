import re
from collections.abc import AsyncGenerator
from typing import ClassVar

from pydantic import BaseModel

from domyn.agents.llm import LLMResponse


class EscapeSequenceParser(BaseModel):
    incomplete_escape_regex: ClassVar[re.Pattern] = re.compile(
        r"(\\+(?:[uU][0-9a-fA-F]{0,3}|x[0-9a-fA-F]?)?$)"
    )
    unescape_streaming: bool

    def process_streaming_chunk(self, chunk: str) -> str:
        if self.unescape_streaming:
            processed = chunk.encode("raw_unicode_escape").decode("unicode_escape")
        else:
            processed = chunk
        return processed

    def split_incomplete_escape(self, chunk: str) -> tuple[str, str]:
        """Splits the chunk into a valid JSON string body and any trailing incomplete escape sequence."""
        match = self.incomplete_escape_regex.search(chunk)
        if match:
            return self.process_streaming_chunk(chunk[: match.start()]), match.group()
        return self.process_streaming_chunk(chunk), ""


async def partial_pattern_matching(
    generator: AsyncGenerator[LLMResponse, None],
    start_pattern_conditions: list[str],
    end_pattern_conditions: list[str],
    end_pattern_negative_lookaheads: list[str],
    unescape_streamed_chunks: bool = True,
) -> AsyncGenerator[str | dict[str, str], None]:
    """Yields items from the generator that match all the pattern conditions (not necessarily contiguous).
    Args:
        generator: The generator producing the string pieces.
        start_pattern_conditions: A list of strings that must appear in order in the accumulated string for it to be considered a match. Each string in the list represents a level of pattern matching.
        end_pattern_conditions: A list of strings that indicate the end of a pattern match for each corresponding level in start_pattern_conditions.
        end_pattern_negative_lookaheads: A list of strings that indicate negative lookaheads for the end pattern conditions. If the end pattern condition is preceded by any of the negative lookaheads, it will not be considered a match. Each string in the list corresponds to the end pattern condition at the same index.
        unescape_streamed_chunks: If True, attempts to unescape JSON string escape sequences in the emitted chunks. This is useful when the patterns are meant to match JSON strings that may contain escaped characters.
    """
    if not (
        len(start_pattern_conditions)
        == len(end_pattern_conditions)
        == len(end_pattern_negative_lookaheads)
    ):
        raise ValueError("All pattern condition lists must have the same length")

    start_patterns = [re.compile(p) for p in start_pattern_conditions]
    end_patterns = [re.compile(p) for p in end_pattern_conditions]
    end_patterns_w_lookaheads = [
        re.compile(f"(?<!{re.escape(neg)}){p}")
        for p, neg in zip(end_pattern_conditions, end_pattern_negative_lookaheads, strict=True)
    ]
    escape_sequence_parser = EscapeSequenceParser(unescape_streaming=unescape_streamed_chunks)

    total = ""
    streamed = ""
    pattern_level_matched = -1
    accumulated = ""

    async for item in generator:
        total += item.content
        accumulated += item.content
        while True:
            if pattern_level_matched == (len(start_pattern_conditions) - 1):
                break
            start_matched = start_patterns[pattern_level_matched + 1].split(accumulated, maxsplit=1)

            if len(start_matched) > 1:
                pattern_level_matched += 1
                accumulated = start_matched[-1]
            else:
                break
        if pattern_level_matched >= 0:
            while True:
                negative_lookahead = re.escape(
                    end_pattern_negative_lookaheads[pattern_level_matched]
                )
                if negative_lookahead:
                    end_matched = end_patterns_w_lookaheads[pattern_level_matched].split(
                        accumulated
                    )
                else:
                    end_matched = end_patterns[pattern_level_matched].split(accumulated)
                if len(end_matched) > 1:
                    accumulated = end_matched[0]
                    if pattern_level_matched == len(start_pattern_conditions) - 1:
                        emitted, accumulated = escape_sequence_parser.split_incomplete_escape(
                            accumulated
                        )
                        if emitted:
                            yield emitted
                            streamed += emitted
                    pattern_level_matched -= 1
                else:
                    break
            if start_pattern_conditions and (
                pattern_level_matched == len(start_pattern_conditions) - 1
            ):
                emitted, accumulated = escape_sequence_parser.split_incomplete_escape(accumulated)
                if emitted:
                    yield emitted
                    streamed += emitted
    yield {"full_response": total, "streamed_response": streamed}


def sanitize_format_string(value: str) -> str:
    """
    Escape only unmatched/dangling curly braces.
    Leaves valid format placeholders like {name}, {0}, {key!r}, {:.2f} intact.
    """
    # Match valid format placeholders: {}, {0}, {name}, {name!r}, {:.2f}, etc.
    _valid_placeholder = re.compile(r"\{[^{}]*\}")

    result = []
    last = 0

    for match in _valid_placeholder.finditer(value):
        # Process the chunk before this valid placeholder
        chunk = value[last : match.start()]
        # Escape any stray braces in the chunk
        chunk = chunk.replace("{", "{{").replace("}", "}}")
        result.append(chunk)
        # Keep the valid placeholder as-is
        result.append(match.group())
        last = match.end()

    # Process remaining tail
    tail = value[last:]
    tail = tail.replace("{", "{{").replace("}", "}}")
    result.append(tail)

    return "".join(result)


def remove_descriptions_from_schema(schema: dict) -> dict:
    """Recursively removes 'description' fields from a JSON schema dictionary."""
    if isinstance(schema, dict):
        return {
            key: remove_descriptions_from_schema(value)
            for key, value in schema.items()
            if key != "description"
        }
    elif isinstance(schema, list):
        return [remove_descriptions_from_schema(item) for item in schema]
    else:
        return schema
