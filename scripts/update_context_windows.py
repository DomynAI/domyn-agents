#!/usr/bin/env python3
"""Regenerate the context-window catalogue from models.dev.

Only for hosted APIs that cannot be asked: OpenAI's ``/v1/models`` reports no window,
nor does Cerebras's (ids only). Anthropic answers ``GET /v1/models/{id}`` itself, and is
kept only as a fallback for when that call fails. Self-hosted models (vLLM, custom
endpoints) are never in it: their window is how the server was started, not a property
of the weights, and their own ``/v1/models`` says it.

A snapshot rather than a live lookup with a cache, because the live part already exists
where it is authoritative: the endpoint itself is asked first, cached five minutes. This
file only answers for APIs that report nothing, and a hosted model keeps its window under
the same id. Read live it would cost every process models.dev's whole catalogue (about
6 MB, 231 providers) for some 70 numbers, a third-party host every deployment must reach
(air-gapped and customer clusters included), and windows that change between two runs of
one release without review. A model newer than the snapshot is treated like any
undeclared window.

Run by hand, and the file it writes committed like any change:

    make context-windows

A release is a tag on main and commits nothing, so it ships the catalogue main has.

It rewrites ``domyn/agents/llm/context_windows.json`` only when a window changed, and
prints what did: for each model, the prompt limit when models.dev states one
(``limit.input`` — gpt-5 takes 272k of its 400k as prompt), else the whole window
(``limit.context``). A provider models.dev lists no window for leaves the file as it was.
"""

import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

SOURCE = "https://models.dev/api.json"
PROVIDERS = ("openai", "anthropic", "cerebras")
TARGET = Path(__file__).resolve().parents[1] / "domyn" / "agents" / "llm" / "context_windows.json"


def window(limit: dict) -> int | None:
    for key in ("input", "context"):
        value = limit.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def differences(before: dict, after: dict) -> list[str]:
    """One line per model whose window was added, changed or removed."""
    lines = []
    for provider in sorted(before.keys() | after.keys()):
        old, new = before.get(provider, {}), after.get(provider, {})
        for model in sorted(old.keys() | new.keys()):
            was, now = old.get(model), new.get(model)
            if was == now:
                continue
            if was is None:
                lines.append(f"{provider}/{model}: {now} (new)")
            elif now is None:
                lines.append(f"{provider}/{model}: removed (was {was})")
            else:
                lines.append(f"{provider}/{model}: {was} -> {now}")
    return lines


def main() -> None:
    request = urllib.request.Request(SOURCE, headers={"User-Agent": "domyn"})
    with urllib.request.urlopen(request, timeout=60) as response:
        catalogue = json.load(response)
    windows = {
        provider: {
            model_id: size
            for model_id, model in sorted(catalogue.get(provider, {}).get("models", {}).items())
            if (size := window(model.get("limit") or {})) is not None
        }
        for provider in PROVIDERS
    }
    if missing := [provider for provider, models in windows.items() if not models]:
        sys.exit(
            f"models.dev lists no window for {', '.join(missing)}: {TARGET.name} left as it was"
        )
    before = json.loads(TARGET.read_text())["windows"] if TARGET.exists() else {}
    if not (changes := differences(before, windows)):
        print(f"{TARGET.name}: unchanged")
        return
    TARGET.write_text(
        json.dumps(
            {"source": SOURCE, "generated": date.today().isoformat(), "windows": windows},
            indent=2,
        )
        + "\n"
    )
    print("\n".join(changes))


if __name__ == "__main__":
    main()
