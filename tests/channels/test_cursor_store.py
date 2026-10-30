"""Tests for InMemoryCursorStore and FileCursorStore.

Covered:
1. InMemoryCursorStore: round-trip, missing key, independent instances don't
   share state (no persistence across a fresh instance == a process restart).
2. FileCursorStore:
   a. load() returns None when the file doesn't exist yet.
   b. save() then load() round-trips the value.
   c. Multiple keys are stored independently in the same file.
   d. A newer save() overwrites an older one for the same key.
   e. A corrupted/non-JSON file is treated as empty rather than raising.
   f. A second instance pointed at the same path picks up prior saves.
"""

from pathlib import Path

import pytest

from domyn.agents.channels import FileCursorStore, InMemoryCursorStore


@pytest.mark.asyncio
async def test_in_memory_load_missing_key_returns_none() -> None:
    store = InMemoryCursorStore()
    assert await store.load("chan-1") is None


@pytest.mark.asyncio
async def test_in_memory_save_then_load_round_trips() -> None:
    store = InMemoryCursorStore()
    await store.save("chan-1", "evt-abc")
    assert await store.load("chan-1") == "evt-abc"


@pytest.mark.asyncio
async def test_in_memory_does_not_survive_a_fresh_instance() -> None:
    """A new InMemoryCursorStore has no memory of a previous instance's saves —
    this is the intentional behavior a process restart falls back to."""
    await InMemoryCursorStore().save("chan-1", "evt-1")
    fresh = InMemoryCursorStore()
    assert await fresh.load("chan-1") is None


@pytest.mark.asyncio
async def test_load_missing_file_returns_none(tmp_path: Path) -> None:
    store = FileCursorStore(tmp_path / "cursors.json")
    assert await store.load("chan-1") is None


@pytest.mark.asyncio
async def test_save_then_load_round_trips(tmp_path: Path) -> None:
    store = FileCursorStore(tmp_path / "cursors.json")
    await store.save("chan-1", "evt-abc")
    assert await store.load("chan-1") == "evt-abc"


@pytest.mark.asyncio
async def test_multiple_keys_independent(tmp_path: Path) -> None:
    store = FileCursorStore(tmp_path / "cursors.json")
    await store.save("chan-1", "evt-1")
    await store.save("chan-2", "evt-2")
    assert await store.load("chan-1") == "evt-1"
    assert await store.load("chan-2") == "evt-2"


@pytest.mark.asyncio
async def test_newer_save_overwrites_older(tmp_path: Path) -> None:
    store = FileCursorStore(tmp_path / "cursors.json")
    await store.save("chan-1", "evt-1")
    await store.save("chan-1", "evt-2")
    assert await store.load("chan-1") == "evt-2"


@pytest.mark.asyncio
async def test_persists_across_store_instances(tmp_path: Path) -> None:
    """A new FileCursorStore instance pointed at the same path picks up prior saves —
    this is what makes the cursor survive a process restart."""
    path = tmp_path / "cursors.json"
    await FileCursorStore(path).save("chan-1", "evt-1")
    fresh = FileCursorStore(path)
    assert await fresh.load("chan-1") == "evt-1"


@pytest.mark.asyncio
async def test_corrupted_file_treated_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "cursors.json"
    path.write_text("{not valid json")
    store = FileCursorStore(path)
    assert await store.load("chan-1") is None
    # Saving after a corrupt read must still work (starts fresh).
    await store.save("chan-1", "evt-1")
    assert await store.load("chan-1") == "evt-1"
