"""``update_conversation_state`` replaces the stored state; it does not merge.

Replace is the contract because of what the callers do with it. Each one hands
over the whole of ``context.state`` — the runner writes it back after every
non-token event — so a key missing from the argument is not an omission, it is a
removal, and it has to reach storage.

``SqliteConversationManager`` has always replaced (``SET state = ?`` over
``json.dumps(state)``). ``InMemoryConversationManager`` merged, via
``ConversationInfo.update_state``, until that method was changed to assign a
deep copy.
"""

import tempfile
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from domyn.agents.conversation_manager import (
    BaseConversationManager,
    InMemoryConversationManager,
    SqliteConversationManager,
)

APP = "test_app"
USER = "user_1"


@pytest_asyncio.fixture
async def backends():
    """One manager of each kind, keyed by name so a test can parametrize over them.

    The key is what pytest prints in the test id, so a failure names the backend
    that broke rather than leaving the reader to work it out.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
        temp_db_path = temp_db.name

    sqlite_manager = SqliteConversationManager(path=temp_db_path)
    await sqlite_manager._create_table()

    yield {"in_memory": InMemoryConversationManager(), "sqlite": sqlite_manager}

    Path(temp_db_path).unlink(missing_ok=True)


async def _state_after_update(
    manager: BaseConversationManager,
    initial: dict[str, Any],
    update: dict[str, Any],
) -> dict[str, Any]:
    """Create a conversation, update its state once, and read back what was stored.

    The dicts are copied on the way in: the in-memory backend stores the object it
    is handed, and the parametrized cases below are shared across every backend, so
    without the copies one run could contaminate the next.
    """
    conversation = await manager.create_conversation(APP, USER, initial_state=dict(initial))
    updated = await manager.update_conversation_state(
        APP, USER, conversation.conversation_id, dict(update)
    )
    assert updated, f"{type(manager).__name__} reported the update as failed"

    stored = await manager.get_conversation(APP, USER, conversation.conversation_id)
    assert stored is not None
    return stored.state


class TestTheUpdateReplacesTheState:
    """``update_conversation_state`` replaces the stored state, on every backend."""

    @pytest.mark.parametrize("backend", ["in_memory", "sqlite"])
    @pytest.mark.parametrize(
        ("initial", "update", "expected"),
        [
            # The discriminating case: "drop" exists before the update and the
            # update does not mention it. Replace removes it; merge would keep it.
            pytest.param(
                {"keep": 1, "drop": 2},
                {"keep": 1},
                {"keep": 1},
                id="key-absent-from-the-update",
            ),
            # The same claim, in the form a caller is most likely to write by accident.
            pytest.param({"a": 1}, {}, {}, id="empty-update"),
            # These two hold under either semantics, and are here so that a change
            # breaking them is caught as well.
            pytest.param({"a": 1}, {"a": 2}, {"a": 2}, id="overwrite-an-existing-key"),
            pytest.param({"a": 1}, {"a": 1, "b": 2}, {"a": 1, "b": 2}, id="add-a-new-key"),
            pytest.param(
                {"a": [1, 2, 3]},
                {"a": [1, 2, 3], "b": [4, 5, 6]},
                {"a": [1, 2, 3], "b": [4, 5, 6]},
                id="update-list-with-new-key",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_the_stored_state_is_the_update(
        self,
        backends,
        backend: str,
        initial: dict[str, Any],
        update: dict[str, Any],
        expected: dict[str, Any],
    ) -> None:
        stored = await _state_after_update(backends[backend], initial, update)

        assert stored == expected, (
            f"{backend} did not replace the conversation state\n"
            f"  initial:  {initial}\n"
            f"  update:   {update}\n"
            f"  expected: {expected}\n"
            f"  stored:   {stored}"
        )


class TestTheStoredStateIsASnapshot:
    """What is stored is a copy, not an alias to the caller's dict.

    Callers hand over ``context.state`` and keep mutating it for the rest of the
    turn. An in-memory store that assigned the argument directly would publish
    those later edits with no write behind them, and so drift from the
    persistent backends, which can only ever see what was passed.
    """

    @pytest.mark.parametrize("backend", ["in_memory", "sqlite"])
    @pytest.mark.asyncio
    async def test_mutating_the_argument_afterwards_does_not_reach_the_store(
        self, backends, backend: str
    ) -> None:
        manager = backends[backend]
        # Nested, because a shallow copy would pass a flat check.
        caller_state: dict[str, Any] = {"items": [1, 2, 3]}

        conversation = await manager.create_conversation(APP, USER)
        assert await manager.update_conversation_state(
            APP, USER, conversation.conversation_id, caller_state
        )

        caller_state["items"].append(4)
        caller_state["added_later"] = True

        stored = await manager.get_conversation(APP, USER, conversation.conversation_id)
        assert stored is not None
        assert stored.state == {"items": [1, 2, 3]}, (
            f"{backend} kept a reference to the caller's dict instead of a snapshot\n"
            f"  stored: {stored.state}"
        )
