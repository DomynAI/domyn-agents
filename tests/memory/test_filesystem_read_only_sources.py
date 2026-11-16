"""A filesystem memory that reads other files beside the one it writes (`read_only_sources`).

What can go wrong: a file only read being written to, forgotten from or cleared; its
entries hidden from an agent named otherwise than the one that wrote them; a fact it
holds being stored again; the two files' entries out of order.
"""

from datetime import UTC, datetime

import pytest

from domyn.agents.memory.managers import FilesystemMemoryManager, Memory


class _Agent:
    def __init__(self, name):
        self.name = name
        self.sub_agents = []


def _write(path, memories):
    lines = "".join(
        f"- {text} <!-- id={mid} at={at} user=u by={by} -->\n" for mid, text, at, by in memories
    )
    path.write_text(f"# Memory\n\n{lines}", encoding="utf-8")


@pytest.fixture
def files(tmp_path):
    own, other = tmp_path / "project.md", tmp_path / "global.md"
    _write(own, [("p1", "project fact", "2026-01-02T00:00:00+00:00", "Coder")])
    _write(
        other,
        [
            ("g1", "older global fact", "2026-01-01T00:00:00+00:00", "Assistant"),
            ("g2", "newer global fact", "2026-01-03T00:00:00+00:00", "Assistant"),
        ],
    )
    return own, other


@pytest.fixture
def manager(files):
    own, other = files
    return FilesystemMemoryManager(file_path=str(own), read_only_sources=[str(other)])


class TestReading:
    @pytest.mark.asyncio
    async def test_both_files_are_read_newest_first(self, manager):
        page = await manager.recall_page("u", _Agent("Coder"))
        assert [m.id for m in page.items] == ["g2", "p1", "g1"]

    @pytest.mark.asyncio
    async def test_entries_only_read_are_marked_with_their_file(self, manager, files):
        _, other = files
        page = await manager.recall_page("u", _Agent("Coder"))
        sources = {m.id: m.read_only_source for m in page.items}
        assert sources == {"p1": None, "g1": str(other), "g2": str(other)}

    @pytest.mark.asyncio
    async def test_another_configurations_agent_names_do_not_hide_them(self, manager):
        # The global memory was written by "Assistant"; this agent is "Coder".
        page = await manager.recall_page("u", _Agent("Coder"), query="global")
        assert {m.id for m in page.items} == {"g1", "g2"}

    @pytest.mark.asyncio
    async def test_another_users_entries_stay_hidden(self, manager):
        assert (await manager.recall_page("someone-else", _Agent("Coder"))).items == []

    @pytest.mark.asyncio
    async def test_a_missing_file_has_nothing(self, files, tmp_path):
        own, _ = files
        manager = FilesystemMemoryManager(
            file_path=str(own), read_only_sources=[str(tmp_path / "nowhere.md")]
        )
        assert [m.id for m in (await manager.recall_page("u")).items] == ["p1"]

    @pytest.mark.asyncio
    async def test_the_written_file_listed_again_is_read_once(self, files):
        own, _ = files
        manager = FilesystemMemoryManager(file_path=str(own), read_only_sources=[str(own)])
        page = await manager.recall_page("u")
        assert [(m.id, m.read_only_source) for m in page.items] == [("p1", None)]

    def test_the_mark_is_never_serialized(self):
        memory = Memory(timestamp=datetime.now(UTC), memory="m", read_only_source="/x.md")
        assert "read_only_source" not in memory.model_dump()


class TestWriting:
    @pytest.mark.asyncio
    async def test_remember_writes_to_its_own_file_only(self, manager, files):
        own, other = files
        before = other.read_text()
        assert await manager.remember(_Agent("Coder"), "likes tea", {"user_id": "u"}) == (
            "Memory stored."
        )
        assert "likes tea" in own.read_text()
        assert other.read_text() == before

    @pytest.mark.asyncio
    async def test_a_fact_in_a_file_only_read_is_already_remembered(self, manager, files):
        own, _ = files
        before = own.read_text()
        answer = await manager.remember(_Agent("Coder"), "Newer global fact.", {"user_id": "u"})
        assert answer == "Already remembered."
        assert own.read_text() == before


class TestRemoving:
    @pytest.mark.asyncio
    async def test_forget_refuses_an_entry_only_read_and_says_where_it_is(self, manager, files):
        _, other = files
        before = other.read_text()
        answer = await manager.forget(_Agent("Coder"), "g1", {"user_id": "u"})
        assert answer.startswith("Not forgotten") and str(other) in answer
        assert other.read_text() == before

    @pytest.mark.asyncio
    async def test_forget_still_removes_its_own(self, manager):
        assert await manager.forget(_Agent("Coder"), "p1", {"user_id": "u"}) == "Forgotten."
        assert await manager.get_memory("p1") is None

    @pytest.mark.asyncio
    async def test_delete_never_touches_a_file_only_read(self, manager, files):
        _, other = files
        before = other.read_text()
        assert await manager.delete_memory("g1") is False
        assert other.read_text() == before

    @pytest.mark.asyncio
    async def test_clearing_leaves_the_files_only_read(self, manager, files):
        own, other = files
        before = other.read_text()
        await manager.reset_memory("u")
        await manager.complete_reset()
        assert other.read_text() == before
        assert "project fact" not in own.read_text()
        assert {m.id for m in (await manager.recall_page("u")).items} == {"g1", "g2"}

    @pytest.mark.asyncio
    async def test_an_id_in_both_files_is_the_one_it_can_change(self, files):
        own, other = files
        _write(other, [("p1", "same id elsewhere", "2026-01-05T00:00:00+00:00", "Assistant")])
        manager = FilesystemMemoryManager(file_path=str(own), read_only_sources=[str(other)])
        assert (await manager.get_memory("p1")).read_only_source is None
        assert await manager.forget(_Agent("Coder"), "p1", {"user_id": "u"}) == "Forgotten."
