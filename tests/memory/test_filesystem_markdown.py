"""The filesystem manager writes markdown: a file a person can read and edit.

What can go wrong: pages that skip or repeat entries, a cursor that is not ours, a
line the user added by hand being ignored or hidden from an agent, a project reaching
outside its folder, and a write that leaves a half-written file.
"""

import pytest
from pydantic import ValidationError

from domyn.agents.memory.managers import BaseMemoryManager, FilesystemMemoryManager


class _Agent:
    def __init__(self, name, sub=()):
        self.name = name
        self.sub_agents = list(sub)


@pytest.fixture
def manager(tmp_path):
    return FilesystemMemoryManager(file_path=str(tmp_path / "memory.md"))


async def fill(manager, n, agent=None):
    agent = agent or _Agent("A")
    for i in range(n):
        await manager.write_single_memory("u", agent, f"fact number {i}")


class TestWritingAndReading:
    @pytest.mark.asyncio
    async def test_the_newest_comes_first(self, manager):
        await fill(manager, 3)
        page = await manager.recall_page("u", limit=10)
        assert [m.memory for m in page.items] == ["fact number 2", "fact number 1", "fact number 0"]
        assert page.total == 3 and page.next_cursor is None

    @pytest.mark.asyncio
    async def test_pages_neither_skip_nor_repeat(self, manager):
        await fill(manager, 7)
        seen, cursor, pages = [], None, 0
        while True:
            page = await manager.recall_page("u", limit=3, cursor=cursor)
            seen += [m.memory for m in page.items]
            pages += 1
            cursor = page.next_cursor
            if cursor is None:
                break
        assert pages == 3
        assert seen == [f"fact number {i}" for i in reversed(range(7))]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cursor", ["abc", "-1"])
    async def test_a_cursor_that_is_not_ours_is_refused(self, manager, cursor):
        await fill(manager, 2)
        with pytest.raises(ValueError):
            await manager.recall_page("u", cursor=cursor)

    @pytest.mark.asyncio
    async def test_an_empty_memory_is_an_empty_page(self, manager):
        page = await manager.recall_page("u")
        assert page.items == [] and page.next_cursor is None and page.total == 0

    @pytest.mark.asyncio
    async def test_recall_memories_keeps_the_old_contract(self, manager):
        await fill(manager, 8)
        recent = await manager.recall_memories("u", recency_limit=3)
        assert [m.memory for m in recent] == ["fact number 7", "fact number 6", "fact number 5"]

    @pytest.mark.asyncio
    async def test_a_text_with_a_line_break_stays_one_entry(self, manager):
        await manager.write_single_memory("u", _Agent("A"), "one\ntwo --> three")
        (memory,) = (await manager.recall_page("u")).items
        assert "\n" not in memory.memory and "-->" not in memory.memory

    @pytest.mark.asyncio
    async def test_an_empty_text_stores_nothing(self, manager):
        await manager.write_single_memory("u", _Agent("A"), "   ")
        assert (await manager.recall_page("u")).total == 0


class TestWhichFileIsWhichMemory:
    @pytest.mark.asyncio
    async def test_a_different_path_is_a_different_memory(self, tmp_path):
        glob = FilesystemMemoryManager(file_path=str(tmp_path / "memory.md"))
        proj = FilesystemMemoryManager(file_path=str(tmp_path / "projects" / "demo" / "memory.md"))
        await glob.write_single_memory("u", _Agent("A"), "global fact")
        await proj.write_single_memory("u", _Agent("A"), "project fact")
        assert [m.memory for m in (await glob.recall_page("u")).items] == ["global fact"]
        assert [m.memory for m in (await proj.recall_page("u")).items] == ["project fact"]

    def test_it_is_built_from_yaml_like_any_manager(self, tmp_path):
        built = BaseMemoryManager._registry["filesystem"].from_config(
            {"type": "filesystem", "file_path": str(tmp_path / "m.md"), "max_chars": 500}
        )
        assert isinstance(built, FilesystemMemoryManager) and built.max_chars == 500


class TestWhoseMemoryItIs:
    @pytest.mark.asyncio
    async def test_a_user_sees_only_their_own(self, manager):
        await manager.write_single_memory("u1", _Agent("A"), "for one")
        await manager.write_single_memory("u2", _Agent("A"), "for two")
        assert [m.memory for m in (await manager.recall_page("u1")).items] == ["for one"]

    @pytest.mark.asyncio
    async def test_a_line_nobody_wrote_is_seen_by_every_user(self, manager, tmp_path):
        (tmp_path / "memory.md").write_text("# Memory\n\n- written by hand\n")
        for user in ("u1", "u2"):
            assert [m.memory for m in (await manager.recall_page(user)).items] == [
                "written by hand"
            ]

    @pytest.mark.asyncio
    async def test_names_with_spaces_and_symbols_survive_the_file(self, manager):
        await manager.write_single_memory("user one@x.com", _Agent("My Agent"), "a fact")
        (memory,) = (await manager.recall_page("user one@x.com")).items
        assert memory.user_id == "user one@x.com" and memory.agent_name == "My Agent"


class TestFilesWrittenByEarlierVersions:
    LEGACY = (
        '{"id": "old1", "timestamp": "2026-01-01T10:00:00", "memory": "from the old format", '
        '"agent_name": "A", "user_id": "u"}\n'
        "not a line of either format\n"
    )

    @pytest.mark.asyncio
    async def test_a_json_lines_file_is_read(self, manager, tmp_path):
        (tmp_path / "memory.md").write_text(self.LEGACY)
        (memory,) = (await manager.recall_page("u")).items
        assert memory.memory == "from the old format" and memory.id == "old1"
        assert memory.timestamp.tzinfo is not None

    @pytest.mark.asyncio
    async def test_a_new_entry_is_appended_as_markdown(self, manager, tmp_path):
        (tmp_path / "memory.md").write_text(self.LEGACY)
        await manager.write_single_memory("u", _Agent("A"), "a new fact")
        text = (tmp_path / "memory.md").read_text()
        assert text.startswith(self.LEGACY) and "\n- a new fact <!-- " in text
        assert [m.memory for m in (await manager.recall_page("u")).items] == [
            "a new fact",
            "from the old format",
        ]

    @pytest.mark.asyncio
    async def test_the_next_removal_rewrites_it_as_markdown(self, manager, tmp_path):
        (tmp_path / "memory.md").write_text(self.LEGACY)
        await manager.write_single_memory("u", _Agent("A"), "a new fact")
        (fact,) = [m for m in (await manager.recall_page("u")).items if m.memory == "a new fact"]
        assert await manager.delete_memory(fact.id)
        text = (tmp_path / "memory.md").read_text()
        assert text.startswith("# Memory\n") and "{" not in text
        assert [m.memory for m in (await manager.recall_page("u")).items] == ["from the old format"]


class TestWhichAgentSees:
    @pytest.mark.asyncio
    async def test_an_agent_sees_its_own_and_its_sub_agents(self, manager):
        child = _Agent("child")
        parent = _Agent("parent", sub=[child])
        await manager.write_single_memory("u", child, "from the child")
        await manager.write_single_memory("u", _Agent("other"), "from another agent")
        texts = [m.memory for m in (await manager.recall_page("u", parent)).items]
        assert texts == ["from the child"]

    @pytest.mark.asyncio
    async def test_a_line_nobody_wrote_is_seen_by_every_agent(self, manager, tmp_path):
        (tmp_path / "memory.md").write_text("# Memory\n\n- written by hand\n")
        texts = [m.memory for m in (await manager.recall_page("u", _Agent("any"))).items]
        assert texts == ["written by hand"]


class TestTheFileIsTheUsers:
    @pytest.mark.asyncio
    async def test_the_file_is_markdown_with_a_bullet_per_entry(self, manager, tmp_path):
        await manager.write_single_memory("u", _Agent("A"), "likes tea")
        text = (tmp_path / "memory.md").read_text()
        assert text.startswith("# Memory\n")
        assert "- likes tea <!-- id=" in text

    @pytest.mark.asyncio
    async def test_a_hand_written_line_keeps_its_id_once_the_file_is_rewritten(
        self, manager, tmp_path
    ):
        (tmp_path / "memory.md").write_text("# Memory\n\n- written by hand\n")
        (entry,) = (await manager.recall_page("u")).items
        await manager.write_single_memory("u", _Agent("A"), "added by the agent")
        again = {m.memory: m.id for m in (await manager.recall_page("u")).items}
        assert again["written by hand"] == entry.id

    @pytest.mark.asyncio
    async def test_lines_that_are_not_bullets_are_left_out(self, manager, tmp_path):
        (tmp_path / "memory.md").write_text("# Memory\n\nsome notes\n\n- a fact\n")
        assert [m.memory for m in (await manager.recall_page("u")).items] == ["a fact"]

    @pytest.mark.asyncio
    async def test_a_new_entry_leaves_everything_written_by_hand_as_it_was(self, manager, tmp_path):
        by_hand = "# Memory\n\n## Work\n\nsome notes\n\n* prefers tabs\n1. a list\n- a fact\n"
        (tmp_path / "memory.md").write_text(by_hand)
        await manager.write_single_memory("u", _Agent("A"), "added by the agent")
        text = (tmp_path / "memory.md").read_text()
        assert text.startswith(by_hand)
        assert text[len(by_hand) :].startswith("- added by the agent <!-- id=")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "text", ["uses <!-- for HTML comments", "a <!-- b --> c", "ends with <!--"]
    )
    async def test_a_comment_in_the_text_is_kept_whole(self, manager, text):
        await manager.write_single_memory("u", _Agent("A"), text)
        (memory,) = (await manager.recall_page("u")).items
        assert memory.memory == text.replace("-->", "-- >")
        assert memory.user_id == "u" and memory.agent_name == "A"

    @pytest.mark.asyncio
    async def test_a_file_not_ending_in_a_newline_gets_one(self, manager, tmp_path):
        (tmp_path / "memory.md").write_text("# Memory\n\n- a fact")
        await manager.write_single_memory("u", _Agent("A"), "another")
        texts = [m.memory for m in (await manager.recall_page("u")).items]
        assert texts == ["another", "a fact"]

    @pytest.mark.asyncio
    async def test_managers_sharing_a_file_keep_each_others_entries(self, tmp_path):
        # Two sessions on the global memory: each has its own lock, none sees the other.
        import asyncio

        path = str(tmp_path / "memory.md")
        one, two = FilesystemMemoryManager(file_path=path), FilesystemMemoryManager(file_path=path)
        await asyncio.gather(
            *(
                manager.write_single_memory("u", _Agent("A"), f"fact {i} of {k}")
                for i in range(20)
                for k, manager in (("one", one), ("two", two))
            )
        )
        assert (await one.recall_page("u")).total == 40

    @pytest.mark.asyncio
    async def test_a_write_leaves_no_temporary_file(self, manager, tmp_path):
        await fill(manager, 3)
        assert [p.name for p in tmp_path.iterdir()] == ["memory.md"]


class TestSearch:
    @pytest.mark.asyncio
    async def test_the_best_match_comes_first(self, manager):
        a = _Agent("A")
        for text in ["likes green tea", "owns a green bicycle in Rome", "drinks coffee"]:
            await manager.write_single_memory("u", a, text)
        page = await manager.recall_page("u", query="green tea", limit=10)
        assert page.items[0].memory == "likes green tea"
        assert "drinks coffee" not in [m.memory for m in page.items]

    @pytest.mark.asyncio
    async def test_accents_and_case_do_not_matter(self, manager):
        await manager.write_single_memory("u", _Agent("A"), "Abita a Perugia, città")
        assert (await manager.recall_page("u", query="CITTA")).total == 1


class TestDeleting:
    @pytest.mark.asyncio
    async def test_one_entry_goes_and_an_unknown_id_changes_nothing(self, manager):
        await fill(manager, 3)
        target = (await manager.recall_page("u")).items[1]
        assert await manager.delete_memory(target.id) is True
        assert await manager.delete_memory("nope") is False
        assert target.memory not in [m.memory for m in (await manager.recall_page("u")).items]
        assert await manager.get_memory(target.id) is None

    @pytest.mark.asyncio
    async def test_removing_keeps_every_line_that_is_not_removed(self, manager, tmp_path):
        by_hand = "# Memory\n\n## Work\n\nsome notes\n* prefers tabs\n- by hand\n"
        (tmp_path / "memory.md").write_text(by_hand)
        await manager.write_single_memory("u", _Agent("A"), "goes")
        await manager.write_single_memory("u", _Agent("A"), "stays")
        goes = next(m for m in (await manager.recall_page("u")).items if m.memory == "goes")
        assert await manager.delete_memory(goes.id)
        text = (tmp_path / "memory.md").read_text()
        assert text.startswith(by_hand) and "- stays <!-- " in text and "goes" not in text

    @pytest.mark.asyncio
    async def test_clearing_keeps_what_is_not_an_entry(self, manager, tmp_path):
        (tmp_path / "memory.md").write_text("# Memory\n\n## Work\nsome notes\n- by hand\n")
        await manager.write_single_memory("u", _Agent("A"), "mine")
        await manager.reset_memory("u")
        assert (
            tmp_path / "memory.md"
        ).read_text() == "# Memory\n\n## Work\nsome notes\n- by hand\n"
        await manager.complete_reset()
        assert (tmp_path / "memory.md").read_text() == "# Memory\n\n## Work\nsome notes\n"
        assert (await manager.recall_page("u")).total == 0

    @pytest.mark.asyncio
    async def test_nothing_to_remove_leaves_the_file_unwritten(self, manager, tmp_path):
        (tmp_path / "memory.md").write_text("# Memory\n\n- by hand\n")
        before = (tmp_path / "memory.md").stat().st_mtime_ns
        assert await manager.delete_memory("nope") is False
        await manager.reset_memory("u")
        assert (tmp_path / "memory.md").stat().st_mtime_ns == before

    @pytest.mark.asyncio
    async def test_reset_can_be_limited_to_one_agent(self, manager):
        await manager.write_single_memory("u", _Agent("A"), "a")
        await manager.write_single_memory("u", _Agent("B"), "b")
        await manager.reset_memory("u", agent_name="A")
        assert [m.memory for m in (await manager.recall_page("u")).items] == ["b"]
        await manager.reset_memory("u")
        assert (await manager.recall_page("u")).total == 0
