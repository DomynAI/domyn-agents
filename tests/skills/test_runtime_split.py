"""The runtime splits into a load half and a select half.

``load()`` is I/O and runs once per turn, in the runner. ``select()`` is a
decision and runs once per agent entry, in the agent. ``resolve()`` remains the
two composed, for callers that own no turn.

The budgets split with them: the index budget belongs to the load (the selector
must see the trimmed index, and the "excluded" warning should fire once), while
the expansion and body budgets belong to the selection.
"""

import pytest

from domyn.agents.skills import (
    AllSkillsSelector,
    LoadedSkills,
    Selection,
    SkillBudget,
    SkillContent,
    SkillDescriptor,
    SkillIndex,
    SkillNotFoundError,
    SkillQuery,
    SkillRef,
    SkillSelector,
    SkillSource,
    SkillsRuntime,
)


class PreloadingSource(SkillSource):
    """Local-shaped: hands back bodies with the index, and refuses to be fetched."""

    type: str = "preloading"
    fetches: int = 0

    async def index(self, query: SkillQuery) -> SkillIndex:
        return (await self.load(query)).index

    async def fetch(self, ref: SkillRef, query: SkillQuery) -> SkillContent:
        self.fetches += 1
        raise AssertionError(f"fetch('{ref.name}') — the body was already loaded")

    async def load(self, query: SkillQuery) -> LoadedSkills:
        return LoadedSkills(
            index=SkillIndex(
                skills=[SkillDescriptor(name="sql", description="d", sections=["examples"])]
            ),
            bodies={"sql": SkillContent(name="sql", body="intro text\n\n## examples\n\nSELECT 1")},
        )


class RemoteShapedSource(SkillSource):
    """Remote-shaped: an index up front, bodies only on request."""

    type: str = "remote_shaped"
    fetches: int = 0

    async def index(self, query: SkillQuery) -> SkillIndex:
        return SkillIndex(skills=[SkillDescriptor(name="sql", description="d")])

    async def fetch(self, ref: SkillRef, query: SkillQuery) -> SkillContent:
        self.fetches += 1
        if ref.name != "sql":
            raise SkillNotFoundError(ref.name)
        return SkillContent(name="sql", body="fetched body")


class TestResolveStillComposes:
    @pytest.mark.asyncio
    async def test_resolve_equals_load_then_select(self) -> None:
        runtime = SkillsRuntime(source=PreloadingSource(), selector=AllSkillsSelector())

        whole = await runtime.resolve(SkillQuery())
        halves = await runtime.select(await runtime.load(SkillQuery()), SkillQuery())

        assert [c.body for c in whole[1]] == [c.body for c in halves[1]]
        assert [d.name for d in whole[0].index] == [d.name for d in halves[0].index]


class TestPreloadedBodiesAreUsed:
    @pytest.mark.asyncio
    async def test_expansion_does_not_refetch_a_loaded_body(self) -> None:
        source = PreloadingSource()
        runtime = SkillsRuntime(source=source, selector=AllSkillsSelector())

        _, contents = await runtime.select(await runtime.load(SkillQuery()), SkillQuery())

        assert [c.body for c in contents] == ["intro text\n\n## examples\n\nSELECT 1"]
        assert source.fetches == 0

    @pytest.mark.asyncio
    async def test_a_section_is_sliced_from_the_preloaded_body(self) -> None:
        """L2 costs no I/O either: the slice is the same one the source would do."""

        class SectionSelector(SkillSelector):
            """No stock selector emits a section ref, so this is the way in."""

            type: str = "section_only"

            async def select(self, index: SkillIndex, query: SkillQuery) -> Selection:
                return Selection(
                    index=list(index.skills),
                    expand=[SkillRef(name="sql", section="examples")],
                )

        source = PreloadingSource()
        runtime = SkillsRuntime(source=source, selector=SectionSelector())

        _, contents = await runtime.select(await runtime.load(SkillQuery()), SkillQuery())

        assert [c.body for c in contents] == ["SELECT 1"]
        assert [c.section for c in contents] == ["examples"]
        assert source.fetches == 0

    @pytest.mark.asyncio
    async def test_a_body_that_was_not_preloaded_falls_back_to_fetch(self) -> None:
        """The remote shape: an index up front, bodies only for what is expanded."""
        source = RemoteShapedSource()
        runtime = SkillsRuntime(source=source, selector=AllSkillsSelector())

        loaded = await runtime.load(SkillQuery())
        assert loaded.bodies == {}, "an index-only source must not invent bodies"

        _, contents = await runtime.select(loaded, SkillQuery())

        assert [c.body for c in contents] == ["fetched body"]
        assert source.fetches == 1


class TestBudgetsSplitWithTheHalves:
    @pytest.mark.asyncio
    async def test_the_index_budget_is_not_applied_at_load_time(self) -> None:
        """Loading is the whole catalogue: the budget belongs to the rendered index,
        and applied here it hid skills from the selector and ``skill_view`` too."""

        class ManySource(SkillSource):
            type: str = "many"

            async def index(self, query: SkillQuery) -> SkillIndex:
                return SkillIndex(
                    skills=[SkillDescriptor(name=f"s{n}", description="x" * 100) for n in range(10)]
                )

            async def fetch(self, ref: SkillRef, query: SkillQuery) -> SkillContent:
                raise SkillNotFoundError(ref.name)

        runtime = SkillsRuntime(source=ManySource(), budget=SkillBudget(max_index_chars=250))

        loaded = await runtime.load(SkillQuery())

        assert len(loaded.index.skills) == 10

    @pytest.mark.asyncio
    async def test_the_selector_sees_what_the_budget_left_out(self) -> None:
        """So a router can pick a skill the index shows by name only, or not at all."""
        seen: list[int] = []

        class RecordingSelector(SkillSelector):
            type: str = "recording"

            async def select(self, index: SkillIndex, query: SkillQuery):
                seen.append(len(index.skills))
                return await AllSkillsSelector().select(index, query)

        class ManySource(SkillSource):
            type: str = "many_2"

            async def index(self, query: SkillQuery) -> SkillIndex:
                return SkillIndex(
                    skills=[SkillDescriptor(name=f"s{n}", description="x" * 100) for n in range(10)]
                )

            async def fetch(self, ref: SkillRef, query: SkillQuery) -> SkillContent:
                raise SkillNotFoundError(ref.name)

        runtime = SkillsRuntime(
            source=ManySource(),
            selector=RecordingSelector(),
            budget=SkillBudget(max_index_chars=250),
        )
        await runtime.resolve(SkillQuery())

        assert seen == [10], "the selector was handed a trimmed index"

    @pytest.mark.asyncio
    async def test_the_expansion_cap_applies_at_select_time(self) -> None:
        class TwoSource(SkillSource):
            type: str = "two"

            async def load(self, query: SkillQuery) -> LoadedSkills:
                names = ["a", "b", "c"]
                return LoadedSkills(
                    index=SkillIndex(
                        skills=[SkillDescriptor(name=n, description="d") for n in names]
                    ),
                    bodies={n: SkillContent(name=n, body=f"body {n}") for n in names},
                )

            async def index(self, query: SkillQuery) -> SkillIndex:
                return (await self.load(query)).index

            async def fetch(self, ref: SkillRef, query: SkillQuery) -> SkillContent:
                raise SkillNotFoundError(ref.name)

        runtime = SkillsRuntime(
            source=TwoSource(),
            selector=AllSkillsSelector(),
            budget=SkillBudget(max_expansions_per_turn=2),
        )
        loaded = await runtime.load(SkillQuery())

        _, contents = await runtime.select(loaded, SkillQuery())

        assert len(contents) == 2
