"""``type: resource`` skills and instructions sources, over local and other resources."""

from pathlib import Path

import pytest

from domyn.agents.instructions import InstructionKind
from domyn.agents.instructions.sources import (
    LocalDirInstructionsSource,
    ResourceInstructionsSource,
)
from domyn.agents.resources import LocalDirResource, Resource, ResourceError
from domyn.agents.skills import LocalDirSkillSource, SkillQuery, SkillRef
from domyn.agents.skills.sources import ResourceSkillSource, local_source_paths
from tests.resources.fake import InMemoryResource


def _skill(name: str, body: str = "body") -> str:
    return f"---\nname: {name}\ndescription: {name} skill\n---\n\n{body}\n\n## Part\n\npart\n"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _bound(source: ResourceSkillSource, *resources: Resource) -> ResourceSkillSource:
    return source.bind({r.id: r for r in resources})


class TestSkills:
    @pytest.mark.asyncio
    async def test_skills_are_read_under_each_root_as_a_local_dir_source_reads_them(
        self, tmp_path: Path
    ) -> None:
        _write(tmp_path / "a" / "skills" / "one" / "SKILL.md", _skill("one"))
        _write(tmp_path / "a" / "skills" / "group" / "two" / "SKILL.md", _skill("two"))
        _write(tmp_path / "b" / "skills" / "one" / "SKILL.md", _skill("one", "shadowed"))
        _write(tmp_path / "b" / "skills" / "three.md", _skill("three"))
        resource = LocalDirResource(id="p", paths=[str(tmp_path / "a"), str(tmp_path / "b")])
        source = _bound(ResourceSkillSource(resource="p"), resource)

        loaded = await source.load(SkillQuery())
        local = await LocalDirSkillSource(
            paths=[tmp_path / "a" / "skills", tmp_path / "b" / "skills"]
        ).load(SkillQuery())

        assert [s.name for s in loaded.index.skills] == [s.name for s in local.index.skills]
        assert loaded.body("one").body.strip().startswith("body")  # type: ignore[union-attr]
        assert local_source_paths(source) == [
            tmp_path / "a" / "skills",
            tmp_path / "b" / "skills",
        ]

    @pytest.mark.asyncio
    async def test_path_dot_reads_each_root_whole(self, tmp_path: Path) -> None:
        _write(tmp_path / "one" / "SKILL.md", _skill("one"))
        resource = LocalDirResource(id="p", paths=[str(tmp_path / "one")])
        source = _bound(ResourceSkillSource(resource="p", path="."), resource)
        assert [s.name for s in (await source.index(SkillQuery())).skills] == ["one"]

    @pytest.mark.asyncio
    async def test_the_earlier_resource_wins_a_name(self, tmp_path: Path) -> None:
        _write(tmp_path / "skills" / "one" / "SKILL.md", _skill("one", "local"))
        local = LocalDirResource(id="local", paths=[str(tmp_path)])
        far = InMemoryResource(
            id="far",
            files={"skills/one/SKILL.md": _skill("one", "far"), "skills/two.md": _skill("two")},
        )
        source = _bound(ResourceSkillSource(resource=["far", "local"]), local, far)

        loaded = await source.load(SkillQuery())

        assert sorted(s.name for s in loaded.index.skills) == ["one", "two"]
        assert "far" in loaded.body("one").body  # type: ignore[union-attr]
        assert local_source_paths(source) is None

    @pytest.mark.asyncio
    async def test_a_resource_not_on_this_machine_is_read_with_the_same_layout(self) -> None:
        far = InMemoryResource(
            id="far",
            files={
                "skills/loose.md": _skill("loose"),
                "skills/own/SKILL.md": _skill("own"),
                "skills/own/nested/SKILL.md": _skill("not-entered"),
                "skills/group/deep/SKILL.md": _skill("deep"),
                "skills/group/not-a-skill.md": _skill("not-loose-below-the-top"),
                "skills/.hidden/SKILL.md": _skill("hidden"),
                "skills/node_modules/x/SKILL.md": _skill("dependency"),
                "elsewhere/SKILL.md": _skill("outside-the-path"),
            },
        )
        source = _bound(ResourceSkillSource(resource="far"), far)

        loaded = await source.load(SkillQuery())

        assert sorted(s.name for s in loaded.index.skills) == ["deep", "loose", "own"]
        part = await source.fetch(SkillRef(name="own", section="Part"), SkillQuery())
        assert part.body.strip() == "part"

    @pytest.mark.asyncio
    async def test_an_unbound_source_refuses_rather_than_reads_nothing(self) -> None:
        with pytest.raises(ResourceError, match="never bound"):
            await ResourceSkillSource(resource="p").load(SkillQuery())

    def test_binding_an_undeclared_id_is_refused(self) -> None:
        with pytest.raises(ResourceError, match="'nope'"):
            ResourceSkillSource(resource="nope").bind({})

    @pytest.mark.parametrize("path", ["../x", "/abs"])
    def test_a_path_outside_the_resource_is_refused(self, path: str) -> None:
        with pytest.raises(ValueError):
            ResourceSkillSource(resource="p", path=path)


class TestInstructions:
    @pytest.mark.asyncio
    async def test_local_roots_are_read_as_a_local_dir_source_reads_them(
        self, tmp_path: Path
    ) -> None:
        soul = _write(tmp_path / "me" / "SOUL.md", "be kind")
        _write(tmp_path / "project" / "SOUL.md", "be terse")
        brief = _write(tmp_path / "project" / "AGENTS.md", "the brief")
        resource = LocalDirResource(id="i", paths=[str(tmp_path / "me"), str(tmp_path / "project")])
        source = ResourceInstructionsSource(resource="i").bind({"i": resource})

        loaded = await source.load()
        local = await LocalDirInstructionsSource(
            paths=[tmp_path / "me", tmp_path / "project"]
        ).load()

        assert loaded == local
        assert [p.path for p in loaded.parts] == [str(soul.resolve()), str(brief.resolve())]

    @pytest.mark.asyncio
    async def test_the_first_soul_wins_across_resources(self, tmp_path: Path) -> None:
        _write(tmp_path / "AGENTS.md", "local brief")
        _write(tmp_path / "SOUL.md", "local soul")
        far = InMemoryResource(id="far", files={"SOUL.md": "far soul", "AGENT.md": "far brief"})
        local = LocalDirResource(id="local", paths=[str(tmp_path)])
        source = ResourceInstructionsSource(resource=["far", "local"]).bind(
            {"far": far, "local": local}
        )

        loaded = await source.load()

        assert [p.text for p in loaded.soul] == ["far soul"]
        assert [p.text for p in loaded.agent] == ["far brief", "local brief"]
        assert loaded.superseded == (str((tmp_path / "SOUL.md").resolve()),)
        assert loaded.soul[0].path == "far:SOUL.md"

    @pytest.mark.asyncio
    async def test_a_subpath_of_a_remote_resource(self) -> None:
        far = InMemoryResource(id="far", files={"team/AGENTS.md": "team", "AGENTS.md": "top"})
        source = ResourceInstructionsSource(resource="far", path="team").bind({"far": far})
        loaded = await source.load()
        assert [(p.text, p.instruction_kind) for p in loaded.parts] == [
            ("team", InstructionKind.AGENT)
        ]

    @pytest.mark.asyncio
    async def test_an_unbound_source_refuses(self) -> None:
        with pytest.raises(ResourceError, match="never bound"):
            await ResourceInstructionsSource(resource="i").load()


class TestEntries:
    """``resource:`` entries can say what they read, when it differs from the source's."""

    @pytest.mark.asyncio
    async def test_each_entry_reads_its_own_path(self, tmp_path: Path) -> None:
        _write(tmp_path / "project" / ".domyn" / "skills" / "local" / "SKILL.md", _skill("local"))
        _write(tmp_path / "mine" / "review" / "SKILL.md", _skill("review"))
        auto = LocalDirResource(id="auto", paths=[str(tmp_path / "project" / ".domyn")])
        mine = LocalDirResource(id="mine", paths=[str(tmp_path / "mine")])
        source = ResourceSkillSource(
            resource=[{"id": "auto", "path": "skills"}, "mine"], path="."
        ).bind({"auto": auto, "mine": mine})

        loaded = await source.load(SkillQuery())

        assert sorted(s.name for s in loaded.index.skills) == ["local", "review"]
        assert local_source_paths(source) == [
            tmp_path / "project" / ".domyn" / "skills",
            tmp_path / "mine",
        ]
        assert source.describe() == "resource(auto/skills, mine)"

    def test_walk_is_for_instructions_only(self) -> None:
        with pytest.raises(ValueError, match="walk"):
            ResourceSkillSource(resource=[{"id": "auto", "walk": True}])

    @pytest.mark.parametrize("entry", [{"id": "a", "other": 1}, {"path": "x"}, 3])
    def test_a_malformed_entry_is_refused(self, entry: object) -> None:
        with pytest.raises(ValueError):
            ResourceSkillSource(resource=[entry])

    @pytest.mark.asyncio
    async def test_instructions_walk_only_the_entries_that_say_so(self, tmp_path: Path) -> None:
        for folder in ("walked", "flat"):
            _write(tmp_path / folder / "AGENTS.md", f"{folder} top")
            _write(tmp_path / folder / "deep" / "AGENTS.md", f"{folder} deep")
        walked = LocalDirResource(id="walked", paths=[str(tmp_path / "walked")])
        flat = LocalDirResource(id="flat", paths=[str(tmp_path / "flat")])
        source = ResourceInstructionsSource(resource=["flat", {"id": "walked", "walk": True}]).bind(
            {"walked": walked, "flat": flat}
        )

        loaded = await source.load()

        assert [p.text for p in loaded.agent] == ["flat top", "walked top", "walked deep"]
        assert source.walked_paths() == [tmp_path / "walked"]

    @pytest.mark.asyncio
    async def test_a_resource_not_on_this_machine_is_walked_through_its_listing(self) -> None:
        far = InMemoryResource(
            id="far",
            files={
                "SOUL.md": "soul",
                "sub/AGENTS.md": "deep",
                "skills/x/AGENTS.md": "in a skill folder",
                ".hidden/AGENTS.md": "hidden",
            },
        )
        source = ResourceInstructionsSource(resource=[{"id": "far", "walk": True}]).bind(
            {"far": far}
        )

        loaded = await source.load()

        # Nothing is skipped by name, a skill folder included: breadth first.
        assert [p.text for p in loaded.parts] == ["soul", "deep", "in a skill folder"]
