"""A configuration's ``resources:``: what loading checks, and what it binds."""

from pathlib import Path
from typing import Any

import pytest

from domyn.agents.agents import Agent
from domyn.agents.core.interfaces import BaseAgent
from domyn.agents.loader import ConfigurationV1
from domyn.agents.resources import LocalDirResource, ResourceError
from tests.resources.fake import InMemoryResource

_LLM = {
    "name": "p",
    "type": "vllm",
    "model_name": "m",
    "url": "http://localhost:8000/v1/chat/completions",
}


def _agent(name: str, **fields: Any) -> dict[str, Any]:
    return {"name": name, "description": "d", "instruction": "i", "llm_provider": "p", **fields}


def _config(tmp_path: Path, *agents: dict[str, Any], **extra: Any) -> dict[str, Any]:
    (tmp_path / "work").mkdir(exist_ok=True)
    (tmp_path / "sub").mkdir(exist_ok=True)
    return {
        "version": "1.0",
        "entrypoint": agents[0]["name"],
        "llm_providers": [_LLM],
        "resources": [
            {"id": "project", "type": "local_dir", "paths": [str(tmp_path / "work")]},
            {"id": "sub_files", "type": "local_dir", "paths": [str(tmp_path / "sub")]},
        ],
        "tool_bundles": [{"name": "cli_tools", "bundles": ["files", "shell"]}],
        "agents": list(agents),
        **extra,
    }


def _agents(config: ConfigurationV1) -> dict[str, Agent]:
    return {a.name: a for a in config.agents if isinstance(a, Agent)}


class TestBinding:
    def test_one_bundle_is_bound_to_each_agents_own_resource(self, tmp_path: Path) -> None:
        config = ConfigurationV1.from_dict(
            _config(
                tmp_path,
                _agent(
                    "Main",
                    tool_bundles=[{"use": "cli_tools", "workspace": "project"}],
                    sub_agents=["Researcher"],
                ),
                _agent("Researcher", tool_bundles=[{"use": "cli_tools", "workspace": "sub_files"}]),
            )
        )
        agents = _agents(config)
        main_space = agents["Main"].builtin_tools[0].workspace
        sub_space = agents["Researcher"].builtin_tools[0].workspace
        assert main_space is not None and main_space.root == (tmp_path / "work").resolve()
        assert sub_space is not None and sub_space.root == (tmp_path / "sub").resolve()
        assert [r.id for r in config.resources] == ["project", "sub_files"]

    def test_the_other_roots_are_reachable(self, tmp_path: Path) -> None:
        (tmp_path / "more").mkdir()
        raw = _config(
            tmp_path, _agent("Main", tool_bundles=[{"use": "cli_tools", "workspace": "project"}])
        )
        raw["resources"][0]["paths"].append(str(tmp_path / "more"))
        space = _agents(ConfigurationV1.from_dict(raw))["Main"].builtin_tools[0].workspace
        assert space is not None
        assert space.extra_roots == ((tmp_path / "more").resolve(),)

    def test_the_effective_set_is_the_list_then_every_reference(self, tmp_path: Path) -> None:
        config = ConfigurationV1.from_dict(
            _config(
                tmp_path,
                _agent(
                    "Main",
                    resources=["sub_files", "project"],
                    tool_bundles=[{"use": "cli_tools", "workspace": "project"}],
                    skills={"source": {"type": "resource", "resource": "sub_files"}},
                ),
            )
        )
        assert [r.id for r in _agents(config)["Main"].resources] == ["sub_files", "project"]

    def test_a_reference_alone_grants_access(self, tmp_path: Path) -> None:
        config = ConfigurationV1.from_dict(
            _config(
                tmp_path,
                _agent(
                    "Main",
                    tool_bundles=[{"use": "cli_tools", "workspace": "project"}],
                    standing_instructions={"source": {"type": "resource", "resource": "sub_files"}},
                ),
            )
        )
        assert [r.id for r in _agents(config)["Main"].resources] == ["project", "sub_files"]

    def test_a_resource_nothing_reads_is_harmless(self, tmp_path: Path) -> None:
        config = ConfigurationV1.from_dict(_config(tmp_path, _agent("Main")))
        assert _agents(config)["Main"].resources == []

    def test_the_sources_are_bound_to_their_resources(self, tmp_path: Path) -> None:
        config = ConfigurationV1.from_dict(
            _config(
                tmp_path,
                _agent(
                    "Main",
                    skills={"source": {"type": "resource", "resource": ["sub_files", "project"]}},
                ),
            )
        )
        source = _agents(config)["Main"].skills.source  # type: ignore[union-attr]
        assert [r.id for r in source.bound] == ["sub_files", "project"]  # type: ignore[attr-defined]

    def test_to_dict_keeps_them(self, tmp_path: Path) -> None:
        config = ConfigurationV1.from_dict(_config(tmp_path, _agent("Main")))
        assert [r["id"] for r in config.to_dict()["resources"]] == ["project", "sub_files"]

    @pytest.mark.parametrize("bundles", [["shell"], ["files", "browser"]])
    def test_an_execution_bundle_may_work_in_a_read_only_resource(
        self, tmp_path: Path, bundles: list[str]
    ) -> None:
        # Enforcing read-only is the resource's job, through its mount: the shell is
        # still there for everything else.
        raw = _config(tmp_path, _agent("Main", tool_bundles=[{"use": "b", "workspace": "project"}]))
        raw["resources"][0]["read_only"] = True
        raw["tool_bundles"].append({"name": "b", "bundles": bundles})
        names = {t.name for t in _agents(ConfigurationV1.from_dict(raw))["Main"].tools}
        assert ("run_command" in names) == ("shell" in bundles)
        assert ("browser_open" in names) == ("browser" in bundles)

    @pytest.mark.asyncio
    async def test_a_read_only_resource_keeps_the_files_bundle_from_writing(
        self, tmp_path: Path
    ) -> None:
        raw = _config(
            tmp_path, _agent("Main", tool_bundles=[{"use": "readers", "workspace": "project"}])
        )
        raw["resources"][0]["read_only"] = True
        raw["tool_bundles"].append({"name": "readers", "bundles": ["files"]})
        (tmp_path / "work" / "a.md").write_text("old")
        agent = _agents(ConfigurationV1.from_dict(raw))["Main"]
        tools = {t.name: t for t in agent.tools}

        written = await tools["write_file"].execute(path="a.md", content="new")
        assert not written.success and "read-only" in str(written.error)
        assert (tmp_path / "work" / "a.md").read_text() == "old"
        read = await tools["read_file"].execute(path="a.md")
        assert read.success and "old" in str(read.data)


class TestRefused:
    def test_a_duplicate_id(self, tmp_path: Path) -> None:
        raw = _config(tmp_path, _agent("Main"))
        raw["resources"].append({"id": "project", "type": "local_dir", "paths": ["."]})
        with pytest.raises(ResourceError, match="'project' is declared twice"):
            ConfigurationV1.from_dict(raw)

    def test_an_id_that_is_an_agents_name(self, tmp_path: Path) -> None:
        raw = _config(tmp_path, _agent("project"))
        with pytest.raises(ResourceError, match="'project' is also an agent's name"):
            ConfigurationV1.from_dict(raw)

    @pytest.mark.parametrize(
        "fields",
        [
            {"tool_bundles": [{"use": "cli_tools", "workspace": "nope"}]},
            {"resources": ["nope"]},
            {"skills": {"source": {"type": "resource", "resource": "nope"}}},
            {"standing_instructions": {"source": {"type": "resource", "resource": ["nope"]}}},
        ],
        ids=["bundle", "list", "skills", "instructions"],
    )
    def test_an_undeclared_id(self, tmp_path: Path, fields: dict[str, Any]) -> None:
        with pytest.raises(ResourceError, match="'nope', which no 'resources:' entry declares"):
            ConfigurationV1.from_dict(_config(tmp_path, _agent("Main", **fields)))

    def test_a_bundle_named_without_a_workspace(self, tmp_path: Path) -> None:
        raw = _config(tmp_path, _agent("Main", tool_bundles=["cli_tools"]))
        with pytest.raises(ResourceError, match=r"'Main' uses tool bundle 'cli_tools' without"):
            ConfigurationV1.from_dict(raw)

    def test_a_bundle_entry_without_a_workspace(self, tmp_path: Path) -> None:
        raw = _config(tmp_path, _agent("Main", tool_bundles=[{"use": "cli_tools"}]))
        with pytest.raises(ResourceError, match="add `workspace: <resource id>`"):
            ConfigurationV1.from_dict(raw)

    def test_a_resource_source_naming_none(self, tmp_path: Path) -> None:
        raw = _config(tmp_path, _agent("Main", skills={"source": {"type": "resource"}}))
        with pytest.raises(ResourceError, match="names none"):
            ConfigurationV1.from_dict(raw)

    def test_a_workspace_on_the_bundle_itself(self, tmp_path: Path) -> None:
        raw = _config(tmp_path, _agent("Main"))
        raw["tool_bundles"][0]["workspace"] = "."
        with pytest.raises(ValueError, match="'workspace' is no longer a key of the bundle"):
            ConfigurationV1.from_dict(raw)

    @pytest.mark.parametrize("bundles", [["shell"], ["browser"], ["files"]])
    def test_a_bundle_on_a_resource_not_on_this_machine(
        self, tmp_path: Path, bundles: list[str]
    ) -> None:
        raw = _config(tmp_path, _agent("Main", tool_bundles=[{"use": "b", "workspace": "far"}]))
        raw["resources"].append(InMemoryResource(id="far"))
        raw["tool_bundles"].append({"name": "b", "bundles": bundles})
        with pytest.raises(ResourceError, match="to a local_dir resource"):
            ConfigurationV1.from_dict(raw)


class TestTheSchema:
    # Both pieces on their own: the whole schema does not build on main either (an
    # agent's `decisions` field has no JSON schema).
    def test_it_documents_resources_and_the_bundle_binding(self) -> None:
        resources = ConfigurationV1._resources_schema()
        variants = resources["items"]["anyOf"]
        local = next(v for v in variants if v["properties"]["type"]["const"] == "local_dir")
        assert {"id", "paths", "read_only"} <= set(local["properties"])
        assert {"id", "type", "paths"} <= set(local["required"])
        agent = ConfigurationV1._customize_agent_schema_properties({})
        assert agent["tool_bundles"]["items"]["required"] == ["use", "workspace"]
        assert agent["resources"]["items"]["type"] == "string"


class TestNoAgentId:
    def test_an_agent_is_its_name(self) -> None:
        assert "id" not in BaseAgent.model_fields

    def test_a_resource_is_named_by_its_own_id(self, tmp_path: Path) -> None:
        resource = LocalDirResource(id="res_7c1e", paths=[str(tmp_path)])
        assert resource.id == "res_7c1e"


class TestLocalDirSourcesAreRefused:
    @pytest.mark.parametrize("key", ["skills", "standing_instructions"])
    def test_a_file_names_its_folders_through_a_resource(self, tmp_path: Path, key: str) -> None:
        raw = _config(
            tmp_path, _agent("Main", **{key: {"source": {"type": "local_dir", "paths": ["x"]}}})
        )
        with pytest.raises(ResourceError, match=r"'local_dir' is not accepted.*resources:"):
            ConfigurationV1.from_dict(raw)

    def test_the_classes_still_serve_code(self, tmp_path: Path) -> None:
        from domyn.agents.skills import LocalDirSkillSource, SkillsRuntime

        runtime = SkillsRuntime(source=LocalDirSkillSource(paths=[tmp_path]))
        assert runtime.source.paths == [tmp_path]  # type: ignore[attr-defined]


class TestMissingFolders:
    def test_a_relative_root_is_not_warned_about(self, tmp_path: Path) -> None:
        """A project's `.domyn` is missing from most folders the process runs in."""
        relative = LocalDirResource(id="a", paths=[".nowhere-at-all"])
        absolute = LocalDirResource(id="b", paths=[str(tmp_path / "missing")])
        assert relative.check() == []
        assert absolute.check() == [f"resource 'b': {tmp_path / 'missing'} does not exist"]
