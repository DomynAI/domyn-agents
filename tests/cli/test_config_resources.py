"""What `domyn setup` writes: its folders as resources, named by the bundle and sources."""

from pathlib import Path
from typing import Any

import yaml

from domyn.agents.interfaces.cli import config as cfg
from domyn.agents.interfaces.cli.export import config_paths
from domyn.agents.loader import ConfigurationV1

_REAL_INSTALL_DIR = cfg.skills_install_dir
"""The real one: the CLI conftest points it at a temporary folder for every test."""


def _written(**agent: Any) -> dict[str, Any]:
    return cfg.configuration_v1_dict_from(
        cfg.ProviderConfig(type="openai", model_name="m"),
        cfg.AgentConfig(**agent),
        cfg.CompactionConfig(),
    )


def _local(*paths: str) -> dict[str, Any]:
    return {"source": {"type": "local_dir", "paths": list(paths)}}


class TestWhatSetupWrites:
    def test_the_bundle_is_bound_to_a_workspace_resource(self) -> None:
        written = _written(tools=["files", "shell"], workspace="/srv/project")
        agent = written["agents"][0]

        assert agent["tool_bundles"] == [{"use": "cli_tools", "workspace": "workspace"}]
        assert "workspace" not in written["tool_bundles"][0]
        assert written["resources"] == [
            {"id": "workspace", "type": "local_dir", "paths": ["/srv/project"]}
        ]

    def test_skills_and_instructions_read_resources_of_their_own(self) -> None:
        written = _written(
            skills=_local(".domyn/skills", "/abs/review"),
            standing_instructions=_local("/me/SOUL.md", ".domyn"),
        )
        agent = written["agents"][0]

        assert agent["skills"] == {
            "source": {"type": "resource", "resource": "skills", "path": "."}
        }
        assert agent["standing_instructions"] == {
            "source": {"type": "resource", "resource": "instructions"}
        }
        assert written["resources"] == [
            {"id": "skills", "type": "local_dir", "paths": [".domyn/skills", "/abs/review"]},
            {"id": "instructions", "type": "local_dir", "paths": ["/me/SOUL.md", ".domyn"]},
        ]
        # The agent lists none: a reference already grants access.
        assert "resources" not in agent

    def test_nothing_chosen_writes_no_resource(self) -> None:
        written = _written(skills=_local(), standing_instructions=None)
        assert "resources" not in written
        assert "skills" not in written["agents"][0]

    def test_another_source_is_kept_as_it_is(self) -> None:
        memory = {"source": {"type": "memory", "skills": {}}}
        written = _written(skills=memory)
        assert written["agents"][0]["skills"] == memory
        assert "resources" not in written

    def test_it_loads(self, tmp_path: Path) -> None:
        (tmp_path / "SOUL.md").write_text("be kind")
        written = _written(
            tools=["files"],
            workspace=str(tmp_path),
            standing_instructions=_local(str(tmp_path / "SOUL.md")),
        )
        config = ConfigurationV1.from_dict(written)
        assert [r.id for r in config.agents[0].resources] == ["workspace", "instructions"]  # type: ignore[union-attr]


class TestReadingItBack:
    def test_setup_edits_the_paths_it_wrote(self) -> None:
        written = _written(
            tools=["files"],
            workspace="/srv/project",
            skills=_local(".domyn/skills"),
            standing_instructions=_local("/me/SOUL.md"),
        )
        _, agent, _ = cfg.current_settings_from(written)

        assert agent.workspace == "/srv/project"
        assert agent.tools == ["files"]
        assert agent.skills == _local(".domyn/skills")
        assert agent.standing_instructions == _local("/me/SOUL.md")

    def test_a_workspace_of_dot_reads_as_none(self) -> None:
        _, agent, _ = cfg.current_settings_from(_written(tools=["files"]))
        assert agent.workspace is None

    def test_a_source_reading_another_resource_is_left_to_its_author(self) -> None:
        raw = _written()
        raw["resources"] = [{"id": "project", "type": "local_dir", "paths": ["."]}]
        hand_written = {"source": {"type": "resource", "resource": "project"}}
        raw["agents"][0]["skills"] = hand_written

        _, agent, _ = cfg.current_settings_from(raw)

        assert agent.skills == hand_written
        assert cfg.source_paths(raw, hand_written, "skills") == ["./skills"]

    def test_a_rerun_keeps_resources_it_did_not_write_and_rewrites_its_own(self) -> None:
        existing = _written(skills=_local("/old"))
        existing["resources"].append({"id": "stash", "type": "local_dir", "paths": ["/stash"]})
        existing["agents"][0]["resources"] = ["stash"]

        rerun = cfg.configuration_v1_dict_from(
            cfg.ProviderConfig(type="openai", model_name="m"),
            cfg.AgentConfig(skills=_local("/new")),
            cfg.CompactionConfig(),
            existing=existing,
        )

        assert rerun["resources"] == [
            {"id": "skills", "type": "local_dir", "paths": ["/new"]},
            {"id": "stash", "type": "local_dir", "paths": ["/stash"]},
        ]
        assert rerun["agents"][0]["resources"] == ["stash"]

    def test_unticking_everything_drops_setups_resources(self) -> None:
        existing = _written(tools=["files"], skills=_local("/old"))
        rerun = cfg.configuration_v1_dict_from(
            cfg.ProviderConfig(type="openai", model_name="m"),
            cfg.AgentConfig(),
            cfg.CompactionConfig(),
            existing=existing,
        )
        assert "resources" not in rerun


class TestEditingInPlace:
    def test_set_agent_source_rewrites_the_resource_where_it_was(self) -> None:
        raw = _written(skills=_local("/a"), standing_instructions=_local("/s"))
        agent = raw["agents"][0]

        cfg.set_agent_source(raw, agent, "skills", _local(".domyn/skills", "/a"))

        assert [r["id"] for r in raw["resources"]] == ["skills", "instructions"]
        assert raw["resources"][0]["paths"] == [".domyn/skills", "/a"]

    def test_set_agent_source_with_nothing_removes_both(self) -> None:
        raw = _written(skills=_local("/a"))
        cfg.set_agent_source(raw, raw["agents"][0], "skills", None)
        assert "resources" not in raw
        assert "skills" not in raw["agents"][0]

    def test_a_legacy_local_dir_source_moves_onto_a_resource(self) -> None:
        raw = _written()
        raw["agents"][0]["skills"] = _local("/a")
        local = cfg.as_local_source(raw, "skills", raw["agents"][0]["skills"])
        cfg.set_agent_source(raw, raw["agents"][0], "skills", local)
        assert raw["agents"][0]["skills"]["source"]["type"] == "resource"
        assert cfg.source_paths(raw, raw["agents"][0]["skills"], "skills") == ["/a"]


class TestPathsTheConfigNames:
    def test_resource_roots_are_named_except_the_working_folder(self) -> None:
        raw = _written(tools=["files"], skills=_local("/abs/review"))
        assert config_paths(raw) == [("resource 'skills'", "/abs/review")]

    def test_project_terms_rewrite_resource_paths(self, tmp_path: Path) -> None:
        raw = _written(skills=_local(str(tmp_path / "skills" / "one")))
        written = cfg.in_project_terms(raw, tmp_path)
        assert written["resources"][0]["paths"] == [f"${{{cfg.PROJECT_DIR_VAR}}}/skills/one"]
        assert cfg.expand_project_dir(written, tmp_path)["resources"][0]["paths"] == [
            str(tmp_path / "skills" / "one")
        ]


def _skill(folder: Path, name: str, external: str = "") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    extra = f"external_files: [{external}]\n" if external else ""
    (folder / "SKILL.md").write_text(f"---\nname: {name}\ndescription: d\n{extra}---\nbody\n")


def _raw(resources: list[dict[str, Any]], **agent: Any) -> dict[str, Any]:
    return {
        "version": "1.0",
        "entrypoint": "A",
        "llm_providers": [{"name": "p", "type": "openai", "model_name": "m"}],
        "resources": resources,
        "agents": [
            {"name": "A", "description": "d", "instruction": "i", "llm_provider": "p", **agent}
        ],
    }


def _exported(raw: dict[str, Any], tmp_path: Path) -> tuple[dict[str, Any], list[str]]:
    from domyn.agents.interfaces.cli.export import export_configuration

    out = tmp_path / "out"
    export_configuration(raw, out, cwd=tmp_path)
    written = yaml.safe_load((out / "config.yaml").read_text())
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    return written, files


class TestExportingResourcesByPosition:
    def test_files_of_one_folder_stay_in_one_folder_across_resources(self, tmp_path: Path) -> None:
        notes = tmp_path / "team" / "notes"
        notes.mkdir(parents=True)
        (notes / "SOUL.md").write_text("soul")
        (notes / "AGENTS.md").write_text("brief")
        raw = _raw(
            [
                {"id": "souls", "type": "local_dir", "paths": [str(notes / "SOUL.md")]},
                {"id": "briefs", "type": "local_dir", "paths": [str(notes / "AGENTS.md")]},
            ],
            standing_instructions={"source": {"type": "resource", "resource": ["souls", "briefs"]}},
        )

        written, files = _exported(raw, tmp_path)

        assert "resources/notes/SOUL.md" in files and "resources/notes/AGENTS.md" in files
        paths = {r["id"]: r["paths"] for r in written["resources"]}
        assert paths == {
            "souls": [f"${{{cfg.PROJECT_DIR_VAR}}}/resources/notes/SOUL.md"],
            "briefs": [f"${{{cfg.PROJECT_DIR_VAR}}}/resources/notes/AGENTS.md"],
        }

    def test_a_subpath_still_reads_what_it_read_and_shared_files_keep_their_place(
        self, tmp_path: Path
    ) -> None:
        team = tmp_path / "team"
        _skill(team / "skills" / "review", "review", "../shared/style.md")
        (team / "skills" / "shared").mkdir()
        (team / "skills" / "shared" / "style.md").write_text("style")
        (team / "unrelated.txt").write_text("not read, not copied")
        raw = _raw(
            [{"id": "team", "type": "local_dir", "paths": [str(team), str(tmp_path / "nothing")]}],
            skills={"source": {"type": "resource", "resource": "team"}},
        )

        written, files = _exported(raw, tmp_path)

        assert files == [
            ".info",
            "config.yaml",
            "resources/team/skills/review/SKILL.md",
            "resources/team/skills/shared/style.md",
        ]
        # A root that names nothing here is left out, as the loader would skip it.
        assert written["resources"][0]["paths"] == [f"${{{cfg.PROJECT_DIR_VAR}}}/resources/team"]
        assert written["agents"][0]["skills"] == {
            "source": {"type": "resource", "resource": "team"}
        }

    def test_setups_own_shape_is_still_a_list_of_paths(self, tmp_path: Path) -> None:
        _skill(tmp_path / "mine" / "review", "review")
        written, files = _exported(
            _written(skills=_local(str(tmp_path / "mine" / "review"))), tmp_path
        )
        assert "resources/review/SKILL.md" in files
        skills = next(r for r in written["resources"] if r["id"] == "skills")
        assert skills["paths"] == [f"${{{cfg.PROJECT_DIR_VAR}}}/resources/review"]


class TestTheBuiltIns:
    def test_they_are_read_through_their_install_folder(self) -> None:
        install = cfg.skills_install_dir()
        _skill(install / "data" / "charts", "charts")
        _skill(install / "data" / "tables", "tables")
        written = _written(
            skills=_local(
                ".domyn/skills", str(install / "data" / "charts"), str(install / "data" / "tables")
            )
        )

        assert written["agents"][0]["skills"] == {
            "source": {"type": "resource", "resource": ["skills", "builtin_skills"], "path": "."}
        }
        assert written["resources"] == [
            {"id": "skills", "type": "local_dir", "paths": [".domyn/skills"]},
            {"id": "builtin_skills", "type": "local_dir", "paths": [str(install.absolute())]},
        ]
        # Setup reads them back one by one, the user's first.
        _, agent, _ = cfg.current_settings_from(written)
        assert agent.skills == _local(
            ".domyn/skills",
            str(install.absolute() / "data" / "charts"),
            str(install.absolute() / "data" / "tables"),
        )

    def test_none_chosen_writes_no_resource_for_them(self) -> None:
        written = _written(skills=_local(".domyn/skills"))
        assert [r["id"] for r in written["resources"]] == ["skills"]
        assert written["agents"][0]["skills"]["source"]["resource"] == "skills"


class TestWhereTheyAreInstalled:
    @staticmethod
    def _config_with(folder: str | None) -> dict[str, Any]:
        raw = _written()
        if folder is not None:
            raw["resources"] = [{"id": "builtin_skills", "type": "local_dir", "paths": [folder]}]
        return raw

    def test_the_config_s_resource_says_where(self, tmp_path: Path, monkeypatch: Any) -> None:
        config = tmp_path / "config.yaml"
        config.write_text(yaml.safe_dump(self._config_with(str(tmp_path / "elsewhere"))))
        monkeypatch.setattr(cfg, "config_path", lambda **_: config)

        assert _REAL_INSTALL_DIR() == tmp_path / "elsewhere"

    def test_without_one_it_is_the_home_s_builtin_skills(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(cfg, "config_path", lambda **_: tmp_path / "none.yaml")
        assert _REAL_INSTALL_DIR() == cfg.home_dir() / cfg.BUILTIN_SKILLS_DIRNAME

    def test_while_setup_runs_it_follows_the_config_being_edited(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """An import's config is not on disk until the wizard saves it."""
        monkeypatch.setattr(cfg, "config_path", lambda **_: tmp_path / "none.yaml")
        with cfg.editing_config(self._config_with(str(tmp_path / "imported"))):
            assert _REAL_INSTALL_DIR() == tmp_path / "imported"
        assert _REAL_INSTALL_DIR() == cfg.home_dir() / cfg.BUILTIN_SKILLS_DIRNAME

    def test_a_project_names_it_in_its_own_terms(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.setattr(cfg, "_active_project", tmp_path / "proj")
        monkeypatch.setattr(cfg, "config_path", lambda **_: tmp_path / "none.yaml")
        raw = self._config_with(f"${{{cfg.PROJECT_DIR_VAR}}}/resources/builtin-skills")
        with cfg.editing_config(raw):
            assert _REAL_INSTALL_DIR() == tmp_path / "proj" / "resources" / "builtin-skills"


class TestAutodiscovery:
    """`.domyn` and `.agents` are one resource, `autodiscovery`: skills read its
    `skills/`, instructions walk it."""

    @staticmethod
    def _real(monkeypatch: Any) -> None:
        monkeypatch.setattr(
            cfg, "LOCAL_SKILLS_DIRS", (Path(".domyn/skills"), Path(".agents/skills"))
        )
        monkeypatch.setattr(cfg, "AUTODISCOVERY_DIRS", (Path(".domyn"), Path(".agents")))

    def test_one_resource_serves_skills_and_instructions(self, monkeypatch: Any) -> None:
        self._real(monkeypatch)
        written = _written(
            skills=_local("/me/review", ".domyn/skills", ".agents/skills"),
            standing_instructions=_local(".domyn", "/me/SOUL.md", ".agents"),
        )

        resources = {r["id"]: r["paths"] for r in written["resources"]}
        assert resources == {
            "autodiscovery": [".domyn", ".agents"],
            "skills": ["/me/review"],
            "instructions": ["/me/SOUL.md"],
        }
        agent = written["agents"][0]
        # The project's skills first, the user's soul first: whatever order they came in.
        assert agent["skills"]["source"]["resource"] == [
            {"id": "autodiscovery", "path": "skills"},
            "skills",
        ]
        assert agent["standing_instructions"]["source"]["resource"] == [
            "instructions",
            {"id": "autodiscovery", "walk": True},
        ]

        _, read_back, _ = cfg.current_settings_from(written)
        assert read_back.skills == _local(".domyn/skills", ".agents/skills", "/me/review")
        assert read_back.standing_instructions == _local("/me/SOUL.md", ".domyn", ".agents")

    def test_it_loads_and_walks(self, monkeypatch: Any, tmp_path: Path) -> None:
        import asyncio

        self._real(monkeypatch)
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".domyn" / "deep").mkdir(parents=True)
        (tmp_path / ".domyn" / "deep" / "AGENTS.md").write_text("deep brief", encoding="utf-8")
        written = _written(standing_instructions=_local(".domyn", ".agents"))

        agent = ConfigurationV1.from_dict(written).agents[0]
        loaded = asyncio.run(agent.standing_instructions.load())  # type: ignore[union-attr]

        assert [p.text for p in loaded.parts] == ["deep brief"]

    def test_an_entry_setup_does_not_own_is_kept_in_its_place(self, monkeypatch: Any) -> None:
        self._real(monkeypatch)
        raw = _written(skills=_local(".domyn/skills", "/me/review"))
        raw["resources"].append(
            {"id": "autodiscovery_export", "type": "local_dir", "paths": ["/p/resources/domyn"]}
        )
        raw["agents"][0]["skills"]["source"]["resource"].insert(
            1, {"id": "autodiscovery_export", "path": "skills"}
        )

        _, agent, _ = cfg.current_settings_from(raw)
        rerun = cfg.configuration_v1_dict_from(
            cfg.ProviderConfig(type="openai", model_name="m"),
            agent,
            cfg.CompactionConfig(),
            existing=raw,
        )

        assert rerun["agents"][0]["skills"]["source"]["resource"] == [
            {"id": "autodiscovery", "path": "skills"},
            {"id": "autodiscovery_export", "path": "skills"},
            "skills",
        ]
        assert "autodiscovery_export" in {r["id"] for r in rerun["resources"]}

    def test_autodiscover_off_keeps_the_instructions_use(self, monkeypatch: Any) -> None:
        """The two share the resource: turning skills off leaves it to instructions."""
        self._real(monkeypatch)
        raw = _written(
            skills=_local(".domyn/skills", ".agents/skills"),
            standing_instructions=_local(".domyn", ".agents"),
        )
        agent = raw["agents"][0]

        cfg.set_agent_source(raw, agent, "skills", None)

        assert "skills" not in agent
        assert [r["id"] for r in raw["resources"]] == ["autodiscovery"]
        assert agent["standing_instructions"]["source"]["resource"] == {
            "id": "autodiscovery",
            "walk": True,
        }
