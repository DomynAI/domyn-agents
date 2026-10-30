"""Tests for domyn/agents/cli/commands/init/command.py"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import tomlkit

from domyn.agents import Agent
from domyn.agents.cli.commands.init.command import (
    check_uv_installed,
    create_agent_file,
    create_dockerfile,
    create_readme,
    find_domyn_agents_checkout,
    init_command,
    modify_pyproject_toml,
)
from domyn.agents.cli.commands.init.command import run_command as run_subprocess_command
from domyn.agents.cli.commands.run.command import load_module_from_path

# ---------------------------------------------------------------------------
# check_uv_installed
# ---------------------------------------------------------------------------


class TestCheckUvInstalled:
    def test_returns_true_when_uv_is_available(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert check_uv_installed() is True
            mock_run.assert_called_once_with(["uv", "--version"], capture_output=True, check=True)

    def test_returns_false_when_uv_not_in_path(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("uv not found")):
            assert check_uv_installed() is False

    def test_returns_false_when_uv_returns_nonzero(self):
        with patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "uv"),
        ):
            assert check_uv_installed() is False


# ---------------------------------------------------------------------------
# run_subprocess_command (the local shell helper, not the typer command)
# ---------------------------------------------------------------------------


class TestRunSubprocessCommand:
    def test_returns_true_on_success(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="output", stderr="")
            result = run_subprocess_command(["uv", "sync"])
            assert result is True

    def test_returns_false_on_called_process_error(self):
        with patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "uv", stderr="error output"),
        ):
            result = run_subprocess_command(["uv", "sync"])
            assert result is False

    def test_returns_false_when_executable_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = run_subprocess_command(["nonexistent-tool", "arg"])
            assert result is False

    def test_passes_cwd_to_subprocess(self, tmp_path: Path):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            run_subprocess_command(["echo", "test"], cwd=tmp_path)
            _, kwargs = mock_run.call_args
            assert kwargs["cwd"] == tmp_path


# ---------------------------------------------------------------------------
# create_agent_file
# ---------------------------------------------------------------------------


class TestCreateAgentFile:
    def test_creates_agent_py_in_project_dir(self, tmp_path: Path):
        create_agent_file(tmp_path, "my-project")
        assert (tmp_path / "agent.py").exists()

    def test_file_contains_agent_class_instantiation(self, tmp_path: Path):
        create_agent_file(tmp_path, "test-project")
        content = (tmp_path / "agent.py").read_text()
        assert "Agent(" in content

    def test_generated_file_loads_as_domyn_run_would(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # `domyn run agent.py` loads it this way: the scaffold must import and build.
        monkeypatch.setitem(sys.modules, "user_module", None)  # restored afterwards
        create_agent_file(tmp_path, "test-project")

        module = load_module_from_path(str(tmp_path / "agent.py"))

        assert isinstance(module.agent, Agent)

    def test_file_contains_example_tools(self, tmp_path: Path):
        create_agent_file(tmp_path, "test-project")
        content = (tmp_path / "agent.py").read_text()
        assert "greet_user" in content
        assert "get_current_time" in content

    def test_file_includes_project_name_in_agent(self, tmp_path: Path):
        create_agent_file(tmp_path, "super-bot")
        content = (tmp_path / "agent.py").read_text()
        # project name appears in agent name or description
        assert "super-bot" in content or "super_bot" in content


# ---------------------------------------------------------------------------
# create_readme
# ---------------------------------------------------------------------------


class TestCreateReadme:
    def test_creates_readme_in_project_dir(self, tmp_path: Path):
        create_readme(tmp_path, "my-project")
        assert (tmp_path / "README.md").exists()

    def test_readme_contains_project_name(self, tmp_path: Path):
        create_readme(tmp_path, "awesome-bot")
        content = (tmp_path / "README.md").read_text()
        assert "awesome-bot" in content

    def test_readme_contains_run_instructions(self, tmp_path: Path):
        create_readme(tmp_path, "test")
        content = (tmp_path / "README.md").read_text()
        assert "domyn run" in content

    def test_readme_contains_setup_instructions(self, tmp_path: Path):
        create_readme(tmp_path, "test")
        content = (tmp_path / "README.md").read_text()
        assert "uv sync" in content


# ---------------------------------------------------------------------------
# modify_pyproject_toml
# ---------------------------------------------------------------------------


def _make_minimal_pyproject(tmp_path: Path) -> Path:
    """Write a minimal pyproject.toml suitable for testing."""
    doc = tomlkit.document()
    project = tomlkit.table()
    project.add("name", "test-project")
    project.add("version", "0.1.0")
    project.add("dependencies", tomlkit.array())
    doc.add("project", project)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(tomlkit.dumps(doc))
    return pyproject


class TestCreateDockerfile:
    def test_creates_dockerfile_in_project_dir(self, tmp_path: Path):
        create_dockerfile(tmp_path)
        assert (tmp_path / "Dockerfile").exists()

    def test_dockerfile_uses_python_slim_base(self, tmp_path: Path):
        create_dockerfile(tmp_path)
        content = (tmp_path / "Dockerfile").read_text()
        assert "python:3.12-slim" in content

    def test_dockerfile_installs_uv(self, tmp_path: Path):
        create_dockerfile(tmp_path)
        content = (tmp_path / "Dockerfile").read_text()
        assert "uv" in content

    def test_dockerfile_exposes_port_8000(self, tmp_path: Path):
        create_dockerfile(tmp_path)
        content = (tmp_path / "Dockerfile").read_text()
        assert "EXPOSE 8000" in content

    def test_dockerfile_cmd_runs_domyn_serve(self, tmp_path: Path):
        create_dockerfile(tmp_path)
        content = (tmp_path / "Dockerfile").read_text()
        assert "domyn" in content
        assert "serve" in content
        assert "agent.py" in content

    def test_dockerfile_cmd_binds_to_all_interfaces(self, tmp_path: Path):
        create_dockerfile(tmp_path)
        content = (tmp_path / "Dockerfile").read_text()
        assert "0.0.0.0" in content


class TestModifyPyprojectToml:
    def test_adds_domyn_agents_to_dependencies(self, tmp_path: Path):
        _make_minimal_pyproject(tmp_path)
        modify_pyproject_toml(tmp_path)
        content = (tmp_path / "pyproject.toml").read_text()
        assert "domyn" in content

    def test_depends_on_the_cli_extra(self, tmp_path: Path):
        """``uv sync`` in the new project must install the line editor too, or
        ``domyn run`` there falls back to a bare ``input()``."""
        _make_minimal_pyproject(tmp_path)
        modify_pyproject_toml(tmp_path)
        pyproject = tomlkit.loads((tmp_path / "pyproject.toml").read_text())
        assert "domyn[cli]" in pyproject["project"]["dependencies"]

    def test_local_checkout_becomes_a_relative_editable_source(self, tmp_path: Path):
        project_dir = tmp_path / "repo" / "my-bot"
        project_dir.mkdir(parents=True)
        _make_minimal_pyproject(project_dir)
        modify_pyproject_toml(project_dir, local_checkout=tmp_path / "repo")
        pyproject = tomlkit.loads((project_dir / "pyproject.toml").read_text())
        source = pyproject["tool"]["uv"]["sources"]["domyn"]
        assert source["path"] == ".."
        assert source["editable"] is True

    def test_gracefully_handles_missing_pyproject(self, tmp_path: Path):
        # Should not raise — logs error and returns
        modify_pyproject_toml(tmp_path)  # no pyproject.toml created

    def test_file_is_valid_toml_after_modification(self, tmp_path: Path):
        _make_minimal_pyproject(tmp_path)
        modify_pyproject_toml(tmp_path)
        tomlkit.loads((tmp_path / "pyproject.toml").read_text())  # must not raise


class TestFindDomynAgentsCheckout:
    def test_finds_the_checkout_from_a_subdirectory(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "domyn"\n')
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        assert find_domyn_agents_checkout(nested) == tmp_path

    def test_ignores_other_projects(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "something-else"\n')
        assert find_domyn_agents_checkout(tmp_path) is None

    def test_skips_an_unreadable_pyproject(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text("not = [valid")
        assert find_domyn_agents_checkout(tmp_path) is None

    def test_finds_this_repository(self):
        repo = Path(__file__).resolve().parents[2]
        assert find_domyn_agents_checkout(repo / "tests" / "cli") == repo


# ---------------------------------------------------------------------------
# init_command (integration)
# ---------------------------------------------------------------------------


def _patch_run_command(return_value: bool = True):
    """Patch the subprocess helper inside init/command.py."""
    return patch(
        "domyn.agents.cli.commands.init.command.run_command",
        return_value=return_value,
    )


def _patch_uv_check(installed: bool = True):
    return patch(
        "domyn.agents.cli.commands.init.command.check_uv_installed",
        return_value=installed,
    )


class TestInitCommand:
    def test_creates_agent_py_readme_and_dockerfile(self, tmp_path: Path):
        project_dir = tmp_path / "my-bot"
        with _patch_uv_check(), _patch_run_command():
            init_command(project_name="my-bot", directory=str(project_dir))
        assert (project_dir / "agent.py").exists()
        assert (project_dir / "README.md").exists()
        assert (project_dir / "Dockerfile").exists()

    def test_creates_gitignore(self, tmp_path: Path):
        project_dir = tmp_path / "my-bot"
        with _patch_uv_check(), _patch_run_command():
            init_command(project_name="my-bot", directory=str(project_dir))
        assert (project_dir / ".gitignore").exists()

    def test_exits_when_uv_not_installed(self, tmp_path: Path):
        project_dir = tmp_path / "my-bot"
        with _patch_uv_check(installed=False), pytest.raises(SystemExit) as exc_info:
            init_command(project_name="my-bot", directory=str(project_dir))
        assert exc_info.value.code == 1

    def test_exits_when_directory_exists_and_not_force(self, tmp_path: Path):
        project_dir = tmp_path / "existing"
        project_dir.mkdir()
        (project_dir / "existing_file.txt").write_text("content")
        with _patch_uv_check(), _patch_run_command(), pytest.raises(SystemExit) as exc_info:
            init_command(project_name="existing", directory=str(project_dir), force=False)
        assert exc_info.value.code == 1

    def test_force_flag_overwrites_existing_directory(self, tmp_path: Path):
        project_dir = tmp_path / "existing"
        project_dir.mkdir()
        (project_dir / "old_file.txt").write_text("old content")
        with _patch_uv_check(), _patch_run_command():
            init_command(project_name="existing", directory=str(project_dir), force=True)
        assert not (project_dir / "old_file.txt").exists()
        assert (project_dir / "agent.py").exists()

    def test_exits_when_uv_init_fails(self, tmp_path: Path):
        project_dir = tmp_path / "my-bot"
        with (
            _patch_uv_check(),
            _patch_run_command(return_value=False),
            pytest.raises(SystemExit) as exc_info,
        ):
            init_command(project_name="my-bot", directory=str(project_dir))
        assert exc_info.value.code == 1

    def test_uses_project_name_as_directory_when_no_directory_given(self, tmp_path: Path):
        # Change cwd so the default directory resolves predictably
        original_cwd = Path.cwd()
        import os

        os.chdir(tmp_path)
        try:
            with _patch_uv_check(), _patch_run_command():
                init_command(project_name="fresh-bot", directory=None)
            assert (tmp_path / "fresh-bot" / "agent.py").exists()
        finally:
            os.chdir(original_cwd)
