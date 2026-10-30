"""The built-in ``domyn-cli`` skill teaches an agent how to write a skill: what it says
about a skill's own files has to match what the code does with them."""

import re
from pathlib import Path

from domyn.agents.skills import parse_skill_file
from domyn.agents.skills.catalog import BUILTIN_SKILLS_ROOT

SKILL = BUILTIN_SKILLS_ROOT / "cli" / "domyn-cli" / "SKILL.md"


def _section(title: str) -> str:
    text = SKILL.read_text(encoding="utf-8")
    start = text.index(f"## {title}\n")
    end = text.find("\n## ", start + 1)
    return text[start : end if end != -1 else None]


class TestWhatItSaysAboutASkillsFiles:
    def test_it_says_every_file_must_be_listed_and_why(self) -> None:
        text = " ".join(_section("Adding a skill").split())

        assert "`external_files` must list every file the skill uses" in text
        assert "domyn export" in text  # an unlisted file stays behind
        assert "marked unavailable" in text  # a missing listed file

    def test_it_never_teaches_the_old_key(self) -> None:
        text = " ".join(SKILL.read_text(encoding="utf-8").split())

        assert "`required` lists" not in text
        assert "`required` files" not in text
        assert "`required` list" not in text


class TestWhereANewSkillGoes:
    """Autodiscovery on: ./.domyn/skills. Off: the configuration's skills/ folder, listed in
    config.yaml. Either way loaded from the next message, with no restart."""

    def test_it_tells_the_agent_how_to_tell_which_case_it_is(self) -> None:
        text = " ".join(_section("Adding a skill").split())

        assert "lists `.domyn/skills` and `.agents/skills`" in text
        assert "`/config`" in text

    def test_with_autodiscovery_on_it_goes_in_dot_domyn_skills_and_needs_nothing_else(
        self,
    ) -> None:
        text = " ".join(_section("Adding a skill").split())

        assert "create it in **`./.domyn/skills/<name>/`**" in text
        assert "it is loaded from the next message" in text

    def test_with_autodiscovery_off_it_goes_in_the_project_and_into_the_config(
        self,
    ) -> None:
        text = " ".join(_section("Adding a skill").split())

        assert "`$DOMYN_PROJECT_DIR/skills/<name>/` in a project" in text
        assert "`${DOMYN_PROJECT_DIR}/skills/<name>`" in text
        assert "**and add it to the config**" in text

    def test_an_unlisted_skill_is_said_never_to_load_and_the_config_is_checked(
        self,
    ) -> None:
        text = " ".join(_section("Adding a skill").split())

        assert "not listed in `config.yaml` is **never loaded**" in text
        assert "read the config back" in text

    def test_a_config_change_is_said_to_need_no_restart(self) -> None:
        text = " ".join(_section("Adding a skill").split())

        assert "**Both are read again before every message**" in text
        assert "**Check, then tell the user it is loaded from their next message.**" in text
        assert "restart" not in text.replace("no restart", "").replace("only after a restart", "")

    def test_it_warns_that_setup_rewrites_that_folder(self) -> None:
        text = " ".join(_section("Adding a skill").split())

        assert "every `domyn setup` rewrites that folder" in text


class TestWhichConfigurationIsInUse:
    """The agent cannot type ``/config``, and a project replaces ``~/.domyn`` entirely:
    it is told how to find the session's configuration, and to pass ``--project`` to
    any ``domyn`` command it runs, or that command changes ``~/.domyn`` instead."""

    def test_it_is_told_to_read_the_project_folder_from_the_shell(self) -> None:
        text = " ".join(_section("Where things live").split())

        assert 'echo "$DOMYN_PROJECT_DIR"' in text
        assert "**Nothing in `~/.domyn/` is used**" in text

    def test_every_domyn_command_it_runs_takes_the_project(self) -> None:
        text = " ".join(_section("Where things live").split())

        assert '`--project "$DOMYN_PROJECT_DIR"` when the variable is set' in text

    def test_remove_is_no_longer_said_to_lack_project_support(self) -> None:
        assert "not supported yet" not in SKILL.read_text(encoding="utf-8")


class TestWhereInstructionsGo:
    """The same rule as a new skill: ``./.domyn`` with autodiscovery on, the
    configuration's own folder otherwise, listed in the config: nothing is read
    without being listed."""

    def test_with_autodiscovery_on_they_go_in_dot_domyn(self) -> None:
        text = " ".join(_section("Writing SOUL.md or AGENTS.md").split())

        assert "the paths list `.domyn` or `.agents`" in text
        assert "**`./.domyn/SOUL.md`** or **`./.domyn/AGENTS.md`**" in text

    def test_with_autodiscovery_off_they_go_in_the_configuration_folder(self) -> None:
        text = " ".join(_section("Writing SOUL.md or AGENTS.md").split())

        assert "`$DOMYN_PROJECT_DIR/SOUL.md` in a project, `~/.domyn/SOUL.md` without one" in text
        assert "**and add it to the config**" in text
        assert "`${DOMYN_PROJECT_DIR}/SOUL.md`" in text
        assert "is **never loaded**" in text
        assert "no change to the config" not in text

    def test_an_existing_file_is_edited_not_doubled(self) -> None:
        text = " ".join(_section("Writing SOUL.md or AGENTS.md").split())

        assert "a second file adds to the first instead of replacing it" in text
