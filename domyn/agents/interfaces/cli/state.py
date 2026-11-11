"""What the CLI remembers between sessions, in one typed file.

``state.yaml`` lives in the folder that plays ``~/.domyn`` (`home_dir`): the selected
``--project``'s own folder, else ``~/.domyn`` itself, so each project has its own and
the global one serves a session with none. It is for settings a person turns on
while using the tool and expects to find again — not configuration (that is
``config.yaml``, edited deliberately), not secrets (``credentials``) and not grants
(``approvals.yaml``). Never in the repo, so a clone cannot ship a setting that
loosens a safeguard, and never exported (`export.is_private`).

The file is always read and written as a whole `CliState` (`load`, `save`): a new
setting is a new field on it, with its default, and nothing else here changes. A file that is missing, unreadable or invalid is the defaults; a key this
version does not know is kept when another is written, so a newer version's
settings are not erased by an older one.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from domyn.agents.interfaces.cli.config import PrivateFiles, home_dir
from domyn.agents.logger import get_logger


class CliState(BaseModel):
    """The settings remembered between sessions. Every field has a default, which is
    what a session with no file starts with."""

    model_config = ConfigDict(extra="ignore")

    verbose: bool = False
    """Show internal events and library logs (`/verbose`, `domyn run -v`)."""

    auto_approve: bool = False
    """Answer yes to every approval prompt (`/auto`, Shift+Tab).

    Still not a standing grant: nothing is written to an approval store and the
    hardline rules still apply. Only whether the toggle starts on is remembered."""

    @classmethod
    def path(cls) -> Path:
        return home_dir() / PrivateFiles.STATE

    @classmethod
    def _read_raw(cls) -> dict[str, Any]:
        path = cls.path()
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None
        except (OSError, yaml.YAMLError) as exc:
            get_logger().warning(f"cannot read {PrivateFiles.STATE}: {type(exc).__name__}")
            get_logger().debug(f"state read detail: {exc}")
            return {}
        return raw if isinstance(raw, dict) else {}

    @classmethod
    def load(cls) -> Self:
        """The remembered settings. Never raises: what cannot be read is the defaults."""
        try:
            return cls.model_validate(cls._read_raw())
        except ValidationError as exc:
            get_logger().warning(f"invalid {PrivateFiles.STATE}: using the defaults")
            get_logger().debug(f"state validation detail: {exc}")
            return cls()

    def save(self) -> None:
        """Remember these settings, keeping any key this version does not know as it was.

        A failure to write is not the session's problem: the settings still hold for
        this run, they just are not remembered.
        """
        raw = {**self._read_raw(), **self.model_dump()}
        path = self.path()
        tmp: str | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Written beside and renamed, as the approval store does: a reader sees the
            # old file or the new one, never half of one.
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                yaml.safe_dump(raw, handle, sort_keys=False)
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        except (OSError, yaml.YAMLError) as exc:
            if tmp is not None:
                Path(tmp).unlink(missing_ok=True)
            get_logger().warning(f"cannot save {PrivateFiles.STATE}: {type(exc).__name__}")
            get_logger().debug(f"state write detail: {exc}")
