import logging
from importlib.metadata import PackageNotFoundError, version

package_name = "domyn"


def get_version() -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"


LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def setup_logging(log_level: str) -> int:
    lvl = LOG_LEVELS.get(log_level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # basicConfig installs a handler and sets the level, but only the first time:
    # once the root logger has handlers it returns without doing either. The
    # level is therefore set explicitly, so a later call actually changes it
    # rather than silently keeping whatever the first caller picked.
    logging.getLogger().setLevel(lvl)
    return lvl
