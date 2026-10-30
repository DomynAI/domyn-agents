from dotenv import find_dotenv, load_dotenv

from domyn.agents.cli.cli import app


def _load_dotenv() -> None:
    """Load a .env file from the working directory (or a parent of it), if present.

    Existing process env vars always win (override=False) - a .env file
    supplements the shell environment, it doesn't shadow it.
    """
    load_dotenv(find_dotenv(usecwd=True), override=False)


def _load_credentials() -> None:
    """Expose ``~/.domyn/credentials`` as environment variables.

    The default config is a plain ``ConfigurationV1`` file: it can only refer to
    a secret via ``${VAR_NAME}`` interpolation, resolved once against
    ``os.environ`` when the YAML loads (``domyn/agents/loader/__init__.py``).
    There is no credentials-file fallback at that layer. Doing the injection
    here — before the YAML is ever read — lets `api_key: "${OPENAI_API_KEY}"`
    keep working exactly as it does for a var set in the shell, so the 0600
    credentials store (`interfaces/cli/config.py`'s `write_credential`) stays a
    transparent alternative to exporting the variable yourself.

    A real environment variable always wins, matching `resolve_api_key`'s
    existing priority (env first, then the credentials file) for the code paths
    that still call it directly.

    This runs before any option is parsed, so it reads ``~/.domyn/credentials``;
    ``--project`` puts the project's keys over it once it is known (`select_project`).
    """
    from domyn.agents.interfaces.cli.config import load_credentials_into_environment

    load_credentials_into_environment()


def cli() -> None:
    _load_dotenv()
    _load_credentials()
    app()


if __name__ == "__main__":
    cli()
