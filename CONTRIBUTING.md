# Contributing to Domyn Agents

Thank you for helping. This page explains how to propose a change, set up the project, check your work and open a pull request. It exists to save time on both sides: following it is the quickest way to a merged pull request.

## Before you start

**Search the [issues](https://github.com/domynai/domyn-agents/issues?q=is%3Aissue) first**, open and closed. If yours is there, add what is new to it rather than opening another.

**For contributors outside the maintainers, anything beyond a small fix starts with an issue that a maintainer has accepted.** Open one with the *Proposal* template, say what you want to change and why, and wait for the `accepted` label before writing the code: a new feature, a change to the Python API, the YAML format or the CLI, or a change in how the agent behaves is a design decision first. Small fixes do not need one: a typo, a documentation fix, a bug fix of a few lines with its test. A larger pull request without an accepted issue may be closed, however good the code. Maintainers (the code owners, [`.github/CODEOWNERS`](.github/CODEOWNERS)) open pull requests without one.

### Skill, tool, plugin or core?

The core (`domyn/agents/`) is kept small, so that it stays maintainable and others can build on it. Before proposing a change to it, check whether it can live outside:

- **A skill** (`SKILL.md`, see the documentation) when it is knowledge or a procedure the agent should follow: a workflow, a document format, a way of using an existing tool. Skills ship without a release and anyone can keep their own. A builtin skill (`domyn/agents/skills/builtin/`) ships to everyone who installs the package, so it has to be useful to most of them and carry no company's name or brand.
- **A tool** (a Python function with `@tool`, or an MCP server) when the agent needs to *do* something it cannot do with the built-in file, shell and browser tools. A tool for one service belongs in your project or in an MCP server, not in the core.
- **A provider, channel or plugin** when it connects Domyn Agents to something else: an LLM API, a messaging platform, another agent framework.
- **The core** when the framework itself has to change: the runner, the planners, context management, approvals, the loader, the CLI. Even a new extension point is discussed in an issue first.

## Issues

Use one of the templates: **Bug**, **Proposal** or **Question** (questions are welcome as issues). Keep it short and concrete:

- If it does not fit on one screen, it is too long.
- Write in your own words. If you used an LLM to write or research it, say so.
- State the problem, why it matters, and the smallest way to reproduce it.
- Say if you would like to implement it yourself.

Remove API keys, tokens and anything private from logs and configurations before pasting them.

## Using a coding agent

Coding agents are welcome, and you are responsible for everything yours produces:

- **Run it from the repository root, so that it reads [`AGENTS.md`](AGENTS.md)**, and have it follow that file: it holds the architecture, the conventions and the rules a change here has to respect.
- **Check what your agent did, and what it implies,** before opening the pull request: read the whole diff, and think through how it changes behaviour, security (what the agent may run or read, what reaches the logs) and the parts of the system it touches without saying so. You should be able to explain every line; a pull request whose author cannot is closed.
- Do not send issues or pull requests in bulk through automation.

## Setting up

You need Python 3.11 or later and [uv](https://docs.astral.sh/uv/). Clone the repository and install every dependency, extras included:

```sh
git clone https://github.com/domynai/domyn-agents.git
cd domyn-agents
uv sync --all-extras
```

The SDK of the Domyn platform is not part of that: it is distributed with the platform, not on an index this repo can reach, so the lock does not hold it. Install it next to the rest only if you work on the platform integration (`domyn expose`, the platform tools, the relay channel):

```sh
uv pip install "domyn-platform-sdk>=0.37.0"
```

A later `uv sync` removes it again; `uv sync --inexact` and `uv run` leave it. The minimum version is `MIN_SDK_VERSION` in `domyn/agents/integrations/domyn_platform_sdk.py`, which also says so when it is missing or too old.

## Running the tests

```sh
uv run pytest tests/ examples/ -q --no-cov
```

The suite runs in parallel across your cores. Add `-n0` to run it in one process, for a debugger or to read the output in order. To run part of it:

```sh
uv run pytest tests/cli/test_repl_session.py -q --no-cov     # one file
uv run pytest tests/ -q --no-cov -k "compaction"             # by name
```

`tests/` mirrors the layout of `domyn/agents/`, so the tests for a module are in the same place under `tests/`. Async tests carry an explicit `@pytest.mark.asyncio`.

## Checking a change

Before opening a pull request, run every check:

```sh
make check-all
```

This runs the linter, the format check and the tests, the same checks CI runs on a pull request. Two of them can fix what they find: `make lint-fix` and `make format`.

New code in `domyn/agents/` must be fully type-annotated. `make mypy` runs mypy in strict mode; it still reports errors in existing code, so it is not part of `check-all` yet, but it should report no new errors in the files you change.

Pull requests run the same lint, format and test checks in CI. On `main`, the tests also run on every supported Python version, 3.11 to 3.15.

## Working on the documentation

The documentation site is in [`docs/`](docs/), built with [Docusaurus](https://docusaurus.io/). It needs Node 22.19+ and pnpm, in the version pinned by `packageManager` in `docs/package.json` (`corepack enable` picks it up):

```sh
cd docs
make install   # once
make serve     # http://localhost:3000, reloads on save
make build     # the static site, in build/
```

Pages are the `.md` files under `docs/docs/`. [`docs/README.md`](docs/README.md) explains how to add one.

When a change alters behaviour a page describes, update the page in the same pull request. A broken link fails the build, and pull requests that touch `docs/` build the site in CI. The site is published to <https://domynai.github.io/domyn-agents/> with every release tag.

The CLI agent reads the same pages, through the `domyn-docs` builtin skill (`domyn/agents/skills/builtin/cli/domyn-docs/`), whose `SKILL.md` lists every page with its title and description. When you add, remove or rename a page, or change its title or description, update that list too; a test fails until it matches and prints the lines it expects.

## Opening a pull request

- Link the accepted issue it implements (`Fixes #123`), unless it is a small fix or you are a maintainer (see [Before you start](#before-you-start)).
- Keep a pull request to one change, and say in its description what it changes and why. The pull request template has the sections to fill in.
- Title it as a [Conventional Commit](https://www.conventionalcommits.org/): `type: what it does`, or `type(scope): …`, with `!` after the type for a breaking change (`feat!: …`). The types are `feat` (or `add`) for a feature, `fix` for a bug fix, `perf`, `remove` for a feature or behaviour taken out, `docs`, and for the rest `refactor`, `test`, `build`, `ci`, `chore` (or `change`) and `revert`. A required check fails on any other title, and the pull request cannot merge until it is fixed. Pull requests are squash-merged with their title as the commit message, and every release lists them by title.
- Add tests for new behaviour, and a test that fails without the fix for a bug. The `test` check, required to merge, fails when fewer than 80% of the lines the pull request adds or changes run in a test (the job summary lists the ones that do not), or when the total coverage drops below `main`'s by more than 0.1%. For code that can only run against a real service, a maintainer adds the `coverage-exempt` label, says why in the pull request, and re-runs the job.
- Make sure `make check-all` passes.
- Every pull request needs the approval of a maintainer other than its author (the code owners, [`.github/CODEOWNERS`](.github/CODEOWNERS)); GitHub asks them for the review. Only maintainers can merge.

## Reporting bugs and security issues

Open an [issue](https://github.com/domynai/domyn-agents/issues/new/choose) for a bug, a proposal or a question (see [Issues](#issues)). Security vulnerabilities are reported privately instead: see [SECURITY.md](SECURITY.md).

## Code of conduct

Everyone taking part in the project is expected to follow the [code of conduct](CODE_OF_CONDUCT.md).

Ignoring this guide repeatedly, or filling the tracker with issues or pull requests generated in bulk, gets an account blocked from the project.

## License

By contributing, you agree that your contributions are licensed under the project's [Apache License 2.0](LICENSE).
