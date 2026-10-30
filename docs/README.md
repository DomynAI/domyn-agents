# Domyn Agents documentation

The documentation site for Domyn Agents, built with [Docusaurus 3](https://docusaurus.io/) and
published on GitHub Pages at <https://domynai.github.io/domyn-agents/>.

## Local development

Requires Node 22.19+ and pnpm (the version in `package.json`'s `packageManager`; `corepack enable`
picks it up). From this directory:

```bash
make install   # once
make serve     # http://localhost:3000, reloads on save
make build     # static site in build/
```

## Writing pages

- Pages are `.md`/`.mdx` files under `docs/`; the sidebars are in `sidebars.js`.
- Link to another page by its relative path to the file, `.md` included:
  `[Tools](../guides/tools.md)`.
- The CLI agent answers from these pages through the `domyn-docs` builtin skill
  (`domyn/agents/skills/builtin/cli/domyn-docs/`): its `pages` is a link to `docs/docs`, and its
  `SKILL.md` lists every page with the title and description of its front matter, in the order of
  `sidebars.js`. A page added, removed or retitled, or a description changed, needs its line there
  updated; `tests/skills/test_agent_docs.py` fails until it matches and prints the lines it expects.

A broken link fails the build, locally and in CI.

## Publishing

`.github/workflows/docs.yml` builds the site on every pull request that touches `docs/` and on every
push to `main`, to check it builds (a broken link fails it).

A release tag (`v*.*.*`) also publishes it, through `.github/workflows/release.yml` once the tests pass: the build goes to `<site>/<tag>/` on the `gh-pages`
branch, which GitHub Pages serves, and the root of the site redirects to the latest release
(pre-releases such as `v1.6.0-rc1` are published but never become the default; the nightly alphas, `v1.6.3a1`, are not published at all). Every version's
navbar has a dropdown that reads `versions.json` from the root of the site. The branch is
maintained by `scripts/publish_docs_version.py`, tested in `tests/scripts/`.

One-time setting: Settings > Pages > Source: *Deploy from a branch*, branch `gh-pages`, folder `/ (root)`.

`make serve` shows the dev version; the dropdown there lists only `dev`, since there is no
`versions.json` to read.
