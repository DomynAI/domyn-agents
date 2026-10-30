# braces (patched copy)

`braces` 3.0.3 as published (MIT, see `LICENSE`), plus one change: the parser rejects brace
nesting deeper than `MAX_DEPTH` (256) with a `SyntaxError`, in `lib/parse.js` and
`lib/constants.js`.

Why: CVE-2026-93687 / GHSA-vfj7-8cjw-p6xm. The tree walkers in `compile.js` and `expand.js` recurse
once per nesting level, and `{`×4900 + `a` + `}`×4900, under the 10 000 character limit, overflows
the stack. No patched release exists (3.0.3 is the last, from September 2024); the fixes in the
upstream repository are open pull requests.

It replaces the registry package through `overrides` in `../pnpm-workspace.yaml`, for the
dependencies of Docusaurus (`micromatch`, `chokidar`). Drop this directory and the override once a
patched `braces` is published. `braces.test.js` checks the limit, and CI runs it:

```bash
pnpm run test:braces
```
