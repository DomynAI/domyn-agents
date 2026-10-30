<!--
The title is the commit on main and the line in the release notes:
`type(scope): description`, see CONTRIBUTING.md. A security fix does not go in
a public pull request: see SECURITY.md.
-->

## Summary

<!-- One or two sentences: what changes, for whom. -->

## Why

<!-- The problem this solves, or what it makes possible. Link the accepted issue:
Fixes #123. Only a small fix (a typo, docs, a few-line bug fix) or a maintainer's
pull request goes without one. -->

## What changed

<!-- What a user sees first (CLI, configuration, Python API, YAML), then what is
internal. Name the files where it helps a reviewer find their way. -->

-

## Breaking changes

<!-- "None", or what stops working and what to do instead. A breaking change
also takes a `!` in the title (`feat!: ...`), which puts it first in the notes. -->

None.

## How it was tested

<!-- The tests added or changed, and anything run by hand: the commands, and the
model or provider for a change in how the agent behaves. Output or a screenshot
helps for anything shown in the terminal. -->

## Not in this pull request

<!-- Optional: what was left out on purpose, and follow-ups. Delete if empty. -->

## Checklist

- [ ] It implements an accepted issue, or it is a small fix, or I am a maintainer
- [ ] One change per pull request, with nothing unrelated in it
- [ ] If a coding agent wrote it: it followed `AGENTS.md`, and I checked the whole diff and what it implies
- [ ] `make check-all` passes
- [ ] Tests cover the new behaviour (for a bug, a test that fails without the fix)
- [ ] The pages under `docs/docs/` say what the change does to them (a page added or retitled also needs its line in `domyn-docs/SKILL.md`)
- [ ] No secrets, private URLs or customer data in code, tests or examples
