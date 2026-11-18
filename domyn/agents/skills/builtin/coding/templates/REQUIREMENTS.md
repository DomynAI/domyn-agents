# Requirements: <task title>

## Goal

<One or two sentences: what will be true when this is done, and for whom.>

## Requirements

| # | Requirement | Source | Checked by |
|---|---|---|---|
| R1 | <what must be true> | <user's request / test / doc / config, with path> | <test name or command> |
| R2 | | | |

## Exact names

<Paths, function and command names, flags, ports, formats and output wording the request
fixes, copied verbatim. Write "none" if the request fixes none.>

## Out of scope

<What this task does not change. Anything noticed along the way that is out of scope goes
here too, to be mentioned in the final message.>

## Open questions

<Questions not yet settled, each with where to look for the answer.>

### Resolved

- <question> → <answer>. Evidence: <file, test or doc that settles it>.

### Assumptions

<Only what the repository could not settle: the reading chosen, and why it is the one most
consistent with the existing code.>

## Baseline

- Test command: `<command>`
- Result before any change: <N passed, M failed; names of the failing tests>
- Lint / format / type checks: `<commands>`, <result>

## Definition of DONE

The task is finished only when every item is checked, with its evidence written beside it.
An item without evidence is not met. `BLOCKED` only for causes outside the agent's control,
with the reason and what was tried.

- [ ] Every requirement's check passes (R1…Rn). Evidence:
- [ ] What was built ran the way it will be used; its output matches the request. Evidence:
- [ ] The whole test suite ran after the last edit; nothing fails that did not fail in the
      baseline. Evidence:
- [ ] Lint, format and type checks pass on the changed code. Evidence:
- [ ] The full diff has been read; it holds nothing unrelated to the task. Evidence:
- [ ] `PLAN.md` is up to date and every step is done. Evidence:
- [ ] <task-specific condition, checkable by a command or by reading something>. Evidence:
