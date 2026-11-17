---
name: coding
version: 1.0.0
description: >
  Carry out a coding task in an existing or new codebase: a feature, a fix or
  refactor across several files or modules, a new component or service, or work likely to
  outlast one session. Keeps REQUIREMENTS.md (with a Definition of DONE), ARCHITECTURE.md and
  PLAN.md in a task folder and does not finish until DONE is met. 
tags: [coding, software-engineering, planning, testing]
external_files:
  - templates/
---

# Complex coding tasks

A method for coding work too large to hold in one head: the task is written down before it
is built, built in verified steps, and finished only when a written **Definition of DONE**
is met. Three files keep the work on track, survive a long conversation (the history may be
summarized, the files are not) and let a later session pick up where this one stopped:

| File | Holds | Changes |
|---|---|---|
| `REQUIREMENTS.md` | what must be true at the end, where each requirement comes from, the **Definition of DONE** | when the task itself changes |
| `ARCHITECTURE.md` | the shape of the solution: components, interfaces, data flow, decisions | rarely, each change logged with its reason |
| `PLAN.md` | the steps to get there, the current step, a log of what was tried | after every step |

Templates for the three are in `templates/`, beside this file (see the `[file: …]` line
above this text). Read them when you create the folder.

## When to use it, and when not

Use it when at least one holds: the change spans several files or modules, it adds a
component or a public interface, the requirements need working out, or the work will not
fit in one sitting. For a script, a snippet, a one-file fix or a question about code, do not
create the folder: do the work directly, and still apply **Working in the code** and
**Verifying** below.

## Phase 1: the task folder

1. Find the project root (the repository's top, or the folder the request names).
2. Look in `<root>/.plans/` for a folder about this same task. If there is one, **resume**:
   read its `PLAN.md` first, then `REQUIREMENTS.md` and `ARCHITECTURE.md`, and continue from
   the plan's current step. Do not start a second folder for the same task.
3. Otherwise create `<root>/.plans/<YYYY-MM-DD>-<slug>/`, where the slug is three to six
   words saying what the task is (`2026-11-17-oauth-token-refresh`, not `task1` or `feature`).
4. Keep the folder out of the user's commits: in a git repository, add `.plans/` to
   `.git/info/exclude` (local to this clone, unlike `.gitignore`) unless it is already ignored.
   Never name a file in it `AGENTS.md` or `SOUL.md`.
5. Tell the user, in one line, where the folder is.

Everything the task needs to remember goes in these files, not only in the conversation.

## Phase 2: orient, briefly

Find what decides the task before deciding anything. Keep this short: the aim is to know
enough to write the requirements, not to read the whole codebase.

- **How the project is tested and built**: the test command (CI config, `Makefile`,
  `pyproject.toml`, `package.json`, `go.mod`, `Cargo.toml`, the README). Run the suite once
  now and record the result as the **baseline**, failures included: a test failing before
  your change is not yours to fix, and one failing after it is.
- **The code the task touches** and its tests. Existing tests are the most reliable
  statement of what the code is meant to do.
- **The conventions**: naming, layout, error handling, how similar features were done.

Make independent searches and reads in the same response, so they run together.

## Phase 3: REQUIREMENTS.md

Copy `templates/REQUIREMENTS.md` and fill it in. One page is usually enough.

- **Every requirement names its source**: the user's words, a test, a doc, a config file.
  A requirement with no source is a guess; find the source or move it to open questions.
- **Every requirement names its check**: the test or command that shows it is met.
- **Exact names are copied verbatim** from the request: paths, function and command names,
  flags, ports, formats, output wording.
- **Open questions are resolved from the repository**, not assumed: search for the test, doc
  or config that settles it, and move the question to *Resolved* with that evidence. If the
  user is present and a question is a real choice only they can make, ask once, all such
  questions together, after writing the file. If the request says you work alone, or nobody
  will answer, do not ask: settle it from the repository, or pick the reading most
  consistent with the existing code and record it under *Assumptions*, with why.

### The Definition of DONE

The last section of `REQUIREMENTS.md` is a checklist of conditions that, all together,
mean the task is finished. Write it before writing any code. Each item must be something
you can **check by running or reading something**, not a feeling ("the code is clean" is
not an item; "`ruff check src/` reports nothing" is). It always contains, besides the
task's own conditions:

- every requirement's check passes;
- the project's **whole** test suite has been run after the last edit, and nothing fails
  that did not fail in the baseline;
- the project's lint, format and type checks pass on the changed code, where the project
  has them;
- the full diff has been read once, and holds nothing unrelated to the task (no debug
  output, no stray files, no drive-by changes);
- `PLAN.md` is up to date and every step in it is done.

Change the Definition of DONE only when the task itself changes, and say so in the log.
Never weaken it to fit what was built.

## Phase 4: ARCHITECTURE.md

Copy `templates/ARCHITECTURE.md`. Describe the shape of the solution, not its code: the
components and what each is responsible for, the interfaces between them, how data flows,
and which existing parts are reused or changed. Follow the patterns the codebase already
uses; a new pattern needs a reason.

Change this file only when the shape has to change: a component added or split, an
interface changed, an approach abandoned. Every change adds an entry to its *Decisions* log
saying what changed and why. Editing a detail of the implementation does not belong here.

For a task that only changes existing code without changing its structure, a few lines
saying so are enough.

## Phase 5: PLAN.md

Copy `templates/PLAN.md`. Break the architecture into steps small enough to verify one at a
time, in an order where each step leaves the code working. For each step: what it does,
the files it touches, and the command that verifies it.

Keep it current; it is the file a resumed session reads first:

- exactly **one** step is in progress at a time, named on the *Current step* line;
- a step is marked done only after its verification ran and passed, never before;
- update the file after every step, whenever the approach changes, and before finishing;
- the *Log* records what was tried and what failed, with why, so the same thing is not
  tried twice.

## Phase 6: implement, step by step

For each step of the plan:

1. Read every file before editing it, unless you created or edited it in this session.
2. Make the change: the smallest one that does the step, in the style of the code around
   it. Fix causes rather than symptoms. Make all the edits one file needs together.
3. Run the step's verification, starting with the narrowest test and widening as it
   passes. When it fails, read the failure and fix the cause; do not change a test to
   match your code unless the requirements say the test is wrong.
4. Update `PLAN.md`: mark the step, move *Current step*, add to the log.

If a step shows the architecture was wrong, stop, update `ARCHITECTURE.md` (with its
decision entry) and the plan, then continue.

## Working in the code

These apply to every coding task, with or without the task folder:

- **Do not create branches, commit, push or rewrite git history** unless the user asked.
  The user reviews and commits.
- **Stay in scope.** Do not fix unrelated bugs, reformat untouched code, rename things the
  task does not need renamed, or upgrade dependencies, unless asked. Note such findings in
  the final message instead.
- **No new dependency** without a reason recorded in `ARCHITECTURE.md`.
- **Do not add license headers**, and comment only what the code cannot say itself.
- **Treat files you did not create as the user's work**: read before overwriting, and do
  not delete them unless the task requires it.
- **Check every command's result.** A non-zero exit code is a failure to investigate before
  moving on, not noise.

## Verifying

A claim that something works is worth only the command that showed it. Before finishing:

1. Run the project's **whole** test suite, not only the tests you wrote or the file you
   changed, and compare with the baseline.
2. Run the lint, format and type checks the project uses.
3. Run what you built the way it will be used (the command, the endpoint, the import),
   when the tests do not already do so.
4. Read the whole diff (`git diff`, plus any new files) as a reviewer would.

When a check fails, fix it and run the checks again from the start: a fix can break
something already verified.

## Finishing: the DONE gate

**Do not finish until every item of the Definition of DONE is met.** Before your final
message, go through the checklist in `REQUIREMENTS.md` item by item, and for each one write
next to it the evidence: the command you ran and what it reported, or what you read. An item
without evidence is not met. If any item is not met, do not write a final message: go back
to the plan, add the step that meets it, and continue.

Finishing with an item unmet is allowed only when it is **blocked by something outside your
control** that you have tried to get around (a service that cannot be reached, a tool that
cannot be installed, a decision only the user can make). Then mark the item `BLOCKED` in the
checklist with the reason and what you tried, and open the final message with it. "Ran out
of ideas" and "probably fine" are not blocked.

The final message, short and in this order:

- what was done, in a few lines;
- the evidence: the verification commands and their results, the whole suite's included;
- anything `BLOCKED`, any assumption still standing, and anything noticed but left out of
  scope;
- where the task folder is.

Size it to the change: a few sentences for a small one, a short list for a large one.
