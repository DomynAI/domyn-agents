---
name: coding
version: 1.2.0
description: >
  A skill for coding tasks. Read it before starting one: how to orient, plan in a task
  folder (REQUIREMENTS.md, PLAN.md, and ARCHITECTURE.md when needed), work in verified
  steps, and decide when the task is done.
tags: [coding, software-engineering, debugging, testing, planning]
external_files:
  - templates/
---

# Coding tasks

A method for coding tasks: the task is written down before it is built,
built in verified steps, and finished only when a written **Definition of DONE** is met and
shown to be met. A task folder keeps the work on track and can be picked up by a later session:

| File | Holds | Changes | When |
|---|---|---|---|
| `REQUIREMENTS.md` | what must be true at the end, where each requirement comes from, the **Definition of DONE** | when the task itself changes | always |
| `PLAN.md` | the steps to get there, the current step, a log of what was tried | after every step | always |
| `ARCHITECTURE.md` | the shape of the solution: components, interfaces, data flow, decisions | rarely, each change logged with its reason | only when the solution has a shape worth writing down (Phase 4) |

Templates for the three are in `templates/`, beside this file (see the `[file: …]` line
above this text). Read them when you create the folder.

The files scale with the task: for a one-file fix, `REQUIREMENTS.md` is a few lines and
`PLAN.md` two or three steps. They are never skipped: writing down what done means and
the steps to get there is what keeps a small task from being finished half way.

## Phase 1: the task folder

1. Find the project root (the repository's top, the folder the request names, or the
   working directory when there is no project yet).
2. Look in `<root>/.plans/` for a folder about this same task. If there is one, **resume**:
   read its `PLAN.md` first, then `REQUIREMENTS.md` and `ARCHITECTURE.md` if there is one,
   and continue from the plan's current step. Do not start a second folder for the same task.
3. Otherwise create `<root>/.plans/<YYYY-MM-DD>-<slug>/`, where the slug is three to six
   words saying what the task is (`2026-11-17-oauth-token-refresh`, not `task1` or `feature`).
4. Keep the folder out of the user's commits: in a git repository, add `.plans/` to
   `.git/info/exclude` (local to this clone, unlike `.gitignore`) unless it is already ignored.
   Never name a file in it `AGENTS.md` or `SOUL.md`, and never put the task's deliverables
   in it: they go where the request says.
5. Tell the user, in one line, where the folder is.

Everything the task needs to remember goes in these files, not only in the conversation.

## Phase 2: orient, briefly

Find what decides the task before deciding anything. Keep this short: the aim is to know
enough to write the requirements, not to read the whole codebase.

- **The request, read twice.** Note every condition the result must meet, stated or implied
  (where an output file goes, what a command must print, which tests must pass), and copy
  exact names verbatim: paths, file names, function and command names, flags, ports,
  formats, output wording.
- **How the project is built and tested**: the test command (CI config, `Makefile`,
  `pyproject.toml`, `package.json`, `go.mod`, `Cargo.toml`, the README). When there is a
  test suite, run it once now and record the result as the **baseline**, failures included:
  a test failing before your change is not yours to fix, and one failing after it is.
- **The code the task touches** and its tests. Existing tests are the most reliable
  statement of what the code is meant to do.
- **The conventions**: naming, layout, error handling, how similar features were done.
- **The environment**, when it matters: the language and tool versions installed, what is
  already running.

Make independent searches and reads in the same response, so they run together.

## Phase 3: REQUIREMENTS.md

Copy `templates/REQUIREMENTS.md` and fill it in. Leave out the sections that would be empty;
one page is usually enough, a few lines for a small task.

- **Every requirement names its source**: the user's words, a test, a doc, a config file.
  A requirement with no source is a guess; find the source or move it to open questions.
- **Every requirement names its check**: the test or command that shows it is met.
- **Exact names are copied verbatim** from the request.
- **Open questions are settled from the repository**, not assumed: search for the test, doc
  or config that answers them, and move the question to *Resolved* with that evidence. If
  the user is present and a question is a real choice only they can make, ask once, all
  such questions together, after writing the file. If you work alone, or nobody will
  answer, do not ask and do not stop to wait: pick the reading most consistent with the
  request and the existing code, record it under *Assumptions* with why, and carry on.

### The Definition of DONE

The last section of `REQUIREMENTS.md` is a checklist of conditions that, all together,
mean the task is finished. Write it before writing any code. Each item must be something
you can **check by running or reading something**, not a feeling ("the code is clean" is
not an item; "`ruff check src/` reports nothing" is). It always contains, besides the
task's own conditions:

- every requirement's check passes;
- what was built has been run the way it will be used, and its output matches the request;
- the project's **whole** test suite has been run after the last edit, and nothing fails
  that did not fail in the baseline (where the project has one);
- the project's lint, format and type checks pass on the changed code, where the project
  has them;
- the full diff has been read once, and holds nothing unrelated to the task (no debug
  output, no stray files, no drive-by changes);
- `PLAN.md` is up to date and every step in it is done.

Change the Definition of DONE only when the task itself changes, and say so in the plan's
log. Never weaken it to fit what was built.

## Phase 4: ARCHITECTURE.md, when needed

Write it when the solution has a shape worth recording: a new component, module or
service; a new or changed public interface (functions others call, an endpoint, a command,
a file or message format); data moving between several parts; or a choice between
approaches that a later session would need the reason for. **Skip it** when the code is
small or simple enough that the plan says it all: a fix inside existing code, a change to
one or two files that leaves the structure as it is, a single script. When you skip it,
say so in one line of `PLAN.md` (`Architecture: none needed, <why>`).

When you write it, copy `templates/ARCHITECTURE.md`. Describe the shape of the solution,
not its code: the components and what each is responsible for, the interfaces between
them, how data flows, and which existing parts are reused or changed. Follow the patterns
the codebase already uses; a new pattern, or a new dependency, needs a reason here.

Change it only when the shape has to change: a component added or split, an interface
changed, an approach abandoned. Every change adds an entry to its *Decisions* log saying
what changed and why. If a task that skipped it turns out to need one, write it then.

## Phase 5: PLAN.md

Copy `templates/PLAN.md`. Break the work into steps small enough to verify one at a time,
in an order where each step leaves the code working. For each step: what it does, the
files it touches, and the command that verifies it. The last step is always the final
verification.

Keep it current; it is the file a resumed session reads first:

- exactly **one** step is in progress at a time, named on the *Current step* line;
- a step is marked done only after its verification ran and passed, never before;
- update the file after every step, whenever the approach changes, and before finishing;
- the *Log* records what was tried and what failed, with why, so the same thing is not
  tried twice.

## Phase 6: implement, step by step

For each step of the plan:

1. Make the change, following *Working in the code*.
2. Run the step's verification, starting with the narrowest check and widening as it
   passes. When it fails, read the failure and fix the cause.
3. Update `PLAN.md`: mark the step, move *Current step*, add to the log.

If a step shows the approach was wrong, stop, update the plan (and `ARCHITECTURE.md` with
its decision entry, or write one if the shape now needs it), then continue.

## Working in the code

- **The result is what is on disk and in the environment**, not what your message says:
  files written where the request says, with the names it gives; programs built; tools
  installed; services started and left running if the task needs them. Describing a change
  is not making it.
- **Read every file before editing it**, unless you created or edited it in this session.
  Make all the edits one file needs together.
- **Make the smallest change that does the step**, in the style of the code around it.
  Follow the patterns the codebase already uses. Fix causes rather than symptoms.
- **Do not change a test to match your code** unless the requirements say the test is wrong.
- **Check every command's result.** A non-zero exit code or an error in the output is a
  failure to investigate before moving on, not noise. Read the whole error message.
- **When something fails twice the same way, stop and change approach**: re-read the error,
  check your assumptions (versions, paths, working directory, environment), and log what you
  tried in `PLAN.md` so you do not repeat it.
- **Do not create branches, commit, push or rewrite git history** unless the user asked.
- **Stay in scope.** Do not fix unrelated bugs, reformat untouched code, rename things the
  task does not need renamed, or upgrade dependencies, unless asked. Note such findings
  under *Out of scope* in `REQUIREMENTS.md` and in the final message instead.
- **Do not add license headers**, and comment only what the code cannot say itself.
- **Treat files you did not create as the user's work**: read before overwriting, and do
  not delete them unless the task requires it.

## Verifying

A claim that something works is worth only the command that showed it. The plan's last
step:

1. Run the project's **whole** test suite, not only the tests you wrote or the file you
   changed, and compare with the baseline.
2. Run the lint, format and type checks the project uses.
3. Run what you built the way it will be used (the command, the script, the endpoint, the
   import), with the inputs the request names, and check the output against the request
   word for word, when the tests do not already do so.
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
