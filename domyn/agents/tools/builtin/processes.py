"""Commands that outlive the call that started them.

A benchmark, a build, a training run: minutes to hours, and it talks while it
works. Until now the call waited for it and, at the timeout, *killed* it — so
the choice was between throwing away two minutes of work and blocking the
conversation for two hours with nothing to show.

**The call gets a deadline, the process does not**: run to the deadline, hand
back a handle, let it carry on. That is codex's `yield_time_ms`. Where we differ
is how the process is kept: codex holds a handle in memory, which works because
codex owns the process. We often do not — a backend may reach a remote executor
through a one-shot endpoint, where there is nowhere to hold anything.

So the state lives in the filesystem of whoever runs the command, and every
operation is an ordinary shell command:

    start     a detached process whose output is redirected to files
    watch     read those files
    stop      signal the recorded process group

That is what makes it portable — identical locally, against a remote executor
and over ssh, with **no new backend method and nothing added to its API**. It
also means `grep` and `read_file` already reach the logs.

**The redirection is load-bearing, not tidiness.** A backend using
`Popen(..., stdout=PIPE)` and `communicate()` returns when the *pipes close*,
not when the child exits, so a background child that inherits them blocks the
call until it dies — the trap `_terminate_tree` documents in ``backends.py``.
With the output in files the child holds nothing: measured, the call returns in
~8 ms with the process still running.

**No PTY**, deliberately: it would have to reach inside whatever runs the
command, and without a terminal most tools stop drawing progress bars anyway.
The cost is that programs reading `/dev/tty` — `sudo`, `ssh` asking for a
password — cannot be driven, which is the right thing to lose.
"""

# No `from __future__ import annotations`: this module is imported by shell.py,
# whose tools are built from real annotations at construction time.

import re
import secrets
import shlex
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from domyn.agents.tools.builtin.session import safe_id

# How long the *call* waits before handing back a handle. Not how long the
# process may live.
DEFAULT_YIELD_S = 120.0

# The backstop. Kill-on-timeout was the only thing stopping a runaway command,
# and yielding gives it up; this buys it back. Enforced by a watchdog inside the
# process group rather than by us, because "us" may be a CLI that has since been
# closed. Long by default: the point of the feature is work
# that takes hours, and a service that pays for its own CPU should lower it.
DEFAULT_MAX_LIFETIME_S = 24 * 60 * 60

# Between SIGTERM and SIGKILL. Codex allows 50ms, which is enough to close a file
# and not enough to write a checkpoint; a training run deserves better.
DEFAULT_GRACE_S = 5

# Per conversation. Codex allows 64, but that is one person on their own machine.
MAX_PROCESSES = 8

# The longest any single call may block — the initial yield and an explicit
# `check_command(wait=...)` alike, since from outside they are the same thing.
# Bounded by what the transport tolerates, not by what would be useful: a remote
# backend makes this a request, and one held open for several minutes is one
# something in between is entitled to drop.
MAX_BLOCK_S = 300

# Kept as the old name for the wait path, so the two uses read differently at the
# call site even though the bound is shared.
MAX_WAIT_S = MAX_BLOCK_S

# The total any one conversation may spend *deliberately waiting*. The per-call
# bound alone does not stop a model asking for another five minutes over and
# over; this does, and it states the intent: waiting is for minutes, and an
# hours-long run should be reported and looked at again later.
#
# Only explicit waits are charged, and only for time actually spent. The initial
# yield of `run_command` is capped instead of charged: a conversation out of
# budget would otherwise get a handle back from `ls` rather than its output.
#
# Half an hour, not the ten minutes first guessed — one five-minute run watched
# twice used that up, and from outside it looks like a model asking to wait and
# being answered instantly for no reason.
WAIT_BUDGET_S = 1800
WAIT_BUDGET_KEY = "background_wait_spent"

# What one process may leave on disk, per stream — about 10 MB a process. Small
# on purpose: what is worth keeping is the beginning (what it was doing) and the
# end (how it went), never the middle of hour three.
#
# The head is captured **once** and then never touched: it cannot change, so
# there is nothing to race with. Only the tail is rewritten.
LOG_HEAD_BYTES = 1_000_000
LOG_COMPACT_AT = 4_000_000
LOG_TAIL_BYTES = 2_000_000
LOG_CHECK_EVERY_S = 30

# How much of a log the *shell* sends back. Python clips again, harder; this
# only keeps a megabyte of build log off the wire.
HEAD_BYTES = 20_000
TAIL_BYTES = 20_000

HOME = ".domyn/processes"
"""Relative to the working directory the backend already uses: the workspace
root locally, the backend's own working directory elsewhere. Processes are grouped
below it, so one conversation cannot read another's logs through the file
tools, and so everything a conversation left behind can be found — and one
day removed — as a unit."""


class ProcessRecord(BaseModel):
    """What survives a turn, and therefore what has to be JSON.

    Kept in ``InteractionContext.state``, which the runner serialises to a
    ``TEXT`` column after every event: no paths, no datetimes, no live objects,
    and nothing that grows. In particular **no output text** — it would be
    re-serialised on every event of every turn. The output is in the files.

    Liveness is deliberately absent: it is discovered by probing, which is what
    lets a resumed conversation pick up a process that outlived the session.
    """

    handle: str
    command: str
    conversation: str = ""
    finished: bool = False
    """Set once observed finished, so the cap can count live ones without
    paying a probe per record just to ask."""
    started_at: int = 0
    """Epoch seconds, filled in by the launch itself."""
    cursor: int = 0
    """Bytes of stdout already shown to the model."""
    err_cursor: int = 0

    @property
    def dir(self) -> str:
        return f"{HOME}/{safe_id(self.conversation)}/{self.handle}"


class ProcessState(BaseModel):
    """One observation of a process. Never stored; produced by every probe."""

    handle: str = ""
    running: bool = True
    exit_code: int | None = None
    killed: bool = False
    """Whether *we* stopped it, read from the file the killer writes — never
    inferred from the exit code.

    The exit code is not trustworthy for this. Two wrappers were measured
    reporting **``exited 0``** after being killed: a plain ``echo $? >`` and one
    with a ``TERM`` trap. What a shell reports about being signalled depends on
    what it happened to be doing at the time — sometimes 143, sometimes 0 — and a
    model reading 0 would tell the user their benchmark finished successfully.
    Since the kill goes through us, we already know; there is nothing to infer.
    """
    killed_by_lifetime: bool = False
    """Stopped by the safety limit rather than by anyone's decision. Worth
    distinguishing: it means the work was not finished, and that the limit — not
    the command — is what needs changing."""
    elapsed_s: int = 0
    idle_s: int = 0
    """Since the last byte of output. This is the number that makes "it has been
    stuck for ten minutes" expressible; without it, silence and progress look
    alike."""
    out_bytes: int = 0
    err_bytes: int = 0
    out: str = ""
    err: str = ""
    omitted_bytes: int = 0
    log_path: str = ""

    model_config = {"extra": "ignore"}


def new_record(command: str, conversation: str = "") -> ProcessRecord:
    """A record for a command that has not been started yet."""
    return ProcessRecord(handle=secrets.token_hex(4), command=command, conversation=conversation)


# --------------------------------------------------------------------------- #
# Building shell
# --------------------------------------------------------------------------- #


# Every operation reports through this, so a launch that finishes in time and a
# poll ten minutes later parse identically. The handle is the nonce: a command
# would have to print this exact line to confuse the parser.
def _markers(handle: str) -> dict[str, str]:
    return {
        key: f"###DOMYN:{handle}:{key}###"
        for key in ("META", "OUT_HEAD", "OUT_TAIL", "ERR_HEAD", "ERR_TAIL", "END")
    }


def _stat_mtime(path: str) -> str:
    """Modification time in epoch seconds, on both GNU and BSD.

    `stat` takes different flags on Linux and macOS and there is no portable
    third option, so both are tried. `date -r` is not the escape hatch it looks
    like: GNU reads it as a file, BSD as a number of seconds.
    """
    return f"(stat -c %Y {path} 2>/dev/null || stat -f %m {path} 2>/dev/null || echo 0)"


def _bytes_of(path: str) -> str:
    # `wc -c` pads with spaces on BSD.
    return f'(wc -c < {path} 2>/dev/null | tr -d " " || echo 0)'


def _report(record: ProcessRecord, *, from_cursor: bool) -> str:
    """The shell that emits one observation.

    Output is sent in two chunks — a head and a tail — rather than whole: the
    head is where a run says what it is doing (versions, device, sample counts)
    and the tail is where it says how it is going. With the byte count the caller
    can tell whether they overlap and stitch them back together exactly.

    When following a running process the head is worthless — what matters is what
    happened since last time — so `from_cursor` sends only the new bytes,
    themselves clipped to the tail.
    """
    m = _markers(record.handle)
    d = shlex.quote(record.dir)
    out, err = f"{d}/out.log", f"{d}/err.log"
    exit_file = f"{d}/run.exit"

    # **Every** read is bounded by the size measured once, above, and reported
    # to the caller — the head as much as the tail. The file grows between
    # counting it and reading it, because a running command writes the whole
    # time, so an unbounded read returns more than was counted: the caller then
    # advances its cursor to the smaller number and the next poll repeats what
    # it has already shown. Bounding only the tail left the head doing exactly
    # that, and it took a CI run on a small file to show it.
    if from_cursor:
        # `tail -c +N` is 1-based: +1 is the whole file.
        head_cmd_out = "true"
        tail_cmd_out = (
            f"tail -c +$((CURSOR + 1)) {out} 2>/dev/null "
            f"| head -c $((OUTB - CURSOR)) | tail -c {TAIL_BYTES}"
        )
        head_cmd_err = "true"
        tail_cmd_err = (
            f"tail -c +$((ERRCURSOR + 1)) {err} 2>/dev/null "
            f"| head -c $((ERRB - ERRCURSOR)) | tail -c {TAIL_BYTES}"
        )
    else:
        # The head comes from the file the compactor set aside, when there is
        # one: the live log no longer starts where the run did. That file never
        # changes once written, so it needs no bound; the live log does — see
        # the note above about every read being bounded by the measured size.
        head_cmd_out = (
            f"if [ -s {out}.head ]; then head -c {HEAD_BYTES} {out}.head; "
            f"else head -c $OUTB {out} 2>/dev/null | head -c {HEAD_BYTES}; fi"
        )
        tail_cmd_out = f"head -c $OUTB {out} 2>/dev/null | tail -c {TAIL_BYTES}"
        head_cmd_err = (
            f"if [ -s {err}.head ]; then head -c {HEAD_BYTES} {err}.head; "
            f"else head -c $ERRB {err} 2>/dev/null | head -c {HEAD_BYTES}; fi"
        )
        tail_cmd_err = f"head -c $ERRB {err} 2>/dev/null | tail -c {TAIL_BYTES}"

    return (
        f'echo "{m["META"]}"; '
        # Liveness comes from the OS, not from a file. A process killed by the
        # watchdog never gets to write its exit code — it is killed *before*
        # that line — so a report that trusted the file alone would call it
        # "still running" forever.
        f"PID=$(cat {d}/run.pid 2>/dev/null); "
        f'if kill -0 "$PID" 2>/dev/null; then echo "alive=1"; else echo "alive=0"; fi; '
        f'if [ -f {exit_file} ]; then echo "exit_code=$(cat {exit_file})"; fi; '
        f'echo "killed=$([ -f {d}/run.killed ] && echo 1 || echo 0)"; '
        f'echo "reason=$(cat {d}/run.killed 2>/dev/null || echo none)"; '
        f'echo "now=$(date +%s)"; '
        f'echo "started_at=$(cat {d}/run.started 2>/dev/null || echo 0)"; '
        f'echo "out_mtime=${_stat_mtime(out)}"; '
        f'echo "err_mtime=${_stat_mtime(err)}"; '
        f"OUTB=${_bytes_of(out)}; ERRB=${_bytes_of(err)}; "
        # A compaction rewrites the log shorter, and a cursor is an offset
        # into it — so afterwards the caller points past the end and would
        # read nothing at all, silently, for the rest of the run.
        '[ "${CURSOR:-0}" -gt "$OUTB" ] && CURSOR=0; '
        '[ "${ERRCURSOR:-0}" -gt "$ERRB" ] && ERRCURSOR=0; '
        f'echo "out_bytes=$OUTB"; '
        f'echo "err_bytes=$ERRB"; '
        f'echo "out_cut=$(cat {out}.cut 2>/dev/null || echo 0)"; '
        f'echo "err_cut=$(cat {err}.cut 2>/dev/null || echo 0)"; '
        f'echo "{m["OUT_HEAD"]}"; {head_cmd_out}; echo; '
        f'echo "{m["OUT_TAIL"]}"; {tail_cmd_out}; echo; '
        f'echo "{m["ERR_HEAD"]}"; {head_cmd_err}; echo; '
        f'echo "{m["ERR_TAIL"]}"; {tail_cmd_err}; echo; '
        f'echo "{m["END"]}"'
    )


# The detached half, kept as a script rather than concatenated into the command
# line that starts it. Three reasons, and the third is the one that matters:
# quoting happens once instead of at every join; the source stays readable enough
# to review; and **it lands on disk**, so when something behaves oddly the thing
# that ran can be opened and read instead of reconstructed from memory.
_WRAPPER = """#!/usr/bin/env bash
# Written by domyn-agents. Runs detached; nothing waits for it.
echo $$ > "{dir}/run.pid"
ps -o pgid= -p $$ | tr -d ' ' > "{dir}/run.pgid"

# The backstop for the guarantee detaching gives up, inside the group it
# polices: when it fires, the session that started it may be long closed.
#
# `trap '' TERM` is not caution — without it the watchdog dies to its own first
# signal and never escalates. Nor is the SIGKILL politeness: a bash loop that
# receives TERM while waiting on a child simply carries on. Both were measured.
# Waits in steps and stops on its own when the command is done, rather than
# sleeping through the whole limit. A single long sleep cannot be cleaned up:
# killing the subshell orphans the sleep, which is then reparented and no longer
# findable by its parent — so every command ever run left a day-long sleep
# behind. Killing the sleep first is worse: it *wakes* the watchdog into firing.
#
# Done is the exit file *or* the wrapper gone: the wrapper outlives the command
# it runs, so once it has left there is nothing to police. The file alone was
# not enough. It lives in a directory that can be removed under the helpers —
# pytest removes a test's three sessions later, and HOME is meant to go as a
# unit — and then it never appears: measured, both helpers outlived their
# command by a day, the compactor spinning all the while (see compact_once).
# The wrapper's absence misleads in one case only: a wrapper killed on its own
# leaves its command running, now with no limit — a command check_command,
# stop_command and /ps have already lost, since all three find it through
# run.pid.
# In a subshell `$$` is the wrapper's pid, not the subshell's.
( trap '' TERM
  waited=0
  while [ "$waited" -lt {max_lifetime} ]; do
    [ -f "{dir}/run.exit" ] && exit 0
    kill -0 $$ 2>/dev/null || exit 0
    sleep {watch_step}
    waited=$(( waited + {watch_step} ))
  done
  [ -f "{dir}/run.exit" ] && exit 0
  echo lifetime > "{dir}/run.killed"
  kill -TERM 0
  sleep {grace}
  kill -KILL 0 ) &

# Keeps each log under its cap with nothing outside the group alive.
#
# The `>>` below is what makes it safe: in append mode every write goes to the
# current end, so the file can be shrunk underneath the writer. With `>` the
# writer holds an offset and a shrunk file leaves a hole.
#
# The head is copied once and left alone. Only the tail is rewritten, and that
# seam is the one race: whatever arrives between reading the tail and replacing
# the file is lost, so a line can be cut in half. Bounded — it happens only
# above the cap — and reported rather than hidden.
#
# One pass at a time. The timer's pass can still be running when the command
# ends and the wrapper starts its last one; two at once share `$f.part` and both
# add to `$f.cut`, so one can write back the other's half-read tail, emptying the
# log, and the count can exceed what the command ever wrote. Seen in CI: a log
# whose tail was gone and 5.1 MB reported dropped from 3.5 MB written. `mkdir`
# is the lock because it is atomic and everywhere; `flock` is not on macOS.
compact_once() {{
  # Waits only while another pass holds the lock. Any other failure — the
  # directory removed, a full disk, a read-only file system — is one retrying
  # cannot cure. A lock released between the failed mkdir and the look is taken
  # at once: giving that pass up could skip the wrapper's last one, and leave
  # what the command wrote at its end above the cap.
  until mkdir "{dir}/compact.lock" 2>/dev/null; do
    if [ ! -d "{dir}/compact.lock" ]; then
      mkdir "{dir}/compact.lock" 2>/dev/null && break
      return 1
    fi
    sleep 0.1
  done
  for f in "{dir}/out.log" "{dir}/err.log"; do
    s=$(wc -c < "$f" 2>/dev/null | tr -d ' ')
    [ "${{s:-0}}" -gt {compact_at} ] || continue
    [ -s "$f.head" ] || head -c {head_bytes} "$f" > "$f.head"
    if tail -c {tail_bytes} "$f" > "$f.part"; then
      prev=$(cat "$f.cut" 2>/dev/null || echo 0)
      cat "$f.part" > "$f"
      rm -f "$f.part"
      echo $(( prev + s - {tail_bytes} )) > "$f.cut"
    fi
  done
  rmdir "{dir}/compact.lock"
}}

# Leaves with the command, on the same two signs as the watchdog.
( while :; do
    sleep {check_every}
    [ -f "{dir}/run.exit" ] && exit 0
    kill -0 $$ 2>/dev/null || exit 0
    compact_once
  done ) &

{stdin_open}env PYTHONUNBUFFERED=1 TERM=dumb COLORTERM= \
    bash "{dir}/cmd.sh" >> "{dir}/out.log" 2>> "{dir}/err.log" {stdin_redirect}
echo $? > "{dir}/run.exit"

# Once more, now that nothing is writing. The loop above leaves when the command
# does, so without this everything written since its last pass would stay — and
# a command that is short and very noisy, finishing well inside one interval,
# would never be compacted at all.
compact_once

"""


# How long the fine-grained phase lasts, and how often it looks. Almost every
# command finishes inside this window, which is the whole reason it exists.
_FINE_S = 2
_FINE_STEP = "0.1"


def _wait_loop(seconds: float, marker: str) -> str:
    """Wait for a file to appear: often at first, once a second afterwards.

    A fixed interval must choose between returning promptly and spawning a
    process five times a second for two minutes. It need not: nearly every
    command ends in the first seconds, and nothing is gained by asking that
    often about a build in its second minute. Two minutes costs ~140 sleeps
    instead of 600, and a 0.3s command still returns in 0.3s.

    Integer arithmetic, no `bc`: it is not everywhere, and missing it would
    degrade into a loop that never waits.

    Not a knob for the model — the decision with a meaning, how long to wait, is
    already `wait`.
    """
    # Bounded by what was asked for, not only by the phase length: a yield of one
    # second must not wait two. Measured — a `sleep 2` asked to yield after 1s
    # ran to completion inside the wait and never produced a handle at all.
    fine_seconds = min(_FINE_S, max(0.0, seconds))
    fine = int(fine_seconds / float(_FINE_STEP))
    coarse = max(0, int(seconds - fine_seconds))
    return (
        f"i=0; while [ $i -lt {fine} ]; do "
        f"[ -f {marker} ] && break; sleep {_FINE_STEP}; i=$((i+1)); done; "
        f"i=0; while [ ! -f {marker} ] && [ $i -lt {coarse} ]; do "
        f"sleep 1; i=$((i+1)); done"
    )


def build_launch(
    record: ProcessRecord,
    *,
    yield_s: float = DEFAULT_YIELD_S,
    max_lifetime: float = DEFAULT_MAX_LIFETIME_S,
    grace: int = DEFAULT_GRACE_S,
    stdin: str | None = None,
) -> str:
    """One shell command that starts the process, waits a while, and reports.

    Deliberately a single call rather than start-then-poll: against a remote
    executor every call is a round trip, and the overwhelming majority of commands finish in
    under a second. Making `ls` pay for two requests to buy something only a
    benchmark needs would be a bad trade taken on every command.

    The model writes none of this. If it forgot the redirection the call would
    block until the process died — the failure being removed here, coming back
    as a hang, which is the worst shape for it to come back in.
    """
    d = shlex.quote(record.dir)
    raw_dir = record.dir
    cmd = shlex.quote(record.command)

    # Opened, then immediately unlinked. A POSIX fd outlives the name, so the
    # command still reads every byte while nothing is left on disk for anyone —
    # this agent's own file tools included — to read afterwards. Without it the
    # answer to a prompt sits in the workspace until somebody deletes it.
    stdin_open = (
        f'exec 3< "{raw_dir}/stdin.in"\nrm -f "{raw_dir}/stdin.in"\n' if stdin is not None else ""
    )
    wrapper = _WRAPPER.format(
        dir=raw_dir,
        max_lifetime=int(max_lifetime),
        grace=grace,
        check_every=LOG_CHECK_EVERY_S,
        # Its own step: the compactor's cadence is about log size, this one
        # about how precisely the lifetime limit lands — and it must never
        # overshoot a limit shorter than itself.
        watch_step=max(1, min(LOG_CHECK_EVERY_S, int(max_lifetime))),
        compact_at=LOG_COMPACT_AT,
        head_bytes=LOG_HEAD_BYTES,
        tail_bytes=LOG_TAIL_BYTES,
        stdin_open=stdin_open,
        stdin_redirect="<&3" if stdin is not None else "< /dev/null",
    )

    # The user's command and the wrapper both go to files instead of being
    # interpolated into the line that starts them: one is arbitrary text from a
    # model, and nesting either inside the other's quoting is how broken quoting
    # and command injection both happen.
    setup = (
        f"mkdir -p {d} && "
        f"printf '%s' {cmd} > {d}/cmd.sh && "
        f"printf '%s' {shlex.quote(wrapper)} > {d}/run.sh && "
        f"date +%s > {d}/run.started && "
        f": > {d}/out.log && : > {d}/err.log"
    )
    if stdin is not None:
        setup += f" && printf '%s' {shlex.quote(stdin)} > {d}/stdin.in"

    # `setsid` makes the wrapper its own session leader, so its group id is its
    # own pid and cannot be recycled while it lives. macOS has none at all — and
    # the first attempt failed *silently* there, the error landing in the log
    # with an exit status of 0 — so its absence is expected, not an error.
    launch = (
        f"SETSID=$(command -v setsid || true); "
        f"nohup $SETSID bash {d}/run.sh > /dev/null 2>&1 < /dev/null &"
    )

    return (
        f"{setup} && {{ {launch} }}; "
        f"{_wait_loop(yield_s, f'{d}/run.exit')}; "
        f"{_report(record, from_cursor=False)}"
    )


def build_probe(record: ProcessRecord, wait_s: int = 0) -> str:
    """One shell command that reports on an already-started process.

    With ``wait_s`` it waits for the process to end before reporting, and
    returns the moment it does. That is one call that costs one turn, where a
    model polling in a loop would spend a turn per look and re-read the whole
    conversation each time — the difference between watching something for two
    minutes and burning forty turns to do it.

    The waiting happens in the shell, on the machine the process is on, which is
    also why it works unchanged against an executor behind a one-shot endpoint.
    """
    d = shlex.quote(record.dir)
    wait = ""
    if wait_s > 0:
        wait = _wait_loop(min(wait_s, MAX_WAIT_S), f"{d}/run.exit") + "; "
    return (
        f"CURSOR={record.cursor}; ERRCURSOR={record.err_cursor}; "
        f"{wait}{_report(record, from_cursor=True)}"
    )


def build_sweep(records: list[ProcessRecord]) -> str:
    """One command that reports every process at once, facts only.

    Written as a batch because the caller is a front-end asking "what is still
    going" before drawing a prompt or starting a turn, and doing that one
    process at a time would be one round trip each against a remote executor. One call
    regardless of how many there are, and no output text: this answers "is it
    still alive", not "what did it say".
    """
    if not records:
        return "true"
    parts = []
    for record in records:
        d = shlex.quote(record.dir)
        parts.append(
            f"P=$(cat {d}/run.pid 2>/dev/null); "
            f'if kill -0 "$P" 2>/dev/null; then A=1; else A=0; fi; '
            f'echo "{record.handle} $A $(cat {d}/run.exit 2>/dev/null || echo -) '
            f"$([ -f {d}/run.killed ] && echo 1 || echo 0) "
            f"$(cat {d}/run.started 2>/dev/null || echo 0) $(date +%s) "
            f'${_stat_mtime(f"{d}/out.log")} ${_bytes_of(f"{d}/out.log")}"'
        )
    return "; ".join(parts)


def build_discover(conversation: str = "") -> str:
    """Ask the machine what this conversation left running, without being told.

    ``build_sweep`` needs the records; this needs nothing. Each process has a
    directory holding its pid, its command and its log, so the filesystem is the
    source of truth and ``state`` a cache of it.

    Not tidiness: a process outlives the session that started it, so anything
    that loses the conversation's state used to orphan it permanently — still
    running, still writing, unreachable. Now it is one listing away.
    """
    d = shlex.quote(f"{HOME}/{safe_id(conversation)}")
    return (
        f"for p in {d}/*/; do "
        f'[ -d "$p" ] || continue; '
        f'h=$(basename "$p"); '
        f'P=$(cat "$p/run.pid" 2>/dev/null); '
        f'if kill -0 "$P" 2>/dev/null; then A=1; else A=0; fi; '
        f'echo "P $h $A $(cat "$p/run.exit" 2>/dev/null || echo -) '
        f'$([ -f "$p/run.killed" ] && echo 1 || echo 0) '
        f'$(cat "$p/run.started" 2>/dev/null || echo 0) $(date +%s) '
        f'$(stat -c %Y "$p/out.log" 2>/dev/null || stat -f %m "$p/out.log" 2>/dev/null || echo 0) '
        f'$(wc -c < "$p/out.log" 2>/dev/null | tr -d " " || echo 0)"; '
        f'echo "C $h $(head -c 200 "$p/cmd.sh" 2>/dev/null | tr "\n" " ")"; '
        f"done"
    )


def parse_discover(output: str) -> dict[str, dict[str, Any]]:
    """Read a listing back: the same facts a sweep gives, plus the command.

    Unparseable lines are skipped. This reads a directory that other versions of
    this code may have written, so a shape we do not recognise means one process
    we cannot describe, never a failure.
    """
    found: dict[str, dict[str, Any]] = {}
    for line in output.splitlines():
        bits = line.split(" ", 2)
        if len(bits) < 2:
            continue
        kind, handle = bits[0], bits[1]
        if kind == "P":
            facts = parse_sweep(f"{handle} {bits[2] if len(bits) > 2 else ''}")
            if handle in facts:
                found.setdefault(handle, {}).update(facts[handle])
        elif (kind == "C" and handle in found) or kind == "C":
            found.setdefault(handle, {})["command"] = (bits[2] if len(bits) > 2 else "").strip()
    return {h: v for h, v in found.items() if "running" in v}


def parse_sweep(output: str) -> dict[str, dict[str, Any]]:
    """Read a sweep back. Unparseable lines are skipped, never raised on."""
    found: dict[str, dict[str, Any]] = {}
    for line in output.splitlines():
        bits = line.split()
        if len(bits) != 8:
            continue
        handle, alive, exit_code, killed, started, now, mtime, out_bytes = bits
        began, clock, touched = _as_int(started) or 0, _as_int(now) or 0, _as_int(mtime) or 0
        found[handle] = {
            "running": alive == "1",
            "exit_code": None if exit_code == "-" else _as_int(exit_code),
            "killed": killed == "1",
            "elapsed_s": max(0, clock - began) if began else 0,
            "idle_s": max(0, clock - touched) if touched else 0,
            "out_bytes": _as_int(out_bytes) or 0,
        }
    return found


def _as_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def block_seconds(asked: float, default: float) -> float:
    """How long a call may block, whatever the model asked for.

    An upper bound has to exist here and not only on the explicit wait: without
    it `run_command(..., timeout=7200)` blocks for two hours and walks straight
    past the waiting budget, which only ever saw `check_command`. A remote backend's
    own ceiling might catch it there; nothing would catch it locally.
    """
    chosen = asked if asked > 0 else default
    return max(1.0, min(chosen, MAX_BLOCK_S))


def wait_allowance(state: dict[str, Any] | None, asked: int) -> tuple[int, int]:
    """How long this call may wait, and what is left afterwards.

    The budget is per conversation and lives in the same persisted state as the
    processes, so it survives the turn that spent it.
    """
    asked = max(0, min(asked, MAX_WAIT_S))
    spent = 0
    if isinstance(state, dict):
        raw = state.get(WAIT_BUDGET_KEY)
        spent = raw if isinstance(raw, int) else 0
    remaining = max(0, WAIT_BUDGET_S - spent)
    return min(asked, remaining), remaining


def spend_wait(state: dict[str, Any] | None, seconds: int) -> None:
    """Charge the budget for time **actually** spent, never for time asked for.

    The allowance is an upper bound, and a wait ends the moment the command does
    — waiting 300s for something that takes two costs two. Charging the
    allowance instead would let a handful of quick checks exhaust the budget, and
    the model would be refused a wait it had never really had.
    """
    if not isinstance(state, dict) or seconds <= 0:
        return
    raw = state.get(WAIT_BUDGET_KEY)
    state[WAIT_BUDGET_KEY] = (raw if isinstance(raw, int) else 0) + seconds


def build_kill(record: ProcessRecord, grace: int = DEFAULT_GRACE_S) -> str:
    """Stop the process group, and record that *we* did.

    Takes a record, never a pid — which is why this needs no approval of its
    own: it can only reach what we started. A signature, not a policy.

    The group, not the process: a benchmark forks workers.

    The recorded pid is checked alive *and still in the recorded group* first.
    Without `setsid` the group leader is a shell that exited long ago, so the
    group id is a pid that could have been reused.

    The outcome is written here, not by the process: both a plain `echo $? >`
    and a version with a TERM trap were measured writing **`exited 0`** after
    being killed, and a model reading that would report success.
    """
    d = shlex.quote(record.dir)
    return (
        f"PID=$(cat {d}/run.pid 2>/dev/null); PGID=$(cat {d}/run.pgid 2>/dev/null); "
        f'if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null && '
        f'[ "$(ps -o pgid= -p $PID | tr -d \' \')" = "$PGID" ]; then '
        f'kill -TERM -"$PGID" 2>/dev/null; sleep {grace}; kill -KILL -"$PGID" 2>/dev/null; '
        f'echo "[stopped by the agent at $(date -u +%Y-%m-%dT%H:%M:%SZ)]" >> {d}/out.log; '
        f"echo killed > {d}/run.killed; echo stopped; "
        f'else echo "not running"; fi'
    )


# --------------------------------------------------------------------------- #
# Reading it back
# --------------------------------------------------------------------------- #

_META_LINE = re.compile(r"^([a-z_]+)=(-?\d+)$")
_REASON_LINE = re.compile(r"^reason=(\w+)$")


@dataclass
class _Chunks:
    out_head: str = ""
    out_tail: str = ""
    err_head: str = ""
    err_tail: str = ""


def _split(text: str, handle: str) -> tuple[dict[str, int], _Chunks]:
    m = _markers(handle)
    if m["META"] not in text:
        return {}, _Chunks()

    body = text.split(m["META"], 1)[1]
    body = body.split(m["END"], 1)[0]

    sections: list[str] = [body]
    for key in ("OUT_HEAD", "OUT_TAIL", "ERR_HEAD", "ERR_TAIL"):
        head, _, rest = sections[-1].partition(m[key])
        sections[-1] = head
        sections.append(rest)

    meta: dict[str, int] = {}
    for line in sections[0].splitlines():
        stripped = line.strip()
        found = _META_LINE.match(stripped)
        if found:
            meta[found.group(1)] = int(found.group(2))
            continue
        reason = _REASON_LINE.match(stripped)
        if reason and reason.group(1) != "none":
            meta["_reason_" + reason.group(1)] = 1

    # Each chunk carries the newline this module's `echo` added after it.
    def clean(chunk: str) -> str:
        return chunk[1:] if chunk.startswith("\n") else chunk

    return meta, _Chunks(
        out_head=clean(sections[1])[:-1] if sections[1] else "",
        out_tail=clean(sections[2])[:-1] if sections[2] else "",
        err_head=clean(sections[3])[:-1] if sections[3] else "",
        err_tail=clean(sections[4])[:-1] if sections[4] else "",
    )


def _stitch(head: str, tail: str, total: int) -> tuple[str, int]:
    """Put a head and a tail back together, exactly when they overlap.

    Three cases, and only the third loses anything: the file fits in the head;
    the file fits in head plus tail, so the overlap can be removed by arithmetic
    and the result is byte-exact; or it does not, and the gap is reported rather
    than hidden.
    """
    if total <= 0:
        return "", 0
    if total <= len(head.encode("utf-8", "replace")) or not tail:
        return head, 0
    head_bytes = len(head.encode("utf-8", "replace"))
    tail_bytes = len(tail.encode("utf-8", "replace"))
    missing = total - head_bytes - tail_bytes
    if missing <= 0:
        # They overlap: drop from the tail what the head already showed.
        overlap = -missing
        return head + tail[overlap:], 0
    # A real gap. Both ends are kept and the hole is named: the head says what
    # the run was doing, and the tail is where a failure is. Returning the head
    # alone — which this did at first — throws away the half that explains the
    # exit code.
    return f"{head}\n\n... [{missing} bytes omitted] ...\n\n{tail}", missing


def parse_report(output: str, record: ProcessRecord, *, from_cursor: bool) -> ProcessState:
    """Turn one shell report into a state. Never raises.

    A malformed report means the observation is lost, which costs a poll. Raising
    would cost the conversation, and the shell involved is running on a machine
    we do not control.
    """
    meta, chunks = _split(output, record.handle)
    if not meta:
        return ProcessState(handle=record.handle, running=True, log_path=f"{record.dir}/out.log")

    now = meta.get("now", 0)
    started = meta.get("started_at", 0)
    out_bytes = meta.get("out_bytes", 0)
    err_bytes = meta.get("err_bytes", 0)
    last_touch = max(meta.get("out_mtime", 0), meta.get("err_mtime", 0))

    cut = meta.get("out_cut", 0)
    if from_cursor:
        out, omitted = chunks.out_tail, max(0, out_bytes - record.cursor - len(chunks.out_tail))
        err = chunks.err_tail
    elif cut:
        # The middle is gone from the disk, not merely unsent, so there is
        # nothing to stitch: the two ends come from two different files now.
        out = (
            f"{chunks.out_head}\n\n"
            f"... [{cut} bytes dropped to keep the log under its size cap] ...\n\n"
            f"{chunks.out_tail}"
        )
        omitted = cut
        err, _ = _stitch(chunks.err_head, chunks.err_tail, err_bytes)
    else:
        out, omitted = _stitch(chunks.out_head, chunks.out_tail, out_bytes)
        err, _ = _stitch(chunks.err_head, chunks.err_tail, err_bytes)

    return ProcessState(
        handle=record.handle,
        # Alive *and* no exit file. The wrapper outlives the command by a
        # moment — it writes the exit code, then compacts the log one last time
        # — and asking only the pid reports a finished command as running for
        # that window. Which half is authoritative is not symmetric: an exit
        # code means it is over, whatever else is still tidying up.
        running=bool(meta.get("alive", 0)) and "exit_code" not in meta,
        exit_code=meta.get("exit_code"),
        killed=bool(meta.get("killed", 0)),
        killed_by_lifetime=bool(meta.get("_reason_lifetime", 0)),
        elapsed_s=max(0, now - started) if started else 0,
        idle_s=max(0, now - last_touch) if last_touch else 0,
        out_bytes=out_bytes,
        err_bytes=err_bytes,
        out=out,
        err=err,
        omitted_bytes=max(0, omitted),
        log_path=f"{record.dir}/out.log",
    )


def registry(state: dict[str, Any] | None) -> dict[str, ProcessRecord]:
    """The conversation's processes, read out of ``state``.

    Follows ``CompactionRecord``: a malformed entry is dropped rather than
    raised on. Losing one costs the ability to poll that process — its files are
    still on disk — while raising would make a resumed conversation unusable.
    """
    if not isinstance(state, dict):
        # Not only "absent". This is read from whatever the conversation store
        # handed back — a different process wrote it, possibly a different
        # version — and from front-ends that keep their own idea of state. A
        # shape we do not recognise means no processes, never an exception.
        return {}
    out: dict[str, ProcessRecord] = {}
    entries = state.get("background_processes")
    if not isinstance(entries, dict):
        return {}
    for handle, raw in entries.items():
        try:
            out[handle] = ProcessRecord.model_validate(raw)
        except Exception:
            continue
    return out


# How many records a conversation keeps, finished ones included. Records are
# small, but `state` is re-serialised after every event of every turn, so an
# unbounded list of them is a cost paid continuously for information nobody will
# read again. Well above the live cap, so nothing still running is ever at risk.
MAX_RECORDS = 32


def refresh(
    state: dict[str, Any] | None, observed: dict[str, dict[str, Any]]
) -> list[ProcessRecord]:
    """Fold a sweep back into the registry and return what is still alive.

    The cap is enforced against the live count, which is only as fresh as the
    last time anybody looked — and nothing looks on its own. So a conversation
    that started several commands and never checked them refused to start
    another one forever. The fix is to check *before refusing*: one call, only
    on the path that was about to fail.
    """
    known = registry(state)
    for handle, record in known.items():
        seen = observed.get(handle)
        if seen is not None and not seen["running"]:
            record.finished = True
        remember(state, record)
    prune(state)
    return [r for r in registry(state).values() if not r.finished]


def prune(state: dict[str, Any] | None) -> None:
    """Drop the oldest finished records once there are too many.

    Insertion order is the age order: dicts keep it, and a JSON round trip keeps
    it too. Only finished records are ever dropped — losing a live one would
    orphan a running process, which is the failure this module exists to avoid.
    """
    if not isinstance(state, dict):
        return
    entries = state.get("background_processes")
    if not isinstance(entries, dict) or len(entries) <= MAX_RECORDS:
        return
    known = registry(state)
    droppable = [h for h, r in known.items() if r.finished]
    for handle in droppable[: len(entries) - MAX_RECORDS]:
        entries.pop(handle, None)


def remember(state: dict[str, Any] | None, record: ProcessRecord) -> None:
    if not isinstance(state, dict):
        return
    state.setdefault("background_processes", {})[record.handle] = record.model_dump()


# --------------------------------------------------------------------------- #
# Turning it into text a model reads
# --------------------------------------------------------------------------- #

# What one observation may cost. Far below the 20 000 that `ToolUseContextManager`
# clips tool results to by default, and deliberately: following a long run must
# not be the thing that fills the window and triggers a compaction. The whole log
# stays one `grep` away.
POLL_CHARS = 4_000


def collapse_repeats(text: str) -> str:
    """Fold runs of the same line into one.

    A retry loop, a warning printed once per item, a progress line that a
    non-terminal turned into one line per update: thousands of identical lines
    that say one thing. The count is kept because "it printed this 3412 times"
    is itself the diagnosis.
    """
    lines = text.splitlines()
    if len(lines) < 3:
        return text
    out: list[str] = []
    previous: str | None = None
    run = 0
    for line in lines:
        if line == previous:
            run += 1
            continue
        if run > 1:
            out.append(f"    ... (x {run + 1})")
        previous, run = line, 0
        out.append(line)
    if run > 1:
        out.append(f"    ... (x {run + 1})")
    return "\n".join(out)


def clip(text: str, limit: int = POLL_CHARS, *, tail_only: bool) -> tuple[str, int]:
    """Cut text to size, from the right end.

    `tail_only` is the difference between the two things this is used for. A
    command that has finished is read for *what happened*: the head has the
    invocation and the setup, the tail has the failure, and both are wanted. A
    command still running is read for *where it is now*, and the start of an
    increment is of no interest at all.
    """
    if len(text) <= limit:
        return text, 0
    if tail_only:
        return text[-limit:], len(text) - limit
    half = limit // 2
    return text[:half] + "\n…\n" + text[-half:], len(text) - 2 * half


def _one_line(command: str, limit: int = 80) -> str:
    flat = " ".join(command.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def tail_line(text: str) -> str:
    """The last line worth showing from a chunk of output.

    For a progress display, which wants one line rather than a screenful. Blank
    lines are skipped: a run that just printed a separator has not stopped
    saying anything, and showing an empty line would read as if it had.
    """
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()[:160]
    return ""


def _duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def _size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def _body(text: str, label: str, *, tail_only: bool, omitted_bytes: int = 0) -> str:
    """One stream, folded, clipped, and honest about what is missing."""
    folded = collapse_repeats(text)
    shown, dropped = clip(folded, tail_only=tail_only)
    if not shown.strip():
        return ""
    note = ""
    total_missing = dropped + omitted_bytes
    if total_missing:
        note = f" ({_size(total_missing)} not shown — grep the log for the rest)"
    return f"{label}{note}:\n{shown.rstrip()}"


def describe(state: ProcessState, *, following: bool, command: str = "") -> str:
    """The text a model reads about one process.

    **Named by what it runs, not by its handle.** A handle is a pid by another
    name, and leading with it taught the model to lead with it too: people were
    shown tables of hex strings to describe "the benchmark". It comes last, and
    labelled, because it is only for calling these tools and for `/stop`.

    Then the facts, which are small and are what a decision is made on: alive or
    not, how long it has been, and — the one that makes "stuck for ten minutes"
    sayable at all — how long since it last said anything.
    """
    name = f"`$ {_one_line(command)}`" if command else f"`{state.handle}`"

    if state.killed_by_lifetime:
        head = (
            f"{name} — stopped by the maximum runtime limit after "
            f"{_duration(state.elapsed_s)}. It had not finished."
        )
    elif state.killed:
        head = f"{name} — stopped, after {_duration(state.elapsed_s)}."
    elif not state.running:
        head = (
            f"{name} — finished with exit code {state.exit_code} "
            f"after {_duration(state.elapsed_s)}."
        )
    else:
        head = (
            f"{name} — still running after {_duration(state.elapsed_s)}, "
            f"last output {_duration(state.idle_s)} ago, "
            f"{_size(state.out_bytes)} written.\n"
            f"It keeps running whatever you do next. Check it again with "
            f'check_command("{state.handle}"), stop it with '
            f'stop_command("{state.handle}").\n'
            f"Full log: {state.log_path}"
        )
    if command:
        head += f"\n(handle `{state.handle}` — for these tools, and for the person's /stop)"

    label = "new output" if following else "output"
    parts = [head]
    body = _body(state.out, label, tail_only=following, omitted_bytes=state.omitted_bytes)
    if body:
        parts.append(body)
    errors = _body(state.err, "errors", tail_only=following)
    if errors:
        parts.append(errors)
    if len(parts) == 1 and state.running:
        parts.append("(nothing new since the last check)")
    return "\n\n".join(parts)
