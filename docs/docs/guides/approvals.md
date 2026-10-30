---
title: Tool approvals
sidebar_label: Tool approvals
description: "Require a person's decision before a tool runs, remember it, and decide some calls automatically"
---

# Tool approvals

Some tools should not run just because a model decided to call them: a payment, a deletion, a message sent on someone's behalf. A tool marked with an `approval_type` stops the turn before it runs and waits for a person to approve or refuse the call. An approval can be remembered for the rest of the conversation or longer, and rules and hooks can decide some calls without asking.

## Requiring approval

Set `approval_type` on the tool:

```python
from domyn.agents import ApprovalType
from domyn.agents import FunctionTool

delete_tool = FunctionTool(func=delete_file, approval_type=ApprovalType.USER_CONFIRM)
```

```yaml
tools:
  - name: transfer
    type: function
    path: tools.py
    approval_type: params_update
```

| `approval_type` | The person can |
|---|---|
| `USER_CONFIRM` (`user_confirm`) | Approve or refuse the call as the model made it. |
| `PARAMS_UPDATE` (`params_update`) | Also change the arguments before approving. |

The `@tool` decorator takes `approval_type` too:

```python
from domyn.agents import ApprovalType
from domyn.agents.core.decorators import tool


@tool(approval_type=ApprovalType.USER_CONFIRM)
def delete_file(path: str) -> str:
    """Delete a file."""
    ...
```

It does not take `approval_subject`, `approval_pattern` or `approval_parts` (described below): a tool that needs those is built with `FunctionTool`. A tool without one never waits for approval: rules and hooks decide only for tools that ask for a decision.

## Answering a request

When the model calls such a tool, the turn yields `tool_start`, then `tool_needs_feedback`, and ends. To answer, run the next turn with a `TOOL_FEEDBACK` event that carries the action of the `tool_needs_feedback` event:

```python
from domyn.agents import BaseEvent, ExecutionEventType

feedback = BaseEvent(
    event_type=ExecutionEventType.TOOL_FEEDBACK,
    author="user",
    action=pending.action.model_copy(update={"approve": True}),
)

async for event in runner.run(
    application_name="app",
    user_id="user-42",
    root=agent,
    user_input=feedback,
    conversation_id=conversation_id,
):
    ...
```

`pending` is the `tool_needs_feedback` event. The turn continues from the tool call:

- **Approved**: the tool runs, and the stream yields its `tool_end`.
- **Refused** (`approve=False`): the tool does not run. `tool_end` carries a refusal as the result, which the model reads, and the agent continues without the tool.

The `tool_needs_feedback` event carries what to show the person:

| Where | What |
|---|---|
| `action.name` | The tool. |
| `action.parameters` | The arguments, as a list of `name`, `value`, `type`, `is_required`. |
| `metadata["approval_subject"]` | The line that describes the call, and that a remembered approval is matched on. See [What an approval covers](#what-an-approval-covers). |
| `metadata["approval_scopes"]` | The scopes the approval can be remembered for. See [Remembering an approval](#remembering-an-approval). |
| `metadata["approval_pattern"]` | A pattern the tool proposes, to approve similar calls too, or `None`. |
| `metadata["approval_parts"]` | The parts of the call that a pattern must each cover, or `None`. See [Patterns](#patterns). |

### Changing the arguments

For a `PARAMS_UPDATE` tool, send back the parameters with new values:

```python
parameters = [
    p.model_copy(update={"value": 50.0}) if p.name == "amount" else p
    for p in pending.action.parameters
]
feedback = BaseEvent(
    event_type=ExecutionEventType.TOOL_FEEDBACK,
    author="user",
    action=pending.action.model_copy(update={"approve": True, "parameters": parameters}),
)
```

The tool runs with the new values, and `tool_end` shows them. The edited call is checked again by the [deny rules](#deny-rules) and [approval hooks](#deciding-in-code): an edit cannot turn a call into one that is denied. For a `USER_CONFIRM` tool, the parameters sent back are ignored, and the call runs as the model made it.

### Several calls in one answer

When the model asks for several tools at once, a call that needs approval stops the turn after the calls before it have run. The `TOOL_NEEDS_FEEDBACK` is about that call; the calls after it are held. Your answer runs the approved (or refused) call, then the held ones, one of which may need an answer in its turn, and only then asks the model again: it sees a result for every call it made.

If a new message arrives instead of an answer, the waiting call and the held ones do not run. Each is given a result saying so, `The tool … was not executed: the user sent a new message instead of answering…`, so the model knows they did not happen and can ask again.

## Remembering an approval

Asking for every call trains people to approve without reading. An approval can instead be remembered, by setting `remember_scope` in the feedback:

```python
action=pending.action.model_copy(update={"approve": True, "remember_scope": "conversation"})
```

Later calls that the approval covers then run without asking. Each scope is kept in a **store**:

| Scope | Lasts | Store |
|---|---|---|
| `once` | This call only. Nothing is stored. | |
| `conversation` | The rest of this conversation. | `ConversationApprovalStore`, in the conversation's state |
| `user` | Every conversation of this user. | `InMemoryApprovalStore` (this process) or `FileApprovalStore` (a YAML file) |

A runner uses only a `ConversationApprovalStore` unless it is given an `ApprovalPolicy` with more:

```python
from pathlib import Path

from domyn.agents.approvals import (
    ApprovalPolicy,
    ConversationApprovalStore,
    FileApprovalStore,
)
from domyn.agents import Runner

runner = Runner(
    approval_policy=ApprovalPolicy(
        stores=[
            ConversationApprovalStore(),
            FileApprovalStore(Path.home() / ".myapp" / "approvals.yaml"),
        ],
    ),
)
```

`metadata["approval_scopes"]` lists the scopes that have a store, so an interface offers only those. A `remember_scope` with no store behind it applies to the one call.

Approvals are kept per user and per agent: one user's approval never covers another user's call, and approving a call for the main agent does not approve it for a team member.

| Store | Arguments | Notes |
|---|---|---|
| `ConversationApprovalStore()` | | Scope `conversation`. Kept with the conversation, so it follows it wherever the conversation is stored. |
| `InMemoryApprovalStore()` | | Scope `user`. Lost when the process ends. |
| `FileApprovalStore(path, scope=ApprovalScope.USER)` | `path`: `str` or `Path`, used as given (`~` is not expanded); `scope`: `ApprovalScope`, default `USER` | A YAML file, written atomically, readable and editable by hand. One file can hold the grants of many users. |

### Your own store

To keep approvals somewhere else, such as your application's database, subclass `ApprovalStore`. Its methods are `async`, so a store can wait on a database or a network call. This one keeps them in a SQLite table:

```python
import asyncio
import sqlite3

from domyn.agents.approvals import (
    ApprovalGrant,
    ApprovalRequest,
    ApprovalScope,
    ApprovalStore,
    is_already_approved,
)


class SqliteApprovalStore(ApprovalStore):
    """Approvals that last for a user, kept in a SQLite table."""

    scope = ApprovalScope.USER

    def __init__(self, path: str) -> None:
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS grants (user_id TEXT, grant TEXT)")

    def _rows(self, user_id: str | None) -> list[ApprovalGrant]:
        rows = self._db.execute("SELECT grant FROM grants WHERE user_id IS ?", (user_id,))
        return [ApprovalGrant.model_validate_json(row[0]) for row in rows]

    async def granted(self, context=None) -> list[ApprovalGrant]:
        user_id = getattr(context, "user_id", None)
        return await asyncio.to_thread(self._rows, user_id)

    async def is_granted(self, request: ApprovalRequest, context=None) -> bool:
        grants = await asyncio.to_thread(self._rows, request.user_id)
        return is_already_approved(request, grants)

    async def store(self, grant: ApprovalGrant, request: ApprovalRequest, context=None) -> None:
        def insert() -> None:
            self._db.execute(
                "INSERT INTO grants VALUES (?, ?)", (request.user_id, grant.model_dump_json())
            )
            self._db.commit()

        await asyncio.to_thread(insert)

    async def revoke(self, grant: ApprovalGrant, context=None) -> bool:
        def delete() -> bool:
            cursor = self._db.execute(
                "DELETE FROM grants WHERE grant = ?", (grant.model_dump_json(),)
            )
            self._db.commit()
            return cursor.rowcount > 0

        return await asyncio.to_thread(delete)

    async def revoke_all(self, context=None) -> None:
        def clear() -> None:
            self._db.execute("DELETE FROM grants")
            self._db.commit()

        await asyncio.to_thread(clear)
```

`scope` is the scope the store keeps. The runner asks `is_granted` before asking the person, and calls `store` when the person chooses to remember an approval. A store that serves many users filters by `request.user_id`, as this one does. Implement `revoke` and `revoke_all` too: a remembered approval must be possible to take back, and the base class only raises `NotImplementedError`. Pass the store to the `ApprovalPolicy` like the built-in ones.

## What an approval covers

An approval is remembered for the call's **subject**, the line in `metadata["approval_subject"]`. By default the subject is the tool's name followed by every argument the model sent, so a remembered approval covers only an identical call. The `reasoning` argument that the `tool_use_thought` planner adds to each call is not part of it.

To make an approval cover later calls, set `approval_subject`. It says which arguments carry the risk, so an approval covers every call that agrees on those:

```python
FunctionTool(
    func=transfer,
    approval_type=ApprovalType.PARAMS_UPDATE,
    approval_subject=["account"],
)
```

Here the subject is `transfer account=IT01`: approving a transfer to `IT01` for the conversation approves later transfers to that account, of any amount. A function receives the arguments and returns the subject, for a value derived from them:

```python
approval_subject=lambda path, **_: f"write {path}"
```

### Patterns

To approve a family of calls, such as every file in a directory, a tool can propose a pattern with `approval_pattern`. It is shown in `metadata["approval_pattern"]`, and the person can accept it, or a narrower one, by sending it as `remember_pattern` together with `remember_scope`:

```python
import posixpath

write_tool = FunctionTool(
    func=write_file,
    approval_type=ApprovalType.USER_CONFIRM,
    approval_subject=lambda path, **_: f"write {path}",
    approval_pattern=lambda path, **_: f"write {posixpath.dirname(path)}/*",
)
```

```python
action=pending.action.model_copy(update={
    "approve": True,
    "remember_scope": "conversation",
    "remember_pattern": pending.metadata["approval_pattern"],   # "write src/*"
})
```

After that, writes to `src/b.py` run without asking, and a write to `docs/c.md` still asks. A pattern is a shell-style wildcard (`*`, `?`, `[…]`) matched against the subject. A proposal that does not match the call it was made for is not offered.

A call that does several things, such as a batch of statements, is covered by patterns only if each of its parts is. `approval_parts` splits the call into parts:

```python
FunctionTool(
    func=run_sql,
    approval_type=ApprovalType.USER_CONFIRM,
    approval_subject=["query"],
    approval_parts=lambda query, **_: [s.strip() for s in query.split(";") if s.strip()],
)
```

With `SELECT *` approved, `SELECT 2; SELECT 3` runs without asking, and `SELECT 4; DROP TABLE t` asks.

## Deny rules

An `ApprovalPolicy` can refuse calls without asking anyone. `deny_rules` are patterns matched against the subject and every part; a call that matches one is refused, even if an approval covers it:

```python
ApprovalPolicy(
    stores=[ConversationApprovalStore()],
    deny_rules=["transfer account=XX*"],
)
```

A refused call does not run. Its `tool_end` carries a result telling the model that an automatic rule blocked it, not the user, and `metadata["denied_by"]` names the rule, as `deny:<rule>` (`deny:transfer account=XX*` here).

Before the deny rules, a short built-in list refuses a few catastrophic shell commands in any subject: deleting a root or home directory recursively, formatting a disk, writing to a device with `dd`, piping a download into a shell, and a fork bomb. It applies even to approved calls, and its refusals carry `hardline:<pattern>` in `metadata["denied_by"]`. It is a guard against accidents, not a security boundary. `ApprovalPolicy(enforce_hardline=False)` turns it off.

| `ApprovalPolicy` argument | Type | Default | Description |
|---|---|---|---|
| `stores` | `list[ApprovalStore]` | `[]` | Where approvals are remembered, one store per scope. The runner's default policy has a `ConversationApprovalStore`. |
| `deny_rules` | `list[str]` | `[]` | Patterns of calls to refuse. |
| `enforce_hardline` | `bool` | `True` | Refuse the built-in list of catastrophic commands. |

## Deciding in code

A `PRE_TOOL_APPROVAL` hook decides a call from its arguments: it returns `ApprovalVerdict.allow(...)` to run it without asking, `ApprovalVerdict.deny(...)` to refuse it, or `None` to leave the decision to the person.

```python
from domyn.agents.approvals import ApprovalVerdict
from domyn.agents.core.decorators import hook
from domyn.agents.hooks import HookPhase


@hook(trigger=HookPhase.PRE_TOOL_APPROVAL)
def scratch_files(tool, tool_args):
    if tool_args["path"].startswith("/tmp/"):
        return ApprovalVerdict.allow(reason="scratch files")
    if tool_args["path"].startswith("/etc/"):
        return ApprovalVerdict.deny(reason="system configuration is off limits")
    return None


write_tool = FunctionTool(
    func=write_file,
    approval_type=ApprovalType.USER_CONFIRM,
    hooks=[scratch_files],
)
```

The verdict can also be returned as `{"approval": verdict}`. A hook that raises has no opinion, and the person is asked. A call a hook denies carries `"policy"` in `metadata["denied_by"]`.

In a [YAML configuration](configuration.md), the hook is declared under `hooks` with `trigger: pre_tool_approval` and attached by name in the tool's `hooks` list, like any other tool hook.

The order of the checks is fixed: the built-in list, then deny rules, then remembered approvals, then hooks (the first that allows or denies decides), and otherwise the person is asked. A deny at any step wins over an allow at a later one. See [Hooks](hooks.md) for hooks in general.

## In the interactive CLI

The [interactive CLI](../cli/tools-and-approvals.md) asks before every file write and every command, offers to remember the approval for the conversation, the project or always, and proposes patterns for commands. `/approvals` lists and revokes remembered approvals, and `/auto` approves everything for the rest of the session.

## Next steps

- [Events and streaming](events-and-streaming.md): the rest of the event stream around an approval.
- [Event types](../reference/events.md): `tool_needs_feedback` and `tool_feedback` field by field.
- [Tools](tools.md): writing the tools that need approval.
