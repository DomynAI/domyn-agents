---
title: Decision models
sidebar_label: Decision models
description: "Let a small model answer typed questions with probabilities, where a heuristic or a prompt would otherwise decide"
---

# Decision models

Several things an agent does in a turn are not writing problems: whether a tool call is safe enough to run without asking, which skills are worth reading, which old tool output can be dropped first. Each is a fixed rule or a question put to you.

A **decision model** answers typed questions with probabilities instead of text, in roughly 150 milliseconds. It does not generate prose. You give it a state and a set of questions, and it returns a number for each, in a single pass. Two exist: **Jev** (TypeSafe) and **Clef** (Cloudflare, which speaks Jev's API on purpose).

The number is the point. A model that says "allow" tells you nothing about how sure it is; one that says `0.53` tells you to ask a person.

:::info It is off unless you configure it
An agent with no decision model behaves exactly as it does today — nothing is asked, nothing is logged per turn, and no code path exists that only runs when a model is present. Configuring a provider switches nothing on by itself: every use case starts at `off`.
:::

## Setting one up

`domyn setup` asks. Four levels, in order of what they hand over:

| level | what it does |
|---|---|
| `no` | no decision model |
| `shadow` | ask and record, change nothing |
| `active` | pick skills and conditional instructions; the approval gate still only records |
| `active, approvals too` | also let it allow or deny a tool call |

The approval gate is a level of its own on purpose. Everywhere else a wrong answer changes what the model is *shown* and the turn carries on; at the gate, allowing removes a human confirmation.

The wizard asks for the endpoint and the model name, then sends a real request to check both. A failure warns and saves anyway — the endpoint may come up later.

```
  decision model:
     1) no                     — no decision model
     2) shadow                 — ask and record, change nothing
   * 3) active                 — pick skills and instructions; the approval gate only records
     4) active, approvals too  — also let it allow or deny a tool call

  decision endpoint [http://localhost:28080]:
  decision model name [Cloudflare/clef]:
  endpoint: answering
```

### In YAML

The wizard writes this, and you can write more of it by hand:

```yaml
decision_providers:
  - name: local_clef
    type: system_one
    base_url: http://localhost:28080
    model: Cloudflare/clef        # the name the SERVER exposes
    max_state_tokens: 73728

agents:
  - name: Assistant
    decisions:
      provider: local_clef
      approvals: {mode: active, min_confidence: 0.8}
      skills: {mode: active}
      instructions: {mode: active}
      tools: {mode: off}
      context_ladder: {mode: off}
      sensitivity: {mode: active, on_unknown: block}
```

:::warning `model` is the name the server exposes
A server that serves `Cloudflare/clef` rejects `clef` with a 404 naming the one it does serve. The name is configuration and is never derived, because the short form is the one people try first.
:::

`/model` and re-running `domyn setup` keep what you wrote by hand: the wizard replaces only the keys it asked about.

## Where it is allowed to matter

Six places, each with one direction it may move in. A name that is not in this list is refused when the config loads, so a typo cannot quietly switch on something nobody reviewed.

| use case | what it decides | it may only |
|---|---|---|
| `approvals` | whether a gated call still needs you | allow, or deny |
| `skills` | which skills to read up front | add |
| `instructions` | which `##` sections apply | remove, on a confident no |
| `tools` | which tools the turn is shown | hide |
| `context_ladder` | which old outputs compaction drops first | mark for dropping |
| `sensitivity` | how sensitive this turn's data is | refuse the turn |

### Modes

Every use case is `off`, `shadow` or `active`, set separately.

- **`off`** — not asked. No request, no latency.
- **`shadow`** — asked and recorded, and the turn is **unchanged**. This is how you watch a model on your own work before trusting it.
- **`active`** — the answer is used, if it clears the bar.

An `active` use case whose backend cannot produce good enough probabilities stays in `shadow` and says so once.

## Confidence, and where the number came from

Each use case has a `min_confidence` and an answer below it is ignored. The default differs per use case because the cost of being wrong does.

The threshold is only as good as the number it reads, so every answer also carries **where that number came from**:

| provenance | meaning |
|---|---|
| `native` | a real decision model's probability |
| `logprobs` | derived from an ordinary LLM's token probabilities |
| `stated` | the LLM was asked how sure it was, and said so |
| `none` | no answer |

A use case declares the weakest provenance it will act on. Approvals declare **two**: allowing removes a human confirmation and needs a real probability, while denying only adds friction you can override, so it runs on the weaker bar.

:::tip Why this distinction is in the product and not just in the code
Measured on 13 shell commands: the decision model got 12 right, and its one wrong answer carried **0.53**, against 0.83–0.98 for the right ones — so a 0.8 threshold rejects exactly the wrong one. The same test against an ordinary LLM asked to state its confidence also got 12 right, but its wrong answers sat at **0.78–0.85**, above the same threshold. Same accuracy, useless number.
:::

## Using an ordinary LLM instead

With no decision model available, `type: llm` points a use case at a normal provider:

```yaml
decision_providers:
  - name: fallback
    type: llm
    provider: default_provider
    method: logprobs        # or `stated`
```

It asks one question per request and reads the token probabilities. Expect less: on a reasoning model the probabilities describe the *thinking*, not the answer, so the provider reads them backwards and downgrades itself to `stated` rather than report a confidence it cannot see. Use it to try the shape of the thing, not to remove confirmations.

## The approval gate

This is the use case worth understanding in detail, because it is the one that can skip a question you would otherwise be asked.

The model's verdict joins the **hook verdicts**, in the position the approval policy already defines:

1. hardline rules (never negotiable)
2. deny rules
3. a grant you already gave
4. **hooks, and the decision model**
5. ask you

Two consequences follow from that ordering, and both are deliberate:

- A grant returns at step 3, before hooks are read, so **a model can never revoke a permission you gave**.
- The only case the model ever speaks in is the one whose answer would otherwise have been *ask the user*.

A hardline rule still wins outright. `rm -rf /` is refused whatever the model says — in fact, with a decision model configured it is refused at step 1 and the model's opinion is never reached.

## The data boundary

`sensitivity` is the one use case that does not fall back to today's behaviour, because here today's behaviour is "send it anyway".

A provider declares the highest level it may receive:

```yaml
llm_providers:
  - name: a_cloud_model
    type: openai
    model_name: gpt-4o
    max_sensitivity: internal
```

Each turn is rated `public`, `internal`, `confidential` or `regulated`, and one rated above the ceiling is refused before the agent produces anything. A provider that declares no ceiling is never checked, never asked, and costs nothing.

Because a guardrail that fails open is not a guardrail, an unobtainable rating **blocks**. `on_unknown: allow` is available and sits in the file where a reviewer can see it. A fractional rating rounds **up**: a turn between internal and confidential is confidential.

## Conditional instructions

A `##` section in `SOUL.md` or `AGENTS.md` can say when it applies, as the first line under its heading:

```markdown
## SQL migrations
when: the task involves a database migration or a schema change

Always check the rollback path before the forward one.
```

Sections without a `when:` are unconditional. A `when:` further down is prose, not a directive.

The section is dropped only on a **confident no**. Between "yes" and "no" sits *it does not know*, and that case keeps your text — a 0.45 is a coin flip, and deleting a line you wrote on a coin flip is the mistake this rule exists to prevent. What was left out is recorded rather than vanishing, and `/config` shows it.

## Watching it work

A decision that *acted* prints one line in the session:

```
  decision  tools   4 tool(s) hidden this turn   0.96   151ms
```

`/verbose` shows the rest, including decisions that were taken in `shadow` and not applied. `/config` has a **Decision Models** section with each use case's mode and threshold — the threshold is what actually decides, so it is shown rather than hidden.

Two use cases do not print anything: `skills` and `sensitivity`. They still work, and still log at `info`.

## When it cannot answer

A runtime failure never breaks a turn. A timeout, a dead endpoint, a malformed answer and an oversized state all arrive as "I don't know", and the turn proceeds exactly as it would with no model at all. After repeated failures the provider stops asking until the next turn.

A **configuration** mistake is different and raises when the config loads: an unknown provider type, an unknown use case name, a `provider:` nothing declares. A feature you believe you configured and which is silently off is worse than a failed start.

:::caution One timeout matters more than the others
The approval phase has no timeout of its own, so the provider's `timeout_s` is the only thing between a hung endpoint and a hung turn. The default is deliberately short.
:::

## Next steps

- [Tool approvals](approvals.md) — the gate this plugs into, and what it decides on its own
- [Standing instructions](standing-instructions.md) — where conditional `##` sections live
- [Skills](skills.md) — what the `skills` use case is choosing between
- [Context compaction](context-compaction.md) — what `context_ladder` feeds
