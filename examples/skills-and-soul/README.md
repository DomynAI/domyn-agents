# Skills and SOUL.md

This example shows three things:

- how to give an agent skills, which are files of know-how that the agent reads only when a
  task needs them;
- how to give it a `SOUL.md`, which holds standing instructions that set the tone of every
  reply;
- how to load Python tools from a YAML configuration.

The configuration defines a single agent, the support agent of an online coffee shop. It
answers questions about orders and payments, and it decides refunds.

## Features

- **Skills.** In `config.yaml`, the agent's `skills` point at the `skills/` folder, where
  each skill is a `SKILL.md` file. The agent's prompt lists every skill by name and
  description. When the agent needs a skill, it opens it with the `skill_view` tool.
  - `order-lookup` is an advisory skill. It explains how to read an order's status back to a
    customer.
  - `refund-policy` is a required skill (`enforcement: required`). It explains when a refund
    is due, and how to word a yes or a no. If a required skill cannot be used, the turn is
    refused rather than run without it.
- **SOUL.md.** The agent's `standing_instructions` point at `SOUL.md`, which is added to the
  system prompt on every message. It asks for short, plain replies that state bad news
  first.
- **Python tools in YAML.** The tools are Python functions in `tools.py`, and `config.yaml`
  loads each one by its `path` and name. `refund_payment` asks for your approval before it
  runs. It also receives `refund_time` from the configuration (`config_params`), and the
  model never sees that value.

## Files

- `config.yaml`: the configuration of the system, made of the model, the tools and the
  agent, together with the agent's skills and its `SOUL.md`.
- `tools.py`: the tools. Their data is made up, and stays the same for the same name or id.
- `skills/`: the two skills.
- `SOUL.md`: the agent's tone.

## Run it

Copy `.env.example` to `.env` in this folder and fill it in (see [Setup](../README.md#setup)).

```bash
cd examples/skills-and-soul
uv run domyn run config.yaml
```

## Try it

- "I'm Anna Green, my last order arrived damaged: the lid is cracked. I want a refund": the
  agent reads `refund-policy` and refunds the order once you approve.
- In a new session, "I'm Jane Smith, my parcel never arrived and I want my money back": her
  payment is still pending, so the policy says no.

## Learn more

- [Skills](../../docs/docs/guides/skills.md)
- [Standing instructions](../../docs/docs/guides/standing-instructions.md)
