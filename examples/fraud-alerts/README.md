# Fraud alerts

This example shows an agent that works with nobody typing to it. Alerts from a bank's fraud
engine arrive on a Kafka topic; for each one, a fraud analyst agent looks at the customer
and their transactions, decides, and acts. The bank's rules decide which actions the agent
may take alone and which wait for a person, who answers from a console in the browser.

The agent needs no code of its own to run: `domyn serve` serves `config.yaml` on Kafka.

It shows five things:

- how to serve an agent on Kafka, many alerts at a time, with only configuration;
- how a case that waits for a person is parked, and resumed when the answer arrives;
- how to write the bank's rules as hooks that allow, deny or ask before an action runs;
- how to give an agent existing REST services as tools, through MCP;
- skills, one per fraud pattern, and an audit log of every action.

## How it works

```
console ──fraud.alerts─────▶ domyn serve ──▶ FraudAnalyst ──MCP──▶ services.py
                                │   ▲                │
                     fraud.events   fraud.decisions  └── policy.py: allow, deny or ask
                                ▼   │
                             console.py (the analyst, on localhost:8080)
```

1. From the console you send the five sample alerts, or raise your own: each goes to
   `fraud.alerts`, keyed by its id.
2. `domyn serve` reads them with the Kafka channel and starts one conversation per alert,
   named by the key, three at a time.
3. The agent reads the skill for the alert's type, then calls the services: the customer's
   profile, recent transactions, spending metrics, a held payment.
4. It takes one action: block the card, close the alert as a false alarm, or release a
   held payment.
5. Before the action runs, `policy.py` rules on it:

   | Action | Rule |
   |---|---|
   | `block_card` | allowed |
   | `close_alert` | allowed when the alert's risk score is under the bank's threshold (50, changed from the console), else ask |
   | `release_payment` | always ask; the analyst can change the amount; more than the payment holds is refused |

6. When the rule asks, the turn ends with the waiting call, written to `fraud.events`. The
   case is parked; the other alerts carry on.
7. `console.py` follows the three topics and shows every case in the browser. A case that
   waits shows the action and the agent's reason; the analyst approves, denies or edits
   it. The answer goes to `fraud.decisions` under the case's key, and the agent resumes
   the case where it stopped.

## Features

- **The Kafka channel.** `serve.channels` in `config.yaml` names the topics to read and the
  one to write. A message's key is the conversation; its value is any text (here, an
  alert's JSON) or an event as JSON (here, the analyst's answer). An offset is committed
  only once its case has ended or parked, so an alert cut short by a crash is read again.
- **Pause and resume.** An action that needs approval ends the turn with a
  `TOOL_NEEDS_FEEDBACK` event. The answer is a `TOOL_FEEDBACK` event, and the same
  conversation continues with it. `domyn serve` keeps conversations in memory: a case
  parked when the agent restarts is lost, and its answer finds nothing to resume.
- **Policy hooks: the bank's rules are code, not instructions.** The skills tell the model
  how to investigate, and it may follow them or not. `fraud_policy` in `policy.py` is a
  `pre_tool_approval` hook: it runs before each action, answers allow, deny or ask, and the
  model cannot argue with it. It reads the risk score and the payment from the services,
  never from the agent's arguments. Each verdict is recorded with its rule and reason, and
  the console shows it beside the action ("Bank rule allows block card…"), so what the
  model chose and what the bank allowed are told apart. **Bank rules** in the console
  changes the threshold, and the next verdicts follow it.
- **Approval types.** `block_card` and `close_alert` are `user_confirm`: yes or no.
  `release_payment` is `params_update`: the analyst can change the amount, and the changed
  call goes through the policy again.
- **Services as tools.** `services.py` is an ordinary FastAPI app. `FastMCP.from_fastapi`
  serves its routes as MCP tools, and `config.yaml` lists the ones the agent gets.
- **Skills.** One per alert type in `skills/`; the agent reads the one it needs.
- **New alerts from the console.** "New alert" raises one more, from a scenario (card
  testing, impossible travel, account takeover, a large transfer), for the customer and
  risk score you choose. The services write what the agent will find before the alert, so
  the two agree; a card alert cannot be raised on a card already blocked. These routes,
  under `/engine`, are not given to the agent: it works alerts, it does not raise them.
- **When something fails.** A case that ends in an error (the free models are often rate
  limited) shows "Send the alert again": the agent takes it up in the same conversation,
  with what it had found.
- **A console with no database.** `console.py` reads the three topics from their start
  and builds each case's story from them, so it can be restarted at any time.
- **Audit log.** `audit` in `policy.py` is a listener hook that writes every action that
  ran to `domyn_fraud_audit.jsonl`, in the temporary folder of the machine the agent runs on.

## Files

- `config.yaml`: the agent, its tools, its hooks, and how `domyn serve` serves it.
- `services.py`: the bank's services, with made-up data, as REST and as MCP tools, and the
  fraud engine's scenarios for new alerts.
- `policy.py`: the approval rules and the audit log.
- `skills/`: card testing, account takeover, impossible travel, unusual payment.
- `alerts.py`: the five sample alerts, published from the console or from a terminal.
- `console.py`, `console.html`: the analyst's console, a FastAPI app and its page.
- `docker-compose.yaml`, `Dockerfile`: the whole example, in containers.

## Run it

Copy `.env.example` to `.env` in this folder and fill in the key (see
[Setup](../README.md#setup)). You need Docker.

```bash
cd examples/fraud-alerts
docker compose up --build
```

Then open http://localhost:8080 and send the five sample alerts, or raise your own. This
starts Kafka, the services, the agent and the console; the image is built from this
checkout, so the first build takes a minute.

The terminal shows what happens, one line for each thing the agent does:

```
console-1  | A-1005  new alert: impossible_travel · risk 70 · Card used in Tokyo, then in Milan three hours later.
console-1  | A-1005  -> get_customer(customer_id=C-205)
console-1  | A-1005  -> close_alert(alert_id=A-1005, note=False alarm: customer has travel notice for Tokyo…)
console-1  | A-1005  !! close_alert waits for an analyst
console-1  | A-1005  Analyst denied close_alert (alert_id=A-1005, note=…)
console-1  | A-1005  case note: The customer has an active travel notice for Tokyo…
```

with the agent's warnings and errors among them. Everything keeps running, waiting for
more alerts, until Ctrl-C. `docker compose up -d` starts it without the log instead, and
`docker compose logs -f` follows it then.

### Without Docker for the agent

To change the code and run it again quickly, keep only Kafka in Docker and run the rest
here, each in its own terminal, from this folder:

```bash
docker compose up -d kafka
uv run python services.py
uv run --with aiokafka domyn serve config.yaml --channel multi
uv run --with aiokafka python console.py         # then open http://localhost:8080
```

The sample alerts can also be sent from a terminal: `uv run --with aiokafka python alerts.py`,
or only some, `alerts.py A-1001 A-1004`.

`.env` tells the agent where Kafka and the services are; Docker Compose sets its own.

## Try it

Send the five sample alerts, and watch the console while the agent works them:

- **A-1001**, card testing: the agent blocks the card on its own.
- **A-1002**, impossible travel with a risk score of 35: the Singapore charge is an airline
  bought online, so the agent closes the alert on its own.
- **A-1003**, account takeover: the agent blocks the card and leaves the transfer held.
- **A-1004**, a large transfer to a regular supplier: the agent wants to release it, and the
  console asks you. Lower the amount and approve: the agent goes on with your amount.
  Type an amount larger than 12,000 instead, and the policy refuses it.
- **A-1005**, impossible travel with a risk score of 70: the customer has a travel notice for
  Tokyo, so the agent wants to close the alert, and the console asks you.

Then raise your own with **New alert**. Try the same scenario with two risk scores: an
airline ticket bought abroad at 35, which the agent closes on its own, and at 70, which
asks you. Or keep it at 70 and raise the threshold in **Bank rules** to 80: the same alert
is now closed without you, and the timeline says which rule allowed it. Or block a customer's card, and see that card alerts can no longer be raised
for them.

`docker compose down` stops everything and starts the next run over: Kafka's topics and
the services' data live in the containers. Running without Docker, restart `services.py`
and the agent instead.

## Learn more

- [Approvals](../../docs/docs/guides/approvals.md)
- [Hooks](../../docs/docs/guides/hooks.md)
- [Events and streaming](../../docs/docs/guides/events-and-streaming.md)
- [Conversations and memory](../../docs/docs/guides/conversations-and-memory.md)
- [Skills](../../docs/docs/guides/skills.md)
- [The Kafka channel](../../domyn/agents/channels/kafka.py)
