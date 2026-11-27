# KYC onboarding

This example shows several agents working on one task at the same time. A company asks a
bank to open an account; before saying yes, the bank must know who it is dealing with
(KYC, "know your customer"). A coordinator gives two checks to two specialists, who work
in parallel, then writes the KYC memo. A rule of the bank, written as code, decides which
decisions the agents may not take, and an analyst confirms the memo in the terminal.

It shows five things:

- sub-agents called in one response and working at the same time, with the caller waiting
  for both;
- sanctions screening against list files on this machine: a fictional list by default, the
  real EU and OFAC lists with one setting, and no client name ever sent anywhere;
- a hook that rules on an action from the bank's own data, whatever the agents concluded;
- the built-in file tools, limited to the case folders;
- one skill per agent, read when its task starts.

## How it works

```
                      domyn run config.yaml
                               │
                       ┌───────▼────────┐
                       │ KYCCoordinator │
                       └───┬────────┬───┘
              at the same  │        │  time
            ┌──────────────▼─┐    ┌─▼───────────────┐
            │ScreeningAnalyst│    │DocumentsAnalyst │
            │ lookup_company │    │ lookup_company  │
            │ screen_name ───┼─┐  │ read_file ──────┼──► cases/<company>/request.md
            └────────────────┘ │  └─────────────────┘
                               ▼
                lists/ (sanctions list + internal watchlist)
                               │
                       KYCCoordinator
                       save_memo ──► policy.py ──► the analyst confirms ──► memos/
```

1. You ask the coordinator to review a company, naming its case folder.
2. It calls both specialists in one response, and they work at the same time:
   - **ScreeningAnalyst** looks the company up in the business registry and screens every
     party behind it (the company, its directors, its shareholders down the chain) with
     `screen_name`. The tool finds candidates by name; the agent decides, from birth
     dates and countries, whether each is the same party or a namesake.
   - **DocumentsAnalyst** finds the beneficial owners (more than 25%, through every
     holding) from the registry and checks the application in `cases/` against them:
     missing, expired or inconsistent documents.
3. With both answers, the coordinator decides (approve, approve with conditions, reject,
   escalate), rates the risk and writes the memo.
4. Before the memo is saved, `policy.py` screens every party again, itself:
   - approving a company with a party on the internal watchlist, or a close match nothing
     tells apart (same birth date, or an entity with the same name), is **refused**;
   - approving with possible matches still open **asks you**, naming the ones to clear;
   - otherwise you confirm the memo as usual.

## The sanctions lists

| `SANCTIONS_LISTS` | What is screened |
|---|---|
| `fictional` (default) | `lists/fictional_eu_list.csv`: made-up entries in the exact format of the EU consolidated list. Offline, and the same result every time |
| `official` | The real [EU consolidated list](https://data.europa.eu/data/datasets/consolidated-list-of-persons-groups-and-entities-subject-to-eu-financial-sanctions) and the US [OFAC SDN list](https://ofac.treasury.gov/sanctions-list-service), downloaded from their official sources on first use and kept for a day in your temporary folder |

The bank's internal watchlist, `lists/internal_watchlist.csv`, is fictional and screened in
both modes. Internet is used only to download the list files: the names screened never
leave your machine.

The four cases are built on the fictional list. With the official lists they will most
likely come out clean, since every name in them is made up: the setting is there to screen
names of your own.

## Files

- `config.yaml`: the three agents, their tools, skills and the hook.
- `screening.py`: reading the lists and finding candidates (`screen_name`).
- `registry.py`: a made-up business registry (`lookup_company`).
- `memo.py`: saving the memo (`save_memo`), the one action.
- `policy.py`: the bank's rule on decisions.
- `skills/`: screening, beneficial ownership, the memo.
- `cases/`: four account opening requests.
- `lists/`: the fictional sanctions list and the internal watchlist.

## Run it

Copy `.env.example` to `.env` in this folder and fill in the key (see
[Setup](../README.md#setup)), then:

```bash
cd examples/kyc-onboarding
uv run domyn run config.yaml
```

The memos are written to `memos/`.

## Try it

One case at a time:

- *Review the onboarding of Lanterna Meccanica Srl, case folder lanterna-meccanica.* Nobody
  on any list, documents complete: approved, you confirm.
- *Review Velarte Design Srl, case folder velarte-design.* The managing director has the
  name of a listed person, born 27 years earlier and of another country: the screening
  calls it a namesake, and approving asks you to clear the match first.
- *Review Nordavia Logistics Srl, case folder nordavia-logistics.* 70% belongs to a holding
  on the bank's watchlist, owned by a listed group and run by a listed person, and the
  application declares only the 30% owner. Try to have it approved: the rule refuses.
- *Review Studio Calvari Associati Srl, case folder studio-calvari.* Nobody on any list, but
  no beneficial ownership declaration, one director without an identity document and the
  other's expired: approved with conditions at most.

Then set `SANCTIONS_LISTS=official` in `.env` and ask for a review of a company of your own
making, or screen a name you know is listed, to see the real lists at work.

## Using your own data

The cases, the registry and the internal watchlist are made up. To try a case of your own,
add its company to `COMPANIES` in `registry.py`, put a folder with a `request.md` in
`cases/`, and set `SANCTIONS_LISTS=official`.

The screening runs on this machine, but the model reads the whole case. With real customer
data, use a model your organisation allows to receive it: not the free hosted one this
example is configured with. Changing `llm_providers` in `config.yaml` is enough.

This is an example, not a compliance tool: its name matching is simple (no transliteration
or phonetic matching), and it screens the EU and OFAC lists only, with no other national
lists, politically exposed persons or adverse media.

## Learn more

- [Orchestration](../../docs/docs/guides/orchestration.md)
- [Approvals](../../docs/docs/guides/approvals.md)
- [Hooks](../../docs/docs/guides/hooks.md)
- [Skills](../../docs/docs/guides/skills.md)
- [Configuration](../../docs/docs/guides/configuration.md)
