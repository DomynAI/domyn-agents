---
name: kyc-memo
description: How to turn the screening and documents reviews into a KYC memo with a risk rating and a decision, and save it. Read it before calling save_memo.
---

# The KYC memo

## Decision
| Situation | Decision | Risk |
|---|---|---|
| A match on a sanctions list or the internal watchlist, for any party | **reject** | high |
| A beneficial owner the registry cannot show (a chain ending abroad, owners withheld), or one left out of the declaration | **escalate** | high |
| Possible matches unresolved | **escalate** | medium |
| All candidates cleared as false positives, documents complete | **approve** | low, or medium if a false positive needed clearing |
| No match, beneficial owners clear from the registry, but documents missing or expired (the ownership declaration included) | **approve_with_conditions**: list the documents to obtain before the account is used | medium |
| No match, documents complete, no inconsistency | **approve** | low |

High-risk countries among the expected counterparties raise the risk one level.

## The memo
Markdown, in this order:
1. **Summary**: the decision and the reason, in two sentences.
2. **Parties screened**: one line each, with the result and the reason.
3. **Beneficial owners**: who, with what share, how it was computed.
4. **Documents**: complete, or what is missing and inconsistent.
5. **Conditions or next steps**, when there are any.

Name the lists screened, and say plainly when they were the fictional sample.

## Saving it
Call `save_memo` with the company's name as in the registry, the decision, the risk rating
and the memo. The bank's rules check the decision first: if they refuse it, say why and
save the decision they allow instead (reject or escalate), with the memo updated.
