---
name: sanctions-screening
description: How to screen the parties of a company onboarding and decide whether each candidate the lists return is the same party or a false positive. Read it before screening.
---

# Sanctions screening

## Who to screen
Every party behind the company, from the business registry, not from the application:
- the company itself;
- every director;
- every shareholder; when a shareholder is a company, look it up in the registry and screen
  its own directors and shareholders too, down the chain.

## Screening
Call `screen_name` once per party, with `kind` ("person" or "entity"), and for a person
the birth date and citizenship from the registry. It returns candidates, never matches:
a candidate only has a name close to a listed one.

## Deciding on a candidate
For each candidate, compare what the list says with what you know of the party:

| What you see | Conclusion |
|---|---|
| On the internal watchlist, same name | **Match**: the bank already decided against this party |
| Person, same name, birth date agrees | **Match** |
| Person, same name, birth date and citizenship differ | **False positive**: a namesake |
| Person, same name, birth date unknown on either side | **Unresolved**: needs an identity document or more data |
| Entity, same name (score 0.95 or more) | **Match**, unless the registry shows a clearly different entity |
| Name only similar (score under 0.95) | Usually a **false positive**: say what tells them apart |

Always say why: "listed person born 1961-07-02, citizen of CY; our director born
1988-03-14, citizen of IT: a namesake". A false positive without a reason is not cleared.

## What to report
One line per party: name, role, result (no candidates / false positive / match /
unresolved) and the reason. Name the lists searched and say when they are the fictional
sample rather than the official lists.
