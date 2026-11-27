---
name: beneficial-ownership
description: How to find a company's beneficial owners from the registry, check the application's documents against them, and list what is missing or inconsistent. Read it before reviewing a case.
---

# Beneficial ownership and documents

## Finding the beneficial owners
A beneficial owner is a natural person who owns or controls more than 25% of the company,
directly or through other companies.
- Start from the registry (`lookup_company`), never from the application.
- When a shareholder is a company, look it up and multiply the shares down the chain:
  70% of a holding that is 100% owned by X gives X 70%.
- When the chain ends in a company the registry does not know, or abroad with no owners
  shown, the beneficial owner is **unknown**: say so, do not guess.
- If no person reaches 25%, the directors are treated as the beneficial owners.

## Checking the documents
Read the case's `request.md` in the case folder. For each item, say whether it is fine:
- the declared beneficial owners match the ones you found in the registry;
- there is a signed beneficial ownership declaration;
- there is an identity document for every beneficial owner and every director, valid on
  the request date written at the top of `request.md`;
- the declared activity and expected turnover fit the registry's activity;
- countries of counterparties: name any high-risk ones the application mentions.

## What to report
The beneficial owners with their share and how it was computed, then a list of
**missing documents** and **inconsistencies**, each in one line. A beneficial owner left
out of the declaration is an inconsistency, and a serious one.
