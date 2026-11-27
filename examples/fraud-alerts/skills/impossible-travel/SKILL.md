---
name: impossible-travel
description: How to judge card use in two places too far apart in too little time. Read it for alerts of type impossible_travel.
---

The fraud engine raises this alert when the same card is used in two places that nobody
could travel between in the time between the two uses. Most of these alerts are false
alarms.

## Common false alarms

- **Online purchases.** An online merchant is registered where its company is, not where
  the customer is. An airline, a hotel chain or a marketplace abroad says nothing about
  where the card was. Check the `channel` of each transaction.
- **A merchant the customer already uses**, in the same country, in the last months.
- **A travel notice** on the customer's profile that covers the place and the dates. Then
  a chip payment there is expected, and an online payment from home is too: bills and
  subscriptions keep running.

## Real fraud

Two **chip** (card present) payments in places too far apart, with no travel notice: the
card was copied. Block it.

## What to do

When one of the false alarms explains the alert, close it with a note naming which one.
