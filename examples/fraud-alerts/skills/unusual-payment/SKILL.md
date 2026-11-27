---
name: unusual-payment
description: How to judge a large transfer that was held for being unusual. Read it for alerts of type unusual_payment.
---

The fraud engine holds a transfer that is much larger than what the customer usually
pays. Most of these are legitimate: a business grows, a bill is larger than usual.

## Legitimate

- The payee is one the customer has paid regularly, for months.
- The amount is larger, but in line with the customer's average monthly spend.
- No sign of an account takeover around it: no password reset, new device or new payee.

## Suspicious

- A payee added in the last days.
- Any sign of an account takeover: read the account-takeover skill.

## What to do

When the payment is legitimate, release it with `release_payment`, in full, with a note
saying why. An analyst confirms every release and may lower the amount. When it is
suspicious, leave it held and explain why: do not close the alert.
