# Annotation Handbook V1

Status: normative for V1 human annotation of the current PocketFinancer
bank/card extraction contract.

This handbook defines what the human label means. The locked Android profile and
executable contract remain the product source of truth; if they change, version
this handbook instead of silently changing V1 labels.

All examples below are invented for this handbook. They are not copied or adapted
from private messages, regression rows, or model output. `Example Bank`, its
senders, and every named party are fictional.

## 1. Label one posted bank/card transaction

Choose `transaction` only when the SMS explicitly confirms a completed movement
of money on a bank account or card and supplies all three required components:

1. one positive transaction amount;
2. the direction relative to the customer's account or card; and
3. an exact account/card mention in the SMS.

A transaction annotation contains:

- `decision=transaction`;
- the exact normalized decimal in `amount_decimal` and its source span;
- `type=debit` or `type=credit`;
- the exact account/card source span;
- an exact counterparty source span, or explicit counterparty absence;
- optional concise notes; and
- the `uncertain` state.

Choose `not_transaction` for everything else. A `not_transaction` annotation must
leave amount, type, account, and counterparty fields empty. Do not fill plausible
fields on a null label.

An amount, masked account, or transaction word alone is not proof of posting.
Read the whole SMS, including status and negation. Never use sender reputation,
outside knowledge, another SMS, or a model proposal to supply missing evidence.

## 2. Decision rules

Apply these rules in order. A specific exclusion wins over a generic transaction
verb.

### Completed debits and credits

A **debit** is money explicitly posted out of the customer's named account/card:
a purchase, payment, transfer sent, or cash withdrawal. A **credit** is money
explicitly posted into or back to the customer's named account/card: a deposit,
transfer received, or qualifying refund/reversal.

Judge direction from the customer's account/card, not from the bank, merchant,
sender, or grammatical subject.

Invented example:

> INR 840.00 was debited from A/c XX0420 at Orbit Bazaar.

Label `transaction`, `amount_decimal=840.00`, `type=debit`, account `A/c
XX0420`, and counterparty `Orbit Bazaar`.

Invented example:

> INR 125.50 was credited to A/c XX0420 from Cedar Studio.

Label `transaction`, `amount_decimal=125.50`, `type=credit`, account `A/c
XX0420`, and counterparty `Cedar Studio`.

### Refunds and reversals

A refund or reversal is a `credit` only when this SMS explicitly confirms that
money was credited to the named bank account/card. Words such as *initiated*,
*approved*, *requested*, *processed*, *expected*, *will be refunded*, *reversed*,
or *released* do not by themselves confirm a credit.

- “Refund of INR 60.00 initiated for Card **8842” is `not_transaction`.
- “Refund of INR 60.00 credited to Card **8842” is a `credit` transaction.
- “The failed debit will be reversed” is `not_transaction`.
- A released card hold is `not_transaction` unless a separate explicit credit is
  stated.

### Pending, failed, declined, blocked, and cancelled events

These are `not_transaction`, even if the SMS contains a transaction amount,
account, or words such as *spent* or *debited*:

- pending or awaiting confirmation;
- failed, unsuccessful, not processed, or timed out;
- declined, blocked, rejected, voided, or cancelled; and
- a warning that money *may* have left and *will* be returned.

A later SMS that unambiguously says the event has now posted is judged on that
later SMS alone and can be a transaction.

### Holds and preauthorizations

Authorizations, preauthorizations, verification charges, liens, temporary holds,
and reserved/blocked amounts are `not_transaction`. A reduced available balance
does not turn a hold into a posted debit. Releasing or voiding a hold is also
`not_transaction`; do not label it as a refund without an explicit credit.

### OTP and security messages

OTP, one-time-password, verification-code, authentication, approval, login,
activate/block-card, suspected-fraud, and other security messages are
`not_transaction`. Under the current contract, this exclusion applies even when
the message also mentions an amount, account, or the word *transaction*. Never
copy an OTP into notes.

### Requests, bills, card notices, and mandates

The following are `not_transaction` because they request, schedule, summarize, or
acknowledge an obligation rather than report the target posting:

- collect requests, payment requests, invoices, and pay/approve links;
- amount-due, minimum-due, statement-generated, bill, EMI, and due-date reminders;
- credit-limit, available-limit, and card-spend summaries;
- a generic “payment received toward your credit card” acknowledgement; and
- mandate/autopay setup, registration, approval, modification, revocation, or
  scheduled-debit notices.

A separate notice explicitly confirming “INR 300.00 debited from A/c XX0420 under
the mandate” is a debit transaction. A card-bill payment is also a transaction
when the SMS explicitly reports the debit from a named bank account/card; the bill
or reminder itself is not.

### Wallet and BNPL exclusion

Posted activity in a wallet, stored-value balance, BNPL account, or pay-later
ledger is `not_transaction` under the current contract. This is a product boundary,
not a claim that the event is economically unimportant. A bank/card transaction
used to fund or repay such a product may qualify only when the SMS explicitly
reports the posting on the named bank account/card.

Invented example:

> INR 90.00 added to Comet Wallet; wallet balance is INR 410.00.

Label `not_transaction`. Do not select either amount.

### Multiple completed events

The V1 schema represents one transaction. When one SMS appears to report two
distinct completed postings and neither is clearly the single notified event, do
not choose arbitrarily or combine them. Enter the best provisional decision, mark
it uncertain, note “multiple completed events,” and send it to adjudication.

## 3. Choose the fields from the SMS

### Amount

Select the positive amount directly and grammatically tied to the qualifying
posting. Normalize only its written decimal value: remove the currency marker and
digit-group separators, but do not round, convert currency, infer missing digits,
apply a sign, or calculate a total. `INR 1,240.50` therefore has
`amount_decimal=1240.50`.

Do not select:

- available/current/closing balance;
- fee, tax, or tip when it is separately stated from the posted amount;
- credit or spending limit and remaining limit;
- amount due, minimum due, cumulative spend, rewards, or cashback;
- an exchange-rate equivalent; or
- an earlier, original, requested, failed, or pending amount.

Invented example:

> INR 840.00 debited from Card **8842 at Orbit Bazaar. Fee INR 8.00. Available
> limit INR 4,152.00.

Choose `840.00`; never choose `8.00` or `4,152.00`, and never add them. If a fee is
itself explicitly reported as a separate completed debit, treat the message as a
possible multiple-event case and adjudicate. When two plausible amounts remain
and grammar does not resolve them, mark uncertainty; do not guess.

### Account/card, not bank

The account field is the exact SMS mention of the customer's account or card
involved in the posting, normally its type cue plus masked identifier. It is not:

- the bank or issuer name;
- the sender ID;
- a merchant, payee, payer, or wallet;
- a payment rail, VPA, transaction/reference number, or phone number; or
- an ungrounded account inferred from prior messages.

In “Example Bank: INR 25.00 credited to A/c XX0420,” select `A/c XX0420`, not
`Example Bank`. If no exact account/card mention supports the posting, the current
contract is incomplete: use `not_transaction`, mark uncertainty when appropriate,
and explain the missing component without copying identifiers into notes.

When multiple accounts/cards appear, choose the one explicitly attached to this
amount and direction. If the SMS does not disambiguate them, adjudicate.

### Counterparty or explicit absence

The counterparty is the other named party in this posting: for example, the
merchant/payee on a debit or the payer/source on a credit. Select only a name or
address that appears in the SMS and is grammatically linked to the transaction.

Do not use the bank, issuer, sender ID, payment rail, generic words such as
“merchant,” a location, reference number, account, or purpose merely to avoid a
null. A bank name may be the counterparty only when the SMS explicitly presents it
as the other party (for example, the named collector of a posted fee), not merely
because it sent or issued the notice.

Use explicit counterparty absence when no other party is named, such as an ATM cash
withdrawal with no operator named. Absence is a valid label, not uncertainty. If
two named parties are both plausible, choose the one linked to the qualifying
amount/direction; otherwise mark uncertainty and adjudicate.

## 4. Exact source spans

All spans refer to the UTF-8 encoding of the unmodified SMS body, never the sender.
They are zero-based, half-open **UTF-8 byte offsets** `[start, end)`. Both offsets
must fall on character boundaries, and decoding the selected bytes must reproduce
the visible source substring exactly. Do not count Unicode code points or UTF-16
units; use the workbench selection rather than calculating offsets by hand.

For every transaction:

- select one contiguous, complete currency expression exactly as written, including
  its currency marker and grouping/decimal punctuation (for example,
  `INR 1,240.50`), while storing `1240.50` separately in `amount_decimal`;
- select the complete account/card mention, including its account/card cue and
  masked identifier;
- select only the counterparty value, excluding relation cues such as `to`, `from`,
  `at`, or `by` and excluding trailing sentence punctuation; or
- record explicit counterparty absence with no fabricated span.

The normalized amount comes from its amount span. Preserve original case,
whitespace, masking symbols, punctuation, and Unicode in every span. Do not trim
by rewriting text, correct a typo, expand an abbreviation, join discontinuous
text, or select from a normalized display. If the same text occurs more than once,
select the occurrence linked to the posting. Use uncertainty when occurrence or
boundary choice could change the label.

## 5. Uncertainty and adjudication

`uncertain` means that a reasonable reviewer could choose a different decision,
field, or span under this handbook. It is not a third class and must not be used to
avoid a provisional annotation.

When uncertain:

1. enter the best provisional label supported by the SMS;
2. set `uncertain` before saving;
3. add a short categorical note such as “amount attachment ambiguous” or “two
   possible account spans”; do not repeat the SMS, identifiers, or secrets; and
4. route the row to adjudication.

The workbench must retain that the row was ever uncertain even if a later reviewer
resolves it. Clearing the current uncertainty flag does not remove the QC
obligation. Corrections and adjudication must preserve the prior annotation,
reviewer identity, timestamps, and reason; never silently overwrite history.

An adjudicator applies this handbook to the source evidence. Blinded-test
adjudication remains proposal-blind. In training mode, preserve any prior proposal
exposure and never describe an exposed adjudication as blind. The adjudicator
records a reasoned final annotation and escalates a genuine policy gap rather than
inventing a one-row exception. No unresolved uncertainty or disagreement may enter
final gold. Repeated policy gaps require a new handbook version and a deliberate
review of affected V1 rows.

## 6. Quick reference

| SMS meaning | V1 decision | Direction/amount rule |
|---|---|---|
| Posted purchase, transfer out, withdrawal | `transaction` | `debit`; posted amount |
| Posted deposit or transfer in | `transaction` | `credit`; posted amount |
| Refund/reversal explicitly credited | `transaction` | `credit`; credited amount |
| Refund initiated or debit merely “reversed” | `not_transaction` | No fields |
| Pending, failed, declined, blocked, cancelled | `not_transaction` | No fields |
| Hold, preauthorization, or hold release | `not_transaction` | No fields |
| OTP, verification, security, or approval | `not_transaction` | No fields |
| Request, bill/due reminder, or mandate setup | `not_transaction` | No fields |
| Wallet or BNPL ledger activity | `not_transaction` | No fields |
| Balance, fee, or limit beside one posted amount | `transaction` | Select the posted amount only |
