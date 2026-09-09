# SMS Workbench Labeling Guide

Status: **active practical guide**

This guide explains how to use the current PocketFinancer SMS Review workbench and
how to create canonical human labels. It is a practical companion to the active
[taxonomy](../architecture/DATA_TAXONOMY_AND_CANONICAL_LABELS.md) and executable
rules in `src/pocketfinancer_sms/labels.py`. If this guide and the executable
contract ever disagree, stop labeling and resolve the policy gap instead of
inventing a one-message exception.

All examples below are fictional. Do not copy private message text, account
identifiers, OTPs, or other sensitive values into notes or documentation.

## 1. The mental model

The workbench displays two different kinds of information:

1. **Machine-generated diagnostic hints** help you browse and understand why a
   message was routed a certain way. These include weak class, disposition,
   selector action, candidates, cues, reason codes, and processing trace.
2. **Canonical human outcome** is the label you create after reading the original
   SMS. This is the authoritative human judgment.

Machine hints are deliberately called **weak** because they can be wrong. Never
copy them automatically into the human outcome. Read the complete SMS first.

For each message, answer these questions in order:

1. Is this a reliable outgoing message or invalid input?
2. Does it report a completed movement of money?
3. If not completed, does it describe a financial attempt, request, obligation,
   pending state, or failure?
4. If it is a completed movement, how many distinct events are reported?
5. For each completed event, what exact text proves amount, direction, account,
   and counterparty?
6. What kind of financial event is it, and which payment rail was used?

## 2. Starting and stopping the workbench

From the `pF_slm_selection` repository, start the local server with:

```bash
python scripts/run_sms_processing.py serve-workbench
```

Open the local URL printed by the command. Keep that terminal running while you
review. Enter a stable local reviewer name; the name separates your revision
history from another reviewer’s work. The app and all SMS data stay on
`127.0.0.1` and use no remote assets or telemetry.

To stop the workbench, return to the terminal and press `Control-C`. Stopping the
server does not remove saved drafts or submitted labels.

The screen has three independently scrolling columns:

- **Find messages** on the left filters the corpus.
- **Messages** in the middle lists matching SMS rows.
- The right side shows **Message evidence**, machine diagnostics, and the
  **Canonical human outcome** form for the selected SMS.

The top progress strip counts saved work. Expand **Progress and coverage by pool
and weak class** for aggregate coverage; “weak class” there refers to the
machine’s initial grouping, not final human truth.

## 3. Recommended order of work

Start with `annotation_training`. It is the best pool for learning the interface
because analyzer details are visible. Use `annotation_development` after the
labeling policy feels stable. Treat `protected_test` and `later_time_holdout` as
blind evaluation pools: decide from the source message before revealing any
machine suggestions or earlier labels.

For every SMS:

1. Read the entire message, including words such as *pending*, *failed*,
   *declined*, *requested*, *due*, *reversed*, or *credited*.
2. Make your own decision before expanding the analyzer panel.
3. Choose the canonical **Decision**. The interface fills the usual operational
   class and event state; verify them rather than assuming they are correct.
4. For `posted` or `multiple_event`, select exact evidence and add the event or
   events.
5. Choose financial family and payment rail when supported by the SMS.
6. Mark uncertainty and add a short categorical note when needed.
7. Use **Save draft** to preserve current work. Once `posted` or
   `multiple_event` is selected, the current UI requires a complete event draft
   before it can save. Use **Submit label** only when the outcome is complete and
   defensible from the visible SMS.

## 4. Canonical decision, operational class, and event state

These fields describe related but different aspects of the same human judgment.

- **Decision** is the overall annotation outcome.
- **Operational class** describes how the message should be handled as a type of
  input.
- **Event state** states whether a financial movement actually posted.

The allowed combinations are fixed:

| Decision | Operational class | Event state | Event records |
|---|---|---|---:|
| `posted` | `posted_candidate` | `posted` | exactly 1 |
| `not_posted` | `financial_non_posted` | `not_posted` | 0 |
| `non_financial` | `non_financial` | `no_event` | 0 |
| `non_financial` for outgoing/invalid input | `invalid_outgoing` | `no_event` | 0 |
| `ambiguous` | `ambiguous` | `unknown` | 0 |
| `multiple_event` | `posted_candidate` | `posted` | at least 2 |

The separate fields may look redundant, but they make different errors measurable.
For example, a message can contain financial language while still having a
`not_posted` event state. The word `candidate` in `posted_candidate` means the
operational category contains apparent completed movements; the human `posted`
decision is what confirms the truth.

### `posted`

Use when the SMS explicitly reports one completed movement of money. A posted
event needs an exact amount and direction. Account and counterparty must each be
marked present, absent, or unknown.

Examples include a completed purchase, debit, credit, transfer, withdrawal,
deposit, fee, salary credit, or refund credit.

Do not use `posted` merely because a message contains an amount, account, or the
word “transaction.” Status language can override those clues.

### `not_posted`

Use when the message is financial but the movement did not complete or is not yet
a completed posting. Common cases are:

- pending, processing, or awaiting confirmation;
- failed, declined, rejected, blocked, cancelled, or timed out;
- collect/payment requests and approval requests;
- bill, amount-due, or payment reminders;
- mandate setup or a future scheduled debit;
- refund initiated or expected, without an explicit completed credit; and
- a hold, authorization, or preauthorization rather than a final posting.

Do not add event evidence for a `not_posted` label.

### `non_financial`

Use when no target financial movement is expressed. The name does not mean the
SMS cannot mention money, a bank, or a balance. Examples include promotions,
administrative notices, balance-only information, OTP/security messages, device
registration, and service notifications.

When reliable metadata says the message was sent **from the user**, choose
`invalid_outgoing` as the operational class. Outgoing UPI activation strings such
as opaque `UPIACT` or bank-app device-binding payloads belong here. They are real
outgoing SMS database records, but they are outside the incoming transaction-alert
path.

### `ambiguous`

Use only when the visible evidence genuinely cannot determine whether an event
posted or what operational outcome applies. Examples include corrupt/truncated
text, directly conflicting status statements, or language with no safe reading.

Do not use `ambiguous` simply because an optional account or counterparty is
missing. A clearly completed event can still be `posted` with that field marked
`absent` or `unknown`.

### `multiple_event`

Use when one SMS clearly reports at least two distinct completed movements. Add a
separate event for each movement. Do not combine amounts or choose one event
arbitrarily.

## 5. Event evidence

The message body is immutable evidence. Select text directly in **Message
evidence**, then use the Amount, Direction, Account, or Counterparty button. Never
type a rewritten version of source evidence.

For a single `posted` event, set the top-level Family and Payment rail to the same
values as Event family and Event rail. For `multiple_event`, classify every event
individually; use the top-level value only when one description applies to the
whole message, otherwise use `unknown` or None consistently.

### Amount

Select the exact amount tied to this event, preferably including its explicit
currency code or marker, for example `INR 1,250.00` or `₹750`.

- Select one contiguous expression containing exactly one money number.
- Do not select an available balance, credit limit, reward, amount due, or fee
  when it is not the event amount.
- Do not add amounts, convert currencies, round values, or infer missing digits.
- If two amounts represent two completed movements, use `multiple_event`.

### Currency

Enter the three-letter uppercase ISO currency code, such as `INR`, `USD`, `EUR`,
or `GBP`, and record how it was established:

| Currency source | Use when |
|---|---|
| `explicit_code` | The message explicitly says `INR`, `USD`, `EUR`, and so on. |
| `explicit_unambiguous_symbol_or_marker` | The message uses an unambiguous configured marker such as `₹`, `Rs`, `€`, or `£`. |
| `user_primary_default` | The message gives a bare amount with clear transaction context, so the configured primary currency applies. |

Do not guess globally ambiguous markers such as `$` or `¥`. If context does not
resolve the currency safely, use an uncertain/ambiguous outcome rather than
inventing a currency.

### Direction

Direction is always relative to the user’s account or balance:

- **Debit**: money moved out—paid, spent, debited, transferred out, withdrawn.
- **Credit**: money moved in or back—credited, received, deposited, salary paid
  in, refund credited.

Select the exact word or phrase that proves the direction, such as `debited from`,
`credited to`, `paid`, or `received`. Do not infer direction only from the sender,
merchant, or transaction family.

### Account

The account is the user-side financial instrument explicitly involved in the
event: a masked bank account, card, VPA, or other supported account reference.

- `present`: a specific account reference is visible; select its exact text.
- `absent`: after reading the message, no account reference is present.
- `unknown`: the text is present but corrupted, ambiguous, or cannot safely be
  identified as the account.

Do not select the bank name, sender ID, transaction/reference number, phone
number, or counterparty merely to fill this field.

### Counterparty

The counterparty is the other party to the movement: merchant/payee for a debit,
payer/source for a credit, or another explicitly named party.

- `present`: a counterparty is explicitly named; select its exact text.
- `absent`: the event is clear but no counterparty is named.
- `unknown`: possible counterparty text exists but cannot be resolved safely.

Do not use the bank/issuer, payment rail, generic word “merchant,” reference
number, or location unless the SMS explicitly presents it as the other party.

`absent` means “the message clearly does not provide it.” `unknown` means “the
message may provide it, but I cannot safely determine it.” This distinction is
important: unknown optional fields intentionally prevent automatic target
projection.

## 6. Financial family

Financial family answers **what kind of event this is**. Use the most specific
supported family that describes the event’s economic purpose:

| Family | Practical meaning |
|---|---|
| `bank_transfer` | Generic account-to-account transfer not better described below. |
| `bill_payment` | Payment of a bill or billed obligation. |
| `card_purchase` | Purchase explicitly made using a debit/credit card. |
| `cash_deposit` | Cash deposited into an account. |
| `cash_withdrawal` | Cash withdrawn, normally at an ATM or branch. |
| `fee_charge` | Bank, card, service, or transaction fee posted as the event. |
| `insurance` | Insurance premium, payout, or other insurance movement. |
| `interest` | Interest credited or charged. |
| `investment` | Investment contribution, purchase, redemption, or proceeds. |
| `loan` | Loan disbursal, repayment, or loan-related movement. |
| `merchant_payment` | Purchase/payment to a merchant without a more specific card classification. |
| `refund` | Money explicitly returned and credited. |
| `salary_income` | Salary or payroll credit. |
| `upi_transfer` | Person/account transfer whose event is fundamentally a UPI transfer. |
| `wallet` | Movement into, out of, or within a stored-value wallet. |
| `other_financial` | Clearly financial but none of the named families fits. |
| `unknown` | A financial family applies but the message cannot resolve which one. |

A family and a rail can differ. For example, a merchant purchase can be
`merchant_payment` with rail `upi`; a bill payment can be `bill_payment` with rail
`nach`; and a refund can be `refund` with rail `card`.

Use **None** when family is not applicable, especially for non-financial input.
Use `unknown` when it is applicable but cannot be determined.

## 7. Payment rail

Payment rail answers **how the money moved**:

| Rail | Meaning |
|---|---|
| `bank_internal` | Transfer within the same bank’s internal system. |
| `card` | Debit/credit card network. |
| `cash` | Physical cash deposit or withdrawal. |
| `imps` | Immediate Payment Service. |
| `nach` | NACH/e-mandate clearing. |
| `neft` | National Electronic Funds Transfer. |
| `rtgs` | Real Time Gross Settlement. |
| `upi` | Unified Payments Interface. |
| `wallet` | Stored-value wallet rail. |
| `other` | A clear rail exists but is outside the named list. |
| `unknown` | A rail applies but cannot be determined. |

Do not infer a rail from sender reputation or prior messages. Use only the current
SMS. Use **None** if rail is not applicable.

## 8. Quick decision examples

| Fictional SMS meaning | Decision | Class / state | Family / rail |
|---|---|---|---|
| “₹850 debited from A/c XX42 at Blue Café via UPI” | `posted`, debit | `posted_candidate` / `posted` | `merchant_payment` / `upi` |
| “₹2,000 credited to A/c XX42 as salary” | `posted`, credit | `posted_candidate` / `posted` | `salary_income` / `bank_internal` or `unknown` if unstated |
| “UPI collect request for ₹850 awaits approval” | `not_posted` | `financial_non_posted` / `not_posted` | `upi_transfer` / `upi` |
| “Card purchase of ₹850 was declined” | `not_posted` | `financial_non_posted` / `not_posted` | `card_purchase` / `card` |
| “Your OTP is 123456” | `non_financial` | `non_financial` / `no_event` | None / None |
| Outgoing opaque UPI activation payload | `non_financial` | `invalid_outgoing` / `no_event` | None / None |
| One SMS reports a ₹500 debit and a separate ₹200 credit | `multiple_event` | `posted_candidate` / `posted` | one family/rail per event |
| Truncated text with irreconcilable posted and failed status | `ambiguous` | `ambiguous` / `unknown` | `unknown` where applicable |

## 9. Machine terminology in the workbench

### Weak class and weak facets

The analyzer’s initial, revisable categorization. Weak facets include operational
class, event state, family, rail, confidence, and reason codes. They exist for
browsing and diagnostics and must not be treated as a human label.

### Disposition

The deterministic routing recommendation:

- `invoke`: one grounded event candidate is strong enough for the selector’s
  normal path;
- `retain_review`: something is incomplete, conflicting, or ambiguous and should
  stay in review; and
- `discard`: terminally irrelevant to the incoming transaction path, such as a
  reliable outgoing message, invalid input, standalone OTP, request, failure,
  promotion, or administrative notice.

`discard` does not delete the row. Every source row remains in the offline corpus.

### Selector action

Whether the future small language model would be used:

- `run_normal`: normal single-event selection;
- `run_assistive`: model output may help a reviewer but cannot auto-persist; and
- `skip`: do not run the model for this row.

This is independent from human truth. A `skip` row can still deserve a meaningful
human label.

### Analyzer

The deterministic, non-model parser that preserves the SMS and identifies
clauses, candidate spans, status cues, currency context, and structural reasons.

### Candidate

A source-grounded piece of text that could fill amount, direction, account, or
counterparty. **Use** copies that exact candidate into the event editor. A missing
candidate is an analyzer-coverage issue; it does not change the human truth.

### Clause

A local segment of the SMS used to keep an amount and direction attached to the
same event rather than accidentally mixing evidence from different statements.

### Cue

A source-backed signal such as failure, pending, negation, request, balance,
promotion, credential/OTP, due amount, or payment rail.

### Queue reason / reason code

A stable machine explanation for why the row was invoked, retained, or discarded.
It is diagnostic, not an instruction to the reviewer.

### Candidate-oracle coverage

Shows whether the analyzer offered candidates for fields a correct human event
might need. “Complete core clauses” means an amount and direction candidate occur
in the same clause; it does not prove a transaction posted.

### Processing trace

An audit trail of analyzer, selector, validation, reconstruction, and persistence
stages. It explains system behavior without changing the source or revealing
hidden model reasoning.

## 10. Pools, groups, and filters

### Pools

- `annotation_training`: ordinary human-labeling pool used to develop training
  evidence; machine diagnostics are visible.
- `annotation_development`: development/validation pool used to improve policy and
  measure behavior without touching the protected test.
- `protected_test`: blind evaluation pool; suggestions and prior labels remain
  hidden until initial submission and explicit reveal.
- `later_time_holdout`: newer-time blind holdout used to check temporal
  generalization.
- `legacy_review`: rows linked to an older review asset; legacy labels are not
  silently converted into current truth.
- `regression_only`: compatibility/challenge rows retained for regression work,
  not ordinary training truth.

### Template and sender groups

Group buttons show messages with similar normalized text, the same sender family,
or the same sender-template combination. They help find repeated mistakes and
coverage gaps. Do not let another message in the group supply evidence missing
from the current SMS.

### Review state

- `unreviewed`: no saved work by the current reviewer;
- `draft`: incomplete work saved locally;
- `submitted`: a complete canonical label;
- `adjudicated`: a final resolution of disagreeing submitted labels.

## 11. Uncertainty, notes, revision, and adjudication

Mark **uncertain** when a reasonable reviewer could choose a different decision,
event boundary, field, or span under this guide. Still enter the best provisional
label; uncertainty is not a substitute for a decision.

Notes should be short and categorical, for example:

- `status language conflicts`;
- `amount attachment ambiguous`;
- `possible second completed event`;
- `account reference corrupted`; or
- `family unclear`.

Do not copy OTPs, full account numbers, private names, or the SMS body into notes.

Saving again creates a new revision; history is append-only. Adjudication becomes
available only after at least two submitted human labels disagree. The
adjudicator reads the source, resolves the disagreement, and records a new final
revision rather than overwriting either reviewer.

## 12. Protected blind review

For `protected_test` and `later_time_holdout`:

1. Read only the original message and create the initial human decision.
2. Save drafts if necessary; drafts do not reveal suggestions.
3. Submit a complete initial label.
4. Explicitly choose **Reveal after submission** if analysis is needed.
5. Only then inspect weak facets, candidates, group context, and disagreement
   information.

Never describe a decision made after reveal as blind.

## 13. Weak segregation correction versus human truth

The **Canonical human outcome** is the actual annotation. The separate **Correct
weak segregation** panel fixes the analyzer’s browsing facets when they are wrong.

For example, if the analyzer says `posted_candidate` but the SMS is a failed
payment:

1. submit the human outcome as `not_posted`;
2. optionally record a weak correction explaining that the deterministic class
   should have been `financial_non_posted`.

A weak correction never edits the SMS and never replaces the canonical label.

## 14. Draft, submit, preview, backup, and export

- **Save draft**: preserve work without asserting final truth. The current UI
  still requires a complete event draft after choosing `posted` or
  `multiple_event`.
- **Submit label**: validate and commit a complete canonical label.
- **Preview selector target**: show how a submitted rich label would project to
  `none`, `abstain`, or exact candidate IDs. A preview failure usually means
  candidate coverage is incomplete; do not change a truthful label merely to make
  preview pass.
- **Create backup**: create and verify a local SQLite snapshot.
- **Export labels**: create a local, corpus-bound export of submitted/adjudicated
  labels. It does not publish or upload anything.

## 15. Final checklist before submitting

- I read the complete current SMS rather than relying on sender or another row.
- My decision, operational class, and event state form an allowed combination.
- I did not treat a pending, failed, requested, due, or security message as posted.
- Every posted event has exact amount and direction evidence.
- Present account/counterparty fields have exact spans; absent and unknown fields
  have no span.
- Currency and its provenance are supported by the message or configured default.
- Family describes what happened; rail describes how it moved.
- Multiple completed events are separate event records.
- I marked genuine uncertainty and wrote only a short non-sensitive note.
- I did not change human truth merely to match analyzer candidates or preview.
