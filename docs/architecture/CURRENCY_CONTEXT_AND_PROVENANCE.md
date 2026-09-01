# Currency Context and Provenance

Status: **active**
Contract sources: `currency.py`, `profiles.py`, and
`configs/sms_processing/currency/iso-4217.json`

## Configuration snapshot

Each processing operation receives an explicit `CurrencyContext` containing:

- `primary_currency`: an uppercase ISO-4217 code selected by the user;
- `profile_ids`: the deterministic structural profiles enabled for the operation;
- versioned profile revisions and a canonical configuration hash recorded with
  analysis and trace state.

The Indian archive rebuild passes `INR` in
`configs/sms_processing/archive-india-inr.json`. INR is neither inferred from the
archive nor embedded as a universal default. A later app currency change affects
future operations only; historical operations retain their snapshot.

## Precedence

1. An explicit supported ISO code in the message controls that amount.
2. An explicit unambiguous symbol or locale marker controls that amount.
3. A bare amount with valid transaction context uses the configured primary
   currency.
4. Conflicting currencies or an ambiguous marker retain the row for review.

`$` and `¥` are globally ambiguous and never silently become USD or JPY, even
when that currency is the user primary. A three-letter token is called an
unsupported currency only when it appears in the current ISO 4217 List One; this
prevents payment rails such as UPI from being misclassified as currencies. The
checked-in List One code set was verified against the official
[SIX Maintenance Agency XML](https://www.six-group.com/dam/download/financial-information/data-center/iso-currrency/lists/list-one.xml)
on 2026-09-01 and is contract-tested against runtime declarations.

Currency provenance is one of:

- `explicit_code`;
- `explicit_unambiguous_symbol_or_marker`;
- `user_primary_default`.

The provenance travels with the amount candidate and reconstructed transaction.
The persistence gate has an explicit allowlist; it does not assume every parsed
provenance is safe for automatic persistence.

## Exact money

The host parses decimal text into positive integer minor units using the currency's
declared scale. Non-finite values, zero/negative amounts, unsupported codes, and
excess decimal precision fail. The model selects an amount ID and cannot rewrite
the number, currency, scale, or provenance.

## Generalization boundary

`core-en` owns country-neutral English movement language and globally
unambiguous markers. `india` owns INR/Rs/₹, lakh-style grouping,
and UPI/IMPS/NEFT/RTGS/NACH cues. Any India-specific sender conventions added
later must live in that extension as well; sender shape is not a current
invocation requirement.
Additional country/locale behavior must enter through a reviewed profile plus
synthetic tests, never through a hidden primary-currency conditional.

The checked-in currency table is the runtime-supported ISO-4217 subset, not a
claim to implement every currency today. Adding a code requires minor-unit,
marker-ambiguity, explicit/default-path, conflict, and persistence tests.

## Later native responsibility

Android and iOS must provide onboarding/settings for primary currency, snapshot it
per operation, surface provenance in the processing trace, and avoid retroactively
rewriting saved transactions. Those app changes are intentionally deferred to the
native-integration session.
