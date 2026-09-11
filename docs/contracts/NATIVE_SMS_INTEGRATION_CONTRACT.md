# Native SMS Integration Contract

Status: **frozen for Kotlin and Swift implementation**

Release manifest: `configs/sms_processing/contracts/releases/native-integration-v2.json`

## Compatibility boundary

Existing `/1` readers and stored artifacts remain supported. Native integration uses
`sms-analysis/2`, `processing-result/2`, `processing-trace/2`, and
`user-feedback/2`, plus `processing-config/2`, selector output `/1`, selector
validation profile `/3`, and native trace bundle `/1`. Changing any frozen asset
requires a new release manifest and golden bundle; an existing release is never
rewritten in place.

## Frozen behavior

- Source text is immutable. Matching uses per-code-point Unicode 14.0.0 NFKC,
  case folding, and Unicode-whitespace collapse; returned evidence is unchanged
  source text with host-computed character and UTF-8 offsets.
- IDs, configuration hashes, event hashes, and canonical JSON use the algorithms
  declared by the release manifest and parity vectors.
- Money is exact signed 64-bit minor units with currency scale. Floating-point
  values are not part of the contract.
- Expected refunds and authorization holds are recognized state, not posted ledger
  truth. A completed financial clause may coexist with an unrelated credential
  clause without making the completed event disappear.
- Timestamp provenance and account resolution are explicit. Missing, ambiguous, or
  silently inferred core values fail closed to review.

## Selector and persistence

The selector is one direct, greedy, non-thinking attempt per operation with no
wall-clock deadline. Explicit user cancellation and operation interruption remain
effective. It may return only `none`, `abstain`, or one source-grounded `posted`
selection. Duplicate
JSON keys, extra text, unknown fields, type coercion, unknown candidate IDs, mixed
clauses, and inconsistent selections are invalid.

Recognition and persistence are separate. The typed persistence gate requires a
known frozen contract/configuration, exactly one posted event, exact money,
currency, timestamp, a uniquely resolved existing account, consistent family and
evidence, no blocking conflicts, and an enabled rollout mode. Automatic
persistence is disabled in this release; native apps operate in shadow/review-only
mode until a separately reviewed release enables it.

## Trace, feedback, and transfer

Trace events are sequence-numbered and hash chained to immutable operation inputs.
Feedback is append-only, revision-bound, idempotent by action ID, and distinguishes
source-grounded, inferred, user-supplied, and unknown values. Feedback never
silently becomes a canonical training label.

Private source text, model output, traces, labels, and feedback remain on device.
Any diagnostic transfer requires explicit consent and an encrypted trace bundle;
the contract does not authorize analytics, telemetry, cloud inference, training,
or export.
