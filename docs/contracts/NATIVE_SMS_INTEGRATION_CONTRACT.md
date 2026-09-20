# Native SMS Integration Contract

Status: **release v4 frozen; native mandatory steps 3–5 implemented; review pending**

Release manifest: `configs/sms_processing/contracts/releases/native-integration-v4.json`

Post-step-5 checkpoint:
`docs/plans/NATIVE_SMS_V4_POST_STEP_5_REVIEW.md`

## Compatibility boundary

Existing v1/v2 readers and stored artifacts remain supported byte for byte.
Native integration v4 succeeds v3 and binds `sms-extractor-input/1`, `sms-extractor/1`,
`extractor-validation-profile/1`, `extractor-prompt/1`,
`processing-config/4`, `processing-result/3`, `processing-trace/3`,
`reason-code-registry/2`, `account-resolution-profile/1`,
`review-case/1`, `user-feedback/3`, and `canonical-label/2`.

Android and iOS now contain the step 3–5 v4 parser, Unicode-scalar, money,
account-resolution, operation-routing, durable-state, migration, and recovery
ports. Android automated and emulator recovery gates pass. The iOS XCTest and
migration coverage is present but still requires execution on Xcode 26; review
drafts, confirmation, and UI work from steps 6 onward remain unimplemented. The
shared Python package remains the behavioral authority. Changing any frozen
asset requires a new release manifest and golden bundle.

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

## Extractor and persistence

The extractor is one direct, greedy, non-thinking attempt per operation with no
wall-clock deadline. Explicit user cancellation and operation interruption
remain effective. It may return only `none`, `abstain`, or one source-grounded
`posted` extraction. The deterministic analyzer is advisory, not an answer
allowlist. Duplicate JSON keys, extra text, unknown fields, type coercion, invalid
scalar spans, and inconsistent selections are invalid.

Recognition and persistence are separate. The typed persistence gate requires a
known frozen contract/configuration, exactly one posted event, exact money,
currency, timestamp, a uniquely resolved existing account, consistent family and
evidence, no blocking conflicts, and an enabled rollout mode. Automatic
persistence is disabled in v4; native apps must operate in shadow/review-only mode
until a separately reviewed release enables it.

## Trace, feedback, and transfer

Trace events are sequence-numbered and hash chained to immutable operation inputs.
Feedback is append-only, revision-bound, idempotent by action ID, and distinguishes
source-grounded, inferred, user-supplied, and unknown values. Feedback never
silently becomes a canonical training label.

Private source text, model output, traces, labels, and feedback remain on device.
Any diagnostic transfer requires explicit consent and an encrypted trace bundle;
the contract does not authorize analytics, telemetry, cloud inference, training,
or export.

## Native review and scalar conversion

The review screen displays the complete immutable source, receipt time as
read-only, separate amount/direction/account/counterparty highlights, analyzer
suggestions identified as advisory, the extractor suggestion, and stable reason
messages. Reviewers select source text for corrected evidence, may use an
explicit debit/credit control when direction is not selectable, and choose an
existing account before confirmation. Feedback is append-only and revision-bound.

Contract spans count Unicode scalar values. Kotlin walks code points to convert
scalar indices to UTF-16 offsets and rejects surrogate-pair splits. Swift derives
`String.Index` values by walking `unicodeScalars`, not `Character` graphemes.
Both ports must pass the frozen emoji and combining-mark golden vectors.

## Historical releases

Native releases v1/v2/v3 and the candidate-selector assets are frozen historical
compatibility artifacts. They remain valid for their stored operations and
reproducibility only. Release v4 is additive and changes no historical byte.


## V4 model identity provenance

V4 leaves every v3 byte unchanged. It adds model_identity_kind to the extractor configuration: An eligible file_sha256 runtime requires a real model_file_sha256. An ineligible file_sha256 runtime may use null when the file is unavailable or unreadable, or a real hash when observed; it must never fabricate one. system_managed_runtime always requires that field to be null. Both require a nonempty model_identifier when eligible. This supports Apple Foundation Models without inventing a file hash while preserving file-backed Android provenance.

V4 exports made with manifest 293086a80b437dea7301ebfa218b537aa3d7b8c40ccea118f3091de1e7e264ed are obsolete. The only changed bundle paths are configs/sms_processing/contracts/releases/native-integration-v4.json and configs/sms_processing/contracts/v4/processing-config.schema.json. The package exporter intentionally refuses overwrites: before replacing those two paths, verify their old hashes (manifest 293086a80b437dea7301ebfa218b537aa3d7b8c40ccea118f3091de1e7e264ed; schema 20be50184176295446c627dbe9d36a86553e28d3f07dd687ae7dafabe252e9d3), replace only those verified files with the new exported bytes, then run package_native_contract_v4.py check. Any other mismatch fails closed and requires a clean export root.
