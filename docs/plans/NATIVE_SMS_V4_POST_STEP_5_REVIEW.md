# Native SMS v4 post-step-5 review checkpoint

Date: 2026-09-16

Status: **implementation present; paused for required review; not accepted as a
completed app integration**

> Historical checkpoint. Execution resumed after this review and implemented
> steps 6–8. Current evidence and remaining acceptance gaps are recorded in
> [NATIVE_SMS_V4_IMPLEMENTATION_REVIEW.md](NATIVE_SMS_V4_IMPLEMENTATION_REVIEW.md).
> The checkpoint details below are preserved as the state observed on
> 2026-09-16.

This record covers mandatory execution steps 3–5 from
`NATIVE_SMS_V3_APP_IMPLEMENTATION_PLAN.md`, plus the contract correction needed
to execute them truthfully. Steps 6 and later are deliberately untouched.
Automatic persistence remains disabled.

## Contract decision

Frozen `processing-config/3` requires every eligible runtime to identify a
model file with its SHA-256. Apple Foundation Models is system managed and does
not expose an app-readable model file. Supplying any substitute value would
fabricate provenance.

The additive `native-integration-v4` release therefore introduces
`processing-config/4` and `model_identity_kind`:

- `file_sha256`: an eligible runtime must stream and record the SHA-256 of the
  actual model file. An unavailable or unreadable file makes the runtime
  ineligible; no hash is invented.
- `system_managed_runtime`: `model_file_sha256` must be null, while the model
  identifier, runtime, OS, device, prompt, grammar, and validation provenance
  remain explicit.

Android uses `file_sha256`. iOS uses `system_managed_runtime`. Historical
v1/v2/v3 artifacts and readers are unchanged.

Verified shared hashes:

| Artifact | SHA-256 |
|---|---|
| Frozen v3 release manifest | `61609c3336374c8b96b1e36cb90b5af01039ceecefcfc1b0091fd9359804436b` |
| Current v4 release manifest | `0d3bf18f91d0a197c7bb56b5e082fd2851ce072f854452a9647c52d45b3433d8` |
| v4 processing-config schema | `257405f8cff35db2131dc2325388cc9845cb1e62775b5df1b45c321908642406` |
| Release-manifest schema | `ae86ae96dd65df7c1be9a4478b825e1c035d61bbbfda32aa1023da19e92b7b30` |

The authoritative rationale is recorded in
`docs/architecture/SMS_PROCESSING_DECISION_LOG.md`.

## Step 3 — parsing, normalization, and grounding

| Requirement | Android evidence | iOS evidence | Checkpoint state |
|---|---|---|---|
| Strict extractor parsing | `SmsExtractorValidator.kt` rejects duplicate keys, trailing content, coercion, unknown/missing fields, invalid numbers, and invalid nullable combinations | `SmsExtractorValidator.swift` implements the same fail-closed boundary | Android executed; iOS test source present but not executed on this host |
| Unicode scalar spans | `UnicodeScalarSpan.kt` converts code-point ranges to safe UTF-16 offsets | `UnicodeScalarSpan.swift` walks `unicodeScalars` and derives Swift/NSRange boundaries | Frozen emoji, combining-mark, repeated-text, and boundary vectors are included on both platforms |
| Exact money | `SmsExtractorNormalizer.kt` produces checked `Long` minor units without floating point | `SmsExtractorNormalizer.swift` produces checked `Int64` minor units without floating point | Currency, scale, precision, positivity, and overflow are validated |
| Account resolution | Android hashes normalized VPA/suffix aliases and requires one live owned-account match | iOS applies the same normalized alias and ownership rules | Unique, unresolved, and ambiguous cases have focused tests |
| Sanitized vectors | `SmsExtractorValidatorTest.kt` covers all five shared vectors | `SmsExtractorValidatorTests.swift` covers all five shared vectors | Android passed; iOS requires Xcode execution |

Counterparty value/span nullability is enforced. No default account and no
balance behavior were added.

The post-checkpoint source audit tightened four fail-closed edges before review:

- Android now accepts only the four JSON whitespace bytes, rejects malformed
  UTF-16 source before scalar conversion, and enforces the frozen canonical
  declared-decimal grammar.
- The Android exact-money regression now exercises the actual evidence number;
  it covers precision, signs, leading zeros, overflow, and `Long.MAX_VALUE`
  rather than failing earlier on an unrelated missing number.
- iOS now rejects decimal numeric tokens such as `start_scalar: 1.0` instead of
  allowing `JSONSerialization` to erase that coercion, and its VPA grammar now
  matches the shared/Android grammar.
- Both ports apply the shared counterparty scalar-length boundary and the
  required case-fold behavior for the covered multi-scalar cases.

## Step 4 — v4 operation routing

Both apps now build immutable `processing-config/4` snapshots, call the direct
extractor exactly once with greedy/non-thinking settings and no parser/model
wall-clock timeout, treat analyzer output as advisory only, normalize into
`processing-result/3`, and apply the frozen gate order in review-only mode.
Android passes the 512-token limit to its GGUF runtime. iOS now passes
`maximumResponseTokens: 512` to Foundation Models instead of recording the
limit only in provenance.

Routing is explicit:

| Stored/new work | Route |
|---|---|
| New or “retry with current” work | v4 direct extractor |
| Historical v2 operation | existing v2 coordinator |
| Historical v1, incompatible v3, or unknown release | retained failure/review; no silent release upgrade |
| Historical v4 operation | v4 using its stable event and receipt identity |

Android computes the model SHA from the actual GGUF at snapshot and execution
time and compares the observations. iOS records the Foundation Models identity
as system managed with a null file hash. Neither platform fabricates a digest.

## Step 5 — durable state, migration, and recovery

Android advances Room from schema 6 to 7, adds the nullable indexed transaction
fingerprint, makes the account-alias composite index non-unique so ambiguity can
be represented, persists v4 normalized result/gate data, and retains durable
operation/review/recovery behavior. The exported schema is
`data/schemas/com.pocketfinancer.data.db.AppDatabase/7.json`.

iOS adds the explicit SwiftData V6-to-V7 lightweight migration boundary,
persists idempotent v4 result/gate and duplicate state through the existing
durable JSON fields, and adds migration/recovery coverage for an interrupted
attempt retained for review.

Recovery does not silently change a stored release. New retries use a new v4
operation; compatible stored operations keep their original configuration and
stable identities.

## Verification completed

### Shared foundation

- Repository safety check: passed.
- Ruff full and strict error selections: passed.
- Python tests: **794 passed**.
- Contract packaging check against both native bundles: **44 artifacts and 45
  files including the manifest verified** per bundle.
- Whitespace/error diff check: passed.

### Android

- Unit tests for the pipeline and data modules: passed.
- Focused strict parser/scalar/money/case-fold suite: **7 passed**.
- Pipeline/data lint and debug APK assembly: passed.
- Room schema regenerated after a clean data-module build; migration tests:
  **73 passed**.
- Pixel 9 emulator, API 15: focused durable recovery instrumentation:
  **4 passed**.
- Packaged v4 contract bundle: passed the shared package checker.
- Whitespace/error diff check: passed; only line-ending warnings were emitted.

### iOS

- Packaged v4 contract bundle: passed the shared package checker.
- Bundle property-list metadata parses and identifies v4 correctly.
- Swift/XCTest coverage for parsing, vectors, account resolution, v4 JSON,
  migration, and recovery is present.
- Whitespace/error diff check: passed; only line-ending warnings were emitted.

## Open verification and scope boundaries

This Windows/WSL host has no Swift compiler, `xcodebuild`, SourceKit, or iOS
simulator. Therefore Swift 6 compilation, XCTest, V6-to-V7 migration execution,
Release build, and simulator/device recovery remain **unverified**, not passed.
They require an Xcode 26 macOS runner.

The following are intentionally outside this checkpoint and remain undone:

- step 6 review drafts, validation, and atomic confirmation;
- step 7 native source-selection UI and Transactions sections;
- later full automated, simulator, physical-device, and local review-all gates;
- any automatic-persistence rollout decision.

The three working trees remain uncommitted. No deployment, upload, telemetry,
private-data movement, model publication, or contract hash fabrication occurred.

## Review decision

This checkpoint is ready for inspection of steps 3–5. It must not be treated as
approval of the full implementation. The next permitted action after review is
either correction within steps 3–5 or explicit continuation to step 6.
