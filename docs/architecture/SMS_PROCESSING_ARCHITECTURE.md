# PocketFinancer SMS processing architecture

Status: **canonical cross-platform architecture**
Last reconciled: 2026-09-22

This document defines the product boundary shared by Android and iOS. Executable
shared behavior, schemas, frozen releases, parity vectors, and evaluators live in
this repository. The native repositories implement that boundary with their own
runtime, storage, and UI frameworks.

## Product invariant

The local small language model is the central classifier and extractor. It decides
`none`, `abstain`, or one `posted` event and, for a posted event, proposes the
transaction fields and exact source spans.

The deterministic analyzer is advisory evidence. Its cues and candidate spans may
help the model and the reviewer, but they are not an allowlist, cannot suppress a
model call by themselves, and cannot override the unchanged source SMS.

The host remains authoritative for safety:

- strict JSON parsing with duplicate-key, unknown-field, coercion, and trailing-data
  rejection;
- zero-based, half-open Unicode scalar spans grounded against the unchanged SMS;
- exact decimal-to-minor-unit normalization without floating point;
- account resolution against versioned local aliases, with missing or ambiguous
  matches failing closed;
- receipt time, operation ownership, duplicate assessment, persistence, and
  recovery; and
- durable traces, review revisions, and local feedback.

```text
immutable source SMS + receipt time + configuration
                         |
             advisory deterministic analysis
                         |
              one local SLM classification/extraction
             /                 |                 \
          none              abstain          posted + scalar spans
             \                 |                 /
               strict host validation and grounding
                         |
        exact money + account resolution + duplicate assessment
                         |
             versioned routing and durable local state
```

## Routing policy

The frozen `native-integration-v4` release is implemented in both native source
trees in `review_only` mode. Under that release, even a complete, valid posted
result is retained for owner review. Android automated/emulator evidence and iOS
source/XCTest coverage are recorded in the historical implementation reviews;
they do not prove physical-device release readiness, and iOS execution still
requires the Mac/Xcode lane.

The shared `native-integration-v5` release now freezes the successor routing
policy below. Android adoption is still in progress, so this describes executable
shared behavior rather than current Android runtime verification:

| Result | Destination |
| --- | --- |
| Complete, strictly valid, uniquely resolved, non-duplicate posted result | `Transactions` |
| Incomplete or invalid result | `Review` |
| Missing or ambiguous account resolution | `Review` |
| `abstain`, runtime failure, interruption, or incompatible provenance | `Review` |
| `none` with a valid terminal classification | No transaction; retain only the evidence required by the active privacy policy |

`processing-config/5` binds `pocketfinancer.persistence-policy/2` in `automatic`
mode. `review-case/2` retains independently grounded fields with explicit `slm`
or `advisory_analyzer` provenance. Stored v1-v4 operations remain bound to their
original release and stored rollout behavior; v4 remains review-only. A retry
creates a distinct operation with an explicit parent operation ID. Native
migration, recovery, parity, and device evidence remain required.

## Review and correction

Review is for exceptions, not the normal destination for a complete valid result.
A review case must show the complete immutable source SMS, receipt time, stable
reason codes, advisory analyzer evidence, and the model proposal. Amount,
direction, account, and counterparty use accessible field-specific highlights.
Exactly one native text selection is active at a time so selection handles,
screen-reader focus, and field assignment remain unambiguous.

A reviewer may select exact source evidence, choose debit or credit when the
direction cannot be selected, and choose an existing account. Confirmation is one
atomic local transaction. Corrections append revision-bound `UserFeedbackEvent`
records. They remain local label evidence until explicit export and adjudication;
only source-grounded, split-safe approved labels may enter SLM fine-tuning or
deterministic-component improvement datasets.

## Processing transparency

Owner-visible processing must distinguish source evidence, advisory analysis,
model input, observable generation, raw output, strict parsing, grounding,
normalization, routing, persistence, and later owner correction. It must never
present fabricated chain of thought.

Real-time decoding transparency is required where the runtime exposes it. Android
can display decoded token deltas and cumulative structured output while generation
is active. Apple Foundation Models does not expose decoded token pieces or IDs via
the public API used by the iOS app; iOS must instead display each observable
cumulative structured-generation snapshot in real time and label token-level data
as unavailable. Reconstructed text must not be called token decoding.

## Model identity and the v3 incompatibility

`processing-config/3` requires a SHA-256 for a readable model file. That is valid
for Android GGUF files but impossible for Apple's system-managed model, which has
no app-readable model artifact. The evidence-backed decision is the additive
`processing-config/4` contract:

- `file_sha256` requires a real observed SHA-256 when the runtime is eligible;
- `system_managed_runtime` requires `model_file_sha256: null`; and
- runtime, OS, device, model identifier, prompt, grammar, validation, and release
  provenance remain explicit.

No platform may fabricate a hash. Releases v1-v3 remain byte-for-byte frozen.

## Ownership and operating model

| Repository/lane | Responsibility |
| --- | --- |
| `pF_slm_selection` on WSL | Architecture, frozen contracts, schemas, sanitized vectors, Python parity oracle, host GGUF evaluation, native-trace import, future Android/iOS native scorers, corpus/workbench, and aggregate model evidence |
| `pocket-financer-android` on Windows | Kotlin/JNI/Room/Compose implementation, Gradle verification, emulator evidence, and Android physical-device acceptance |
| `pocket-financer-ios` on macOS | Swift/Foundation Models/SwiftData/SwiftUI implementation, Xcode build/XCTest/simulator evidence, and iPhone acceptance |

Cross-platform changes start with a versioned shared contract and sanitized
vectors, then land independently in the native repositories. A pass in one lane
does not stand in for another. Private SMS and per-row outputs remain local.

## Compatibility and evidence

Stored v1/v2 candidate-selector operations and v3/v4 direct-extractor operations
must continue to load under their original release. New work must not silently
upgrade a stored operation or rewrite historical traces. Dated checkpoint and
implementation reports are indexed under `docs/history`; they are evidence, not
the current plan.

Continue with the [cross-platform roadmap](../plans/CROSS_PLATFORM_SMS_ROADMAP.md),
[evaluation strategy](../plans/SMS_EVALUATION_STRATEGY.md),
[direct extractor contract](../contracts/DIRECT_SMS_EXTRACTOR_CONTRACT.md), and
[native integration contract](../contracts/NATIVE_SMS_INTEGRATION_CONTRACT.md).
