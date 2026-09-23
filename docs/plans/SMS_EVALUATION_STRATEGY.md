# SMS evaluation strategy

Status: **canonical evaluation plan**  
Last reconciled: 2026-09-23

Evaluation separates contract correctness, model quality, product routing,
review usefulness, runtime behavior, and release acceptance. Passing one category
does not substitute for another.

## Current evaluation capability

| Lane | Current state | What it proves |
| --- | --- | --- |
| Shared Python contracts and sanitized vectors | Successor release implemented; shared lightweight-CI gate passes locally | Parser, grounding, normalization, configuration, compatibility, partial Review evidence, and deterministic policy behavior |
| Android app integration | Automatic routing, partial Review, and live output implemented; complete debug unit/lint/build and Pixel_9 connected gates pass; Pixel_9 confirms decoded/cumulative display, a retained partial-highlight pair, exception Review, retry lineage, and controlled process replay with one Review case after the fix | Deterministic connected SQLCipher tests prove save/rollback/reopen/duplicate fencing, both Review replay orders, and edited-draft protection; sampled models did not produce a complete valid posted result, so model-driven automatic persistence and runtime duplicate fencing remain unverified; full accessibility and physical device remain open |
| Android-target GGUF direct extractor (`evaluate-extractor`) | Implemented | Reproducible semantic quality for one exact local GGUF/configuration on the host; not Android device proof |
| Encrypted native-trace import | Implemented | Consent-bound local ingestion of Android/iOS traces into the workbench with provenance and blind-pool protection |
| Android app/device runner and scorer | Planned extension | Exact app/runtime/model behavior on emulator and physical Android hardware |
| Apple Foundation Models evaluator and scorer | Missing; planned in the active roadmap | Exact iOS contract behavior through macOS/Xcode and supported physical iPhone hardware |

Host GGUF results are not Android measurements. Android results are not iOS
results. Simulator/emulator evidence is not physical-device evidence.

## Apple Foundation Models pipeline implementation

The Apple lane is a required deliverable, not merely a future comparison idea. It
has two cooperating parts:

1. A macOS/Xcode runner executes the exact iOS Foundation Models contract locally:
   instructions, guided schema, explicit model-processing locale check, observable
   cumulative snapshots, mapped draft, strict source grounding, validation, and
   terminal disposition.
2. `pF_slm_selection` owns the versioned suite manifest, protected package
   identity, encrypted result import, canonical-label comparison, aggregate
   scorer, and evidence report.

Each run is bound to suite, contract, prompt, schema, runner, OS, hardware, locale,
and observable system-runtime identity. The runner must support fresh execution,
resume by completed case ID, interruption, retry, partial-result quarantine, and
configuration-mismatch rejection. It records decoded tokens, token throughput,
logits, confidence, context size, and model-file hash as unavailable when Apple
does not expose them.

Start with sanitized synthetic fixtures on the Mac. Private-suite execution
requires explicit local packaging and encrypted return; no raw row or per-row
prediction may enter Git, CI, telemetry, or a hosted service. Product claims then
require the supported physical iPhone matrix. The same declared semantic cohort
may be compared with the Android-target GGUF lane, but runtime measurements and
model identity remain platform-specific.

## 1. Contract parity

Run the same sanitized fixtures through Python, Kotlin, and Swift and compare
canonical results. Cover strict JSON rejection, duplicate keys, trailing content,
Unicode scalar spans with emoji and combining marks, exact minor units and
overflow, account alias ambiguity, duplicate assessment, stable IDs/hashes,
stored-operation compatibility, migration, interruption, claim fencing, retry,
and atomic rollback.

## 2. Classification, extraction, and hints

Measure whether the SLM correctly returns `none`, `abstain`, or `posted`, then
measure exact amount, direction, account, optional counterparty, and evidence
spans. Include cases where analyzer hints are absent, incomplete, or wrong. A
valid grounded SLM extraction must not fail merely because the analyzer missed or
disagreed with it.

Use a newly adjudicated, sender/template-held-out human-gold set for product
claims. Report precision/recall, false transactions, missed transactions,
abstain rate, exact fields, grounding failures, and hint coverage. The 203-row
fixture remains regression-only.

## 3. Routing and persistence

For the successor contract, prove:

- every complete valid uniquely resolved non-duplicate posted result enters
  Transactions exactly once;
- incomplete, invalid, ambiguous, abstained, interrupted, incompatible, and
  failed operations enter Review and never the ledger;
- valid `none` outcomes create no transaction and follow the declared evidence
  retention policy; and
- retries, process death, replay, concurrent confirmation, and migration never
  duplicate or silently reinterpret an operation.

## 4. Review and labeling quality

Verify that Review shows the complete source and every valid partial field from
the last completed stage, with accessible field-specific highlights and one
active native selection. Test clear/reselect, debit/credit choice, account
resolution, retry, not-a-transaction, confirmation, and draft restoration.

Measure review rate, fields already correct, selections adjusted, time/actions to
resolve, abandoned reviews, and correction causes. Confirm that feedback remains
revision-bound and local, requires adjudication before becoming canonical truth,
and can be attributed to the responsible SLM/analyzer/host/UI component.

Evaluate the personal-corpus workbench separately from native Review. Measure
time and actions per annotation, save-and-next reliability, draft recovery,
validation-error recovery, skip/resume behavior, category and pool coverage,
disagreement handling, and backup/restore/export integrity. Verify that quick
actions never convert weak segregation into truth, alter pool assignment, reveal
protected hints early, or bypass grounding and revision checks.

Verify annotation-contract migration explicitly: v1 revisions remain readable
and hash-valid, new edits declare v2, v2 `posted` labels require all grounded
mandatory fields, `none` and `abstain` contain no event, and target preview
projects directly to the extractor without requiring analyzer candidate IDs.

## 5. Transparency

Verify that source, advisory analysis, request, observable generation, raw output,
parser result, validation, route, persistence, and later owner correction are
separate facts. Android must show live decoded token deltas without logging
private text. iOS must show live cumulative structured snapshots and explicitly
state that decoded tokens are unavailable. Neither platform may invent chain of
thought, confidence, token counts, model hashes, or model versions.

## 6. Runtime and device evidence

Measure end-to-end and generation latency, peak memory, thermals, battery,
cancellation, background/foreground transitions, locked-device behavior where
applicable, migration, and recovery. Record exact contract/configuration/evaluator
hashes, real model-file hashes where observable, system-managed identity where no
file exists, OS/device/build, decode settings, and every unverified lane.

## 7. Privacy

Keep raw SMS, identifiers, prompts, per-row outputs, trace keys, and databases
local. Checked-in material contains sanitized fixtures or aggregate results only.
Native transfer is explicit, encrypted, provenance-bound, size-limited, and
blocked from blind protected pools. Never fabricate or omit provenance to make
cross-platform results look comparable.

## Acceptance rule

A release decision requires all mandatory contract/routing/recovery tests, both
native evaluation lanes, relevant simulator/emulator gates, target physical
devices, protected human-gold results, privacy/data-rights/model-license review,
and an explicit owner decision. Until then, describe only the individual pieces
that were actually implemented and verified.
