# SMS evaluation strategy

Status: **canonical evaluation plan**  
Last reconciled: 2026-09-22

Evaluation separates contract correctness, model quality, product routing,
review usefulness, runtime behavior, and release acceptance. Passing one category
does not substitute for another.

## Current evaluation capability

| Lane | Current state | What it proves |
| --- | --- | --- |
| Shared Python contracts and sanitized vectors | Implemented | Parser, grounding, normalization, configuration, compatibility, and deterministic policy behavior |
| Local GGUF direct extractor (`evaluate-extractor`) | Implemented | Reproducible semantic quality for one exact local GGUF/configuration on the host |
| Encrypted native-trace import | Implemented | Consent-bound local ingestion of Android/iOS traces into the workbench with provenance and blind-pool protection |
| Android native evaluator and scorer | Planned | Exact app/runtime/model behavior on emulator and physical Android hardware |
| iOS native evaluator and scorer | Planned | Exact Foundation Models behavior on simulator-supported paths and physical iPhone hardware |

Host GGUF results are not Android measurements. Android results are not iOS
results. Simulator/emulator evidence is not physical-device evidence.

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
