# Cross-platform SMS roadmap

Status: **canonical active roadmap**  
Last reconciled: 2026-09-24

This is the only forward-looking SMS implementation plan. Frozen contracts and
dated evidence remain authoritative for the versions they describe, but they do
not define the next product behavior.

## Product outcome

PocketFinancer should turn a complete, trustworthy SMS extraction into a normal
transaction without asking the owner to review it. Review is the exception path
for incomplete, invalid, ambiguous, abstained, interrupted, incompatible, or
failed work.

The local SLM is the central classifier and extractor. The deterministic analyzer
supplies useful source-grounded hints; it never limits what the SLM may extract.
Host code remains authoritative for strict parsing, evidence grounding, exact
money, receipt time, account resolution, duplicate protection, persistence,
operation ownership, and recovery.

## Current baseline and observed gaps

The frozen `native-integration-v4` source is present in both apps. It includes
direct extraction, strict validation, durable review state, review drafts, atomic
confirmation, and native span-selection UI. Stored v4 operations remain
`review_only`; the additive Android release now uses the final automatic policy.

The latest Android emulator trial exposed product gaps that source-level and
automated checks did not settle:

- the previously visible live decoded-token stream was not visible;
- the expected source-SMS span review experience was not visible in the tested
  flow; and
- the tested routing did not match the intended exception-only review policy.

The original Pixel_9 emulator now exercises the Android runtime and navigation
path. Direct Review-card navigation, explicit direction fallback, existing-account
selection, and live decoded/cumulative output are verified there. Complete-valid
automatic saving remains unverified there because sampled local models returned
invalid extractor output. iOS behavior remains unverified on the Mac/Xcode lane.

The shared repository now contains frozen `native-integration-v5`,
`processing-config/5`, `persistence-policy/2`, `reason-code-registry/3`, and
`review-case/2`. The existing `ExtractionCoordinator` remains the routing oracle;
the successor binding selects its automatic mode rather than introducing another
routing engine. Shared tests and Android debug unit, lint, and build gates pass
locally. Pixel_9 verifies exception Review and part of the UI/runtime path, not
successful model-driven automatic persistence. Physical-device behavior remains
unverified.

## 2026-09-24 Android Review and settings correction

The owner tested the Android PR on the emulator. The visible grammar toggle is
off by default, but v5 inference always uses grammar; this is a real settings
disconnect. The separate latest-token-delta card is unwanted. The desired
observable output is one live, growing model-output view. A future operation
must capture grammar on/off in a new versioned configuration; stored v5 behavior
and original-configuration retries remain frozen.

The Review interaction should be selection-first: select exact source wording,
then tap a field to assign it. Offer short, source-backed one-tap choices for
missing fields so precise drag selection is optional. The native selection should
clear after assignment or dismissal. A field-specific, accessible evidence cue
must remain visible.
Distinct colored source highlights are preferred if they work with touch
selection; colored chips plus a source excerpt or tap-to-focus preview are
acceptable when they give a clearer interaction. Avoid text-entry forms for
ordinary correction. Resolve Direction once, do not repeat analyzer "Paid" as a
competing main-flow suggestion, and move raw analyzer suggestions out of the
primary confirmation area.

A grounded account reference may create a new local account during owner
confirmation when no unique existing account matches. That account and the
transaction must commit atomically and remain idempotent under retry/replay.
The current automatic-save rule still requires unique existing-account
resolution; the owner-confirmed Review creation path is a separate decision.
Amount remains required. Render it in major units (INR 125.00 for 12,500 minor
units) and keep exact minor-unit storage. A disabled Confirm action must reveal
the missing or invalid field. Remove the unnecessary receipt-time explanation
while preserving the immutable timestamp.

Acceptance is an emulator interaction using invented SMS: source selection and
dismissal, each field assignment, readable evidence, no pre-existing account,
correct amount, successful confirmation, duplicate replay, and grammar off/on
diagnostics. Follow with accessibility and physical-device checks. The Android
implementation checklist lives in docs/sms-processing-next-steps.md.
## Workstream 1 — freeze the successor behavior

Status: **implemented and verified in the shared lightweight-CI lane**

1. Add a new processing configuration, routing policy, and reason-code release.
   Do not edit v1-v4 assets or reinterpret stored operations.
2. Define one typed success route: a complete, strictly valid, uniquely resolved,
   non-duplicate posted result enters `Transactions` atomically.
3. Define exception routes for incomplete, invalid, ambiguous, abstained,
   interrupted, incompatible, and failed work. Every exception retains the
   evidence needed for review and recovery.
4. Define terminal `none` behavior and its minimum evidence-retention policy
   separately from Transactions and Review.
5. Add migration, retry, and recovery vectors proving that old operations keep
   their original release and new-policy retries create explicit new operations.

Implemented bindings include exception-only Review, terminal valid `none`, atomic
transaction-settlement intent, original-release compatibility, explicit retry
lineage, partial grounded field evidence, and sanitized routing vectors. Android
migration, atomic routing, and recovery paths now have local automated coverage;
emulator and device execution remain open in Workstreams 2 and 3.

Shared verification on 2026-09-22 ran `python scripts/check_repo_safety.py`, both
required Ruff commands, `pytest -q`, and `git diff --check` from the activated WSL
environment. The result was 820 passing tests with clean safety, lint, and diff
checks. This is local shared-contract verification only: Android Gradle, emulator,
and physical-device lanes remain unverified. The next implementation start is the
Android release binding, Room migration/atomic route, and focused routing tests.

Checkpoint handoff (2026-09-22): commit `ba90c01` completes the shared successor
release, partial Review contract, frozen bundle, vectors, and compatibility tests.
The exact verification lane was WSL lightweight CI using the commands above; no
emulator or physical device was used. Remaining work is Android asset binding,
version-aware routing, atomic persistence, Review projection/navigation, and live
token presentation. Start with `configuration_v5.py`, `processing_v3.py`,
`test_successor_routing.py`, the v5 manifest, and Android's existing
`DefaultSmsV4ProcessingCoordinator`/`SmsProcessingStore` tests.

Checkpoint handoff (2026-09-23): Android commits `fcf6614`, `f6079ff`, and
`01f3861` implement automatic routing, partial Review in the existing UI, direct
case navigation, and live decoded/cumulative output; `fd8b8b0` corrects the
legacy migration test fixture. Focused Android tests and
`./gradlew.bat testDebugUnitTest lintDebug assembleDebug --no-daemon` passed
locally. Pixel_9 then passed 15 Compose and 4 encrypted-recovery instrumentation
tests. Its synthetic runtime audit observed 7 operations (5 realtime, 2 manual),
6 invalid selector completions, 7 Review cases, and no transactions; one
operation was interrupted by an audit foreground switch. Both Qwen3-0.6B Q8_0
and debug-upgraded Qwen3-1.7B Q4_K_M returned invalid output in this small
sample. Review extensions retained 2 valid SLM amount fields and 1 valid
direction field. Decoded-token deltas and growing cumulative output were
observed during active inference without recording their text. A later Pixel_9
Review inspection confirmed separate retained SLM amount and direction
highlights while the account remained unassigned. An
additional synthetic prompt-example alert entered Review with malformed JSON.
The aggregate-only audit found six short open JSON fragments and four readable
posted objects with strict grounding failures; it recorded no SMS or output
text. The frozen Android runtime uses grammar-constrained greedy decoding and
a 512-token answer limit, but its stored evidence does not identify the stop
condition for each open fragment. Strict grounding was not relaxed.

An existing Review Retry created a parent-linked operation and reused its case
on Pixel_9. A controlled process kill during synthetic inference exposed
duplicate Review cases for one source after WorkManager replay. Android now
reuses an unedited open v5 same-source Review case and preserves a newer case
when an older claim recovers. Encrypted connected tests covered both replay orders and protected an
owner-edited draft. After installing the fix, another controlled kill and replay added two
operations but only one Review case; the earlier duplicate synthetic cases
remain in emulator data. A separate connected SQLCipher fixture passed atomic
automatic-save rollback, persistence, encrypted reopen, and retry duplicate
fencing. The complete Android debug unit/lint/build gate, 15 app UI tests,
8 encrypted data tests, and 2 pipeline connected tests passed; the opt-in audit
passed separately and skipped the ordinary suite. This is deterministic device
store evidence, not a successful model-driven save. Complete-valid model-driven
saving, runtime duplicate fencing after such a save, full accessibility, and
physical-device behavior remain open. Relevant Android sources are
`DefaultSmsV4ProcessingCoordinator`, `SmsProcessingStore`,
`GroundedReviewContent`, `ReviewDetailScreen`, `TransactionsScreen`, and
`SmsTelemetryViewer` and `SmsSyntheticRuntimeAuditTest`; the platform handoff is
`docs/sms-processing-next-steps.md` in the Android repository.

## Workstream 2 — restore processing transparency

Android status: **implemented; verified by local tests and Pixel_9 live output**. The
decoded-token callback reaches the existing processing surface, where the latest
delta and cumulative structured output are distinct and lifecycle/owner cleanup
is covered by local tests. During active emulator inference, both output panes
were non-placeholder and cumulative output grew. Physical-device behavior and
full lifecycle cleanup on-device remain unverified. iOS work is
unchanged and outside the current Android task.

1. Trace the Android generation callback from llama.cpp/JNI through the runtime,
   durable attempt state, presentation model, and visible processing surface.
2. Restore live decoded-token deltas and cumulative structured output while the
   Android SLM is running. Keep them distinct from reconstructed JSON, validation,
   routing, and saved data.
3. On iOS, display each cumulative structured-generation snapshot that Foundation
   Models exposes. State explicitly that decoded token pieces/IDs are unavailable;
   never reconstruct text and label it as token decoding.
4. Preserve source evidence, advisory analyzer output, exact request, observable
   generation, raw response, parser result, field validation, routing, persistence,
   retry, and later owner correction as separate facts.
5. Keep all private text local and out of logs, telemetry, notifications, crash
   reports, screenshots, and checked-in fixtures.

## Workstream 3 — make Review a source-labeling interaction

Android status: **implemented; partially verified on Pixel_9**. The
existing Review UI projects partial grounded fields, labels analyzer suggestions,
requires valid mandatory fields and deliberate account choice, and opens a
selected card directly. Direct navigation, direction fallback, account selection,
and disabled confirmation with a missing amount were exercised on the emulator.
One retained amount/direction highlight pair is visually verified on
Pixel_9. Synthetic Review retry and controlled process recovery were exercised,
including one fixed same-source replay duplication. Full accessibility remains
unverified there.

1. Put Processing and Needs Review above the confirmed ledger on Transactions;
   do not create a disconnected manual-entry form as the main workflow.
2. Show sender context, the complete immutable SMS, read-only receipt time, and
   stable reason messages.
3. Project every successfully grounded field from the last completed stage into
   the SMS body even when a later field or gate failed. A missing account must not
   hide a valid amount or direction span.
4. Use accessible field-specific styling for Amount, Direction, Account, and
   Counterparty. Keep all assigned spans visible; exactly one active field owns
   native selection handles at a time.
5. Let the owner clear/reselect a span, choose debit or credit when source
   selection is not suitable, choose or resolve an account, retry, or mark the
   message as not a transaction. Do not expose epoch milliseconds or allow the
   model/user to overwrite receipt time.
6. Confirm account changes, transaction creation, review resolution, and feedback
   in one idempotent local transaction.

## Workstream 4 — turn corrections into governed learning data

1. Append every owner correction as revision-bound local feedback while
   preserving the original model result and analyzer evidence.
2. Import explicitly exported encrypted Android/iOS trace bundles into the local
   workbench without crossing blind protected-pool boundaries.
3. Present corrections for adjudication. User feedback is label evidence, not
   automatic canonical truth.
4. After adjudication, project canonical labels into separate error slices for
   SLM classification/extraction, analyzer hint coverage, grounding, account
   resolution, duplicate handling, routing, and UI/recovery failures.
5. Only approved, source-grounded, split-safe labels may enter fine-tuning or
   component-improvement datasets. Keep a fresh sender/template-held-out test set.

## Workstream 5 — make personal-corpus annotation fast

The shared repository already contains the private canonical corpus, leakage-safe
pool assignments, weak operational segregation, annotation queues, and a local
SQLite workbench. Preserve that foundation and add a focused annotation mode for
quickly labeling the owner's personal SMS dataset.

1. Version the workbench annotation contract from its implemented
   `canonical-label/1`/Candidate Selector form to `canonical-label/2` and
   direct-extractor target projection. Preserve old revisions as readable
   historical evidence; never rewrite them in place or silently reinterpret an
   old decision.
2. Let the owner enter a queue and annotate one complete SMS at a time with a
   keyboard-first save-and-next workflow. Persist drafts continuously and resume
   at the last unfinished item without losing selection state.
3. Make the existing segregation useful for navigation: pool, weak operational
   class, event state, financial family, payment rail, sender/template group,
   review state, disagreement, and candidate-coverage filters. Weak categories
   remain browsing aids and never become human truth automatically.
4. For `posted`, support exact source selection for Amount, Direction, Account,
   and optional Counterparty on the unchanged SMS. Support quick `none` and
   `abstain` decisions, explicit uncertainty, notes, family, and rail without
   forcing irrelevant fields.
5. Offer analyzer candidates and imported native corrections as clearly labeled
   suggestions in ordinary annotation pools. Never constrain the human label to
   those suggestions.
6. Preserve blind-first review for `protected_test` and
   `later_time_holdout`: hide weak categories, analyzer hints, queue rationale,
   prior labels, and model output until the initial label is submitted and the
   owner explicitly reveals them.
7. Show progress, remaining counts, category/pool coverage, validation failures,
   disagreements, and annotation throughput. Provide safe undo through a new
   append-only revision, never destructive mutation of an earlier label.
8. Keep the service localhost-only, encrypted, ignored by Git, free of remote
   assets/telemetry, and backed up with corpus/run/hash binding. Export only
   explicitly selected, revision-bound labels for adjudication or downstream
   dataset construction.

Acceptance requires a private end-to-end trial covering queue selection, rapid
labeling, legacy-revision readability, direct-extractor projection,
interruption/resume, conflicting revisions, blind reveal, backup, restore, and
consent-bound export. Annotation speed must not weaken grounding, privacy,
provenance, or protected-pool isolation.

Checkpoint 1 (2026-09-24): The focused `canonical-label/2` editor,
source-span highlights, local draft autosave, queue navigation, historical v1
readability, and direct-extractor preview are implemented. Synthetic tests
cover grounding, revision transition, blind reveal, and protected aggregates.

Checkpoint 2 (2026-09-24): Reviewer-specific queue position, filters, and search
now persist in the local workbench database across browser sessions. A completed
resume item redirects to the reviewer's unfinished queue. Synthetic store,
service, and HTTP tests cover reopening, reviewer isolation, and completion.
The production secure store encrypts this state with the workbench database.
Remaining build work: disagreement/candidate/imported-feedback queue views,
one-click imported native suggestions, richer progress/validation/throughput,
and append-only label correction controls. Then run the full private acceptance
trial.

Checkpoint 3 (2026-09-24): Local encrypted export now requires a deliberate
selection of submitted or adjudicated revisions with exact revision hashes.
The UI can select the current or a historical submitted revision; the CLI
requires a private selection manifest. Stale, duplicate, empty, or unconsented
selections fail closed. Synthetic tests cover the selection gate.

Checkpoint 4 (2026-09-24): Reviewer queues now filter by disagreement,
candidate core coverage, and imported native feedback, with protected blind
review exclusion. Imported v2 native correction spans are checked against the
original source and offered as labeled, optional one-click evidence. The
dashboard counts reviewer remaining work, non-protected disagreements,
validation failures, and recent submissions. The editor offers an explicit
append-only correction draft. Synthetic tests cover each boundary.

Checkpoint 5 (2026-09-24): Full WSL gate passes: repository safety, both
Ruff checks, 836 tests, and diff whitespace. The JavaScript syntax check also
passes. Field-level canonical disagreements now enter the adjudication queue.
The in-app browser connection failed at the local sandbox setup, so a visual
click-through has not been verified. This WSL checkout has no private canonical
manifest; no private rows were used for these checkpoints. The remaining
acceptance work is a visual click-through and the private end-to-end trial
using the manifest through the owner's normal local workflow.

Checkpoint 6 (2026-09-25): WSL integration checks uncovered a SQLCipher
row-factory mismatch and an empty optional-filter error in message-list
requests. Both were fixed with synthetic regression tests. The full WSL gate
passes 844 tests. Human acceptance of annotation, restart/resume, blind
reveal, adjudication, restore, and selected export remains. In-app browser
automation remains unavailable because its local sandbox setup fails, so no
visual click-through was observed by the agent. Private corpus metrics and
trial records stay outside version control.

## Workstream 6 — add the Apple Foundation Models evaluation pipeline

The shared repository already has the Android-target local GGUF
direct-extractor evaluator and encrypted native-trace import. That provides the
SLM evaluation baseline used for Android-oriented model work, although an
emulator/device run remains a separate runtime gate. The missing evaluation
implementation is the Apple Foundation Models lane.

1. Preserve `evaluate-extractor` as the reproducible Android-target GPU/GGUF
   semantic evaluator. Do not describe host results as Android device results.
2. Define one hash-bound evaluation-suite manifest in `pF_slm_selection` with
   cohort identity, contract/prompt/schema hashes, protected-pool policy, expected
   canonical labels, and aggregate scoring rules. The Apple and Android-target
   lanes must consume the same declared semantic cohort when comparison is
   intended.
3. Add a macOS/Xcode Foundation Models runner that invokes the exact iOS
   instructions, guided-generation schema, locale check, parser, grounding, and
   validation path. It must run locally against sanitized fixtures or an
   explicitly packaged private suite and never send SMS data to a hosted service.
4. Capture one provenance-bound result per case: suite/case identity, contract
   and prompt hashes, observable system-managed model identifier, macOS/iOS
   version, hardware, locale support, cumulative structured-generation snapshots,
   mapped draft, validation stages, disposition, safe error category, and timing.
   Record decoded tokens, logits, confidence, model-file hash, and token
   throughput as unavailable rather than inventing them.
5. Make the Apple runner resumable and deterministic around completed case IDs.
   Define fresh run, interruption, retry, partial output, duplicate case,
   configuration mismatch, and incompatible result-bundle behavior.
6. Export Apple results through an explicit encrypted local bundle and import them
   into `pF_slm_selection` using the existing protected trace boundary. Raw
   messages, prompts, snapshots, and per-row predictions remain private; checked-in
   reports contain sanitized fixtures or aggregates only.
7. Add shared scoring and report generation for classification, exact fields,
   Unicode-scalar grounding, account resolution, abstention, false/missed
   transactions, routing, review rate, correction burden, failures, latency, and
   recovery. Keep Foundation Models system-runtime facts separate from GGUF facts.
8. Run the pipeline first on macOS with synthetic/sanitized cases, then on the
   supported physical iPhone matrix for device claims. Simulator-only execution
   cannot establish Foundation Models inference parity.

Acceptance requires a repeatable suite package, successful Xcode runner execution,
encrypted import, deterministic aggregate scoring, interruption/resume coverage,
privacy review, and an evidence report that states every unavailable or unverified
runtime fact.

## Delivery order

1. Completed locally: freeze the additive shared routing and partial Review
   contracts; implement Android exception-only routing, Review projection, and
   transparent live output. Preserve the v1-v4 compatibility paths.
2. Next: inspect retained partial fields in Review on Pixel_9; obtain a valid
   complete local-model result to prove automatic persistence and duplicate
   fencing there, and test retry/recovery; then implement and run the Android
   native evaluation lane. Keep JVM and emulator evidence separate.
3. Implement and verify the equivalent iOS behavior on the Mac, respecting the
   Foundation Models observability limits; then implement and run the Apple
   Foundation Models evaluation pipeline above.
4. Exercise the fast personal-corpus annotation flow, including blind review,
   interruption/resume, backup/restore, adjudication, and governed export.
5. Exercise native correction export, adjudication, and component error
   attribution end to end.
6. Run physical Android and iPhone acceptance, compare aggregate evidence, and
   make a separate owner-controlled rollout decision.

Do not enable rollout, publish data, or describe the SMS product as complete until
the [evaluation strategy](SMS_EVALUATION_STRATEGY.md) passes and the owner makes
an explicit release decision.
