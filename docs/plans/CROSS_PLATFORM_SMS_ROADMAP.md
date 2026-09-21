# Cross-platform SMS roadmap

Status: **canonical active roadmap**  
Last reconciled: 2026-09-22

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
confirmation, and native span-selection UI. V4 is deliberately `review_only`, so
even a complete valid result is currently retained for review.

The latest Android emulator trial exposed product gaps that source-level and
automated checks did not settle:

- the previously visible live decoded-token stream was not visible;
- the expected source-SMS span review experience was not visible in the tested
  flow; and
- the tested routing did not match the intended exception-only review policy.

These are open observations, not fixed defects. The first implementation session
must reproduce them on the current branch and trace the actual runtime/navigation
path before changing code. iOS source exists, but its equivalent behavior remains
unverified until the Mac/Xcode lane runs it.

## Workstream 1 — freeze the successor behavior

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

## Workstream 2 — restore processing transparency

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

## Workstream 5 — build both native evaluation lanes in `pF_slm_selection`

The shared repository already has the local GGUF direct-extractor evaluator and
encrypted native-trace import. It does not yet have complete native Android and
iOS scoring pipelines.

1. Keep `evaluate-extractor` as the reproducible GPU/GGUF semantic evaluator. It
   is the closest desktop analogue to Android, not Android runtime proof.
2. Add an Android evaluation lane that packages a hash-bound sanitized/private
   suite, runs the exact app contract/model on emulator or device, exports an
   encrypted native trace bundle, and scores aggregate results in the shared repo.
3. Add an iOS evaluation lane with the same suite identity and scoring contract,
   executed from macOS/Xcode against Foundation Models and imported through the
   same protected boundary.
4. Keep platform-specific facts—model identity, token/snapshot availability,
   latency, memory, OS/device, interruption, and background behavior—rather than
   forcing false runtime parity.
5. Report classification, exact fields, grounding, account resolution,
   duplicates, routing, review rate, correction burden, recovery, and privacy-safe
   latency/resource aggregates on identical declared cohorts.

## Delivery order

1. Reproduce the Android observations and audit both native paths against v4.
2. Freeze the additive successor routing/feedback/evaluation contracts and
   sanitized vectors in the shared repository.
3. Restore Android transparency and prove it on a fresh emulator state.
4. Implement exception-only routing and partial-field review projection on
   Android; then run the Android native evaluation lane.
5. Implement and verify the equivalent iOS behavior on the Mac, respecting the
   Foundation Models observability limits; then run the iOS evaluation lane.
6. Exercise correction export, adjudication, and error attribution end to end.
7. Run physical Android and iPhone acceptance, compare aggregate evidence, and
   make a separate owner-controlled rollout decision.

Do not enable rollout, publish data, or describe the SMS product as complete until
the [evaluation strategy](SMS_EVALUATION_STRATEGY.md) passes and the owner makes
an explicit release decision.
