# PocketFinancer native SMS integration and local workbench plan

Status: proposed implementation plan; saved for the next implementation session.

Prepared: 2026-09-06. This document records the planning audit and the proposed implementation decisions. Creating this document does not implement or enable any processing change.

Intended implementation setup: the user's selected GPT-5.6 model with High reasoning, working through the milestones below. The live user-selected model settings override the dated agent profile in repository instructions. Model choice does not replace tests, physical-device evidence, privacy boundaries, or rollout approval.

## 1. Scope, authority, and handoff

Implement the shared SMS-processing foundation in native Kotlin and Swift, converge every acquisition path on one processing coordinator per app, preserve evidence in durable encrypted review storage, replace mutable edit flags with feedback and revision history, and extend the local workbench to inspect those records.

The repositories are independent:

| Name | Workspace path | Responsibility |
|---|---|---|
| Foundation | /Users/toji/Projects/pocket-financer/pF_slm_selection | Production-intended contracts, deterministic Python reference, sanitized fixtures, local workbench, research |
| Android | /Users/toji/Projects/pocket-financer/pocket-financer-android | Native Kotlin implementation, Android acquisition, encrypted storage, local inference and UI |
| iOS | /Users/toji/Projects/pocket-financer/pocket-financer-ios | Native Swift implementation, App Intent acquisition, protected local storage, platform inference and UI |

Within each repository, paths in this document are relative to that repository unless a full workspace path is shown.

### 1.1 Immutable inspected baseline

The planning session read the applicable instructions and synchronized the checked-out branches with fast-forward-only pulls. These were the resulting inspected commits; they are provenance anchors, never reset targets.

| Repository | Inspected branch | Inspected HEAD | Inspected upstream | Upstream commit at that synchronization | State then |
|---|---|---|---|---|---|
| Foundation | codex/feat-sms-processing-foundation | 2cf8f49c732acf248365d786e7f9e17a809895ae | origin/codex/feat-sms-processing-foundation | 2cf8f49c732acf248365d786e7f9e17a809895ae | Clean; synchronized |
| Android | main | bc837ead283db0ac5eb2b2ced1f79a020c0dd1b7 | origin/main | 006dd4322a04b7e3c11391eb1c82392d8204f8dc | Clean; one known local commit ahead, preserved |
| iOS | main | 7a7a1463f38acac15dedf077a7aafdd111589997 | origin/main | 04f770b235f860080ecd96adad1a5d011f3c2c2c | Clean; one known local commit ahead, preserved |

**Subsequent user update:** the PRs containing the Android and iOS ahead commits have now been merged. Their new remote commits have not been inspected for this saved handoff. The next implementation session must fetch and pull those native changes before editing implementation files.

This plan itself is a new, uncommitted documentation file in the foundation repository. It is not part of the inspected foundation commit. Preserve it; do not discard it to obtain a clean status.

### 1.2 Required next-session synchronization

1. Read every applicable AGENTS.md and repository-specific instruction file completely before any Git operation. Read this plan and the active foundation architecture and contracts.
2. Inspect each repository's branch, HEAD, configured upstream, remotes, status, and ahead/behind relationship. Check remote identity against the intended PocketFinancer repository.
3. Require clean Android and iOS worktrees. Fetch and prune their configured remote references, then compare the updated histories with the known anchors and the merged PR changes.
4. Pull only each currently checked-out native branch from its configured upstream, fast-forward only. Preserve any known local commits already represented by the merged history.
5. If a PR was squash-merged or rebased, the local ahead commit may no longer be an ancestor of remote main. That is divergence, even if file contents look equivalent. Stop and report the exact condition; do not reset, rebase, merge, cherry-pick, switch branches, drop commits, stash, clean, or force-update references automatically.
6. Also stop on unexpected local commits, an unexpected branch or remote, missing upstream, unrelated dirty changes, or a failed fast-forward.
7. Inspect the foundation branch and upstream. Preserve this handoff document and the SMS foundation branch. Do not merge or pull default-branch changes into that branch. If foundation synchronization is needed, it requires its own clean-worktree preflight; do not stash or discard this plan to make it possible.
8. Record the actual post-pull native branch names, HEADs, upstream commits, and status in the implementation handoff. Read the relevant delta from the inspected commits before implementation. Do not silently claim this document already describes unseen merged changes.
9. Once synchronization is safe, create the required short-lived implementation branches under the repository rules. Start from the synchronized native branches and the existing foundation lineage. Do not implement directly on native main.

The initial planning synchronization is complete. This section is a new implementation preflight prompted by the user's later PR-merge update.

### 1.3 Verification already completed during planning

- The current private run was resolved and validated locally through the protected pointer and provenance mechanisms. Source/configuration/artifact integrity, schema consistency, pool boundaries, and workbench revision integrity were checked without exposing private identifiers or row content.
- Repository safety, Ruff, ShellCheck, and whitespace checks passed.
- The lightweight suite had 654 passing tests and one skip in the initial run; three synthetic loopback tests were blocked by sandbox socket permissions and passed when rerun with the appropriate local permission. This is 657 passing tests and one skip across those runs, not a claim that one unrestricted full-suite run was performed.
- No private model inference, protected scoring, training, deployment, or model downloads were performed.
- Native physical-device behavior and the proposed integration have not been verified or implemented.

Do not repeat broad discovery or verification merely to recreate this planning history. In the implementation session, perform checks justified by changed code and repository requirements, and distinguish synthetic, host, simulator, and physical-device evidence.

## 2. What needs changing, and in which repository

The foundation remains the production-intended authority. Android and iOS currently implement older direct-extraction pipelines and need the substantial integration work. This is not a proposal to replace the foundation with the native apps' existing behavior.

### 2.1 Foundation: preserve architecture; make narrow versioned changes

The completed foundation already supplies structural analysis, operation-bound candidates, tri-state disposition, selector actions, selector validation, host reconstruction, independent persistence checks, traces, feedback primitives, canonical labels, and the local workbench.

The planning audit identified these bounded issues or extensions:

| Finding | Repository | Required treatment |
|---|---|---|
| Expected refund wording and authorization/hold wording can currently produce completed-looking candidates | Foundation | Add clause-aware state cues and version the affected analysis/gating behavior |
| A pending financial clause followed by a credential clause can reach credential discard | Foundation | Give uncertain financial evidence retention precedence over unrelated credential-only discard |
| Duplicate selector JSON keys can be accepted by the current parser | Foundation | Reject duplicates and malformed/unhashable discriminator values explicitly |
| Runtime reconstruction/gating lacks sufficient family and timestamp/account-resolution provenance for native policy | Foundation | Add versioned result/configuration/gate contracts; keep canonical truth separate from persistence policy |
| Current trace and feedback contracts are too small for native recovery and field-level corrections | Foundation | Extend via new versions; keep original revisions readable |
| Workbench SQLite permissions do not themselves prove database encryption | Foundation/workbench | Introduce encrypted workbench storage and verify protection for other raw artifacts |

These are proposed, reviewable versioned changes. They do not invalidate the completed foundation architecture, authorize relabeling an existing corpus, or authorize rewriting the current private run.

### 2.2 Android: replace the older direct-extraction path

The inspected Android path builds extraction prompts, can enable thinking from model capability metadata, resolves or creates accounts including a default-account fallback, and writes floating-point transactions. Manual processing duplicates important parts of the pipeline. The edit path mutates the transaction and sets isEdited without preserving an append-only correction history.

Preserve the useful runtime lease coordinator, durable queue admission, scoped observer isolation, bounded activity presentation, batch progress, cancellation settlement, SQLCipher/Keystore storage, and established Compose interaction infrastructure.

### 2.3 iOS: preserve acquisition and storage strengths

The existing App Intent and user-created Messages automation/Shortcut are the authoritative acquisition boundary. ImportTransactionAlertIntent durably enqueues before filtering or inference. Preserve that behavior.

The inspected app uses SwiftData schema V4, protected local files, no CloudKit, durable extraction snapshots, and integer minor-unit transactions. It still uses direct extraction, mutable edits/replacements, process-local claim bookkeeping, and a near-time content duplicate heuristic. It needs shared processing and review semantics, stronger recovery ownership, currency-scale correctness, and immutable history.

The existing iOS SMS path did not contain Android-style thinking machinery. Keep it free of such machinery.

## 3. Frozen architecture and implementation order

Every processing operation follows:

    durable admission
      → immutable operation/configuration snapshot
      → deterministic structural analyzer
      → storage disposition + selector action
      → at most one direct, greedy Candidate Selector pass
      → strict schema parsing and candidate grounding
      → host reconstruction
      → independent account resolution and automatic-persistence gate
      → atomic persistence, durable review, or justified terminal discard

The selector is untrusted. Its only accepted answers are none, abstain, or one posted selection containing operation-bound candidate IDs. It never creates canonical amounts, currencies, accounts, counterparties, dates, or evidence offsets.

The implementation order is mandatory:

1. Freeze the versioned contracts, reason registry, configuration profiles, and sanitized golden vectors.
2. Build native encrypted storage, migrations, state transitions, and claim ownership.
3. Implement native analyzers and coordinators, direct selector adapters, and durable traces.
4. Implement review, correction history, primary-currency settings, and trace UI.
5. Verify parity, failure recovery, migrations, and physical-device behavior in shadow and review-only modes.
6. Consider automatic persistence only after the separate rollout gates and explicit enablement approval.

No Python interpreter or Python reference execution belongs in either app.

## 4. Shared contract freeze

### 4.1 Version manifest

Introduce a release manifest that binds schemas, analyzer behavior, Unicode normalization data, currency/locale profiles, reason-code registry, selector validation profile, prompt, decoding policy, persistence policy, and golden-fixture hashes.

| Contract | Planned version | Decision |
|---|---|---|
| sms-analysis | /2 | Version clause-state, family, timestamp, and configuration-hash extensions |
| grounded-candidate-selector-input | /1 | Preserve candidate-selector semantics; do not add model-generated semantic fields |
| grounded-candidate-selector output | /1 | Preserve none/abstain/one-posted-ID-selection wire shape |
| selector-validation-profile | /2 | Strict duplicate-key, type, foreign-ID, size, and grounding behavior |
| processing-config | /1 | New complete immutable native operation configuration |
| processing-result | /2 | Reconstructed semantic truth plus separately represented policy outcomes |
| processing-trace | /2 | Native stages, durable sequencing, runtime provenance, recovery history |
| user-feedback | /2 | Append-only user actions and field-level grounding classifications |
| native-trace-bundle | /1 | Explicit-consent local interchange with integrity and provenance |
| canonical-label | /1 | Preserve the richer canonical human annotation schema |

Keep the existing /1 schema files and readers available. Add versioned files for the new schemas rather than changing historical /1 meaning in place. Additive host analysis metadata must not accidentally widen the selector's permitted output language.

If the current selector input schema cannot express a required host-side addition without changing its declared shape, keep that addition outside selector input. Only introduce selector-input /2 if a concrete fixture proves it necessary and the contract decision is reviewed before either native port.

Native apps vendor the released schemas, immutable profiles, and sanitized fixture bundle with its manifest. CI compares manifest hashes and fixture results. Apps do not retrieve contracts or configuration from a remote service at runtime.

### 4.2 Immutable configuration snapshot

Each operation captures:

- Operation UUID, parent operation UUID when retrying, source reference, trigger, and creation time.
- Contract manifest version and hashes; analyzer implementation/profile version; Unicode behavior version.
- Primary currency, enabled locale/currency profiles, and exact profile asset hashes.
- Source timestamp availability, claimed timestamp origin, admission time, timezone context, and timestamp policy.
- Selector eligibility decision, model identifier, model file hash/version when available, runtime version, OS/model cohort when applicable.
- Prompt version/hash, validation profile, persistence policy, and rollout mode.
- Generation mode DIRECT_NON_THINKING, greedy decoding, answer limit, output byte limit, and timeout policy.

Persist the snapshot before analysis. Changes to settings, runtime configuration, or model selection do not mutate an existing operation. A retry preserving its context clones the previous configuration into a new linked operation; “retry with current settings” is an explicit separate action.

Hash the canonical serialized snapshot and all behavior-affecting assets. Do not rely on profile names alone. If a platform cannot pin a system model revision, record the available OS/runtime facts and the limitation; do not invent a stable model-file identity.

### 4.3 Normalization, evidence, clauses, and IDs

- Freeze the reference's per-code-point NFKC/case-fold and whitespace mapping behavior with Unicode vectors. Do not substitute a platform's whole-string normalization without parity proof.
- Keep the original source immutable. Every evidence span has original Unicode code-point boundaries and UTF-8 byte boundaries. UI highlighting converts those boundaries safely to Kotlin/Swift string indices.
- Clause segmentation is deterministic. Preserve current boundary behavior where valid; version refinements for completed, pending, failed, authorization, expectation, and security clauses.
- Freeze clause ordering and IDs, cue ordering and IDs, candidate ordering, normalization, JSON canonicalization, and hashing in byte-level fixtures.
- Preserve the existing candidate ID recipe within its version: kind prefix plus the truncated digest over analysis identity, kind, original span or explicit absence, and canonical candidate value. Preserve the existing analysis-ID ingredients within /1; /2 binds the expanded configuration hash.
- Include vectors for non-ASCII values, escaped characters, empty/absent candidates, repeated amounts, and different operation IDs over identical text.
- Reject collisions or duplicate candidate IDs; never overwrite a candidate silently.
- IDs must bind to this operation and analysis. A candidate from another operation is invalid even if its semantic value matches.
- Preserve explicit-absence account and counterparty candidates. Absence has no fabricated source span.
- Optional semantic fields distinguish present, explicitly absent, and unknown. Unknown is not zero, an empty string, a default account, or an invented candidate.

### 4.4 Money, currency, and timestamps

New native extraction records use signed 64-bit integer minor units plus ISO currency and the frozen currency scale. The amount magnitude is positive; direction carries debit/credit semantics. Reject overflow and unsupported precision before persistence.

Initial supported currencies are AED, AUD, CAD, CHF, EUR, GBP, INR, JPY, SGD, and USD. JPY uses scale zero; the other listed profiles use scale two. Expansion is a versioned profile change.

JSON tooling must preserve full integer precision; the workbench must not round Int64 values through JavaScript Number. Use lossless decoding and string-backed display/editing where needed while preserving the declared schema representation.

Currency provenance distinguishes explicit source code, approved unambiguous source symbol/marker, and user-primary default. Explicit source evidence overrides the primary currency. Globally ambiguous symbols such as dollar or yen symbols require an enabled disambiguating profile or sufficient source evidence; otherwise retain for review.

Timestamp provenance distinguishes source-supplied transaction time, acquisition-supplied message time, user/Shortcut-supplied time, admission time, and user-corrected time. Do not call a Shortcut parameter an authenticated Messages timestamp. Missing source time stays missing even though admission time is always recorded.

The automatic timestamp policy accepts only the versioned, sufficiently grounded timestamp cases in the fixture bundle. A fallback to admission time is visible and defaults to review; it must not silently pretend to be the transaction time.

### 4.5 Reason codes and persistence-gate results

Freeze a namespaced registry with stable meanings, severity, stage, and allowed outcome effects. Existing codes retain their historical meaning. New codes cover expected refunds, authorization/hold, mixed credentials, missing coverage, runtime ineligibility, unknown IDs, invalid output, missing timestamp provenance, account ambiguity, and policy restrictions.

Consumers must not infer disposition by matching human-readable text. An unknown contract version or unknown safety-relevant code fails closed to retained review.

The gate returns a typed result with all evaluated checks and a primary reason:

| Gate result | Meaning |
|---|---|
| eligible | All automatic-persistence checks passed |
| review_required | Truth may be useful, but coverage, account, timestamp, runtime, or product policy prevents automation |
| not_posted | Canonical semantics identify no completed movement; no ledger insertion |
| multiple_events | More than one completed event requires explicit review |
| blocked_by_mode | Shadow or review-only rollout mode prevents automatic insertion |
| invalid_operation | Contract, ownership, trace, configuration, or grounding integrity failed |

The final automatic gate requires: one completed grounded event; valid positive exact money/currency; acceptable timestamp provenance; exactly one present, grounded and uniquely resolved owned account; supported family; invoke disposition; normal selector action; valid direct selector output; completed reconstruction; current claim ownership; no unresolved conflict; and an enabled rollout policy.

Assistive output is never eligible for automatic persistence.

### 4.6 Financial truth versus product policy

| Source meaning | Canonical truth | MVP behavior |
|---|---|---|
| Completed debit | Posted debit | May persist only after every gate |
| Completed credit | Posted credit | May persist only after every gate |
| Explicitly completed refund credit | Posted credit, refund family | May persist only after every gate |
| Refund initiated, expected, or promised within a period | Not posted | No ledger transaction; retain when financial uncertainty or user review remains |
| Completed wallet movement | Posted, wallet family | Explicit unsupported-auto-persistence result; review |
| Failed, declined, requested, pending, due, held, or merely authorized movement | Not posted | No ledger transaction; discard only when deterministic policy is terminally certain |
| Several completed events | Multiple events | Review; no automatic split into ledger records |
| Unsupported financial family | Preserve truthful family/event semantics | Explicit policy rejection or review, never relabel as non-financial |
| Standalone high-confidence credential message | Non-financial credential-only | May erase source under terminal-discard policy |
| Completed transaction plus security clause, including “without OTP” | Preserve completed financial clause | Do not reject merely because credential terminology appears elsewhere |

Expected-refund reminders, scheduling, obligations, and cross-message reconciliation are excluded. They need their own future schema, candidates, and state machine.

### 4.7 Strict selector contract

Accept exactly one JSON document and exactly the declared fields/types. Reject duplicate keys, additional properties, leading/trailing prose, invalid discriminators, foreign/unknown IDs, wrong candidate kinds, incompatible clause selections, malformed numbers, and truncated output.

Do not repair malformed output, extract a JSON-looking substring, use a second model pass, or convert model-provided semantic values into candidates. none and abstain are valid selector responses but do not independently authorize deleting admitted financial evidence.

Record invalidity precisely and retain the operation for review. Host reconstruction consumes only validated candidates and deterministic metadata.

## 5. One native coordinator per app

### 5.1 Application-facing interfaces

Kotlin interface, implemented in the pipeline module:

    interface SmsProcessingCoordinator {
        suspend fun process(
            source: AdmittedMessageRef,
            operation: SmsOperationSnapshot,
            observer: SmsProcessingObserver = SmsProcessingObserver.None
        ): SmsProcessingOutcome

        suspend fun requestStop(operationId: OperationId): StopReceipt
        suspend fun resolveReview(command: ReviewCommand): ReviewReceipt
    }

Swift interface, with Sendable value types and storage isolated behind an actor:

    protocol SmsProcessingCoordinating: Sendable {
        func process(
            source: AdmittedMessageRef,
            operation: SmsOperationSnapshot,
            observer: any SmsProcessingObserver
        ) async -> SmsProcessingOutcome

        func requestStop(operationID: OperationID) async -> StopReceipt
        func resolveReview(_ command: ReviewCommand) async -> ReviewReceipt
    }

The admitted reference contains a durable source ID and admission receipt, never an unpersisted raw body. The snapshot contains the operation ID and the immutable configuration. A small snapshot factory persists configuration atomically with operation creation before process is called.

Repeated process calls with the same operation ID resume or return its settled receipt. A different source/configuration for that ID is an integrity failure, never an implicit overwrite.

Typed outcomes:

    TerminallyDiscarded(operationId, reason, deletionReceipt)
    RetainedForReview(operationId, reviewCaseId, reasons)
    Persisted(operationId, transactionIds, alreadyCommitted)
    RetryableFailure(operationId, reviewCaseId, reason, retryCondition)
    Stopped(operationId, reviewCaseId, reason)

Review commands are Confirm, Correct, Reject, ResolveMultipleEvents, SaveDraft, and Retry. Every command includes an action UUID and expected review/projection revision. Correct carries per-field provenance and the selected account action. Retry explicitly chooses original or current configuration.

### 5.2 Ownership boundaries

| Component | Owns |
|---|---|
| Acquisition/admission service | Input limits, original metadata, durable source insertion, authoritative delivery identity |
| Snapshot factory | Immutable configuration, operation identity, parent/retry linkage |
| Processing coordinator | Analyzer → triage → selector → validation → reconstruction → gate orchestration and typed settlement |
| Analyzer | Deterministic clauses, cues, candidates, provenance and recall limitations |
| Selector adapter | One direct request, exact available output, runtime facts; no semantic persistence decisions |
| Runtime lease/execution gate | Model lifetime, serialization, unload/reload, cancellation ownership |
| Processing store | Claims, transitions, idempotency, encrypted records and atomic settlement |
| Account resolver | Grounded owned-account matching with zero/one/many result; no default fallback |
| Review command handler | Append-only feedback/revisions and explicit user-authorized ledger changes in one transaction |
| UI/observer | Truthful presentation; no independent parsing, account creation, or ledger writes |

Observers receive immutable stage events and bounded live selector updates. Observer exceptions or a disappeared screen do not alter processing or suppress durable settlement. Raw detail remains inside the protected app surface, never ordinary logs.

### 5.3 Processing state machine

Persist state, transition sequence, owner token, generation/fencing token, and expected previous state with each durable transition.

| From | Trigger | To | Atomic durable effect |
|---|---|---|---|
| No source | Accepted acquisition | admitted | Insert immutable source and admission receipt |
| admitted | Configuration unavailable | awaiting_configuration | Preserve source; record required user action |
| admitted / awaiting_configuration | Snapshot created | ready | Insert immutable operation/configuration |
| ready | Claim succeeds | claimed | Acquire owner token, fencing generation and lease |
| claimed | Analysis completes | analyzed | Store immutable analysis, clauses, cues and candidates |
| analyzed | Triage completes | triaged | Store disposition, selector action and reasons |
| triaged | Certain terminal discard | discarded | Settle receipt; erase permitted sensitive payload atomically |
| triaged | Selector skipped but evidence unresolved | retain_review | Create/update durable review case with explicit skip reason |
| triaged | Eligible normal/assistive pass begins | selector_running | Insert one attempt and runtime/request provenance |
| selector_running | Output available | selector_recorded | Store exact available terminal output and completion status |
| selector_recorded | Strict validation passes | validated | Store parsed selection and grounding report |
| validated | Host reconstruction succeeds | reconstructed | Store semantic result and provenance |
| reconstructed | All automatic gates pass | persisted | Resolve account, insert ledger/revision and settle trace/review together |
| Any unsettled processing state | Invalid/uncertain/gate rejection | retain_review | Preserve all evidence; record reasons and review draft |
| Any unsettled processing state | Transient operational failure | retry_wait | Preserve source and trace; create visible review case and retry condition |
| Any unsettled processing state | Stop, timeout or lost ownership | interrupted | Fence writers; preserve source, partial trace and recovery action |
| retry_wait / interrupted / retain_review | Retry authorized or due | new linked ready operation | Preserve old operation; create a new immutable attempt context |
| Any settled state | Duplicate process call | Same state | Return existing settlement receipt |

An assistive pass can prefill review, including a posted selection. It always settles to review. A normal pass yielding none, abstain, invalid output, missing coverage, failed reconstruction or gate rejection also retains financial evidence.

### 5.4 Review state machine

Review state is durable and distinct from processing state.

| From | Action | To | Required effect |
|---|---|---|---|
| No case | retain/retry/interruption requiring attention | open | Link complete source and operation history |
| open / draft | Save partial correction | draft | Persist encrypted draft revision; no ledger change |
| open / draft | Confirm grounded valid proposal | confirmed | Append confirmation, transaction revision and ledger projection atomically |
| open / draft | Correct values | corrected | Append field classifications, correction revision and ledger projection atomically |
| open / draft | Reject | rejected | Append rejection; suppress retry/recreation; preserve evidence under retention policy |
| open / draft | Resolve multiple events | confirmed / corrected | Create explicitly approved event identities and all ledger rows atomically |
| open / draft | Retry | waiting_retry | Preserve draft/history; link new operation |
| waiting_retry | Processing settles | open / confirmed | Apply result only if the current review revision still permits it |
| confirmed / corrected | Later edit/rejection | corrected / rejected | Append a new revision; update current projection without rewriting history |

Optimistic revision checks prevent two screens or a late model result from overwriting a user's action. Repeated action UUIDs return the previous receipt. A rejected source is not resurrected by background retry.

### 5.5 Claiming, cancellation, and recovery

- Use a two-minute renewable claim lease and a 15-second heartbeat during active work. Lease expiry alone is insufficient to permit a second live writer; enforce owner generation and native process/execution ownership.
- Android uses Room compare-and-set transitions and the existing runtime lease coordinator.
- iOS uses a dedicated processing-store actor, fresh contexts for bounded transactions, and a short interprocess protected-file lock around claim/settlement operations when app and intent can overlap. Never hold that lock or a database transaction during inference.
- Every settlement rechecks ownership, source deletion epoch, review revision, configuration hash and idempotency key.
- A 60-second parser deadline revokes permission to settle. An uncooperative model task keeps the runtime lane occupied until it actually ends; timing out an await must not start a second overlapping generation.
- Stop acknowledges either a committed receipt or a safely fenced, retained operation. Use only a short cancellation-shielded atomic settlement section.
- If exact selector output is already durably recorded, recovery may resume host validation/reconstruction without rerunning the model.
- If the process died during generation, retain the interrupted attempt. A retry gets a new linked operation; do not claim a second pass occurred within the original operation.
- Cap automatic transient retries at three. Model unavailability waits for a relevant availability/foreground event; it must not burn retries in a tight loop. Exhaustion goes to review, never deletion.
- Process restart, device restart, app upgrade, and model unload/reload preserve pending evidence. Foreground draining continues bounded batches until no eligible work remains or the app loses its execution opportunity.

## 6. Storage model, accounts, feedback, and migrations

### 6.1 Logical entities

Use native entities with equivalent serialized contracts:

| Entity | Contents and mutability |
|---|---|
| AdmittedSource | Immutable original body and original metadata, admission timestamp, source identity/digest, retention state |
| SourceMetadataEvent | Append-only later enrichment or duplicate-delivery evidence; never overwrite original metadata |
| ProcessingOperation | Immutable source/configuration linkage plus transactional state, claim and settlement receipt |
| ProcessingAnalysis | Immutable versioned analysis, clauses, cues, candidates and source/config hashes |
| SelectorAttempt | Immutable request/runtime provenance, exact available raw output, completion/limit status and validated output |
| ProcessingTraceEvent | Append-only sequence, stage/status, reason codes, truthful measurements, previous-event hash |
| ReconstructedResult | Immutable canonical semantic result, evidence and provenance |
| PersistenceDecision | Gate checks, account resolution, policy mode and final reason/result |
| ReviewCase / ReviewDraft | Durable lifecycle, optimistic revision, encrypted saved draft and recovery action |
| UserFeedbackEvent | Append-only action, actor class, operation/transaction revision references and per-field classifications |
| TransactionRevision | Append-only original extraction and subsequent changes, provenance and feedback reference |
| CurrentTransaction | Materialized active ledger projection for ordinary screens and queries |
| AccountAlias | Explicitly confirmed owned-account identifiers and matching scope |
| LegacyTransactionSnapshot | Immutable migration-time copy with honest legacy provenance and unknown original history |
| TraceImportReceipt | Bundle manifest/integrity, consent/import provenance and idempotent import result |

Encrypted payloads include original messages, raw selector output, evidence spans, review drafts, correction text, account details and feedback. They must never be copied into generic diagnostic strings.

### 6.2 Idempotency and atomicity

- Source identity: authoritative connector/provider identity where available. Content fingerprint is supporting evidence, not a universal unique key.
- Operation: operation UUID.
- Trace: operation UUID plus monotonically increasing sequence.
- Review action: action UUID plus expected review/projection revision.
- Ledger insertion: source UUID plus stable event UUID. Automatic single-event processing uses the same event identity across retries.
- Multiple-event confirmation: user-approved event UUIDs are persisted before/reused across repeated submission.
- Import: transfer UUID plus manifest hash.

Account resolution, any explicitly authorized account creation, ledger insertion/update, transaction revision, feedback event, final trace, and review settlement belong to one native database transaction/save. A failure rolls back all of them.

Automatic SMS processing never creates an account opportunistically. User-approved account creation in review is allowed only inside the same atomic settlement.

### 6.3 Account resolution

Return Missing, Unresolved, Ambiguous(matches), or UniquelyResolved(accountId, matchedAlias, provenance).

Automatic persistence requires a present, source-grounded account candidate matching exactly one confirmed owned account under the versioned matching policy. An identifier belonging to the counterparty is not the user's account.

Remove default-account fallback, first-match ambiguity resolution, and account creation from automatic paths. Do not infer ownership solely from a sender label or a common last-four suffix. Existing accounts can continue serving legacy transactions; ambiguous aliases require user confirmation before becoming eligible.

Move lazy account consolidation out of ledger reads. Any future consolidation must be an explicit atomic operation preserving references, aliases and history. It must not run as a side effect while an SMS operation is settling.

### 6.4 Feedback and revisions

Replace isEdited as the authority with append-only history. Keep the legacy field only for old-reader compatibility during migration.

Each correction records the field, previous/current revision reference, new value and one of:

1. selected_existing_candidate;
2. changed_interpretation_among_candidates;
3. supplied_source_supported_candidate_miss;
4. supplied_manual_ungrounded_value.

Other action types are confirmed_unchanged and rejected. A correction containing several fields may contain several classifications.

A source-supported candidate miss stores the user's proposed evidence and a candidate-generation-miss record. It does not fabricate an analyzer candidate retroactively. A manually supplied value has explicit ungrounded provenance.

Native user feedback is not canonical human dataset annotation. A canonical label reference is optional for native feedback /2; canonical-label validation remains required when an authorized annotation workflow produces canonical truth. Candidate Selector targets may be derived only from eligible grounded candidates under a separately authorized export/curation process.

### 6.5 Android Room/SQLCipher migration

The inspected database is Room version 5. Plan version 6 as an additive, transactional migration:

1. Preserve all historical schema exports and existing 1→2→3→4→5 migrations.
2. Add the new operation, analysis, selector, trace, result, gate, review, feedback, revision, alias and source-metadata tables.
3. Add nullable exact-money, currency/scale/provenance, source/event, current-revision and projection-state columns to transactions.
4. Insert migration-time legacy snapshots without pretending to recover the original pre-edit extraction.
5. Preserve transaction IDs, account IDs, raw evidence, legacy Double amounts, dates, and existing source identity/fingerprint information.
6. Move queued candidates into the durable source/operation model without losing claim/retry/source metadata. Coordinate old worker shutdown with schema activation so no old writer can settle into the new model.
7. Replace uniqueness assumptions that permit only one transaction per source with source-plus-event uniqueness. Change source queries to return all associated events.
8. Preserve provider IDs as authoritative where present; do not collapse historical transactions by a nonunique content fingerprint.
9. Keep SQLCipher and the current Keystore-backed key behavior. Preserve fail-closed key/database handling and avoid destructive migration fallback.
10. Enable new processing only after migration invariants pass. Existing ledger browsing remains available if currency onboarding is still pending.

Legacy floating-point amounts remain legacy values. Do not multiply and round them into supposedly historically exact money. New records use exact fields; old records expose an explicit legacy precision status. Editing an old record creates a new exact user-confirmed revision while preserving its legacy snapshot.

Queries, summaries, and charts group by ISO currency. Do not sum different currencies as one total without a separately designed conversion policy.

### 6.6 iOS SwiftData migration

The inspected store is schema V4. Plan V5 using SchemaMigrationPlan:

1. Freeze historical schema definitions so later model edits cannot accidentally change V1–V4 migration interpretation.
2. Add the new models and optional links needed to migrate existing stores safely.
3. Preserve InboxAlert evidence, transactions, accounts, extraction runs, deterministic filter runs, and generation snapshots.
4. Preserve existing Int64 amounts and currency strings. Fix formatting/arithmetic to use currency scale instead of a universal divide-by-100 assumption.
5. Create legacy snapshots/revisions with explicit limits: current stored values are known; lost original edits are not recoverable.
6. Map pending/needs-review alerts into durable review/recovery state. Convert abandoned processing records to recoverable interruption; retain their original attempts.
7. Stop replacement-on-retry from mutating an original extraction or overwriting a user revision. Update only through a new revision and atomic projection settlement.
8. Perform any backfill too complex for the schema migration as resumable, idempotent batches; block new processing until required invariants are complete.
9. Keep each logical settlement within one save transaction with rollback on error and ownership revalidation.
10. Preserve file protection, backup exclusions, local-only configuration and fail-closed startup. Do not delete or recreate a store merely because a version or key is unfamiliar.

The iOS protection baseline is Data Protection with completeUntilFirstUserAuthentication and protected SQLite/WAL/SHM files. This allows the existing after-first-unlock background use case; it does not mean the database is available before the first unlock after restart.

Do not claim separate application-level SQLCipher encryption exists on iOS. If an additional encryption layer is later required beyond the current protected local store, treat it as a separately reviewed storage change.

### 6.7 Upgrade, downgrade, and rollback

Before release, prove every supported old schema can migrate without losing saved transactions or evidence. Preserve encrypted recovery material according to platform policy.

An older binary encountering a newer schema must fail safely and preserve the store. Do not promise reverse migrations. Operational rollback uses a forward-compatible build or policy that retains/reviews work and disables automatic persistence while preserving all new tables and user transactions.

## 7. Direct, non-thinking runtime policy

### 7.1 Android

- Remove thinking fields/callbacks from the SMS request/response types, capability-driven enablement, thinking-token budgets, thinking deltas/buffers, two-phase prompts, and all thinking UI/toggles.
- Explicitly request DIRECT_NON_THINKING for every Candidate Selector operation. A zero thinking budget is not proof of direct generation.
- The adapter must use a verified direct chat-template path and greedy constrained selector JSON generation from the first generated token.
- Baseline answer limit: 512 generated tokens; raw-output capture limit: 16 KiB UTF-8; operation parser deadline: 60 seconds. Freeze these in the configuration profile and change them only through versioned configuration.
- Grammar/runtime capability failure is a recorded ineligibility or failure. Do not silently fall back to unconstrained direct extraction.
- Models/templates requiring mandatory reasoning before their answer are ineligible. Capability metadata may describe a model, but cannot enable reasoning.
- If a mode violation is detected, stop and retain review with a mode-violation reason; do not collect/persist scratch text or salvage a final answer from it.
- The inspected scope did not establish an unrelated non-SMS consumer that requires thinking machinery. Remove it. If later code proves such a consumer exists, document the exact unreachable-from-SMS boundary and a final retirement milestone; do not retain it for hypothetical SMS experiments.
- Preserve runtime leases, model ownership, unload/reload and safe cancellation. Do not persist SMS-derived KV/session caches; reusable caches must contain only verified source-free static material.

### 7.2 iOS

- Every application request declares DIRECT_NON_THINKING. Map it to the platform's ordinary greedy generation path; do not invent a reasoning flag the SDK does not expose.
- Use a guided/dynamic schema for the exact selector alternatives and the current operation's candidate IDs where supported.
- Validate the actual returned representation against the selector contract. If a platform wrapper cannot produce a conforming representation, retain with unsupported-runtime/contract reason; do not silently reinterpret it as valid output.
- Preserve the platform's actual cumulative structured snapshots. Label them as platform-exposed structured JSON snapshots, not token-by-token bytes.
- Record exact final platform-exposed selector output. Mark unavailable token counts, model file identifiers, throughput or memory metrics as unavailable.
- The inspected iOS 26.0 SDK does not provide every metric introduced in later SDKs. Availability-gate newer APIs and verify the relevant OS/model cohort before using them.
- Do not request, infer, fabricate or display hidden chain-of-thought. Platform-internal computation is not an observable reasoning trace.
- Keep inference local through the intended on-device platform API; no hosted fallback or Private Cloud Compute workflow is introduced.

### 7.3 Bounded output and truthful traces

Live output is bounded in memory. Persist the exact terminal output available within the configured limit. If the runtime exceeds the limit, cancel, record incomplete/truncated status and the limit reason, and retain review; never label a truncated buffer “complete exact output.”

Invalid selector output may be useful evidence, but reasoning scratch text is not a supported trace field. Production trace schemas have no thinking/scratch channel.

Future thinking comparisons belong exclusively to separately authorized, versioned research inside the foundation repository.

## 8. Native call sites and file-by-file work

### 8.1 Android existing files

| File | Required change |
|---|---|
| data/src/main/java/com/pocketfinancer/data/db/AppDatabase.kt | Room V6 migration, new DAOs/entities, atomic store integration |
| data/src/main/java/com/pocketfinancer/data/db/entity/TransactionEntity.kt | Exact-money/provenance/revision fields and current projection semantics |
| data/src/main/java/com/pocketfinancer/data/db/dao/TransactionDao.kt | Source-plus-event idempotency, revision-aware projection writes, currency-aware queries |
| data/src/main/java/com/pocketfinancer/data/repository/TransactionRepository.kt | Replace mutable updateTransaction authority with atomic revision/feedback commands; preserve legacy reads |
| data/src/main/java/com/pocketfinancer/data/repository/AccountRepository.kt | Explicit zero/one/many resolution, confirmed aliases, remove automatic fallback/creation and lazy consolidation side effects |
| data/src/main/java/com/pocketfinancer/data/repository/SmsIngestionRepository.kt | Durable source lifecycle, claims, review retention and recovery; no deletion on uncertainty or exhausted retries |
| sms/src/main/java/com/pocketfinancer/sms/SmsReceiver.kt | Preserve durable admission before processing; pass admission identity to scheduling |
| pipeline/src/main/java/com/pocketfinancer/pipeline/PipelineService.kt | Convert to a thin compatibility facade over the coordinator, then remove duplicated orchestration |
| pipeline/src/main/java/com/pocketfinancer/pipeline/SmsParserWorker.kt | Use coordinator; remove independent prefilter/prompt/account/save path; handle typed outcomes and claims |
| pipeline/src/main/java/com/pocketfinancer/pipeline/AutomaticProcessingPreferences.kt | Explicit rollout mode and future-operation snapshot inputs; disabling automation preserves admitted evidence |
| pipeline/src/main/java/com/pocketfinancer/pipeline/AutomaticSmsProcessingActivity.kt | Map coordinator events to bounded live activity; preserve stop and committed receipts |
| inference/src/main/java/com/pocketfinancer/inference/SlmRuntime.kt | Direct-only Candidate Selector request/outcome types; remove SMS thinking channels |
| inference/src/main/java/com/pocketfinancer/inference/SlmRuntimeBindings.kt | Wire direct adapter and truthful runtime metadata |
| inference/src/main/java/com/pocketfinancer/inference/LlamaEngine.kt | One greedy direct constrained pass; remove thinking phase and callbacks |
| inference/src/main/java/com/pocketfinancer/inference/SlmRuntimeCoordinator.kt | Preserve leases; verify timeout/unload/reload fencing and source-free cache policy |
| hardware/src/main/java/com/pocketfinancer/hardware/SlmSelector.kt | Select only verified direct-eligible models; capability metadata cannot enable reasoning |
| app/src/main/java/com/pocketfinancer/ui/home/HomeSyncManager.kt | Route manual processing through coordinator; remove duplicate extraction, account and insertion code |
| app/src/main/java/com/pocketfinancer/ui/onboarding/OnboardingService.kt | Route historical processing through coordinator and immutable batch operation snapshots |
| app/src/main/java/com/pocketfinancer/ui/onboarding/HistoricalSmsProcessingActivity.kt | Preserve batch progress, stop/retry, and settlement; consume new events |
| app/src/main/java/com/pocketfinancer/ui/home/HomeViewModel.kt | Read stored trace; stop reconstructing historical prompts/output from current settings |
| app/src/main/java/com/pocketfinancer/ui/transactions/TransactionsViewModel.kt | Review/revision commands replace direct edits/account creation; load exact stored provenance |
| app/src/main/java/com/pocketfinancer/ui/settings/SettingsViewModel.kt | Coordinator-based diagnostics with no automatic persistence; remove thinking buffers/toggles |
| app/src/main/java/com/pocketfinancer/ui/smsprocessing/SmsProcessingModels.kt | Typed decision stages and durable review outcomes |
| app/src/main/java/com/pocketfinancer/ui/smsprocessing/SmsTelemetryModels.kt | Truthful metric availability and stored/live trace distinctions |
| app/src/main/java/com/pocketfinancer/ui/smsprocessing/SmsTelemetryViewer.kt | Replace Thinking Output with Decision Trace and bounded JSON inspection |
| app/src/main/java/com/pocketfinancer/ui/smsprocessing/SmsPipelineActivityCard.kt | Accessible stage timeline, loading, stop, retry and review actions |
| app/src/main/java/com/pocketfinancer/ui/home/HomeScreen.kt | Durable review entry, automatic/historical status and saved trace navigation |
| app/src/main/java/com/pocketfinancer/ui/transactions/TransactionsScreen.kt | Current projection plus revision/review actions and honest legacy states |
| app/src/main/java/com/pocketfinancer/ui/settings/SettingsScreen.kt | Primary currency/profiles and truthful diagnostic trace; no reasoning toggle |
| app/src/main/java/com/pocketfinancer/ui/onboarding/OnboardingScreen.kt | Explicit primary-currency choice before new processing |
| app/src/main/java/com/pocketfinancer/ui/navigation/Screen.kt | Review inbox/detail and trace destinations |
| app/src/main/java/com/pocketfinancer/ui/PocketFinancerRoot.kt | Recovery scheduling, navigation, privacy and processing ownership integration |
| app/src/main/java/com/pocketfinancer/ui/settings/LocalFinancialEraseBoundary.kt | Invalidate new operation claims/writers and erase protected new entities |
| app/src/main/java/com/pocketfinancer/ui/settings/LocalFinancialEraseRecovery.kt | Resume interrupted erase safely across new schema |
| data/schemas/com.pocketfinancer.data.db.AppDatabase/6.json | New exported schema; keep versions 1–5 |
| Module build.gradle.kts files and native inference binding source, if required | Dependencies/resources and direct-template capability changes only where actual integration requires them |

### 8.2 Android new files

Under pipeline/src/main/java/com/pocketfinancer/pipeline/sms/:

- SmsProcessingCoordinator.kt
- DefaultSmsProcessingCoordinator.kt
- SmsProcessingContracts.kt
- SmsOperationSnapshotFactory.kt
- SmsProcessingObserver.kt
- SmsProcessingRecovery.kt
- StructuralSmsAnalyzer.kt
- StructuralClauseSegmenter.kt
- CurrencyProfileRegistry.kt
- GroundedSelectorValidator.kt
- SemanticReconstructor.kt
- AutomaticPersistenceGate.kt

Under inference/src/main/java/com/pocketfinancer/inference/:

- DirectCandidateSelector.kt
- CandidateSelectorRuntimeProfile.kt

Under data/src/main/java/com/pocketfinancer/data/:

- db/entity/SmsProcessingEntities.kt
- db/entity/TransactionRevisionEntity.kt
- db/dao/SmsProcessingDao.kt
- db/dao/TransactionRevisionDao.kt
- repository/SmsProcessingStore.kt
- repository/SmsReviewRepository.kt
- repository/GroundedAccountResolver.kt
- repository/ProcessingConfigurationRepository.kt

Under app/src/main/java/com/pocketfinancer/ui/:

- review/ReviewInboxScreen.kt
- review/ReviewDetailScreen.kt
- review/ReviewViewModel.kt
- smsprocessing/DecisionTraceTimeline.kt
- smsprocessing/DecisionTraceViewModel.kt

Add the pinned shared contract/profile/fixture resources to the responsible modules and tests. Keep deterministic implementation independent of Compose and Android acquisition APIs.

### 8.3 iOS existing files

| File | Required change |
|---|---|
| PocketFinancer/Intents/ImportTransactionAlertIntent.swift | Preserve enqueue-only background acquisition and truthful admission receipts |
| PocketFinancer/Services/AlertIngestionService.swift | Admission and draining facade; route pending, needs-review retry and foreground recovery through coordinator |
| PocketFinancer/Services/FoundationModelTransactionParser.swift | Replace direct semantic extraction with direct Candidate Selector adapter |
| PocketFinancer/Services/FoundationModelExtractionContract.swift | Version selector instructions/schema and remove legacy extraction semantics from the new path |
| PocketFinancer/Services/FoundationModelExecutionGate.swift | Preserve single execution ownership; integrate deadlines, fencing and actual task completion |
| PocketFinancer/Data/Models.swift | Extend InboxAlert/Transaction/Account relationships without overwriting source or extraction history |
| PocketFinancer/Data/PocketFinancerSchema.swift | Freeze V1–V4 and add V5 migration plan |
| PocketFinancer/Data/AppDatabase.swift | Register models, protected store actor integration and fail-closed migration/backfill startup |
| PocketFinancer/Data/SecureStoreFilePolicy.swift | Apply protection/backup policy to every new database/supporting artifact |
| PocketFinancer/Data/ExtractionRun.swift | Preserve legacy runs; link new operation/result/trace records |
| PocketFinancer/Data/StructuredGenerationSnapshot.swift | Preserve actual platform snapshots and completion/availability metadata |
| PocketFinancer/Data/DeterministicFilterRun.swift | Preserve legacy history; new analyzer/triage trace replaces body-wide filtering |
| PocketFinancer/Features/Transactions/TransactionsView.swift | Durable review inbox entry, state/result display and currency-aware projection |
| PocketFinancer/Features/Transactions/TransactionDetailView.swift | Replace direct mutable save with feedback/revision command; rejection prevents retry resurrection |
| PocketFinancer/Features/Transactions/AlertProcessingDetailView.swift | Decision timeline, actual selector snapshots, review/correction, coverage misses and recovery actions |
| PocketFinancer/Features/Settings/SettingsView.swift | Primary currency/profiles, pending/review recovery and truthful runtime diagnostics |
| PocketFinancer/Features/Onboarding/OnboardingView.swift | Explicit primary-currency selection and durable admission explanation |
| PocketFinancer/App/AppRootView.swift | Foreground bounded-batch recovery, privacy shield and availability changes |
| PocketFinancer/Shared/CurrencyFormatter.swift | ISO currency scale-aware display/parsing; no hardcoded INR/divide-by-100 |
| PocketFinancer/Services/LocalDataService.swift | Fence operations before erasing all source/trace/review/feedback entities and caches |
| PocketFinancer/Services/ModelSelfTestService.swift | Synthetic selector-contract diagnostics through the new coordinator/runtime |
| PocketFinancer/Features/Settings/ModelSelfTestReportView.swift | Truthful selector capabilities and unavailable metrics, no hidden-reasoning representation |

### 8.4 iOS new files

Under PocketFinancer/Services/SmsProcessing/:

- SmsProcessingCoordinator.swift
- SmsProcessingContracts.swift
- SmsOperationSnapshotFactory.swift
- SmsProcessingObserver.swift
- SmsProcessingRecovery.swift
- StructuralSmsAnalyzer.swift
- StructuralClauseSegmenter.swift
- CurrencyProfileRegistry.swift
- GroundedSelectorValidator.swift
- SemanticReconstructor.swift
- AutomaticPersistenceGate.swift
- DirectCandidateSelector.swift
- GroundedAccountResolver.swift
- SmsReviewService.swift

Under PocketFinancer/Data/:

- SmsProcessingModels.swift
- SmsProcessingStore.swift
- TransactionRevision.swift
- UserFeedbackEvent.swift
- ProcessingConfigurationStore.swift

Under PocketFinancer/Features/Transactions/:

- ReviewInboxView.swift
- ReviewCorrectionView.swift
- DecisionTraceTimeline.swift

Add versioned contract/profile resources and sanitized fixtures to the app/test targets. Use Swift 6 concurrency-safe value types and explicit actor isolation.

## 9. iOS acquisition and duplicate/recovery policy

Keep the App Intent and user-created Shortcut as the input boundary. Do not request Android-style SMS inbox access.

- Durably save an accepted alert before deterministic filtering or model availability checks.
- Preserve existing bounded input validation. If an input is not admitted, the intent must report failure honestly; do not say it was saved.
- Keep original optional sender, timestamp and source-app values, including missingness. Admission time is a separate field.
- Use authoritative delivery IDs when available. Re-delivery of the same authoritative identity returns the existing admission/settlement receipt.
- The inspected 15-second near-time body-digest heuristic is insufficient to prove two deliveries are the same event. Change heuristic matches to possible-duplicate review, preserving both admission records and their provenance.
- Distinct authoritative IDs remain distinct admissions even if their bodies match. Final ledger idempotency remains source/event-based.
- A possible duplicate cannot auto-persist until resolved. An explicit user decision links duplicate provenance or confirms separate events.
- Pending and needs-review draining share the coordinator. Avoid a single eight-item foreground batch leaving additional work indefinitely stranded.
- Model discovery/unavailability cannot delete or mutate source truth. Retry later when the model becomes available.
- A user-edited or rejected transaction cannot be overwritten or recreated by a late/retried parser.
- A safe intent delivery while locked is still subject to file protection and platform execution limits; expose only the admission outcome actually achieved.

## 10. UI, review ergonomics, and primary currency

### 10.1 Decision Trace

Replace Android's Thinking Output card with Decision Trace. Preserve useful existing cards, sheets, batch progress, stop/retry interactions and privacy shielding.

The chronological timeline exposes, when applicable:

1. Durable admission and original metadata availability.
2. Operation identity, immutable configuration and analyzer/profile versions.
3. Clauses, cues, candidates, original evidence and candidate coverage.
4. Triage reason codes, disposition and selector action.
5. Explicit skipped/ineligible/unavailable selector reason.
6. Model/runtime identity, available file/version information, contract/prompt/decoding profile and DIRECT_NON_THINKING mode.
7. Bounded live compact selector JSON where available.
8. Exact final available selector output, clearly marked complete/incomplete.
9. Strict parsing, grounding and host reconstruction.
10. Account resolution and each persistence-gate check.
11. Persisted, discarded, review, retry, stopped or failed result.
12. Runtime-exposed token counts, latency, throughput and memory, each with availability/provenance.
13. Recovery/retry attempts and confirmation/correction/rejection history.

Stage states are pending, running, completed, skipped, failed, interrupted and retained. A missing metric is “not available,” never zero. A skipped selector explains why.

Historical ledger records lacking original output show “original selector output unavailable.” Do not reconstruct historical prompts or fake output from transaction fields.

### 10.2 Review interactions

- Show completed/not-posted/ambiguous/multiple-event state and grounded amount, direction, owned account and counterparty first.
- Allow confirmation of a valid proposal, candidate selection, corrected interpretation, source-supported candidate miss, explicit manual value, rejection, and explicit multiple-event resolution.
- Require account selection/confirmation when resolution is missing or ambiguous.
- Show whether a proposal came from normal or assistive selection. Assistive proposals cannot auto-save.
- Save partial drafts locally and encrypted. Navigation, backgrounding, termination or restart must not lose an outstanding correction.
- Clearly distinguish retry with original configuration from retry with current settings.
- Do not auto-confirm an unknown field or turn blank controls into canonical absence.
- Rejection closes the active case and prevents background recreation; erasing source evidence is a separate explicit action unless terminal-discard policy applies.

### 10.3 Accessibility and long-content behavior

- Preserve Dynamic Type/font scaling and responsive layouts.
- TalkBack/VoiceOver reading order follows the chronological trace and then the available actions.
- Use textual stage/outcome labels and accessible semantics; never rely only on color or icons.
- Announce major stage changes, not every token/snapshot.
- Keep long JSON/evidence expandable, selectable where appropriate, wrapped or deliberately scrollable, with a clear collapsed summary.
- Provide a follow-live control; do not repeatedly steal scroll position or screen-reader focus.
- Support reduced motion, keyboard/focus navigation where applicable, adequate touch targets and contrast.
- Define loading, empty review inbox, no model, no eligible profile, stopped, malformed output, save conflict, storage failure and recovery-required states with actionable recovery.

### 10.4 Primary-currency onboarding/settings

Require an explicit user-selected primary currency before new processing. INR may be a suggested initial choice for the current target audience, but must be confirmed; do not silently migrate a default into asserted source truth.

Existing users can continue viewing their ledger while onboarding is outstanding. Newly admitted alerts wait durably for configuration.

Settings expose primary currency and approved locale/currency profiles. Changes affect future operations only. Existing traces and in-flight operations retain their captured snapshot.

## 11. Foundation contracts and unified local workbench

### 11.1 Existing files that require later changes

| File or directory | Required change |
|---|---|
| src/pocketfinancer_sms/types.py | Versioned analysis/result/configuration types and explicit provenance |
| src/pocketfinancer_sms/structural_text.py | Versioned clause refinements and Unicode parity vectors |
| src/pocketfinancer_sms/analyzer.py | Expected/held/authorized/security clause handling and deterministic family/time metadata |
| src/pocketfinancer_sms/currency.py | Profile hashes, exact-money bounds and frozen native portability behavior |
| src/pocketfinancer_sms/selector.py | Strict parsing and grounding safety fixes without semantic generation |
| src/pocketfinancer_sms/persistence.py | Independent explicit family, account, timestamp and rollout policy results |
| src/pocketfinancer_sms/trace.py | Version /2 events and native recovery/runtime provenance |
| src/pocketfinancer_sms/feedback.py | Version /2 native feedback and field-level grounding classes |
| src/pocketfinancer_sms/labels.py | Compatibility/eligibility validation only; do not weaken canonical-label /1 |
| configs/sms_processing/contracts/ | Add versioned configuration/result/analysis/trace/feedback/import schemas and validation profile |
| src/pocketfinancer_sms/workbench/store.py | Encrypted storage migration, native trace imports, distinct provenance and revision history |
| src/pocketfinancer_sms/workbench/service.py | Eligibility, protected-pool guards, candidate coverage and feedback inspection |
| src/pocketfinancer_sms/workbench/web.py | Local authenticated import/inspection APIs with redaction and bounded payloads |
| src/pocketfinancer_sms/workbench/assets/index.html | Unified inspection and optional streamlined labeling layout |
| src/pocketfinancer_sms/workbench/assets/app.js | Native trace/history navigation, lossless money, explicit annotation actions |
| src/pocketfinancer_sms/workbench/assets/styles.css | Accessible progressive disclosure and long-content handling |
| scripts/run_sms_processing.py | Explicit local migration/import commands and safe aggregate-only status, if required by the chosen CLI boundary |
| docs/architecture/SMS_PROCESSING_ARCHITECTURE.md | Document native integration/version boundaries |
| docs/contracts/GROUNDED_CANDIDATE_SELECTOR_CONTRACT.md | Freeze strict validation and direct-runtime requirements |
| docs/architecture/CURRENCY_CONTEXT_AND_PROVENANCE.md | Complete native snapshot and provenance rules |
| docs/architecture/DATA_TAXONOMY_AND_CANONICAL_LABELS.md | Feedback versus annotation and truthful family/policy distinctions |
| docs/architecture/SMS_PROCESSING_DECISION_LOG.md | Record approved versioned decisions and supersession boundaries |
| docs/plans/SMS_PROCESSING_EXECUTION_PLAN.md | Link native milestones and current implementation status |

Add:

- configs/sms_processing/contracts/releases/native-integration-v1.json
- configs/sms_processing/contracts/v2/ for the new /2 schemas and reason registry
- configs/sms_processing/contracts/processing-config.schema.json
- configs/sms_processing/contracts/native-trace-bundle.schema.json
- tests/sms_processing/golden/native-v1/ for wholly invented, sanitized fixtures
- src/pocketfinancer_sms/workbench/native_import.py
- src/pocketfinancer_sms/workbench/secure_store.py
- docs/contracts/NATIVE_SMS_INTEGRATION_CONTRACT.md

Keep historical model profiles, experiments, and protected run artifacts intact. Do not repoint the historical Android extraction profile to the new app path without a deliberate versioned historical-parity decision.

### 11.2 Unified inspection and provenance

The workbench must join, for authorized local inspection:

- Canonical imported source rows.
- Android/iOS operation/configuration snapshots and processing traces.
- Analyzer output, original selector output, reconstructed result and gate decision.
- Model/runtime provenance and availability limitations.
- User confirmations, corrections and rejections.
- Canonical annotation revision history.
- Candidate coverage, candidate-generation misses and export eligibility.

Maintain separate provenance classes for imported source, deterministic analysis, machine proposal, native user feedback, and canonical human annotation. Linking them does not merge their authority.

Machine suggestions never automatically become canonical truth. Native feedback never silently becomes a training target. A candidate miss is a recall defect even when a human can supply the correct value.

### 11.3 Optional streamlined labeling view

Offer a compact first screen for posted, not posted, ambiguous or multiple-event decisions, with a clear non-financial classification where appropriate. Show grounded amount, direction, account and counterparty first. Reveal family, rail, timestamp and advanced provenance progressively.

Retain the full canonical schema and its unknown/absent distinctions. Do not fill unknown truth to save clicks. Label submission remains an explicit human action.

This plan does not authorize actual labeling or machine-proposal generation.

### 11.4 Protected pools and local transfer

- Protected-test and later-time-holdout rows remain blind. No machine proposals are generated or imported for those pools before authorized reveal.
- An incoming native trace bundle matching a blind pool is rejected or quarantined before proposal content becomes available to the labeling surface.
- Reveal remains an explicit authorized event with preserved history. Hidden UI alone is not sufficient protection.
- Export requires an explicit user action, a clear selection/consent boundary, an encrypted local bundle and integrity manifest.
- Import verifies contract versions, hashes, source mapping, provenance class, pool eligibility and idempotency before activation.
- No background sync, telemetry, hosted labeling, browser automation over private data, remote model, Antigravity or Gemini integration is introduced.

### 11.5 Workbench encryption and existing-run compatibility

The inspected workbench uses local SQLite with restrictive permissions; those permissions are not database encryption. Plan a versioned SQLCipher-backed store with an OS-protected key.

Migrate by creating an encrypted replacement, verifying record counts and revision/hash chains, then atomically activating it. Retain a protected encrypted recovery copy until verification succeeds. Never expose keys or private database paths in output.

Canonical corpus files and backups outside the database also require verified encrypted local storage. Verify the relevant volume/container protection before claiming the whole workbench is encrypted; do not assume FileVault or equivalent is enabled.

Preserve the current private run and pointer. Readers must validate artifacts against the producer version recorded for that run; changing current code hashes must not silently rebuild, relabel or invalidate historical artifacts. New contract generation gets a new explicit lineage when separately authorized.

## 12. Privacy, retention, and deletion

- Raw sources, evidence, model output, feedback, annotations, corrections, identifiers and private paths stay local and encrypted/protected.
- Use only wholly invented fixtures in source control, CI, screenshots, demos and implementation-agent examples. Never copy a private row into a synthetic test.
- Do not print private pointer values, run IDs, source paths, private verification output or per-row records. Safe verification emits only non-identifying aggregate status.
- Keep logs, notifications, crash reports and analytics free of raw content and sensitive identifiers. Avoid model-spec stringification that exposes local model/private paths.
- Disable remote services and telemetry for the private corpus/workbench. Do not send private data to the implementation model as debugging context.
- Unresolved review has no automatic expiry. Preserve original source, immutable operations, terminal available output, review drafts and feedback while user action is outstanding.
- Terminal standalone credential discard may erase its source, candidates and related sensitive payload, retaining only a minimal non-content deletion receipt.
- Ambiguous financial evidence, none/abstain, model rejection, invalid output, missing models and exhausted retries never independently authorize deletion.
- Confirmation or rejection does not silently erase history. Retention/deletion controls are explicit and local.
- User erase invalidates all operation epochs and writers, removes protected data/drafts/caches and local app-managed exports, and records only a safe completion receipt. Do not claim cryptographic erasure of arbitrary previously copied files or guaranteed flash overwrite.
- Encrypted exports leave the app only through explicit user action and consent. No export, publication, dataset release or model deployment is authorized by implementing these features.

## 13. Verification matrix

All shared fixtures are wholly invented and versioned. Differential tests compare canonical serialized results, not merely whether each platform returns “some transaction.”

| Area | Required cases | Pass condition |
|---|---|---|
| Analyzer parity | Unicode mapping, clause boundaries, cues, candidate ordering/IDs, repeated values, explicit absence, currency profiles | Python/Kotlin/Swift byte-level fixture agreement |
| Selector input/output | none, abstain, one posted selection; duplicate keys, extra fields, wrong types, prose, truncation | Exact contract compliance; every invalid case fails closed |
| Grounding | Foreign operation IDs, wrong-kind IDs, unknown IDs, incompatible clauses, hash mismatch | Zero accepted foreign/invalid selections |
| Candidate recall | Correct user value absent from candidates, source-supported miss, ungrounded manual value | Correct miss/provenance event; no fabricated selector target |
| Triage | invoke/discard/retain_review × normal/assistive/skip valid combinations | Correct action/reason; assistive never auto-persists |
| Credentials | Standalone OTP; completed without OTP; appended security clause; failed/pending/mixed clauses | No body-wide false discard of unresolved financial evidence |
| Financial state | Completed debit/credit/refund; expected/initiated refund; hold/authorized; wallet; unsupported family; multiple events | Truth preserved; exact policy outcome; no false ledger insertion |
| Accounts | Absent, unmatched, duplicate aliases, ambiguous suffix, counterparty identifier, one confirmed owned match | Only unique grounded owned account can pass automatic gate |
| Money/currency/time | Int64 limits, precision/scale, ambiguous symbols, source override, missing timestamp, settings changed mid-operation | Exact/provenance-preserving result or review |
| Runtime | No model, ineligible mandatory reasoning, timeout, unload/reload, cancellation, invalid grammar/template, oversized output | Durable review/retry; no scratch persistence or overlapping stale writer |
| Idempotency | Repeated admission, repeated operation/action, duplicate Shortcut delivery, multiple-event submit | At most one ledger result per approved source/event |
| Crash recovery | Kill/restart before and after every durable transition and final commit | No raw-data loss, duplicate ledger, orphan account or partial settlement |
| Concurrency | Two workers/screens, late result after edit/reject/erase, expired lease with live owner | Fencing/revision checks prevent overwrite or resurrection |
| Migration | Every supported Android schema through V6; every supported iOS schema through V5; legacy edits/raw evidence | Counts/IDs/value provenance preserved; no destructive fallback |
| Feedback | Confirm/correct/reject, mixed field classifications, action replay, revision conflict | Append-only intact history and correct current projection |
| Workbench | Encrypted migration/import, integrity failure, lossless integers, provenance separation, blind pools | No silent truth promotion or protected proposal reveal |
| Privacy | Logs, diagnostics, notifications, crash strings, caches, export consent, erase race | No private payload leakage; explicit transfer boundary |
| UI/accessibility | Loading/empty/error/stop/retry/review; long JSON; large text; TalkBack/VoiceOver | Usable truthful states without inaccessible or misleading output |

### 13.1 Android physical-device tests

Use the supported on-device runtime/model profile and synthetic messages. Verify one direct greedy pass, no thinking phase, real observed timing/token facts, automatic/manual/historical/retry convergence, process kill/restart, model unload/reload, and stop versus commit races.

Desktop/HF timing does not count as GGUF phone evidence. Existing installed model assets may be used when authorized; model downloads and private inference remain separately scoped.

### 13.2 iOS physical-device tests

Run the actual App Intent through the user's Messages automation/Shortcut boundary using synthetic alerts:

- Unlocked delivery; locked after first unlock; locked before first unlock after reboot.
- Foreground and background delivery and truthful admission receipt.
- Duplicate delivery and similar-but-distinct events.
- Missing sender, timestamp and source-app metadata.
- Termination after admission, during claim, during generation, and during settlement.
- Model unavailable, parser timeout, uncooperative cancellation, later model availability.
- Foreground recovery with more than one batch of pending alerts.
- Retry after failure/commit without duplicate persistence.
- Correction/rejection before a late result returns.
- OS/model cohort differences and unavailable runtime metrics.

Simulator tests do not replace these ingestion and runtime checks. Before-first-unlock inability to access protected storage must be reported truthfully as non-admission, not silently treated as queued.

### 13.3 Test execution discipline

Run the repository-required checks for each actual change, targeted tests while developing, and the appropriate complete suite at milestone integration. Broaden or repeat testing only for changed code, failures or unresolved concerns.

Do not run protected evaluation, label private data, generate machine proposals, train, download models or deploy merely because a test gate mentions future quality evidence. Report unavailable device/evaluation evidence as pending.

## 14. Staged rollout and measurable gates

These are proposed release gates, not claims about current accuracy. Automatic-persistence precision and fail-closed behavior take priority over coverage.

| Stage | Entry gate | Completion gate |
|---|---|---|
| 1. Shared fixture parity | Approved version manifest, schemas and reason semantics | 100% Python/Kotlin/Swift golden agreement; zero accepted invalid/foreign candidate cases |
| 2. Native development diagnostics | Stage 1 plus migrations/state foundation | Full recovery/atomicity matrix passes; direct-only runtime verified on each supported device cohort; no privacy or migration defect |
| 3. Shadow processing | Stage 2; automatic ledger insertion disabled | At least 200 synthetic/device operations per platform plus complete failure matrix; zero new automatic ledger writes, evidence losses, duplicate settlements or stale-writer mutations |
| 4. Retained-review-only | Stage 3; durable review UI and feedback complete | At least 100 end-to-end confirmation/correction/retry/restart cases per platform; zero automatic insertions and zero lost drafts/history |
| 5. Narrow automatic persistence | Separate user enablement approval; supported model/profile cohort; all gates enabled | At least 600 independently adjudicated eligible would-persist cases per platform/model/profile cohort with zero false insertions; one-sided 95% precision lower bound at least 99.5%; zero safety/idempotency defects |
| 6. Monitored general rollout | Stage 5 evidence; device/recovery gates complete | At least 14 days and 200 locally reviewed automatic cases per platform with no false persistence, duplication or evidence-loss incident before cohort expansion |
| 7. Reversible rollback | Rollback tested before Stage 5 | Switch to retained-review-only in a compatible build/policy; stop automatic settlement, preserve operations/evidence/user transactions, and recover pending work successfully |

The 600-case precision gate is an acceptance target, not authorization to reveal protected data or run protected evaluation. A suitable independently adjudicated dataset and any private evaluation require their own authorization. If evidence is insufficient, remain review-only.

Monitor locally: gate outcomes, review reasons, selector invalidity, candidate misses, duplicate prevention, recovery success and user-confirmed errors. Aggregate export is optional and explicit-consent only; there is no telemetry dependency.

Changing model, template, contract, analyzer, currency profile or OS cohort can invalidate eligibility evidence. Version the cohort and repeat the affected gates.

Rollback must not re-enable the old direct extractor, restore default-account fallback, delete new history, reverse migrations destructively or automatically remove user transactions. Incorrect ledger entries are corrected through explicit revision history.

## 15. Cross-repository milestones and commit sequence

Each milestone has a bounded completion gate and a concise handoff. Do not run the entire project as one undifferentiated implementation pass.

### Milestone A — Synchronize and freeze shared behavior

- [ ] Pull merged native updates safely and record new baselines.
- [ ] Review relevant merged deltas against this plan.
- [ ] Add sanitized regression vectors for the narrow foundation safety findings.
- [ ] Freeze schemas, reason codes, ID/Unicode/money rules, configuration and persistence policy.
- [ ] Publish the local versioned manifest/fixture bundle inside the foundation repository.

Completion: all contract decisions represented by executable schemas and vectors; no private run rewrite; native ports can implement without inventing semantics.

Foundation commit sequence:

1. docs(plan): record native SMS integration handoff
2. test(sms): add native parity and financial-state regression vectors
3. feat(contracts): version native operation result trace and feedback contracts
4. fix(sms): harden clause state handling and strict selector validation
5. docs(contracts): freeze native integration profiles and compatibility rules

### Milestone B — Native storage and state foundations

- [ ] Android Room V6 and iOS SwiftData V5 migrations.
- [ ] Immutable sources/configuration/analysis/output and append-only feedback/revisions.
- [ ] Claims, fencing, idempotency and atomic account/ledger/trace settlement.
- [ ] Legacy transaction compatibility and fail-closed startup.

Completion: migration and crash-state tests pass before model integration or new automatic persistence.

Android commits:

1. feat(data): add durable SMS operation review and revision storage
2. fix(data): preserve legacy money and enforce atomic source event settlement

iOS commits:

1. feat(data): add versioned SMS operation review and revision models
2. fix(data): preserve legacy extraction history and atomic recovery state

### Milestone C — Native parity, direct adapters and coordinators

- [ ] Kotlin/Swift analyzer and grounding/reconstruction/gate parity.
- [ ] Direct-only Candidate Selector adapters.
- [ ] Android automatic/manual/historical/retry/diagnostic callers converge.
- [ ] iOS intent admission/pending/review retry/foreground recovery converge.
- [ ] Complete durable traces and typed outcomes.

Completion: parity and failure matrix pass; one selector attempt per operation; no app-side thinking path; no automatic ledger writes in shadow mode.

Android commits:

3. feat(sms): implement versioned analyzer and selector validation parity
4. refactor(inference): enforce direct non-thinking candidate selection
5. refactor(pipeline): route all SMS processing through one coordinator

iOS commits:

3. feat(sms): implement native analyzer and candidate selector parity
4. refactor(ingestion): coordinate durable selection recovery and settlement
5. fix(ingestion): preserve uncertain duplicates and fence late parser results

### Milestone D — Review, trace UI, currency and workbench

- [ ] Durable review/correction, candidate misses, feedback and revision projection.
- [ ] Decision Trace replaces Thinking Output; truthful platform visibility.
- [ ] Primary-currency onboarding/settings with immutable snapshots.
- [ ] Encrypted local workbench import and provenance-separated inspection.
- [ ] Optional streamlined labeling UI without performing labeling.

Completion: accessibility/review/restart tests pass; no fake historical output; no silent annotation/target promotion; blind-pool boundaries preserved.

Android commits:

6. feat(review): add grounded corrections and append-only transaction history
7. feat(ui): replace thinking output with an accessible decision trace
8. feat(settings): snapshot primary currency and approved profiles

iOS commits:

6. feat(review): add durable corrections and transaction revision history
7. feat(ui): expose truthful decision traces and recovery actions
8. feat(settings): add primary currency and provenance-aware formatting

Foundation commits:

6. feat(workbench): add encrypted native trace import and provenance separation
7. feat(workbench): inspect feedback candidate coverage and annotation history
8. feat(workbench): add optional streamlined canonical labeling view

### Milestone E — Shadow and review-only verification

- [ ] Complete shared, native migration, privacy and recovery suites.
- [ ] Run synthetic Android physical-device inference checks.
- [ ] Run iOS physical-device Shortcut ingestion/recovery matrix.
- [ ] Meet shadow and retained-review-only counts and invariants.
- [ ] Record unresolved cohort/accuracy evidence honestly.

Android commit:

9. test(sms): verify migration recovery privacy and device processing flows

iOS commit:

9. test(sms): verify migration Shortcut recovery and selector device parity

Foundation commit:

9. docs(verification): record native parity evidence and rollout readiness

### Milestone F — Separate rollout decision

- [ ] Present concrete eligibility/precision evidence.
- [ ] Obtain explicit approval to enable narrow automatic persistence.
- [ ] Verify rollback preserves evidence and user transactions.
- [ ] Expand only through the measured stages in Section 14.

Code implementation approval does not automatically authorize this milestone's enablement, protected evaluation, model deployment, app publication, training or dataset release. Release Please continues to own repository versions/changelogs/tags where configured.

## 16. Risks and decisions requiring approval

### Proposed decisions to approve with this plan

- Narrow foundation /2 extensions and strict-validation fixes while preserving Candidate Selector output /1.
- Native storage versions Room V6 and SwiftData V5, subject to the next-session merged-delta check.
- Unique confirmed owned-account requirement and no automatic account creation/fallback.
- Completed wallet movements remain truthful but review-only in the MVP.
- Missing/uncertain timestamp provenance defaults to review.
- Primary-currency onboarding is explicit; settings affect future operations only.
- One direct pass, 512 answer tokens, 16 KiB output cap, 60-second deadline and versioned runtime eligibility.
- Two-minute claim lease, 15-second heartbeat, fencing and bounded retries.
- No automatic expiry of unresolved review evidence.
- SQLCipher workbench migration plus verification of encryption for raw artifacts outside the database.
- The measurable staged rollout gates above.

These are concrete defaults for implementation, not open-ended requests to redesign the system.

### Material risks

- Analyzer recall limits automation; more model reasoning cannot supply missing grounded candidates safely.
- A mandatory-reasoning Android model/template may become ineligible, leaving review available until a separately approved direct-compatible runtime exists.
- iOS system model behavior and API visibility can change across OS cohorts and cannot always be pinned to a model file hash.
- Account ownership, common suffixes and missing metadata reduce coverage; fail closed rather than broaden matching silently.
- Historical edits and floating-point values cannot be reconstructed into exact original history.
- Background execution/file protection can prevent iOS admission or immediate processing; truthful receipts and later recovery are essential.
- Encryption migration and old-store compatibility need failure injection and protected recovery artifacts.
- Sufficient independent quality evidence may not yet exist for automatic persistence. Remain review-only rather than weakening the gate.
- The newly merged native PRs may change inspected files or create Git divergence; resolve that preflight explicitly before code work.

### Separate authorization still required

Private labeling, machine proposals, protected-test/holdout reveal or scoring, training/fine-tuning, model downloads not already authorized, research thinking experiments, private export/publication, model/app deployment, and automatic-persistence enablement are outside routine implementation authorization.

If a genuine contract contradiction appears after synchronization, document the concrete fixture/code evidence and the smallest versioned decision required. Do not silently change the foundation architecture or proceed past an unresolved safety boundary.

## 17. Copyable next-session implementation prompt

Use the following prompt with the user's selected GPT-5.6 model and High reasoning:

> Implement the PocketFinancer plan at:
>
> /Users/toji/Projects/pocket-financer/pF_slm_selection/docs/plans/NATIVE_SMS_INTEGRATION_PLAN.md
>
> Work across pF_slm_selection, pocket-financer-android and pocket-financer-ios under /Users/toji/Projects/pocket-financer/. Read this plan and every applicable AGENTS.md and repository-specific instruction file completely before Git operations or edits.
>
> Important synchronization update: the Android and iOS PRs containing the previously local ahead commits have been merged. Before implementation, inspect each native repository, require a clean worktree, fetch/prune its configured remote, and pull only its currently checked-out branch from its configured upstream using fast-forward only. Record the actual new HEAD/upstream/status and inspect relevant changes since the plan's audited commits. The plan's listed remote commits are historical anchors, not the latest remote state or reset targets.
>
> If the PR merge strategy caused divergence, or there is an unexpected branch/remote/upstream, unexpected local commit or dirty worktree, stop and report the exact Git condition. Do not reset, stash, clean, rebase, merge, switch branches to evade the condition, discard commits or force-update references. Preserve the foundation SMS branch and this uncommitted plan document; do not merge main into the foundation branch.
>
> After safe synchronization, create the required implementation branches and follow the plan contract-first: shared schemas/reason codes/golden fixtures, then native migrations and durable state, then Kotlin/Swift analyzers and coordinators/direct selector adapters, then review/correction/trace UI and currency settings, then shadow and review-only verification. Treat the shared foundation as the production-intended authority. Every native SMS path must converge on one coordinator per app.
>
> Remove app-side thinking machinery from SMS processing. The selector is one greedy direct pass returning only none, abstain, or grounded candidate IDs. It is not the semantic or persistence authority. Preserve the iOS App Intent durable admission boundary. Retain uncertain evidence, prohibit default-account fallback, and use append-only feedback/revisions with exact money for new records.
>
> Implement milestone by milestone with focused changes and tests required by the affected code and repository instructions. Reuse the planning audit; do not spend the session repeating broad discovery or unrelated tests. Keep a concise milestone handoff with changed files, checks, remaining work and any actual blocker. Do not claim physical-device verification unless it was performed.
>
> Use only wholly invented fixtures in source, logs, screenshots and model context. Never expose private run IDs/paths, SMS, senders, accounts, annotations, correction text or private verification output. Do not use hosted services on the private corpus/workbench. Do not label private data, generate private machine proposals, reveal/score protected pools, train, download models, export private data, deploy, publish, push, open PRs or enable automatic persistence without separate explicit authorization. Keep automatic persistence disabled through implementation and shadow/review-only verification.
>
> This message authorizes implementation of the plan's engineering milestones, not the separately gated rollout/data/model actions. If synchronized code reveals a genuine contradiction, explain the smallest versioned contract decision required instead of redesigning the architecture. Start by reporting the synchronization result and the first bounded implementation milestone, then proceed.
