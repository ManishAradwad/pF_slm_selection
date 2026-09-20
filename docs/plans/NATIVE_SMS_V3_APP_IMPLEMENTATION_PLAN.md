# PocketFinancer native SMS v3 app implementation plan

Status: **mandatory execution steps 1–8 implemented; Android automated and
emulator gates passed; Apple and physical-device acceptance remains pending**

Prepared: 2026-09-14

Execution update: 2026-09-16. The evidence-backed model-provenance decision
required the additive `native-integration-v4` / `processing-config/4` successor;
v1/v2/v3 remain frozen. New eligible Android work uses a real runtime-observed
model-file SHA-256, while the Apple system-managed runtime records no file hash.
See [NATIVE_SMS_V4_POST_STEP_5_REVIEW.md](NATIVE_SMS_V4_POST_STEP_5_REVIEW.md)
for the preserved step-5 checkpoint and
[NATIVE_SMS_V4_IMPLEMENTATION_REVIEW.md](NATIVE_SMS_V4_IMPLEMENTATION_REVIEW.md)
for the continued steps 6–10 implementation, verification, and remaining-gap
record. Android completed the synthetic local review-all trial. This host cannot
run Xcode, an iOS simulator, or either platform's physical-device acceptance.

This is the executable plan for integrating the frozen SLM-primary SMS
extractor into the Android and iOS apps and building the source-span review
experience on the Transactions page. It supersedes the implementation portions
of NATIVE_SMS_INTEGRATION_PLAN.md, which remains historical evidence for the
completed v1/v2 candidate-selector foundation.

Creating this document does not change either app, enable automatic
persistence, deploy a model, move private data, or authorize telemetry.

## 1. Inspected baseline

This plan is based on the live code in all three repositories.

| Repository | Live path | Inspected branch and HEAD | State |
|---|---|---|---|
| Shared foundation | /home/tojinotzenin/pF_slm_selection | codex/slm-primary-sms-extraction at f6cd955 | Clean; v3 authority |
| Android | D:\Personal_Projects\pocket-financer\pocket-financer-android | codex/fix-windows-sms-assets at 1e8641e | Clean |
| iOS | D:\Personal_Projects\pocket-financer\pocket-financer-ios | codex/feat-native-sms-integration at f524e8a | Clean; one commit ahead |

The Android and iOS checkpoint commits are both named
fix: align native SMS selector runtime policy. They align historical v2 with
the no-deadline runtime policy. They do not implement v3.

Current code facts:

- The shared foundation implements and freezes native integration v3.
- Both apps still create v2 candidate-selector operations.
- Both apps already have durable admission, claims, traces, review storage,
  feedback, and transaction revisions. Extend those systems.
- Android review shows the SMS but edits plain values and records corrections
  without source grounding.
- iOS review is a Form of text fields, pickers, and an editable date. It has no
  selectable source evidence.
- Android has a separate Reviews destination. iOS includes pending alerts in
  Transactions, but without the intended Processing/Needs Review hierarchy.
- Neither app has an account-balance model.
- iOS intake is its App Intent used from a user-created Messages Shortcut plus
  manual import. Public iOS APIs do not permit passive SMS inbox access.

## 2. Locked product decisions

1. Every valid posted v3 extraction requires review in the first rollout.
2. Processing and Needs Review appear above the ledger on Transactions.
3. All assigned fields remain highlighted; only one active field displays
   draggable native selection handles.
4. Amount, direction, and account require exact non-empty source spans.
   Counterparty is optional.
5. The platform receipt time is read-only and cannot be model-generated or
   edited in this rollout.
6. Successful confirmation returns to Transactions, removes the review card,
   and reveals the saved ledger transaction.
7. One account alias match is reused. No match proposes an account created only
   on confirmation. Multiple matches block as an account-integrity defect.
8. Do not add balance extraction, storage, derivation, or display.
9. Processing and evidence remain local. No hosted inference, CloudKit,
   telemetry, analytics, or sensitive logs.
10. Frozen v1/v2 assets and stored operations remain readable. A retry never
    silently changes its release.

## 3. Scope

In scope:

- Exact v3 assets and manifest verification in both apps.
- New-operation processing-config/3 snapshots.
- One direct SLM extraction using the analyzer only as advisory context.
- Strict parsing, scalar grounding, exact money, account resolution, duplicate
  assessment, and typed persistence decisions.
- Durable v3 results, traces, review cases, drafts, feedback, and revisions.
- Native span selection with extractor spans preselected.
- Atomic confirmation, account reuse/creation, and transaction insertion.
- Backward-compatible Android Room and iOS SwiftData migrations.
- Parity, recovery, UI, accessibility, emulator/simulator, and device checks.

Out of scope:

- Automatic SMS transaction persistence.
- Model training, selection, download-policy changes, or deployment.
- Private phone data in source control or CI.
- Passive iOS inbox access.
- Four simultaneously editable handle pairs.
- Account balances or statement reconciliation.
- Cloud synchronization.

## 4. Contract authority and compatibility

The implementation authority is:

- configs/sms_processing/contracts/releases/native-integration-v3.json
- configs/sms_processing/contracts/v3/
- configs/sms_processing/prompts/sms-extractor-v1.txt
- configs/sms_processing/grammars/sms-extractor-v1.gbnf
- tests/sms_processing/golden/extractor-v1/sanitized-vectors.json
- src/pocketfinancer_sms/extractor.py
- src/pocketfinancer_sms/processing_v3.py
- src/pocketfinancer_sms/account_resolution.py
- docs/contracts/NATIVE_SMS_INTEGRATION_CONTRACT.md

Native code reproduces the frozen bytes and executable behavior. Do not edit a
frozen shared artifact to make a native port pass. A contract correction needs
a new versioned release and separate review.

Version routing:

- New work uses native-integration-v3 and processing-config/3.
- Historical records retain their original parser, validator, display, and
  context-preserving retry behavior.
- Retry with current v3 is a separate explicit action linked to its parent.
- Unknown releases, missing assets, or hash mismatches fail closed to review.

## 5. Target flow and durable state

    received/imported
      -> durable admission
      -> immutable v3 configuration
      -> claim
      -> advisory analysis
      -> extractor input
      -> one local SLM attempt
      -> strict validation and normalization
      -> account resolution
      -> duplicate assessment
      -> persistence gate
           -> not-posted terminal
           -> interrupted/retryable
           -> retained review
      -> review draft
      -> atomic confirmation
           -> account reused or created
           -> transaction inserted
           -> feedback appended
           -> review resolved
      -> ledger

All confirmation-side changes occur in one local database transaction. A crash
cannot leave a resolved review without its transaction or create two
transactions from one action ID.

## 6. Work package A: package frozen v3

For each app:

1. Vendor the exact v3 manifest and required referenced artifacts without
   overwriting v1/v2.
2. Bind new operation snapshots to the v3 manifest and asset hashes.
3. Verify every packaged artifact in unit tests.
4. Verify the production subset before admitting a new v3 operation.
5. Route missing or altered assets to durable review using a stable
   configuration-integrity reason.
6. Compare both apps against the same release ID, policy, paths, and SHA-256
   table.

Gate A:

- No new operation starts with an unreproducible configuration.
- Historical resources remain readable.
- Android and iOS report identical v3 bindings.

## 7. Work package B: direct extractor transport

Build sms-extractor-input/1 from the complete unchanged message, derived sender
family, snapshotted currency/profile IDs, advisory analyzer candidates and cues,
and the frozen output rules. Analyzer evidence is not an answer allowlist.

Runtime policy:

- exactly one local model call per operation;
- DIRECT_NON_THINKING and greedy decoding;
- 512 answer tokens and 16,384 UTF-8 output bytes at most;
- no model/parser wall-clock timeout;
- real user cancellation, claim loss, and process interruption;
- Android uses its serialized SlmRuntime lease;
- iOS uses FoundationModelExecutionGate.

The model may return exactly none, abstain, or one posted transaction containing
amount value/currency/span, direction value/span, account reference/span, and a
nullable counterparty with a consistent nullable span. It cannot emit receipt
time, account database ID, reason code, confidence, minor units, duplicate
status, or persistence.

Gate B:

- Both apps record one bounded v3 attempt.
- Cancellation is distinguishable from runtime failure.
- New v3 operations never use candidate-ID selection.

## 8. Work package C: strict validation and normalization

Reject invalid UTF-8, oversized/empty/truncated output, non-object or trailing
documents, duplicate keys, non-standard numeric constants, unknown/missing
fields, type coercion, unknown decisions/currencies/directions, malformed
decimals, unsupported precision, non-positive amounts, Int64 overflow, invalid
scalar spans, mismatched span text, inconsistent account evidence, and
counterparty value/span nullability mismatches.

Normalize valid posted output into positive signed 64-bit minor units, frozen
ISO currency/scale, debit or credit, normalized account reference with original
evidence, and optional counterparty with original evidence. Never use floating
point for money.

Contract ranges count Unicode scalars:

- Android walks code points to create safe UTF-16 offsets and rejects surrogate
  splits.
- iOS walks unicodeScalars to create String.Index and NSRange values.
- UI receives only validated converted source ranges.

Gate C:

- Both apps pass every sanitized extractor vector.
- Emoji, combining marks, repeated strings, and boundaries match Python.
- Malformed-output fuzz tests never produce a semantic transaction.

## 9. Work package D: accounts and duplicates

Account resolution:

1. NFKC, trim, and case-fold the extracted account evidence.
2. Prefer a normalized VPA; otherwise use exactly one masked/bare 3-8 digit
   suffix.
3. Match only owned account aliases.
4. One match is uniquely resolved.
5. Zero matches becomes a proposed new-account review state.
6. Multiple matches is ambiguous and blocks confirmation.
7. Never use a default account.

Last four digits are not treated as globally unique. Issuer/bank, account kind,
and normalized alias participate in integrity checks. A display proposal may
look like ICICI Credit Card ••3489.

For an unresolved review, confirmation atomically rechecks the catalog, reuses a
new unique match if one now exists, otherwise creates the proposed account and
alias, records that account ID in feedback, inserts the transaction, and resolves
the review. The model never creates accounts.

Duplicate checks run in order: review action ID, source idempotency key,
platform source-event key, then a fingerprint over minor units, currency,
direction, resolved account ID, and receipt time.

Gate D:

- Unique, unresolved, ambiguous, concurrent-create, repeated-submit, and
  possible-duplicate tests pass.
- No default account or balance behavior appears.

## 10. Work package E: storage and migration

Persist in the current encrypted/protected store:

- immutable source reference and hash;
- v3 configuration and hash;
- advisory analysis;
- bounded extractor attempt and raw response;
- validation/normalization result;
- account and duplicate assessments;
- typed persistence decision;
- sequence-numbered hash-chained trace;
- immutable review case, draft revisions, feedback, and transaction revision.

Never put SMS bodies, senders, account labels, or model output in ordinary logs,
notifications, crash text, analytics, or test reports.

Migration rules:

- Android adds the next Room migration and exported schema.
- iOS adds SwiftData V7 and an explicit migration stage.
- Historical payloads are never decoded as v3.
- Existing open reviews remain usable through historical projection.
- Only migrated historical rows may lack v3 fields.
- Test upgrade, failure, interruption, and reopen/recovery paths.

Gate E:

- Existing user data opens after upgrade.
- V3 work survives termination at every durable boundary.
- Recovery cannot duplicate attempts, feedback, accounts, or transactions.

## 11. Work package F: review domain

Each app stores the same field state:

| Property | Meaning |
|---|---|
| field | amount, direction, account, or counterparty |
| scalar range | half-open source range or absent |
| exact text | verified unchanged source slice |
| normalized value | host-derived value or absent |
| provenance | extractor, analyzer advisory, or user selected |

Rules:

- Valid extractor spans prefill fields.
- Analyzer suggestions remain visibly advisory.
- Range changes rerun host normalization.
- Amount, direction, and account are required.
- Counterparty may be cleared.
- Receipt time stays immutable.
- Confirmation revalidates source hash, revision, every range/value, account
  state, and duplicate state.
- Feedback is append-only, revision-bound, and idempotent by action ID.

Gate F:

- Draft reload preserves exact selections after restart.
- Stale simultaneous submissions fail without partial changes.
- Success returns one transaction ID and resolves exactly one review.

## 12. Work package G: review UI

The screen contains:

1. sender family and read-only receipt time;
2. stable reason summary;
3. complete immutable SMS body;
4. Amount, Direction, Account, and Counterparty chips;
5. persistent distinct highlights for assigned fields;
6. native selection handles for the active field;
7. normalized preview and account reuse/new proposal;
8. Confirm transaction, Retry extraction, and Not a transaction.

Interaction:

1. Open with extractor spans preselected.
2. Tap a field chip to activate it.
3. Drag native handles over exact source wording.
4. Keep inactive fields highlighted without extra handles.
5. Provide Clear and Reselect per field.
6. Disable confirmation until required ranges normalize.
7. On success, return to Transactions and reveal the ledger item.

Four simultaneous handle pairs are deliberately excluded because each platform
owns one active text selection. Persistent labeled highlights preserve the
multi-field context.

Accessibility:

- Do not use color as the only field signal.
- Each chip announces selected/missing state and selected text.
- Expose focus, clear, and reselect actions.
- Support screen readers, font scaling, high contrast, RTL, and reduced motion.

Gate G:

- Users adjust fields without editing SMS text.
- Long messages, emoji, RTL, scrolling, rotation/backgrounding, and large fonts
  preserve valid state.

## 13. Work package H: Transactions workflow

Transactions becomes:

1. **Processing** for claimed/in-flight operations.
2. **Needs Review** for open v3 cases.
3. **Transactions** for the confirmed ledger.

Processing cards use truthful stages: Alert saved, Inspecting message, Running
on-device AI, Validating details, Matching account, Ready for review, and
Interrupted — retry available.

There is one durable state owner. Home may retain a compact summary/link, but
the canonical actionable list lives on Transactions. Keep Android's Reviews
route temporarily as a deep-link compatibility alias and remove it only after
navigation/back-stack tests show that no entry point is stranded.

Gate H:

- Admission appears in Processing before a transaction exists.
- It moves once to Needs Review.
- Confirmation removes the review and adds exactly one ledger row.
- Retry and restart states remain understandable and actionable.

## 14. Android implementation map

Pipeline/inference:

- pipeline/.../sms/SmsProcessingContracts.kt
- pipeline/.../sms/SmsOperationSnapshotFactory.kt
- pipeline/.../sms/DefaultSmsProcessingCoordinator.kt
- pipeline/.../SmsParserWorker.kt
- inference/.../SlmRuntime.kt

Add focused v3 types rather than changing historical meanings:

- SmsExtractorContracts.kt
- DirectSmsExtractor.kt
- SmsExtractorValidator.kt
- UnicodeScalarSpan.kt
- SmsExtractorNormalizer.kt

LlamaEngine remains internal.

Data:

- data/.../db/entity/SmsProcessingEntities.kt
- data/.../db/dao/SmsProcessingDao.kt
- data/.../repository/SmsProcessingStore.kt
- data/.../repository/SmsReviewRepository.kt
- data/.../repository/AccountRepository.kt
- data/.../db/AppDatabase.kt

UI:

- app/.../ui/review/ReviewViewModel.kt
- app/.../ui/review/ReviewDetailScreen.kt
- app/.../ui/review/ReviewInboxScreen.kt
- the existing Transactions screen and ViewModel
- app/.../ui/smsprocessing/SmsPipelineActivityCard.kt
- app/.../ui/PocketFinancerRoot.kt
- app/.../ui/navigation/Screen.kt

Add a read-only Compose evidence-selection component. Its active TextRange owns
native handles; AnnotatedString styles draw inactive selections. Never use
normalized offsets to highlight source text.

## 15. iOS implementation map

Pipeline:

- PocketFinancer/Services/SmsProcessing/SmsProcessingContracts.swift
- PocketFinancer/Services/SmsProcessing/SmsOperationSnapshotFactory.swift
- PocketFinancer/Services/SmsProcessing/SmsProcessingCoordinator.swift
- PocketFinancer/Services/SmsProcessing/DirectCandidateSelector.swift
- PocketFinancer/Services/SmsProcessing/AlertIngestionService.swift

Add:

- SmsExtractorContracts.swift
- FoundationSmsExtractor.swift
- SmsExtractorValidator.swift
- UnicodeScalarSpan.swift
- SmsExtractorNormalizer.swift

Prefer a raw JSON response so one-document parsing remains observable. If
Foundation Models requires guided structured output, isolate that transport,
serialize the v3 wire shape, run every host semantic/span validation, and record
the distinction without weakening the contract.

Data:

- PocketFinancer/Data/Models.swift
- PocketFinancer/Data/PocketFinancerSchema.swift
- PocketFinancer/Data/AppDatabase.swift
- PocketFinancer/Services/SmsProcessing/SmsProcessingStore.swift

UI:

- PocketFinancer/Views/TransactionsView.swift
- PocketFinancer/Views/ReviewCorrectionView.swift
- PocketFinancer/Views/AlertProcessingDetailView.swift
- PocketFinancer/AppRootView.swift

Add EvidenceSelectionTextView.swift as a UIViewRepresentable around a selectable,
non-editable UITextView. Attributed ranges show all highlights; selectedRange
owns active handles. Explicitly convert scalar ranges to UTF-16 NSRange.

Preserve durable enqueue-before-model behavior in
ImportTransactionAlertIntent.swift and the existing Shortcut/manual intake.

## 16. Verification matrix

| Layer | Android | iOS | Required result |
|---|---|---|---|
| Assets | JVM hash tests | XCTest hash tests | Exact v3 bindings |
| Parser | malformed-output suite | malformed-output suite | Duplicate/trailing/type/size rejection |
| Spans | Kotlin code-point vectors | Swift scalar vectors | Sanitized Unicode parity |
| Money | exact Long tests | exact Int64 tests | Scale, precision, overflow |
| Coordinator | coroutine/claim tests | async actor/store tests | none/abstain/posted/failure/cancel/restart |
| Accounts | Room repository tests | SwiftData store tests | unique/new/ambiguous/concurrent |
| Duplicates | DAO/idempotency tests | store/idempotency tests | Retry/submit cannot duplicate |
| Migration | Room migration test | V6-to-V7 test | Existing data remains readable |
| Review | ViewModel tests | state/store tests | prefill/edit/clear/draft/stale revision |
| UI | Compose instrumentation | XCUITest | handles/highlights/validation/navigation |
| Accessibility | Compose semantics | VoiceOver identifiers | Meaning is not color-only |
| Build | unit, lint, schema, APK | Swift 6 unit/UI/Release on macOS | Exact environment reported |
| Runtime | emulator then phone | simulator then iPhone | Local model, latency, memory, recovery |

No private SMS, sender, account identifier, or per-row output enters fixtures or
CI. Device reports contain only aggregate/non-sensitive results.

## 17. Commit sequence

Shared foundation: no contract change is planned. Add parity tooling only if a
deterministic export/check command is missing. Frozen behavior changes require a
new release and separate review.

Android:

1. build: vendor native SMS v3 contract assets
2. feat: add direct grounded SMS extractor
3. feat: persist native SMS v3 review state
4. feat: add source-span transaction review
5. feat: integrate SMS actions into transactions
6. test: cover native SMS v3 parity and recovery

Keep 1e8641e intact. Before coding, either land its focused branch and create
codex/feat-native-sms-v3, or explicitly continue it as a stacked dependency.

iOS:

1. build: vendor native SMS v3 contract assets
2. feat: add Foundation Models SMS extractor
3. feat: migrate native SMS v3 review state
4. feat: add source-span transaction review
5. feat: integrate SMS actions into transactions
6. test: cover native SMS v3 parity and recovery

The existing iOS feature branch is correctly scoped, but f524e8a must be pushed
or reconciled before its PR is complete.

Each PR reports outcome, migration impact, privacy impact, commands/results,
simulator/emulator evidence, physical-device gaps, and automatic-persistence
status.

## 18. Mandatory execution order

1. Preflight branch/upstream state without discarding or rebasing local work.
2. Finish asset/hash parity on both platforms.
3. Implement parser, Unicode, money, accounts, and sanitized vectors.
4. Route new operations through v3 while preserving v1/v2.
5. Migrate durable operation/review/feedback state and recovery.
6. Implement review drafts, validation, and atomic confirmation.
7. Build native selection UI and Transactions sections.
8. Pass repository automated gates.
9. Verify emulator/simulator, then physical devices.
10. Run a local review-all trial without telemetry.

Pause for review after steps 2, 5, 7, and 9.

## 19. Final acceptance criteria

- [ ] New Android and iOS operations snapshot exact v3.
- [ ] Historical v1/v2 work remains readable and recoverable.
- [ ] Both apps pass every sanitized extractor vector.
- [ ] The model runs once locally without a wall-clock timeout or thinking pass.
- [ ] Invalid output cannot reach persistence.
- [ ] Every valid posted result is retained for review.
- [ ] Amount, direction, and account prefill from exact source spans.
- [ ] One field has native handles while all assigned fields stay highlighted.
- [ ] Counterparty can be changed or absent.
- [ ] Receipt time remains immutable.
- [ ] Unique accounts are reused, new accounts are confirmation-only, and
      ambiguity blocks.
- [ ] Confirmation is atomic/idempotent and returns to Transactions.
- [ ] Processing and Needs Review appear above the ledger.
- [ ] Restart preserves queued, in-flight, draft, and resolved state.
- [ ] No balance, remote inference, telemetry, or private fixtures were added.
- [ ] Android passes unit, migration, lint, schema, and APK checks.
- [ ] iOS passes Swift 6, migration, unit/UI, and Release checks on Xcode 26.
- [ ] Physical-device evidence and remaining gaps are documented truthfully.

## 20. Rollout boundary

The first release remains review-only. Revision-bound local feedback may be
evaluated later after privacy and data-rights review. Automatic persistence
needs a separate versioned release, protected human-gold evidence, both
platforms' device evidence, and an explicit product decision. Completing this
plan does not make that decision.
