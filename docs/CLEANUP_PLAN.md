# Cross-repository cleanup plan and handoff

Status: **pure prefilter pilot locally committed and source-reviewed; PR-base confirmation and execution gates pending**
Last updated: **2026-11-07** (pilot baseline/live PR refresh: **2026-11-07**)

This is the single temporary work plan for `pocket-financer-intelligence`,
`pocket-financer-android`, and `pocket-financer-ios`. It records the owner's
approved direction, not a claim that the target is implemented. Read the three
`AGENTS.md` files as well. For this cleanup, the target below supersedes older
roadmaps, experimental product choices, and blanket historical-code retention
requirements. Existing code/tests establish actual behavior, not future intent.
Privacy, source-data protection, branch protection, and release controls remain.

Update this document in place; do not create session diaries or parallel plans.
After completion, move lasting guidance into the few permanent docs and remove
this temporary plan. Older roadmaps are reference material pending consolidation,
not competing task lists.

## 1. Owner-approved target

### Scope and ownership

- Make all three repos lean around one current processing flow. Remove obsolete
  selectors, experimental pipelines, unused assets/dependencies, version adapters,
  scripts, tests solely for removed flows, and irrelevant Markdown documentation
  after their dependencies and useful unmerged work have been audited. Git history
  retains committed experiments; do not keep runnable historical pipelines merely
  for archival purposes. Current testing/evaluation is not obsolete baggage.
- Disposable development app databases and generated development datasets may be
  reset. Identify the exact artifacts before a reset. This does **not** authorize
  deletion of original SMS archives, human labels/corrections, provenance mappings,
  encryption keys, or their necessary backups. No reset is needed for the audit.
- Intelligence owns the shared design/contracts/prompts and Python evaluation,
  dataset/annotation, and improvement tooling. Android and iOS own their native
  intake, inference, validation, persistence, and UI. Native apps link contributors
  to the design; they do not call Python or depend on this repo at runtime.
- No shared UI framework, wholesale rewrite, automatic training-label promotion,
  publication, model deployment, or release is authorized by this plan.
- **Preserve unfinished SMS Review-page UI work.** The owner wants that work
  carried into the retained implementation, not removed because it is incomplete,
  experimental-looking, or tied to a superseded pipeline version. Keeping it only
  in Git history is not sufficient. Preserve the screens, evidence-selection and
  partial-field editing behavior, supporting state/validation and tests while
  consolidating backend contracts. Do not redesign, replace, hide or remove that
  work as a cleanup side effect; any such change requires a separate owner
  decision. Preservation is not a claim that the UI is finished or safe to ship.

### Product flow

```text
Receive/discover SMS or supported imported alert
  -> same-source identity / existing-outcome check
  -> narrow deterministic prefilter
  -> durable eligible work + per-operation settings snapshot
  -> advisory analysis
  -> one local SLM classification/extraction
  -> strict parsing, source grounding, exact money normalization
  -> account resolution + possible same-payment duplicate check
  -> Transactions / No transaction / Review
```

Platform intake may persist evidence earlier for crash safety; eligible evidence
must be durable before inference. Filtering must precede inference and behave
consistently across realtime intake, historical import, and manual scans.

- Prefilter confidently irrelevant standalone OTPs, ads, and similarly tested
  non-events without calling the model or creating Review. Uncertainty passes
  through. Missing regex matches are not proof of irrelevance; an `OTP` keyword
  alone cannot reject a completed transaction. Exclusion is not deletion from the
  phone's SMS inbox. Define minimal retention for excluded/`none` outcomes during
  implementation; do not invent a new retention policy during this bootstrap.
- The analyzer supplies optional evidence, not an extraction allowlist or an
  overriding semantic decision. The unchanged message is the grounding authority.
- The model returns `posted`, `none`, or `abstain`. `Review` is a host destination,
  not a model classification. The host never equates `posted` with permission to
  save without validation.
- Fields: required amount, currency, debit/credit direction, and account reference;
  optional counterparty/merchant. Explicit message currency overrides configured
  primary currency. Receipt/acquisition time is host-owned, not model-generated.
  Preserve exact minor-unit money and source-backed evidence.
- Automatically save complete, valid, uniquely resolved, non-duplicate results.
  Valid `none` creates no transaction or Review. Uncertainty, invalid extraction,
  unresolved accounts, and failures remain reviewable/retryable, not false `none`.
- First unknown account goes to Review. Owner confirmation can atomically reuse
  or create an account/alias and save the transaction. Later messages can resolve
  automatically. Missing counterparty alone must not block saving.
- Supported scope is one transaction per SMS. Do not build a multi-event extractor.
- Review shows original evidence and independently valid partial fields. Keep
  analyzer suggestions separate from model output; support source-backed owner
  correction and atomic confirmation. Corrections remain local feedback until
  explicit export/adjudication, not automatic approved training truth.

### Two duplicate checks, not one late check

1. **Same SMS before inference:** use reliable source identity and durable state
   to reuse the outcome or resume work. No unnecessary model call, second ledger
   entry, or duplicate Review case. Identical text alone is not reliable identity.
   An explicit owner-requested extraction retry is distinct from ingestion replay.
2. **Different SMS, possibly same payment after extraction:** compare grounded
   fields/resolved account and available timing/counterparty context. Flag credible
   uncertainty for Review rather than silently merge/delete. Same amount alone is
   insufficient. Offer a duplicate-versus-separate-payment resolution. Exact match
   rules and time windows remain to be specified/tested, not silently assumed.

Retain atomic writes and database uniqueness as final race/retry protection.

### Evaluation is a first-class deliverable

- Build one Python-based replica of the current end-to-end app behavior, with
  platform runtime adapters, datasets, isolated account/ledger/Review state, shared
  scoring, and reproducible configuration. Do not require app deployment for the
  normal prompt/model comparison loop.
- Android-target lane: retain useful local GGUF/llama.cpp and existing evaluation
  framework integration, including `lm-evaluation-harness` where appropriate.
  Audit reuse; do not mandate a new framework or preserve obsolete flows for it.
- Apple lane: use **Apple's official Python Foundation Models SDK on the Mac**.
  A Swift/Xcode runner is **not** the required evaluation architecture.
- Match each app's filtering, prompt/request, output protocol, generation settings
  where supported, parsing, grounding, normalization, accounts, duplicates, and
  routing. Test deterministic Python/Kotlin/Swift parity with the same supplied
  response and state; separate model calls need not produce identical responses.
  Document API differences rather than claiming a different protocol is parity.
- Include stateful sequences: unknown account, explicit simulated confirmation,
  subsequent auto-save, same-SMS replay, possible duplicate, interruption/retry.
  Simulated owner actions are not model successes. Define fresh run, resume,
  partial output, and changed-configuration rejection; keep per-run state isolated.
- Score **all input messages**, including filtered rows: false filter exclusions,
  avoided model calls, classification/field/grounding quality, correct/incorrect
  saves, missed transactions, Review rate/useful partial fields, account/duplicate
  behavior, failures, and host stage timings. Grounding alone does not prove the
  model selected the right amount or transaction semantics.
- Use a fixed labelled development cohort for tuning and a protected held-out
  cohort for quality claims. The 203-row regression fixture is not fresh gold.
  Keep private inputs/outputs local and ignored; do not use hosted inference.
- **The evaluation dataset is not ready. The owner will annotate the data using
  the existing workbench later.** Keep that annotation workflow available and
  protect work-in-progress labels, revisions, source mappings, keys and backups.
  Initial filter/unit/parity and fake-runtime state tests may use invented
  fixtures now; they do not establish real-data filter recall, model quality or
  end-to-end pipeline quality. Dataset-backed pipeline evaluation follows owner
  annotation and explicit cohort readiness. Do not fabricate gold, use model
  outputs as human labels, or make preliminary work wait for the full dataset.
- Improvement loop: baseline -> controlled change -> compare -> adopt in native
  app -> parity check. Desktop quality/timing is not phone speed, battery,
  background execution, or native storage/device acceptance evidence.

### Documentation end state

Keep a small useful set: README/getting started, contributor/agent rules, shared
architecture, evaluation guide, and necessary native build/device/privacy/release
instructions. Consolidate duplicates; remove obsolete plans/status reports and
fix links/tooling references. Review legal/license obligations separately. Do not
replace old document sprawl with a new document per session or version.

## 2. Verified starting points, not completion claims

The source observations below came from these **unmerged feature tips**, not
`main`. Preserve and inspect them during the audit:

| Repository | Starting checkout branch | Observed commit |
| --- | --- | --- |
| Intelligence | `codex/workbench-eval-set-filter` | `0fd9b8aece174f28f2b390d8d7fe99bc0ebc58fe` |
| Android | `codex/feat-native-sms-v3` | `54a8aa7ab9c0c4bc4c640c5534d7e637d27eecfd` |
| iOS | `codex/feat-native-sms-integration` | `050010e78afff3712750c67e206833df11170415` |

- Android `PipelineService` uses the automatic direct-extraction coordinator.
  `OnboardingService` and `HomeSyncManager` still apply old `SmsFilterPipeline`
  rules; the realtime `SmsParserWorker`/current coordinator do not use that same
  terminal prefilter. Filtering is **inconsistent**, not entirely absent.
- Android source-identity/atomic-save protections exist. The payment fingerprint
  includes exact receipt time, so it is not a complete separate-alert duplicate
  solution. Review account creation and partial-evidence handling are useful work.
- iOS's `SmsV4ProcessingCoordinator` is still review-only. Its
  `FoundationSmsExtractor` requests raw text; do not assume guided-schema parity.
- Intelligence `src/pocketfinancer_sms/evaluation.py` (`evaluate-extractor`)
  implements GGUF extraction/scoring, not the full stateful product flow.
  `apple_evaluation.py` (`evaluate-apple-fm`) has an official Python SDK adapter
  restricted to synthetic cases and a different guided protocol from the Swift
  app. Live Mac results were not verified in this session.
- `scripts/evaluate_lfm25_android_gguf.py` reflects an older Android contract.
  Audit useful runtime/framework pieces without treating that old flow as current.

No new app, model, emulator, or device verification supports this bootstrap.

**Unfinished Review UI preservation reference (2026-11-07):** Android #42's
`codex/feat-native-sms-v3` contains
`app/src/main/java/com/pocketfinancer/ui/review/{ReviewInboxScreen,ReviewDetailScreen,ReviewViewModel,GroundedReviewContent,EvidenceSelectionText}.kt`.
Its history includes `f6079ff` (partial-evidence Review), `f7f5301` (Review/settings
alignment) and `cc1eb79` (field contrast); the partial-evidence and contrast
commits were verified as ancestors of `54a8aa7`. Keep this unfinished work present
in the integration candidate with its supporting repository/grounding/tests.
iOS #8 also contains `GroundedReviewCorrectionView.swift`, `ReviewCorrectionView.swift`
and `EvidenceSelectionTextView.swift`; retain useful native Review work there as
well. These are verified preservation locations, not proof that a specific one
is the exact revision the owner recalls. If another branch/revision contains
additional owner UI changes, locate and carry those forward before any affected
removal; do not assume the current checkout or these filenames are exhaustive.

## 3. Open PR inventory

Metadata, paginated incremental file diffs, commit graphs, reviews, review threads,
and head-SHA checks rechecked through GitHub on **2026-11-05**. Remote heads were
matched to freshly fetched refs. The following inventory remains open; proposed
dispositions below are **not** merge/closure authorization. Checks are observations
of existing heads, not verification of the proposed cleanup.

| Repo / PR | Head branch @ observed SHA | Base | Scope |
| --- | --- | --- | --- |
| Intelligence [#5](https://github.com/pocket-financer/pocket-financer-intelligence/pull/5) | `codex/feat-sms-processing-foundation` @ `11c84903026b` | `main` | Foundation/workbench |
| Intelligence [#6](https://github.com/pocket-financer/pocket-financer-intelligence/pull/6) | `codex/slm-primary-sms-extraction` @ `4c80dd5cb911` | `main` | Direct extraction/routing |
| Intelligence [#7](https://github.com/pocket-financer/pocket-financer-intelligence/pull/7) | `codex/sms-model-quality-eval` @ `bc5f32a10c2f` | #6 branch | Evaluation lanes |
| Intelligence [#8](https://github.com/pocket-financer/pocket-financer-intelligence/pull/8) | `codex/workbench-eval-set-filter` @ `0fd9b8aece17` | #7 branch | Workbench saved-message sets |
| Intelligence [#9](https://github.com/pocket-financer/pocket-financer-intelligence/pull/9) | `codex/docs-session-bootstrap` @ `858a7b0cf6eb` | #8 branch | Access/session guidance |
| Android [#40](https://github.com/pocket-financer/pocket-financer-android/pull/40) (draft) | `codex/feat-native-sms-integration` @ `cd4f135ed4c7` | `main` | Earlier integration |
| Android [#42](https://github.com/pocket-financer/pocket-financer-android/pull/42) | `codex/feat-native-sms-v3` @ `54a8aa7ab9c0` | `main` | Automatic routing/review |
| Android [#45](https://github.com/pocket-financer/pocket-financer-android/pull/45) | `codex/docs-session-bootstrap` @ `e118afa3e015` | #42 branch | Access/session guidance |
| iOS [#8](https://github.com/pocket-financer/pocket-financer-ios/pull/8) (draft) | `codex/feat-native-sms-integration` @ `050010e78aff` | `main` | Native integration |
| iOS [#9](https://github.com/pocket-financer/pocket-financer-ios/pull/9) | `codex/docs-session-bootstrap` @ `93dd70c7ce22` | #8 branch | Access/session guidance |

Leave Android dependency PRs #44/#46 and iOS dependency PR #10/release PR #3
separate. Never merge a release PR or publish as a cleanup side effect.

For each relevant PR, record retain/update, consolidate, or supersede with the
exact destination commits/files and reason. Inspect incremental diffs against its
own base and ancestry/overlap with other PRs; check review comments and checks.
Do not merge everything just to simplify the graph. Do not discard useful work
by considering only `main`. Close/delete nothing until preservation and the
integration strategy are reviewed. Keep focused PRs; no direct pushes to `main`,
force pushes, history rewriting, or unapproved publication.

## 4. Audit findings and proposed dispositions

### Graph and overlap evidence

`git merge-base --is-ancestor`, `rev-list --left-right --count`, incremental
`diff base...head`, direct tip-to-tip diffs, and `log --cherry-mark` establish:

- **Intelligence:** main `7850964` -> #5 `11c8490` -> shared extractor history
  through `d9388be`. #6 `4c80dd5` and #7 `bc5f32a` then diverge (2 left / 4
  right commits). #7 is **not** a literal descendant of the #6 tip: #6's
  `cf308b0` / `4c80dd5` are patch-equivalent to #7's `079807e` / `bc5f32a`
  (encrypted WSL workbench loading and back-navigation fixes). The remaining
  #7 commits `dda5e73` / `fee58e7` add/fix evaluation. Direct tip diff is 13
  evaluation-related files, not lost extractor work. #8 is two commits beyond
  #7: saved-message sets `c5c0e3b` and repository rename `0fd9b8a`.
- Intelligence #9 forked at `c5c0e3b`; it has `4711a37` / `858a7b0` but lacks
  #8's rename. Its own merge-base diff is six guidance files, not the whole
  feature stack. #11 is an independent two-commit docs branch from main; its
  `AGENTS.md` and `docs/README.md` overlap the feature/guidance edits.
- **Android:** main `a58fcad` -> all 12 commits of #40 `cd4f135` -> all 31
  main-relative commits of #42 `54a8aa7` -> three guidance commits in #45
  `e118afa`. Thus #40 has **no unique commits outside #42**. #48 `ca25105`
  is independent/main-based and touches the same `AGENTS.md` insertion area
  as #45. #42 includes native v3-v6 bundles, not just one new coordinator.
- **iOS:** main `c79b7b5` -> 17 commits in #8 `050010e`. #9 forked at
  `20891a4`; its two guidance commits `9b83002` / `93dd70c` omit #8's later
  rename. #12 `1343ab7` is independent/main-based and overlaps #9 in
  `AGENTS.md`. Preserve the rename plus the useful guidance, not one full file
  from either branch at the expense of the other.
- Incremental PR sizes: intelligence #5 138 files, #6 211, #7 19, #8 25,
  #9 6, #11 3; Android #40 114, #42 369, #45 3, #48 1; iOS #8 180,
  #9 1, #12 1. #6 and #42 are broad cumulative integrations; merging their
  predecessors separately adds review/graph churn without preserving extra work.

### Proposed PR destination ledger

Destinations below identify **existing source commits/files**, not nonexistent
future integration commits. New destination SHAs must be recorded when approved
integration work is actually committed. No PR was changed in this audit.

| PR | Proposal | Preservation destination and reason |
| --- | --- | --- |
| Intelligence #5 | Consolidate/supersede after preservation verified | All commits are ancestors of #8 `0fd9b8a`; preserve `src/pocketfinancer_sms/{corpus,workbench}`, currency/provenance/labels and safety tests there. Do not merge the old candidate-selector product as a separate current flow. |
| Intelligence #6 | Consolidate/supersede after preservation verified | Use #8's extractor/routing/configuration/Review files. Its two final workbench fixes are patch-equivalent in #7 and hence #8; do not cherry-pick duplicates. |
| Intelligence #7 | Consolidate into #8, retaining evaluation | #8 contains its exact tip `bc5f32a`; keep `evaluation.py`, `apple_evaluation.py`, manifest, optional SDK dependency, CLI and tests. Do not remove evaluation because the feature stack is consolidated. |
| Intelligence #8 | Retain/update as integration candidate | Baseline `0fd9b8a` preserves foundation, extractor, both Python lanes, saved-set filtering and rename. Its current base is #7, not main; proposed later main-target integration must disclose the whole inherited diff and pass new gates. Not ready to merge unchanged. |
| Intelligence #9 | Consolidate guidance, then supersede | Apply only useful guidance from `4711a37` / `858a7b0` to baseline `AGENTS.md`, `README.md`, `docs/README.md`, architecture and the WSL access guide. Fold roadmap-only changes into architecture; retain #8 rename and #11 cleanup authority. |
| Intelligence #11 | Retain/update, docs-only coordination | Current branch `5c37cb9` plus this plan-only audit commit. Carry its plan and cleanup pointers into the integration baseline without replacing the feature docs index with main's older index. |
| Android #40 (draft) | Supersede after preservation verified | Exact tip `cd4f135` is contained in #42 `54a8aa7`; preserve admission, recovery, claims, currency and Review foundations there. No independent integration of #40 needed. |
| Android #42 | Retain/update as integration candidate | `54a8aa7`: preserve automatic routing, transactional save, partial Review/account creation, current grammar snapshot, runtime safeguards and tests. Fix inconsistent filtering and replay gaps before claiming target compliance. |
| Android #45 | Consolidate guidance, then supersede | All three commits beyond #42 are docs-only: `AGENTS.md`, `copilot-instructions.md`, `docs/sms-processing-next-steps.md`. Move access/build guidance into contributor/native guidance; obsolete next-step prose is not another task list. |
| Android #48 | Retain docs-only coordination | Carry `ca25105`'s cleanup pointer alongside #45 guidance; no feature content exists here. |
| Android #44 | Keep separate | `243d39f`: two workflow action-version changes, not a pipeline dependency; revalidate independently. |
| Android #46 | Keep separate, blocked | `32e8f70`: version catalog and Gradle wrapper changes; failing unit-test job. Do not absorb dependency churn into cleanup or assert its cause without logs. |
| iOS #8 (draft) | Retain/update, blocked | `050010e`: native ingestion/store/recovery, direct Foundation Models transport, evidence/Review UI, currency and tests are useful. Review-only routing and absent terminal prefilter are not the approved finished flow. |
| iOS #9 | Consolidate guidance, then supersede | Only incremental `AGENTS.md` access guidance from `9b83002` / `93dd70c`; preserve #8 rename and #12 authority pointer. Do not replace with this divergent branch wholesale. |
| iOS #12 | Retain docs-only coordination | Carry `1343ab7` cleanup pointer into #8-based guidance, not its older runtime baseline. |
| iOS #10 | Keep separate | `1d54bba`: one upload-artifact action change; no cleanup dependency. |
| iOS #3 (draft release) | Leave untouched | Release Please's `fade7b5`, four release/version files; no merge, release, manual version edits or publication. |

### Reviews and checks

This is a bounded integration/deletion audit, not a line-by-line correctness
approval of tens of thousands of inherited additions. Diff inventories were
checked across all relevant PRs; source/test inspection concentrated on intake,
model transport, validation, settlement, annotation protection and evaluation.
The selected candidate PRs still need focused implementation review and native
acceptance before merge.

- Intelligence #5's schema `$ref` review thread is resolved/outdated. Fix
  `11c8490` is in the selected baseline: corpus schema embeds `$defs/analysis`;
  `tests/sms_processing/test_json_contracts.py` covers standalone validation.
- iOS #8's three posted review threads are resolved/outdated. Fix `182a7b9`
  is contained in the tip: writable error alert bindings, explicit `@MainActor`
  tasks with `defer`, and corrected drain-loop comment remain in
  `ReviewCorrectionView.swift`, `TransactionDetailView.swift`, `AppRootView.swift`.
  This does not resolve the currently failing test job.
- Other implementation/guidance/handoff reviews mostly report Copilot quota
  exhaustion. There are **no submitted approving reviews** in the fetched review
  records; quota messages and `mergeable=clean` are not correctness approval.
  Issue-comment lists were empty. Review-comment and review-thread APIs were both
  checked, including resolved replies rather than treating old comments as open.

| Repo / PRs | Existing head check-run observations |
| --- | --- |
| Intelligence #5/#6/#7/#8/#9/#11 | All returned Python 3.11 jobs succeeded; returned GitGuardian jobs also succeeded. Duplicate check names reflect separate runs, not extra local testing. |
| Android #40/#42/#44/#48 | Build/lint/unit job succeeded. #48 completed during this audit (initially in progress). Returned GitGuardian jobs succeeded. |
| Android #45 | No check-runs returned; **not a passed build**. |
| Android #46 | Build/lint/unit job failed at `Run all debug unit tests`, exit 1; lint/APK steps skipped. [Run 38181207167](https://github.com/pocket-financer/pocket-financer-android/actions/runs/38181207167). |
| iOS #8 | Title checks succeeded; format and Release build steps succeeded, `Run unit and UI smoke tests` failed, exit 65. [Run 41490942618](https://github.com/pocket-financer/pocket-financer-ios/actions/runs/41490942618). |
| iOS #9 | Title/security succeeded; no format/build/test job returned for that head. |
| iOS #10/#12/#3 | Returned title and format/build/test jobs succeeded; returned security jobs succeeded. Release #3 remains out of scope. |

Legacy combined commit-status endpoints returned `pending` with no status
contexts; this is **not** evidence that the completed check-runs above are still
running. Do not infer branch-protection approval from either endpoint. Job-step
metadata and annotations were readable; downloading the two failed job logs
returned **HTTP 401**. Annotations contain only exit codes/runner notices, not a
root cause. Obtain test reports/log access or reproduce on the native toolchain
before diagnosing or weakening any test.

## 5. Concrete cleanup map (proposal, no files removed)

Paths in each table are repository-relative to the **feature baseline**, not the
main-based handoff checkout. K = keep; S = simplify/consolidate; R = remove after
replacement, reference checks and owner approval. Braces/globs denote the named
families, not permission to delete unrelated files. K preserves useful behavior,
not every old version or a claim that implementation is complete. No R row may
remove original evidence, labels, correction history, source mappings or keys.

### Intelligence: code, tests, assets and dependencies

| Action | Paths / useful work | Destination, dependents and removal gate |
| --- | --- | --- |
| K | `src/pocketfinancer_sms/corpus/{manifest,grouping,pools,reports}.py`, `provenance.py`, `labels.py`, `feedback.py`; `tests/sms_processing/{test_corpus_builder,test_labels_and_trace}.py` | One private corpus/annotation workflow. Keep held-out grouping, provenance, append-only human revisions, weak-vs-gold separation and fail-closed paths/permissions; evaluation and workbench depend on these. |
| K/S | `src/pocketfinancer_sms/workbench/{store,secure_store,service,web,native_import,source_sets}.py`, `workbench/assets/*`; `tests/sms_processing/{test_workbench,test_workbench_web,test_native_workbench_import}.py`, `tests/test_sms_workbench_assets.py` | Preserve loopback/token protection, SQLCipher/key handling, backup/restore, blind held-out review, saved-source sets and back-navigation fixes. One current UI; consolidate candidate preview/version branches only after historical labels/native feedback remain readable. No store conversion/reset in cleanup by implication. The owner will use this workbench for later annotation: preserve the usable workflow and unfinished labels; do not retire it before an owner-approved replacement is verified. |
| K/S | `extractor.py`, `currency.py`, `structural_text.py`, `account_resolution.py`, `processing_v3.py`, `analyzer.py`, `types.py`, `profiles.py`, `trace.py` under `src/pocketfinancer_sms/` | Consolidate into one current host flow. Keep exact minor units, scalar/UTF-8 grounding, optional counterparty, explicit-currency precedence, account aliases, independent partial fields and failure/retry fencing. Analyzer is advisory; add narrow prefilter and pre-model source-outcome lookup. Current Python duplicate lookup occurs after model work and exact-time fingerprinting is insufficient for different alerts. |
| K/S | `tests/sms_processing/test_extractor_{contracts,pipeline,state,evaluation}.py`, `test_successor_routing.py`; `tests/sms_processing/golden/{extractor-v1,native-v5}` | Keep semantic/grounding/failure/none/partial-field tests. Update assertions for the approved flow; add same-source zero-call replay and stateful account-confirm/save/duplicate/interruption sequences. Do not retain review-only rollout as an alternative product or remove assertions merely to pass cleanup. |
| S | `src/pocketfinancer_sms/{configuration,configuration_v3,configuration_v4,configuration_v5,configuration_v6}.py`, `configs/sms_processing/contracts/{v2,v3,v4,v5,v6}`, `policies/native-persistence-v*.json` | One current configuration/result/Review/feedback contract and explicit runtime capability settings. Preserve stored snapshots as evidence; current retry may report old executable adapter unavailable and offer explicit current-config retry. Migrate/read historical annotations first; remove runnable old variants only after CLI/workbench/native import references move. Do not edit hashes to pretend old evidence used the new contract. |
| K/S | `src/pocketfinancer_sms/evaluation.py`, `apple_evaluation.py`, `configs/sms_processing/evaluations/*`, `requirements-apple-fm.txt`, CLI `evaluate-extractor` / `evaluate-apple-fm`; `test_apple_evaluation.py` | Retain local GGUF and **official `apple_fm_sdk` Python adapter on Mac**, shared scoring, locks, checkpoints and hash mismatch refusal. Consolidate duplicate orchestration behind platform adapters, not a Swift runner. Both currently evaluate isolated extractions, not stateful end-to-end saves/Review. Apple is synthetic-only and guided-flat; Swift uses raw text. Match the native request/transport explicitly and expand private local cohorts under the existing privacy boundary. No hosted fallback. |
| S | `configs/sms_processing/prompts/{sms-extractor-v1,apple-fm-sms-extractor-v1}.txt`, `configs/sms_processing/grammars/sms-extractor-v1.gbnf`, `contracts/v3/{sms-extractor*,extractor-validation-profile*,account-resolution-profile*,canonical-label*}` | One semantic protocol with documented transport capabilities. Android current config/6 snapshots default-off GBNF; GGUF evaluator currently always constrains grammar. Align settings rather than falsely report parity. Preserve prompt/schema/config/runtime hashes and immutable old evidence bindings while replacing active assets deliberately. |
| S -> R | `scripts/build_native_contract_release*.py`, `scripts/package_native_contract_v*.py`, `configs/sms_processing/contracts/releases/{native-integration-v*.json,v2,v3,v4,v5,v6}`, corresponding `test_native_contract*` / `test_processing_config_v*.py` | Consolidate packaging/verification into one current generator and neutral integrity test; then remove superseded per-version builders, asset copies and tests asserting only their continued presence. Native apps and snapshots consume their hashes; migrate those references first, retain data-read compatibility where necessary. |
| R | `src/pocketfinancer_sms/selector.py`, `triage.py`, candidate-only portions of `persistence.py`; grounded-selector schemas/prompt and old selector golden tests | Remove candidate-ID selection/reconstruction as a runnable alternative after workbench preview, native imports, core tests and current extractor imports are disentangled. Retain reusable types/evidence helpers in current code; candidate absence must never become an extraction allowlist. |
| S | `DATA/llamacpp_model.py`, `run_gguf_eval.py`, `DATA/{sms_extraction.yaml,utils.py,sms_extraction.gbnf}`, `requirements.txt` pinned `lm_eval[gguf]` | Preserve out-of-tree harness registration, local llama.cpp runtime/tokenizer/chat-template and metric integration where useful. Route a current task through the same host/scorer; retire old two-pass thinking/task protocol, permissive parsing and 203-row-only comparison entry point once replaced. Do not delete the harness as unused just because new `evaluation.py` imports `llama_cpp` directly. No vendored upstream edits. |
| S -> R | `scripts/evaluate_lfm25_android_gguf.py`, `lfm25/{android_contract,android_profile_sync,android_baseline,android_protocol_lab,android_runtime_evidence}.py`, `configs/contracts/pocketfinancer-android-*.json`, `configs/baselines/*`, `configs/experiments/*phase-d*`; their old profile/lab tests | Extract useful runtime/provenance/chat-format checks into current evaluation. Remove old Android-profile/Phase C-D runners and locks as active paths after callers, safety/catalog tests and docs are updated. Existing results remain historical, not current app-parity evidence. |
| S -> R | `lfm25/{annotation_*,workbench_v2,semantic_v2,candidate_protocol*,candidate_curriculum,candidate_sft,candidates,component_evaluation,evaluation_v2}.py`, legacy annotation assets, `scripts/run_lfm25_annotation_workbench*.py`, `run_pocketfinancer_workbench_v2.py`, candidate-only builders/evaluators, `configs/{history,programs}/*`, candidate/semantic/workbench-v2 contracts | First preserve import/projection of original labels, human corrections and provenance in the current workbench/corpus. Then remove duplicate annotation frontends, candidate/semantic pipelines and their exclusive tests. Reuse coverage/scoring utilities where current evaluation needs them; no deletion of legacy private databases to simplify this code. |
| K/S | `lfm25/{private_data,blinded_review,public_candidate,training_provenance,training_loss,provenance,metrics}.py`, `scripts/{prepare_lfm25_private_data,review_lfm25_blinded_test,audit_lfm25_public_candidate,probe_lfm25_memorization,train_lfm25_lora,merge_lfm25_lora}.py`, conversion/setup scripts and relevant tests | Keep useful local dataset/improvement, leakage/privacy/license, training/merge/conversion/provenance primitives; decouple model identity and old schema from machinery. One current label projection and improvement workflow, not automatic promotion of corrections. No training run, new model choice, publication or data rebuild authorized here. |
| S -> R | `lfm25/pipeline.py`, `scripts/run_pocketfinancer_pipeline.py`, `scripts/run_lfm25_experiments.py`, `scripts/evaluate_lfm25_{hf,gguf,candidate_hf,candidate_protocol_v1_hf}.py`, `configs/{pipelines,lfm25,models,experiments}`, `run_gguf_eval.py`'s old CLI modes; exclusive pipeline/catalog tests | Consolidate reusable local runtime/training mechanics above; remove obsolete model-slate/short-prompt/candidate/Phase D orchestration and configs after current commands replace consumers. Do not retain runnable experiments just for archival. Do not remove the only working current GGUF comparison path before its replacement is tested. |
| K/S | `requirements-ci.txt`, `pyproject.toml`, `.github/workflows/ci.yml`, `scripts/check_repo_safety.py`, `.gitignore`, `tests/test_repo_safety.py` | Keep lightweight pytest/Ruff/ShellCheck/jsonschema, src coverage and privacy checks; update tracked-path/catalog expectations for approved removals rather than weakening guards. `rapidfuzz` currently supports legacy public-candidate leakage review: retain until that useful review is migrated or its replacement verified. |
| S | `requirements.txt`, `requirements-training.txt`, `requirements-apple-fm.txt`; optional workbench imports | Separate lightweight core/tests, local GGUF/harness, optional ML and Mac SDK/workbench extras. Keep `lm_eval`, `llama-cpp-python`, `apple-fm-sdk`; SQLCipher/keyring and `cryptography` serve protected workbench/import, not dead dependencies. HF/accelerate/datasets/PEFT/quantization/tokenizer packages belong only to retained local improvement tooling. Remove TRL/sentencepiece/other pins only after a current call-site/tokenizer/build check; import absence alone is not proof. |
| K/S | `export_sms.py`, `find_sms_db.py` | Retain useful local archive discovery/import mechanics only within the current corpus workflow, with safe paths/logging and no remote processing. Do not run them against original sources for this audit. Removing duplicate scripts later cannot remove or rewrite the original archive or source mappings. |
| K (protected) | `DATA/extraction_ds.jsonl`; ignored `PRIVATE_DATA/`, `PUBLIC_CANDIDATE/`, `MODELS/`, `TRAINING_ARTIFACTS/`, `RESULTS/`, `UPSTREAM/`; original archive locations and key vaults | Do not inspect/publish/reset them during this audit. Preserve 203-row fixture as regression only, never training/fresh gold. No private directory's generated-looking name authorizes deleting human annotations, source mappings, encryption keys or backups. No exact reset inventory is approved. |

### Android: native flow, tests and build

For compact paths below: `P` = `pipeline/src/main/java/com/pocketfinancer/pipeline`,
`D` = `data/src/main/java/com/pocketfinancer/data`, `I` =
`inference/src/main/java/com/pocketfinancer/inference`, `A` =
`app/src/main/java/com/pocketfinancer`.

| Action | Paths / useful work | Destination, dependents and removal gate |
| --- | --- | --- |
| K | `sms/src/main/java/com/pocketfinancer/sms/SmsReceiver.kt`, `SmsReader.kt`, source identity; `D/repository/SmsIngestionRepository.kt`, `P/{SmsParserWorker,SmsWorkController,AutomaticProcessingPreferences}.kt` | Preserve opt-in realtime intake, opaque WorkManager payloads, encrypted durable outbox, reconciliation, claims/cancellation and source metadata. Unify with manual/historical intake without moving model work into receiver. Keep receiver/worker/ingestion/recovery tests. |
| S | `P/PipelineService.kt`, `P/sms/DefaultSmsV5ProcessingCoordinator.kt`, `A/ui/{home/HomeSyncManager,onboarding/OnboardingService}.kt`, `P/SmsFilterPipeline.kt`; filter-injected Home/Transactions ViewModels | One admission/outcome lookup -> narrow prefilter -> snapshot/analyzer/extractor route. Old filter rejects missing amount/account/verb and any OTP keyword (even a completed transaction); remove those rejection rules. Both historical/manual services call it, realtime/current coordinator do not. Remove duplicate callers only after shared boundary coverage, not by disabling filtering. |
| S | `P/sms/SmsV5OperationSnapshotFactory.kt`, `D/repository/SmsProcessingStore.kt`, `D/db/dao/{SmsProcessingDao,TransactionDao}.kt` | Preserve durable state/atomic uniqueness; add a pre-model source/outcome lookup for all entrypoints. Factory currently generates a fresh operation and stable-event UUID for ordinary calls; coordinator's settled lookup is operation-specific. Existing outbox suppression/atomic save is useful but **does not prove all-entrypoint zero-call replay**. Retain explicit owner retry distinction and test concurrent/restart/replay states. |
| K/S | `D/repository/{SmsReviewRepository,SmsReviewGrounding,GroundedAccountResolver,CurrencyScaleRegistry}.kt`, `A/ui/review/*`, `P/sms/SmsExtractor{Contracts,Validator,Normalizer}.kt`, `UnicodeScalarSpan.kt` | Keep exact money, source-backed independently valid fields, account reuse/create, transactional confirmation/rollback and no second save. Preserve the unfinished Review UI identified in section 2; consolidate backend v4/v5 branches without discarding or silently replacing the owner's page work. Missing counterparty alone cannot require Review. Duplicate resolution must not silently merge separate payments. |
| K | `D/db/{AppDatabase,entity/SmsProcessingEntities,entity/TransactionEntity}.kt`, `D/repository/SmsProcessingStore.kt`, transaction revisions and protected key initialization | Preserve SQLCipher/Keystore protection, ownership fencing, durable receipts, atomic ledger/gate/result writes and history. Keep `data/schemas/com.pocketfinancer.data.db.AppDatabase/{6,7,8}.json` and migration tests until a separately identified disposable DB reset or migration decision; code cleanup is not permission to erase old Review evidence or keys. |
| K/S | `P/sms/DirectSmsExtractor.kt`, `I/{SlmRuntime,SlmRuntimeCoordinator,LlamaEngine,DirectCandidateSelector,NonThinkingOutputGuard,CandidateSelectorRuntimeProfile}.kt`, `inference/src/main/cpp/{llama_jni.cpp,CMakeLists.txt}` | Keep serialized lease/native access, stop/cancel/output bounds, one direct non-thinking call, cache/provenance and context settings. Rename selector transport types if useful, but do **not** delete `DirectCandidateSelector` wholesale: current direct extractor uses it. Native/transport/timeout tests remain. |
| S -> R | `P/sms/{DefaultSmsProcessingCoordinator,DefaultSmsV4ProcessingCoordinator,SmsOperationSnapshotFactory,SmsV4OperationSnapshotFactory,SmsV4ProcessingJson,GroundedSelectorValidator,SemanticReconstructor,SmsTriageEvaluator,NativeSmsV3Assets,NativeSmsV4Assets,NativeSmsV5Assets,NativeSmsV6Assets}.kt`; `pipeline/src/main/assets/sms_processing/` (including `native-integration-v3` through `native-integration-v6`) | One current coordinator/snapshot/asset verifier; extract shared binding types currently named V3, preserve current config/6 grammar snapshot and required schemas. Remove legacy original-config execution/review-only/candidate routes and old bundles only after `retryReview`, DI, tests and packaged-resource references migrate. Keep old snapshot readable; explicit current-config retry replaces execution of old adapters. |
| R | `P/{PromptBuilder,ExtractionParser}.kt`, `inference/src/main/assets/{system_prompt.txt,few_shot_examples.json,sms_extraction.gbnf}` and exclusive prompt/parser tests | Old two-phase protocol, not current direct route. Confirm remaining diagnostic/native prompt consumers have moved before removal; don't confuse with the current bundled `sms-extractor-v1` prompt/grammar. |
| S | `hardware/.../{DeviceCapabilities,SlmSelector}.kt`, `A/SlmModelSpecs.kt`, `I/SlmModelStorage.kt`, settings/onboarding model surfaces | Keep device feasibility/download/integrity and configured model identity; consolidate speculative family tiers/manual switches and unsupported quality/speed claims. A single processing flow need not hard-code one model. No new model/default choice or removal of all hardware safeguards is authorized. |
| K/S | `data/src/test/.../{SmsAutomaticPersistenceTest,SmsReviewRepositoryV5Test,SmsReviewGroundingTest,GroundedAccountResolverTest,SmsIngestionRepositoryTest}.kt`, pipeline extractor/runtime tests, `data/src/androidTest/java/com/pocketfinancer/data/repository/SmsProcessingRecoveryDeviceTest.kt`, app Review/processing Compose tests | Preserve rollback, exactly-once ledger, account-confirm and partial-evidence cases; port v4/v5 tests into current behavior rather than duplicate version suites. Add zero-call same-source replay, retry, prefilter across all intake paths and missing-counterparty auto-save tests. Deleted historical Home/import tests in #42 need equivalent cancellation/settlement coverage, not blind restoration of removed architecture. |
| K/S | `gradle/libs.versions.toml`, module `build.gradle.kts`, Gradle wrapper, `.github/workflows/{ci,release}.yml` | Room/SQLCipher/Keystore, WorkManager, Hilt, Coroutines, native llama.cpp and Compose/test dependencies all have live uses. No dependency is proven removable wholesale. Remove orphan catalog aliases/version-only test pins after call-site/build verification; keep #44/#46 separate and release automation unchanged. |
| R (after consumer check) | `android_mock_ui/`, obsolete diagnostic-only assets, `.hermes/plans/*.md` | Native app UI stays. Check build/source references before removing mock/prototype assets; keep sanitized emulator/SMS injection tooling needed for acceptance. No generated app DB, inbox, key or model file deletion. |

### iOS: native flow, tests and configuration

Here `S` in a path means `PocketFinancer/Services/SmsProcessing` (the Action
column's S still means simplify). Unprefixed `App/`, `Data/`, `Domain/`,
`Features/`, `Intents/`, `Resources/`, `Services/` and `Shared/` paths in this table
are under `PocketFinancer/`; other paths are repository-relative.

| Action | Paths / useful work | Destination, dependents and removal gate |
| --- | --- | --- |
| K/S | `PocketFinancer/Services/AlertIngestionService.swift`, `Intents/ImportTransactionAlertIntent.swift`, `Features/Transactions/ManualAlertImportView.swift`, `App/AppRootView.swift` | Keep evidence-first App Intent boundary, pending unavailable-model behavior and bounded foreground recovery. Add shared narrow prefilter before inference. Ingestion currently creates a new alert and flags content/time heuristics for Review, not reliable same-source outcome reuse; distinguish reliable connector identity from heuristics, do not turn identical bodies into identity. |
| S | `S/{SmsV4ProcessingCoordinator,SmsV4OperationSnapshotFactory,FoundationSmsExtractor,SmsV4ProcessingJSON}.swift` | Starting point for one direct Foundation Models flow. Retain fresh local session, execution gate, greedy/512-token settings and strict host validation; evolve review-only posted routing to approved automatic save only with grounded fields, unique account and duplicate safety. Preserve unavailable/truncated/error reviewability. Python SDK lane must reproduce Swift request/transport, not replace it with a Swift evaluator. |
| K/S | `S/{SmsExtractorContracts,SmsExtractorValidator,SmsExtractorNormalizer,SmsReviewProjection,GroundedAccountResolver,CurrencyProfileRegistry,UnicodeScalarSpan,StructuralSmsAnalyzer}.swift`; `Features/Transactions/{GroundedReviewCorrectionView,EvidenceSelectionTextView,ReviewCorrectionView,TransactionDetailView}.swift` | Keep scalar/source grounding, exact currency money, owner correction and valid partial fields. Preserve unfinished native Review UI; backend/candidate cleanup must not silently remove or replace it. Disallow mandatory optional-counterparty checks. Add atomic account/alias confirm and subsequent automatic routing tests rather than assume Android's behavior exists on iOS. |
| K/S | `PocketFinancer/Data/{SmsProcessingStore,SmsProcessingModels,PocketFinancerSchema,AppDatabase,SecureStoreFilePolicy}.swift`, revision/feedback records | Keep evidence/recovery/fencing, atomic correction history, file protection and schema reading. No deletion/migration of human corrections to discard version adapters. Simplify current operation contracts only after preserving/read-testing persisted Review and ledger state. |
| S -> R | `S/{SmsProcessingCoordinator,SmsOperationSnapshotFactory,DirectCandidateSelector,GroundedSelectorValidator,SemanticReconstructor,SmsTriageEvaluator,AutomaticPersistenceGate,NativeSmsV3Assets,NativeSmsV4Assets}.swift`; `Resources/{SmsProcessing,NativeSmsV3.bundle,NativeSmsV4.bundle}` | Consolidate verifier/binding types, currency, current direct extractor assets into one bundle. Old original-config retry and diagnostics reference candidate selector; V4 asset verification reuses V3 binding types. Move those consumers/shared types before deleting obsolete executors/bundles. Remove candidate-only tests/resources only after current tests replace integrity/grounding coverage. |
| S -> R | `Services/FoundationModel{TransactionParser,ExtractionContract}.swift`, `Domain/{AlertFilter,EvidenceValidator,ParsedAlertDraft,AlertDateParser,AmountParser}.swift`, `Data/{DeterministicFilterRun,ExtractionRun,StructuredGenerationSnapshot}.swift` | Retire obsolete extraction generation/filter rules and diagnostic re-computation; preserve needed model availability, current diagnostics, exact-money/grounding utilities and old stored records until references are migrated. `AlertProcessingDetailView.swift` still re-runs `AlertFilter` for display: do not present that as proof the active v4 ingress filtered. |
| K/S | `PocketFinancerTests/{AlertIngestionServiceTests,AlertSourceIdentityTests,AppDatabaseStartupTests,SmsProcessingStoreTests,SmsExtractorValidatorTests,SmsReviewProjectionTests,GroundedAccountResolverTests,FoundationModelExecutionGateTests}.swift`, `NativeSmsProcessingTests.swift`, parity tests, `PocketFinancerUITests/` | Keep claim/recovery/correction/idempotency/Unicode/currency/UI cases; port selector/shadow/bundle-presence assertions to current deterministic parity. Add same-source replay, no-model exclusions, valid none, unknown-account-confirm-next-auto-save, optional merchant and interrupted retry acceptance. Fix #8's failing tests first; resolved UI review comments are not a green suite. |
| K | `Config/*.xcconfig`, `PocketFinancer.xcodeproj`, `Resources/PrivacyInfo.xcprivacy`, `Shared/PrimaryCurrencySettings.swift`, `Services/FoundationModelExecutionGate.swift`, `scripts/{build,format,test}.sh` | Native SwiftUI/SwiftData/FoundationModels only, iOS 26+, strict concurrency, local-only CI guard. No Swift package dependency was found in the audited project; do not add one or a shipping prerelease API. Keep release/version files and workflows; #10/#3 stay independent. |

### Markdown end state and consolidation destinations

This is a removal **proposal** after useful content/link/legal checks, not a
historical-code retention mandate. Keep current docs truthful as features land;
no claim that proposed future behavior already works. Existing evidence files
outside Markdown and private annotation history are not covered by doc removal.

| Repo / action | Markdown paths | Destination / dependent updates |
| --- | --- | --- |
| All K/S | `README.md`, `AGENTS.md`, `CONTRIBUTING.md`, `.github/pull_request_template.md` | Concise getting started, authority, native/contributor checks and safety. Keep shared-plan pointers during cleanup; native docs link shared design, never Python runtime dependency. Preserve #9/#45 access guidance without duplicate workstation tables in every file. Keep release-owned `CHANGELOG.md` in app repos. |
| Intelligence S | `docs/architecture/{SMS_PROCESSING_ARCHITECTURE,SMS_PROCESSING_DECISION_LOG,CURRENCY_CONTEXT_AND_PROVENANCE,DATA_TAXONOMY_AND_CANONICAL_LABELS,WORKBENCH_REQUIREMENTS_AND_DATA_FLOW,REPOSITORY_LAYOUT}.md`, `docs/contracts/{DIRECT_SMS_EXTRACTOR_CONTRACT,NATIVE_SMS_INTEGRATION_CONTRACT}.md` | One lasting shared `docs/architecture/SMS_PROCESSING_ARCHITECTURE.md` with contract links, currency/identity/Review/account/duplicate semantics and platform gaps. Merge useful decisions and remove redundant source pages only after links/test/catalog references move. |
| Intelligence S | `docs/guides/{SMS_MODEL_QUALITY_EVALUATION,SMS_WORKBENCH_LABELING_GUIDE,ANNOTATION_HANDBOOK_V1,LOCAL_ANNOTATION_WORKBENCH}.md`, `scripts/README.md` | Keep `SMS_MODEL_QUALITY_EVALUATION.md` for full Python stateful comparison, runtime capability/phone limits, cohorts/checkpoints/scoring. Keep one `SMS_WORKBENCH_LABELING_GUIDE.md` for annotation/export/adjudication/private restore. Fold duplicated commands into these and root getting-started; remove obsolete legacy workbench instructions. |
| Intelligence S | `docs/guides/POCKETFINANCER_WINDOWS_WSL_ACCESS.md` from #9 | Keep one useful native-tooling/private-workspace access section in contributor/getting-started guidance; small native pointers instead of repeated workstation-specific agent instructions. Never imply this Linux audit ran on WSL/Mac. |
| Intelligence R | `docs/plans/{CROSS_PLATFORM_SMS_ROADMAP,SMS_EVALUATION_STRATEGY}.md`; `docs/architecture/{MODEL_IMPROVEMENT_ROADMAP,ANDROID_RUNTIME_AUDIT,POCKETFINANCER_ANDROID_BASELINE_PHASE_C,POCKETFINANCER_ANDROID_PROTOCOL_LAB_PHASE_D}.md`; `docs/contracts/GROUNDED_CANDIDATE_SELECTOR_CONTRACT.md` | Extract still-useful parity, provenance and improvement constraints to the two current guides/shared architecture, then remove competing plans and obsolete protocol guidance. `docs/README.md` becomes a short index of surviving pages. |
| Intelligence R | `docs/LFM25_350M_{EXPERIMENT_REPORT,PIPELINE_V2}.md`, `docs/experiments/*.md`, `docs/history/**/*.md` and direct history Markdown, `docs/guides/{EXTRACTION_V2_ANDROID_PHASE_D_LAB,EXTRACTION_V2_WORKBENCH,FINE_TUNING_PRIMER}.md` | Retain only actionable current setup/provenance caveats in evaluation/contributor guidance, then retire obsolete experiment reports/status/catalog/phase docs to Git history. Do not retain executable pipelines solely to support those links. Update `test_project_catalogs.py` and related docs-link checks deliberately, not privacy checks. |
| Intelligence K pending legal review | `docs/lfm25_public_candidate/{LICENSE_AND_DATA_RIGHTS_REVIEW,REVIEW_CHECKLIST,README}.md`, any license/notice obligations | Preserve data-rights/publication gates until legal/provenance review identifies the minimum durable obligation. Cleanup does not grant rights or authorize publication. `docs/CLEANUP_PLAN.md` stays the sole temporary plan until completion. |
| Android S | `copilot-instructions.md`, `docs/{emulator_testing,windows-sms-sync-validation,sms-processing-next-steps}.md` | Keep concise native architecture/build instructions in contributor guidance and sanitized device/emulator validation in `docs/emulator_testing.md`; replace obsolete next-step lists with shared-design links then remove them. Fix `file:///d:/...` links and old two-phase/mandatory regex/quality claims. |
| Android K | `docs/releasing.md`, useful privacy/build/device guidance in README/contributor docs | Native release/Keystore/signing/verification safeguards remain. No release workflow simplification in this slice. |
| Android R | `docs/{design_spec_typography_spacing,tier3_inline_override_elimination,typography_m3_audit_2026-06-26,release-notes-v1.0.0}.md`, `.hermes/plans/*.md`, `android_mock_ui/README.md` | Move genuinely current accessibility/build instructions to contributors, then remove prototype/session/audit/status docs and fix inbound links; do not replace them with more per-session pages. |
| iOS S | `docs/{architecture,processing-transparency,sms-processing-next-steps}.md` | `docs/architecture.md` keeps only native intake/storage/runtime/UI and links to shared semantics; fold durable transparency behavior there and remove duplicate roadmap/status prose. |
| iOS K/S | `docs/{device-validation,privacy-and-security,shortcuts-setup,releasing}.md`, `SECURITY.md` | Keep necessary native device, App Intent setup, privacy/security disclosure and release controls; trim duplicated shared semantics and fix links. No unsupported background execution/automatic-save claims. |

## 6. Integration recommendation and first implementation slice

### Proposed baselines (not executed)

Use the latest audited feature tip **as the preservation baseline**, not a blind
main-based rewrite:

| Repo | Preservation baseline | Guidance/handoff combination |
| --- | --- | --- |
| Intelligence | #8 `0fd9b8aece174f28f2b390d8d7fe99bc0ebc58fe` | Incremental #9 guidance plus #11 cleanup authority/current plan; resolve `AGENTS.md` and docs index by content, preserving rename and feature guidance. No separate cherry-pick of #6's equivalent fixes or #5/#7 ancestry. |
| Android | #42 `54a8aa7ab9c0c4bc4c640c5534d7e637d27eecfd` | Incremental three #45 doc commits plus #48 pointer. #40 already contained; no dependency PR content mixed in. |
| iOS | #8 `050010e78afff3712750c67e206833df11170415` | Incremental #9 `AGENTS.md` guidance plus #12 pointer and retained rename. Explicitly blocked on test failure and current-flow gaps. |

These feature tips are **starting versions to preserve, not versions approved
for merge**. Refresh main/PR refs before implementation. After explicit scope and
branch/review-arrangement approval, make bounded fixes and cleanup on separate
short-lived working branches that retain the useful feature work and combined
guidance, including the unfinished Review UI; leave `main` and the original
feature branches unchanged. Review/test each small change against its recorded
starting version. Separately review the complete inherited diff against refreshed
`main` and pass the agreed checks before proposing integration. **Do not merge
the large feature PRs unchanged merely to establish a baseline**, or merge their
predecessors just to flatten the graph. This does not require finishing every
cleanup item before any integration; it requires an explicitly agreed scope and
acceptance gates for what enters `main`. Any pre-merge dependent branch/PR setup
needs explicit approval rather than silently overriding repo branch rules.
Once the preservation baseline is accepted into `main`, normal focused PRs use
short-lived `codex/<type>-...` branches from refreshed main. Retargeting,
superseding, merging and PR closure remain separately gated; no history rewrite
is proposed. Native handoff PRs remain docs-only. No integration branch was
created and no merge was performed during this audit or docs clarification.

### First small slice proposed for approval

**Narrow deterministic prefilter + supplied-vector parity, without live model,
DB resets, retention changes, UI work or broad deletion.** This is a bounded
prerequisite for the full flow, not a replacement for full Python evaluation.

- In intelligence, add a pure prefilter module under `src/pocketfinancer_sms/`
  and one sanitized vector set under `tests/sms_processing/golden/`. Specify
  pass/exclude and safe reason codes; optional analyzer evidence cannot turn
  a failed amount/account/sender regex into terminal exclusion.
- Port only that policy to pure Kotlin/Swift functions in the current pipeline
  namespaces; one shared set of inputs/expected decisions, no device/model calls.
  Start with confidently standalone verification/OTP/promotional non-events;
  ambiguous or mixed completed-transaction/OTP messages must pass. Do not copy
  private examples into fixtures. Do not connect to destructive erasure yet.
- Tests: standalone irrelevant cases exclude; completed transaction mentioning
  OTP passes; missing amount/account/verb or numeric sender alone passes;
  uncertain collect/payment text passes; Unicode and explicit foreign-currency
  cases pass unchanged. Assert input preservation, deterministic reason/decision,
  and no inference dependency in the pure policy.
- Gates: intelligence safety, both Ruff gates (including `src`), targeted
  `tests/sms_processing/` and full pytest; Android `:pipeline:testDebugUnitTest`
  plus normal `testDebugUnitTest`, `lintDebug`, `:app:assembleDebug`; iOS
  `scripts/format.sh --lint`, `scripts/build.sh`, `scripts/test.sh` on Mac.
  Report unavailable native lanes, do not substitute a Python pass for them.
- Follow-on, separately bounded: wire **one common admission/outcome boundary**
  for all native intake paths and the Python stateful runner, with fake-runtime
  call counts proving exclusions/replays invoke zero models and resume eligible
  work. Do not leave the old Android filter ahead of the new one. This wiring
  needs the retention decision below and tests for durable excluded outcomes,
  same-source retry distinction, cancellation and atomic settlement.

### Verification sequence and owner annotation

1. **Now, after scope approval:** preliminary filter tests, deterministic
   Python/Kotlin/Swift parity, existing unit/build regressions and fake-runtime
   stateful tests using invented cases. Evaluation tooling may be developed and
   tested with supplied responses before the real dataset is ready. These checks
   establish mechanics and specific regressions, not real-world quality.
2. **Owner annotation later:** keep the workbench usable, including saved-message
   sets, resume, protected-pool rules and durable human revisions. The owner labels
   the private corpus there; do not require completion of annotation before the
   first small filter slice, and do not reset annotation work to support cleanup.
   Declare the labelled development and protected held-out cohorts ready only
   after annotation/adjudication and provenance/split checks, not merely because
   an export file exists.
3. **Dataset-backed pipeline validation later:** use those ready cohorts with
   the complete Python end-to-end evaluation and native parity/device acceptance
   as applicable. Until then, record dataset-backed quality validation as
   **pending owner annotation**, not passed, abandoned, or replaced by synthetic
   tests/the 203-row regression fixture. No release/quality claim follows from
   preliminary checks.

### Full evaluation acceptance remains required

The integration's dataset-backed quality validation remains incomplete until a
single Python end-to-end runner exercises
all messages, local GGUF/harness and the official Mac Python SDK through platform
adapters, with native-supplied-response parity for request/settings, filtering,
strict parsing, grounding/money, accounts, duplicates and routing. Preserve run
isolation/locks, fresh/resume/partial/hash-mismatch behavior, labelled development
and protected held-out cohorts, and all-input denominators (including exclusions,
false exclusions, avoided calls, correct/incorrect saves, missed transactions,
Review/partial fields, failures and host-stage timings). Exercise unknown account
-> simulated explicit confirmation -> later auto-save, replay, possible duplicate
and interrupted retry. Simulated owner decisions are not model successes. Desktop
quality/timing is not device latency/battery/background/storage acceptance.

### Blockers and decisions to carry forward

1. The owner explicitly authorized the section 6 pure prefilter/synthetic-vector
   pilot on 2026-11-07, superseding the prior setup-only restriction. This is not
   approval of the full disposition/deletion map, intake wiring or retention.
   Dependent PR bases require confirmation before publication (section 10).
   No deletion, PR closure or reset is preauthorized by an R row.
2. Specify minimal excluded/valid-none outcome/evidence retention and reconcile
   iOS guidance that deterministically rejected bodies are erased immediately;
   preserve enough non-sensitive durable identity/outcome state for replay without
   retaining rejected bodies unnecessarily. No retention policy chosen here.
3. Specify different-alert duplicate context/window and owner resolution UI.
   Exact receipt-time fingerprints are not sufficient; amount alone is not safe.
   iOS imports may lack reliable source IDs: preserve uncertainty instead of
   silently deduplicating text or inventing connector identity.
4. Match Apple Python/Swift transport and Android GBNF settings explicitly;
   support documented API limitations, not fabricated cross-runtime identity.
   Mac official SDK/live quality and native device lanes have not been verified.
5. Obtain failing iOS #8 and Android #46 test details; #46 stays independent.
   No logs/root-cause diagnosis or native reproduction is available here.
6. Inventory private archive/annotations/keys/backup dependencies with the owner
   before future migrations or resets. This audit opened no private dataset and
   approves **no reset artifacts**. Old version adapters may be removed only
   without stranding evidence/labels, unfinished Review UI or implicitly running
   old inference.
7. Dataset-backed quality evaluation is blocked on the owner's later workbench
   annotation and cohort readiness. This does not block preliminary invented-case
   tests, evaluation-tooling development or other separately approved work that
   does not require real-data quality claims. Preserve the workbench and Review
   UI while that dataset is prepared.

## 7. Work checklist and next bounded task

- [x] Agree on product, evaluation, cleanup, and documentation expectations.
- [x] Record initial source observations and open-PR metadata.
- [x] Commit/push docs-only handoff branches and open the three coordination PRs.
- [x] Audit relevant PR diffs/ancestry and produce the proposed file-level cleanup map.
- [x] Owner authorizes only the section 6 pure prefilter/shared invented-vector pilot.
- [x] Refresh live feature/main refs and verify the existing clean pilot worktrees.
- [x] Add the pure Python policy, invented shared vectors and native Kotlin/Swift ports.
- [x] Complete fresh independent source review; no actionable defects identified.
- [x] Commit the focused pilot changes locally in all three isolated worktrees.
- [ ] Confirm dependent PR bases before publication.
- [ ] Obtain applicable CI/native execution evidence; no local test/build suites on this host.
- [ ] Resolve retention/admission decisions before separately authorized wiring or deletion.
- [ ] Review the proposed integration branches, PR dispositions, reset scope,
      documentation consolidation, and acceptance tests with the owner.
- [ ] Implement the selected flow and Python evaluation in bounded, tested slices;
      align native platforms and remove superseded paths without deleting useful work.
- [ ] Owner annotates the evaluation data in the preserved workbench; declare
      development/held-out cohorts ready and run dataset-backed end-to-end quality
      evaluation. This is later work, not a prerequisite for preliminary tests.
- [ ] Complete docs/dependency cleanup, parity checks, and platform verification;
      reconcile PRs and retire this temporary plan.

The bounded audit deliverables remain in sections 4-6. The newly authorized pure
pilot and its remaining gates are recorded in section 10. The broader cleanup,
intake wiring, resets, merges, PR closure and deletion remain unapproved. Record
exact selected baseline/destination commits and native-test evidence in this same
document as approved slices progress.

## 8. Historical docs-only checkpoint

Docs clarification environment: sibling checkouts under
`/home/ubuntu/pocket-financer`, generic Linux, **2026-11-07 UTC**. All three working
trees were clean at entry and remain on `codex/docs-cleanup-handoff`; no feature
checkout or `main` was modified. This is not WSL/Mac/device verification.

| Repository | Handoff head at clarification entry | Handoff PR (base `main`) |
| --- | --- | --- |
| Intelligence | `4450187e441b` (committed/pushed bounded audit) | [#11](https://github.com/pocket-financer/pocket-financer-intelligence/pull/11) |
| Android | `ca251056b829` (unchanged) | [#48](https://github.com/pocket-financer/pocket-financer-android/pull/48) |
| iOS | `1343ab79d2c0` (unchanged) | [#12](https://github.com/pocket-financer/pocket-financer-ios/pull/12) |

Only `docs/CLEANUP_PLAN.md` changes in this session. The owner clarified that
unfinished SMS Review UI must remain present and that they will annotate the
not-yet-ready evaluation dataset in the existing workbench later. Sections 1, 2,
5 and 6 now make those preservation and sequencing requirements explicit. The
integration paragraph also separates starting versions, small reviewed working
changes and eventual complete main integration; no merge-first shortcut is
approved. This clarification does **not** authorize runtime implementation,
branch setup, deletion, resets or merges.

Verification and limits:

- Git status/head checked in each repo. Intelligence `git fetch origin` succeeded;
  its remote handoff head matched `4450187` before this docs update. Android/iOS
  file lists and Review UI commit history were inspected via refs, without
  checking out or modifying the feature work. Android partial-evidence and
  contrast commits were verified as ancestors of the recorded #42 tip.
- Repository safety (tracked pathnames only), unstaged/staged `git diff --check`,
  document headings/fences/local links, exact plan-only scope and named Review
  source/commit preservation references are the checks for this docs-only change.
  Normal model/native code suites are not a prerequisite or a claimed result.
- The 2026-11-05 PR/ancestry/check audit remains in sections 3-4 and commit
  `4450187`. **Live PR checks were not re-audited in this clarification.** The
  recorded iOS #8 and Android #46 failures/log-access blocker must be refreshed,
  not assumed resolved. Prior audit tools were missing Ruff/pytest and native
  build tooling; no installation, bypass or runtime test is part of this update.
- No private messages, annotations, source mappings, credentials, keys or backups
  were opened or changed. No runtime code, UI, asset, data or release file was
  changed; no new plan/document, integration branch, deletion, reset, merge,
  closure, force-push, model download or release occurred.

Commit subject for this update: `docs: preserve review UI and stage evaluation`.
Use the latest intelligence handoff head for this checkpoint; the final response
records commit/push results. Only the normal fast-forward #11 plan update is in
scope; other PRs remain unchanged. Next: owner approval of the remaining proposed
scope/branch arrangement, then explicitly authorized small slices. Dataset-backed
quality validation stays pending the owner's later annotation, without blocking
preliminary invented-case tests. Refresh status/live PR state before acting.

## 9. Resume prompt

> Read all three repositories' AGENTS.md files and this CLEANUP_PLAN.md. Sections
> 4-6 retain the bounded audit; section 10 records the explicitly authorized pure
> prefilter pilot, current commits and remaining gates. Reuse existing pilot
> branches/worktrees without discarding work. Confirm dependent PR bases before
> publication; do not infer native CI or executed parity from source review or
> fixture byte equality. No test/build suites or native tooling installation on
> this host. Preserve useful feature work, unfinished Review UI, the annotation
> workbench, private evidence/labels/provenance, keys and backups. Dataset-backed
> quality remains pending owner annotation and cohort readiness. Wiring requires
> separately bounded approval and retention/admission decisions. No resets,
> merges, closures, force pushes, model/data publication or releases. Update this
> single plan; do not create another planning/session document.

## 10. Authorized pure prefilter pilot outcome

### Authority, baseline and scope

The owner's 2026-11-07 request explicitly authorizes application implementation
of the **section 6 pure prefilter plus shared invented-vector slice only**. The
prior setup-only restriction is superseded for that slice. It does not authorize
production admission wiring, erasure/retention changes, database resets, Review
UI changes, broad deletion, inherited-stack integration, model/data publication
or release. No old PR is retargeted, closed, superseded or merged.

Fresh fetches and live GitHub PR metadata match the previously prepared clean
pilot branches; all three remain `codex/feat-prefilter-pilot` in their respective
repos. Existing branches/worktrees were reused, not reset or recreated. Main
and feature source branches remain unchanged. The docs handoff remains separate;
combining older guidance stacks is deferred rather than mixed into pure code.

| Repo | Exact preservation baseline / live feature head | Proposed dependent draft PR base | Local pilot commit |
| --- | --- | --- | --- |
| Intelligence | `0fd9b8aece174f28f2b390d8d7fe99bc0ebc58fe` (#8) | `codex/workbench-eval-set-filter` | `4688d2eabc8f278ca8bd3906da622dc28894ca47` |
| Android | `54a8aa7ab9c0c4bc4c640c5534d7e637d27eecfd` (#42) | `codex/feat-native-sms-v3` | `2f19cc29e7c4d7ff1f6a90b499268761068aa232` |
| iOS | `050010e78afff3712750c67e206833df11170415` (#8, draft) | `codex/feat-native-sms-integration` | `fe336182eb87f689eadee9000d7ad76767968851` |

Refreshed main heads are intelligence `78509644dad110219e7a53d7c503cd746f9e0ab6`,
Android `a58fcad7903f99484cc98f16006b00cf02fa127c`, and iOS
`c79b7b574fc1d390adbc59dc4771b656015c1144`. Open PR inventory is unchanged
from section 3. No merge-first baseline or main-based feature rewrite was used.
The three bases above are **proposed, pending owner confirmation before push/PR
publication**. No pilot branch has been pushed and no pilot PR has been opened.

### Contract and changed files

`conservative-prefilter-v1` takes only an unchanged body and returns `pass` /
`uncertain`, or `exclude` / `standalone_verification` or
`standalone_promotion`. Exclusion requires a whole ASCII message matching one of
these closed templates (digits shown as placeholders):

- `Your OTP is <4-8 ASCII digits>. Do not share it.`
- `Your login verification code is <4-8 ASCII digits>. Do not share it.`
- `Advertisement: Save <1-2 ASCII digits>% in our weekend sale. Shop now.`

Only ASCII case and runs of space/tab/CR/LF are ignored, for matching only.
Non-ASCII, unrecognized templates and additional content always pass. There is
no broad keyword exclusion, negative amount/account/verb/sender test, analyzer
allowlist, model dependency or retention action. This is intentionally limited
coverage, not a claim to filter every OTP or advertisement. Passing is not a
transaction classification. The shared contract/vectors were settled before
porting to native functions; native apps do not depend on Python at runtime.

| Repo | Additive files only |
| --- | --- |
| Intelligence | `src/pocketfinancer_sms/prefilter.py`; `tests/sms_processing/test_prefilter.py`; `tests/sms_processing/golden/prefilter-v1.json` |
| Android | `pipeline/src/main/java/com/pocketfinancer/pipeline/sms/ConservativeSmsPrefilter.kt`; matching `pipeline/src/test/java/com/pocketfinancer/pipeline/sms/ConservativeSmsPrefilterTest.kt`; `pipeline/src/test/resources/prefilter-v1.json` |
| iOS | `PocketFinancer/Services/SmsProcessing/ConservativeSmsPrefilter.swift`; `PocketFinancerTests/ConservativeSmsPrefilterTests.swift`; `PocketFinancerTests/Fixtures/prefilter-v1.json` |

Intelligence owns the fixture. All three copies are byte-identical, SHA-256
`373337f584cd70d4d165e7ff557fb45ece8967f3795d0d2d05a02f61db5053c9`.
It has 38 unique invented cases: 5 expected excludes and 33 expected passes,
including mixed posted/OTP/promotion text, missing fields, numeric sender,
uncertain collect requests, foreign currency, Unicode, control characters and
near-template misses. Tests pin the hash, decision/reason, repeatability and
source preservation. Native resource loading and execution remain unverified.

### Review, preservation and checks

Parent implementation/integration uses configured OpenAI `gpt-6.1-sol`; one fresh
read-only contract reviewer uses the same explicitly selected OpenAI model.
No Astra, API-key fallback, priority route or model/data evaluation is used.
Independent review of all nine additions completed with **no actionable defects**.
The parent inspected the saved review and verified the same additive scope,
fixture hashes and absence of intake callers. No fix round was needed. Review
supports source-level policy alignment, not executed parity or native acceptance.
Nonblocking future coverage opportunities (uppercase login in the shared set,
three-digit promotion percentage and ASCII DEL suffix) were not treated as
permission to expand this settled fixture slice.

Only the nine additive pilot files and this existing plan are in scope. Parent
inspection confirms no existing production source/UI/storage/annotation asset
was modified. Android partial-Review `f6079ff` and contrast `cc1eb79` remain
ancestors. The baseline histories and existing Review/workbench files remain
intact. No private data, labels, provenance, keys or backups were opened.

Performed on the editing host: status/ref/worktree/ancestry inspection, live PR
and check metadata refresh, JSON data inventory, `sha256sum`, `cmp`, and staged
`git diff --check`. No application code, tests, builds, formatter, safety suite,
model runtime or native tooling was executed or installed.

| Gate / evidence | Actual state |
| --- | --- |
| Pilot Python safety, both Ruff gates including `src`, targeted/full pytest | **Not run**. Existing intelligence CI triggers on branch push/PR; pending publication and its new-head results. |
| Android focused pipeline/full unit tests, lint, debug APK | **Not run**. Existing CI only triggers push/PR against `main`, not the proposed dependent base. Pending appropriate machine/later approved integration checks. |
| iOS format, Release build, unit/UI smoke tests | **Not run**. Existing CI only targets `main`; Mac/Xcode execution required. Pending appropriate machine/later approved integration checks. |
| Native supplied-vector parity | **Pending**. Byte equality/source review does not establish native execution. |
| Real-data recall / end-to-end pipeline quality | **Pending owner annotation/cohort readiness**, not established by invented vectors. |

Live baseline results remain intelligence #8 Python checks successful and Android
#42 build/lint/unit successful **for their old heads only**, not this pilot.
iOS #8 still fails `Format, build, and test` in run `41490942618`; its title
checks succeeded. Separate Android dependency #46 still fails run `38181207167`.
No root cause or fix is claimed. No workflow trigger was changed or dispatched,
and no release workflow was touched.

### Draft delivery and next bounded task

Prepared scope/title for each draft: **`feat: add conservative standalone SMS
prefilter`**, targeting only its proposed feature base above. Each draft must
state: pure/unwired implementation, intelligence-owned fixture/hash, inherited
feature dependency, source-review outcome, no local suite execution, pending
native lanes and pending dataset quality. A main-target PR would inherit the
large unreviewed feature stack and is not an acceptable shortcut.

All three code commits use `feat: add conservative standalone SMS prefilter`,
contain exactly the three additions listed for their repo, and leave their pilot
worktrees clean. The plan-only handoff update uses
`docs: record reviewed prefilter pilot outcome`; its final SHA is recorded in the
delivery response rather than self-referenced inside its own commit.

Next: obtain owner confirmation of the three dependent bases before push/draft
PR creation. After confirmation, observe existing intelligence CI
without changing native CI triggers; leave unavailable checks pending. The next
implementation slice is **not started**: decide retention/admission contracts,
then separately authorize the common source-outcome/admission boundary with
fake-runtime zero-call/replay/retry/cancellation/atomic-settlement coverage. Do
not retain the old Android terminal filter ahead of that future boundary.
