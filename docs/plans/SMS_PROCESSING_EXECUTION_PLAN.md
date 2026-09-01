# SMS Processing Execution Plan

Status: **single active execution plan**
Last updated: 2026-09-01

## Completed foundation

- [x] Establish explicit repository/private-data boundaries and focused branch.
- [x] Preserve the exploratory pre-filter/taxonomy snapshot before replacement.
- [x] Implement source-preserving deterministic analysis and core/India profiles.
- [x] Implement explicit/default currency provenance and exact minor-unit parsing.
- [x] Implement tri-state high-recall triage plus normal/assistive/skip selector action.
- [x] Implement the Grounded Candidate Selector and strict host reconstruction.
- [x] Separate recognition, semantic reconstruction, and automatic persistence.
- [x] Implement canonical label, processing trace, and user-feedback contracts.
- [x] Rebuild every one of 17,830 rows into one ignored canonical manifest.
- [x] Assign normalized-template components to protected annotation pools.
- [x] Produce queue views, grouping, coverage, pre-filter, leakage, and provenance reports.
- [x] Build and initialize the local SQLite workbench with backup/recovery/export tests.

## Current cleanup and authority phase

1. [x] Make the active architecture documents authoritative and index historical
   experiment evidence explicitly.
2. [x] Remove obsolete active V2 plans/configs/tests and exploratory replacement
   code now covered by `src/pocketfinancer_sms`.
3. [x] Verify canonical copies/member hashes and remove old derived datasets after
   explicit confirmation. Preserve the raw archive, 1,436 reviewed asset, 203
   regression fixture, and measured reports.
4. [x] Run the full repository gate and open a focused pull request.

## Implementation file map

Active files added by this route:

- `src/pocketfinancer_sms/{analyzer,currency,feedback,labels,persistence,profiles,provenance,selector,structural_text,trace,triage,types}.py`;
- `src/pocketfinancer_sms/corpus/{grouping,manifest,pools,reports}.py`;
- `src/pocketfinancer_sms/workbench/{service,store,web}.py` and its checked-in
  local `assets/{index.html,app.js,styles.css}`;
- `configs/sms_processing/archive-india-inr.json`, all schemas in
  `configs/sms_processing/contracts`, the supported-currency declaration, and the
  `core-en`/`india` profiles;
- `scripts/run_sms_processing.py` and synthetic tests under
  `tests/sms_processing`;
- the seven descriptive architecture/contract/plan documents indexed from
  `docs/README.md`, plus `docs/history/SMS_PROCESSING_EVIDENCE_INDEX.md`.

Repository authority and checks modified:

- `.github/workflows/ci.yml`, `AGENTS.md`, `CONTRIBUTING.md`, `README.md`,
  `requirements-ci.txt`, `scripts/check_repo_safety.py`, and `scripts/README.md`;
- `docs/README.md`, `docs/architecture/REPOSITORY_LAYOUT.md`,
  `docs/experiments/EXPERIMENT_CATALOG.md`, and
  `docs/guides/EXTRACTION_V2_WORKBENCH.md`.

Historical relocations:

- `LFM25_350M_GOAL_BRIEF.md` -> `docs/history/LFM25_350M_GOAL_BRIEF.md`;
- `TRAINING_READINESS.md` -> `docs/history/TRAINING_READINESS.md`;
- `docs/plans/POCKETFINANCER_EXTRACTION_PROGRAM_V2.md` and
  `docs/plans/POCKETFINANCER_EXTRACTION_V2_STATUS.md` ->
  `docs/history/extraction_program/`;
- `configs/programs/pocketfinancer-extraction-v2.json` and its decision policy ->
  `configs/history/`, with the policy's old path retained as a compatibility
  symlink only.

Tracked active-path deletions:

- `configs/contracts/unified-prefilter-spec-v1.json`;
- `lfm25/prefilter_simulator.py` and `lfm25/taxonomy.py`;
- `scripts/auto_label_obvious_non_transactions.py`,
  `scripts/auto_label_otp_not_transaction.py`,
  `scripts/build_universal_v2_datasets.py`,
  `scripts/export_stratified_evaluation_splits.py`, and
  `scripts/segregate_sms_dataset.py`;
- `tests/test_dataset_segregation.py` and
  `tests/test_extraction_v2_program_state.py`.

The ignored private deletions are recorded in the Decision Log's artifact matrix;
they never enter a commit.

## Commit sequence

1. Establish repository and private-data boundaries.
2. Add deterministic analysis, currency context, and locale profiles.
3. Add grounded selector, reconstruction/persistence, canonical labels, traces,
   and feedback contracts.
4. Add canonical grouping, protected annotation pools, reports, and provenance.
5. Add the unified SQLite workbench and synthetic browser/service tests.
6. Harden source binding, schemas, backup/recovery, revision chains, privacy, and
   contract conformance.
7. Relocate historical governance, remove superseded active paths, and make the
   descriptive documents authoritative.
8. Push the focused branch and open the review only after the final verification
   matrix passes.

## Human segregation phase after this implementation

1. Start with the annotation-training pool; calibrate reviewers on diverse
   sender/template/time groups.
2. Correct weak operational class, event state, family, and rail separately from
   canonical outcomes.
3. Submit exact rich labels and track candidate-oracle failures by field.
4. Improve analyzer/profile rules only from aggregate error evidence and reviewed
   examples, preserving rule provenance and synthetic regression cases.
5. Rebuild a new immutable corpus run when executable analysis changes; never edit
   the old run in place.
6. Use annotation-development for policy calibration. Keep protected test and
   later-time holdout blind until policy/model work is frozen.
7. Do not materialize SFT data until the label and projection gates pass without
   fallback.

## Native integration planning phase (next session)

The next session begins in Plan mode and inspects Android/iOS before changing
them. It must plan:

- primary-currency onboarding/settings and per-operation snapshots;
- shared analyzer/profile parity and high-recall triage on both platforms;
- a user review inbox for every `retain_review` outcome, including assistive SLM
  results when candidates are complete;
- one-pass non-thinking Grounded Candidate Selector invocation and token-by-token
  compact JSON visibility;
- strict reconstruction/persistence without default-account fallback;
- a truthful processing trace screen showing analyzer output, candidate IDs and
  evidence, model JSON stream/raw output, validation, reconstruction, and
  persistence reasons—never fabricated chain of thought;
- append-only user confirmations/corrections/rejections bound to trace and
  canonical label revisions for future local evaluation/fine-tuning;
- migration from Android's current editable result and hard OTP behavior;
- how native feedback synchronizes into the same local workbench without hosted
  APIs, telemetry, or raw-data logs.

No native implementation begins until that plan is approved.

## Later model/evaluation phases

Only after human truth is sufficient:

1. Freeze projection contracts and build hash-bound Candidate Selector targets.
2. Train/compare models outside this session under one-pass greedy decoding.
3. Evaluate on annotation-development; freeze decisions.
4. Open protected test/later-time holdout exactly once for the declared gate.
5. Run aligned host and native-device evaluation before any deployment decision.

## Completion gates

- [x] All 17,830 source rows represented exactly once.
- [x] No raw private data tracked.
- [x] Zero source-ID, exact-body, normalized-template, and sender-template overlap.
- [x] Existing mapped human-positive rows are never deterministically discarded.
- [x] No SFT targets generated.
- [x] Candidate coverage reported per field and complete event.
- [x] Missing candidates route to review, not negative truth.
- [x] Mixed-clause, negation, standalone OTP, and posted-plus-OTP tests pass.
- [x] Selector output is strictly ID-grounded and `posted` requires candidates.
- [x] Explicit and configured-default currency paths are tested.
- [x] Workbench blind-first, revision, validation, backup, recovery, export, and
  synthetic HTTP smoke tests pass.
- [x] Obsolete derived private artifacts removed after approved verification.
- [x] Full Ruff, targeted lint, complete pytest, safety, schema, corpus, and diff
  checks pass on the final tree.
- [x] Active documentation and executable contracts pass a final agreement audit.
- [x] Focused branch pushed and pull request created.

## Next-session planning prompt

Use the following request in the next session:

> Start in Plan mode. Plan one unified local workbench and native integration
> across the PocketFinancer Android and iOS sibling repositories and the shared
> `pF_slm_selection` SMS-processing foundation. Read
> every applicable AGENTS.md and inspect current native SMS filters, inference,
> persistence, editing/feedback, currency settings, and UI flows before proposing
> changes. Produce one executable plan—do not edit yet—for integrating the shared
> deterministic analyzer, configured primary currency, tri-state review behavior,
> one-pass Grounded Candidate Selector, strict reconstruction/persistence gate,
> append-only ProcessingTrace/UserFeedbackEvent contracts, and a user review inbox
> for retain-review cases. Preserve maximum truthful transparency: show analyzer
> cues/candidates, compact JSON token decoding, raw selector output, validation,
> reconstruction, and persistence reasons, but do not expose or fabricate chain of
> thought. Replace Android's body-wide OTP discard and default-account fallback,
> migrate existing user corrections into revisioned feedback, and plan how both
> apps feed the single workbench in which source rows, deterministic processing,
> selector traces, persistence decisions, user corrections, canonical labels,
> review queues, and aggregate coverage can all be observed without telemetry,
> hosted APIs, raw logs, or private screenshots. Keep model training, deployment,
> and production-default
> changes out of scope. End with exact files, migrations, state transitions,
> tests, rollout gates, and a native-by-native commit sequence for approval.
