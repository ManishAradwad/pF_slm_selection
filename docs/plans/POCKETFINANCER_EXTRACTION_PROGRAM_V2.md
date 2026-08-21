# PocketFinancer transaction-extraction program, V2

## Status and authority

This is the normative program plan for reliable local extraction of completed Indian financial transactions. It governs the Android app, iOS app, and the `pF_slm_selection` research/evaluation repository. Semantic truth and evaluation requirements are shared; application code, model prompts, parser adapters, runtime settings, and user interfaces remain platform-specific.

PocketFinancer Android's committed source and its locked profile remain the authority for existing Android behaviour. Phase A was foundation work in this repository and did not alter either app, the locked Android profile, or production defaults. Later phases do authorize focused Android and iOS implementation when their entry conditions and user approvals are satisfied; evaluation evidence by itself never changes a production default. The machine state is in `configs/programs/pocketfinancer-extraction-v2.json`; the task handoff is in `docs/plans/POCKETFINANCER_EXTRACTION_V2_STATUS.md`.

## Program definition of done

The engineering integration is done only when the selected, version-pinned Android runtime variants and the selected iOS pipeline are implemented end to end in their owning repositories and default-off UAT candidates are ready for the user's hands-on testing. This includes native platform adapters, deterministic filtering, model/runtime invocation, source grounding, posting decisions, exact-money and timestamp-provenance persistence, required data migrations, review/retry and recovery paths, UI observability, and unit, conformance, integration, UI, migration, and relevant device tests.

The SLM repository remains the coordination authority. It must contain a versioned three-repository compatibility manifest that binds the exact compatible Android and iOS commits to the preceding verified SLM implementation commit, Semantic V2 schema/reference hashes, selected platform profile IDs, prompts/output protocols, filters, parser adapters, runtime/decode settings, model or system-runtime identity, artefact hashes where available, migrations, evaluator and split hashes, threshold decision, and verification evidence. The later commit that adds or updates the manifest cannot contain its own Git object ID; that manifest-record commit is reported externally in the handoff and may also be bound by a tag or attestation. A compatible set is not inferred from branch names or latest commits.

Shared semantics do not require identical model protocols, runtimes, or UIs. Android and iOS may select different prompts, output forms, adapters, recovery behaviour, and presentation appropriate to llama.cpp-style token generation and Apple Foundation Models respectively, provided both map to and conform with the same Semantic V2 contract. Completion of the engineering integration does not itself authorize deployment: user acceptance and any default cutover or release are Phase J decisions.

## Shared semantic boundary

Semantic V2 (`pocketfinancer_semantic_v2`, version 2) is the shared host semantic truth, not a universal model protocol. Its versioned JSON Schema is `configs/contracts/pocketfinancer-semantic-v2.schema.json`; the dependency-free Python reference is `lfm25/semantic_v2.py`.

Every evaluated interpretation must state all of the following:

- scope: `bank_card`, `wallet_bnpl`, or `other`;
- posting status: `posted` or `not_posted`;
- event cardinality: `none`, `single`, or `multiple`;
- for each event, optional amount, direction, and account evidence;
- an explicit counterparty state: `present` with evidence or `absent`.

Evidence ranges are zero-based, half-open UTF-8 byte offsets into the original message. Exact `decimal_text` is authoritative. Currency is derived from and validated against selected amount evidence. Minor units are deterministic host derivatives using the versioned currency scale and are `null` when their calculation would need rounding; rounding is forbidden. Automatic persistence requires non-null exact minor units. Platform storage must use integer minor units or an equivalently lossless decimal representation; a binary floating-point value is not the source of truth. Any existing incompatible storage receives an explicit, tested migration and rollback path.

The initial automatic-post projection is deliberately narrow: exactly one `posted`, `single`, INR `bank_card` event with grounded amount, direction, and account evidence and exact minor units. It is an evaluation/reference boundary only in Phase A; it does not authorize an application behaviour change.

## Timestamp and privacy policy

Transaction date is source metadata, never model output. Android supplies `SmsMessage.date`. iOS uses the persisted `InboxAlert.receivedAt` and records whether it was supplied or assigned during ingestion. Semantic V2 injects this metadata after interpretation and preserves provenance. Each native pipeline must carry the source timestamp and provenance through retries and persistence, expose it to diagnostics without exposing private content, and migrate storage when provenance cannot otherwise be represented. No prompt, model target, output schema, or parser acceptance rule may ask a model to infer or emit a date.

Raw alerts, senders, account labels, private annotations, row-level predictions, stable source mappings, credentials, and model weights stay local. Before explicit user authorization they may not be inspected or copied. A later authorized local workflow may store scoped private material only in ignored private roots such as `PRIVATE_DATA/` and other paths allowed by the applicable `AGENTS.md`; it must never enter Git, committed or published datasets, logs, hosted APIs, telemetry, cloud inference, screenshots, or task transcripts. The synthetic fixture committed in Phase A is entirely invented. The user exclusively owns real-data annotation, adjudication, split, and migration decisions.

## Model and runtime roles

Each model/runtime family receives its own versioned prompt, output, parser, decoding, and runtime profile. All profiles map into Semantic V2; none is presumed portable merely because the semantics are shared. Direct V2 and Candidate V2 are both unselected hypotheses. They must be compared independently per family and platform; Phase A makes neither a universal protocol.

Android's presently deployed selectable fleet is Qwen3-0.6B Q8, Qwen3-1.7B Q4/Q8, and Gemma 4 E2B Q4/Q8. Their chat-template handling, thinking behaviour, quantization, grammar use, parser behaviour, and device limits are separate experimental variables. LFM2.5-350M is the primary small trainable research candidate, not an Android-selectable production model. LFM2.5-2.6B Base and Post are research ceilings, not mobile deployment targets.

Every currently selectable Android runtime variant receives an independent Phase H disposition: a selected V2 profile ID or an explicit no-selection/exclusion reason. At least one Android variant and the iOS pipeline must be selected to enter Phase I. Phase I implements every selected Android variant; an excluded variant remains on its existing path, is outside the Semantic V2 UAT claim, and cannot be removed or made default without the user's Phase J decision. The program cannot claim complete Android fleet convergence while a still-UAT-selectable variant lacks a recorded disposition.

iOS uses `Apple SystemLanguageModel.default`. Its installed revision is opaque, OS-managed, and device-dependent. The iOS program evaluates availability, structured-generation behaviour, parser grounding, latency, and recovery without claiming a fixed model revision or equating it to Android model evidence.

## Controlled experiments and selection gates

Experiments must pin the Semantic V2 schema/reference hash, prompt/output profile ID, model revision, chat template, quantization, decode settings, parser/evaluator revision, filter version, data/split hash, seed, selected checkpoint, and runtime environment. Hold out sender/template families where the user has approved data. Compare one material variable class at a time, or mark the comparison non-causal. Store aggregate-safe metrics and provenance, never raw private rows.

Host/HF, host/GGUF CPU, host/GGUF accelerator, Windows emulator, macOS emulator/simulator, and physical Android/iOS device measures are separate evidence classes. No host, emulator, or simulator timing is physical-phone timing. A profile is not selected from a single model family, a single device, or an unblinded review.

Before protected or blinded scoring, Phase B must freeze a versioned, user-approved threshold artifact. Every candidate reports scope, posting-status, and event-cardinality accuracy; amount/currency, direction, account, and counterparty exactness; evidence-grounding validity; automatic-post precision and coverage; false-automatic-post taxonomy; invalid-output and fail-closed rates; latency; memory; availability; recovery; and, at the relevant device tier, battery impact. Confidence intervals and sample-size requirements are fixed before comparison, not chosen after results are seen.

Selection requires all applicable gates, not merely the best headline score:

1. 100% conformance-vector and contract validation with zero uncaught parser/validator exceptions;
2. zero observed automatic persistence of ungrounded, ambiguous, not-posted, none/multiple-event, non-INR, inexact-money, or missing-timestamp-provenance cases in the conformance and approved gold suites;
3. the predeclared lower confidence bound for automatic-post precision, plus predeclared accuracy and useful-coverage floors, passes independently for each selected platform profile;
4. no disallowed regression against that platform's reproducible baseline and all predeclared device budgets pass;
5. privacy, provenance, and compatibility-hash checks pass completely; and
6. ties use a predeclared rule; a failed gate produces no selection and returns the platform to the relevant laboratory phase.

## Cross-platform implementation and test matrix

| Layer | Shared/SLM repository | Android native repository | iOS native repository | Required gate |
| --- | --- | --- | --- | --- |
| Contract and adapters | Schema/reference tests, invented vectors, hash/drift checks | Every selected runtime variant's native adapter runs shared invented vectors | Native adapter runs shared invented vectors | Semantic equivalence and 100% conformance; protocols may differ |
| End-to-end decision path | Workbench fixtures and aggregate evaluator | SMS intake through filter, runtime, adapter, grounding, eligibility, review/retry, and persistence | Inbox intake through availability/filter, structured generation, adapter, grounding, eligibility, review/retry, and persistence | Expected Semantic V2 result and fail-closed outcome at every boundary |
| Money, time, and storage | Exact minor-unit and timestamp-provenance oracles | Exact-money storage plus migration/rollback and source-timestamp tests | Exact-money storage plus migration/rollback and received-at provenance tests | No rounding, floating-point source of truth, timestamp inference, or provenance loss |
| UI observability | Versioned stage/outcome semantics | Live emitted thinking and JSON-token progress, processing stages, and saved/review/error result | Cumulative structured-generation snapshots, processing timeline, and saved/review/error result | UI and accessibility tests; no invented hidden reasoning or private-data leakage |
| Recovery and device | Profile, failure taxonomy, and aggregate evidence | Interruption/retry/idempotency, model load, latency, memory, battery, and supported-device tests | Unavailable/interrupted/retry/idempotency, latency, memory, battery, OS/device tests | Predeclared recovery and device budgets pass independently |
| Compatibility and UAT | Three-repository manifest and verification record | Reproducible build at pinned commit/profile | Reproducible build at pinned commit/profile | Exact compatible commits/hashes pass and candidate remains default-off |

Android must preserve or deliberately adapt the existing display of runtime-emitted thinking and JSON token-generation progress, followed by filter, model load, generation, grounding, persistence, and final saved/review/error stages. This is observable model output and pipeline state, not a claim that the app can reveal private hidden reasoning.

iOS must present the cumulative structured-generation snapshots that Foundation Models exposes and a truthful process timeline from durable intake through availability/filtering, generation, field mapping, grounding, eligibility, database write, and final saved/review/retry state. It must not invent token decoding, chain-of-thought, or hidden reasoning that the API does not provide. Both apps keep diagnostic content local and apply their privacy/redaction rules.

## Phases

### A. Foundation and Semantic V2

Entry: the three worktrees and guidance are recorded; Android and iOS are read-only evidence; no private data is inspected.

Exit: committed program/state/handoff artifacts; a versioned platform-neutral schema and reference validator; host timestamp provenance; exact-money and UTF-8 evidence rules; invented conformance vectors; unit coverage; and the lightweight repository gate. No production default changes and no model protocol selection. Phase A itself makes no app change.

### B. Workbench V2, evaluation packages, and frozen decision policy

Entry: Phase A is complete, the schema and synthetic vectors are frozen by hash, and the user has explicitly authorized any private annotation work that is proposed.

Exit: a local-only workbench/evaluation package can create, validate, anonymize, and score Semantic V2 annotations without raw data in Git; split/adjudication/provenance interfaces are defined; synthetic smoke tests prove fail-closed handling; the cross-platform matrix is executable at its shared layer; and the metric definitions, confidence method, sample-size rules, thresholds, and tie/no-selection policy are versioned before protected evaluation. No training, model selection, or application integration follows automatically.

### C. Android baseline reproducibility and runtime instrumentation

Entry: Phase B's shared evaluator is available and the current Android source/profile relationship has been audited without changing the locked production defaults.

Exit: reproducible Android baseline captures filter, prompt assets, parser, chat template, decode/runtime settings, model identifiers, aggregate device instrumentation, current persistence/time semantics, and current UI stage behaviour. Host and device measures are labelled separately, and baseline gaps are recorded.

### D. Android prompt/output-protocol laboratory

Entry: Phase C baseline evidence and fixtures are reproducible for each Android family under test.

Exit: controlled, evaluation-only comparisons independently test Direct V2 and Candidate V2 (and any justified alternatives) for Qwen and Gemma. Each result has protocol/parser/runtime provenance, aggregate failure taxonomy, threshold-gate status, and device evidence. A result may select no protocol.

### E. LFM2.5-350M data, fine-tuning and quantization program

Entry: Phase B supplies user-approved local data interfaces and the chosen experiment design explicitly names its LFM profile. Any real labels/splits have user approval and remain private.

Exit: reproducible, non-overwriting data build, fine-tuning, checkpoint selection, merge, conversion, and quantization evidence for LFM2.5-350M. LFM2.5-2.6B Base/Post may be measured only as research ceilings. Results do not establish Android deployability.

### F. Android evaluation-only model integration and device validation

Entry: a Phase D or E candidate has a complete profile and artefact provenance, and Android integration remains evaluation-only behind an explicit non-default path.

Exit: native adapter, filter, prompt, runtime, memory, latency, recovery, UI-observability, and device validation are recorded for the exact artefact. The production fleet/defaults remain unchanged pending Phase I.

### G. iOS Foundation Models evaluator and prompt/output-protocol laboratory

Entry: Phase B evaluator semantics are available, and tests handle `SystemLanguageModel.default` unavailability without data loss.

Exit: iOS evaluation-only profiles independently compare prompt/output approaches, native adapter grounding, availability/recovery, cumulative structured-generation observability, and device behaviour with OS/device provenance. No fixed Apple model revision, hidden reasoning, token-decoding parity, or Android protocol parity is claimed.

### H. Blinded cross-platform evaluation and profile selection

Entry: the intended Android and iOS candidates have completed their evaluation-only evidence paths; the threshold artifact is frozen; and the user has approved blinded human evaluation and test governance.

Exit: a blinded, aggregate-safe cross-platform report uses the shared Semantic V2 evaluator and reports every gate independently. The Android decision map covers Qwen3-0.6B Q8, Qwen3-1.7B Q4/Q8, and Gemma 4 E2B Q4/Q8 with a selected profile ID or explicit no-selection/exclusion reason for each runtime variant; iOS records a selected profile ID or no-selection decision. Known runtime/parity gaps are disclosed. Phase I requires at least one selected Android runtime variant, a selected iOS profile, and a user-approved Android UAT inclusion/exclusion set.

### I. Production-quality three-repository implementation and default-off UAT candidate

Entry: Phase H has selected at least one Android runtime-variant profile and an iOS profile, recorded the disposition of every current Android runtime variant, every pre-implementation promotion gate passes, and the user explicitly authorizes focused implementation branches in all affected repositories.

Exit: every selected Android runtime-variant pipeline and the selected iOS pipeline are implemented end to end in their native apps, including adapters, filters, prompts/protocols, runtime integration, grounding, posting decisions, exact-money and timestamp-provenance storage, migrations and rollback, review/retry/recovery, and the platform-appropriate UI observability defined above. The cross-platform matrix passes at the required tiers; native unit, conformance, integration, UI, migration, build, and relevant device tests pass; the SLM compatibility manifest pins the exact reviewed commits/hashes and evidence; and reproducible Android and iOS UAT builds are available through an explicit default-off path. Existing production defaults and ordinary upgrade behaviour remain unchanged.

### J. User acceptance and optional default cutover/release

Entry: Phase I's manifest-bound UAT candidates are reproducible and the user explicitly chooses to begin hands-on acceptance testing. Use of private alerts and any migration of existing user data remain the user's decisions.

Exit: UAT findings and disposition are recorded. A rejected or deferred candidate remains default-off and returns to Phase I or the relevant earlier phase; no release is implied. If the user accepts both platform candidates, separately reviewed cutover changes define default and upgrade behaviour, staged migration, rollback, supported OS/devices, privacy/licensing disposition, observability, and final build/device evidence. Deployment or release occurs only after a separate explicit user authorization, and Phase J may validly end with a no-release decision.

## Branch strategy and verification

Use a short-lived `codex/extraction-v2-<phase>` branch per repository. Phase work changes only the repository that owns the implementation; cross-repository coordination is versioned in the SLM repository. Preserve unrelated work, never reset/stash/clean, and use Conventional Commits. Phase I uses coordinated but independently reviewed branches, and the compatibility manifest is finalized only from exact commits that passed their owning repository's checks. Its SLM pin names the preceding verified implementation commit; the subsequent manifest-record commit is reported by the handoff/tag/attestation rather than self-recorded.

Every Phase A/B repository change runs the SLM lightweight gate: `python scripts/check_repo_safety.py`, `ruff check .`, `ruff check --select E4,E7,E9,F lfm25 scripts tests`, `pytest -q`, and `git diff --check`. Profile/pipeline/lock/manifest changes additionally run `python scripts/run_pocketfinancer_pipeline.py check`. Model downloads, HF inference, GGUF conversion, and device checks run only in the phases that need them and report their tier honestly.

Standing user authorization recorded on 2026-08-22 requires each affected verification tier to use the best available applicable hardware. GPU-suitable local ML/GGUF work should use the available NVIDIA CUDA GPU with explicit offload configuration and observed utilization evidence. Windows-hosted Android emulators and, when a macOS execution host is available, Android emulators or iOS simulators are authorized for applicable tests; attached physical devices may also be used when available and in scope. Host CPU, host GPU, each emulator/simulator platform, and physical devices remain distinct evidence classes. This policy does not make accelerators or devices prerequisites for documentation-only or pure unit work, and it does not itself authorize private data, model downloads, training/fine-tuning, conversion, quantization, selection, deployment, or release.

Before consuming shared GPU capacity, tell the user which workload needs it and why. If the user reports an interactive GPU workload, defer new GPU work until the user says the device is available; do not create avoidable contention.

Android and iOS implementation phases run the owning repository's unit, conformance, integration, migration, UI/accessibility, build, and applicable device gates. Phase I cannot exit on SLM tests alone: every row of the cross-platform matrix must link to its result or an explicitly approved, non-promotion gap.

## Production-promotion gates

A proposed default cutover or release must demonstrate all of the following:

1. a frozen, validated Semantic V2 contract and native parser adapters with source-grounded evidence checks and hash drift protection;
2. user-approved data rights, private adjudication, held-out evaluation, and reproducible aggregate-safe provenance;
3. blinded per-platform evidence that passes the frozen threshold artifact, including error taxonomy, confidence requirements, and useful coverage rather than only headline accuracy;
4. a three-repository compatibility manifest with exact app/SLM commits, profiles, model/runtime, prompt/template, quantization, decode, filter, parser, evaluator, migration, and artefact hashes where available;
5. exact-money and timestamp-provenance persistence, tested migrations/rollback, idempotent retry/recovery, and no automatic persistence outside the narrow eligibility boundary;
6. Android and iOS native conformance, pipeline, UI, availability/recovery, memory, latency, battery, persistence, privacy, and supported-device/OS checks at the relevant tier;
7. no unresolved safety, privacy, data-rights, licensing, compatibility, or runtime-parity blocker; and
8. a completed default-off Phase I UAT candidate, recorded user acceptance, separately reviewed cutover changes, and an explicit Phase J release decision.
