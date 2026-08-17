# PocketFinancer transaction-extraction program, V2

## Status and authority

This is the normative program plan for reliable local extraction of completed Indian financial transactions. It governs the Android app, iOS app, and the `pF_slm_selection` research/evaluation repository. Semantic truth and evaluation requirements are shared; application code, model prompts, parser adapters, runtime settings, and user interfaces remain platform-specific.

PocketFinancer Android's committed source and its locked profile remain the authority for existing Android behaviour. This plan does not alter production defaults, the locked Android profile, the Android app, or the iOS app. The machine state is in `configs/programs/pocketfinancer-extraction-v2.json`; the task handoff is in `docs/plans/POCKETFINANCER_EXTRACTION_V2_STATUS.md`.

## Shared semantic boundary

Semantic V2 (`pocketfinancer_semantic_v2`, version 2) is the shared host semantic truth, not a universal model protocol. Its versioned JSON Schema is `configs/contracts/pocketfinancer-semantic-v2.schema.json`; the dependency-free Python reference is `lfm25/semantic_v2.py`.

Every evaluated interpretation must state all of the following:

- scope: `bank_card`, `wallet_bnpl`, or `other`;
- posting status: `posted` or `not_posted`;
- event cardinality: `none`, `single`, or `multiple`;
- for each event, optional amount, direction, and account evidence;
- an explicit counterparty state: `present` with evidence or `absent`.

Evidence ranges are zero-based, half-open UTF-8 byte offsets into the original message. Exact `decimal_text` is authoritative. Currency is derived from and validated against selected amount evidence. Minor units are deterministic host derivatives and are `null` when their calculation would need rounding; rounding is forbidden.

The initial automatic-post projection is deliberately narrow: exactly one `posted`, `single`, INR `bank_card` event with amount, direction, and account evidence. It is an evaluation/reference boundary only in Phase A; it does not authorize an application behaviour change.

## Timestamp and privacy policy

Transaction date is source metadata, never model output. Android supplies `SmsMessage.date`. iOS uses the persisted `InboxAlert.receivedAt` and records whether it was supplied or assigned during ingestion. Semantic V2 injects this metadata after interpretation and preserves provenance. No prompt, model target, output schema, or parser acceptance rule may ask a model to infer or emit a date.

Raw alerts, senders, account labels, private annotations, row-level predictions, stable source mappings, credentials, and model weights stay local. They must not enter Git, logs, hosted APIs, telemetry, cloud inference, datasets, screenshots, or task transcripts. The synthetic fixture committed in Phase A is entirely invented. The user exclusively owns real-data annotation, adjudication, split, and migration decisions.

## Model and runtime roles

Each model/runtime family receives its own versioned prompt, output, parser, decoding, and runtime profile. All profiles map into Semantic V2; none is presumed portable merely because the semantics are shared. Direct V2 and Candidate V2 are both unselected hypotheses. They must be compared independently per family and platform; Phase A makes neither a universal protocol.

Android's presently deployed selectable fleet is Qwen3-0.6B Q8, Qwen3-1.7B Q4/Q8, and Gemma 4 E2B Q4/Q8. Their chat-template handling, thinking behaviour, quantization, grammar use, parser behaviour, and device limits are separate experimental variables. LFM2.5-350M is the primary small trainable research candidate, not an Android-selectable production model. LFM2.5-2.6B Base and Post are research ceilings, not mobile deployment targets.

iOS uses `Apple SystemLanguageModel.default`. Its installed revision is opaque, OS-managed, and device-dependent. The iOS program evaluates availability, structured-generation behaviour, parser grounding, latency, and recovery without claiming a fixed model revision or equating it to Android model evidence.

## Controlled experiments and evidence

Experiments must pin the Semantic V2 schema/reference hash, prompt/output profile ID, model revision, chat template, quantization, decode settings, parser/evaluator revision, filter version, data/split hash, seed, selected checkpoint, and runtime environment. Hold out sender/template families where the user has approved data. Compare one material variable class at a time, or mark the comparison non-causal. Store aggregate-safe metrics and provenance, never raw private rows.

Host/HF measures, GGUF host measures, Android-device measures, and iOS-device measures are separate evidence classes. No host timing is phone timing. A profile is not selected from a single model family, a single device, or an unblinded review.

## Phases

### A. Foundation and Semantic V2

Entry: the three worktrees and guidance are recorded; Android and iOS are read-only evidence; no private data is inspected.

Exit: committed program/state/handoff artifacts; a versioned platform-neutral schema and reference validator; host timestamp provenance; exact-money and UTF-8 evidence rules; invented conformance vectors; unit coverage; and the lightweight repository gate. No production default changes and no model protocol selection.

### B. Workbench V2 and evaluation packages

Entry: Phase A is complete, the schema and synthetic vectors are frozen by hash, and the user has explicitly authorized any private annotation work that is proposed.

Exit: a local-only workbench/evaluation package can create, validate, anonymize, and score Semantic V2 annotations without raw data in Git; split/adjudication/provenance interfaces are defined; and synthetic smoke tests prove fail-closed handling. No training, model selection, or application integration follows automatically.

### C. Android baseline reproducibility and runtime instrumentation

Entry: Phase B's shared evaluator is available and the current Android source/profile relationship has been audited without changing the locked production defaults.

Exit: reproducible Android baseline captures filter, prompt assets, parser, chat template, decode/runtime settings, model identifiers, and aggregate device instrumentation. Host and device measures are labelled separately, and baseline gaps are recorded.

### D. Android prompt/output-protocol laboratory

Entry: Phase C baseline evidence and fixtures are reproducible for each Android family under test.

Exit: controlled, evaluation-only comparisons independently test Direct V2 and Candidate V2 (and any justified alternatives) for Qwen and Gemma. Each result has protocol/parser/runtime provenance, aggregate failure taxonomy, and device evidence. A result may select no protocol.

### E. LFM2.5-350M data, fine-tuning and quantization program

Entry: Phase B supplies user-approved local data interfaces and the chosen experiment design explicitly names its LFM profile. Any real labels/splits have user approval and remain private.

Exit: reproducible, non-overwriting data build, fine-tuning, checkpoint selection, merge, conversion, and quantization evidence for LFM2.5-350M. LFM2.5-2.6B Base/Post may be measured only as research ceilings. Results do not establish Android deployability.

### F. Android evaluation-only model integration and device validation

Entry: a Phase D or E candidate has a complete profile and artefact provenance, and Android integration remains evaluation-only behind an explicit non-default path.

Exit: parser, filter, prompt, runtime, memory, latency, recovery, and device validation are recorded for the exact artefact. The production fleet/defaults remain unchanged pending Phase I.

### G. iOS Foundation Models evaluator and prompt/output-protocol laboratory

Entry: Phase B evaluator semantics are available, and tests handle `SystemLanguageModel.default` unavailability without data loss.

Exit: iOS evaluation-only profiles independently compare prompt/output approaches, parser grounding, availability/recovery, and device behaviour with OS/device provenance. No fixed Apple model revision or Android parity is claimed.

### H. Blinded cross-platform evaluation and profile selection

Entry: the intended Android and iOS candidates have completed their evaluation-only evidence paths, and the user has approved blinded human evaluation and test governance.

Exit: a blinded, aggregate-safe cross-platform report uses the shared Semantic V2 evaluator; selected profile IDs (or an explicit no-selection decision) are recorded; and known runtime/parity gaps are disclosed.

### I. Separate production-cutover phase

Entry: Phase H selects a profile, every production-promotion gate below passes, and the user explicitly authorizes a cutover proposal.

Exit: separately reviewed platform changes have migration, rollback, default/upgrade behaviour, observability, privacy, parser, and device validation plans. Deployment occurs only after the user explicitly decides to ship; evaluation success alone never changes a production default.

## Branch strategy and verification

Use a short-lived `codex/extraction-v2-<phase>` branch per repository. Phase work changes only the repository that owns the implementation; cross-repository coordination is versioned in the SLM repository. Preserve unrelated work, never reset/stash/clean, and use Conventional Commits. Android and iOS changes, when authorized in later phases, use their own focused branches and checks.

Every Phase A/B repository change runs the SLM lightweight gate: `python scripts/check_repo_safety.py`, `ruff check .`, `ruff check --select E4,E7,E9,F lfm25 scripts tests`, `pytest -q`, and `git diff --check`. Profile/pipeline/lock changes additionally run `python scripts/run_pocketfinancer_pipeline.py check`. CUDA, model downloads, HF inference, GGUF conversion, and device checks run only in the phases that need them and report their tier honestly.

## Production-promotion gates

A proposed production cutover must demonstrate all of the following:

1. a frozen, validated Semantic V2 contract and parser adapter with source-grounded evidence checks;
2. user-approved data rights, private adjudication, held-out evaluation, and reproducible provenance;
3. blinded cross-platform evidence with acceptable error taxonomy, not only headline accuracy;
4. exact profile/model/prompt/template/quantization/decode/parser/runtime versions and rollback capability;
5. Android and iOS device behaviour, availability/recovery, memory, latency, battery, persistence, and privacy checks at the relevant tier;
6. no unresolved safety, privacy, licensing, or runtime-parity blocker; and
7. an explicit user release decision after a separately reviewed Phase I change.
