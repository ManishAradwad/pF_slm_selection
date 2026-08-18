# PocketFinancer Extraction V2 status

## Completed and reviewed: Phase A — Foundation and Semantic V2

- Created the normative three-repository program plan and machine-readable state.
- Added `pocketfinancer_semantic_v2` version 2, including its platform-neutral JSON Schema and dependency-free Python reference implementation.
- Enforced scope, posting status, none/single/multiple cardinality, exact amount and currency derivation, explicit counterparty state, trusted host timestamp provenance, and zero-based half-open UTF-8 byte evidence.
- Closed the Phase A review findings: evidence ranges must be non-empty; account and present-counterparty values must exactly match their decoded evidence; direction must match the version 1 source lexicon; uppercase currency codes are handled generically but not when embedded inside larger words; a bare dollar sign fails closed; every record requires a text message even when it has no evidence; exact minor-unit scaling is bounded string arithmetic; INR auto-posting requires exact minor units; and none/multiple cardinality no longer emits false missing-field reasons.
- Added 13 fully invented conformance vectors and regression coverage for hallucinated, empty, reversed, unrelated, ambiguous-currency, inexact-money, signed-64-bit boundary, Unicode, repeated-value, and timestamp cases.
- Android and iOS were read-only evidence. No real SMS, private annotation, hosted service, model download, CUDA operation, inference, app code, app database, or production default was changed.

## Finalized program end state

- Phase I now requires every selected Android runtime-variant pipeline and the selected iOS pipeline to be implemented end to end in their owning repositories, not merely described in change plans.
- Phase H must record a selected profile ID or an explicit no-selection/exclusion reason for each of the five current Android runtime variants. At least one Android variant and iOS must be selected for Phase I; excluded variants remain outside the Semantic V2 UAT claim.
- Required work includes native adapters, filters, prompts/protocols, runtime integration, grounding and eligibility, exact-money and timestamp-provenance persistence, migrations and rollback, recovery/retry, and native unit, conformance, integration, UI, migration, build, and relevant device tests.
- Android must preserve or deliberately adapt its existing thinking/JSON token progress and pipeline-stage presentation through the final saved/review/retry/error result.
- iOS must expose the cumulative structured-generation snapshots Apple makes available and a truthful decision/persistence timeline. It must not claim token decoding, hidden reasoning, or chain-of-thought access.
- A versioned compatibility manifest must bind the preceding verified SLM implementation commit and exact compatible Android/iOS commits, profiles, contract and artefact hashes, platform protocols, migrations, selection decision, and verification evidence. The later manifest-record commit is reported externally because a commit cannot contain its own object ID.
- Phase I ends with reproducible default-off UAT candidates for the user's hands-on testing. Phase J records user acceptance and is the only phase that may separately authorize a default cutover or release.

## Branches, bases, and commits

- SLM branch: `codex/extraction-v2-foundation` from `78509644dad110219e7a53d7c503cd746f9e0ab6`.
- Android read-only base: `main` at `552ffbdfbd41773980aa249789b0cb508fdb19fd`.
- iOS read-only base: `main` at `04f770b235f860080ecd96adad1a5d011f3c2c2c`.
- Original program record: `2e50800` (`docs: record extraction V2 program`).
- Semantic V2 foundation: `823ce714e0015aac44415a842653ab744d776c53` (`feat: add Semantic V2 foundation`).
- Phase A review hardening: `95cb4287076642867596fbbfe2835cb105ff6399` (`fix: harden Semantic V2 contract`).
- Final review-gap closure: `0d1abf1be91a528baf0364d3475f2b4f25d72f5e` (`fix: close Semantic V2 review gaps`).

The follow-on plan/state/handoff commit intentionally records the immutable hardening commit above. Its own hash is reported by the task that creates it; a commit cannot contain its own final Git object ID.

## Frozen SHA-256 artifacts

- Normative plan: `6eccc6c7d5df21b4998079ff40c2d5c92e51af53a83a6cbad5fc15c29c7203f5`.
- Semantic V2 schema: `6c65b29543a85ca314f45620ed6300300d802ecab58d11371cc85af314a27bf8`.
- Semantic V2 reference: `ae0da990999035b8e04999d81854ed7ce4a95914a9079ff08eb50da03874c018`.
- Invented conformance fixture: `1bb7053d7b49066830eb0c08f546c8b3be5490e292f3497928c832f25b03f2fa`.

`tests/test_extraction_v2_program_state.py` recomputes these hashes and fails on unrecorded drift.

## Verification

- `pytest -q tests/test_extraction_v2_program_state.py tests/test_semantic_v2.py` — 8 passed.
- `python scripts/check_repo_safety.py` — passed; it reported only the pre-existing publication-review exception for `DATA/extraction_ds.jsonl`.
- `ruff check .` — passed.
- `ruff check --select E4,E7,E9,F lfm25 scripts tests` — passed.
- `pytest -q` — 563 passed.
- `git diff --check` — passed.

Only the lightweight repository tier was run. CUDA, model downloads, HF/GGUF inference, Android build/device validation, and iOS build/device validation were intentionally not run because Phase A did not change those layers.

## Unresolved empirical and implementation questions

- Which Direct V2, Candidate V2, or other protocol best serves each Qwen, Gemma, LFM, and Apple Foundation Models family remains unselected.
- Phase B must define and freeze the metric, confidence, sample-size, threshold, tie, and no-selection policy before protected or blinded evaluation.
- The current Android fleet's per-device parser acceptance, latency, memory, battery, recovery, persistence, and UI evidence has not been regenerated under Semantic V2.
- Android's current amount persistence uses binary floating point; the exact-money storage migration and rollback design must be baselined and tested before Phase I can complete.
- Apple Foundation Models behaviour remains OS/device dependent; no installed model revision is known or claimed.
- Real-data annotation policy, adjudication, data rights, sender/template-held-out splits, and acceptance thresholds require the user's explicit decisions.
- LFM2.5-350M trainability/quantization and LFM2.5-2.6B ceiling comparisons are unmeasured for this program.
- Neither native app has a Semantic V2 production adapter or manifest-bound UAT build yet; those are explicit later-phase deliverables.

## Next phase: B — Workbench V2, evaluation packages, and frozen decision policy

Prerequisites:

1. Start from `codex/extraction-v2-foundation` after the final plan/state/handoff commit reported by this task.
2. Read the applicable `AGENTS.md` guidance completely and verify the hash-lock test before acting.
3. Keep Semantic V2 and its frozen artifacts as the shared semantic boundary; version any intentional contract change instead of silently replacing a frozen hash.
4. Keep Android and iOS read-only throughout Phase B.
5. Do not inspect, print, modify, or transmit real SMS or private annotations. Obtain explicit user authorization before introducing any local private-data workflow.
6. Build the local workbench/evaluator, invented synthetic tests, and predeclared selection-policy artifact only. Do not train, select a profile, change production defaults, or proceed into Phase C.

Copy-paste prompt for the next task:

```text
Continue the PocketFinancer Extraction V2 program with Phase B only: Workbench V2, evaluation packages, and frozen decision policy. Start from the branch, commits, finalized plan, and hash-locked state recorded in docs/plans/POCKETFINANCER_EXTRACTION_V2_STATUS.md and configs/programs/pocketfinancer-extraction-v2.json. Read every applicable AGENTS.md completely before acting and run the state hash-lock test first. Preserve all unrelated changes. Keep Android and iOS read-only throughout Phase B. Use Semantic V2 as the shared semantic truth, but do not select, implement, or declare Direct V2 or Candidate V2 as a universal protocol. Build only local, privacy-safe workbench/evaluation foundations and the versioned metric, confidence, sample-size, threshold, tie, and no-selection policy, using invented synthetic tests. Do not inspect, print, modify, or transmit real SMS or private annotation rows, and do not train, download models, run CUDA/inference, change production defaults, or proceed into Phase C. Make focused local Conventional Commits, run the relevant lightweight checks, and update the program state and handoff with exact completed-commit IDs and the next allowed phase. Do not push or open a pull request.
```
