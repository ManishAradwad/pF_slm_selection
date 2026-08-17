# PocketFinancer Extraction V2 status

## Completed: Phase A — Foundation and Semantic V2

- Created the normative cross-platform program plan and machine-readable state.
- Added `pocketfinancer_semantic_v2` version 2, including its platform-neutral JSON Schema and dependency-free Python reference implementation.
- Enforced posted/not-posted, scope, none/single/multiple cardinality, optional evidence-backed amount/direction/account, explicit present/absent counterparty state, zero-based half-open UTF-8 byte evidence, exact decimal validation, derived currency, and non-rounding minor-unit derivation.
- Added the intentionally narrow initial auto-post eligibility projection. It is evaluation/reference logic only; no Android or iOS default changed.
- Added 12 fully invented synthetic golden vectors and unit coverage for validation, invalid combinations, exact decimal conversion, eligibility, UTF-8/unicode slicing, repeated-equal values, and host timestamp injection.
- Android and iOS were read-only evidence. No real SMS, private annotation, hosted service, model download, CUDA operation, or inference was used.

## Branches, bases, and commits

- SLM branch: `codex/extraction-v2-foundation` from `78509644dad110219e7a53d7c503cd746f9e0ab6`.
- Android read-only base: `main` at `552ffbdfbd41773980aa249789b0cb508fdb19fd`.
- iOS read-only base: `main` at `04f770b235f860080ecd96adad1a5d011f3c2c2c`.
- Semantic V2 implementation commit: `823ce714e0015aac44415a842653ab744d776c53` (`feat: add Semantic V2 foundation`).

The follow-on documentation/state commit intentionally records the immutable implementation commit above. Its own hash is reported by the task that creates it; a commit cannot contain its own final Git object ID.

## Verification

- `pytest -q tests/test_semantic_v2.py` — 6 passed.
- `ruff check --select E4,E7,E9,F lfm25/semantic_v2.py tests/test_semantic_v2.py` — passed.
- `python scripts/check_repo_safety.py` — passed; it reported only the pre-existing publication-review exception for `DATA/extraction_ds.jsonl`.
- `ruff check .` — passed.
- `ruff check --select E4,E7,E9,F lfm25 scripts tests` — passed.
- `pytest -q` — 561 passed.
- `git diff --check` — passed.

Only the lightweight repository tier was run. CUDA, model downloads, HF/GGUF inference, Android build/device validation, and iOS build/device validation were intentionally not run because Phase A does not change those layers.

## Unresolved empirical questions

- Which Direct V2, Candidate V2, or other protocol best serves each Qwen, Gemma, LFM, and Apple Foundation Models family remains unselected.
- The current Android fleet's per-device parser acceptance, latency, memory, battery, and recovery evidence has not been regenerated under Semantic V2.
- Apple Foundation Models behaviour remains OS/device dependent; no installed model revision is known or claimed.
- Real-data annotation policy, adjudication, data rights, sender/template-held-out splits, and acceptance thresholds require the user's explicit decisions.
- LFM2.5-350M trainability/quantization and LFM2.5-2.6B ceiling comparisons are unmeasured for this program.

## Next phase: B — Workbench V2 and evaluation packages

Prerequisites:

1. Start from `codex/extraction-v2-foundation` after the documentation/state commit recorded by the current task.
2. Read the applicable `AGENTS.md` guidance completely and keep Android/iOS read-only unless a new task explicitly authorizes changes.
3. Treat `pocketfinancer_semantic_v2` and its synthetic vectors as the shared semantic boundary; do not select Direct V2 or Candidate V2.
4. Do not inspect, print, modify, or transmit real SMS or private annotations. Obtain explicit user authorization before introducing any local private-data workflow.
5. Keep workbench outputs local/ignored and add only invented synthetic tests to Git.

Copy-paste prompt for the next task:

```text
Continue the PocketFinancer Extraction V2 program with Phase B only: Workbench V2 and evaluation packages. Start from the branch/state recorded in docs/plans/POCKETFINANCER_EXTRACTION_V2_STATUS.md and configs/programs/pocketfinancer-extraction-v2.json. Read every applicable AGENTS.md completely before acting. Preserve all unrelated changes. Treat Android and iOS as read-only unless I explicitly authorize changes. Use Semantic V2 as the shared semantic truth, but do not select, implement, or declare Direct V2 or Candidate V2 as a universal protocol. Build only local, privacy-safe workbench/evaluation foundations with invented synthetic tests; do not inspect, print, modify, or transmit real SMS or private annotation rows, and do not train, download models, run CUDA/inference, change production defaults, or proceed into Phase C. Make focused local Conventional Commits, run the relevant lightweight checks, and update the program state and handoff with exact completed-commit IDs and the next allowed phase. Do not push or open a pull request.
```
