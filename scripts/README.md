# Command map

Run commands from `/home/tojinotzenin/pF_slm_selection` inside WSL2 after
`source scripts/activate_wsl.sh`.

## Primary PocketFinancer pipeline

`scripts/run_pocketfinancer_pipeline.py` is the single supported entry point for
new app-facing work. The default and canonical declaration is
`configs/pipelines/pocketfinancer-lfm2.5-350m.json`. The parallel
`configs/pipelines/pocketfinancer-lfm2.5-2.6b-base.json` declaration applies the
same workflow to the pinned LFM2.5-2.6B-Base checkpoint and must be selected
explicitly with `--config`. Both declarations expose these stages:

| Stage | Purpose |
|---|---|
| `check` | Report local readiness and exact inputs without modifying anything |
| `plan` | Print every command as an argv array |
| `build-data` | Materialize app-filtered, source-grounded private train/dev rows |
| `evaluate-base-hf` | Evaluate the locked, untuned HF checkpoint before adaptation |
| `train` | Train completion-only LoRA with the exact PocketFinancer messages |
| `evaluate-hf` | Evaluate the trained HF adapter using the app prompt and prefilter |
| `merge` | Merge the chosen adapter into the HF base |
| `convert` | Produce reference and deployable GGUF files |
| `evaluate-gguf` | Evaluate the deployable artifact under the app runtime profile |

Execution order is `build-data`, `evaluate-base-hf`, `train`, `evaluate-hf`,
`merge`, `convert`, then `evaluate-gguf`. Use `--dry-run` with an execution stage
to inspect only that command. `all` exists for an intentional end-to-end run, but
stage-by-stage execution is preferable while developing because each artifact can
be inspected before the next expensive step.

When thinking mode is enabled, the HF and GGUF evaluators retain raw reasoning
only in `samples.jsonl` beneath ignored local `RESULTS/` directories; aggregate
metrics do not contain it. Evaluation timing is measured on the WSL host, not on
an Android device, and is not Android runtime-performance evidence.

## Environment and safety

| Command | Purpose |
|---|---|
| `python scripts/verify_gpu.py` | Verify CUDA/PyTorch and local runtime readiness |
| `python scripts/verify_lfm25_backward.py` | Run a real short LFM backward-pass probe |
| `python scripts/probe_lfm25_lora_memory.py --help` | Inspect the one-backward-pass LoRA capacity gate; it is not training or quality evidence |
| `python scripts/check_repo_safety.py` | Check that private/generated artifacts cannot be committed |
| `bash scripts/setup_wsl.sh` | Bootstrap the pinned native WSL environment |

## Current data builders

| Command | Purpose |
|---|---|
| `python scripts/review_lfm25_blinded_test.py export` | Freeze all test rows into a reviewer-blind, test-only local package |
| `python scripts/review_lfm25_blinded_test.py validate` | Validate complete or resumable partial human annotations without writing outputs |
| `python scripts/review_lfm25_blinded_test.py import` | Write completed labels to a separate reviewed manifest plus aggregate-only report |
| `python scripts/build_lfm25_private_sft_v2.py` | Rebuild the private source-grounded direct SFT set |
| `python scripts/build_lfm25_candidate_sft.py` | Convert grounded direct rows to candidate-selector rows |
| `python scripts/build_lfm25_candidate_curriculum.py` | Build low-weight semantic curriculum and mixed training data |
| `python scripts/audit_lfm25_candidate_coverage.py` | Audit deterministic candidate coverage without training |

The blinded workflow writes only below ignored `PRIVATE_DATA/lfm25`. Give reviewers
only `blinded_test_review.jsonl`; the ID map and provenance metadata are internal.
Untouched rows stay blank so review can resume later. Import never edits the frozen
`split_manifest.jsonl`: it writes `split_manifest_human_reviewed.jsonl` and an
aggregate report separately. Existing nonempty outputs require an explicit
`--force`. Package metadata and the import report are committed last as validity
markers; a detected source change rolls the group back to its prior state.

For each completed row, set `decision` to `transaction` or `not_transaction`,
then fill `reviewer` and an ISO-8601 `reviewed_at` timestamp with a timezone.
Transactions also require a numeric `amount`, `type` (`debit` or `credit`), and
nonempty `account`; `counterparty` may be null. Non-transactions keep all four
extraction fields null. Leave every annotation field null on untouched rows;
`notes` is optional only after a decision is complete.

The filenames preserve compatibility with historical manifests. Dataset status and
actual semantic versions are recorded in [the experiment catalog](../docs/experiments/EXPERIMENT_CATALOG.md).

## Underlying training and evaluation commands

| Command | Purpose |
|---|---|
| `python scripts/train_lfm25_lora.py --help` | Low-level LoRA trainer; PocketFinancer is the default profile |
| `python scripts/evaluate_lfm25_candidate_hf.py --help` | Evaluate candidate selection and strict reconstruction |
| `python scripts/evaluate_lfm25_android_hf.py --help` | HF prompt/training proxy; not a llama.cpp/Android runtime proof |
| `python scripts/evaluate_lfm25_android_gguf.py --help` | Evaluate a deployable GGUF against the current runtime contract |
| `python scripts/merge_lfm25_lora.py --help` | Merge a selected adapter into its HF base |
| `bash scripts/convert_lfm25_gguf.sh --help` | Convert/quantize a merged model with the pinned toolchain |
| `python scripts/compare_lfm25_predictions.py --help` | Compare paired per-row predictions |

## Historical/general model slate

`bash scripts/evaluate.sh` and root `run_gguf_eval.py` are the older broad GGUF
benchmark. They remain useful for exploration but are not the default app-facing
pipeline.

The dated results and failure signatures from that work are preserved in the
[general GGUF benchmark history](../docs/history/GENERAL_GGUF_BENCHMARK_2026-04-25.md).

## Historical commands

`scripts/run_lfm25_experiments.py`, `evaluate_lfm25_hf.py`,
`evaluate_lfm25_gguf.py`, `build_lfm25_synthetic_sft.py`, and
`build_lfm25_public_candidate.py` reproduce older short-prompt/synthetic work. Do
not use them to claim current app parity. The runner intentionally rejects nonlegacy
prompt profiles and fails closed on stale cached provenance.
