# Command map

Run commands from `/home/tojinotzenin/pF_slm_selection` inside WSL2 after
`source scripts/activate_wsl.sh`.

## Primary PocketFinancer pipeline

`scripts/run_pocketfinancer_pipeline.py` is the single supported entry point for
new app-facing work. It reads
`configs/pipelines/pocketfinancer-lfm2.5-350m.json` and exposes these stages:

| Stage | Purpose |
|---|---|
| `check` | Report local readiness and exact inputs without modifying anything |
| `plan` | Print every command as an argv array |
| `build-data` | Materialize app-filtered, source-grounded private train/dev rows |
| `train` | Train completion-only LoRA with the exact PocketFinancer messages |
| `evaluate-hf` | Fast BF16/adapter diagnostic using the app prompt and prefilter |
| `merge` | Merge the chosen adapter into the HF base |
| `convert` | Produce reference and deployable GGUF files |
| `evaluate-gguf` | Evaluate the deployable artifact under the app runtime profile |

Use `--dry-run` with an execution stage to inspect only that command. `all` exists
for an intentional end-to-end run, but stage-by-stage execution is preferable while
developing because each artifact can be inspected before the next expensive step.

## Environment and safety

| Command | Purpose |
|---|---|
| `python scripts/verify_gpu.py` | Verify CUDA/PyTorch and local runtime readiness |
| `python scripts/verify_lfm25_backward.py` | Run a real short LFM backward-pass probe |
| `python scripts/check_repo_safety.py` | Check that private/generated artifacts cannot be committed |
| `bash scripts/setup_wsl.sh` | Bootstrap the pinned native WSL environment |

## Current data builders

| Command | Purpose |
|---|---|
| `python scripts/build_lfm25_private_sft_v2.py` | Rebuild the private source-grounded direct SFT set |
| `python scripts/build_lfm25_candidate_sft.py` | Convert grounded direct rows to candidate-selector rows |
| `python scripts/build_lfm25_candidate_curriculum.py` | Build low-weight semantic curriculum and mixed training data |
| `python scripts/audit_lfm25_candidate_coverage.py` | Audit deterministic candidate coverage without training |

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
