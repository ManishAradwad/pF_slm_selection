# PocketFinancer SLM training readiness

> Historical readiness snapshot. Training has since completed under the current
> Android `a9b7df44` profile. Use
> `docs/experiments/POCKETFINANCER_A9_LORA_R16_S17.md` for measured results and
> `configs/pipelines/pocketfinancer-lfm2.5-350m.json` for the active pipeline.

This note is the handoff for the dedicated fine-tuning goal. It records what is
actually present in the WSL checkout, what the existing experiments prove, and
which controls are required before any training result can be trusted.

## Working environment

- Canonical working copy: `/home/tojinotzenin/pF_slm_selection` on the native
  WSL2 Linux filesystem.
- Hardware visible to WSL: RTX 4070 with 12 GB VRAM, 15 GB RAM, 4 GB swap.
- Use the user-owned Python 3.11 environment at `.venv`; activate it with
  `source scripts/activate_wsl.sh`.
- Keep datasets, HF weights, checkpoints, and GGUFs under `/home`. The mounted
  Windows `D:` drive was already 95% full during the audit.
- Docker is not part of the active workflow. The obsolete project container,
  images, volume, build cache, and `pf_docker` shim were removed on 2026-08-03.

Verified on 2026-08-03:

- PyTorch `2.6.0+cu124` reports CUDA available and identifies the RTX 4070.
- `llama-cpp-python 0.3.20` found CUDA compute capability 8.9 and offloaded all
  19 layers of the smallest local GGUF.
- A one-example Gemma-4 E2B Q4 run completed through the real task loader,
  cached tokenizer, GBNF grammar, and current metrics with valid exact output.
- `bitsandbytes 0.50.0` executed an NF4 `Linear4bit` layer on `cuda:0`; PEFT
  `0.20.0` and TRL `1.9.2` import successfully together.
- All 105 installed packages pass `uv pip check`.

The original Hugging Face safetensors for the finalists are not intentionally
downloaded yet. Download them after the training goal chooses its base model;
the existing GGUFs are sufficient for baseline evaluation but not training.

## Data available

The private source archive is a useful annotation pool, not ready-made
supervised training data.

| Artifact/pool | Count | Meaning |
|---|---:|---|
| Exported messages | 17,830 | CSV and JSON exactly mirror the valid SQLite export |
| Incoming messages | 17,584 | Candidate source pool |
| Strict heuristic positives | 2,372 | Weak labels, not ground truth |
| Strict heuristic rejects | 14,949 | Weak labels, not ground truth |
| Broad heuristic positives | 10,975 | Shows how uncertain the weak labeling is |
| Structured labeled benchmark | 203 | 114 transactions, 89 nulls; repeatedly tuned evaluation set |

Important properties:

- The 203 labeled rows overwhelmingly came through the same deterministic
  filter used to construct the pool: all 114 transactions pass it and 88/89
  nulls fail it. The benchmark therefore contains too few hard boundary cases.
- Repeated bank templates are common. Random row splitting would leak template
  structure across train and test.
- The archive appears to represent one person's financial ecosystem. It cannot
  by itself establish generalization across users, banks, regions, or devices.
- Do not train on the current 203 rows while continuing to report the historic
  scores. Doing so would turn the existing benchmark into training data.

## Current model-selection result

The trustworthy current-schema artifacts evaluate the same 203 messages with
grammar-constrained decoding and the production `extract_json_nonnull` filter.

| Variant | Exact | Ghosts | Misses | Runtime | GGUF size |
|---|---:|---:|---:|---:|---:|
| Gemma-4 E2B Q8 | 176/203 (86.70%) | 15/89 negatives | 0/114 | 210.3 s | 5.05 GB |
| **Gemma-4 E2B Q4** | **175/203 (86.21%)** | **17/89** | **0/114** | **208.3 s** | **3.11 GB** |
| Qwen3-1.7B Q8 | 170/203 (83.74%) | 7/89 | 10/114 | 905.8 s | 1.83 GB |
| Qwen3-1.7B Q4 | 170/203 (83.74%) | 9/89 | 16/114 | 714.4 s | 1.11 GB |

Practical default before fine-tuning: **Gemma-4 E2B Q4_K_M**. Q8 adds one
correct example for 62.5% more storage. Qwen Q8 is smaller and emits fewe
ghost transactions, but is 4.35x slower than Gemma Q4 and misses ten real
transactions. The five-example Gemma-Q4 advantage over Qwen-Q8 is not
statistically decisive on this small, biased test set, so both are reasonable
fine-tuning challengers.

## Required dataset build

Before SFT/LoRA:

1. Preserve source message ID and date, normalized-template group, annotation
   method, annotator decision, and confidence.
2. Deduplicate/group messages by normalized bank template before splitting.
3. Human-label strict positives plus stratified negatives, especially amount +
   account + transaction-verb near misses, OTPs, reversals, failed transactions,
   mandates, balance alerts, and payment requests.
4. Create train/dev/test splits by template/sender and time, not random row.
5. Freeze an untouched recent test set containing unseen templates/banks before
   the first training run.
6. Keep the binary transaction decision distinct from conditional extraction
   fields so classification and field failures can be diagnosed separately.
7. Add synthetic or separately sourced, human-validated templates for coverage;
   never allow paraphrases of a test template into training.

## Training experiment shape

- Start from original Hugging Face safetensors, not the existing GGUF files.
- With 12 GB VRAM, use LoRA/QLoRA for 1-2B models, gradient checkpointing, and
  short sequence lengths based on the observed SMS distribution. Do not begin
  with full-parameter AdamW training.
- Run the same frozen dataset and hyperparameter budget on both finalists where
  library support permits; do not choose a winner from differently curated data.
- Keep a zero-shot/base checkpoint result beside every tuned checkpoint to
  measure genuine gain and catastrophic regressions.
- Merge the winning adapter, export to GGUF, sweep practical quantizations, then
  evaluate through the existing llama.cpp/GBNF path used by `llama.rn`.

## Acceptance metrics

Report counts and conditional rates, not only means over all 203 rows:

- transaction precision/recall/F1;
- ghost rate over gold-null examples;
- missed rate over gold-transaction examples;
- exact four-field match;
- amount, type, account, and counterparty accuracy on relevant transactions;
- JSON validity, few-shot leakage, model size, latency, and peak memory;
- confidence intervals or paired bootstrap/McNemar comparisons on the frozen
  test set;
- final latency and memory on the target Android device via `llama.rn`.

## Privacy boundary

Raw SMS, prompts, result samples, checkpoints, and model caches are private and
must stay local and gitignored. Redact/tokenize phone numbers, VPAs, references,
links, names, and account suffixes in any artifact leaving this machine. Avoid
cloud experiment tracking unless explicitly configured for private, redacted
data. Fine-tuned weights may memorize rare strings, so treat checkpoints as
sensitive until memorization tests pass.

## Known cleanup/fixes from the readiness pass

- Native WSL entry points now use `.venv` instead of the deleted `pf_docker`.
- The task YAML uses a repository-relative dataset path.
- The HTML report now reads `counterparty_accuracy`; it previously requested the
  obsolete `merchant_accuracy` key and displayed zero for current runs.
- Repository and HF cache ownership were restored from `root` to the WSL user.
- `HF_TOKEN` was not set during the audit. It is needed only when a gated model
  artifact is absent from the local cache.
