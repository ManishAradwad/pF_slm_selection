# CLAUDE.md - SLM Evaluation for Financial SMS

## Project Overview
Evaluation playground for selecting the optimal Small Language Model (SLM) to power a mobile financial helper app called **pocket-financer**. The app tracks the user's bank accounts and credit/debit cards, then uses an SLM to process transaction SMS alerts and maintain balance/spend data. Wallets (Simpl, slice, PayTM wallet) are out of scope — only direct bank account and card transactions. Production runtime is **`llama.rn`** (React Native binding over `llama.cpp`) on Android.

## Evaluation approach
The task is **structured JSON extraction**, not classification. Given an Indian bank/card SMS, the model outputs either the literal word `null` (not a real transaction) or a JSON object with 5 fields: `amount`, `merchant`, `date`, `type` (`debit`|`credit`), `account`. Pipeline is built out-of-tree on `lm-evaluation-harness`.

**Backend**: `llama-cpp-python` — the same engine as production `llama.rn`. This is the whole reason to avoid the HF `transformers` backend: quantization, tokenization, and sampler match on-device, so dev metrics honestly predict production behavior.

**Grammar-constrained decoding**: A GBNF grammar (`DATA/sms_extraction.gbnf`) enforces output shape. `amount`, `type`, `account` are non-nullable; `merchant`, `date` are nullable. The same grammar ships on-device.

**Two metric filter pipelines** run in parallel on every eval:
- `extract_json` — raw model output (cleaned to first JSON object or `null`).
- `extract_json_nonnull` — applies a reject rule: if the output dict has `null` in any of `amount`, `type`, or `account`, treat the whole prediction as `null`. The model's own null-emission is used as an implicit confidence signal — no per-model calibration needed, works across any SLM. See `_REQUIRED_NONNULL_FIELDS` in `DATA/utils.py`.

Every metric appears twice in results files with `,extract_json` and `,extract_json_nonnull` suffixes. Compare the two columns to see raw-model vs production-filtered behavior.

**Merchant match is intentionally loose** — case-insensitive + whitespace-collapsed + substring either direction (3-char floor). Bank SMS wrap the same entity in cosmetic boilerplate (`VPA x@y`, `mobile 9XXX-APIBANKING`, `UPI-<ref>-Compass`, trailing city names) and the model shouldn't be punished for failing to strip it. `None` vs non-`None` stays strict — over-extraction is still an error. See `_merchant_match` in `DATA/utils.py`.

## Entry point
```bash
source pf_docker/bin/activate
python run_gguf_eval.py \
  --model google/gemma-4-E2B-it \
  --gguf MODELS/google_gemma-4-E2B-it-Q4_K_M.gguf \
  --grammar DATA/sms_extraction.gbnf \
  [--limit 10]    # smoke test
```
Outputs land in `RESULTS/llamacpp/<model_slug>/`. Omit `--grammar` to compare grammar-free behavior.

## Key files

### Evaluation pipeline
- `run_gguf_eval.py` — entry script; registers the adapter and calls `lm_eval.simple_evaluate`.
- `DATA/llamacpp_model.py` — out-of-tree `@register_model("llamacpp")` adapter over `llama-cpp-python`. Accepts `grammar_file=...` via `model_args`, passes `grammar=...` into `Llama.create_completion`. Uses HF `AutoTokenizer` (via `--model <hf_id>`) for chat-template rendering so prompts are byte-compatible with prior HF runs. Strips a leading `<bos>` from the rendered template because `llama-cpp-python` adds its own BOS and a duplicate hurts quality.
- `DATA/sms_extraction.yaml` — task config. `output_type: generate_until`, two parallel filter pipelines, ~11 metrics per filter.
- `DATA/sms_extraction.gbnf` — GBNF grammar for JSON output.
- `DATA/utils.py` — `SYSTEM_PROMPT`, `FEW_SHOT_EXAMPLES`, `doc_to_text`, `extract_json_filter`, `extract_json_nonnull_filter`, metric functions (full_match, ghost_rate, missed_rate, per-field accuracy, few-shot leakage). Has a `__main__` self-test suite — run `python DATA/utils.py` after any metric/normalizer edit.
- `DATA/extraction_ds.jsonl` — 203 labeled samples (114 real transactions, 89 non-transactions). **Do NOT edit.**

### SMS filtering pipeline (upstream — builds the raw data that feeds the eval)
- `old_pipeline.ipynb` — v1 filtering (negative filters; superseded).
- `new_pipeline.py` — v2 filtering; positive filters (masked account + transaction verb) for far fewer false negatives on real bank messages.
- `build_datasets.py`, `expand_dataset.py`, `export_sms.py`, `find_sms_db.py` — dataset construction utilities. `find_sms_db.py` extracts `sms.db` from an iPhone iTunes backup.
- `copilot-instructions.md` — full original project context and goals.

### Models and results
- `MODELS/` — GGUF weights (gitignored). Use Q4_K_M to match what `llama.rn` ships on-device.
- `RESULTS/llamacpp/<model_slug>/` — per-run output (`results_<ts>.json` + `samples_<task>_<ts>.jsonl`).
- `RESULTS/new_pipeline/` — legacy HF-backend baselines; kept for reference but not the current source of truth.

## Environment
- **Docker dev container**: `pf_docker/` — `source pf_docker/bin/activate`. Python 3.11 (bullseye), Node 20, GPU passthrough works for torch / HF.
- **WSL2 bare**: `pf/` — separate venv with hardcoded paths; not usable inside Docker.
- **Hardware**: WSL2, NVIDIA RTX 4070 (12 GB VRAM), 32 GB RAM.
- **`llama-cpp-python` is CPU-only in the Docker container.** Prebuilt CUDA wheels need glibc ≥ 2.32 and GLIBCXX_3.4.29/3.4.30; Debian bullseye ships glibc 2.31. GPU support would require installing the CUDA toolkit in the container and rebuilding from source. CPU completes a full 203-sample run in ~10 min, which is fine for iteration.

## Conventions
- Install dependencies into the active venv, not globally.
- Evaluation datasets in `DATA/`, results in `RESULTS/`, GGUF weights in `MODELS/` (gitignored).
- **`lm-evaluation-harness/` is an upstream clone.** Prefer out-of-tree extensions — custom tasks via `--include_path`, custom models via `@register_model` in a module outside the clone — so it stays rebasable. Edit inside the clone only when the extension point genuinely doesn't exist.
- Data files (csv, json, db, xlsx, gguf) are gitignored — never commit them.
- **Do not name dataset senders in `SYSTEM_PROMPT`.** The pre-2026-04-17 prompt listed `JK-620016`, `VM-OFFERZ`, `VK-GOIBIB`, `VM-NOBRKR` as example promo senders — all of which appear in the eval dataset. The current prompt describes the XX-YYYYYY Indian SMS sender format generically instead, at a small accuracy cost. Keep it generic.
- Keep the eval pipeline model-agnostic: filters and rules should express properties of the domain (bank SMS always contain amount/type/account), not properties of any specific SLM.

## Known state
- Evaluation pipeline is clean and operational for Gemma-4-E2B-it Q4_K_M.
- SLMs still to benchmark with the same pipeline: Qwen3.5-0.8B, Qwen3-0.6B, LFM2.5-1.2B-Instruct.
- SMS data covers 2021–2026 with coverage tapering after 2024.
- `pf_docker` has torch, transformers, accelerate, lm-eval, llama-cpp-python 0.3.20 (CPU build).
