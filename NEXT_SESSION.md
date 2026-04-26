# 2026-04-25 session — first full model sweep

## What we did

### Candidate slate finalised
Added three models to the original four:
- **Gemma-3-270M-it** (270M, Q4_K_M) — gated; user accepted licence at `huggingface.co/google/gemma-3-270m-it`
- **Qwen3-1.7B** (1.7B, Q4_K_M) — added as a mid-size Qwen reference point
- **Bonsai-1.7B / Bonsai-4B** (Q1_0) — kept in the table for completeness but **blocked** (see below)

Full slate now in `CLAUDE.md § Candidate models`.

### Pipeline improvements shipped

**`DATA/llamacpp_model.py`**
1. BOS stripping is now dynamic: uses `hf_tokenizer.bos_token` instead of the hard-coded `"<bos>"` string. Fixes LFM2.5 which uses `<|startoftext|>` as its BOS.
2. `n_ctx` auto-detection: new `_read_gguf_n_ctx_train()` parses GGUF metadata without loading the model and sets context to `min(native, 32768)`. The 32k cap is a VRAM budget — llama.cpp pre-allocates the full KV cache, so Qwen3.5-0.8B's native 262k or LFM2.5's 128k would OOM on our 12GB GPU. Pass `--n-ctx N` to override.
3. Default `n_ctx` in `run_gguf_eval.py` changed from 131072 to 0 (meaning auto-detect).

**`scripts/`**
- `fetch_models.sh` — added Gemma-3-270M-it and Qwen3-1.7B download entries
- `run_all_evals.sh` — new script; runs all non-Gemma-4 models sequentially, grammar-constrained, auto n_ctx. Passes through any `--limit N` flag for smoke tests. Continues on per-model failure. Usage: `bash scripts/run_all_evals.sh`

### All GGUFs downloaded
All 9 model files now present in `MODELS/`. See `scripts/fetch_models.sh` for sources.

### Eval results — full 203-sample grammar run

| Model | full_match | ghost_rate | missed_rate |
|---|---|---|---|
| Gemma-4-E2B-it Q4_K_M | **0.640** | — | — |
| arcee-lite Q4_K_M | 0.458 | 0.340 | 0.000 |
| Qwen3-1.7B Q4_K_M | 0.453 | 0.438 | 0.000 |
| Qwen3-0.6B Q4_K_M | 0.438 | 0.000 | 0.562 |
| LFM2.5-1.2B Q4_K_M | 0.434 | 0.005 | 0.557 |
| Qwen3.5-0.8B Q4_K_M | 0.310 | 0.291 | 0.000 |
| Gemma-3-270M-it Q4_K_M | 0.005 | 0.438 | 0.000 |
| Bonsai-1.7B Q1_0 | ❌ blocked | — | — |
| Bonsai-4B Q1_0 | ❌ blocked | — | — |

Results live in `RESULTS/llamacpp/<model_slug>/`. Most recent timestamped file = canonical result.

### Bonsai blocked — do not chase this next session without a clear reason

Bonsai Q1_0 GGUFs use `GGML_TYPE_Q1_0 = 41`. The installed `llama-cpp-python 0.3.20` (cu124 pre-built wheel) has `GGML_TYPE_COUNT = 41` — type 41 is out of range, model fails to load on both GPU and CPU. Fix requires rebuilding against a newer llama.cpp. Decision this session: **not worth it** — Q1_0 is a research quant that `llama.rn` does not support, so Bonsai scores wouldn't inform the deployment decision anyway.

## What to do next

### Immediate: analyse the results
Gemma-4-E2B-it leads at 0.640 but there are clear error signatures worth understanding:
- **Qwen3-0.6B / LFM2.5**: high `missed_transaction_rate` (~0.56) — overly conservative, calling real transactions null
- **Qwen3-1.7B / arcee-lite / Qwen3.5-0.8B**: zero missed rate but significant ghost rate — hallucinating transactions from non-transaction SMS
- **Gemma-3-270M-it**: near-zero full_match at 270M — too small for this task

Look at `samples_sms_extraction_<ts>.jsonl` in each model's results directory for per-sample failures. Also check `few_shot_leakage_rate` and compare `extract_json` vs `extract_json_nonnull` columns to see the impact of the nonnull filter per model.

### Per-model tuning
- **Qwen3 family**: try `/no_think` in system prompt or adjusting few-shot examples to reduce missed transactions
- **arcee-lite**: ghost rate 0.340 is high — investigate whether nonnull filter closes it or it needs prompt work
- **Gemma-4-E2B-it**: already leading — check its ghost rate and whether nonnull filter helps further

### Quantisation sweep on winner
Once a clear winner emerges, run Q5_K_M, Q3_K_M, and Q8_0 variants to find the best quality/size tradeoff for the target device.
