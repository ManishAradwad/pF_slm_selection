# CLAUDE.md - SLM Evaluation for Financial SMS

## Project Overview
Evaluation playground for selecting the optimal Small Language Model (SLM) to power a mobile financial helper app called **pocket-financer**. The app tracks the user's bank accounts and credit/debit cards, then uses an SLM to process transaction SMS alerts and maintain balance/spend data. Wallets (Simpl, slice, PayTM wallet) are out of scope — only direct bank account and card transactions. Production runtime is **`llama.rn`** (React Native binding over `llama.cpp`) on Android.

## Evaluation approach
The task is **structured JSON extraction**, not classification. Given an Indian bank/card SMS, the model outputs either the literal word `null` (not a real transaction) or a JSON object with 5 fields: `amount`, `merchant`, `date`, `type` (`debit`|`credit`), `account`. Pipeline is built out-of-tree on `lm-evaluation-harness`.

**Backend**: `llama-cpp-python` — the same engine as production `llama.rn`. This is the whole reason to avoid the HF `transformers` backend: quantization, tokenization, and sampler match on-device, so dev metrics honestly predict production behavior.

**Grammar-constrained decoding**: A GBNF grammar (`DATA/sms_extraction.gbnf`) enforces output shape. `amount`, `type`, `account` are non-nullable; `merchant`, `date` are nullable. The same grammar ships on-device.

**Two-phase generation for thinking models** (Qwen3 family). When the tokenizer's chat template responds to `enable_thinking` (i.e. rendering with `True` differs from `False`), we run two passes per sample:
1. **Phase 1** — render prompt with `enable_thinking=True`, force-open the block by appending `<think>\n`, generate freely with `repeat_penalty=1.1` (breaks Qwen3.5-class loops). Two stop conditions: `max_tokens=thinking_max_tokens` (the budget — default 4096) and `stop=["</think>"]`. Whichever fires first wins. If the budget hits first, we forcibly append `</think>\n` ourselves before phase 2 — i.e. the model gets cut off mid-thought.
2. **Phase 2** — append the closed think-block to the prompt, attach the GBNF grammar, decode the JSON answer.

Without phase 1, GBNF suppresses every `<think>` token (the grammar disallows `<`) and the model collapses into copying few-shot demos. The two-phase split is auto-applied; `_template_lets_model_emit_think()` in `DATA/llamacpp_model.py` handles detection so non-thinking models (LFM2.5, Gemma, arcee-lite) skip the extra call. The `<think>…</think>` block is stripped before metrics by `extract_json_filter`.

**Two metric filter pipelines** run in parallel on every eval:
- `extract_json` — raw model output (cleaned to first JSON object or `null`).
- `extract_json_nonnull` — applies a reject rule: if the output dict has `null` in any of `amount`, `type`, or `account`, treat the whole prediction as `null`. The model's own null-emission is used as an implicit confidence signal — no per-model calibration needed, works across any SLM. See `_REQUIRED_NONNULL_FIELDS` in `DATA/utils.py`.

Every metric appears twice in results files with `,extract_json` and `,extract_json_nonnull` suffixes. Compare the two columns to see raw-model vs production-filtered behavior.

**Prompt structure**: `SYSTEM_PROMPT + "### EXAMPLES" block (6 demos) + "### YOUR TASK" delimiter + (sender, sms, "Output: ")`. The explicit delimiter is a domain-level fix: without it, thinking models read few-shot demos as in-flight chat turns and answer one of those instead of the actual query (observed on Qwen3-1.7B with 23.6% few-shot leakage pre-fix). The delimiter helps any model and is not a per-SLM hack.

**Merchant match is intentionally loose** — case-insensitive + whitespace-collapsed + substring either direction (3-char floor). Bank SMS wrap the same entity in cosmetic boilerplate (`VPA x@y`, `mobile 9XXX-APIBANKING`, `UPI-<ref>-Compass`, trailing city names) and the model shouldn't be punished for failing to strip it. `None` vs non-`None` stays strict — over-extraction is still an error. See `_merchant_match` in `DATA/utils.py`.

**Run isolation**: each (model, quant, thinking_budget) tuple runs in a **fresh Python subprocess** spawned by `scripts/evaluate.sh`. New CUDA context, new GGUF load, new KV cache, no shared in-memory state with the previous quant. Only read-only files (dataset, grammar, GGUF, HF tokenizer cache) are shared — none are mutated. So one model's behavior cannot bleed into the next.

## Candidate models

Slate to compare for `pocket-financer` deployment. The goal is to find the best model+quantization combination for the user's phone — nothing here is locked in. Q4_K_M is a reasonable starting quant for most candidates (good size/quality balance, widely available), but if a model wins at Q5_K_M, Q3_K_M, or even Q8_0 within the device budget, that's the one we pick. Bonsai is shown at Q1_0 because that's the only form prism-ml publishes.

| Model | Released | Params | Quant | GGUF source |
|---|---|---|---|---|
| Gemma-3-270M-it | 2025 | 270M | Q4_K_M | `unsloth/gemma-3-270m-it-GGUF` |
| Gemma-4-E2B-it | 2026 | ~2B | Q4_K_M | `unsloth/gemma-4-E2B-it-GGUF` |
| Qwen3-0.6B | 2025 | 0.6B | Q4_K_M | `unsloth/Qwen3-0.6B-GGUF` |
| Qwen3.5-0.8B | 2026 | 0.8B | Q4_K_M | `unsloth/Qwen3.5-0.8B-GGUF` |
| LFM2.5-1.2B-Instruct | 2025 | 1.2B | Q4_K_M | `unsloth/LFM2.5-1.2B-Instruct-GGUF` |
| Qwen3-1.7B | 2025 | 1.7B | Q4_K_M | `unsloth/Qwen3-1.7B-GGUF` |
| Bonsai-1.7B | 2026 | 1.7B | Q1_0 | `prism-ml/Bonsai-1.7B-gguf` |
| Bonsai-4B | 2026 | 4B | Q1_0 | `prism-ml/Bonsai-4B-gguf` |
| arcee-lite | 2024 | 2B | Q4_K_M | `arcee-ai/arcee-lite-GGUF` |

Expect per-model tuning. Structured-JSON extraction stresses each model's chat-template handling, BOS/EOS behavior, sampler defaults, and grammar compatibility differently — a config that's good for Gemma may need adjustment for Qwen3 (thinking-mode tokens), LFM2 (Liquid's hybrid arch), or Bonsai (1-bit quant has unusual sampler dynamics). Treat the eval as per-model: tune, then compare.

## Entry point

**Whole-slate evaluation** (default workflow):

```bash
source pf_docker/bin/activate
bash scripts/evaluate.sh                # smoke (10) + prompt + full (203) + report
bash scripts/evaluate.sh --auto-full    # smoke → full → report, no prompt
bash scripts/evaluate.sh --full-only    # full + report (skip smoke)
bash scripts/evaluate.sh --report-only  # regenerate RESULTS/report.html only
```

The script sweeps **two quants per model** (`Q4_K_M`, `Q8_0`) — the practical on-device quant and the high-quality reference. For thinking-aware models (Qwen3 family) it also sweeps the **thinking-token budget** (`1024`, `4096`) so we can see how each model behaves with truncated vs full reasoning. Bonsai is excluded.

Final HTML report lands at `RESULTS/report.html` (self-contained, open in any browser). Per-run artefacts at `RESULTS/llamacpp/<model_slug>/results_<ts>.json` + `samples_sms_extraction_<ts>.jsonl`.

**Single-model evaluation** (one-off / debugging):

```bash
python run_gguf_eval.py \
  --model google/gemma-4-E2B-it \
  --gguf MODELS/gemma-4-E2B-it-Q4_K_M.gguf \
  --grammar DATA/sms_extraction.gbnf \
  [--thinking auto|on|off] \
  [--thinking-max-tokens 4096] \
  [--limit 10]
```

`--model` is the HF repo id (used for tokenizer + chat template); `--gguf` is the local weights fetched by `scripts/fetch_models.sh`. Omit `--grammar` to compare grammar-free behavior. `--thinking auto` (default) enables two-phase generation when the chat template supports it.

## Key files

### Evaluation pipeline
- `scripts/evaluate.sh` — **single entry point**. Iterates the slate × quants × thinking budgets, spawns one fresh `python run_gguf_eval.py` per combination, then calls `build_report.py` at the end. See § Entry point for flags.
- `scripts/build_report.py` — walks `RESULTS/llamacpp/*/results_*.json` + matching samples files, groups by (model, quant, thinking_budget), emits a single self-contained `RESULTS/report.html`: aggregate metrics table with best-per-column highlighted, side-by-side response viewer for representative SMS, failure-mode badges, per-model quirks blurbs.
- `run_gguf_eval.py` — single-model entry script; registers the adapter and calls `lm_eval.simple_evaluate`. Accepts `--thinking {auto,on,off}` and `--thinking-max-tokens N`.
- `DATA/llamacpp_model.py` — out-of-tree `@register_model("llamacpp")` adapter over `llama-cpp-python`. Accepts `grammar_file`, `thinking`, `thinking_max_tokens`, `thinking_repeat_penalty` via `model_args`. Uses HF `AutoTokenizer` (via `--model <hf_id>`) for chat-template rendering, with `enable_thinking` forwarded when supported. Strips the model's `bos_token` from the rendered template (using `hf_tokenizer.bos_token` dynamically) because `llama-cpp-python` adds its own BOS and a duplicate hurts quality. Auto-detects `n_ctx` from GGUF metadata at load (capped at 32768 for our GPU). For thinking models, runs phase 1 (no grammar, stop on `</think>`, `repeat_penalty=1.1` to break loops) → phase 2 (grammar + JSON).
- `DATA/sms_extraction.yaml` — task config. `output_type: generate_until`, two parallel filter pipelines, ~11 metrics per filter.
- `DATA/sms_extraction.gbnf` — GBNF grammar for JSON output.
- `DATA/utils.py` — `SYSTEM_PROMPT`, `FEW_SHOT_EXAMPLES`, `doc_to_text` (with explicit `### EXAMPLES` / `### YOUR TASK` delimiters), `extract_json_filter` (strips `<think>...</think>`), `extract_json_nonnull_filter`, metric functions (full_match, ghost_rate, missed_rate, per-field accuracy, few-shot leakage). Has a `__main__` self-test suite — run `python DATA/utils.py` after any metric/normalizer edit.
- `DATA/extraction_ds.jsonl` — 203 labeled samples (114 real transactions, 89 non-transactions). **Do NOT edit.**

### SMS filtering pipeline (upstream — builds the raw data that feeds the eval)
- `old_pipeline.ipynb` — v1 filtering (negative filters; superseded).
- `new_pipeline.py` — v2 filtering; positive filters (masked account + transaction verb) for far fewer false negatives on real bank messages.
- `build_datasets.py`, `expand_dataset.py`, `export_sms.py`, `find_sms_db.py` — dataset construction utilities. `find_sms_db.py` extracts `sms.db` from an iPhone iTunes backup.
- `copilot-instructions.md` — full original project context and goals.

### Models and results
- `MODELS/` — GGUF weights (gitignored). The slate sweeps `Q4_K_M` and `Q8_0` per model. Other quants on disk are not part of the standard sweep but can be evaluated ad-hoc with `run_gguf_eval.py`.
- `RESULTS/llamacpp/<model_slug>/` — per-run output (`results_<ts>.json` + `samples_<task>_<ts>.jsonl`). Multiple (quant, thinking_budget) variants of the same HF model coexist here, distinguished by timestamp; `scripts/build_report.py` parses the GGUF path + `model_args.thinking_max_tokens` to identify them.
- `RESULTS/report.html` — self-contained HTML report rebuilt by `scripts/evaluate.sh` (or `build_report.py --report-only`).
- `RESULTS/new_pipeline/` — legacy HF-backend baselines; kept for reference but not the current source of truth.
- `RESULTS/llamacpp_pre_thinking_*` / `RESULTS/llamacpp_smoke_*` — archived runs from earlier pipeline iterations; never read by `build_report.py` unless `--include-archives` is passed.

### Bootstrap (Dockerfile + scripts)
- `.devcontainer/Dockerfile` — `nvidia/cuda:12.4.1-devel-ubuntu22.04` + Python 3.11 (deadsnakes) + heavy deps from `requirements.txt` + from-source CUDA build of `llama-cpp-python` (GitHub main branch — see Environment note).
- `.devcontainer/devcontainer.json` — forwards `HF_TOKEN` from host, sets `HF_HOME=/workspaces/.../.hf_cache`, runs `.devcontainer/post-create.sh`.
- `.devcontainer/post-create.sh` — runs once on container creation. Installs Claude Code, (re)creates `pf_docker/` venv if stale, runs `verify_gpu.py`.
- `requirements.txt` — pinned versions for transformers / accelerate / huggingface_hub / lm_eval. torch and llama-cpp-python are NOT here (different install paths — see comments in the file).
- `scripts/verify_gpu.py` — asserts `torch.cuda.is_available()` and that `llama-cpp-python` loads a GGUF with GPU offload. Run after each container creation by post-create.sh.
- `scripts/fetch_models.sh` — `hf download` calls for the candidate-slate GGUFs. Sweeps `Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0` per model so any quant in the range is on disk for ad-hoc runs, even though the standard `evaluate.sh` sweep only uses `Q4_K_M` and `Q8_0`. Idempotent.

## Environment
- **Docker dev container**: built from `nvidia/cuda:12.4.1-devel-ubuntu22.04`. Python 3.11 (deadsnakes), Node 20 (devcontainer feature), GPU passthrough via `runArgs: ["--gpus", "all"]`. Activate the workflow venv with `source pf_docker/bin/activate`.
- **Hardware**: WSL2, NVIDIA RTX 4070 (12 GB VRAM), 32 GB RAM. Host driver 591.74 / max CUDA 13.1.
- **`pf_docker/` is a `--system-site-packages` shim venv** created by `.devcontainer/post-create.sh`. Heavy deps (torch, transformers, llama-cpp-python, lm-eval) live in the image's system Python; the venv just inherits them and provides the activation entry point. Means rebuilds reconstruct it in seconds.
- **GPU-backed `llama-cpp-python`**: the Dockerfile installs a pre-built CUDA 12.4 wheel (`llama-cpp-python==0.3.20` from `abetlen.github.io/llama-cpp-python/whl/cu124`). If a future rebuild silently regresses to CPU, verify the `--extra-index-url` in `.devcontainer/Dockerfile` still points to the correct CUDA wheel index.
- **`HF_TOKEN` is required** for the gated Gemma tokenizer. Set on the WSL host (`echo 'export HF_TOKEN=hf_...' >> ~/.bashrc`); `devcontainer.json` forwards it via `containerEnv: {"HF_TOKEN": "${localEnv:HF_TOKEN}"}`. Tokenizer/model cache lives at `.hf_cache/` (gitignored, persisted across rebuilds via the workspace bind mount).

## Conventions
- Pin every heavy dep in `requirements.txt` + the Dockerfile, not via ad-hoc `pip install` in the running container — manual installs don't survive rebuild.
- Evaluation datasets in `DATA/`, results in `RESULTS/`, GGUF weights in `MODELS/` (gitignored).
- **`lm-evaluation-harness/` is an upstream clone.** Prefer out-of-tree extensions — custom tasks via `--include_path`, custom models via `@register_model` in a module outside the clone — so it stays rebasable. Edit inside the clone only when the extension point genuinely doesn't exist.
- Data files (csv, json, db, xlsx, gguf) are gitignored — never commit them.
- **Do not name dataset senders in `SYSTEM_PROMPT`.** The pre-2026-04-17 prompt listed `JK-620016`, `VM-OFFERZ`, `VK-GOIBIB`, `VM-NOBRKR` as example promo senders — all of which appear in the eval dataset. The current prompt describes the XX-YYYYYY Indian SMS sender format generically instead, at a small accuracy cost. Keep it generic.
- Keep the eval pipeline model-agnostic: filters and rules should express properties of the domain (bank SMS always contain amount/type/account), not properties of any specific SLM.

## Known state
- Evaluation pipeline operates over the full slate via `scripts/evaluate.sh`. Two-phase thinking generation is in for Qwen3 family. Full-layer GPU offload via `n_gpu_layers=-1`; n_ctx auto-detected from GGUF metadata, capped at 32768 for the 4070's 12 GB VRAM budget.
- Per-sample timing observed on RTX 4070: prefill ~6600 TPS (with KV-cache reuse across consecutive samples — only the SMS suffix is re-prefilled, ~80 tokens). Decode TPS varies 50–340 tok/s under GBNF grammar (grammar checking + sampling overhead is the dominant non-thinking cost). Per-call Python wrapper overhead ~1.2s/sample is unavoidable in `llama-cpp-python` short of switching to `llama-server`.
- Output token counts (avg from smoke pass): Gemma-3-270M 53, Gemma-4-E2B 24, Qwen3-0.6B 361 (thinking), Qwen3-1.7B 521 (thinking, one outlier at 2103), **Qwen3.5-0.8B 3794** (thinking, almost always hits the 4096 cap), LFM2.5-1.2B 2 (over-rejects), arcee-lite 47.
- See § Candidate models for the rest of the slate to evaluate.
- SMS data covers 2021–2026 with coverage tapering after 2024.
- Container deps (system Python): torch 2.7.0 (cu126), transformers 4.57.6, accelerate 1.13.0, huggingface_hub 1.12.0, lm-eval pinned to commit `c1c4bea3`, llama-cpp-python 0.3.20 (built from source with `-DGGML_CUDA=on`).
