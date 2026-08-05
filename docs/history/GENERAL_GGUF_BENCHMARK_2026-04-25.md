# Historical general GGUF benchmark - 2026-04-25

Status: historical pre-app-contract evidence retained for audit. These scores use
the older broad evaluation path, not the current PocketFinancer Android profile,
Kotlin parser, custom JNI runtime, or a fresh test set. Do not compare them as
deployment-equivalent results.

This document replaces the old root session handoff. Current work starts from the
[experiment catalog](../experiments/EXPERIMENT_CATALOG.md) and the unified pipeline.

## Slate and evaluator changes at the time

The sweep added Gemma-3-270M-it, Qwen3-1.7B, and two Bonsai Q1_0 models to the
existing candidates. `scripts/fetch_models.sh` remains the executable historical
source list.

The legacy `DATA/llamacpp_model.py` adapter was changed to:

1. Strip `hf_tokenizer.bos_token` dynamically instead of assuming `<bos>`.
2. Read the GGUF trained context and use `min(native_context, 32768)` by default
   to keep KV-cache allocation within the 12 GB host GPU budget.
3. Treat `n_ctx=0` in `run_gguf_eval.py` as automatic context selection.

Downloaded weights stayed under ignored `MODELS/`. The broad evaluator used the
legacy grammar-constrained task and did not reproduce the later Android filter,
prompt/decode profile, or app parser.

## Saved 203-row sweep

| Model | Full match | Ghost rate | Missed rate |
|---|---:|---:|---:|
| Gemma-4-E2B-it Q4_K_M | 0.640 | Not recorded | Not recorded |
| arcee-lite Q4_K_M | 0.458 | 0.340 | 0.000 |
| Qwen3-1.7B Q4_K_M | 0.453 | 0.438 | 0.000 |
| Qwen3-0.6B Q4_K_M | 0.438 | 0.000 | 0.562 |
| LFM2.5-1.2B Q4_K_M | 0.434 | 0.005 | 0.557 |
| Qwen3.5-0.8B Q4_K_M | 0.310 | 0.291 | 0.000 |
| Gemma-3-270M-it Q4_K_M | 0.005 | 0.438 | 0.000 |
| Bonsai-1.7B Q1_0 | Blocked | Not applicable | Not applicable |
| Bonsai-4B Q1_0 | Blocked | Not applicable | Not applicable |

Per-row outputs remain local under ignored `RESULTS/llamacpp/`. The 203-row
fixture was already repeatedly consulted, so even these original scores are
regression evidence rather than an unbiased test.

## Historical failure signatures

- Qwen3-0.6B and LFM2.5-1.2B were overly conservative and missed many labeled
  transactions.
- Qwen3-1.7B, arcee-lite, and Qwen3.5-0.8B emitted transactions for many labeled
  null rows.
- Gemma-3-270M-it was effectively unusable under this prompt/evaluator.
- Gemma-4-E2B-it led this non-equivalent broad benchmark.

These observations motivated later work on app-aligned prompting, fail-closed
interpretation, clean training data, and direct versus candidate architectures.
They are not current model-selection conclusions.

## Bonsai blocker

The available Bonsai files used `GGML_TYPE_Q1_0 = 41`, while the installed
`llama-cpp-python 0.3.20` build had `GGML_TYPE_COUNT = 41`; type 41 was therefore
out of range and failed to load on CPU and GPU. Resolving it required a newer
llama.cpp build, while the target Android runtime did not support that research
quant. The experiment intentionally stopped there.

## Reproduction boundary

The historical command path is documented in [the script map](../../scripts/README.md):

```bash
bash scripts/evaluate.sh
```

Use it only for broad exploration. New app-facing experiments must use
`scripts/run_pocketfinancer_pipeline.py` and the current versioned Android profile.
