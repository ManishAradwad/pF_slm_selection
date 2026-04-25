#!/usr/bin/env bash
# Run full eval (203 samples, grammar-constrained) for all candidate models.
# Gemma is excluded — it has existing full runs in RESULTS/llamacpp/.
# Bonsai models use the upstream Qwen3 tokenizer (same vocab/template, no
# tokenizer files in the prism-ml GGUF repos).
#
# Usage: bash scripts/run_all_evals.sh [--limit N]
#   --limit N   cap samples per model (smoke test, e.g. --limit 5)

set -uo pipefail
cd "$(dirname "$0")/.."

source pf_docker/bin/activate

GRAMMAR="DATA/sms_extraction.gbnf"
LIMIT_ARGS="${@}"   # pass through any --limit flag as-is

FAILED=()

run() {
  local model="$1"
  local gguf="$2"
  echo ""
  echo "================================================================"
  echo "[run_all_evals] START: $model  (gguf=$(basename $gguf))"
  echo "================================================================"
  if [ ! -f "$gguf" ]; then
    echo "[run_all_evals] SKIP — GGUF not found: $gguf"
    return
  fi
  if python run_gguf_eval.py \
       --model "$model" \
       --gguf  "$gguf" \
       --grammar "$GRAMMAR" \
       $LIMIT_ARGS; then
    echo "[run_all_evals] DONE: $model"
  else
    echo "[run_all_evals] FAILED: $model — continuing with next model"
    FAILED+=("$model ($(basename $gguf))")
  fi
}

run google/gemma-3-270m-it     MODELS/gemma-3-270m-it-Q4_K_M.gguf
run Qwen/Qwen3-0.6B           MODELS/Qwen3-0.6B-Q4_K_M.gguf
run Qwen/Qwen3.5-0.8B         MODELS/Qwen3.5-0.8B-Q4_K_M.gguf
run LiquidAI/LFM2.5-1.2B-Instruct MODELS/LFM2.5-1.2B-Instruct-Q4_K_M.gguf
run Qwen/Qwen3-1.7B           MODELS/Qwen3-1.7B-Q4_K_M.gguf
run arcee-ai/arcee-lite        MODELS/arcee-lite-Q4_K_M.gguf
run Qwen/Qwen3-1.7B           MODELS/Bonsai-1.7B-Q1_0.gguf
run Qwen/Qwen3-4B             MODELS/Bonsai-4B-Q1_0.gguf

echo ""
echo "================================================================"
echo "[run_all_evals] ALL DONE"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "[run_all_evals] FAILED models:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
else
  echo "[run_all_evals] All models completed successfully."
fi
echo "================================================================"
