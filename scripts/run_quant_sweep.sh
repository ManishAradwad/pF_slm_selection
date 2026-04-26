#!/usr/bin/env bash
# Quantisation sweep for Gemma-4-E2B-it.
# Runs the full 203-sample grammar eval for each quant variant.
# Q4_K_M is the baseline (already evaluated); all others are new.
#
# Usage: bash scripts/run_quant_sweep.sh [--limit N]

set -uo pipefail
cd "$(dirname "$0")/.."

source pf_docker/bin/activate

MODEL="google/gemma-4-E2B-it"
GRAMMAR="DATA/sms_extraction.gbnf"
LIMIT_ARGS="${@}"

FAILED=()

run() {
  local quant="$1"
  local gguf="MODELS/gemma-4-E2B-it-${quant}.gguf"
  echo ""
  echo "================================================================"
  echo "[quant_sweep] START: ${quant}"
  echo "================================================================"
  if [ ! -f "$gguf" ]; then
    echo "[quant_sweep] SKIP — GGUF not found: $gguf"
    return
  fi
  if python run_gguf_eval.py \
       --model  "$MODEL" \
       --gguf   "$gguf" \
       --grammar "$GRAMMAR" \
       $LIMIT_ARGS; then
    echo "[quant_sweep] DONE: ${quant}"
  else
    echo "[quant_sweep] FAILED: ${quant} — continuing"
    FAILED+=("$quant")
  fi
}

# Ordered high→low quality
run Q8_0
run Q6_K
run Q5_K_M
run Q4_K_M   # baseline — re-runs for a same-session comparison point
run Q3_K_M
run Q3_K_S
run IQ4_XS

echo ""
echo "================================================================"
echo "[quant_sweep] ALL DONE"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "[quant_sweep] FAILED: ${FAILED[*]}"
else
  echo "[quant_sweep] All variants completed."
fi
echo "================================================================"
