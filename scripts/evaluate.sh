#!/usr/bin/env bash
# Single entry point for the candidate-slate evaluation.
#
# Sweeps every model in MODELS in two quantizations (Q4_K_M and Q8_0). For
# thinking-aware models (Qwen3 family), additionally sweeps the thinking-token
# budget (1024 vs 4096) so we can see how each model behaves with truncated
# vs full reasoning. Bonsai is excluded per user.
#
# Usage:
#   bash scripts/evaluate.sh                  # smoke (10), prompt y/n, full (203), report
#   bash scripts/evaluate.sh --auto-full      # smoke → full → report, no prompt
#   bash scripts/evaluate.sh --smoke-only     # smoke only, no full, no report
#   bash scripts/evaluate.sh --full-only      # full only, no smoke; report at end
#   bash scripts/evaluate.sh --report-only    # only regenerate RESULTS/report.html
#
# Per-model thinking is auto-detected from each tokenizer's chat template
# (Qwen3 family → on; everything else → off). Override with THINKING env var.
#
# Each (model, quant, budget) gets a fresh Python subprocess — see CLAUDE.md
# § Pipeline for the run-isolation contract and end-to-end metric flow.

set -uo pipefail
cd "$(dirname "$0")/.."

source pf_docker/bin/activate

GRAMMAR="${GRAMMAR:-DATA/sms_extraction.gbnf}"
THINKING="${THINKING:-auto}"

MODE="full+smoke"
case "${1:-}" in
  --smoke-only) MODE="smoke" ;;
  --full-only)  MODE="full" ;;
  --auto-full)  MODE="auto" ;;
  --report-only) MODE="report" ;;
  --help|-h) sed -n '1,21p' "$0"; exit 0 ;;
  "") ;;
  *) echo "Unknown flag: $1"; exit 2 ;;
esac

build_report() {
  echo ""
  echo "================================================================"
  echo "[evaluate] Generating HTML report"
  echo "================================================================"
  python scripts/build_report.py
}

# ── Slate: each line is `<HF repo id>  <gguf-basename-prefix>` ─────────────────
# Smoke runs use only Q4_K_M per model (one-per-model sanity check —
# all quants of the same model share chat-template, BOS/EOS, sampler,
# grammar compatibility).
#
# Full pass sweeps two endpoints per model: Q4_K_M (the practical on-device
# quant) and Q8_0 (the high-quality reference). We're not running every quant
# in between — the slope from Q4 → Q8 tells us if going up the quant curve
# would change a winner without paying for the intermediate experiments.
SMOKE_QUANTS=(Q4_K_M)
FULL_QUANTS=(Q4_K_M Q8_0)

# For thinking models (Qwen3 family) we also sweep the thinking-token budget,
# so we see how each model behaves when its reasoning is forcibly truncated
# vs given full headroom. Tight (1024) often forces a mid-thought close;
# loose (4096) lets the model converge naturally. For Qwen3-0.6B/1.7B the
# two budgets produce nearly identical results (they converge well under
# 1024); for Qwen3.5-0.8B (avg 3794 thinking tokens) the delta is large.
THINKING_BUDGETS=(1024 4096)
# Single budget for the smoke pass — just verify the pipeline.
SMOKE_THINKING_BUDGETS=(4096)
# Models that are thinking-aware (chat template responds to enable_thinking).
THINKING_MODELS_RE='^(Qwen/Qwen3-0\.6B|Qwen/Qwen3\.5-0\.8B|Qwen/Qwen3-1\.7B)$'

# Models in roughly ascending parameter count for fast feedback first.
MODELS=(
  "google/gemma-3-270m-it          gemma-3-270m-it"
  "Qwen/Qwen3-0.6B                 Qwen3-0.6B"
  "Qwen/Qwen3.5-0.8B               Qwen3.5-0.8B"
  "LiquidAI/LFM2.5-1.2B-Instruct   LFM2.5-1.2B-Instruct"
  "Qwen/Qwen3-1.7B                 Qwen3-1.7B"
  "arcee-ai/arcee-lite             arcee-lite"
  "google/gemma-4-E2B-it           gemma-4-E2B-it"
)

run_one() {
  local model="$1" prefix="$2" quant="$3" limit_arg="$4" tag="$5" budget="${6:-}" thinking_arg="${7:-$THINKING}"
  local gguf="MODELS/${prefix}-${quant}.gguf"
  local budget_label=""
  local budget_arg=()
  if [ -n "$budget" ]; then
    budget_arg=(--thinking-max-tokens "$budget")
    budget_label=" budget=${budget}"
  fi
  echo ""
  echo "================================================================"
  echo "[${tag}] ${model}  (${quant}${budget_label})"
  echo "================================================================"
  if [ ! -f "$gguf" ]; then
    echo "[${tag}] SKIP — GGUF not found: $gguf"
    return 0
  fi
  if python run_gguf_eval.py \
       --model "$model" \
       --gguf  "$gguf" \
       --grammar "$GRAMMAR" \
       --thinking "$thinking_arg" \
       "${budget_arg[@]}" \
       $limit_arg; then
    echo "[${tag}] DONE: ${model} ${quant}${budget_label}"
  else
    echo "[${tag}] FAILED: ${model} ${quant}${budget_label} — continuing"
    FAILED+=("${model} (${quant}${budget_label})")
  fi
}

run_pass() {
  local pass_kind="$1"  # "smoke" or "full"
  local limit_arg=""
  local quants_var budgets_var
  if [ "$pass_kind" = "smoke" ]; then
    limit_arg="--limit 10"
    quants_var=("${SMOKE_QUANTS[@]}")
    budgets_var=("${SMOKE_THINKING_BUDGETS[@]}")
  else
    quants_var=("${FULL_QUANTS[@]}")
    budgets_var=("${THINKING_BUDGETS[@]}")
  fi
  FAILED=()

  for entry in "${MODELS[@]}"; do
    # shellcheck disable=SC2086
    set -- $entry
    local model="$1" prefix="$2"
    local is_thinking=0
    if [[ "$model" =~ $THINKING_MODELS_RE ]]; then is_thinking=1; fi
    # Non-thinking models get --thinking off explicitly — THINKING_MODELS_RE is
    # the authoritative list and we don't want auto-detection to silently flip
    # a model into two-phase mode if its tokenizer changes upstream.
    local model_thinking_arg="off"
    if [ "$is_thinking" = "1" ]; then model_thinking_arg="$THINKING"; fi
    for q in "${quants_var[@]}"; do
      if [ "$is_thinking" = "1" ]; then
        for b in "${budgets_var[@]}"; do
          run_one "$model" "$prefix" "$q" "$limit_arg" "$pass_kind" "$b" "$model_thinking_arg"
        done
      else
        run_one "$model" "$prefix" "$q" "$limit_arg" "$pass_kind" "" "$model_thinking_arg"
      fi
    done
  done

  echo ""
  echo "================================================================"
  echo "[${pass_kind}] ALL DONE"
  if [ ${#FAILED[@]} -gt 0 ]; then
    echo "[${pass_kind}] FAILED:"
    for f in "${FAILED[@]}"; do echo "  - $f"; done
  else
    echo "[${pass_kind}] All variants completed."
  fi
  echo "================================================================"
}

case "$MODE" in
  smoke)
    run_pass smoke
    ;;
  full)
    run_pass full
    build_report
    ;;
  auto)
    run_pass smoke
    run_pass full
    build_report
    ;;
  report)
    build_report
    ;;
  full+smoke)
    run_pass smoke
    echo ""
    read -p "[evaluate] Smoke runs done. Run full 203-sample pass? [y/N] " ans
    if [[ "$ans" =~ ^[Yy]$ ]]; then
      run_pass full
      build_report
    else
      echo "[evaluate] Skipping full pass."
    fi
    ;;
esac
