#!/usr/bin/env bash
# Download GGUFs for the candidate SLM slate (see CLAUDE.md § Candidate models).
# Per user direction we now sweep multiple quantizations per model so each
# candidate's quality/size tradeoff curve can be measured directly. Bonsai is
# excluded — Bonsai is published only at Q1_0 and the run currently fails for
# it.
#
# Quant sweep per model: Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0
#   (covers the meaningful range; finer-grained variants like Q3_K_S, IQ4_XS
#    are still pulled for Gemma-4-E2B since they're already on disk and
#    historically informative for a 2B-class model.)
#
# Idempotent: skips files already present.
# Requires HF_TOKEN in env for gated repos (Gemma is gated).

set -uo pipefail

cd "$(dirname "$0")/.."
mkdir -p MODELS

dl() {
  local repo="$1"
  local file="$2"
  local out="MODELS/$3"
  if [ -s "$out" ]; then
    echo "[fetch_models] $3 already present — skipping"
    return 0
  fi
  echo "[fetch_models] $repo / $file"
  if hf download "$repo" "$file" --local-dir MODELS 2>&1 | tail -3; then
    if [ "MODELS/$file" != "$out" ] && [ -s "MODELS/$file" ]; then
      mv "MODELS/$file" "$out"
    fi
  else
    echo "[fetch_models] WARN: failed to download $repo/$file — skipping"
    return 0
  fi
}

# Standard 5-point quant sweep applied to each candidate model.
# Each entry: <repo> <basename-prefix> (file + out share the same name).
sweep() {
  local repo="$1"
  local prefix="$2"
  for q in Q3_K_M Q4_K_M Q5_K_M Q6_K Q8_0; do
    dl "$repo" "${prefix}-${q}.gguf" "${prefix}-${q}.gguf"
  done
}

# ── Candidate models ────────────────────────────────────────────────────────────

sweep unsloth/gemma-3-270m-it-GGUF        gemma-3-270m-it
sweep unsloth/gemma-4-E2B-it-GGUF         gemma-4-E2B-it
sweep unsloth/Qwen3-0.6B-GGUF             Qwen3-0.6B
sweep unsloth/Qwen3.5-0.8B-GGUF           Qwen3.5-0.8B
sweep unsloth/LFM2.5-1.2B-Instruct-GGUF   LFM2.5-1.2B-Instruct
sweep unsloth/Qwen3-1.7B-GGUF             Qwen3-1.7B
sweep arcee-ai/arcee-lite-GGUF            arcee-lite

# ── Extra Gemma-4-E2B variants already on disk historically ────────────────────
# Kept so re-running fetch is a no-op for the existing collection.
dl unsloth/gemma-4-E2B-it-GGUF gemma-4-E2B-it-Q3_K_S.gguf gemma-4-E2B-it-Q3_K_S.gguf
dl unsloth/gemma-4-E2B-it-GGUF gemma-4-E2B-it-IQ4_XS.gguf gemma-4-E2B-it-IQ4_XS.gguf

echo ""
echo "[fetch_models] done. MODELS/:"
ls -lh MODELS/*.gguf 2>/dev/null || echo "(empty)"
