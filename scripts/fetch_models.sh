#!/usr/bin/env bash
# Download GGUFs for the SLMs the eval is meant to compare.
# Q4_K_M only — matches what `llama.rn` ships on-device (see CLAUDE.md).
#
# Idempotent: skips files already present.
# Requires HF_TOKEN in env for gated repos (Gemma is gated).
# Uses the `hf` CLI from huggingface_hub >= 1.0 (the older `huggingface-cli`
# entry point is deprecated; see hf.co/docs/huggingface_hub/guides/cli).

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p MODELS

dl() {
  local repo="$1"
  local file="$2"
  local out="MODELS/$3"
  if [ -s "$out" ]; then
    echo "[fetch_models] $3 already present — skipping"
    return
  fi
  echo "[fetch_models] $repo / $file"
  hf download "$repo" "$file" --local-dir MODELS
  # `hf download` writes the file at MODELS/<repo-internal-path>; flatten
  # if the in-repo name differs from the desired output name.
  if [ "MODELS/$file" != "$out" ] && [ -s "MODELS/$file" ]; then
    mv "MODELS/$file" "$out"
  fi
}

# Gemma-4-E2B-it — only repo currently confirmed. Verify others before adding.
dl bartowski/google_gemma-4-E2B-it-GGUF \
   google_gemma-4-E2B-it-Q4_K_M.gguf \
   google_gemma-4-E2B-it-Q4_K_M.gguf

# TODO: add Qwen3.5-0.8B, Qwen3-0.6B, LFM2.5-1.2B-Instruct once Q4_K_M GGUF
# repos are confirmed (see CLAUDE.md "SLMs still to benchmark").

echo "[fetch_models] done. MODELS/:"
ls -lh MODELS/*.gguf 2>/dev/null || echo "(empty)"
