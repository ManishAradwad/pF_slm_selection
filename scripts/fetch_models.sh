#!/usr/bin/env bash
# Download GGUFs for the candidate SLM slate (see CLAUDE.md § Candidate models).
# Quantization choice is open — Q4_K_M is just a reasonable starting point per
# model. Final pick depends on what scores best inside the device size budget.
# Bonsai is at Q1_0 because that's the only form prism-ml publishes.
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

dl unsloth/gemma-3-270m-it-GGUF \
   gemma-3-270m-it-Q4_K_M.gguf \
   gemma-3-270m-it-Q4_K_M.gguf

dl unsloth/gemma-4-E2B-it-GGUF \
   gemma-4-E2B-it-Q4_K_M.gguf \
   gemma-4-E2B-it-Q4_K_M.gguf

dl unsloth/Qwen3-0.6B-GGUF \
   Qwen3-0.6B-Q4_K_M.gguf \
   Qwen3-0.6B-Q4_K_M.gguf

dl unsloth/Qwen3.5-0.8B-GGUF \
   Qwen3.5-0.8B-Q4_K_M.gguf \
   Qwen3.5-0.8B-Q4_K_M.gguf

dl unsloth/LFM2.5-1.2B-Instruct-GGUF \
   LFM2.5-1.2B-Instruct-Q4_K_M.gguf \
   LFM2.5-1.2B-Instruct-Q4_K_M.gguf

dl unsloth/Qwen3-1.7B-GGUF \
   Qwen3-1.7B-Q4_K_M.gguf \
   Qwen3-1.7B-Q4_K_M.gguf

dl prism-ml/Bonsai-1.7B-gguf \
   Bonsai-1.7B-Q1_0.gguf \
   Bonsai-1.7B-Q1_0.gguf

dl prism-ml/Bonsai-4B-gguf \
   Bonsai-4B-Q1_0.gguf \
   Bonsai-4B-Q1_0.gguf

dl arcee-ai/arcee-lite-GGUF \
   arcee-lite-Q4_K_M.gguf \
   arcee-lite-Q4_K_M.gguf

echo "[fetch_models] done. MODELS/:"
ls -lh MODELS/*.gguf 2>/dev/null || echo "(empty)"
