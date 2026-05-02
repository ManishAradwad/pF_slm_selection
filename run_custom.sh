#!/usr/bin/env bash
source pf_docker/bin/activate
export GRAMMAR="DATA/sms_extraction.gbnf"

echo "Running Qwen3-1.7B Q8..."
python run_gguf_eval.py --model Qwen/Qwen3-1.7B --gguf MODELS/Qwen3-1.7B-Q8_0.gguf --grammar "$GRAMMAR" --thinking auto --thinking-max-tokens 1024

echo "Running Gemma-4-E2B-it Q4..."
python run_gguf_eval.py --model google/gemma-4-E2B-it --gguf MODELS/gemma-4-E2B-it-Q4_K_M.gguf --grammar "$GRAMMAR" --thinking off

echo "Building report..."
python scripts/build_report.py
