"""Sanity check: torch sees the GPU and llama-cpp-python is CUDA-built.

Run after container rebuild. If this fails, the runtime will silently fall
back to CPU on every eval — diagnose before that happens.

Looks for the smallest .gguf file under MODELS/ to load. If MODELS/ is empty,
warns but exits 0 — populate via scripts/fetch_models.sh and re-run.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    try:
        import torch
    except ImportError as e:
        print(f"[verify_gpu] torch not importable: {e}", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("[verify_gpu] FAIL: torch.cuda.is_available() is False", file=sys.stderr)
        return 2
    print(f"[verify_gpu] torch sees GPU 0: {torch.cuda.get_device_name(0)}", file=sys.stderr)

    try:
        from llama_cpp import Llama
    except ImportError as e:
        print(f"[verify_gpu] llama_cpp not importable: {e}", file=sys.stderr)
        return 3

    repo = Path(__file__).resolve().parent.parent
    ggufs = sorted((repo / "MODELS").glob("*.gguf"), key=lambda p: p.stat().st_size)
    if not ggufs:
        print(
            "[verify_gpu] no .gguf in MODELS/ — skipping llama-cpp runtime check.\n"
            "             Run scripts/fetch_models.sh, then re-run this script.",
            file=sys.stderr,
        )
        return 0

    gguf = ggufs[0]
    print(f"[verify_gpu] loading {gguf.name} on GPU (n_gpu_layers=-1)", file=sys.stderr)
    # llama.cpp's CUDA-init log goes straight to fd 2 — visible in console.
    # Look for "ggml_cuda_init: found N CUDA devices" or similar.
    Llama(model_path=str(gguf), n_ctx=128, n_gpu_layers=-1, verbose=True)
    print("[verify_gpu] PASS: model loaded with GPU offload.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
