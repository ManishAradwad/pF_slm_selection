#!/usr/bin/env python3
"""Run a real synthetic completion-only BF16 backward pass on LFM2.5-350M."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lfm25.provenance import code_fingerprints, fingerprint_named_files  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TRAINING_ARTIFACTS/base/LFM2.5-350M")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "RESULTS" / "lfm25" / "verification" / "backward.json")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this verification")
    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
    ).to("cuda")
    model.train()
    model.config.use_cache = False

    prompt = [
        {"role": "system", "content": "Return exactly null or a four-field transaction JSON object."},
        {"role": "user", "content": "Sender: VM-DEMOXX\nSMS: Your one-time code is 123456. Do not share it."},
    ]
    complete = prompt + [{"role": "assistant", "content": "null"}]
    prompt_ids = tokenizer.apply_chat_template(
        prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    full_ids = tokenizer.apply_chat_template(
        complete, tokenize=True, add_generation_prompt=False, return_tensors="pt"
    )
    if hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids.input_ids
    if hasattr(full_ids, "input_ids"):
        full_ids = full_ids.input_ids
    labels = full_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    completion_tokens = int((labels != -100).sum().item())
    if completion_tokens <= 0:
        raise RuntimeError("completion-only masking left no supervised tokens")

    output = model(input_ids=full_ids.to("cuda"), labels=labels.to("cuda"))
    if not torch.isfinite(output.loss):
        raise RuntimeError("loss is not finite")
    output.loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    finite = sum(int(torch.isfinite(gradient).all().item()) for gradient in gradients)
    nonzero = sum(int(torch.count_nonzero(gradient).item() > 0) for gradient in gradients)
    if not gradients or finite != len(gradients) or nonzero == 0:
        raise RuntimeError("gradient verification failed")

    report = {
        "model_class": model.__class__.__name__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "sequence_tokens": int(full_ids.numel()),
        "completion_tokens": completion_tokens,
        "loss": round(float(output.loss.detach().cpu()), 6),
        "finite_gradient_tensors": finite,
        "nonzero_gradient_tensors": nonzero,
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "seed": args.seed,
        "model": fingerprint_named_files(
            Path(args.model),
            ("model.safetensors", "config.json", "tokenizer.json", "chat_template.jinja"),
        ),
        "code_sha256": code_fingerprints(REPO_ROOT),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, args.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
