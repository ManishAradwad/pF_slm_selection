#!/usr/bin/env python3
"""Merge a local LoRA adapter into the pinned BF16 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _tree_hash(
    root: Path,
    *,
    excluded_relative_paths: frozenset[str] = frozenset(),
) -> dict[str, str]:
    hashes = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative_path = path.relative_to(root).as_posix()
        if relative_path in excluded_relative_paths:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[relative_path] = digest
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="TRAINING_ARTIFACTS/base/LFM2.5-350M")
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    merged = PeftModel.from_pretrained(base, args.adapter, local_files_only=True).merge_and_unload()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size="2GB")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    tokenizer.save_pretrained(args.output_dir)
    manifest = {
        "base_model": str(Path(args.base_model).resolve()),
        "adapter": str(args.adapter.resolve()),
        "adapter_hashes": _tree_hash(args.adapter),
        # Excluding the manifest makes an identical rerun stable: on the first run
        # it does not exist yet, while on later runs it must not become a model input.
        "merged_hashes": _tree_hash(
            args.output_dir,
            excluded_relative_paths=frozenset({"merge_manifest.json"}),
        ),
        "dtype": "bfloat16",
        "push_to_hub": False,
    }
    (args.output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "files": len(manifest["merged_hashes"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
