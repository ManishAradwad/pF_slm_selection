#!/usr/bin/env python3
"""Evaluate a local BF16 base, adapter, or merged model with the short prompt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lfm25.metrics import score_records  # noqa: E402
from lfm25.provenance import (  # noqa: E402
    code_fingerprints,
    fingerprint_file,
    fingerprint_named_files,
)
from lfm25.prompts import extraction_messages  # noqa: E402


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _gold(row: dict[str, Any]) -> Any:
    if "expected" in row:
        return row["expected"]
    if "label" in row:
        return row["label"]
    if isinstance(row.get("messages"), list) and row["messages"]:
        return row["messages"][-1].get("content")
    raise ValueError("evaluation row has no target")


def _percentile(values: list[float], proportion: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(proportion * (len(ordered) - 1))))
    return round(ordered[index], 3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
    model.to("cuda").eval()

    rows = _read_rows(args.dataset)
    if args.limit is not None:
        rows = rows[: args.limit]
    output_records: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    output_lengths: list[int] = []
    prompt_lengths: list[int] = []

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        chats = [extraction_messages(str(row.get("sender", "")), str(row.get("sms", ""))) for row in batch]
        encoded = tokenizer.apply_chat_template(
            chats,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        )
        encoded = {key: value.to("cuda") for key, value in encoded.items()}
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        prompt_width = encoded["input_ids"].shape[1]
        completions = generated[:, prompt_width:]
        decoded = tokenizer.batch_decode(completions, skip_special_tokens=True)
        for offset, (row, prediction) in enumerate(zip(batch, decoded)):
            completion_ids = completions[offset]
            completion_length = int((completion_ids != tokenizer.pad_token_id).sum().item())
            prompt_length = int(encoded["attention_mask"][offset].sum().item())
            per_item_latency = elapsed_ms / len(batch)
            latencies_ms.append(per_item_latency)
            output_lengths.append(completion_length)
            prompt_lengths.append(prompt_length)
            output_records.append(
                {
                    "id": row.get("id", row.get("public_id", start + offset)),
                    "gold": _gold(row),
                    "prediction": prediction.strip(),
                    "prompt_tokens": prompt_length,
                    "output_tokens": completion_length,
                    "latency_ms_batch_amortized": round(per_item_latency, 3),
                    **{
                        key: row[key]
                        for key in ("class", "template_family")
                        if key in row
                    },
                }
            )

    scored = score_records(output_records, slice_keys=("template_family", "class"))
    scored["runtime"] = {
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "latency_ms_p50_batch_amortized": _percentile(latencies_ms, 0.5),
        "latency_ms_p95_batch_amortized": _percentile(latencies_ms, 0.95),
        "prompt_tokens_mean": round(statistics.fmean(prompt_lengths), 2) if prompt_lengths else None,
        "output_tokens_mean": round(statistics.fmean(output_lengths), 2) if output_lengths else None,
        "output_tokens_max": max(output_lengths) if output_lengths else None,
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    }
    model_path = Path(args.model)
    model_evidence: dict[str, Any]
    if model_path.is_dir():
        model_evidence = fingerprint_named_files(
            model_path,
            (
                "model.safetensors",
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
            ),
        )
    else:
        model_evidence = {"identifier": args.model}
    scored["provenance"] = {
        "model": model_evidence,
        "adapter": fingerprint_named_files(
            args.adapter, ("adapter_model.safetensors", "adapter_config.json")
        ) if args.adapter else None,
        "dataset": fingerprint_file(args.dataset),
        "code_sha256": code_fingerprints(REPO_ROOT),
        "decode": {
            "engine": "transformers",
            "grammar_constrained": False,
            "do_sample": False,
            "repetition_penalty": 1.05,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "row_limit": args.limit,
        "row_count": len(rows),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for record in output_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(scored, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(scored, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
