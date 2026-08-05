#!/usr/bin/env python3
"""Evaluate a model with the PocketFinancer app input and output behavior.

For a non-thinking model such as LFM2.5-350M this uses the same direct-answer
shape as the app. Transformers still cannot reproduce the custom Android JNI,
GGUF quantization, optional GBNF sampler, or phone performance, so final
deployment validation belongs to the GGUF/device evaluator.
"""

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

from lfm25.android_contract import (  # noqa: E402
    ANDROID_DECODE_DEFAULTS,
    android_extraction_messages,
    context_compatibility,
    contract_provenance,
    decode_defaults,
    pocketfinancer_normalize_prediction,
    pocketfinancer_parse_prediction,
    selection_prefilter_sms,
    should_apply_prefilter,
    summarize_prefilter,
)
from lfm25.metrics import score_records  # noqa: E402
from lfm25.prompts import extraction_messages  # noqa: E402
from lfm25.provenance import (  # noqa: E402
    code_fingerprints,
    fingerprint_file,
    fingerprint_named_files,
)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


def _contract_metadata(contract: str) -> dict[str, Any]:
    if contract in {"pocketfinancer", "android-prompt-proxy"}:
        return {
            "name": "pocketfinancer_hf_training_evaluation",
            "runtime_parity": False,
            "limitation": (
                "Transformers does not reproduce Android JNI, GGUF quantization, "
                "or optional llama.cpp GBNF sampling."
            ),
            "android_current_prompt_contract": contract_provenance(),
            "decode_defaults": decode_defaults(contract),
        }
    return {
        "name": "legacy_short_prompt",
        "runtime_parity": False,
        "message_builder": "lfm25.prompts.extraction_messages",
        "decode_defaults": decode_defaults("legacy"),
    }


def _context_summary(prompt_lengths: list[int]) -> dict[str, Any]:
    n_ctx = int(ANDROID_DECODE_DEFAULTS["n_ctx"])
    answer_tokens = int(ANDROID_DECODE_DEFAULTS["answer_max_tokens"])
    compatible = sum(
        context_compatibility(
            length,
            n_ctx=n_ctx,
            generation_tokens=answer_tokens,
        )["compatible"]
        for length in prompt_lengths
    )
    return {
        "android_current_n_ctx": n_ctx,
        "android_current_answer_budget": answer_tokens,
        "measured_rows": len(prompt_lengths),
        "prompt_tokens_min": min(prompt_lengths) if prompt_lengths else None,
        "prompt_tokens_max": max(prompt_lengths) if prompt_lengths else None,
        "compatible_rows": compatible,
        "incompatible_rows": len(prompt_lengths) - compatible,
        "runtime_parity": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "HF evaluation using PocketFinancer's exact prompt, prefilter, and "
            "direct-answer profile. Final runtime parity requires GGUF/device testing."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--contract",
        choices=("pocketfinancer", "android-prompt-proxy", "legacy"),
        default="pocketfinancer",
    )
    parser.add_argument(
        "--apply-prefilter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "PocketFinancer SMS prefilter; enabled by default for the app profile"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--repeat-penalty", type=float)
    parser.add_argument("--n-ctx", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    defaults = decode_defaults(args.contract)
    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else defaults.get("answer_max_tokens", defaults.get("max_tokens", 256))
    )
    repeat_penalty = (
        args.repeat_penalty
        if args.repeat_penalty is not None
        else defaults.get("repeat_penalty", 1.0)
    )
    n_ctx = args.n_ctx if args.n_ctx is not None else defaults["n_ctx"]
    apply_prefilter = should_apply_prefilter(args.contract, args.apply_prefilter)
    pocketfinancer_profile = args.contract in {
        "pocketfinancer",
        "android-prompt-proxy",
    }
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if repeat_penalty <= 0:
        parser.error("--repeat-penalty must be positive")
    if n_ctx <= max_new_tokens:
        parser.error("--n-ctx must be greater than --max-new-tokens")

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

    records_by_index: dict[int, dict[str, Any]] = {}
    pending: list[tuple[int, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        disposition = (
            selection_prefilter_sms(str(row.get("sender", "")), str(row.get("sms", "")))
            if apply_prefilter
            else None
        )
        passed = disposition.accepted if disposition is not None else True
        common = {
            "id": row.get("id", row.get("public_id", index)),
            "gold": _gold(row),
            "selection_prefilter_applied": apply_prefilter,
            "prefilter_applied": apply_prefilter,
            "prefilter_passed": passed,
            "prefilter_rejection_stage": (
                disposition.rejection_stage if disposition is not None else None
            ),
            "prefilter_rejection_reason": (
                disposition.rejection_reason if disposition is not None else None
            ),
            "model_invoked": passed,
            **{key: row[key] for key in ("class", "template_family") if key in row},
        }
        if passed:
            pending.append((index, row))
            records_by_index[index] = common
        else:
            records_by_index[index] = {
                **common,
                "prediction": "null",
                "app_prediction": "null",
                "prompt_tokens": 0,
                "output_tokens": 0,
                "latency_ms_batch_amortized": 0.0,
            }

    latencies_ms: list[float] = []
    output_lengths: list[int] = []
    prompt_lengths: list[int] = []
    for start in range(0, len(pending), args.batch_size):
        batch_items = pending[start : start + args.batch_size]
        batch = [row for _, row in batch_items]
        if args.contract in {"pocketfinancer", "android-prompt-proxy"}:
            chats = [
                android_extraction_messages(
                    str(row.get("sender", "")), str(row.get("sms", ""))
                )
                for row in batch
            ]
        else:
            chats = [
                extraction_messages(str(row.get("sender", "")), str(row.get("sms", "")))
                for row in batch
            ]
        encoded = tokenizer.apply_chat_template(
            chats,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        )
        encoded = {key: value.to("cuda") for key, value in encoded.items()}
        longest_prompt = int(encoded["attention_mask"].sum(dim=1).max().item())
        if longest_prompt + max_new_tokens > n_ctx:
            raise ValueError(
                f"proxy prompt ({longest_prompt}) + completion ({max_new_tokens}) "
                f"exceeds proxy n_ctx={n_ctx}; this guard does not change context size"
            )

        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=repeat_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        prompt_width = encoded["input_ids"].shape[1]
        completions = generated[:, prompt_width:]
        decoded = tokenizer.batch_decode(completions, skip_special_tokens=True)
        for offset, ((index, _row), prediction) in enumerate(zip(batch_items, decoded)):
            completion_ids = completions[offset]
            completion_length = int((completion_ids != tokenizer.pad_token_id).sum().item())
            prompt_length = int(encoded["attention_mask"][offset].sum().item())
            per_item_latency = elapsed_ms / len(batch)
            latencies_ms.append(per_item_latency)
            output_lengths.append(completion_length)
            prompt_lengths.append(prompt_length)
            records_by_index[index].update(
                {
                    "prediction": prediction.strip(),
                    "app_prediction": pocketfinancer_normalize_prediction(
                        prediction.strip()
                    )
                    if pocketfinancer_profile
                    else prediction.strip(),
                    "prompt_tokens": prompt_length,
                    "output_tokens": completion_length,
                    "latency_ms_batch_amortized": round(per_item_latency, 3),
                }
            )

    output_records = [records_by_index[index] for index in range(len(rows))]
    model_records = [record for record in output_records if record["model_invoked"]]
    strict_whole_pipeline = score_records(
        output_records, slice_keys=("template_family", "class")
    )
    strict_conditional_model = score_records(
        model_records, slice_keys=("template_family", "class")
    )
    if pocketfinancer_profile:
        whole_pipeline = score_records(
            output_records,
            prediction_parser=pocketfinancer_parse_prediction,
            slice_keys=("template_family", "class"),
        )
        conditional_model = score_records(
            model_records,
            prediction_parser=pocketfinancer_parse_prediction,
            slice_keys=("template_family", "class"),
        )
    else:
        whole_pipeline = strict_whole_pipeline
        conditional_model = strict_conditional_model
    scored = dict(whole_pipeline)
    scored["whole_pipeline"] = whole_pipeline
    scored["conditional_model"] = conditional_model
    scored["app_interpreted"] = {
        "whole_pipeline": whole_pipeline,
        "conditional_model": conditional_model,
    }
    scored["strict_model_output"] = {
        "whole_pipeline": strict_whole_pipeline,
        "conditional_model": strict_conditional_model,
    }
    scored["selection_prefilter"] = summarize_prefilter(
        output_records, enabled=apply_prefilter
    )
    scored["prefilter"] = scored["selection_prefilter"]
    scored["android_current_context_compatibility"] = _context_summary(prompt_lengths)
    scored["runtime"] = {
        "profile": args.contract,
        "android_runtime_parity": False,
        "prefilter_applied": apply_prefilter,
        "rows": len(output_records),
        "model_invocations": len(model_records),
        "batch_size": args.batch_size,
        "n_ctx": n_ctx,
        "max_new_tokens": max_new_tokens,
        "latency_ms_p50_batch_amortized": _percentile(latencies_ms, 0.5),
        "latency_ms_p95_batch_amortized": _percentile(latencies_ms, 0.95),
        "prompt_tokens_mean": round(statistics.fmean(prompt_lengths), 2)
        if prompt_lengths
        else None,
        "output_tokens_mean": round(statistics.fmean(output_lengths), 2)
        if output_lengths
        else None,
        "output_tokens_max": max(output_lengths) if output_lengths else None,
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    }

    model_path = Path(args.model)
    if model_path.is_dir():
        model_evidence: dict[str, Any] = fingerprint_named_files(
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
        "profile": _contract_metadata(args.contract),
        "evaluator": fingerprint_file(Path(__file__)),
        "model": model_evidence,
        "adapter": fingerprint_named_files(
            args.adapter, ("adapter_model.safetensors", "adapter_config.json")
        )
        if args.adapter
        else None,
        "dataset": fingerprint_file(args.dataset),
        "code_sha256": code_fingerprints(REPO_ROOT),
        "decode": {
            "engine": "transformers_prompt_training_proxy",
            "android_runtime_parity": False,
            "grammar_constrained": False,
            "do_sample": False,
            "repetition_penalty": repeat_penalty,
            "max_new_tokens": max_new_tokens,
            "n_ctx": n_ctx,
            "seed": args.seed,
        },
        "selection_prefilter": {
            "applied": apply_prefilter,
            "part_of_android_current": args.contract == "pocketfinancer",
            "rejected_prediction": "null",
        },
        "output_interpretation": {
            "primary_metrics": "pocketfinancer_kotlin_parser"
            if pocketfinancer_profile
            else "strict_research_parser",
            "strict_raw_metrics_also_reported": True,
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
