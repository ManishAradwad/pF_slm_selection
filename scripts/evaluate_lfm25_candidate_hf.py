#!/usr/bin/env python3
"""Evaluate the grounded candidate-selector LFM track with Android prefiltering."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lfm25.android_contract import (  # noqa: E402
    contract_provenance,
    prefilter_sms,
    summarize_prefilter,
)
from lfm25.candidates import (  # noqa: E402
    candidate_selector_messages,
    extract_candidates,
    oracle_selection,
    resolve_selector_prediction,
)
from lfm25.contract import parse_gold  # noqa: E402
from lfm25.metrics import score_records  # noqa: E402
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
    raise ValueError("candidate evaluation row has no expected or label field")


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return round(ordered[index], 3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--n-ctx", type=int, default=1024)
    parser.add_argument("--repeat-penalty", type=float, default=1.0)
    parser.add_argument(
        "--apply-prefilter",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--hybrid-safety",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply source-grounded VPA recovery and counterparty distractor ranking",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        parser.error("batch size and max new tokens must be positive")
    if args.n_ctx <= args.max_new_tokens:
        parser.error("--n-ctx must exceed --max-new-tokens")
    if args.repeat_penalty <= 0:
        parser.error("--repeat-penalty must be positive")

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

    records: dict[int, dict[str, Any]] = {}
    pending: list[tuple[int, dict[str, Any], Any]] = []
    coverage_counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        disposition = (
            prefilter_sms(str(row.get("sender", "")), str(row.get("sms", "")))
            if args.apply_prefilter
            else None
        )
        passed = disposition.accepted if disposition is not None else True
        gold = _gold(row)
        common = {
            "id": row.get("id", row.get("public_id", index)),
            "gold": gold,
            "prefilter_applied": args.apply_prefilter,
            "prefilter_passed": passed,
            "prefilter_rejection_stage": (
                disposition.rejection_stage if disposition is not None else None
            ),
            "model_invoked": passed,
            **{key: row[key] for key in ("class", "template_family") if key in row},
        }
        if not passed:
            records[index] = {
                **common,
                "prediction": "null",
                "selector_status": "prefiltered_null",
                "prompt_tokens": 0,
                "output_tokens": 0,
                "latency_ms_batch_amortized": 0.0,
            }
            continue

        candidates = extract_candidates(str(row.get("sms", "")))
        pending.append((index, row, candidates))
        records[index] = common
        parsed_gold = parse_gold(gold)
        if parsed_gold is not None:
            coverage_counts["transactions"] += 1
            oracle = oracle_selection(parsed_gold, candidates)
            coverage_counts["joint_covered"] += int(oracle.covered)
            for field in ("amount", "account", "counterparty"):
                coverage_counts[f"{field}_covered"] += int(field not in oracle.missing_fields)

    latencies: list[float] = []
    prompt_lengths: list[int] = []
    output_lengths: list[int] = []
    selector_statuses: Counter[str] = Counter()
    intervention_counts: Counter[str] = Counter()
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        chats = [
            candidate_selector_messages(
                str(row.get("sender", "")), str(row.get("sms", ""))
            )
            for _, row, _ in batch
        ]
        encoded = tokenizer.apply_chat_template(
            chats,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        )
        longest = int(encoded["attention_mask"].sum(dim=1).max().item())
        if longest + args.max_new_tokens > args.n_ctx:
            raise ValueError(
                f"prompt ({longest}) + completion ({args.max_new_tokens}) exceeds "
                f"n_ctx={args.n_ctx}"
            )
        encoded = {key: value.to("cuda") for key, value in encoded.items()}
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                repetition_penalty=args.repeat_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        prompt_width = encoded["input_ids"].shape[1]
        completions = generated[:, prompt_width:]
        decoded = tokenizer.batch_decode(completions, skip_special_tokens=True)
        for offset, ((index, _row, candidates), raw_prediction) in enumerate(
            zip(batch, decoded)
        ):
            parsed, interventions = resolve_selector_prediction(
                raw_prediction,
                candidates,
                hybrid_safety=args.hybrid_safety,
            )
            completion_ids = completions[offset]
            output_length = int(
                (completion_ids != tokenizer.pad_token_id).sum().item()
            )
            prompt_length = int(encoded["attention_mask"][offset].sum().item())
            latency = elapsed_ms / len(batch)
            selector_statuses[parsed.status] += 1
            intervention_counts.update(interventions)
            latencies.append(latency)
            prompt_lengths.append(prompt_length)
            output_lengths.append(output_length)
            records[index].update(
                {
                    # Invalid raw generations are deliberately not persisted: a
                    # failed model can echo private prompt content.
                    "prediction": (
                        parsed.extracted
                        if parsed.status != "invalid"
                        else "__invalid_selector__"
                    ),
                    "selector_status": parsed.status,
                    "selector_error": parsed.error,
                    "prompt_tokens": prompt_length,
                    "output_tokens": output_length,
                    "latency_ms_batch_amortized": round(latency, 3),
                }
            )

    output_records = [records[index] for index in range(len(rows))]
    model_records = [record for record in output_records if record["model_invoked"]]
    whole = score_records(output_records, slice_keys=("template_family", "class"))
    conditional = score_records(model_records, slice_keys=("template_family", "class"))
    metrics = dict(whole)
    metrics["whole_pipeline"] = whole
    metrics["conditional_model"] = conditional
    metrics["prefilter"] = summarize_prefilter(
        output_records, enabled=args.apply_prefilter
    )
    transaction_count = coverage_counts["transactions"]
    metrics["candidate_oracle"] = {
        "transactions": transaction_count,
        "joint_covered": coverage_counts["joint_covered"],
        "joint_coverage": (
            round(coverage_counts["joint_covered"] / transaction_count, 6)
            if transaction_count
            else None
        ),
        "field_coverage": {
            field: (
                round(coverage_counts[f"{field}_covered"] / transaction_count, 6)
                if transaction_count
                else None
            )
            for field in ("amount", "account", "counterparty")
        },
    }
    metrics["selector_status_counts"] = dict(sorted(selector_statuses.items()))
    metrics["hybrid_safety"] = {
        "enabled": args.hybrid_safety,
        "intervention_counts": dict(sorted(intervention_counts.items())),
    }
    metrics["runtime"] = {
        "rows": len(rows),
        "model_invocations": len(model_records),
        "batch_size": args.batch_size,
        "n_ctx": args.n_ctx,
        "max_new_tokens": args.max_new_tokens,
        "latency_ms_p50_batch_amortized": _percentile(latencies, 0.5),
        "latency_ms_p95_batch_amortized": _percentile(latencies, 0.95),
        "prompt_tokens_mean": (
            round(statistics.fmean(prompt_lengths), 2) if prompt_lengths else None
        ),
        "output_tokens_mean": (
            round(statistics.fmean(output_lengths), 2) if output_lengths else None
        ),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    }

    model_path = Path(args.model)
    model_evidence = (
        fingerprint_named_files(
            model_path,
            (
                "model.safetensors",
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
            ),
        )
        if model_path.is_dir()
        else {"identifier": args.model}
    )
    metrics["provenance"] = {
        "pipeline": "android_prefilter_then_grounded_candidate_selector_v1",
        "hybrid_safety": {
            "enabled": args.hybrid_safety,
            "rules": [
                "vpa_unambiguous_transaction_fallback",
                "counterparty_currency_contamination_override",
            ]
            if args.hybrid_safety
            else [],
        },
        "android_contract": contract_provenance(),
        "candidate_code": fingerprint_file(REPO_ROOT / "lfm25" / "candidates.py"),
        "evaluator": fingerprint_file(Path(__file__)),
        "model": model_evidence,
        "adapter": (
            fingerprint_named_files(
                args.adapter,
                ("adapter_model.safetensors", "adapter_config.json"),
            )
            if args.adapter
            else None
        ),
        "dataset": fingerprint_file(args.dataset),
        "code_sha256": code_fingerprints(REPO_ROOT),
        "decode": {
            "engine": "transformers",
            "do_sample": False,
            "repeat_penalty": args.repeat_penalty,
            "max_new_tokens": args.max_new_tokens,
            "n_ctx": args.n_ctx,
            "seed": args.seed,
        },
        "privacy": {
            "raw_selector_failures_persisted": False,
            "source_sms_persisted_in_results": False,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for record in output_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
