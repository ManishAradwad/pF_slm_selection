#!/usr/bin/env python3
"""
Segregate raw SMS dataset into 3-tier Indian financial taxonomy and benchmark
cross-platform pre-SLM filter performance for Android, iOS, and Unified pipelines.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lfm25.prefilter_simulator import (
    run_android_prefilter,
    run_ios_prefilter,
    run_unified_prefilter,
)
from lfm25.taxonomy import CATEGORIES_METADATA, classify_sms_record


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segregate Indian SMS dataset into fine-grained taxonomy and benchmark pre-SLM filters."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "all_sms.json",
        help="Path to raw all_sms.json (default: pF_slm_selection/all_sms.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "PRIVATE_DATA" / "segregated_datasets",
        help="Directory to write segregated datasets and reports (default: PRIVATE_DATA/segregated_datasets)",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=None,
        help="Optional path to output the markdown report (defaults to output-dir/taxonomy_and_prefilter_report.md)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    subcategories_dir = output_dir / "subcategories"
    subcategories_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Error: Input dataset not found at {input_path}", file=sys.stderr)
        return 1

    print(f"Loading raw SMS dataset from: {input_path}")
    with input_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    total_records = len(raw_data)
    print(f"Total messages to process: {total_records}")

    manifest_records: list[dict[str, Any]] = []
    by_primary: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_sub: dict[str, list[dict[str, Any]]] = defaultdict(list)

    # Statistics accumulators
    primary_counts: Counter[str] = Counter()
    sub_counts: Counter[str] = Counter()

    android_pass_by_sub: Counter[str] = Counter()
    ios_pass_by_sub: Counter[str] = Counter()
    unified_pass_by_sub: Counter[str] = Counter()

    android_rejection_stages: Counter[str] = Counter()
    ios_rejection_codes: Counter[str] = Counter()
    unified_rejection_stages: Counter[str] = Counter()

    for idx, item in enumerate(raw_data):
        sender = str(item.get("sender") or "")
        text = str(item.get("text") or "")
        is_from_me = bool(item.get("is_from_me", False))
        service = str(item.get("service") or "SMS")
        sms_id = item.get("id", idx + 1)
        date_str = str(item.get("date") or "")

        # 1. Run taxonomy classification
        cls_res = classify_sms_record(sender, text, is_from_me)
        primary_cat = cls_res.primary_category
        sub_cat = cls_res.subcategory

        # 2. Run pre-SLM filters
        android_res = run_android_prefilter(sender, text)
        ios_res = run_ios_prefilter(sender, text)
        unified_res = run_unified_prefilter(sender, text)

        # Update stats
        primary_counts[primary_cat] += 1
        sub_counts[sub_cat] += 1

        if android_res.passed:
            android_pass_by_sub[sub_cat] += 1
        elif android_res.rejection_stage:
            android_rejection_stages[android_res.rejection_stage] += 1

        if ios_res.passed:
            ios_pass_by_sub[sub_cat] += 1
        elif ios_res.rejection_code:
            ios_rejection_codes[ios_res.rejection_code] += 1

        if unified_res.passed:
            unified_pass_by_sub[sub_cat] += 1
        elif unified_res.rejection_stage:
            unified_rejection_stages[unified_res.rejection_stage] += 1

        enriched_record = {
            "id": sms_id,
            "date": date_str,
            "sender": sender,
            "text": text,
            "is_from_me": is_from_me,
            "service": service,
            "taxonomy": {
                "primary_category": primary_cat,
                "subcategory": sub_cat,
                "action_type": cls_res.action_type,
                "description": cls_res.description,
                "ground_truth_target": cls_res.ground_truth_target,
            },
            "prefilters": {
                "android": android_res.as_dict(),
                "ios": ios_res.as_dict(),
                "unified": unified_res.as_dict(),
            },
        }

        manifest_records.append(enriched_record)
        by_primary[primary_cat].append(enriched_record)
        by_sub[sub_cat].append(enriched_record)

    # 3. Write segregated_manifest.jsonl
    manifest_path = output_dir / "segregated_manifest.jsonl"
    print(f"Writing master segregated manifest to: {manifest_path}")
    with manifest_path.open("w", encoding="utf-8") as f:
        for record in manifest_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 4. Write primary category splits
    primary_files = {
        "TRANSACTIONAL": output_dir / "transactional.jsonl",
        "EDGE_FINANCIAL": output_dir / "edge_financial.jsonl",
        "NON_TRANSACTIONAL": output_dir / "non_transactional.jsonl",
        "UNCATEGORIZED": output_dir / "uncategorized_residuals.jsonl",
    }
    for p_name, p_path in primary_files.items():
        records = by_primary.get(p_name, [])
        with p_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 5. Write individual subcategory files
    for sub_name, records in by_sub.items():
        safe_filename = sub_name.replace(".", "_") + ".jsonl"
        sub_path = subcategories_dir / safe_filename
        with sub_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 6. Build benchmark matrix JSON
    benchmark_matrix: dict[str, Any] = {
        "dataset_total": total_records,
        "primary_breakdown": {
            k: {"count": v, "pct": round(v / total_records * 100, 2)}
            for k, v in primary_counts.most_common()
        },
        "subcategories": {},
        "overall_filter_pass_rates": {
            "android": {
                "total_passed": sum(android_pass_by_sub.values()),
                "pass_pct": round(sum(android_pass_by_sub.values()) / total_records * 100, 2),
                "rejection_stage_counts": dict(android_rejection_stages.most_common()),
            },
            "ios": {
                "total_passed": sum(ios_pass_by_sub.values()),
                "pass_pct": round(sum(ios_pass_by_sub.values()) / total_records * 100, 2),
                "rejection_code_counts": dict(ios_rejection_codes.most_common()),
            },
            "unified": {
                "total_passed": sum(unified_pass_by_sub.values()),
                "pass_pct": round(sum(unified_pass_by_sub.values()) / total_records * 100, 2),
                "rejection_stage_counts": dict(unified_rejection_stages.most_common()),
            },
        },
    }

    for sub_name, cnt in sorted(sub_counts.items(), key=lambda x: x[1], reverse=True):
        meta = CATEGORIES_METADATA.get(sub_name, {})
        android_pass = android_pass_by_sub[sub_name]
        ios_pass = ios_pass_by_sub[sub_name]
        unified_pass = unified_pass_by_sub[sub_name]

        benchmark_matrix["subcategories"][sub_name] = {
            "primary": meta.get("primary", "UNKNOWN"),
            "description": meta.get("description", ""),
            "total_count": cnt,
            "prevalence_pct": round(cnt / total_records * 100, 2),
            "android_pass_count": android_pass,
            "android_pass_pct": round(android_pass / cnt * 100, 2) if cnt else 0.0,
            "ios_pass_count": ios_pass,
            "ios_pass_pct": round(ios_pass / cnt * 100, 2) if cnt else 0.0,
            "unified_pass_count": unified_pass,
            "unified_pass_pct": round(unified_pass / cnt * 100, 2) if cnt else 0.0,
        }

    matrix_path = output_dir / "prefilter_benchmark_matrix.json"
    with matrix_path.open("w", encoding="utf-8") as f:
        json.dump(benchmark_matrix, f, indent=2)

    # 7. Generate markdown report
    report_file_path = args.report_file or (output_dir / "taxonomy_and_prefilter_report.md")
    generate_markdown_report(report_file_path, benchmark_matrix, by_sub, total_records)
    print(f"Report generated at: {report_file_path}")

    print("\n" + "=" * 65)
    print(f"{'PRIMARY CATEGORY':25s} | {'COUNT':8s} | {'PERCENTAGE':10s}")
    print("=" * 65)
    for p, cnt in primary_counts.most_common():
        print(f"{p:25s} | {cnt:8d} | {cnt / total_records * 100:8.2f}%")
    print("=" * 65)

    return 0


def generate_markdown_report(
    report_path: Path,
    benchmark_matrix: dict[str, Any],
    by_sub: dict[str, list[dict[str, Any]]],
    total_records: int,
) -> None:
    lines: list[str] = [
        "# Indian SMS Taxonomy Segregation & Pre-SLM Filter Benchmark Report",
        "",
        "## Executive Summary",
        f"- **Dataset Source**: iPhone SMS Archive (`all_sms.json`)",
        f"- **Total SMS Records Analyzed**: **{total_records:,}**",
        f"- **Taxonomy Coverage**: 3 Master Tiers, 32 Fine-Grained Subcategories",
        f"- **Unclassified Residuals**: **{benchmark_matrix['primary_breakdown'].get('UNCATEGORIZED', {}).get('count', 0)}** ({benchmark_matrix['primary_breakdown'].get('UNCATEGORIZED', {}).get('pct', 0.0)}%)",
        "",
        "---",
        "",
        "## 1. Master Category Distribution",
        "",
        "| Primary Category | Total SMS Count | Prevalence (%) | Role in PocketFinancer Pipeline |",
        "|---|---:|---:|---|",
    ]

    for p_name, data in benchmark_matrix["primary_breakdown"].items():
        role = {
            "TRANSACTIONAL": "Target for 4-field extraction (`amount`, `counterparty`, `type`, `account`)",
            "EDGE_FINANCIAL": "Hard Negatives containing money amounts; must output `null` to prevent False Positives",
            "NON_TRANSACTIONAL": "Operational/telecom/OTP traffic; filtered before SLM or outputs `null`",
            "UNCATEGORIZED": "Residual messages reserved for manual review",
        }.get(p_name, "")
        lines.append(f"| **`{p_name}`** | **{data['count']:,}** | **{data['pct']:.2f}%** | {role} |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Pre-SLM Filter Benchmarking: Android vs. iOS vs. Unified",
        "",
        "In PocketFinancer on-device runtime, incoming SMS messages pass through a deterministic prefilter before invoking the SLM (`LFM2.5-350M`).",
        "",
        "| Metric | Android (`SmsFilterPipeline.kt`) | iOS (`AlertFilter.swift`) | Proposed Unified Filter |",
        "|---|---:|---:|---:|",
        f"| **Overall Passed to SLM** | {benchmark_matrix['overall_filter_pass_rates']['android']['total_passed']:,} ({benchmark_matrix['overall_filter_pass_rates']['android']['pass_pct']}%) | {benchmark_matrix['overall_filter_pass_rates']['ios']['total_passed']:,} ({benchmark_matrix['overall_filter_pass_rates']['ios']['pass_pct']}%) | {benchmark_matrix['overall_filter_pass_rates']['unified']['total_passed']:,} ({benchmark_matrix['overall_filter_pass_rates']['unified']['pass_pct']}%) |",
        "",
        "### Key Architectural Insights:",
        "1. **False Positives Triage**: iOS rejects failed/declined transactions and standalone marketing before the model, preventing false positives, while Android forwards failed transactions to the SLM.",
        "2. **Sender Filtering**: Android enforces sender validation against 10-15 digit personal numbers, eliminating peer-to-peer chat spam from SLM evaluation.",
        "3. **Unified Filter Advantage**: The Unified specification combines Android's sender safety with iOS's hard-negative rejection while extending VPA/UPI account recognition.",
        "",
        "---",
        "",
        "## 3. Subcategory Breakdown & Filter Pass Rates",
        "",
        "| Subcategory | Primary | Count | Prev (%) | Android Pass (%) | iOS Pass (%) | Unified Pass (%) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])

    for sub_name, data in benchmark_matrix["subcategories"].items():
        p_badge = f"`{data['primary']}`"
        lines.append(
            f"| `{sub_name}` | {p_badge} | {data['total_count']:,} | {data['prevalence_pct']:.2f}% | "
            f"{data['android_pass_pct']:.1f}% | {data['ios_pass_pct']:.1f}% | {data['unified_pass_pct']:.1f}% |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Subcategory Sample Messages",
        "",
    ])

    for sub_name, data in benchmark_matrix["subcategories"].items():
        sample_records = by_sub.get(sub_name, [])[:2]
        lines.append(f"### `{sub_name}` ({data['total_count']:,} records)")
        lines.append(f"_{data['description']}_")
        lines.append("")
        for idx, sample in enumerate(sample_records, 1):
            text_preview = sample["text"].replace("\n", " ")
            if len(text_preview) > 160:
                text_preview = text_preview[:160] + "..."
            lines.append(f"- **Sample {idx}** `[{sample['sender']}]`: `{text_preview}`")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
