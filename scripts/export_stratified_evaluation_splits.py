#!/usr/bin/env python3
"""
Generate stratified training, test, and hard-negative evaluation splits from the segregated SMS manifest.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate stratified training and hard-negative evaluation splits."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "PRIVATE_DATA" / "segregated_datasets" / "segregated_manifest.jsonl",
        help="Path to segregated_manifest.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "PRIVATE_DATA" / "segregated_datasets" / "evaluation_splits",
        help="Directory to save generated evaluation splits",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible stratified sampling",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}. Please run segregate_sms_dataset.py first.", file=sys.stderr)
        return 1

    random.seed(args.seed)

    print(f"Reading manifest from: {manifest_path}")
    records_by_sub: dict[str, list[dict[str, Any]]] = defaultdict(list)
    records_by_primary: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_records: list[dict[str, Any]] = []

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            all_records.append(item)
            sub = item["taxonomy"]["subcategory"]
            prim = item["taxonomy"]["primary_category"]
            records_by_sub[sub].append(item)
            records_by_primary[prim].append(item)

    print(f"Loaded {len(all_records)} records across {len(records_by_sub)} subcategories.")

    # 1. HARD NEGATIVE EVALUATION SUITE
    # Contains all high-risk edge cases + sampled OTPs & Telecom to test model rejection
    hard_negatives: list[dict[str, Any]] = []
    for sub, items in records_by_sub.items():
        if sub.startswith("edge."):
            hard_negatives.extend(items)
        elif sub == "non_tx.otp_auth_2fa":
            hard_negatives.extend(random.sample(items, min(len(items), 100)))

    random.shuffle(hard_negatives)
    hard_neg_path = output_dir / "hard_negative_eval_suite.jsonl"
    with hard_neg_path.open("w", encoding="utf-8") as f:
        for r in hard_negatives:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Exported {len(hard_negatives)} hard-negative evaluation cases to: {hard_neg_path}")

    # 2. BALANCED TRANSACTION TRAINING POOL
    # Sample up to 100 examples per transactional subcategory to avoid UPI or Card dominance
    balanced_tx_train: list[dict[str, Any]] = []
    for sub, items in records_by_sub.items():
        if sub.startswith("tx."):
            sample_size = min(len(items), 100)
            balanced_tx_train.extend(random.sample(items, sample_size))

    random.shuffle(balanced_tx_train)
    balanced_train_path = output_dir / "balanced_tx_train_pool.jsonl"
    with balanced_train_path.open("w", encoding="utf-8") as f:
        for r in balanced_tx_train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Exported {len(balanced_tx_train)} balanced transaction training candidates to: {balanced_train_path}")

    # 3. STRATIFIED HOLDOUT BENCHMARK (10% sample across all subcategories)
    stratified_holdout: list[dict[str, Any]] = []
    for sub, items in records_by_sub.items():
        holdout_count = max(1, int(len(items) * 0.10))
        stratified_holdout.extend(random.sample(items, min(len(items), holdout_count)))

    random.shuffle(stratified_holdout)
    holdout_path = output_dir / "stratified_holdout_test_suite.jsonl"
    with holdout_path.open("w", encoding="utf-8") as f:
        for r in stratified_holdout:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Exported {len(stratified_holdout)} stratified holdout test cases to: {holdout_path}")

    # 4. Summary Metadata
    summary = {
        "seed": args.seed,
        "total_source_records": len(all_records),
        "hard_negative_eval_count": len(hard_negatives),
        "balanced_tx_train_count": len(balanced_tx_train),
        "stratified_holdout_test_count": len(stratified_holdout),
        "hard_negative_composition": dict(Counter(r["taxonomy"]["subcategory"] for r in hard_negatives)),
        "balanced_train_composition": dict(Counter(r["taxonomy"]["subcategory"] for r in balanced_tx_train)),
        "holdout_composition": dict(Counter(r["taxonomy"]["subcategory"] for r in stratified_holdout)),
    }
    summary_path = output_dir / "splits_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Splits summary saved to: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
