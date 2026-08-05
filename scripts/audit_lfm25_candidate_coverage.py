#!/usr/bin/env python3
"""Report aggregate candidate-oracle coverage without emitting SMS content."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lfm25.candidates import extract_candidates, oracle_selection  # noqa: E402
from lfm25.contract import parse_gold  # noqa: E402


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--label-field", default="expected")
    parser.add_argument("--split")
    parser.add_argument("--split-field", default="split")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    counts: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    candidate_counts: dict[str, list[int]] = {"amount": [], "account": [], "counterparty": []}
    for row in _rows(args.dataset):
        if args.split is not None and row.get(args.split_field) != args.split:
            continue
        if args.label_field not in row:
            counts["missing_label_field"] += 1
            continue
        label = row[args.label_field]
        try:
            gold = parse_gold(label)
        except ValueError:
            counts["invalid_label"] += 1
            continue
        counts["rows"] += 1
        if gold is None:
            counts["null"] += 1
            continue
        counts["transactions"] += 1
        candidates = extract_candidates(str(row.get("sms", "")))
        candidate_counts["amount"].append(len(candidates.amounts))
        candidate_counts["account"].append(len(candidates.accounts))
        candidate_counts["counterparty"].append(len(candidates.counterparties) - 1)
        oracle = oracle_selection(gold, candidates)
        if oracle.covered:
            counts["joint_covered"] += 1
        for field in oracle.missing_fields:
            missing[field] += 1

    tx = counts["transactions"]
    report = {
        "dataset": str(args.dataset.resolve()),
        "label_field": args.label_field,
        "split": args.split,
        "counts": dict(counts),
        "coverage": {
            field: round((tx - missing[field]) / tx, 6) if tx else None
            for field in ("amount", "account", "counterparty")
        },
        "joint_coverage": round(counts["joint_covered"] / tx, 6) if tx else None,
        "missing_counts": dict(missing),
        "candidate_count_mean": {
            key: round(sum(values) / len(values), 3) if values else None
            for key, values in candidate_counts.items()
        },
        "privacy": {"sms_emitted": False, "row_level_failures_emitted": False},
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
