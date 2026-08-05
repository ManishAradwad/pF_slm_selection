#!/usr/bin/env python3
"""Compare two generic prediction JSONLs while emitting aggregate results only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lfm25.contract import parse_prediction  # noqa: E402
from lfm25.metrics import paired_exact_comparison, score_records  # noqa: E402
from lfm25.provenance import fingerprint_file  # noqa: E402


def _read(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"non-object row at {path}:{line_number}")
        if "id" not in value or "gold" not in value or "prediction" not in value:
            raise ValueError(f"missing generic prediction fields at {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"no prediction rows in {path}")
    return rows


def _normalized_prediction(value: Any) -> tuple[str, str]:
    parsed = parse_prediction(str(value))
    normalized = json.dumps(parsed.value, ensure_ascii=False, sort_keys=True)
    return parsed.status, normalized


def compare(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> dict[str, Any]:
    first_by_id = {str(row["id"]): row for row in first}
    second_by_id = {str(row["id"]): row for row in second}
    if len(first_by_id) != len(first) or len(second_by_id) != len(second):
        raise ValueError("prediction IDs must be unique")
    shared_ids = sorted(set(first_by_id) & set(second_by_id))
    if len(shared_ids) != len(first_by_id) or len(shared_ids) != len(second_by_id):
        raise ValueError("prediction files must contain the same IDs")

    semantic_differences = 0
    string_differences = 0
    for row_id in shared_ids:
        first_prediction = first_by_id[row_id]["prediction"]
        second_prediction = second_by_id[row_id]["prediction"]
        string_differences += int(str(first_prediction) != str(second_prediction))
        semantic_differences += int(
            _normalized_prediction(first_prediction)
            != _normalized_prediction(second_prediction)
        )

    first_score = score_records(first, include_per_example=True)
    second_score = score_records(second, include_per_example=True)
    paired = paired_exact_comparison(first_score, second_score)
    return {
        "n_shared": len(shared_ids),
        "prediction_string_differences": string_differences,
        "semantic_prediction_differences": semantic_differences,
        "first_exact": first_score["counts"].get("four_field_exact", 0),
        "second_exact": second_score["counts"].get("four_field_exact", 0),
        "paired_exact": paired,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", required=True, type=Path)
    parser.add_argument("--second", required=True, type=Path)
    parser.add_argument("--first-name", default="first")
    parser.add_argument("--second-name", default="second")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = {
        "first": args.first_name,
        "second": args.second_name,
        "provenance": {
            "first": fingerprint_file(args.first),
            "second": fingerprint_file(args.second),
        },
        **compare(_read(args.first), _read(args.second)),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
