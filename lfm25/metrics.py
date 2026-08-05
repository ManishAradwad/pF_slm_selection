"""Conditional metrics for the LFM2.5 extraction experiments."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Callable, Iterable, Mapping

from .contract import FIELDS, ParsedOutput, field_matches, parse_gold, parse_prediction, transaction_matches


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float | None]:
    if total <= 0:
        return [None, None]
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return [round(max(0.0, center - margin), 6), round(min(1.0, center + margin), 6)]


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def score_records(
    records: Iterable[Mapping[str, Any]],
    *,
    gold_key: str = "gold",
    prediction_key: str = "prediction",
    prediction_parser: Callable[[str], ParsedOutput] = parse_prediction,
    slice_keys: tuple[str, ...] = (),
    include_per_example: bool = False,
) -> dict[str, Any]:
    """Score rows without letting gold-null cases inflate extraction fields."""
    counts: Counter[str] = Counter()
    field_hits: Counter[str] = Counter()
    slice_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_example: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        gold = parse_gold(record.get(gold_key))
        parsed = prediction_parser(str(record.get(prediction_key, "")))
        gold_transaction = gold is not None
        predicted_transaction = parsed.status == "transaction"

        counts["rows"] += 1
        counts["gold_transactions"] += int(gold_transaction)
        counts["gold_nulls"] += int(not gold_transaction)
        counts["valid_json"] += int(parsed.status != "invalid")
        counts["schema_valid"] += int(parsed.status != "invalid")

        if gold_transaction and predicted_transaction:
            counts["tp"] += 1
        elif gold_transaction:
            counts["fn"] += 1
        elif predicted_transaction:
            counts["fp"] += 1
        else:
            counts["tn"] += 1

        exact = False
        if gold_transaction and predicted_transaction:
            assert gold is not None and parsed.value is not None
            exact = transaction_matches(gold, parsed.value)
            counts["transaction_exact"] += int(exact)
            for field in FIELDS:
                field_hits[field] += int(field_matches(field, gold, parsed.value))
        elif not gold_transaction and parsed.status == "null":
            exact = True

        counts["four_field_exact"] += int(exact)
        example_summary = {
            "id": record.get("id", index),
            "exact": exact,
            "gold_transaction": gold_transaction,
            "predicted_transaction": predicted_transaction,
            "valid": parsed.status != "invalid",
        }
        per_example.append(example_summary)

        for slice_key in slice_keys:
            value = record.get(slice_key)
            if value is not None:
                slice_rows[f"{slice_key}={value}"].append(
                    {"gold": record.get(gold_key), "prediction": record.get(prediction_key), "id": record.get("id", index)}
                )

    total = counts["rows"]
    positives = counts["gold_transactions"]
    negatives = counts["gold_nulls"]
    predicted_positives = counts["tp"] + counts["fp"]
    precision = counts["tp"] / predicted_positives if predicted_positives else 0.0
    recall = counts["tp"] / positives if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    result: dict[str, Any] = {
        "n": total,
        "counts": dict(sorted(counts.items())),
        "json_validity": _ratio(counts["valid_json"], total),
        "schema_validity": _ratio(counts["schema_valid"], total),
        "four_field_exact_match": _ratio(counts["four_field_exact"], total),
        "four_field_exact_match_ci95": wilson_interval(counts["four_field_exact"], total),
        "transaction_only_exact_match": _ratio(counts["transaction_exact"], positives),
        "conditional_ghost_rate": _ratio(counts["fp"], negatives),
        "conditional_ghost_rate_ci95": wilson_interval(counts["fp"], negatives),
        "conditional_miss_rate": _ratio(counts["fn"], positives),
        "conditional_miss_rate_ci95": wilson_interval(counts["fn"], positives),
        "transaction_precision": round(precision, 6),
        "transaction_recall": round(recall, 6),
        "transaction_f1": round(f1, 6),
        "field_accuracy_on_transactions": {
            field: _ratio(field_hits[field], positives) for field in FIELDS
        },
    }
    if slice_rows:
        result["slices"] = {
            name: score_records(
                rows,
                gold_key="gold",
                prediction_key="prediction",
                prediction_parser=prediction_parser,
            )
            for name, rows in sorted(slice_rows.items())
        }
    if include_per_example:
        result["_per_example"] = per_example
    return result


def _binomial_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    cutoff = min(k, n - k)
    probability = sum(math.comb(n, i) for i in range(cutoff + 1)) / (2**n)
    return min(1.0, 2 * probability)


def paired_exact_comparison(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    first_rows = {row["id"]: row for row in first.get("_per_example", [])}
    second_rows = {row["id"]: row for row in second.get("_per_example", [])}
    shared = sorted(set(first_rows) & set(second_rows), key=str)
    first_only = sum(first_rows[key]["exact"] and not second_rows[key]["exact"] for key in shared)
    second_only = sum(second_rows[key]["exact"] and not first_rows[key]["exact"] for key in shared)
    return {
        "n_shared": len(shared),
        "first_only_correct": first_only,
        "second_only_correct": second_only,
        "ties": len(shared) - first_only - second_only,
        "mcnemar_exact_p": round(_binomial_two_sided(first_only, first_only + second_only), 8),
    }
