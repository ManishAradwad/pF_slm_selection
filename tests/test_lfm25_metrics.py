from __future__ import annotations

import json

from lfm25.metrics import paired_exact_comparison, score_records


TXN = {
    "amount": 125.0,
    "counterparty": "DEMO STORE",
    "type": "debit",
    "account": "A/c XX1234",
}


def _text(value: dict | None) -> str:
    return "null" if value is None else json.dumps(value)


def test_conditional_rates_and_fields_exclude_gold_null_rows() -> None:
    records = [
        {"id": "a", "gold": _text(TXN), "prediction": _text(TXN)},
        {"id": "b", "gold": _text(TXN), "prediction": "null"},
        {"id": "c", "gold": "null", "prediction": _text(TXN)},
        {"id": "d", "gold": "null", "prediction": "null"},
    ]
    metrics = score_records(records)

    assert metrics["counts"]["tp"] == 1
    assert metrics["counts"]["fn"] == 1
    assert metrics["counts"]["fp"] == 1
    assert metrics["counts"]["tn"] == 1
    assert metrics["conditional_ghost_rate"] == 0.5
    assert metrics["conditional_miss_rate"] == 0.5
    assert metrics["four_field_exact_match"] == 0.5
    assert metrics["transaction_only_exact_match"] == 0.5
    assert metrics["field_accuracy_on_transactions"] == {
        "amount": 0.5,
        "counterparty": 0.5,
        "type": 0.5,
        "account": 0.5,
    }


def test_invalid_output_is_a_miss_not_valid_json() -> None:
    metrics = score_records([{"gold": _text(TXN), "prediction": "not-json"}])
    assert metrics["counts"]["fn"] == 1
    assert metrics["json_validity"] == 0.0
    assert metrics["conditional_miss_rate"] == 1.0


def test_always_null_does_not_get_transaction_field_credit() -> None:
    metrics = score_records(
        [
            {"gold": _text(TXN), "prediction": "null"},
            {"gold": "null", "prediction": "null"},
        ]
    )
    assert metrics["four_field_exact_match"] == 0.5
    assert metrics["transaction_only_exact_match"] == 0.0
    assert all(value == 0.0 for value in metrics["field_accuracy_on_transactions"].values())


def test_paired_exact_comparison_counts_discordance() -> None:
    first = score_records(
        [
            {"id": "a", "gold": _text(TXN), "prediction": _text(TXN)},
            {"id": "b", "gold": "null", "prediction": _text(TXN)},
        ],
        include_per_example=True,
    )
    second = score_records(
        [
            {"id": "a", "gold": _text(TXN), "prediction": "null"},
            {"id": "b", "gold": "null", "prediction": "null"},
        ],
        include_per_example=True,
    )
    comparison = paired_exact_comparison(first, second)
    assert comparison["first_only_correct"] == 1
    assert comparison["second_only_correct"] == 1
    assert comparison["ties"] == 0
