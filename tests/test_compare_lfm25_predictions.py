from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
COMPARER = REPO_ROOT / "scripts" / "compare_lfm25_predictions.py"
TRANSACTION = {
    "account": "A/c XX0000",
    "amount": 10.0,
    "counterparty": "DEMO",
    "type": "debit",
}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _run_comparer(first: Path, second: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(COMPARER),
            "--first",
            str(first),
            "--second",
            str(second),
            *args,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def _comparison_files(tmp_path: Path) -> tuple[Path, Path]:
    gold = json.dumps(TRANSACTION, sort_keys=True)
    prediction = json.dumps(TRANSACTION, sort_keys=True)
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_jsonl(
        first,
        [
            {
                "id": "private-row-id",
                "gold": gold,
                "prediction": prediction,
                "app_prediction": prediction,
            }
        ],
    )
    _write_jsonl(
        second,
        [
            {
                "id": "private-row-id",
                "gold": gold,
                "prediction": prediction,
                "app_prediction": "null",
            }
        ],
    )
    return first, second


def test_default_prediction_field_preserves_existing_comparison_behavior(
    tmp_path: Path,
) -> None:
    first, second = _comparison_files(tmp_path)

    completed = _run_comparer(first, second)

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["prediction_field"] == "prediction"
    assert report["prediction_string_differences"] == 0
    assert report["semantic_prediction_differences"] == 0
    assert report["first_exact"] == 1
    assert report["second_exact"] == 1
    assert report["paired_exact"]["ties"] == 1
    assert "private-row-id" not in completed.stdout
    assert "A/c XX0000" not in completed.stdout


def test_app_prediction_field_controls_differences_and_scoring(tmp_path: Path) -> None:
    first, second = _comparison_files(tmp_path)

    completed = _run_comparer(
        first,
        second,
        "--prediction-field",
        "app_prediction",
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["prediction_field"] == "app_prediction"
    assert report["prediction_string_differences"] == 1
    assert report["semantic_prediction_differences"] == 1
    assert report["first_exact"] == 1
    assert report["second_exact"] == 0
    assert report["paired_exact"]["first_only_correct"] == 1
    assert "private-row-id" not in completed.stdout
    assert "A/c XX0000" not in completed.stdout


def test_selected_field_must_be_present_in_both_files(tmp_path: Path) -> None:
    first, second = _comparison_files(tmp_path)
    second_rows = [
        {
            "id": "private-row-id",
            "gold": json.dumps(TRANSACTION, sort_keys=True),
            "prediction": "null",
        }
    ]
    _write_jsonl(second, second_rows)

    completed = _run_comparer(
        first,
        second,
        "--prediction-field",
        "app_prediction",
    )

    assert completed.returncode != 0
    assert "missing required field(s) app_prediction" in completed.stderr
    assert "private-row-id" not in completed.stderr
    assert "A/c XX0000" not in completed.stderr


def test_mismatched_gold_labels_fail_closed_without_disclosing_rows(
    tmp_path: Path,
) -> None:
    first, second = _comparison_files(tmp_path)
    mismatched_transaction = {
        **TRANSACTION,
        "amount": 20.0,
        "counterparty": "SYNTHETIC OTHER",
    }
    _write_jsonl(
        second,
        [
            {
                "id": "private-row-id",
                "gold": json.dumps(mismatched_transaction, sort_keys=True),
                "prediction": json.dumps(mismatched_transaction, sort_keys=True),
                "app_prediction": json.dumps(mismatched_transaction, sort_keys=True),
            }
        ],
    )

    completed = _run_comparer(first, second)

    assert completed.returncode != 0
    assert "same gold label for each ID" in completed.stderr
    assert completed.stdout == ""
    assert "private-row-id" not in completed.stderr
    assert "SYNTHETIC OTHER" not in completed.stderr
    assert "A/c XX0000" not in completed.stderr


def test_equivalent_gold_serializations_are_accepted(tmp_path: Path) -> None:
    first, second = _comparison_files(tmp_path)
    _write_jsonl(
        second,
        [
            {
                "id": "private-row-id",
                "gold": TRANSACTION,
                "prediction": json.dumps(TRANSACTION, sort_keys=True),
                "app_prediction": "null",
            }
        ],
    )

    completed = _run_comparer(first, second)

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["n_shared"] == 1
