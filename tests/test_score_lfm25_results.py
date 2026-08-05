from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
SCORER = REPO_ROOT / "scripts" / "score_lfm25_results.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_scorer(*args: str) -> tuple[dict, str]:
    completed = subprocess.run(
        [sys.executable, str(SCORER), *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout), completed.stdout


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_always_null_summary_has_content_free_provenance(tmp_path: Path) -> None:
    secret = "SENSITIVE_SAMPLE_CONTENT_MUST_NOT_BE_LOGGED"
    dataset = tmp_path / "evaluation.jsonl"
    _write_jsonl(
        dataset,
        [
            {"expected": None, "sms": secret},
            {
                "expected": {
                    "account": "A/c XX0000",
                    "amount": 10.0,
                    "counterparty": "DEMO",
                    "type": "debit",
                },
                "sms": "another private value",
            },
        ],
    )
    output = tmp_path / "summary.json"

    report, stdout = _run_scorer(
        "--always-null",
        str(dataset),
        "--out",
        str(output),
    )

    assert report == json.loads(output.read_text(encoding="utf-8"))
    assert secret not in stdout
    assert "another private value" not in stdout
    assert report["variants"]["always_null"]["counts"]["four_field_exact"] == 1
    assert report["variants"]["always_null"]["counts"]["rows"] == 2

    provenance = report["provenance"]
    assert provenance["schema"] == "lfm25-baseline-provenance-v1"
    assert provenance["hash_algorithm"] == "sha256"
    assert provenance["source"]["kind"] == "always_null_input"
    assert provenance["source"]["files"][0]["sha256"] == _sha256(dataset)
    assert provenance["dataset"]["sha256"] == _sha256(dataset)
    assert provenance["filter"]["name"] == "always_null_literal"
    assert provenance["filter"]["stage"] == "generated_by_scorer"
    assert len(provenance["filter"]["sha256"]) == 64
    assert len(provenance["scorer"]["sha256"]) == 64
    assert {item["role"] for item in provenance["scorer"]["files"]} == {
        "contract",
        "entrypoint",
        "metrics",
    }


def test_lm_eval_summary_hashes_samples_dataset_filter_and_preserves_pairing(
    tmp_path: Path,
) -> None:
    secret = "FILTERED_SAMPLE_CONTENT_MUST_NOT_BE_LOGGED"
    dataset = tmp_path / "canonical-regression.jsonl"
    _write_jsonl(dataset, [{"expected": None, "sms": secret}])
    transaction = json.dumps(
        {
            "account": "A/c XX0000",
            "amount": 10.0,
            "counterparty": "DEMO",
            "type": "debit",
        },
        sort_keys=True,
    )
    sample_rows_q4 = [
        {
            "doc_id": 0,
            "filter": "extract_json_nonnull",
            "filtered_resps": ["null"],
            "target": "null",
        },
        {
            "doc_id": 1,
            "filter": "extract_json_nonnull",
            "filtered_resps": ["null"],
            "target": transaction,
        },
        {
            "doc_id": 99,
            "filter": "other_filter",
            "filtered_resps": [secret],
            "target": "null",
        },
    ]
    sample_rows_q8 = [
        {
            "doc_id": 0,
            "filter": "extract_json_nonnull",
            "filtered_resps": ["null"],
            "target": "null",
        },
        {
            "doc_id": 1,
            "filter": "extract_json_nonnull",
            "filtered_resps": [transaction],
            "target": transaction,
        },
    ]
    q4 = tmp_path / "lfm25" / "untouched_q4" / "model" / "samples.jsonl"
    q8 = tmp_path / "lfm25" / "untouched_q8" / "model" / "samples.jsonl"
    _write_jsonl(q4, sample_rows_q4)
    _write_jsonl(q8, sample_rows_q8)

    report, stdout = _run_scorer(
        "--samples",
        str(q4),
        "--samples",
        str(q8),
        "--dataset",
        str(dataset),
        "--filter",
        "extract_json_nonnull",
    )

    assert secret not in stdout
    assert report["variants"]["lfm25:untouched_q4"]["counts"]["rows"] == 2
    assert report["variants"]["lfm25:untouched_q8"]["counts"]["rows"] == 2
    assert report["paired_exact"] == {
        "first": "lfm25:untouched_q4",
        "first_only_correct": 0,
        "mcnemar_exact_p": 1.0,
        "n_shared": 2,
        "second": "lfm25:untouched_q8",
        "second_only_correct": 1,
        "ties": 1,
    }

    provenance = report["provenance"]
    assert provenance["source"]["kind"] == "lm_eval_samples"
    assert [item["sha256"] for item in provenance["source"]["files"]] == [
        _sha256(q4),
        _sha256(q8),
    ]
    assert len(provenance["source"]["sha256"]) == 64
    assert provenance["dataset"]["sha256"] == _sha256(dataset)
    assert provenance["filter"]["name"] == "extract_json_nonnull"
    assert provenance["filter"]["stage"] == "upstream_filtered_resps"
    assert {item["path"] for item in provenance["filter"]["files"]} == {
        "DATA/sms_extraction.yaml",
        "DATA/utils.py",
    }
    assert len(provenance["filter"]["sha256"]) == 64
