from __future__ import annotations

from collections import Counter
import hashlib
import json
from decimal import Decimal
from pathlib import Path
import subprocess
import sys

import pytest

from lfm25 import metrics as metrics_module
from lfm25.candidate_protocol import (
    build_protocol_request,
    serialize_selector_target,
)
from lfm25.contract import parse_gold as historical_parse_gold
from scripts import evaluate_lfm25_candidate_protocol_v1_hf as evaluator

DIAGNOSTIC_DATASET = Path(__file__).resolve().parents[1] / "DATA" / "extraction_ds.jsonl"

SMS = "INR 1,234.50 debited from A/c XX7890 to ACME STORE."
GOLD = {
    "type": "debit",
    "amount": 1234.5,
    "account": "7890",
    "counterparty": "ACME STORE",
}


def _locked_model(tmp_path: Path) -> tuple[Path, Path]:
    model = tmp_path / "model"
    model.mkdir()
    contents = {
        "LICENSE": "license",
        "README.md": "readme",
        "chat_template.jinja": "template",
        "config.json": "{}",
        "generation_config.json": "{}",
        "model.safetensors": "weights",
        "tokenizer.json": "{}",
        "tokenizer_config.json": "{}",
    }
    for name, content in contents.items():
        (model / name).write_text(content, encoding="utf-8")
    model_lock = tmp_path / "model.lock.json"
    model_lock.write_text(
        json.dumps(
            {
                "model": {
                    "files": {
                        name: hashlib.sha256(content.encode()).hexdigest()
                        for name, content in contents.items()
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return model, model_lock


def test_resolve_uses_strict_v1_parser() -> None:
    request = build_protocol_request("VK-BANK", SMS)
    target = serialize_selector_target(GOLD, request)

    parsed, interventions = evaluator._resolve(target, request)

    assert parsed.status == "transaction"
    assert parsed.value == {
        **GOLD,
        "account": request.candidates.accounts[0].value,
    }
    assert interventions == ()

    rejected, interventions = evaluator._resolve(target + " trailing", request)
    assert rejected.status == "invalid"
    assert rejected.error == "trailing_content"
    assert interventions == ()


def test_diagnostic_gold_projector_preserves_exact_amount_and_drops_only_date() -> None:
    legacy = evaluator._project_diagnostic_gold(
        '{"type":"debit","amount":1234.50,"account":"7890",'
        '"counterparty":"ACME STORE","date":"2026-08-08"}'
    )
    assert legacy == {
        "type": "debit",
        "amount": Decimal("1234.50"),
        "account": "7890",
        "counterparty": "ACME STORE",
    }
    assert evaluator._project_diagnostic_gold("null") is None
    assert evaluator._project_diagnostic_gold(None) is None
    assert evaluator._project_diagnostic_gold(
        {
            "type": "credit",
            "amount": 7,
            "account": None,
            "counterparty": None,
        }
    ) == {
        "type": "credit",
        "amount": 7,
        "account": None,
        "counterparty": None,
    }


@pytest.mark.parametrize(
    "gold",
    [
        (
            '{"type":"debit","amount":1,"amount":2,"account":"1",'
            '"counterparty":"X","date":"2026-08-08"}'
        ),
        ('{"type":"debit","amount":NaN,"account":"1","counterparty":"X","date":"2026-08-08"}'),
        ('{"type":"debit","amount":1,"account":"1","date":"2026-08-08"}'),
        (
            '{"type":"debit","amount":1,"account":"1","counterparty":"X",'
            '"date":"2026-08-08","memo":"extra"}'
        ),
        "[]",
    ],
)
def test_diagnostic_gold_projector_rejects_non_strict_shapes(gold: str) -> None:
    with pytest.raises(ValueError):
        evaluator._project_diagnostic_gold(gold)


def test_live_diagnostic_gold_projects_with_locked_candidate_coverage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    totals: Counter[str] = Counter()
    for line in DIAGNOSTIC_DATASET.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        projected = evaluator._project_diagnostic_gold(evaluator.base._gold(row))
        totals["rows"] += 1
        if projected is None:
            totals["nulls"] += 1
            continue

        totals["transactions"] += 1
        assert set(projected) == {"type", "amount", "account", "counterparty"}
        coverage = evaluator._oracle(
            projected,
            evaluator._request(str(row.get("sms", ""))),
        )
        totals["amount_covered"] += int("amount" not in coverage.missing_fields)
        totals["account_covered"] += int("account" not in coverage.missing_fields)
        totals["counterparty_covered"] += int("counterparty" not in coverage.missing_fields)
        totals["joint_covered"] += int(coverage.covered)

    assert totals == Counter(
        {
            "rows": 203,
            "transactions": 114,
            "nulls": 89,
            "amount_covered": 114,
            "account_covered": 114,
            "counterparty_covered": 113,
            "joint_covered": 113,
        }
    )
    assert capsys.readouterr().out == ""


def test_install_patches_only_historical_evaluator_gold_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluator.base, "parse_gold", historical_parse_gold)
    score_parser = metrics_module.parse_gold

    evaluator._install_v1_contract()

    assert evaluator.base.parse_gold is evaluator._project_diagnostic_gold
    assert metrics_module.parse_gold is score_parser
    assert score_parser is historical_parse_gold


def test_main_forwards_argv_and_rewrites_v1_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "evidence"
    model, model_lock = _locked_model(tmp_path)
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"id":"one"}\n{"id":"two"}\n', encoding="utf-8")
    forwarded: list[str] = []

    hook_names = (
        "candidate_selector_messages",
        "extract_candidates",
        "oracle_selection",
        "resolve_selector_prediction",
        "contract_provenance",
        "parse_gold",
    )
    for name in hook_names:
        monkeypatch.setattr(evaluator.base, name, getattr(evaluator.base, name))

    def fake_base_main() -> int:
        forwarded.extend(sys.argv[1:])
        destination = Path(forwarded[forwarded.index("--output-dir") + 1])
        destination.mkdir(parents=True)
        (destination / "samples.jsonl").write_text(
            "\n".join(
                (
                    json.dumps({"selector_error": "accepted_transaction"}),
                    json.dumps({"selector_error": "trailing_content"}),
                )
            )
            + "\n",
            encoding="utf-8",
        )
        (destination / "metrics.json").write_text(
            json.dumps(
                {
                    "provenance": {
                        "android_contract": {"legacy": True},
                        "candidate_code": {"legacy": True},
                        "evaluator": {"path": "base", "bytes": 1, "sha256": "0" * 64},
                    },
                    "runtime": {"model_invocations": 2},
                    "prefilter": {"enabled": True, "model_invocations": 2},
                    "hybrid_safety": {"enabled": True},
                }
            ),
            encoding="utf-8",
        )
        return 7

    monkeypatch.setattr(evaluator.base, "main", fake_base_main)
    arguments = [
        "--model",
        str(model),
        "--model-lock",
        str(model_lock),
        "--dataset",
        str(dataset),
        "--output-dir",
        str(output_dir),
        "--seed",
        "29",
    ]

    assert evaluator.main(arguments) == 7
    assert forwarded == evaluator._without_model_lock(arguments)

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["provenance"]["pipeline"] == ("pocketfinancer_candidate_protocol_v1_hf")
    assert metrics["provenance"]["candidate_protocol"]["name"] == ("candidate_protocol_v1")
    assert "android_contract" not in metrics["provenance"]
    assert "candidate_code" not in metrics["provenance"]
    assert metrics["selector_reason_counts"] == {
        "accepted_transaction": 1,
        "trailing_content": 1,
    }
    assert metrics["candidate_protocol_acceptance"] == {
        "model_invocations": 2,
        "accepted_outputs": 1,
        "rejected_outputs": 1,
        "strict_schema_acceptance_rate": 0.5,
        "accepted_transactions": 1,
        "source_grounded_transactions": 1,
        "source_grounded_transaction_rate": 1.0,
    }
    assert metrics["provenance"]["generation_engine"]["path"].endswith(
        "evaluate_lfm25_candidate_hf.py"
    )
    assert metrics["provenance"]["evaluator"]["path"].endswith(
        "evaluate_lfm25_candidate_protocol_v1_hf.py"
    )
    assert metrics["provenance"]["training_run"] is None
    assert set(metrics["provenance"]["model"]["files"]) == {
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    assert metrics["provenance"]["model_lock"]["filename"] == "model.lock.json"
    assert set(metrics["provenance"]["model_lock"]) == {"filename", "bytes", "sha256"}
    assert metrics["provenance"]["dataset"]["row_count"] == 2
    assert metrics["provenance"]["dataset"]["row_limit"] is None
    assert set(metrics["provenance"]["candidate_profile"]) == {
        "candidate",
        "baseline",
        "golden_vectors",
    }
    assert all(
        set(item) == {"filename", "bytes", "sha256"}
        for item in metrics["provenance"]["candidate_profile"].values()
    )
    assert metrics["provenance"]["platform_gates"] == evaluator.PLATFORM_GATES

    assert metrics["provenance"]["selection_prefilter"] == {
        "applied": True,
        "part_of_android_current": True,
        "rejected_prediction": "null",
    }
    assert metrics["runtime"]["model_output_protocol"] == "candidate_protocol_v1"
    assert json.loads(capsys.readouterr().out) == metrics


def test_main_requires_model_lock_before_historical_evaluator(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        evaluator.main(
            [
                "--model",
                str(tmp_path / "model"),
                "--dataset",
                str(tmp_path / "dataset.jsonl"),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )

    assert error.value.code == 2


def test_wrapper_filters_model_lock_for_real_historical_parser(tmp_path: Path) -> None:
    arguments = [
        "--model",
        "local-model",
        "--model-lock",
        str(tmp_path / "model.lock.json"),
        "--dataset",
        str(tmp_path / "dataset.jsonl"),
        "--output-dir",
        str(tmp_path / "output"),
        "--batch-size",
        "0",
    ]
    unfiltered = subprocess.run(
        [sys.executable, str(Path(evaluator.base.__file__)), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    assert unfiltered.returncode == 2
    assert "unrecognized arguments: --model-lock" in unfiltered.stderr

    filtered = subprocess.run(
        [
            sys.executable,
            str(Path(evaluator.base.__file__)),
            *evaluator._without_model_lock(arguments),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert filtered.returncode == 2
    assert "batch size and max new tokens must be positive" in filtered.stderr
    assert "unrecognized arguments" not in filtered.stderr


def test_main_rejects_dataset_replacement_after_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, model_lock = _locked_model(tmp_path)
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"id":"one"}\n', encoding="utf-8")
    output_dir = tmp_path / "evidence"

    def fake_base_main() -> int:
        output_dir.mkdir()
        (output_dir / "samples.jsonl").write_text(
            json.dumps({"selector_error": "accepted_transaction"}) + "\n",
            encoding="utf-8",
        )
        (output_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "provenance": {"evaluator": {}},
                    "runtime": {"model_invocations": 1},
                    "prefilter": {"enabled": True, "model_invocations": 1},
                }
            ),
            encoding="utf-8",
        )
        replacement = tmp_path / "dataset.replacement"
        replacement.write_text('{"id":"replacement"}\n', encoding="utf-8")
        replacement.replace(dataset)
        return 0

    monkeypatch.setattr(evaluator.base, "main", fake_base_main)
    with pytest.raises(RuntimeError, match="changed after the pre-inference snapshot"):
        evaluator.main(
            [
                "--model",
                str(model),
                "--model-lock",
                str(model_lock),
                "--dataset",
                str(dataset),
                "--output-dir",
                str(output_dir),
            ]
        )


def test_candidate_contract_stability_check_rejects_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = evaluator._candidate_execution_contract_evidence()
    monkeypatch.setattr(evaluator, "_candidate_execution_contract_evidence", lambda: {})

    with pytest.raises(RuntimeError, match="code or contract changed"):
        evaluator._assert_candidate_execution_contract_unchanged(expected)


def test_main_forbids_historical_hybrid_override(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="does not permit --hybrid-safety"):
        evaluator.main(
            [
                "--model",
                "local-model",
                "--dataset",
                str(tmp_path / "dataset.jsonl"),
                "--output-dir",
                str(tmp_path / "evidence"),
                "--hybrid-safety",
            ]
        )


def test_rewrite_metrics_handles_empty_limited_evaluation(tmp_path: Path) -> None:
    output_dir = tmp_path / "empty"
    model, model_lock = _locked_model(tmp_path)
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("", encoding="utf-8")
    input_evidence = evaluator._evaluation_input_evidence(
        model=str(model),
        adapter=None,
        dataset=dataset,
        model_lock=model_lock,
    )
    output_dir.mkdir()
    (output_dir / "samples.jsonl").write_text("", encoding="utf-8")
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "provenance": {"evaluator": {}},
                "runtime": {"model_invocations": 0},
                "prefilter": {"enabled": True, "model_invocations": 0},
            }
        ),
        encoding="utf-8",
    )

    metrics = evaluator._rewrite_metrics(
        output_dir,
        input_evidence=input_evidence,
        execution_contract_evidence=evaluator._candidate_execution_contract_evidence(),
        row_count=0,
        row_limit=0,
    )

    assert metrics["candidate_protocol_acceptance"]["strict_schema_acceptance_rate"] == 1.0
    assert metrics["candidate_protocol_acceptance"]["accepted_outputs"] == 0
    assert metrics["candidate_protocol_acceptance"]["source_grounded_transaction_rate"] is None
