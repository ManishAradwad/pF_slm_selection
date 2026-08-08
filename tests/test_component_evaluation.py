from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from lfm25.component_evaluation import (
    ComponentEvaluationError,
    adapt_workbench_annotation,
    evaluate_component_rows,
    evaluate_jsonl,
    load_contract,
    read_paired_jsonl,
)
from scripts import evaluate_lfm25_components as cli


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "DATA" / "annotation_component_v1_synthetic.jsonl"
CONTRACT = REPO_ROOT / "configs" / "contracts" / "annotation-component-evaluation-v1.json"


def _assert_no_float(value: Any) -> None:
    assert not isinstance(value, float)
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_no_float(key)
            _assert_no_float(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_float(item)


def _span(sms: str, text: str, occurrence: int = 0) -> dict[str, Any]:
    start = -1
    next_start = 0
    for _ in range(occurrence + 1):
        start = sms.index(text, next_start)
        next_start = start + len(text)
    return {
        "text": text,
        "start_utf8_byte": len(sms[:start].encode("utf-8")),
        "end_utf8_byte": len(sms[: start + len(text)].encode("utf-8")),
    }


def _decimal_pair(
    gold_decimal: str,
    candidate_decimal: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sms = (
        f"Invented Juniper Bank: INR {gold_decimal} debited from A/c XX8080 "
        f"to PIXEL CANTEEN; receipt INR {candidate_decimal}."
    )
    annotation = {
        "id": "decimal-boundary",
        "source": {
            "sender": "VK-INVENT",
            "sms": sms,
            "message_timestamp_epoch_ms": 1786175400900,
        },
        "gold": {
            "type": "debit",
            "amount": {
                "decimal": gold_decimal,
                "span": _span(sms, f"INR {gold_decimal}"),
            },
            "account": {
                "value": "A/c XX8080",
                "span": _span(sms, "A/c XX8080"),
            },
            "counterparty": {
                "value": "PIXEL CANTEEN",
                "span": _span(sms, "PIXEL CANTEEN"),
            },
        },
    }
    component = {
        "id": "decimal-boundary",
        "prefilter": {"model_invoked": True, "rejection_stage": None},
        "candidates": {
            "amounts": [
                {
                    "id": "A0",
                    "decimal": candidate_decimal,
                    "span": _span(
                        sms,
                        f"INR {candidate_decimal}",
                        occurrence=int(candidate_decimal == gold_decimal),
                    ),
                }
            ],
            "accounts": [
                {
                    "id": "C0",
                    "value": "A/c XX8080",
                    "span": _span(sms, "A/c XX8080"),
                }
            ],
            "counterparties": [
                {
                    "id": "PT0",
                    "value": "PIXEL CANTEEN",
                    "span": _span(sms, "PIXEL CANTEEN"),
                },
                {"id": "PN", "value": None, "span": None},
            ],
            "type_hints": ["D"],
        },
        "parser": {
            "model_output": (
                '{"transaction":1,"type":"D","amount":"A0","account":"C0","counterparty":"PT0"}'
            )
        },
    }
    return annotation, component


def test_synthetic_fixture_reports_requested_aggregate_metrics_only() -> None:
    report = evaluate_jsonl(FIXTURE, contract_path=CONTRACT)

    assert report["rows"] == 8
    assert report["prefilter"] == {
        "gold_transactions": 5,
        "gold_nulls": 3,
        "transactions_invoked": 4,
        "transaction_recall": "0.800000",
        "false_rejection_count": 1,
        "nulls_rejected": 1,
        "null_rejection_rate": "0.333333",
        "model_invocations": 6,
        "model_invocation_rate": "0.750000",
        "rejection_counts_by_stage": {
            "personal_mobile_sender": 0,
            "currency_amount": 1,
            "masked_account_or_card": 0,
            "transaction_verb": 1,
            "otp": 0,
            "collect_or_mandate_request": 0,
        },
    }
    assert report["candidates"]["oracle_coverage"] == {
        "amount": {"count": 4, "total": 5, "rate": "0.800000"},
        "account": {"count": 4, "total": 5, "rate": "0.800000"},
        "counterparty": {"count": 4, "total": 5, "rate": "0.800000"},
        "joint": {"count": 2, "total": 5, "rate": "0.400000"},
    }
    assert report["candidates"]["exact_span_grounding"] == {
        "amount": {"count": 3, "total": 5, "rate": "0.600000"},
        "account": {"count": 4, "total": 5, "rate": "0.800000"},
        "counterparty": {"count": 4, "total": 5, "rate": "0.800000"},
        "joint": {"count": 1, "total": 5, "rate": "0.200000"},
    }
    assert report["parser"]["status_counts"] == {
        "transaction": 2,
        "not_transaction": 1,
        "rejected": 3,
    }
    assert report["parser"]["rejection_counts"] == {
        "duplicate_key": 1,
        "schema_mismatch": 1,
        "unknown_amount_id": 1,
    }
    for case in ("duplicate_key", "unknown_id", "reordered_members"):
        assert report["parser"]["strict_behavior"][case] == {
            "cases": 1,
            "rejected": 1,
            "correct_reason": 1,
            "rejection_rate": "1.000000",
        }
    assert report["parser"]["exact_reconstruction"] == {
        "count": 1,
        "total": 2,
        "rate": "0.500000",
    }
    assert report["parser"]["timestamp_preservation"] == {
        "count": 6,
        "total": 6,
        "rate": "1.000000",
    }
    assert report["pipeline"]["transaction_exact"] == 1
    assert report["pipeline"]["transaction_exact_rate"] == "0.200000"
    assert report["pipeline"]["whole_pipeline_exact"] == 3
    assert report["pipeline"]["whole_pipeline_exact_rate"] == "0.375000"
    assert report["pipeline"]["ghosts"] == 1
    assert report["pipeline"]["misses"] == 4
    assert {
        field: metric["rate"] for field, metric in report["pipeline"]["field_accuracy"].items()
    } == {
        "amount": "0.200000",
        "counterparty": "0.200000",
        "type": "0.200000",
        "account": "0.200000",
    }
    _assert_no_float(report)
    serialized = json.dumps(report, sort_keys=True)
    for private_key in ("sender", "sms", "id", "model_output", "source_span"):
        assert f'"{private_key}"' not in serialized


def test_decimal_comparison_never_collapses_adjacent_large_values() -> None:
    annotation, component = _decimal_pair(
        "9007199254740993.01",
        "9007199254740992.01",
    )

    report = evaluate_component_rows([annotation], [component])

    assert report["candidates"]["oracle_coverage"]["amount"] == {
        "count": 0,
        "total": 1,
        "rate": "0.000000",
    }
    assert report["pipeline"]["field_accuracy"]["amount"] == {
        "count": 0,
        "total": 1,
        "rate": "0.000000",
    }
    _assert_no_float(report)


def test_exact_reconstruction_preserves_decimal_text_precision() -> None:
    annotation, component = _decimal_pair("1.00", "1.0")

    report = evaluate_component_rows([annotation], [component])

    assert report["candidates"]["oracle_coverage"]["amount"]["count"] == 1
    assert report["candidates"]["exact_span_grounding"]["amount"]["count"] == 0
    assert report["parser"]["exact_reconstruction"]["count"] == 0
    assert report["pipeline"]["field_accuracy"]["amount"]["count"] == 0


def test_ids_must_be_unique_and_order_aligned() -> None:
    annotations, components = read_paired_jsonl(FIXTURE)
    misaligned = deepcopy(components)
    misaligned[0]["id"] = "different-id"
    with pytest.raises(ComponentEvaluationError, match="not order-aligned"):
        evaluate_component_rows(annotations, misaligned)

    duplicate_annotations = deepcopy(annotations)
    duplicate_components = deepcopy(components)
    duplicate_annotations[1]["id"] = duplicate_annotations[0]["id"]
    duplicate_components[1]["id"] = duplicate_components[0]["id"]
    with pytest.raises(ComponentEvaluationError, match="duplicates an earlier"):
        evaluate_component_rows(duplicate_annotations, duplicate_components)

    bad_candidate_id = deepcopy(components)
    bad_candidate_id[0]["candidates"]["amounts"][0]["id"] = "A1"
    with pytest.raises(ComponentEvaluationError, match="aligned with array order"):
        evaluate_component_rows(annotations, bad_candidate_id)


def test_source_spans_must_be_exact_utf8_byte_ranges() -> None:
    annotations, components = read_paired_jsonl(FIXTURE)
    forged = deepcopy(components)
    # The second fixture amount starts with the three-byte U+20B9 rupee sign.
    forged[1]["candidates"]["amounts"][0]["span"]["start_utf8_byte"] += 1

    with pytest.raises(ComponentEvaluationError, match="UTF-8 code-point boundaries"):
        evaluate_component_rows(annotations, forged)


def test_jsonl_reader_rejects_duplicate_members_without_echoing_row(
    tmp_path: Path,
) -> None:
    path = tmp_path / "duplicate.jsonl"
    secret = "INVENTED-DO-NOT-ECHO"
    path.write_text(
        '{"contract":{"name":"annotation_component_evaluation","version":1},'
        f'"annotation":{{"id":"one","id":"{secret}"}},'
        '"component_output":{}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ComponentEvaluationError) as raised:
        read_paired_jsonl(path)

    assert secret not in str(raised.value)


def test_contract_binding_is_versioned() -> None:
    contract = load_contract(CONTRACT)
    assert contract["contract_name"] == "annotation_component_evaluation"
    assert contract["contract_version"] == 1
    assert contract["offset_convention"] == "utf8_bytes"
    assert contract["rate_representation"] == "fixed_6_decimal_string_or_null"


def test_cli_dry_run_prints_aggregate_json_and_no_row_material(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["--input", str(FIXTURE), "--contract", str(CONTRACT), "--dry-run"]) == 0

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["rows"] == 8
    assert captured.err == ""
    assert "Fictional Aster Bank" not in captured.out
    assert "synthetic-001" not in captured.out
    assert '"amount":"A0"' not in captured.out


def test_cli_writes_only_aggregate_json_below_private_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    destination = tmp_path / "RESULTS" / "component-eval" / "metrics.json"

    assert (
        cli.main(
            [
                "--input",
                str(FIXTURE),
                "--contract",
                str(CONTRACT),
                "--output",
                str(destination),
            ]
        )
        == 0
    )

    report = json.loads(destination.read_text(encoding="utf-8"))
    assert report["rows"] == 8
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "aggregate_output_written": True,
        "rows": 8,
    }
    assert "Fictional" not in destination.read_text(encoding="utf-8")

    outside = tmp_path / "elsewhere" / "metrics.json"
    with pytest.raises(SystemExit) as raised:
        cli.main(
            [
                "--input",
                str(FIXTURE),
                "--contract",
                str(CONTRACT),
                "--output",
                str(outside),
            ]
        )
    assert raised.value.code == 2


def test_cli_resolves_symlinks_before_authorizing_output(
    tmp_path: Path,
) -> None:
    results = tmp_path / "RESULTS"
    outside = tmp_path / "outside"
    results.mkdir()
    outside.mkdir()
    (results / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ComponentEvaluationError, match="below PRIVATE_DATA or RESULTS"):
        cli._output_path(results / "escape" / "metrics.json", repo_root=tmp_path)


def _workbench_span(sms: str, text: str) -> dict[str, Any]:
    evaluator_span = _span(sms, text)
    return {
        "text": evaluator_span["text"],
        "start": evaluator_span["start_utf8_byte"],
        "end": evaluator_span["end_utf8_byte"],
    }


def test_workbench_adapter_preserves_decimal_utf8_spans_and_field_accuracy() -> None:
    decimal_text = "9007199254740993.01"
    sms = f"सूचना: INR {decimal_text} debited from A/c XX7788 to CAFÉ Ω."
    workbench = {
        "decision": "transaction",
        "amount_decimal": decimal_text,
        "amount_span": _workbench_span(sms, f"INR {decimal_text}"),
        "type": "debit",
        "account_span": _workbench_span(sms, "A/c XX7788"),
        "counterparty_span": _workbench_span(sms, "CAFÉ Ω"),
        "counterparty_absent": False,
        "notes": "invented adapter fixture",
        "uncertain": False,
    }

    adapted = adapt_workbench_annotation(
        workbench,
        row_id="adapter-transaction",
        sender="VK-INVENT",
        sms=sms,
        message_timestamp_epoch_ms=1786175400999,
    )

    gold = adapted["gold"]
    assert gold is not None
    assert gold["type"] == "debit"
    assert gold["amount"]["decimal"] == decimal_text
    assert gold["amount"]["span"] == {
        "text": f"INR {decimal_text}",
        "start_utf8_byte": workbench["amount_span"]["start"],
        "end_utf8_byte": workbench["amount_span"]["end"],
    }
    assert gold["amount"]["span"]["start_utf8_byte"] > sms.index("INR")
    assert gold["counterparty"]["value"] == "CAFÉ Ω"
    assert adapted["source"]["message_timestamp_epoch_ms"] == 1786175400999

    component = {
        "id": "adapter-transaction",
        "prefilter": {"model_invoked": True, "rejection_stage": None},
        "candidates": {
            "amounts": [
                {
                    "id": "A0",
                    "decimal": decimal_text,
                    "span": gold["amount"]["span"],
                }
            ],
            "accounts": [
                {
                    "id": "C0",
                    "value": gold["account"]["value"],
                    "span": gold["account"]["span"],
                }
            ],
            "counterparties": [
                {
                    "id": "PT0",
                    "value": gold["counterparty"]["value"],
                    "span": gold["counterparty"]["span"],
                },
                {"id": "PN", "value": None, "span": None},
            ],
            "type_hints": ["D"],
        },
        "parser": {
            "model_output": (
                '{"transaction":1,"type":"D","amount":"A0","account":"C0","counterparty":"PT0"}'
            )
        },
    }
    report = evaluate_component_rows([adapted], [component])
    assert report["pipeline"]["transaction_exact"] == 1
    assert all(metric["count"] == 1 for metric in report["pipeline"]["field_accuracy"].values())
    assert report["parser"]["timestamp_preservation"]["count"] == 1


def test_workbench_adapter_allows_grounded_account_without_candidate_identity() -> None:
    sms = "Invented INR 25.00 debited from Primary savings account at Demo Mart."
    workbench = {
        "decision": "transaction",
        "amount_decimal": "25.00",
        "amount_span": _workbench_span(sms, "INR 25.00"),
        "type": "debit",
        "account_span": _workbench_span(sms, "Primary savings account"),
        "counterparty_span": _workbench_span(sms, "Demo Mart"),
        "counterparty_absent": False,
        "notes": None,
        "uncertain": False,
    }

    adapted = adapt_workbench_annotation(
        workbench,
        row_id="adapter-unmasked-account",
        sender="VK-INVENT",
        sms=sms,
        message_timestamp_epoch_ms=None,
    )
    gold = adapted["gold"]
    assert gold is not None
    component = {
        "id": "adapter-unmasked-account",
        "prefilter": {"model_invoked": True, "rejection_stage": None},
        "candidates": {
            "amounts": [
                {
                    "id": "A0",
                    "decimal": "25.00",
                    "span": gold["amount"]["span"],
                }
            ],
            "accounts": [],
            "counterparties": [
                {
                    "id": "PT0",
                    "value": "Demo Mart",
                    "span": gold["counterparty"]["span"],
                },
                {"id": "PN", "value": None, "span": None},
            ],
            "type_hints": ["D"],
        },
        "parser": {"model_output": '{"transaction":0}'},
    }
    report = evaluate_component_rows([adapted], [component])
    coverage = report["candidates"]["oracle_coverage"]
    assert coverage["amount"]["count"] == 1
    assert coverage["account"]["count"] == 0
    assert coverage["counterparty"]["count"] == 1
    assert coverage["joint"]["count"] == 0


def test_workbench_adapter_preserves_null_and_rejects_incomplete_labels() -> None:
    sms = "नमस्ते – invented profile notice."
    workbench = {
        "decision": "not_transaction",
        "amount_decimal": None,
        "amount_span": None,
        "type": None,
        "account_span": None,
        "counterparty_span": None,
        "counterparty_absent": False,
        "notes": None,
        "uncertain": False,
    }

    adapted = adapt_workbench_annotation(
        workbench,
        row_id="adapter-null",
        sender="VK-INVENT",
        sms=sms,
        message_timestamp_epoch_ms=1786175401000,
    )

    assert adapted == {
        "id": "adapter-null",
        "source": {
            "sender": "VK-INVENT",
            "sms": sms,
            "message_timestamp_epoch_ms": 1786175401000,
        },
        "gold": None,
    }

    incomplete = dict(workbench)
    incomplete["decision"] = None
    with pytest.raises(ComponentEvaluationError, match="complete valid human label"):
        adapt_workbench_annotation(
            incomplete,
            row_id="adapter-invalid",
            sender="VK-INVENT",
            sms=sms,
            message_timestamp_epoch_ms=None,
        )
