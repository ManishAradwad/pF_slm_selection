from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path

import pytest

from lfm25.annotation_sources import (
    load_blinded_workspace,
    load_training_workspace,
    public_row,
    qc_requirements,
    source_prefill_for_sms,
    training_proposals,
)
from lfm25.annotation_workbench import (
    ANNOTATION_KEYS,
    BLINDED_MODE,
    SOURCE_PREFILL_UNAMBIGUOUS,
    TRAINING_MODE,
    AnnotationValidationError,
    WorkbenchError,
    annotation_to_legacy_fields,
    annotations_equal,
    empty_annotation,
    exact_json_dumps,
    exact_json_loads,
    validate_annotation,
)
from lfm25.blinded_review import run_export


def _span(sms: str, source_text: str) -> dict[str, object]:
    character_start = sms.index(source_text)
    character_end = character_start + len(source_text)
    return {
        "text": source_text,
        "start": len(sms[:character_start].encode("utf-8")),
        "end": len(sms[:character_end].encode("utf-8")),
    }


def _transaction_annotation(sms: str) -> dict[str, object]:
    value = empty_annotation()
    value.update(
        {
            "decision": "transaction",
            "amount_decimal": "1234.50",
            "amount_span": _span(sms, "INR 1,234.50"),
            "type": "debit",
            "account_span": _span(sms, "A/c XX0042"),
            "counterparty_span": _span(sms, "CAFÉ NILA"),
            "counterparty_absent": False,
            "notes": "Invented fixture note.",
        }
    )
    return value


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_handbook(repo: Path, text: str = "# Invented Annotation Handbook V1\n") -> Path:
    handbook = repo / "docs/guides/ANNOTATION_HANDBOOK_V1.md"
    handbook.parent.mkdir(parents=True, exist_ok=True)
    handbook.write_text(text, encoding="utf-8")
    return handbook


def _manifest_row(
    row_id: str,
    split: str,
    *,
    sms: str,
    template: str | None = None,
    sender_group: str | None = None,
) -> dict:
    return {
        "schema_version": 2,
        "record_hash": row_id,
        "split": split,
        "sender": "ZZ-SYNTH",
        "sms": sms,
        "template_group": template or f"template-{row_id}",
        "private_hashes": {"sender": sender_group or f"sender-{row_id}"},
        "review_status": "required" if split == "test" else "pending",
        "human_review_required": split == "test",
        "human_label": None,
        "human_reviewer": None,
        "human_reviewed_at": None,
        "silver_label": None,
        "confidence": 0.9,
        "heuristic_reason_codes": [],
        "local_model_proposals": [],
    }


def test_source_prefill_is_off_by_default_and_prefilter_gated_without_leaks() -> None:
    accepted_sms = (
        "Synthetic Demo Bank: INR 42.50 was debited from A/c XX1234 "
        "at Paper Kite Cafe."
    )
    rejected_sms = "Your invented verification code is 654321. Do not share it."

    assert source_prefill_for_sms("SYNTH-DEMO", accepted_sms) is None
    assert (
        source_prefill_for_sms(
            "SYNTH-SECURITY",
            rejected_sms,
            source_prefill=SOURCE_PREFILL_UNAMBIGUOUS,
        )
        is None
    )

    suggestion = source_prefill_for_sms(
        "SYNTH-DEMO",
        accepted_sms,
        source_prefill=SOURCE_PREFILL_UNAMBIGUOUS,
    )
    assert suggestion is not None
    assert set(suggestion).issubset(
        {
            "policy_version",
            "amount_decimal",
            "amount_span",
            "type",
            "account_span",
            "counterparty_span",
        }
    )
    assert not (
        {
            "decision",
            "counterparty_absent",
            "accepted",
            "reason",
            "confidence",
            "candidate_id",
            "candidate_count",
        }
        & set(suggestion)
    )


def test_exact_decimal_and_utf8_spans_round_trip_without_float() -> None:
    sms = "Alert: INR 1,234.50 debited from A/c XX0042 at CAFÉ NILA."
    annotation = _transaction_annotation(sms)
    reordered = {key: annotation[key] for key in reversed(ANNOTATION_KEYS)}

    validated = validate_annotation(reordered, sms, require_complete=True)
    projected = annotation_to_legacy_fields(validated)
    encoded = exact_json_dumps(projected)

    assert projected["amount"] == Decimal("1234.50")
    assert '"amount":1234.50' in encoded
    assert exact_json_loads(encoded)["amount"] == Decimal("1234.50")
    assert validated["counterparty_span"]["text"] == "CAFÉ NILA"
    assert annotations_equal(annotation, reordered)

def test_amount_grounding_preserves_written_fractional_digits() -> None:
    sms = "Alert: INR 1 debited from A/c XX0042 at Demo Mart."
    annotation = empty_annotation()
    annotation.update(
        {
            "decision": "transaction",
            "amount_decimal": "1",
            "amount_span": _span(sms, "INR 1"),
            "type": "debit",
            "account_span": _span(sms, "A/c XX0042"),
            "counterparty_span": _span(sms, "Demo Mart"),
            "counterparty_absent": False,
        }
    )

    assert validate_annotation(annotation, sms, require_complete=True)[
        "amount_decimal"
    ] == "1"

    annotation["amount_decimal"] = "1.00"
    with pytest.raises(AnnotationValidationError) as captured:
        validate_annotation(annotation, sms, require_complete=True)
    assert any(
        problem.field == "amount_span" and problem.code == "amount_mismatch"
        for problem in captured.value.problems
    )



@pytest.mark.parametrize(
    ("update", "field", "code"),
    (
        ({"amount_decimal": "0"}, "amount_decimal", "invalid"),
        ({"amount_decimal": "1234.51"}, "amount_span", "amount_mismatch"),
        ({"account_span": {"text": "OTHER", "start": 0, "end": 5}}, "account_span", "not_grounded"),
        ({"counterparty_span": None, "counterparty_absent": False}, "counterparty_span", "required"),
        ({"counterparty_absent": True}, "counterparty_span", "exclusive"),
        ({"uncertain": True}, "uncertain", "unresolved"),
    ),
)
def test_complete_transaction_validation_is_actionable(
    update: dict[str, object], field: str, code: str
) -> None:
    sms = "Alert: INR 1,234.50 debited from A/c XX0042 at CAFÉ NILA."
    annotation = _transaction_annotation(sms)
    annotation.update(update)

    with pytest.raises(AnnotationValidationError) as captured:
        validate_annotation(annotation, sms, require_complete=True)

    assert any(item.field == field and item.code == code for item in captured.value.problems)


def test_not_transaction_requires_empty_extractions_and_pending_draft_is_allowed() -> None:
    sms = "Invented bill reminder for INR 400."
    value = empty_annotation()
    value["decision"] = "not_transaction"
    assert validate_annotation(value, sms, require_complete=True)["decision"] == "not_transaction"

    value["amount_decimal"] = "400"
    with pytest.raises(AnnotationValidationError, match="validation errors"):
        validate_annotation(value, sms, require_complete=True)

    draft = empty_annotation()
    draft["notes"] = "Need handbook check."
    assert validate_annotation(draft, sms, require_complete=False)["notes"] == (
        "Need handbook check."
    )


def test_blinded_loader_exposes_only_reviewer_fields_and_binds_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    handbook = _write_handbook(repo)
    manifest = repo / "PRIVATE_DATA/lfm25/split_manifest.jsonl"
    rows = [
        _manifest_row("train-a", "train", sms="Invented training reminder."),
        _manifest_row(
            "test-a",
            "test",
            sms="Invented INR 7.25 debited from A/c XX0007.",
        ),
    ]
    _write_jsonl(manifest, rows)
    monkeypatch.setattr("lfm25.blinded_review.require_private_ignore", lambda *_args: None)
    run_export(repo)

    workspace = load_blinded_workspace(repo)
    row = workspace.rows[0]
    dto = public_row(
        row.store_dict(),
        {"status": "pending", "revision": 0, "annotation": None},
        mode=BLINDED_MODE,
        total_rows=workspace.row_count,
    )

    assert workspace.row_count == 1
    assert workspace.binding["row_count"] == 1
    assert workspace.binding["handbook_version"] == "v1"
    first_handbook_hash = workspace.binding["handbook_sha256"]
    assert set(dto) == {
        "review_id",
        "position",
        "total_rows",
        "sender",
        "sms",
        "status",
        "revision",
        "annotation",
        "qc_required",
        "qc_status",
    }
    forbidden = "silver_label confidence record_hash template_group local_model_proposals"
    assert all(name not in json.dumps(dto) for name in forbidden.split())
    with pytest.raises(WorkbenchError, match="unavailable"):
        training_proposals("{}", mode=BLINDED_MODE)
    handbook.write_text("# Changed invented handbook\n", encoding="utf-8")
    assert load_blinded_workspace(repo).binding["handbook_sha256"] != first_handbook_hash


def test_training_loader_rejects_test_overlap_and_reveals_proposals_only_explicitly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _write_handbook(repo)
    private = repo / "PRIVATE_DATA/lfm25"
    sealed = private / "split_manifest.jsonl"
    test = _manifest_row(
        "test-a",
        "test",
        sms="Invented INR 8 credited to A/c XX0008.",
        template="sealed-template",
        sender_group="sealed-sender",
    )
    train = _manifest_row(
        "train-a",
        "train",
        sms="OTP 123456 for invented purchase of INR 10.",
    )
    train["confidence"] = 0.4
    train["local_model_proposals"] = [
        {
            "model_id": "local-fixture-model",
            "model_family": "fixture-family",
            "label": None,
            "confidence": 0.4,
            "schema_valid": True,
        }
    ]
    _write_jsonl(sealed, [train, test])
    pool = private / "explicit_train_pool.jsonl"
    _write_jsonl(pool, [train])
    monkeypatch.setattr("lfm25.annotation_sources.require_private_ignore", lambda *_args: None)

    workspace = load_training_workspace(repo, pool_file=pool)
    assert workspace.row_count == 1
    assert workspace.binding["active_learning_queue_policy_version"] == 1
    assert workspace.binding["active_learning_queue_tie_break"] == "original_pool_position"
    assert workspace.binding["active_learning_queue_priority"][0] == "model_disagreement"
    assert "otp_or_security" in workspace.rows[0].queue_tags
    assert "low_confidence_output" in workspace.rows[0].queue_tags
    revealed = training_proposals(workspace.rows[0].source_json or "", mode=TRAINING_MODE)
    assert revealed["model_proposals"][0]["model_id"] == "local-fixture-model"

    overlapping = dict(train)
    overlapping["record_hash"] = "train-overlap"
    overlapping["template_group"] = "sealed-template"
    bad_pool = private / "overlapping_train_pool.jsonl"
    _write_jsonl(bad_pool, [overlapping])
    with pytest.raises(WorkbenchError, match="sealed test"):
        load_training_workspace(repo, pool_file=bad_pool)

    for field, bad_value in (("sender", ""), ("sms", None), ("template_group", "")):
        malformed_test = dict(test)
        malformed_test[field] = bad_value
        _write_jsonl(sealed, [train, malformed_test])
        with pytest.raises(WorkbenchError, match="invalid split identity"):
            load_training_workspace(repo, pool_file=pool)


def test_training_loader_rejects_cross_split_sender_or_template_leakage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _write_handbook(repo)
    private = repo / "PRIVATE_DATA/lfm25"
    sealed = private / "split_manifest.jsonl"
    test = _manifest_row("test-a", "test", sms="Invented security alert.")
    train = _manifest_row(
        "train-a",
        "train",
        sms="Invented INR 5 debited from A/c XX0005.",
        sender_group="shared-sender",
    )
    dev = _manifest_row(
        "dev-a",
        "dev",
        sms="Invented INR 6 credited to A/c XX0006.",
        sender_group="shared-sender",
    )
    _write_jsonl(sealed, [train, dev, test])
    pool = private / "explicit_pool.jsonl"
    _write_jsonl(pool, [train, dev])
    monkeypatch.setattr("lfm25.annotation_sources.require_private_ignore", lambda *_args: None)

    with pytest.raises(WorkbenchError, match="split isolation"):
        load_training_workspace(repo, pool_file=pool)


def test_qc_queue_covers_transactions_notes_uncertainty_and_ten_percent_nulls() -> None:
    rows = []
    for index in range(21):
        annotation = empty_annotation()
        annotation["decision"] = "transaction" if index == 0 else "not_transaction"
        if index == 1:
            annotation["notes"] = "Invented note."
        rows.append(
            {
                "row_id": f"row-{index:02d}",
                "annotation": annotation,
                "ever_uncertain": index == 2,
            }
        )

    first = qc_requirements(rows, deterministic_seed="synthetic-seed")
    second = qc_requirements(rows, deterministic_seed="synthetic-seed")
    sampled = [
        row_id
        for row_id, reasons in first.items()
        if "deterministic_null_sample_10pct" in reasons
    ]

    assert first == second
    assert "transaction_second_pass" in first["row-00"]
    assert "noted_second_pass" in first["row-01"]
    assert "uncertain_second_pass" in first["row-02"]
    assert len(sampled) == 2  # ceil(10% of 20 final nulls)
