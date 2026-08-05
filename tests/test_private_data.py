import json
from copy import deepcopy
from pathlib import Path

import pytest

from lfm25.private_data import (
    LABEL_FIELDS,
    RegressionIndex,
    accepted_training_label,
    aggregate_validation_report,
    file_sha256,
    normalize_template,
    prepare_private_dataset,
    read_source_rows,
    run_materialize,
    run_prepare,
    run_validate,
    safe_summary,
)


SYNTHETIC_KEY = b"synthetic-private-test-key-000000"


def _config(regression_rows: int = 1) -> dict:
    return {
        "schema_version": 1,
        "source": "all_sms.json",
        "allowed_source_names": ["all_sms.json", "all_sms.csv"],
        "regression_dataset": "DATA/extraction_ds.jsonl",
        "expected_regression_rows": regression_rows,
        "private_root": "PRIVATE_DATA/lfm25",
        "outputs": {
            "manifest": "split_manifest.jsonl",
            "review_queue": "review_queue.jsonl",
            "validation_report": "validation_report.json",
            "materialization_report": "materialization_report.json",
            "sft_prefix": "sft",
        },
        "source_fields": {
            "id": "id",
            "date": "date",
            "sender": "sender",
            "text": "text",
            "is_from_me": "is_from_me",
        },
        "label_fields": list(LABEL_FIELDS),
        "near_relative": {
            "threshold": 0.72,
            "character_ngram_size": 3,
            "minimum_template_characters": 12,
        },
        "splits": {
            "train": 0.34,
            "dev": 0.33,
            "test": 0.33,
            "rule_version": "synthetic-split-v1",
        },
        "consensus_policy": {
            "policy_version": "synthetic-consensus-v1",
            "eligible_splits": ["train", "dev"],
            "test_requires_human_review": True,
            "minimum_proposals": 3,
            "minimum_agreeing_models": 2,
            "minimum_independent_model_families": 2,
            "minimum_proposal_confidence": 0.9,
            "require_exact_four_fields": True,
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _label(amount: float = 12.5) -> dict:
    return {
        "amount": amount,
        "counterparty": "SYNTHETIC SHOP",
        "type": "debit",
        "account": "Card XX0000",
    }


def _proposal(model_id: str, family: str, label: dict | None, confidence: float = 0.97) -> dict:
    return {
        "model_id": model_id,
        "model_family": family,
        "label": label,
        "confidence": confidence,
        "inference_config_hash": f"config-{model_id}",
    }


def test_template_normalization_replaces_sensitive_families() -> None:
    message = (
        "₹1,250.00 debited from A/c XX1234 on 03-Aug-2026 at DEMO PERSON "
        "Ref ZXCV123456; user demo.user@example.test, phone 9876543210."
    )

    template = normalize_template(message)

    assert {"<AMOUNT>", "<ACCOUNT>", "<DATE>", "<REFERENCE>", "<PII>"} <= set(
        template.split()
    )
    for sensitive_value in (
        "1,250.00",
        "xx1234",
        "03-aug-2026",
        "zxcv123456",
        "demo.user@example.test",
        "9876543210",
    ):
        assert sensitive_value not in template.casefold()


def test_regression_index_has_exact_template_and_near_boundaries() -> None:
    regression = "INR 10.00 debited from A/c XX0001 at ALPHA LAB on 01/01/2025 Ref ABCD1000."
    template_relative = (
        "INR 99.00 debited from A/c XX9999 at BETA LAB on 02/02/2025 Ref WXYZ9999."
    )
    near_relative = (
        "Alert: INR 77.00 was debited from A/c XX7777 at GAMMA LAB on 03/03/2025 "
        "Ref LMNO7777."
    )
    index = RegressionIndex([regression], ngram_size=3)

    assert index.exclusion_reason(regression, normalize_template(regression), 0.72, 12) == "exact"
    assert (
        index.exclusion_reason(
            template_relative,
            normalize_template(template_relative),
            0.72,
            12,
        )
        == "normalized_template"
    )
    assert (
        index.exclusion_reason(near_relative, normalize_template(near_relative), 0.72, 12)
        == "near_relative"
    )


def test_preparation_is_immutable_decontaminated_and_split_before_silver_labeling(
    tmp_path: Path,
) -> None:
    regression_message = (
        "INR 10.00 debited from A/c XX0001 at ALPHA LAB on 01/01/2025 Ref ABCD1000."
    )
    source_rows = [
        {
            "id": 1,
            "date": "2025-01-01T00:00:00+00:00",
            "sender": "ZZ-SYNTH1",
            "text": regression_message,
        },
        {
            "id": 2,
            "date": "2025-01-02T00:00:00+00:00",
            "sender": "ZZ-SYNTH1",
            "text": (
                "INR 99.00 debited from A/c XX9999 at BETA LAB on 02/02/2025 "
                "Ref WXYZ9999."
            ),
        },
        {
            "id": 3,
            "date": "2025-01-03T00:00:00+00:00",
            "sender": "ZZ-SYNTH1",
            "text": (
                "Alert: INR 77.00 was debited from A/c XX7777 at GAMMA LAB on 03/03/2025 "
                "Ref LMNO7777."
            ),
        },
        {
            "id": 4,
            "date": "2025-02-01T00:00:00+00:00",
            "sender": "ZZ-SYNTH2",
            "text": "Synthetic library reminder: return the blue manual tomorrow.",
        },
        {
            "id": 5,
            "date": "2025-03-01T00:00:00+00:00",
            "sender": "ZZ-SYNTH3",
            "text": "Synthetic OTP 246810 is for signing in to the test sandbox.",
        },
        {
            "id": 6,
            "date": "2025-04-01T00:00:00+00:00",
            "sender": "ZZ-SYNTH4",
            "text": "Synthetic parcel notice: the demonstration parcel reached the lab.",
        },
    ]
    for row in source_rows:
        row["is_from_me"] = False
    source_rows.append(
        {
            "id": 7,
            "date": "2025-05-01T00:00:00+00:00",
            "sender": "ZZ-SYNTH-OUTGOING",
            "text": "Synthetic outgoing note that must never enter the private manifest.",
            "is_from_me": True,
        }
    )
    source_path = tmp_path / "all_sms.json"
    source_path.write_text(json.dumps(source_rows), encoding="utf-8")
    regression_path = tmp_path / "regression.jsonl"
    _write_jsonl(regression_path, [{"sender": "ZZ-REGRES", "sms": regression_message}])
    source_before = file_sha256(source_path)
    regression_before = file_sha256(regression_path)

    manifest, review_queue, report = prepare_private_dataset(
        source_path,
        regression_path,
        _config(),
        SYNTHETIC_KEY,
    )

    assert file_sha256(source_path) == source_before
    assert file_sha256(regression_path) == regression_before
    assert report["valid"] is True
    assert report["exclusions_by_reason"] == {
        "exact": 1,
        "near_relative": 1,
        "normalized_template": 1,
    }
    assert report["source_row_count"] == 7
    assert report["source_boundary_counts"] == {
        "total": 7,
        "incoming": 6,
        "outgoing_excluded": 1,
    }
    assert all(row["is_from_me"] is False for row in manifest)
    assert len(manifest) == len(review_queue) == 3
    assert {row["split"] for row in manifest} == {"train", "dev", "test"}
    assert all(row["label_tier"] == "silver" for row in manifest)
    assert all(row["split_provenance"]["assigned_before_labeling"] for row in manifest)
    assert all(
        {
            "source_id",
            "date",
            "sender",
            "template_group",
            "private_hashes",
            "provenance",
            "confidence",
            "review_status",
        }
        <= set(row)
        for row in manifest
    )
    test_rows = [row for row in manifest if row["split"] == "test"]
    assert test_rows and all(row["human_review_required"] for row in test_rows)
    assert all(row["review_status"] == "required" for row in test_rows)


def test_split_assignment_is_order_independent_for_template_groups(tmp_path: Path) -> None:
    regression_path = tmp_path / "regression.jsonl"
    _write_jsonl(
        regression_path,
        [{"sender": "ZZ-REGRES", "sms": "Unrelated synthetic regression fixture."}],
    )
    rows = [
        {
            "id": index,
            "date": f"2025-0{index}-01T00:00:00+00:00",
            "sender": f"ZZ-SYNTH{index}",
            "text": f"Synthetic category {word} notice for the offline fixture.",
        }
        for index, word in enumerate(("alpha", "bravo", "charlie", "delta"), start=1)
    ]
    for row in rows:
        row["is_from_me"] = False
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_source = first_dir / "all_sms.json"
    second_source = second_dir / "all_sms.json"
    first_source.write_text(json.dumps(rows), encoding="utf-8")
    second_source.write_text(json.dumps(list(reversed(rows))), encoding="utf-8")

    first_manifest, _, _ = prepare_private_dataset(
        first_source, regression_path, _config(), SYNTHETIC_KEY
    )
    second_manifest, _, _ = prepare_private_dataset(
        second_source, regression_path, _config(), SYNTHETIC_KEY
    )

    first_assignments = {row["source_id"]: row["split"] for row in first_manifest}
    second_assignments = {row["source_id"]: row["split"] for row in second_manifest}
    assert first_assignments == second_assignments


def test_materialization_accepts_consensus_silver_but_never_for_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(regression_rows=0)
    repo_root = tmp_path
    regression_path = repo_root / "DATA" / "extraction_ds.jsonl"
    _write_jsonl(regression_path, [])
    private_root = repo_root / "PRIVATE_DATA" / "lfm25"
    manifest_path = private_root / "split_manifest.jsonl"
    consensus_proposals = [
        _proposal("model-a", "family-a", _label()),
        _proposal("model-b", "family-b", _label()),
        _proposal("model-c", "family-c", _label(99.0)),
    ]
    base = {
        "record_hash": "record-a",
        "sender": "ZZ-SYNTH",
        "sms": "Synthetic INR 12.50 spent on Card XX0000 at SYNTHETIC SHOP.",
        "review_status": "pending",
        "human_label": None,
        "local_model_proposals": consensus_proposals,
    }
    train_silver = {**base, "split": "train"}
    test_silver = {**deepcopy(base), "record_hash": "record-b", "split": "test"}
    test_gold = {
        **deepcopy(base),
        "record_hash": "record-c",
        "split": "test",
        "review_status": "human_approved",
        "human_label": _label(),
        "human_reviewer": "synthetic-reviewer",
        "human_reviewed_at": "2026-08-03T00:00:00+00:00",
        "local_model_proposals": [],
    }
    pending_heuristic = {
        **deepcopy(base),
        "record_hash": "record-d",
        "split": "dev",
        "local_model_proposals": [],
    }
    _write_jsonl(manifest_path, [train_silver, test_silver, test_gold, pending_heuristic])
    monkeypatch.setattr("lfm25.private_data.is_git_ignored", lambda *_: True)

    accepted, _, tier = accepted_training_label(train_silver, config["consensus_policy"])
    assert (accepted, tier) == (True, "silver")
    assert accepted_training_label(test_silver, config["consensus_policy"])[0] is False

    report = run_materialize(repo_root, config)

    assert report["materialized_row_count"] == 2
    assert report["split_row_counts"] == {"train": 1, "dev": 0, "test": 1}
    train_row = json.loads((private_root / "sft_train.jsonl").read_text(encoding="utf-8"))
    test_row = json.loads((private_root / "sft_test.jsonl").read_text(encoding="utf-8"))
    for row in (train_row, test_row):
        assert set(row) == {"messages"}
        assert [message["role"] for message in row["messages"]] == [
            "system",
            "user",
            "assistant",
        ]
        completion = json.loads(row["messages"][-1]["content"])
        assert tuple(completion) == LABEL_FIELDS
        assert "date" not in completion
        assert "merchant" not in completion


def test_console_summary_cannot_include_private_row_values() -> None:
    report = {
        "valid": True,
        "source_row_count": 4,
        "sender": "ZZ-PRIVATE",
        "source_id": "secret-id",
        "sms": "secret message",
        "split_row_counts": {"train": 3, "dev": 1, "test": 0},
    }

    rendered = json.dumps(safe_summary(report))

    assert "ZZ-PRIVATE" not in rendered
    assert "secret-id" not in rendered
    assert "secret message" not in rendered


def test_csv_direction_flags_are_normalized_without_filtering(tmp_path: Path) -> None:
    source_path = tmp_path / "all_sms.csv"
    source_path.write_text(
        "id,date,sender,text,is_from_me\n"
        "1,2025-01-01T00:00:00+00:00,ZZ-IN,Synthetic incoming row,false\n"
        "2,2025-01-02T00:00:00+00:00,ZZ-OUT,Synthetic outgoing row,TRUE\n",
        encoding="utf-8",
    )

    rows, _ = read_source_rows(source_path, _config()["source_fields"])

    assert [row["is_from_me"] for row in rows] == [False, True]


def test_sender_component_fallback_is_reported_truthfully(tmp_path: Path) -> None:
    regression_path = tmp_path / "regression.jsonl"
    _write_jsonl(
        regression_path,
        [{"sender": "ZZ-REGRES", "sms": "Unrelated regression-only fixture."}],
    )
    rows = [
        {
            "id": index,
            "date": f"2025-0{index}-01T00:00:00+00:00",
            "sender": "ZZ-SHARED-SENDER",
            "text": f"Synthetic {word} category notice for a private split fixture.",
            "is_from_me": False,
        }
        for index, word in enumerate(("alpha", "bravo", "charlie"), start=1)
    ]
    source_path = tmp_path / "all_sms.json"
    source_path.write_text(json.dumps(rows), encoding="utf-8")

    manifest, _, report = prepare_private_dataset(
        source_path,
        regression_path,
        _config(),
        SYNTHETIC_KEY,
    )

    assert report["valid"] is True
    assert report["sender_component_lock_applied"] is False
    assert report["sender_split_diagnostics"]["cross_split_sender_count"] == 1
    assert report["sender_split_diagnostics"]["sender_held_out"] is False
    assert {row["split"] for row in manifest} == {"train", "dev", "test"}
    assert all(
        row["split_provenance"]["sender_component_locked"] is False
        for row in manifest
    )

    outgoing_manifest = deepcopy(manifest)
    outgoing_manifest[0]["is_from_me"] = True
    outgoing_report = aggregate_validation_report(outgoing_manifest)
    assert outgoing_report["valid"] is False
    assert outgoing_report["error_counts"]["outgoing_or_unknown_source_direction"] == 1
    assert outgoing_report["invariants"]["outgoing_rows_present"] is True


def test_run_validate_recomputes_durable_preparation_aggregates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    source_rows = [
        {
            "id": index,
            "date": f"2025-0{index}-01T00:00:00+00:00",
            "sender": f"ZZ-SYNTH-{index}",
            "text": f"Synthetic {word} archive notice for durable validation.",
            "is_from_me": False,
        }
        for index, word in enumerate(("alpha", "bravo", "charlie", "delta"), start=1)
    ]
    source_rows.append(
        {
            "id": 5,
            "date": "2025-05-01T00:00:00+00:00",
            "sender": "ZZ-OUTGOING",
            "text": "Synthetic outgoing archive notice.",
            "is_from_me": True,
        }
    )
    (tmp_path / "all_sms.json").write_text(json.dumps(source_rows), encoding="utf-8")
    _write_jsonl(
        tmp_path / "DATA" / "extraction_ds.jsonl",
        [{"sender": "ZZ-REGRES", "sms": "Unrelated regression-only fixture."}],
    )
    monkeypatch.setattr("lfm25.private_data.is_git_ignored", lambda *_: True)

    prepared = run_prepare(tmp_path, config)
    validated = run_validate(tmp_path, config)

    assert prepared["valid"] is validated["valid"] is True
    assert validated["source_boundary_counts"] == {
        "total": 5,
        "incoming": 4,
        "outgoing_excluded": 1,
    }
    assert validated["eligible_row_count"] == 4
    assert validated["excluded_row_count"] == 0
    assert validated["exclusions_by_reason"] == {}
    assert validated["split_group_counts"] == prepared["split_group_counts"]
    assert validated["split_assignment_unit_counts"] == prepared[
        "split_assignment_unit_counts"
    ]
    assert validated["invariants"]["manifest_membership_mismatch"] is False
    assert validated["invariants"]["preparation_provenance_mismatch"] is False
    persisted = json.loads(
        (tmp_path / "PRIVATE_DATA" / "lfm25" / "validation_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted == validated
