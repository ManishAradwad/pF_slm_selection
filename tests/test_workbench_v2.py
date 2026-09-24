from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from lfm25.workbench_v2 import (
    SYNTHETIC_FIXTURE_PATH,
    WorkbenchV2Error,
    anonymize_annotation_package,
    create_annotation_package,
    read_annotation_package,
    resolved_annotation_rows,
    validate_annotation_package,
    write_aggregate_report,
    write_private_artifact,
)


def _fixture() -> dict:
    value = json.loads(SYNTHETIC_FIXTURE_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_invented_package_validates_split_adjudication_and_semantic_records() -> None:
    package = _fixture()
    summary = validate_annotation_package(package)

    assert summary["privacy_classification"] == "invented_synthetic_only"
    assert summary["counts"] == {"rows": 4, "pending": 0, "resolved": 4, "excluded": 0}
    assert summary["split_counts"] == {"train": 0, "development": 0, "protected_test": 4}
    assert len(summary["package_sha256"]) == 64
    rows = resolved_annotation_rows(package, split="protected_test")
    assert [row.row_id for row in rows] == [
        "synth_eligible_debit",
        "synth_not_posted",
        "synth_fault_injection",
        "synth_non_inr",
    ]
    assert all(row.message for row in rows)


def test_empty_private_package_can_be_created_without_reading_source_data() -> None:
    package = create_annotation_package(
        package_id="local_workspace_v2",
        created_at="2026-08-19T00:00:00Z",
        annotation_policy_id="human_gold_v2",
        annotation_policy_version=1,
        source_revision="not_materialized",
        source_digest_sha256="0" * 64,
    )

    summary = validate_annotation_package(package)
    assert summary["counts"]["rows"] == 0
    assert summary["privacy_classification"] == "private_local_only"


def test_sender_or_template_group_cannot_cross_split() -> None:
    package = _fixture()
    package["rows"][3]["split"] = "development"
    package["rows"][3]["groups"]["sender_family"] = package["rows"][0]["groups"][
        "sender_family"
    ]

    with pytest.raises(WorkbenchV2Error, match="crosses split"):
        validate_annotation_package(package)


def test_resolved_row_requires_independent_adjudicator() -> None:
    package = _fixture()
    package["rows"][0]["adjudication"]["adjudicator_id"] = "synthetic_reviewer_a"

    with pytest.raises(WorkbenchV2Error, match="adjudicator must be independent"):
        validate_annotation_package(package)


def test_semantic_evidence_failure_is_rejected_without_exposing_message() -> None:
    package = _fixture()
    package["rows"][0]["annotations"][0]["semantic_record"]["events"][0]["account"][
        "evidence"
    ]["end_utf8_byte"] = 9999

    with pytest.raises(ValueError, match="evidence") as captured:
        validate_annotation_package(package)
    assert "Orbit Bank" not in str(captured.value)
    assert "XX1234" not in str(captured.value)


def test_anonymization_removes_raw_values_and_uses_export_scoped_ids() -> None:
    package = _fixture()
    first = anonymize_annotation_package(
        package,
        secret=b"s" * 32,
        export_nonce="synthetic_export_nonce_001",
    )
    second = anonymize_annotation_package(
        package,
        secret=b"s" * 32,
        export_nonce="synthetic_export_nonce_002",
    )

    rendered = json.dumps(first, sort_keys=True)
    assert "Orbit Bank" not in rendered
    assert "MAPLE BOOKS" not in rendered
    assert "XX1234" not in rendered
    assert "1250.50" not in rendered
    assert "received_at_epoch_ms" not in rendered
    assert "sender_family" not in rendered
    assert first["privacy"]["publication_approved"] is False
    assert all(len(row["anonymous_row_id"]) == 64 for row in first["rows"])
    assert first["rows"][0]["anonymous_row_id"] != second["rows"][0]["anonymous_row_id"]


def test_filesystem_guards_allow_only_private_or_exact_synthetic_inputs(tmp_path: Path) -> None:
    package = read_annotation_package(SYNTHETIC_FIXTURE_PATH, allow_synthetic_fixture=True)
    assert validate_annotation_package(package)["counts"]["rows"] == 4

    outside = tmp_path / "annotation.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(WorkbenchV2Error, match="PRIVATE_DATA"):
        read_annotation_package(outside)
    with pytest.raises(WorkbenchV2Error, match="PRIVATE_DATA"):
        write_private_artifact(outside, deepcopy(package))
    with pytest.raises(WorkbenchV2Error, match="RESULTS"):
        write_aggregate_report(outside, {"counts": {"rows": 4}})


def test_aggregate_writer_rejects_row_level_fields_before_path_checks(tmp_path: Path) -> None:
    with pytest.raises(WorkbenchV2Error, match="row-level"):
        write_aggregate_report(tmp_path / "report.json", {"rows": [{"row_id": "hidden"}]})
