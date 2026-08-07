from __future__ import annotations

import json
from pathlib import Path

import pytest

import lfm25.blinded_review as blinded_review
import scripts.review_lfm25_blinded_test as review_cli
from lfm25.blinded_review import (
    REVIEW_FIELDS,
    BlindedReviewError,
    resolve_review_paths,
    run_export,
    run_import,
    run_validate,
    safe_console_summary,
)
from lfm25.private_data import PrivateDataError, file_sha256, read_jsonl


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _source_row(index: int, split: str, record_hash: str) -> dict:
    return {
        "schema_version": 2,
        "record_hash": record_hash,
        "split": split,
        "sender": f"ZZ-SYNTH-{index}",
        "sms": f"Synthetic INR {index}.50 debited from A/c XX{index:04d}.",
        "silver_label": {
            "amount": index + 0.5,
            "counterparty": "SYNTHETIC SHOP",
            "type": "debit",
            "account": f"A/c XX{index:04d}",
        },
        "heuristic_reason_codes": [f"private-heuristic-{index}"],
        "confidence": 0.91,
        "hard_negative_category": f"private-negative-{index}",
        "template_group": f"private-template-{index}",
        "private_hashes": {"sender": f"private-sender-hash-{index}"},
        "local_model_proposals": [{"model_id": f"private-proposal-{index}"}],
        "review_status": "required" if split == "test" else "pending",
        "human_review_required": split == "test",
        "human_label": None,
        "human_reviewer": None,
        "human_reviewed_at": None,
    }


def _setup_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, list[dict]]:
    repo_root = tmp_path / "repo"
    manifest = repo_root / "PRIVATE_DATA" / "lfm25" / "split_manifest.jsonl"
    rows = [
        _source_row(90, "train", "record-train"),
        _source_row(2, "test", "record-z"),
        _source_row(1, "test", "record-a"),
        _source_row(91, "dev", "record-dev"),
    ]
    _write_jsonl(manifest, rows)
    monkeypatch.setattr("lfm25.blinded_review.require_private_ignore", lambda *_args: None)
    return repo_root, manifest, rows


def _review_path(repo_root: Path) -> Path:
    return repo_root / "PRIVATE_DATA" / "lfm25" / "blinded_test_review.jsonl"


def test_export_is_test_only_blinded_deterministic_and_non_overwriting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root, manifest, _ = _setup_repo(tmp_path, monkeypatch)
    source_before = manifest.read_bytes()

    report = run_export(repo_root)

    review_file = _review_path(repo_root)
    mapping_file = (
        repo_root
        / "PRIVATE_DATA"
        / "lfm25"
        / "blinded_test_review_internal_map.jsonl"
    )
    metadata_file = (
        repo_root / "PRIVATE_DATA" / "lfm25" / "blinded_test_review_metadata.json"
    )
    review_rows = read_jsonl(review_file)
    assert report["test_rows"] == 2
    assert manifest.read_bytes() == source_before
    assert [row["review_id"] for row in review_rows] == ["test-000001", "test-000002"]
    assert [row["sender"] for row in review_rows] == ["ZZ-SYNTH-1", "ZZ-SYNTH-2"]
    assert all(tuple(row) == REVIEW_FIELDS for row in review_rows)
    assert all(row[field] is None for row in review_rows for field in REVIEW_FIELDS[3:])
    forbidden_fields = {
        "silver_label",
        "heuristic_reason_codes",
        "confidence",
        "hard_negative_category",
        "template_group",
        "private_hashes",
        "local_model_proposals",
        "record_hash",
    }
    assert all(forbidden_fields.isdisjoint(row) for row in review_rows)
    mapping_rows = read_jsonl(mapping_file)
    assert mapping_rows == [
        {"review_id": "test-000001", "record_hash": "record-a"},
        {"review_id": "test-000002", "record_hash": "record-z"},
    ]
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    assert metadata["test_row_count"] == 2
    assert metadata["source_manifest_sha256"] == file_sha256(manifest)

    frozen_bytes = {
        "review": review_file.read_bytes(),
        "mapping": mapping_file.read_bytes(),
        "metadata": metadata_file.read_bytes(),
    }
    with pytest.raises(BlindedReviewError, match="nonempty"):
        run_export(repo_root)
    run_export(repo_root, force=True)
    assert review_file.read_bytes() == frozen_bytes["review"]
    assert mapping_file.read_bytes() == frozen_bytes["mapping"]
    assert metadata_file.read_bytes() == frozen_bytes["metadata"]


def test_partial_review_validates_and_imports_without_mutating_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root, manifest, source_rows = _setup_repo(tmp_path, monkeypatch)
    run_export(repo_root)
    source_hash = file_sha256(manifest)
    review_file = _review_path(repo_root)
    review_rows = read_jsonl(review_file)
    review_rows[0].update(
        {
            "decision": "transaction",
            "amount": 1.5,
            "counterparty": "SYNTHETIC SHOP",
            "type": "debit",
            "account": "A/c XX0001",
            "reviewer": "fixture-reviewer",
            "reviewed_at": "2026-08-05T12:30:00+05:30",
            "notes": "synthetic fixture annotation",
        }
    )
    _write_jsonl(review_file, review_rows)

    validation = run_validate(repo_root)
    assert validation["completed_rows"] == 1
    assert validation["pending_rows"] == 1
    assert validation["ready_for_evaluation"] is False
    summary = safe_console_summary(validation)
    assert "reviewer" not in json.dumps(summary)
    assert "sms" not in json.dumps(summary).casefold()

    imported = run_import(repo_root)
    assert imported["completed_rows"] == 1
    assert imported["source_manifest_unchanged"] is True
    assert file_sha256(manifest) == source_hash
    reviewed_path = (
        repo_root / "PRIVATE_DATA" / "lfm25" / "split_manifest_human_reviewed.jsonl"
    )
    reviewed = read_jsonl(reviewed_path)
    assert len(reviewed) == len(source_rows)
    by_hash = {row["record_hash"]: row for row in reviewed}
    assert by_hash["record-a"]["review_status"] == "human_approved"
    assert by_hash["record-a"]["human_label"] == {
        "amount": 1.5,
        "counterparty": "SYNTHETIC SHOP",
        "type": "debit",
        "account": "A/c XX0001",
    }
    original_pending = next(row for row in source_rows if row["record_hash"] == "record-z")
    assert by_hash["record-z"] == original_pending
    report_path = (
        repo_root
        / "PRIVATE_DATA"
        / "lfm25"
        / "blinded_test_review_import_report.json"
    )
    report_text = report_path.read_text(encoding="utf-8")
    assert "Synthetic INR" not in report_text
    assert "fixture-reviewer" not in report_text

    with pytest.raises(BlindedReviewError, match="nonempty"):
        run_import(repo_root)
    run_import(repo_root, force=True)


def test_completed_not_transaction_uses_an_approved_null_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root, _, _ = _setup_repo(tmp_path, monkeypatch)
    run_export(repo_root)
    review_file = _review_path(repo_root)
    rows = read_jsonl(review_file)
    for row in rows:
        row.update(
            {
                "decision": "not_transaction",
                "reviewer": "fixture-reviewer",
                "reviewed_at": "2026-08-05T12:30:00Z",
                "notes": None,
            }
        )
    _write_jsonl(review_file, rows)

    validation = run_validate(repo_root)
    assert validation["ready_for_evaluation"] is True
    assert validation["not_transaction_rows"] == 2
    run_import(repo_root)
    reviewed = read_jsonl(
        repo_root / "PRIVATE_DATA" / "lfm25" / "split_manifest_human_reviewed.jsonl"
    )
    test_rows = [row for row in reviewed if row["split"] == "test"]
    assert all(row["review_status"] == "human_approved" for row in test_rows)
    assert all(row["human_label"] is None for row in test_rows)


@pytest.mark.parametrize(
    ("case", "error"),
    (
        ("duplicate", "duplicate review IDs"),
        ("missing", "missing or unknown IDs"),
        ("schema", "invalid schema"),
        ("partial", "partial annotation"),
        ("invalid_label", "invalid canonical transaction label"),
        ("altered_source", "source fields do not match"),
    ),
)
def test_validation_fails_closed_on_invalid_review_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    error: str,
) -> None:
    repo_root, _, _ = _setup_repo(tmp_path, monkeypatch)
    run_export(repo_root)
    review_file = _review_path(repo_root)
    rows = read_jsonl(review_file)
    if case == "duplicate":
        rows.append(dict(rows[0]))
    elif case == "missing":
        rows.pop()
    elif case == "schema":
        rows[0].pop("notes")
    elif case == "partial":
        rows[0]["reviewer"] = "fixture-reviewer"
    elif case == "invalid_label":
        rows[0].update(
            {
                "decision": "transaction",
                "amount": "1.50",
                "counterparty": None,
                "type": "debit",
                "account": "A/c XX0001",
                "reviewer": "fixture-reviewer",
                "reviewed_at": "2026-08-05T12:30:00Z",
            }
        )
    else:
        rows[0]["sms"] = "Altered synthetic source field"
    _write_jsonl(review_file, rows)

    with pytest.raises(BlindedReviewError, match=error):
        run_validate(repo_root)


def test_validation_rejects_stale_source_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root, manifest, source_rows = _setup_repo(tmp_path, monkeypatch)
    run_export(repo_root)
    source_rows.append(_source_row(92, "train", "record-new"))
    _write_jsonl(manifest, source_rows)

    with pytest.raises(BlindedReviewError, match="changed after blinded export"):
        run_validate(repo_root)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("review_id_format", "tampered-format", "ID format metadata"),
        ("review_template_sha256", "0" * 64, "template provenance"),
    ),
)
def test_validation_rejects_tampered_frozen_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    error: str,
) -> None:
    repo_root, _, _ = _setup_repo(tmp_path, monkeypatch)
    run_export(repo_root)
    metadata_path = (
        repo_root / "PRIVATE_DATA" / "lfm25" / "blinded_test_review_metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(BlindedReviewError, match=error):
        run_validate(repo_root)


def test_export_source_race_does_not_publish_package_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root, manifest, _ = _setup_repo(tmp_path, monkeypatch)
    private_root = repo_root / "PRIVATE_DATA" / "lfm25"
    review_path = private_root / "blinded_test_review.jsonl"
    real_file_sha256 = blinded_review.file_sha256

    def unstable_source_hash(path: Path) -> str:
        observed = real_file_sha256(path)
        if path.resolve() == manifest.resolve() and review_path.exists():
            return "0" * 64
        return observed

    monkeypatch.setattr(blinded_review, "file_sha256", unstable_source_hash)
    with pytest.raises(BlindedReviewError, match="changed during export"):
        run_export(repo_root)

    assert not (private_root / "blinded_test_review.jsonl").exists()
    assert not (private_root / "blinded_test_review_internal_map.jsonl").exists()
    assert not (private_root / "blinded_test_review_metadata.json").exists()


def test_forced_export_source_race_restores_previous_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root, manifest, _ = _setup_repo(tmp_path, monkeypatch)
    run_export(repo_root)
    private_root = repo_root / "PRIVATE_DATA" / "lfm25"
    package_paths = (
        private_root / "blinded_test_review.jsonl",
        private_root / "blinded_test_review_internal_map.jsonl",
        private_root / "blinded_test_review_metadata.json",
    )
    previous = {path: path.read_bytes() for path in package_paths}
    real_file_sha256 = blinded_review.file_sha256
    source_hash_calls = 0

    def unstable_source_hash(path: Path) -> str:
        nonlocal source_hash_calls
        observed = real_file_sha256(path)
        if path.resolve() == manifest.resolve():
            source_hash_calls += 1
            if source_hash_calls >= 4:
                return "0" * 64
        return observed

    monkeypatch.setattr(blinded_review, "file_sha256", unstable_source_hash)
    with pytest.raises(BlindedReviewError, match="changed during export"):
        run_export(repo_root, force=True)

    assert {path: path.read_bytes() for path in package_paths} == previous


def test_import_source_race_rolls_back_reviewed_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root, manifest, _ = _setup_repo(tmp_path, monkeypatch)
    run_export(repo_root)
    private_root = repo_root / "PRIVATE_DATA" / "lfm25"
    reviewed_path = private_root / "split_manifest_human_reviewed.jsonl"
    report_path = private_root / "blinded_test_review_import_report.json"
    real_file_sha256 = blinded_review.file_sha256

    def unstable_source_hash(path: Path) -> str:
        observed = real_file_sha256(path)
        if path.resolve() == manifest.resolve() and reviewed_path.exists():
            return "0" * 64
        return observed

    monkeypatch.setattr(blinded_review, "file_sha256", unstable_source_hash)
    with pytest.raises(BlindedReviewError, match="changed during import"):
        run_import(repo_root)

    assert not reviewed_path.exists()
    assert not report_path.exists()


@pytest.mark.parametrize(
    ("failure", "hidden_text"),
    (
        (PrivateDataError("safe private-data failure"), None),
        (OSError("SYNTHETIC PRIVATE ROW TEXT"), "SYNTHETIC PRIVATE ROW TEXT"),
    ),
)
def test_cli_reports_local_failures_without_tracebacks_or_oserror_details(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: BaseException,
    hidden_text: str | None,
) -> None:
    def fail_validate(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(review_cli, "run_validate", fail_validate)
    with pytest.raises(SystemExit) as exited:
        review_cli.main(["validate"])

    assert exited.value.code == 2
    stderr = capsys.readouterr().err
    assert "Traceback" not in stderr
    if hidden_text is not None:
        assert hidden_text not in stderr


def test_paths_outside_private_root_are_rejected(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    with pytest.raises(BlindedReviewError, match="outside"):
        resolve_review_paths(
            repo_root,
            review_file=repo_root / "RESULTS" / "review.jsonl",
        )
