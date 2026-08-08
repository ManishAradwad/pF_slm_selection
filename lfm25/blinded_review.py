"""Local-only blinded adjudication for the frozen private test partition.

The reviewer-facing JSONL deliberately contains only the source prompt fields,
an opaque deterministic ID, and blank human annotation fields.  Source hashes,
heuristic labels, model proposals, confidence, template groups, and sender hashes
remain in an internal mapping or in the unchanged source manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

from .annotation_workbench import (
    DEFAULT_BLINDED_DB,
    WORKBENCH_CONTRACT,
    WORKBENCH_SCHEMA_VERSION,
    exact_json_dumps,
    exact_json_loads,
    validate_annotation,
)
from .private_data import (
    PrivateDataError,
    _atomic_write_json,
    _atomic_write_jsonl,
    ensure_within,
    file_sha256,
    read_jsonl,
    require_private_ignore,
)


SCHEMA_VERSION = 1
PACKAGE_VERSION = "lfm25-blinded-test-review-v1"
PRIVATE_ROOT = Path("PRIVATE_DATA/lfm25")
DEFAULT_SOURCE_MANIFEST = PRIVATE_ROOT / "split_manifest.jsonl"
DEFAULT_REVIEW_FILE = PRIVATE_ROOT / "blinded_test_review.jsonl"
DEFAULT_MAPPING_FILE = PRIVATE_ROOT / "blinded_test_review_internal_map.jsonl"
DEFAULT_METADATA_FILE = PRIVATE_ROOT / "blinded_test_review_metadata.json"
DEFAULT_REVIEWED_MANIFEST = PRIVATE_ROOT / "split_manifest_human_reviewed.jsonl"
DEFAULT_IMPORT_REPORT = PRIVATE_ROOT / "blinded_test_review_import_report.json"
DEFAULT_WORKBENCH_DB = DEFAULT_BLINDED_DB

REVIEW_FIELDS = (
    "review_id",
    "sender",
    "sms",
    "decision",
    "amount",
    "counterparty",
    "type",
    "account",
    "reviewer",
    "reviewed_at",
    "notes",
)
MAPPING_FIELDS = ("review_id", "record_hash")
DECISIONS = ("transaction", "not_transaction")
_EXTRACTION_FIELDS = ("amount", "counterparty", "type", "account")
_ANNOTATION_FIELDS = (
    "decision",
    *_EXTRACTION_FIELDS,
    "reviewer",
    "reviewed_at",
    "notes",
)
REVIEW_ID_FORMAT = "test-{one_based_ordinal:06d_or_wider}"
_METADATA_FIELDS = {
    "schema_version",
    "package_version",
    "source_manifest",
    "source_manifest_sha256",
    "source_manifest_row_count",
    "test_row_count",
    "ordering",
    "review_id_format",
    "review_fields",
    "decision_values",
    "test_record_hash_set_sha256",
    "mapping_file_sha256",
    "review_template_sha256",
}


class BlindedReviewError(PrivateDataError):
    """A blinded-review failure whose message does not expose private row data."""


@dataclass(frozen=True)
class ReviewPaths:
    """Resolved paths constrained to the ignored private-data boundary."""

    private_root: Path
    source_manifest: Path
    review_file: Path
    mapping_file: Path
    metadata_file: Path
    reviewed_manifest: Path
    import_report: Path


@dataclass(frozen=True)
class ReviewAnnotation:
    """A validated pending or completed human annotation."""

    review_id: str
    decision: str | None
    label: dict[str, Any] | None
    reviewer: str | None
    reviewed_at: str | None
    notes: str | None

    @property
    def completed(self) -> bool:
        return self.decision is not None


@dataclass(frozen=True)
class _ValidatedPackage:
    paths: ReviewPaths
    source_records: tuple[dict[str, Any], ...]
    test_records: tuple[dict[str, Any], ...]
    mapping_rows: tuple[dict[str, str], ...]
    annotations: tuple[ReviewAnnotation, ...]
    source_manifest_sha256: str
    report: dict[str, Any]


def _resolve_path(repo_root: Path, private_root: Path, value: Path) -> Path:
    candidate = value if value.is_absolute() else repo_root / value
    try:
        return ensure_within(candidate, private_root)
    except PrivateDataError as exc:
        raise BlindedReviewError(
            "a requested blinded-review path is outside the private root"
        ) from exc


def resolve_review_paths(
    repo_root: Path,
    *,
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST,
    review_file: Path = DEFAULT_REVIEW_FILE,
    mapping_file: Path = DEFAULT_MAPPING_FILE,
    metadata_file: Path = DEFAULT_METADATA_FILE,
    reviewed_manifest: Path = DEFAULT_REVIEWED_MANIFEST,
    import_report: Path = DEFAULT_IMPORT_REPORT,
) -> ReviewPaths:
    """Resolve every artifact below ``PRIVATE_DATA/lfm25`` and reject aliases."""

    root = (repo_root / PRIVATE_ROOT).resolve()
    paths = ReviewPaths(
        private_root=root,
        source_manifest=_resolve_path(repo_root, root, source_manifest),
        review_file=_resolve_path(repo_root, root, review_file),
        mapping_file=_resolve_path(repo_root, root, mapping_file),
        metadata_file=_resolve_path(repo_root, root, metadata_file),
        reviewed_manifest=_resolve_path(repo_root, root, reviewed_manifest),
        import_report=_resolve_path(repo_root, root, import_report),
    )
    resolved = {
        paths.source_manifest,
        paths.review_file,
        paths.mapping_file,
        paths.metadata_file,
        paths.reviewed_manifest,
        paths.import_report,
    }
    if len(resolved) != 6:
        raise BlindedReviewError("blinded-review input and output paths must be distinct")
    return paths


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(exact_json_dumps(dict(row)) + "\n" for row in rows).encode("utf-8")


def _read_exact_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read reviewer JSONL while preserving fractional numbers as Decimal."""

    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for row_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = exact_json_loads(line)
                if not isinstance(value, dict):
                    raise BlindedReviewError(f"reviewer-facing row {row_number} is not an object")
                rows.append(value)
    except BlindedReviewError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise BlindedReviewError("the reviewer-facing annotation file could not be parsed") from exc
    return rows


def _atomic_write_exact_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Atomically write JSONL without converting Decimal values through float."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_jsonl_bytes(rows))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _record_hash_set_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(f"{record['record_hash']}\n" for record in records).encode("utf-8")
    return _sha256_bytes(payload)


def _load_source_records(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise BlindedReviewError("the source split manifest is missing or empty")
    records = read_jsonl(path)
    seen_hashes: set[str] = set()
    test_records: list[dict[str, Any]] = []
    for record in records:
        record_hash = record.get("record_hash")
        if not isinstance(record_hash, str) or not record_hash:
            raise BlindedReviewError("the source split manifest contains a missing record ID")
        if record_hash in seen_hashes:
            raise BlindedReviewError("the source split manifest contains duplicate record IDs")
        seen_hashes.add(record_hash)
        if record.get("split") != "test":
            continue
        sender = record.get("sender")
        sms = record.get("sms")
        if not isinstance(sender, str) or not isinstance(sms, str) or not sms:
            raise BlindedReviewError("a test row is missing required reviewer source fields")
        test_records.append(record)
    if not test_records:
        raise BlindedReviewError("the source split manifest contains no test rows")
    test_records.sort(key=lambda record: str(record["record_hash"]))
    return records, test_records


def build_blinded_rows(
    test_records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Create deterministic reviewer rows and a separate private ID mapping."""

    ordered = sorted(test_records, key=lambda record: str(record["record_hash"]))
    width = max(6, len(str(len(ordered))))
    review_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, str]] = []
    for ordinal, record in enumerate(ordered, start=1):
        review_id = f"test-{ordinal:0{width}d}"
        review_rows.append(
            {
                "review_id": review_id,
                "sender": record["sender"],
                "sms": record["sms"],
                "decision": None,
                "amount": None,
                "counterparty": None,
                "type": None,
                "account": None,
                "reviewer": None,
                "reviewed_at": None,
                "notes": None,
            }
        )
        mapping_rows.append(
            {"review_id": review_id, "record_hash": str(record["record_hash"])}
        )
    return review_rows, mapping_rows


def _relative_private_path(repo_root: Path, path: Path) -> str:
    try:
        return path.relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:  # Guarded by resolve_review_paths; retain a safe failure.
        raise BlindedReviewError("a blinded-review path is outside the repository") from exc


def _metadata(
    repo_root: Path,
    paths: ReviewPaths,
    source_records: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
    review_rows: Sequence[Mapping[str, Any]],
    mapping_rows: Sequence[Mapping[str, Any]],
    source_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "package_version": PACKAGE_VERSION,
        "source_manifest": _relative_private_path(repo_root, paths.source_manifest),
        "source_manifest_sha256": source_manifest_sha256,
        "source_manifest_row_count": len(source_records),
        "test_row_count": len(test_records),
        "ordering": "record_hash_lexicographic",
        "review_id_format": REVIEW_ID_FORMAT,
        "review_fields": list(REVIEW_FIELDS),
        "decision_values": list(DECISIONS),
        "test_record_hash_set_sha256": _record_hash_set_sha256(test_records),
        "mapping_file_sha256": _sha256_bytes(_jsonl_bytes(mapping_rows)),
        "review_template_sha256": _sha256_bytes(_jsonl_bytes(review_rows)),
    }


def _require_replaceable(paths: Sequence[Path], force: bool) -> None:
    if any(path.exists() and not path.is_file() for path in paths):
        raise BlindedReviewError("a blinded-review output path is not a regular file")
    nonempty = [path for path in paths if path.exists() and path.stat().st_size > 0]
    if nonempty and not force:
        raise BlindedReviewError(
            "one or more blinded-review outputs are nonempty; use --force to replace them"
        )


def _protect_private_files(paths: Sequence[Path]) -> None:
    for path in paths:
        path.chmod(0o600)


def _commit_staged_outputs(
    pairs: Sequence[tuple[Path, Path]],
    *,
    source_manifest: Path,
    expected_source_sha256: str,
    source_change_message: str,
) -> None:
    """Replace a group with rollback; the final pair is the validity marker."""

    backups: list[tuple[Path, Path]] = []
    committed: list[Path] = []
    staging_root = pairs[0][0].parent

    def require_frozen_source() -> None:
        if file_sha256(source_manifest) != expected_source_sha256:
            raise BlindedReviewError(source_change_message)

    try:
        require_frozen_source()
        for index, (staged, final) in enumerate(pairs):
            if final.exists():
                backup = staging_root / f".previous-{index:03d}-{final.name}"
                final.replace(backup)
                backups.append((backup, final))
            staged.replace(final)
            committed.append(final)
            require_frozen_source()
    except BaseException as exc:
        rollback_errors: list[BaseException] = []
        for final in reversed(committed):
            try:
                final.unlink(missing_ok=True)
            except BaseException as rollback_exc:
                rollback_errors.append(rollback_exc)
        for backup, final in reversed(backups):
            try:
                if backup.exists():
                    backup.replace(final)
            except BaseException as rollback_exc:
                rollback_errors.append(rollback_exc)
        if rollback_errors:
            raise BlindedReviewError(
                "a blinded-review staged-output rollback failed; outputs require inspection"
            ) from exc
        raise


def run_export(
    repo_root: Path,
    *,
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST,
    review_file: Path = DEFAULT_REVIEW_FILE,
    mapping_file: Path = DEFAULT_MAPPING_FILE,
    metadata_file: Path = DEFAULT_METADATA_FILE,
    force: bool = False,
) -> dict[str, Any]:
    """Export all and only frozen test rows into a blinded local review package."""

    paths = resolve_review_paths(
        repo_root,
        source_manifest=source_manifest,
        review_file=review_file,
        mapping_file=mapping_file,
        metadata_file=metadata_file,
    )
    require_private_ignore(repo_root, paths.private_root)
    _require_replaceable(
        (paths.review_file, paths.mapping_file, paths.metadata_file),
        force,
    )
    source_hash_before = file_sha256(paths.source_manifest)
    source_records, test_records = _load_source_records(paths.source_manifest)
    review_rows, mapping_rows = build_blinded_rows(test_records)
    metadata = _metadata(
        repo_root,
        paths,
        source_records,
        test_records,
        review_rows,
        mapping_rows,
        source_hash_before,
    )
    with TemporaryDirectory(
        prefix=".blinded-review-export.",
        dir=paths.private_root,
    ) as temporary_name:
        staging = Path(temporary_name)
        staged_review = staging / paths.review_file.name
        staged_mapping = staging / paths.mapping_file.name
        staged_metadata = staging / paths.metadata_file.name
        _atomic_write_jsonl(staged_review, review_rows)
        _atomic_write_jsonl(staged_mapping, mapping_rows)
        _atomic_write_json(staged_metadata, metadata)
        _protect_private_files((staged_review, staged_mapping, staged_metadata))
        if file_sha256(paths.source_manifest) != source_hash_before:
            raise BlindedReviewError("the source split manifest changed during export")
        _require_replaceable(
            (paths.review_file, paths.mapping_file, paths.metadata_file),
            force,
        )
        _commit_staged_outputs(
            (
                (staged_review, paths.review_file),
                (staged_mapping, paths.mapping_file),
                (staged_metadata, paths.metadata_file),
            ),
            source_manifest=paths.source_manifest,
            expected_source_sha256=source_hash_before,
            source_change_message="the source split manifest changed during export",
        )
    return {
        "operation": "export",
        "valid": True,
        "schema_version": SCHEMA_VERSION,
        "source_rows": len(source_records),
        "test_rows": len(test_records),
        "completed_rows": 0,
        "pending_rows": len(test_records),
        "transaction_rows": 0,
        "not_transaction_rows": 0,
        "ready_for_evaluation": False,
        "wrote_files": True,
        "source_manifest_unchanged": True,
    }


def _load_metadata(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BlindedReviewError("the blinded-review metadata could not be parsed") from exc
    if not isinstance(value, dict) or set(value) != _METADATA_FIELDS:
        raise BlindedReviewError("the blinded-review metadata schema is invalid")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise BlindedReviewError("the blinded-review schema version is unsupported")
    if value.get("package_version") != PACKAGE_VERSION:
        raise BlindedReviewError("the blinded-review package version is unsupported")
    if value.get("ordering") != "record_hash_lexicographic":
        raise BlindedReviewError("the blinded-review ordering policy is invalid")
    if value.get("review_fields") != list(REVIEW_FIELDS):
        raise BlindedReviewError("the reviewer-facing field schema is invalid")
    if value.get("decision_values") != list(DECISIONS):
        raise BlindedReviewError("the blinded-review decision schema is invalid")
    integer_fields = ("source_manifest_row_count", "test_row_count")
    if any(
        isinstance(value.get(field), bool)
        or not isinstance(value.get(field), int)
        or value[field] < 1
        for field in integer_fields
    ):
        raise BlindedReviewError("the blinded-review metadata row counts are invalid")
    hash_fields = (
        "source_manifest_sha256",
        "test_record_hash_set_sha256",
        "mapping_file_sha256",
        "review_template_sha256",
    )
    if any(
        not isinstance(value.get(field), str) or len(value[field]) != 64
        for field in hash_fields
    ):
        raise BlindedReviewError("the blinded-review metadata hashes are invalid")
    if not isinstance(value.get("source_manifest"), str) or not value["source_manifest"]:
        raise BlindedReviewError("the blinded-review source provenance is invalid")
    if value.get("review_id_format") != REVIEW_ID_FORMAT:
        raise BlindedReviewError("the blinded-review ID format metadata is invalid")
    return value


def _validate_mapping(
    mapping_rows: Sequence[Mapping[str, Any]],
    expected_rows: Sequence[Mapping[str, str]],
) -> None:
    review_ids: set[str] = set()
    record_hashes: set[str] = set()
    for row in mapping_rows:
        if set(row) != set(MAPPING_FIELDS):
            raise BlindedReviewError("the internal blinded-review mapping schema is invalid")
        review_id = row.get("review_id")
        record_hash = row.get("record_hash")
        if not isinstance(review_id, str) or not review_id:
            raise BlindedReviewError("the internal mapping contains a missing review ID")
        if not isinstance(record_hash, str) or not record_hash:
            raise BlindedReviewError("the internal mapping contains a missing record ID")
        if review_id in review_ids or record_hash in record_hashes:
            raise BlindedReviewError("the internal mapping contains duplicate IDs")
        review_ids.add(review_id)
        record_hashes.add(record_hash)
    if len(mapping_rows) != len(expected_rows):
        raise BlindedReviewError("the internal mapping has missing or unknown IDs")
    if list(mapping_rows) != list(expected_rows):
        raise BlindedReviewError("the internal mapping does not match the frozen test order")


def _valid_reviewed_at(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _canonical_review_label(row: Mapping[str, Any]) -> dict[str, Any]:
    invalid_message = "a review row contains an invalid canonical transaction label"
    amount = row["amount"]
    if isinstance(amount, bool) or not isinstance(amount, (int, float, Decimal)):
        raise BlindedReviewError(invalid_message)
    try:
        exact_amount = Decimal(str(amount)) if isinstance(amount, float) else Decimal(amount)
    except (TypeError, ValueError):
        raise BlindedReviewError(invalid_message) from None
    if not exact_amount.is_finite() or exact_amount <= 0:
        raise BlindedReviewError(invalid_message)
    counterparty = row["counterparty"]
    account = row["account"]
    if (
        (
            counterparty is not None
            and (not isinstance(counterparty, str) or not counterparty.strip())
        )
        or row["type"] not in {"debit", "credit"}
        or not isinstance(account, str)
        or not account.strip()
    ):
        raise BlindedReviewError(invalid_message)
    return {
        "amount": exact_amount,
        "counterparty": counterparty,
        "type": row["type"],
        "account": account,
    }


def _validate_annotation(row: Mapping[str, Any]) -> ReviewAnnotation:
    review_id = str(row["review_id"])
    decision = row["decision"]
    notes = row["notes"]
    if notes is not None and not isinstance(notes, str):
        raise BlindedReviewError("a review row contains invalid notes")
    if decision is None:
        if any(row[field] is not None for field in _ANNOTATION_FIELDS[1:]):
            raise BlindedReviewError("a pending review row contains a partial annotation")
        return ReviewAnnotation(review_id, None, None, None, None, None)
    if decision not in DECISIONS:
        raise BlindedReviewError("a review row contains an invalid transaction decision")
    reviewer = row["reviewer"]
    reviewed_at = row["reviewed_at"]
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise BlindedReviewError("a completed review row is missing its reviewer")
    if not _valid_reviewed_at(reviewed_at):
        raise BlindedReviewError("a completed review row has an invalid reviewed_at timestamp")
    if decision == "not_transaction":
        if any(row[field] is not None for field in _EXTRACTION_FIELDS):
            raise BlindedReviewError(
                "a not_transaction review row must leave extraction fields null"
            )
        label = None
    else:
        label = _canonical_review_label(row)
    return ReviewAnnotation(review_id, str(decision), label, reviewer, reviewed_at, notes)


def _validate_review_rows(
    review_rows: Sequence[Mapping[str, Any]],
    mapping_rows: Sequence[Mapping[str, str]],
    source_by_hash: Mapping[str, Mapping[str, Any]],
) -> list[ReviewAnnotation]:
    seen_ids: set[str] = set()
    annotations_by_id: dict[str, ReviewAnnotation] = {}
    observed_order: list[str] = []
    for row in review_rows:
        if set(row) != set(REVIEW_FIELDS):
            raise BlindedReviewError("a reviewer-facing row has an invalid schema")
        review_id = row.get("review_id")
        if not isinstance(review_id, str) or not review_id:
            raise BlindedReviewError("a reviewer-facing row has a missing review ID")
        if review_id in seen_ids:
            raise BlindedReviewError("the reviewer-facing file contains duplicate review IDs")
        seen_ids.add(review_id)
        observed_order.append(review_id)
        annotations_by_id[review_id] = _validate_annotation(row)

    expected_ids = [str(row["review_id"]) for row in mapping_rows]
    if set(observed_order) != set(expected_ids):
        raise BlindedReviewError("the reviewer-facing file contains missing or unknown IDs")
    if observed_order != expected_ids:
        raise BlindedReviewError("the reviewer-facing file order no longer matches the frozen order")

    ordered_annotations: list[ReviewAnnotation] = []
    for review_row, mapping_row in zip(review_rows, mapping_rows, strict=True):
        source = source_by_hash[str(mapping_row["record_hash"])]
        if review_row["sender"] != source["sender"] or review_row["sms"] != source["sms"]:
            raise BlindedReviewError(
                "reviewer-facing source fields do not match the frozen source manifest"
            )
        ordered_annotations.append(annotations_by_id[str(mapping_row["review_id"])])
    return ordered_annotations


def _validate_package(repo_root: Path, paths: ReviewPaths) -> _ValidatedPackage:
    require_private_ignore(repo_root, paths.private_root)
    for required in (
        paths.source_manifest,
        paths.review_file,
        paths.mapping_file,
        paths.metadata_file,
    ):
        if not required.is_file() or required.stat().st_size == 0:
            raise BlindedReviewError("one or more blinded-review package files are missing or empty")

    source_hash = file_sha256(paths.source_manifest)
    source_records, test_records = _load_source_records(paths.source_manifest)
    metadata = _load_metadata(paths.metadata_file)
    if metadata["source_manifest"] != _relative_private_path(repo_root, paths.source_manifest):
        raise BlindedReviewError("the blinded-review source path does not match its provenance")
    if metadata["source_manifest_sha256"] != source_hash:
        raise BlindedReviewError("the source split manifest changed after blinded export")
    if metadata["source_manifest_row_count"] != len(source_records):
        raise BlindedReviewError("the source split manifest row count changed after export")
    if metadata["test_row_count"] != len(test_records):
        raise BlindedReviewError("the frozen test row count changed after export")
    if metadata["test_record_hash_set_sha256"] != _record_hash_set_sha256(test_records):
        raise BlindedReviewError("the frozen test record set changed after export")
    if metadata["mapping_file_sha256"] != file_sha256(paths.mapping_file):
        raise BlindedReviewError("the internal blinded-review mapping changed after export")

    mapping_rows = read_jsonl(paths.mapping_file)
    expected_review, expected_mapping = build_blinded_rows(test_records)
    if metadata["review_template_sha256"] != _sha256_bytes(
        _jsonl_bytes(expected_review)
    ):
        raise BlindedReviewError("the blinded-review template provenance is invalid")
    _validate_mapping(mapping_rows, expected_mapping)
    source_by_hash = {str(record["record_hash"]): record for record in test_records}
    review_rows = _read_exact_jsonl(paths.review_file)
    annotations = _validate_review_rows(review_rows, mapping_rows, source_by_hash)

    completed = [annotation for annotation in annotations if annotation.completed]
    transaction_count = sum(
        annotation.decision == "transaction" for annotation in completed
    )
    not_transaction_count = sum(
        annotation.decision == "not_transaction" for annotation in completed
    )
    report = {
        "operation": "validate",
        "valid": True,
        "schema_version": SCHEMA_VERSION,
        "source_rows": len(source_records),
        "test_rows": len(test_records),
        "completed_rows": len(completed),
        "pending_rows": len(annotations) - len(completed),
        "transaction_rows": transaction_count,
        "not_transaction_rows": not_transaction_count,
        "ready_for_evaluation": len(completed) == len(annotations),
        "wrote_files": False,
        "source_manifest_unchanged": file_sha256(paths.source_manifest) == source_hash,
    }
    if not report["source_manifest_unchanged"]:
        raise BlindedReviewError("the source split manifest changed during validation")
    return _ValidatedPackage(
        paths=paths,
        source_records=tuple(source_records),
        test_records=tuple(test_records),
        mapping_rows=tuple(dict(row) for row in mapping_rows),
        annotations=tuple(annotations),
        source_manifest_sha256=source_hash,
        report=report,
    )


def run_validate(
    repo_root: Path,
    *,
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST,
    review_file: Path = DEFAULT_REVIEW_FILE,
    mapping_file: Path = DEFAULT_MAPPING_FILE,
    metadata_file: Path = DEFAULT_METADATA_FILE,
) -> dict[str, Any]:
    """Validate a possibly partial review package without writing any artifact."""

    paths = resolve_review_paths(
        repo_root,
        source_manifest=source_manifest,
        review_file=review_file,
        mapping_file=mapping_file,
        metadata_file=metadata_file,
    )
    return dict(_validate_package(repo_root, paths).report)


def _load_workbench_annotations_for_import(
    repo_root: Path,
    package: _ValidatedPackage,
    db_path: Path,
) -> dict[str, dict[str, Any]]:
    from .annotation_sources import load_blinded_workspace
    from .annotation_store import WorkbenchStore

    definition = load_blinded_workspace(
        repo_root,
        source_manifest=package.paths.source_manifest,
        review_file=package.paths.review_file,
        mapping_file=package.paths.mapping_file,
        metadata_file=package.paths.metadata_file,
        include_initial_annotations=False,
    )
    result: dict[str, dict[str, Any]] = {}
    with WorkbenchStore(
        db_path,
        workspace_binding=definition.binding,
    ) as store:
        rows = store.get_rows(limit=definition.row_count + 1)
        if len(rows) != definition.row_count:
            raise BlindedReviewError("workbench provenance row count does not match")
        for stored, mapping in zip(
            rows,
            package.mapping_rows,
            strict=True,
        ):
            if stored["row_id"] != mapping["review_id"]:
                raise BlindedReviewError("workbench provenance ordering does not match")
            if stored["status"] != "completed":
                raise BlindedReviewError("workbench provenance is incomplete")
            fields = stored["reviewer_fields"]
            annotation = validate_annotation(
                stored["annotation"],
                str(fields["sms"]),
                require_complete=True,
            )
            history = store.get_history(str(stored["row_id"]))
            if not history:
                raise BlindedReviewError("workbench provenance history is missing")
            record_hash = str(mapping["record_hash"])
            result[record_hash] = {
                "contract": WORKBENCH_CONTRACT,
                "schema_version": WORKBENCH_SCHEMA_VERSION,
                "annotation": annotation,
                "reviewer": stored["reviewer"],
                "reviewed_at": stored["recorded_at"],
                "blind_first_event_hash": stored["initial_event_hash"],
                "final_event_hash": history[-1]["event_hash"],
                "final_phase": stored["phase"],
                "history_event_count": len(history),
                "qc_required": stored["qc_required"],
                "qc_status": stored["qc_status"],
                "qc_event_hash": stored["qc_event_hash"],
                "qc_for_initial_hash": stored["qc_for_initial_hash"],
            }
    expected_hashes = {
        str(mapping["record_hash"])
        for mapping in package.mapping_rows
    }
    if set(result) != expected_hashes:
        raise BlindedReviewError("workbench provenance identity set does not match")
    return result


def _merge_annotations(
    package: _ValidatedPackage,
    workbench_annotations: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    annotation_by_hash = {
        str(mapping["record_hash"]): annotation
        for mapping, annotation in zip(
            package.mapping_rows,
            package.annotations,
            strict=True,
        )
    }
    merged: list[dict[str, Any]] = []
    for source in package.source_records:
        row = dict(source)
        record_hash = str(source["record_hash"])
        annotation = annotation_by_hash.get(record_hash)
        if annotation is not None and annotation.completed:
            workbench = workbench_annotations.get(record_hash)
            if workbench is None:
                raise BlindedReviewError("workbench annotation provenance is missing")
            row["review_status"] = "human_approved"
            row["human_review_required"] = True
            row["human_label"] = annotation.label
            row["human_reviewer"] = annotation.reviewer
            row["human_reviewed_at"] = annotation.reviewed_at
            row["human_review_notes"] = annotation.notes
            row["human_annotation_workbench"] = dict(workbench)
        merged.append(row)
    return merged


def run_import(
    repo_root: Path,
    *,
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST,
    review_file: Path = DEFAULT_REVIEW_FILE,
    mapping_file: Path = DEFAULT_MAPPING_FILE,
    metadata_file: Path = DEFAULT_METADATA_FILE,
    reviewed_manifest: Path = DEFAULT_REVIEWED_MANIFEST,
    import_report: Path = DEFAULT_IMPORT_REPORT,
    workbench_db: Path = DEFAULT_WORKBENCH_DB,
    force: bool = False,
) -> dict[str, Any]:
    """Validate annotations and write a separate reviewed manifest and report."""

    paths = resolve_review_paths(
        repo_root,
        source_manifest=source_manifest,
        review_file=review_file,
        mapping_file=mapping_file,
        metadata_file=metadata_file,
        reviewed_manifest=reviewed_manifest,
        import_report=import_report,
    )
    package = _validate_package(repo_root, paths)
    if not package.report["ready_for_evaluation"]:
        raise BlindedReviewError(
            "final import requires every reviewer-facing annotation to be completed"
        )
    resolved_workbench_db = _resolve_path(repo_root, paths.private_root, workbench_db)
    if resolved_workbench_db in vars(paths).values():
        raise BlindedReviewError(
            "the workbench database must be distinct from review package artifacts"
        )
    if not resolved_workbench_db.is_file() or resolved_workbench_db.stat().st_size == 0:
        raise BlindedReviewError("the blinded-test workbench database is missing or empty")
    _require_replaceable((paths.reviewed_manifest, paths.import_report), force)
    from .annotation_service import validate_blinded_import_gate

    gate_report = validate_blinded_import_gate(
        repo_root,
        db_path=resolved_workbench_db,
        source_manifest=paths.source_manifest,
        review_file=paths.review_file,
        mapping_file=paths.mapping_file,
        metadata_file=paths.metadata_file,
    )
    qc_required_rows = gate_report.get("qc_required_rows")
    if (
        gate_report.get("valid") is not True
        or gate_report.get("ready_for_import") is not True
        or gate_report.get("completed_rows") != len(package.annotations)
        or gate_report.get("pending_rows") != 0
        or gate_report.get("unresolved_uncertain_rows") != 0
        or isinstance(qc_required_rows, bool)
        or not isinstance(qc_required_rows, int)
        or qc_required_rows < 0
        or gate_report.get("qc_passed_rows") != qc_required_rows
    ):
        raise BlindedReviewError("the blinded-test workbench final import gate did not pass")
    workbench_annotations = _load_workbench_annotations_for_import(
        repo_root,
        package,
        resolved_workbench_db,
    )
    merged = _merge_annotations(package, workbench_annotations)
    with TemporaryDirectory(
        prefix=".blinded-review-import.",
        dir=paths.private_root,
    ) as temporary_name:
        staging = Path(temporary_name)
        staged_manifest = staging / paths.reviewed_manifest.name
        staged_report = staging / paths.import_report.name
        _atomic_write_exact_jsonl(staged_manifest, merged)
        reviewed_hash = file_sha256(staged_manifest)
        report = dict(package.report)
        report.update(
            {
                "operation": "import",
                "wrote_files": True,
                "reviewed_manifest_sha256": reviewed_hash,
                "source_manifest_sha256": package.source_manifest_sha256,
                "source_manifest_unchanged": True,
                "workbench_gate_passed": True,
                "workbench_annotation_contract": WORKBENCH_CONTRACT,
                "workbench_annotation_rows": len(workbench_annotations),
                "unresolved_uncertain_rows": gate_report["unresolved_uncertain_rows"],
                "qc_required_rows": gate_report["qc_required_rows"],
                "qc_passed_rows": gate_report["qc_passed_rows"],
            }
        )
        _atomic_write_json(staged_report, report)
        _protect_private_files((staged_manifest, staged_report))
        if file_sha256(paths.source_manifest) != package.source_manifest_sha256:
            raise BlindedReviewError("the source split manifest changed during import")
        _require_replaceable((paths.reviewed_manifest, paths.import_report), force)
        _commit_staged_outputs(
            (
                (staged_manifest, paths.reviewed_manifest),
                (staged_report, paths.import_report),
            ),
            source_manifest=paths.source_manifest,
            expected_source_sha256=package.source_manifest_sha256,
            source_change_message="the source split manifest changed during import",
        )
    return report


def safe_console_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return aggregate-only fields suitable for stdout or a pull-request log."""

    fields = (
        "operation",
        "valid",
        "schema_version",
        "source_rows",
        "test_rows",
        "completed_rows",
        "pending_rows",
        "transaction_rows",
        "not_transaction_rows",
        "ready_for_evaluation",
        "workbench_gate_passed",
        "workbench_annotation_contract",
        "workbench_annotation_rows",
        "unresolved_uncertain_rows",
        "qc_required_rows",
        "qc_passed_rows",
        "wrote_files",
        "source_manifest_unchanged",
    )
    return {field: report[field] for field in fields if field in report}
