"""Build the private, app-aligned training intersection for Candidate Protocol V1."""

from __future__ import annotations

from collections import Counter
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from lfm25.android_contract import pocketfinancer_prefilter_sms
from lfm25.candidate_protocol import (
    PROTOCOL_VERSION,
    build_protocol_request,
    oracle_coverage,
    serialize_selector_target,
)
from lfm25.private_data import (
    _atomic_write_json,
    _atomic_write_jsonl,
    ensure_within,
    file_sha256,
    require_private_ignore,
)


BUILDER_VERSION = "pocketfinancer-candidate-protocol-v1-sft-v1"
INPUT_FILENAMES = {
    "train": "private_sft_v2_train.jsonl",
    "dev": "private_sft_v2_dev.jsonl",
}
OUTPUT_FILENAMES = {
    "train": "candidate_protocol_v1_train.jsonl",
    "dev": "candidate_protocol_v1_dev.jsonl",
    "report": "candidate_protocol_v1_report.json",
}


class CandidateProtocolDataError(ValueError):
    """Aggregate-safe failure while materializing private protocol rows."""


def _reject_nonfinite_number(_value: str) -> Any:
    raise ValueError("non-finite JSON numbers are not allowed")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _read_exact_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read private source rows without projecting fractional JSON numbers to float."""

    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(
                    line,
                    parse_float=Decimal,
                    parse_constant=_reject_nonfinite_number,
                    object_pairs_hook=_unique_object,
                )
                if not isinstance(row, dict):
                    raise ValueError("Candidate Protocol V1 source row is not an object")
                rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise CandidateProtocolDataError(
            "Candidate Protocol V1 private source could not be parsed"
        ) from error
    return rows


def _json_compatible(value: Any) -> Any:
    """Return the ordinary JSON-native projection consumed by both training arms."""

    if isinstance(value, Decimal):
        projected = float(value)
        if not math.isfinite(projected):
            raise CandidateProtocolDataError(
                "Candidate Protocol V1 source has no finite JSON-native projection"
            )
        return projected
    if isinstance(value, float) and not math.isfinite(value):
        raise CandidateProtocolDataError(
            "Candidate Protocol V1 source has no finite JSON-native projection"
        )
    if isinstance(value, Mapping):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _sha256_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        serialized = json.dumps(
            dict(row),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        digest.update((serialized + "\n").encode("utf-8"))
    return digest.hexdigest()


def select_protocol_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return only original-train rows representable by source-backed candidates."""

    accepted: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    tiers: Counter[str] = Counter()
    seen_hashes: set[str] = set()
    for row in rows:
        source = row.get("source")
        if not isinstance(source, Mapping) or source.get("original_split") != "train":
            exclusions["not_original_train"] += 1
            continue
        record_hash = source.get("record_hash")
        if not isinstance(record_hash, str) or not record_hash or record_hash in seen_hashes:
            exclusions["missing_or_duplicate_record_hash"] += 1
            continue
        split_identities = (
            source.get("private_sender_hash"),
            source.get("template_group"),
        )
        if any(not isinstance(value, str) or not value for value in split_identities):
            exclusions["missing_split_identity"] += 1
            continue
        seen_hashes.add(record_hash)
        sender = row.get("sender")
        sms = row.get("sms")
        if not isinstance(sender, str) or not isinstance(sms, str):
            exclusions["missing_sender_or_sms"] += 1
            continue
        provenance = row.get("provenance")
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("android_prefilter_accepted") is not True
            or not pocketfinancer_prefilter_sms(sender, sms).accepted
        ):
            exclusions["android_prefilter_not_accepted"] += 1
            continue
        if "expected" not in row:
            exclusions["missing_label"] += 1
            continue
        try:
            request = build_protocol_request(sender, sms)
            oracle = oracle_coverage(row["expected"], request)
        except (TypeError, ValueError):
            exclusions["invalid_label"] += 1
            continue

        try:
            if not oracle.covered:
                for field in oracle.missing_fields:
                    exclusions[f"candidate_missing_{field}"] += 1
                continue
            target = serialize_selector_target(row["expected"], request)
        except (TypeError, ValueError) as error:
            exclusions[f"protocol_target_{type(error).__name__.lower()}"] += 1
            continue

        prepared = _json_compatible(row)
        if not isinstance(prepared, dict):  # Defensive: source rows are objects.
            raise CandidateProtocolDataError(
                "Candidate Protocol V1 source row has no JSON-native projection"
            )
        prepared["candidate_protocol_v1_target"] = target
        prepared["provenance"] = {
            **dict(provenance),
            "candidate_protocol": PROTOCOL_VERSION,
            "candidate_protocol_builder_version": BUILDER_VERSION,
            "candidate_oracle_covered": True,
        }
        accepted.append(prepared)
        kinds["transaction" if oracle.is_transaction else "null"] += 1
        tiers[str(row.get("label_tier", "unknown"))] += 1

    return accepted, {
        "input_rows": len(rows),
        "output_rows": len(accepted),
        "excluded_rows": len(rows) - len(accepted),
        "exclusion_reasons": dict(sorted(exclusions.items())),
        "label_kind_counts": dict(sorted(kinds.items())),
        "label_tier_counts": dict(sorted(tiers.items())),
    }


def build_candidate_protocol_data(
    *,
    repo_root: Path,
    input_dir: Path,
    output_dir: Path,
    implementation_root: Path | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Materialize a private candidate-covered train/dev intersection."""

    repo = repo_root.resolve()
    implementation = (
        repo
        if implementation_root is None
        else (
            implementation_root
            if implementation_root.is_absolute()
            else repo / implementation_root
        ).resolve()
    )
    extractor_path = implementation / "lfm25" / "candidates.py"
    protocol_path = implementation / "lfm25" / "candidate_protocol.py"
    if not extractor_path.is_file() or not protocol_path.is_file():
        raise CandidateProtocolDataError("Candidate Protocol V1 implementation is incomplete")
    private_root = (repo / "PRIVATE_DATA" / "lfm25").resolve()
    source_dir = ensure_within(
        input_dir if input_dir.is_absolute() else repo / input_dir,
        private_root,
    )
    destination = ensure_within(
        output_dir if output_dir.is_absolute() else repo / output_dir,
        private_root,
    )
    require_private_ignore(repo, private_root)
    if destination == source_dir:
        raise CandidateProtocolDataError("protocol output directory must differ from input")

    selected: dict[str, list[dict[str, Any]]] = {}
    split_reports: dict[str, Any] = {}
    inputs: dict[str, Any] = {}
    for split, filename in INPUT_FILENAMES.items():
        path = source_dir / filename
        if not path.is_file():
            raise CandidateProtocolDataError(f"missing app-aligned {split} input")
        rows = _read_exact_jsonl(path)
        selected[split], split_reports[split] = select_protocol_rows(rows)
        inputs[split] = {
            "filename": filename,
            "sha256": file_sha256(path),
            "rows": len(rows),
        }

    train_senders = {str(row["source"]["private_sender_hash"]) for row in selected["train"]}
    dev_senders = {str(row["source"]["private_sender_hash"]) for row in selected["dev"]}
    train_templates = {str(row["source"]["template_group"]) for row in selected["train"]}
    dev_templates = {str(row["source"]["template_group"]) for row in selected["dev"]}
    train_records = {str(row["source"]["record_hash"]) for row in selected["train"]}
    dev_records = {str(row["source"]["record_hash"]) for row in selected["dev"]}
    if (
        train_records & dev_records
        or train_senders & dev_senders
        or train_templates & dev_templates
    ):
        raise CandidateProtocolDataError(
            "protocol subset introduced record/sender/template leakage"
        )

    artifacts = {
        split: {
            "filename": OUTPUT_FILENAMES[split],
            "rows": len(rows),
            "sha256": _sha256_rows(rows),
        }
        for split, rows in selected.items()
    }
    report = {
        "builder_version": BUILDER_VERSION,
        "candidate_protocol": PROTOCOL_VERSION,
        "candidate_implementations": {
            "extractor": {
                "path": "lfm25/candidates.py",
                "sha256": file_sha256(extractor_path),
            },
            "protocol": {
                "path": "lfm25/candidate_protocol.py",
                "sha256": file_sha256(protocol_path),
            },
        },
        "valid": True,
        "private_local_only": True,
        "release_authorized": False,
        "inputs": inputs,
        "splits": split_reports,
        "artifacts": artifacts,
        "invariants": {
            "only_original_train_rows": True,
            "candidate_oracle_covered": True,
            "android_prefilter_accepted": True,
            "same_rows_for_direct_and_selector": True,
            "historical_candidate_data_reused": False,
            "record_overlap_count": 0,
            "sender_overlap_count": 0,
            "template_overlap_count": 0,
            "sealed_test_rows_materialized": 0,
            "raw_values_emitted_to_stdout": False,
        },
    }
    if dry_run:
        return {"dry_run": True, "wrote_files": False, "report": report}

    paths = {split: destination / OUTPUT_FILENAMES[split] for split in ("train", "dev")}
    paths["report"] = destination / OUTPUT_FILENAMES["report"]
    if not force and any(path.exists() for path in paths.values()):
        raise CandidateProtocolDataError(
            "Candidate Protocol V1 outputs exist; pass --force to replace"
        )
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    _atomic_write_jsonl(paths["train"], selected["train"])
    _atomic_write_jsonl(paths["dev"], selected["dev"])
    _atomic_write_json(paths["report"], report)
    for split in ("train", "dev"):
        if file_sha256(paths[split]) != artifacts[split]["sha256"]:
            raise CandidateProtocolDataError(f"written {split} Candidate Protocol V1 hash mismatch")
    return {"dry_run": False, "wrote_files": True, "report": report}


def safe_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return aggregate-only materialization evidence safe for stdout."""

    report = result["report"]
    return {
        "dry_run": result["dry_run"],
        "wrote_files": result["wrote_files"],
        "valid": report["valid"],
        "candidate_protocol": report["candidate_protocol"],
        "splits": report["splits"],
        "artifacts": report["artifacts"],
        "invariants": report["invariants"],
    }
