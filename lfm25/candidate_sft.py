"""Materialize the source-grounded subset for candidate-selector training."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from lfm25.candidates import extract_candidates, oracle_selection, selector_target
from lfm25.contract import parse_gold
from lfm25.private_data import (
    _atomic_write_json,
    _atomic_write_jsonl,
    ensure_within,
    file_sha256,
    read_jsonl,
    require_private_ignore,
)


BUILDER_VERSION = "lfm25-candidate-sft-v3"
INPUT_FILENAMES = {
    "train": "private_sft_v2_train.jsonl",
    "dev": "private_sft_v2_dev.jsonl",
}
OUTPUT_FILENAMES = {
    "train": "candidate_sft_train.jsonl",
    "dev": "candidate_sft_dev.jsonl",
    "report": "candidate_sft_report.json",
}


class CandidateSFTError(ValueError):
    """An aggregate-safe candidate materialization failure."""


def _sha256_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            (json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def select_grounded_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep rows whose selector target can be reconstructed from current-SMS spans."""

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
        seen_hashes.add(record_hash)
        if not isinstance(row.get("sender"), str) or not isinstance(row.get("sms"), str):
            exclusions["missing_sender_or_sms"] += 1
            continue
        try:
            gold = parse_gold(row.get("expected"))
        except ValueError:
            exclusions["invalid_label"] += 1
            continue
        candidates = extract_candidates(row["sms"])
        oracle = oracle_selection(gold, candidates)
        if not oracle.covered:
            for field in oracle.missing_fields:
                exclusions[f"candidate_missing_{field}"] += 1
            continue
        # This second check prevents the audit and trainer paths from drifting.
        try:
            selector_target(gold, candidates)
        except ValueError:
            exclusions["selector_target_invalid"] += 1
            continue
        prepared = dict(row)
        prepared["provenance"] = {
            **(
                dict(row["provenance"])
                if isinstance(row.get("provenance"), Mapping)
                else {}
            ),
            "candidate_builder_version": BUILDER_VERSION,
            "candidate_oracle_covered": True,
        }
        accepted.append(prepared)
        kinds["transaction" if gold is not None else "null"] += 1
        tiers[str(row.get("label_tier", "unknown"))] += 1
    return accepted, {
        "input_rows": len(rows),
        "output_rows": len(accepted),
        "excluded_rows": len(rows) - len(accepted),
        "exclusion_reasons": dict(sorted(exclusions.items())),
        "label_kind_counts": dict(sorted(kinds.items())),
        "label_tier_counts": dict(sorted(tiers.items())),
    }


def build_candidate_sft(
    *,
    repo_root: Path,
    input_dir: Path,
    output_dir: Path,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    repo = repo_root.resolve()
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
        raise CandidateSFTError("candidate output directory must differ from its input")

    selected: dict[str, list[dict[str, Any]]] = {}
    split_reports: dict[str, Any] = {}
    inputs: dict[str, Any] = {}
    for split, filename in INPUT_FILENAMES.items():
        path = source_dir / filename
        if not path.is_file():
            raise CandidateSFTError(f"missing fixed private v2 {split} input")
        rows = read_jsonl(path)
        selected[split], split_reports[split] = select_grounded_rows(rows)
        inputs[split] = {"filename": filename, "sha256": file_sha256(path), "rows": len(rows)}

    train_senders = {
        str(row["source"]["private_sender_hash"]) for row in selected["train"]
    }
    dev_senders = {str(row["source"]["private_sender_hash"]) for row in selected["dev"]}
    train_templates = {str(row["source"]["template_group"]) for row in selected["train"]}
    dev_templates = {str(row["source"]["template_group"]) for row in selected["dev"]}
    if train_senders & dev_senders or train_templates & dev_templates:
        raise CandidateSFTError("candidate subset introduced sender/template leakage")

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
        "candidate_implementation": {
            "path": "lfm25/candidates.py",
            "sha256": file_sha256(Path(__file__).with_name("candidates.py")),
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
            "sender_overlap_count": 0,
            "template_overlap_count": 0,
            "sealed_test_rows_materialized": 0,
            "raw_values_emitted_to_stdout": False,
        },
    }
    if dry_run:
        return {"dry_run": True, "wrote_files": False, "report": report}

    paths = {
        split: destination / OUTPUT_FILENAMES[split] for split in ("train", "dev")
    }
    paths["report"] = destination / OUTPUT_FILENAMES["report"]
    if not force and any(path.exists() for path in paths.values()):
        raise CandidateSFTError("candidate SFT outputs exist; pass --force to replace")
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    _atomic_write_jsonl(paths["train"], selected["train"])
    _atomic_write_jsonl(paths["dev"], selected["dev"])
    _atomic_write_json(paths["report"], report)
    for split in ("train", "dev"):
        if file_sha256(paths[split]) != artifacts[split]["sha256"]:
            raise CandidateSFTError(f"written {split} candidate artifact hash mismatch")
    return {"dry_run": False, "wrote_files": True, "report": report}


def safe_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    report = result["report"]
    return {
        "dry_run": result["dry_run"],
        "wrote_files": result["wrote_files"],
        "valid": report["valid"],
        "splits": report["splits"],
        "artifacts": report["artifacts"],
        "invariants": report["invariants"],
    }
