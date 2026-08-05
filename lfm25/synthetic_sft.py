"""Build ignored, programmatic-only SFT splits without private-row derivation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from .public_candidate import validate_candidate_row


DEFAULT_HOLDOUT_FAMILIES = (
    "merchant_refund",
    "utility_bill",
    "failed_payment",
    "bill_due",
)
SFT_PROVENANCE = {
    "label_source": "programmatic_synthetic",
    "human_gold": False,
    "consensus_silver": False,
    "private_row_derived": False,
}


def split_by_template_family(
    rows: Sequence[Mapping[str, Any]],
    holdout_families: Sequence[str] = DEFAULT_HOLDOUT_FAMILIES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Place complete template families in train or unseen-template dev."""

    holdouts = frozenset(holdout_families)
    if not holdouts:
        raise ValueError("at least one held-out template family is required")
    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    for row in rows:
        errors = validate_candidate_row(row)
        if errors:
            raise ValueError("synthetic source row violates the candidate contract")
        prepared = dict(row)
        prepared["sft_label_provenance"] = dict(SFT_PROVENANCE)
        if str(row["template_family"]) in holdouts:
            dev.append(prepared)
        else:
            train.append(prepared)
    if not train or not dev:
        raise ValueError("template-family split produced an empty partition")
    train_families = {str(row["template_family"]) for row in train}
    dev_families = {str(row["template_family"]) for row in dev}
    if train_families & dev_families:
        raise RuntimeError("template-family leakage between train and dev")
    missing = holdouts - dev_families
    if missing:
        raise ValueError(f"held-out template families were not generated: {sorted(missing)}")
    return train, dev


def ensure_ignored_private_output(repo_root: Path, output_dir: Path) -> None:
    """Fail closed unless generated SFT artifacts stay in ignored private storage."""

    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    required_root = (repo_root / "PRIVATE_DATA" / "lfm25").resolve()
    try:
        output_dir.relative_to(required_root)
    except ValueError as error:
        raise RuntimeError("output must stay under PRIVATE_DATA/lfm25") from error
    probe = (output_dir / "synthetic_sft_train.jsonl").relative_to(repo_root)
    ignored = subprocess.run(
        ["git", "-C", str(repo_root), "check-ignore", "--no-index", "-q", "--", probe.as_posix()],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if ignored.returncode != 0:
        raise RuntimeError("PRIVATE_DATA/lfm25 is not protected by .gitignore")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def write_synthetic_sft_artifacts(
    train: Sequence[Mapping[str, Any]],
    dev: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path,
    output_dir: Path,
    seed: int,
    generated_rows: int,
    audit_report: Mapping[str, Any],
    holdout_families: Sequence[str],
    force: bool = False,
) -> tuple[Path, Path, Path]:
    """Atomically write ignored training data plus an aggregate provenance manifest."""

    ensure_ignored_private_output(repo_root, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        output_dir.chmod(0o700)
    except OSError:
        pass
    train_path = output_dir / "synthetic_sft_train.jsonl"
    dev_path = output_dir / "synthetic_sft_dev.jsonl"
    manifest_path = output_dir / "synthetic_sft_manifest.json"
    paths = (train_path, dev_path, manifest_path)
    if not force and any(path.exists() for path in paths):
        raise FileExistsError("refusing to overwrite existing synthetic SFT artifacts")

    train_content = _jsonl_bytes(train)
    dev_content = _jsonl_bytes(dev)
    train_families = sorted({str(row["template_family"]) for row in train})
    dev_families = sorted({str(row["template_family"]) for row in dev})
    manifest = {
        "dataset_state": "local_training_only",
        "release_authorized": False,
        "seed": seed,
        "generated_rows": generated_rows,
        "accepted_rows": len(train) + len(dev),
        "train_rows": len(train),
        "dev_rows": len(dev),
        "train_template_families": train_families,
        "dev_template_families": dev_families,
        "holdout_template_families": sorted(holdout_families),
        "template_family_overlap": sorted(set(train_families) & set(dev_families)),
        "label_provenance": dict(SFT_PROVENANCE),
        "train_sha256": _sha256(train_content),
        "dev_sha256": _sha256(dev_content),
        "privacy_audit": dict(audit_report),
    }
    contents = (
        train_content,
        dev_content,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    temporaries: list[Path] = []
    try:
        for path, content in zip(paths, contents):
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            temporary.write_bytes(content)
            try:
                temporary.chmod(0o600)
            except OSError:
                pass
            temporaries.append(temporary)
        for temporary, path in zip(temporaries, paths):
            os.replace(temporary, path)
    finally:
        for temporary in temporaries:
            if temporary.exists():
                temporary.unlink()
    return paths
