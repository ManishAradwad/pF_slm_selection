#!/usr/bin/env python3
"""Aggregate-only strict scoring for lm-eval samples or the always-null baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lfm25.metrics import paired_exact_comparison, score_records  # noqa: E402

PROVENANCE_SCHEMA = "lfm25-baseline-provenance-v1"
DEFAULT_DATASET = REPO_ROOT / "DATA" / "extraction_ds.jsonl"
FILTER_PROVENANCE_FILES = (
    ("task_config", REPO_ROOT / "DATA" / "sms_extraction.yaml"),
    ("filter_implementation", REPO_ROOT / "DATA" / "utils.py"),
)
SCORER_PROVENANCE_FILES = (
    ("entrypoint", Path(__file__).resolve()),
    ("metrics", REPO_ROOT / "lfm25" / "metrics.py"),
    ("contract", REPO_ROOT / "lfm25" / "contract.py"),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _file_record(path: Path, role: str) -> dict[str, str]:
    return {
        "role": role,
        "path": _display_path(path),
        "sha256": _sha256_file(path),
    }


def _manifest_sha256(
    kind: str,
    metadata: dict[str, Any],
    files: list[dict[str, str]],
) -> str:
    payload = {
        "files": files,
        "kind": kind,
        "metadata": metadata,
        "schema": PROVENANCE_SCHEMA,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_provenance(
    *,
    always_null: Path | None,
    samples: list[Path] | None,
    dataset: Path,
    selected_filter: str,
) -> dict[str, Any]:
    if always_null is not None:
        source_kind = "always_null_input"
        source_files = [_file_record(always_null, "always_null_input")]
        filter_name = "always_null_literal"
        filter_stage = "generated_by_scorer"
        filter_files = [_file_record(Path(__file__).resolve(), "implementation")]
    else:
        assert samples
        source_kind = "lm_eval_samples"
        source_files = [
            _file_record(path, f"sample_{index}") for index, path in enumerate(samples)
        ]
        filter_name = selected_filter
        filter_stage = "upstream_filtered_resps"
        filter_files = [_file_record(path, role) for role, path in FILTER_PROVENANCE_FILES]

    source_metadata = {"file_count": len(source_files)}
    source = {
        "kind": source_kind,
        "files": source_files,
        "sha256": _manifest_sha256(source_kind, source_metadata, source_files),
    }
    dataset_record = _file_record(dataset, "evaluation_dataset")
    dataset_provenance = {
        "path": dataset_record["path"],
        "sha256": dataset_record["sha256"],
    }
    filter_metadata = {"name": filter_name, "stage": filter_stage}
    filter_provenance = {
        **filter_metadata,
        "files": filter_files,
        "sha256": _manifest_sha256("filter", filter_metadata, filter_files),
    }
    scorer_files = [_file_record(path, role) for role, path in SCORER_PROVENANCE_FILES]
    scorer_metadata = {"entrypoint": "scripts/score_lfm25_results.py"}
    scorer = {
        "files": scorer_files,
        "sha256": _manifest_sha256("scorer", scorer_metadata, scorer_files),
    }
    return {
        "schema": PROVENANCE_SCHEMA,
        "hash_algorithm": "sha256",
        "source": source,
        "dataset": dataset_provenance,
        "filter": filter_provenance,
        "scorer": scorer,
    }


def _load_samples(path: Path, selected_filter: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("filter") != selected_filter:
            continue
        responses = row.get("filtered_resps") or []
        records.append(
            {
                "id": row.get("doc_id"),
                "gold": row.get("target"),
                "prediction": responses[0] if responses else "",
            }
        )
    if not records:
        raise ValueError(f"no rows for filter {selected_filter!r} in {path}")
    return records


def _load_always_null(path: Path) -> list[dict[str, Any]]:
    records = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        records.append(
            {
                "id": index,
                "gold": row.get("expected"),
                "prediction": "null",
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--samples", action="append", type=Path)
    source.add_argument("--always-null", type=Path)
    parser.add_argument("--filter", default="extract_json_nonnull")
    parser.add_argument(
        "--dataset",
        type=Path,
        help=(
            "Dataset whose gold rows produced the samples. Defaults to the always-null "
            "input for --always-null and DATA/extraction_ds.jsonl for --samples."
        ),
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    named_scores: dict[str, dict[str, Any]] = {}
    if args.always_null:
        records = _load_always_null(args.always_null)
        named_scores["always_null"] = score_records(records, include_per_example=True)
    else:
        assert args.samples
        for path in args.samples:
            records = _load_samples(path, args.filter)
            named_scores[path.parent.parent.parent.name + ":" + path.parent.parent.name] = score_records(
                records, include_per_example=True
            )

    dataset = args.dataset
    if dataset is None:
        dataset = args.always_null if args.always_null is not None else DEFAULT_DATASET
    report: dict[str, Any] = {
        "provenance": _build_provenance(
            always_null=args.always_null,
            samples=args.samples,
            dataset=dataset,
            selected_filter=args.filter,
        ),
        "variants": {},
    }
    for name, score in named_scores.items():
        report["variants"][name] = {key: value for key, value in score.items() if key != "_per_example"}
    if len(named_scores) == 2:
        first_name, second_name = named_scores
        report["paired_exact"] = {
            "first": first_name,
            "second": second_name,
            **paired_exact_comparison(named_scores[first_name], named_scores[second_name]),
        }

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
