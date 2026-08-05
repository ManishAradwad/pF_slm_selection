#!/usr/bin/env python3
"""Build a low-weight semantic-ID curriculum mixed with private selector rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lfm25.candidate_curriculum import (  # noqa: E402
    CURRICULUM_VERSION,
    audit_curriculum_overlap,
    generate_candidate_curriculum,
    mix_private_and_curriculum,
)
from lfm25.private_data import (  # noqa: E402
    _atomic_write_json,
    _atomic_write_jsonl,
    ensure_within,
    file_sha256,
    read_jsonl,
    require_private_ignore,
)


def _rows_sha256(rows) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--private-train",
        type=Path,
        default=Path("PRIVATE_DATA/lfm25/candidate_sft_v4/candidate_sft_train.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("PRIVATE_DATA/lfm25/candidate_curriculum_v2"),
    )
    parser.add_argument("--rows-per-relation", type=int, default=20)
    parser.add_argument("--rows-per-negative", type=int, default=10)
    parser.add_argument("--sample-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=35_025)
    parser.add_argument(
        "--decontamination-corpus",
        action="append",
        type=Path,
        help="JSONL corpus checked for aggregate exact/template overlap; repeatable",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = REPO_ROOT.resolve()
    private_root = (repo / "PRIVATE_DATA" / "lfm25").resolve()
    source = ensure_within(repo / args.private_train, private_root)
    output = ensure_within(repo / args.output_dir, private_root)
    require_private_ignore(repo, private_root)
    if not source.is_file() or source.name != "candidate_sft_train.jsonl":
        parser.error("private train input must be the fixed candidate SFT artifact")
    curriculum = generate_candidate_curriculum(
        seed=args.seed,
        rows_per_relation=args.rows_per_relation,
        rows_per_negative=args.rows_per_negative,
        sample_weight=args.sample_weight,
    )
    mixed, counts = mix_private_and_curriculum(
        read_jsonl(source), curriculum, seed=args.seed
    )
    corpus_paths = args.decontamination_corpus or [
        Path("DATA/extraction_ds.jsonl"),
        Path("PRIVATE_DATA/lfm25/split_manifest.jsonl"),
    ]
    references = {}
    reference_inputs = {}
    for corpus_path in corpus_paths:
        resolved = ensure_within(repo / corpus_path, repo)
        if not resolved.is_file():
            parser.error(f"missing decontamination corpus: {corpus_path}")
        name = str(resolved.relative_to(repo))
        references[name] = read_jsonl(resolved)
        reference_inputs[name] = {
            "rows": len(references[name]),
            "sha256": file_sha256(resolved),
        }
    decontamination = audit_curriculum_overlap(curriculum, references)
    if (
        decontamination["exact_overlap_count"]
        or decontamination["normalized_template_overlap_count"]
    ):
        parser.error("candidate curriculum overlaps a decontamination corpus")
    report = {
        "curriculum_version": CURRICULUM_VERSION,
        "valid": True,
        "private_local_only": True,
        "release_authorized": False,
        "config": {
            "seed": args.seed,
            "rows_per_relation": args.rows_per_relation,
            "rows_per_negative": args.rows_per_negative,
            "sample_weight": args.sample_weight,
        },
        "counts": counts,
        "input": {"sha256": file_sha256(source)},
        "decontamination_inputs": reference_inputs,
        "decontamination": decontamination,
        "code": {
            "candidate_curriculum_sha256": file_sha256(
                REPO_ROOT / "lfm25" / "candidate_curriculum.py"
            ),
            "candidates_sha256": file_sha256(REPO_ROOT / "lfm25" / "candidates.py"),
        },
        "artifacts": {
            "synthetic": {
                "filename": "candidate_curriculum.jsonl",
                "rows": len(curriculum),
                "sha256": _rows_sha256(curriculum),
            },
            "mixed_train": {
                "filename": "candidate_mixed_train.jsonl",
                "rows": len(mixed),
                "sha256": _rows_sha256(mixed),
            },
        },
        "privacy": {
            "programmatic_curriculum_private_derived": False,
            "raw_values_emitted_to_stdout": False,
            "hosted_services_used": False,
        },
    }
    summary = {
        "dry_run": args.dry_run,
        "wrote_files": not args.dry_run,
        "valid": True,
        "counts": counts,
        "artifacts": report["artifacts"],
        "decontamination": decontamination,
        "privacy": report["privacy"],
    }
    if not args.dry_run:
        if output.exists():
            parser.error("curriculum output already exists")
        output.mkdir(parents=True, mode=0o700)
        _atomic_write_jsonl(output / "candidate_curriculum.jsonl", curriculum)
        _atomic_write_jsonl(output / "candidate_mixed_train.jsonl", mixed)
        _atomic_write_json(output / "candidate_curriculum_report.json", report)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
