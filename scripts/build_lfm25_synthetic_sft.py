#!/usr/bin/env python3
"""Generate, privacy-audit, and group-split a local synthetic SFT pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lfm25.public_candidate import (  # noqa: E402
    audit_candidate_rows,
    generate_candidate_rows,
    load_private_text_sources,
)
from lfm25.synthetic_sft import (  # noqa: E402
    DEFAULT_HOLDOUT_FAMILIES,
    split_by_template_family,
    write_synthetic_sft_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an ignored programmatic-only SFT pool with held-out templates."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "PRIVATE_DATA" / "lfm25",
    )
    parser.add_argument("--count", type=int, default=900)
    parser.add_argument("--seed", type=int, default=25_052_027)
    parser.add_argument(
        "--holdout-family",
        action="append",
        dest="holdout_families",
        help="Whole template family assigned to dev; repeatable.",
    )
    parser.add_argument("--private-jsonl", action="append", type=Path)
    parser.add_argument("--private-text-field", default="sms")
    parser.add_argument("--private-sqlite", action="append", type=Path)
    parser.add_argument("--private-export", action="append", type=Path)
    parser.add_argument("--private-export-text-field", default="text")
    parser.add_argument("--private-sqlite-table", default="message")
    parser.add_argument("--private-sqlite-text-column", default="text")
    parser.add_argument("--similarity-threshold", type=float, default=0.72)
    parser.add_argument("--ngram-size", type=int, default=4)
    parser.add_argument("--rewrite-attempts", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.private_jsonl and not args.private_export and not args.private_sqlite:
        parser.error("at least one private source is required for leakage auditing")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = generate_candidate_rows(count=args.count, seed=args.seed)
    private_texts = load_private_text_sources(
        jsonl_paths=args.private_jsonl or [],
        export_paths=args.private_export or [],
        sqlite_paths=args.private_sqlite or [],
        jsonl_text_field=args.private_text_field,
        export_text_field=args.private_export_text_field,
        sqlite_table=args.private_sqlite_table,
        sqlite_text_column=args.private_sqlite_text_column,
    )
    audit = audit_candidate_rows(
        rows,
        private_texts,
        similarity_threshold=args.similarity_threshold,
        ngram_size=args.ngram_size,
        rewrite_attempts=args.rewrite_attempts,
    )
    if audit.report["rejected_candidate_rows"]:
        raise RuntimeError("synthetic SFT audit rejected rows; inspect aggregate report")
    if len(audit.accepted_rows) != len(rows):
        raise RuntimeError("synthetic SFT audit did not accept the complete generated pool")
    holdouts = tuple(args.holdout_families or DEFAULT_HOLDOUT_FAMILIES)
    train, dev = split_by_template_family(audit.accepted_rows, holdouts)
    paths = write_synthetic_sft_artifacts(
        train,
        dev,
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        seed=args.seed,
        generated_rows=len(rows),
        audit_report=audit.report,
        holdout_families=holdouts,
        force=args.force,
    )
    similarity = audit.report["private_similarity_audit"]
    print(
        json.dumps(
            {
                "dataset_state": "local_training_only",
                "release_authorized": False,
                "train_rows": len(train),
                "dev_rows": len(dev),
                "train_template_families": len({row["template_family"] for row in train}),
                "dev_template_families": len({row["template_family"] for row in dev}),
                "private_documents_compared_in_memory": similarity[
                    "private_documents_compared_in_memory"
                ],
                "rows_rewritten": similarity["rows_rewritten"],
                "artifacts": [path.name for path in paths],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
