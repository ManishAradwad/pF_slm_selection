#!/usr/bin/env python3
"""Audit a synthetic candidate against local private texts without logging them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lfm25.public_candidate import (  # noqa: E402
    DEFAULT_NGRAM_SIZE,
    DEFAULT_SIMILARITY_THRESHOLD,
    audit_candidate_rows,
    ensure_ignored_output_tree,
    load_private_text_sources,
    read_candidate_jsonl,
    write_audit_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit an unreleased synthetic candidate using aggregate-only results."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=REPO_ROOT / "PUBLIC_CANDIDATE" / "lfm25" / "candidate_unreviewed.jsonl",
    )
    parser.add_argument(
        "--private-jsonl",
        action="append",
        type=Path,
        help="Local private JSONL used read-only; may be provided more than once.",
    )
    parser.add_argument("--private-text-field", default="sms")
    parser.add_argument(
        "--private-export",
        action="append",
        type=Path,
        help="Complete private JSON-array or CSV export; may be repeated.",
    )
    parser.add_argument(
        "--private-export-text-field",
        default="text",
        help="Text field/column in --private-export sources (default: text).",
    )
    parser.add_argument(
        "--private-sqlite",
        action="append",
        type=Path,
        help="Local private SQLite archive opened immutable/read-only; repeatable.",
    )
    parser.add_argument(
        "--private-sqlite-table",
        default="message",
        help="SQLite table containing the private text (default: message).",
    )
    parser.add_argument(
        "--private-sqlite-text-column",
        default="text",
        help="SQLite text column to read (default: text).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "PUBLIC_CANDIDATE" / "lfm25",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
    )
    parser.add_argument("--ngram-size", type=int, default=DEFAULT_NGRAM_SIZE)
    parser.add_argument("--rewrite-attempts", type=int, default=2)
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace only this auditor's existing ignored artifacts.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.private_jsonl and not args.private_export and not args.private_sqlite:
        parser.error(
            "at least one --private-jsonl, --private-export, or --private-sqlite source is required"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_ignored_output_tree(args.repo_root, args.output_dir)
    rows = read_candidate_jsonl(args.candidate)
    private_texts = load_private_text_sources(
        jsonl_paths=args.private_jsonl or [],
        export_paths=args.private_export or [],
        sqlite_paths=args.private_sqlite or [],
        jsonl_text_field=args.private_text_field,
        export_text_field=args.private_export_text_field,
        sqlite_table=args.private_sqlite_table,
        sqlite_text_column=args.private_sqlite_text_column,
    )
    bundle = audit_candidate_rows(
        rows,
        private_texts,
        similarity_threshold=args.similarity_threshold,
        ngram_size=args.ngram_size,
        rewrite_attempts=args.rewrite_attempts,
    )
    paths = write_audit_artifacts(
        bundle,
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        preview_count=args.preview_count,
        force=args.force,
    )
    similarity = bundle.report["private_similarity_audit"]
    summary = {
        "candidate_state": "unreleased",
        "manual_review": "pending",
        "accepted_rows": bundle.report["accepted_candidate_rows"],
        "rejected_rows": bundle.report["rejected_candidate_rows"],
        "rewritten_rows": similarity["rows_rewritten"],
        "private_documents_compared_in_memory": similarity["private_documents_compared_in_memory"],
        "artifacts": [path.name for path in paths],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
