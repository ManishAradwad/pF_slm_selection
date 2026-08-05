#!/usr/bin/env python3
"""Build an ignored, unreleased synthetic candidate pool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lfm25.public_candidate import (  # noqa: E402
    DEFAULT_ROW_COUNT,
    DEFAULT_SEED,
    generate_candidate_rows,
    validate_candidate_row,
    write_generation_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a fully synthetic, unreleased LFM2.5 SMS candidate pool."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "PUBLIC_CANDIDATE" / "lfm25",
    )
    parser.add_argument("--count", type=int, default=DEFAULT_ROW_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace only this generator's existing ignored artifacts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = generate_candidate_rows(count=args.count, seed=args.seed)
    invalid_count = sum(bool(validate_candidate_row(row)) for row in rows)
    if invalid_count:
        raise RuntimeError(f"generated row validation failed for {invalid_count} rows")
    paths = write_generation_artifacts(
        rows,
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        seed=args.seed,
        force=args.force,
    )
    summary = {
        "candidate_state": "unreleased",
        "manual_review": "pending",
        "rows_generated": len(rows),
        "artifacts": [path.name for path in paths],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
