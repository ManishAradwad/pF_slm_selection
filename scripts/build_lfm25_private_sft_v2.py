#!/usr/bin/env python3
"""Build the local-only v2 private SFT train/dev artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lfm25.private_sft_v2 import (  # noqa: E402
    BuildConfig,
    PrivateSFTV2Error,
    run_private_sft_v2,
    safe_console_summary,
)
from lfm25.private_data import PrivateDataError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a privacy-guarded v2 SFT set from the already-local split manifest. "
            "Console output contains aggregates and hashes only."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("PRIVATE_DATA/lfm25/split_manifest.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("PRIVATE_DATA/lfm25/private_sft_v3"),
    )
    parser.add_argument("--dev-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=25_052_027)
    parser.add_argument("--minimum-silver-confidence", type=float, default=0.86)
    parser.add_argument("--max-per-template", type=int, default=8)
    parser.add_argument("--max-per-category", type=int, default=512)
    parser.add_argument("--max-null-to-transaction-ratio", type=float, default=1.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all checks and print aggregate counts/hashes without writing artifacts.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Atomically replace only the four fixed v2 outputs inside the private directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = BuildConfig(
        dev_fraction=args.dev_fraction,
        seed=args.seed,
        minimum_silver_confidence=args.minimum_silver_confidence,
        max_per_template=args.max_per_template,
        max_per_category=args.max_per_category,
        max_null_to_transaction_ratio=args.max_null_to_transaction_ratio,
    )
    try:
        value = run_private_sft_v2(
            repo_root=args.repo_root,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            config=config,
            dry_run=args.dry_run,
            force=args.force,
        )
    except (PrivateSFTV2Error, PrivateDataError) as error:
        parser.error(str(error))
    print(json.dumps(safe_console_summary(value), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
