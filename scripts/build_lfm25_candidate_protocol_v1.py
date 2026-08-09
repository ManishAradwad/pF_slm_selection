#!/usr/bin/env python3
"""Build private, source-grounded Candidate Protocol V1 train/dev rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lfm25.candidate_protocol_data import (  # noqa: E402
    CandidateProtocolDataError,
    build_candidate_protocol_data,
    safe_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--implementation-root",
        type=Path,
        help="root containing the Candidate V1 implementation to fingerprint",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("PRIVATE_DATA/lfm25/pocketfinancer-android-a9b7df4-direct-v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("PRIVATE_DATA/lfm25/pocketfinancer-candidate-protocol-v1"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        result = build_candidate_protocol_data(
            repo_root=args.repo_root,
            implementation_root=args.implementation_root,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            force=args.force,
        )
    except CandidateProtocolDataError as error:
        parser.error(str(error))
    print(json.dumps(safe_summary(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
