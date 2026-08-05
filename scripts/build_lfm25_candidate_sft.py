#!/usr/bin/env python3
"""Build private, oracle-covered SFT inputs for the grounded selector track."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lfm25.candidate_sft import (  # noqa: E402
    CandidateSFTError,
    build_candidate_sft,
    safe_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("PRIVATE_DATA/lfm25/private_sft_v3"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("PRIVATE_DATA/lfm25/candidate_sft_v4"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        result = build_candidate_sft(
            repo_root=args.repo_root,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            force=args.force,
        )
    except CandidateSFTError as error:
        parser.error(str(error))
    print(json.dumps(safe_summary(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
