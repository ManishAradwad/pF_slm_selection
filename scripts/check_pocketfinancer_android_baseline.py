#!/usr/bin/env python3
"""Verify or hash the immutable PocketFinancer Android Phase C baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lfm25.android_baseline import (  # noqa: E402
    AndroidBaselineError,
    capture_committed_source_hashes,
    load_android_baseline,
    verify_android_baseline,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the immutable, source-only Phase C Android baseline"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(
            "configs/baselines/pocketfinancer-android-552ffbdf-phase-c.json"
        ),
    )
    parser.add_argument("--android-repo", type=Path, required=True)
    parser.add_argument(
        "--capture-source-hashes",
        action="store_true",
        help=(
            "print committed source hashes named by the baseline "
            "(bootstrap/review only)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    baseline_path = (
        arguments.baseline
        if arguments.baseline.is_absolute()
        else REPOSITORY_ROOT / arguments.baseline
    )
    try:
        if arguments.capture_source_hashes:
            baseline = load_android_baseline(baseline_path)
            report = {
                "revision": baseline["source_snapshot"]["revision"],
                "files_sha256": capture_committed_source_hashes(
                    baseline,
                    arguments.android_repo,
                ),
            }
        else:
            report = verify_android_baseline(
                baseline_path,
                repository_root=REPOSITORY_ROOT,
                android_repo=arguments.android_repo,
            )
    except AndroidBaselineError as error:
        print(f"Android baseline check failed safely: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
