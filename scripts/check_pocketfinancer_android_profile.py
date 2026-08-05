#!/usr/bin/env python3
"""Check the selection repo's app profile against a local Android checkout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lfm25.android_profile_sync import (  # noqa: E402
    AndroidProfileError,
    verify_current_android_profile,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the pinned PocketFinancer Android source profile"
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("configs/contracts/pocketfinancer-android-current.json"),
    )
    parser.add_argument("--android-repo", type=Path)
    parser.add_argument(
        "--allow-head-mismatch",
        action="store_true",
        help="verify the pinned revision blobs without requiring checkout HEAD to match",
    )
    args = parser.parse_args()
    profile = args.profile if args.profile.is_absolute() else REPO_ROOT / args.profile
    try:
        report = verify_current_android_profile(
            profile,
            android_repo=args.android_repo,
            require_head=not args.allow_head_mismatch,
        )
    except AndroidProfileError as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
