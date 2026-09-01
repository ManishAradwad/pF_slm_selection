#!/usr/bin/env python3
"""Validate aggregate Android runtime evidence without row-level output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lfm25.android_runtime_evidence import (  # noqa: E402
    AndroidRuntimeEvidenceError,
    read_runtime_evidence_package,
    validate_runtime_evidence_package,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate aggregate-only host or Android-device runtime evidence"
        )
    )
    parser.add_argument("path", type=Path)
    parser.add_argument(
        "--invented-fixture",
        action="store_true",
        help=(
            "allow only the exact committed invented Phase C fixture "
            "outside RESULTS"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        package = read_runtime_evidence_package(
            arguments.path,
            allow_synthetic_fixture=arguments.invented_fixture,
        )
        summary = validate_runtime_evidence_package(package)
    except AndroidRuntimeEvidenceError as error:
        print(
            f"Android runtime evidence failed safely: {error}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
