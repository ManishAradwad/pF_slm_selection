"""Aggregate-safe command line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .corpus import build_private_corpus
from .provenance import PrivateArtifactError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PocketFinancer local SMS processing tools")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    corpus = subparsers.add_parser("build-corpus", help="build the ignored canonical corpus")
    corpus.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sms_processing/archive-india-inr.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build-corpus":
            summary = build_private_corpus(args.repo_root, args.config)
            print(json.dumps(summary, sort_keys=True))
            return 0
    except PrivateArtifactError as exc:
        print(json.dumps({"status": "failed", "reason": str(exc)}, sort_keys=True))
        return 2
    return 2
