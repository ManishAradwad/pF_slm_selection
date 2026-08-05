#!/usr/bin/env python3
"""Prepare, validate, or materialize the local private LFM2.5 SMS corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lfm25.private_data import (  # noqa: E402
    PrivateDataError,
    load_config,
    run_materialize,
    run_prepare,
    run_validate,
    safe_summary,
)


def _private_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local-only LFM2.5 private data preparation (aggregate console output only)."
    )
    parser.add_argument(
        "--config",
        type=_private_path,
        default=REPO_ROOT / "configs" / "lfm25_data.json",
        help="Private-data policy configuration.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Decontaminate, split, propose silver labels, and emit a review queue.",
    )
    prepare.add_argument(
        "--source",
        help="Override with a local all_sms.json or all_sms.csv path.",
    )
    prepare.add_argument("--force", action="store_true", help="Atomically replace private outputs.")

    validate = subparsers.add_parser(
        "validate",
        help="Re-run aggregate-only manifest validation.",
    )
    validate.add_argument("--manifest", type=_private_path, help="Private manifest override.")

    validate.add_argument(
        "--source",
        help="Override the local all_sms.json or all_sms.csv used for recomputation.",
    )
    materialize = subparsers.add_parser(
        "materialize",
        help="Create assistant-last SFT JSONL from accepted gold/silver rows only.",
    )
    materialize.add_argument("--manifest", type=_private_path, help="Private manifest override.")
    materialize.add_argument(
        "--output-dir",
        type=_private_path,
        help="Output directory; it must remain below PRIVATE_DATA/lfm25.",
    )
    materialize.add_argument(
        "--force",
        action="store_true",
        help="Atomically replace existing SFT outputs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        config = load_config(arguments.config)
        if arguments.command == "prepare":
            report = run_prepare(
                REPO_ROOT,
                config,
                source_override=arguments.source,
                force=arguments.force,
            )
        elif arguments.command == "validate":
            report = run_validate(
                REPO_ROOT,
                config,
                manifest_override=arguments.manifest,
                source_override=arguments.source,
            )
        else:
            report = run_materialize(
                REPO_ROOT,
                config,
                manifest_override=arguments.manifest,
                output_dir_override=arguments.output_dir,
                force=arguments.force,
            )
    except PrivateDataError as exc:
        parser.exit(2, f"private-data command failed: {exc}\n")
    print(json.dumps(safe_summary(report), sort_keys=True))
    return 0 if report.get("valid", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
