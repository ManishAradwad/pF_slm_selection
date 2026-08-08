#!/usr/bin/env python3
"""Export, validate, and import the local reviewer-blind private test package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lfm25.blinded_review import (  # noqa: E402
    DEFAULT_IMPORT_REPORT,
    DEFAULT_MAPPING_FILE,
    DEFAULT_METADATA_FILE,
    DEFAULT_REVIEWED_MANIFEST,
    DEFAULT_REVIEW_FILE,
    DEFAULT_SOURCE_MANIFEST,
    DEFAULT_WORKBENCH_DB,
    run_export,
    run_import,
    run_validate,
    safe_console_summary,
)
from lfm25.private_data import PrivateDataError  # noqa: E402


def _path(value: str) -> Path:
    return Path(value)


def _add_package_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--manifest",
        type=_path,
        default=DEFAULT_SOURCE_MANIFEST,
        help="Frozen source split manifest below PRIVATE_DATA/lfm25.",
    )
    parser.add_argument(
        "--review-file",
        type=_path,
        default=DEFAULT_REVIEW_FILE,
        help="Reviewer-facing JSONL below PRIVATE_DATA/lfm25.",
    )
    parser.add_argument(
        "--mapping-file",
        type=_path,
        default=DEFAULT_MAPPING_FILE,
        help="Internal ID mapping below PRIVATE_DATA/lfm25; do not give this to reviewers.",
    )
    parser.add_argument(
        "--metadata-file",
        type=_path,
        default=DEFAULT_METADATA_FILE,
        help="Frozen package provenance below PRIVATE_DATA/lfm25.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Local-only blinded test adjudication. Console output contains aggregate counts only."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser(
        "export",
        help="Export all and only frozen test rows with blank human labels.",
    )
    _add_package_paths(export)
    export.add_argument(
        "--force",
        action="store_true",
        help="Explicitly replace nonempty package outputs, including any in-progress review file.",
    )

    validate = subparsers.add_parser(
        "validate",
        help="Validate a complete or resumable partial annotation file without writing outputs.",
    )
    _add_package_paths(validate)

    import_parser = subparsers.add_parser(
        "import",
        help="Validate annotations and write a separate reviewed manifest and aggregate report.",
    )
    _add_package_paths(import_parser)
    import_parser.add_argument(
        "--reviewed-manifest",
        type=_path,
        default=DEFAULT_REVIEWED_MANIFEST,
        help="Separate reviewed-manifest output below PRIVATE_DATA/lfm25.",
    )
    import_parser.add_argument(
        "--report",
        type=_path,
        default=DEFAULT_IMPORT_REPORT,
        help="Aggregate-only import report below PRIVATE_DATA/lfm25.",
    )
    import_parser.add_argument(
        "--workbench-db",
        type=_path,
        default=DEFAULT_WORKBENCH_DB,
        help="Completed local workbench database below PRIVATE_DATA/lfm25.",
    )
    import_parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly replace nonempty reviewed-manifest and report outputs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    common = {
        "source_manifest": arguments.manifest,
        "review_file": arguments.review_file,
        "mapping_file": arguments.mapping_file,
        "metadata_file": arguments.metadata_file,
    }
    try:
        if arguments.command == "export":
            report = run_export(REPO_ROOT, force=arguments.force, **common)
        elif arguments.command == "validate":
            report = run_validate(REPO_ROOT, **common)
        else:
            report = run_import(
                REPO_ROOT,
                reviewed_manifest=arguments.reviewed_manifest,
                import_report=arguments.report,
                workbench_db=arguments.workbench_db,
                force=arguments.force,
                **common,
            )
    except PrivateDataError as exc:
        parser.exit(2, f"blinded-review command failed: {exc}\n")
    except OSError:
        parser.exit(2, "blinded-review command failed: a local artifact could not be read or written\n")
    print(json.dumps(safe_console_summary(report), sort_keys=True))
    return 0 if report.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
