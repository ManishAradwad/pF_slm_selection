#!/usr/bin/env python3
"""Operate the local-only PocketFinancer Workbench V2 without row-level output."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from lfm25.evaluation_v2 import EvaluationV2Error, score_evaluation_package  # noqa: E402
from lfm25.workbench_v2 import (  # noqa: E402
    PRIVATE_DATA_ROOT,
    WorkbenchV2Error,
    anonymize_annotation_package,
    create_annotation_package,
    read_annotation_package,
    validate_annotation_package,
    write_aggregate_report,
    write_private_artifact,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create, validate, anonymize, or score local Semantic V2 packages. "
            "Raw and row-level material is restricted to PRIVATE_DATA; aggregate reports "
            "are restricted to RESULTS."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Create an empty private annotation package")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--package-id", required=True)
    create.add_argument("--created-at", required=True, help="UTC RFC3339 timestamp ending in Z")
    create.add_argument("--annotation-policy-id", required=True)
    create.add_argument("--annotation-policy-version", type=int, default=1)
    create.add_argument("--source-revision", required=True)
    create.add_argument("--source-digest-sha256", required=True)

    validate = subparsers.add_parser("validate", help="Validate without printing row content")
    validate.add_argument("path", type=Path)
    validate.add_argument(
        "--invented-fixture",
        action="store_true",
        help="Allow only the exact committed invented Phase B fixture outside PRIVATE_DATA",
    )

    anonymize = subparsers.add_parser(
        "anonymize", help="Create a still-local row-level categorical export"
    )
    anonymize.add_argument("path", type=Path)
    anonymize.add_argument("--output", type=Path, required=True)
    anonymize.add_argument("--export-nonce")
    anonymize.add_argument(
        "--secret-env",
        default="PF_WORKBENCH_V2_HMAC_KEY",
        help="Environment variable containing at least 32 UTF-8 bytes; never pass the key as an argument",
    )

    score = subparsers.add_parser("score", help="Write an aggregate-only Semantic V2 report")
    score.add_argument("--annotations", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--split", default="protected_test")
    return parser


def _safe_print(value: Any) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _read_private_mapping(path: Path) -> Mapping[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(PRIVATE_DATA_ROOT.resolve())
    except ValueError as error:
        raise WorkbenchV2Error("row-level input must remain under PRIVATE_DATA") from error

    def reject_constant(_value: str) -> None:
        raise WorkbenchV2Error("prediction package contains a non-finite JSON constant")

    try:
        value = json.loads(resolved.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, json.JSONDecodeError) as error:
        raise WorkbenchV2Error("prediction package is unavailable or invalid JSON") from error
    if not isinstance(value, Mapping):
        raise WorkbenchV2Error("prediction package root must be an object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "create":
            package = create_annotation_package(
                package_id=arguments.package_id,
                created_at=arguments.created_at,
                annotation_policy_id=arguments.annotation_policy_id,
                annotation_policy_version=arguments.annotation_policy_version,
                source_revision=arguments.source_revision,
                source_digest_sha256=arguments.source_digest_sha256,
            )
            write_private_artifact(arguments.output, package)
            _safe_print({"status": "created", "counts": {"rows": 0}})
            return 0

        if arguments.command == "validate":
            package = read_annotation_package(
                arguments.path,
                allow_synthetic_fixture=arguments.invented_fixture,
            )
            summary = validate_annotation_package(package)
            _safe_print(
                {
                    "status": "valid",
                    "contract_id": summary["contract_id"],
                    "contract_version": summary["contract_version"],
                    "privacy_classification": summary["privacy_classification"],
                    "counts": summary["counts"],
                    "split_counts": summary["split_counts"],
                    "package_sha256": summary["package_sha256"],
                }
            )
            return 0

        if arguments.command == "anonymize":
            secret_value = os.environ.get(arguments.secret_env)
            if secret_value is None:
                raise WorkbenchV2Error("anonymization secret environment variable is not set")
            package = read_annotation_package(arguments.path)
            anonymized = anonymize_annotation_package(
                package,
                secret=secret_value.encode("utf-8"),
                export_nonce=arguments.export_nonce,
            )
            write_private_artifact(arguments.output, anonymized)
            _safe_print({"status": "anonymized", "counts": {"rows": len(anonymized["rows"])}})
            return 0

        annotations = read_annotation_package(arguments.annotations)
        predictions = _read_private_mapping(arguments.predictions)
        report = score_evaluation_package(annotations, predictions, split=arguments.split)
        write_aggregate_report(arguments.output, report)
        _safe_print(
            {
                "status": "scored",
                "profile_id": report["profile"]["profile_id"],
                "resolved_rows": report["sample"]["resolved_rows"],
                "false_automatic_posts": report["safety"]["false_automatic_posts"],
            }
        )
        return 0
    except (WorkbenchV2Error, EvaluationV2Error) as error:
        print(f"Workbench V2 failed safely: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
