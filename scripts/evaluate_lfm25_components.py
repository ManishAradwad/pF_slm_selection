#!/usr/bin/env python3
"""Evaluate paired annotation/component JSONL without exposing row data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lfm25.component_evaluation import (  # noqa: E402
    ComponentEvaluationError,
    evaluate_jsonl,
)


DEFAULT_CONTRACT = REPO_ROOT / "configs" / "contracts" / "annotation-component-evaluation-v1.json"
OUTPUT_ROOT_NAMES = ("PRIVATE_DATA", "RESULTS")


def _output_path(value: Path, *, repo_root: Path) -> Path:
    candidate = value.resolve(strict=False)
    if candidate.suffix.lower() != ".json":
        raise ComponentEvaluationError("aggregate output must be a .json file")
    for name in OUTPUT_ROOT_NAMES:
        root = (repo_root / name).resolve(strict=False)
        try:
            relative = candidate.relative_to(root)
        except ValueError:
            continue
        if relative.parts:
            return candidate
    raise ComponentEvaluationError("aggregate output must be below PRIVATE_DATA or RESULTS")


def _write_aggregate(path: Path, report: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise ComponentEvaluationError(
            "aggregate output already exists; pass --overwrite to replace it"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            report,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except OSError as error:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass
        raise ComponentEvaluationError("could not write aggregate output") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a local paired annotation/component JSONL file and report "
            "aggregate metrics only."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument(
        "--dry-run",
        action="store_true",
        help="perform read-only validation and print aggregate JSON to stdout",
    )
    destination.add_argument(
        "--output",
        type=Path,
        help="write aggregate JSON below PRIVATE_DATA or RESULTS",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly replace an existing aggregate output file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.dry_run and args.overwrite:
        parser.error("--overwrite requires --output")
    try:
        report = evaluate_jsonl(args.input, contract_path=args.contract)
        if args.dry_run:
            print(
                json.dumps(
                    report,
                    ensure_ascii=True,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
        else:
            output = _output_path(args.output, repo_root=REPO_ROOT)
            _write_aggregate(output, report, overwrite=args.overwrite)
            print(
                json.dumps(
                    {
                        "aggregate_output_written": True,
                        "rows": report["rows"],
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
    except ComponentEvaluationError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
