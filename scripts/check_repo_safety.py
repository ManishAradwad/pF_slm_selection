#!/usr/bin/env python3
"""Reject private or generated artifacts that have been added to Git.

The check deliberately examines only pathnames returned by ``git ls-files``.
It never opens a dataset, database, result, or model file.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import NamedTuple


# These legacy tracked artifacts predate the guard and remain untouched here. The
# exceptions are exact and case-sensitive so similarly named files cannot bypass
# the policy. Their presence requires separate review before repository publication.
GRANDFATHERED_PATHS = frozenset(
    {"DATA/extraction_ds.jsonl", "error_analysis.txt", "results_analysis.ipynb"}
)

# These exact, case-sensitive top-level trees contain only local inputs,
# generated outputs, model artifacts, or pinned upstream working copies.  A
# basename/extension heuristic must never allow a force-added file beneath one
# of them.  Case sensitivity is intentional: ordinary source packages such as
# ``models/runtime.py`` remain valid.
_PROTECTED_TOP_LEVEL_TREES = frozenset(
    {
        "MODELS",
        "PRIVATE_DATA",
        "PUBLIC_CANDIDATE",
        "RESULTS",
        "TRAINING_ARTIFACTS",
        "UPSTREAM",
    }
)

_DATABASE_ENDINGS = (
    ".accdb",
    ".db",
    ".db-journal",
    ".db-shm",
    ".db-wal",
    ".duckdb",
    ".mdb",
    ".sqlite",
    ".sqlite3",
)
_TABULAR_EXPORT_ENDINGS = (".csv", ".tsv")
_RAW_EXPORT_ENDINGS = (
    ".arrow",
    ".avro",
    ".feather",
    ".json",
    ".jsonl",
    ".ndjson",
    ".parquet",
)
_RAW_EXPORT_DIRECTORIES = frozenset(
    {
        "backups",
        "data",
        "dataset",
        "datasets",
        "dumps",
        "exports",
        "raw",
        "raw-data",
        "raw_data",
    }
)
_RAW_EXPORT_NAME = re.compile(
    r"(?:^|[._-])(?:backup|candidate|dump|export(?:ed|s)?|messages?|raw|sms)(?:[._-]|$)"
)
_PATCH_BACKUP_ENDINGS = (".orig", ".rej", "~")

_MODEL_WEIGHT_ENDINGS = (
    ".bin",
    ".ckpt",
    ".gguf",
    ".h5",
    ".keras",
    ".onnx",
    ".pb",
    ".pt",
    ".pth",
    ".safetensors",
    ".tflite",
)
_CHECKPOINT_DIRECTORIES = frozenset({"checkpoint", "checkpoints", "model_weights", "weights"})

_RESULT_DIRECTORIES = frozenset(
    {"artifacts", "logs", "output", "outputs", "predictions", "reports", "result", "results", "runs"}
)
_RESULT_ENDINGS = (
    ".arrow",
    ".csv",
    ".html",
    ".jpeg",
    ".ipynb",
    ".jpg",
    ".json",
    ".jsonl",
    ".log",
    ".ndjson",
    ".parquet",
    ".png",
    ".tsv",
    ".txt",
    ".webp",
    ".xls",
    ".xlsx",
)
_RESULT_NAME = re.compile(
    r"^(?:error[._-]?analysis|eval(?:uation)?[._-]?results?|metrics?|predictions?|reports?|results?[._-]?analysis|sample[._-]?results?|samples?)(?:[._-]|$)"
)


class Violation(NamedTuple):
    """A tracked path that violates the repository artifact policy."""

    path: str
    category: str


def _normalize_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def classify_path(path: str) -> str | None:
    """Return the policy category for a tracked pathname, or ``None`` if safe."""

    normalized = _normalize_path(path)
    if normalized in GRANDFATHERED_PATHS:
        return None

    pure_path = PurePosixPath(normalized)
    lowered_parts = tuple(part.casefold() for part in pure_path.parts)
    if not lowered_parts:
        return None

    name = lowered_parts[-1]
    stem = PurePosixPath(name).stem
    directories = lowered_parts[:-1]

    if name.endswith(_PATCH_BACKUP_ENDINGS):
        return "patch backup"

    if name.endswith(_DATABASE_ENDINGS):
        return "database"

    if name.endswith(_TABULAR_EXPORT_ENDINGS):
        return "tabular export"

    if name.endswith(_MODEL_WEIGHT_ENDINGS):
        return "model weight"

    if any(
        part in _CHECKPOINT_DIRECTORIES or part.startswith("checkpoint-")
        for part in directories
    ):
        return "checkpoint"

    if name.endswith(_RESULT_ENDINGS) and (
        any(part in _RESULT_DIRECTORIES for part in directories)
        or _RESULT_NAME.search(stem)
    ):
        return "result artifact"

    if name.endswith(_RAW_EXPORT_ENDINGS) and (
        any(part in _RAW_EXPORT_DIRECTORIES for part in directories)
        or _RAW_EXPORT_NAME.search(stem)
    ):
        return "raw export"

    top_level = pure_path.parts[0]
    if len(pure_path.parts) > 1 and top_level in _PROTECTED_TOP_LEVEL_TREES:
        return f"protected output tree: {top_level}"

    return None


def find_violations(paths: Iterable[str]) -> list[Violation]:
    """Classify synthetic or Git-provided pathnames without opening any files."""

    violations = []
    for path in paths:
        category = classify_path(path)
        if category is not None:
            violations.append(Violation(_normalize_path(path), category))
    return violations


def tracked_paths(repo_root: Path | str = ".") -> list[str]:
    """Return tracked Git pathnames using NUL delimiters for unusual filenames."""

    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
    )
    return [
        item.decode("utf-8", errors="surrogateescape")
        for item in completed.stdout.split(b"\0")
        if item
    ]


def main() -> int:
    try:
        paths = tracked_paths()
        violations = find_violations(paths)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"repository safety check could not inspect Git paths: {exc}", file=sys.stderr)
        return 2

    if not violations:
        print("Repository safety check passed (tracked pathnames only).")
        grandfathered = sorted(set(paths) & GRANDFATHERED_PATHS)
        if grandfathered:
            print(
                "Explicit legacy path exceptions require publication review: "
                + ", ".join(grandfathered)
            )
        return 0

    print("Repository safety check failed; remove these artifacts from Git:", file=sys.stderr)
    for violation in sorted(violations):
        print(f"  - {violation.path} ({violation.category})", file=sys.stderr)
    print(
        "Keep private data, generated results, and model files outside version control.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
