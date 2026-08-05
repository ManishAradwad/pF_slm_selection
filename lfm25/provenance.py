"""Small deterministic fingerprints for local experiment evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 without loading model artifacts into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_file(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def fingerprint_named_files(directory: Path, names: Iterable[str]) -> dict[str, Any]:
    """Fingerprint selected model files while making absent optional files explicit."""

    resolved = directory.resolve(strict=True)
    files: dict[str, Any] = {}
    for name in names:
        candidate = resolved / name
        if candidate.is_file():
            files[name] = {
                "bytes": candidate.stat().st_size,
                "sha256": file_sha256(candidate),
            }
    return {"path": str(resolved), "files": files}


def code_fingerprints(repo_root: Path) -> dict[str, str]:
    """Bind results to the prompt, parser/contract, and aggregate scorer."""

    relative_paths = (
        "lfm25/prompts.py",
        "lfm25/contract.py",
        "lfm25/metrics.py",
        "DATA/sms_extraction.gbnf",
    )
    return {
        relative: file_sha256(repo_root / relative)
        for relative in relative_paths
    }
