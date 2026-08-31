"""Hashing, private-path, permission, and atomic-write helpers."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class PrivateArtifactError(RuntimeError):
    """Aggregate-safe private artifact failure."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_private_output(repo_root: Path, path: Path) -> Path:
    root = (repo_root / "PRIVATE_DATA" / "sms_processing").resolve()
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise PrivateArtifactError("private output is outside the SMS processing private root")
    completed = subprocess.run(
        ["git", "check-ignore", "-q", str(resolved)],
        cwd=repo_root,
        check=False,
    )
    if completed.returncode != 0:
        raise PrivateArtifactError("private output path is not protected by Git ignore rules")
    return resolved


def ensure_private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path, 0o700)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    ensure_private_directory(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value) + b"\n")


def atomic_write_jsonl(path: Path, values: Iterable[Any]) -> None:
    payload = b"".join(canonical_json_bytes(value) + b"\n" for value in values)
    atomic_write_bytes(path, payload)


def load_or_create_secret(path: Path, *, length: int = 32) -> bytes:
    ensure_private_directory(path.parent)
    if path.exists():
        value = path.read_bytes()
        if len(value) < length:
            raise PrivateArtifactError("private source-ID key is invalid")
        os.chmod(path, 0o600)
        return value
    value = os.urandom(length)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, value)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return value
