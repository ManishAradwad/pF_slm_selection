#!/usr/bin/env python3
"""Export or verify the exact frozen native-integration-v4 app asset bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PurePosixPath(
    "configs/sms_processing/contracts/releases/native-integration-v4.json"
)
MANIFEST_SHA256 = "0d3bf18f91d0a197c7bb56b5e082fd2851ce072f854452a9647c52d45b3433d8"
RELEASE_ID = "native-integration-v4"
INTEGRITY_REASON = "configuration_integrity"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class BundleIntegrityError(ValueError):
    """The frozen source release or packaged bundle failed closed."""


@dataclass(frozen=True)
class FrozenAsset:
    path: PurePosixPath
    sha256: str
    content: bytes


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _safe_relative_path(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise BundleIntegrityError("release contains an invalid artifact path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or chr(92) in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise BundleIntegrityError(f"release contains an unsafe artifact path: {value!r}")
    return path


def _read_frozen_file(repo_root: Path, relative_path: PurePosixPath) -> bytes:
    candidate = repo_root.joinpath(*relative_path.parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(repo_root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise BundleIntegrityError(
            f"frozen release asset is missing or outside the repository: {relative_path}"
        ) from exc
    if candidate.is_symlink() or not resolved.is_file():
        raise BundleIntegrityError(f"frozen release asset is not a regular file: {relative_path}")
    try:
        return resolved.read_bytes()
    except OSError as exc:
        raise BundleIntegrityError(
            f"frozen release asset could not be read: {relative_path}"
        ) from exc


def load_frozen_assets(repo_root: Path = REPO_ROOT) -> tuple[FrozenAsset, ...]:
    """Load only the pinned manifest and its hash-bound, repository-local allowlist."""

    repo_root = repo_root.resolve(strict=True)
    manifest_content = _read_frozen_file(repo_root, MANIFEST_PATH)
    if _sha256(manifest_content) != MANIFEST_SHA256:
        raise BundleIntegrityError("native-integration-v4 manifest does not match its frozen hash")
    try:
        manifest = json.loads(manifest_content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleIntegrityError("native-integration-v4 manifest is not valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("release_id") != RELEASE_ID:
        raise BundleIntegrityError("native-integration-v4 manifest has the wrong release ID")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise BundleIntegrityError("native-integration-v4 manifest has no artifact table")

    frozen = [FrozenAsset(MANIFEST_PATH, MANIFEST_SHA256, manifest_content)]
    seen = {MANIFEST_PATH}
    for item in artifacts:
        if not isinstance(item, dict):
            raise BundleIntegrityError("native-integration-v4 contains an invalid artifact entry")
        relative_path = _safe_relative_path(item.get("path"))
        expected_sha256 = item.get("sha256")
        if not isinstance(expected_sha256, str) or not _SHA256_PATTERN.fullmatch(
            expected_sha256
        ):
            raise BundleIntegrityError(f"artifact has an invalid SHA-256: {relative_path}")
        if relative_path in seen:
            raise BundleIntegrityError(f"release contains a duplicate path: {relative_path}")
        seen.add(relative_path)
        content = _read_frozen_file(repo_root, relative_path)
        if _sha256(content) != expected_sha256:
            raise BundleIntegrityError(f"frozen release asset hash mismatch: {relative_path}")
        frozen.append(FrozenAsset(relative_path, expected_sha256, content))
    return tuple(frozen)


def _target_path(bundle_root: Path, relative_path: PurePosixPath) -> Path:
    target = bundle_root.joinpath(*relative_path.parts)
    try:
        target.resolve(strict=False).relative_to(bundle_root)
    except (OSError, ValueError) as exc:
        raise BundleIntegrityError(f"bundle path escapes its root: {relative_path}") from exc
    return target


def _verify_target(target: Path, asset: FrozenAsset) -> str | None:
    if not target.exists():
        return "missing"
    if target.is_symlink() or not target.is_file():
        return "not_a_regular_file"
    try:
        actual_sha256 = _sha256(target.read_bytes())
    except OSError:
        return "unreadable"
    if actual_sha256 != asset.sha256:
        return "hash_mismatch"
    return None


def _report(status: str, *, copied_files: int, existing_files: int) -> dict[str, Any]:
    file_count = copied_files + existing_files
    return {
        "artifact_count": file_count - 1,
        "copied_files": copied_files,
        "existing_files": existing_files,
        "file_count": file_count,
        "manifest_path": MANIFEST_PATH.as_posix(),
        "manifest_sha256": MANIFEST_SHA256,
        "release_id": RELEASE_ID,
        "status": status,
    }


def check_bundle(bundle_root: Path, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Verify the packaged manifest and every referenced artifact byte-for-byte."""

    assets = load_frozen_assets(repo_root)
    bundle_root = bundle_root.resolve(strict=False)
    failures = []
    for asset in assets:
        target = _target_path(bundle_root, asset.path)
        issue = _verify_target(target, asset)
        if issue is not None:
            failures.append(f"{asset.path}:{issue}")
    if failures:
        raise BundleIntegrityError("packaged v3 bundle failed verification: " + ", ".join(failures))
    return _report("verified", copied_files=0, existing_files=len(assets))


def _copy_exclusive(target: Path, content: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, target)
        except FileExistsError as exc:
            raise BundleIntegrityError(f"bundle target changed during export: {target.name}") from exc
    except OSError as exc:
        raise BundleIntegrityError(f"could not export bundle asset: {target.name}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def export_bundle(bundle_root: Path, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Add absent frozen assets, refusing every conflicting existing destination."""

    assets = load_frozen_assets(repo_root)
    bundle_root = bundle_root.resolve(strict=False)
    targets = [(asset, _target_path(bundle_root, asset.path)) for asset in assets]

    conflicts = []
    existing = 0
    for asset, target in targets:
        issue = _verify_target(target, asset)
        if issue is None:
            existing += 1
        elif issue != "missing":
            conflicts.append(f"{asset.path}:{issue}")
    if conflicts:
        raise BundleIntegrityError(
            "refusing to overwrite conflicting app bundle assets: " + ", ".join(conflicts)
        )

    copied = 0
    for asset, target in targets:
        if target.exists():
            continue
        _copy_exclusive(target, asset.content)
        copied += 1
    check_bundle(bundle_root, repo_root)
    return _report("exported", copied_files=copied, existing_files=existing)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export or verify the pinned native-integration-v4 app asset bundle."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("export", "check"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument(
            "--bundle-root",
            type=Path,
            required=True,
            help="Root below which manifest-relative paths are preserved",
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "export":
            report = export_bundle(args.bundle_root)
        else:
            report = check_bundle(args.bundle_root)
    except BundleIntegrityError as exc:
        print(
            json.dumps(
                {"reason": INTEGRITY_REASON, "status": "failed", "summary": str(exc)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
