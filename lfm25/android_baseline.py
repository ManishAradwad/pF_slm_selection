"""Reproduce the immutable Phase C Android source baseline from Git blobs.

The verifier deliberately hashes committed objects rather than working-tree files.
This makes a baseline reproducible while preserving unrelated local Android edits.
It never reads SMS storage, model artifacts, app databases, or device logs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from lfm25.android_profile_sync import verify_android_repository


COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class AndroidBaselineError(ValueError):
    """Raised when a baseline declaration or committed Android source drifts."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AndroidBaselineError(f"{name} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        raise AndroidBaselineError(
            f"{name} keys differ: missing={missing}, unexpected={unexpected}"
        )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_android_baseline(path: Path) -> dict[str, Any]:
    def reject_constant(_value: str) -> None:
        raise AndroidBaselineError("baseline contains a non-finite JSON constant")

    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, json.JSONDecodeError) as error:
        raise AndroidBaselineError(f"could not read Android baseline {path}") from error
    return dict(_mapping(value, "baseline"))


def _validate_relative_path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise AndroidBaselineError(f"{name} must be a nonempty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise AndroidBaselineError(f"{name} must remain repository-relative")
    return value


def validate_android_baseline(baseline: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the static, aggregate-safe Phase C baseline declaration."""

    _exact_keys(
        baseline,
        {
            "schema_version",
            "baseline_id",
            "baseline_version",
            "status",
            "captured_at",
            "source_snapshot",
            "production_profile_relationship",
            "behavior",
            "instrumentation",
            "evidence_classes",
            "selection",
            "privacy",
        },
        "baseline",
    )
    if baseline.get("schema_version") != 1 or baseline.get("baseline_version") != 1:
        raise AndroidBaselineError("unsupported Android baseline version")
    if baseline.get("status") != "captured_static_baseline":
        raise AndroidBaselineError("baseline status must be captured_static_baseline")

    source = _mapping(baseline.get("source_snapshot"), "source_snapshot")
    _exact_keys(
        source,
        {"repository", "revision", "hash_basis", "paths", "files_sha256"},
        "source_snapshot",
    )
    revision = source.get("revision")
    if not isinstance(revision, str) or COMMIT_RE.fullmatch(revision) is None:
        raise AndroidBaselineError("source_snapshot.revision must be a full commit")
    if source.get("hash_basis") != "sha256_of_git_blob_content_at_revision":
        raise AndroidBaselineError("source snapshot must hash canonical Git blob bytes")
    raw_paths = source.get("paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise AndroidBaselineError("source_snapshot.paths must be a nonempty list")
    paths = [_validate_relative_path(item, "source_snapshot path") for item in raw_paths]
    if len(paths) != len(set(paths)):
        raise AndroidBaselineError("source_snapshot.paths contains duplicates")
    hashes = dict(_mapping(source.get("files_sha256"), "source_snapshot.files_sha256"))
    if set(hashes) != set(paths):
        raise AndroidBaselineError("source snapshot path and hash keys differ")
    if any(
        not isinstance(value, str) or SHA256_RE.fullmatch(value) is None
        for value in hashes.values()
    ):
        raise AndroidBaselineError("every source snapshot hash must be SHA-256")

    relationship = _mapping(
        baseline.get("production_profile_relationship"),
        "production_profile_relationship",
    )
    _exact_keys(
        relationship,
        {
            "profile_path",
            "profile_sha256",
            "profile_revision",
            "baseline_revision",
            "profile_is_ancestor",
            "commit_distance",
            "profiled_paths_changed",
            "production_defaults_changed",
        },
        "production_profile_relationship",
    )
    _validate_relative_path(relationship.get("profile_path"), "profile_path")
    if SHA256_RE.fullmatch(str(relationship.get("profile_sha256"))) is None:
        raise AndroidBaselineError("profile_sha256 must be SHA-256")
    if COMMIT_RE.fullmatch(str(relationship.get("profile_revision"))) is None:
        raise AndroidBaselineError("profile_revision must be a full commit")
    if relationship.get("baseline_revision") != revision:
        raise AndroidBaselineError(
            "baseline relationship revision differs from source snapshot"
        )
    if relationship.get("profile_is_ancestor") is not True:
        raise AndroidBaselineError("the declared production profile must be an ancestor")
    if (
        not isinstance(relationship.get("commit_distance"), int)
        or relationship["commit_distance"] < 0
    ):
        raise AndroidBaselineError("commit_distance must be a nonnegative integer")
    changed = relationship.get("profiled_paths_changed")
    if not isinstance(changed, list) or len(changed) != len(set(changed)):
        raise AndroidBaselineError("profiled_paths_changed must be a unique list")
    for item in changed:
        _validate_relative_path(item, "profiled_paths_changed item")
    if relationship.get("production_defaults_changed") is not False:
        raise AndroidBaselineError(
            "Phase C must not declare a production-default change"
        )

    evidence = _mapping(baseline.get("evidence_classes"), "evidence_classes")
    _exact_keys(
        evidence,
        {"source_static", "host", "android_device"},
        "evidence_classes",
    )
    if _mapping(evidence["source_static"], "source_static").get("status") != "captured":
        raise AndroidBaselineError("source-static evidence must be captured")
    if _mapping(evidence["host"], "host").get("evidence_class") != "host":
        raise AndroidBaselineError("host evidence must be explicitly labelled")
    device = _mapping(evidence["android_device"], "android_device")
    if device.get("evidence_class") != "android_device":
        raise AndroidBaselineError("device evidence must be explicitly labelled")
    if device.get("status") not in {"measured", "not_measured_no_device"}:
        raise AndroidBaselineError("unexpected Android-device evidence status")

    selection = _mapping(baseline.get("selection"), "selection")
    if selection.get("selected_profile_id") is not None:
        raise AndroidBaselineError("Phase C cannot select a profile")
    if selection.get("direct_v2") != "unselected_hypothesis":
        raise AndroidBaselineError("Direct V2 must remain unselected")
    if selection.get("candidate_v2") != "unselected_hypothesis":
        raise AndroidBaselineError("Candidate V2 must remain unselected")

    privacy = _mapping(baseline.get("privacy"), "privacy")
    if privacy.get("classification") != "aggregate_safe_source_only":
        raise AndroidBaselineError(
            "baseline privacy classification is not aggregate-safe"
        )
    if privacy.get("private_data_accessed") is not False:
        raise AndroidBaselineError("Phase C baseline cannot claim private-data access")

    return {
        "baseline_id": baseline["baseline_id"],
        "baseline_version": baseline["baseline_version"],
        "revision": revision,
        "source_files": len(paths),
        "device_evidence_status": device["status"],
    }


def _git(
    repo: Path,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            capture_output=True,
        )
    except OSError as error:
        raise AndroidBaselineError(
            f"could not execute Git for Android repository {repo}"
        ) from error
    if check and completed.returncode != 0:
        raise AndroidBaselineError(
            f"Git baseline verification failed for arguments {args!r}"
        )
    return completed


def capture_committed_source_hashes(
    baseline: Mapping[str, Any],
    android_repo: Path,
) -> dict[str, str]:
    """Hash only committed blobs named by a baseline source list."""

    source = _mapping(baseline.get("source_snapshot"), "source_snapshot")
    revision = str(source.get("revision"))
    paths = source.get("paths")
    if not isinstance(paths, list):
        raise AndroidBaselineError("source_snapshot.paths must be a list")
    hashes: dict[str, str] = {}
    for raw_path in paths:
        relative = _validate_relative_path(raw_path, "source_snapshot path")
        content = _git(android_repo, "show", f"{revision}:{relative}").stdout
        hashes[relative] = hashlib.sha256(content).hexdigest()
    return hashes


def verify_android_baseline(
    baseline_path: Path,
    *,
    repository_root: Path,
    android_repo: Path,
) -> dict[str, Any]:
    """Verify the profile relationship and immutable baseline source snapshot."""

    baseline = load_android_baseline(baseline_path)
    declaration = validate_android_baseline(baseline)
    source = _mapping(baseline["source_snapshot"], "source_snapshot")
    relationship = _mapping(
        baseline["production_profile_relationship"],
        "production_profile_relationship",
    )
    revision = str(source["revision"])
    profile_revision = str(relationship["profile_revision"])

    _git(android_repo, "cat-file", "-e", f"{revision}^{{commit}}")
    ancestor = _git(
        android_repo,
        "merge-base",
        "--is-ancestor",
        profile_revision,
        revision,
        check=False,
    )
    if ancestor.returncode != 0:
        raise AndroidBaselineError(
            "profile revision is not an ancestor of baseline revision"
        )
    distance_text = (
        _git(
            android_repo,
            "rev-list",
            "--count",
            f"{profile_revision}..{revision}",
        )
        .stdout.decode("utf-8")
        .strip()
    )
    try:
        distance = int(distance_text)
    except ValueError as error:
        raise AndroidBaselineError("could not parse Android commit distance") from error
    if distance != relationship["commit_distance"]:
        raise AndroidBaselineError(
            "Android commit distance drifted: "
            f"expected {relationship['commit_distance']}, got {distance}"
        )

    profile_path = repository_root / str(relationship["profile_path"])
    if _sha256_file(profile_path) != relationship["profile_sha256"]:
        raise AndroidBaselineError("locked production profile hash drifted")
    profile_report = verify_android_repository(
        profile_path,
        android_repo,
        require_head=False,
    )

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile_paths = sorted(profile["android_source"]["files"])
    changed_output = (
        _git(
            android_repo,
            "diff",
            "--name-only",
            profile_revision,
            revision,
            "--",
            *profile_paths,
        )
        .stdout.decode("utf-8")
    )
    changed = sorted(item for item in changed_output.splitlines() if item)
    expected_changed = sorted(relationship["profiled_paths_changed"])
    if changed != expected_changed:
        raise AndroidBaselineError(
            "profiled source relationship drifted: "
            f"expected {expected_changed}, got {changed}"
        )

    actual_hashes = capture_committed_source_hashes(baseline, android_repo)
    expected_hashes = dict(_mapping(source["files_sha256"], "files_sha256"))
    mismatched = sorted(
        path
        for path, expected in expected_hashes.items()
        if actual_hashes.get(path) != expected
    )
    if mismatched:
        raise AndroidBaselineError(
            f"Android baseline source hashes drifted: {mismatched}"
        )

    head = _git(android_repo, "rev-parse", "HEAD").stdout.decode("utf-8").strip()
    dirty_entries = _git(
        android_repo,
        "status",
        "--porcelain=v1",
    ).stdout.splitlines()
    return {
        **declaration,
        "status": "verified",
        "profile_revision": profile_report["revision"],
        "profile_files_verified": profile_report["files_verified"],
        "baseline_files_verified": len(actual_hashes),
        "profiled_paths_changed": changed,
        "checkout": {
            "head": head,
            "head_matches_baseline": head == revision,
            "dirty": bool(dirty_entries),
            "dirty_entry_count": len(dirty_entries),
            "verification_basis": "committed_git_blobs_not_worktree_files",
        },
        "evidence": {
            "source_static": "verified",
            "host": _mapping(
                baseline["evidence_classes"], "evidence_classes"
            )["host"],
            "android_device": _mapping(
                baseline["evidence_classes"], "evidence_classes"
            )["android_device"],
        },
    }
