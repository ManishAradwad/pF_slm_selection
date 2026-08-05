"""Verify the checked-in PocketFinancer profile against Android git blobs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

from lfm25.android_contract import (
    ANDROID_DECODE_DEFAULTS,
    ANDROID_SOURCE_REVISION,
    ANDROID_SOURCE_SHA256,
    POCKETFINANCER_CONTRACT,
)


class AndroidProfileError(ValueError):
    """Raised when the app profile or pinned Android source has drifted."""


KNOWN_ANDROID_REPOSITORIES = (
    Path("/home/tojinotzenin/pocket-financer-android"),
    Path("/mnt/d/Personal_Projects/pocket-financer-android"),
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AndroidProfileError(f"{name} must be an object")
    return value


def load_android_profile(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AndroidProfileError(f"could not read app profile {path}: {error}") from error
    return dict(_mapping(value, "app profile"))


def verify_profile_declaration(path: Path) -> dict[str, Any]:
    """Ensure the static JSON profile agrees with the executable contract."""

    profile = load_android_profile(path)
    if profile.get("profile") != POCKETFINANCER_CONTRACT:
        raise AndroidProfileError("app profile must be 'pocketfinancer'")
    source = _mapping(profile.get("android_source"), "android_source")
    if source.get("revision") != ANDROID_SOURCE_REVISION:
        raise AndroidProfileError("app profile revision does not match android_contract.py")
    files = dict(_mapping(source.get("files"), "android_source.files"))
    if files != ANDROID_SOURCE_SHA256:
        missing = sorted(set(ANDROID_SOURCE_SHA256) - set(files))
        unexpected = sorted(set(files) - set(ANDROID_SOURCE_SHA256))
        mismatched = sorted(
            path_name
            for path_name in set(files) & set(ANDROID_SOURCE_SHA256)
            if files[path_name] != ANDROID_SOURCE_SHA256[path_name]
        )
        raise AndroidProfileError(
            "app profile source hashes differ from android_contract.py: "
            f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
        )
    runtime = _mapping(profile.get("runtime"), "runtime")
    generation = _mapping(profile.get("generation"), "generation")
    grammar = _mapping(generation.get("grammar"), "generation.grammar")
    preprocessing = _mapping(profile.get("preprocessing"), "preprocessing")
    if runtime.get("n_ctx") != ANDROID_DECODE_DEFAULTS["n_ctx"]:
        raise AndroidProfileError("app profile n_ctx differs from executable contract")
    if generation.get("thinking_mode") != ANDROID_DECODE_DEFAULTS["thinking_mode"]:
        raise AndroidProfileError("app profile thinking mode differs from executable contract")
    if grammar.get("default_enabled") != ANDROID_DECODE_DEFAULTS["grammar"]:
        raise AndroidProfileError("app profile grammar default differs from executable contract")
    if preprocessing.get("enabled") is not ANDROID_DECODE_DEFAULTS["prefilter"]:
        raise AndroidProfileError("app profile prefilter default differs from executable contract")
    return {
        "declaration_verified": True,
        "profile": profile["profile"],
        "revision": source["revision"],
        "files_declared": len(files),
    }


def _repository_head(repo: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _required_repository(path: Path, source: str) -> Path:
    resolved = path.expanduser().resolve()
    if not (resolved / ".git").exists():
        raise AndroidProfileError(f"Android repository from {source} not found: {path}")
    return resolved


def discover_android_repository(explicit: Path | None = None) -> Path | None:
    """Find the checkout whose HEAD matches the pinned Android profile.

    An explicit CLI path or environment path is authoritative and is therefore
    returned even when its HEAD has drifted; the verifier will then report that
    drift. Auto-discovery may see multiple local clones, so it prefers the clone
    checked out at the pinned revision instead of whichever hard-coded path happens
    to appear first.
    """

    if explicit is not None:
        return _required_repository(explicit, "--android-repo")
    configured = os.environ.get("POCKETFINANCER_ANDROID_REPO")
    if configured:
        return _required_repository(Path(configured), "POCKETFINANCER_ANDROID_REPO")

    repositories = [
        candidate.expanduser().resolve()
        for candidate in KNOWN_ANDROID_REPOSITORIES
        if (candidate.expanduser().resolve() / ".git").exists()
    ]
    for repository in repositories:
        if _repository_head(repository) == ANDROID_SOURCE_REVISION:
            return repository
    return repositories[0] if repositories else None


def _git(repo: Path, *args: str, text: bool = False) -> str | bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=text,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise AndroidProfileError(f"git verification failed in {repo}: {error}") from error
    return completed.stdout


def verify_android_repository(
    profile_path: Path,
    android_repo: Path,
    *,
    require_head: bool = True,
) -> dict[str, Any]:
    """Hash canonical git blobs so checkout line endings cannot hide drift."""

    declaration = verify_profile_declaration(profile_path)
    profile = load_android_profile(profile_path)
    source = _mapping(profile["android_source"], "android_source")
    revision = str(source["revision"])
    head = str(_git(android_repo, "rev-parse", "HEAD", text=True)).strip()
    if require_head and head != revision:
        raise AndroidProfileError(
            f"Android HEAD {head} differs from pinned profile revision {revision}"
        )
    mismatched: list[str] = []
    missing: list[str] = []
    for relative, expected_hash in sorted(
        _mapping(source["files"], "android_source.files").items()
    ):
        try:
            content = _git(android_repo, "show", f"{revision}:{relative}")
        except AndroidProfileError:
            missing.append(str(relative))
            continue
        if hashlib.sha256(content).hexdigest() != expected_hash:
            mismatched.append(str(relative))
    if missing or mismatched:
        raise AndroidProfileError(
            f"Android source profile drift: missing={missing}, mismatched={mismatched}"
        )
    return {
        **declaration,
        "repository_verified": True,
        "repository": str(android_repo),
        "head": head,
        "files_verified": len(source["files"]),
    }


def verify_current_android_profile(
    profile_path: Path,
    *,
    android_repo: Path | None = None,
    require_head: bool = True,
) -> dict[str, Any]:
    """Verify declaration always and local source blobs whenever available."""

    declaration = verify_profile_declaration(profile_path)
    repository = discover_android_repository(android_repo)
    if repository is None:
        return {
            **declaration,
            "repository_verified": False,
            "repository": None,
            "note": "no local Android checkout found; static declaration only",
        }
    return verify_android_repository(
        profile_path,
        repository,
        require_head=require_head,
    )
