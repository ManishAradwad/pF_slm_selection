from __future__ import annotations

import json
from pathlib import Path

import pytest

import lfm25.android_profile_sync as profile_sync
from lfm25.android_profile_sync import (
    AndroidProfileError,
    discover_android_repository,
    verify_profile_declaration,
)
from lfm25.android_contract import ANDROID_SOURCE_REVISION


REPO_ROOT = Path(__file__).resolve().parents[1]
CURRENT_PROFILE = REPO_ROOT / "configs/contracts/pocketfinancer-android-current.json"


def test_current_profile_matches_executable_contract() -> None:
    report = verify_profile_declaration(CURRENT_PROFILE)

    assert report["declaration_verified"] is True
    assert report["profile"] == "pocketfinancer"
    assert report["files_declared"] == 13


def test_profile_revision_drift_is_rejected(tmp_path: Path) -> None:
    profile = json.loads(CURRENT_PROFILE.read_text(encoding="utf-8"))
    profile["android_source"]["revision"] = "0" * 40
    drifted = tmp_path / "drifted.json"
    drifted.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(AndroidProfileError, match="revision"):
        verify_profile_declaration(drifted)


def test_auto_discovery_prefers_checkout_at_pinned_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = tmp_path / "stale"
    current = tmp_path / "current"
    (stale / ".git").mkdir(parents=True)
    (current / ".git").mkdir(parents=True)
    monkeypatch.delenv("POCKETFINANCER_ANDROID_REPO", raising=False)
    monkeypatch.setattr(profile_sync, "KNOWN_ANDROID_REPOSITORIES", (stale, current))
    monkeypatch.setattr(
        profile_sync,
        "_repository_head",
        lambda repo: ANDROID_SOURCE_REVISION if repo == current else "0" * 40,
    )

    assert discover_android_repository() == current
