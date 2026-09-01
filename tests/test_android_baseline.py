from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from lfm25.android_baseline import (
    AndroidBaselineError,
    capture_committed_source_hashes,
    load_android_baseline,
    validate_android_baseline,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = (
    REPOSITORY_ROOT
    / "configs/baselines/pocketfinancer-android-552ffbdf-phase-c.json"
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
    )


def test_phase_c_baseline_declaration_is_source_only_and_unselected() -> None:
    baseline = load_android_baseline(BASELINE_PATH)
    summary = validate_android_baseline(baseline)

    assert (
        summary["revision"]
        == "552ffbdfbd41773980aa249789b0cb508fdb19fd"
    )
    assert summary["source_files"] >= 30
    assert summary["device_evidence_status"] == "not_measured_no_device"
    assert baseline["selection"] == {
        "direct_v2": "unselected_hypothesis",
        "candidate_v2": "unselected_hypothesis",
        "selected_profile_id": None,
    }
    assert baseline["privacy"]["private_data_accessed"] is False


def test_capture_hashes_committed_blob_not_dirty_worktree(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "android"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "phase-c@example.invalid")
    _git(repo, "config", "user.name", "Phase C Test")
    source = repo / "Runtime.kt"
    source.write_text("committed\n", encoding="utf-8")
    _git(repo, "add", "Runtime.kt")
    _git(repo, "commit", "-m", "test: add runtime")
    revision = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source.write_text("unrelated dirty edit\n", encoding="utf-8")
    baseline = {
        "source_snapshot": {
            "revision": revision,
            "paths": ["Runtime.kt"],
        }
    }

    hashes = capture_committed_source_hashes(baseline, repo)

    assert hashes == {
        "Runtime.kt": hashlib.sha256(b"committed\n").hexdigest()
    }
    assert (
        source.read_text(encoding="utf-8")
        == "unrelated dirty edit\n"
    )


def test_baseline_rejects_a_selected_profile() -> None:
    baseline = load_android_baseline(BASELINE_PATH)
    baseline["selection"]["selected_profile_id"] = "forbidden-selection"

    with pytest.raises(AndroidBaselineError, match="cannot select"):
        validate_android_baseline(baseline)


def test_baseline_source_hash_keys_are_exact() -> None:
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    baseline["source_snapshot"]["files_sha256"].pop(
        baseline["source_snapshot"]["paths"][0]
    )

    with pytest.raises(AndroidBaselineError, match="path and hash keys"):
        validate_android_baseline(baseline)
