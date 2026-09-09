from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from lfm25.android_runtime_evidence import (
    AndroidRuntimeEvidenceError,
    read_runtime_evidence_package,
    validate_runtime_evidence_package,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    REPOSITORY_ROOT
    / "tests/fixtures/pocketfinancer_android_runtime_evidence_synthetic.json"
)
SCRIPT = (
    REPOSITORY_ROOT
    / "scripts/validate_pocketfinancer_android_runtime_evidence.py"
)


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_invented_runtime_fixture_keeps_host_and_device_evidence_distinct() -> None:
    summary = validate_runtime_evidence_package(_fixture())

    assert summary["status"] == "valid"
    assert summary["evidence_classes"] == ["android_device", "host"]
    assert summary["captures_by_class"] == {
        "android_device": 1,
        "host": 1,
    }
    assert summary["sample_total_by_class"] == {
        "android_device": 3,
        "host": 4,
    }


def test_runtime_package_rejects_row_or_message_fields() -> None:
    package = _fixture()
    package["captures"][0]["sms_body"] = "invented but still forbidden"

    with pytest.raises(AndroidRuntimeEvidenceError, match="unexpected"):
        validate_runtime_evidence_package(package)


def test_host_capture_cannot_claim_android_device_provenance() -> None:
    package = _fixture()
    package["captures"][0]["provenance"][
        "device_fingerprint_sha256"
    ] = "4" * 64

    with pytest.raises(
        AndroidRuntimeEvidenceError,
        match="Android-device provenance",
    ):
        validate_runtime_evidence_package(package)


def test_not_measured_capture_cannot_smuggle_measurements() -> None:
    package = _fixture()
    capture = copy.deepcopy(package["captures"][1])
    capture["status"] = "not_measured"
    capture["gaps"] = ["device_measurement_not_run"]
    capture["provenance"]["device_fingerprint_sha256"] = None
    capture["provenance"]["android_api_level"] = None
    package["captures"] = [capture]

    with pytest.raises(
        AndroidRuntimeEvidenceError,
        match="cannot contain measurements",
    ):
        validate_runtime_evidence_package(package)


def test_unmeasured_device_capture_is_content_free() -> None:
    package = _fixture()
    capture = copy.deepcopy(package["captures"][1])
    capture["status"] = "not_measured"
    capture["provenance"]["device_fingerprint_sha256"] = None
    capture["provenance"]["android_api_level"] = None
    capture["sample"] = {
        "total": 0,
        "success": 0,
        "null": 0,
        "error": 0,
        "stopped": 0,
        "cache_attempts": 0,
        "cache_hits": 0,
    }
    capture["measurements"] = {
        key: None for key in capture["measurements"]
    }
    capture["gaps"] = [
        "no_attached_android_device",
        "device_measurement_not_run",
    ]
    package["captures"] = [capture]

    summary = validate_runtime_evidence_package(package)

    assert summary["sample_total_by_class"] == {"android_device": 0}


def test_runtime_package_rejects_a_different_baseline_hash() -> None:
    package = _fixture()
    package["baseline"]["baseline_manifest_sha256"] = "f" * 64

    with pytest.raises(AndroidRuntimeEvidenceError, match="exact Phase C"):
        validate_runtime_evidence_package(package)


def test_runtime_reader_requires_results_or_exact_invented_fixture(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "copied.json"
    copied.write_text(
        FIXTURE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.raises(AndroidRuntimeEvidenceError, match="RESULTS"):
        read_runtime_evidence_package(
            copied,
            allow_synthetic_fixture=True,
        )

    assert read_runtime_evidence_package(
        FIXTURE,
        allow_synthetic_fixture=True,
    )["contract_version"] == 1


def test_runtime_cli_prints_only_aggregate_summary() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(FIXTURE),
            "--invented-fixture",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    output = json.loads(completed.stdout)

    assert output["status"] == "valid"
    assert output["capture_count"] == 2
    assert "capture_id" not in completed.stdout
    assert "environment_id" not in completed.stdout
