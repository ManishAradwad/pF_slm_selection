"""Freeze and compatibility checks for native-integration-v6."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema

from scripts.build_native_contract_release_v6 import build_manifest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    "configs/sms_processing/contracts/releases/native-integration-v6.json"
)
V5_MANIFEST = (
    "configs/sms_processing/contracts/releases/native-integration-v5.json"
)
V5_SHA256 = "971ac758729916e8dfdd8ea165d4e8344b5d3ab6e4d190473c8e91e2541c5f6d"


def _json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_v6_release_rebuilds_and_preserves_v5() -> None:
    stored = _json(MANIFEST)
    assert stored == build_manifest()
    assert stored["release_id"] == "native-integration-v6"
    assert stored["routing_policy"]["processing_config_contract"] == (
        "pocketfinancer.processing-config/6"
    )
    assert stored["compatibility"]["frozen_release_ids"][-1] == (
        "native-integration-v5"
    )
    assert hashlib.sha256((ROOT / V5_MANIFEST).read_bytes()).hexdigest() == V5_SHA256
    jsonschema.validate(
        stored,
        _json(
            "configs/sms_processing/contracts/releases/v6/release-manifest.schema.json"
        ),
    )
