"""Freeze and compatibility checks for native-integration-v5."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema

from scripts.build_native_contract_release_v5 import build_manifest


ROOT = Path(__file__).resolve().parents[2]
FROZEN_RELEASE_HASHES = {
    "native-integration-v1.json": "e07ac6d2f6e90fac914db824d104141a20e49fc40f8b8f02c8fec4c0614e680a",
    "native-integration-v2.json": "637013f0988a20eb070e10b07f68ddf9172b847262024a50676c90022234019d",
    "native-integration-v3.json": "61609c3336374c8b96b1e36cb90b5af01039ceecefcfc1b0091fd9359804436b",
    "native-integration-v4.json": "0d3bf18f91d0a197c7bb56b5e082fd2851ce072f854452a9647c52d45b3433d8",
}


def _json(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def test_manifest_rebuilds_exactly_and_binds_final_policy() -> None:
    stored = _json(
        "configs/sms_processing/contracts/releases/native-integration-v5.json"
    )
    assert stored == build_manifest()
    assert stored["automatic_persistence_enabled"] is True
    assert stored["routing_policy"] == {
        "complete_posted": "transactions",
        "duplicate": "review",
        "exception": "review",
        "policy_contract": "pocketfinancer.persistence-policy/2",
        "processing_config_contract": "pocketfinancer.processing-config/5",
        "processing_result_contract": "pocketfinancer.processing-result/3",
        "retry": "new_operation_with_parent_operation_id",
        "review_contract": "pocketfinancer.review-case/2",
        "transaction_write": "atomic_with_operation_settlement",
        "valid_none": "no_transaction",
    }
    schema = _json(
        "configs/sms_processing/contracts/releases/v5/release-manifest.schema.json"
    )
    jsonschema.validate(stored, schema)


def test_every_predecessor_release_remains_byte_for_byte_frozen() -> None:
    release_root = ROOT / "configs/sms_processing/contracts/releases"
    for name, expected in FROZEN_RELEASE_HASHES.items():
        assert hashlib.sha256((release_root / name).read_bytes()).hexdigest() == expected


def test_successor_policy_and_reason_registry_validate() -> None:
    jsonschema.validate(
        _json("configs/sms_processing/policies/native-persistence-v2.json"),
        _json("configs/sms_processing/policies/native-persistence-v2.schema.json"),
    )
    registry = _json("configs/sms_processing/contracts/v5/reason-code-registry.json")
    jsonschema.validate(
        registry,
        _json("configs/sms_processing/contracts/v5/reason-code-registry.schema.json"),
    )
    codes = [item["code"] for group in registry["namespaces"] for item in group["codes"]]
    assert len(codes) == len(set(codes))
    assert {
        "persistence_atomic_write_failed",
        "operation_release_incompatible",
        "retry_operation_lineage_invalid",
    }.issubset(codes)
