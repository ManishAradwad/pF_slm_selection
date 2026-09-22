#!/usr/bin/env python3
"""Build the final automatic-routing native release without altering v1-v4."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import jsonschema


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pocketfinancer_sms.provenance import file_sha256  # noqa: E402

try:
    from scripts.build_native_contract_release_v4 import (  # noqa: E402
        ARTIFACTS as V4_ARTIFACTS,
        build_manifest as build_v4_manifest,
    )
except ModuleNotFoundError:
    from build_native_contract_release_v4 import (  # type: ignore[no-redef]  # noqa: E402
        ARTIFACTS as V4_ARTIFACTS,
        build_manifest as build_v4_manifest,
    )


NEW_ARTIFACTS = (
    (
        "configs/sms_processing/contracts/v5/processing-config.schema.json",
        "pocketfinancer.processing-config/5",
    ),
    (
        "configs/sms_processing/contracts/v5/review-case.schema.json",
        "pocketfinancer.review-case/2",
    ),
    (
        "configs/sms_processing/contracts/v5/reason-code-registry.json",
        "pocketfinancer.reason-code-registry/3",
    ),
    (
        "configs/sms_processing/contracts/v5/reason-code-registry.schema.json",
        "pocketfinancer.reason-code-registry-schema/3",
    ),
    (
        "configs/sms_processing/policies/native-persistence-v2.json",
        "pocketfinancer.persistence-policy/2",
    ),
    (
        "configs/sms_processing/policies/native-persistence-v2.schema.json",
        "pocketfinancer.persistence-policy-schema/2",
    ),
    (
        "configs/sms_processing/contracts/releases/v5/release-manifest.schema.json",
        "pocketfinancer.contract-release-manifest-schema/5",
    ),
    (
        "tests/sms_processing/golden/native-v5/routing-policy.json",
        "pocketfinancer.native-routing-golden/2",
    ),
)
ARTIFACTS = V4_ARTIFACTS + NEW_ARTIFACTS

_FROZEN_RELEASE_HASHES = {
    "configs/sms_processing/contracts/releases/native-integration-v1.json": "e07ac6d2f6e90fac914db824d104141a20e49fc40f8b8f02c8fec4c0614e680a",
    "configs/sms_processing/contracts/releases/native-integration-v2.json": "637013f0988a20eb070e10b07f68ddf9172b847262024a50676c90022234019d",
    "configs/sms_processing/contracts/releases/native-integration-v3.json": "61609c3336374c8b96b1e36cb90b5af01039ceecefcfc1b0091fd9359804436b",
    "configs/sms_processing/contracts/releases/native-integration-v4.json": "0d3bf18f91d0a197c7bb56b5e082fd2851ce072f854452a9647c52d45b3433d8",
}


def _read_json(path: str) -> Any:
    with (REPO_ROOT / path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _validate_frozen_releases() -> None:
    for path, expected in _FROZEN_RELEASE_HASHES.items():
        actual = file_sha256(REPO_ROOT / path)
        if actual != expected:
            raise ValueError(f"frozen release changed: {path} expected {expected}, found {actual}")
    stored_v4 = _read_json(
        "configs/sms_processing/contracts/releases/native-integration-v4.json"
    )
    if stored_v4 != build_v4_manifest():
        raise ValueError("native-integration-v4 no longer matches its deterministic rebuild")


def _validate_reason_registry() -> None:
    previous = _read_json("configs/sms_processing/contracts/v3/reason-code-registry.json")
    current = _read_json("configs/sms_processing/contracts/v5/reason-code-registry.json")
    previous_codes = {
        item["code"]: item
        for namespace in previous["namespaces"]
        for item in namespace["codes"]
    }
    current_codes = {
        item["code"]: item
        for namespace in current["namespaces"]
        for item in namespace["codes"]
    }
    if len(current_codes) != sum(len(item["codes"]) for item in current["namespaces"]):
        raise ValueError("reason-code-registry/3 contains duplicate codes")
    for code, frozen in previous_codes.items():
        if current_codes.get(code) != frozen:
            raise ValueError(f"frozen reason-code meaning changed: {code}")
    expected_new = {
        "persistence_atomic_write_failed",
        "operation_release_incompatible",
        "retry_operation_lineage_invalid",
    }
    if set(current_codes) - set(previous_codes) != expected_new:
        raise ValueError("reason-code-registry/3 additive code set is unexpected")


def _validate_routing_vectors() -> None:
    golden = _read_json("tests/sms_processing/golden/native-v5/routing-policy.json")
    destinations = {case["id"]: case["destination"] for case in golden["cases"]}
    if destinations["frozen_v4_complete_posted"] != "review":
        raise ValueError("v4 routing compatibility vector changed")
    if destinations["successor_complete_posted"] != "transactions":
        raise ValueError("successor success route is not Transactions")
    if destinations["successor_none"] != "no_transaction":
        raise ValueError("successor none route is not terminal")
    exception_ids = set(destinations) - {
        "frozen_v4_complete_posted",
        "successor_complete_posted",
        "successor_none",
    }
    if any(destinations[item] != "review" for item in exception_ids):
        raise ValueError("successor exception route does not retain Review")
    partial = golden["partial_review"]
    source = partial["source"]
    for field in ("amount", "direction"):
        span = partial["invalid_model_output"][field]["evidence"]
        if source[span["start_scalar"] : span["end_scalar"]] != span["text"]:
            raise ValueError(f"partial Review vector has an invalid {field} span")


def validate_freeze_inputs() -> None:
    missing = [path for path, _ in NEW_ARTIFACTS if not (REPO_ROOT / path).is_file()]
    if missing:
        raise ValueError(f"native-integration-v5 assets are incomplete: {', '.join(missing)}")
    if len({path for path, _ in ARTIFACTS}) != len(ARTIFACTS):
        raise ValueError("native-integration-v5 contains duplicate artifact paths")

    for path, contract in NEW_ARTIFACTS:
        if not path.endswith(".json"):
            continue
        value = _read_json(path)
        declared = value.get("$id", value.get("contract")) if isinstance(value, dict) else None
        if declared != contract:
            raise ValueError(f"{path} declares {declared!r}, expected {contract!r}")
        if path.endswith(".schema.json"):
            jsonschema.Draft202012Validator.check_schema(value)

    jsonschema.validate(
        _read_json("configs/sms_processing/contracts/v5/reason-code-registry.json"),
        _read_json("configs/sms_processing/contracts/v5/reason-code-registry.schema.json"),
    )
    jsonschema.validate(
        _read_json("configs/sms_processing/policies/native-persistence-v2.json"),
        _read_json("configs/sms_processing/policies/native-persistence-v2.schema.json"),
    )
    _validate_frozen_releases()
    _validate_reason_registry()
    _validate_routing_vectors()


def build_manifest() -> dict[str, Any]:
    validate_freeze_inputs()
    predecessor = build_v4_manifest()
    manifest = {
        "contract": "pocketfinancer.contract-release-manifest/1",
        "release_id": "native-integration-v5",
        "status": "frozen_for_native_implementation",
        "automatic_persistence_enabled": True,
        "routing_policy": {
            "policy_contract": "pocketfinancer.persistence-policy/2",
            "processing_config_contract": "pocketfinancer.processing-config/5",
            "review_contract": "pocketfinancer.review-case/2",
            "processing_result_contract": "pocketfinancer.processing-result/3",
            "complete_posted": "transactions",
            "exception": "review",
            "duplicate": "review",
            "valid_none": "no_transaction",
            "transaction_write": "atomic_with_operation_settlement",
            "retry": "new_operation_with_parent_operation_id",
        },
        "compatibility": {
            "stored_operation_behavior": "original_release",
            "frozen_release_ids": [
                "native-integration-v1",
                "native-integration-v2",
                "native-integration-v3",
                "native-integration-v4",
            ],
        },
        "algorithms": predecessor["algorithms"],
        "runtime_policy": predecessor["runtime_policy"],
        "artifacts": [
            {"path": path, "contract": contract, "sha256": file_sha256(REPO_ROOT / path)}
            for path, contract in ARTIFACTS
        ],
    }
    jsonschema.validate(
        manifest,
        _read_json("configs/sms_processing/contracts/releases/v5/release-manifest.schema.json"),
    )
    return manifest


def main() -> None:
    print(json.dumps(build_manifest(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
