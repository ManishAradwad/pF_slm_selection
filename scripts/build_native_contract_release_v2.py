#!/usr/bin/env python3
"""Render the native-integration-v2 release manifest from checked-in assets."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pocketfinancer_sms.provenance import file_sha256  # noqa: E402


ARTIFACTS = (
    ("configs/sms_processing/contracts/sms-analysis.schema.json", "pocketfinancer.sms-analysis/1"),
    ("configs/sms_processing/contracts/v2/sms-analysis.schema.json", "pocketfinancer.sms-analysis/2"),
    (
        "configs/sms_processing/contracts/grounded-candidate-selector-input.schema.json",
        "pocketfinancer.grounded-candidate-selector-input/1",
    ),
    (
        "configs/sms_processing/contracts/grounded-candidate-selector.schema.json",
        "pocketfinancer.grounded-candidate-selector/1",
    ),
    (
        "configs/sms_processing/contracts/v3/selector-validation-profile.json",
        "pocketfinancer.selector-validation-profile/3",
    ),
    (
        "configs/sms_processing/contracts/v3/selector-validation-profile.schema.json",
        "pocketfinancer.selector-validation-profile-schema/3",
    ),
    (
        "configs/sms_processing/contracts/v2/processing-config.schema.json",
        "pocketfinancer.processing-config/2",
    ),
    (
        "configs/sms_processing/contracts/processing-result.schema.json",
        "pocketfinancer.processing-result/1",
    ),
    (
        "configs/sms_processing/contracts/v2/processing-result.schema.json",
        "pocketfinancer.processing-result/2",
    ),
    (
        "configs/sms_processing/contracts/processing-trace.schema.json",
        "pocketfinancer.processing-trace/1",
    ),
    (
        "configs/sms_processing/contracts/v2/processing-trace.schema.json",
        "pocketfinancer.processing-trace/2",
    ),
    (
        "configs/sms_processing/contracts/user-feedback.schema.json",
        "pocketfinancer.user-feedback/1",
    ),
    (
        "configs/sms_processing/contracts/v2/user-feedback.schema.json",
        "pocketfinancer.user-feedback/2",
    ),
    (
        "configs/sms_processing/contracts/canonical-label.schema.json",
        "pocketfinancer.canonical-label/1",
    ),
    (
        "configs/sms_processing/contracts/native-trace-bundle.schema.json",
        "pocketfinancer.native-trace-bundle/1",
    ),
    (
        "configs/sms_processing/contracts/v2/reason-code-registry.json",
        "pocketfinancer.reason-code-registry/1",
    ),
    (
        "configs/sms_processing/contracts/v2/reason-code-registry.schema.json",
        "pocketfinancer.reason-code-registry-schema/1",
    ),
    ("configs/sms_processing/currency/iso-4217.json", "pocketfinancer.supported-currencies/1"),
    ("configs/sms_processing/profiles/core-en.json", "pocketfinancer.analyzer-profile/1:core-en"),
    ("configs/sms_processing/profiles/india.json", "pocketfinancer.analyzer-profile/1:india"),
    (
        "configs/sms_processing/policies/native-persistence-v1.json",
        "pocketfinancer.persistence-policy/1",
    ),
    (
        "configs/sms_processing/policies/timestamp-v1.json",
        "pocketfinancer.timestamp-policy/1",
    ),
    (
        "configs/sms_processing/prompts/grounded-candidate-selector-v1.txt",
        "pocketfinancer.selector-prompt/1",
    ),
    (
        "tests/sms_processing/golden/native-v1/parity-bundle.json",
        "pocketfinancer.native-parity-golden/1",
    ),
    (
        "tests/sms_processing/golden/native-v1/financial-state.json",
        "pocketfinancer.native-golden-financial-state/1",
    ),
    (
        "tests/sms_processing/golden/native-v1/selector-validation.json",
        "pocketfinancer.native-golden-selector-validation/1",
    ),
)


def build_manifest() -> dict:
    artifacts = [
        {"path": path, "contract": contract, "sha256": file_sha256(REPO_ROOT / path)}
        for path, contract in ARTIFACTS
    ]
    return {
        "contract": "pocketfinancer.contract-release-manifest/1",
        "release_id": "native-integration-v2",
        "status": "frozen_for_native_implementation",
        "automatic_persistence_enabled": False,
        "algorithms": {
            "analysis_id_v1": "sha256(operation_id NUL source_sha256 NUL currency_context_hash)[:24]",
            "analysis_id_v2": "sha256(operation_id NUL source_sha256 NUL operation_config_hash NUL analyzer_behavior_version)[:24]",
            "candidate_id": "kind_prefix + '_' + sha256(analysis_id '|' kind '|' span_or_absent '|' canonical_value_json)[:12]",
            "canonical_json": "utf8-json-sort-keys-compact-no-ascii-escaping",
            "unicode_behavior": "Unicode 14.0.0 per-code-point NFKC then casefold; collapse Unicode whitespace",
        },
        "runtime_policy": {
            "generation_mode": "DIRECT_NON_THINKING",
            "decoding": "greedy",
            "answer_token_limit": 512,
            "raw_output_utf8_byte_limit": 16_384,
            "parser_deadline_ms": 0,
            "claim_lease_ms": 120_000,
            "claim_heartbeat_ms": 15_000,
            "automatic_retry_limit": 3,
        },
        "artifacts": artifacts,
    }


def main() -> None:
    print(json.dumps(build_manifest(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
