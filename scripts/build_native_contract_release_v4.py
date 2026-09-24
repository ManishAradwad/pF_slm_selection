#!/usr/bin/env python3
"""Build native-integration-v4 after every extractor freeze check passes."""

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
    from scripts.build_native_contract_release_v2 import (  # noqa: E402
        ARTIFACTS as LEGACY_ARTIFACTS,
    )
except ModuleNotFoundError:
    from build_native_contract_release_v2 import (  # type: ignore[no-redef]  # noqa: E402
        ARTIFACTS as LEGACY_ARTIFACTS,
    )


NEW_ARTIFACTS = (
    (
        "configs/sms_processing/contracts/v3/sms-extractor-input.schema.json",
        "pocketfinancer.sms-extractor-input/1",
    ),
    (
        "configs/sms_processing/contracts/v3/sms-extractor.schema.json",
        "pocketfinancer.sms-extractor/1",
    ),
    (
        "configs/sms_processing/contracts/v3/extractor-validation-profile.json",
        "pocketfinancer.extractor-validation-profile/1",
    ),
    (
        "configs/sms_processing/contracts/v3/extractor-validation-profile.schema.json",
        "pocketfinancer.extractor-validation-profile-schema/1",
    ),
    (
        "configs/sms_processing/prompts/sms-extractor-v1.txt",
        "pocketfinancer.extractor-prompt/1",
    ),
    (
        "configs/sms_processing/grammars/sms-extractor-v1.gbnf",
        "pocketfinancer.extractor-grammar/1",
    ),
    (
        "configs/sms_processing/contracts/v3/processing-config.schema.json",
        "pocketfinancer.processing-config/3",
    ),
    (
        "configs/sms_processing/contracts/v3/processing-result.schema.json",
        "pocketfinancer.processing-result/3",
    ),
    (
        "configs/sms_processing/contracts/v3/processing-trace.schema.json",
        "pocketfinancer.processing-trace/3",
    ),
    (
        "configs/sms_processing/contracts/v3/reason-code-registry.json",
        "pocketfinancer.reason-code-registry/2",
    ),
    (
        "configs/sms_processing/contracts/v3/reason-code-registry.schema.json",
        "pocketfinancer.reason-code-registry-schema/2",
    ),
    (
        "configs/sms_processing/contracts/v3/account-resolution-profile.json",
        "pocketfinancer.account-resolution-profile/1",
    ),
    (
        "configs/sms_processing/contracts/v3/account-resolution-profile.schema.json",
        "pocketfinancer.account-resolution-profile-schema/1",
    ),
    (
        "configs/sms_processing/contracts/v3/review-case.schema.json",
        "pocketfinancer.review-case/1",
    ),
    (
        "configs/sms_processing/contracts/v3/user-feedback.schema.json",
        "pocketfinancer.user-feedback/3",
    ),
    (
        "configs/sms_processing/contracts/v3/canonical-label.schema.json",
        "pocketfinancer.canonical-label/2",
    ),
    (
        "configs/sms_processing/contracts/releases/v3/release-manifest.schema.json",
        "pocketfinancer.contract-release-manifest-schema/3",
    ),
    (
        "tests/sms_processing/golden/extractor-v1/sanitized-vectors.json",
        "pocketfinancer.sanitized-extractor-golden/1",
    ),
)
V4_ARTIFACTS = (
    ("configs/sms_processing/contracts/v4/processing-config.schema.json", "pocketfinancer.processing-config/4"),
    ("configs/sms_processing/contracts/releases/v4/release-manifest.schema.json", "pocketfinancer.contract-release-manifest-schema/4"),
)
_SUPERSEDED_V3_PATHS = {
    "configs/sms_processing/contracts/v3/processing-config.schema.json",
    "configs/sms_processing/contracts/releases/v3/release-manifest.schema.json",
}
ARTIFACTS = tuple(item for item in LEGACY_ARTIFACTS + NEW_ARTIFACTS if item[0] not in _SUPERSEDED_V3_PATHS) + V4_ARTIFACTS

_JSON_ASSET_CONTRACTS = {
    path: contract
    for path, contract in NEW_ARTIFACTS + V4_ARTIFACTS
    if path.startswith("configs/") and path.endswith(".json")
}
_SCHEMA_PATHS = tuple(path for path, _ in NEW_ARTIFACTS + V4_ARTIFACTS if path.endswith(".schema.json"))
_SCHEMA_INSTANCE_PAIRS = (
    (
        "configs/sms_processing/contracts/v3/extractor-validation-profile.json",
        "configs/sms_processing/contracts/v3/extractor-validation-profile.schema.json",
    ),
    (
        "configs/sms_processing/contracts/v3/account-resolution-profile.json",
        "configs/sms_processing/contracts/v3/account-resolution-profile.schema.json",
    ),
    (
        "configs/sms_processing/contracts/v3/reason-code-registry.json",
        "configs/sms_processing/contracts/v3/reason-code-registry.schema.json",
    ),
)
_HISTORICAL_FILE_HASHES = {
    "configs/sms_processing/contracts/releases/native-integration-v1.json": "e07ac6d2f6e90fac914db824d104141a20e49fc40f8b8f02c8fec4c0614e680a",
    "configs/sms_processing/contracts/releases/native-integration-v2.json": "637013f0988a20eb070e10b07f68ddf9172b847262024a50676c90022234019d",
    "tests/sms_processing/golden/native-v1/financial-state.json": "9c9cdb06c030f6e55ba9177cb3158ce26b70adeb3b1c6052e3623e8e6bbc3d68",
    "tests/sms_processing/golden/native-v1/parity-bundle.json": "6f34fdb2ba992ff7144acc41424d8cb90dde78ed0fea3af8a69431b3fa6a6c35",
    "tests/sms_processing/golden/native-v1/selector-validation.json": "47c960df0791c5e93c4911be630e127cdcd46b09b046ad77e8cc7c6498fc7c66",
}


def _read_json(path: str) -> Any:
    with (REPO_ROOT / path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _validate_historical_assets() -> None:
    for path, expected_sha256 in _HISTORICAL_FILE_HASHES.items():
        actual_sha256 = file_sha256(REPO_ROOT / path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"historical asset changed: {path} "
                f"expected {expected_sha256}, found {actual_sha256}"
            )

    legacy_release = _read_json(
        "configs/sms_processing/contracts/releases/native-integration-v2.json"
    )
    frozen = {
        item["path"]: (item["contract"], item["sha256"]) for item in legacy_release["artifacts"]
    }
    for path, contract in LEGACY_ARTIFACTS:
        expected = frozen.get(path)
        actual = (contract, file_sha256(REPO_ROOT / path))
        if expected != actual:
            raise ValueError(
                "historical native v2 asset changed: "
                f"{path} expected {expected!r}, found {actual!r}"
            )


def _span(source: str, text: str) -> dict[str, Any]:
    start = source.index(text)
    return {
        "start_scalar": start,
        "end_scalar": start + len(text),
        "text": text,
    }


def build_sanitized_vectors() -> dict[str, Any]:
    """Reconstruct the invented extractor vectors without private inputs."""

    emoji_source = "Alert \U0001f4b3: INR 1,250.00 debited from a/c XX1234 at Caf\u00e9."
    combining_source = "Cafe\u0301 INR 42.00 credited to account XX4321."
    negative_source = "Your statement is ready to view in the app."
    missing_account_source = "INR 75.00 was debited."
    mismatch_source = "INR 1,250.00 debited from account XX9999."
    return {
        "span_coordinate_system": "zero-based-half-open-unicode-scalars",
        "cases": [
            {
                "id": "unicode_emoji_zero_evidence",
                "sms_body": emoji_source,
                "advisory_evidence": [],
                "model_output": {
                    "decision": "posted",
                    "amount": {
                        "value": "1250.00",
                        "currency": "INR",
                        "evidence": _span(emoji_source, "INR 1,250.00"),
                    },
                    "direction": {
                        "value": "debit",
                        "evidence": _span(emoji_source, "debited"),
                    },
                    "account": {
                        "reference": "XX1234",
                        "evidence": _span(emoji_source, "XX1234"),
                    },
                    "counterparty": {
                        "value": "Caf\u00e9",
                        "evidence": _span(emoji_source, "Caf\u00e9"),
                    },
                },
                "expected": {
                    "decision": "posted",
                    "amount_minor_units": 125000,
                    "amount_currency": "INR",
                    "direction": "debit",
                    "account_reference": "1234",
                    "counterparty": "caf\u00e9",
                },
            },
            {
                "id": "combining_mark_conflicting_advice",
                "sms_body": combining_source,
                "advisory_evidence": [
                    {
                        "kind": "amount",
                        "source_span": _span(combining_source, "INR"),
                        "clause": "wrong hint",
                        "suggested_interpretation": {"value": "999.00"},
                        "provenance": {
                            "candidate_id": "amt_deadbeef0000",
                            "analyzer_kind": "candidate",
                        },
                        "analyzer_version": "pocketfinancer.sms-analysis/2",
                    }
                ],
                "model_output": {
                    "decision": "posted",
                    "amount": {
                        "value": "42.00",
                        "currency": "INR",
                        "evidence": _span(combining_source, "INR 42.00"),
                    },
                    "direction": {
                        "value": "credit",
                        "evidence": _span(combining_source, "credited"),
                    },
                    "account": {
                        "reference": "XX4321",
                        "evidence": _span(combining_source, "XX4321"),
                    },
                    "counterparty": None,
                },
                "expected": {
                    "decision": "posted",
                    "amount_minor_units": 4200,
                    "amount_currency": "INR",
                    "direction": "credit",
                    "account_reference": "4321",
                    "counterparty": None,
                },
            },
            {
                "id": "sanitized_negative",
                "sms_body": negative_source,
                "advisory_evidence": [],
                "model_output": {"decision": "none"},
                "expected": {"decision": "none"},
            },
            {
                "id": "mandatory_account_missing",
                "sms_body": missing_account_source,
                "advisory_evidence": [],
                "model_output": {"decision": "abstain"},
                "expected": {"decision": "abstain"},
            },
            {
                "id": "adversarial_mismatched_span",
                "sms_body": mismatch_source,
                "advisory_evidence": [],
                "model_output": {
                    "decision": "posted",
                    "amount": {
                        "value": "1250.00",
                        "currency": "INR",
                        "evidence": _span(mismatch_source, "INR 1,250.00"),
                    },
                    "direction": {
                        "value": "debit",
                        "evidence": _span(mismatch_source, "debited"),
                    },
                    "account": {
                        "reference": "XX0000",
                        "evidence": {
                            "start_scalar": mismatch_source.index("XX9999"),
                            "end_scalar": mismatch_source.index("XX9999") + len("XX9999"),
                            "text": "XX0000",
                        },
                    },
                    "counterparty": None,
                },
                "expected_reason": "extractor_evidence_mismatch",
            },
        ],
    }


def _validate_goldens() -> None:
    golden = _read_json("tests/sms_processing/golden/extractor-v1/sanitized-vectors.json")
    rebuilt = build_sanitized_vectors()
    if golden != rebuilt:
        raise ValueError("extractor sanitized goldens do not match deterministic rebuild")
    if golden.get("span_coordinate_system") != ("zero-based-half-open-unicode-scalars"):
        raise ValueError("extractor golden span coordinate system is not frozen")
    output_schema = _read_json("configs/sms_processing/contracts/v3/sms-extractor.schema.json")
    for case in golden.get("cases", []):
        source = case["sms_body"]
        output = case["model_output"]
        jsonschema.validate(output, output_schema)
        mismatches = 0
        for field in ("amount", "direction", "account", "counterparty"):
            item = output.get(field)
            if not isinstance(item, dict):
                continue
            span = item["evidence"]
            start = span["start_scalar"]
            end = span["end_scalar"]
            exact = 0 <= start < end <= len(source) and source[start:end] == span["text"]
            mismatches += not exact
        mismatch_expected = case.get("expected_reason") == "extractor_evidence_mismatch"
        if mismatch_expected != bool(mismatches):
            raise ValueError(f"extractor golden span expectation is invalid: {case['id']}")


def validate_freeze_inputs() -> None:
    """Refuse partial, invalid, or historically destructive v3 releases."""

    missing = [path for path, _ in NEW_ARTIFACTS + V4_ARTIFACTS if not (REPO_ROOT / path).is_file()]
    if missing:
        raise ValueError(f"native-integration-v4 assets are incomplete: {', '.join(missing)}")
    if len({path for path, _ in ARTIFACTS}) != len(ARTIFACTS):
        raise ValueError("native-integration-v4 contains duplicate artifact paths")

    for path, contract in _JSON_ASSET_CONTRACTS.items():
        value = _read_json(path)
        declared = value.get("$id", value.get("contract")) if isinstance(value, dict) else None
        if declared != contract:
            raise ValueError(f"{path} declares {declared!r}, expected {contract!r}")
    for path in _SCHEMA_PATHS:
        jsonschema.Draft202012Validator.check_schema(_read_json(path))
    for instance_path, schema_path in _SCHEMA_INSTANCE_PAIRS:
        jsonschema.validate(_read_json(instance_path), _read_json(schema_path))
    registry = _read_json("configs/sms_processing/contracts/v3/reason-code-registry.json")
    codes = [item["code"] for namespace in registry["namespaces"] for item in namespace["codes"]]
    if len(codes) != len(set(codes)):
        raise ValueError("reason-code-registry/2 contains duplicate codes")

    prompt = (REPO_ROOT / "configs/sms_processing/prompts/sms-extractor-v1.txt").read_text(
        encoding="utf-8"
    )
    if prompt.splitlines()[0] != "pocketfinancer.extractor-prompt/1":
        raise ValueError("extractor prompt contract identifier is missing")
    grammar = (REPO_ROOT / "configs/sms_processing/grammars/sms-extractor-v1.gbnf").read_text(
        encoding="utf-8"
    )
    grammar_tokens = (
        "none ::= ",
        "abstain ::= ",
        "posted ::= ",
        r"\"start_scalar\"",
        r"\"end_scalar\"",
    )
    for token in grammar_tokens:
        if token not in grammar:
            raise ValueError(f"extractor grammar is incomplete: missing {token}")

    _validate_goldens()
    _validate_historical_assets()


def build_manifest() -> dict[str, Any]:
    validate_freeze_inputs()
    artifacts = [
        {
            "path": path,
            "contract": contract,
            "sha256": file_sha256(REPO_ROOT / path),
        }
        for path, contract in ARTIFACTS
    ]
    manifest = {
        "contract": "pocketfinancer.contract-release-manifest/1",
        "release_id": "native-integration-v4",
        "status": "frozen_for_native_implementation",
        "automatic_persistence_enabled": False,
        "algorithms": {
            "analysis_id_v1": "sha256(operation_id NUL source_sha256 NUL currency_context_hash)[:24]",
            "analysis_id_v2": "sha256(operation_id NUL source_sha256 NUL operation_config_hash NUL analyzer_behavior_version)[:24]",
            "candidate_id": "kind_prefix + '_' + sha256(analysis_id '|' kind '|' span_or_absent '|' canonical_value_json)[:12]",
            "canonical_json": "utf8-json-sort-keys-compact-no-ascii-escaping",
            "unicode_behavior": "Unicode 14.0.0 per-code-point NFKC then casefold; collapse Unicode whitespace",
            "source_spans": "zero-based-half-open-Unicode-scalar-offsets-with-exact-text",
            "kotlin_span_conversion": "convert UTF-16 indices through code points; reject split surrogate pairs",
            "swift_span_conversion": "convert String.Index selections through unicodeScalars; grapheme offsets are not scalar offsets",
            "account_resolution": "NFKC-trim-casefold; first VPA else exactly one masked/bare 3-8 digit suffix; vpa:/suffix: key; exactly one owned alias; no default",
            "duplicate_assessment": "idempotency/source-event key then transaction fingerprint; only clear is eligible",
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
    jsonschema.validate(
        manifest,
        _read_json("configs/sms_processing/contracts/releases/v4/release-manifest.schema.json"),
    )
    return manifest


def main() -> None:
    print(json.dumps(build_manifest(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
