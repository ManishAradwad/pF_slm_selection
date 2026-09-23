"""Configuration and compatibility tests for the final automatic-routing release."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import jsonschema
import pytest

from pocketfinancer_sms.configuration import AssetBinding, ProcessingTrigger, RolloutMode
from pocketfinancer_sms.configuration_v4 import (
    MODEL_IDENTITY_FILE_SHA256,
    ExtractorRuntimeConfigV4,
)
from pocketfinancer_sms.configuration_v6 import (
    NATIVE_INTEGRATION_RELEASE_V6,
    ProcessingConfigSnapshotV6,
    rollout_mode_for_stored_release,
)
from pocketfinancer_sms.types import TimestampProvenance


ROOT = Path(__file__).resolve().parents[2]


def _runtime() -> ExtractorRuntimeConfigV4:
    return ExtractorRuntimeConfigV4(
        eligible=True,
        ineligibility_reason=None,
        model_identifier="synthetic.gguf",
        model_file_sha256="9" * 64,
        model_identity_kind=MODEL_IDENTITY_FILE_SHA256,
        runtime_version="synthetic-runtime",
        os_version="synthetic-os",
        device_cohort="synthetic-device",
        prompt_sha256="b" * 64,
        grammar_sha256="c" * 64,
        validation_profile_sha256="d" * 64,
    )


def _snapshot(
    *,
    operation_id: str = "11111111-1111-4111-8111-111111111111",
    parent_operation_id: str | None = None,
    trigger: ProcessingTrigger = ProcessingTrigger.REALTIME,
) -> ProcessingConfigSnapshotV6:
    return ProcessingConfigSnapshotV6(
        operation_id=operation_id,
        parent_operation_id=parent_operation_id,
        source_ref_hash="a" * 64,
        trigger=trigger,
        created_at_epoch_ms=2,
        admission_epoch_ms=1,
        release_id=NATIVE_INTEGRATION_RELEASE_V6,
        release_manifest_sha256="e" * 64,
        analyzer_behavior_version="pocketfinancer.structural-sms-analyzer/2",
        unicode_behavior_version="unicode-scalar-nfkc-casefold",
        currency_asset_sha256="f" * 64,
        profile_assets=(AssetBinding("core-en", "1" * 64),),
        primary_currency="INR",
        enabled_profile_ids=("core-en",),
        received_at_epoch_ms=1,
        received_timestamp_provenance=TimestampProvenance.PLATFORM_RECEIVED,
        timezone_id="Asia/Kolkata",
        extractor=_runtime(),
        grammar_enabled=False,
    )


def _schema() -> dict:
    return json.loads(
        (ROOT / "configs/sms_processing/contracts/v6/processing-config.schema.json").read_text()
    )


def test_successor_configuration_binds_automatic_policy() -> None:
    snapshot = _snapshot()
    payload = snapshot.to_dict()
    jsonschema.validate(payload, _schema())
    assert payload["contract"] == "pocketfinancer.processing-config/6"
    assert payload["contract_release"]["release_id"] == "native-integration-v6"
    assert payload["persistence_policy"] == {
        "version": "pocketfinancer.persistence-policy/2",
        "rollout_mode": "automatic",
    }
    with pytest.raises(ValueError, match="automatic routing"):
        replace(snapshot, rollout_mode=RolloutMode.REVIEW_ONLY)


def test_retry_is_a_distinct_operation_with_explicit_parent() -> None:
    parent = "11111111-1111-4111-8111-111111111111"
    retry = _snapshot(
        operation_id="22222222-2222-4222-8222-222222222222",
        parent_operation_id=parent,
        trigger=ProcessingTrigger.RETRY,
    )
    jsonschema.validate(retry.to_dict(), _schema())
    assert retry.operation_id != parent
    assert retry.parent_operation_id == parent
    with pytest.raises(ValueError, match="explicit parent"):
        _snapshot(trigger=ProcessingTrigger.RETRY)
    with pytest.raises(ValueError, match="own parent"):
        _snapshot(parent_operation_id=parent, trigger=ProcessingTrigger.RETRY)


def test_stored_operations_retain_release_owned_routing() -> None:
    assert (
        rollout_mode_for_stored_release("native-integration-v1", RolloutMode.SHADOW)
        == RolloutMode.SHADOW
    )
    assert (
        rollout_mode_for_stored_release(
            "native-integration-v4", RolloutMode.REVIEW_ONLY
        )
        == RolloutMode.REVIEW_ONLY
    )
    assert (
        rollout_mode_for_stored_release(
            "native-integration-v6", RolloutMode.AUTOMATIC
        )
        == RolloutMode.AUTOMATIC
    )
    with pytest.raises(ValueError, match="incompatible rollout"):
        rollout_mode_for_stored_release(
            "native-integration-v6", RolloutMode.REVIEW_ONLY
        )
    with pytest.raises(ValueError, match="incompatible"):
        rollout_mode_for_stored_release(
            "native-integration-v999", RolloutMode.REVIEW_ONLY
        )


def test_grammar_mode_is_required_and_bound_to_the_configuration_hash() -> None:
    off = _snapshot()
    on = replace(off, grammar_enabled=True)
    jsonschema.validate(off.to_dict(), _schema())
    jsonschema.validate(on.to_dict(), _schema())
    assert off.to_dict()["extractor"]["grammar_enabled"] is False
    assert on.to_dict()["extractor"]["grammar_enabled"] is True
    assert off.config_hash != on.config_hash

    missing = off.to_dict()
    del missing["extractor"]["grammar_enabled"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(missing, _schema())
