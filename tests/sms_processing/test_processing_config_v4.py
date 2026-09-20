"""Contract tests for system-managed and file-backed extractor identity provenance."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from pocketfinancer_sms.configuration import AssetBinding, ProcessingTrigger, RolloutMode
from pocketfinancer_sms.configuration_v4 import (
    MODEL_IDENTITY_FILE_SHA256,
    MODEL_IDENTITY_SYSTEM_MANAGED_RUNTIME,
    ExtractorRuntimeConfigV4,
    ProcessingConfigSnapshotV4,
)
from pocketfinancer_sms.types import TimestampProvenance

ROOT = Path(__file__).resolve().parents[2]
SHA = "a" * 64


def _runtime(kind: str, model_file_sha256: str | None, *, eligible: bool = True) -> ExtractorRuntimeConfigV4:
    return ExtractorRuntimeConfigV4(
        eligible=eligible,
        ineligibility_reason=None if eligible else "runtime_unavailable",
        model_identifier="com.apple.foundationmodels.system-language-model",
        model_file_sha256=model_file_sha256,
        model_identity_kind=kind,
        runtime_version="FoundationModels/1",
        os_version="iOS 26",
        device_cohort="synthetic-device",
        prompt_sha256="b" * 64,
        grammar_sha256="c" * 64,
        validation_profile_sha256="d" * 64,
    )


def _snapshot(runtime: ExtractorRuntimeConfigV4) -> ProcessingConfigSnapshotV4:
    return ProcessingConfigSnapshotV4(
        operation_id="11111111-1111-4111-8111-111111111111",
        parent_operation_id=None,
        source_ref_hash=SHA,
        trigger=ProcessingTrigger.REALTIME,
        created_at_epoch_ms=2,
        admission_epoch_ms=1,
        release_id="native-integration-v4",
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
        extractor=runtime,
        persistence_policy_version="pocketfinancer.persistence-policy/2",
        rollout_mode=RolloutMode.SHADOW,
    )


def _schema() -> dict:
    return json.loads((ROOT / "configs/sms_processing/contracts/v4/processing-config.schema.json").read_text())


def test_eligible_system_managed_runtime_uses_explicit_identity_without_fabricated_file_hash() -> None:
    payload = _snapshot(_runtime(MODEL_IDENTITY_SYSTEM_MANAGED_RUNTIME, None)).to_dict()
    jsonschema.validate(payload, _schema())
    assert payload["contract"] == "pocketfinancer.processing-config/4"
    assert payload["extractor"]["model_file_sha256"] is None


def test_eligible_file_backed_runtime_requires_real_sha256() -> None:
    jsonschema.validate(_snapshot(_runtime(MODEL_IDENTITY_FILE_SHA256, "9" * 64)).to_dict(), _schema())
    with pytest.raises(ValueError, match="requires a model hash"):
        _runtime(MODEL_IDENTITY_FILE_SHA256, None)


def test_ineligible_file_backed_runtime_may_omit_an_unavailable_file_hash_without_fabrication() -> None:
    jsonschema.validate(_snapshot(_runtime(MODEL_IDENTITY_FILE_SHA256, None, eligible=False)).to_dict(), _schema())
    jsonschema.validate(_snapshot(_runtime(MODEL_IDENTITY_FILE_SHA256, "9" * 64, eligible=False)).to_dict(), _schema())
    with pytest.raises(ValueError, match="model hash"):
        _runtime(MODEL_IDENTITY_FILE_SHA256, "not-a-hash", eligible=False)


@pytest.mark.parametrize("eligible", [True, False])
def test_system_managed_runtime_forbids_file_hash_at_every_eligibility_state(eligible: bool) -> None:
    jsonschema.validate(_snapshot(_runtime(MODEL_IDENTITY_SYSTEM_MANAGED_RUNTIME, None, eligible=eligible)).to_dict(), _schema())
    with pytest.raises(ValueError, match="must not claim"):
        _runtime(MODEL_IDENTITY_SYSTEM_MANAGED_RUNTIME, "9" * 64, eligible=eligible)