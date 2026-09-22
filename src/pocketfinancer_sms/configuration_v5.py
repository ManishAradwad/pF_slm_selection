"""Final automatic-routing configuration layered over frozen v1-v4 assets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from .configuration import AssetBinding, ProcessingTrigger, RolloutMode
from .configuration_v4 import ExtractorRuntimeConfigV4
from .currency import ISO_MINOR_UNITS
from .provenance import object_sha256
from .types import TimestampProvenance


PROCESSING_CONFIG_CONTRACT_V5 = "pocketfinancer.processing-config/5"
NATIVE_INTEGRATION_RELEASE_V5 = "native-integration-v5"
PERSISTENCE_POLICY_V2 = "pocketfinancer.persistence-policy/2"

_FROZEN_PREDECESSOR_RELEASES = {
    "native-integration-v1",
    "native-integration-v2",
    "native-integration-v3",
    "native-integration-v4",
}


def rollout_mode_for_stored_release(
    release_id: str,
    stored_rollout_mode: RolloutMode,
) -> RolloutMode:
    """Preserve a predecessor's stored mode and enforce the successor binding."""

    if release_id in _FROZEN_PREDECESSOR_RELEASES:
        return stored_rollout_mode
    if release_id == NATIVE_INTEGRATION_RELEASE_V5:
        if stored_rollout_mode != RolloutMode.AUTOMATIC:
            raise ValueError("successor stored operation has an incompatible rollout mode")
        return RolloutMode.AUTOMATIC
    raise ValueError("stored operation release is incompatible")


@dataclass(frozen=True, slots=True)
class ProcessingConfigSnapshotV5:
    operation_id: str
    parent_operation_id: str | None
    source_ref_hash: str
    trigger: ProcessingTrigger
    created_at_epoch_ms: int
    admission_epoch_ms: int
    release_id: str
    release_manifest_sha256: str
    analyzer_behavior_version: str
    unicode_behavior_version: str
    currency_asset_sha256: str
    profile_assets: tuple[AssetBinding, ...]
    primary_currency: str
    enabled_profile_ids: tuple[str, ...]
    received_at_epoch_ms: int
    received_timestamp_provenance: TimestampProvenance
    timezone_id: str
    extractor: ExtractorRuntimeConfigV4
    persistence_policy_version: str = PERSISTENCE_POLICY_V2
    rollout_mode: RolloutMode = RolloutMode.AUTOMATIC
    contract: str = PROCESSING_CONFIG_CONTRACT_V5

    def __post_init__(self) -> None:
        _require_uuid(self.operation_id)
        if self.parent_operation_id is not None:
            _require_uuid(self.parent_operation_id)
            if self.parent_operation_id == self.operation_id:
                raise ValueError("operation cannot be its own parent")
        if self.trigger == ProcessingTrigger.RETRY and self.parent_operation_id is None:
            raise ValueError("successor retry requires an explicit parent operation")
        if not _is_sha256(self.source_ref_hash):
            raise ValueError("configuration source reference hash is invalid")
        if (
            not _is_nonnegative_int(self.created_at_epoch_ms)
            or not _is_nonnegative_int(self.admission_epoch_ms)
            or not _is_nonnegative_int(self.received_at_epoch_ms)
            or self.created_at_epoch_ms < self.admission_epoch_ms
        ):
            raise ValueError("configuration timestamps are invalid")
        if (
            self.release_id != NATIVE_INTEGRATION_RELEASE_V5
            or not _is_sha256(self.release_manifest_sha256)
        ):
            raise ValueError("configuration release provenance is invalid")
        if (
            self.analyzer_behavior_version
            != "pocketfinancer.structural-sms-analyzer/2"
            or not self.unicode_behavior_version
        ):
            raise ValueError("configuration analyzer provenance is incomplete")
        if not _is_sha256(self.currency_asset_sha256):
            raise ValueError("configuration currency hash is invalid")
        if self.primary_currency not in ISO_MINOR_UNITS:
            raise ValueError("configuration primary currency is unsupported")
        if not self.enabled_profile_ids or len(set(self.enabled_profile_ids)) != len(
            self.enabled_profile_ids
        ):
            raise ValueError("configuration profiles are invalid")
        if tuple(asset.asset_id for asset in self.profile_assets) != self.enabled_profile_ids:
            raise ValueError("configuration profile hashes do not match enabled profiles")
        if self.received_timestamp_provenance not in {
            TimestampProvenance.PLATFORM_RECEIVED,
            TimestampProvenance.ACQUISITION_SUPPLIED_MESSAGE_TIME,
        }:
            raise ValueError("extractor receipt timestamp provenance is unsupported")
        if not self.timezone_id:
            raise ValueError("configuration timezone is missing")
        if self.persistence_policy_version != PERSISTENCE_POLICY_V2:
            raise ValueError("successor persistence policy is unsupported")
        if self.rollout_mode != RolloutMode.AUTOMATIC:
            raise ValueError("successor operations require automatic routing")
        if self.contract != PROCESSING_CONFIG_CONTRACT_V5:
            raise ValueError("configuration contract is unsupported")

    def payload(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "operation_id": self.operation_id,
            "parent_operation_id": self.parent_operation_id,
            "source_ref_hash": self.source_ref_hash,
            "trigger": self.trigger.value,
            "created_at_epoch_ms": self.created_at_epoch_ms,
            "admission_epoch_ms": self.admission_epoch_ms,
            "contract_release": {
                "release_id": self.release_id,
                "manifest_sha256": self.release_manifest_sha256,
            },
            "analyzer": {
                "behavior_version": self.analyzer_behavior_version,
                "unicode_behavior_version": self.unicode_behavior_version,
                "currency_asset_sha256": self.currency_asset_sha256,
                "profile_assets": [asset.to_dict() for asset in self.profile_assets],
            },
            "currency_context": {
                "primary_currency": self.primary_currency,
                "enabled_profile_ids": list(self.enabled_profile_ids),
            },
            "received_timestamp": {
                "epoch_ms": self.received_at_epoch_ms,
                "provenance": self.received_timestamp_provenance.value,
                "timezone_id": self.timezone_id,
                "read_only": True,
            },
            "extractor": self.extractor.to_dict(),
            "persistence_policy": {
                "version": self.persistence_policy_version,
                "rollout_mode": self.rollout_mode.value,
            },
        }

    @property
    def config_hash(self) -> str:
        return object_sha256(self.payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "config_hash": self.config_hash}


def _require_uuid(value: str) -> None:
    try:
        parsed = UUID(value)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("configuration operation ID is invalid") from exc
    if str(parsed) != value.lower():
        raise ValueError("configuration operation ID is not canonical")


def _is_sha256(value: str | None) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _is_nonnegative_int(value: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0
