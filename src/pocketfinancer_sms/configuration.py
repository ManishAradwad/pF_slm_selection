"""Immutable, canonically hashed processing configuration snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from uuid import UUID

from .currency import ISO_MINOR_UNITS
from .provenance import object_sha256
from .selector import SELECTOR_VALIDATION_PROFILE
from .types import TimestampProvenance


PROCESSING_CONFIG_CONTRACT = "pocketfinancer.processing-config/1"


class ProcessingTrigger(StrEnum):
    REALTIME = "realtime"
    HISTORICAL = "historical"
    MANUAL = "manual"
    RETRY = "retry"
    DIAGNOSTIC = "diagnostic"
    APP_INTENT = "app_intent"
    BACKGROUND_RECOVERY = "background_recovery"


class RolloutMode(StrEnum):
    SHADOW = "shadow"
    REVIEW_ONLY = "review_only"
    AUTOMATIC = "automatic"


@dataclass(frozen=True, slots=True)
class AssetBinding:
    asset_id: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.asset_id or not _is_sha256(self.sha256):
            raise ValueError("configuration asset binding is invalid")

    def to_dict(self) -> dict[str, str]:
        return {"asset_id": self.asset_id, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class SelectorRuntimeConfig:
    eligible: bool
    ineligibility_reason: str | None
    model_identifier: str | None
    model_file_sha256: str | None
    runtime_version: str
    os_version: str
    device_cohort: str
    prompt_version: str
    prompt_sha256: str
    validation_profile: str = SELECTOR_VALIDATION_PROFILE
    generation_mode: str = "DIRECT_NON_THINKING"
    decoding: str = "greedy"
    answer_token_limit: int = 512
    raw_output_utf8_byte_limit: int = 16_384
    parser_deadline_ms: int = 60_000

    def __post_init__(self) -> None:
        if self.eligible:
            if self.ineligibility_reason is not None or not self.model_identifier:
                raise ValueError("eligible selector runtime metadata is inconsistent")
        elif not self.ineligibility_reason:
            raise ValueError("ineligible selector runtime requires a reason")
        if self.model_file_sha256 is not None and not _is_sha256(self.model_file_sha256):
            raise ValueError("selector model file hash is invalid")
        required = (
            self.runtime_version,
            self.os_version,
            self.device_cohort,
            self.prompt_version,
        )
        if not all(required) or not _is_sha256(self.prompt_sha256):
            raise ValueError("selector runtime provenance is incomplete")
        if (
            self.validation_profile != SELECTOR_VALIDATION_PROFILE
            or self.generation_mode != "DIRECT_NON_THINKING"
            or self.decoding != "greedy"
            or self.answer_token_limit != 512
            or self.raw_output_utf8_byte_limit != 16_384
            or self.parser_deadline_ms != 60_000
        ):
            raise ValueError("selector runtime policy does not match the frozen profile")

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "ineligibility_reason": self.ineligibility_reason,
            "model_identifier": self.model_identifier,
            "model_file_sha256": self.model_file_sha256,
            "runtime_version": self.runtime_version,
            "os_version": self.os_version,
            "device_cohort": self.device_cohort,
            "prompt_version": self.prompt_version,
            "prompt_sha256": self.prompt_sha256,
            "validation_profile": self.validation_profile,
            "generation_mode": self.generation_mode,
            "decoding": self.decoding,
            "answer_token_limit": self.answer_token_limit,
            "raw_output_utf8_byte_limit": self.raw_output_utf8_byte_limit,
            "parser_deadline_ms": self.parser_deadline_ms,
        }


@dataclass(frozen=True, slots=True)
class ProcessingConfigSnapshot:
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
    source_timestamp_epoch_ms: int | None
    source_timestamp_provenance: TimestampProvenance
    timezone_id: str
    timestamp_policy_version: str
    selector: SelectorRuntimeConfig
    persistence_policy_version: str
    rollout_mode: RolloutMode
    contract: str = PROCESSING_CONFIG_CONTRACT

    def __post_init__(self) -> None:
        _require_uuid(self.operation_id, "operation")
        if self.parent_operation_id is not None:
            _require_uuid(self.parent_operation_id, "parent operation")
            if self.parent_operation_id == self.operation_id:
                raise ValueError("operation cannot be its own parent")
        if not _is_sha256(self.source_ref_hash):
            raise ValueError("configuration source reference hash is invalid")
        if not _is_nonnegative_int(self.created_at_epoch_ms) or not _is_nonnegative_int(
            self.admission_epoch_ms
        ):
            raise ValueError("configuration timestamps are invalid")
        if self.created_at_epoch_ms < self.admission_epoch_ms:
            raise ValueError("configuration cannot predate admission")
        hashes = (self.release_manifest_sha256, self.currency_asset_sha256)
        if not self.release_id or not all(_is_sha256(value) for value in hashes):
            raise ValueError("configuration release provenance is invalid")
        if not self.analyzer_behavior_version or not self.unicode_behavior_version:
            raise ValueError("configuration analyzer provenance is incomplete")
        if self.primary_currency not in ISO_MINOR_UNITS:
            raise ValueError("configuration primary currency is unsupported")
        if not self.enabled_profile_ids or len(set(self.enabled_profile_ids)) != len(
            self.enabled_profile_ids
        ):
            raise ValueError("configuration profiles are invalid")
        if tuple(asset.asset_id for asset in self.profile_assets) != self.enabled_profile_ids:
            raise ValueError("configuration profile hashes do not match enabled profiles")
        if not isinstance(self.source_timestamp_provenance, TimestampProvenance):
            raise ValueError("configuration timestamp provenance is unsupported")
        if self.source_timestamp_epoch_ms is None:
            if self.source_timestamp_provenance != TimestampProvenance.UNKNOWN:
                raise ValueError("configuration timestamp provenance requires a timestamp")
        elif not _is_nonnegative_int(self.source_timestamp_epoch_ms):
            raise ValueError("configuration source timestamp is invalid")
        required = (self.timezone_id, self.timestamp_policy_version, self.persistence_policy_version)
        if not all(required):
            raise ValueError("configuration policy provenance is incomplete")
        if self.contract != PROCESSING_CONFIG_CONTRACT:
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
            "source_timestamp": {
                "epoch_ms": self.source_timestamp_epoch_ms,
                "provenance": self.source_timestamp_provenance.value,
                "timezone_id": self.timezone_id,
                "policy_version": self.timestamp_policy_version,
            },
            "selector": self.selector.to_dict(),
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


def _require_uuid(value: str, label: str) -> None:
    try:
        parsed = UUID(value)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"configuration {label} ID is invalid") from exc
    if str(parsed) != value.lower():
        raise ValueError(f"configuration {label} ID is not canonical")


def _is_sha256(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _is_nonnegative_int(value: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0
