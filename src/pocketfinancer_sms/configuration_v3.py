"""Immutable, canonically hashed extractor-era processing configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from .configuration import AssetBinding, ProcessingTrigger, RolloutMode
from .currency import ISO_MINOR_UNITS
from .extractor import EXTRACTOR_PROMPT, EXTRACTOR_VALIDATION_PROFILE
from .provenance import object_sha256
from .types import TimestampProvenance


PROCESSING_CONFIG_CONTRACT_V3 = "pocketfinancer.processing-config/3"


@dataclass(frozen=True, slots=True)
class ExtractorRuntimeConfig:
    eligible: bool
    ineligibility_reason: str | None
    model_identifier: str | None
    model_file_sha256: str | None
    runtime_version: str
    os_version: str
    device_cohort: str
    prompt_sha256: str
    grammar_sha256: str
    validation_profile_sha256: str
    prompt_version: str = EXTRACTOR_PROMPT
    validation_profile: str = EXTRACTOR_VALIDATION_PROFILE
    grammar_version: str = "pocketfinancer.extractor-grammar/1"
    generation_mode: str = "DIRECT_NON_THINKING"
    decoding: str = "greedy"
    answer_token_limit: int = 512
    raw_output_utf8_byte_limit: int = 16_384
    parser_deadline_ms: int = 0

    def __post_init__(self) -> None:
        if self.eligible:
            if (
                self.ineligibility_reason is not None
                or not self.model_identifier
                or self.model_file_sha256 is None
            ):
                raise ValueError("eligible extractor runtime metadata is inconsistent")
        elif not self.ineligibility_reason:
            raise ValueError("ineligible extractor runtime requires a reason")
        for value, label in (
            (self.prompt_sha256, "prompt"),
            (self.grammar_sha256, "grammar"),
            (self.validation_profile_sha256, "validation profile"),
        ):
            if not _is_sha256(value):
                raise ValueError(f"extractor {label} hash is invalid")
        if self.model_file_sha256 is not None and not _is_sha256(self.model_file_sha256):
            raise ValueError("extractor model hash is invalid")
        if not all((self.runtime_version, self.os_version, self.device_cohort)):
            raise ValueError("extractor runtime provenance is incomplete")
        if (
            self.prompt_version != EXTRACTOR_PROMPT
            or self.validation_profile != EXTRACTOR_VALIDATION_PROFILE
            or self.grammar_version != "pocketfinancer.extractor-grammar/1"
            or self.generation_mode != "DIRECT_NON_THINKING"
            or self.decoding != "greedy"
            or self.answer_token_limit != 512
            or self.raw_output_utf8_byte_limit != 16_384
            or self.parser_deadline_ms != 0
        ):
            raise ValueError("extractor runtime policy does not match the frozen profile")

    def to_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class ProcessingConfigSnapshotV3:
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
    extractor: ExtractorRuntimeConfig
    persistence_policy_version: str
    rollout_mode: RolloutMode
    contract: str = PROCESSING_CONFIG_CONTRACT_V3

    def __post_init__(self) -> None:
        _require_uuid(self.operation_id)
        if self.parent_operation_id is not None:
            _require_uuid(self.parent_operation_id)
            if self.parent_operation_id == self.operation_id:
                raise ValueError("operation cannot be its own parent")
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
            self.release_id != "native-integration-v3"
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
        if not self.timezone_id or not self.persistence_policy_version:
            raise ValueError("configuration policy provenance is incomplete")
        if self.contract != PROCESSING_CONFIG_CONTRACT_V3:
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
