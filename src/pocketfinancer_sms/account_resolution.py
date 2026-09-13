"""Deterministic local account resolution and duplicate safety."""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass
from enum import StrEnum

from .extractor import normalize_account_reference


ACCOUNT_RESOLUTION_PROFILE = "pocketfinancer.account-resolution-profile/1"


class ResolutionStatus(StrEnum):
    MISSING = "missing"
    UNRESOLVED = "unresolved"
    AMBIGUOUS = "ambiguous"
    UNIQUELY_RESOLVED = "uniquely_resolved"


class DuplicateStatus(StrEnum):
    CLEAR = "clear"
    POSSIBLE_DUPLICATE = "possible_duplicate"
    ALREADY_PERSISTED = "already_persisted"


@dataclass(frozen=True, slots=True)
class AccountCatalogEntry:
    account_id: str
    account_type: str
    aliases: tuple[str, ...]
    owned: bool = True

    def __post_init__(self) -> None:
        if not self.account_id or self.account_type not in {"bank_account", "card", "vpa"}:
            raise ValueError("account catalog entry is invalid")
        if not self.aliases or not all(isinstance(alias, str) and alias for alias in self.aliases):
            raise ValueError("account catalog aliases are invalid")
        if not isinstance(self.owned, bool):
            raise ValueError("account ownership flag is invalid")


@dataclass(frozen=True, slots=True)
class ExtractorAccountResolution:
    status: ResolutionStatus
    match_count: int
    account_id: str | None = None
    normalized_reference: str | None = None
    matched_alias_hash: str | None = None
    provenance: str = ACCOUNT_RESOLUTION_PROFILE

    def __post_init__(self) -> None:
        if isinstance(self.match_count, bool) or self.match_count < 0:
            raise ValueError("account resolution count is invalid")
        if self.status in {ResolutionStatus.MISSING, ResolutionStatus.UNRESOLVED}:
            if self.match_count != 0 or self.account_id is not None:
                raise ValueError("empty account resolution is inconsistent")
        elif self.status == ResolutionStatus.AMBIGUOUS:
            if self.match_count < 2 or self.account_id is not None:
                raise ValueError("ambiguous account resolution is inconsistent")
        elif (
            self.match_count != 1
            or not self.account_id
            or not self.normalized_reference
            or not _is_sha256(self.matched_alias_hash)
        ):
            raise ValueError("unique account resolution is incomplete")


@dataclass(frozen=True, slots=True)
class DuplicateAssessment:
    status: DuplicateStatus
    idempotency_key: str
    source_event_key: str
    transaction_fingerprint: str

    def __post_init__(self) -> None:
        if not self.idempotency_key or not self.source_event_key:
            raise ValueError("duplicate assessment keys are required")
        if not _is_sha256(self.transaction_fingerprint):
            raise ValueError("duplicate transaction fingerprint is invalid")


def resolve_account(
    reference: str | None,
    catalog: tuple[AccountCatalogEntry, ...] | list[AccountCatalogEntry],
) -> ExtractorAccountResolution:
    """Resolve exactly one owned catalog match; never choose a default."""

    if reference is None:
        return ExtractorAccountResolution(ResolutionStatus.MISSING, 0)
    normalized = normalize_account_alias(reference)
    if not normalized:
        return ExtractorAccountResolution(
            ResolutionStatus.UNRESOLVED, 0, normalized_reference=normalized or None
        )
    matches: list[tuple[AccountCatalogEntry, str]] = []
    for entry in catalog:
        if not entry.owned:
            continue
        for alias in entry.aliases:
            normalized_alias = normalize_account_alias(alias)
            if normalized_alias == normalized:
                matches.append((entry, normalized_alias))
                break
    if not matches:
        return ExtractorAccountResolution(
            ResolutionStatus.UNRESOLVED, 0, normalized_reference=normalized
        )
    if len(matches) > 1:
        return ExtractorAccountResolution(
            ResolutionStatus.AMBIGUOUS,
            len(matches),
            normalized_reference=normalized,
        )
    entry, alias = matches[0]
    return ExtractorAccountResolution(
        ResolutionStatus.UNIQUELY_RESOLVED,
        1,
        account_id=entry.account_id,
        normalized_reference=normalized,
        matched_alias_hash=hashlib.sha256(alias.encode("utf-8")).hexdigest(),
    )


def normalize_account_alias(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    reference = normalize_account_reference(normalized)
    if not reference:
        return ""
    return f"vpa:{reference}" if "@" in reference else f"suffix:{reference}"


def transaction_fingerprint(
    *,
    minor_units: int,
    currency: str,
    direction: str,
    account_id: str | None,
    received_at_epoch_ms: int,
) -> str:
    value = (
        f"{minor_units}\0{currency}\0{direction}\0{account_id or ''}\0"
        f"{received_at_epoch_ms}"
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def assess_duplicate(
    *,
    idempotency_key: str,
    source_event_key: str,
    transaction_fingerprint_value: str,
    persisted_idempotency_keys: frozenset[str] = frozenset(),
    persisted_source_event_keys: frozenset[str] = frozenset(),
    known_transaction_fingerprints: frozenset[str] = frozenset(),
) -> DuplicateAssessment:
    if (
        idempotency_key in persisted_idempotency_keys
        or source_event_key in persisted_source_event_keys
    ):
        status = DuplicateStatus.ALREADY_PERSISTED
    elif transaction_fingerprint_value in known_transaction_fingerprints:
        status = DuplicateStatus.POSSIBLE_DUPLICATE
    else:
        status = DuplicateStatus.CLEAR
    return DuplicateAssessment(
        status,
        idempotency_key,
        source_event_key,
        transaction_fingerprint_value,
    )


def _is_sha256(value: str | None) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )
