"""Append-only user feedback contract for future native-app integration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any
from uuid import UUID

from .types import EvidenceSpan


FEEDBACK_V2_CONTRACT = "pocketfinancer.user-feedback/2"


class FieldGroundingClassification(StrEnum):
    SELECTED_EXISTING_CANDIDATE = "selected_existing_candidate"
    CHANGED_INTERPRETATION_AMONG_CANDIDATES = "changed_interpretation_among_candidates"
    SUPPLIED_SOURCE_SUPPORTED_CANDIDATE_MISS = "supplied_source_supported_candidate_miss"
    SUPPLIED_MANUAL_UNGROUNDED_VALUE = "supplied_manual_ungrounded_value"


@dataclass(frozen=True, slots=True)
class UserFeedbackEvent:
    contract: str
    event_id: str
    operation_id_hash: str
    trace_hash: str
    revision: int
    action: str
    canonical_label_id: str | None
    canonical_label_revision: int | None
    created_at_epoch_ms: int
    actor_id_hash: str
    previous_event_hash: str | None = None

    @classmethod
    def create(
        cls,
        *,
        event_id: str,
        operation_id: str,
        trace_hash: str,
        revision: int,
        action: str,
        canonical_label_id: str | None,
        canonical_label_revision: int | None,
        created_at_epoch_ms: int,
        actor_id: str,
        previous_event_hash: str | None = None,
    ) -> UserFeedbackEvent:
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise ValueError("feedback revision must be positive")
        if action not in {"confirm", "correct", "reject", "request_review"}:
            raise ValueError("feedback action is unsupported")
        if not event_id or not operation_id or not actor_id or created_at_epoch_ms < 0:
            raise ValueError("feedback identity or timestamp is invalid")
        if not _is_sha256(trace_hash):
            raise ValueError("feedback trace hash is invalid")
        if revision == 1 and previous_event_hash is not None:
            raise ValueError("first feedback revision cannot reference a predecessor")
        if revision > 1 and not _is_sha256(previous_event_hash or ""):
            raise ValueError("later feedback revision requires a predecessor hash")
        has_label_id = bool(canonical_label_id)
        has_label_revision = (
            isinstance(canonical_label_revision, int)
            and not isinstance(canonical_label_revision, bool)
            and canonical_label_revision > 0
        )
        if has_label_id != has_label_revision:
            raise ValueError("feedback canonical label ID and revision must appear together")
        if action in {"confirm", "correct"} and not has_label_id:
            raise ValueError(f"{action} feedback requires a canonical label reference")
        return cls(
            contract="pocketfinancer.user-feedback/1",
            event_id=event_id,
            operation_id_hash=hashlib.sha256(operation_id.encode()).hexdigest(),
            trace_hash=trace_hash,
            revision=revision,
            action=action,
            canonical_label_id=canonical_label_id,
            canonical_label_revision=canonical_label_revision,
            created_at_epoch_ms=created_at_epoch_ms,
            actor_id_hash=hashlib.sha256(actor_id.encode()).hexdigest(),
            previous_event_hash=previous_event_hash,
        )

    @property
    def event_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class FieldCorrectionV2:
    field: str
    classification: FieldGroundingClassification
    previous_revision_id: str | None
    candidate_id: str | None
    evidence: EvidenceSpan | None
    new_value: Any

    def __post_init__(self) -> None:
        if not self.field:
            raise ValueError("feedback correction field is required")
        if self.classification in {
            FieldGroundingClassification.SELECTED_EXISTING_CANDIDATE,
            FieldGroundingClassification.CHANGED_INTERPRETATION_AMONG_CANDIDATES,
        } and not self.candidate_id:
            raise ValueError("candidate-based correction requires a candidate ID")
        if (
            self.classification
            == FieldGroundingClassification.SUPPLIED_SOURCE_SUPPORTED_CANDIDATE_MISS
            and self.evidence is None
        ):
            raise ValueError("source-supported candidate miss requires evidence")

    def to_dict(self) -> dict[str, Any]:
        evidence = None
        if self.evidence is not None:
            evidence = asdict(self.evidence)
        return {
            "field": self.field,
            "classification": self.classification.value,
            "previous_revision_id": self.previous_revision_id,
            "candidate_id": self.candidate_id,
            "evidence": evidence,
            "new_value": self.new_value,
        }


@dataclass(frozen=True, slots=True)
class UserFeedbackEventV2:
    contract: str
    action_id: str
    operation_id_hash: str
    review_case_id_hash: str
    transaction_revision_id: str | None
    expected_review_revision: int
    resulting_review_revision: int
    action: str
    actor_class: str
    actor_id_hash: str
    field_corrections: tuple[FieldCorrectionV2, ...]
    retry_configuration: str | None
    canonical_label_id: str | None
    canonical_label_revision: int | None
    created_at_epoch_ms: int
    previous_event_hash: str | None = None

    @classmethod
    def create(
        cls,
        *,
        action_id: str,
        operation_id: str,
        review_case_id: str,
        transaction_revision_id: str | None,
        expected_review_revision: int,
        resulting_review_revision: int,
        action: str,
        actor_class: str,
        actor_id: str,
        field_corrections: tuple[FieldCorrectionV2, ...] = (),
        retry_configuration: str | None = None,
        canonical_label_id: str | None = None,
        canonical_label_revision: int | None = None,
        created_at_epoch_ms: int,
        previous_event_hash: str | None = None,
    ) -> UserFeedbackEventV2:
        for value, label in ((action_id, "action"), (operation_id, "operation")):
            _require_uuid(value, label)
        if not review_case_id or not actor_id:
            raise ValueError("feedback review or actor identity is missing")
        if (
            not _is_nonnegative_int(expected_review_revision)
            or not _is_nonnegative_int(resulting_review_revision)
            or resulting_review_revision != expected_review_revision + 1
        ):
            raise ValueError("feedback review revision transition is invalid")
        actions = {
            "confirm",
            "correct",
            "reject",
            "resolve_multiple_events",
            "save_draft",
            "retry",
        }
        if action not in actions or actor_class not in {"user", "migration", "system"}:
            raise ValueError("feedback action or actor class is unsupported")
        if action == "correct" and not field_corrections:
            raise ValueError("correction feedback requires field corrections")
        if action not in {"correct", "save_draft", "resolve_multiple_events"} and field_corrections:
            raise ValueError("feedback action cannot contain field corrections")
        if action == "retry":
            if retry_configuration not in {"original", "current"}:
                raise ValueError("retry feedback requires a configuration choice")
        elif retry_configuration is not None:
            raise ValueError("non-retry feedback cannot choose retry configuration")
        has_label_id = bool(canonical_label_id)
        has_label_revision = _is_positive_int(canonical_label_revision)
        if has_label_id != has_label_revision:
            raise ValueError("feedback canonical label reference is incomplete")
        if not _is_nonnegative_int(created_at_epoch_ms):
            raise ValueError("feedback timestamp is invalid")
        if expected_review_revision == 0 and previous_event_hash is not None:
            raise ValueError("first feedback event cannot reference a predecessor")
        if expected_review_revision > 0 and not _is_sha256(previous_event_hash or ""):
            raise ValueError("later feedback event requires a predecessor hash")
        return cls(
            contract=FEEDBACK_V2_CONTRACT,
            action_id=action_id,
            operation_id_hash=hashlib.sha256(operation_id.encode()).hexdigest(),
            review_case_id_hash=hashlib.sha256(review_case_id.encode()).hexdigest(),
            transaction_revision_id=transaction_revision_id,
            expected_review_revision=expected_review_revision,
            resulting_review_revision=resulting_review_revision,
            action=action,
            actor_class=actor_class,
            actor_id_hash=hashlib.sha256(actor_id.encode()).hexdigest(),
            field_corrections=field_corrections,
            retry_configuration=retry_configuration,
            canonical_label_id=canonical_label_id,
            canonical_label_revision=canonical_label_revision,
            created_at_epoch_ms=created_at_epoch_ms,
            previous_event_hash=previous_event_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "field_corrections": [item.to_dict() for item in self.field_corrections],
        }

    @property
    def event_hash(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(payload.encode()).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _require_uuid(value: str, label: str) -> None:
    try:
        parsed = UUID(value)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"feedback {label} ID is invalid") from exc
    if str(parsed) != value.lower():
        raise ValueError(f"feedback {label} ID is not canonical")


def _is_nonnegative_int(value: int | None) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_positive_int(value: int | None) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0
