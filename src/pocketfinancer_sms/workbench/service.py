"""Workbench workflows, blind-review policy, validation, and target preview."""

from __future__ import annotations

import time
from typing import Any

from ..labels import (
    CanonicalDecision,
    CanonicalEvent,
    CanonicalLabel,
    EventState,
    FINANCIAL_FAMILIES,
    LabelValidationError,
    OperationalClass,
    PAYMENT_RAILS,
    PresenceState,
    ReviewStatus,
    canonical_label_to_dict,
    project_selector_target,
    validate_canonical_label,
)
from ..provenance import object_sha256
from ..types import Analysis, CurrencyProvenance, Direction, EvidenceSpan
from .store import WorkbenchStore


PROTECTED_POOLS = frozenset({"protected_test", "later_time_holdout"})


class WorkbenchValidationError(ValueError):
    """Actionable aggregate-safe validation error."""


class WorkbenchService:
    def __init__(self, store: WorkbenchStore) -> None:
        self.store = store

    def list_rows(
        self,
        *,
        reviewer_id: str,
        filters: dict[str, str | None],
        search: str | None = None,
        sort: str = "timestamp",
        descending: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        if not reviewer_id:
            raise WorkbenchValidationError("reviewer_id is required")
        pool = filters.get("pool")
        hidden_filters = {
            "operational_class",
            "event_state",
            "financial_family",
            "payment_rail",
            "disposition",
            "selector_action",
            "review_state",
            "normalized_template_group",
            "sender_family_group",
            "sender_template_group",
        }
        has_hidden_filter = any(filters.get(name) for name in hidden_filters)
        if pool in PROTECTED_POOLS and has_hidden_filter:
            raise WorkbenchValidationError(
                "suggestion, group, and prior-review filters are unavailable during protected blind review"
            )
        if pool is None and has_hidden_filter:
            filters = {**filters, "exclude_protected": "true"}
        result = self.store.list_rows(
            filters=filters,
            search=search,
            sort=sort,
            descending=descending,
            limit=limit,
            offset=offset,
        )
        for row in result["rows"]:
            if row["pool"] in PROTECTED_POOLS and not self._may_reveal(
                row["source_id"], reviewer_id
            ):
                for name in hidden_filters:
                    row[name] = None
                row["blind_locked"] = True
            else:
                row["blind_locked"] = False
        return result

    def view_row(self, source_id: str, reviewer_id: str) -> dict[str, Any]:
        record = self.store.get_record(source_id)
        if record is None:
            raise WorkbenchValidationError("source row does not exist")
        latest = self.store.latest_annotation(source_id, reviewer_id)
        may_reveal = record["pool"] not in PROTECTED_POOLS or self._may_reveal(
            source_id, reviewer_id
        )
        result = {
            "source_id": source_id,
            "source": record["source"],
            "source_metadata": record["source_metadata"],
            "pool": record["pool"],
            "review_state": (
                (latest["status"] if latest is not None else "unreviewed")
                if not may_reveal
                else record["review_state"]
            ),
            "latest_annotation": latest,
            "blind_locked": not may_reveal,
            "can_reveal": (
                record["pool"] in PROTECTED_POOLS
                and self.store.has_initial_submission(source_id, reviewer_id)
                and not self.store.is_revealed(source_id, reviewer_id)
            ),
        }
        if may_reveal:
            result.update(
                {
                    "analysis": record["analysis"],
                    "weak_facets": record["weak_facets"],
                    "grouping": record["grouping"],
                    "processing_trace": record.get("processing_trace"),
                    "latest_weak_correction": self.store.latest_weak_correction(source_id),
                    "candidate_coverage": _candidate_coverage(record["analysis"]),
                }
            )
        return result

    def save_draft(
        self,
        *,
        source_id: str,
        reviewer_id: str,
        expected_revision: int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_record_and_reviewer(source_id, reviewer_id)
        if not isinstance(payload, dict):
            raise WorkbenchValidationError("draft payload must be an object")
        return self.store.append_annotation_revision(
            source_id=source_id,
            reviewer_id=reviewer_id,
            expected_revision=expected_revision,
            status="draft",
            payload=payload,
            canonical_label=None,
            created_at_epoch_ms=_now_ms(),
        )

    def submit(
        self,
        *,
        source_id: str,
        reviewer_id: str,
        expected_revision: int,
        payload: dict[str, Any],
        adjudicated: bool = False,
    ) -> dict[str, Any]:
        record = self._require_record_and_reviewer(source_id, reviewer_id)
        revision = expected_revision + 1
        status = ReviewStatus.ADJUDICATED if adjudicated else ReviewStatus.SUBMITTED
        stored_payload = payload
        if adjudicated:
            disagreement = self.disagreements(source_id, reviewer_id)
            if not disagreement["has_disagreement"]:
                raise WorkbenchValidationError(
                    "adjudication requires at least two disagreeing submitted labels"
                )
            stored_payload = {
                **payload,
                "adjudication_of": [
                    item["revision_hash"] for item in disagreement["annotations"]
                ],
            }
        label = _build_label(
            source_id=source_id,
            reviewer_id=reviewer_id,
            revision=revision,
            status=status,
            payload=payload,
            source=record["source"]["body"],
        )
        canonical = canonical_label_to_dict(label)
        return self.store.append_annotation_revision(
            source_id=source_id,
            reviewer_id=reviewer_id,
            expected_revision=expected_revision,
            status=status.value,
            payload=stored_payload,
            canonical_label=canonical,
            created_at_epoch_ms=label.created_at_epoch_ms,
        )

    def reveal(self, source_id: str, reviewer_id: str) -> dict[str, Any]:
        self._require_record_and_reviewer(source_id, reviewer_id)
        self.store.reveal(source_id, reviewer_id, _now_ms())
        return self.view_row(source_id, reviewer_id)

    def correct_weak_facets(
        self,
        *,
        source_id: str,
        reviewer_id: str,
        expected_revision: int,
        facets: dict[str, Any],
    ) -> dict[str, Any]:
        record = self._require_record_and_reviewer(source_id, reviewer_id)
        if record["pool"] in PROTECTED_POOLS and not self._may_reveal(
            source_id, reviewer_id
        ):
            raise WorkbenchValidationError(
                "weak segregation cannot be corrected before protected review is revealed"
            )
        _validate_weak_correction(facets)
        return self.store.append_weak_correction(
            source_id=source_id,
            reviewer_id=reviewer_id,
            expected_revision=expected_revision,
            facets=facets,
            created_at_epoch_ms=_now_ms(),
        )

    def target_preview(self, source_id: str, reviewer_id: str) -> dict[str, Any]:
        record = self._require_record_and_reviewer(source_id, reviewer_id)
        if record["pool"] in PROTECTED_POOLS and not self._may_reveal(
            source_id, reviewer_id
        ):
            raise WorkbenchValidationError(
                "selector target preview remains hidden until protected review is revealed"
            )
        latest = self.store.latest_annotation(source_id, reviewer_id)
        if latest is None or latest["canonical_label"] is None:
            return {
                "convertible": False,
                "reason_code": "projection_submitted_canonical_label_missing",
            }
        label = _label_from_dict(latest["canonical_label"], record["source"]["body"])
        try:
            analysis = Analysis.from_dict(
                record["analysis"], source=record["source"]["body"]
            )
        except ValueError as exc:
            raise WorkbenchValidationError("stored analysis cannot be validated") from exc
        try:
            target = project_selector_target(label, analysis, record["source"]["body"])
        except LabelValidationError as exc:
            return {"convertible": False, "reason_code": exc.reason_code}
        return {"convertible": True, "target": target}

    def disagreements(self, source_id: str, reviewer_id: str) -> dict[str, Any]:
        record = self._require_record_and_reviewer(source_id, reviewer_id)
        if record["pool"] in PROTECTED_POOLS and not self._may_reveal(
            source_id, reviewer_id
        ):
            raise WorkbenchValidationError(
                "disagreement details remain hidden until protected review is revealed"
            )
        annotations = self.store.submitted_annotations(source_id)
        decisions = {
            item["canonical_label"]["decision"]
            for item in annotations
            if item["canonical_label"] is not None
        }
        return {
            "source_id": source_id,
            "has_disagreement": len(decisions) > 1,
            "review_count": len(annotations),
            "annotations": annotations,
        }

    def _may_reveal(self, source_id: str, reviewer_id: str) -> bool:
        return self.store.is_revealed(source_id, reviewer_id)

    def _require_record_and_reviewer(
        self, source_id: str, reviewer_id: str
    ) -> dict[str, Any]:
        if not reviewer_id:
            raise WorkbenchValidationError("reviewer_id is required")
        record = self.store.get_record(source_id)
        if record is None:
            raise WorkbenchValidationError("source row does not exist")
        return record


def _build_label(
    *,
    source_id: str,
    reviewer_id: str,
    revision: int,
    status: ReviewStatus,
    payload: dict[str, Any],
    source: str,
) -> CanonicalLabel:
    try:
        decision = CanonicalDecision(payload["decision"])
        operational = OperationalClass(payload["operational_class"])
        event_state = EventState(payload["event_state"])
        events = tuple(_build_event(item, source) for item in payload.get("events", []))
        uncertain = payload["uncertain"]
        notes = payload.get("notes", "")
    except WorkbenchValidationError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkbenchValidationError("annotation payload has invalid or missing fields") from exc
    if not isinstance(uncertain, bool) or not isinstance(notes, str):
        raise WorkbenchValidationError("annotation uncertainty or notes are invalid")
    family = payload.get("financial_family")
    rail = payload.get("payment_rail")
    created = _now_ms()
    label_basis = {
        "source_id": source_id,
        "reviewer_id": reviewer_id,
        "revision": revision,
        "payload": payload,
    }
    label = CanonicalLabel(
        contract="pocketfinancer.canonical-label/1",
        label_id="lbl_" + object_sha256(label_basis)[:24],
        source_id=source_id,
        revision=revision,
        status=status,
        decision=decision,
        operational_class=operational,
        event_state=event_state,
        financial_family=family,
        payment_rail=rail,
        events=events,
        uncertain=uncertain,
        notes=notes,
        reviewer_id=reviewer_id,
        created_at_epoch_ms=created,
        supersedes_revision=revision - 1 if revision > 1 else None,
    )
    try:
        validate_canonical_label(label, source)
    except LabelValidationError as exc:
        raise WorkbenchValidationError(exc.reason_code) from exc
    return label


def _build_event(value: dict[str, Any], source: str) -> CanonicalEvent:
    if not isinstance(value, dict):
        raise WorkbenchValidationError("event must be an object")
    required = {
        "amount_span",
        "currency",
        "currency_provenance",
        "direction",
        "direction_span",
        "account_state",
        "counterparty_state",
    }
    missing = sorted(required - set(value))
    if missing:
        raise WorkbenchValidationError(f"event is missing required field: {missing[0]}")
    try:
        account_state = PresenceState(value["account_state"])
        counterparty_state = PresenceState(value["counterparty_state"])
        return CanonicalEvent(
            amount_span=_span(value["amount_span"], source),
            currency=str(value["currency"]),
            currency_provenance=CurrencyProvenance(value["currency_provenance"]),
            direction=Direction(value["direction"]),
            direction_span=_span(value["direction_span"], source),
            account_state=account_state,
            account_span=(
                _span(value["account_span"], source) if value.get("account_span") is not None else None
            ),
            counterparty_state=counterparty_state,
            counterparty_span=(
                _span(value["counterparty_span"], source)
                if value.get("counterparty_span") is not None
                else None
            ),
            financial_family=value.get("financial_family"),
            payment_rail=value.get("payment_rail"),
        )
    except WorkbenchValidationError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkbenchValidationError(
            "event currency source, direction, or presence state is invalid"
        ) from exc


def _span(value: dict[str, Any], source: str) -> EvidenceSpan:
    if not isinstance(value, dict):
        raise WorkbenchValidationError("evidence span must be an object")
    try:
        start = value["start_char"]
        end = value["end_char"]
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
        ):
            raise TypeError
        return EvidenceSpan.from_source(source, start, end)
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkbenchValidationError("evidence span is outside the source message") from exc


def _label_from_dict(value: dict[str, Any], source: str) -> CanonicalLabel:
    payload = {
        "decision": value["decision"],
        "operational_class": value["operational_class"],
        "event_state": value["event_state"],
        "financial_family": value.get("financial_family"),
        "payment_rail": value.get("payment_rail"),
        "events": value.get("events", []),
        "uncertain": value["uncertain"],
        "notes": value.get("notes", ""),
    }
    label = _build_label(
        source_id=value["source_id"],
        reviewer_id=value["reviewer_id"],
        revision=int(value["revision"]),
        status=ReviewStatus(value["status"]),
        payload=payload,
        source=source,
    )
    return CanonicalLabel(
        **{
            **{field: getattr(label, field) for field in label.__dataclass_fields__},
            "label_id": value["label_id"],
            "created_at_epoch_ms": int(value["created_at_epoch_ms"]),
            "supersedes_revision": value.get("supersedes_revision"),
        }
    )


def _validate_weak_correction(facets: dict[str, Any]) -> None:
    try:
        OperationalClass(facets["operational_class"])
        EventState(facets["event_state"])
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkbenchValidationError("weak correction taxonomy axes are invalid") from exc
    family = facets.get("financial_family")
    rail = facets.get("payment_rail")
    if family is not None and family not in FINANCIAL_FAMILIES:
        raise WorkbenchValidationError("weak correction financial family is invalid")
    if rail is not None and rail not in PAYMENT_RAILS:
        raise WorkbenchValidationError("weak correction payment rail is invalid")
    if not isinstance(facets.get("reason"), str) or not facets["reason"].strip():
        raise WorkbenchValidationError("weak correction requires a reason")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _candidate_coverage(analysis: dict[str, Any]) -> dict[str, Any]:
    counts = {kind: 0 for kind in ("amount", "direction", "account", "counterparty")}
    amount_clauses: set[str] = set()
    direction_clauses: set[str] = set()
    for candidate in analysis.get("candidates", []):
        kind = candidate.get("kind")
        if kind in counts and not candidate.get("explicit_absence"):
            counts[kind] += 1
        clause = candidate.get("clause_id")
        if kind == "amount" and clause:
            amount_clauses.add(clause)
        elif kind == "direction" and clause:
            direction_clauses.add(clause)
    return {
        "field_candidate_counts": counts,
        "complete_core_clause_count": len(amount_clauses & direction_clauses),
    }
