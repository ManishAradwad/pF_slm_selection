"""Workbench workflows, blind-review policy, validation, and target preview."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Any

from ..extractor import SourceSpan
from ..labels import (
    CanonicalDecision,
    CanonicalEvent,
    CanonicalLabel,
    CanonicalLabelV2,
    ExtractorCanonicalEvent,
    EventState,
    FINANCIAL_FAMILIES,
    LabelValidationError,
    OperationalClass,
    PAYMENT_RAILS,
    PresenceState,
    ReviewStatus,
    canonical_label_to_dict,
    canonical_label_v2_to_dict,
    project_extractor_target,
    project_selector_target,
    validate_canonical_label,
    validate_canonical_label_v2,
)
from ..provenance import object_sha256
from ..types import Analysis, CurrencyProvenance, Direction, EvidenceSpan
from .native_import import NativeTraceImporter
from .store import WorkbenchStore


PROTECTED_POOLS = frozenset({"protected_test", "later_time_holdout"})
V2_CONTRACT = "pocketfinancer.canonical-label/2"
RESUME_FILTERS = frozenset({
    "pool", "operational_class", "event_state", "financial_family",
    "payment_rail", "normalized_template_group", "sender_family_group",
    "sender_template_group", "time_group", "time_from", "time_to",
    "disposition", "selector_action", "review_state", "reviewer_state",
    "candidate_coverage", "disagreement", "imported_feedback",
})
RESUME_SORTS = frozenset({
    "timestamp", "source_id", "pool", "review_state", "operational_class",
    "payment_rail", "time_group", "sender_template_group",
})


class WorkbenchValidationError(ValueError):
    """Actionable aggregate-safe validation error."""


class WorkbenchService:
    def __init__(
        self,
        store: WorkbenchStore,
        *,
        native_trace_importer: NativeTraceImporter | None = None,
    ) -> None:
        self.store = store
        self.native_trace_importer = native_trace_importer

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
            "candidate_coverage",
            "disagreement",
            "imported_feedback",
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
        if filters.get("imported_feedback") and self.native_trace_importer is None:
            raise WorkbenchValidationError("imported feedback view is unavailable")
        result = self.store.list_rows(
            filters=filters,
            search=search,
            sort=sort,
            descending=descending,
            limit=limit,
            offset=offset,
            reviewer_id=reviewer_id,
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

    def save_resume(
        self,
        *,
        reviewer_id: str,
        source_id: str,
        offset: int,
        filters: dict[str, str | None],
        search: str = "",
        sort: str = "timestamp",
        descending: bool = False,
    ) -> dict[str, Any]:
        if not reviewer_id or len(reviewer_id) > 128:
            raise WorkbenchValidationError("reviewer_id is invalid")
        if self.store.get_record(source_id) is None:
            raise WorkbenchValidationError("source row does not exist")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise WorkbenchValidationError("resume offset is invalid")
        if not isinstance(filters, dict) or any(
            key not in RESUME_FILTERS or (
                value is not None and (not isinstance(value, str) or len(value) > 256)
            )
            for key, value in filters.items()
        ):
            raise WorkbenchValidationError("resume filters are invalid")
        if not isinstance(search, str) or len(search) > 256:
            raise WorkbenchValidationError("resume search is invalid")
        if sort not in RESUME_SORTS or not isinstance(descending, bool):
            raise WorkbenchValidationError("resume sort is invalid")
        self.list_rows(
            reviewer_id=reviewer_id, filters=filters, search=search,
            sort=sort, descending=descending, limit=1, offset=0,
        )
        state = {
            "source_id": source_id, "offset": offset, "filters": filters,
            "search": search, "sort": sort, "descending": descending,
        }
        self.store.save_resume(reviewer_id, state)
        return state

    def load_resume(self, reviewer_id: str) -> dict[str, Any] | None:
        if not reviewer_id or len(reviewer_id) > 128:
            raise WorkbenchValidationError("reviewer_id is invalid")
        return self.store.load_resume(reviewer_id)

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
            "annotation_history": self.store.annotation_history(source_id, reviewer_id),
            "blind_locked": not may_reveal,
            "can_reveal": (
                record["pool"] in PROTECTED_POOLS
                and self.store.has_initial_submission(source_id, reviewer_id)
                and not self.store.is_revealed(source_id, reviewer_id)
            ),
        }
        if may_reveal:
            native_traces = (
                self.native_trace_importer.records_for_source(source_id)
                if self.native_trace_importer is not None
                else []
            )
            result.update(
                {
                    "analysis": record["analysis"],
                    "weak_facets": record["weak_facets"],
                    "grouping": record["grouping"],
                    "processing_trace": record.get("processing_trace"),
                    "latest_weak_correction": self.store.latest_weak_correction(source_id),
                    "candidate_coverage": _candidate_coverage(record["analysis"]),
                    "native_traces": native_traces,
                    "native_suggestions": _native_evidence_suggestions(
                        native_traces, record["source"]["body"]
                    ),
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
        _validate_payload_contract(payload)
        _prevent_contract_downgrade(
            self.store.latest_annotation(source_id, reviewer_id), payload
        )
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
        try:
            return self._submit_impl(
                source_id=source_id,
                reviewer_id=reviewer_id,
                expected_revision=expected_revision,
                payload=payload,
                adjudicated=adjudicated,
            )
        except WorkbenchValidationError:
            self.store.record_validation_failure()
            raise

    def _submit_impl(
        self,
        *,
        source_id: str,
        reviewer_id: str,
        expected_revision: int,
        payload: dict[str, Any],
        adjudicated: bool = False,
    ) -> dict[str, Any]:
        record = self._require_record_and_reviewer(source_id, reviewer_id)
        if not isinstance(payload, dict):
            raise WorkbenchValidationError("annotation payload must be an object")
        contract = _validate_payload_contract(payload)
        _prevent_contract_downgrade(
            self.store.latest_annotation(source_id, reviewer_id), payload
        )
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
        builder = _build_label_v2 if contract == V2_CONTRACT else _build_label
        label = builder(
            source_id=source_id,
            reviewer_id=reviewer_id,
            revision=revision,
            status=status,
            payload=payload,
            source=record["source"]["body"],
        )
        canonical = (
            canonical_label_v2_to_dict(label)
            if isinstance(label, CanonicalLabelV2)
            else canonical_label_to_dict(label)
        )
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
        canonical = latest["canonical_label"]
        source = record["source"]["body"]
        if canonical.get("contract") == V2_CONTRACT:
            try:
                label_v2 = _label_v2_from_dict(canonical, source)
                target = project_extractor_target(label_v2, source)
            except (LabelValidationError, WorkbenchValidationError) as exc:
                reason = exc.reason_code if isinstance(exc, LabelValidationError) else str(exc)
                return {"convertible": False, "reason_code": reason}
            return {
                "convertible": True,
                "target": target,
                "target_contract": "pocketfinancer.sms-extractor/1",
            }
        label = _label_from_dict(canonical, source)
        try:
            analysis = Analysis.from_dict(record["analysis"], source=source)
        except ValueError as exc:
            raise WorkbenchValidationError("stored analysis cannot be validated") from exc
        try:
            target = project_selector_target(label, analysis, source)
        except LabelValidationError as exc:
            return {"convertible": False, "reason_code": exc.reason_code}
        return {"convertible": True, "target": target, "target_contract": "historical-candidate-selector"}

    def disagreements(self, source_id: str, reviewer_id: str) -> dict[str, Any]:
        record = self._require_record_and_reviewer(source_id, reviewer_id)
        if record["pool"] in PROTECTED_POOLS and not self._may_reveal(
            source_id, reviewer_id
        ):
            raise WorkbenchValidationError(
                "disagreement details remain hidden until protected review is revealed"
            )
        annotations = self.store.submitted_annotations(source_id)
        semantic_labels = {
            object_sha256(_semantic_label(item["canonical_label"]))
            for item in annotations
            if item["canonical_label"] is not None
        }
        return {
            "source_id": source_id,
            "has_disagreement": len(semantic_labels) > 1,
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



def _validate_payload_contract(payload: dict[str, Any]) -> str:
    contract = payload.get("contract", "pocketfinancer.canonical-label/1")
    if contract not in {"pocketfinancer.canonical-label/1", V2_CONTRACT}:
        raise WorkbenchValidationError("annotation contract is unsupported")
    return contract


def _prevent_contract_downgrade(latest: dict[str, Any] | None, payload: dict[str, Any]) -> None:
    if latest is None:
        return
    previous = latest.get("canonical_label") or latest.get("payload") or {}
    if previous.get("contract") == V2_CONTRACT and _validate_payload_contract(payload) != V2_CONTRACT:
        raise WorkbenchValidationError("v2 annotation cannot be edited as legacy v1")


def _source_span(value: Any, source: str) -> SourceSpan:
    try:
        return SourceSpan.from_payload(value, source)
    except ValueError as exc:
        raise WorkbenchValidationError("annotation source span is invalid") from exc


def _build_v2_event(value: Any, source: str) -> ExtractorCanonicalEvent:
    if not isinstance(value, dict):
        raise WorkbenchValidationError("posted annotation requires one event")
    required = {
        "amount_value", "currency", "amount_span", "direction", "direction_span",
        "account_reference", "account_span", "existing_account_id",
        "counterparty", "counterparty_span",
    }
    if set(value) != required:
        raise WorkbenchValidationError("posted event fields are incomplete or unsupported")
    try:
        return ExtractorCanonicalEvent(
            amount_value=value["amount_value"],
            currency=value["currency"],
            amount_span=_source_span(value["amount_span"], source),
            direction=Direction(value["direction"]),
            direction_span=_source_span(value["direction_span"], source),
            account_reference=value["account_reference"],
            account_span=_source_span(value["account_span"], source),
            existing_account_id=value["existing_account_id"],
            counterparty=value["counterparty"],
            counterparty_span=(
                _source_span(value["counterparty_span"], source)
                if value["counterparty_span"] is not None else None
            ),
        )
    except WorkbenchValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise WorkbenchValidationError("posted event value is invalid") from exc


def _build_label_v2(
    *,
    source_id: str,
    reviewer_id: str,
    revision: int,
    status: ReviewStatus,
    payload: dict[str, Any],
    source: str,
) -> CanonicalLabelV2:
    if payload.get("contract") != V2_CONTRACT:
        raise WorkbenchValidationError("v2 annotation contract must be explicit")
    try:
        decision = payload["decision"]
        operational = OperationalClass(payload["operational_class"])
        event_state = EventState(payload["event_state"])
        uncertain = payload["uncertain"]
        notes = payload.get("notes", "")
        event_value = payload.get("event")
        event = _build_v2_event(event_value, source) if decision == "posted" else None
        if decision != "posted" and event_value is not None:
            raise WorkbenchValidationError("non-posted annotation cannot contain an event")
    except WorkbenchValidationError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkbenchValidationError("annotation payload has invalid or missing fields") from exc
    if not isinstance(uncertain, bool) or not isinstance(notes, str):
        raise WorkbenchValidationError("annotation uncertainty or notes are invalid")
    try:
        label = CanonicalLabelV2(
            label_id="lbl_" + object_sha256({
                "source_id": source_id, "reviewer_id": reviewer_id,
                "revision": revision, "payload": payload,
            })[:24],
            source_id=source_id,
            revision=revision,
            status=status,
            decision=decision,
            operational_class=operational,
            event_state=event_state,
            financial_family=payload.get("financial_family"),
            payment_rail=payload.get("payment_rail"),
            event=event,
            uncertain=uncertain,
            notes=notes,
            reviewer_id=reviewer_id,
            created_at_epoch_ms=_now_ms(),
            supersedes_revision=revision - 1 if revision > 1 else None,
        )
        validate_canonical_label_v2(label, source)
    except LabelValidationError as exc:
        raise WorkbenchValidationError(exc.reason_code) from exc
    return label


def _label_v2_from_dict(value: dict[str, Any], source: str) -> CanonicalLabelV2:
    event = value.get("event")
    label = CanonicalLabelV2(
        label_id=value["label_id"],
        source_id=value["source_id"],
        revision=value["revision"],
        status=ReviewStatus(value["status"]),
        decision=value["decision"],
        operational_class=OperationalClass(value["operational_class"]),
        event_state=EventState(value["event_state"]),
        financial_family=value.get("financial_family"),
        payment_rail=value.get("payment_rail"),
        event=_build_v2_event(event, source) if event is not None else None,
        uncertain=value["uncertain"],
        notes=value["notes"],
        reviewer_id=value["reviewer_id"],
        created_at_epoch_ms=value["created_at_epoch_ms"],
        supersedes_revision=value.get("supersedes_revision"),
    )
    validate_canonical_label_v2(label, source)
    return label


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


SEMANTIC_METADATA = frozenset({
    "label_id", "source_id", "reviewer_id", "revision", "status",
    "created_at_epoch_ms", "supersedes_revision", "notes",
})


def _semantic_label(label: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in label.items() if key not in SEMANTIC_METADATA}


def _native_evidence_suggestions(
    traces: list[dict[str, Any]], source: str
) -> list[dict[str, Any]]:
    """Show only source-grounded native corrections as advisory span suggestions."""

    suggestions: list[dict[str, Any]] = []
    for trace in traces:
        try:
            record = json.loads(trace["record_json"])
        except (KeyError, TypeError, ValueError):
            continue
        if not isinstance(record, dict) or not isinstance(record.get("native_feedback"), list):
            continue
        for feedback in record["native_feedback"]:
            if (
                not isinstance(feedback, dict)
                or feedback.get("contract") != "pocketfinancer.user-feedback/2"
                or not isinstance(feedback.get("field_corrections"), list)
            ):
                continue
            for correction in feedback["field_corrections"]:
                if not isinstance(correction, dict):
                    continue
                field = correction.get("field")
                evidence = correction.get("evidence")
                if (
                    field not in {"amount", "direction", "account", "counterparty"}
                    or correction.get("classification")
                    == "supplied_manual_ungrounded_value"
                    or not isinstance(evidence, dict)
                ):
                    continue
                if any(
                    isinstance(evidence.get(key), bool)
                    or not isinstance(evidence.get(key), int)
                    for key in ("start_char", "end_char")
                ):
                    continue
                try:
                    expected = EvidenceSpan.from_source(
                        source, evidence["start_char"], evidence["end_char"]
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                if evidence != asdict(expected) or not expected.text:
                    continue
                suggestions.append({
                    "provenance_class": "native_feedback_correction",
                    "source_platform": trace["source_platform"],
                    "field": field,
                    "classification": correction.get("classification"),
                    "evidence": asdict(expected),
                })
                if len(suggestions) >= 200:
                    return suggestions
    return suggestions


def _candidate_coverage(analysis: dict[str, Any]) -> dict[str, Any]:
    counts = {kind: 0 for kind in ("amount", "direction", "account", "counterparty")}
    amount_clauses: set[str] = set()
    direction_clauses: set[str] = set()
    for candidate in analysis.get("candidates", []):
        kind = candidate.get("kind")
        if kind in counts and not candidate.get("explicit_absence"):
            counts[kind] += 1
        clause = candidate.get("clause_id")
        if candidate.get("explicit_absence"):
            continue
        if kind == "amount" and clause:
            amount_clauses.add(clause)
        elif kind == "direction" and clause:
            direction_clauses.add(clause)
    return {
        "field_candidate_counts": counts,
        "complete_core_clause_count": len(amount_clauses & direction_clauses),
    }
