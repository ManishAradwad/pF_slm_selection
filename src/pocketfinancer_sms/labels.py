"""Orthogonal taxonomy and rich canonical human-label validation."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import StrEnum

from .currency import parse_money
from .types import Analysis, Candidate, CandidateKind, CurrencyProvenance, Direction, EvidenceSpan


class OperationalClass(StrEnum):
    POSTED_CANDIDATE = "posted_candidate"
    FINANCIAL_NON_POSTED = "financial_non_posted"
    NON_FINANCIAL = "non_financial"
    AMBIGUOUS = "ambiguous"
    INVALID_OUTGOING = "invalid_outgoing"


class EventState(StrEnum):
    POSTED = "posted"
    NOT_POSTED = "not_posted"
    NO_EVENT = "no_event"
    UNKNOWN = "unknown"


class CanonicalDecision(StrEnum):
    POSTED = "posted"
    NOT_POSTED = "not_posted"
    NON_FINANCIAL = "non_financial"
    AMBIGUOUS = "ambiguous"
    MULTIPLE_EVENT = "multiple_event"


class ReviewStatus(StrEnum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    ADJUDICATED = "adjudicated"
    SUPERSEDED = "superseded"


class PresenceState(StrEnum):
    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"


class WeakConfidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


FINANCIAL_FAMILIES = frozenset(
    {
        "bank_transfer",
        "bill_payment",
        "card_purchase",
        "cash_deposit",
        "cash_withdrawal",
        "fee_charge",
        "insurance",
        "interest",
        "investment",
        "loan",
        "merchant_payment",
        "refund",
        "salary_income",
        "upi_transfer",
        "wallet",
        "other_financial",
        "unknown",
    }
)

PAYMENT_RAILS = frozenset(
    {
        "bank_internal",
        "card",
        "cash",
        "imps",
        "nach",
        "neft",
        "other",
        "rtgs",
        "upi",
        "wallet",
        "unknown",
    }
)

_EXACT_MONEY_NUMBER = re.compile(
    r"(?:\d{1,3}(?:,\d{2})*,\d{3}|\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
)


@dataclass(frozen=True, slots=True)
class CanonicalEvent:
    amount_span: EvidenceSpan
    currency: str
    currency_provenance: CurrencyProvenance
    direction: Direction
    direction_span: EvidenceSpan
    account_state: PresenceState
    account_span: EvidenceSpan | None
    counterparty_state: PresenceState
    counterparty_span: EvidenceSpan | None
    financial_family: str | None
    payment_rail: str | None


@dataclass(frozen=True, slots=True)
class CanonicalLabel:
    contract: str
    label_id: str
    source_id: str
    revision: int
    status: ReviewStatus
    decision: CanonicalDecision
    operational_class: OperationalClass
    event_state: EventState
    financial_family: str | None
    payment_rail: str | None
    events: tuple[CanonicalEvent, ...]
    uncertain: bool
    notes: str
    reviewer_id: str
    created_at_epoch_ms: int
    supersedes_revision: int | None = None


@dataclass(frozen=True, slots=True)
class WeakFacets:
    operational_class: OperationalClass
    event_state: EventState
    financial_family: str | None
    payment_rail: str | None
    confidence: WeakConfidence
    reason_codes: tuple[str, ...]


class LabelValidationError(ValueError):
    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)

    def __str__(self) -> str:
        return self.reason_code


def validate_canonical_label(label: CanonicalLabel, source: str) -> None:
    if label.contract != "pocketfinancer.canonical-label/1":
        raise LabelValidationError("label_contract_invalid")
    if label.revision < 1 or label.created_at_epoch_ms < 0:
        raise LabelValidationError("label_provenance_invalid")
    if not label.label_id or not label.source_id or not label.reviewer_id:
        raise LabelValidationError("label_identity_missing")
    if label.revision == 1 and label.supersedes_revision is not None:
        raise LabelValidationError("label_first_revision_cannot_supersede")
    if label.revision > 1 and label.supersedes_revision != label.revision - 1:
        raise LabelValidationError("label_revision_chain_invalid")

    expected_event_counts = {
        CanonicalDecision.POSTED: 1,
        CanonicalDecision.NOT_POSTED: 0,
        CanonicalDecision.NON_FINANCIAL: 0,
        CanonicalDecision.AMBIGUOUS: 0,
    }
    if label.decision == CanonicalDecision.MULTIPLE_EVENT:
        if len(label.events) < 2:
            raise LabelValidationError("label_multiple_event_requires_two_events")
    elif len(label.events) != expected_event_counts[label.decision]:
        raise LabelValidationError("label_event_count_inconsistent")

    expected_axes = {
        CanonicalDecision.POSTED: {
            (OperationalClass.POSTED_CANDIDATE, EventState.POSTED)
        },
        CanonicalDecision.NOT_POSTED: {
            (OperationalClass.FINANCIAL_NON_POSTED, EventState.NOT_POSTED)
        },
        CanonicalDecision.NON_FINANCIAL: {
            (OperationalClass.NON_FINANCIAL, EventState.NO_EVENT),
            (OperationalClass.INVALID_OUTGOING, EventState.NO_EVENT),
        },
        CanonicalDecision.AMBIGUOUS: {
            (OperationalClass.AMBIGUOUS, EventState.UNKNOWN)
        },
        CanonicalDecision.MULTIPLE_EVENT: {
            (OperationalClass.POSTED_CANDIDATE, EventState.POSTED)
        },
    }
    if (label.operational_class, label.event_state) not in expected_axes[label.decision]:
        raise LabelValidationError("label_taxonomy_axes_inconsistent")
    _validate_facets(label.financial_family, label.payment_rail)
    for event in label.events:
        _validate_event(event, source)
    if label.decision == CanonicalDecision.POSTED:
        event = label.events[0]
        if (
            label.financial_family != event.financial_family
            or label.payment_rail != event.payment_rail
        ):
            raise LabelValidationError("label_event_facets_inconsistent")


def project_selector_target(
    label: CanonicalLabel, analysis: Analysis, source: str
) -> dict[str, str]:
    """Project valid rich truth into the compact selector contract or fail explicitly."""

    validate_canonical_label(label, source)
    if label.status not in {ReviewStatus.SUBMITTED, ReviewStatus.ADJUDICATED}:
        raise LabelValidationError("projection_label_not_committed")
    if label.operational_class == OperationalClass.INVALID_OUTGOING:
        raise LabelValidationError("projection_invalid_outgoing_excluded")
    if label.decision in {CanonicalDecision.NOT_POSTED, CanonicalDecision.NON_FINANCIAL}:
        return {"decision": "none"}
    if label.decision in {CanonicalDecision.AMBIGUOUS, CanonicalDecision.MULTIPLE_EVENT}:
        return {"decision": "abstain"}

    event = label.events[0]
    if event.account_state == PresenceState.UNKNOWN:
        raise LabelValidationError("projection_account_unknown")
    if event.counterparty_state == PresenceState.UNKNOWN:
        raise LabelValidationError("projection_counterparty_unknown")
    amount = _match_span(
        analysis, CandidateKind.AMOUNT, event.amount_span, "projection_missing_amount_candidate"
    )
    direction = _match_span(
        analysis,
        CandidateKind.DIRECTION,
        event.direction_span,
        "projection_missing_direction_candidate",
    )
    if amount.value.get("currency") != event.currency:
        raise LabelValidationError("projection_currency_mismatch")
    if amount.value.get("currency_provenance") != event.currency_provenance.value:
        raise LabelValidationError("projection_currency_provenance_mismatch")
    if direction.value.get("direction") != event.direction.value:
        raise LabelValidationError("projection_direction_mismatch")
    account = _presence_candidate(
        analysis,
        CandidateKind.ACCOUNT,
        event.account_state,
        event.account_span,
        "projection_missing_account_candidate",
    )
    counterparty = _presence_candidate(
        analysis,
        CandidateKind.COUNTERPARTY,
        event.counterparty_state,
        event.counterparty_span,
        "projection_missing_counterparty_candidate",
    )
    return {
        "decision": "posted",
        "amount": amount.candidate_id,
        "direction": direction.candidate_id,
        "account": account.candidate_id,
        "counterparty": counterparty.candidate_id,
    }


def canonical_label_to_dict(label: CanonicalLabel) -> dict:
    """Return the JSON contract form without duplicating private evidence text."""

    value = asdict(label)
    for event in value["events"]:
        for field in (
            "amount_span",
            "direction_span",
            "account_span",
            "counterparty_span",
        ):
            if event[field] is not None:
                event[field].pop("text", None)
    return value


def _validate_event(event: CanonicalEvent, source: str) -> None:
    _validate_span(event.amount_span, source, "label_amount_span_invalid")
    if len(event.currency) != 3 or event.currency != event.currency.upper():
        raise LabelValidationError("label_currency_invalid")
    number_matches = _EXACT_MONEY_NUMBER.findall(event.amount_span.text)
    if len(number_matches) != 1:
        raise LabelValidationError("label_amount_evidence_not_exact_money")
    try:
        parse_money(
            number_matches[0],
            currency=event.currency,
            provenance=event.currency_provenance,
        )
    except ValueError as exc:
        raise LabelValidationError("label_amount_evidence_not_exact_money") from exc
    _validate_span(event.direction_span, source, "label_direction_span_invalid")
    _validate_presence(event.account_state, event.account_span, source, "account")
    _validate_presence(event.counterparty_state, event.counterparty_span, source, "counterparty")
    _validate_facets(event.financial_family, event.payment_rail)


def _validate_presence(
    state: PresenceState, span: EvidenceSpan | None, source: str, field: str
) -> None:
    if state == PresenceState.PRESENT and span is None:
        raise LabelValidationError(f"label_{field}_present_without_span")
    if state != PresenceState.PRESENT and span is not None:
        raise LabelValidationError(f"label_{field}_nonpresent_with_span")
    if span is not None:
        _validate_span(span, source, f"label_{field}_span_invalid")


def _validate_span(span: EvidenceSpan, source: str, reason: str) -> None:
    if span.start_char < 0 or span.end_char <= span.start_char or span.end_char > len(source):
        raise LabelValidationError(reason)
    expected = EvidenceSpan.from_source(source, span.start_char, span.end_char)
    if span != expected:
        raise LabelValidationError(reason)


def _validate_facets(family: str | None, rail: str | None) -> None:
    if family is not None and family not in FINANCIAL_FAMILIES:
        raise LabelValidationError("label_financial_family_invalid")
    if rail is not None and rail not in PAYMENT_RAILS:
        raise LabelValidationError("label_payment_rail_invalid")


def _match_span(
    analysis: Analysis, kind: CandidateKind, span: EvidenceSpan, reason: str
) -> Candidate:
    matches = [
        candidate
        for candidate in analysis.candidates_of(kind)
        if candidate.evidence is not None
        and candidate.evidence.start_char == span.start_char
        and candidate.evidence.end_char == span.end_char
    ]
    if len(matches) != 1:
        raise LabelValidationError(reason)
    return matches[0]


def _presence_candidate(
    analysis: Analysis,
    kind: CandidateKind,
    state: PresenceState,
    span: EvidenceSpan | None,
    reason: str,
) -> Candidate:
    if state == PresenceState.ABSENT:
        matches = [candidate for candidate in analysis.candidates_of(kind) if candidate.explicit_absence]
        if len(matches) != 1:
            raise LabelValidationError(reason)
        return matches[0]
    if state == PresenceState.PRESENT and span is not None:
        return _match_span(analysis, kind, span, reason)
    raise LabelValidationError(reason)
