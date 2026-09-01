"""Synthetic canonical-label, projection, trace, and feedback tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from pocketfinancer_sms.analyzer import DeterministicSmsAnalyzer
from pocketfinancer_sms.currency import CurrencyContext
from pocketfinancer_sms.feedback import UserFeedbackEvent
from pocketfinancer_sms.labels import (
    CanonicalDecision,
    CanonicalEvent,
    CanonicalLabel,
    EventState,
    LabelValidationError,
    OperationalClass,
    PresenceState,
    ReviewStatus,
    project_selector_target,
    validate_canonical_label,
)
from pocketfinancer_sms.trace import ProcessingTrace, TraceStage
from pocketfinancer_sms.types import CandidateKind, CurrencyProvenance, Direction, EvidenceSpan


def _posted_fixture():
    source = "INR 42.50 was credited to account **7788 from FRIEND."
    analysis = DeterministicSmsAnalyzer(
        CurrencyContext("INR", ("core-en", "india"))
    ).analyze(source, operation_id="synthetic-label")
    amount = analysis.candidates_of(CandidateKind.AMOUNT)[0]
    direction = analysis.candidates_of(CandidateKind.DIRECTION)[0]
    account = next(
        item for item in analysis.candidates_of(CandidateKind.ACCOUNT) if not item.explicit_absence
    )
    counterparty = next(
        item
        for item in analysis.candidates_of(CandidateKind.COUNTERPARTY)
        if not item.explicit_absence
    )
    assert amount.evidence and direction.evidence and account.evidence and counterparty.evidence
    event = CanonicalEvent(
        amount_span=amount.evidence,
        currency="INR",
        currency_provenance=CurrencyProvenance.EXPLICIT_CODE,
        direction=Direction.CREDIT,
        direction_span=direction.evidence,
        account_state=PresenceState.PRESENT,
        account_span=account.evidence,
        counterparty_state=PresenceState.PRESENT,
        counterparty_span=counterparty.evidence,
        financial_family="bank_transfer",
        payment_rail="bank_internal",
    )
    label = CanonicalLabel(
        contract="pocketfinancer.canonical-label/1",
        label_id="synthetic-label-1",
        source_id="synthetic-source",
        revision=1,
        status=ReviewStatus.SUBMITTED,
        decision=CanonicalDecision.POSTED,
        operational_class=OperationalClass.POSTED_CANDIDATE,
        event_state=EventState.POSTED,
        financial_family="bank_transfer",
        payment_rail="bank_internal",
        events=(event,),
        uncertain=False,
        notes="",
        reviewer_id="synthetic-reviewer",
        created_at_epoch_ms=1_700_000_000_000,
    )
    return source, analysis, label


def test_valid_rich_label_projects_to_grounded_ids() -> None:
    source, analysis, label = _posted_fixture()
    validate_canonical_label(label, source)
    target = project_selector_target(label, analysis, source)

    assert target["decision"] == "posted"
    assert target["amount"].startswith("amt_")
    assert target["direction"].startswith("dir_")
    assert target["account"].startswith("acc_")
    assert target["counterparty"].startswith("cp_")


def test_invalid_or_incomplete_label_never_becomes_negative_target() -> None:
    source, analysis, label = _posted_fixture()
    incomplete = CanonicalLabel(
        **{
            **{field: getattr(label, field) for field in label.__dataclass_fields__},
            "events": (),
        }
    )

    with pytest.raises(LabelValidationError, match="label_event_count_inconsistent"):
        project_selector_target(incomplete, analysis, source)


def test_unknown_optional_state_blocks_posted_projection() -> None:
    source, analysis, label = _posted_fixture()
    original = label.events[0]
    unknown_event = CanonicalEvent(
        amount_span=original.amount_span,
        currency=original.currency,
        currency_provenance=original.currency_provenance,
        direction=original.direction,
        direction_span=original.direction_span,
        account_state=PresenceState.UNKNOWN,
        account_span=None,
        counterparty_state=original.counterparty_state,
        counterparty_span=original.counterparty_span,
        financial_family=original.financial_family,
        payment_rail=original.payment_rail,
    )
    unknown = CanonicalLabel(
        **{
            **{field: getattr(label, field) for field in label.__dataclass_fields__},
            "events": (unknown_event,),
        }
    )

    with pytest.raises(LabelValidationError, match="projection_account_unknown"):
        project_selector_target(unknown, analysis, source)


def test_amount_evidence_must_contain_one_exact_money_value() -> None:
    source, _analysis_value, label = _posted_fixture()
    start = source.index("credited")
    invalid_event = replace(
        label.events[0],
        amount_span=EvidenceSpan.from_source(source, start, start + len("credited")),
    )

    with pytest.raises(LabelValidationError, match="label_amount_evidence_not_exact_money"):
        validate_canonical_label(replace(label, events=(invalid_event,)), source)


def test_invalid_or_outgoing_truth_is_valid_but_never_becomes_selector_negative() -> None:
    source, analysis, label = _posted_fixture()
    invalid_outgoing = replace(
        label,
        decision=CanonicalDecision.NON_FINANCIAL,
        operational_class=OperationalClass.INVALID_OUTGOING,
        event_state=EventState.NO_EVENT,
        financial_family=None,
        payment_rail=None,
        events=(),
    )

    validate_canonical_label(invalid_outgoing, source)
    with pytest.raises(LabelValidationError, match="projection_invalid_outgoing_excluded"):
        project_selector_target(invalid_outgoing, analysis, source)


def test_processing_trace_is_hash_bound_and_contains_truthful_stages() -> None:
    trace = ProcessingTrace.create(
        "synthetic-operation",
        "a" * 64,
        (
            TraceStage(0, "analysis", "completed", ("amount_candidate_present",)),
            TraceStage(1, "selector_decode", "completed", detail={"raw_output": "{synthetic}"}),
        ),
    )

    assert len(trace.trace_hash) == 64
    changed = ProcessingTrace.create(
        "synthetic-operation",
        "a" * 64,
        (TraceStage(0, "analysis", "completed", ("different",)),),
    )
    assert trace.trace_hash != changed.trace_hash


def test_feedback_is_append_only_provenance_and_correction_requires_label() -> None:
    event = UserFeedbackEvent.create(
        event_id="synthetic-feedback",
        operation_id="synthetic-operation",
        trace_hash="b" * 64,
        revision=1,
        action="correct",
        canonical_label_id="synthetic-label-1",
        canonical_label_revision=1,
        created_at_epoch_ms=1_700_000_000_000,
        actor_id="synthetic-user",
    )
    assert event.action == "correct"
    assert len(event.operation_id_hash) == 64

    with pytest.raises(ValueError, match="correct feedback requires"):
        UserFeedbackEvent.create(
            event_id="synthetic-feedback-2",
            operation_id="synthetic-operation",
            trace_hash="b" * 64,
            revision=2,
            action="correct",
            canonical_label_id=None,
            canonical_label_revision=None,
            created_at_epoch_ms=1_700_000_000_001,
            actor_id="synthetic-user",
            previous_event_hash="c" * 64,
        )
    assert len(event.event_hash) == 64
