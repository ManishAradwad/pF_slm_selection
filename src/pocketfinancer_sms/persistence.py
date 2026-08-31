"""Automatic-persistence safety gate, deliberately narrower than recognition."""

from __future__ import annotations

from .types import (
    AccountState,
    Analysis,
    PersistenceContext,
    PersistenceDecision,
    SelectorResult,
    TriageDecision,
    TimestampProvenance,
)


def evaluate_persistence(
    selector_result: SelectorResult,
    analysis: Analysis,
    triage: TriageDecision,
    context: PersistenceContext,
) -> PersistenceDecision:
    reasons: set[str] = set()
    transaction = selector_result.transaction
    if selector_result.decision != "posted" or transaction is None:
        reasons.add("persistence_not_posted")
    else:
        if transaction.minor_units <= 0:
            reasons.add("persistence_invalid_money")
        if transaction.currency_provenance not in context.approved_currency_provenance:
            reasons.add("persistence_currency_provenance_not_approved")
        if transaction.account_state != AccountState.PRESENT:
            reasons.add("persistence_account_not_present")
        if context.account_resolution_count != 1:
            reasons.add("persistence_account_not_uniquely_resolved")
    if context.timestamp_epoch_ms is None or context.timestamp_provenance == TimestampProvenance.UNKNOWN:
        reasons.add("persistence_timestamp_provenance_invalid")
    if triage.disposition.value != "invoke":
        reasons.add("persistence_triage_requires_review")
    if any(
        cue.kind in {"failure", "negation", "pending", "request"} for cue in analysis.cues
    ):
        reasons.add("persistence_conflicting_non_posted_evidence")
    if analysis.metadata.get("completed_event_clause_count") != 1:
        reasons.add("persistence_not_exactly_one_event")
    return PersistenceDecision(not reasons, tuple(sorted(reasons)))
