"""Automatic-persistence safety gate, deliberately narrower than recognition."""

from __future__ import annotations

from typing import Any

from .currency import ISO_MINOR_UNITS, MAX_SIGNED_64
from .types import (
    AccountState,
    Analysis,
    GateCheck,
    GateResult,
    PersistenceContext,
    PersistenceContextV2,
    PersistenceDecision,
    PersistenceDecisionV2,
    SelectorResult,
    TriageDecision,
    TimestampProvenance,
)


PROCESSING_RESULT_CONTRACT_V2 = "pocketfinancer.processing-result/2"


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
    if (
        analysis.metadata.get("completed_event_clause_count") != 1
        or analysis.metadata.get("completed_event_candidate_count") != 1
    ):
        reasons.add("persistence_not_exactly_one_event")
    return PersistenceDecision(not reasons, tuple(sorted(reasons)))


def processing_result_payload(
    selector_result: SelectorResult,
    persistence: PersistenceDecision,
) -> dict[str, Any]:
    """Serialize recognition, reconstructed semantics, and save safety separately."""

    transaction = selector_result.transaction
    if selector_result.decision == "posted":
        if transaction is None:
            raise ValueError("posted recognition is missing its reconstructed transaction")
        semantic_result: dict[str, Any] | None = {
            "amount_candidate_id": transaction.amount_candidate_id,
            "direction_candidate_id": transaction.direction_candidate_id,
            "account_candidate_id": transaction.account_candidate_id,
            "counterparty_candidate_id": transaction.counterparty_candidate_id,
            "minor_units": transaction.minor_units,
            "currency": transaction.currency,
            "currency_provenance": transaction.currency_provenance.value,
            "direction": transaction.direction.value,
            "account_state": transaction.account_state.value,
            "counterparty_state": transaction.counterparty_state.value,
        }
    elif selector_result.decision in {"none", "abstain"} and transaction is None:
        semantic_result = None
    else:
        raise ValueError("recognition decision and reconstructed transaction are inconsistent")
    return {
        "recognition_decision": selector_result.decision,
        "semantic_result": semantic_result,
        "automatic_persistence": {
            "safe": persistence.safe_to_persist,
            "reason_codes": list(persistence.reason_codes),
        },
    }


def evaluate_persistence_v2(
    selector_result: SelectorResult,
    analysis: Analysis,
    triage: TriageDecision,
    context: PersistenceContextV2,
) -> PersistenceDecisionV2:
    """Evaluate every native automatic-persistence gate without conflating truth and policy."""

    transaction = selector_result.transaction
    checks = (
        _check(
            "known_analysis_contract",
            analysis.contract == "pocketfinancer.sms-analysis/2",
            "persistence_unknown_analysis_contract",
        ),
        _check(
            "configuration_hash",
            context.configuration_hash_matches,
            "persistence_configuration_hash_mismatch",
        ),
        _check(
            "claim_ownership",
            context.claim_ownership_current,
            "persistence_claim_ownership_invalid",
        ),
        _check(
            "selector_mode",
            context.selector_mode_valid,
            "persistence_selector_mode_invalid",
        ),
        _check(
            "posted_reconstruction",
            selector_result.decision == "posted" and transaction is not None,
            "persistence_not_posted",
        ),
        _check(
            "exactly_one_event",
            analysis.metadata.get("completed_event_clause_count") == 1
            and analysis.metadata.get("completed_event_candidate_count") == 1,
            "persistence_not_exactly_one_event",
        ),
        _check(
            "exact_money",
            transaction is not None and 0 < transaction.minor_units <= MAX_SIGNED_64,
            "persistence_invalid_money",
        ),
        _check(
            "currency_provenance",
            transaction is not None
            and transaction.currency in ISO_MINOR_UNITS
            and transaction.currency_provenance in context.approved_currency_provenance,
            "persistence_currency_provenance_not_approved",
        ),
        _check(
            "timestamp_provenance",
            context.timestamp_epoch_ms is not None
            and context.timestamp_provenance in context.approved_timestamp_provenance,
            "persistence_timestamp_provenance_invalid",
        ),
        _check(
            "account_present",
            transaction is not None and transaction.account_state == AccountState.PRESENT,
            "persistence_account_not_present",
        ),
        _check(
            "account_resolution",
            context.account_resolution.status.value == "uniquely_resolved"
            and context.account_resolution.match_count == 1
            and context.account_resolution.account_id_hash is not None,
            "persistence_account_not_uniquely_resolved",
        ),
        _check(
            "financial_family",
            context.financial_family is not None
            and context.financial_family in context.supported_automatic_families,
            "persistence_financial_family_not_supported",
        ),
        _check(
            "triage_disposition",
            triage.disposition.value == "invoke",
            "persistence_triage_requires_review",
        ),
        _check(
            "selector_action",
            triage.selector_action.value == "run_normal",
            "persistence_assistive_selection_not_eligible",
        ),
        _check(
            "non_posted_conflict",
            not any(
                cue.kind
                in {
                    "failure",
                    "negation",
                    "pending",
                    "request",
                    "expectation",
                    "authorization_hold",
                }
                for cue in analysis.cues
            ),
            "persistence_conflicting_non_posted_evidence",
        ),
        _check(
            "rollout_mode",
            context.rollout_mode == "automatic",
            "persistence_blocked_by_rollout_mode",
        ),
    )
    failed = tuple(check for check in checks if not check.passed)
    primary_reason = failed[0].reason_code if failed else "persistence_all_gates_passed"
    assert primary_reason is not None

    integrity_checks = {"known_analysis_contract", "configuration_hash", "claim_ownership"}
    if any(check.check in integrity_checks for check in failed):
        result = GateResult.INVALID_OPERATION
    elif selector_result.decision == "none":
        result = GateResult.NOT_POSTED
    elif analysis.metadata.get("completed_event_candidate_count", 0) > 1:
        result = GateResult.MULTIPLE_EVENTS
    elif any(check.check != "rollout_mode" for check in failed):
        result = GateResult.REVIEW_REQUIRED
    elif failed:
        result = GateResult.BLOCKED_BY_MODE
    else:
        result = GateResult.ELIGIBLE
    return PersistenceDecisionV2(result, primary_reason, checks)


def processing_result_payload_v2(
    selector_result: SelectorResult,
    persistence: PersistenceDecisionV2,
    context: PersistenceContextV2,
) -> dict[str, Any]:
    transaction = selector_result.transaction
    semantic_result = None
    if selector_result.decision == "posted":
        if transaction is None:
            raise ValueError("posted recognition is missing its reconstructed transaction")
        semantic_result = {
            "analysis_id": transaction.analysis_id,
            "candidate_ids": {
                "amount": transaction.amount_candidate_id,
                "direction": transaction.direction_candidate_id,
                "account": transaction.account_candidate_id,
                "counterparty": transaction.counterparty_candidate_id,
            },
            "money": {
                "minor_units": transaction.minor_units,
                "currency": transaction.currency,
                "scale": ISO_MINOR_UNITS[transaction.currency],
                "provenance": transaction.currency_provenance.value,
            },
            "direction": transaction.direction.value,
            "account_state": transaction.account_state.value,
            "counterparty_state": transaction.counterparty_state.value,
            "financial_family": context.financial_family,
            "timestamp": {
                "epoch_ms": context.timestamp_epoch_ms,
                "provenance": context.timestamp_provenance.value,
            },
            "account_resolution": {
                "status": context.account_resolution.status.value,
                "match_count": context.account_resolution.match_count,
                "account_id_hash": context.account_resolution.account_id_hash,
                "matched_alias_hash": context.account_resolution.matched_alias_hash,
                "provenance": context.account_resolution.provenance,
            },
            "evidence": {
                "amount": _evidence_payload(transaction.amount_evidence),
                "direction": _evidence_payload(transaction.direction_evidence),
                "account": _evidence_payload(transaction.account_evidence),
                "counterparty": _evidence_payload(transaction.counterparty_evidence),
            },
        }
    elif selector_result.decision not in {"none", "abstain"} or transaction is not None:
        raise ValueError("recognition decision and reconstructed transaction are inconsistent")
    return {
        "contract": PROCESSING_RESULT_CONTRACT_V2,
        "recognition_decision": selector_result.decision,
        "semantic_result": semantic_result,
        "automatic_persistence": {
            "result": persistence.result.value,
            "primary_reason": persistence.primary_reason,
            "checks": [
                {
                    "check": check.check,
                    "passed": check.passed,
                    "reason_code": check.reason_code,
                }
                for check in persistence.checks
            ],
        },
    }


def _check(name: str, passed: bool, reason_code: str) -> GateCheck:
    return GateCheck(name, passed, None if passed else reason_code)


def _evidence_payload(evidence: Any) -> dict[str, Any] | None:
    if evidence is None:
        return None
    return {
        "start_char": evidence.start_char,
        "end_char": evidence.end_char,
        "start_utf8": evidence.start_utf8,
        "end_utf8": evidence.end_utf8,
        "text": evidence.text,
    }
