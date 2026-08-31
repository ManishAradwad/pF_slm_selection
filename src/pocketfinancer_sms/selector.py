"""Strict Grounded Candidate Selector parsing and host reconstruction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .types import (
    AccountState,
    Analysis,
    Candidate,
    CandidateKind,
    CounterpartyState,
    CurrencyProvenance,
    Direction,
    ReconstructedTransaction,
    SelectorResult,
)


SELECTOR_CONTRACT = "pocketfinancer.grounded-candidate-selector/1"


@dataclass(frozen=True, slots=True)
class SelectorValidationError(ValueError):
    reason_code: str

    def __str__(self) -> str:
        return self.reason_code


def model_candidate_payload(analysis: Analysis) -> dict[str, Any]:
    """Return compact source-backed options for one non-thinking model pass."""

    candidates = []
    for candidate in analysis.candidates:
        item: dict[str, Any] = {
            "id": candidate.candidate_id,
            "kind": candidate.kind.value,
            "clause": candidate.clause_id,
            "absent": candidate.explicit_absence,
        }
        if candidate.evidence is not None:
            item["evidence"] = candidate.evidence.text
        if candidate.kind == CandidateKind.DIRECTION:
            item["direction"] = candidate.value["direction"]
        candidates.append(item)
    return {"contract": SELECTOR_CONTRACT, "candidates": candidates}


def parse_and_reconstruct(raw_output: str, analysis: Analysis) -> SelectorResult:
    try:
        payload = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError) as exc:
        raise SelectorValidationError("selector_malformed_json") from exc
    if not isinstance(payload, dict):
        raise SelectorValidationError("selector_output_not_object")
    decision = payload.get("decision")
    if decision in {"none", "abstain"}:
        if set(payload) != {"decision"}:
            raise SelectorValidationError("selector_non_posted_extra_fields")
        return SelectorResult(decision=decision)
    if decision != "posted":
        raise SelectorValidationError("selector_unknown_decision")

    expected = {"decision", "amount", "direction", "account", "counterparty"}
    if set(payload) != expected:
        raise SelectorValidationError("selector_posted_field_set_invalid")
    if any(not isinstance(payload[field], str) or not payload[field] for field in expected - {"decision"}):
        raise SelectorValidationError("selector_candidate_id_invalid")

    candidate_map = analysis.candidate_map()
    selected = {
        field: _resolve_candidate(candidate_map, payload[field], CandidateKind(field))
        for field in ("amount", "direction", "account", "counterparty")
    }
    amount = selected["amount"]
    direction = selected["direction"]
    account = selected["account"]
    counterparty = selected["counterparty"]
    if amount.explicit_absence or direction.explicit_absence:
        raise SelectorValidationError("selector_required_candidate_absent")
    if amount.evidence is None or direction.evidence is None:
        raise SelectorValidationError("selector_required_evidence_missing")
    if amount.clause_id != direction.clause_id:
        raise SelectorValidationError("selector_cross_clause_core_selection")

    try:
        provenance = CurrencyProvenance(amount.value["currency_provenance"])
        parsed_direction = Direction(direction.value["direction"])
    except (KeyError, ValueError) as exc:
        raise SelectorValidationError("selector_candidate_metadata_invalid") from exc

    account_state, account_evidence = _optional_state(account, AccountState)
    counterparty_state, counterparty_evidence = _optional_state(counterparty, CounterpartyState)
    transaction = ReconstructedTransaction(
        analysis_id=analysis.analysis_id,
        amount_candidate_id=amount.candidate_id,
        direction_candidate_id=direction.candidate_id,
        account_candidate_id=account.candidate_id,
        counterparty_candidate_id=counterparty.candidate_id,
        minor_units=int(amount.value["minor_units"]),
        currency=str(amount.value["currency"]),
        currency_provenance=provenance,
        direction=parsed_direction,
        account_state=account_state,
        account_evidence=account_evidence,
        counterparty_state=counterparty_state,
        counterparty_evidence=counterparty_evidence,
        amount_evidence=amount.evidence,
        direction_evidence=direction.evidence,
    )
    return SelectorResult(decision="posted", transaction=transaction)


def _resolve_candidate(
    candidate_map: dict[str, Candidate], candidate_id: str, expected_kind: CandidateKind
) -> Candidate:
    candidate = candidate_map.get(candidate_id)
    if candidate is None:
        raise SelectorValidationError("selector_unknown_or_cross_message_candidate")
    if candidate.kind != expected_kind:
        raise SelectorValidationError("selector_candidate_kind_mismatch")
    return candidate


def _optional_state(candidate: Candidate, state_type: type) -> tuple[Any, Any]:
    if candidate.explicit_absence:
        return state_type.ABSENT, None
    if candidate.evidence is None:
        raise SelectorValidationError("selector_optional_evidence_missing")
    return state_type.PRESENT, candidate.evidence
