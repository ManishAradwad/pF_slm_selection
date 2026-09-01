"""Strict Grounded Candidate Selector parsing and host reconstruction."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .currency import ISO_MINOR_UNITS
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
SELECTOR_INPUT_CONTRACT = "pocketfinancer.grounded-candidate-selector-input/1"


class SelectorValidationError(ValueError):
    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)

    def __str__(self) -> str:
        return self.reason_code


def model_candidate_payload(source: str, analysis: Analysis) -> dict[str, Any]:
    """Return compact source-backed options for one non-thinking model pass."""

    if hashlib.sha256(source.encode("utf-8")).hexdigest() != analysis.source_fingerprint:
        raise ValueError("selector source does not match the current analysis")

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
    return {
        "contract": SELECTOR_INPUT_CONTRACT,
        "analysis_id": analysis.analysis_id,
        "message": source,
        "candidates": candidates,
    }


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
    if len(candidate_map) != len(analysis.candidates):
        raise SelectorValidationError("selector_candidate_ids_ambiguous")
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
        minor_units = amount.value["minor_units"]
        currency = amount.value["currency"]
        if isinstance(minor_units, bool) or not isinstance(minor_units, int) or minor_units <= 0:
            raise ValueError
        if not isinstance(currency, str) or currency not in ISO_MINOR_UNITS:
            raise ValueError
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
        minor_units=minor_units,
        currency=currency,
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
        if candidate.evidence is not None or candidate.clause_id is not None or candidate.value != {"state": "absent"}:
            raise SelectorValidationError("selector_absent_candidate_metadata_invalid")
        return state_type.ABSENT, None
    if candidate.evidence is None or candidate.clause_id is None:
        raise SelectorValidationError("selector_optional_evidence_missing")
    if candidate.kind == CandidateKind.ACCOUNT:
        if candidate.value.get("account_type") not in {"bank_account", "card", "vpa"} or not isinstance(
            candidate.value.get("identifier"), str
        ):
            raise SelectorValidationError("selector_candidate_metadata_invalid")
    elif candidate.kind == CandidateKind.COUNTERPARTY and not isinstance(
        candidate.value.get("surface"), str
    ):
        raise SelectorValidationError("selector_candidate_metadata_invalid")
    return state_type.PRESENT, candidate.evidence
