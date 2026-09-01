"""Synthetic-only tests for the production-intended SMS processing foundation."""

from __future__ import annotations

import json

import pytest

from pocketfinancer_sms.analyzer import DeterministicSmsAnalyzer
from pocketfinancer_sms.currency import CurrencyContext
from pocketfinancer_sms.persistence import evaluate_persistence
from pocketfinancer_sms.selector import (
    SelectorValidationError,
    model_candidate_payload,
    parse_and_reconstruct,
)
from pocketfinancer_sms.triage import evaluate_triage
from pocketfinancer_sms.types import (
    CandidateKind,
    CurrencyProvenance,
    Disposition,
    PersistenceContext,
    SelectorAction,
    TimestampProvenance,
)


def _analysis(message: str, *, primary: str = "INR", operation_id: str = "synthetic-op"):
    analyzer = DeterministicSmsAnalyzer(CurrencyContext(primary, ("core-en", "india")))
    return analyzer.analyze(message, operation_id=operation_id, is_outgoing=False)


def _posted_payload(analysis, *, absent_account: bool = False) -> str:
    def first(kind: CandidateKind, *, absent: bool | None = None):
        matches = [candidate for candidate in analysis.candidates_of(kind)]
        if absent is not None:
            matches = [candidate for candidate in matches if candidate.explicit_absence is absent]
        return matches[0].candidate_id

    return json.dumps(
        {
            "decision": "posted",
            "amount": first(CandidateKind.AMOUNT),
            "direction": first(CandidateKind.DIRECTION),
            "account": first(CandidateKind.ACCOUNT, absent=absent_account),
            "counterparty": first(CandidateKind.COUNTERPARTY, absent=False),
        }
    )


def test_single_completed_event_invokes_and_reconstructs_only_selected_ids() -> None:
    analysis = _analysis(
        "INR 125.40 was debited from account **4321 at NORTH MARKET. Available balance INR 900."
    )
    triage = evaluate_triage(analysis)

    assert triage.disposition == Disposition.INVOKE
    assert triage.selector_action == SelectorAction.RUN_NORMAL
    result = parse_and_reconstruct(_posted_payload(analysis), analysis)
    assert result.decision == "posted"
    assert result.transaction is not None
    assert result.transaction.minor_units == 12540
    assert result.transaction.currency == "INR"
    assert result.transaction.currency_provenance == CurrencyProvenance.EXPLICIT_CODE


def test_explicit_currency_overrides_primary_and_bare_amount_uses_primary() -> None:
    explicit = _analysis("EUR 19.95 was paid from account **1234 at CAFE.", primary="USD")
    bare = _analysis("75.50 was paid from account **1234 at CAFE.", primary="USD")

    explicit_amount = explicit.candidates_of(CandidateKind.AMOUNT)[0]
    bare_amount = bare.candidates_of(CandidateKind.AMOUNT)[0]
    assert explicit_amount.value["currency"] == "EUR"
    assert explicit_amount.value["currency_provenance"] == "explicit_code"
    assert bare_amount.value["currency"] == "USD"
    assert bare_amount.value["currency_provenance"] == "user_primary_default"


def test_mixed_completed_and_failed_clauses_are_retained_and_may_run_assistively() -> None:
    analysis = _analysis(
        "INR 100 was debited from account **1234 at STORE. However INR 50 transfer failed."
    )
    triage = evaluate_triage(analysis)

    assert triage.disposition == Disposition.RETAIN_REVIEW
    assert triage.selector_action == SelectorAction.RUN_ASSISTIVE
    assert "review_conflicting_or_ambiguous_context" in triage.reason_codes


def test_standalone_otp_discards_but_completed_event_with_otp_language_does_not() -> None:
    standalone = _analysis(
        "482910 is your OTP to authorize an online transaction of INR 500. Do not share it."
    )
    posted = _analysis(
        "INR 500 was debited from account **1234 at STORE; no OTP was required."
    )

    assert evaluate_triage(standalone).disposition == Disposition.DISCARD
    assert "discard_standalone_credential_otp" in evaluate_triage(standalone).reason_codes
    assert evaluate_triage(posted).disposition == Disposition.INVOKE


def test_unknown_or_cross_message_candidate_fails_closed() -> None:
    first = _analysis(
        "INR 10 was debited from account **1111 at STORE.", operation_id="synthetic-one"
    )
    second = _analysis(
        "INR 10 was debited from account **1111 at STORE.", operation_id="synthetic-two"
    )
    payload = json.loads(_posted_payload(first))
    payload["amount"] = second.candidates_of(CandidateKind.AMOUNT)[0].candidate_id

    with pytest.raises(
        SelectorValidationError, match="selector_unknown_or_cross_message_candidate"
    ):
        parse_and_reconstruct(json.dumps(payload), first)


def test_posted_requires_exact_field_set_and_available_grounded_candidates() -> None:
    analysis = _analysis("INR 10 was debited from account **1111 at STORE.")
    payload = json.loads(_posted_payload(analysis))
    payload["amount_minor_units"] = 1000

    with pytest.raises(SelectorValidationError, match="selector_posted_field_set_invalid"):
        parse_and_reconstruct(json.dumps(payload), analysis)


def test_model_payload_contains_ids_and_host_evidence_but_never_offsets_or_money_values() -> None:
    message = "INR 10 was credited to account **1111 from FRIEND."
    analysis = _analysis(message)
    payload = model_candidate_payload(message, analysis)
    rendered = json.dumps(payload)

    assert payload["message"] == message
    assert payload["analysis_id"] == analysis.analysis_id
    assert '"id"' in rendered
    assert '"evidence"' in rendered
    assert "start_utf8" not in rendered
    assert "minor_units" not in rendered
    assert "currency_provenance" not in rendered

    with pytest.raises(ValueError, match="source does not match"):
        model_candidate_payload("different synthetic message", analysis)


def test_recognition_is_broader_than_automatic_persistence() -> None:
    analysis = _analysis("INR 10 was debited at STORE.")
    result = parse_and_reconstruct(_posted_payload(analysis, absent_account=True), analysis)
    triage = evaluate_triage(analysis)
    persistence = evaluate_persistence(
        result,
        analysis,
        triage,
        PersistenceContext(
            timestamp_epoch_ms=1_700_000_000_000,
            timestamp_provenance=TimestampProvenance.PLATFORM_RECEIVED,
            account_resolution_count=0,
            approved_currency_provenance=frozenset(CurrencyProvenance),
        ),
    )

    assert result.decision == "posted"
    assert persistence.safe_to_persist is False
    assert "persistence_account_not_present" in persistence.reason_codes


def test_conflicting_explicit_currencies_route_to_review() -> None:
    analysis = _analysis(
        "INR 100 was debited from account **1234 at STORE. Available balance USD 20."
    )
    triage = evaluate_triage(analysis)
    assert triage.disposition == Disposition.RETAIN_REVIEW
    assert "conflicting_currencies" in triage.reason_codes


def test_invalid_and_reliable_outgoing_messages_are_terminal_discards() -> None:
    analyzer = DeterministicSmsAnalyzer(CurrencyContext("INR", ("core-en", "india")))
    invalid = analyzer.analyze("", operation_id="invalid", input_valid=False)
    outgoing = analyzer.analyze(
        "INR 10 was paid at STORE.", operation_id="outgoing", is_outgoing=True
    )

    assert evaluate_triage(invalid).disposition == Disposition.DISCARD
    assert evaluate_triage(outgoing).disposition == Disposition.DISCARD


def test_ambiguous_or_unsupported_currency_never_falls_back_to_primary() -> None:
    ambiguous = _analysis("$ 10 was paid from account **1234 at STORE.", primary="USD")
    unsupported = _analysis("CNY 10 was paid from account **1234 at STORE.", primary="INR")

    assert not ambiguous.candidates_of(CandidateKind.AMOUNT)
    assert "ambiguous_currency_marker" in ambiguous.reason_codes
    assert evaluate_triage(ambiguous).disposition == Disposition.RETAIN_REVIEW
    assert not unsupported.candidates_of(CandidateKind.AMOUNT)
    assert "unsupported_currency_code" in unsupported.reason_codes
    assert evaluate_triage(unsupported).disposition == Disposition.RETAIN_REVIEW


def test_negated_direction_is_not_completed_evidence_and_discards_terminally() -> None:
    analysis = _analysis("INR 10 has not been debited from account **1234.")

    assert not analysis.candidates_of(CandidateKind.DIRECTION)
    triage = evaluate_triage(analysis)
    assert triage.disposition == Disposition.DISCARD
    assert "discard_explicit_non_posted_movement" in triage.reason_codes


def test_future_direction_is_not_completed_and_stays_available_for_review() -> None:
    analysis = _analysis("INR 10 will be debited from account **1234 tomorrow.")

    assert not analysis.candidates_of(CandidateKind.DIRECTION)
    triage = evaluate_triage(analysis)
    assert triage.disposition == Disposition.RETAIN_REVIEW
    assert "pending_event" in triage.reason_codes


def test_known_account_identifier_is_not_enumerated_as_bare_money() -> None:
    analysis = _analysis("Account **1234 was debited at STORE.")

    assert not analysis.candidates_of(CandidateKind.AMOUNT)
    assert evaluate_triage(analysis).disposition == Disposition.RETAIN_REVIEW


def test_multiple_completed_directions_in_one_clause_never_invoke_normally() -> None:
    analysis = _analysis("INR 10 was debited and INR 20 was credited to account **1234.")
    triage = evaluate_triage(analysis)

    assert len(analysis.candidates_of(CandidateKind.DIRECTION)) == 2
    assert triage.disposition == Disposition.RETAIN_REVIEW
    assert triage.selector_action == SelectorAction.RUN_ASSISTIVE
    assert "review_multiple_completed_event_candidates" in triage.reason_codes


def test_structural_normalization_matches_fullwidth_text_but_preserves_evidence() -> None:
    message = "ＩＮＲ １０ was credited to account **１２３４ from FRIEND."
    analysis = _analysis(message)
    amount = analysis.candidates_of(CandidateKind.AMOUNT)[0]
    account = next(
        candidate
        for candidate in analysis.candidates_of(CandidateKind.ACCOUNT)
        if not candidate.explicit_absence
    )

    assert amount.value["minor_units"] == 1000
    assert amount.evidence is not None and amount.evidence.text == "ＩＮＲ １０"
    assert account.evidence is not None and "１２３４" in account.evidence.text
    assert len(analysis.metadata["normalized_structural_fingerprint"]) == 64
