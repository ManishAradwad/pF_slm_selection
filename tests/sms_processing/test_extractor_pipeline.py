"""Synthetic and adversarial checks for the SLM-primary extractor path."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from pocketfinancer_sms.account_resolution import (
    AccountCatalogEntry,
    DuplicateStatus,
    ResolutionStatus,
    assess_duplicate,
    resolve_account,
)
from pocketfinancer_sms.analyzer import ANALYSIS_CONTRACT_V2, DeterministicSmsAnalyzer
from pocketfinancer_sms.currency import CurrencyContext
from pocketfinancer_sms.extractor import (
    ExtractionValidationError,
    SourceSpan,
    build_extractor_input,
    parse_and_normalize_extraction,
)
from pocketfinancer_sms.processing_v3 import (
    CancellationToken,
    ExtractionCoordinator,
    RuntimeUnavailableError,
    RuntimeTruncatedError,
)
from pocketfinancer_sms.types import TimestampProvenance


CONFIG_HASH = "a" * 64
RECEIVED_AT = 1_700_000_000_000


def _analyzer() -> DeterministicSmsAnalyzer:
    return DeterministicSmsAnalyzer(
        CurrencyContext("INR", ("core-en", "india")),
        analysis_contract=ANALYSIS_CONTRACT_V2,
    )


def _analysis(source: str):
    return _analyzer().analyze(
        source,
        operation_id="synthetic-extractor",
        operation_config_hash=CONFIG_HASH,
        source_timestamp_epoch_ms=RECEIVED_AT,
        source_timestamp_provenance=TimestampProvenance.PLATFORM_RECEIVED,
    )


def _span(source: str, text: str) -> dict[str, object]:
    start = source.index(text)
    return {"start_scalar": start, "end_scalar": start + len(text), "text": text}


def _posted(source: str, *, amount: str = "INR 1,250.00", account: str = "XX1234") -> str:
    return json.dumps(
        {
            "decision": "posted",
            "amount": {
                "value": "1250.00",
                "currency": "INR",
                "evidence": _span(source, amount),
            },
            "direction": {
                "value": "debit",
                "evidence": _span(source, "debited"),
            },
            "account": {
                "reference": account,
                "evidence": _span(source, account),
            },
            "counterparty": None,
        }
    )


def _coordinator(model, *, catalog=(), rollout_mode="automatic") -> ExtractionCoordinator:
    return ExtractionCoordinator(
        _analyzer(),
        model,
        primary_currency="INR",
        enabled_profile_ids=("core-en", "india"),
        account_catalog=tuple(catalog),
        rollout_mode=rollout_mode,
    )


def _process(coordinator: ExtractionCoordinator, source: str, **kwargs):
    return coordinator.process(
        source=source,
        raw_sender="AD-SYNTH1",
        operation_id="11111111-1111-4111-8111-111111111111",
        operation_config_hash=CONFIG_HASH,
        received_at_epoch_ms=RECEIVED_AT,
        received_timestamp_provenance=TimestampProvenance.PLATFORM_RECEIVED,
        source_event_key="synthetic-source-event",
        idempotency_key="synthetic-idempotency",
        **kwargs,
    )


def test_successful_posted_extraction_is_independent_of_analyzer_candidates() -> None:
    source = "INR 1,250.00 was debited from account XX1234."
    analysis = replace(_analysis(source), candidates=(), cues=())
    payload = build_extractor_input(
        source,
        analysis,
        sender_family="synth#",
        primary_currency="INR",
        enabled_profile_ids=("core-en", "india"),
    )
    assert payload["advisory_evidence"] == []

    result = parse_and_normalize_extraction(
        _posted(source),
        source,
        primary_currency="INR",
        enabled_profile_ids=("core-en", "india"),
    )
    assert result.transaction is not None
    assert result.transaction.minor_units == 125_000
    assert result.transaction.account_reference == "1234"


def test_conflicting_or_incomplete_advisory_evidence_does_not_validate_output() -> None:
    source = "INR 1,250.00 was debited from account XX1234."
    analysis = _analysis(source)
    wrong_candidate = replace(
        analysis.candidates[0],
        value={"minor_units": 1, "currency": "USD", "currency_provenance": "explicit_code"},
    )
    payload = build_extractor_input(
        source,
        replace(analysis, candidates=(wrong_candidate,)),
        sender_family="synth#",
        primary_currency="INR",
        enabled_profile_ids=("core-en", "india"),
    )
    assert len(payload["advisory_evidence"]) >= 1
    result = parse_and_normalize_extraction(
        _posted(source),
        source,
        primary_currency="INR",
        enabled_profile_ids=("core-en", "india"),
    )
    assert result.transaction is not None and result.transaction.currency == "INR"


@pytest.mark.parametrize(
    ("missing", "reason"),
    [
        ("amount", "extractor_missing_amount"),
        ("direction", "extractor_missing_direction"),
        ("account", "extractor_missing_account"),
    ],
)
def test_posted_mandatory_fields_are_required(missing: str, reason: str) -> None:
    source = "INR 1,250.00 was debited from account XX1234."
    payload = json.loads(_posted(source))
    del payload[missing]
    with pytest.raises(ExtractionValidationError, match=reason):
        parse_and_normalize_extraction(
            json.dumps(payload),
            source,
            primary_currency="INR",
            enabled_profile_ids=("core-en", "india"),
        )


    outcome = _process(
        _coordinator(lambda _request, _token: json.dumps(payload)),
        source,
    )
    assert outcome.status == "review"
    assert outcome.reason_codes == (reason,)


    source = "INR 1,250.00 was debited from account XX1234."
    mismatch = json.loads(_posted(source))
    mismatch["amount"]["evidence"]["text"] = "INR 9,999"
    with pytest.raises(ExtractionValidationError, match="extractor_evidence_mismatch"):
        parse_and_normalize_extraction(
            json.dumps(mismatch),
            source,
            primary_currency="INR",
            enabled_profile_ids=("core-en", "india"),
        )
    outcome = _process(
        _coordinator(lambda _request, _token: json.dumps(mismatch)),
        source,
    )
    assert outcome.status == "review"
    assert outcome.reason_codes == ("extractor_evidence_mismatch",)

    invalid = json.loads(_posted(source))
    invalid["account"]["evidence"]["end_scalar"] = len(source) + 1
    with pytest.raises(ExtractionValidationError, match="extractor_evidence_out_of_bounds"):
        parse_and_normalize_extraction(
            json.dumps(invalid),
            source,
            primary_currency="INR",
            enabled_profile_ids=("core-en", "india"),
        )


def test_none_settles_without_review_and_abstain_is_retained() -> None:
    source = "This is a synthetic service notice."
    none = _process(_coordinator(lambda _payload, _token: '{"decision":"none"}'), source)
    assert none.model_invoked is True
    assert none.status == "not_posted"
    assert none.review_case is None

    abstain = _process(_coordinator(lambda _payload, _token: '{"decision":"abstain"}'), source)
    assert abstain.status == "review"
    assert abstain.review_case is not None
    assert abstain.reason_codes == ("extractor_abstained",)


def test_all_valid_incoming_messages_invoke_even_when_legacy_triage_would_discard() -> None:
    calls = []

    def model(payload, _token):
        calls.append(payload)
        return '{"decision":"none"}'

    result = _process(_coordinator(model), "Your synthetic OTP is 123456.")
    assert result.status == "not_posted"
    assert len(calls) == 1
    assert calls[0]["message"] == "Your synthetic OTP is 123456."


def test_invalid_and_reliably_outgoing_inputs_bypass_the_model() -> None:
    calls = []
    coordinator = _coordinator(lambda *_args: calls.append(True) or '{"decision":"none"}')
    outgoing = _process(coordinator, "Synthetic outgoing message.", is_outgoing=True)
    invalid = _process(coordinator, "", input_valid=False)
    assert outgoing.status == invalid.status == "bypassed"
    assert outgoing.reason_codes == ("reliable_outgoing_metadata",)
    assert invalid.reason_codes == ("invalid_input",)
    assert calls == []


def test_receipt_timestamp_is_authoritative_and_model_time_fields_are_rejected() -> None:
    source = "INR 1,250.00 was debited from account XX1234."
    payload = json.loads(_posted(source))
    payload["transaction_time"] = "2026-01-01"
    with pytest.raises(ExtractionValidationError, match="extractor_posted_field_set_invalid"):
        parse_and_normalize_extraction(
            json.dumps(payload),
            source,
            primary_currency="INR",
            enabled_profile_ids=("core-en", "india"),
        )
    catalog = (AccountCatalogEntry("acct-1", "bank_account", ("XX1234",)),)
    outcome = _process(_coordinator(lambda *_args: _posted(source), catalog=catalog), source)
    assert outcome.received_at_epoch_ms == RECEIVED_AT
    assert outcome.received_timestamp_provenance == TimestampProvenance.PLATFORM_RECEIVED


def test_offline_acquisition_time_cannot_claim_native_persistence_eligibility() -> None:
    source = "INR 1,250.00 was debited from account XX1234."
    catalog = (AccountCatalogEntry("acct-1", "bank_account", ("XX1234",)),)
    outcome = _coordinator(lambda *_args: _posted(source), catalog=catalog).process(
        source=source,
        raw_sender="AD-SYNTH1",
        operation_id="11111111-1111-4111-8111-111111111111",
        operation_config_hash=CONFIG_HASH,
        received_at_epoch_ms=RECEIVED_AT,
        received_timestamp_provenance=TimestampProvenance.ACQUISITION_SUPPLIED_MESSAGE_TIME,
        source_event_key="synthetic-source-event",
        idempotency_key="synthetic-idempotency",
    )
    assert outcome.status == "review"
    assert "persistence_receipt_timestamp_provenance_invalid" in outcome.reason_codes
    assert outcome.review_case is not None
    assert outcome.review_case.to_dict()["receipt_timestamp"]["provenance"] == (
        "acquisition_supplied_message_time"
    )


def test_no_wall_clock_deadline_is_supplied_to_runtime() -> None:
    source = "INR 1,250.00 was debited from account XX1234."
    observed = {}

    def simulated_sixty_one_second_runtime(payload, token):
        observed["payload"] = payload
        observed["cancelled"] = token.cancelled
        observed["simulated_elapsed_ms"] = 61_000
        return _posted(source)

    catalog = (AccountCatalogEntry("acct-1", "bank_account", ("XX1234",)),)
    outcome = _process(_coordinator(simulated_sixty_one_second_runtime, catalog=catalog), source)
    assert observed == {
        "payload": observed["payload"],
        "cancelled": False,
        "simulated_elapsed_ms": 61_000,
    }
    assert "deadline" not in observed["payload"]
    assert outcome.status == "eligible"


def test_cancellation_fences_late_output_and_preserves_review() -> None:
    source = "INR 1,250.00 was debited from account XX1234."
    token = CancellationToken()

    def late_runtime(_payload, cancellation):
        cancellation.cancel()
        return _posted(source)

    outcome = _process(_coordinator(late_runtime), source, cancellation=token)
    assert outcome.status == "interrupted"
    assert outcome.result is None
    assert outcome.reason_codes == ("user_cancelled",)
    assert outcome.review_case is not None


def test_pre_cancelled_operation_never_invokes_the_model() -> None:
    source = "INR 1,250.00 was debited from account XX1234."
    token = CancellationToken()
    token.cancel()
    calls: list[bool] = []
    outcome = _process(
        _coordinator(lambda *_args: calls.append(True) or '{"decision":"none"}'),
        source,
        cancellation=token,
    )
    assert outcome.status == "interrupted"
    assert outcome.model_invoked is False
    assert outcome.reason_codes == ("user_cancelled",)
    assert calls == []


def test_runtime_truncation_is_retained_for_review() -> None:
    def truncated_runtime(_payload, _token):
        raise RuntimeTruncatedError

    outcome = _process(_coordinator(truncated_runtime), "Synthetic service message.")
    assert outcome.status == "review"
    assert outcome.result is None
    assert outcome.reason_codes == ("runtime_output_truncated",)
    assert outcome.review_case is not None


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (RuntimeUnavailableError(), "runtime_unavailable"),
        (RuntimeError(), "runtime_failure"),
    ],
)
def test_runtime_failures_are_retained_for_review(
    error: Exception, reason: str
) -> None:
    def failing_runtime(_payload, _token):
        raise error

    outcome = _process(_coordinator(failing_runtime), "Synthetic service message.")
    assert outcome.status == "review"
    assert outcome.result is None
    assert outcome.reason_codes == (reason,)
    assert outcome.review_case is not None


def test_process_interruption_is_resumable_and_retained() -> None:
    def interrupted_runtime(_payload, _token):
        raise KeyboardInterrupt

    outcome = _process(_coordinator(interrupted_runtime), "Synthetic service message.")
    assert outcome.status == "interrupted"
    assert outcome.result is None
    assert outcome.reason_codes == ("process_interrupted",)
    assert outcome.review_case is not None


    assert resolve_account(None, ()).status == ResolutionStatus.MISSING
    assert resolve_account("XX1234", ()).status == ResolutionStatus.UNRESOLVED
    unique = resolve_account(
        "XX1234", (AccountCatalogEntry("acct-1", "bank_account", ("A/c **1234",)),)
    )
    assert unique.status == ResolutionStatus.UNIQUELY_RESOLVED
    assert unique.account_id == "acct-1"
    ambiguous = resolve_account(
        "XX1234",
        (
            AccountCatalogEntry("acct-1", "bank_account", ("**1234",)),
            AccountCatalogEntry("acct-2", "card", ("XX1234",)),
        ),
    )
    assert ambiguous.status == ResolutionStatus.AMBIGUOUS
    assert ambiguous.account_id is None


def test_account_ambiguity_and_duplicate_statuses_block_eligibility() -> None:
    source = "INR 1,250.00 was debited from account XX1234."
    catalog = (
        AccountCatalogEntry("acct-1", "bank_account", ("**1234",)),
        AccountCatalogEntry("acct-2", "card", ("XX1234",)),
    )
    ambiguous = _process(_coordinator(lambda *_args: _posted(source), catalog=catalog), source)
    assert ambiguous.status == "review"
    assert "account_resolution_ambiguous" in ambiguous.reason_codes
    unresolved = _process(_coordinator(lambda *_args: _posted(source)), source)
    assert unresolved.status == "review"
    assert unresolved.account_resolution is not None
    assert unresolved.account_resolution.status == ResolutionStatus.UNRESOLVED
    assert "account_resolution_unresolved" in unresolved.reason_codes

    persisted = _process(
        _coordinator(
            lambda *_args: _posted(source),
            catalog=(AccountCatalogEntry("acct-1", "bank_account", ("XX1234",)),),
        ),
        source,
        persisted_source_event_keys=frozenset({"synthetic-source-event"}),
    )
    assert persisted.duplicate_assessment is not None
    assert persisted.duplicate_assessment.status == DuplicateStatus.ALREADY_PERSISTED
    assert "duplicate_already_persisted" in persisted.reason_codes


def test_duplicate_assessment_is_idempotency_keyed() -> None:
    fingerprint = hashlib.sha256(b"synthetic").hexdigest()
    possible = assess_duplicate(
        idempotency_key="idempotency",
        source_event_key="source-event",
        transaction_fingerprint_value=fingerprint,
        known_transaction_fingerprints=frozenset({fingerprint}),
    )
    assert possible.status == DuplicateStatus.POSSIBLE_DUPLICATE
    already_by_idempotency = assess_duplicate(
        idempotency_key="idempotency",
        source_event_key="different-source-event",
        transaction_fingerprint_value=fingerprint,
        persisted_idempotency_keys=frozenset({"idempotency"}),
    )
    assert already_by_idempotency.status == DuplicateStatus.ALREADY_PERSISTED
    already_by_source_event = assess_duplicate(
        idempotency_key="different-idempotency",
        source_event_key="source-event",
        transaction_fingerprint_value=fingerprint,
        persisted_source_event_keys=frozenset({"source-event"}),
    )
    assert already_by_source_event.status == DuplicateStatus.ALREADY_PERSISTED


@pytest.mark.parametrize(
    "source",
    [
        "😀 INR 1,250.00 was debited from account XX1234.",
        "e\u0301 INR 1,250.00 was debited from account XX1234.",
        "Ａ INR 1,250.00 was debited from account XX1234.",
    ],
)
def test_unicode_scalar_spans_cover_emoji_combining_marks_and_bmp_text(source: str) -> None:
    result = parse_and_normalize_extraction(
        _posted(source),
        source,
        primary_currency="INR",
        enabled_profile_ids=("core-en", "india"),
    )
    assert result.transaction is not None
    span = result.transaction.amount_span
    assert source[span.start_scalar : span.end_scalar] == span.text
    assert SourceSpan.from_source(source, span.start_scalar, span.end_scalar) == span


def test_escaped_json_preserves_unicode_scalar_span_text() -> None:
    source = "INR 10.00 was credited to account XX1234 from CAF\u00c9."
    payload = {
        "decision": "posted",
        "amount": {
            "value": "10.00",
            "currency": "INR",
            "evidence": _span(source, "INR 10.00"),
        },
        "direction": {
            "value": "credit",
            "evidence": _span(source, "credited"),
        },
        "account": {
            "reference": "XX1234",
            "evidence": _span(source, "XX1234"),
        },
        "counterparty": {
            "value": "CAF\u00c9",
            "evidence": _span(source, "CAF\u00c9"),
        },
    }
    result = parse_and_normalize_extraction(
        json.dumps(payload, ensure_ascii=True),
        source,
        primary_currency="INR",
        enabled_profile_ids=("core-en", "india"),
    )
    assert result.transaction is not None
    assert result.transaction.counterparty == "caf\u00e9"

@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ('{"decision":"none","decision":"abstain"}', "extractor_duplicate_json_key"),
        (
            '{"decision":"posted","amount":{"value":"1","value":"2"}}',
            "extractor_duplicate_json_key",
        ),
        (
            '{"decision":"posted","amount":{"value":"1","currency":"INR","evidence":{"start_scalar":0,"start_scalar":1}}}',
            "extractor_duplicate_json_key",
        ),
        ('{"decision":"none"} trailing', "extractor_extra_content"),
        ('[{"decision":"none"}]', "extractor_output_not_object"),
        ('{"decision":NaN}', "extractor_malformed_json"),
    ],
)
def test_duplicate_keys_at_nested_depth_and_trailing_content_fail_closed(
    raw: str, reason: str
) -> None:
    with pytest.raises(ExtractionValidationError, match=reason):
        parse_and_normalize_extraction(
            raw,
            "Synthetic source",
            primary_currency="INR",
            enabled_profile_ids=("core-en", "india"),
        )


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (" " * 16_385, "runtime_output_truncated"),
        ("\ud800", "extractor_malformed_json"),
    ],
)
def test_raw_output_limits_and_invalid_unicode_fail_closed(raw: str, reason: str) -> None:
    with pytest.raises(ExtractionValidationError, match=reason):
        parse_and_normalize_extraction(
            raw,
            "Synthetic source",
            primary_currency="INR",
            enabled_profile_ids=("core-en", "india"),
        )


@pytest.mark.parametrize(
    ("amount_text", "declared", "reason"),
    [
        ("INR 1.234", "1.234", "extractor_amount_invalid"),
        (
            "INR 92233720368547758.08",
            "92233720368547758.08",
            "extractor_amount_invalid",
        ),
        ("INR 1,250.00", "1251.00", "extractor_amount_value_disagreement"),
        ("INR 1,250.00", "01250.00", "extractor_amount_invalid"),
    ],
)
def test_money_precision_overflow_and_value_disagreement(
    amount_text: str, declared: str, reason: str
) -> None:
    source = f"{amount_text} was debited from account XX1234."
    payload = json.loads(_posted(source, amount=amount_text))
    payload["amount"]["value"] = declared
    with pytest.raises(ExtractionValidationError, match=reason):
        parse_and_normalize_extraction(
            json.dumps(payload),
            source,
            primary_currency="INR",
            enabled_profile_ids=("core-en", "india"),
        )
