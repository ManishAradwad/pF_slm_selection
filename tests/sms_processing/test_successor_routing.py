"""Focused policy and partial-Review tests using the existing routing engine."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from pocketfinancer_sms.account_resolution import AccountCatalogEntry, DuplicateStatus
from pocketfinancer_sms.analyzer import ANALYSIS_CONTRACT_V2, DeterministicSmsAnalyzer
from pocketfinancer_sms.currency import CurrencyContext
from pocketfinancer_sms.processing_v3 import (
    CancellationToken,
    ExtractionCoordinator,
    ReviewCaseV2,
    RuntimeUnavailableError,
    SuccessorExtractionCoordinator,
)
from pocketfinancer_sms.types import TimestampProvenance


ROOT = Path(__file__).resolve().parents[2]
SOURCE = "INR 125.00 was debited from account XX1234."
RECEIVED_AT = 1_700_000_000_000


def _span(source: str, text: str) -> dict[str, object]:
    start = source.index(text)
    return {"start_scalar": start, "end_scalar": start + len(text), "text": text}


def _posted(source: str = SOURCE) -> str:
    return json.dumps(
        {
            "decision": "posted",
            "amount": {
                "value": "125.00",
                "currency": "INR",
                "evidence": _span(source, "INR 125.00"),
            },
            "direction": {"value": "debit", "evidence": _span(source, "debited")},
            "account": {"reference": "XX1234", "evidence": _span(source, "XX1234")},
            "counterparty": None,
        }
    )


def _analyzer() -> DeterministicSmsAnalyzer:
    return DeterministicSmsAnalyzer(
        CurrencyContext("INR", ("core-en", "india")),
        analysis_contract=ANALYSIS_CONTRACT_V2,
    )


def _catalog(*, ambiguous: bool = False) -> tuple[AccountCatalogEntry, ...]:
    entries = [AccountCatalogEntry("acct-1", "bank_account", ("XX1234",))]
    if ambiguous:
        entries.append(AccountCatalogEntry("acct-2", "card", ("**1234",)))
    return tuple(entries)


def _successor(model, *, catalog: tuple[AccountCatalogEntry, ...] | None = None):
    return SuccessorExtractionCoordinator(
        _analyzer(),
        model,
        primary_currency="INR",
        enabled_profile_ids=("core-en", "india"),
        account_catalog=_catalog() if catalog is None else catalog,
    )


def _process(coordinator, source: str = SOURCE, **kwargs):
    return coordinator.process(
        source=source,
        raw_sender="AD-SYNTH1",
        operation_id="11111111-1111-4111-8111-111111111111",
        operation_config_hash="a" * 64,
        received_at_epoch_ms=RECEIVED_AT,
        received_timestamp_provenance=TimestampProvenance.PLATFORM_RECEIVED,
        source_event_key="synthetic-source-event",
        idempotency_key="synthetic-idempotency",
        **kwargs,
    )


def test_v4_remains_review_only_while_successor_is_eligible() -> None:
    frozen = ExtractionCoordinator(
        _analyzer(),
        lambda *_args: _posted(),
        primary_currency="INR",
        enabled_profile_ids=("core-en", "india"),
        account_catalog=_catalog(),
        rollout_mode="review_only",
    )
    old = _process(frozen)
    new = _process(_successor(lambda *_args: _posted()))
    assert old.status == "blocked"
    assert old.review_case is not None
    assert old.review_case.contract == "pocketfinancer.review-case/1"
    assert new.status == "eligible"
    assert new.review_case is None


def test_missing_field_retains_other_grounded_slm_fields_without_model_repair() -> None:
    source = "INR 125.00 was debited."
    payload = json.loads(_posted(SOURCE))
    del payload["account"]
    payload["amount"]["evidence"] = _span(source, "INR 125.00")
    payload["direction"]["evidence"] = _span(source, "debited")
    outcome = _process(_successor(lambda *_args: json.dumps(payload)), source)
    assert outcome.status == "review"
    assert isinstance(outcome.review_case, ReviewCaseV2)
    assert outcome.review_case.extractor_suggestion is None
    slm = [item for item in outcome.review_case.field_evidence if item.origin == "slm"]
    assert [(item.field, item.validation_state) for item in slm] == [
        ("amount", "valid"),
        ("direction", "valid"),
    ]
    assert all(item.origin != "advisory_analyzer" for item in slm)
    serialized = json.loads(json.dumps(outcome.review_case.to_dict()))
    schema = json.loads(
        (ROOT / "configs/sms_processing/contracts/v5/review-case.schema.json").read_text()
    )
    jsonschema.validate(serialized, schema)


def test_account_and_duplicate_exceptions_remain_review_only() -> None:
    missing = _process(_successor(lambda *_args: _posted(), catalog=()))
    ambiguous = _process(
        _successor(lambda *_args: _posted(), catalog=_catalog(ambiguous=True))
    )
    duplicate = _process(
        _successor(lambda *_args: _posted()),
        persisted_source_event_keys=frozenset({"synthetic-source-event"}),
    )
    assert missing.status == ambiguous.status == duplicate.status == "review"
    assert "account_resolution_unresolved" in missing.reason_codes
    assert "account_resolution_ambiguous" in ambiguous.reason_codes
    assert duplicate.duplicate_assessment is not None
    assert duplicate.duplicate_assessment.status == DuplicateStatus.ALREADY_PERSISTED
    assert "duplicate_already_persisted" in duplicate.reason_codes


@pytest.mark.parametrize(
    ("model", "reason"),
    [
        (lambda *_args: '{"decision":"abstain"}', "extractor_abstained"),
        (lambda *_args: "not-json", "extractor_malformed_json"),
        (
            lambda *_args: (_ for _ in ()).throw(RuntimeUnavailableError()),
            "runtime_unavailable",
        ),
        (lambda *_args: (_ for _ in ()).throw(RuntimeError()), "runtime_failure"),
    ],
)
def test_abstain_invalid_and_runtime_failures_enter_review(model, reason: str) -> None:
    outcome = _process(_successor(model))
    assert outcome.status == "review"
    assert outcome.review_case is not None
    assert outcome.reason_codes == (reason,)


def test_interruption_and_incompatible_configuration_enter_review() -> None:
    token = CancellationToken()
    token.cancel()
    interrupted = _process(_successor(lambda *_args: _posted()), cancellation=token)
    incompatible = _process(
        _successor(lambda *_args: _posted()), configuration_hash_matches=False
    )
    assert interrupted.status == "interrupted"
    assert interrupted.review_case is not None
    assert incompatible.status == "review"
    assert incompatible.review_case is not None
    assert "persistence_configuration_hash_mismatch" in incompatible.reason_codes


def test_valid_none_settles_without_transaction_or_review() -> None:
    outcome = _process(_successor(lambda *_args: '{"decision":"none"}'))
    assert outcome.status == "not_posted"
    assert outcome.review_case is None
    assert outcome.result is not None and outcome.result.decision == "none"


def test_sanitized_routing_vector_matches_executable_partial_case() -> None:
    golden = json.loads(
        (ROOT / "tests/sms_processing/golden/native-v5/routing-policy.json").read_text()
    )
    partial = golden["partial_review"]
    outcome = _process(
        _successor(lambda *_args: json.dumps(partial["invalid_model_output"])),
        partial["source"],
    )
    assert isinstance(outcome.review_case, ReviewCaseV2)
    fields = [
        item.field
        for item in outcome.review_case.field_evidence
        if item.origin == "slm" and item.validation_state == "valid"
    ]
    assert fields == partial["expected_valid_slm_fields"]
