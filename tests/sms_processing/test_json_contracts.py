"""Synthetic conformance checks binding executable objects to versioned schemas."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import jsonschema
import pytest

from pocketfinancer_sms.analyzer import ANALYSIS_CONTRACT, DeterministicSmsAnalyzer
from pocketfinancer_sms.currency import (
    ISO_4217_CURRENT_CODES,
    ISO_MINOR_UNITS,
    CurrencyContext,
)
from pocketfinancer_sms.feedback import UserFeedbackEvent
from pocketfinancer_sms.persistence import processing_result_payload
from pocketfinancer_sms.profiles import PROFILES
from pocketfinancer_sms.selector import (
    SELECTOR_CONTRACT,
    SELECTOR_INPUT_CONTRACT,
    model_candidate_payload,
)
from pocketfinancer_sms.trace import ProcessingTrace, TraceStage
from pocketfinancer_sms.types import (
    AccountState,
    Analysis,
    CounterpartyState,
    CurrencyProvenance,
    Direction,
    EvidenceSpan,
    PersistenceDecision,
    ReconstructedTransaction,
    SelectorResult,
)


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_ROOT = ROOT / "configs" / "sms_processing" / "contracts"


def _schema(name: str) -> dict:
    return json.loads((SCHEMA_ROOT / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "name",
    [
        "canonical-label.schema.json",
        "corpus-record.schema.json",
        "grounded-candidate-selector.schema.json",
        "grounded-candidate-selector-input.schema.json",
        "processing-result.schema.json",
        "processing-trace.schema.json",
        "sms-analysis.schema.json",
        "user-feedback.schema.json",
    ],
)
def test_contract_schema_is_valid_draft_2020_12(name: str) -> None:
    jsonschema.Draft202012Validator.check_schema(_schema(name))


def test_executable_analysis_conforms_to_schema() -> None:
    analysis = DeterministicSmsAnalyzer(
        CurrencyContext("INR", ("core-en", "india"))
    ).analyze(
        "INR 12 was debited from account **1234 at SYNTH STORE.",
        operation_id="synthetic-contract",
        is_outgoing=False,
    )
    assert analysis.contract == ANALYSIS_CONTRACT
    assert _schema("sms-analysis.schema.json")["$id"] == ANALYSIS_CONTRACT
    jsonschema.validate(analysis.to_dict(), _schema("sms-analysis.schema.json"))
    assert Analysis.from_dict(analysis.to_dict(), source="INR 12 was debited from account **1234 at SYNTH STORE.") == analysis


def test_checked_in_currency_and_profile_declarations_match_runtime() -> None:
    currency_config = json.loads(
        (ROOT / "configs/sms_processing/currency/iso-4217.json").read_text(
            encoding="utf-8"
        )
    )
    assert {
        code: value["minor_units"]
        for code, value in currency_config["currencies"].items()
    } == ISO_MINOR_UNITS
    assert frozenset(currency_config["current_codes"]) == ISO_4217_CURRENT_CODES
    assert ISO_MINOR_UNITS.keys() <= ISO_4217_CURRENT_CODES
    for profile_id, runtime in PROFILES.items():
        declared = json.loads(
            (ROOT / f"configs/sms_processing/profiles/{profile_id}.json").read_text(
                encoding="utf-8"
            )
        )
        declared_markers = {
            currency: tuple(marker.casefold() for marker in markers)
            for currency, markers in declared["currency_markers"].items()
        }
        runtime_markers = {
            currency: tuple(marker.casefold() for marker in markers)
            for currency, markers in runtime.explicit_markers.items()
        }
        assert declared_markers == runtime_markers
        assert tuple(
            marker.casefold() for marker in declared["ambiguous_currency_markers"]
        ) == runtime.ambiguous_markers


def test_selector_schema_accepts_only_three_semantic_branches() -> None:
    schema = _schema("grounded-candidate-selector.schema.json")
    assert schema["$id"] == SELECTOR_CONTRACT
    jsonschema.validate({"decision": "none"}, schema)
    jsonschema.validate({"decision": "abstain"}, schema)
    jsonschema.validate(
        {
            "decision": "posted",
            "amount": "amt_0123456789ab",
            "direction": "dir_0123456789ab",
            "account": "acc_0123456789ab",
            "counterparty": "cp_0123456789ab",
        },
        schema,
    )
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"decision": "posted", "amount": "12.00"}, schema)


def test_selector_input_payload_conforms_without_host_canonical_values() -> None:
    source = "INR 12 was debited from account **1234 at SYNTH STORE."
    analysis = DeterministicSmsAnalyzer(
        CurrencyContext("INR", ("core-en", "india"))
    ).analyze(source, operation_id="synthetic-selector-input")
    payload = model_candidate_payload(source, analysis)
    schema = _schema("grounded-candidate-selector-input.schema.json")
    assert schema["$id"] == SELECTOR_INPUT_CONTRACT
    jsonschema.validate(payload, schema)
    rendered = json.dumps(payload)
    assert "minor_units" not in rendered
    assert "start_utf8" not in rendered


def test_trace_and_feedback_objects_conform_to_schemas() -> None:
    trace = ProcessingTrace.create(
        "synthetic-operation",
        "a" * 64,
        (TraceStage(0, "analysis", "completed", ("synthetic_reason",)),),
    )
    feedback = UserFeedbackEvent.create(
        event_id="synthetic-feedback",
        operation_id="synthetic-operation",
        trace_hash=trace.trace_hash,
        revision=1,
        action="confirm",
        canonical_label_id="synthetic-label",
        canonical_label_revision=1,
        created_at_epoch_ms=1_700_000_000_000,
        actor_id="synthetic-reviewer",
    )
    trace_value = json.loads(json.dumps(asdict(trace)))
    feedback_value = json.loads(json.dumps(asdict(feedback)))
    jsonschema.validate(trace_value, _schema("processing-trace.schema.json"))
    jsonschema.validate(feedback_value, _schema("user-feedback.schema.json"))


def test_processing_result_schema_keeps_recognition_and_persistence_separate() -> None:
    evidence = EvidenceSpan.from_source("INR debit", 0, 3)
    direction_evidence = EvidenceSpan.from_source("INR debit", 4, 9)
    transaction = ReconstructedTransaction(
        analysis_id="a" * 24,
        amount_candidate_id="amt_0123456789ab",
        direction_candidate_id="dir_0123456789ab",
        account_candidate_id="acc_0123456789ab",
        counterparty_candidate_id="cp_0123456789ab",
        minor_units=1200,
        currency="INR",
        currency_provenance=CurrencyProvenance.EXPLICIT_CODE,
        direction=Direction.DEBIT,
        account_state=AccountState.PRESENT,
        account_evidence=evidence,
        counterparty_state=CounterpartyState.ABSENT,
        counterparty_evidence=None,
        amount_evidence=evidence,
        direction_evidence=direction_evidence,
    )
    value = processing_result_payload(
        SelectorResult("posted", transaction),
        PersistenceDecision(
            False, ("persistence_account_not_uniquely_resolved",)
        ),
    )
    jsonschema.validate(value, _schema("processing-result.schema.json"))
