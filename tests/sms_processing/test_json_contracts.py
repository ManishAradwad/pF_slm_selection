"""Synthetic conformance checks binding executable objects to versioned schemas."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import jsonschema
import pytest

from pocketfinancer_sms.analyzer import (
    ANALYZER_BEHAVIOR_V2,
    ANALYSIS_CONTRACT,
    ANALYSIS_CONTRACT_V2,
    DeterministicSmsAnalyzer,
)
from pocketfinancer_sms.currency import (
    ISO_4217_CURRENT_CODES,
    ISO_MINOR_UNITS,
    CurrencyContext,
)
from pocketfinancer_sms.configuration import (
    AssetBinding,
    ProcessingConfigSnapshot,
    ProcessingTrigger,
    RolloutMode,
    SelectorRuntimeConfig,
)
from pocketfinancer_sms.feedback import UserFeedbackEvent
from pocketfinancer_sms.persistence import processing_result_payload
from pocketfinancer_sms.profiles import PROFILES
from pocketfinancer_sms.selector import (
    SELECTOR_CONTRACT,
    SELECTOR_INPUT_CONTRACT,
    SELECTOR_VALIDATION_PROFILE,
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
    TimestampProvenance,
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
        "processing-config.schema.json",
        "processing-trace.schema.json",
        "sms-analysis.schema.json",
        "user-feedback.schema.json",
        "v2/selector-validation-profile.schema.json",
        "v2/sms-analysis.schema.json",
    ],
)
def test_contract_schema_is_valid_draft_2020_12(name: str) -> None:
    jsonschema.Draft202012Validator.check_schema(_schema(name))


def test_executable_analysis_conforms_to_schema() -> None:
    source = "INR 12 was debited from account **1234 at SYNTH STORE."
    analysis = DeterministicSmsAnalyzer(CurrencyContext("INR", ("core-en", "india"))).analyze(
        source,
        operation_id="synthetic-contract",
        is_outgoing=False,
    )
    assert analysis.contract == ANALYSIS_CONTRACT
    assert _schema("sms-analysis.schema.json")["$id"] == ANALYSIS_CONTRACT
    jsonschema.validate(analysis.to_dict(), _schema("sms-analysis.schema.json"))
    assert Analysis.from_dict(analysis.to_dict(), source=source) == analysis


def test_executable_v2_analysis_conforms_to_versioned_schema() -> None:
    source = "INR 12 will be refunded to account **1234 by SYNTH STORE."
    analysis = DeterministicSmsAnalyzer(
        CurrencyContext("INR", ("core-en", "india")),
        analysis_contract=ANALYSIS_CONTRACT_V2,
    ).analyze(
        source,
        operation_id="synthetic-v2-contract",
        operation_config_hash="a" * 64,
        source_timestamp_epoch_ms=1_700_000_000_000,
        source_timestamp_provenance=(
            TimestampProvenance.ACQUISITION_SUPPLIED_MESSAGE_TIME
        ),
    )

    jsonschema.validate(analysis.to_dict(), _schema("v2/sms-analysis.schema.json"))
    assert analysis.contract == ANALYSIS_CONTRACT_V2
    assert Analysis.from_dict(analysis.to_dict(), source=source) == analysis


def test_checked_in_currency_and_profile_declarations_match_runtime() -> None:
    currency_config = json.loads(
        (ROOT / "configs/sms_processing/currency/iso-4217.json").read_text(encoding="utf-8")
    )
    assert {
        code: value["minor_units"] for code, value in currency_config["currencies"].items()
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
        assert declared["revision"] == runtime.revision
        assert declared_markers == runtime_markers
        assert (
            tuple(marker.casefold() for marker in declared["ambiguous_currency_markers"])
            == runtime.ambiguous_markers
        )
        assert tuple(declared["grouping"]) == runtime.grouping_styles
        assert tuple(declared["transaction_terms"]) == runtime.transaction_terms
        assert {rail: tuple(terms) for rail, terms in declared["rails"].items()} == runtime.rails


def _processing_config(*, primary_currency: str = "INR") -> ProcessingConfigSnapshot:
    return ProcessingConfigSnapshot(
        operation_id="11111111-1111-4111-8111-111111111111",
        parent_operation_id=None,
        source_ref_hash="1" * 64,
        trigger=ProcessingTrigger.APP_INTENT,
        created_at_epoch_ms=1_700_000_000_100,
        admission_epoch_ms=1_700_000_000_000,
        release_id="native-integration-v1",
        release_manifest_sha256="2" * 64,
        analyzer_behavior_version=ANALYZER_BEHAVIOR_V2,
        unicode_behavior_version="unicode-15.0-per-code-point-nfkc-casefold",
        currency_asset_sha256="3" * 64,
        profile_assets=(
            AssetBinding("core-en", "4" * 64),
            AssetBinding("india", "5" * 64),
        ),
        primary_currency=primary_currency,
        enabled_profile_ids=("core-en", "india"),
        source_timestamp_epoch_ms=1_699_999_999_000,
        source_timestamp_provenance=(
            TimestampProvenance.ACQUISITION_SUPPLIED_MESSAGE_TIME
        ),
        timezone_id="Asia/Kolkata",
        timestamp_policy_version="pocketfinancer.timestamp-policy/1",
        selector=SelectorRuntimeConfig(
            eligible=True,
            ineligibility_reason=None,
            model_identifier="synthetic-local-model",
            model_file_sha256=None,
            runtime_version="synthetic-runtime-1",
            os_version="synthetic-os-1",
            device_cohort="synthetic-device",
            prompt_version="pocketfinancer.selector-prompt/1",
            prompt_sha256="6" * 64,
        ),
        persistence_policy_version="pocketfinancer.persistence-policy/1",
        rollout_mode=RolloutMode.SHADOW,
    )


def test_processing_configuration_is_immutable_schema_bound_and_canonically_hashed() -> None:
    config = _processing_config()
    value = config.to_dict()

    jsonschema.validate(value, _schema("processing-config.schema.json"))
    assert value["config_hash"] == config.config_hash
    assert config.config_hash == _processing_config().config_hash
    assert config.config_hash != _processing_config(primary_currency="USD").config_hash
    assert value["selector"]["generation_mode"] == "DIRECT_NON_THINKING"
    assert value["persistence_policy"]["rollout_mode"] == "shadow"


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


def test_selector_validation_profile_v2_is_frozen_and_schema_valid() -> None:
    schema = _schema("v2/selector-validation-profile.schema.json")
    profile = _schema("v2/selector-validation-profile.json")

    jsonschema.validate(profile, schema)
    assert profile["contract"] == SELECTOR_VALIDATION_PROFILE
    assert profile["selector_output_contract"] == SELECTOR_CONTRACT
    assert profile["generation"] == {
        "mode": "DIRECT_NON_THINKING",
        "decoding": "greedy",
        "answer_token_limit": 512,
        "raw_output_utf8_byte_limit": 16_384,
        "parser_deadline_ms": 60_000,
    }
    assert profile["json_policy"]["reject_duplicate_keys"] is True
    assert profile["json_policy"]["reject_non_string_discriminator"] is True


def test_selector_input_payload_conforms_without_host_canonical_values() -> None:
    source = "INR 12 was debited from account **1234 at SYNTH STORE."
    analysis = DeterministicSmsAnalyzer(CurrencyContext("INR", ("core-en", "india"))).analyze(
        source, operation_id="synthetic-selector-input"
    )
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
        PersistenceDecision(False, ("persistence_account_not_uniquely_resolved",)),
    )
    jsonschema.validate(value, _schema("processing-result.schema.json"))
