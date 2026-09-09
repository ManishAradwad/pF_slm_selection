"""Synthetic conformance checks binding executable objects to versioned schemas."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import jsonschema
import pytest
from scripts.build_native_contract_release import build_manifest
from scripts.build_native_sms_golden import build_fixture

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
from pocketfinancer_sms.feedback import (
    FieldCorrectionV2,
    FieldGroundingClassification,
    UserFeedbackEvent,
    UserFeedbackEventV2,
)
from pocketfinancer_sms.persistence import (
    evaluate_persistence_v2,
    processing_result_payload,
    processing_result_payload_v2,
)
from pocketfinancer_sms.profiles import PROFILES
from pocketfinancer_sms.selector import (
    SELECTOR_CONTRACT,
    SELECTOR_INPUT_CONTRACT,
    SELECTOR_VALIDATION_PROFILE,
    model_candidate_payload,
    parse_and_reconstruct,
)
from pocketfinancer_sms.trace import (
    ProcessingTrace,
    ProcessingTraceV2,
    TraceEventV2,
    TraceStage,
)
from pocketfinancer_sms.triage import evaluate_triage
from pocketfinancer_sms.types import (
    AccountResolution,
    AccountResolutionStatus,
    AccountState,
    Analysis,
    CandidateKind,
    CounterpartyState,
    CurrencyProvenance,
    Direction,
    EvidenceSpan,
    PersistenceDecision,
    PersistenceContextV2,
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
        "native-trace-bundle.schema.json",
        "grounded-candidate-selector.schema.json",
        "grounded-candidate-selector-input.schema.json",
        "processing-result.schema.json",
        "processing-config.schema.json",
        "processing-trace.schema.json",
        "sms-analysis.schema.json",
        "user-feedback.schema.json",
        "v2/selector-validation-profile.schema.json",
        "v2/sms-analysis.schema.json",
        "v2/processing-result.schema.json",
        "v2/processing-trace.schema.json",
        "v2/user-feedback.schema.json",
        "v2/reason-code-registry.schema.json",
        "releases/release-manifest.schema.json",
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


def test_corpus_record_schema_is_self_contained() -> None:
    source = "INR 12 was debited from account **1234 at SYNTH STORE."
    analysis = DeterministicSmsAnalyzer(CurrencyContext("INR", ("core-en", "india"))).analyze(
        source,
        operation_id="synthetic-corpus-contract",
        is_outgoing=False,
    )
    corpus_schema = _schema("corpus-record.schema.json")
    analysis_schema = _schema("sms-analysis.schema.json")
    embedded_analysis = {
        **corpus_schema["$defs"]["analysis"],
        "$defs": {
            name: corpus_schema["$defs"][name]
            for name in ("evidence", "clause", "candidate", "cue")
        },
    }
    canonical_analysis = {
        key: value
        for key, value in analysis_schema.items()
        if key not in {"$schema", "$id", "title"}
    }
    assert embedded_analysis == canonical_analysis

    jsonschema.validate(
        {
            "contract": "pocketfinancer.corpus-record/1",
            "source_id": "src_" + "a" * 32,
            "source": {"body": source, "sender": "SYNTH-BANK"},
            "source_metadata": {},
            "analysis": analysis.to_dict(),
            "weak_facets": {},
            "grouping": {},
            "pool": "annotation_development",
            "review_state": "unreviewed",
            "provenance": {},
        },
        corpus_schema,
    )


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


def test_reason_registry_is_unique_complete_for_frozen_profiles_and_fails_closed() -> None:
    schema = _schema("v2/reason-code-registry.schema.json")
    registry = _schema("v2/reason-code-registry.json")
    jsonschema.validate(registry, schema)
    entries = [entry for namespace in registry["namespaces"] for entry in namespace["codes"]]
    codes = [entry["code"] for entry in entries]

    assert len(codes) == len(set(codes))
    assert registry["unknown_safety_code_policy"] == "retain_review"
    selector_profile = _schema("v2/selector-validation-profile.json")
    assert set(selector_profile["reason_codes"]) <= set(codes)
    financial_vectors = json.loads(
        (ROOT / "tests/sms_processing/golden/native-v1/financial-state.json").read_text(
            encoding="utf-8"
        )
    )
    assert {vector["reason_code"] for vector in financial_vectors["vectors"]} <= set(codes)
    assert {
        "expected_refund_not_posted",
        "authorization_or_hold_not_posted",
        "runtime_mode_violation",
        "candidate_coverage_missing",
        "account_resolution_ambiguous",
        "persistence_timestamp_provenance_invalid",
        "persistence_blocked_by_rollout_mode",
    } <= set(codes)


def test_native_parity_bundle_matches_canonical_byte_level_outputs() -> None:
    stored = json.loads(
        (ROOT / "tests/sms_processing/golden/native-v1/parity-bundle.json").read_text(
            encoding="utf-8"
        )
    )
    assert stored == build_fixture()

    repeated = {
        vector["id"]: json.loads(vector["expected_analysis_json"])
        for vector in stored["vectors"]
        if vector["id"].startswith("repeated-amounts")
    }
    first_ids = [
        item["candidate_id"] for item in repeated["repeated-amounts-first-operation"]["candidates"]
    ]
    second_ids = [
        item["candidate_id"]
        for item in repeated["repeated-amounts-second-operation"]["candidates"]
    ]
    assert first_ids != second_ids


def test_native_trace_bundle_requires_encryption_integrity_and_explicit_consent() -> None:
    schema = _schema("native-trace-bundle.schema.json")
    bundle = {
        "contract": "pocketfinancer.native-trace-bundle/1",
        "transfer_id": "11111111-1111-4111-8111-111111111111",
        "created_at_epoch_ms": 1_700_000_000_000,
        "source_platform": "ios",
        "release_id": "native-integration-v1",
        "release_manifest_sha256": "a" * 64,
        "explicit_consent": True,
        "purpose": "local_workbench_inspection",
        "encryption": {
            "algorithm": "AES-256-GCM",
            "key_protection": "os_protected",
            "nonce_base64": "MTIzNDU2Nzg5MDEy",
        },
        "source_ref_hashes": ["b" * 64],
        "record_count": 1,
        "payload_ciphertext_base64": "c3ludGhldGljLWNpcGhlcnRleHQ=",
        "payload_ciphertext_sha256": "c" * 64,
    }

    jsonschema.validate(bundle, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**bundle, "explicit_consent": False}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**bundle, "payload_ciphertext_base64": "raw plaintext"}, schema)


def test_native_release_manifest_hashes_every_frozen_artifact() -> None:
    stored = _schema("releases/native-integration-v1.json")
    schema = _schema("releases/release-manifest.schema.json")

    jsonschema.validate(stored, schema)
    assert stored == build_manifest()
    paths = [artifact["path"] for artifact in stored["artifacts"]]
    assert len(paths) == len(set(paths))
    assert stored["automatic_persistence_enabled"] is False
    assert {
        "configs/sms_processing/contracts/v2/sms-analysis.schema.json",
        "configs/sms_processing/contracts/v2/processing-result.schema.json",
        "configs/sms_processing/contracts/v2/processing-trace.schema.json",
        "configs/sms_processing/contracts/v2/user-feedback.schema.json",
        "tests/sms_processing/golden/native-v1/parity-bundle.json",
    } <= set(paths)


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


def test_v2_trace_and_feedback_are_append_only_and_schema_bound() -> None:
    first = TraceEventV2(
        sequence=0,
        event_id="11111111-1111-4111-8111-111111111111",
        occurred_at_epoch_ms=1_700_000_000_000,
        stage="admission",
        status="completed",
        reason_codes=("source_admitted",),
    )
    second = TraceEventV2(
        sequence=1,
        event_id="22222222-2222-4222-8222-222222222222",
        occurred_at_epoch_ms=1_700_000_000_100,
        stage="analysis",
        status="completed",
        reason_codes=("completed_direction_candidate_present",),
        previous_event_hash=first.event_hash,
    )
    trace = ProcessingTraceV2.create(
        "33333333-3333-4333-8333-333333333333",
        "a" * 64,
        1,
        (first, second),
    )
    correction = FieldCorrectionV2(
        field="amount",
        classification=(
            FieldGroundingClassification.SUPPLIED_SOURCE_SUPPORTED_CANDIDATE_MISS
        ),
        previous_revision_id="synthetic-revision-1",
        candidate_id=None,
        evidence=EvidenceSpan.from_source("INR 12", 0, 6),
        new_value={"minor_units": "1200", "currency": "INR"},
    )
    feedback = UserFeedbackEventV2.create(
        action_id="44444444-4444-4444-8444-444444444444",
        operation_id="33333333-3333-4333-8333-333333333333",
        review_case_id="synthetic-review-case",
        transaction_revision_id="synthetic-revision-2",
        expected_review_revision=0,
        resulting_review_revision=1,
        action="correct",
        actor_class="user",
        actor_id="synthetic-user",
        field_corrections=(correction,),
        created_at_epoch_ms=1_700_000_000_200,
    )

    trace_value = json.loads(json.dumps(asdict(trace)))
    feedback_value = feedback.to_dict()
    jsonschema.validate(trace_value, _schema("v2/processing-trace.schema.json"))
    jsonschema.validate(feedback_value, _schema("v2/user-feedback.schema.json"))
    assert len(trace.trace_hash) == 64
    assert len(feedback.event_hash) == 64

    broken_second = TraceEventV2(
        sequence=1,
        event_id="55555555-5555-4555-8555-555555555555",
        occurred_at_epoch_ms=1_700_000_000_300,
        stage="triage",
        status="completed",
        previous_event_hash="b" * 64,
    )
    with pytest.raises(ValueError, match="hash chain"):
        ProcessingTraceV2.create(
            "33333333-3333-4333-8333-333333333333",
            "a" * 64,
            1,
            (first, broken_second),
        )


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


def test_processing_result_v2_exposes_typed_gate_and_provenance() -> None:
    source = "INR 12 was paid from account **1234 at SYNTH STORE."
    analysis = DeterministicSmsAnalyzer(
        CurrencyContext("INR", ("core-en", "india")),
        analysis_contract=ANALYSIS_CONTRACT_V2,
    ).analyze(
        source,
        operation_id="synthetic-v2-result",
        operation_config_hash="a" * 64,
        source_timestamp_epoch_ms=1_700_000_000_000,
        source_timestamp_provenance=(
            TimestampProvenance.ACQUISITION_SUPPLIED_MESSAGE_TIME
        ),
    )
    candidates = {
        kind: analysis.candidates_of(kind)[0]
        for kind in (CandidateKind.AMOUNT, CandidateKind.DIRECTION)
    }
    account = next(
        item
        for item in analysis.candidates_of(CandidateKind.ACCOUNT)
        if not item.explicit_absence
    )
    counterparty = next(
        item
        for item in analysis.candidates_of(CandidateKind.COUNTERPARTY)
        if not item.explicit_absence
    )
    raw = json.dumps(
        {
            "decision": "posted",
            "amount": candidates[CandidateKind.AMOUNT].candidate_id,
            "direction": candidates[CandidateKind.DIRECTION].candidate_id,
            "account": account.candidate_id,
            "counterparty": counterparty.candidate_id,
        }
    )
    selector_result = parse_and_reconstruct(raw, analysis)
    context = PersistenceContextV2(
        timestamp_epoch_ms=1_700_000_000_000,
        timestamp_provenance=TimestampProvenance.ACQUISITION_SUPPLIED_MESSAGE_TIME,
        approved_timestamp_provenance=frozenset(
            {TimestampProvenance.ACQUISITION_SUPPLIED_MESSAGE_TIME}
        ),
        account_resolution=AccountResolution(
            AccountResolutionStatus.UNIQUELY_RESOLVED,
            1,
            "b" * 64,
            "c" * 64,
            "confirmed_owned_alias",
        ),
        approved_currency_provenance=frozenset(CurrencyProvenance),
        financial_family="merchant_payment",
        supported_automatic_families=frozenset({"merchant_payment"}),
        rollout_mode="shadow",
        selector_mode_valid=True,
        claim_ownership_current=True,
        configuration_hash_matches=True,
    )
    decision = evaluate_persistence_v2(
        selector_result,
        analysis,
        evaluate_triage(analysis),
        context,
    )
    value = processing_result_payload_v2(selector_result, decision, context)

    jsonschema.validate(value, _schema("v2/processing-result.schema.json"))
    assert decision.result.value == "blocked_by_mode"
    assert decision.safe_to_persist is False
    assert value["semantic_result"]["money"] == {
        "minor_units": 1200,
        "currency": "INR",
        "scale": 2,
        "provenance": "explicit_code",
    }


def test_persistence_v2_fails_integrity_closed_before_policy() -> None:
    analysis = DeterministicSmsAnalyzer(CurrencyContext("INR", ("core-en", "india"))).analyze(
        "INR 12 was paid from account **1234 at SYNTH STORE.",
        operation_id="synthetic-v1-invalid-result",
    )
    context = PersistenceContextV2(
        timestamp_epoch_ms=None,
        timestamp_provenance=TimestampProvenance.UNKNOWN,
        approved_timestamp_provenance=frozenset(),
        account_resolution=AccountResolution(AccountResolutionStatus.UNRESOLVED, 0),
        approved_currency_provenance=frozenset(),
        financial_family=None,
        supported_automatic_families=frozenset(),
        rollout_mode="review_only",
        selector_mode_valid=False,
        claim_ownership_current=False,
        configuration_hash_matches=False,
    )
    decision = evaluate_persistence_v2(
        SelectorResult("abstain"),
        analysis,
        evaluate_triage(analysis),
        context,
    )

    assert decision.result.value == "invalid_operation"
    assert decision.primary_reason == "persistence_unknown_analysis_contract"
