"""Schema, freeze-manifest, and sanitized-vector checks for extractor contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import jsonschema
import pytest

from pocketfinancer_sms.account_resolution import AccountCatalogEntry
from pocketfinancer_sms.analyzer import ANALYSIS_CONTRACT_V2, DeterministicSmsAnalyzer
from pocketfinancer_sms.configuration import AssetBinding, ProcessingTrigger, RolloutMode
from pocketfinancer_sms.configuration_v3 import ExtractorRuntimeConfig, ProcessingConfigSnapshotV3
from pocketfinancer_sms.currency import CurrencyContext
from pocketfinancer_sms.extractor import (
    ExtractionValidationError,
    SourceSpan,
    build_extractor_input,
    parse_and_normalize_extraction,
)
from pocketfinancer_sms.feedback import (
    FieldRevisionProvenance,
    FieldRevisionV3,
    UserFeedbackEventV3,
)
from pocketfinancer_sms.processing_v3 import (
    ExtractionCoordinator,
    ReviewCase,
    ReviewUserRevision,
    processing_result_payload_v3,
)
from pocketfinancer_sms.provenance import file_sha256
from pocketfinancer_sms.trace import ProcessingTraceV3, TraceEventV3
from pocketfinancer_sms.types import TimestampProvenance
from scripts.build_native_contract_release_v3 import (
    build_manifest,
    build_sanitized_vectors,
    validate_freeze_inputs,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "configs" / "sms_processing" / "contracts"
V3 = CONTRACTS / "v3"
SHA = "a" * 64
OPERATION_ID = "11111111-1111-4111-8111-111111111111"
RECEIVED_AT = 1_700_000_000_000


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _schema(name: str) -> dict:
    return _json(V3 / name)


def _analyzer() -> DeterministicSmsAnalyzer:
    return DeterministicSmsAnalyzer(
        CurrencyContext("INR", ("core-en", "india")),
        analysis_contract=ANALYSIS_CONTRACT_V2,
    )


def test_every_new_schema_is_valid_draft_2020_12() -> None:
    schemas = (
        "sms-extractor-input.schema.json",
        "sms-extractor.schema.json",
        "extractor-validation-profile.schema.json",
        "processing-config.schema.json",
        "processing-result.schema.json",
        "processing-trace.schema.json",
        "reason-code-registry.schema.json",
        "account-resolution-profile.schema.json",
        "review-case.schema.json",
        "user-feedback.schema.json",
        "canonical-label.schema.json",
    )
    for name in schemas:
        jsonschema.Draft202012Validator.check_schema(_schema(name))
    jsonschema.Draft202012Validator.check_schema(
        _json(CONTRACTS / "releases" / "v3" / "release-manifest.schema.json")
    )


def test_profiles_and_reason_registry_are_schema_valid_and_codes_unique() -> None:
    jsonschema.validate(
        _json(V3 / "extractor-validation-profile.json"),
        _schema("extractor-validation-profile.schema.json"),
    )
    jsonschema.validate(
        _json(V3 / "account-resolution-profile.json"),
        _schema("account-resolution-profile.schema.json"),
    )
    registry = _json(V3 / "reason-code-registry.json")
    jsonschema.validate(registry, _schema("reason-code-registry.schema.json"))
    codes = [item["code"] for namespace in registry["namespaces"] for item in namespace["codes"]]
    assert len(codes) == len(set(codes))
    legacy_registry = _json(V3.parent / "v2" / "reason-code-registry.json")
    legacy_by_code = {
        item["code"]: item
        for namespace in legacy_registry["namespaces"]
        for item in namespace["codes"]
    }
    current_by_code = {
        item["code"]: item for namespace in registry["namespaces"] for item in namespace["codes"]
    }
    assert legacy_by_code.keys() <= current_by_code.keys()
    frozen_fields = ("meaning", "severity", "safety_relevant", "allowed_outcomes")
    for code, legacy_item in legacy_by_code.items():
        assert {field: current_by_code[code][field] for field in frozen_fields} == {
            field: legacy_item[field] for field in frozen_fields
        }
    required = {
        "extractor_malformed_json",
        "extractor_duplicate_json_key",
        "extractor_extra_content",
        "extractor_missing_amount",
        "extractor_missing_direction",
        "extractor_missing_account",
        "extractor_evidence_out_of_bounds",
        "extractor_evidence_mismatch",
        "extractor_amount_value_disagreement",
        "extractor_abstained",
        "runtime_unavailable",
        "runtime_failure",
        "runtime_output_truncated",
        "user_cancelled",
        "process_interrupted",
        "account_resolution_unresolved",
        "account_resolution_ambiguous",
        "duplicate_possible",
        "duplicate_already_persisted",
        "persistence_configuration_hash_mismatch",
        "persistence_claim_ownership_invalid",
        "persistence_receipt_timestamp_provenance_invalid",
        "persistence_blocked_by_rollout_mode",
        "extractor_none",
        "extractor_posted_valid",
        "review_retained",
        "persistence_blocked",
    }
    assert required <= set(codes)


def test_sanitized_goldens_are_grounded_and_parse_under_the_frozen_output_schema() -> None:
    validate_freeze_inputs()
    golden = _json(
        ROOT / "tests" / "sms_processing" / "golden" / "extractor-v1" / "sanitized-vectors.json"
    )
    schema = _schema("sms-extractor.schema.json")
    assert golden == build_sanitized_vectors()
    for case in golden["cases"]:
        jsonschema.validate(case["model_output"], schema)
        encoded = json.dumps(case["model_output"], ensure_ascii=False)
        if "expected_reason" in case:
            with pytest.raises(ExtractionValidationError, match=case["expected_reason"]):
                parse_and_normalize_extraction(
                    encoded,
                    case["sms_body"],
                    primary_currency="INR",
                    enabled_profile_ids=("core-en", "india"),
                )
            continue
        result = parse_and_normalize_extraction(
            encoded,
            case["sms_body"],
            primary_currency="INR",
            enabled_profile_ids=("core-en", "india"),
        )
        assert result.decision == case["expected"]["decision"]
        if result.transaction is not None:
            assert result.transaction.minor_units == case["expected"]["amount_minor_units"]
            assert result.transaction.account_reference == case["expected"]["account_reference"]


def test_executable_input_config_result_review_feedback_and_trace_match_schemas() -> None:
    source = "INR 42.50 was credited to account XX7788."
    analysis = _analyzer().analyze(
        source,
        operation_id=OPERATION_ID,
        operation_config_hash=SHA,
        source_timestamp_epoch_ms=RECEIVED_AT,
        source_timestamp_provenance=TimestampProvenance.PLATFORM_RECEIVED,
    )
    request = build_extractor_input(
        source,
        analysis,
        sender_family="synthetic-family",
        primary_currency="INR",
        enabled_profile_ids=("core-en", "india"),
    )
    jsonschema.validate(request, _schema("sms-extractor-input.schema.json"))

    config = ProcessingConfigSnapshotV3(
        operation_id=OPERATION_ID,
        parent_operation_id=None,
        source_ref_hash="1" * 64,
        trigger=ProcessingTrigger.REALTIME,
        created_at_epoch_ms=RECEIVED_AT + 100,
        admission_epoch_ms=RECEIVED_AT,
        release_id="native-integration-v3",
        release_manifest_sha256="2" * 64,
        analyzer_behavior_version="pocketfinancer.structural-sms-analyzer/2",
        unicode_behavior_version="unicode-scalar-nfkc-casefold",
        currency_asset_sha256="3" * 64,
        profile_assets=(AssetBinding("core-en", "4" * 64), AssetBinding("india", "5" * 64)),
        primary_currency="INR",
        enabled_profile_ids=("core-en", "india"),
        received_at_epoch_ms=RECEIVED_AT,
        received_timestamp_provenance=TimestampProvenance.PLATFORM_RECEIVED,
        timezone_id="Asia/Kolkata",
        extractor=ExtractorRuntimeConfig(
            eligible=True,
            ineligibility_reason=None,
            model_identifier="synthetic-local-model",
            model_file_sha256="9" * 64,
            runtime_version="synthetic-runtime",
            os_version="synthetic-os",
            device_cohort="synthetic-device",
            prompt_sha256="6" * 64,
            grammar_sha256="7" * 64,
            validation_profile_sha256="8" * 64,
        ),
        persistence_policy_version="pocketfinancer.persistence-policy/2",
        rollout_mode=RolloutMode.SHADOW,
    )
    jsonschema.validate(config.to_dict(), _schema("processing-config.schema.json"))

    coordinator = ExtractionCoordinator(
        _analyzer(),
        lambda _request, _token: '{"decision":"none"}',
        primary_currency="INR",
        enabled_profile_ids=("core-en", "india"),
        rollout_mode="shadow",
    )
    outcome = coordinator.process(
        source="Synthetic service notice.",
        raw_sender="AD-SYNTH",
        operation_id=OPERATION_ID,
        operation_config_hash=SHA,
        received_at_epoch_ms=RECEIVED_AT,
        received_timestamp_provenance=TimestampProvenance.PLATFORM_RECEIVED,
        source_event_key="synthetic-event",
        idempotency_key="synthetic-idempotency",
    )
    jsonschema.validate(
        processing_result_payload_v3(outcome), _schema("processing-result.schema.json")
    )
    posted_output = {
        "decision": "posted",
        "amount": {
            "value": "42.50",
            "currency": "INR",
            "evidence": {
                "start_scalar": 0,
                "end_scalar": 9,
                "text": "INR 42.50",
            },
        },
        "direction": {
            "value": "credit",
            "evidence": {
                "start_scalar": source.index("credited"),
                "end_scalar": source.index("credited") + len("credited"),
                "text": "credited",
            },
        },
        "account": {
            "reference": "XX7788",
            "evidence": {
                "start_scalar": source.index("XX7788"),
                "end_scalar": source.index("XX7788") + len("XX7788"),
                "text": "XX7788",
            },
        },
        "counterparty": None,
    }
    posted_coordinator = ExtractionCoordinator(
        _analyzer(),
        lambda _request, _token: json.dumps(posted_output),
        primary_currency="INR",
        enabled_profile_ids=("core-en", "india"),
        account_catalog=(AccountCatalogEntry("opaque-account-1", "bank_account", ("XX7788",)),),
        rollout_mode="automatic",
    )
    posted_outcome = posted_coordinator.process(
        source=source,
        raw_sender="AD-SYNTH",
        operation_id=OPERATION_ID,
        operation_config_hash=SHA,
        received_at_epoch_ms=RECEIVED_AT,
        received_timestamp_provenance=TimestampProvenance.PLATFORM_RECEIVED,
        source_event_key="synthetic-posted-event",
        idempotency_key="synthetic-posted-idempotency",
    )
    posted_payload = processing_result_payload_v3(posted_outcome)
    assert posted_payload["status"] == "eligible"
    assert posted_payload["account_resolution"]["status"] == "uniquely_resolved"
    assert posted_payload["duplicate_assessment"]["status"] == "clear"
    assert posted_payload["automatic_persistence"]["result"] == "eligible"
    jsonschema.validate(posted_payload, _schema("processing-result.schema.json"))

    review = ReviewCase(
        review_case_id="synthetic-review",
        operation_id_hash=hashlib.sha256(OPERATION_ID.encode()).hexdigest(),
        raw_sender="AD-SYNTH",
        source=source,
        received_at_epoch_ms=RECEIVED_AT,
        primary_reason="extractor_abstained",
        reason_codes=("extractor_abstained",),
        furthest_stage="extractor_validation",
        analyzer_suggestions=(
            {
                "kind": "amount",
                "span": {
                    "start_scalar": 0,
                    "end_scalar": 9,
                    "text": "INR 42.50",
                },
                "clause": source,
                "suggested_interpretation": {"minor_units": 4250},
                "provenance": {
                    "candidate_id": "amt_synthetic",
                    "analyzer_kind": "candidate",
                },
                "analyzer_version": "pocketfinancer.sms-analysis/2",
            },
            {
                "kind": "credential",
                "span": None,
                "clause": None,
                "suggested_interpretation": {"present": False},
                "provenance": {
                    "cue_id": "cue_synthetic",
                    "analyzer_kind": "cue",
                },
                "analyzer_version": "pocketfinancer.sms-analysis/2",
            },
        ),
        extractor_suggestion=None,
        account_resolution=None,
        user_revisions=(
            ReviewUserRevision(
                revision=1,
                action="correct",
                actor_id_hash="c" * 64,
                field_revisions=(
                    FieldRevisionV3(
                        "direction",
                        "credit",
                        FieldRevisionProvenance.USER_DIRECTION_CONTROL,
                    ),
                ),
                created_at_epoch_ms=RECEIVED_AT,
            ),
        ),
        revision=2,
        previous_revision_hash="e" * 64,
    )
    jsonschema.validate(
        json.loads(json.dumps(review.to_dict())),
        _schema("review-case.schema.json"),
    )

    span = SourceSpan.from_source(source, source.index("credited"), source.index("credited") + 8)
    revision = FieldRevisionV3(
        "direction", "credit", FieldRevisionProvenance.USER_DIRECTION_CONTROL
    )
    feedback = UserFeedbackEventV3.create(
        action_id="22222222-2222-4222-8222-222222222222",
        operation_id=OPERATION_ID,
        review_case_id="synthetic-review",
        expected_review_revision=0,
        resulting_review_revision=1,
        action="correct",
        actor_id="synthetic-user",
        field_revisions=(revision,),
        created_at_epoch_ms=RECEIVED_AT,
    )
    assert span.text == "credited"
    jsonschema.validate(feedback.to_dict(), _schema("user-feedback.schema.json"))

    event = TraceEventV3(
        sequence=0,
        event_id="33333333-3333-4333-8333-333333333333",
        occurred_at_epoch_ms=RECEIVED_AT,
        stage="analysis_advisory",
        status="completed",
    )
    trace = ProcessingTraceV3.create(
        operation_id=OPERATION_ID,
        config_hash=SHA,
        owner_generation=1,
        events=(event,),
    )
    jsonschema.validate(
        json.loads(json.dumps(asdict(trace))),
        _schema("processing-trace.schema.json"),
    )


def test_canonical_label_v2_retains_taxonomy_and_requires_grounded_posted_fields() -> None:
    source = "INR 42.50 was credited to account XX7788."

    def span(text: str) -> dict:
        start = source.index(text)
        return {"start_scalar": start, "end_scalar": start + len(text), "text": text}

    payload = {
        "contract": "pocketfinancer.canonical-label/2",
        "label_id": "synthetic-label",
        "source_id": "synthetic-source",
        "revision": 1,
        "status": "submitted",
        "decision": "posted",
        "operational_class": "posted_candidate",
        "event_state": "posted",
        "financial_family": "bank_transfer",
        "payment_rail": "bank_internal",
        "event": {
            "amount_value": "42.50",
            "currency": "INR",
            "amount_span": span("INR 42.50"),
            "direction": "credit",
            "direction_span": span("credited"),
            "account_reference": "XX7788",
            "account_span": span("XX7788"),
            "existing_account_id": "opaque-account-1",
            "counterparty": None,
            "counterparty_span": None,
        },
        "uncertain": False,
        "notes": "",
        "reviewer_id": "synthetic-reviewer",
        "created_at_epoch_ms": RECEIVED_AT,
        "supersedes_revision": None,
    }
    schema = _schema("canonical-label.schema.json")
    jsonschema.validate(payload, schema)
    del payload["event"]["direction_span"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, schema)


def test_manifest_rebuilds_exactly_and_preserves_every_legacy_hash() -> None:
    stored = _json(CONTRACTS / "releases" / "native-integration-v3.json")
    assert stored == build_manifest()
    assert stored["automatic_persistence_enabled"] is False
    jsonschema.validate(
        stored,
        _json(CONTRACTS / "releases" / "v3" / "release-manifest.schema.json"),
    )
    for artifact in stored["artifacts"]:
        assert file_sha256(ROOT / artifact["path"]) == artifact["sha256"]

    legacy = _json(CONTRACTS / "releases" / "native-integration-v2.json")
    for artifact in legacy["artifacts"]:
        assert file_sha256(ROOT / artifact["path"]) == artifact["sha256"]


def test_output_contract_rejects_time_fields_and_utf_offsets() -> None:
    schema = _schema("sms-extractor.schema.json")
    payload = {
        "decision": "posted",
        "amount": {
            "value": "1.00",
            "currency": "INR",
            "evidence": {"start_scalar": 0, "end_scalar": 8, "text": "INR 1.00"},
        },
        "direction": {
            "value": "debit",
            "evidence": {"start_scalar": 9, "end_scalar": 16, "text": "debited"},
        },
        "account": {
            "reference": "XX1234",
            "evidence": {"start_scalar": 22, "end_scalar": 28, "text": "XX1234"},
        },
        "counterparty": None,
        "message_time": 0,
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, schema)
    del payload["message_time"]
    payload["account"]["evidence"]["start_utf16"] = 22
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, schema)


def test_actual_coordinator_review_includes_candidate_and_cue_advice() -> None:
    source = "OTP 123456. INR 42.50 credited to account XX7788."
    coordinator = ExtractionCoordinator(
        _analyzer(),
        lambda _request, _token: '{"decision":"abstain"}',
        primary_currency="INR",
        enabled_profile_ids=("core-en", "india"),
        rollout_mode="shadow",
    )
    outcome = coordinator.process(
        source=source,
        raw_sender="AD-SYNTH",
        operation_id=OPERATION_ID,
        operation_config_hash=SHA,
        received_at_epoch_ms=RECEIVED_AT,
        received_timestamp_provenance=(TimestampProvenance.ACQUISITION_SUPPLIED_MESSAGE_TIME),
        source_event_key="synthetic-review-event",
        idempotency_key="synthetic-review-idempotency",
    )
    assert outcome.review_case is not None
    payload = json.loads(json.dumps(outcome.review_case.to_dict()))
    advice_kinds = {item["provenance"]["analyzer_kind"] for item in payload["analyzer_suggestions"]}
    assert {"candidate", "cue"} <= advice_kinds
    assert payload["receipt_timestamp"]["provenance"] == "acquisition_supplied_message_time"
    jsonschema.validate(payload, _schema("review-case.schema.json"))
