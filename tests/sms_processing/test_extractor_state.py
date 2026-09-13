"""Versioned state and revision-chain tests for direct extraction."""

from __future__ import annotations

from dataclasses import replace

import pytest

from pocketfinancer_sms.configuration import AssetBinding, ProcessingTrigger, RolloutMode
from pocketfinancer_sms.configuration_v3 import (
    ExtractorRuntimeConfig,
    ProcessingConfigSnapshotV3,
)
from pocketfinancer_sms.extractor import SourceSpan
from pocketfinancer_sms.feedback import (
    FieldRevisionProvenance,
    FieldRevisionV3,
    UserFeedbackEventV3,
)
from pocketfinancer_sms.labels import (
    CanonicalLabelV2,
    EventState,
    ExtractorCanonicalEvent,
    LabelValidationError,
    OperationalClass,
    ReviewStatus,
    project_extractor_target,
)
from pocketfinancer_sms.processing_v3 import ReviewCase, ReviewUserRevision
from pocketfinancer_sms.trace import ProcessingTraceV3, TraceEventV3
from pocketfinancer_sms.types import Direction, TimestampProvenance


def _source_and_event() -> tuple[str, ExtractorCanonicalEvent]:
    source = "INR 42.50 was credited to account XX7788 from FRIEND."

    def span(text: str) -> SourceSpan:
        start = source.index(text)
        return SourceSpan.from_source(source, start, start + len(text))

    return source, ExtractorCanonicalEvent(
        amount_value="42.50",
        currency="INR",
        amount_span=span("INR 42.50"),
        direction=Direction.CREDIT,
        direction_span=span("credited"),
        account_reference="XX7788",
        account_span=span("XX7788"),
        existing_account_id="account-opaque-1",
        counterparty="FRIEND",
        counterparty_span=span("FRIEND"),
    )


def _config() -> ProcessingConfigSnapshotV3:
    return ProcessingConfigSnapshotV3(
        operation_id="11111111-1111-4111-8111-111111111111",
        parent_operation_id=None,
        source_ref_hash="1" * 64,
        trigger=ProcessingTrigger.REALTIME,
        created_at_epoch_ms=1_700_000_000_100,
        admission_epoch_ms=1_700_000_000_000,
        release_id="native-integration-v3",
        release_manifest_sha256="2" * 64,
        analyzer_behavior_version="pocketfinancer.structural-sms-analyzer/2",
        unicode_behavior_version="unicode-scalar-nfkc-casefold",
        currency_asset_sha256="3" * 64,
        profile_assets=(
            AssetBinding("core-en", "4" * 64),
            AssetBinding("india", "5" * 64),
        ),
        primary_currency="INR",
        enabled_profile_ids=("core-en", "india"),
        received_at_epoch_ms=1_699_999_999_000,
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


def test_processing_config_v3_is_immutable_hashed_and_timestamp_read_only() -> None:
    config = _config()
    payload = config.to_dict()
    assert payload["contract"] == "pocketfinancer.processing-config/3"
    assert payload["extractor"]["parser_deadline_ms"] == 0
    assert payload["received_timestamp"] == {
        "epoch_ms": 1_699_999_999_000,
        "provenance": "platform_received",
        "timezone_id": "Asia/Kolkata",
        "read_only": True,
    }
    assert payload["config_hash"] == config.config_hash
    assert replace(config, primary_currency="USD").config_hash != config.config_hash
    with pytest.raises(ValueError, match="release provenance"):
        replace(config, release_id="native-integration-v2")
    with pytest.raises(ValueError, match="analyzer provenance"):
        replace(config, analyzer_behavior_version="pocketfinancer.structural-sms-analyzer/1")


def test_canonical_label_v2_projects_direct_json_without_analyzer_candidates() -> None:
    source, event = _source_and_event()
    label = CanonicalLabelV2(
        label_id="synthetic-label",
        source_id="synthetic-source",
        revision=1,
        status=ReviewStatus.SUBMITTED,
        decision="posted",
        operational_class=OperationalClass.POSTED_CANDIDATE,
        event_state=EventState.POSTED,
        financial_family="bank_transfer",
        payment_rail="bank_internal",
        event=event,
        uncertain=False,
        notes="synthetic",
        reviewer_id="synthetic-reviewer",
        created_at_epoch_ms=1_700_000_000_000,
    )
    target = project_extractor_target(label, source)
    assert target["decision"] == "posted"
    assert target["amount"]["value"] == "42.50"
    assert target["account"]["evidence"]["text"] == "XX7788"
    assert "candidate" not in repr(target)

    invalid = replace(label, event=replace(event, account_span=event.direction_span))
    with pytest.raises(LabelValidationError, match="label_extractor_event_invalid"):
        project_extractor_target(invalid, source)


def test_review_case_records_read_only_receipt_and_revision_hashes() -> None:
    review = ReviewCase(
        review_case_id="review-synthetic",
        operation_id_hash="a" * 64,
        raw_sender="SYNTH-BANK",
        source="Synthetic body",
        received_at_epoch_ms=1_700_000_000_000,
        primary_reason="extractor_abstained",
        reason_codes=("extractor_abstained",),
        furthest_stage="extractor_validation",
        analyzer_suggestions=(),
        extractor_suggestion=None,
        account_resolution=None,
    )
    payload = review.to_dict()
    assert payload["receipt_timestamp"]["read_only"] is True
    assert payload["source_sha256"]
    assert payload["revision_hash"] == review.content_hash
    with pytest.raises(ValueError, match="predecessor"):
        replace(review, revision=2)

    direction = FieldRevisionV3(
        "direction",
        "credit",
        FieldRevisionProvenance.USER_DIRECTION_CONTROL,
    )
    first_user_revision = ReviewUserRevision(
        revision=1,
        action="correct",
        actor_id_hash="b" * 64,
        field_revisions=(direction,),
        created_at_epoch_ms=1_700_000_000_100,
    )
    revised = replace(
        review,
        revision=2,
        user_revisions=(first_user_revision,),
        previous_revision_hash=review.content_hash,
    )
    revised_payload = revised.to_dict()
    assert revised_payload["user_revisions"][0]["revision_hash"] == (
        first_user_revision.revision_hash
    )
    assert revised_payload["previous_revision_hash"] == review.content_hash

    broken_second = ReviewUserRevision(
        revision=2,
        action="confirm",
        actor_id_hash="c" * 64,
        field_revisions=(),
        created_at_epoch_ms=1_700_000_000_200,
        previous_revision_hash="d" * 64,
    )
    with pytest.raises(ValueError, match="hash chain"):
        replace(
            review,
            revision=3,
            user_revisions=(first_user_revision, broken_second),
            previous_revision_hash=revised.content_hash,
        )


def test_feedback_v3_uses_spans_and_direction_control_without_timestamp_actions() -> None:
    source, event = _source_and_event()
    amount = FieldRevisionV3(
        "amount",
        "42.50",
        FieldRevisionProvenance.SOURCE_SPAN_SELECTION,
        event.amount_span,
    )
    direction = FieldRevisionV3(
        "direction",
        "credit",
        FieldRevisionProvenance.USER_DIRECTION_CONTROL,
    )
    account = FieldRevisionV3(
        "account",
        "7788",
        FieldRevisionProvenance.SOURCE_SPAN_SELECTION,
        event.account_span,
        "account-opaque-1",
    )
    feedback = UserFeedbackEventV3.create(
        action_id="22222222-2222-4222-8222-222222222222",
        operation_id="11111111-1111-4111-8111-111111111111",
        review_case_id="review-synthetic",
        source=source,
        expected_review_revision=0,
        resulting_review_revision=1,
        action="correct",
        actor_id="synthetic-user",
        field_revisions=(amount, direction, account),
        created_at_epoch_ms=1_700_000_000_100,
    )
    assert feedback.to_dict()["field_revisions"][1]["span"] is None
    assert len(feedback.event_hash) == 64
    with pytest.raises(ValueError, match="unsupported"):
        FieldRevisionV3(
            "transaction_time",
            0,
            FieldRevisionProvenance.SOURCE_SPAN_SELECTION,
            SourceSpan.from_source(source, 0, 3),
        )


def test_feedback_v3_rejects_source_spans_not_bound_to_the_review_sms() -> None:
    source, event = _source_and_event()
    common = {
        "action_id": "22222222-2222-4222-8222-222222222222",
        "operation_id": "11111111-1111-4111-8111-111111111111",
        "review_case_id": "review-synthetic",
        "source": source,
        "expected_review_revision": 0,
        "resulting_review_revision": 1,
        "action": "correct",
        "actor_id": "synthetic-user",
        "created_at_epoch_ms": 1_700_000_000_100,
    }
    fabricated = FieldRevisionV3(
        "amount",
        "99.99",
        FieldRevisionProvenance.SOURCE_SPAN_SELECTION,
        SourceSpan(
            event.amount_span.start_scalar,
            event.amount_span.end_scalar,
            "fabricated",
        ),
    )
    with pytest.raises(ValueError, match="does not match source"):
        UserFeedbackEventV3.create(field_revisions=(fabricated,), **common)

    out_of_bounds = FieldRevisionV3(
        "amount",
        "42.50",
        FieldRevisionProvenance.SOURCE_SPAN_SELECTION,
        SourceSpan(0, len(source) + 1, source),
    )
    with pytest.raises(ValueError, match="span is invalid"):
        UserFeedbackEventV3.create(field_revisions=(out_of_bounds,), **common)


def test_processing_trace_v3_hash_chain_and_extractor_stages() -> None:
    first = TraceEventV3(
        sequence=0,
        event_id="33333333-3333-4333-8333-333333333333",
        occurred_at_epoch_ms=1_700_000_000_000,
        stage="analysis_advisory",
        status="completed",
    )
    second = TraceEventV3(
        sequence=1,
        event_id="44444444-4444-4444-8444-444444444444",
        occurred_at_epoch_ms=1_700_000_000_100,
        stage="extractor_execution",
        status="completed",
        previous_event_hash=first.event_hash,
    )
    trace = ProcessingTraceV3.create(
        operation_id="11111111-1111-4111-8111-111111111111",
        config_hash="a" * 64,
        owner_generation=1,
        events=(first, second),
    )
    assert trace.contract == "pocketfinancer.processing-trace/3"
    assert len(trace.trace_hash) == 64
    with pytest.raises(ValueError, match="hash chain"):
        ProcessingTraceV3.create(
            operation_id="11111111-1111-4111-8111-111111111111",
            config_hash="a" * 64,
            owner_generation=1,
            events=(first, replace(second, previous_event_hash="b" * 64)),
        )
