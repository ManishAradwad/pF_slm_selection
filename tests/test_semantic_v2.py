from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from lfm25.semantic_v2 import (
    EVIDENCE_OFFSET_CONVENTION,
    SEMANTIC_CONTRACT_ID,
    SEMANTIC_CONTRACT_VERSION,
    EvidenceSpan,
    SemanticV2Error,
    TimestampProvenance,
    derive_currency_from_amount_evidence,
    derive_decimal_text_from_amount_evidence,
    derive_direction_from_evidence,
    derive_minor_units,
    inject_source_timestamp,
    project_initial_auto_post,
    semantic_v2_schema,
    slice_utf8_evidence,
    validate_semantic_v2,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPOSITORY_ROOT / "configs/contracts/pocketfinancer-semantic-v2.schema.json"
GOLDEN_PATH = REPOSITORY_ROOT / "tests/fixtures/pocketfinancer_semantic_v2_synthetic_golden.json"


def _golden() -> dict[str, Any]:
    value = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _full_record(vector: dict[str, Any]) -> dict[str, Any]:
    return inject_source_timestamp(
        vector["semantic_core"],
        received_at_epoch_ms=vector["source_metadata"]["received_at_epoch_ms"],
        received_at_provenance=vector["source_metadata"]["received_at_provenance"],
    )


def _vector(vector_id: str) -> dict[str, Any]:
    return next(item for item in _golden()["vectors"] if item["id"] == vector_id)


def test_schema_is_versioned_platform_neutral_and_excludes_model_dates() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    loaded = semantic_v2_schema()

    assert schema == loaded
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["properties"]["semantic_contract_id"]["const"] == SEMANTIC_CONTRACT_ID
    assert schema["properties"]["semantic_contract_version"]["const"] == SEMANTIC_CONTRACT_VERSION
    assert schema["x-pf-evidence-offset-convention"] == EVIDENCE_OFFSET_CONVENTION
    assert "date" not in schema["$defs"]["event"]["properties"]
    assert "timestamp" not in schema["$defs"]["event"]["properties"]
    assert schema["$defs"]["sourceMetadata"]["required"] == [
        "received_at_epoch_ms",
        "received_at_provenance",
    ]
    assert schema["$defs"]["evidenceSpan"]["properties"]["start_utf8_byte"]["maximum"] == (
        (1 << 63) - 1
    )
    assert schema["$defs"]["money"]["properties"]["minor_units"]["maximum"] == (1 << 63) - 1
    assert schema["x-pf-direction-evidence-lexicon-version"] == 1
    assert "non-empty" in schema["x-pf-evidence-policy"]
    assert "bare dollar sign is ambiguous" in schema["x-pf-currency-evidence-policy"]
    assert "requires non-null exact minor_units" in schema["x-pf-money-policy"]


def test_synthetic_golden_vectors_are_complete_and_project_as_declared() -> None:
    golden = _golden()

    assert golden["privacy"]["classification"] == "invented_synthetic_only"
    assert golden["privacy"]["contains_private_messages"] is False
    assert golden["privacy"]["contains_real_senders_or_accounts"] is False
    assert golden["semantic_contract"] == {
        "id": SEMANTIC_CONTRACT_ID,
        "version": SEMANTIC_CONTRACT_VERSION,
        "schema_path": "configs/contracts/pocketfinancer-semantic-v2.schema.json",
    }
    identifiers = {vector["id"] for vector in golden["vectors"]}
    assert identifiers == {
        "posted_inr_debit",
        "posted_inr_credit",
        "not_transaction",
        "refund_is_credit",
        "wallet_bnpl_out_of_scope",
        "multiple_posted_events",
        "missing_account",
        "explicitly_absent_counterparty",
        "non_inr_currency",
        "repeated_equal_amount_uses_first_utf8_span",
        "unicode_multibyte_evidence",
        "system_timestamp_injection",
        "inr_fractional_minor_unit_unavailable",
    }

    for vector in golden["vectors"]:
        assert "source_metadata" not in vector["semantic_core"]
        record = validate_semantic_v2(_full_record(vector), message=vector["message"])
        projection = project_initial_auto_post(record)
        expected = vector["expected_projection"]
        assert projection.eligible is expected["eligible"], vector["id"]
        assert [reason.value for reason in projection.reasons] == expected["reasons"], vector["id"]
        if projection.eligible:
            assert projection.transaction is not None
            assert projection.transaction.minor_units is not None
            assert (
                projection.transaction.received_at_epoch_ms
                == vector["source_metadata"]["received_at_epoch_ms"]
            )
        else:
            assert projection.transaction is None


def test_invalid_schema_and_semantic_combinations_fail_closed() -> None:
    vector = _vector("posted_inr_debit")
    baseline = _full_record(vector)

    unexpected = deepcopy(baseline)
    unexpected["date"] = "2030-01-01"
    with pytest.raises(SemanticV2Error, match="invalid keys"):
        validate_semantic_v2(unexpected, message=vector["message"])

    posted_none = deepcopy(baseline)
    posted_none["posting_status"] = "posted"
    posted_none["event_cardinality"] = "none"
    posted_none["events"] = []
    with pytest.raises(SemanticV2Error, match="none cardinality"):
        validate_semantic_v2(posted_none, message=vector["message"])

    counterparty_with_evidence = deepcopy(baseline)
    counterparty_with_evidence["events"][0]["counterparty"] = {
        "state": "absent",
        "evidence": {"start_utf8_byte": 52, "end_utf8_byte": 63},
    }
    with pytest.raises(SemanticV2Error, match="invalid keys"):
        validate_semantic_v2(counterparty_with_evidence, message=vector["message"])

    currency_mismatch = deepcopy(baseline)
    currency_mismatch["events"][0]["amount"]["currency"] = "USD"
    with pytest.raises(SemanticV2Error, match="currency does not match"):
        validate_semantic_v2(currency_mismatch, message=vector["message"])

    rounded_minor_units = deepcopy(baseline)
    rounded_minor_units["events"][0]["amount"]["minor_units"] = 125051
    with pytest.raises(SemanticV2Error, match="deterministic derivative"):
        validate_semantic_v2(rounded_minor_units, message=vector["message"])

    empty_account_evidence = deepcopy(baseline)
    empty_account_evidence["events"][0]["account"]["evidence"] = {
        "start_utf8_byte": 37,
        "end_utf8_byte": 37,
    }
    with pytest.raises(SemanticV2Error, match="non-empty range"):
        validate_semantic_v2(empty_account_evidence, message=vector["message"])

    hallucinated_account = deepcopy(baseline)
    hallucinated_account["events"][0]["account"]["value"] = "Card XX0000"
    with pytest.raises(SemanticV2Error, match="does not exactly match evidence"):
        validate_semantic_v2(hallucinated_account, message=vector["message"])

    hallucinated_counterparty = deepcopy(baseline)
    hallucinated_counterparty["events"][0]["counterparty"]["value"] = "FICTIONAL SHOP"
    with pytest.raises(SemanticV2Error, match="does not exactly match evidence"):
        validate_semantic_v2(hallucinated_counterparty, message=vector["message"])

    reversed_direction = deepcopy(baseline)
    reversed_direction["events"][0]["direction"]["value"] = "credit"
    with pytest.raises(SemanticV2Error, match="does not match direction evidence"):
        validate_semantic_v2(reversed_direction, message=vector["message"])

    unrelated_direction_evidence = deepcopy(baseline)
    unrelated_direction_evidence["events"][0]["direction"]["evidence"] = {
        "start_utf8_byte": 37,
        "end_utf8_byte": 48,
    }
    with pytest.raises(SemanticV2Error, match="direction lexicon"):
        validate_semantic_v2(unrelated_direction_evidence, message=vector["message"])


def test_exact_decimal_derivation_never_rounds() -> None:
    assert derive_decimal_text_from_amount_evidence("₹1,25,000.50") == "125000.50"
    assert derive_minor_units("125000.50", "INR") == 12500050
    assert derive_minor_units("100.000", "INR") == 10000
    assert derive_minor_units("100.005", "INR") is None
    assert derive_minor_units("92233720368547758.07", "INR") == (1 << 63) - 1
    assert derive_minor_units("1.00", "CAD") is None
    with pytest.raises(SemanticV2Error, match="signed 64-bit range"):
        derive_minor_units("92233720368547758.08", "INR")
    with pytest.raises(SemanticV2Error, match="signed 64-bit range"):
        derive_minor_units("99999999999999999999999999999", "INR")


def test_currency_and_direction_evidence_are_explicit_and_conservative() -> None:
    assert derive_currency_from_amount_evidence("CAD 12.34") == "CAD"
    assert derive_currency_from_amount_evidence("USD $12.34") == "USD"
    assert derive_direction_from_evidence("  debited  ").value == "debit"
    assert derive_direction_from_evidence("refunded").value == "credit"
    with pytest.raises(SemanticV2Error, match="bare dollar sign"):
        derive_currency_from_amount_evidence("$12.34")
    with pytest.raises(SemanticV2Error, match="exactly one currency"):
        derive_currency_from_amount_evidence("INR 10.00 or EUR 10.00")
    with pytest.raises(SemanticV2Error, match="direction lexicon"):
        derive_direction_from_evidence("transaction")

    vector = _vector("non_inr_currency")
    cad_message = vector["message"].replace("USD", "CAD")
    cad_record = _full_record(vector)
    cad_record["events"][0]["amount"]["currency"] = "CAD"
    cad_record["events"][0]["amount"]["minor_units"] = None
    record = validate_semantic_v2(cad_record, message=cad_message)
    projection = project_initial_auto_post(record)
    assert record.events[0].amount.currency == "CAD"
    assert [reason.value for reason in projection.reasons] == ["currency_not_inr"]


def test_utf8_evidence_slicing_selects_exact_span_and_rejects_split_code_point() -> None:
    repeated = _vector("repeated_equal_amount_uses_first_utf8_span")
    record = validate_semantic_v2(_full_record(repeated), message=repeated["message"])
    amount_span = record.events[0].amount.evidence
    second_span = repeated["additional_assertions"]["other_equal_amount_span"]

    assert slice_utf8_evidence(repeated["message"], amount_span) == "₹50.00"
    assert slice_utf8_evidence(repeated["message"], EvidenceSpan(**second_span)) == "₹50.00"
    assert amount_span != EvidenceSpan(**second_span)
    with pytest.raises(SemanticV2Error, match="empty"):
        slice_utf8_evidence(repeated["message"], EvidenceSpan(0, 0))

    unicode_vector = _vector("unicode_multibyte_evidence")
    unicode_record = validate_semantic_v2(_full_record(unicode_vector), message=unicode_vector["message"])
    assert slice_utf8_evidence(
        unicode_vector["message"], unicode_record.events[0].account.evidence
    ) == "कार्ड XX1234"
    with pytest.raises(SemanticV2Error, match="splits a UTF-8 code point"):
        slice_utf8_evidence(unicode_vector["message"], EvidenceSpan(38, 59))


def test_host_timestamp_injection_uses_provenance_and_rejects_duplicate_metadata() -> None:
    vector = _vector("system_timestamp_injection")
    injected = inject_source_timestamp(
        vector["semantic_core"],
        received_at_epoch_ms=1760000000999,
        received_at_provenance=TimestampProvenance.IOS_INBOX_ALERT_RECEIVED_AT_ASSIGNED_DURING_INGESTION,
    )

    assert injected["source_metadata"] == {
        "received_at_epoch_ms": 1760000000999,
        "received_at_provenance": "ios_inbox_alert_received_at_assigned_during_ingestion",
    }
    with pytest.raises(SemanticV2Error, match="exactly once"):
        inject_source_timestamp(injected, received_at_epoch_ms=1, received_at_provenance="android_sms_message_date")
