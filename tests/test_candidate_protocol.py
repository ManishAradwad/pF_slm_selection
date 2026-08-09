from dataclasses import replace
from decimal import Decimal
import json

import pytest

from lfm25.candidate_protocol import (
    PORTABLE_SIGNED_INT64_MAX,
    PROTOCOL_VERSION,
    AccountHintState,
    CandidateProtocolError,
    ExactMoney,
    LocalAccount,
    OracleCoverage,
    OutcomeReason,
    OutcomeStatus,
    ProtocolRequest,
    build_protocol_request,
    candidate_protocol_messages,
    canonical_candidate_evidence_bytes,
    canonical_candidate_payload_bytes,
    canonical_model_messages_bytes,
    canonical_request_bytes,
    contract_provenance,
    oracle_coverage,
    parse_selector_output,
    resolve_account_hint,
    selector_target_mapping,
    serialize_selector_target,
)
from lfm25.candidates import Candidate, CandidateSet


SMS = "INR 1,234.50 spent on your Credit Card ending XX0000 at DEMO MARKET on 03-JAN."
GOLD = {
    "amount": 1234.5,
    "counterparty": "DEMO MARKET",
    "type": "debit",
    "account": "Credit Card ending XX0000",
}


@pytest.fixture
def protocol_request() -> ProtocolRequest:
    return build_protocol_request("VD-DEMBNK", SMS, 1_767_225_600_000)


def _target(protocol_request: ProtocolRequest, **updates: object) -> str:
    value = selector_target_mapping(GOLD, protocol_request)
    value.update(updates)
    return json.dumps(value, separators=(",", ":"))


def test_protocol_request_binds_host_metadata_but_projects_only_model_inputs(protocol_request):
    assert protocol_request.protocol_version == PROTOCOL_VERSION == "candidate_protocol_v1"
    assert protocol_request.message_timestamp_epoch_ms == 1_767_225_600_000
    assert protocol_request.exact_amounts[0].id == "A0"
    assert protocol_request.exact_amounts[0].money == ExactMoney("1234.50", 123_450)
    assert protocol_request.prompt_payload()["amounts"] == {"A0": "1234.50"}

    messages = candidate_protocol_messages(protocol_request)
    assert [message["role"] for message in messages] == ["system", "user"]
    rendered = json.dumps(messages, ensure_ascii=False)
    assert "candidate_protocol_v1" not in rendered
    assert "1767225600000" not in rendered
    assert '"PN":null' in messages[1]["content"]
    assert '"A0":"1234.50"' in messages[1]["content"]


def test_timestamp_is_optional_offline_but_invalid_values_are_rejected():
    assert build_protocol_request("VD-DEMBNK", SMS).message_timestamp_epoch_ms is None
    for invalid in (-1, 1.5, True, "1000"):
        with pytest.raises(ValueError, match="message_timestamp_epoch_ms"):
            build_protocol_request("VD-DEMBNK", SMS, invalid)  # type: ignore[arg-type]


def test_oracle_and_target_serialization_are_canonical(protocol_request):
    coverage = oracle_coverage(GOLD, protocol_request)
    assert coverage == OracleCoverage(
        covered=True,
        is_transaction=True,
        amount_id="A0",
        account_id="C0",
        counterparty_id="PA0",
        type_code="D",
        missing_fields=(),
    )
    assert coverage.selection == {
        "transaction": 1,
        "type": "D",
        "amount": "A0",
        "account": "C0",
        "counterparty": "PA0",
    }
    assert serialize_selector_target(GOLD, protocol_request) == (
        '{"transaction":1,"type":"D","amount":"A0","account":"C0","counterparty":"PA0"}'
    )
    assert selector_target_mapping(None, protocol_request) == {"transaction": 0}
    assert serialize_selector_target(None, protocol_request) == '{"transaction":0}'


def test_oracle_reports_uncovered_fields_instead_of_inventing_ids(protocol_request):
    uncovered = {**GOLD, "amount": 999.0, "counterparty": "MISSING PARTY"}
    coverage = oracle_coverage(uncovered, protocol_request)
    assert not coverage.covered
    assert coverage.selection is None
    assert coverage.missing_fields == ("amount", "counterparty")
    with pytest.raises(CandidateProtocolError, match="amount, counterparty"):
        serialize_selector_target(uncovered, protocol_request)


def test_parser_reconstructs_existing_four_field_value_and_host_metadata(protocol_request):
    local_accounts = [LocalAccount("local-card", "Credit Card XX0000")]
    outcome = parse_selector_output(
        " \n" + serialize_selector_target(GOLD, protocol_request) + "\t ",
        protocol_request,
        local_accounts=local_accounts,
    )
    assert outcome.status is OutcomeStatus.TRANSACTION
    assert outcome.reason is OutcomeReason.ACCEPTED_TRANSACTION
    assert outcome.transaction == GOLD
    assert tuple(outcome.transaction) == ("amount", "counterparty", "type", "account")
    assert outcome.selection == selector_target_mapping(GOLD, protocol_request)
    assert outcome.exact_amount == ExactMoney("1234.50", 123_450)
    assert outcome.message_timestamp_epoch_ms == 1_767_225_600_000
    assert outcome.account_resolution is not None
    assert outcome.account_resolution.state is AccountHintState.UNIQUE_MATCH
    assert outcome.account_resolution.unique_account_id == "local-card"
    assert outcome.accepted
    assert outcome.persistence_ready


def test_parser_accepts_not_transaction_without_reconstructing(protocol_request):
    outcome = parse_selector_output('\n { "transaction" : 0 } \t', protocol_request)
    assert outcome.status is OutcomeStatus.NOT_TRANSACTION
    assert outcome.reason is OutcomeReason.ACCEPTED_NOT_TRANSACTION
    assert outcome.selection == {"transaction": 0}
    assert outcome.transaction is None
    assert outcome.exact_amount is None
    assert outcome.account_resolution is None
    assert outcome.message_timestamp_epoch_ms == protocol_request.message_timestamp_epoch_ms
    assert outcome.accepted
    assert not outcome.persistence_ready


@pytest.mark.parametrize(
    ("output", "reason"),
    [
        (None, OutcomeReason.OUTPUT_NOT_TEXT),
        (" \n\t", OutcomeReason.EMPTY_OUTPUT),
        ('answer: {"transaction":0}', OutcomeReason.INVALID_JSON),
        ('```json\n{"transaction":0}\n```', OutcomeReason.INVALID_JSON),
        ('{"transaction":', OutcomeReason.INVALID_JSON),
        ('{"transaction":0} prose', OutcomeReason.TRAILING_CONTENT),
        ('\u00a0{"transaction":0}', OutcomeReason.INVALID_JSON),
        ('{"transaction":0}\u00a0', OutcomeReason.TRAILING_CONTENT),
        ('{"transaction":0}{"transaction":0}', OutcomeReason.TRAILING_CONTENT),
        ("null", OutcomeReason.OUTPUT_NOT_OBJECT),
        ("[]", OutcomeReason.OUTPUT_NOT_OBJECT),
    ],
)
def test_parser_rejects_every_non_full_object_output(protocol_request, output, reason):
    outcome = parse_selector_output(output, protocol_request)
    assert outcome.status is OutcomeStatus.REJECTED
    assert outcome.reason is reason
    assert outcome.selection is None
    assert outcome.transaction is None
    assert outcome.exact_amount is None
    assert not outcome.accepted


@pytest.mark.parametrize(
    "output",
    [
        '{"transaction":0,"transaction":0}',
        (
            '{"transaction":1,"type":"D","amount":"A0",'
            '"account":"C0","counterparty":"PA0","amount":"A0"}'
        ),
    ],
)
def test_parser_rejects_duplicate_members(protocol_request, output):
    assert parse_selector_output(output, protocol_request).reason is OutcomeReason.DUPLICATE_KEY


@pytest.mark.parametrize(
    "output",
    [
        "{}",
        '{"transaction":0,"version":"candidate_protocol_v1"}',
        '{"transaction":0,"extra":1}',
        '{"type":"D","transaction":1,"amount":"A0","account":"C0","counterparty":"PA0"}',
        '{"transaction":1,"amount":"A0","type":"D","account":"C0","counterparty":"PA0"}',
        '{"transaction":1,"type":"D","amount":"A0","account":"C0"}',
    ],
)
def test_parser_requires_exact_fields_and_canonical_member_order(protocol_request, output):
    assert parse_selector_output(output, protocol_request).reason is OutcomeReason.SCHEMA_MISMATCH


@pytest.mark.parametrize(
    "flag",
    [False, True, -1, 2, 1.0, "1", None],
)
def test_parser_requires_integer_zero_or_one_transaction_flag(protocol_request, flag):
    output = json.dumps(
        {
            "transaction": flag,
            "type": "D",
            "amount": "A0",
            "account": "C0",
            "counterparty": "PA0",
        },
        separators=(",", ":"),
    )
    assert (
        parse_selector_output(output, protocol_request).reason
        is OutcomeReason.INVALID_TRANSACTION_FLAG
    )


@pytest.mark.parametrize("type_code", ["d", "debit", "", None, 1, [], {}])
def test_parser_requires_exact_direction_codes(protocol_request, type_code):
    outcome = parse_selector_output(_target(protocol_request, type=type_code), protocol_request)
    assert outcome.reason is OutcomeReason.INVALID_TYPE_CODE


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"amount": "A9"}, OutcomeReason.UNKNOWN_AMOUNT_ID),
        ({"amount": 0}, OutcomeReason.UNKNOWN_AMOUNT_ID),
        ({"account": "C9"}, OutcomeReason.UNKNOWN_ACCOUNT_ID),
        ({"account": None}, OutcomeReason.UNKNOWN_ACCOUNT_ID),
        ({"counterparty": "PA9"}, OutcomeReason.UNKNOWN_COUNTERPARTY_ID),
        ({"counterparty": "PN0"}, OutcomeReason.UNKNOWN_COUNTERPARTY_ID),
    ],
)
def test_parser_rejects_unknown_or_non_string_candidate_ids(protocol_request, updates, reason):
    outcome = parse_selector_output(_target(protocol_request, **updates), protocol_request)
    assert outcome.status is OutcomeStatus.REJECTED
    assert outcome.reason is reason
    assert outcome.transaction is None


def test_pn_is_the_only_explicit_no_counterparty_choice():
    sms = "INR 500 withdrawn from A/c XX6254 on 01-JAN."
    gold = {
        "amount": 500.0,
        "counterparty": None,
        "type": "debit",
        "account": "A/c XX6254",
    }
    protocol_request = build_protocol_request("VM-DEMBNK", sms, 1_700_000_000_000)
    assert [item.id for item in protocol_request.candidates.counterparties].count("PN") == 1
    assert selector_target_mapping(gold, protocol_request)["counterparty"] == "PN"
    outcome = parse_selector_output(
        serialize_selector_target(gold, protocol_request), protocol_request
    )
    assert outcome.transaction is not None
    assert outcome.transaction["counterparty"] is None

    without_pn = replace(
        protocol_request.candidates,
        counterparties=tuple(
            item for item in protocol_request.candidates.counterparties if item.id != "PN"
        ),
    )
    with pytest.raises(CandidateProtocolError, match="exactly one null PN"):
        build_protocol_request("VM-DEMBNK", sms, candidates=without_pn)

    second_null = replace(
        protocol_request.candidates,
        counterparties=protocol_request.candidates.counterparties
        + (Candidate("PA9", None, None, None, None),),
    )
    with pytest.raises(CandidateProtocolError, match="only null"):
        build_protocol_request("VM-DEMBNK", sms, candidates=second_null)


def test_exact_money_preserves_source_precision_without_changing_app_shape():
    sms = "INR 7.125 spent on Card XX1234 at DEMO CAFE."
    gold = {
        "amount": 7.125,
        "counterparty": "DEMO CAFE",
        "type": "debit",
        "account": "Card XX1234",
    }
    protocol_request = build_protocol_request("VD-DEMBNK", sms, 1_700_000_000_000)
    assert protocol_request.exact_amounts[0].money == ExactMoney("7.125", None)
    assert protocol_request.prompt_payload()["amounts"] == {"A0": "7.125"}
    outcome = parse_selector_output(
        serialize_selector_target(gold, protocol_request), protocol_request
    )
    assert outcome.transaction == gold
    assert outcome.exact_amount == ExactMoney("7.125", None)
    assert not outcome.persistence_ready


def test_account_hint_resolution_has_explicit_zero_unique_and_multiple_states():
    hint = "Credit Card ending XX0000"
    zero = resolve_account_hint(hint, [])
    assert zero.state is AccountHintState.ZERO_MATCHES
    assert zero.matching_account_ids == ()
    assert zero.unique_account_id is None

    unique = resolve_account_hint(hint, {"card-a": "Credit Card XX0000"})
    assert unique.state is AccountHintState.UNIQUE_MATCH
    assert unique.matching_account_ids == ("card-a",)
    assert unique.unique_account_id == "card-a"

    multiple = resolve_account_hint(
        hint,
        [
            LocalAccount("card-a", "Card *0000"),
            LocalAccount("card-b", "Debit Card ending 0000"),
            LocalAccount("account-a", "A/c XX0000"),
        ],
    )
    assert multiple.state is AccountHintState.MULTIPLE_MATCHES
    assert multiple.matching_account_ids == ("card-a", "card-b")
    assert multiple.unique_account_id is None


def test_parser_exposes_ambiguous_account_resolution_without_guessing(protocol_request):
    outcome = parse_selector_output(
        serialize_selector_target(GOLD, protocol_request),
        protocol_request,
        local_accounts=[
            LocalAccount("a", "Credit Card XX0000"),
            LocalAccount("b", "Card ending 0000"),
        ],
    )
    assert outcome.status is OutcomeStatus.TRANSACTION
    assert outcome.account_resolution is not None
    assert outcome.account_resolution.state is AccountHintState.MULTIPLE_MATCHES
    assert not outcome.persistence_ready


def test_candidate_evidence_uses_utf8_bytes_and_stays_outside_model_io():
    sms = "Prefix ₹ and é marker. INR 10.00 spent on Card XX1234 at DEMO SHOP."
    request = build_protocol_request("VD-DEMBNK", sms)
    non_pn_ids = {
        item.id
        for items in (
            request.candidates.amounts,
            request.candidates.accounts,
            request.candidates.counterparties,
        )
        for item in items
        if item.id != "PN"
    }
    assert {item.id for item in request.candidate_evidence} == non_pn_ids
    assert request.evidence_for("PN") is None

    encoded_sms = sms.encode("utf-8")
    for evidence in request.candidate_evidence:
        assert evidence.offset_convention == "utf8_bytes"
        assert (
            encoded_sms[evidence.start_utf8_byte : evidence.end_utf8_byte].decode("utf-8")
            == evidence.source_text
        )

    amount = request.evidence_for("A0")
    amount_candidate = request.candidates.amounts[0]
    assert amount is not None
    assert amount.start_utf8_byte > amount_candidate.start
    rendered = json.dumps(candidate_protocol_messages(request), ensure_ascii=False)
    assert "candidate_evidence" not in rendered
    assert "start_utf8_byte" not in rendered
    assert "offset_convention" not in rendered


def test_invalid_character_span_is_rejected_before_byte_evidence_is_built():
    sms = "Prefix ₹: INR 10.00 spent on Card XX1234 at DEMO SHOP."
    request = build_protocol_request("VD-DEMBNK", sms)
    amount = request.candidates.amounts[0]
    bad_amount = replace(amount, start=amount.start + 1)
    bad_candidates = replace(request.candidates, amounts=(bad_amount,))
    with pytest.raises(CandidateProtocolError, match="invalid source evidence"):
        build_protocol_request("VD-DEMBNK", sms, candidates=bad_candidates)


def test_reconstruction_failure_is_a_structured_rejection(protocol_request):
    invalid_candidates = CandidateSet(
        amounts=protocol_request.candidates.amounts,
        accounts=(Candidate("C0", "   ", "   ", 0, 3),),
        counterparties=protocol_request.candidates.counterparties,
        type_hints=protocol_request.candidates.type_hints,
    )
    trusted_but_invalid_protocol_request = ProtocolRequest(
        sender=protocol_request.sender,
        sms=protocol_request.sms,
        message_timestamp_epoch_ms=protocol_request.message_timestamp_epoch_ms,
        candidates=invalid_candidates,
        exact_amounts=protocol_request.exact_amounts,
        candidate_evidence=protocol_request.candidate_evidence,
    )
    outcome = parse_selector_output(_target(protocol_request), trusted_but_invalid_protocol_request)
    assert outcome.status is OutcomeStatus.REJECTED
    assert outcome.reason is OutcomeReason.RECONSTRUCTION_FAILED
    assert outcome.transaction is None


def test_contract_provenance_is_json_safe_and_binds_both_implementations():
    provenance = contract_provenance()
    assert provenance["name"] == PROTOCOL_VERSION
    assert provenance["version"] == 1
    assert set(provenance) == {
        "name",
        "version",
        "offset_convention",
        "protocol_module_sha256",
        "candidate_extractor_sha256",
        "system_prompt_utf8_sha256",
        "selector_schema_sha256",
    }
    for key, value in provenance.items():
        if key.endswith("sha256"):
            assert len(value) == 64
            int(value, 16)
    assert json.loads(json.dumps(provenance)) == provenance


def test_canonical_byte_serializers_lock_schema_order_and_utf8():
    sender = "e\u0301-SENDER"
    sms = "₹ marker. " + SMS
    request = build_protocol_request(sender, sms, PORTABLE_SIGNED_INT64_MAX)

    expected_request = {
        "sender": sender,
        "sms": sms,
        "message_timestamp_epoch_ms": PORTABLE_SIGNED_INT64_MAX,
    }
    request_bytes = canonical_request_bytes(request)
    assert request_bytes == json.dumps(
        expected_request, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    assert tuple(json.loads(request_bytes)) == (
        "sender",
        "sms",
        "message_timestamp_epoch_ms",
    )

    payload_bytes = canonical_candidate_payload_bytes(request)
    assert payload_bytes == json.dumps(
        request.prompt_payload(), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    assert tuple(json.loads(payload_bytes)) == (
        "amounts",
        "accounts",
        "counterparties",
        "type_hints",
    )

    evidence_object = [
        {
            "id": item.id,
            "candidate_kind": item.candidate_kind,
            "source_text": item.source_text,
            "start_utf8_byte": item.start_utf8_byte,
            "end_utf8_byte": item.end_utf8_byte,
            "offset_convention": item.offset_convention,
        }
        for item in request.candidate_evidence
    ]
    evidence_bytes = canonical_candidate_evidence_bytes(request)
    assert evidence_bytes == json.dumps(
        evidence_object, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    assert all(
        tuple(item)
        == (
            "id",
            "candidate_kind",
            "source_text",
            "start_utf8_byte",
            "end_utf8_byte",
            "offset_convention",
        )
        for item in json.loads(evidence_bytes)
    )

    messages = candidate_protocol_messages(request)
    messages_bytes = canonical_model_messages_bytes(request)
    assert messages_bytes == json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    assert all(tuple(item) == ("role", "content") for item in json.loads(messages_bytes))
    payload_text = payload_bytes.decode("utf-8")
    assert f"Candidates: {payload_text}\nOutput:" in messages[1]["content"]

    for serialized in (request_bytes, payload_bytes, evidence_bytes, messages_bytes):
        assert not serialized.startswith(b"\xef\xbb\xbf")
        assert not serialized.endswith(b"\n")
    assert b"e\xcc\x81-SENDER" in request_bytes
    assert "₹".encode() in request_bytes
    assert b"\\u20b9" not in request_bytes


def test_request_rejects_non_unicode_scalar_sender_and_sms():
    with pytest.raises(ValueError, match="Unicode scalar"):
        build_protocol_request("\ud800", SMS)
    with pytest.raises(ValueError, match="Unicode scalar"):
        build_protocol_request("VD-DEMBNK", SMS + "\udfff")


def test_timestamp_accepts_only_nonnegative_portable_signed_int64():
    request = build_protocol_request("VD-DEMBNK", SMS, PORTABLE_SIGNED_INT64_MAX)
    assert request.message_timestamp_epoch_ms == PORTABLE_SIGNED_INT64_MAX
    with pytest.raises(ValueError, match="signed Int64"):
        build_protocol_request("VD-DEMBNK", SMS, PORTABLE_SIGNED_INT64_MAX + 1)


def test_minor_units_use_portable_signed_int64_availability_semantics():
    at_limit = build_protocol_request(
        "VD-DEMBNK",
        "INR 92233720368547758.07 spent on Card XX1234 at DEMO SHOP.",
    )
    assert at_limit.exact_amounts[0].money == ExactMoney(
        "92233720368547758.07", PORTABLE_SIGNED_INT64_MAX
    )

    above_limit = build_protocol_request(
        "VD-DEMBNK",
        "INR 92233720368547758.08 spent on Card XX1234 at DEMO SHOP.",
    )
    assert above_limit.exact_amounts[0].money == ExactMoney("92233720368547758.08", None)


def test_nonfinite_app_projection_is_a_structured_reconstruction_rejection():
    huge_amount = "9" * 400
    request = build_protocol_request(
        "VD-DEMBNK",
        f"INR {huge_amount} spent on Card XX1234 at DEMO SHOP.",
    )
    exact = request.exact_amounts[0].money
    assert exact.minor_units is None
    with pytest.raises(CandidateProtocolError, match="finite app projection"):
        exact.app_amount()

    selection = {
        "transaction": 1,
        "type": "D",
        "amount": request.candidates.amounts[0].id,
        "account": request.candidates.accounts[0].id,
        "counterparty": next(
            item.id for item in request.candidates.counterparties if item.id != "PN"
        ),
    }
    outcome = parse_selector_output(json.dumps(selection, separators=(",", ":")), request)
    assert outcome.status is OutcomeStatus.REJECTED
    assert outcome.reason is OutcomeReason.RECONSTRUCTION_FAILED


def test_supplied_candidate_values_must_be_derived_from_their_source_spans(
    protocol_request,
):
    candidates = protocol_request.candidates
    forged_sets = [
        replace(
            candidates,
            amounts=(replace(candidates.amounts[0], value="1234.51"),),
        ),
        replace(
            candidates,
            accounts=(replace(candidates.accounts[0], value="Credit Card ending XX9999"),),
        ),
        replace(
            candidates,
            counterparties=tuple(
                replace(item, value="FORGED MARKET") if item.id == "PA0" else item
                for item in candidates.counterparties
            ),
        ),
    ]
    for forged in forged_sets:
        with pytest.raises(CandidateProtocolError, match="source span"):
            build_protocol_request(
                protocol_request.sender,
                protocol_request.sms,
                candidates=forged,
            )


def test_amount_candidate_has_no_metadata_fallback_without_source_decimal(
    protocol_request,
):
    candidates = protocol_request.candidates
    amount = candidates.amounts[0]
    start = protocol_request.sms.index("DEMO MARKET")
    forged_amount = replace(
        amount,
        value="1234.50",
        source_text="DEMO MARKET",
        start=start,
        end=start + len("DEMO MARKET"),
    )
    with pytest.raises(CandidateProtocolError, match="one source decimal"):
        build_protocol_request(
            protocol_request.sender,
            protocol_request.sms,
            candidates=replace(candidates, amounts=(forged_amount,)),
        )


def test_oracle_preserves_exact_gold_amounts_beyond_binary_float_precision():
    sms = "INR 9007199254740992 debited from A/c XX0000. Available balance INR 9007199254740993."
    request = build_protocol_request("ZX-DEMO", sms)
    common_gold = {
        "counterparty": None,
        "type": "debit",
        "account": "A/c XX0000",
    }

    for exact_amount in (9007199254740993, Decimal("9007199254740993")):
        coverage = oracle_coverage(
            {"amount": exact_amount, **common_gold},
            request,
        )
        assert coverage.covered
        assert coverage.amount_id == "A1"

    string_gold = (
        '{"amount":9007199254740993.00,"counterparty":null,"type":"debit","account":"A/c XX0000"}'
    )
    assert oracle_coverage(string_gold, request).amount_id == "A1"


def test_oracle_fails_closed_when_float_amount_projection_is_ambiguous():
    sms = "INR 9007199254740992 debited from A/c XX0000. Available balance INR 9007199254740993."
    request = build_protocol_request("ZX-DEMO", sms)
    lossy_gold = {
        "amount": float(9007199254740993),
        "counterparty": None,
        "type": "debit",
        "account": "A/c XX0000",
    }

    coverage = oracle_coverage(lossy_gold, request)
    assert not coverage.covered
    assert coverage.amount_id is None
    assert coverage.missing_fields == ("amount",)
    with pytest.raises(CandidateProtocolError, match="amount"):
        serialize_selector_target(lossy_gold, request)


def test_v1_exact_gold_path_preserves_legacy_validation_of_other_fields():
    request = build_protocol_request("VD-DEMBNK", SMS)
    invalid_type = {**GOLD, "amount": Decimal("1234.50"), "type": "pending"}

    with pytest.raises(ValueError, match="four-field contract"):
        oracle_coverage(invalid_type, request)


def test_v1_gold_json_accepts_only_rfc_json_boundary_whitespace():
    request = build_protocol_request("VD-DEMBNK", SMS)
    serialized = json.dumps(GOLD, separators=(",", ":"))

    assert oracle_coverage(f" \t{serialized}\r\n", request).covered
    with pytest.raises(ValueError, match="valid JSON"):
        oracle_coverage(f"\u00a0{serialized}", request)


def test_v1_gold_json_requires_exact_null_and_unique_exact_members():
    request = build_protocol_request("VD-DEMBNK", SMS)
    duplicate_amount = (
        '{"amount":1234.50,"amount":999.00,"counterparty":"DEMO MARKET",'
        '"type":"debit","account":"Credit Card ending XX0000"}'
    )

    assert oracle_coverage("null", request).is_transaction is False
    for invalid_json in ("NULL", duplicate_amount):
        with pytest.raises(ValueError, match="valid JSON"):
            oracle_coverage(invalid_json, request)

    extra_mapping = {**GOLD, "unexpected": "value"}
    extra_json = json.dumps(extra_mapping, separators=(",", ":"))
    for invalid_shape in (extra_mapping, extra_json):
        with pytest.raises(ValueError, match="four-field contract"):
            oracle_coverage(invalid_shape, request)


def test_supplied_candidates_must_equal_deterministic_protocol_extraction():
    sms = "INR 10 debited from A/c XX0000. Available balance INR 20."
    request = build_protocol_request("ZX-DEMO", sms)
    candidates = request.candidates
    amounts = candidates.amounts
    pn = next(item for item in candidates.counterparties if item.id == "PN")
    non_pn = tuple(item for item in candidates.counterparties if item.id != "PN")

    rebuilt = build_protocol_request(
        "ZX-DEMO",
        sms,
        candidates=candidates,
    )
    assert rebuilt.candidates == candidates

    noncanonical_sets = [
        replace(candidates, amounts=tuple(reversed(amounts))),
        replace(
            candidates,
            amounts=(replace(amounts[0], id="A7"), *amounts[1:]),
        ),
        replace(
            candidates,
            amounts=amounts + (replace(amounts[0], id="A2"),),
        ),
        replace(candidates, counterparties=(pn, *non_pn)),
    ]
    for noncanonical in noncanonical_sets:
        with pytest.raises(CandidateProtocolError, match="deterministic protocol extraction"):
            build_protocol_request(
                "ZX-DEMO",
                sms,
                candidates=noncanonical,
            )
