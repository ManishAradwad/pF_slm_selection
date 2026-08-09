from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from lfm25.android_contract import pocketfinancer_prefilter_sms
from lfm25.candidate_protocol import (
    EVIDENCE_OFFSET_CONVENTION,
    PROTOCOL_REVISION,
    PROTOCOL_VERSION,
    AccountHintState,
    OutcomeReason,
    OutcomeStatus,
    build_protocol_request,
    candidate_protocol_messages,
    canonical_candidate_evidence_bytes,
    canonical_candidate_payload_bytes,
    canonical_model_messages_bytes,
    canonical_request_bytes,
    contract_provenance,
    parse_selector_output,
    resolve_account_hint,
    selector_target_mapping,
    serialize_selector_target,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO_ROOT / "configs/contracts/pocketfinancer-candidate-v1.json"
GOLDEN_PATH = REPO_ROOT / "DATA/candidate_protocol_v1_golden.json"
PREFILTER_PROFILE_PATH = REPO_ROOT / "configs/contracts/pocketfinancer-android-current.json"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _candidate_ids(request: Any) -> dict[str, list[str]]:
    return {
        "amount": [item.id for item in request.candidates.amounts],
        "account": [item.id for item in request.candidates.accounts],
        "counterparty": [item.id for item in request.candidates.counterparties],
    }


def _candidate_evidence(request: Any) -> list[dict[str, Any]]:
    return [
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


def test_candidate_v1_profile_matches_the_executable_semantic_contract() -> None:
    profile = _load_json(PROFILE_PATH)
    provenance = contract_provenance()

    assert profile["profile"] == PROTOCOL_VERSION == "candidate_protocol_v1"
    assert profile["profile_version"] == PROTOCOL_REVISION == 1
    assert profile["protocol_binding"] == {
        "name": PROTOCOL_VERSION,
        "version": PROTOCOL_REVISION,
        "version_location": "profile",
        "version_emitted_by_model": False,
    }
    assert provenance["name"] == PROTOCOL_VERSION
    assert provenance["version"] == PROTOCOL_REVISION
    assert all(
        len(provenance[key]) == 64
        for key in (
            "protocol_module_sha256",
            "candidate_extractor_sha256",
            "system_prompt_utf8_sha256",
            "selector_schema_sha256",
        )
    )
    assert provenance["offset_convention"] == EVIDENCE_OFFSET_CONVENTION

    application = profile["application_input"]
    assert application["schema"]["properties"]["message_timestamp_epoch_ms"]["maximum"] == 2**63 - 1
    assert (
        application["portable_integer_ranges"]["message_timestamp_epoch_ms"]["maximum"] == 2**63 - 1
    )

    conventions = profile["candidate_id_conventions"]
    assert "exact Decimal" in conventions["amount"]["source_enumeration"]
    assert conventions["counterparty"]["source_tie_break_prefix_order"] == [
        "PL",
        "PV",
        "PA",
        "PT",
        "PB",
        "PF",
        "PW",
        "PU",
        "PR",
        "PO",
    ]
    assert "forged-value" in conventions["validation"]
    lexical_domain = conventions["enumeration_lexical_domain"]
    assert lexical_domain["regex_classes"].startswith("\\b, \\d, \\s, and \\w use ASCII")
    assert "ASCII A through Z" in lexical_domain["case_insensitive_matching"]
    assert "U+0009 through U+000D" in lexical_domain["semantic_text_normalization"]
    assert lexical_domain["unicode_dependency"].startswith(
        "Candidate enumeration has no Unicode database"
    )
    assert "exactly equal" in conventions["caller_supplied_candidate_sets"]

    oracle = profile["supervision_oracle"]
    assert "Decimal-preserving" in oracle["exact_amount_inputs"]
    assert oracle["ambiguous_binary_float"] == "candidate_missing_amount_fail_closed"
    assert oracle["exact_json_whitespace_code_points"] == ["U+0009", "U+000A", "U+000D", "U+0020"]
    assert oracle["duplicate_members_allowed"] is False
    assert oracle["additional_members_allowed"] is False

    evidence = profile["candidate_source_evidence"]
    assert evidence["scope"] == "host_only"
    assert evidence["model_visible"] is False
    assert evidence["required_for"] == "every_non_PN_candidate"
    assert evidence["PN_evidence"] is None
    assert evidence["offset_convention"] == EVIDENCE_OFFSET_CONVENTION
    assert evidence["range_semantics"] == ("half_open_start_inclusive_end_exclusive")
    assert (
        "candidate_evidence" in profile["application_input"]["model_message_projection"]["excludes"]
    )

    output = profile["model_output"]
    assert output["member_order_enforced"] is True
    assert output["surrounding_json_whitespace_allowed"] is True
    assert output["internal_json_whitespace_allowed"] is True
    assert output["accepted_json_whitespace_code_points"] == [
        "U+0009",
        "U+000A",
        "U+000D",
        "U+0020",
    ]
    assert output["all_other_unicode_whitespace"].startswith("not JSON whitespace")
    assert output["prose_markdown_fences_and_trailing_content_allowed"] is False
    assert output["schemas"]["not_transaction"]["member_order"] == ["transaction"]
    assert output["schemas"]["not_transaction"]["canonical_example"] == ('{"transaction":0}')
    assert output["schemas"]["transaction"]["member_order"] == [
        "transaction",
        "type",
        "amount",
        "account",
        "counterparty",
    ]
    assert output["schemas"]["transaction"]["canonical_example"] == (
        '{"transaction":1,"type":"D","amount":"A0","account":"C0","counterparty":"PN"}'
    )
    serialization = profile["canonical_serialization"]
    assert serialization["character_encoding"] == "UTF-8"
    assert serialization["byte_order_mark"] is False
    assert serialization["trailing_newline"] is False
    assert serialization["ensure_ascii"] is False
    assert (
        serialization["object_member_separator"],
        serialization["name_value_separator"],
    ) == (",", ":")
    assert serialization["insignificant_whitespace"] is False
    assert serialization["unicode_normalization"].startswith("none;")
    escaping = serialization["string_escaping"]
    assert escaping["quotation_mark_U+0022"] == '\\"'
    assert escaping["reverse_solidus_U+005C"] == "\\\\"
    assert escaping["solidus_U+002F"] == "literal_unescaped"
    assert escaping["short_control_escapes"] == {
        "U+0008": "\\b",
        "U+0009": "\\t",
        "U+000A": "\\n",
        "U+000C": "\\f",
        "U+000D": "\\r",
    }
    assert "lowercase hexadecimal" in escaping["other_U+0000_through_U+001F"]
    assert escaping["U+2028_and_U+2029"].startswith("literal UTF-8")

    documents = serialization["byte_documents"]
    assert tuple(documents) == (
        "application_request",
        "candidate_payload",
        "model_messages",
        "candidate_evidence",
        "selector_target",
    )
    assert documents["application_request"]["member_order"] == [
        "sender",
        "sms",
        "message_timestamp_epoch_ms",
    ]
    assert documents["candidate_payload"]["member_order"] == [
        "amounts",
        "accounts",
        "counterparties",
        "type_hints",
    ]
    assert documents["model_messages"]["array_order"] == ["system", "user"]
    assert documents["model_messages"]["message_member_order"] == ["role", "content"]
    assert documents["candidate_evidence"]["PN_included"] is False
    assert serialization["hashing"]["algorithm"] == "SHA-256"
    system_content = documents["model_messages"]["system_content"]
    assert _sha256_bytes(system_content.encode("utf-8")) == provenance["system_prompt_utf8_sha256"]
    assert (
        documents["model_messages"]["user_content_framing"]
        == "Sender: {sender}\nSMS: {sms}\nCandidates: {canonical_candidate_payload_json}\nOutput:"
    )

    runtime = profile["model_runtime_defaults"]
    assert runtime["model"] == "LiquidAI/LFM2.5-350M"
    assert runtime["thinking"] is False
    assert runtime["temperature"] == 0.0
    assert runtime["answer_max_tokens"] == 64

    accepted_reasons = {
        OutcomeReason.ACCEPTED_NOT_TRANSACTION.value,
        OutcomeReason.ACCEPTED_TRANSACTION.value,
    }
    declared_accepted = set(profile["outcomes"]["accepted_reason_codes"])
    declared_rejected = set(profile["outcomes"]["fail_closed_reason_codes"])
    assert declared_accepted == accepted_reasons
    assert declared_rejected == {
        reason.value for reason in OutcomeReason if reason.value not in accepted_reasons
    }


def test_profile_resource_hashes_prefilter_reference_and_parity_claims() -> None:
    profile = _load_json(PROFILE_PATH)

    prefilter = profile["prefilter"]
    assert prefilter["required"] is True
    assert prefilter["on_reject"] == "do_not_invoke_model"
    assert prefilter["profile_path"] == ("configs/contracts/pocketfinancer-android-current.json")
    assert prefilter["profile_sha256"] == _sha256(PREFILTER_PROFILE_PATH)
    assert prefilter["implementation"] == ("lfm25.android_contract.pocketfinancer_prefilter_sms")

    resource = profile["resources"]["golden_vectors"]
    assert resource["path"] == "DATA/candidate_protocol_v1_golden.json"
    assert resource["schema_version"] == 2
    assert resource["privacy"] == "invented_synthetic_only"
    assert resource["sha256"] == _sha256(GOLDEN_PATH)

    parity = profile["platform_parity"]
    assert parity["host_reference_implemented"] is True
    assert "candidate_evidence_utf8_sha256" in parity["required_comparison_fields"]
    for key in (
        "android_implemented",
        "ios_implemented",
        "wire_compatible_with_pocketfinancer_android_current",
        "gguf_runtime_validated",
        "device_validated",
    ):
        assert parity[key] is False
    assert any("No iOS source tree" in gap for gap in parity["known_gaps"])
    assert "protocol evidence only" in parity["claim_policy"]


def test_golden_vectors_are_explicitly_synthetic_and_cover_the_risk_surface() -> None:
    artifact = _load_json(GOLDEN_PATH)
    privacy = artifact["privacy"]
    vectors = artifact["vectors"]

    assert artifact["artifact"] == "candidate_protocol_v1_golden"
    assert artifact["schema_version"] == 2
    assert artifact["protocol_profile"] == {
        "name": PROTOCOL_VERSION,
        "version": PROTOCOL_REVISION,
        "path": "configs/contracts/pocketfinancer-candidate-v1.json",
    }
    assert privacy["classification"] == "invented_synthetic_only"
    assert privacy["contains_private_messages"] is False
    assert privacy["contains_real_senders_or_accounts"] is False
    assert "invented" in privacy["construction"]
    semantics = artifact["vector_semantics"]
    assert semantics["candidate_evidence_offset_convention"] == (EVIDENCE_OFFSET_CONVENTION)
    assert semantics["candidate_evidence_ranges"] == "half_open"
    assert semantics["candidate_evidence_model_visible"] is False
    assert semantics["canonical_hash_algorithm"] == "SHA-256"
    assert semantics["canonical_model_messages_are_exact"] is True
    assert semantics["canonical_byte_documents"] == [
        "application_request",
        "candidate_payload",
        "model_messages",
        "candidate_evidence",
        "raw_model_output",
    ]
    assert all(vector["input"]["sender"].startswith("ZX-DEMO") for vector in vectors)
    assert len({vector["id"] for vector in vectors}) == len(vectors)
    assert len({vector["input"]["message_timestamp_epoch_ms"] for vector in vectors}) == len(
        vectors
    )

    coverage = {tag for vector in vectors for tag in vector["covers"]}
    assert {
        "amount_ambiguity",
        "account_ambiguity",
        "counterparty_ambiguity",
        "null",
        "filter",
        "PN",
        "unicode",
        "quotation_mark_escaping",
        "reverse_solidus_escaping",
        "utf8_byte_offsets",
        "host_only_candidate_evidence",
        "invalid_output",
        "fail_closed",
        "large_exact_amounts",
        "binary_float_collision_resistance",
        "portable_int64_minor_units",
        "solidus_unescaped",
        "u2028_literal",
        "u2029_literal",
        "control_short_escapes",
        "control_unicode_escape_lowercase",
        "exact_gold_integer",
        "second_colliding_candidate_A1",
    } <= coverage


def test_all_golden_vectors_conform_to_prefilter_candidates_and_parser() -> None:
    vectors = _load_json(GOLDEN_PATH)["vectors"]

    for vector in vectors:
        input_value = vector["input"]
        expected = vector["expected"]
        prefilter = pocketfinancer_prefilter_sms(input_value["sender"], input_value["sms"])
        assert prefilter.accepted is expected["prefilter"]["accepted"], vector["id"]
        assert prefilter.rejection_stage == expected["prefilter"]["rejection_stage"], vector["id"]
        assert prefilter.stage_index == expected["prefilter"]["stage_index"], vector["id"]

        request = build_protocol_request(
            input_value["sender"],
            input_value["sms"],
            message_timestamp_epoch_ms=input_value["message_timestamp_epoch_ms"],
        )
        wire = expected["canonical_wire"]
        assert _sha256_bytes(canonical_request_bytes(request)) == wire["request_utf8_sha256"]

        if not expected["model_invoked"]:
            assert vector["model_output"] is None
            assert expected["candidate_ids"] is None
            assert expected["outcome"] is None
            assert wire["candidate_payload_utf8_sha256"] is None
            assert wire["model_messages_utf8_sha256"] is None
            assert wire["candidate_evidence_utf8_sha256"] is None
            assert wire["raw_model_output_utf8_sha256"] is None
            assert wire["model_messages"] is None
            continue
        assert request.protocol_version == PROTOCOL_VERSION
        assert request.message_timestamp_epoch_ms == input_value["message_timestamp_epoch_ms"]
        assert _candidate_ids(request) == expected["candidate_ids"], vector["id"]

        payload_bytes = canonical_candidate_payload_bytes(request)
        messages_bytes = canonical_model_messages_bytes(request)
        evidence_bytes = canonical_candidate_evidence_bytes(request)
        assert _sha256_bytes(payload_bytes) == wire["candidate_payload_utf8_sha256"]
        assert _sha256_bytes(messages_bytes) == wire["model_messages_utf8_sha256"]
        assert _sha256_bytes(evidence_bytes) == wire["candidate_evidence_utf8_sha256"]
        assert isinstance(vector["model_output"], str)
        assert (
            _sha256_bytes(vector["model_output"].encode("utf-8"))
            == wire["raw_model_output_utf8_sha256"]
        )
        assert wire["model_messages"] == candidate_protocol_messages(request)
        assert json.loads(messages_bytes.decode("utf-8")) == wire["model_messages"]
        for canonical_bytes in (
            canonical_request_bytes(request),
            payload_bytes,
            messages_bytes,
            evidence_bytes,
        ):
            assert not canonical_bytes.startswith(b"\xef\xbb\xbf")
            assert not canonical_bytes.endswith(b"\n")

        null_counterparties = [
            candidate for candidate in request.candidates.counterparties if candidate.value is None
        ]
        assert [candidate.id for candidate in null_counterparties] == ["PN"]

        actual_evidence = _candidate_evidence(request)
        expected_evidence_ids = [
            *expected["candidate_ids"]["amount"],
            *expected["candidate_ids"]["account"],
            *[
                candidate_id
                for candidate_id in expected["candidate_ids"]["counterparty"]
                if candidate_id != "PN"
            ],
        ]
        assert [item["id"] for item in actual_evidence] == expected_evidence_ids
        assert request.evidence_for("PN") is None
        sms_utf8 = request.sms.encode("utf-8")
        for evidence in request.candidate_evidence:
            assert evidence.offset_convention == EVIDENCE_OFFSET_CONVENTION
            assert request.evidence_for(evidence.id) == evidence
            source_slice = sms_utf8[evidence.start_utf8_byte : evidence.end_utf8_byte]
            assert source_slice.decode("utf-8") == evidence.source_text

        if "candidate_evidence" in expected:
            assert actual_evidence == expected["candidate_evidence"]

        if "exact_amounts" in expected:
            assert [
                {
                    "id": item.id,
                    "decimal_text": item.money.decimal_text,
                    "minor_units": item.money.minor_units,
                }
                for item in request.exact_amounts
            ] == expected["exact_amounts"]

        assert selector_target_mapping(vector["gold"], request) == expected["target"], vector["id"]
        assert (
            serialize_selector_target(vector["gold"], request) == expected["serialized_target"]
        ), vector["id"]

        outcome = parse_selector_output(vector["model_output"], request)
        assert outcome.status.value == expected["outcome"]["status"], vector["id"]
        assert outcome.reason.value == expected["outcome"]["reason_code"], vector["id"]
        assert outcome.transaction == expected["transaction"], vector["id"]
        assert outcome.message_timestamp_epoch_ms == input_value["message_timestamp_epoch_ms"]
        assert outcome.persistence_ready is False

        if outcome.status is OutcomeStatus.TRANSACTION:
            assert outcome.selection == expected["target"]
            assert outcome.exact_amount == request.exact_money(expected["target"]["amount"])
            assert outcome.account_resolution is not None
            assert outcome.account_resolution.state is AccountHintState.ZERO_MATCHES
        elif outcome.status is OutcomeStatus.NOT_TRANSACTION:
            assert outcome.selection == {"transaction": 0}
            assert outcome.transaction is None
        else:
            assert outcome.selection is None
            assert outcome.transaction is None


def test_large_exact_amounts_remain_distinct_before_float_projection() -> None:
    vector = next(
        item
        for item in _load_json(GOLDEN_PATH)["vectors"]
        if item["id"] == "valid_large_exact_amounts"
    )
    input_value = vector["input"]
    request = build_protocol_request(
        input_value["sender"],
        input_value["sms"],
        message_timestamp_epoch_ms=input_value["message_timestamp_epoch_ms"],
    )
    first, second = request.exact_amounts
    assert [first.money.decimal_text, second.money.decimal_text] == [
        "9007199254740992.00",
        "9007199254740993.00",
    ]
    assert [first.money.minor_units, second.money.minor_units] == [
        900719925474099200,
        900719925474099300,
    ]
    assert first.money.decimal != second.money.decimal
    assert first.money.app_amount() == second.money.app_amount()
    assert vector["expected"]["candidate_ids"]["amount"] == ["A0", "A1"]

    collider = next(
        item
        for item in _load_json(GOLDEN_PATH)["vectors"]
        if item["id"] == "valid_large_exact_amount_a1"
    )
    collider_input = collider["input"]
    collider_request = build_protocol_request(
        collider_input["sender"],
        collider_input["sms"],
        message_timestamp_epoch_ms=collider_input["message_timestamp_epoch_ms"],
    )
    collider_target = selector_target_mapping(collider["gold"], collider_request)
    assert collider_target["amount"] == "A1"
    collider_outcome = parse_selector_output(collider["model_output"], collider_request)
    assert collider_outcome.exact_amount is not None
    assert collider_outcome.exact_amount.decimal_text == "9007199254740993.00"
    assert collider_outcome.exact_amount.decimal == second.money.decimal


def test_utf8_projection_and_strict_complete_output_boundary() -> None:
    vector = next(
        item
        for item in _load_json(GOLDEN_PATH)["vectors"]
        if item["id"] == "valid_unicode_and_escaping"
    )
    input_value = vector["input"]
    request = build_protocol_request(
        input_value["sender"],
        input_value["sms"],
        message_timestamp_epoch_ms=input_value["message_timestamp_epoch_ms"],
    )
    messages = candidate_protocol_messages(request)
    user_content = messages[1]["content"]
    encoded = user_content.encode("utf-8")

    assert encoded.decode("utf-8") == user_content
    assert not encoded.startswith(b"\xef\xbb\xbf")
    assert "café\\trial" in user_content
    assert "नमस्ते" in user_content
    assert "✅" in user_content
    assert PROTOCOL_VERSION not in user_content
    assert str(input_value["message_timestamp_epoch_ms"]) not in user_content
    assert "candidate_evidence" not in user_content
    assert "start_utf8_byte" not in user_content
    assert messages == candidate_protocol_messages(request)

    request_bytes = canonical_request_bytes(request)
    assert rb"Memo: \"caf" in request_bytes
    assert rb"\\trial/path" in request_bytes
    assert b"/path" in request_bytes
    assert rb"\/" not in request_bytes
    assert chr(0x2028).encode("utf-8") in request_bytes
    assert rb"\u2028" not in request_bytes
    assert chr(0x2029).encode("utf-8") in request_bytes
    assert rb"\u2029" not in request_bytes
    assert rb"controls:\b\t\n\f\r\u0001\u001e" in request_bytes
    assert rb"\u001E" not in request_bytes

    whitespace = parse_selector_output(' \n { "transaction" : 0 }\t', request)
    reordered = parse_selector_output(
        '{"type":"D","transaction":1,"amount":"A0","account":"C0","counterparty":"PA0"}',
        request,
    )
    prose = parse_selector_output('{"transaction":0} trailing', request)
    assert whitespace.reason is OutcomeReason.ACCEPTED_NOT_TRANSACTION
    assert reordered.reason is OutcomeReason.SCHEMA_MISMATCH
    assert prose.reason is OutcomeReason.TRAILING_CONTENT


def test_local_account_resolution_is_host_only_and_fail_closed() -> None:
    zero = resolve_account_hint("Card XX9080", {})
    unique = resolve_account_hint("Card XX9080", {"local-a": "Card XX9080"})
    multiple = resolve_account_hint(
        "Card XX9080",
        {
            "local-a": "Card XX9080",
            "local-b": "Credit Card ending 9080",
        },
    )

    assert zero.state is AccountHintState.ZERO_MATCHES
    assert zero.unique_account_id is None
    assert unique.state is AccountHintState.UNIQUE_MATCH
    assert unique.unique_account_id == "local-a"
    assert multiple.state is AccountHintState.MULTIPLE_MATCHES
    assert multiple.unique_account_id is None
