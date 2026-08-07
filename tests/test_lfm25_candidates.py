import pytest

from lfm25.candidates import (
    _dedupe_source_text_candidates,
    COUNTERPARTY_SOURCE_TIE_BREAK_ORDER,
    Candidate,
    CandidateSet,
    candidate_selector_messages,
    canonical_amount_token,
    extract_candidates,
    extract_protocol_candidates,
    oracle_selection,
    parse_selector_prediction,
    reconstruct_transaction,
    resolve_selector_prediction,
    selector_target,
)


SMS = (
    "INR 23.00 spent on your Credit Card ending XX0000 at DEMO SHOP DAILY "
    "on 03-JAN. Avbl limit: Rs.10000.00."
)
GOLD = {
    "amount": 23.0,
    "counterparty": "DEMO SHOP DAILY",
    "type": "debit",
    "account": "Credit Card ending XX0000",
}


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("+01,23,456.070", "123456.070"),
        ("0007", "7"),
        ("7.00", "7.00"),
    ],
)
def test_canonical_amount_token_preserves_source_precision(token, expected):
    assert canonical_amount_token(token) == expected


@pytest.mark.parametrize("token", ["0", "0.00", "-1", "1e3", "NaN", "1,"])
def test_canonical_amount_token_rejects_nonpositive_or_non_source_forms(token):
    with pytest.raises(ValueError):
        canonical_amount_token(token)


def test_protocol_amounts_are_exact_above_binary_float_precision():
    sms = "INR 9007199254740992 debited from A/c XX0000. Available balance INR 9007199254740993."
    amounts = extract_protocol_candidates(sms).amounts
    assert [item.id for item in amounts] == ["A0", "A1"]
    assert [item.value for item in amounts] == [
        "9007199254740992",
        "9007199254740993",
    ]
    assert [item.start for item in amounts] == sorted(item.start for item in amounts)
    assert all(sms[item.start : item.end] == item.source_text for item in amounts)


def test_protocol_amount_dedupe_keeps_the_first_source_precision():
    sms = "INR 01.00 debited from A/c XX0000. Balance INR 1.0."
    amounts = extract_protocol_candidates(sms).amounts
    assert [(item.id, item.value, item.source_text) for item in amounts] == [
        ("A0", "1.00", "INR 01.00")
    ]


def test_protocol_counterparties_are_source_ordered_with_explicit_ties():
    sms = "INR 10 paid from A/c XX0000 to demo@bank on 01-JAN."
    counterparties = [
        item for item in extract_protocol_candidates(sms).counterparties if item.id != "PN"
    ]
    starts = [item.start for item in counterparties]
    assert starts == sorted(starts)
    assert [item.id for item in counterparties if str(item.value).casefold() == "demo@bank"] == [
        "PV0"
    ]
    assert COUNTERPARTY_SOURCE_TIE_BREAK_ORDER.index(
        "PV"
    ) < COUNTERPARTY_SOURCE_TIE_BREAK_ORDER.index("PT")


def test_candidate_oracle_and_reconstruction_are_source_grounded():
    candidates = extract_candidates(SMS)
    oracle = oracle_selection(GOLD, candidates)
    assert oracle.covered
    assert oracle.counterparty_id == "PA0"
    assert len(candidates.amounts) == 2
    assert candidates.counterparties[-1].id == "PN"

    target = selector_target(GOLD, candidates)
    assert target == {
        "transaction": 1,
        "type": "D",
        "amount": oracle.amount_id,
        "account": oracle.account_id,
        "counterparty": oracle.counterparty_id,
    }
    assert reconstruct_transaction(target, candidates) == GOLD


def test_null_counterparty_is_an_explicit_candidate():
    sms = "INR 500 withdrawn from A/c XX6254 on 01-01-2026."
    gold = {"amount": 500.0, "counterparty": None, "type": "debit", "account": "A/c XX6254"}
    candidates = extract_candidates(sms)
    target = selector_target(gold, candidates)
    assert target["counterparty"] == "PN"
    assert reconstruct_transaction(target, candidates)["counterparty"] is None


def test_counterparty_ids_encode_the_source_relation_instead_of_list_position():
    cases = {
        "INR 10 paid from A/c XX0000 to DEMO USER on 01-JAN.": "PT0",
        "INR 10 credited to A/c XX0000 by DEMO USER on 01-JAN.": "PB0",
        "INR 10 spent on Card XX0000 at DEMO SHOP on 01-JAN.": "PA0",
    }
    for sms, expected_id in cases.items():
        assert expected_id in {item.id for item in extract_candidates(sms).counterparties}


def test_null_target_and_bad_ids_are_strict():
    candidates = extract_candidates(SMS)
    assert selector_target(None, candidates) == {"transaction": 0}
    assert reconstruct_transaction({"transaction": 0}, candidates) is None
    with pytest.raises(ValueError, match="unknown amount"):
        reconstruct_transaction(
            {
                "transaction": 1,
                "type": "D",
                "amount": "A999",
                "account": "C0",
                "counterparty": "PN",
            },
            candidates,
        )


def test_selector_prompt_contains_only_candidate_ids_for_the_answer_contract():
    messages = candidate_selector_messages("VD-IDFCFB", SMS)
    assert [item["role"] for item in messages] == ["system", "user"]
    assert '"A0":23.0' in messages[1]["content"]
    assert '"PN":null' in messages[1]["content"]


def test_uncovered_gold_is_rejected_instead_of_becoming_hallucination_training():
    candidates = extract_candidates("INR 10 credited to A/c XX0000.")
    uncovered = {
        "amount": 11.0,
        "counterparty": None,
        "type": "credit",
        "account": "A/c XX0000",
    }
    oracle = oracle_selection(uncovered, candidates)
    assert not oracle.covered
    assert oracle.missing_fields == ("amount",)
    with pytest.raises(ValueError, match="amount"):
        selector_target(uncovered, candidates)


def test_selector_parser_reconstructs_and_rejects_out_of_set_values():
    candidates = extract_candidates(SMS)
    target = selector_target(GOLD, candidates)
    parsed = parse_selector_prediction(
        "answer: " + __import__("json").dumps(target) + " trailing text",
        candidates,
    )
    assert parsed.status == "transaction"
    assert parsed.value == GOLD
    assert parse_selector_prediction('{"transaction":0}', candidates).status == "null"
    assert parse_selector_prediction('{"transaction":false}', candidates).status == "invalid"
    assert (
        parse_selector_prediction(
            '{"transaction":1,"type":"D","amount":"A404","account":"C0","counterparty":"PN"}',
            candidates,
        ).status
        == "invalid"
    )


def test_hybrid_safety_recovers_unambiguous_vpa_null_and_currency_contamination():
    vpa_sms = "INR 10 credited to A/c XX0000 from VPA fixture@okdemo on 12-JAN."
    vpa_candidates = extract_candidates(vpa_sms)
    recovered, events = resolve_selector_prediction(
        '{"transaction":0}',
        vpa_candidates,
        hybrid_safety=True,
    )
    assert recovered.status == "transaction"
    assert recovered.value["counterparty"] == "fixture@okdemo"
    assert events == ("vpa_unambiguous_transaction_fallback",)

    candidates = extract_candidates(SMS)
    candidates = CandidateSet(
        amounts=candidates.amounts,
        accounts=candidates.accounts,
        counterparties=candidates.counterparties
        + (Candidate("PR0", "INR 5.00", "INR 5.00", 0, 8),),
        type_hints=candidates.type_hints,
    )
    target = selector_target(GOLD, candidates)
    target["counterparty"] = "PR0"
    resolved, events = resolve_selector_prediction(
        __import__("json").dumps(target),
        candidates,
        hybrid_safety=True,
    )
    assert resolved.value["counterparty"] == GOLD["counterparty"]
    assert events == ("counterparty_currency_contamination_override",)


def test_hybrid_safety_does_not_override_null_or_generic_distractor_choices():
    candidates = extract_candidates(SMS)
    target = selector_target(GOLD, candidates)
    target["counterparty"] = "PN"
    resolved, events = resolve_selector_prediction(
        __import__("json").dumps(target),
        candidates,
        hybrid_safety=True,
    )
    assert resolved.value["counterparty"] is None
    assert events == ()


def test_candidate_extraction_drops_currency_amount_counterparty_spans():
    sms = "INR 10 debited from A/c XX0000 to DEMO USER on 12-JAN for INR 5.00."
    candidates = extract_candidates(sms)
    assert all(
        "INR 5" not in str(item.value)
        for item in candidates.counterparties
        if item.value is not None
    )

    trailing_amount = extract_candidates("INR 10 spent on Card XX0000 at DEMO USER INR 5.00.")
    assert "DEMO USER" in {
        item.value for item in trailing_amount.counterparties if item.value is not None
    }

    repeated_relation = extract_candidates(
        "INR 10 credited to A/c XX0000 by INR 5 fee by DEMO USER on 12-JAN."
    )
    assert "DEMO USER" in {
        item.value for item in repeated_relation.counterparties if item.value is not None
    }
    assert all(
        not str(item.value).startswith("INR 5")
        for item in repeated_relation.counterparties
        if item.value is not None
    )


def test_canonical_amount_token_does_not_depend_on_host_integer_digit_limits():
    token = "0" * 5_000 + "7.00"
    assert canonical_amount_token(token) == "7.00"


def test_protocol_vpa_drops_terminal_sentence_punctuation_from_its_span():
    sms = "INR 10 credited to A/c XX0000 from fixture@okdemo."
    candidates = extract_protocol_candidates(sms)
    vpa = [item for item in candidates.counterparties if item.id.startswith("PV")]

    assert [(item.id, item.value, item.source_text) for item in vpa] == [
        ("PV0", "fixture@okdemo", "fixture@okdemo")
    ]
    assert sms[vpa[0].start : vpa[0].end] == vpa[0].source_text


def test_protocol_amounts_consume_the_complete_source_fraction():
    sms = "INR 1,23,456.0700 spent on Card XX1234 at DEMO SHOP."
    amount = extract_protocol_candidates(sms).amounts[0]

    assert (amount.value, amount.source_text) == (
        "123456.0700",
        "INR 1,23,456.0700",
    )
    assert sms[amount.start : amount.end] == amount.source_text


@pytest.mark.parametrize(
    "token",
    [
        "7.12.50",
        "1,2345.00",
        "12,3456",
        "1e3",
        "7.12abc",
        "1..",
    ],
)
def test_protocol_amounts_do_not_partially_accept_malformed_numeric_tokens(token):
    sms = f"INR {token} spent on Card XX1234 at DEMO SHOP."

    assert extract_protocol_candidates(sms).amounts == ()


def test_protocol_extraction_uses_ascii_classes_but_keeps_literal_rupee():
    rupee = extract_protocol_candidates(
        "\u20b910.1234 spent on Card XX1234 at DEMO SHOP."
    )
    assert [(item.value, item.source_text) for item in rupee.amounts] == [
        ("10.1234", "\u20b910.1234")
    ]

    kelvin = extract_protocol_candidates(
        "INR 10 spent on Card XX1234 at \u212aIOSK."
    )
    assert all("\u212a" not in str(item.value) for item in kelvin.counterparties)

    nonbreaking_amount_space = extract_protocol_candidates(
        "INR\u00a010 spent on Card XX1234 at DEMO SHOP."
    )
    assert nonbreaking_amount_space.amounts == ()

    nonbreaking_account_space = extract_protocol_candidates(
        "INR 10 spent on Card\u00a0XX1234 at DEMO SHOP."
    )
    assert nonbreaking_account_space.accounts == ()


def test_protocol_text_identity_does_not_trim_nonbreaking_spaces():
    plain = "DEMO ACCOUNT"
    nonbreaking = "\u00a0DEMO ACCOUNT\u00a0"
    candidates = _dedupe_source_text_candidates(
        [
            (plain, plain, 0, len(plain)),
            (
                nonbreaking,
                nonbreaking,
                len(plain) + 1,
                len(plain) + 1 + len(nonbreaking),
            ),
        ],
        "C",
    )

    assert [(item.id, item.value) for item in candidates] == [
        ("C0", plain),
        ("C1", nonbreaking),
    ]
