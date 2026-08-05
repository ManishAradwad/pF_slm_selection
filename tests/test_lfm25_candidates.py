import pytest

from lfm25.candidates import (
    Candidate,
    CandidateSet,
    candidate_selector_messages,
    extract_candidates,
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
    assert parse_selector_prediction(
        '{"transaction":1,"type":"D","amount":"A404","account":"C0","counterparty":"PN"}',
        candidates,
    ).status == "invalid"


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

    trailing_amount = extract_candidates(
        "INR 10 spent on Card XX0000 at DEMO USER INR 5.00."
    )
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
