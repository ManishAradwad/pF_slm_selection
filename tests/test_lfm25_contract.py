from __future__ import annotations

import json

import pytest

from lfm25.contract import (
    canonical_transaction,
    counterparty_matches,
    normalize_account,
    parse_gold,
    parse_prediction,
    transaction_matches,
)


TRANSACTION = {
    "amount": 125.0,
    "counterparty": "DEMO STORE",
    "type": "debit",
    "account": "A/c XX1234",
}


def test_prediction_contract_accepts_only_null_or_exact_four_fields() -> None:
    assert parse_prediction("null").status == "null"
    parsed = parse_prediction(json.dumps(TRANSACTION))
    assert parsed.status == "transaction"
    assert parsed.value == TRANSACTION

    with_date = {**TRANSACTION, "date": "01-01-2026"}
    assert parse_prediction(json.dumps(with_date)).status == "invalid"
    assert parse_prediction('{"amount":125,"counterparty":null,"type":"debit"}').status == "invalid"
    assert parse_prediction("not json").status == "invalid"


def test_gold_can_drop_legacy_date_but_not_missing_contract_fields() -> None:
    legacy = {**TRANSACTION, "date": "01-01-2026"}
    assert parse_gold(json.dumps(legacy)) == TRANSACTION
    with pytest.raises(ValueError):
        parse_gold('{"amount":125,"type":"debit","account":"A/c XX1234"}')


def test_required_values_are_nonnull_and_typed() -> None:
    assert canonical_transaction({**TRANSACTION, "amount": True}) is None
    assert canonical_transaction({**TRANSACTION, "account": "null"}) is None
    assert canonical_transaction({**TRANSACTION, "type": "pending"}) is None
    assert canonical_transaction({**TRANSACTION, "counterparty": None}) is not None


def test_repository_matching_rules_are_preserved() -> None:
    predicted = {
        "amount": 125,
        "counterparty": "UPI-DEMO STORE",
        "type": "DEBIT",
        "account": "A/C XXXXX1234",
    }
    assert normalize_account("A/c XX1234") == ("account", "1234")
    assert counterparty_matches("DEMO STORE", "UPI-DEMO STORE")
    assert transaction_matches(TRANSACTION, predicted)
