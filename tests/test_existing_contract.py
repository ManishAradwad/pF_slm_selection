import json

from DATA import utils


def test_prompt_contract_places_the_synthetic_query_last() -> None:
    document = {
        "sender": "ZZ-TESTBK",
        "sms": "Synthetic notice: Rs.12.50 debited from Card XX0000 at DEMO STORE.",
    }

    prompt = utils.doc_to_text(document)
    task = prompt.split("### YOUR TASK", maxsplit=1)[1]

    assert "Sender: ZZ-TESTBK" in task
    assert document["sms"] in task
    assert prompt.endswith("Output: ")


def test_response_filter_removes_reasoning_and_keeps_the_json_object() -> None:
    expected = {
        "amount": 12.5,
        "counterparty": "DEMO STORE",
        "type": "debit",
        "account": "Card XX0000",
    }
    response = f"<think>synthetic reasoning</think>\nResult: {json.dumps(expected)}"

    filtered = utils.extract_json_filter([[response]], [{}])

    assert json.loads(filtered[0][0]) == expected


def test_nonnull_filter_rejects_missing_required_synthetic_field() -> None:
    incomplete = {
        "amount": 12.5,
        "counterparty": "DEMO STORE",
        "type": "debit",
        "account": None,
    }

    filtered = utils.extract_json_nonnull_filter([[json.dumps(incomplete)]], [{}])

    assert filtered == [["null"]]


def test_full_match_contract_normalizes_synthetic_account_formatting() -> None:
    reference = json.dumps(
        {
            "amount": 12.5,
            "counterparty": "DEMO STORE",
            "type": "debit",
            "account": "Card XX0000",
        }
    )
    prediction = json.dumps(
        {
            "amount": "12.50",
            "counterparty": "demo store",
            "type": "DEBIT",
            "account": "Credit Card ending 0000",
        }
    )

    assert utils.full_match_accuracy([reference], [prediction]) == 1.0
