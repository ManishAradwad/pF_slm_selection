from lfm25.candidate_sft import BUILDER_VERSION, select_grounded_rows


def _row(index, expected):
    return {
        "sender": "AX-BANKXX",
        "sms": "INR 23.00 spent on Credit Card XX0000 at DEMO SHOP on 01-JAN.",
        "expected": expected,
        "sample_weight": 0.8,
        "label_tier": "grounded_silver",
        "source": {
            "original_split": "train",
            "record_hash": f"{index:064x}",
            "template_group": f"template-{index}",
            "private_sender_hash": f"sender-{index}",
        },
    }


def test_candidate_builder_keeps_only_oracle_covered_rows():
    gold = {
        "amount": 23.0,
        "counterparty": "DEMO SHOP",
        "type": "debit",
        "account": "Credit Card XX0000",
    }
    missing = {**gold, "counterparty": "ABSENT MERCHANT"}
    rows, report = select_grounded_rows([_row(1, gold), _row(2, missing), _row(3, None)])
    assert len(rows) == 2
    assert report["label_kind_counts"] == {"null": 1, "transaction": 1}
    assert report["exclusion_reasons"] == {"candidate_missing_counterparty": 1}
    assert all(row["provenance"]["candidate_oracle_covered"] for row in rows)
    assert all(
        row["provenance"]["candidate_builder_version"] == BUILDER_VERSION
        for row in rows
    )


def test_candidate_builder_rejects_non_train_provenance():
    row = _row(1, None)
    row["source"]["original_split"] = "test"
    rows, report = select_grounded_rows([row])
    assert rows == []
    assert report["exclusion_reasons"] == {"not_original_train": 1}
