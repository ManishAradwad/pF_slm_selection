from __future__ import annotations

from copy import deepcopy

from lfm25.private_data import empty_consensus
from scripts.propose_lfm25_private_labels import (
    _merge_proposals,
    select_proposal_records,
)


POLICY = {
    "policy_version": "test-v1",
    "eligible_splits": ["train", "dev"],
    "minimum_proposals": 3,
    "minimum_agreeing_models": 2,
    "minimum_independent_model_families": 2,
    "minimum_proposal_confidence": 0.9,
}


def _row(index: int, split: str, *, transaction: bool, category: str | None = None):
    return {
        "record_hash": f"record-{index:03d}",
        "split": split,
        "silver_label": {
            "amount": 10.0,
            "counterparty": None,
            "type": "debit",
            "account": "A/c XX0001",
        } if transaction else None,
        "hard_negative_category": category,
        "confidence": 0.95 if transaction else 0.6,
        "consensus_acceptance": empty_consensus(POLICY),
    }


def _proposal(model: str, family: str, label):
    return {
        "model_id": model,
        "model_family": family,
        "label": label,
        "confidence": 0.99,
        "inference_config_hash": f"config-{model}",
        "schema_valid": True,
    }


def test_selection_includes_all_eligible_transactions_and_blinds_test():
    rows = [
        _row(1, "train", transaction=True),
        _row(2, "dev", transaction=True),
        _row(3, "test", transaction=True),
        _row(4, "train", transaction=False, category="otp"),
        _row(5, "dev", transaction=False, category="failed"),
        _row(6, "train", transaction=False, category="otp"),
    ]
    selected = select_proposal_records(
        rows,
        {
            "eligible_splits": ["train", "dev"],
            "include_all_heuristic_transactions": True,
            "maximum_rows": 4,
            "test_rows_are_blinded": True,
        },
    )
    hashes = {row["record_hash"] for row in selected}
    assert {"record-001", "record-002"}.issubset(hashes)
    assert "record-003" not in hashes
    assert len(selected) == 4


def test_merge_requires_cross_family_consensus_and_prioritizes_disagreement():
    label = _row(1, "train", transaction=True)["silver_label"]
    records = [
        _row(1, "train", transaction=True),
        _row(2, "dev", transaction=False, category="otp"),
        _row(3, "test", transaction=False, category="failed"),
    ]
    proposals = [
        {
            "record-001": _proposal("gemma-q4", "gemma", label),
            "record-002": _proposal("gemma-q4", "gemma", None),
        },
        {
            "record-001": _proposal("gemma-q8", "gemma", label),
            "record-002": _proposal("gemma-q8", "gemma", None),
        },
        {
            "record-001": _proposal("qwen-q8", "qwen", label),
            "record-002": _proposal("qwen-q8", "qwen", label),
        },
    ]
    merged, queue = _merge_proposals(
        deepcopy(records),
        records[:2],
        proposals,
        POLICY,
    )
    by_hash = {row["record_hash"]: row for row in merged}
    assert by_hash["record-001"]["consensus_acceptance"]["accepted"] is True
    assert by_hash["record-002"]["consensus_acceptance"]["accepted"] is False
    assert by_hash["record-003"]["consensus_acceptance"]["status"] == (
        "not_selected_for_local_proposals"
    )
    queue_by_hash = {row["record_hash"]: row for row in queue}
    assert queue_by_hash["record-003"]["review_priority"] == 0
    assert queue_by_hash["record-002"]["review_priority"] == 1


def test_selection_rejects_limit_smaller_than_transaction_pool():
    rows = [_row(index, "train", transaction=True) for index in range(3)]
    try:
        select_proposal_records(
            rows,
            {
                "eligible_splits": ["train", "dev"],
                "include_all_heuristic_transactions": True,
                "maximum_rows": 2,
                "test_rows_are_blinded": True,
            },
        )
    except ValueError as error:
        assert "smaller" in str(error)
    else:
        raise AssertionError("expected the too-small selection cap to fail")
