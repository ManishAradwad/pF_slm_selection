from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lfm25.android_contract import prefilter_sms
from lfm25.private_data import PrivateDataError, file_sha256
from lfm25.private_sft_v2 import (
    BuildConfig,
    PrivateSFTV2Error,
    _resolve_paths,
    build_private_sft_v2,
    grounding_errors,
    run_private_sft_v2,
    safe_console_summary,
)


def _label(
    amount: float,
    account: str,
    counterparty: str | None,
    transaction_type: str = "debit",
) -> dict:
    return {
        "amount": amount,
        "counterparty": counterparty,
        "type": transaction_type,
        "account": account,
    }


def _sms(index: int, *, transaction_type: str = "debit") -> str:
    verb = "debited" if transaction_type == "debit" else "credited"
    return (
        f"INR {index}.00 {verb} from A/c XX{index:04d} "
        f"at SYNTHETIC SHOP {index} on 01/01/2026."
    )


def _base_row(
    index: int,
    *,
    split: str = "train",
    sender_hash: str | None = None,
    template_group: str | None = None,
    sms: str | None = None,
) -> dict:
    message = sms or _sms(index)
    return {
        "record_hash": f"record-{index:04d}",
        "split": split,
        "sender": f"ZZ-SYNTH-{index}",
        "sms": message,
        "template_group": template_group or f"tg-{index:04d}",
        "private_hashes": {"sender": sender_hash or f"sender-{index:04d}"},
        "provenance": {
            "source_file": "synthetic_fixture.json",
            "source_row": index,
            "source_file_sha256": "a" * 64,
        },
        "review_status": "pending",
        "human_label": None,
        "human_reviewer": None,
        "human_reviewed_at": None,
        "consensus_acceptance": {
            "status": "not_evaluated",
            "accepted": False,
            "accepted_label": None,
            "valid_proposal_count": 0,
            "agreeing_model_count": 0,
            "independent_model_family_count": 0,
        },
        "local_model_proposals": [],
        "silver_label": _label(index, f"A/c XX{index:04d}", f"SYNTHETIC SHOP {index}"),
        "confidence": 0.9,
        "heuristic_reason_codes": ["synthetic_grounded_fixture"],
        "hard_negative_category": None,
    }


def _accept(_sender: str, _sms_text: str) -> SimpleNamespace:
    return SimpleNamespace(accepted=True, rejection_stage=None, rejection_reason=None)


def _test_contract_provenance() -> dict:
    return {"name": "synthetic-test-contract", "version": 1, "prefilter_sha256": "b" * 64}


def test_grounding_requires_amount_account_counterparty_and_type() -> None:
    sms = "INR 12.50 debited from A/c XX1234 at SYNTHETIC CAFE."
    assert grounding_errors(sms, _label(12.5, "A/c XX1234", "SYNTHETIC CAFE")) == ()
    assert grounding_errors(sms, None) == ()
    assert "amount_not_grounded" in grounding_errors(
        sms, _label(99, "A/c XX1234", "SYNTHETIC CAFE")
    )
    assert "account_not_grounded" in grounding_errors(
        sms, _label(12.5, "A/c XX9999", "SYNTHETIC CAFE")
    )
    assert "counterparty_not_grounded" in grounding_errors(
        sms, _label(12.5, "A/c XX1234", "ABSENT MERCHANT")
    )
    assert "counterparty_contains_currency_amount" in grounding_errors(
        "INR 12.50 debited from A/c XX1234 by INR 2 FEE.",
        _label(12.5, "A/c XX1234", "INR 2 FEE"),
    )
    assert "type_not_grounded" in grounding_errors(
        sms, _label(12.5, "A/c XX1234", "SYNTHETIC CAFE", "credit")
    )


def test_builder_seals_original_dev_test_and_preserves_label_tiers() -> None:
    human = _base_row(1, sender_hash="shared", template_group="tg-a")
    human.update(
        {
            "review_status": "human_approved",
            "human_label": human["silver_label"],
            "human_reviewer": "fixture-reviewer",
            "human_reviewed_at": "2026-01-01T00:00:00+00:00",
        }
    )
    consensus = _base_row(2, sender_hash="sender-b", template_group="tg-b")
    consensus_label = consensus["silver_label"]
    consensus.update(
        {
            "consensus_acceptance": {
                "policy_version": "fixture-consensus-v1",
                "status": "accepted",
                "accepted": True,
                "accepted_label": consensus_label,
                "valid_proposal_count": 3,
                "agreeing_model_count": 2,
                "independent_model_family_count": 2,
            },
            "local_model_proposals": [
                {
                    "model_id": "fixture-a",
                    "model_family": "family-a",
                    "label": consensus_label,
                    "confidence": 0.98,
                },
                {
                    "model_id": "fixture-b",
                    "model_family": "family-b",
                    "label": consensus_label,
                    "confidence": 0.96,
                },
                {
                    "model_id": "fixture-c",
                    "model_family": "family-c",
                    "label": None,
                    "confidence": 0.97,
                },
            ],
        }
    )
    silver = _base_row(3, sender_hash="shared", template_group="tg-c")
    fourth = _base_row(4, sender_hash="sender-d", template_group="tg-d")
    original_dev = _base_row(90, split="dev", sms="SYNTHETIC ORIGINAL DEV SECRET")
    original_test = _base_row(91, split="test", sms="SYNTHETIC SEALED TEST SECRET")
    calls: list[str] = []

    def recording_prefilter(sender: str, sms_text: str) -> SimpleNamespace:
        calls.append(sms_text)
        return _accept(sender, sms_text)

    records = [human, consensus, silver, fourth, original_dev, original_test]
    result = build_private_sft_v2(
        records,
        manifest_sha256="c" * 64,
        config=BuildConfig(max_per_template=10, max_per_category=10),
        prefilter=recording_prefilter,
        android_provenance=_test_contract_provenance(),
    )

    assert len(calls) == 4
    assert original_dev["sms"] not in calls
    assert original_test["sms"] not in calls
    output = result.train_rows + result.dev_rows
    assert {row["label_tier"] for row in output} == {
        "human_gold",
        "consensus_silver",
        "grounded_silver",
    }
    assert all(row["source"]["original_split"] == "train" for row in output)
    encoded = json.dumps(output)
    assert "SYNTHETIC ORIGINAL DEV SECRET" not in encoded
    assert "SYNTHETIC SEALED TEST SECRET" not in encoded
    assert result.report["sealed_test_assertion"]["passed"] is True
    assert result.report["sealed_test_assertion"]["original_test_rows_materialized"] == 0
    assert result.report["invariants"]["silver_dev_is_not_claimed_as_gold"] is True
    assert result.report["invariants"]["sealed_test_exclusion_passed"] is True
    assert result.report["invariants"]["original_test_rows_materialized_count"] == 0
    assert set(result.report["split_label_kind_counts"]) == {"train", "dev"}
    assert result.report["dev_evaluation"]["gold_benchmark_claimed"] is False
    for row in result.dev_rows:
        if row["label_tier"] != "human_gold":
            assert row["provenance"]["label"]["gold"] is False

    train_templates = {row["source"]["template_group"] for row in result.train_rows}
    dev_templates = {row["source"]["template_group"] for row in result.dev_rows}
    train_senders = {row["source"]["private_sender_hash"] for row in result.train_rows}
    dev_senders = {row["source"]["private_sender_hash"] for row in result.dev_rows}
    assert train_templates.isdisjoint(dev_templates)
    assert train_senders.isdisjoint(dev_senders)


def test_builder_is_order_independent_caps_templates_and_falls_back_to_silver() -> None:
    first = _base_row(1, template_group="tg-repeated", sender_hash="sender-a")
    first["consensus_acceptance"] = {
        "policy_version": "fixture",
        "status": "accepted",
        "accepted": True,
        "accepted_label": _label(999, "A/c XX0001", "SYNTHETIC SHOP 1"),
        "valid_proposal_count": 3,
        "agreeing_model_count": 2,
        "independent_model_family_count": 2,
    }
    first["local_model_proposals"] = [
        {
            "model_id": "a",
            "model_family": "a",
            "label": first["consensus_acceptance"]["accepted_label"],
            "confidence": 0.99,
        },
        {
            "model_id": "b",
            "model_family": "b",
            "label": first["consensus_acceptance"]["accepted_label"],
            "confidence": 0.98,
        },
    ]
    repeated = _base_row(2, template_group="tg-repeated", sender_hash="sender-a")
    rows = [
        first,
        repeated,
        _base_row(3, template_group="tg-three", sender_hash="sender-c"),
        _base_row(4, template_group="tg-four", sender_hash="sender-d"),
    ]
    config = BuildConfig(max_per_template=1, max_per_category=10, seed=7)
    first_result = build_private_sft_v2(
        rows,
        manifest_sha256="d" * 64,
        config=config,
        prefilter=_accept,
        android_provenance=_test_contract_provenance(),
    )
    reversed_result = build_private_sft_v2(
        list(reversed(rows)),
        manifest_sha256="d" * 64,
        config=config,
        prefilter=_accept,
        android_provenance=_test_contract_provenance(),
    )

    def hashes_by_split(result) -> dict[str, set[str]]:
        return {
            "train": {row["source"]["record_hash"] for row in result.train_rows},
            "dev": {row["source"]["record_hash"] for row in result.dev_rows},
        }

    assert hashes_by_split(first_result) == hashes_by_split(reversed_result)
    assert first_result.report["exclusion_reasons"]["template_cap"] == 1
    kept_first = [
        row
        for row in first_result.train_rows + first_result.dev_rows
        if row["source"]["record_hash"] == first["record_hash"]
    ]
    assert kept_first and kept_first[0]["label_tier"] == "grounded_silver"
    assert any(
        reason.startswith("consensus_silver:amount_not_grounded")
        for reason in first_result.report["label_candidate_rejections"]
    )


def test_real_android_prefilter_is_required_for_model_facing_rows() -> None:
    accepted = prefilter_sms("ZZ-SYNTH", _sms(7))
    rejected = prefilter_sms("ZZ-SYNTH", "Synthetic reminder without a transaction amount.")
    assert accepted.accepted is True
    assert rejected.accepted is False
    assert rejected.rejection_stage == "currency_amount"


def test_run_dry_run_writes_nothing_and_non_dry_run_hashes_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    private_root = repo_root / "PRIVATE_DATA" / "lfm25"
    private_root.mkdir(parents=True)
    manifest = private_root / "split_manifest.jsonl"
    rows = [_base_row(index) for index in range(1, 5)]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output_dir = private_root / "private_sft_v2"
    monkeypatch.setattr("lfm25.private_sft_v2.require_private_ignore", lambda *_args: None)

    dry_result = run_private_sft_v2(
        repo_root=repo_root,
        manifest_path=Path("PRIVATE_DATA/lfm25/split_manifest.jsonl"),
        output_dir=Path("PRIVATE_DATA/lfm25/private_sft_v2"),
        config=BuildConfig(max_per_template=10, max_per_category=10),
        dry_run=True,
    )
    assert dry_result["dry_run"] is True
    assert not output_dir.exists()
    summary = safe_console_summary(dry_result)
    assert set(summary["artifact_hashes"]) == {"train", "dev"}
    assert set(summary["split_label_kind_counts"]) == {"train", "dev"}
    assert summary["dev_evaluation"]["role"] == "silver_tuning_split_not_gold_benchmark"
    assert "sms" not in json.dumps(summary).casefold()

    written = run_private_sft_v2(
        repo_root=repo_root,
        manifest_path=Path("PRIVATE_DATA/lfm25/split_manifest.jsonl"),
        output_dir=Path("PRIVATE_DATA/lfm25/private_sft_v2"),
        config=BuildConfig(max_per_template=10, max_per_category=10),
    )
    assert written["wrote_files"] is True
    for split in ("train", "dev"):
        path = output_dir / f"private_sft_v2_{split}.jsonl"
        assert file_sha256(path) == written["metadata"]["artifacts"][split]["sha256"]


def test_guardrail_rejects_output_outside_private_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    private_root = repo_root / "PRIVATE_DATA" / "lfm25"
    private_root.mkdir(parents=True)
    manifest = private_root / "split_manifest.jsonl"
    manifest.write_text("", encoding="utf-8")
    with pytest.raises((PrivateSFTV2Error, PrivateDataError), match="outside"):
        _resolve_paths(repo_root, manifest, repo_root / "TRAINING_ARTIFACTS")
