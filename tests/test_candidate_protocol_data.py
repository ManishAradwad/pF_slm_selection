from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

import lfm25.candidate_protocol_data as candidate_data
from lfm25.candidate_protocol_data import (
    BUILDER_VERSION,
    CandidateProtocolDataError,
    _read_exact_jsonl,
    build_candidate_protocol_data,
    select_protocol_rows,
)


SMS = (
    "INR 23.00 spent on Credit Card ending XX0000 at DEMO SHOP DAILY "
    "on 03-JAN. Avbl limit: Rs.10000.00."
)
GOLD = {
    "amount": 23.0,
    "counterparty": "DEMO SHOP DAILY",
    "type": "debit",
    "account": "Credit Card ending XX0000",
}


def _row(record_hash: str, *, expected=GOLD, sms: str = SMS) -> dict:
    return {
        "sender": "AX-DEMOXX",
        "sms": sms,
        "expected": expected,
        "label_tier": "grounded_silver",
        "source": {
            "original_split": "train",
            "record_hash": record_hash,
            "private_sender_hash": f"sender-{record_hash}",
            "template_group": f"template-{record_hash}",
        },
        "provenance": {"android_prefilter_accepted": True},
    }


def _exact_source_line(
    record_hash: str,
    *,
    sms: str,
    amount_lexeme: str,
    counterparty: str,
    account: str,
) -> str:
    row = _row(
        record_hash,
        sms=sms,
        expected={
            "amount": "__EXACT_AMOUNT__",
            "counterparty": counterparty,
            "type": "debit",
            "account": account,
        },
    )
    serialized = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
    return serialized.replace('"__EXACT_AMOUNT__"', amount_lexeme)


def test_select_protocol_rows_materializes_exact_v1_target() -> None:
    accepted, report = select_protocol_rows([_row("one"), _row("two", expected=None)])

    assert report["input_rows"] == 2
    assert report["output_rows"] == 2
    assert report["exclusion_reasons"] == {}
    assert report["label_kind_counts"] == {"null": 1, "transaction": 1}
    assert accepted[0]["candidate_protocol_v1_target"].startswith(
        '{"transaction":1,"type":"D","amount":"A0"'
    )
    assert accepted[1]["candidate_protocol_v1_target"] == '{"transaction":0}'
    assert all(
        row["provenance"]["candidate_protocol_builder_version"] == BUILDER_VERSION
        for row in accepted
    )
    assert all(row["provenance"]["candidate_oracle_covered"] for row in accepted)


def test_select_protocol_rows_excludes_uncovered_and_non_train_rows() -> None:
    uncovered = _row(
        "uncovered",
        expected={**GOLD, "amount": 999.0},
    )
    wrong_split = _row("wrong-split")
    wrong_split["source"]["original_split"] = "test"

    accepted, report = select_protocol_rows([uncovered, wrong_split])

    assert accepted == []
    assert report["output_rows"] == 0
    assert report["exclusion_reasons"] == {
        "candidate_missing_amount": 1,
        "not_original_train": 1,
    }


def test_select_protocol_rows_rejects_duplicate_source_records() -> None:
    accepted, report = select_protocol_rows([_row("same"), _row("same")])

    assert len(accepted) == 1
    assert report["exclusion_reasons"] == {"missing_or_duplicate_record_hash": 1}


def test_select_protocol_rows_does_not_treat_a_missing_label_as_null() -> None:
    row = _row("missing-label")
    row.pop("expected")

    accepted, report = select_protocol_rows([row])

    assert accepted == []
    assert report["exclusion_reasons"] == {"missing_label": 1}


def test_select_protocol_rows_requires_prefilter_and_split_identities() -> None:
    false_prefilter = _row("false-prefilter")
    false_prefilter["provenance"]["android_prefilter_accepted"] = False
    missing_prefilter = _row("missing-prefilter")
    missing_prefilter["provenance"].pop("android_prefilter_accepted")
    rejected_by_prefilter = _row("rejected-by-prefilter")
    rejected_by_prefilter["sender"] = "9876543210"
    missing_sender_identity = _row("missing-sender-identity")
    missing_sender_identity["source"]["private_sender_hash"] = ""
    missing_template_identity = _row("missing-template-identity")
    missing_template_identity["source"]["template_group"] = None

    accepted, report = select_protocol_rows(
        [
            false_prefilter,
            missing_prefilter,
            missing_sender_identity,
            rejected_by_prefilter,
            missing_template_identity,
        ]
    )

    assert accepted == []
    assert report["output_rows"] == 0
    assert report["exclusion_reasons"] == {
        "android_prefilter_not_accepted": 3,
        "missing_split_identity": 2,
    }


def test_select_protocol_rows_preserves_exact_large_integer_gold_identity() -> None:
    sms = (
        "INR 9,007,199,254,740,992.00 debited from A/c XX5500 at QUARTZ LABS. "
        "Available balance INR 9,007,199,254,740,993.00."
    )
    exact = _row(
        "large-exact",
        sms=sms,
        expected={
            "amount": 9007199254740993,
            "counterparty": "QUARTZ LABS",
            "type": "debit",
            "account": "A/c XX5500",
        },
    )

    accepted, report = select_protocol_rows([exact])

    assert report["output_rows"] == 1
    assert report["exclusion_reasons"] == {}
    assert '"amount":"A1"' in accepted[0]["candidate_protocol_v1_target"]


def test_select_protocol_rows_rejects_ambiguous_binary_float_gold() -> None:
    sms = (
        "INR 9,007,199,254,740,992.00 debited from A/c XX5500 at QUARTZ LABS. "
        "Available balance INR 9,007,199,254,740,993.00."
    )
    lossy = _row(
        "large-lossy",
        sms=sms,
        expected={
            "amount": 9007199254740993.0,
            "counterparty": "QUARTZ LABS",
            "type": "debit",
            "account": "A/c XX5500",
        },
    )

    accepted, report = select_protocol_rows([lossy])

    assert accepted == []
    assert report["exclusion_reasons"] == {"candidate_missing_amount": 1}


def test_exact_source_fraction_is_selected_before_json_native_projection(tmp_path: Path) -> None:
    sms = "INR 12.3456 debited from A/c XX0000 at DEMO STORE."
    path = tmp_path / "source.jsonl"
    path.write_text(
        _exact_source_line(
            "fractional",
            sms=sms,
            amount_lexeme="12.3456",
            counterparty="DEMO STORE",
            account="A/c XX0000",
        )
        + "\n",
        encoding="utf-8",
    )

    rows = _read_exact_jsonl(path)
    assert rows[0]["expected"]["amount"] == Decimal("12.3456")

    accepted, report = select_protocol_rows(rows)

    assert report["output_rows"] == 1
    assert accepted[0]["expected"]["amount"] == 12.3456
    assert '"amount":"A0"' in accepted[0]["candidate_protocol_v1_target"]
    assert json.loads(json.dumps(accepted[0], allow_nan=False)) == accepted[0]


def test_exact_fractional_float_collision_materializes_a1(tmp_path: Path) -> None:
    sms = (
        "INR 9,007,199,254,740,992.00 debited from A/c XX5500 at QUARTZ LABS. "
        "Available balance INR 9,007,199,254,740,993.00."
    )
    path = tmp_path / "source.jsonl"
    path.write_text(
        _exact_source_line(
            "fractional-collision",
            sms=sms,
            amount_lexeme="9007199254740993.00",
            counterparty="QUARTZ LABS",
            account="A/c XX5500",
        )
        + "\n",
        encoding="utf-8",
    )

    rows = _read_exact_jsonl(path)
    assert rows[0]["expected"]["amount"] == Decimal("9007199254740993.00")

    accepted, report = select_protocol_rows(rows)

    assert report["output_rows"] == 1
    assert accepted[0]["expected"]["amount"] == 9007199254740992.0
    assert '"amount":"A1"' in accepted[0]["candidate_protocol_v1_target"]


@pytest.mark.parametrize(
    "payload",
    [
        '{"marker":"PRIVATE_SENTINEL","marker":"forged"}',
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":-Infinity}',
        '{"unterminated":',
    ],
)
def test_exact_source_parser_fails_closed_without_echoing_private_data(
    tmp_path: Path,
    payload: str,
) -> None:
    path = tmp_path / "source.jsonl"
    path.write_text(payload + "\n", encoding="utf-8")

    with pytest.raises(CandidateProtocolDataError) as raised:
        _read_exact_jsonl(path)

    assert str(raised.value) == "Candidate Protocol V1 private source could not be parsed"
    assert "PRIVATE_SENTINEL" not in str(raised.value)


def test_builder_fingerprints_the_explicit_implementation_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "PRIVATE_DATA" / "lfm25" / "source"
    source.mkdir(parents=True)
    for split in ("train", "dev"):
        (source / f"private_sft_v2_{split}.jsonl").write_text(
            json.dumps(_row(split), ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    implementation = tmp_path / "implementation"
    (implementation / "lfm25").mkdir(parents=True)
    extractor = implementation / "lfm25" / "candidates.py"
    protocol = implementation / "lfm25" / "candidate_protocol.py"
    extractor.write_bytes(b"synthetic extractor\n")
    protocol.write_bytes(b"synthetic protocol\n")
    monkeypatch.setattr(candidate_data, "require_private_ignore", lambda *_args: None)

    result = build_candidate_protocol_data(
        repo_root=tmp_path,
        input_dir=source,
        output_dir=tmp_path / "PRIVATE_DATA" / "lfm25" / "output",
        implementation_root=implementation,
        dry_run=True,
    )

    fingerprints = result["report"]["candidate_implementations"]
    assert fingerprints["extractor"]["sha256"] == hashlib.sha256(
        extractor.read_bytes()
    ).hexdigest()
    assert fingerprints["protocol"]["sha256"] == hashlib.sha256(
        protocol.read_bytes()
    ).hexdigest()
