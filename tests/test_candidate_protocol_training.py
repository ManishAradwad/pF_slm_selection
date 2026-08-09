import json
from pathlib import Path

import pytest

from lfm25.candidate_protocol_data import _read_exact_jsonl, select_protocol_rows
from lfm25.candidate_protocol import (
    build_protocol_request,
    candidate_protocol_messages,
    serialize_selector_target,
)
from scripts.train_lfm25_candidate_protocol_v1 import (
    _has_option,
    candidate_v1_messages,
)


SMS = "INR 10 debited from A/c XX0000 at DEMO STORE."
GOLD = {
    "amount": 10.0,
    "counterparty": "DEMO STORE",
    "type": "debit",
    "account": "A/c XX0000",
}


def _materialized_row(
    tmp_path: Path,
    *,
    sms: str,
    amount_lexeme: str,
    counterparty: str,
    account: str,
) -> dict:
    row = {
        "sender": "AX-DEMOXX",
        "sms": sms,
        "expected": {
            "amount": "__EXACT_AMOUNT__",
            "counterparty": counterparty,
            "type": "debit",
            "account": account,
        },
        "label_tier": "synthetic",
        "source": {
            "original_split": "train",
            "record_hash": "synthetic-record",
            "private_sender_hash": "synthetic-sender",
            "template_group": "synthetic-template",
        },
        "provenance": {"android_prefilter_accepted": True},
    }
    line = json.dumps(row, ensure_ascii=False, separators=(",", ":")).replace(
        '"__EXACT_AMOUNT__"',
        amount_lexeme,
    )
    path = tmp_path / "source.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    accepted, report = select_protocol_rows(_read_exact_jsonl(path))
    assert report["output_rows"] == 1
    return json.loads(json.dumps(accepted[0], allow_nan=False))


def test_training_messages_use_the_exact_v1_contract() -> None:
    request = build_protocol_request("AX-DEMOXX", SMS)
    target = serialize_selector_target(GOLD, request)
    row = {
        "sender": "AX-DEMOXX",
        "sms": SMS,
        "expected": GOLD,
        "candidate_protocol_v1_target": target,
    }
    messages = candidate_v1_messages(row)

    assert messages[:2] == candidate_protocol_messages(request)
    assert messages[-1] == {
        "role": "assistant",
        "content": target,
    }


def test_training_rejects_materialized_target_drift() -> None:
    row = {
        "sender": "AX-DEMOXX",
        "sms": SMS,
        "expected": GOLD,
        "candidate_protocol_v1_target": '{"transaction":0}',
    }
    with pytest.raises(ValueError, match="differs from current contract"):
        candidate_v1_messages(row)


def test_training_requires_the_materialized_target() -> None:
    with pytest.raises(ValueError, match="require a materialized selector target"):
        candidate_v1_messages({"sender": "AX-DEMOXX", "sms": SMS, "expected": GOLD})


def test_fractional_builder_target_survives_into_training_messages(tmp_path: Path) -> None:
    row = _materialized_row(
        tmp_path,
        sms="INR 12.3456 debited from A/c XX0000 at DEMO STORE.",
        amount_lexeme="12.3456",
        counterparty="DEMO STORE",
        account="A/c XX0000",
    )

    messages = candidate_v1_messages(row)

    assert row["expected"]["amount"] == 12.3456
    assert '"amount":"A0"' in row["candidate_protocol_v1_target"]
    assert messages[-1] == {
        "role": "assistant",
        "content": row["candidate_protocol_v1_target"],
    }


def test_float_collision_a1_builder_target_survives_training_messages(
    tmp_path: Path,
) -> None:
    row = _materialized_row(
        tmp_path,
        sms=(
            "INR 9,007,199,254,740,992.00 debited from A/c XX5500 at QUARTZ LABS. "
            "Available balance INR 9,007,199,254,740,993.00."
        ),
        amount_lexeme="9007199254740993.00",
        counterparty="QUARTZ LABS",
        account="A/c XX5500",
    )

    messages = candidate_v1_messages(row)

    assert row["expected"]["amount"] == 9007199254740992.0
    assert '"amount":"A1"' in row["candidate_protocol_v1_target"]
    assert messages[-1] == {
        "role": "assistant",
        "content": row["candidate_protocol_v1_target"],
    }


@pytest.mark.parametrize(
    "materialized",
    [
        None,
        1,
        "not-json",
        '{"transaction":0}\n',
        '{"transaction":0,"transaction":0}',
    ],
)
def test_training_rejects_malformed_materialized_targets(materialized: object) -> None:
    row = {
        "sender": "AX-DEMOXX",
        "sms": SMS,
        "expected": GOLD,
        "candidate_protocol_v1_target": materialized,
    }

    with pytest.raises(ValueError, match="materialized Candidate Protocol V1 target"):
        candidate_v1_messages(row)


def test_training_rejects_ungrounded_materialized_target() -> None:
    request = build_protocol_request("AX-DEMOXX", SMS)
    forged = serialize_selector_target(GOLD, request).replace(
        '"amount":"A0"',
        '"amount":"A99"',
    )
    row = {
        "sender": "AX-DEMOXX",
        "sms": SMS,
        "expected": GOLD,
        "candidate_protocol_v1_target": forged,
    }

    with pytest.raises(ValueError, match="violates the current contract"):
        candidate_v1_messages(row)


def test_training_rejects_forged_grounded_amount_selection() -> None:
    sms = "INR 10 debited from A/c XX0000 at DEMO STORE. Available balance INR 99."
    request = build_protocol_request("AX-DEMOXX", sms)
    canonical = serialize_selector_target(GOLD, request)
    assert '"amount":"A0"' in canonical
    forged = canonical.replace('"amount":"A0"', '"amount":"A1"')
    row = {
        "sender": "AX-DEMOXX",
        "sms": sms,
        "expected": GOLD,
        "candidate_protocol_v1_target": forged,
    }

    with pytest.raises(ValueError, match="differs from current contract"):
        candidate_v1_messages(row)


def test_v1_wrapper_detects_forbidden_profile_overrides() -> None:
    assert _has_option(["--contract", "android"], "--prompt-profile", "--contract")
    assert _has_option(["--prompt-profile=legacy"], "--prompt-profile", "--contract")
    assert not _has_option(["--max-length", "1024"], "--prompt-profile", "--contract")
