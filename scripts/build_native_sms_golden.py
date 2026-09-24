#!/usr/bin/env python3
"""Render the wholly invented native parity bundle to canonical JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pocketfinancer_sms.analyzer import (  # noqa: E402
    ANALYSIS_CONTRACT_V2,
    DeterministicSmsAnalyzer,
)
from pocketfinancer_sms.currency import CurrencyContext  # noqa: E402
from pocketfinancer_sms.provenance import canonical_json_bytes  # noqa: E402
from pocketfinancer_sms.selector import model_candidate_payload  # noqa: E402
from pocketfinancer_sms.triage import evaluate_triage  # noqa: E402
from pocketfinancer_sms.types import TimestampProvenance  # noqa: E402


CASES = (
    (
        "unicode-fullwidth",
        "11111111-1111-4111-8111-111111111111",
        "ＩＮＲ １０ was credited to account **１２３４ from SYNTH FRIEND.",
        True,
    ),
    (
        "escaped-and-multiclause",
        "22222222-2222-4222-8222-222222222222",
        "INR 12 was paid at SYNTH \"STORE\".\nAvailable balance INR 40.",
        True,
    ),
    (
        "explicit-optional-absence",
        "33333333-3333-4333-8333-333333333333",
        "INR 10 was paid.",
        True,
    ),
    (
        "repeated-amounts-first-operation",
        "44444444-4444-4444-8444-444444444444",
        "INR 20 was debited from account **1234 at SYNTH SHOP. Available balance INR 20.",
        True,
    ),
    (
        "repeated-amounts-second-operation",
        "55555555-5555-4555-8555-555555555555",
        "INR 20 was debited from account **1234 at SYNTH SHOP. Available balance INR 20.",
        True,
    ),
    (
        "expected-refund",
        "66666666-6666-4666-8666-666666666666",
        "A refund of INR 75 is expected within five days for account **1234.",
        True,
    ),
    (
        "empty-invalid-input",
        "77777777-7777-4777-8777-777777777777",
        "",
        False,
    ),
)


def build_fixture() -> dict:
    vectors = []
    for vector_id, operation_id, source, input_valid in CASES:
        analysis = DeterministicSmsAnalyzer(
            CurrencyContext("INR", ("core-en", "india")),
            analysis_contract=ANALYSIS_CONTRACT_V2,
        ).analyze(
            source,
            operation_id=operation_id,
            input_valid=input_valid,
            is_outgoing=False,
            operation_config_hash="a" * 64,
            source_timestamp_epoch_ms=1_700_000_000_000,
            source_timestamp_provenance=(
                TimestampProvenance.ACQUISITION_SUPPLIED_MESSAGE_TIME
            ),
        )
        triage = evaluate_triage(analysis)
        vectors.append(
            {
                "id": vector_id,
                "operation_id": operation_id,
                "source": source,
                "input_valid": input_valid,
                "operation_config_hash": "a" * 64,
                "expected_analysis_json": canonical_json_bytes(analysis.to_dict()).decode(),
                "expected_selector_input_json": canonical_json_bytes(
                    model_candidate_payload(source, analysis)
                ).decode(),
                "expected_triage": {
                    "disposition": triage.disposition.value,
                    "selector_action": triage.selector_action.value,
                    "reason_codes": list(triage.reason_codes),
                },
            }
        )
    return {
        "contract": "pocketfinancer.native-parity-golden/1",
        "fixture_class": "wholly_invented_synthetic",
        "analysis_contract": ANALYSIS_CONTRACT_V2,
        "vectors": vectors,
    }


def main() -> None:
    print(json.dumps(build_fixture(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
