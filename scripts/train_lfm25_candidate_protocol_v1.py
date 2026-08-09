#!/usr/bin/env python3
"""Train the versioned Candidate Protocol V1 with the shared LoRA trainer.

The historical ``candidate_selector`` profile remains frozen for provenance.
This entry point injects the V1 prompt, target serializer, and contract evidence
while reusing the current completion-only training implementation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lfm25.candidate_protocol import (  # noqa: E402
    PROTOCOL_VERSION,
    ProtocolRequest,
    build_protocol_request,
    candidate_protocol_messages,
    contract_provenance,
    oracle_coverage,
    parse_selector_output,
)
from scripts import train_lfm25_lora as base  # noqa: E402


PROFILE = PROTOCOL_VERSION
DEFAULT_MAX_LENGTH = 1024
_BASE_MESSAGES = base._messages
_BASE_CONTRACT_PROVENANCE = base._contract_provenance


def _row_gold(row: dict[str, Any]) -> Any:
    for key in ("expected", "label"):
        if key in row:
            return row[key]
    raise ValueError("Candidate Protocol V1 rows require an expected or label target")


def _materialized_target(
    row: dict[str, Any],
    request: ProtocolRequest,
) -> tuple[str, dict[str, Any]]:
    if "candidate_protocol_v1_target" not in row:
        raise ValueError("Candidate Protocol V1 rows require a materialized selector target")
    materialized = row["candidate_protocol_v1_target"]
    if not isinstance(materialized, str):
        raise ValueError("materialized Candidate Protocol V1 target must be text")

    outcome = parse_selector_output(materialized, request)
    if not outcome.accepted or outcome.selection is None:
        raise ValueError(
            "materialized Candidate Protocol V1 target violates the current contract"
        )
    canonical = json.dumps(
        outcome.selection,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if materialized != canonical:
        raise ValueError("materialized Candidate Protocol V1 target is not canonical")
    return canonical, outcome.selection


def _ambiguous_float_amount_ids(
    gold: Any,
    request: ProtocolRequest,
) -> tuple[str, ...]:
    if not isinstance(gold, dict):
        return ()
    amount = gold.get("amount")
    if not isinstance(amount, float) or not math.isfinite(amount):
        return ()

    matches: list[str] = []
    for item in request.exact_amounts:
        try:
            projection = item.money.app_amount()
        except ValueError:
            continue
        if projection == amount:
            matches.append(item.id)
    return tuple(matches) if len(matches) > 1 else ()


def _validate_gold_drift(
    gold: Any,
    request: ProtocolRequest,
    selection: dict[str, Any],
) -> None:
    """Check every gold field still identifiable after ordinary JSON reload."""

    try:
        oracle = oracle_coverage(gold, request)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Candidate Protocol V1 row has an invalid expected or label target"
        ) from error

    if selection["transaction"] != int(oracle.is_transaction):
        raise ValueError(
            "materialized Candidate Protocol V1 target differs from current contract"
        )
    if not oracle.is_transaction:
        return
    if any(field != "amount" for field in oracle.missing_fields):
        raise ValueError(
            "materialized Candidate Protocol V1 target differs from current contract"
        )

    expected_ids = {
        "type": oracle.type_code,
        "account": oracle.account_id,
        "counterparty": oracle.counterparty_id,
    }
    if any(selection[field] != candidate_id for field, candidate_id in expected_ids.items()):
        raise ValueError(
            "materialized Candidate Protocol V1 target differs from current contract"
        )

    if oracle.amount_id is not None:
        amount_matches = selection["amount"] == oracle.amount_id
    else:
        amount_matches = selection["amount"] in _ambiguous_float_amount_ids(gold, request)
    if not amount_matches:
        raise ValueError(
            "materialized Candidate Protocol V1 target differs from current contract"
        )


def candidate_v1_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """Build and validate one exact Candidate Protocol V1 SFT conversation."""

    sender = str(row.get("sender", ""))
    sms = str(row.get("sms", ""))
    request = build_protocol_request(
        sender,
        sms,
        message_timestamp_epoch_ms=None,
    )
    gold = _row_gold(row)
    serialized, selection = _materialized_target(row, request)
    _validate_gold_drift(gold, request, selection)
    messages = base._validate_prompt_messages(
        candidate_protocol_messages(request),
        contract="Candidate Protocol V1",
    )
    messages.append({"role": "assistant", "content": serialized})
    return messages


def _messages(row: dict[str, Any], prompt_profile: str = PROFILE) -> list[dict[str, str]]:
    normalized = base._prompt_profile(prompt_profile)
    if normalized == PROFILE:
        return candidate_v1_messages(row)
    return _BASE_MESSAGES(row, normalized)


def _contract_provenance(prompt_profile: str) -> dict[str, Any]:
    if base._prompt_profile(prompt_profile) == PROFILE:
        return {"profile": PROFILE, **contract_provenance()}
    return _BASE_CONTRACT_PROVENANCE(prompt_profile)


def _has_option(arguments: list[str], *names: str) -> bool:
    return any(
        argument in names or any(argument.startswith(f"{name}=") for name in names)
        for argument in arguments
    )


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if _has_option(arguments, "--prompt-profile", "--contract"):
        raise argparse.ArgumentTypeError(
            "Candidate Protocol V1 training does not accept a different prompt profile"
        )
    arguments.extend(["--prompt-profile", PROFILE])
    if not _has_option(arguments, "--max-length"):
        arguments.extend(["--max-length", str(DEFAULT_MAX_LENGTH)])

    base.PROMPT_PROFILE_ALIASES[PROFILE] = PROFILE
    base.PROMPT_PROFILE_ALIASES["candidate_v1"] = PROFILE
    base._messages = _messages
    base._contract_provenance = _contract_provenance
    return base.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
