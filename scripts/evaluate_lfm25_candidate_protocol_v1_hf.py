#!/usr/bin/env python3
"""Evaluate Candidate Protocol V1 through the existing HF generation engine.

This adapter deliberately reuses the historical candidate evaluator's batching,
metrics, and private-output safeguards while replacing its prompt, oracle,
strict parser, and provenance with the versioned V1 contract.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from collections.abc import Callable
from decimal import Decimal
import io
import json
from pathlib import Path
import sys
from typing import Any, Mapping
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lfm25.android_contract import contract_provenance as prefilter_provenance  # noqa: E402
from lfm25.candidate_protocol import (  # noqa: E402
    OutcomeStatus,
    build_protocol_request,
    candidate_protocol_messages,
    contract_provenance,
    oracle_coverage,
    parse_selector_output,
)
from lfm25.contract import ParsedOutput  # noqa: E402
from lfm25.provenance import code_fingerprints, fingerprint_file  # noqa: E402
from scripts import evaluate_lfm25_candidate_hf as base  # noqa: E402
from scripts.evaluate_lfm25_android_hf import (  # noqa: E402
    _assert_evaluation_inputs_unchanged,
    _evaluation_input_evidence,
    _safe_file_fingerprint,
)

PLATFORM_GATES = {
    "hf_host_reference_only": True,
    "android_implemented": False,
    "ios_implemented": False,
    "android_runtime_parity": False,
    "ios_runtime_parity": False,
    "gguf_runtime_validated": False,
    "android_device_validated": False,
    "ios_device_validated": False,
    "product_promotion_allowed": False,
}
_V1_DIAGNOSTIC_GOLD_KEYS = frozenset({"type", "amount", "account", "counterparty"})
_LEGACY_DIAGNOSTIC_GOLD_KEYS = _V1_DIAGNOSTIC_GOLD_KEYS | {"date"}


def _reject_nonfinite_diagnostic_number(value: str) -> None:
    raise ValueError(f"diagnostic gold contains a non-finite number: {value}")


def _unique_diagnostic_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"diagnostic gold contains a duplicate key: {key}")
        result[key] = value
    return result


def _project_diagnostic_gold(gold: Any) -> dict[str, Any] | None:
    """Project the locked legacy diagnostic label into the exact V1 oracle shape."""

    value = gold
    if isinstance(gold, str):
        try:
            value = json.loads(
                gold,
                parse_float=Decimal,
                parse_constant=_reject_nonfinite_diagnostic_number,
                object_pairs_hook=_unique_diagnostic_object,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError("diagnostic gold is not strict JSON") from error

    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("diagnostic gold must be null or a transaction object")
    keys = frozenset(value)
    if keys not in {
        _V1_DIAGNOSTIC_GOLD_KEYS,
        _LEGACY_DIAGNOSTIC_GOLD_KEYS,
    }:
        raise ValueError("diagnostic gold has an unsupported transaction shape")
    return {
        "type": value["type"],
        "amount": value["amount"],
        "account": value["account"],
        "counterparty": value["counterparty"],
    }


def _candidate_execution_contract_evidence() -> dict[str, Any]:
    return {
        "evaluator": fingerprint_file(Path(__file__)),
        "generation_engine": fingerprint_file(
            REPO_ROOT / "scripts" / "evaluate_lfm25_candidate_hf.py"
        ),
        "candidate_profile": {
            "candidate": _safe_file_fingerprint(
                REPO_ROOT / "configs" / "contracts" / "pocketfinancer-candidate-v1.json",
                filename="pocketfinancer-candidate-v1.json",
            ),
            "baseline": _safe_file_fingerprint(
                REPO_ROOT / "configs" / "contracts" / "pocketfinancer-android-current.json",
                filename="pocketfinancer-android-current.json",
            ),
            "golden_vectors": _safe_file_fingerprint(
                REPO_ROOT / "DATA" / "candidate_protocol_v1_golden.json",
                filename="candidate_protocol_v1_golden.json",
            ),
        },
        "platform_gates": dict(PLATFORM_GATES),
        "candidate_protocol": contract_provenance(),
        "prefilter_contract": prefilter_provenance(),
        "candidate_protocol_code": fingerprint_file(REPO_ROOT / "lfm25" / "candidate_protocol.py"),
        "candidate_extractor": fingerprint_file(REPO_ROOT / "lfm25" / "candidates.py"),
        "code_sha256": code_fingerprints(REPO_ROOT),
    }


def _assert_candidate_execution_contract_unchanged(
    expected: Mapping[str, Any],
) -> None:
    if _candidate_execution_contract_evidence() != expected:
        raise RuntimeError("Candidate Protocol evaluator code or contract changed during inference")


def _messages(sender: str, sms: str) -> list[dict[str, str]]:
    return candidate_protocol_messages(build_protocol_request(sender, sms))


def _request(sms: str) -> Any:
    # The historical evaluator supplies sender only to its message helper. The
    # sender is model-visible but does not affect candidate enumeration, so this
    # host request is used solely for oracle/parsing and carries the same SMS.
    return build_protocol_request("", sms, message_timestamp_epoch_ms=None)


def _oracle(gold: Any, request: Any) -> Any:
    return oracle_coverage(gold, request)


def _resolve(
    text: str,
    request: Any,
    *,
    hybrid_safety: bool = False,
) -> tuple[ParsedOutput, tuple[str, ...]]:
    if hybrid_safety:
        raise ValueError("Candidate Protocol V1 forbids historical hybrid overrides")
    outcome = parse_selector_output(text, request)
    reason = outcome.reason.value
    if outcome.status is OutcomeStatus.TRANSACTION:
        rendered = json.dumps(
            outcome.transaction,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return ParsedOutput("transaction", outcome.transaction, rendered, reason), ()
    if outcome.status is OutcomeStatus.NOT_TRANSACTION:
        return ParsedOutput("null", None, "null", reason), ()
    return ParsedOutput("invalid", None, "", reason), ()


def _parse_evidence_args(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-lock", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--limit", type=int)
    parsed, _unknown = parser.parse_known_args(arguments)
    if parsed.limit is not None and parsed.limit < 0:
        parser.error("--limit cannot be negative")
    return parsed


def _without_model_lock(arguments: list[str]) -> list[str]:
    forwarded: list[str] = []
    skip_value = False
    for argument in arguments:
        if skip_value:
            skip_value = False
            continue
        if argument == "--model-lock":
            skip_value = True
            continue
        if argument.startswith("--model-lock="):
            continue
        forwarded.append(argument)
    if skip_value:
        raise SystemExit("--model-lock requires a value")
    return forwarded


def _frozen_row_reader(
    dataset: Path,
    rows: list[dict[str, Any]],
) -> Callable[[Path], list[dict[str, Any]]]:
    expected = dataset.resolve(strict=True)

    def read_rows(path: Path) -> list[dict[str, Any]]:
        if path.resolve(strict=True) != expected:
            raise RuntimeError("historical evaluator requested an unexpected dataset")
        return [dict(row) for row in rows]

    return read_rows


def _install_v1_contract() -> None:
    base.candidate_selector_messages = _messages
    base.extract_candidates = _request
    base.oracle_selection = _oracle
    base.resolve_selector_prediction = _resolve
    base.parse_gold = _project_diagnostic_gold
    base.contract_provenance = contract_provenance


def _rewrite_metrics(
    output_dir: Path,
    *,
    input_evidence: Mapping[str, Any],
    execution_contract_evidence: Mapping[str, Any],
    row_count: int,
    row_limit: int | None,
) -> dict[str, Any]:
    metrics_path = output_dir / "metrics.json"
    samples_path = output_dir / "samples.jsonl"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    reasons: dict[str, int] = {}
    for line in samples_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        reason = record.get("selector_error")
        if isinstance(reason, str) and reason:
            reasons[reason] = reasons.get(reason, 0) + 1

    runtime = metrics.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("historical evaluator metrics have no runtime object")
    model_invocations = runtime.get("model_invocations")
    if (
        isinstance(model_invocations, bool)
        or not isinstance(model_invocations, int)
        or model_invocations < 0
    ):
        raise ValueError("historical evaluator metrics have no model invocation count")
    prefilter = metrics.get("prefilter")
    if not isinstance(prefilter, dict):
        raise ValueError("historical evaluator metrics have no prefilter summary")
    prefilter_enabled = prefilter.get("enabled")
    if not isinstance(prefilter_enabled, bool):
        raise ValueError("historical evaluator metrics have no prefilter enabled state")
    if prefilter.get("model_invocations") != model_invocations:
        raise ValueError("prefilter and runtime invocation counts disagree")
    runtime["prefilter_applied"] = prefilter_enabled
    if sum(reasons.values()) != model_invocations:
        raise ValueError("selector reason counts do not cover every model invocation")
    accepted_outputs = sum(
        count for reason, count in reasons.items() if reason.startswith("accepted_")
    )
    accepted_transactions = reasons.get("accepted_transaction", 0)
    metrics["candidate_protocol_acceptance"] = {
        "model_invocations": model_invocations,
        "accepted_outputs": accepted_outputs,
        "rejected_outputs": model_invocations - accepted_outputs,
        "strict_schema_acceptance_rate": (
            accepted_outputs / model_invocations if model_invocations else 1.0
        ),
        "accepted_transactions": accepted_transactions,
        "source_grounded_transactions": accepted_transactions,
        "source_grounded_transaction_rate": 1.0 if accepted_transactions else None,
    }

    dataset_evidence = {
        **input_evidence["dataset"],
        "row_count": row_count,
        "row_limit": row_limit,
    }
    provenance = metrics.setdefault("provenance", {})
    provenance.update(
        {
            "pipeline": "pocketfinancer_candidate_protocol_v1_hf",
            "generation_engine": execution_contract_evidence["generation_engine"],
            "evaluator": execution_contract_evidence["evaluator"],
            "model": input_evidence["model"],
            "model_lock": input_evidence["model_lock"],
            "adapter": input_evidence["adapter"],
            "training_run": input_evidence["training_run"],
            "dataset": dataset_evidence,
            "code_sha256": execution_contract_evidence["code_sha256"],
            "selection_prefilter": {
                "applied": prefilter_enabled,
                "part_of_android_current": True,
                "rejected_prediction": "null",
            },
            "candidate_protocol": execution_contract_evidence["candidate_protocol"],
            "prefilter_contract": execution_contract_evidence["prefilter_contract"],
            "candidate_protocol_code": execution_contract_evidence["candidate_protocol_code"],
            "candidate_extractor": execution_contract_evidence["candidate_extractor"],
            "candidate_profile": execution_contract_evidence["candidate_profile"],
            "platform_gates": execution_contract_evidence["platform_gates"],
        }
    )
    provenance.pop("android_contract", None)
    provenance.pop("candidate_code", None)
    metrics["selector_reason_counts"] = dict(sorted(reasons.items()))
    metrics["hybrid_safety"] = {
        "enabled": False,
        "intervention_counts": {},
    }
    metrics["runtime"]["thinking_mode"] = "off"
    metrics["runtime"]["model_output_protocol"] = "candidate_protocol_v1"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--hybrid-safety" in arguments:
        raise SystemExit("Candidate Protocol V1 does not permit --hybrid-safety")
    parsed = _parse_evidence_args(arguments)
    forwarded = _without_model_lock(arguments)
    input_evidence = _evaluation_input_evidence(
        model=parsed.model,
        adapter=parsed.adapter,
        dataset=parsed.dataset,
        model_lock=parsed.model_lock,
    )
    execution_contract_evidence = _candidate_execution_contract_evidence()
    frozen_rows = base._read_rows(parsed.dataset)
    row_count = len(frozen_rows if parsed.limit is None else frozen_rows[: parsed.limit])
    _assert_evaluation_inputs_unchanged(
        input_evidence,
        model=parsed.model,
        adapter=parsed.adapter,
        dataset=parsed.dataset,
        model_lock=parsed.model_lock,
    )
    _assert_candidate_execution_contract_unchanged(execution_contract_evidence)

    _install_v1_contract()
    # Suppress the historical evaluator's pre-rewrite aggregate JSON. It never
    # contains SMS text, but reporting only the corrected V1 evidence avoids a
    # misleading intermediate provenance record.
    with (
        redirect_stdout(io.StringIO()),
        patch.object(sys, "argv", [str(Path(__file__).name), *forwarded]),
        patch.object(base, "_read_rows", _frozen_row_reader(parsed.dataset, frozen_rows)),
    ):
        result = base.main()
    _assert_evaluation_inputs_unchanged(
        input_evidence,
        model=parsed.model,
        adapter=parsed.adapter,
        dataset=parsed.dataset,
        model_lock=parsed.model_lock,
    )
    _assert_candidate_execution_contract_unchanged(execution_contract_evidence)
    metrics = _rewrite_metrics(
        parsed.output_dir,
        input_evidence=input_evidence,
        execution_contract_evidence=execution_contract_evidence,
        row_count=row_count,
        row_limit=parsed.limit,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
