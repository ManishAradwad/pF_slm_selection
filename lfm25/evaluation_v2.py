"""Aggregate-safe Semantic V2 scoring and frozen profile-decision policy.

Predictions enter this layer only after a profile-specific parser maps them into
Semantic V2.  The evaluator neither defines nor prefers a model output protocol.
Every parser-reported or independently detected invalid record fails closed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .metrics import wilson_interval
from .semantic_v2 import SemanticRecord, SemanticV2Error, project_initial_auto_post, validate_semantic_v2
from .workbench_v2 import (
    ResolvedAnnotationRow,
    package_sha256,
    resolved_annotation_rows,
    validate_annotation_package,
    workbench_contract,
)


EVALUATION_CONTRACT_ID = "pocketfinancer_semantic_v2_evaluation"
EVALUATION_CONTRACT_VERSION = 2
DECISION_POLICY_ID = "pocketfinancer_extraction_v2_selection_policy"
DECISION_POLICY_VERSION = 1

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DECISION_POLICY_PATH = (
    _REPOSITORY_ROOT
    / "configs/programs/pocketfinancer-extraction-v2-decision-policy.json"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_OPAQUE_ID_RE = re.compile(r"[a-z][a-z0-9_.-]{2,95}\Z")
_PARSE_STATUSES = ("valid", "invalid")
_DECISIONS = ("auto_post", "review", "reject")
_PLATFORMS = ("shared_synthetic", "android", "ios", "host_hf", "host_gguf")
_EVIDENCE_CLASSES = ("shared_synthetic", "host_hf", "host_gguf", "android_device", "ios_device")


class EvaluationV2Error(ValueError):
    """An evaluation package or decision report violates the frozen contract."""


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationV2Error(f"{path} must be an object")
    return value


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise EvaluationV2Error(f"{path} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    if set(value) != expected:
        raise EvaluationV2Error(f"{path} has invalid keys")


def _text(value: Any, path: str, *, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise EvaluationV2Error(f"{path} must be non-empty text")
    if pattern is not None and not pattern.fullmatch(value):
        raise EvaluationV2Error(f"{path} has an invalid format")
    return value


def _nullable_text(
    value: Any,
    path: str,
    *,
    pattern: re.Pattern[str] | None = None,
) -> str | None:
    if value is None:
        return None
    return _text(value, path, pattern=pattern)


def _integer(value: Any, path: str, *, nullable: bool = False) -> int | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationV2Error(f"{path} must be a non-negative integer")
    return value


def _boolean(value: Any, path: str, *, nullable: bool = False) -> bool | None:
    if value is None and nullable:
        return None
    if not isinstance(value, bool):
        raise EvaluationV2Error(f"{path} must be boolean")
    return value


def _enum_text(value: Any, allowed: Sequence[str], path: str) -> str:
    text = _text(value, path)
    if text not in allowed:
        raise EvaluationV2Error(f"{path} has an unsupported value")
    return text


def _semantic_binding() -> dict[str, Any]:
    semantic = workbench_contract()["semantic_contract"]
    return {
        "id": semantic["id"],
        "version": semantic["version"],
        "schema_sha256": semantic["schema_sha256"],
        "reference_sha256": semantic["reference_sha256"],
        "conformance_sha256": semantic["conformance_sha256"],
    }


def decision_policy() -> dict[str, Any]:
    """Load the frozen policy and verify that it pins the current Semantic V2."""

    try:
        value = json.loads(DECISION_POLICY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationV2Error("decision policy is unavailable or invalid") from error
    policy = _object(value, "decision_policy")
    if (
        policy.get("policy_id") != DECISION_POLICY_ID
        or policy.get("policy_version") != DECISION_POLICY_VERSION
        or policy.get("status") != "frozen"
        or policy.get("frozen_in_phase") != "B"
    ):
        raise EvaluationV2Error("decision policy identity is invalid")
    if dict(_object(policy.get("semantic_contract"), "semantic_contract")) != _semantic_binding():
        raise EvaluationV2Error("decision policy Semantic V2 pins have drifted")
    return dict(policy)


def split_manifest_sha256(annotation_package: Mapping[str, Any]) -> str:
    """Hash only opaque row IDs, split names, and pseudonymous isolation groups."""

    validate_annotation_package(annotation_package)
    rows = []
    for item in annotation_package["rows"]:
        rows.append(
            {
                "row_id": item["row_id"],
                "split": item["split"],
                "groups": item["groups"],
            }
        )
    return package_sha256({"split_manifest_version": 1, "rows": rows})


def _validate_profile(value: Any) -> dict[str, str]:
    profile = _object(value, "profile")
    _exact_keys(profile, {"profile_id", "platform", "runtime_variant", "protocol_id"}, "profile")
    return {
        "profile_id": _text(profile["profile_id"], "profile.profile_id", pattern=_OPAQUE_ID_RE),
        "platform": _enum_text(profile["platform"], _PLATFORMS, "profile.platform"),
        "runtime_variant": _text(
            profile["runtime_variant"], "profile.runtime_variant", pattern=_OPAQUE_ID_RE
        ),
        "protocol_id": _text(profile["protocol_id"], "profile.protocol_id", pattern=_OPAQUE_ID_RE),
    }


def _validate_provenance(value: Any) -> dict[str, Any]:
    provenance = _object(value, "provenance")
    hash_fields = {
        "annotation_package_sha256",
        "split_manifest_sha256",
        "semantic_schema_sha256",
        "semantic_reference_sha256",
        "semantic_conformance_sha256",
        "prompt_sha256",
        "decode_settings_sha256",
        "parser_sha256",
        "filter_sha256",
        "runtime_environment_sha256",
    }
    text_fields = {
        "evaluator_id",
        "model_runtime_id",
        "model_revision",
        "prompt_profile_id",
    }
    nullable_text_fields = {"chat_template_sha256", "quantization", "selected_checkpoint"}
    expected = hash_fields | text_fields | nullable_text_fields | {"evaluator_version", "seed"}
    _exact_keys(provenance, expected, "provenance")
    normalized: dict[str, Any] = {}
    for field in hash_fields:
        normalized[field] = _text(provenance[field], f"provenance.{field}", pattern=_SHA256_RE)
    for field in text_fields:
        normalized[field] = _text(provenance[field], f"provenance.{field}")
    normalized["chat_template_sha256"] = _nullable_text(
        provenance["chat_template_sha256"],
        "provenance.chat_template_sha256",
        pattern=_SHA256_RE,
    )
    normalized["quantization"] = _nullable_text(
        provenance["quantization"], "provenance.quantization"
    )
    normalized["selected_checkpoint"] = _nullable_text(
        provenance["selected_checkpoint"], "provenance.selected_checkpoint"
    )
    normalized["evaluator_version"] = _integer(
        provenance["evaluator_version"], "provenance.evaluator_version"
    )
    if normalized["evaluator_version"] != EVALUATION_CONTRACT_VERSION:
        raise EvaluationV2Error("provenance.evaluator_version is unsupported")
    normalized["seed"] = _integer(provenance["seed"], "provenance.seed", nullable=True)
    return normalized


def _validate_conformance(value: Any) -> dict[str, int]:
    conformance = _object(value, "conformance")
    _exact_keys(conformance, {"passed", "total", "uncaught_exceptions"}, "conformance")
    passed = _integer(conformance["passed"], "conformance.passed")
    total = _integer(conformance["total"], "conformance.total")
    exceptions = _integer(conformance["uncaught_exceptions"], "conformance.uncaught_exceptions")
    assert passed is not None and total is not None and exceptions is not None
    if passed > total:
        raise EvaluationV2Error("conformance.passed cannot exceed total")
    return {"passed": passed, "total": total, "uncaught_exceptions": exceptions}


def _validate_privacy_provenance(value: Any) -> dict[str, bool]:
    evidence = _object(value, "privacy_provenance")
    fields = {
        "local_only",
        "protected_split_locked",
        "contract_hashes_verified",
        "annotation_and_split_hashes_verified",
        "profile_provenance_complete",
    }
    _exact_keys(evidence, fields, "privacy_provenance")
    return {field: bool(_boolean(evidence[field], f"privacy_provenance.{field}")) for field in fields}


def _validate_operational_budget(value: Any) -> dict[str, Any]:
    budget = _object(value, "operational_budget")
    _exact_keys(
        budget,
        {
            "budget_id",
            "budget_sha256",
            "frozen_before_measurement",
            "evidence_class",
            "baseline_profile_id",
            "hard_limits_passed",
        },
        "operational_budget",
    )
    evidence_class = budget["evidence_class"]
    if evidence_class is not None:
        evidence_class = _enum_text(
            evidence_class, _EVIDENCE_CLASSES, "operational_budget.evidence_class"
        )
    return {
        "budget_id": _nullable_text(budget["budget_id"], "operational_budget.budget_id"),
        "budget_sha256": _nullable_text(
            budget["budget_sha256"], "operational_budget.budget_sha256", pattern=_SHA256_RE
        ),
        "frozen_before_measurement": bool(
            _boolean(
                budget["frozen_before_measurement"],
                "operational_budget.frozen_before_measurement",
            )
        ),
        "evidence_class": evidence_class,
        "baseline_profile_id": _nullable_text(
            budget["baseline_profile_id"], "operational_budget.baseline_profile_id"
        ),
        "hard_limits_passed": bool(
            _boolean(budget["hard_limits_passed"], "operational_budget.hard_limits_passed")
        ),
    }


def _validate_measurements(value: Any, path: str) -> dict[str, Any]:
    measurements = _object(value, path)
    fields = {
        "latency_ms",
        "peak_memory_bytes",
        "battery_impact_millipercent",
        "available",
        "recovery_attempted",
        "recovery_succeeded",
    }
    _exact_keys(measurements, fields, path)
    normalized = {
        "latency_ms": _integer(measurements["latency_ms"], f"{path}.latency_ms", nullable=True),
        "peak_memory_bytes": _integer(
            measurements["peak_memory_bytes"], f"{path}.peak_memory_bytes", nullable=True
        ),
        "battery_impact_millipercent": _integer(
            measurements["battery_impact_millipercent"],
            f"{path}.battery_impact_millipercent",
            nullable=True,
        ),
        "available": _boolean(measurements["available"], f"{path}.available", nullable=True),
        "recovery_attempted": _boolean(
            measurements["recovery_attempted"], f"{path}.recovery_attempted", nullable=True
        ),
        "recovery_succeeded": _boolean(
            measurements["recovery_succeeded"], f"{path}.recovery_succeeded", nullable=True
        ),
    }
    attempted = normalized["recovery_attempted"]
    succeeded = normalized["recovery_succeeded"]
    if attempted is True and succeeded is None:
        raise EvaluationV2Error(f"{path}.recovery_succeeded is required after an attempt")
    if attempted is not True and succeeded is not None:
        raise EvaluationV2Error(f"{path}.recovery_succeeded requires an attempted recovery")
    return normalized


def validate_prediction_package(value: Any) -> dict[str, Any]:
    """Validate package structure without trusting parser-reported record validity."""

    package = _object(value, "prediction_package")
    _exact_keys(
        package,
        {
            "evaluation_contract",
            "semantic_contract",
            "profile",
            "provenance",
            "conformance",
            "privacy_provenance",
            "operational_budget",
            "rows",
        },
        "prediction_package",
    )
    identity = _object(package["evaluation_contract"], "evaluation_contract")
    if dict(identity) != {"id": EVALUATION_CONTRACT_ID, "version": EVALUATION_CONTRACT_VERSION}:
        raise EvaluationV2Error("prediction package has an unsupported evaluation identity")
    if dict(_object(package["semantic_contract"], "semantic_contract")) != _semantic_binding():
        raise EvaluationV2Error("prediction package Semantic V2 pins have drifted")
    profile = _validate_profile(package["profile"])
    provenance = _validate_provenance(package["provenance"])
    conformance = _validate_conformance(package["conformance"])
    privacy = _validate_privacy_provenance(package["privacy_provenance"])
    budget = _validate_operational_budget(package["operational_budget"])

    rows = _array(package["rows"], "rows")
    normalized_rows = []
    row_ids: set[str] = set()
    for index, value in enumerate(rows):
        path = f"rows[{index}]"
        row = _object(value, path)
        _exact_keys(
            row,
            {"row_id", "parse_status", "semantic_record", "decision", "failure_code", "measurements"},
            path,
        )
        row_id = _text(row["row_id"], f"{path}.row_id", pattern=_OPAQUE_ID_RE)
        if row_id in row_ids:
            raise EvaluationV2Error(f"{path}.row_id is duplicated")
        row_ids.add(row_id)
        parse_status = _enum_text(row["parse_status"], _PARSE_STATUSES, f"{path}.parse_status")
        semantic_record = row["semantic_record"]
        if parse_status == "invalid" and semantic_record is not None:
            raise EvaluationV2Error(f"{path}.semantic_record must be null when parser reports invalid")
        if parse_status == "valid" and not isinstance(semantic_record, Mapping):
            raise EvaluationV2Error(f"{path}.semantic_record must be an object when parser reports valid")
        normalized_rows.append(
            {
                "row_id": row_id,
                "parse_status": parse_status,
                "semantic_record": semantic_record,
                "decision": _enum_text(row["decision"], _DECISIONS, f"{path}.decision"),
                "failure_code": _nullable_text(row["failure_code"], f"{path}.failure_code"),
                "measurements": _validate_measurements(row["measurements"], f"{path}.measurements"),
            }
        )
    return {
        "profile": profile,
        "provenance": provenance,
        "conformance": conformance,
        "privacy_provenance": privacy,
        "operational_budget": budget,
        "rows": normalized_rows,
        "package_sha256": package_sha256(package),
    }


def _metric(successes: int, total: int) -> dict[str, Any]:
    interval = wilson_interval(successes, total)
    return {
        "successes": successes,
        "total": total,
        "estimate": round(successes / total, 6) if total else None,
        "ci95": {"lower": interval[0], "upper": interval[1]},
    }


def _nearest_rank_p95(values: Sequence[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(0.95 * len(ordered)))
    return ordered[rank - 1]


def _prediction_record(row: Mapping[str, Any], gold: ResolvedAnnotationRow) -> SemanticRecord | None:
    if row["parse_status"] != "valid":
        return None
    try:
        return validate_semantic_v2(row["semantic_record"], message=gold.message)
    except (SemanticV2Error, TypeError, ValueError):
        return None


def _aligned_event(prediction: SemanticRecord | None, index: int):
    if prediction is None or index >= len(prediction.events):
        return None
    return prediction.events[index]


def _field_counts(
    gold: SemanticRecord,
    prediction: SemanticRecord | None,
    counters: dict[str, list[int]],
) -> None:
    for index, gold_event in enumerate(gold.events):
        predicted_event = _aligned_event(prediction, index)
        if gold_event.amount is not None:
            counters["amount_currency_exactness"][1] += 1
            if predicted_event is not None and predicted_event.amount is not None:
                gold_amount = gold_event.amount
                predicted_amount = predicted_event.amount
                counters["amount_currency_exactness"][0] += int(
                    (
                        gold_amount.decimal_text,
                        gold_amount.currency,
                        gold_amount.minor_units,
                    )
                    == (
                        predicted_amount.decimal_text,
                        predicted_amount.currency,
                        predicted_amount.minor_units,
                    )
                )
        if gold_event.direction is not None:
            counters["direction_exactness"][1] += 1
            counters["direction_exactness"][0] += int(
                predicted_event is not None
                and predicted_event.direction is not None
                and predicted_event.direction.value is gold_event.direction.value
            )
        if gold_event.account is not None:
            counters["account_exactness"][1] += 1
            counters["account_exactness"][0] += int(
                predicted_event is not None
                and predicted_event.account is not None
                and predicted_event.account.value == gold_event.account.value
            )
        counters["counterparty_exactness"][1] += 1
        counters["counterparty_exactness"][0] += int(
            predicted_event is not None
            and predicted_event.counterparty.state is gold_event.counterparty.state
            and predicted_event.counterparty.value == gold_event.counterparty.value
        )


def score_evaluation_package(
    annotation_package: Mapping[str, Any],
    prediction_package: Mapping[str, Any],
    *,
    split: str = "protected_test",
) -> dict[str, Any]:
    """Join locally by opaque ID and return an aggregate-only Semantic V2 report."""

    annotation_summary = validate_annotation_package(annotation_package)
    prediction = validate_prediction_package(prediction_package)
    if prediction["provenance"]["annotation_package_sha256"] != annotation_summary["package_sha256"]:
        raise EvaluationV2Error("prediction provenance does not bind the annotation package")
    expected_split_hash = split_manifest_sha256(annotation_package)
    if prediction["provenance"]["split_manifest_sha256"] != expected_split_hash:
        raise EvaluationV2Error("prediction provenance does not bind the split manifest")
    semantic = _semantic_binding()
    for provenance_key, semantic_key in (
        ("semantic_schema_sha256", "schema_sha256"),
        ("semantic_reference_sha256", "reference_sha256"),
        ("semantic_conformance_sha256", "conformance_sha256"),
    ):
        if prediction["provenance"][provenance_key] != semantic[semantic_key]:
            raise EvaluationV2Error("prediction provenance has a mismatched Semantic V2 hash")

    gold_rows = resolved_annotation_rows(annotation_package, split=split)
    gold_by_id = {row.row_id: row for row in gold_rows}
    predictions_by_id = {row["row_id"]: row for row in prediction["rows"]}
    if set(gold_by_id) != set(predictions_by_id):
        raise EvaluationV2Error("prediction rows must exactly match resolved rows in the scored split")

    metric_counts = {
        name: [0, 0]
        for name in (
            "scope_accuracy",
            "posting_status_accuracy",
            "event_cardinality_accuracy",
            "amount_currency_exactness",
            "direction_exactness",
            "account_exactness",
            "counterparty_exactness",
            "evidence_grounding_validity",
            "automatic_post_precision",
            "automatic_post_coverage",
            "invalid_output_rate",
            "fail_closed_rate",
        )
    }
    safety_tags = workbench_contract()["safety_tags"]
    taxonomy = {tag: {"cases": 0, "automatic_post_violations": 0} for tag in safety_tags if tag != "ordinary"}
    false_automatic_posts = 0
    invalid_predictions = 0
    caught_validation_failures = 0
    latency_values: list[int] = []
    memory_values: list[int] = []
    battery_values: list[int] = []
    availability = [0, 0]
    recovery = [0, 0]

    for row_id in sorted(gold_by_id):
        gold_row = gold_by_id[row_id]
        prediction_row = predictions_by_id[row_id]
        predicted_record = _prediction_record(prediction_row, gold_row)
        if prediction_row["parse_status"] == "valid" and predicted_record is None:
            caught_validation_failures += 1
        valid = predicted_record is not None
        invalid_predictions += int(not valid)
        decision_is_auto = prediction_row["decision"] == "auto_post"

        gold = gold_row.gold
        metric_counts["scope_accuracy"][0] += int(valid and predicted_record.scope is gold.scope)
        metric_counts["posting_status_accuracy"][0] += int(
            valid and predicted_record.posting_status is gold.posting_status
        )
        metric_counts["event_cardinality_accuracy"][0] += int(
            valid and predicted_record.event_cardinality is gold.event_cardinality
        )
        for name in (
            "scope_accuracy",
            "posting_status_accuracy",
            "event_cardinality_accuracy",
            "evidence_grounding_validity",
        ):
            metric_counts[name][1] += 1
        metric_counts["evidence_grounding_validity"][0] += int(valid)
        _field_counts(gold, predicted_record, metric_counts)

        gold_projection = project_initial_auto_post(gold)
        if gold_projection.eligible:
            metric_counts["automatic_post_coverage"][1] += 1
        correct_auto_post = False
        if decision_is_auto:
            metric_counts["automatic_post_precision"][1] += 1
            if valid:
                predicted_projection = project_initial_auto_post(predicted_record)
                correct_auto_post = bool(
                    gold_projection.eligible
                    and predicted_projection.eligible
                    and predicted_projection.transaction == gold_projection.transaction
                )
            metric_counts["automatic_post_precision"][0] += int(correct_auto_post)
            false_automatic_posts += int(not correct_auto_post)
        metric_counts["automatic_post_coverage"][0] += int(correct_auto_post)

        metric_counts["invalid_output_rate"][0] += int(not valid)
        metric_counts["invalid_output_rate"][1] += 1
        fault_case = any(tag != "ordinary" for tag in gold_row.safety_tags)
        if fault_case:
            metric_counts["fail_closed_rate"][1] += 1
            metric_counts["fail_closed_rate"][0] += int(not decision_is_auto)
        for tag in gold_row.safety_tags:
            if tag != "ordinary":
                taxonomy[tag]["cases"] += 1
                taxonomy[tag]["automatic_post_violations"] += int(decision_is_auto)

        measurements = prediction_row["measurements"]
        if measurements["latency_ms"] is not None:
            latency_values.append(measurements["latency_ms"])
        if measurements["peak_memory_bytes"] is not None:
            memory_values.append(measurements["peak_memory_bytes"])
        if measurements["battery_impact_millipercent"] is not None:
            battery_values.append(measurements["battery_impact_millipercent"])
        if measurements["available"] is not None:
            availability[1] += 1
            availability[0] += int(measurements["available"])
        if measurements["recovery_attempted"] is True:
            recovery[1] += 1
            recovery[0] += int(measurements["recovery_succeeded"])

    metrics = {name: _metric(*counts) for name, counts in metric_counts.items()}
    semantic_macro_names = (
        "scope_accuracy",
        "posting_status_accuracy",
        "event_cardinality_accuracy",
        "amount_currency_exactness",
        "direction_exactness",
        "account_exactness",
        "counterparty_exactness",
        "evidence_grounding_validity",
    )
    lower_bounds = [metrics[name]["ci95"]["lower"] for name in semantic_macro_names]
    semantic_macro = None if any(value is None for value in lower_bounds) else round(sum(lower_bounds) / len(lower_bounds), 6)
    return {
        "evaluation_contract": {
            "id": EVALUATION_CONTRACT_ID,
            "version": EVALUATION_CONTRACT_VERSION,
        },
        "profile": prediction["profile"],
        "provenance": {
            "annotation_package_sha256": annotation_summary["package_sha256"],
            "split_manifest_sha256": expected_split_hash,
            "prediction_package_sha256": prediction["package_sha256"],
            "evaluator_id": prediction["provenance"]["evaluator_id"],
            "evaluator_version": prediction["provenance"]["evaluator_version"],
        },
        "sample": {
            "split": split,
            "resolved_rows": len(gold_rows),
            "gold_automatic_post_eligible_rows": metric_counts["automatic_post_coverage"][1],
            "predicted_automatic_posts": metric_counts["automatic_post_precision"][1],
            "invalid_predictions": invalid_predictions,
            "fault_injection_cases": metric_counts["fail_closed_rate"][1],
        },
        "metrics": metrics,
        "derived": {"semantic_macro_lower_bound": semantic_macro},
        "safety": {
            "false_automatic_posts": false_automatic_posts,
            "taxonomy": taxonomy,
            "caught_validation_failures": caught_validation_failures,
            "uncaught_parser_or_validator_exceptions": 0,
        },
        "conformance": prediction["conformance"],
        "privacy_provenance": prediction["privacy_provenance"],
        "operational_budget": prediction["operational_budget"],
        "operational": {
            "latency_p95_ms": _nearest_rank_p95(latency_values),
            "latency_runs": len(latency_values),
            "peak_memory_p95_bytes": _nearest_rank_p95(memory_values),
            "memory_runs": len(memory_values),
            "battery_impact_p95_millipercent": _nearest_rank_p95(battery_values),
            "battery_runs": len(battery_values),
            "availability": _metric(*availability),
            "recovery_success": _metric(*recovery),
        },
    }


def _gate(gates: list[dict[str, Any]], name: str, passed: bool, reason: str) -> None:
    gates.append({"gate": name, "passed": passed, "reason": "passed" if passed else reason})


def _bound(metric: Mapping[str, Any], which: str) -> float | None:
    return metric["ci95"][which]


def assess_candidate(
    report: Mapping[str, Any],
    *,
    baseline_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Apply every frozen gate; missing evidence deterministically yields no selection."""

    policy = decision_policy()
    gates: list[dict[str, Any]] = []
    metrics = _object(report.get("metrics"), "report.metrics")
    for name, rule_value in policy["metric_gates"].items():
        rule = _object(rule_value, f"metric_gates.{name}")
        metric = _object(metrics.get(name), f"report.metrics.{name}")
        total = metric.get("total")
        sufficient = isinstance(total, int) and not isinstance(total, bool) and total >= rule["minimum_denominator"]
        _gate(gates, f"sample:{name}", sufficient, "insufficient_denominator")
        bound_value = _bound(metric, rule["bound"])
        threshold_passed = bool(
            sufficient
            and bound_value is not None
            and (
                bound_value >= rule["threshold"]
                if rule["operator"] == ">="
                else bound_value <= rule["threshold"]
            )
        )
        _gate(gates, f"threshold:{name}", threshold_passed, "confidence_bound_failed")

    critical = policy["critical_safety_gate"]
    safety = _object(report.get("safety"), "report.safety")
    _gate(
        gates,
        "safety:false_automatic_posts",
        safety.get("false_automatic_posts") == critical["maximum_total_false_automatic_posts"],
        "false_automatic_post_observed",
    )
    taxonomy = _object(safety.get("taxonomy"), "report.safety.taxonomy")
    for tag in critical["required_tags"]:
        tag_result = _object(taxonomy.get(tag), f"report.safety.taxonomy.{tag}")
        enough = tag_result.get("cases", 0) >= critical["minimum_cases_per_tag"]
        no_violations = (
            tag_result.get("automatic_post_violations")
            == critical["maximum_automatic_post_violations_per_tag"]
        )
        _gate(gates, f"safety_sample:{tag}", enough, "insufficient_safety_cases")
        _gate(gates, f"safety_zero:{tag}", no_violations, "safety_violation_observed")
    _gate(
        gates,
        "safety:uncaught_exceptions",
        safety.get("uncaught_parser_or_validator_exceptions")
        == critical["maximum_uncaught_parser_or_validator_exceptions"],
        "uncaught_exception_observed",
    )

    conformance = _object(report.get("conformance"), "report.conformance")
    required_vectors = policy["sample_sizes"]["conformance_vectors"]
    conformance_passed = bool(
        conformance.get("total") == required_vectors
        and conformance.get("passed") == required_vectors
        and conformance.get("uncaught_exceptions") == 0
    )
    _gate(gates, "conformance", conformance_passed, "conformance_incomplete_or_failed")

    privacy = _object(report.get("privacy_provenance"), "report.privacy_provenance")
    for field in (
        "local_only",
        "protected_split_locked",
        "contract_hashes_verified",
        "annotation_and_split_hashes_verified",
        "profile_provenance_complete",
    ):
        _gate(gates, f"provenance:{field}", privacy.get(field) is True, "required_provenance_missing")

    budget = _object(report.get("operational_budget"), "report.operational_budget")
    complete_budget = bool(
        budget.get("budget_id")
        and isinstance(budget.get("budget_sha256"), str)
        and _SHA256_RE.fullmatch(budget["budget_sha256"])
        and budget.get("frozen_before_measurement") is True
        and budget.get("evidence_class") in _EVIDENCE_CLASSES
        and budget.get("baseline_profile_id")
        and budget.get("hard_limits_passed") is True
    )
    _gate(gates, "operational:budget", complete_budget, "operational_budget_missing_or_failed")

    operational = _object(report.get("operational"), "report.operational")
    minimum_runs = policy["sample_sizes"]["operational_runs_per_device_class"]
    for field in ("latency_runs", "memory_runs", "battery_runs"):
        _gate(
            gates,
            f"operational:sample:{field}",
            isinstance(operational.get(field), int) and operational[field] >= minimum_runs,
            "insufficient_operational_runs",
        )
    recovery = _object(operational.get("recovery_success"), "operational.recovery_success")
    availability = _object(operational.get("availability"), "operational.availability")
    _gate(
        gates,
        "operational:recovery",
        recovery.get("total", 0) >= policy["sample_sizes"]["recovery_trials"]
        and (_bound(recovery, "lower") or 0.0)
        >= policy["operational_gates"]["recovery_success_lower_bound_minimum"],
        "recovery_gate_failed",
    )
    _gate(
        gates,
        "operational:availability",
        availability.get("total", 0) >= policy["sample_sizes"]["availability_trials"]
        and (_bound(availability, "lower") or 0.0)
        >= policy["operational_gates"]["availability_lower_bound_minimum"],
        "availability_gate_failed",
    )

    if baseline_report is None:
        _gate(gates, "baseline", False, "reproducible_baseline_missing")
    else:
        baseline_metrics = _object(baseline_report.get("metrics"), "baseline.metrics")
        same_rows = bool(
            report.get("provenance", {}).get("annotation_package_sha256")
            == baseline_report.get("provenance", {}).get("annotation_package_sha256")
            and report.get("provenance", {}).get("split_manifest_sha256")
            == baseline_report.get("provenance", {}).get("split_manifest_sha256")
        )
        _gate(gates, "baseline:same_rows", same_rows, "baseline_rows_mismatch")
        tolerances = policy["baseline_regression_gate"]
        for name, metric in metrics.items():
            if name not in baseline_metrics:
                continue
            candidate_lower = _bound(metric, "lower")
            baseline_lower = _bound(baseline_metrics[name], "lower")
            candidate_upper = _bound(metric, "upper")
            baseline_upper = _bound(baseline_metrics[name], "upper")
            if name == "invalid_output_rate":
                passed = bool(
                    candidate_upper is not None
                    and baseline_upper is not None
                    and candidate_upper
                    <= baseline_upper + tolerances["invalid_output_upper_bound_tolerance"]
                )
            else:
                tolerance = tolerances["semantic_metric_lower_bound_tolerance"]
                if name == "automatic_post_precision":
                    tolerance = tolerances["automatic_post_precision_lower_bound_tolerance"]
                elif name == "automatic_post_coverage":
                    tolerance = tolerances["automatic_post_coverage_lower_bound_tolerance"]
                passed = bool(
                    candidate_lower is not None
                    and baseline_lower is not None
                    and candidate_lower >= baseline_lower - tolerance
                )
            _gate(gates, f"baseline_regression:{name}", passed, "disallowed_regression")

        baseline_operational = _object(
            baseline_report.get("operational"), "baseline.operational"
        )
        same_evidence_class = bool(
            report.get("operational_budget", {}).get("evidence_class")
            == baseline_report.get("operational_budget", {}).get("evidence_class")
        )
        _gate(
            gates,
            "operational:evidence_class",
            same_evidence_class,
            "operational_evidence_class_mismatch",
        )
        operational_rules = policy["operational_gates"]
        for candidate_key, baseline_key, ratio_key in (
            ("latency_p95_ms", "latency_p95_ms", "latency_p95_maximum_ratio_to_baseline"),
            (
                "peak_memory_p95_bytes",
                "peak_memory_p95_bytes",
                "peak_memory_p95_maximum_ratio_to_baseline",
            ),
        ):
            candidate_value = operational.get(candidate_key)
            baseline_value = baseline_operational.get(baseline_key)
            passed = bool(
                isinstance(candidate_value, int)
                and isinstance(baseline_value, int)
                and baseline_value > 0
                and candidate_value <= baseline_value * operational_rules[ratio_key]
            )
            _gate(gates, f"operational:{candidate_key}", passed, "operational_regression")
        candidate_battery = operational.get("battery_impact_p95_millipercent")
        baseline_battery = baseline_operational.get("battery_impact_p95_millipercent")
        battery_passed = bool(
            isinstance(candidate_battery, int)
            and isinstance(baseline_battery, int)
            and candidate_battery
            <= baseline_battery
            + int(
                operational_rules[
                    "battery_impact_maximum_percentage_point_increase_to_baseline"
                ]
                * 1000
            )
        )
        _gate(gates, "operational:battery", battery_passed, "battery_regression")

    passed = all(item["passed"] for item in gates)
    return {
        "policy_id": policy["policy_id"],
        "policy_version": policy["policy_version"],
        "profile": dict(_object(report.get("profile"), "report.profile")),
        "decision": "eligible_for_tie_break" if passed else "no_selection",
        "passed": passed,
        "gates": gates,
    }


def _material_winner(
    reports: Sequence[Mapping[str, Any]],
    *,
    value_getter,
    preference: str,
    absolute_difference: float | None = None,
    relative_difference: float | None = None,
) -> Mapping[str, Any] | None:
    values = [(report, value_getter(report)) for report in reports]
    if any(value is None for _, value in values):
        return None
    ordered = sorted(values, key=lambda item: item[1], reverse=preference == "higher")
    if len(ordered) < 2:
        return ordered[0][0]
    best_report, best = ordered[0]
    second = ordered[1][1]
    if absolute_difference is not None and abs(best - second) >= absolute_difference:
        return best_report
    if relative_difference is not None and second != 0:
        if abs(best - second) / abs(second) >= relative_difference:
            return best_report
    return None


def select_profile(
    reports: Sequence[Mapping[str, Any]],
    *,
    baselines_by_profile_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Select only within one platform/runtime scope or return explicit no-selection."""

    if not reports:
        return {"decision": "no_selection:no_candidates", "selected_profile_id": None, "assessments": []}
    scopes = {(report["profile"]["platform"], report["profile"]["runtime_variant"]) for report in reports}
    if len(scopes) != 1:
        return {
            "decision": "no_selection:cross_scope_pooling_forbidden",
            "selected_profile_id": None,
            "assessments": [],
        }
    assessments = []
    passing_reports = []
    for report in reports:
        profile_id = report["profile"]["profile_id"]
        assessment = assess_candidate(
            report,
            baseline_report=baselines_by_profile_id.get(profile_id),
        )
        assessments.append(assessment)
        if assessment["passed"]:
            passing_reports.append(report)
    if not passing_reports:
        return {
            "decision": "no_selection:all_candidates_failed",
            "selected_profile_id": None,
            "assessments": assessments,
        }
    if len(passing_reports) == 1:
        return {
            "decision": "selected",
            "selected_profile_id": passing_reports[0]["profile"]["profile_id"],
            "assessments": assessments,
        }

    policy = decision_policy()
    for rule in policy["tie_policy"]["comparison_order"]:
        metric = rule["metric"]
        if metric in ("automatic_post_precision", "automatic_post_coverage"):
            def getter(report, name=metric):
                return report["metrics"][name]["ci95"]["lower"]
        elif metric == "semantic_macro_lower_bound":
            def getter(report):
                return report["derived"]["semantic_macro_lower_bound"]
        else:
            def getter(report, name=metric):
                return report["operational"][name]
        winner = _material_winner(
            passing_reports,
            value_getter=getter,
            preference=rule["preference"],
            absolute_difference=rule.get("minimum_material_difference"),
            relative_difference=rule.get("minimum_material_relative_difference"),
        )
        if winner is not None:
            return {
                "decision": "selected",
                "selected_profile_id": winner["profile"]["profile_id"],
                "tie_break_metric": metric,
                "assessments": assessments,
            }
    return {
        "decision": "no_selection:unresolved_material_tie",
        "selected_profile_id": None,
        "assessments": assessments,
    }
