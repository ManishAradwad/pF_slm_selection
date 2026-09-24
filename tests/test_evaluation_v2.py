from __future__ import annotations

from copy import deepcopy
import json

import pytest

from lfm25.evaluation_v2 import (
    EVALUATION_CONTRACT_ID,
    EVALUATION_CONTRACT_VERSION,
    EvaluationV2Error,
    assess_candidate,
    decision_policy,
    score_evaluation_package,
    select_profile,
    split_manifest_sha256,
)
from lfm25.workbench_v2 import SYNTHETIC_FIXTURE_PATH, package_sha256


def _annotations() -> dict:
    value = json.loads(SYNTHETIC_FIXTURE_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _selected_record(row: dict) -> dict:
    selected = row["adjudication"]["selected_annotation_id"]
    for annotation in row["annotations"]:
        if annotation["annotation_id"] == selected:
            return deepcopy(annotation["semantic_record"])
    raise AssertionError("fixture adjudication is broken")


def _measurements() -> dict:
    return {
        "latency_ms": None,
        "peak_memory_bytes": None,
        "battery_impact_millipercent": None,
        "available": None,
        "recovery_attempted": None,
        "recovery_succeeded": None,
    }


def _predictions(annotations: dict) -> dict:
    semantic = annotations["semantic_contract"]
    rows = []
    for index, source in enumerate(annotations["rows"]):
        record = _selected_record(source)
        if source["row_id"] == "synth_fault_injection":
            record.pop("source_metadata")
        rows.append(
            {
                "row_id": source["row_id"],
                "parse_status": "valid",
                "semantic_record": record,
                "decision": "auto_post" if index == 0 else "review",
                "failure_code": None,
                "measurements": _measurements(),
            }
        )
    return {
        "evaluation_contract": {
            "id": EVALUATION_CONTRACT_ID,
            "version": EVALUATION_CONTRACT_VERSION,
        },
        "semantic_contract": deepcopy(semantic),
        "profile": {
            "profile_id": "synthetic_reference_v2",
            "platform": "shared_synthetic",
            "runtime_variant": "semantic_reference",
            "protocol_id": "synthetic_semantic_mapping",
        },
        "provenance": {
            "annotation_package_sha256": package_sha256(annotations),
            "split_manifest_sha256": split_manifest_sha256(annotations),
            "semantic_schema_sha256": semantic["schema_sha256"],
            "semantic_reference_sha256": semantic["reference_sha256"],
            "semantic_conformance_sha256": semantic["conformance_sha256"],
            "evaluator_id": "pocketfinancer_semantic_v2_evaluation",
            "evaluator_version": 2,
            "model_runtime_id": "invented_no_model",
            "model_revision": "invented_no_revision",
            "prompt_profile_id": "invented_no_prompt",
            "prompt_sha256": "1" * 64,
            "chat_template_sha256": None,
            "quantization": None,
            "decode_settings_sha256": "2" * 64,
            "parser_sha256": "3" * 64,
            "filter_sha256": "4" * 64,
            "runtime_environment_sha256": "5" * 64,
            "seed": 17,
            "selected_checkpoint": None,
        },
        "conformance": {"passed": 4, "total": 4, "uncaught_exceptions": 0},
        "privacy_provenance": {
            "local_only": True,
            "protected_split_locked": True,
            "contract_hashes_verified": True,
            "annotation_and_split_hashes_verified": True,
            "profile_provenance_complete": True,
        },
        "operational_budget": {
            "budget_id": None,
            "budget_sha256": None,
            "frozen_before_measurement": False,
            "evidence_class": None,
            "baseline_profile_id": None,
            "hard_limits_passed": False,
        },
        "rows": rows,
    }


def test_synthetic_smoke_scores_semantics_and_fails_closed() -> None:
    annotations = _annotations()
    report = score_evaluation_package(annotations, _predictions(annotations))

    assert report["sample"] == {
        "split": "protected_test",
        "resolved_rows": 4,
        "gold_automatic_post_eligible_rows": 2,
        "predicted_automatic_posts": 1,
        "invalid_predictions": 1,
        "fault_injection_cases": 3,
    }
    assert report["metrics"]["scope_accuracy"]["successes"] == 3
    assert report["metrics"]["evidence_grounding_validity"]["successes"] == 3
    assert report["metrics"]["automatic_post_precision"]["estimate"] == 1.0
    assert report["metrics"]["automatic_post_coverage"]["estimate"] == 0.5
    assert report["metrics"]["invalid_output_rate"]["estimate"] == 0.25
    assert report["metrics"]["fail_closed_rate"]["estimate"] == 1.0
    assert report["safety"]["false_automatic_posts"] == 0
    assert report["safety"]["caught_validation_failures"] == 1
    assert report["safety"]["uncaught_parser_or_validator_exceptions"] == 0
    assert report["safety"]["taxonomy"]["missing_timestamp_provenance"] == {
        "cases": 1,
        "automatic_post_violations": 0,
    }
    rendered = json.dumps(report, sort_keys=True)
    assert "Orbit Bank" not in rendered
    assert "XX1234" not in rendered
    assert "row_id" not in rendered


def test_invalid_record_requesting_auto_post_is_a_safety_violation() -> None:
    annotations = _annotations()
    predictions = _predictions(annotations)
    predictions["rows"][2]["decision"] = "auto_post"

    report = score_evaluation_package(annotations, predictions)
    assert report["metrics"]["fail_closed_rate"]["successes"] == 2
    assert report["safety"]["false_automatic_posts"] == 1
    assert report["safety"]["taxonomy"]["ungrounded_evidence"][
        "automatic_post_violations"
    ] == 1


def test_prediction_package_must_bind_annotation_and_split_hashes() -> None:
    annotations = _annotations()
    predictions = _predictions(annotations)
    predictions["provenance"]["annotation_package_sha256"] = "0" * 64

    with pytest.raises(EvaluationV2Error, match="annotation package"):
        score_evaluation_package(annotations, predictions)


def test_frozen_policy_has_predeclared_thresholds_and_protocol_neutrality() -> None:
    policy = decision_policy()

    assert policy["status"] == "frozen"
    assert policy["confidence"]["proportion_interval"] == "Wilson score interval"
    assert policy["sample_sizes"]["protected_rows"] == 1000
    assert policy["metric_gates"]["automatic_post_precision"]["threshold"] == 0.99
    assert policy["critical_safety_gate"]["maximum_total_false_automatic_posts"] == 0
    assert policy["comparison_scope"]["direct_v2_status"] == "unselected_hypothesis"
    assert policy["comparison_scope"]["candidate_v2_status"] == "unselected_hypothesis"
    assert policy["no_selection_policy"]["valid_outcome"] is True


def test_small_synthetic_smoke_cannot_be_selected() -> None:
    annotations = _annotations()
    report = score_evaluation_package(annotations, _predictions(annotations))
    assessment = assess_candidate(report, baseline_report=None)

    assert assessment["decision"] == "no_selection"
    assert assessment["passed"] is False
    failed = {gate["gate"] for gate in assessment["gates"] if not gate["passed"]}
    assert "sample:automatic_post_precision" in failed
    assert "conformance" in failed
    assert "operational:budget" in failed
    assert "baseline" in failed


def test_selection_rejects_cross_platform_or_runtime_pooling() -> None:
    annotations = _annotations()
    first = score_evaluation_package(annotations, _predictions(annotations))
    second = deepcopy(first)
    second["profile"]["profile_id"] = "synthetic_other_v2"
    second["profile"]["platform"] = "android"

    result = select_profile([first, second], baselines_by_profile_id={})
    assert result == {
        "decision": "no_selection:cross_scope_pooling_forbidden",
        "selected_profile_id": None,
        "assessments": [],
    }
