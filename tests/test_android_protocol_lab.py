from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from lfm25.android_protocol_lab import (
    ANDROID_COMMIT,
    BASELINE_ID,
    BASELINE_SHA256,
    LAB_ID,
    LAB_VERSION,
    REPOSITORY_ROOT,
    AndroidProtocolLabError,
    candidate_output_from_semantic_core,
    load_lab_manifest,
    load_prompt_profile,
    parse_protocol_output,
    run_parser_conformance,
    validate_result_set,
)


SEMANTIC_FIXTURE = (
    REPOSITORY_ROOT / "tests/fixtures/pocketfinancer_semantic_v2_synthetic_golden.json"
)


def _semantic_vectors():
    return json.loads(SEMANTIC_FIXTURE.read_text(encoding="utf-8"))["vectors"]


def _aggregate_result(manifest):
    profiles = []
    for profile in manifest["profiles"]:
        prompt = load_prompt_profile(profile)
        profiles.append(
            {
                "experiment_profile_id": profile["experiment_profile_id"],
                "runtime_variant_id": profile["runtime_variant_id"],
                "model_family": profile["model_family"],
                "protocol_id": profile["protocol_id"],
                "selected_profile_id": None,
                "provenance": {
                    "android_commit": ANDROID_COMMIT,
                    "baseline_id": BASELINE_ID,
                    "baseline_manifest_sha256": BASELINE_SHA256,
                    "model_revision": profile["model"]["model_revision"],
                    "artifact_sha256": profile["model"]["artifact_sha256"],
                    "chat_template_sha256": profile["model"]["chat_template_sha256"],
                    "prompt_profile_id": prompt["prompt_profile_id"],
                    "prompt_sha256": profile["prompt_profile"]["sha256"],
                    "parser_sha256": manifest["resources"]["parser"]["sha256"],
                    "runtime_profile_sha256": manifest["resources"]["runtime_profile"][
                        "sha256"
                    ],
                    "evaluator_sha256": manifest["resources"]["evaluator"]["sha256"],
                    "fixture_sha256": manifest["resources"]["synthetic_fixture"]["sha256"],
                },
                "evidence": {
                    "source_static": {
                        "status": "measured",
                        "conformance": {
                            "passed": 13,
                            "total": 13,
                            "uncaught_exceptions": 0,
                        },
                    },
                    "host_hf": {
                        "status": "not_applicable",
                        "gap_codes": ["hf_artifact_not_in_phase_d_scope"],
                    },
                    "host_gguf": {
                        "status": "not_measured",
                        "gap_codes": ["gguf_host_run_not_requested"],
                    },
                    "android_device": {
                        "status": "not_measured",
                        "gap_codes": ["android_protocol_harness_unavailable"],
                    },
                },
                "gate_status": {
                    "decision": "no_selection",
                    "passed": False,
                    "assessment": None,
                    "additional_gap_codes": [
                        "insufficient_synthetic_sample",
                        "no_reproducible_quality_baseline",
                        "protected_scoring_not_authorized",
                    ],
                },
            }
        )
    return {
        "schema_version": 1,
        "lab_id": LAB_ID,
        "lab_version": LAB_VERSION,
        "result_set_id": "pocketfinancer-android-phase-d-test-v1",
        "status": "completed_evaluation_only_no_selection",
        "implementation_commit": "1" * 40,
        "bindings": {
            "lab_manifest_sha256": manifest["manifest_sha256"],
            "android_commit": ANDROID_COMMIT,
            "baseline_id": BASELINE_ID,
            "baseline_manifest_sha256": BASELINE_SHA256,
        },
        "privacy": {
            "classification": "aggregate_only_invented_synthetic",
            "contains_private_data": False,
            "contains_row_level_predictions": False,
            "raw_model_output_retained": False,
        },
        "device_runtime_smoke": {
            "status": "not_measured",
            "evidence_class": "android_device",
            "protocol_comparison_claim": False,
            "gap_codes": ["android_protocol_harness_unavailable"],
        },
        "profiles": profiles,
        "selection": {
            "decision": "no_selection",
            "selected_profile_id": None,
            "direct_v2_selected": False,
            "candidate_v2_selected": False,
            "model_selected": False,
            "runtime_variant_selected": False,
            "production_defaults_changed": False,
        },
        "phase_e_started": False,
    }


def test_manifest_declares_independent_protocol_pair_for_every_runtime_variant():
    manifest = load_lab_manifest()
    pairs = {
        (profile["runtime_variant_id"], profile["protocol_id"])
        for profile in manifest["profiles"]
    }
    assert len(manifest["profiles"]) == 10
    assert len(pairs) == 10
    assert all(profile["selected_profile_id"] is None for profile in manifest["profiles"])
    assert manifest["bindings"]["android_commit"] == ANDROID_COMMIT
    assert manifest["bindings"]["baseline_manifest_sha256"] == BASELINE_SHA256


def test_qwen_and_gemma_use_separate_prompt_and_chat_strategies():
    manifest = load_lab_manifest()
    prompts = [load_prompt_profile(profile) for profile in manifest["profiles"]]
    qwen = [prompt for prompt in prompts if prompt["model_family"] == "qwen3"]
    gemma = [prompt for prompt in prompts if prompt["model_family"] == "gemma4"]
    assert all(prompt["chat_messages"]["system"] for prompt in qwen)
    assert all(prompt["chat_messages"]["system"] is None for prompt in gemma)
    assert {prompt["protocol_id"] for prompt in qwen} == {"direct_v2", "candidate_v2"}
    assert {prompt["protocol_id"] for prompt in gemma} == {"direct_v2", "candidate_v2"}


def test_both_adapters_pass_every_frozen_invented_conformance_vector():
    result = run_parser_conformance()
    assert result["protocols"] == {
        "direct_v2": {"passed": 13, "total": 13, "uncaught_exceptions": 0},
        "candidate_v2": {"passed": 13, "total": 13, "uncaught_exceptions": 0},
    }


@pytest.mark.parametrize("protocol_id", ["direct_v2", "candidate_v2"])
def test_protocol_parser_strips_only_a_closed_leading_thinking_block(protocol_id):
    vector = _semantic_vectors()[0]
    output = (
        vector["semantic_core"]
        if protocol_id == "direct_v2"
        else candidate_output_from_semantic_core(vector["semantic_core"])
    )
    raw = "<think>invented reasoning omitted</think>\n" + json.dumps(output)
    parsed = parse_protocol_output(
        raw,
        protocol_id=protocol_id,
        message=vector["message"],
        source_metadata=vector["source_metadata"],
    )
    assert parsed.parse_status == "valid"
    assert parsed.failure_code is None


def test_direct_parser_rejects_model_supplied_source_time():
    vector = _semantic_vectors()[0]
    output = deepcopy(vector["semantic_core"])
    output["source_metadata"] = vector["source_metadata"]
    parsed = parse_protocol_output(
        json.dumps(output),
        protocol_id="direct_v2",
        message=vector["message"],
        source_metadata=vector["source_metadata"],
    )
    assert parsed.parse_status == "invalid"
    assert parsed.failure_code == "forbidden_source_metadata"


def test_direct_parser_rejects_any_model_timestamp_field():
    vector = _semantic_vectors()[0]
    output = deepcopy(vector["semantic_core"])
    output["timestamp"] = "invented"
    parsed = parse_protocol_output(
        json.dumps(output),
        protocol_id="direct_v2",
        message=vector["message"],
        source_metadata=vector["source_metadata"],
    )
    assert parsed.failure_code == "forbidden_timestamp_field"


def test_candidate_parser_rejects_model_derived_values():
    vector = _semantic_vectors()[0]
    output = candidate_output_from_semantic_core(vector["semantic_core"])
    output["events"][0]["amount"] = "1250.50"
    parsed = parse_protocol_output(
        json.dumps(output),
        protocol_id="candidate_v2",
        message=vector["message"],
        source_metadata=vector["source_metadata"],
    )
    assert parsed.failure_code == "protocol_shape_invalid"


def test_candidate_parser_rejects_evidence_outside_the_message():
    vector = _semantic_vectors()[0]
    output = candidate_output_from_semantic_core(vector["semantic_core"])
    output["events"][0]["amount_evidence"] = {
        "start_utf8_byte": 999,
        "end_utf8_byte": 1000,
    }
    parsed = parse_protocol_output(
        json.dumps(output),
        protocol_id="candidate_v2",
        message=vector["message"],
        source_metadata=vector["source_metadata"],
    )
    assert parsed.failure_code == "ungrounded_evidence"


def test_aggregate_result_requires_all_profiles_and_no_selection():
    manifest = load_lab_manifest()
    result = validate_result_set(_aggregate_result(manifest), manifest=manifest)
    assert result["selection"]["selected_profile_id"] is None
    assert result["result_sha256"]


def test_aggregate_result_rejects_raw_content_fields():
    manifest = load_lab_manifest()
    result = _aggregate_result(manifest)
    result["device_runtime_smoke"]["message"] = "forbidden even when invented"
    with pytest.raises(AndroidProtocolLabError, match="forbidden"):
        validate_result_set(result, manifest=manifest)


def test_cli_validates_without_loading_models():
    result = subprocess.run(
        [sys.executable, "scripts/run_pocketfinancer_android_phase_d_lab.py"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["profiles_validated"] == 10
    assert summary["selection"] == "none"
