from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = REPOSITORY_ROOT / "configs/programs/pocketfinancer-extraction-v2.json"
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")


def _state() -> dict[str, Any]:
    value = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(relative_path: str) -> str:
    return hashlib.sha256((REPOSITORY_ROOT / relative_path).read_bytes()).hexdigest()


def test_extraction_v2_program_state_is_finalized_and_hash_locked() -> None:
    state = _state()

    assert state["schema_version"] == 2
    assert state["program_id"] == "pocketfinancer-extraction-v2"
    assert state["current_phase"] == {
        "id": "C",
        "name": "Android baseline reproducibility and runtime instrumentation",
        "status": "not_started",
    }
    assert state["phase_a_review"]["status"] == "closed"
    assert state["phase_a_review"]["hardening_commit"] == state["semantic_contract"][
        "frozen_by_commit"
    ]
    assert state["phase_a_review"]["hardening_commits"] == [
        "95cb4287076642867596fbbfe2835cb105ff6399",
        "0d1abf1be91a528baf0364d3475f2b4f25d72f5e",
    ]

    plan = state["normative_plan"]
    assert plan["status"] == "finalized"
    assert _sha256(plan["path"]) == plan["sha256"]

    contract = state["semantic_contract"]
    assert contract["id"] == "pocketfinancer_semantic_v2"
    assert contract["version"] == 2
    assert contract["hash_algorithm"] == "sha256"
    assert contract["frozen"] is True
    assert contract["frozen_at_phase"] == "A"
    for path_key, hash_key in (
        ("schema_path", "schema_sha256"),
        ("reference_path", "reference_sha256"),
        ("synthetic_conformance_path", "synthetic_conformance_sha256"),
    ):
        assert _sha256(contract[path_key]) == contract[hash_key]

    phase_b = state["phase_b_review"]
    assert phase_b["status"] == "completed_and_reviewed"
    assert phase_b["branch"] == "codex/extraction-v2-phase-b"
    assert COMMIT_RE.fullmatch(phase_b["starting_handoff_commit"])
    assert phase_b["implementation_commits"] == [
        phase_b["workbench_commit"],
        phase_b["evaluation_policy_commit"],
    ]
    assert all(COMMIT_RE.fullmatch(commit) for commit in phase_b["implementation_commits"])
    assert phase_b["verification"]["pytest_full"] == "579 passed"
    assert phase_b["verification"]["verification_tier"] == "lightweight_only"

    workbench = state["workbench_v2"]
    assert workbench["status"] == "completed_and_reviewed"
    assert workbench["contract_id"] == "pocketfinancer_workbench_v2"
    assert workbench["contract_version"] == 2
    for path_key, hash_key in (
        ("contract_path", "contract_sha256"),
        ("reference_path", "reference_sha256"),
        ("evaluator_path", "evaluator_sha256"),
        ("cli_path", "cli_sha256"),
        ("synthetic_fixture_path", "synthetic_fixture_sha256"),
        ("guide_path", "guide_sha256"),
    ):
        assert _sha256(workbench[path_key]) == workbench[hash_key]
    assert workbench["private_input_root"] == "PRIVATE_DATA"
    assert workbench["aggregate_output_root"] == "RESULTS"
    assert workbench["private_workflow_authorized"] is False
    assert workbench["committed_fixture_classification"] == "invented_synthetic_only"

    policy_state = state["selection_decision_policy"]
    assert policy_state["status"] == "frozen_in_phase_B"
    assert policy_state["policy_id"] == "pocketfinancer_extraction_v2_selection_policy"
    assert policy_state["policy_version"] == 1
    assert _sha256(policy_state["threshold_artifact_path"]) == policy_state[
        "threshold_artifact_sha256"
    ]
    assert policy_state["frozen_by_commit"] == phase_b["evaluation_policy_commit"]
    assert policy_state["android_runtime_variant_dispositions_required"] is True
    assert policy_state["minimum_selected_android_variants"] == 1
    assert policy_state["no_selection_is_valid"] is True
    assert policy_state["protected_scoring_authorized"] is False
    assert policy_state["selected_profile_id"] is None
    policy = json.loads(
        (REPOSITORY_ROOT / policy_state["threshold_artifact_path"]).read_text(encoding="utf-8")
    )
    assert policy["status"] == "frozen"
    assert policy["semantic_contract"] == {
        "id": contract["id"],
        "version": contract["version"],
        "schema_sha256": contract["schema_sha256"],
        "reference_sha256": contract["reference_sha256"],
        "conformance_sha256": contract["synthetic_conformance_sha256"],
    }
    assert policy["comparison_scope"]["direct_v2_status"] == "unselected_hypothesis"
    assert policy["comparison_scope"]["candidate_v2_status"] == "unselected_hypothesis"

    phases = state["phase_list"]
    assert [phase["id"] for phase in phases] == list("ABCDEFGHIJ")
    assert phases[0]["status"] == "completed_and_reviewed"
    assert phases[1]["status"] == "completed_and_reviewed"
    assert phases[2]["status"] == "not_started"
    assert all(phase["status"] == "blocked_by_prior_phase" for phase in phases[3:])
    assert phases[8]["name"] == (
        "Production-quality three-repository implementation and default-off UAT candidate"
    )
    assert phases[9]["name"] == "User acceptance and optional default cutover/release"

    definition_of_done = state["program_definition_of_done"]
    assert definition_of_done["status"] == "not_achieved"
    assert definition_of_done["engineering_integration_phase"] == "I"
    assert definition_of_done["user_acceptance_and_release_phase"] == "J"
    assert "explicit Phase J" in definition_of_done["release_requires"]

    manifest = state["compatibility_manifest"]
    assert manifest["required"] is True
    assert manifest["status"] == "required_in_phase_I"
    assert manifest["path"].endswith("-compatibility.json")
    assert any("preceding verified SLM" in item for item in manifest["must_pin"])

    android_variants = state["selected_profile_ids"]["android_by_runtime_variant"]
    assert set(android_variants) == {
        "qwen3-0.6b-q8",
        "qwen3-1.7b-q4",
        "qwen3-1.7b-q8",
        "gemma-4-e2b-q4",
        "gemma-4-e2b-q8",
    }
    assert all(profile_id is None for profile_id in android_variants.values())
    assert (
        state["android_runtime_variant_selection"]["complete_disposition_required_before_phase_I"]
        is True
    )
    assert state["selected_profile_ids"]["ios"] is None
    assert state["last_completed_phase"] == {
        "id": "B",
        "status": "completed_and_reviewed",
        "workbench_commit": phase_b["workbench_commit"],
        "evaluation_policy_commit": phase_b["evaluation_policy_commit"],
        "implementation_commits": phase_b["implementation_commits"],
    }
    assert state["private_data_prohibition"]["active"] is True
    assert state["private_data_prohibition"]["permitted_in_this_program_state"] == (
        "invented_synthetic_only"
    )
    assert state["next_allowed_phase"]["id"] == "C"
    assert any("Do not select Direct V2" in item for item in state["next_allowed_phase"]["requires"])
