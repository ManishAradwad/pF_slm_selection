from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = REPOSITORY_ROOT / "configs/programs/pocketfinancer-extraction-v2.json"


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
    assert state["current_phase"]["id"] == "B"
    assert state["current_phase"]["status"] == "not_started"
    assert state["phase_a_review"]["status"] == "closed"
    assert state["phase_a_review"]["hardening_commit"] == state["last_completed_phase"]["commit"]
    assert state["phase_a_review"]["hardening_commits"] == state["last_completed_phase"][
        "hardening_commits"
    ]
    assert state["last_completed_phase"]["status"] == "completed_and_reviewed"

    plan = state["normative_plan"]
    assert plan["status"] == "finalized"
    assert _sha256(plan["path"]) == plan["sha256"]

    contract = state["semantic_contract"]
    assert contract["id"] == "pocketfinancer_semantic_v2"
    assert contract["version"] == 2
    assert contract["hash_algorithm"] == "sha256"
    assert contract["frozen"] is True
    assert contract["frozen_at_phase"] == "A"
    assert contract["frozen_by_commit"] == state["phase_a_review"]["hardening_commit"]
    for path_key, hash_key in (
        ("schema_path", "schema_sha256"),
        ("reference_path", "reference_sha256"),
        ("synthetic_conformance_path", "synthetic_conformance_sha256"),
    ):
        assert _sha256(contract[path_key]) == contract[hash_key]

    phases = state["phase_list"]
    assert [phase["id"] for phase in phases] == list("ABCDEFGHIJ")
    assert phases[0]["status"] == "completed_and_reviewed"
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

    assert state["selection_decision_policy"]["status"] == "required_in_phase_B"
    assert state["selection_decision_policy"]["android_runtime_variant_dispositions_required"] is True
    assert state["selection_decision_policy"]["minimum_selected_android_variants"] == 1
    assert state["selection_decision_policy"]["no_selection_is_valid"] is True
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
    assert state["private_data_prohibition"]["active"] is True
    assert state["next_allowed_phase"]["id"] == "B"
    assert "Keep Android and iOS read-only throughout Phase B." in state["next_allowed_phase"][
        "requires"
    ]
